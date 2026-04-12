from __future__ import annotations

import base64
import os
import queue
import threading
from pathlib import Path

import cv2

from src.pipeline import LocalizationPipeline
from src.poi_overlay import PoiOverlay, PoiRenderOptions
from src.utils import (
    AppConfig,
    apply_map_metadata_defaults,
    draw_localization,
    ensure_directory,
    format_result_text,
    guess_poi_categories_path,
    list_image_files,
    load_config,
    load_image,
    load_map_metadata_from_poi_data,
    result_json,
)

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - depends on local Python build
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None


class LocalizationGUI:
    def __init__(self) -> None:
        if tk is None or ttk is None:
            raise RuntimeError(
                "当前 Python 环境不可用 Tkinter，"
                "请安装带 Tk 支持的 Python 版本。"
            ) from TK_IMPORT_ERROR

        self.root = tk.Tk()
        self.root.title("ROCO 地图定位器")
        self.root.geometry("1320x860")
        self.root.minsize(1100, 760)

        default_config_path = "configs/rocom_17173.yaml" if Path("configs/rocom_17173.yaml").exists() else "configs/default.yaml"
        self.config_path_var = tk.StringVar(value=default_config_path)
        self.map_path_var = tk.StringVar(value="")
        self.poi_data_path_var = tk.StringVar(value="")
        self.input_mode_var = tk.StringVar(value="frame")
        self.input_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="outputs")
        self.save_visualizations_var = tk.BooleanVar(value=True)
        self.show_poi_overlay_var = tk.BooleanVar(value=False)
        self.show_poi_labels_var = tk.BooleanVar(value=False)
        self.poi_keyword_var = tk.StringVar(value="")
        self.poi_summary_var = tk.StringVar(value="未加载点位数据")
        self.status_var = tk.StringVar(value="就绪")

        self.preview_label = None
        self.log_text = None
        self.category_listbox = None
        self._preview_photo = None
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._poi_overlay: PoiOverlay | None = None
        self._poi_categories: list[tuple[object, int]] = []

        self._build_layout()
        self._load_config_defaults()
        self.root.after(100, self._poll_results)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=16)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="配置文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.config_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="选择", command=self._select_config).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="载入", command=self._load_config_defaults).grid(row=0, column=3)

        ttk.Label(controls, text="完整地图").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.map_path_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(controls, text="选择", command=self._select_map).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(controls, text="点位数据").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.poi_data_path_var).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        poi_action_row = ttk.Frame(controls)
        poi_action_row.grid(row=2, column=2, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(poi_action_row, text="选择", command=self._select_poi_data).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(poi_action_row, text="载入点位", command=self._load_poi_catalog).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(poi_action_row, textvariable=self.poi_summary_var).grid(row=0, column=2, sticky="w")

        ttk.Label(controls, text="输入源").grid(row=3, column=0, sticky="w", pady=(8, 0))
        input_row = ttk.Frame(controls)
        input_row.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        input_row.columnconfigure(3, weight=1)

        ttk.Radiobutton(input_row, text="单张截图", value="frame", variable=self.input_mode_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Radiobutton(input_row, text="截图文件夹", value="frames_dir", variable=self.input_mode_var).grid(row=0, column=1, padx=(0, 8))
        ttk.Radiobutton(input_row, text="视频", value="video", variable=self.input_mode_var).grid(row=0, column=2, padx=(0, 8))
        ttk.Entry(input_row, textvariable=self.input_path_var).grid(row=0, column=3, sticky="ew", padx=(0, 8))
        ttk.Button(input_row, text="选择", command=self._select_input).grid(row=0, column=4)

        ttk.Label(controls, text="输出目录").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(row=4, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(controls, text="选择", command=self._select_output_dir).grid(row=4, column=2, padx=4, pady=(8, 0))
        ttk.Checkbutton(
            controls,
            text="保存可视化结果",
            variable=self.save_visualizations_var,
        ).grid(row=4, column=3, sticky="w", pady=(8, 0))

        filter_row = ttk.Frame(controls)
        filter_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        filter_row.columnconfigure(3, weight=1)
        ttk.Checkbutton(filter_row, text="显示点位", variable=self.show_poi_overlay_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(filter_row, text="显示名称", variable=self.show_poi_labels_var).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(filter_row, text="关键词").grid(row=0, column=2, sticky="w")
        ttk.Entry(filter_row, textvariable=self.poi_keyword_var).grid(row=0, column=3, sticky="ew", padx=(8, 8))
        ttk.Label(filter_row, text="未选分类时不会叠加点位").grid(row=0, column=4, sticky="e")

        action_row = ttk.Frame(controls)
        action_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        action_row.columnconfigure(2, weight=1)
        ttk.Button(action_row, text="开始定位", command=self._start_localization).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(action_row, text="停止", command=self._stop_localization).grid(row=0, column=1)
        ttk.Label(action_row, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        left_panel = ttk.Frame(content, padding=12)
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        ttk.Label(left_panel, text="定位预览").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(left_panel, anchor="center", text="尚无预览")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        content.add(left_panel, weight=3)

        right_panel = ttk.Frame(content, padding=12)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)
        right_panel.rowconfigure(4, weight=2)
        ttk.Label(right_panel, text="点位分类").grid(row=0, column=0, sticky="w")
        category_toolbar = ttk.Frame(right_panel)
        category_toolbar.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        ttk.Button(category_toolbar, text="全选", command=self._select_all_categories).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(category_toolbar, text="清空", command=self._clear_category_selection).grid(row=0, column=1)

        self.category_listbox = tk.Listbox(right_panel, selectmode=tk.MULTIPLE, exportselection=False, height=12)
        self.category_listbox.grid(row=2, column=0, sticky="nsew")
        category_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.category_listbox.yview)
        category_scrollbar.grid(row=2, column=1, sticky="ns")
        self.category_listbox.configure(yscrollcommand=category_scrollbar.set)

        ttk.Label(right_panel, text="结果日志").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.log_text = tk.Text(right_panel, wrap="word", width=48)
        self.log_text.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=4, column=1, sticky="ns", pady=(8, 0))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        content.add(right_panel, weight=2)

    def _load_config_defaults(self) -> None:
        config_path = Path(self.config_path_var.get()).expanduser()
        if not config_path.exists():
            self.status_var.set("未找到配置文件，将使用当前输入")
            return

        try:
            config = load_config(config_path)
        except Exception as exc:
            self.status_var.set("载入配置失败")
            messagebox.showerror("配置错误", str(exc))
            return

        self.map_path_var.set(config.map_path)
        self.poi_data_path_var.set(config.poi_data_path)
        self.output_dir_var.set(config.output_dir)
        self.save_visualizations_var.set(config.save_visualizations)
        self.show_poi_overlay_var.set(config.show_poi_overlay)
        self.show_poi_labels_var.set(config.show_poi_labels)
        self.poi_keyword_var.set(config.poi_keyword)
        self._load_poi_catalog()
        self._apply_category_selection(config.poi_category_ids)
        self.status_var.set("配置已载入")

    def _select_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if selected:
            self.config_path_var.set(selected)

    def _select_map(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择完整地图",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if selected:
            self.map_path_var.set(selected)

    def _select_poi_data(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择点位数据",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.poi_data_path_var.set(selected)

    def _select_input(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "frame":
            selected = filedialog.askopenfilename(
                title="选择局部截图",
                filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
            )
        elif mode == "frames_dir":
            selected = filedialog.askdirectory(title="选择截图文件夹")
        else:
            selected = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv"), ("所有文件", "*.*")],
            )
        if selected:
            self.input_path_var.set(selected)

    def _select_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_dir_var.set(selected)

    def _start_localization(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("提示", "定位任务已经在运行中。")
            return

        try:
            config = self._build_runtime_config()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return

        selected_category_ids = self._get_selected_category_ids()

        self.log_text.delete("1.0", tk.END)
        self._set_preview(None)
        self.status_var.set("运行中...")
        self._stop_event.clear()

        input_mode = self.input_mode_var.get()
        input_path = self.input_path_var.get().strip()
        save_visualizations = self.save_visualizations_var.get()
        output_dir = self.output_dir_var.get().strip() or config.output_dir

        self._worker = threading.Thread(
            target=self._run_pipeline,
            args=(config, input_mode, input_path, output_dir, save_visualizations, selected_category_ids),
            daemon=True,
        )
        self._worker.start()

    def _stop_localization(self) -> None:
        self._stop_event.set()
        self.status_var.set("正在停止...")

    def _run_pipeline(
        self,
        config: AppConfig,
        input_mode: str,
        input_path: str,
        output_dir: str,
        save_visualizations: bool,
        selected_category_ids: list[int],
    ) -> None:
        try:
            pipeline = LocalizationPipeline(config)
            output_path = ensure_directory(output_dir) if save_visualizations else None
            poi_overlay = self._build_overlay_for_run(config)

            for frame_name, frame in self._iterate_inputs(input_mode, input_path):
                if self._stop_event.is_set():
                    break

                result = pipeline.process_frame(frame)
                map_image = pipeline.map_bundle.color
                extra_lines = None
                if poi_overlay is not None:
                    map_image, summary = poi_overlay.render_map(
                        map_image,
                        PoiRenderOptions(
                            enabled=config.show_poi_overlay,
                            selected_category_ids=selected_category_ids,
                            keyword=config.poi_keyword,
                            show_labels=config.show_poi_labels,
                            max_points=config.poi_max_draw,
                            label_limit=config.poi_label_limit,
                        ),
                        focus_xy=(result.x, result.y) if result.x == result.x and result.y == result.y else None,
                    )
                    extra_lines = [summary.text()] if summary is not None else None
                visualization = draw_localization(
                    map_image,
                    frame,
                    result,
                    extra_lines=extra_lines,
                    max_panel_height=620,
                )
                encoded = self._encode_preview(visualization)

                if output_path is not None:
                    cv2.imwrite(str(output_path / f"{frame_name}.png"), visualization)

                self._result_queue.put(
                    (
                        "frame",
                        {
                            "frame_name": frame_name,
                            "message": result_json(frame_name, result),
                            "message_text": format_result_text(frame_name, result),
                            "preview": encoded,
                        },
                    )
                )

            final_status = "已停止" if self._stop_event.is_set() else "已完成"
            self._result_queue.put(("done", final_status))
        except Exception as exc:
            self._result_queue.put(("error", str(exc)))

    def _poll_results(self) -> None:
        while True:
            try:
                message_type, payload = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "frame":
                self.log_text.insert(tk.END, f"{payload['message_text']}\n{payload['message']}\n\n")
                self.log_text.see(tk.END)
                self._set_preview(payload["preview"])
                self.status_var.set(f"正在处理：{payload['frame_name']}")
            elif message_type == "done":
                self.status_var.set(str(payload))
            elif message_type == "error":
                self.status_var.set("运行失败")
                messagebox.showerror("运行错误", str(payload))

        self.root.after(100, self._poll_results)

    def _build_runtime_config(self) -> AppConfig:
        config_path = self.config_path_var.get().strip()
        config = load_config(config_path) if config_path and Path(config_path).exists() else AppConfig()

        map_path = self.map_path_var.get().strip()
        poi_data_path = self.poi_data_path_var.get().strip()
        input_path = self.input_path_var.get().strip()
        if not map_path:
            raise ValueError("请选择完整地图图像。")
        if not input_path:
            raise ValueError("请选择局部截图、截图文件夹或视频。")

        config.map_path = map_path
        config.poi_data_path = poi_data_path
        config.poi_categories_path = guess_poi_categories_path(poi_data_path, config.poi_categories_path)
        config.output_dir = self.output_dir_var.get().strip() or config.output_dir
        config.save_visualizations = self.save_visualizations_var.get()
        config.show_poi_overlay = self.show_poi_overlay_var.get()
        config.show_poi_labels = self.show_poi_labels_var.get()
        config.poi_keyword = self.poi_keyword_var.get().strip()
        return apply_map_metadata_defaults(config)

    def _iterate_inputs(self, input_mode: str, input_path: str):
        if input_mode == "frame":
            frame_path = Path(input_path)
            yield frame_path.stem, load_image(frame_path)
            return

        if input_mode == "frames_dir":
            for image_path in list_image_files(input_path):
                yield image_path.stem, load_image(image_path)
            return

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频文件：{input_path}")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield f"frame_{frame_index:06d}", frame
                frame_index += 1
        finally:
            capture.release()

    def _set_preview(self, encoded_preview: str | None) -> None:
        if not encoded_preview:
            self.preview_label.configure(image="", text="尚无预览")
            self._preview_photo = None
            return

        self._preview_photo = tk.PhotoImage(data=encoded_preview)
        self.preview_label.configure(image=self._preview_photo, text="")

    @staticmethod
    def _encode_preview(image) -> str:
        height, width = image.shape[:2]
        scale = min(1.0, 1080 / width, 620 / height)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("预览图编码失败。")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _on_close(self) -> None:
        self._stop_event.set()
        self.root.destroy()

    def _load_poi_catalog(self) -> None:
        poi_path = self.poi_data_path_var.get().strip()
        if not poi_path:
            self.poi_summary_var.set("未加载点位数据")
            return

        categories_path = guess_poi_categories_path(poi_path)
        bounds = self._guess_map_bounds()
        if bounds is None:
            self.poi_summary_var.set("缺少地图边界，无法映射点位")
            return

        try:
            self._poi_overlay = PoiOverlay(
                pois_path=poi_path,
                categories_path=categories_path or None,
                map_bounds=bounds,
            )
        except Exception as exc:
            self._poi_overlay = None
            self._poi_categories = []
            self.category_listbox.delete(0, tk.END)
            self.poi_summary_var.set("点位载入失败")
            messagebox.showerror("点位数据错误", str(exc))
            return

        self._poi_categories = self._poi_overlay.available_categories()
        self.category_listbox.delete(0, tk.END)
        for category, count in self._poi_categories:
            self.category_listbox.insert(tk.END, f"{category.group_title} / {category.title} ({count})")
        self.poi_summary_var.set(
            f"已载入 {len(self._poi_overlay.records)} 个点位，{len(self._poi_categories)} 个分类"
        )

    def _select_all_categories(self) -> None:
        self.category_listbox.select_set(0, tk.END)

    def _clear_category_selection(self) -> None:
        self.category_listbox.selection_clear(0, tk.END)

    def _apply_category_selection(self, category_ids: list[int]) -> None:
        if not self._poi_categories:
            return

        selected = set(category_ids)
        self.category_listbox.selection_clear(0, tk.END)
        for index, (category, _count) in enumerate(self._poi_categories):
            if category.id in selected:
                self.category_listbox.selection_set(index)

    def _get_selected_category_ids(self) -> list[int]:
        selected_ids = []
        for index in self.category_listbox.curselection():
            selected_ids.append(self._poi_categories[index][0].id)
        return selected_ids

    def _build_overlay_for_run(self, config: AppConfig) -> PoiOverlay | None:
        if not config.poi_data_path or not config.map_bounds or len(config.map_bounds) != 4:
            return None
        return PoiOverlay(
            pois_path=config.poi_data_path,
            categories_path=guess_poi_categories_path(config.poi_data_path, config.poi_categories_path) or None,
            map_bounds=tuple(float(value) for value in config.map_bounds),
            projection_type=config.map_projection,
            tile_zoom=config.map_tile_zoom,
            tile_x_range=tuple(config.map_tile_x_range) if len(config.map_tile_x_range) == 2 else None,
            tile_y_range=tuple(config.map_tile_y_range) if len(config.map_tile_y_range) == 2 else None,
            tile_size=config.map_tile_size,
        )

    def _guess_map_bounds(self) -> tuple[float, float, float, float] | None:
        config_path = self.config_path_var.get().strip()
        if config_path and Path(config_path).exists():
            try:
                config = load_config(config_path)
                if config.map_bounds and len(config.map_bounds) == 4:
                    return tuple(float(value) for value in config.map_bounds)
            except Exception:
                pass
        poi_path = self.poi_data_path_var.get().strip()
        if poi_path:
            metadata = load_map_metadata_from_poi_data(poi_path)
            bounds = metadata.get("bounds") if isinstance(metadata, dict) else None
            if isinstance(bounds, list) and len(bounds) == 4:
                return tuple(float(value) for value in bounds)
        return None


def launch_gui() -> None:
    app = LocalizationGUI()
    app.run()
