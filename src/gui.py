from __future__ import annotations

import base64
import math
import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from src.async_frame_pipeline import LatestFrameAsyncPipeline
from src.collectible_materials import (
    COLLECTIBLE_KIND_MINERAL,
    COLLECTIBLE_KIND_PLANT,
    active_collectible_materials,
    collectible_ids_for_kind,
)
from src.map_alignment import MapAlignment, resolve_map_alignment
from src.pipeline import LocalizationPipeline
from src.poi_overlay import PoiOverlay, PoiRenderOptions
from src.resource_routes import (
    ResourceRoutePlan,
    build_resource_route_plan,
    build_route_cache_signature,
    load_route_plan_cache,
    render_resource_route_plan_viewport,
    route_cache_path,
    save_route_plan_cache,
    summarize_selection_label,
)
from src.resource_sources import (
    ResourceSourceContext,
    build_biliwiki_resource_context,
)
from src.screen_pick import format_capture_region, iterate_screen_region_frames, parse_capture_region, pick_screen_region
from src.utils import (
    AppConfig,
    LocalizationResult,
    apply_map_metadata_defaults,
    draw_localization,
    ensure_directory,
    format_result_text,
    guess_poi_categories_path,
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
        self.root.geometry("1380x900")
        self.root.minsize(1180, 780)

        if Path("configs/rocom_tracker.yaml").exists():
            default_config_path = "configs/rocom_tracker.yaml"
        elif Path("configs/rocom_17173.yaml").exists():
            default_config_path = "configs/rocom_17173.yaml"
        else:
            default_config_path = "configs/default.yaml"
        self.config_path_var = tk.StringVar(value=default_config_path)
        self.map_path_var = tk.StringVar(value="")
        self.poi_data_path_var = tk.StringVar(value="")
        self.input_mode_var = tk.StringVar(value="screen_region")
        self.input_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="outputs")
        self.save_visualizations_var = tk.BooleanVar(value=True)
        self.show_poi_overlay_var = tk.BooleanVar(value=True)
        self.show_poi_labels_var = tk.BooleanVar(value=False)
        self.poi_keyword_var = tk.StringVar(value="")
        self.poi_summary_var = tk.StringVar(value="未加载点位数据")
        self.status_var = tk.StringVar(value="就绪")
        self.capture_interval_var = tk.StringVar(value="250")
        self.recognition_enabled_var = tk.BooleanVar(value=False)
        self.map_zoom_var = tk.DoubleVar(value=1.0)
        self.map_hint_var = tk.StringVar(value="等待加载地图")
        self.resource_route_summary_var = tk.StringVar(value="未生成资源路线")

        self.map_canvas = None
        self.log_text = None
        self.category_listbox = None
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._poi_overlay: PoiOverlay | None = None
        self._poi_categories: list[tuple[object, int]] = []
        self._poi_overlay_signature: tuple | None = None

        self._map_photo = None
        self._map_canvas_image_id: int | None = None
        self._map_source_path = ""
        self._map_source_image: np.ndarray | None = None
        self._map_alignment_cache: dict[tuple[str, str], MapAlignment | None] = {}
        self._map_base_view: np.ndarray | None = None
        self._map_base_key: tuple | None = None
        self._map_view_origin: tuple[int, int] = (0, 0)
        self._map_scale: float = 1.0
        self._map_refresh_pending = False
        self._map_refresh_center = False
        self._suppress_canvas_view_refresh = False
        self._live_map_refresh_interval_ms = 250
        self._last_result: LocalizationResult | None = None
        self._resource_route_plan: ResourceRoutePlan | None = None
        self._resource_route_cache_dir = Path("outputs/route_cache")
        self._biliwiki_resource_context: ResourceSourceContext | None = None

        self._build_layout()
        self._bind_refresh_events()
        self._load_config_defaults()
        self.root.after(100, self._poll_results)
        self.root.after(self._live_map_refresh_interval_ms, self._poll_live_map_refresh)
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

        ttk.Label(controls, text="采集点位").grid(row=2, column=0, sticky="w", pady=(8, 0))
        poi_action_row = ttk.Frame(controls)
        poi_action_row.grid(row=2, column=1, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(poi_action_row, text="自动使用 wiki.biligame.com 的采集点缓存").grid(row=0, column=0, padx=(0, 8))
        ttk.Button(poi_action_row, text="刷新 biliwiki", command=self._reload_biliwiki_collectibles).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(poi_action_row, textvariable=self.poi_summary_var).grid(row=0, column=2, sticky="w")

        ttk.Label(controls, text="输入源").grid(row=3, column=0, sticky="w", pady=(8, 0))
        input_row = ttk.Frame(controls)
        input_row.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        input_row.columnconfigure(2, weight=1)

        ttk.Radiobutton(input_row, text="单张截图", value="frame", variable=self.input_mode_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Radiobutton(input_row, text="屏幕区域", value="screen_region", variable=self.input_mode_var).grid(row=0, column=1, padx=(0, 8))
        ttk.Entry(input_row, textvariable=self.input_path_var).grid(row=0, column=2, sticky="ew", padx=(0, 8))
        ttk.Button(input_row, text="选择", command=self._select_input).grid(row=0, column=3)
        ttk.Label(input_row, text="间隔(ms)").grid(row=0, column=4, padx=(8, 4))
        ttk.Entry(input_row, width=8, textvariable=self.capture_interval_var).grid(row=0, column=5)

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
        ttk.Checkbutton(
            filter_row,
            text="显示点位",
            variable=self.show_poi_overlay_var,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(filter_row, text="点位与路线均基于 biliwiki 采集模块").grid(row=0, column=1, sticky="w")
        ttk.Button(filter_row, text="生成路线", command=self._generate_resource_route).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(filter_row, text="清除路线", command=self._clear_resource_route).grid(row=0, column=3, sticky="w", padx=(8, 0))
        ttk.Label(filter_row, textvariable=self.resource_route_summary_var).grid(row=0, column=4, sticky="e", padx=(12, 0))

        action_row = ttk.Frame(controls)
        action_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        action_row.columnconfigure(2, weight=1)
        ttk.Checkbutton(
            action_row,
            text="启用识别",
            variable=self.recognition_enabled_var,
            command=self._toggle_localization,
        ).grid(row=0, column=0, padx=(0, 12))
        ttk.Label(action_row, text="勾选后按当前输入源开始识别，取消勾选即停止").grid(row=0, column=1, sticky="w")
        ttk.Label(action_row, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        left_panel = ttk.Frame(content, padding=12)
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)

        ttk.Label(left_panel, text="地图视图").grid(row=0, column=0, sticky="w")

        map_toolbar = ttk.Frame(left_panel)
        map_toolbar.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        map_toolbar.columnconfigure(4, weight=1)
        ttk.Button(map_toolbar, text="缩小", command=lambda: self._step_map_zoom(1 / 1.25)).grid(row=0, column=0, padx=(0, 6))
        ttk.Scale(
            map_toolbar,
            from_=0.5,
            to=8.0,
            variable=self.map_zoom_var,
            orient=tk.HORIZONTAL,
            command=self._on_map_zoom_changed,
        ).grid(row=0, column=1, sticky="ew")
        ttk.Button(map_toolbar, text="放大", command=lambda: self._step_map_zoom(1.25)).grid(row=0, column=2, padx=6)
        ttk.Button(map_toolbar, text="适配", command=self._reset_map_zoom).grid(row=0, column=3, padx=(0, 10))
        ttk.Label(map_toolbar, textvariable=self.map_hint_var).grid(row=0, column=4, sticky="e")

        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(canvas_frame, background="#101010", highlightthickness=0)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._on_canvas_yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._on_canvas_xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.map_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.map_canvas.bind("<Configure>", self._on_map_canvas_configure)
        self.map_canvas.bind("<MouseWheel>", self._on_map_mousewheel)
        self.map_canvas.bind("<Button-4>", self._on_map_mousewheel)
        self.map_canvas.bind("<Button-5>", self._on_map_mousewheel)
        content.add(left_panel, weight=3)

        right_panel = ttk.Frame(content, padding=12)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)
        right_panel.rowconfigure(4, weight=2)
        ttk.Label(right_panel, text="采集素材").grid(row=0, column=0, sticky="w")
        category_toolbar = ttk.Frame(right_panel)
        category_toolbar.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        ttk.Button(category_toolbar, text="全选", command=self._select_all_categories).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(category_toolbar, text="只选矿物", command=lambda: self._select_category_kind(COLLECTIBLE_KIND_MINERAL)).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(category_toolbar, text="只选植物", command=lambda: self._select_category_kind(COLLECTIBLE_KIND_PLANT)).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(category_toolbar, text="清空", command=self._clear_category_selection).grid(row=0, column=3)

        self.category_listbox = tk.Listbox(right_panel, selectmode=tk.MULTIPLE, exportselection=False, height=12)
        self.category_listbox.grid(row=2, column=0, sticky="nsew")
        category_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.category_listbox.yview)
        category_scrollbar.grid(row=2, column=1, sticky="ns")
        self.category_listbox.configure(yscrollcommand=category_scrollbar.set)
        self.category_listbox.bind("<<ListboxSelect>>", lambda _event: self._schedule_map_refresh(invalidate_base=True))

        ttk.Label(right_panel, text="结果日志").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.log_text = tk.Text(right_panel, wrap="word", width=48)
        self.log_text.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=4, column=1, sticky="ns", pady=(8, 0))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        content.add(right_panel, weight=2)

    def _bind_refresh_events(self) -> None:
        self.show_poi_overlay_var.trace_add("write", lambda *_args: self._schedule_map_refresh(invalidate_base=True))

    def _load_config_defaults(self) -> None:
        config_path = Path(self.config_path_var.get()).expanduser()
        if not config_path.exists():
            self.status_var.set("未找到配置文件，将使用当前输入")
            self._reload_biliwiki_collectibles(silent=True)
            self._schedule_map_refresh(invalidate_base=True)
            return

        try:
            config = load_config(config_path)
        except Exception as exc:
            self.status_var.set("载入配置失败")
            messagebox.showerror("配置错误", str(exc))
            return

        try:
            resource_context = self._ensure_biliwiki_resource_context(force_refresh=False)
        except Exception as exc:
            self.status_var.set("载入 biliwiki 采集点失败")
            messagebox.showerror("资源错误", str(exc))
            return

        self.map_path_var.set(config.map_path)
        self.poi_data_path_var.set(resource_context.config.poi_data_path)
        self.output_dir_var.set(config.output_dir)
        self.save_visualizations_var.set(config.save_visualizations)
        self.show_poi_overlay_var.set(True)
        self.show_poi_labels_var.set(False)
        self.poi_keyword_var.set("")
        self.capture_interval_var.set(str(config.capture_interval_ms))
        if config.capture_region:
            self.input_mode_var.set("screen_region")
            self.input_path_var.set(format_capture_region(config.capture_region))

        self._last_result = None
        self._clear_resource_route(silent=True)
        self._load_poi_catalog()
        self._apply_category_selection(config.poi_category_ids)
        self.status_var.set("配置已载入")
        self._schedule_map_refresh(invalidate_base=True)

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
            self._last_result = None
            self._clear_resource_route(silent=True)
            self._schedule_map_refresh(invalidate_base=True)

    def _reload_biliwiki_collectibles(self, silent: bool = False) -> None:
        try:
            resource_context = self._ensure_biliwiki_resource_context(force_refresh=True)
        except Exception as exc:
            if not silent:
                messagebox.showerror("刷新 biliwiki", str(exc))
            self.status_var.set("biliwiki 采集点刷新失败")
            return

        self.poi_data_path_var.set(resource_context.config.poi_data_path)
        self._clear_resource_route(silent=True)
        self._load_poi_catalog()
        self.status_var.set("biliwiki 采集点已刷新")
        if resource_context.points_refreshed:
            self.map_hint_var.set("已同步最新 biliwiki 采集点")

    def _select_input(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "frame":
            selected = filedialog.askopenfilename(
                title="选择局部截图",
                filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
            )
        else:
            self.status_var.set("请框选需要持续采集的矩形区域")

            def on_done(region: tuple[int, int, int, int]) -> None:
                self.input_path_var.set(format_capture_region(region))
                self.status_var.set("捕捉区域已选定")

            def on_cancel() -> None:
                self.status_var.set("已取消区域选择")

            try:
                pick_screen_region(self.root, on_done=on_done, on_cancel=on_cancel)
            except Exception as exc:
                messagebox.showerror("区域选择失败", str(exc))
                self.status_var.set("区域选择失败")
            return
        if selected:
            self.input_path_var.set(selected)

    def _select_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_dir_var.set(selected)

    def _toggle_localization(self) -> None:
        if self.recognition_enabled_var.get():
            self._start_localization()
        else:
            self._stop_localization()

    def _start_localization(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        try:
            config = self._build_runtime_config(require_input=True)
        except Exception as exc:
            self.recognition_enabled_var.set(False)
            messagebox.showerror("配置错误", str(exc))
            return

        self.log_text.delete("1.0", tk.END)
        self.status_var.set("识别中...")
        self._stop_event.clear()

        input_mode = self.input_mode_var.get()
        input_path = self.input_path_var.get().strip()
        save_visualizations = self.save_visualizations_var.get()
        output_dir = self.output_dir_var.get().strip() or config.output_dir
        selected_category_ids = self._get_selected_category_ids()

        self._worker = threading.Thread(
            target=self._run_pipeline,
            args=(config, input_mode, input_path, output_dir, save_visualizations, selected_category_ids),
            daemon=True,
        )
        self._worker.start()

    def _stop_localization(self) -> None:
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            self.status_var.set("正在停止识别...")
        else:
            self.status_var.set("识别已关闭")

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
            poi_overlay = self._build_overlay_for_run(config) if save_visualizations else None
            if input_mode == "screen_region":
                async_stats = LatestFrameAsyncPipeline(max_pending_frames=2).run(
                    frame_source=self._iterate_inputs(
                        input_mode,
                        input_path,
                        capture_interval_ms=config.capture_interval_ms,
                    ),
                    frame_processor=lambda frame_name, frame: self._process_localization_frame(
                        pipeline,
                        config,
                        frame_name,
                        frame,
                        output_path,
                        poi_overlay,
                        selected_category_ids,
                    ),
                    stop_event=self._stop_event,
                )
                final_status = (
                    "已停止"
                    if self._stop_event.is_set()
                    else (
                        f"已完成（采集 {async_stats.captured_frames} 帧，"
                        f"识别 {async_stats.processed_frames} 帧，"
                        f"丢弃 {async_stats.dropped_frames} 帧）"
                    )
                )
            else:
                for frame_name, frame in self._iterate_inputs(
                    input_mode,
                    input_path,
                    capture_interval_ms=config.capture_interval_ms,
                ):
                    if self._stop_event.is_set():
                        break
                    self._process_localization_frame(
                        pipeline,
                        config,
                        frame_name,
                        frame,
                        output_path,
                        poi_overlay,
                        selected_category_ids,
                    )

                final_status = "已停止" if self._stop_event.is_set() else "已完成"
            self._result_queue.put(("done", final_status))
        except Exception as exc:
            self._result_queue.put(("error", str(exc)))

    def _process_localization_frame(
        self,
        pipeline: LocalizationPipeline,
        config: AppConfig,
        frame_name: str,
        frame: np.ndarray,
        output_path: Path | None,
        poi_overlay: PoiOverlay | None,
        selected_category_ids: list[int],
    ) -> None:
        result = pipeline.process_frame(frame)

        if output_path is not None:
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
            cv2.imwrite(str(output_path / f"{frame_name}.png"), visualization)

        self._result_queue.put(
            (
                "frame",
                {
                    "frame_name": frame_name,
                    "message": result_json(frame_name, result),
                    "message_text": format_result_text(frame_name, result),
                    "result": result,
                },
            )
        )

    def _poll_results(self) -> None:
        while True:
            try:
                message_type, payload = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "frame":
                self.log_text.insert(tk.END, f"{payload['message_text']}\n{payload['message']}\n\n")
                self.log_text.see(tk.END)
                self._last_result = payload["result"]
                self.status_var.set(f"正在处理：{payload['frame_name']}")
                self.map_hint_var.set(payload["message_text"])
                self._schedule_map_refresh(center_on_result=self.recognition_enabled_var.get())
            elif message_type == "done":
                self._worker = None
                self.status_var.set(str(payload))
                self.recognition_enabled_var.set(False)
            elif message_type == "error":
                self._worker = None
                self.recognition_enabled_var.set(False)
                self.status_var.set("识别失败")
                messagebox.showerror("运行错误", str(payload))

        self.root.after(100, self._poll_results)

    def _poll_live_map_refresh(self) -> None:
        if (
            self.recognition_enabled_var.get()
            and self._worker is not None
            and self._worker.is_alive()
            and self._last_result is not None
        ):
            self._schedule_map_refresh(center_on_result=True)

        self.root.after(self._live_map_refresh_interval_ms, self._poll_live_map_refresh)

    def _build_runtime_config(self, require_input: bool = True) -> AppConfig:
        config_path = self.config_path_var.get().strip()
        config = load_config(config_path) if config_path and Path(config_path).exists() else AppConfig()

        map_path = self.map_path_var.get().strip()
        input_path = self.input_path_var.get().strip()
        input_mode = self.input_mode_var.get().strip()

        if not map_path:
            raise ValueError("请选择完整地图图像。")
        if require_input and not input_path:
            raise ValueError("请选择输入源或屏幕捕捉区域。")

        config.map_path = map_path
        config.display_map_path = ""
        config.poi_data_path = ""
        config.poi_categories_path = ""
        config.poi_icon_dir = ""
        config.output_dir = self.output_dir_var.get().strip() or config.output_dir
        config.save_visualizations = self.save_visualizations_var.get()
        config.show_poi_overlay = False
        config.show_poi_labels = False
        config.poi_keyword = ""
        config.capture_interval_ms = max(0, int(self.capture_interval_var.get().strip() or "250"))
        if input_mode == "screen_region" and input_path:
            config.capture_region = list(parse_capture_region(input_path))
        return apply_map_metadata_defaults(config)

    def _build_display_config(self) -> AppConfig | None:
        try:
            config = self._build_runtime_config(require_input=False)
        except Exception:
            return None
        config.show_poi_overlay = self.show_poi_overlay_var.get()
        config.show_poi_labels = False
        config.poi_keyword = ""
        return self._apply_biliwiki_overlay_defaults(config)

    def _iterate_inputs(
        self,
        input_mode: str,
        input_path: str,
        capture_interval_ms: int,
    ):
        if input_mode == "frame":
            frame_path = Path(input_path)
            yield frame_path.stem, load_image(frame_path)
            return

        yield from iterate_screen_region_frames(
            input_path,
            interval_ms=max(0, int(capture_interval_ms)),
            stop_event=self._stop_event,
        )

    def _on_close(self) -> None:
        self._stop_event.set()
        self.root.destroy()

    def _load_poi_catalog(self) -> None:
        poi_path = self.poi_data_path_var.get().strip()
        previous_selected_ids = self._get_selected_category_ids() if self._poi_categories else []
        if not poi_path:
            self._poi_overlay = None
            self._poi_overlay_signature = None
            self._poi_categories = []
            self.category_listbox.delete(0, tk.END)
            self.poi_summary_var.set("未加载点位数据")
            self._clear_resource_route(silent=True)
            self._schedule_map_refresh(invalidate_base=True)
            return

        config = self._build_overlay_config_for_catalog(poi_path)
        if config is None:
            self.poi_summary_var.set("缺少地图边界，无法映射点位")
            self._clear_resource_route(silent=True)
            self._schedule_map_refresh(invalidate_base=True)
            return

        try:
            overlay = self._build_overlay_from_config(config)
        except Exception as exc:
            self._poi_overlay = None
            self._poi_overlay_signature = None
            self._poi_categories = []
            self.category_listbox.delete(0, tk.END)
            self.poi_summary_var.set("点位载入失败")
            messagebox.showerror("点位数据错误", str(exc))
            self._clear_resource_route(silent=True)
            self._schedule_map_refresh(invalidate_base=True)
            return

        self._poi_overlay = overlay
        self._poi_overlay_signature = self._overlay_signature_for_config(config)
        counts: dict[int, int] = {}
        for record in self._poi_overlay.records:
            counts[int(record.category_id)] = counts.get(int(record.category_id), 0) + 1

        self._poi_categories = [
            (material, counts.get(material.category_id, 0))
            for material in active_collectible_materials()
        ]
        self.category_listbox.delete(0, tk.END)
        for material, count in self._poi_categories:
            group_title = "矿物" if material.kind == COLLECTIBLE_KIND_MINERAL else "植物"
            self.category_listbox.insert(tk.END, f"{group_title} / {material.name} ({count})")
        if previous_selected_ids:
            self._apply_category_selection(previous_selected_ids)
        elif self._poi_categories:
            self.category_listbox.select_set(0, tk.END)
        self.poi_summary_var.set(
            f"已载入 {len(self._poi_overlay.records)} 个点位，{len(self._poi_categories)} 种素材"
        )
        self._schedule_map_refresh(invalidate_base=True)

    def _select_all_categories(self) -> None:
        self.category_listbox.select_set(0, tk.END)
        self._schedule_map_refresh(invalidate_base=True)

    def _clear_category_selection(self) -> None:
        self.category_listbox.selection_clear(0, tk.END)
        self._schedule_map_refresh(invalidate_base=True)

    def _select_category_kind(self, kind: str) -> None:
        selected_ids = set(collectible_ids_for_kind(kind))
        self.category_listbox.selection_clear(0, tk.END)
        for index, (material, _count) in enumerate(self._poi_categories):
            if material.category_id in selected_ids:
                self.category_listbox.selection_set(index)
        self._schedule_map_refresh(invalidate_base=True)

    def _apply_category_selection(self, category_ids: list[int]) -> None:
        if not self._poi_categories:
            return

        selected = set(category_ids)
        self.category_listbox.selection_clear(0, tk.END)
        for index, (material, _count) in enumerate(self._poi_categories):
            if material.category_id in selected:
                self.category_listbox.selection_set(index)
        self._schedule_map_refresh(invalidate_base=True)

    def _get_selected_category_ids(self) -> list[int]:
        selected_ids = []
        for index in self.category_listbox.curselection():
            selected_ids.append(self._poi_categories[index][0].category_id)
        return selected_ids

    def _ensure_biliwiki_resource_context(self, force_refresh: bool = False) -> ResourceSourceContext:
        if force_refresh or self._biliwiki_resource_context is None:
            self._biliwiki_resource_context = build_biliwiki_resource_context(force_refresh_points=force_refresh)
        return self._biliwiki_resource_context

    def _apply_biliwiki_overlay_defaults(self, config: AppConfig) -> AppConfig:
        resource_context = self._ensure_biliwiki_resource_context(force_refresh=False)
        biliwiki_config = resource_context.config
        config.display_map_path = biliwiki_config.display_map_path or biliwiki_config.map_path
        config.poi_data_path = biliwiki_config.poi_data_path
        config.poi_categories_path = biliwiki_config.poi_categories_path
        config.poi_icon_dir = biliwiki_config.poi_icon_dir
        config.poi_pixel_scale = biliwiki_config.poi_pixel_scale
        config.poi_pixel_offset_x = biliwiki_config.poi_pixel_offset_x
        config.poi_pixel_offset_y = biliwiki_config.poi_pixel_offset_y
        config.map_projection = biliwiki_config.map_projection
        config.map_tile_zoom = biliwiki_config.map_tile_zoom
        config.map_tile_x_range = list(biliwiki_config.map_tile_x_range)
        config.map_tile_y_range = list(biliwiki_config.map_tile_y_range)
        config.map_tile_size = biliwiki_config.map_tile_size
        return config

    def _build_overlay_for_run(self, config: AppConfig) -> PoiOverlay | None:
        return None

    def _build_overlay_config_for_catalog(self, poi_path: str) -> AppConfig | None:
        config = self._apply_biliwiki_overlay_defaults(AppConfig())
        config.poi_data_path = poi_path
        if config.map_projection == "linear" and len(config.map_bounds) != 4:
            metadata = load_map_metadata_from_poi_data(poi_path)
            bounds = metadata.get("bounds") if isinstance(metadata, dict) else None
            if isinstance(bounds, list) and len(bounds) == 4:
                config.map_bounds = [float(value) for value in bounds]
        if config.map_projection == "linear" and len(config.map_bounds) != 4:
            return None
        return config

    def _build_overlay_from_config(self, config: AppConfig) -> PoiOverlay | None:
        if not config.poi_data_path:
            return None
        if config.map_projection == "linear" and len(config.map_bounds) != 4:
            return None
        map_bounds = (
            tuple(float(value) for value in config.map_bounds)
            if len(config.map_bounds) == 4
            else (0.0, 0.0, 1.0, 1.0)
        )
        return PoiOverlay(
            pois_path=config.poi_data_path,
            categories_path=guess_poi_categories_path(config.poi_data_path, config.poi_categories_path) or None,
            map_bounds=map_bounds,
            icon_dir=config.poi_icon_dir or None,
            projection_type=config.map_projection,
            tile_zoom=config.map_tile_zoom,
            tile_x_range=tuple(config.map_tile_x_range) if len(config.map_tile_x_range) == 2 else None,
            tile_y_range=tuple(config.map_tile_y_range) if len(config.map_tile_y_range) == 2 else None,
            tile_size=config.map_tile_size,
            pixel_scale=config.poi_pixel_scale,
            pixel_offset_x=config.poi_pixel_offset_x,
            pixel_offset_y=config.poi_pixel_offset_y,
        )

    def _overlay_signature_for_config(self, config: AppConfig) -> tuple:
        return (
            config.poi_data_path,
            config.poi_categories_path,
            config.poi_icon_dir,
            config.map_projection,
            tuple(config.map_bounds),
            config.map_tile_zoom,
            tuple(config.map_tile_x_range),
            tuple(config.map_tile_y_range),
            config.map_tile_size,
            config.poi_pixel_scale,
            config.poi_pixel_offset_x,
            config.poi_pixel_offset_y,
        )

    def _ensure_display_overlay(self, config: AppConfig) -> PoiOverlay | None:
        if not config.poi_data_path:
            self._poi_overlay = None
            self._poi_overlay_signature = None
            return None

        signature = self._overlay_signature_for_config(config)
        if self._poi_overlay is None or self._poi_overlay_signature != signature:
            self._poi_overlay = self._build_overlay_from_config(config)
            self._poi_overlay_signature = signature
        return self._poi_overlay

    def _schedule_map_refresh(self, center_on_result: bool = False, invalidate_base: bool = False) -> None:
        if invalidate_base:
            self._map_base_key = None
        self._map_refresh_center = self._map_refresh_center or center_on_result
        if self._map_refresh_pending:
            return
        self._map_refresh_pending = True
        self.root.after_idle(self._flush_map_refresh)

    def _flush_map_refresh(self) -> None:
        self._map_refresh_pending = False
        center_on_result = self._map_refresh_center
        self._map_refresh_center = False
        self._refresh_map_display(center_on_result=center_on_result)

    def _refresh_map_display(self, center_on_result: bool = False) -> None:
        if self.map_canvas is None:
            return

        config = self._build_display_config()
        if config is None:
            self._show_map_message("请选择完整地图图像")
            return

        display_map_path = self._current_display_map_path(config)
        try:
            source_map = self._ensure_map_source_loaded(display_map_path)
        except Exception as exc:
            self._show_map_message(str(exc))
            return

        canvas_width = max(1, self.map_canvas.winfo_width())
        canvas_height = max(1, self.map_canvas.winfo_height())
        if canvas_width <= 8 or canvas_height <= 8:
            self.root.after(50, lambda: self._schedule_map_refresh(center_on_result=center_on_result))
            return

        scale = self._compute_map_scale(
            map_width=source_map.shape[1],
            map_height=source_map.shape[0],
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        display_width = max(1, int(round(source_map.shape[1] * scale)))
        display_height = max(1, int(round(source_map.shape[0] * scale)))
        selected_category_ids = tuple(self._get_selected_category_ids())
        display_result = self._result_for_display(
            source_map_path=config.map_path,
            display_map_path=display_map_path,
            result=self._last_result,
        )
        view_origin = self._resolve_view_origin(
            display_width=display_width,
            display_height=display_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            center_result=display_result if center_on_result else None,
        )

        base_key = (
            display_map_path,
            self.show_poi_overlay_var.get(),
            selected_category_ids,
            display_width,
            display_height,
            canvas_width,
            canvas_height,
            round(scale, 6),
            view_origin,
            self._resource_route_cache_token(),
        )

        if self._map_base_key != base_key:
            base_view = self._build_map_viewport(
                source_map=source_map,
                scale=scale,
                view_origin=view_origin,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )

            summary_text = f"地图: {source_map.shape[1]}x{source_map.shape[0]} | 显示: {display_width}x{display_height}"
            overlay = self._ensure_display_overlay(config)
            if overlay is not None:
                overlay.render_scale_x = scale
                overlay.render_scale_y = scale
                if self.show_poi_overlay_var.get():
                    base_view, summary = overlay.render_viewport(
                        base_view,
                        PoiRenderOptions(
                            enabled=True,
                            selected_category_ids=list(selected_category_ids),
                            keyword="",
                            show_labels=False,
                            max_points=config.poi_max_draw,
                            label_limit=config.poi_label_limit,
                        ),
                        viewport_origin=view_origin,
                        full_image_size=(display_width, display_height),
                        focus_xy=None,
                    )
                    if summary is not None:
                        summary_text = f"{summary_text} | {summary.text()}"

            base_view = render_resource_route_plan_viewport(
                base_view,
                self._resource_route_plan,
                scale=scale,
                viewport_origin=view_origin,
            )

            self._map_base_view = base_view
            self._map_base_key = base_key
            self._map_view_origin = view_origin
            self._map_scale = scale
            if self._last_result is None:
                self.map_hint_var.set(summary_text)

        if self._map_base_view is None:
            self._show_map_message("地图渲染失败")
            return

        display_image = self._map_base_view.copy()
        self._draw_result_on_map(display_image, display_result, viewport_origin=view_origin)
        self._set_map_canvas_image(
            display_image,
            viewport_origin=view_origin,
            image_width=display_width,
            image_height=display_height,
        )

    def _ensure_map_source_loaded(self, map_path: str) -> np.ndarray:
        if self._map_source_image is None or self._map_source_path != map_path:
            self._map_source_image = load_image(map_path, grayscale=False)
            self._map_source_path = map_path
            self._map_base_key = None
        return self._map_source_image

    def _compute_map_scale(
        self,
        map_width: int,
        map_height: int,
        canvas_width: int,
        canvas_height: int,
    ) -> float:
        fit_scale = min(canvas_width / max(map_width, 1), canvas_height / max(map_height, 1))
        fit_scale = max(fit_scale, 0.01)
        zoom = max(0.5, min(8.0, float(self.map_zoom_var.get())))
        return fit_scale * zoom

    def _draw_result_on_map(
        self,
        image: np.ndarray,
        result: LocalizationResult | None,
        viewport_origin: tuple[int, int] = (0, 0),
    ) -> None:
        if result is None:
            return

        viewport_x, viewport_y = viewport_origin
        if result.corners:
            points = np.array(
                [
                    (
                        int(round(x * self._map_scale)) - viewport_x,
                        int(round(y * self._map_scale)) - viewport_y,
                    )
                    for x, y in result.corners
                ],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(image, [points], isClosed=True, color=(0, 200, 255), thickness=2)

        if result.x == result.x and result.y == result.y:
            center = (
                int(round(result.x * self._map_scale)) - viewport_x,
                int(round(result.y * self._map_scale)) - viewport_y,
            )
            cv2.circle(image, center, 7, (0, 0, 255), -1, cv2.LINE_AA)

    def _set_map_canvas_image(
        self,
        image: np.ndarray,
        viewport_origin: tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> None:
        if self.map_canvas is None:
            return

        encoded = self._encode_png_base64(image)
        self._map_photo = tk.PhotoImage(data=encoded)
        if self._map_canvas_image_id is None:
            self.map_canvas.delete("all")
            self._map_canvas_image_id = self.map_canvas.create_image(
                viewport_origin[0],
                viewport_origin[1],
                anchor=tk.NW,
                image=self._map_photo,
            )
        else:
            self.map_canvas.itemconfigure(self._map_canvas_image_id, image=self._map_photo)
            self.map_canvas.coords(self._map_canvas_image_id, viewport_origin[0], viewport_origin[1])
        self.map_canvas.configure(scrollregion=(0, 0, image_width, image_height))
        self._move_canvas_to_origin(viewport_origin, image_width=image_width, image_height=image_height)

    def _move_canvas_to_origin(
        self,
        viewport_origin: tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> None:
        left, top = viewport_origin
        canvas_width = max(1, self.map_canvas.winfo_width())
        canvas_height = max(1, self.map_canvas.winfo_height())
        if image_width <= canvas_width:
            self.map_canvas.xview_moveto(0.0)
        else:
            self.map_canvas.xview_moveto(left / max(image_width, 1))

        if image_height <= canvas_height:
            self.map_canvas.yview_moveto(0.0)
        else:
            self.map_canvas.yview_moveto(top / max(image_height, 1))

    def _resolve_view_origin(
        self,
        display_width: int,
        display_height: int,
        canvas_width: int,
        canvas_height: int,
        center_result: LocalizationResult | None = None,
    ) -> tuple[int, int]:
        max_left = max(0, display_width - canvas_width)
        max_top = max(0, display_height - canvas_height)

        if (
            center_result is not None
            and center_result.x == center_result.x
            and center_result.y == center_result.y
        ):
            left = int(round(center_result.x * self._map_scale - canvas_width / 2.0))
            top = int(round(center_result.y * self._map_scale - canvas_height / 2.0))
        else:
            x_fraction, y_fraction = self._current_canvas_view_fractions()
            left = int(round(x_fraction * max_left))
            top = int(round(y_fraction * max_top))

        left = max(0, min(left, max_left))
        top = max(0, min(top, max_top))
        return left, top

    def _current_canvas_view_fractions(self) -> tuple[float, float]:
        if self.map_canvas is None or self._map_canvas_image_id is None:
            return 0.0, 0.0
        return self.map_canvas.xview()[0], self.map_canvas.yview()[0]

    def _build_map_viewport(
        self,
        source_map: np.ndarray,
        scale: float,
        view_origin: tuple[int, int],
        canvas_width: int,
        canvas_height: int,
    ) -> np.ndarray:
        view_left, view_top = view_origin
        source_height, source_width = source_map.shape[:2]
        if scale <= 0.0:
            return np.zeros((canvas_height, canvas_width, 3), dtype=source_map.dtype)

        src_x0 = max(0, min(source_width - 1, int(math.floor(view_left / scale))))
        src_y0 = max(0, min(source_height - 1, int(math.floor(view_top / scale))))
        src_x1 = min(source_width, max(src_x0 + 1, int(math.ceil((view_left + canvas_width) / scale)) + 1))
        src_y1 = min(source_height, max(src_y0 + 1, int(math.ceil((view_top + canvas_height) / scale)) + 1))

        crop = source_map[src_y0:src_y1, src_x0:src_x1]
        scaled_crop_width = max(1, int(math.ceil((src_x1 - src_x0) * scale)))
        scaled_crop_height = max(1, int(math.ceil((src_y1 - src_y0) * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled_crop = cv2.resize(crop, (scaled_crop_width, scaled_crop_height), interpolation=interpolation)

        crop_origin_x = int(round(src_x0 * scale))
        crop_origin_y = int(round(src_y0 * scale))
        offset_x = max(0, view_left - crop_origin_x)
        offset_y = max(0, view_top - crop_origin_y)
        viewport = scaled_crop[offset_y : offset_y + canvas_height, offset_x : offset_x + canvas_width]

        if viewport.shape[0] == canvas_height and viewport.shape[1] == canvas_width:
            return viewport

        padded = np.zeros((canvas_height, canvas_width, source_map.shape[2]), dtype=source_map.dtype)
        padded[: viewport.shape[0], : viewport.shape[1]] = viewport
        return padded

    def _resource_route_cache_token(self) -> tuple:
        if self._resource_route_plan is None:
            return ()
        return (
            tuple(self._resource_route_plan.selected_category_ids),
            self._resource_route_plan.total_points,
            round(self._resource_route_plan.total_distance, 3),
        )

    def _on_canvas_xview(self, *args) -> None:
        self.map_canvas.xview(*args)
        if not self._suppress_canvas_view_refresh:
            self._schedule_map_refresh(invalidate_base=True)

    def _on_canvas_yview(self, *args) -> None:
        self.map_canvas.yview(*args)
        if not self._suppress_canvas_view_refresh:
            self._schedule_map_refresh(invalidate_base=True)

    def _show_map_message(self, text: str) -> None:
        if self.map_canvas is None:
            return
        self.map_canvas.delete("all")
        self._map_canvas_image_id = None
        self._map_photo = None
        canvas_width = max(1, self.map_canvas.winfo_width())
        canvas_height = max(1, self.map_canvas.winfo_height())
        self.map_canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=text,
            fill="white",
            font=("Arial", 14),
        )
        self.map_canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        self.map_hint_var.set(text)

    def _step_map_zoom(self, factor: float) -> None:
        current = float(self.map_zoom_var.get())
        self.map_zoom_var.set(max(0.5, min(8.0, current * factor)))
        self._schedule_map_refresh(invalidate_base=True)

    def _reset_map_zoom(self) -> None:
        self.map_zoom_var.set(1.0)
        self._schedule_map_refresh(invalidate_base=True)

    def _on_map_zoom_changed(self, _value=None) -> None:
        self._schedule_map_refresh(invalidate_base=True)

    def _on_map_canvas_configure(self, _event) -> None:
        self._schedule_map_refresh(invalidate_base=True)

    def _on_map_mousewheel(self, event) -> str:
        if event.delta > 0 or getattr(event, "num", None) == 4:
            self._step_map_zoom(1.15)
        else:
            self._step_map_zoom(1 / 1.15)
        return "break"

    @staticmethod
    def _encode_png_base64(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("地图图像编码失败。")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _generate_resource_route(self) -> None:
        try:
            source_context = self._ensure_biliwiki_resource_context(force_refresh=False)
        except Exception as exc:
            messagebox.showerror("资源路线", str(exc))
            return

        overlay = self._build_overlay_from_config(source_context.config)
        if overlay is None:
            messagebox.showerror("资源路线", "当前资源数据源不可用于生成路线。")
            return

        display_map_path = source_context.config.display_map_path or source_context.config.map_path
        try:
            source_map = self._ensure_map_source_loaded(display_map_path)
        except Exception as exc:
            messagebox.showerror("资源路线", str(exc))
            return

        selected_category_ids = sorted({int(category_id) for category_id in self._get_selected_category_ids()})
        if not selected_category_ids:
            messagebox.showerror("资源路线", "请先选择需要生成路线的采集素材。")
            return

        selection_label = summarize_selection_label(selected_category_ids)
        source_path = Path(overlay.pois_path)
        source_mtime_ns = source_path.stat().st_mtime_ns if source_path.exists() else 0
        signature = build_route_cache_signature(
            selected_category_ids=selected_category_ids,
            source_path=str(source_path),
            source_mtime_ns=source_mtime_ns,
            map_path=display_map_path,
            map_width=source_map.shape[1],
            map_height=source_map.shape[0],
            projection_type=source_context.config.map_projection,
            tile_zoom=source_context.config.map_tile_zoom,
            tile_x_range=tuple(source_context.config.map_tile_x_range) if len(source_context.config.map_tile_x_range) == 2 else None,
            tile_y_range=tuple(source_context.config.map_tile_y_range) if len(source_context.config.map_tile_y_range) == 2 else None,
            tile_size=source_context.config.map_tile_size,
            pixel_scale=source_context.config.poi_pixel_scale,
            pixel_offset_x=source_context.config.poi_pixel_offset_x,
            pixel_offset_y=source_context.config.poi_pixel_offset_y,
        )
        cache_path = route_cache_path(self._resource_route_cache_dir, signature)

        route_plan = None
        if cache_path.exists():
            reuse = messagebox.askyesnocancel(
                "资源路线",
                "检测到已缓存的路线。\n选择“是”复用缓存，选择“否”重新生成。",
            )
            if reuse is None:
                return
            if reuse:
                route_plan = load_route_plan_cache(cache_path, signature)

        if route_plan is None:
            start_xy = None
            current_display_config = self._build_display_config()
            if current_display_config is not None:
                display_result = self._result_for_display(
                    source_map_path=current_display_config.map_path,
                    display_map_path=display_map_path,
                    result=self._last_result,
                )
                if (
                    display_result is not None
                    and display_result.x == display_result.x
                    and display_result.y == display_result.y
                ):
                    start_xy = (float(display_result.x), float(display_result.y))

            try:
                route_plan = build_resource_route_plan(
                    records=overlay.records,
                    categories=overlay.categories,
                    overlay=overlay,
                    map_width=source_map.shape[1],
                    map_height=source_map.shape[0],
                    selected_category_ids=selected_category_ids,
                    selection_label=selection_label,
                    start_xy=start_xy,
                    source_label=source_context.source_label,
                )
            except Exception as exc:
                messagebox.showerror("资源路线", str(exc))
                return
            save_route_plan_cache(cache_path, signature, route_plan)

        self._resource_route_plan = route_plan
        if not self.show_poi_overlay_var.get():
            self.show_poi_overlay_var.set(True)
        suffixes = []
        if route_plan.cached:
            suffixes.append("已复用缓存")
        if source_context.auto_downloaded:
            suffixes.append("已自动下载biliwiki资源")
        if source_context.points_refreshed:
            suffixes.append("已同步最新 wiki 点位")
        suffix_text = f" | {' | '.join(suffixes)}" if suffixes else ""
        self.resource_route_summary_var.set(route_plan.text() + suffix_text)
        self.map_hint_var.set(self.resource_route_summary_var.get())
        self._schedule_map_refresh(invalidate_base=True)

    def _clear_resource_route(self, silent: bool = False) -> None:
        self._resource_route_plan = None
        if not silent:
            self.map_hint_var.set("已清除资源路线")
        self.resource_route_summary_var.set("未生成资源路线")
        self._schedule_map_refresh(invalidate_base=True)

    def _result_for_display(
        self,
        source_map_path: str,
        display_map_path: str,
        result: LocalizationResult | None,
    ) -> LocalizationResult | None:
        if result is None:
            return None

        normalized_source = str(Path(source_map_path).expanduser()) if source_map_path else ""
        normalized_target = str(Path(display_map_path).expanduser()) if display_map_path else ""
        if not normalized_source or not normalized_target:
            return None
        if normalized_source == normalized_target:
            return result

        alignment = self._resolve_map_alignment(normalized_source, normalized_target)
        if alignment is None:
            return None
        return alignment.project_result(result)

    def _resolve_map_alignment(self, source_map_path: str, target_map_path: str) -> MapAlignment | None:
        cache_key = (source_map_path, target_map_path)
        if cache_key not in self._map_alignment_cache:
            self._map_alignment_cache[cache_key] = resolve_map_alignment(source_map_path, target_map_path)
        return self._map_alignment_cache[cache_key]

    def _current_display_map_path(self, config: AppConfig) -> str:
        return config.display_map_path or config.map_path


def launch_gui() -> None:
    app = LocalizationGUI()
    app.run()
