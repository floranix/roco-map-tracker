from __future__ import annotations

import base64
import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from src.async_frame_pipeline import LatestFrameAsyncPipeline
from src.map_alignment import MapAlignment, resolve_map_alignment
from src.pipeline import LocalizationPipeline
from src.poi_overlay import PoiOverlay, PoiRenderOptions
from src.resource_routes import (
    RESOURCE_ROUTE_MODE_LABELS,
    RESOURCE_ROUTE_MODE_NONE,
    RESOURCE_ROUTE_MODE_ORE,
    RESOURCE_ROUTE_MODE_ORE_AND_PLANT,
    RESOURCE_ROUTE_MODE_PLANT,
    ResourceRoutePlan,
    build_resource_route_plan,
    build_route_cache_signature,
    infer_resource_kind_from_texts,
    load_route_plan_cache,
    render_resource_route_plan,
    resource_route_mode_label,
    route_cache_path,
    save_route_plan_cache,
)
from src.resource_sources import (
    RESOURCE_SOURCE_17173,
    RESOURCE_SOURCE_BILIWIKI,
    RESOURCE_SOURCE_DEFAULT,
    RESOURCE_SOURCE_LABELS,
    ResourceSourceContext,
    build_resource_source_context,
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
        self.root.geometry("1380x900")
        self.root.minsize(1180, 780)

        default_config_path = "configs/rocom_17173.yaml" if Path("configs/rocom_17173.yaml").exists() else "configs/default.yaml"
        self.config_path_var = tk.StringVar(value=default_config_path)
        self.map_path_var = tk.StringVar(value="")
        self.poi_data_path_var = tk.StringVar(value="")
        self.input_mode_var = tk.StringVar(value="screen_region")
        self.input_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="outputs")
        self.save_visualizations_var = tk.BooleanVar(value=True)
        self.show_poi_overlay_var = tk.BooleanVar(value=False)
        self.show_poi_labels_var = tk.BooleanVar(value=False)
        self.poi_keyword_var = tk.StringVar(value="")
        self.poi_summary_var = tk.StringVar(value="未加载点位数据")
        self.status_var = tk.StringVar(value="就绪")
        self.capture_interval_var = tk.StringVar(value="250")
        self.recognition_enabled_var = tk.BooleanVar(value=False)
        self.map_zoom_var = tk.DoubleVar(value=1.0)
        self.map_hint_var = tk.StringVar(value="等待加载地图")
        self.resource_route_mode_var = tk.StringVar(value=RESOURCE_ROUTE_MODE_LABELS[RESOURCE_ROUTE_MODE_NONE])
        self.resource_route_source_var = tk.StringVar(value=RESOURCE_SOURCE_LABELS[RESOURCE_SOURCE_DEFAULT])
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
        self._map_scale: float = 1.0
        self._map_refresh_pending = False
        self._map_refresh_center = False
        self._last_result: LocalizationResult | None = None
        self._resource_route_plan: ResourceRoutePlan | None = None
        self._resource_route_cache_dir = Path("outputs/route_cache")
        self._resource_route_source_context: ResourceSourceContext | None = None
        self._resource_route_display_map_path = ""

        self._build_layout()
        self._bind_refresh_events()
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
        input_row.columnconfigure(4, weight=1)

        ttk.Radiobutton(input_row, text="单张截图", value="frame", variable=self.input_mode_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Radiobutton(input_row, text="截图文件夹", value="frames_dir", variable=self.input_mode_var).grid(row=0, column=1, padx=(0, 8))
        ttk.Radiobutton(input_row, text="视频", value="video", variable=self.input_mode_var).grid(row=0, column=2, padx=(0, 8))
        ttk.Radiobutton(input_row, text="屏幕区域", value="screen_region", variable=self.input_mode_var).grid(row=0, column=3, padx=(0, 8))
        ttk.Entry(input_row, textvariable=self.input_path_var).grid(row=0, column=4, sticky="ew", padx=(0, 8))
        ttk.Button(input_row, text="选择", command=self._select_input).grid(row=0, column=5)
        ttk.Label(input_row, text="间隔(ms)").grid(row=0, column=6, padx=(8, 4))
        ttk.Entry(input_row, width=8, textvariable=self.capture_interval_var).grid(row=0, column=7)

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
        filter_row.columnconfigure(6, weight=1)
        ttk.Checkbutton(
            filter_row,
            text="显示点位",
            variable=self.show_poi_overlay_var,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(
            filter_row,
            text="显示名称",
            variable=self.show_poi_labels_var,
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(filter_row, text="关键词").grid(row=0, column=2, sticky="w")
        keyword_entry = ttk.Entry(filter_row, textvariable=self.poi_keyword_var)
        keyword_entry.grid(row=0, column=3, sticky="ew", padx=(8, 8))
        ttk.Label(filter_row, text="点位图层与识别解耦，未识别时也可独立查看").grid(row=0, column=4, sticky="e")
        keyword_entry.bind("<Return>", lambda _event: self._schedule_map_refresh(invalidate_base=True))
        ttk.Label(filter_row, text="资源来源").grid(row=1, column=0, sticky="w", pady=(8, 0))
        resource_source_combo = ttk.Combobox(
            filter_row,
            textvariable=self.resource_route_source_var,
            state="readonly",
            values=[
                RESOURCE_SOURCE_LABELS[RESOURCE_SOURCE_BILIWIKI],
                RESOURCE_SOURCE_LABELS[RESOURCE_SOURCE_17173],
            ],
            width=10,
        )
        resource_source_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        resource_source_combo.bind("<<ComboboxSelected>>", self._on_resource_route_source_changed)
        ttk.Label(filter_row, text="资源路线").grid(row=1, column=2, sticky="w", pady=(8, 0))
        resource_route_combo = ttk.Combobox(
            filter_row,
            textvariable=self.resource_route_mode_var,
            state="readonly",
            values=[
                RESOURCE_ROUTE_MODE_LABELS[RESOURCE_ROUTE_MODE_NONE],
                RESOURCE_ROUTE_MODE_LABELS[RESOURCE_ROUTE_MODE_ORE],
                RESOURCE_ROUTE_MODE_LABELS[RESOURCE_ROUTE_MODE_PLANT],
                RESOURCE_ROUTE_MODE_LABELS[RESOURCE_ROUTE_MODE_ORE_AND_PLANT],
            ],
            width=14,
        )
        resource_route_combo.grid(row=1, column=3, sticky="w", pady=(8, 0))
        resource_route_combo.bind("<<ComboboxSelected>>", self._on_resource_route_mode_changed)
        ttk.Button(filter_row, text="生成路线", command=self._generate_resource_route).grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Button(filter_row, text="清除路线", command=self._clear_resource_route).grid(row=1, column=5, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(filter_row, textvariable=self.resource_route_summary_var).grid(row=1, column=6, sticky="e", pady=(8, 0))

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
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.map_canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.map_canvas.xview)
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
        self.show_poi_labels_var.trace_add("write", lambda *_args: self._schedule_map_refresh(invalidate_base=True))
        self.poi_keyword_var.trace_add("write", lambda *_args: self._schedule_map_refresh(invalidate_base=True))

    def _load_config_defaults(self) -> None:
        config_path = Path(self.config_path_var.get()).expanduser()
        if not config_path.exists():
            self.status_var.set("未找到配置文件，将使用当前输入")
            self._schedule_map_refresh(invalidate_base=True)
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

    def _select_poi_data(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择点位数据",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.poi_data_path_var.set(selected)
            self._clear_resource_route(silent=True)
            self._load_poi_catalog()

    def _select_input(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "frame":
            selected = filedialog.askopenfilename(
                title="选择局部截图",
                filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
            )
        elif mode == "frames_dir":
            selected = filedialog.askdirectory(title="选择截图文件夹")
        elif mode == "screen_region":
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

    def _toggle_localization(self) -> None:
        if self.recognition_enabled_var.get():
            self._start_localization()
        else:
            self._stop_localization()

    def _on_resource_route_mode_changed(self, _event=None) -> None:
        if self._selected_resource_route_mode() == RESOURCE_ROUTE_MODE_NONE:
            self._clear_resource_route(silent=True)
        else:
            self.resource_route_summary_var.set(
                f"待生成: {resource_route_mode_label(self._selected_resource_route_mode())}"
            )

    def _on_resource_route_source_changed(self, _event=None) -> None:
        self._clear_resource_route(silent=True)
        self.resource_route_summary_var.set(
            f"资源来源: {self.resource_route_source_var.get().strip()} | 待生成路线"
        )

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
                self._refresh_map_display(center_on_result=self.recognition_enabled_var.get())
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

    def _build_runtime_config(self, require_input: bool = True) -> AppConfig:
        config_path = self.config_path_var.get().strip()
        config = load_config(config_path) if config_path and Path(config_path).exists() else AppConfig()

        map_path = self.map_path_var.get().strip()
        poi_data_path = self.poi_data_path_var.get().strip()
        input_path = self.input_path_var.get().strip()
        input_mode = self.input_mode_var.get().strip()

        if not map_path:
            raise ValueError("请选择完整地图图像。")
        if require_input and not input_path:
            raise ValueError("请选择输入源或屏幕捕捉区域。")

        config.map_path = map_path
        config.poi_data_path = poi_data_path
        config.poi_categories_path = guess_poi_categories_path(poi_data_path, config.poi_categories_path)
        config.output_dir = self.output_dir_var.get().strip() or config.output_dir
        config.save_visualizations = self.save_visualizations_var.get()
        config.show_poi_overlay = self.show_poi_overlay_var.get()
        config.show_poi_labels = self.show_poi_labels_var.get()
        config.poi_keyword = self.poi_keyword_var.get().strip()
        config.capture_interval_ms = max(0, int(self.capture_interval_var.get().strip() or "250"))
        if input_mode == "screen_region" and input_path:
            config.capture_region = list(parse_capture_region(input_path))
        return apply_map_metadata_defaults(config)

    def _build_display_config(self) -> AppConfig | None:
        try:
            return self._build_runtime_config(require_input=False)
        except Exception:
            return None

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

        if input_mode == "frames_dir":
            for image_path in list_image_files(input_path):
                yield image_path.stem, load_image(image_path)
            return

        if input_mode == "screen_region":
            yield from iterate_screen_region_frames(
                input_path,
                interval_ms=max(0, int(capture_interval_ms)),
                stop_event=self._stop_event,
            )
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
        self._poi_categories = self._poi_overlay.available_categories()
        self.category_listbox.delete(0, tk.END)
        for category, count in self._poi_categories:
            self.category_listbox.insert(tk.END, f"{category.group_title} / {category.title} ({count})")
        if previous_selected_ids:
            self._apply_category_selection(previous_selected_ids)
        elif self._poi_categories:
            self.category_listbox.select_set(0, tk.END)
        self.poi_summary_var.set(
            f"已载入 {len(self._poi_overlay.records)} 个点位，{len(self._poi_categories)} 个分类"
        )
        self._schedule_map_refresh(invalidate_base=True)

    def _select_all_categories(self) -> None:
        self.category_listbox.select_set(0, tk.END)
        self._schedule_map_refresh(invalidate_base=True)

    def _clear_category_selection(self) -> None:
        self.category_listbox.selection_clear(0, tk.END)
        self._schedule_map_refresh(invalidate_base=True)

    def _apply_category_selection(self, category_ids: list[int]) -> None:
        if not self._poi_categories:
            return

        selected = set(category_ids)
        self.category_listbox.selection_clear(0, tk.END)
        for index, (category, _count) in enumerate(self._poi_categories):
            if category.id in selected:
                self.category_listbox.selection_set(index)
        self._schedule_map_refresh(invalidate_base=True)

    def _get_selected_category_ids(self) -> list[int]:
        selected_ids = []
        for index in self.category_listbox.curselection():
            selected_ids.append(self._poi_categories[index][0].id)
        return selected_ids

    def _build_overlay_for_run(self, config: AppConfig) -> PoiOverlay | None:
        return self._build_overlay_from_config(config)

    def _build_overlay_config_for_catalog(self, poi_path: str) -> AppConfig | None:
        config_path = self.config_path_var.get().strip()
        config = AppConfig()
        if config_path and Path(config_path).exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = AppConfig()

        config.poi_data_path = poi_path
        config.poi_categories_path = guess_poi_categories_path(poi_path, config.poi_categories_path)
        config = apply_map_metadata_defaults(config)
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
        main_display_map_path = config.display_map_path or config.map_path
        use_main_overlay = display_map_path == main_display_map_path

        base_key = (
            display_map_path,
            self.show_poi_overlay_var.get(),
            self.show_poi_labels_var.get(),
            self.poi_keyword_var.get().strip(),
            selected_category_ids,
            display_width,
            display_height,
            round(scale, 6),
            bool(self._resource_route_plan),
        )

        if self._map_base_key != base_key:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            base_view = cv2.resize(source_map, (display_width, display_height), interpolation=interpolation)

            summary_text = f"地图: {source_map.shape[1]}x{source_map.shape[0]} | 显示: {display_width}x{display_height}"
            overlay = self._ensure_display_overlay(config) if use_main_overlay else None
            if overlay is not None:
                overlay.render_scale_x = scale
                overlay.render_scale_y = scale
                if self.show_poi_overlay_var.get():
                    base_view, summary = overlay.render_map(
                        base_view,
                        PoiRenderOptions(
                            enabled=True,
                            selected_category_ids=list(selected_category_ids),
                            keyword=self.poi_keyword_var.get().strip(),
                            show_labels=self.show_poi_labels_var.get(),
                            max_points=config.poi_max_draw,
                            label_limit=config.poi_label_limit,
                        ),
                        focus_xy=None,
                    )
                    if summary is not None:
                        summary_text = f"{summary_text} | {summary.text()}"

            self._map_base_view = base_view
            self._map_base_key = base_key
            self._map_scale = scale
            if self._last_result is None:
                self.map_hint_var.set(summary_text)

        if self._map_base_view is None:
            self._show_map_message("地图渲染失败")
            return

        display_result = self._result_for_display(
            source_map_path=config.map_path,
            display_map_path=display_map_path,
            result=self._last_result,
        )
        display_image = self._map_base_view.copy()
        display_image = render_resource_route_plan(display_image, self._resource_route_plan, scale=self._map_scale)
        self._draw_result_on_map(display_image, display_result)
        self._set_map_canvas_image(
            display_image,
            center_result=display_result if center_on_result else None,
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

    def _draw_result_on_map(self, image: np.ndarray, result: LocalizationResult | None) -> None:
        if result is None:
            return

        if result.corners:
            points = np.array(
                [
                    (int(round(x * self._map_scale)), int(round(y * self._map_scale)))
                    for x, y in result.corners
                ],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(image, [points], isClosed=True, color=(0, 200, 255), thickness=2)

        if result.x == result.x and result.y == result.y:
            center = (int(round(result.x * self._map_scale)), int(round(result.y * self._map_scale)))
            cv2.circle(image, center, 7, (0, 0, 255), -1, cv2.LINE_AA)

    def _set_map_canvas_image(
        self,
        image: np.ndarray,
        center_result: LocalizationResult | None = None,
    ) -> None:
        if self.map_canvas is None:
            return

        previous_x = self.map_canvas.xview()[0] if self._map_canvas_image_id is not None else 0.0
        previous_y = self.map_canvas.yview()[0] if self._map_canvas_image_id is not None else 0.0
        canvas_width = max(1, self.map_canvas.winfo_width())
        canvas_height = max(1, self.map_canvas.winfo_height())

        encoded = self._encode_png_base64(image)
        self._map_photo = tk.PhotoImage(data=encoded)
        if self._map_canvas_image_id is None:
            self.map_canvas.delete("all")
            self._map_canvas_image_id = self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self._map_photo)
        else:
            self.map_canvas.itemconfigure(self._map_canvas_image_id, image=self._map_photo)

        image_height, image_width = image.shape[:2]
        self.map_canvas.configure(scrollregion=(0, 0, image_width, image_height))

        if center_result is not None and center_result.x == center_result.x and center_result.y == center_result.y:
            self._center_canvas_on(
                x=float(center_result.x * self._map_scale),
                y=float(center_result.y * self._map_scale),
                image_width=image_width,
                image_height=image_height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
        else:
            self.map_canvas.xview_moveto(0.0 if image_width <= canvas_width else previous_x)
            self.map_canvas.yview_moveto(0.0 if image_height <= canvas_height else previous_y)

    def _center_canvas_on(
        self,
        x: float,
        y: float,
        image_width: int,
        image_height: int,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        if image_width <= canvas_width:
            self.map_canvas.xview_moveto(0.0)
        else:
            left = max(0.0, min(x - canvas_width / 2.0, image_width - canvas_width))
            self.map_canvas.xview_moveto(left / max(image_width, 1))

        if image_height <= canvas_height:
            self.map_canvas.yview_moveto(0.0)
        else:
            top = max(0.0, min(y - canvas_height / 2.0, image_height - canvas_height))
            self.map_canvas.yview_moveto(top / max(image_height, 1))

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
        mode = self._selected_resource_route_mode()
        if mode == RESOURCE_ROUTE_MODE_NONE:
            self._clear_resource_route()
            return

        try:
            source_context = build_resource_source_context(self._selected_resource_route_source_key())
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

        source_path = Path(overlay.pois_path)
        source_mtime_ns = source_path.stat().st_mtime_ns if source_path.exists() else 0
        signature = build_route_cache_signature(
            mode=mode,
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
                    mode=mode,
                    start_xy=start_xy,
                    source_label=source_context.source_label,
                )
            except Exception as exc:
                messagebox.showerror("资源路线", str(exc))
                return
            save_route_plan_cache(cache_path, signature, route_plan)

        self._resource_route_plan = route_plan
        self._resource_route_source_context = source_context
        self._resource_route_display_map_path = display_map_path
        if not self.show_poi_overlay_var.get():
            self.show_poi_overlay_var.set(True)
        if source_context.key == RESOURCE_SOURCE_17173:
            self._apply_resource_category_selection(mode)
        suffixes = []
        if route_plan.cached:
            suffixes.append("已复用缓存")
        if source_context.auto_downloaded:
            suffixes.append("已自动下载biliwiki资源")
        if source_context.key != RESOURCE_SOURCE_17173:
            suffixes.append("biliwiki 底图已接入识别坐标")
        suffix_text = f" | {' | '.join(suffixes)}" if suffixes else ""
        self.resource_route_summary_var.set(route_plan.text() + suffix_text)
        self.map_hint_var.set(self.resource_route_summary_var.get())
        self._schedule_map_refresh(invalidate_base=True)

    def _clear_resource_route(self, silent: bool = False) -> None:
        self._resource_route_plan = None
        self._resource_route_source_context = None
        self._resource_route_display_map_path = ""
        if not silent:
            self.map_hint_var.set("已清除资源路线")
        self.resource_route_summary_var.set("未生成资源路线")
        self._schedule_map_refresh(invalidate_base=True)

    def _selected_resource_route_mode(self) -> str:
        selected_label = self.resource_route_mode_var.get().strip()
        for mode, label in RESOURCE_ROUTE_MODE_LABELS.items():
            if label == selected_label:
                return mode
        return RESOURCE_ROUTE_MODE_NONE

    def _selected_resource_route_source_key(self) -> str:
        selected_label = self.resource_route_source_var.get().strip()
        for key, label in RESOURCE_SOURCE_LABELS.items():
            if label == selected_label:
                return key
        return RESOURCE_SOURCE_DEFAULT

    def _apply_resource_category_selection(self, mode: str) -> None:
        if not self._poi_categories:
            return

        matched_indices = []
        for index, (category, _count) in enumerate(self._poi_categories):
            resource_kind = infer_resource_kind_from_texts(
                title=category.title,
                category_title=category.title,
                resolved_category_title=category.title,
                group_title=category.group_title,
            )
            if mode == RESOURCE_ROUTE_MODE_ORE_AND_PLANT:
                matched = resource_kind in {RESOURCE_ROUTE_MODE_ORE, RESOURCE_ROUTE_MODE_PLANT}
            else:
                matched = resource_kind == mode
            if matched:
                matched_indices.append(index)

        if not matched_indices:
            return

        self.category_listbox.selection_clear(0, tk.END)
        for index in matched_indices:
            self.category_listbox.selection_set(index)

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
        if self._resource_route_display_map_path:
            return self._resource_route_display_map_path
        return config.display_map_path or config.map_path


def launch_gui() -> None:
    app = LocalizationGUI()
    app.run()
