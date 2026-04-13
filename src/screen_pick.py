from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np


MIN_CAPTURE_REGION = 32
PICK_OVERLAY_MAX_EDGE = 2400


def parse_capture_region(raw_value: str | list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(raw_value, tuple) and len(raw_value) == 4:
        values = [int(value) for value in raw_value]
    elif isinstance(raw_value, list) and len(raw_value) == 4:
        values = [int(value) for value in raw_value]
    else:
        text = str(raw_value or "").strip()
        if not text:
            raise ValueError("未提供屏幕区域。")
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 4:
            raise ValueError("屏幕区域格式应为 x,y,w,h。")
        values = [int(part) for part in parts]

    left, top, width, height = values
    if width < MIN_CAPTURE_REGION or height < MIN_CAPTURE_REGION:
        raise ValueError(f"屏幕区域至少需要 {MIN_CAPTURE_REGION}x{MIN_CAPTURE_REGION} 像素。")
    return left, top, width, height


def format_capture_region(region: tuple[int, int, int, int] | list[int]) -> str:
    left, top, width, height = parse_capture_region(region)
    return f"{left},{top},{width},{height}"


def iterate_screen_region_frames(
    region: tuple[int, int, int, int] | list[int] | str,
    interval_ms: int = 250,
    stop_event: Any = None,
) -> Iterator[tuple[str, np.ndarray]]:
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("屏幕区域采集需要安装 mss。") from exc

    left, top, width, height = parse_capture_region(region)
    monitor = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }
    interval_seconds = max(0.0, float(interval_ms) / 1000.0)
    next_capture_at = time.monotonic()
    frame_index = 0

    with mss.mss() as sct:
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            frame = _capture_monitor(sct, monitor)
            yield f"screen_{frame_index:06d}", frame
            frame_index += 1

            if interval_seconds <= 0:
                continue

            next_capture_at += interval_seconds
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                remaining = next_capture_at - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.05))


def pick_screen_region(
    parent,
    on_done: Callable[[tuple[int, int, int, int]], None],
    on_cancel: Callable[[], None] | None = None,
) -> None:
    try:
        import mss
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise RuntimeError("屏幕区域采集需要安装 mss。") from exc

    try:
        with mss.mss() as sct:
            monitor = dict(sct.monitors[0])
            screenshot = _capture_monitor(sct, monitor)
    except Exception as exc:  # pragma: no cover - depends on OS permissions
        raise RuntimeError(
            "屏幕抓取失败，请确认已授予当前 Python / Codex 屏幕录制权限。"
        ) from exc

    raw_height, raw_width = screenshot.shape[:2]
    screen_left = int(monitor.get("left", 0))
    screen_top = int(monitor.get("top", 0))
    scale = min(1.0, PICK_OVERLAY_MAX_EDGE / max(raw_width, raw_height, 1))
    display_width = max(1, int(round(raw_width * scale)))
    display_height = max(1, int(round(raw_height * scale)))
    if scale < 1.0:
        screenshot = cv2.resize(screenshot, (display_width, display_height), interpolation=cv2.INTER_AREA)

    overlay = tk.Toplevel(parent)
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)
    overlay.configure(cursor="crosshair", bg="#151515")
    overlay.geometry(f"{display_width}x{display_height}+{screen_left}+{screen_top}")

    canvas = tk.Canvas(
        overlay,
        width=display_width,
        height=display_height,
        highlightthickness=0,
        bg="#151515",
    )
    canvas.pack(fill=tk.BOTH, expand=True)

    photo = tk.PhotoImage(data=_encode_png_base64(screenshot))
    overlay._picker_photo_ref = photo  # type: ignore[attr-defined]
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.create_rectangle(0, 0, display_width, 40, fill="#242424", outline="", width=0)
    canvas.create_text(
        display_width // 2,
        20,
        text="拖拽框选需要持续采集的矩形区域，松开完成，Esc 取消",
        fill="white",
        font=("Arial", 12),
    )

    box: dict[str, float | None] = {"x0": None, "y0": None}
    rect_id: int | None = None

    def close(cancelled: bool) -> None:
        try:
            overlay.destroy()
        except tk.TclError:
            pass
        if cancelled and on_cancel is not None:
            parent.after(0, on_cancel)

    def press(event) -> None:
        box["x0"] = float(event.x)
        box["y0"] = float(event.y)
        nonlocal rect_id
        if rect_id is not None:
            canvas.delete(rect_id)
            rect_id = None

    def motion(event) -> None:
        if box["x0"] is None or box["y0"] is None:
            return
        left = min(box["x0"], float(event.x))
        top = min(box["y0"], float(event.y))
        right = max(box["x0"], float(event.x))
        bottom = max(box["y0"], float(event.y))
        nonlocal rect_id
        if rect_id is None:
            rect_id = canvas.create_rectangle(left, top, right, bottom, outline="#ff4d4f", width=3)
        else:
            canvas.coords(rect_id, left, top, right, bottom)

    def release(event) -> None:
        if box["x0"] is None or box["y0"] is None:
            close(cancelled=True)
            return

        left = max(0.0, min(box["x0"], float(event.x)))
        top = max(0.0, min(box["y0"], float(event.y)))
        right = min(float(display_width), max(box["x0"], float(event.x)))
        bottom = min(float(display_height), max(box["y0"], float(event.y)))

        width = max(1, int(round((right - left) * raw_width / display_width)))
        height = max(1, int(round((bottom - top) * raw_height / display_height)))
        region = (
            screen_left + int(round(left * raw_width / display_width)),
            screen_top + int(round(top * raw_height / display_height)),
            width,
            height,
        )

        close(cancelled=False)
        try:
            on_done(parse_capture_region(region))
        except ValueError as exc:
            messagebox.showwarning("区域无效", str(exc))
            if on_cancel is not None:
                parent.after(0, on_cancel)

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", motion)
    canvas.bind("<ButtonRelease-1>", release)
    overlay.bind("<Escape>", lambda _event: close(cancelled=True))

    overlay.update_idletasks()
    overlay.deiconify()
    overlay.lift()
    try:
        overlay.focus_force()
    except tk.TclError:
        pass


def _capture_monitor(sct, monitor: dict[str, int]) -> np.ndarray:
    shot = np.array(sct.grab(monitor))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def _encode_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("屏幕预览编码失败。")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
