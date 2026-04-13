from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


FrameItem = tuple[str, np.ndarray]


@dataclass
class AsyncFramePipelineStats:
    captured_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0


class LatestFrameAsyncPipeline:
    def __init__(
        self,
        max_pending_frames: int = 2,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.max_pending_frames = max(1, int(max_pending_frames))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))

    def run(
        self,
        frame_source: Iterable[FrameItem],
        frame_processor: Callable[[str, np.ndarray], None],
        stop_event: threading.Event,
    ) -> AsyncFramePipelineStats:
        pending_frames: queue.Queue[FrameItem] = queue.Queue(maxsize=self.max_pending_frames)
        stats = AsyncFramePipelineStats()
        stats_lock = threading.Lock()
        source_done = threading.Event()
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def record_error(exc: Exception) -> None:
            with errors_lock:
                if not errors:
                    errors.append(exc)
            stop_event.set()

        def capture_loop() -> None:
            try:
                for frame_name, frame in frame_source:
                    if stop_event.is_set():
                        break
                    self._put_latest_frame(pending_frames, frame_name, frame, stats, stats_lock)
            except Exception as exc:  # pragma: no cover - exercised via callers
                record_error(exc)
            finally:
                source_done.set()

        def process_loop() -> None:
            try:
                while True:
                    if stop_event.is_set():
                        self._discard_pending_frames(pending_frames, stats, stats_lock)
                        return

                    try:
                        frame_name, frame = pending_frames.get(timeout=self.poll_interval_seconds)
                    except queue.Empty:
                        if source_done.is_set():
                            return
                        continue

                    frame_processor(frame_name, frame)
                    with stats_lock:
                        stats.processed_frames += 1
            except Exception as exc:  # pragma: no cover - exercised via callers
                record_error(exc)

        capture_thread = threading.Thread(target=capture_loop, name="screen-capture", daemon=True)
        process_thread = threading.Thread(target=process_loop, name="screen-recognition", daemon=True)

        capture_thread.start()
        process_thread.start()

        while capture_thread.is_alive() or process_thread.is_alive():
            capture_thread.join(timeout=self.poll_interval_seconds)
            process_thread.join(timeout=self.poll_interval_seconds)
            if errors:
                stop_event.set()

        if errors:
            raise errors[0]
        return stats

    def _put_latest_frame(
        self,
        pending_frames: queue.Queue[FrameItem],
        frame_name: str,
        frame: np.ndarray,
        stats: AsyncFramePipelineStats,
        stats_lock: threading.Lock,
    ) -> None:
        dropped_frames = 0
        while True:
            try:
                pending_frames.put_nowait((frame_name, frame))
                with stats_lock:
                    stats.captured_frames += 1
                    stats.dropped_frames += dropped_frames
                return
            except queue.Full:
                try:
                    pending_frames.get_nowait()
                    dropped_frames += 1
                except queue.Empty:
                    continue

    @staticmethod
    def _discard_pending_frames(
        pending_frames: queue.Queue[FrameItem],
        stats: AsyncFramePipelineStats,
        stats_lock: threading.Lock,
    ) -> None:
        discarded = 0
        while True:
            try:
                pending_frames.get_nowait()
                discarded += 1
            except queue.Empty:
                break

        if discarded:
            with stats_lock:
                stats.dropped_frames += discarded
