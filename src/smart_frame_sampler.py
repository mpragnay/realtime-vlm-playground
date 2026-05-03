"""
Smart frame sampling for egocentric procedure videos.

The sampler assumes the existing harness delivers frames at a fixed low rate
such as 2 FPS. For each 5-second window with 10 candidate frames it keeps:

  - first frame
  - middle anchor frame (index 4 for a 10-frame window)
  - last frame
  - one sharp transition frame between first and middle
  - one sharp transition frame between middle and last

It is intentionally model-free and can be used from the streaming harness via
SmartFrameWindowBuffer.on_frame.

Example:
    .venv/bin/python src/smart_frame_sampler.py \\
        --video data/videos_full/R142-31Aug-RAM/Export_py/Video_pitchshift.mp4 \\
        --start-sec 55 \\
        --end-sec 100 \\
        --target-windows 55-60,80-85,90-95
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SelectedFrame:
    index: int
    timestamp_sec: float
    reason: str
    score: float
    sharpness: float
    distance_to_left_anchor: Optional[float] = None
    distance_to_right_anchor: Optional[float] = None


def resize_gray(frame: np.ndarray, size: Tuple[int, int] = (160, 90)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_abs_distance(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(gray_a, gray_b)) / 255.0)


def ssim_distance(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Small local SSIM implementation to avoid adding a dependency."""
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(a, (7, 7), 1.5)
    mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    ssim = float(np.clip(np.mean(ssim_map), -1.0, 1.0))
    return float(1.0 - ssim)


def frame_distance(gray_a: np.ndarray, gray_b: np.ndarray, metric: str) -> float:
    if metric == "ssim":
        return ssim_distance(gray_a, gray_b)
    if metric == "mad":
        return mean_abs_distance(gray_a, gray_b)
    raise ValueError(f"Unknown metric: {metric}")


def smart_select_frames(
    *,
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    metric: str = "ssim",
) -> List[SelectedFrame]:
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have the same length")
    if len(frames) < 3:
        raise ValueError("Need at least 3 frames for smart selection")

    last_index = len(frames) - 1
    middle_index = 4 if len(frames) >= 10 else len(frames) // 2
    anchor_indices = sorted({0, middle_index, last_index})

    grays = [resize_gray(frame) for frame in frames]
    sharpness = [laplacian_sharpness(gray) for gray in grays]
    max_sharpness = max(sharpness) if sharpness else 1.0
    if max_sharpness <= 0:
        max_sharpness = 1.0

    selected: Dict[int, SelectedFrame] = {}
    for anchor_index in anchor_indices:
        selected[anchor_index] = SelectedFrame(
            index=anchor_index,
            timestamp_sec=float(timestamps[anchor_index]),
            reason="anchor",
            score=0.0,
            sharpness=sharpness[anchor_index],
        )

    def choose_transition(
        candidate_indices: Iterable[int],
        left_anchor: int,
        right_anchor: int,
        reason: str,
    ) -> None:
        best: Optional[SelectedFrame] = None
        for index in candidate_indices:
            if index < 0 or index >= len(frames) or index in selected:
                continue
            distance_left = frame_distance(grays[index], grays[left_anchor], metric)
            distance_right = frame_distance(grays[index], grays[right_anchor], metric)
            distance_score = distance_left + distance_right
            sharpness_factor = 0.5 + 0.5 * min(sharpness[index] / max_sharpness, 1.0)
            score = distance_score * sharpness_factor
            candidate = SelectedFrame(
                index=index,
                timestamp_sec=float(timestamps[index]),
                reason=reason,
                score=score,
                sharpness=sharpness[index],
                distance_to_left_anchor=distance_left,
                distance_to_right_anchor=distance_right,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        if best:
            selected[best.index] = best

    choose_transition(range(1, middle_index), 0, middle_index, "transition_between_start_and_middle")
    choose_transition(range(middle_index + 1, last_index), middle_index, last_index, "transition_between_middle_and_end")

    return [selected[index] for index in sorted(selected)]


class SmartFrameWindowBuffer:
    """
    Harness-facing adapter for selecting frames from streaming callbacks.

    Register buffer.on_frame with StreamingHarness.on_frame. When a complete
    window is available, the optional on_window callback receives:
      (selected_frames, selected_timestamps, selection_log)
    """

    def __init__(
        self,
        *,
        window_sec: float = 5.0,
        input_frames_per_window: int = 10,
        metric: str = "ssim",
        on_window: Optional[Callable[[List[np.ndarray], List[float], Dict[str, Any]], None]] = None,
    ) -> None:
        self.window_sec = window_sec
        self.input_frames_per_window = input_frames_per_window
        self.metric = metric
        self.on_window = on_window
        self._frames: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._window_start: Optional[float] = None

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: Optional[str] = None) -> None:
        del frame_base64
        if self._window_start is None:
            self._window_start = timestamp_sec
        self._frames.append(frame)
        self._timestamps.append(float(timestamp_sec))
        if len(self._frames) >= self.input_frames_per_window:
            self.flush()

    def flush(self) -> Optional[Dict[str, Any]]:
        if not self._frames:
            return None
        selected = smart_select_frames(
            frames=self._frames,
            timestamps=self._timestamps,
            metric=self.metric,
        )
        selected_frames = [self._frames[item.index] for item in selected]
        selected_timestamps = [item.timestamp_sec for item in selected]
        log = {
            "frame_window": [self._timestamps[0], self._timestamps[-1]],
            "metric": self.metric,
            "input_frame_count": len(self._frames),
            "selected_frames": [item.__dict__ for item in selected],
        }
        if self.on_window:
            self.on_window(selected_frames, selected_timestamps, log)
        self._frames = []
        self._timestamps = []
        self._window_start = None
        return log


def read_frames_at_timestamps(
    cap: Any,
    timestamps: Sequence[float],
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame at {timestamp:.3f}s")
        frames.append(frame)
    return frames


def window_timestamps(start_sec: float, frame_fps: float = 2.0, frames_per_window: int = 10) -> List[float]:
    interval = 1.0 / frame_fps
    return [round(start_sec + index * interval, 3) for index in range(frames_per_window)]


def parse_target_windows(value: Optional[str]) -> List[Tuple[float, float]]:
    if not value:
        return []
    windows = []
    for item in value.split(","):
        start, end = item.split("-", 1)
        windows.append((float(start), float(end)))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Test smart frame selection without model calls")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--frames-per-window", type=int, default=10)
    parser.add_argument("--metric", choices=["ssim", "mad"], default="ssim")
    parser.add_argument(
        "--target-windows",
        help="Comma-separated windows to print, e.g. 55-60,80-85,90-95. Defaults to all windows.",
    )
    parser.add_argument("--output-json", help="Optional path for full selection logs")
    parser.add_argument("--export-dir", help="Optional directory to write selected JPEG frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = float(frame_count) / float(fps) if fps > 0 else 0.0
    end_sec = min(args.end_sec if args.end_sec is not None else duration, duration)
    target_windows = parse_target_windows(args.target_windows)

    logs = []
    current = args.start_sec
    while current + args.window_sec <= end_sec + 1e-6:
        frame_window = [round(current, 3), round(current + args.window_sec, 3)]
        if target_windows and not any(abs(frame_window[0] - start) < 1e-6 and abs(frame_window[1] - end) < 1e-6 for start, end in target_windows):
            current = round(current + args.window_sec, 6)
            continue
        timestamps = window_timestamps(current, args.frame_fps, args.frames_per_window)
        frames = read_frames_at_timestamps(cap, timestamps)
        selected = smart_select_frames(frames=frames, timestamps=timestamps, metric=args.metric)
        exported_files = []
        if args.export_dir:
            export_dir = Path(args.export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            for item in selected:
                filename = (
                    f"window_{frame_window[0]:06.2f}_{frame_window[1]:06.2f}"
                    f"_idx_{item.index:02d}_t_{item.timestamp_sec:07.2f}.jpg"
                )
                path = export_dir / filename
                ok = cv2.imwrite(str(path), frames[item.index])
                if not ok:
                    raise ValueError(f"Could not write frame export: {path}")
                exported_files.append(str(path))
        log = {
            "frame_window": frame_window,
            "candidate_timestamps": timestamps,
            "metric": args.metric,
            "selected_frames": [item.__dict__ for item in selected],
            "exported_files": exported_files,
        }
        logs.append(log)

        print(f"window {frame_window[0]:.1f}-{frame_window[1]:.1f}s metric={args.metric}")
        for item in selected:
            left = "" if item.distance_to_left_anchor is None else f" d_left={item.distance_to_left_anchor:.4f}"
            right = "" if item.distance_to_right_anchor is None else f" d_right={item.distance_to_right_anchor:.4f}"
            print(
                f"  idx={item.index:>2} t={item.timestamp_sec:>6.2f}s "
                f"reason={item.reason:<36} score={item.score:.4f} "
                f"sharp={item.sharpness:.1f}{left}{right}"
            )

        current = round(current + args.window_sec, 6)

    cap.release()
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"video": args.video, "windows": logs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
