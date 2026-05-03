"""
Descriptor/reasoner routing submission pipeline.

The harness feeds frames in real time. This file buffers those frames into
5-second descriptor windows, calls the image descriptor model, then sends every
two descriptor windows to the text-only reasoner. Descriptor and detector logic
live in separate modules:

  - src.descriptor_experiment: image-window description prompt/API helpers
  - src.routing_experiment: text-only event detector/reasoner

Usage:
    python src/integrated_pipeline.py \
        --procedure data/clip_procedures/CLIP.json \
        --video path/to/Video_pitchshift.mp4 \
        --output output/events.json \
        --speed 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_procedure_json, validate_procedure_format
from src.descriptor import (
    build_descriptor_prompt,
    call_openrouter_descriptor,
    frame_to_base64,
    normalize_window_description,
    parse_json_response,
)
from src.harness import StreamingHarness
from src.routing import RoutingReasoner, make_window_groups
from src.smart_frame_sampler import smart_select_frames
from src.step_rubric import load_step_rubrics


def append_jsonl(path: Optional[Path], item: Dict[str, Any]) -> None:
    if not path:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(item) + "\n")

# ==========================================================================
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

class Pipeline:
    """
    Harness-facing descriptor/reasoner pipeline.

    The main file only owns streaming state and event emission. The descriptor
    and reasoner modules own the model prompts, JSON parsing, and response
    handling used by the offline experiments.
    """

    def __init__(
        self,
        *,
        harness: StreamingHarness,
        api_key: str,
        procedure: Dict[str, Any],
        descriptor_model: str,
        reasoner_model: str,
        descriptor_temperature: Optional[float],
        reasoner_temperature: Optional[float],
        descriptor_top_p: Optional[float],
        reasoner_top_p: Optional[float],
        descriptor_timeout_sec: int,
        frame_sampler: str,
        sampler_metric: str,
        jpeg_quality: int,
        frames_per_descriptor_window: int,
        reasoner_windows_per_call: int,
        step_confidence_threshold: float,
        error_confidence_threshold: float,
        step_rubric: Optional[str],
        descriptor_log: Optional[str],
        reasoner_log: Optional[str],
    ) -> None:
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]

        self.descriptor_model = descriptor_model
        self.descriptor_temperature = descriptor_temperature
        self.descriptor_top_p = descriptor_top_p
        self.descriptor_timeout_sec = descriptor_timeout_sec
        self.frame_sampler = frame_sampler
        self.sampler_metric = sampler_metric
        self.jpeg_quality = jpeg_quality
        self.frames_per_descriptor_window = max(frames_per_descriptor_window, 3)
        self.reasoner_windows_per_call = max(reasoner_windows_per_call, 1)

        self.descriptor_log_path = Path(descriptor_log) if descriptor_log else None
        if self.descriptor_log_path:
            self.descriptor_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.descriptor_log_path.write_text("")

        rubrics = load_step_rubrics(step_rubric) if step_rubric else []
        self.reasoner = RoutingReasoner(
            procedure=procedure,
            rubrics=rubrics,
            api_key=api_key,
            model=reasoner_model,
            temperature=reasoner_temperature,
            top_p=reasoner_top_p,
            step_confidence_threshold=step_confidence_threshold,
            error_confidence_threshold=error_confidence_threshold,
            windows_per_call=self.reasoner_windows_per_call,
            reasoner_log_path=reasoner_log,
        )

        self._frame_buffer: List[np.ndarray] = []
        self._timestamp_buffer: List[float] = []
        self._descriptor_windows: List[Dict[str, Any]] = []
        self._pending_reasoner_windows: List[Dict[str, Any]] = []
        self._descriptor_call_count = 0
        self._reasoner_call_count = 0

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str) -> None:
        del frame_base64
        self._frame_buffer.append(frame.copy())
        self._timestamp_buffer.append(float(timestamp_sec))
        if len(self._frame_buffer) >= self.frames_per_descriptor_window:
            self.process_descriptor_window()

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float) -> None:
        del audio_bytes, start_sec, end_sec
        # Audio is intentionally ignored for now.

    def on_complete(self) -> None:
        if len(self._frame_buffer) >= 3:
            self.process_descriptor_window()
        if self._pending_reasoner_windows:
            self.process_reasoner_windows(force=True)

    def process_descriptor_window(self) -> None:
        frames = self._frame_buffer
        timestamps = self._timestamp_buffer
        self._frame_buffer = []
        self._timestamp_buffer = []

        if len(frames) < 3:
            return

        self._descriptor_call_count += 1
        window = {
            "source_log_index": self._descriptor_call_count,
            "frame_window": [round(timestamps[0], 3), round(timestamps[-1], 3)],
            "candidate_frame_timestamps": [round(timestamp, 3) for timestamp in timestamps],
            "midpoint_sec": round((timestamps[0] + timestamps[-1]) / 2, 3),
        }
        current_step = self.reasoner.current_step()
        current_step_id = int(current_step["step_id"]) if current_step else int(self.steps[-1]["step_id"])

        try:
            selected_frames, selected_timestamps, selection_log = self.select_descriptor_frames(frames, timestamps)
            encoded_frames = [
                frame_to_base64(frame, self.jpeg_quality)
                for frame in selected_frames
            ]
            prompt = build_descriptor_prompt(
                procedure=self.procedure,
                frame_timestamps=selected_timestamps,
                current_step_id=current_step_id,
            )
            raw_response = call_openrouter_descriptor(
                api_key=self.api_key,
                frames_base64=encoded_frames,
                prompt=prompt,
                model=self.descriptor_model,
                temperature=self.descriptor_temperature,
                top_p=self.descriptor_top_p,
                timeout_sec=self.descriptor_timeout_sec,
            )
            parsed = parse_json_response(raw_response)
            window_description = normalize_window_description(parsed)
            error = None
        except Exception as exc:
            selected_timestamps = timestamps
            selection_log = None
            prompt = build_descriptor_prompt(
                procedure=self.procedure,
                frame_timestamps=selected_timestamps,
                current_step_id=current_step_id,
            )
            raw_response = ""
            parsed = None
            window_description = None
            error = str(exc)
            print(f"  [descriptor] {window['frame_window'][0]:.1f}-{window['frame_window'][1]:.1f}s ERROR: {exc}")

        described_window = {
            **window,
            "frame_timestamps": [round(timestamp, 3) for timestamp in selected_timestamps],
            "current_step_id_hint": current_step_id,
            "window_description": window_description,
        }
        self._descriptor_windows.append(described_window)
        self._pending_reasoner_windows.append(described_window)

        action = (window_description or {}).get("student_action", "")
        print(
            f"  [descriptor] {window['frame_window'][0]:.1f}-{window['frame_window'][1]:.1f}s "
            f"step_hint={current_step_id}: {action[:160]}"
        )
        append_jsonl(self.descriptor_log_path, {
            **window,
            "selected_frame_timestamps": [round(timestamp, 3) for timestamp in selected_timestamps],
            "frame_selection": selection_log,
            "current_step_id_hint": current_step_id,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "window_description": window_description,
            "error": error,
        })
        self.process_reasoner_windows()

    def select_descriptor_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
    ) -> tuple[List[np.ndarray], List[float], Optional[Dict[str, Any]]]:
        if self.frame_sampler == "uniform":
            return frames, timestamps, None
        if self.frame_sampler != "smart":
            raise ValueError(f"Unknown frame sampler: {self.frame_sampler}")

        selected = smart_select_frames(
            frames=frames,
            timestamps=timestamps,
            metric=self.sampler_metric,
        )
        selection_log = {
            "sampler": self.frame_sampler,
            "metric": self.sampler_metric,
            "candidate_timestamps": [round(timestamp, 3) for timestamp in timestamps],
            "selected_frames": [item.__dict__ for item in selected],
        }
        return (
            [frames[item.index] for item in selected],
            [item.timestamp_sec for item in selected],
            selection_log,
        )

    def process_reasoner_windows(self, force: bool = False) -> None:
        if not force and len(self._pending_reasoner_windows) < self.reasoner_windows_per_call:
            return
        while self._pending_reasoner_windows and (
            force or len(self._pending_reasoner_windows) >= self.reasoner_windows_per_call
        ):
            group_windows = self._pending_reasoner_windows[:self.reasoner_windows_per_call]
            self._pending_reasoner_windows = self._pending_reasoner_windows[self.reasoner_windows_per_call:]
            group = make_window_groups(group_windows, len(group_windows))[0]
            self._reasoner_call_count += 1
            try:
                emitted = self.reasoner.process_group(group, index=self._reasoner_call_count)
            except Exception as exc:
                print(
                    f"  [reasoner] {group.get('frame_window')} ERROR: {exc}"
                )
                continue
            for event in emitted:
                self.emit_event(event)
            if emitted:
                print(
                    f"  [reasoner] {group.get('frame_window')} emitted={len(emitted)} "
                    f"completed={sorted(self.reasoner.completed_steps)}"
                )

    def emit_event(self, event: Dict[str, Any]) -> None:
        allowed = {
            "timestamp_sec",
            "type",
            "step_id",
            "confidence",
            "description",
            "source",
            "reason",
            "reasoner_observation",
            "matched_phase",
            "error_type",
            "severity",
            "spoken_response",
        }
        output = {key: value for key, value in event.items() if key in allowed}
        if output.get("type") == "error_detected":
            output.setdefault("severity", "warning")
            output.setdefault("spoken_response", "Stop and return to the current step.")
        self.harness.emit_event(output)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Descriptor/reasoner VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0)
    parser.add_argument("--api-key", help="OpenRouter API key, or set OPENROUTER_API_KEY")
    parser.add_argument("--descriptor-model", default="google/gemini-3.1-flash-image-preview")
    parser.add_argument("--reasoner-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--descriptor-temperature", type=float, default=0.2)
    parser.add_argument("--reasoner-temperature", type=float, default=0.2)
    parser.add_argument("--descriptor-top-p", type=float)
    parser.add_argument("--reasoner-top-p", type=float)
    parser.add_argument("--descriptor-timeout-sec", type=int, default=90)
    parser.add_argument("--frame-sampler", choices=["uniform", "smart"], default="smart")
    parser.add_argument("--sampler-metric", choices=["ssim", "mad"], default="ssim")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--frames-per-descriptor-window", type=int, default=10)
    parser.add_argument("--reasoner-windows-per-call", type=int, default=2)
    parser.add_argument("--step-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--error-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--step-rubric", help="Optional reasoner rubric JSON; descriptor never uses rubrics")
    parser.add_argument("--descriptor-log", help="Optional descriptor JSONL trace")
    parser.add_argument("--reasoner-log", help="Optional reasoner JSONL trace")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    print("=" * 60)
    print("  VLM ORCHESTRATOR - DESCRIPTOR/REASONER ROUTING")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure:  {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:      {args.video}")
    print(f"  Speed:      {args.speed}x")
    print(f"  Descriptor: {args.descriptor_model}")
    print(f"  Reasoner:   {args.reasoner_model}")
    print(f"  Sampler:    {args.frame_sampler}" + (f" ({args.sampler_metric})" if args.frame_sampler == "smart" else ""))
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
        print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )
    pipeline = Pipeline(
        harness=harness,
        api_key=api_key,
        procedure=procedure,
        descriptor_model=args.descriptor_model,
        reasoner_model=args.reasoner_model,
        descriptor_temperature=args.descriptor_temperature,
        reasoner_temperature=args.reasoner_temperature,
        descriptor_top_p=args.descriptor_top_p,
        reasoner_top_p=args.reasoner_top_p,
        descriptor_timeout_sec=args.descriptor_timeout_sec,
        frame_sampler=args.frame_sampler,
        sampler_metric=args.sampler_metric,
        jpeg_quality=args.jpeg_quality,
        frames_per_descriptor_window=args.frames_per_descriptor_window,
        reasoner_windows_per_call=args.reasoner_windows_per_call,
        step_confidence_threshold=args.step_confidence_threshold,
        error_confidence_threshold=args.error_confidence_threshold,
        step_rubric=args.step_rubric,
        descriptor_log=args.descriptor_log,
        reasoner_log=args.reasoner_log,
    )

    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)
    harness.on_complete(pipeline.on_complete)

    results = harness.run()
    harness.save_results(results, args.output)

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()


if __name__ == "__main__":
    main()
