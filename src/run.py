"""
VLM Orchestrator — Starter Template

This is where you implement your pipeline. The harness feeds you frames
and audio in real-time. You call VLMs, detect events, and emit them back.

Usage:
    python src/run.py \\
        --procedure data/clip_procedures/CLIP.json \\
        --video path/to/Video_pitchshift.mp4 \\
        --output output/events.json \\
        --speed 1.0
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import threading

import requests
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format
from src.visual_context import VisualContextManager


# ==========================================================================
# VLM API HELPER (provided — feel free to modify)
# ==========================================================================

def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    """
    Call a VLM via OpenRouter.

    Args:
        api_key: OpenRouter API key
        frame_base64: Base64-encoded JPEG frame
        prompt: Text prompt
        model: OpenRouter model string
        stream: If True, use streaming (SSE) responses for lower time-to-first-token

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
                ],
            }
        ],
    }

    if stream:
        # Streaming: read SSE chunks as they arrive
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return full_text
    else:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def call_vlm_multi_frame(
    api_key: str,
    frames_base64: List[str],
    prompt: str,
    model: str = "google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    """
    Call a VLM with multiple timestamped frame images.

    The prompt is responsible for describing which image corresponds to which
    timestamp. Images are attached in the same order as frames_base64.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    content = [{"type": "text", "text": prompt}]
    for frame_base64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
        })

    payload = {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": content}],
    }

    if stream:
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=45)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return full_text

    resp = requests.post(url, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse strict JSON from a model response, tolerating markdown fences."""
    text = text.strip()
    if not text:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ==========================================================================
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

class Pipeline:
    """
    Your VLM orchestration pipeline.

    The harness calls on_frame() and on_audio() in real-time as the video plays.
    When you detect an event, call self.harness.emit_event({...}).

    Key design decisions you need to make:
    - Which frames to send to the VLM (not every frame — budget is limited)
    - Whether/how to use audio (speech-to-text for instructor corrections?)
    - Which model to use and when (cheap for easy frames, expensive for hard ones?)
    - How to track procedure state (current step, completed steps)
    - How to generate spoken responses for errors
    """

    def __init__(
        self,
        harness: StreamingHarness,
        api_key: str,
        procedure: Dict[str, Any],
        model: str = "google/gemini-2.5-flash",
        vlm_log_path: Optional[str] = None,
        vlm_log_start: Optional[float] = None,
        vlm_log_end: Optional[float] = None,
    ):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]
        self.step_by_id = {step["step_id"]: step for step in self.steps}

        self.model = model
        self.frame_buffer = deque(maxlen=20)
        self.frames_per_call = 10
        self.call_interval_sec = 5.0
        self.last_vlm_call_ts = -self.call_interval_sec

        self.state_lock = threading.RLock()
        self.completed_steps = set()
        self.current_step_index = 0
        self.visual_context = VisualContextManager(self.steps)
        self.emitted_error_times = []
        self.last_raw_vlm_response = ""
        self.last_status = {}
        self.api_calls = 0
        self.vlm_log_path = Path(vlm_log_path) if vlm_log_path else None
        self.vlm_log_start = vlm_log_start
        self.vlm_log_end = vlm_log_end
        self.vlm_log_count = 0
        self.vlm_log_records = []
        self.vlm_log_lock = threading.Lock()
        if self.vlm_log_path:
            self.vlm_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.vlm_log_path.write_text("[]" if self.vlm_log_path.suffix == ".json" else "")

        self.step_confidence_threshold = 0.60
        self.error_confidence_threshold = 0.55
        self.error_cooldown_sec = 2.0

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        """
        Called by the harness for each video frame.

        Args:
            frame: BGR numpy array (raw frame)
            timestamp_sec: Current video timestamp
            frame_base64: Pre-encoded JPEG base64 string (ready for VLM API)

        TODO: Implement your frame processing logic.
        When you detect an event, call:
            self.harness.emit_event({
                "timestamp_sec": timestamp_sec,
                "type": "step_completion",  # or "error_detected" or "idle_detected"
                "step_id": 1,
                "confidence": 0.9,
                "description": "...",
                "source": "video",
                "vlm_observation": "...",
                # For errors, also include:
                "spoken_response": "Stop — you need to turn off the power first.",
            })
        """
        with self.state_lock:
            self.frame_buffer.append({
                "timestamp_sec": timestamp_sec,
                "frame_base64": frame_base64,
            })
            if (
                len(self.frame_buffer) < self.frames_per_call
                or timestamp_sec - self.last_vlm_call_ts < self.call_interval_sec
            ):
                return
            selected_frames = list(self.frame_buffer)[-self.frames_per_call:]
            self.last_vlm_call_ts = timestamp_sec

        prompt = self._build_visual_prompt(selected_frames)
        try:
            response = call_vlm_multi_frame(
                api_key=self.api_key,
                frames_base64=[item["frame_base64"] for item in selected_frames],
                prompt=prompt,
                model=self.model,
            )
            self.api_calls += 1
            self.last_raw_vlm_response = response
        except requests.RequestException as exc:
            print(f"  [pipeline] Visual VLM failed at {timestamp_sec:.1f}s: {exc}")
            return

        parsed = parse_json_response(response)
        if not parsed:
            self._maybe_log_vlm_call("visual_window", selected_frames, prompt, response, None)
            print(f"  [pipeline] Could not parse visual VLM JSON at {timestamp_sec:.1f}s")
            return

        self._maybe_log_vlm_call("visual_window", selected_frames, prompt, response, parsed)
        self._handle_vlm_result(parsed, selected_frames, response, source="video")

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """
        Called by the harness for each audio chunk.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, 16-bit)
            start_sec: Chunk start time in video
            end_sec: Chunk end time in video

        Audio is intentionally ignored in the visual-only baseline. The challenge
        update says instructor audio can leak error labels, so STT experiments
        live in audio_stt_experiment.py instead of this runtime pipeline.
        """
        return

    def finish(self):
        """No async work to finalize in the visual-only baseline."""
        return

    def _maybe_log_vlm_call(
        self,
        source: str,
        selected_frames: List[Dict[str, Any]],
        prompt: str,
        response: str,
        parsed: Optional[Dict[str, Any]],
    ) -> None:
        if not self.vlm_log_path or not selected_frames:
            return

        frame_times = [frame["timestamp_sec"] for frame in selected_frames]
        window_start = min(frame_times)
        window_end = max(frame_times)
        if self.vlm_log_start is not None and window_end < self.vlm_log_start:
            return
        if self.vlm_log_end is not None and window_start > self.vlm_log_end:
            return

        with self.state_lock:
            current_step = self._current_step()
            state = self.visual_context.state_snapshot(
                completed_steps=sorted(self.completed_steps),
                current_step_id=current_step.get("step_id") if current_step else None,
            )

        record = {
            "source": source,
            "frame_window": [window_start, window_end],
            "frame_timestamps": frame_times,
            "state_before_response": state,
            "prompt": prompt,
            "raw_response": response,
            "parsed_response": parsed,
        }
        if isinstance(parsed, dict):
            record["visual_context"] = self.visual_context.response_log_context(parsed)
        with self.vlm_log_lock:
            self.vlm_log_count += 1
            record["log_index"] = self.vlm_log_count
            if self.vlm_log_path.suffix == ".json":
                self.vlm_log_records.append(record)
                self.vlm_log_path.write_text(json.dumps(self.vlm_log_records, indent=2))
            else:
                with self.vlm_log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

    def _build_visual_prompt(self, selected_frames: List[Dict[str, Any]]) -> str:
        with self.state_lock:
            completed = sorted(self.completed_steps)
            current_step = self._current_step()
            next_steps = self.steps[self.current_step_index:self.current_step_index + 1]
            return self.visual_context.build_prompt(
                task_name=self.task_name,
                selected_frames=selected_frames,
                completed_steps=completed,
                current_step=current_step,
                next_steps=next_steps,
            )

    def _handle_vlm_result(
        self,
        parsed: Dict[str, Any],
        selected_frames: List[Dict[str, Any]],
        raw_response: str,
        source: str = "both",
    ) -> List[Dict[str, Any]]:
        events = parsed.get("events", [])
        if not isinstance(events, list):
            events = []
        status = parsed.get("status", {})
        if isinstance(status, dict):
            self.last_status = status

        frame_times = [frame["timestamp_sec"] for frame in selected_frames]
        min_ts = min(frame_times)
        max_ts = max(frame_times)
        self._update_visual_context(parsed, frame_times)

        step_events = [
            event for event in events
            if isinstance(event, dict) and event.get("type") == "step_completion"
        ]
        emitted_events = self._emit_ordered_steps(step_events, min_ts, max_ts, raw_response, source)

        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "error_detected":
                emitted = self._maybe_emit_error(event, min_ts, max_ts, raw_response, source=source)
                if emitted:
                    emitted_events.append(emitted)

        event_proposals = [
            event for event in events
            if isinstance(event, dict) and event.get("type") in {"step_completion", "error_detected"}
        ]
        if event_proposals:
            self._print_event_proposals(event_proposals, frame_times, emitted_events)
        return emitted_events

    def _update_visual_context(self, parsed: Dict[str, Any], frame_times: List[float]) -> None:
        with self.state_lock:
            self.visual_context.update_from_response(parsed, frame_times)

    def _emit_ordered_steps(
        self,
        step_events: List[Dict[str, Any]],
        min_ts: float,
        max_ts: float,
        raw_response: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        emitted_events = []
        events_by_step = {}
        for event in step_events:
            step_id = self._as_int(event.get("step_id"))
            if step_id is None or step_id in self.completed_steps:
                continue
            confidence = self._as_float(event.get("confidence"), default=0.0)
            if confidence < self.step_confidence_threshold:
                continue
            existing = events_by_step.get(step_id)
            if existing is None or confidence > self._as_float(existing.get("confidence"), default=0.0):
                events_by_step[step_id] = event

        while self.current_step_index < len(self.steps):
            expected_step_id = self.steps[self.current_step_index]["step_id"]
            event = events_by_step.get(expected_step_id)
            if event is None:
                break
            emitted = self._emit_step(event, min_ts, max_ts, raw_response, source)
            if emitted:
                emitted_events.append(emitted)
        return emitted_events

    def _emit_step(
        self,
        event: Dict[str, Any],
        min_ts: float,
        max_ts: float,
        raw_response: str,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        step_id = self._as_int(event.get("step_id"))
        if step_id is None:
            return None
        expected_step = self._current_step()
        if not expected_step or step_id != expected_step["step_id"]:
            return None

        confidence = self._as_float(event.get("confidence"), default=0.0)
        timestamp_sec = self._bounded_timestamp(event.get("timestamp_sec"), min_ts, max_ts)
        step = self.step_by_id[step_id]
        emitted = {
            "timestamp_sec": timestamp_sec,
            "type": "step_completion",
            "step_id": step_id,
            "confidence": confidence,
            "description": event.get("description") or step["description"],
            "source": source,
            "vlm_observation": raw_response[:1200],
        }
        self.harness.emit_event(emitted)

        self.completed_steps.add(step_id)
        while (
            self.current_step_index < len(self.steps)
            and self.steps[self.current_step_index]["step_id"] in self.completed_steps
        ):
            self.current_step_index += 1
        return emitted

    def _maybe_emit_error(
        self,
        event: Dict[str, Any],
        min_ts: float,
        max_ts: float,
        raw_response: str,
        source: str = "video",
    ) -> Optional[Dict[str, Any]]:
        confidence = self._as_float(event.get("confidence"), default=0.0)
        if confidence < self.error_confidence_threshold:
            return None

        timestamp_sec = self._bounded_timestamp(event.get("timestamp_sec"), min_ts, max_ts)
        with self.state_lock:
            if any(abs(timestamp_sec - prior) < self.error_cooldown_sec for prior in self.emitted_error_times):
                return None
            self.emitted_error_times.append(timestamp_sec)

        error_type = event.get("error_type") or "other"
        if error_type not in self.harness.VALID_ERROR_TYPES:
            error_type = "other"
        severity = event.get("severity") or "warning"
        if severity not in self.harness.VALID_SEVERITIES:
            severity = "warning"

        emitted = {
            "timestamp_sec": timestamp_sec,
            "type": "error_detected",
            "confidence": confidence,
            "error_type": error_type,
            "severity": severity,
            "description": event.get("description") or "Possible procedure error detected.",
            "spoken_response": event.get("spoken_response") or "Stop and check the procedure before continuing.",
            "source": source,
            "vlm_observation": raw_response[:1200],
        }
        self.harness.emit_event(emitted)
        return emitted

    def _print_event_proposals(
        self,
        event_proposals: List[Dict[str, Any]],
        frame_times: List[float],
        emitted_events: List[Dict[str, Any]],
    ) -> None:
        print(
            f"  [pipeline] VLM event proposal call={self.api_calls} "
            f"window={frame_times[0]:.1f}-{frame_times[-1]:.1f}s "
            f"proposed={len(event_proposals)} emitted={len(emitted_events)}"
        )
        for event in event_proposals:
            event_type = event.get("type")
            timestamp_sec = self._as_float(event.get("timestamp_sec"), default=frame_times[-1])
            confidence = self._as_float(event.get("confidence"), default=0.0)
            description = (event.get("description") or "")[:120]
            if event_type == "step_completion":
                print(
                    f"    proposed step_completion step={event.get('step_id')} "
                    f"t={timestamp_sec:.1f}s conf={confidence:.2f} desc={description}"
                )
            else:
                print(
                    f"    proposed error_detected t={timestamp_sec:.1f}s "
                    f"conf={confidence:.2f} type={event.get('error_type', 'other')} "
                    f"desc={description}"
                )
        for event in emitted_events:
            if event["type"] == "step_completion":
                print(
                    f"    emitted step_completion step={event.get('step_id')} "
                    f"t={event['timestamp_sec']:.1f}s conf={event.get('confidence', 0):.2f}"
                )
            else:
                print(
                    f"    emitted error_detected t={event['timestamp_sec']:.1f}s "
                    f"conf={event.get('confidence', 0):.2f} type={event.get('error_type', 'other')}"
                )

    def _current_step(self) -> Optional[Dict[str, Any]]:
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _bounded_timestamp(self, value: Any, min_ts: float, max_ts: float) -> float:
        timestamp_sec = self._as_float(value, default=max_ts)
        return max(min_ts, min(max_ts, timestamp_sec))


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4 (with audio)")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time, 2.0 = 2x, etc.)")
    parser.add_argument("--frame-fps", type=float, default=2.0,
                        help="Frames per second delivered to pipeline (default: 2)")
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--model", default="google/gemini-2.5-flash",
                        help="OpenRouter model string for VLM calls")
    parser.add_argument("--vlm-log", help="Optional .json or .jsonl path for VLM prompt/response debug logs")
    parser.add_argument("--vlm-log-start", type=float,
                        help="Only log VLM calls whose frame window overlaps this timestamp")
    parser.add_argument("--vlm-log-end", type=float,
                        help="Only log VLM calls whose frame window overlaps this timestamp")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    # Load procedure
    print("=" * 60)
    print("  VLM ORCHESTRATOR")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure: {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Speed:     {args.speed}x")
    print(f"  Model:     {args.model}")
    print("  Audio:     ignored (visual-only baseline)")
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
            print("  [DRY RUN] Procedure validated. Video not checked (file missing).")
        else:
            print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    # Create harness and pipeline
    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(
        harness,
        api_key,
        procedure,
        model=args.model,
        vlm_log_path=args.vlm_log,
        vlm_log_start=args.vlm_log_start,
        vlm_log_end=args.vlm_log_end,
    )

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)
    harness.on_complete(pipeline.finish)

    # Run
    results = harness.run()

    # Save
    harness.save_results(results, args.output)

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()

    if not results.events:
        print("  WARNING: No events detected. Implement Pipeline.on_frame() and Pipeline.on_audio().")


if __name__ == "__main__":
    main()
