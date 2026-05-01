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

import requests
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format


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
    ):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]
        self.step_by_id = {step["step_id"]: step for step in self.steps}

        self.model = model
        self.frame_buffer = deque(maxlen=20)
        self.frames_per_call = 6
        self.call_interval_sec = 3.0
        self.last_vlm_call_ts = -self.call_interval_sec
        self.last_vlm_window_end = None

        self.completed_steps = set()
        self.current_step_index = 0
        self.emitted_error_times = []
        self.last_raw_vlm_response = ""
        self.last_status = {}
        self.last_window_summary = ""
        self.api_calls = 0

        self.step_confidence_threshold = 0.60
        self.catchup_confidence_threshold = 0.85
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
        self.frame_buffer.append({
            "timestamp_sec": timestamp_sec,
            "frame_base64": frame_base64,
        })

        if len(self.frame_buffer) < self.frames_per_call:
            return
        if timestamp_sec - self.last_vlm_call_ts < self.call_interval_sec:
            return

        selected_frames = list(self.frame_buffer)[-self.frames_per_call:]
        self.last_vlm_call_ts = timestamp_sec

        prompt = self._build_prompt(selected_frames)
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
            print(f"  [pipeline] VLM request failed at {timestamp_sec:.1f}s: {exc}")
            return

        parsed = parse_json_response(response)
        if not parsed:
            print(f"  [pipeline] Could not parse VLM JSON at {timestamp_sec:.1f}s")
            return

        self._handle_vlm_result(parsed, selected_frames, response)

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """
        Called by the harness for each audio chunk.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, 16-bit)
            start_sec: Chunk start time in video
            end_sec: Chunk end time in video

        TODO: Implement your audio processing logic.
        Consider: speech-to-text, keyword detection, silence detection.
        The instructor's verbal corrections are a strong signal for errors.
        """
        # Audio is intentionally left out of the first baseline. The callback is
        # kept so the harness can deliver audio without affecting the pipeline.
        return

    def _build_prompt(self, selected_frames: List[Dict[str, Any]]) -> str:
        frame_lines = [
            f"- image_{idx + 1}: timestamp_sec={frame['timestamp_sec']:.2f}"
            for idx, frame in enumerate(selected_frames)
        ]
        completed = sorted(self.completed_steps)
        current_step = self._current_step()
        next_steps = self.steps[self.current_step_index:self.current_step_index + 2]
        all_steps = "\n".join(
            f"{step['step_id']}. {step['description']}" for step in self.steps
        )
        next_step_text = "\n".join(
            f"{step['step_id']}. {step['description']}" for step in next_steps
        )

        current_step_text = "None; all known procedure steps are complete."
        if current_step:
            current_step_text = f"{current_step['step_id']}. {current_step['description']}"

        return f"""
You are monitoring a real-time task video for step completions and mistakes.

Task: {self.task_name}

Full procedure:
{all_steps}

Known state before these images:
- Completed step_ids: {completed}
- Current expected step: {current_step_text}
- Near-future steps to consider:
{next_step_text}
- Previous VLM status: {json.dumps(self.last_status)}
- Previous VLM summary: {self.last_window_summary or "None"}

The attached current evidence images are in chronological order:
{chr(10).join(frame_lines)}

Detect only events visible in the current evidence images image_1 through image_{len(selected_frames)}.
Use previous VLM text only to understand continuity from the prior call.
Do not emit a new event from the previous VLM text.
Most windows have no event. Do not invent an event just because a step is expected.
If the current evidence is ambiguous, return no events and use status instead.
It is better to defer a step completion to a later call than emit it early.

Step completion rules:
- A step_completion means the action for that step is finished by one of these frames.
- Do not report a step_completion when the student has only started the action,
  is holding the relevant object, or is still performing the action.
- If the step appears to be underway but not clearly finished, return status
  type step_in_progress instead of an event.
- Prefer the latest frame timestamp where the completed final state is visible.
- Only report uncompleted step_ids from the procedure.

Error rules:
- An error_detected means the student starts a wrong action, wrong sequence,
  safety violation, or improper technique.
- Prefer the earliest frame timestamp where the wrong action begins.
- Do not report ordinary hesitation or normal progress as an error.
- Be especially alert for the student grabbing, sliding, lifting, or using the
  wrong toolbox, wrong part, wrong switch, wrong button, or doing steps out of
  order compared with the current expected step.

Return exactly one JSON object with this schema and no extra text:
{{
  "events": [
    {{
      "type": "step_completion",
      "timestamp_sec": 12.5,
      "step_id": 1,
      "confidence": 0.0,
      "description": "short reason"
    }},
    {{
      "type": "error_detected",
      "timestamp_sec": 14.0,
      "confidence": 0.0,
      "error_type": "wrong_action",
      "severity": "warning",
      "description": "short reason",
      "spoken_response": "brief corrective instruction"
    }}
  ],
  "status": {{
    "type": "step_in_progress",
    "step_id": 1,
    "confidence": 0.0,
    "description": "what appears to be happening if no event should be emitted",
    "last_frame_state": "what is visible in the latest current evidence image",
    "completion_evidence": "why the current expected step is or is not complete",
    "event_recommendation": "defer"
  }},
  "summary": "brief observation of this window"
}}

Allowed event types: step_completion, error_detected.
Allowed status type values: step_in_progress, no_action, uncertain, possible_error_not_enough_evidence, idle_or_waiting.
Allowed event_recommendation values: defer, emit_if_still_complete, watch_for_error, no_event.
Allowed error_type values: wrong_action, wrong_sequence, safety_violation, improper_technique, other.
Allowed severity values: info, warning, critical.
If there are no step_completion or error_detected events, return {{"events": [], "status": {{"type": "...", "description": "..."}}, "summary": "..."}}.
""".strip()

    def _handle_vlm_result(
        self,
        parsed: Dict[str, Any],
        selected_frames: List[Dict[str, Any]],
        raw_response: str,
    ) -> None:
        events = parsed.get("events", [])
        if not isinstance(events, list):
            return
        status = parsed.get("status", {})
        if isinstance(status, dict):
            self.last_status = status
        summary = parsed.get("summary", "")
        if isinstance(summary, str):
            self.last_window_summary = summary[:500]

        frame_times = [frame["timestamp_sec"] for frame in selected_frames]
        min_ts = min(frame_times)
        max_ts = max(frame_times)

        step_events = [
            event for event in events
            if isinstance(event, dict) and event.get("type") == "step_completion"
        ]
        emitted_events = self._emit_ordered_steps(step_events, min_ts, max_ts, raw_response)

        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "error_detected":
                emitted = self._maybe_emit_error(event, min_ts, max_ts, raw_response)
                if emitted:
                    emitted_events.append(emitted)

        event_proposals = [
            event for event in events
            if isinstance(event, dict) and event.get("type") in {"step_completion", "error_detected"}
        ]
        if event_proposals:
            self._print_event_proposals(event_proposals, frame_times, emitted_events)
        self.last_vlm_window_end = max_ts

    def _emit_ordered_steps(
        self,
        step_events: List[Dict[str, Any]],
        min_ts: float,
        max_ts: float,
        raw_response: str,
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
                catchup = self._maybe_emit_catchup_step(events_by_step, min_ts, max_ts, raw_response)
                if not catchup:
                    break
                emitted_events.extend(catchup)
                continue
            emitted = self._emit_step(event, min_ts, max_ts, raw_response)
            if emitted:
                emitted_events.append(emitted)
                continue
            break
        return emitted_events

    def _maybe_emit_catchup_step(
        self,
        events_by_step: Dict[int, Dict[str, Any]],
        min_ts: float,
        max_ts: float,
        raw_response: str,
    ) -> List[Dict[str, Any]]:
        if self.current_step_index + 1 >= len(self.steps):
            return []

        current_step = self.steps[self.current_step_index]
        next_step = self.steps[self.current_step_index + 1]
        next_event = events_by_step.get(next_step["step_id"])
        if next_event is None:
            return []

        confidence = self._as_float(next_event.get("confidence"), default=0.0)
        if confidence < self.catchup_confidence_threshold:
            return []

        catchup_timestamp = self.last_vlm_window_end
        if catchup_timestamp is None:
            catchup_timestamp = min_ts
        catchup_timestamp = min(catchup_timestamp, max_ts)

        catchup_event = {
            "timestamp_sec": catchup_timestamp,
            "type": "step_completion",
            "step_id": current_step["step_id"],
            "confidence": min(confidence, 0.75),
            "description": (
                "State catch-up: the VLM detected the next step, so the prior "
                f"expected step is treated as completed: {current_step['description']}"
            ),
            "source": "video",
            "vlm_observation": raw_response[:1200],
        }
        self.harness.emit_event(catchup_event)
        self.completed_steps.add(current_step["step_id"])
        self.current_step_index += 1

        emitted_next = self._emit_step(next_event, min_ts, max_ts, raw_response)
        emitted = [catchup_event]
        if emitted_next:
            emitted.append(emitted_next)
        return emitted

    def _emit_step(
        self,
        event: Dict[str, Any],
        min_ts: float,
        max_ts: float,
        raw_response: str,
    ) -> Optional[Dict[str, Any]]:
        step_id = self._as_int(event.get("step_id"))
        if step_id is None:
            return None
        expected_step = self._current_step()
        if not expected_step or step_id != expected_step["step_id"]:
            return None

        confidence = self._as_float(event.get("confidence"), default=0.0)
        timestamp_sec = self._bounded_timestamp(event.get("timestamp_sec"), min_ts, max_ts)
        timestamp_sec = max(timestamp_sec, max_ts)
        step = self.step_by_id[step_id]
        emitted = {
            "timestamp_sec": timestamp_sec,
            "type": "step_completion",
            "step_id": step_id,
            "confidence": confidence,
            "description": event.get("description") or step["description"],
            "source": "video",
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
    ) -> Optional[Dict[str, Any]]:
        confidence = self._as_float(event.get("confidence"), default=0.0)
        if confidence < self.error_confidence_threshold:
            return None

        timestamp_sec = self._bounded_timestamp(event.get("timestamp_sec"), min_ts, max_ts)
        if any(abs(timestamp_sec - prior) < self.error_cooldown_sec for prior in self.emitted_error_times):
            return None

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
            "source": "video",
            "vlm_observation": raw_response[:1200],
        }
        self.harness.emit_event(emitted)
        self.emitted_error_times.append(timestamp_sec)
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
    load_dotenv()

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

    pipeline = Pipeline(harness, api_key, procedure, model=args.model)

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

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
