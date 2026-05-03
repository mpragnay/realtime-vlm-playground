"""
Descriptor-only VLM stage for routing experiments.

This script extracts non-overlapping 5-second windows from a video, sends all
10 sampled frames in each window to an image-capable model, and writes grounded
visual descriptions. It intentionally does not detect steps, errors, or update
procedure state; the output is meant to be consumed by src/routing_experiment.py.

Example:
    .venv/bin/python src/descriptor_experiment.py \
        --procedure data/clip_procedures/z045-june-24-22-dslr.json \
        --video data/videos_full/z045-june-24-22-dslr/Export_py/Video_pitchshift.mp4 \
        --output output/z045-descriptor-3.1-flash.json \
        --descriptor-log output/z045-descriptor-3.1-flash.jsonl \
        --model google/gemini-3.1-flash-image-preview
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.smart_frame_sampler import smart_select_frames


WINDOW_DESCRIPTION_KEYS = (
    "start_state",
    "end_state",
    "motion_or_change",
    "uncertain_inferences",
    "student_action",
    "objects",
    "scene_layout",
    "step_relevance",
)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


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


def frame_to_base64(frame: Any, jpeg_quality: int) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def call_openrouter_descriptor(
    *,
    api_key: str,
    frames_base64: List[str],
    prompt: str,
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    timeout_sec: int,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Descriptor Experiment",
    }
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame_base64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
        })

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    response = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def step_lines(steps: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{step['step_id']}. {step['description']}" for step in steps)


def step_by_id(steps: List[Dict[str, Any]], step_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if step_id is None:
        return None
    for step in steps:
        if int(step["step_id"]) == int(step_id):
            return step
    return None


def format_step(step: Optional[Dict[str, Any]]) -> str:
    if not step:
        return "None"
    return f"{step['step_id']}. {step['description']}"


def load_step_hints(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    data = load_json(path)
    if isinstance(data, list):
        hints = data
    else:
        hints = data.get("hints") or data.get("step_hints") or []
    if not isinstance(hints, list):
        raise ValueError("--step-hints must be a list or contain a hints/step_hints list")
    return hints


def current_step_id_for_window(
    *,
    midpoint_sec: float,
    step_hints: List[Dict[str, Any]],
    default_step_id: int,
) -> int:
    for hint in step_hints:
        start = hint.get("start_sec", hint.get("start"))
        end = hint.get("end_sec", hint.get("end"))
        step_id = hint.get("current_step_id", hint.get("step_id"))
        if start is None or end is None or step_id is None:
            continue
        if float(start) <= midpoint_sec < float(end):
            return int(step_id)
    return default_step_id


def build_descriptor_prompt(
    *,
    procedure: Dict[str, Any],
    frame_timestamps: List[float],
    current_step_id: int,
) -> str:
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    steps = procedure["steps"]
    current_step = step_by_id(steps, current_step_id)
    previous_step = step_by_id(steps, current_step_id - 1)
    next_step = step_by_id(steps, current_step_id + 1)
    frame_lines = "\n".join(
        f"- image_{index}: timestamp_sec={timestamp:.2f}"
        for index, timestamp in enumerate(frame_timestamps, start=1)
    )

    return f"""
You are the descriptor agent for a visual-only procedural task detector.

Your job is ONLY to describe what is visibly happening in the attached frames.
Do not detect step completions. Do not detect errors. Do not decide whether the
student is right or wrong. Do not output events, status, confidence, or running
summary.

Task setting:
- The video shows a student performing a hands-on procedure in a workspace.
- The images are from the student's first-person/egocentric point of view.
- The student's face/body will usually not be visible. Interpret visible hands,
  camera movement, gaze direction, and manipulated objects as evidence of what
  the student appears to be doing.
- If no hands are visible but the camera moves, describe the apparent POV
  action, such as looking toward, moving closer to, backing away from, or
  scanning an object or area.

Task: {task_name}

Full procedure, for object/action context only:
{step_lines(steps)}

Approximate step context hints:
- Previous step: {format_step(previous_step)}
- Current step: {format_step(current_step)}
- Next step: {format_step(next_step)}

How to use these step hints:
- Use them only to know which visible objects and actions may be relevant to
  describe carefully.
- Do not say a step is complete, incomplete, correct, wrong, or an error.
- Avoid phrases like "this completes step X", "the student should", or "the
  correct object". Describe the visible evidence instead.

Current frames in chronological order:
{frame_lines}

Description rules:
- Use only the attached frames.
- Describe visible state at the beginning and end of the ordered frame sequence.
- Describe visible motion/change across the sequence, not only the last frame.
- Describe objects by appearance first: color, shape, size, labels/text,
  location, relation to hands/camera, and uncertainty.
- Do not state hidden state changes as facts. If an internal mechanism, button
  state, latch state, electrical state, or camera setting is not visible, say it
  is unclear.
- Do not use definitive completion/state words such as attached, locked,
  seated, inserted, removed, closed, opened, on, or off unless the stable
  post-action state is clearly visible after the hand releases or moves away.
- If the hand is still manipulating, pressing, rotating, aligning or holding the object, 
  describe the action as in progress along with the last visible state.
- Prefer grounded wording like "appears aligned", "being rotated", "being
  pressed", "partially inserted", or "possibly seated" when the final state is
  not directly visible.
- You may include likely interpretations only in uncertain_inferences.
- Preserve uncertainty. Do not turn "appears to" or "possibly" into facts.
- For step_relevance, only say how visible evidence may relate to the previous,
  current, or next step. Do not make event decisions there.

Return exactly one JSON object with this schema and no extra text:
{{
  "window_description": {{
    "start_state": "what is visible at the beginning of the frame sequence",
    "end_state": "what is visible at the end of the frame sequence",
    "motion_or_change": "how hands, camera viewpoint, and objects changed across the ordered frames",
    "uncertain_inferences": [
      "possible but not directly visible interpretations, or empty list"
    ],
    "student_action": "concise but detailed first-person visual description of what the student/camera/hands are doing",
    "objects": [
      "object descriptions with color, shape, size, labels/text, position, and uncertainty"
    ],
    "scene_layout": "where the student is relative to visible work areas/objects",
    "step_relevance": "how this visual evidence may relate to previous/current/next step without deciding completion or error"
  }}
}}
""".strip()


def normalize_window_description(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return None
    window_description = parsed.get("window_description")
    if not isinstance(window_description, dict):
        if any(key in parsed for key in WINDOW_DESCRIPTION_KEYS):
            window_description = {
                key: parsed.get(key)
                for key in WINDOW_DESCRIPTION_KEYS
                if key in parsed
            }
        else:
            return None

    normalized: Dict[str, Any] = {}
    for key in WINDOW_DESCRIPTION_KEYS:
        value = window_description.get(key)
        if key in ("objects", "uncertain_inferences"):
            if value is None:
                value = []
            elif not isinstance(value, list):
                value = [str(value)]
        elif value is None:
            value = ""
        normalized[key] = value
    return normalized


def video_duration_sec(cap: Any) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0:
        return 0.0
    return float(frame_count) / float(fps)

# Simple Frame window buffer
def extract_window_frames(
    *,
    cap: Any,
    timestamps: List[float],
    jpeg_quality: int,
) -> List[str]:
    frames = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame at {timestamp:.2f}s")
        frames.append(frame_to_base64(frame, jpeg_quality))
    return frames

# Smart Frame Sampler: extract all candidate frames, then select a subset based on visual similarity
def extract_descriptor_frames(
    *,
    cap: Any,
    timestamps: List[float],
    jpeg_quality: int,
    frame_sampler: str,
    sampler_metric: str,
) -> tuple[List[str], List[float], Optional[Dict[str, Any]]]:
    raw_frames = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame at {timestamp:.2f}s")
        raw_frames.append(frame)

    if frame_sampler == "uniform":
        return [frame_to_base64(frame, jpeg_quality) for frame in raw_frames], timestamps, None
    if frame_sampler != "smart":
        raise ValueError(f"Unknown frame sampler: {frame_sampler}")

    selected = smart_select_frames(
        frames=raw_frames,
        timestamps=timestamps,
        metric=sampler_metric,
    )
    selected_frames = [raw_frames[item.index] for item in selected]
    selected_timestamps = [item.timestamp_sec for item in selected]
    selection_log = {
        "sampler": frame_sampler,
        "metric": sampler_metric,
        "candidate_timestamps": timestamps,
        "selected_frames": [item.__dict__ for item in selected],
    }
    return [frame_to_base64(frame, jpeg_quality) for frame in selected_frames], selected_timestamps, selection_log


def build_windows(
    *,
    start_sec: float,
    end_sec: float,
    window_sec: float,
    frame_fps: float,
    frames_per_window: int,
    allow_partial_final_window: bool,
) -> List[Dict[str, Any]]:
    windows = []
    window_start = start_sec
    frame_interval = 1.0 / frame_fps
    while window_start < end_sec:
        timestamps = [
            round(window_start + frame_index * frame_interval, 3)
            for frame_index in range(frames_per_window)
            if window_start + frame_index * frame_interval <= end_sec + 1e-6
        ]
        if not timestamps:
            break
        if len(timestamps) < frames_per_window and not allow_partial_final_window:
            break
        frame_window = [float(timestamps[0]), float(timestamps[-1])]
        windows.append({
            "source_log_index": len(windows) + 1,
            "frame_window": frame_window,
            "frame_timestamps": timestamps,
            "midpoint_sec": round((frame_window[0] + frame_window[1]) / 2, 3),
        })
        window_start = round(window_start + window_sec, 6)
    return windows


def append_jsonl(path: Optional[Path], item: Dict[str, Any]) -> None:
    if not path:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(item) + "\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate descriptor-only VLM window descriptions")
    parser.add_argument("--procedure", required=True, help="Procedure JSON path")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output descriptor JSON path")
    parser.add_argument("--descriptor-log", help="Optional JSONL trace with prompt/raw response")
    parser.add_argument("--api-key", help="OpenRouter API key, or set OPENROUTER_API_KEY")
    parser.add_argument("--model", default="google/gemini-3.1-flash-image-preview", help="Image descriptor model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, help="Exclusive-ish end time; default is video duration")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--frames-per-window", type=int, default=10)
    parser.add_argument("--frame-sampler", choices=["uniform", "smart"], default="uniform")
    parser.add_argument("--sampler-metric", choices=["ssim", "mad"], default="ssim")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--current-step-id",
        type=int,
        default=1,
        help="Fallback current step hint when --step-hints is not supplied",
    )
    parser.add_argument(
        "--step-hints",
        help="Optional JSON with time ranges and current_step_id hints for descriptor prompts",
    )
    parser.add_argument(
        "--allow-partial-final-window",
        action="store_true",
        help="Allow the last window to contain fewer than --frames-per-window frames",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate extraction plan without calling model")
    args = parser.parse_args()

    procedure = load_json(args.procedure)
    step_hints = load_step_hints(args.step_hints)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    duration = video_duration_sec(cap)
    end_sec = min(args.end_sec if args.end_sec is not None else duration, duration)

    windows = build_windows(
        start_sec=max(args.start_sec, 0.0),
        end_sec=end_sec,
        window_sec=args.window_sec,
        frame_fps=args.frame_fps,
        frames_per_window=args.frames_per_window,
        allow_partial_final_window=args.allow_partial_final_window,
    )
    if args.max_windows is not None:
        windows = windows[:args.max_windows]

    print("============================================================")
    print("  DESCRIPTOR EXPERIMENT")
    print("============================================================")
    print(f"  Procedure: {procedure.get('task') or procedure.get('task_name', 'Unknown')} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Duration:  {duration:.1f}s")
    print(f"  Windows:   {len(windows)} ({args.window_sec:.1f}s, {args.frames_per_window} frames each @ {args.frame_fps:.1f}fps)")
    print(f"  Sampler:   {args.frame_sampler}" + (f" ({args.sampler_metric})" if args.frame_sampler == "smart" else ""))
    print(f"  Model:     {args.model}")
    print(f"  Output:    {args.output}")
    print()

    output_path = Path(args.output)
    log_path = Path(args.descriptor_log) if args.descriptor_log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

    if args.dry_run:
        cap.release()
        return

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or --api-key is required")

    described_windows = []
    for window in windows:
        timestamps = window["frame_timestamps"]
        current_step_id = current_step_id_for_window(
            midpoint_sec=float(window["midpoint_sec"]),
            step_hints=step_hints,
            default_step_id=args.current_step_id,
        )
        try:
            frames_base64, selected_timestamps, selection_log = extract_descriptor_frames(
                cap=cap,
                timestamps=timestamps,
                jpeg_quality=args.jpeg_quality,
                frame_sampler=args.frame_sampler,
                sampler_metric=args.sampler_metric,
            )
            prompt = build_descriptor_prompt(
                procedure=procedure,
                frame_timestamps=selected_timestamps,
                current_step_id=current_step_id,
            )
            raw_response = call_openrouter_descriptor(
                api_key=api_key,
                frames_base64=frames_base64,
                prompt=prompt,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_sec=args.timeout_sec,
            )
            parsed = parse_json_response(raw_response)
            window_description = normalize_window_description(parsed)
        except Exception as exc:
            raw_response = ""
            parsed = None
            window_description = None
            selected_timestamps = timestamps
            selection_log = None
            prompt = build_descriptor_prompt(
                procedure=procedure,
                frame_timestamps=selected_timestamps,
                current_step_id=current_step_id,
            )
            print(f"[{window['frame_window'][0]:6.1f}-{window['frame_window'][1]:6.1f}s] ERROR: {exc}")
            append_jsonl(log_path, {
                **window,
                "candidate_frame_timestamps": timestamps,
                "selected_frame_timestamps": selected_timestamps,
                "frame_selection": selection_log,
                "current_step_id_hint": current_step_id,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_response": parsed,
                "window_description": window_description,
                "error": str(exc),
            })
            described_windows.append({
                **window,
                "candidate_frame_timestamps": timestamps,
                "frame_timestamps": selected_timestamps,
                "current_step_id_hint": current_step_id,
                "window_description": None,
            })
            continue

        print(
            f"[{window['frame_window'][0]:6.1f}-{window['frame_window'][1]:6.1f}s] "
            f"{(window_description or {}).get('student_action', '')}"
        )
        described_window = {
            **window,
            "candidate_frame_timestamps": timestamps,
            "frame_timestamps": selected_timestamps,
            "current_step_id_hint": current_step_id,
            "window_description": window_description,
        }
        described_windows.append(described_window)
        append_jsonl(log_path, {
            **window,
            "candidate_frame_timestamps": timestamps,
            "selected_frame_timestamps": selected_timestamps,
            "frame_selection": selection_log,
            "current_step_id_hint": current_step_id,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "window_description": window_description,
        })

    cap.release()

    output = {
        "source": "descriptor_vlm",
        "clip": procedure.get("clip") or Path(args.procedure).stem,
        "task": procedure.get("task") or procedure.get("task_name", "Unknown"),
        "procedure_path": args.procedure,
        "input_video": args.video,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "window_sec": args.window_sec,
        "frame_fps": args.frame_fps,
        "frames_per_window": args.frames_per_window,
        "frame_sampler": args.frame_sampler,
        "sampler_metric": args.sampler_metric if args.frame_sampler == "smart" else None,
        "uses_step_rubric": False,
        "uses_visual_summary": False,
        "windows": described_windows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print()
    print(f"Saved {args.output}")
    if args.descriptor_log:
        print(f"Trace: {args.descriptor_log}")


if __name__ == "__main__":
    main()
