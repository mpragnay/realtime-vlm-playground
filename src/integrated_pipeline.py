"""
Integrated descriptor + reasoner pipeline.

Runs the descriptor and reasoner in a feedback loop so that step completions
detected by the reasoner are propagated back to the descriptor before the next
window is processed. This ensures the descriptor's step-context hint stays
current throughout the video.

Flow per window group:
  1. Reasoner reports current expected step N.
  2. Descriptor processes windows in the group with step N as the hint.
  3. Reasoner reasons over the described windows and may advance to step N+k.
  4. Repeat with the updated step for the next group.

Example:
    .venv/bin/python src/integrated_pipeline.py \\
        --procedure data/clip_procedures/R142-31Aug-RAM.json \\
        --video data/videos_full/R142-31Aug-RAM/Export_py/Video_pitchshift.mp4 \\
        --output output/R142-integrated-events.json \\
        --descriptor-model google/gemini-2.5-flash \\
        --reasoner-model google/gemini-2.5-pro \\
        --descriptor-log output/R142-integrated-descriptor.jsonl \\
        --reasoner-log output/R142-integrated-reasoner.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.descriptor_experiment import (
    build_descriptor_prompt,
    build_windows,
    call_openrouter_descriptor,
    extract_window_frames,
    normalize_window_description,
    parse_json_response,
    video_duration_sec,
)
from src.routing_experiment import RoutingReasoner, make_window_groups, parse_json_response as parse_reasoner_response
from src.step_rubric import load_step_rubrics


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Optional[Path], item: Dict[str, Any]) -> None:
    if not path:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(item) + "\n")


def run_integrated(
    *,
    procedure: Dict[str, Any],
    cap: Any,
    windows: List[Dict[str, Any]],
    reasoner: RoutingReasoner,
    api_key: str,
    descriptor_model: str,
    descriptor_temperature: Optional[float],
    descriptor_top_p: Optional[float],
    descriptor_timeout_sec: int,
    jpeg_quality: int,
    windows_per_call: int,
    descriptor_log_path: Optional[Path],
    reasoner_log_path: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    Process all windows in groups. For each group:
      - Get current expected step from reasoner state.
      - Run descriptor on each window in the group using that step hint.
      - Run reasoner over the described group.
      - Reasoner state (current_step_index) is updated in place.
    """
    all_described_windows: List[Dict[str, Any]] = []
    group_size = max(windows_per_call, 1)

    for group_start in range(0, len(windows), group_size):
        group_windows = windows[group_start:group_start + group_size]

        # Snapshot step hint before describing this group
        current_step = reasoner.current_step()
        current_step_id = int(current_step["step_id"]) if current_step else 1

        described_group: List[Dict[str, Any]] = []
        for window in group_windows:
            timestamps = window["frame_timestamps"]
            frame_window = window["frame_window"]
            prompt = build_descriptor_prompt(
                procedure=procedure,
                frame_timestamps=timestamps,
                current_step_id=current_step_id,
            )

            try:
                frames_base64 = extract_window_frames(
                    cap=cap,
                    timestamps=timestamps,
                    jpeg_quality=jpeg_quality,
                )
                raw_response = call_openrouter_descriptor(
                    api_key=api_key,
                    frames_base64=frames_base64,
                    prompt=prompt,
                    model=descriptor_model,
                    temperature=descriptor_temperature,
                    top_p=descriptor_top_p,
                    timeout_sec=descriptor_timeout_sec,
                )
                parsed = parse_json_response(raw_response)
                window_description = normalize_window_description(parsed)
                error_str = None
            except Exception as exc:
                raw_response = ""
                parsed = None
                window_description = None
                error_str = str(exc)
                print(f"  [descriptor] ERROR {frame_window}: {exc}")

            action_preview = (window_description or {}).get("student_action", "") if window_description else ""
            print(
                f"  [descriptor] step={current_step_id} "
                f"window={frame_window[0]:.1f}-{frame_window[1]:.1f}s  {action_preview[:80]}"
            )

            described_window = {
                **window,
                "current_step_id_hint": current_step_id,
                "window_description": window_description,
            }
            described_group.append(described_window)
            all_described_windows.append(described_window)

            append_jsonl(descriptor_log_path, {
                **window,
                "current_step_id_hint": current_step_id,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_response": parsed,
                "window_description": window_description,
                **({"error": error_str} if error_str else {}),
            })

        # Build a single reasoner group from the described windows
        groups = make_window_groups(described_group, len(described_group))
        if not groups:
            continue
        group = groups[0]
        group_index = group_start // group_size + 1

        reasoner_prompt = reasoner.build_prompt(group)
        try:
            raw_reasoner = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": reasoner.model,
                    "messages": [{"role": "user", "content": reasoner_prompt}],
                    **({"temperature": reasoner.temperature} if reasoner.temperature is not None else {}),
                    **({"top_p": reasoner.top_p} if reasoner.top_p is not None else {}),
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
                    "X-Title": "VLM Orchestrator Integrated Pipeline",
                },
                timeout=90,
            )
            raw_reasoner.raise_for_status()
            reasoner_text = raw_reasoner.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"  [reasoner]   ERROR group={group_index}: {exc}")
            reasoner.log_call(group_index, group, reasoner_prompt, "", None, [])
            continue

        parsed_reasoner = parse_reasoner_response(reasoner_text)
        emitted = reasoner.handle_response(parsed_reasoner, group, reasoner_text)
        reasoner.log_call(group_index, group, reasoner_prompt, reasoner_text, parsed_reasoner, emitted)

        step_after = reasoner.current_step()
        step_id_after = int(step_after["step_id"]) if step_after else "done"
        print(
            f"  [reasoner]   group={group_index} "
            f"window={group.get('frame_window', [0,0])[0]:.1f}-{group.get('frame_window', [0,0])[1]:.1f}s "
            f"emitted={len(emitted)} current_step={step_id_after}"
        )

    return reasoner.events


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Integrated descriptor+reasoner pipeline")
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--descriptor-model", default="google/gemini-3.1-flash-image-preview")
    parser.add_argument("--reasoner-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--step-rubric", help="Optional step rubric JSON path")
    parser.add_argument("--no-step-rubric", action="store_true")
    parser.add_argument("--descriptor-log")
    parser.add_argument("--reasoner-log")
    parser.add_argument("--api-key")
    parser.add_argument("--descriptor-temperature", type=float, default=0.2)
    parser.add_argument("--descriptor-top-p", type=float)
    parser.add_argument("--reasoner-temperature", type=float, default=0.2)
    parser.add_argument("--reasoner-top-p", type=float)
    parser.add_argument("--step-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--error-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--windows-per-call", type=int, default=2,
                        help="Descriptor windows per reasoner call (default 2 = 10s per reasoning step)")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--frames-per-window", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--descriptor-timeout-sec", type=int, default=90)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    procedure = load_json(args.procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")

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
        allow_partial_final_window=False,
    )
    if args.max_windows is not None:
        windows = windows[:args.max_windows]

    print("=" * 60)
    print("  INTEGRATED PIPELINE")
    print("=" * 60)
    print(f"  Procedure:         {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:             {args.video}")
    print(f"  Duration:          {duration:.1f}s")
    print(f"  Windows:           {len(windows)} x {args.window_sec:.1f}s ({args.frames_per_window} frames @ {args.frame_fps:.1f}fps)")
    print(f"  Windows/call:      {args.windows_per_call}")
    print(f"  Descriptor model:  {args.descriptor_model}")
    print(f"  Reasoner model:    {args.reasoner_model}")
    print(f"  Output:            {args.output}")
    print()

    if args.dry_run:
        cap.release()
        return

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or --api-key is required")

    rubrics: List[Dict[str, Any]] = []
    if not args.no_step_rubric:
        rubric_path = args.step_rubric
        if rubric_path and Path(rubric_path).exists():
            rubrics = load_step_rubrics(rubric_path)
            print(f"  Step rubric:       {rubric_path} ({len(rubrics)} steps)")
        else:
            print("  Step rubric:       none (reasoner infers completion cues from step text)")
    else:
        print("  Step rubric:       disabled")
    print()

    descriptor_log_path = Path(args.descriptor_log) if args.descriptor_log else None
    reasoner_log_path = Path(args.reasoner_log) if args.reasoner_log else None
    for log_path in (descriptor_log_path, reasoner_log_path):
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")

    reasoner = RoutingReasoner(
        procedure=procedure,
        rubrics=rubrics,
        api_key=api_key,
        model=args.reasoner_model,
        temperature=args.reasoner_temperature,
        top_p=args.reasoner_top_p,
        step_confidence_threshold=args.step_confidence_threshold,
        error_confidence_threshold=args.error_confidence_threshold,
        windows_per_call=args.windows_per_call,
        reasoner_log_path=None,  # we log manually so we control timing
    )
    # Attach log path directly so log_call works
    if reasoner_log_path:
        reasoner.reasoner_log_path = reasoner_log_path

    events = run_integrated(
        procedure=procedure,
        cap=cap,
        windows=windows,
        reasoner=reasoner,
        api_key=api_key,
        descriptor_model=args.descriptor_model,
        descriptor_temperature=args.descriptor_temperature,
        descriptor_top_p=args.descriptor_top_p,
        descriptor_timeout_sec=args.descriptor_timeout_sec,
        jpeg_quality=args.jpeg_quality,
        windows_per_call=args.windows_per_call,
        descriptor_log_path=descriptor_log_path,
        reasoner_log_path=reasoner_log_path,
    )
    cap.release()

    output = {
        "task": task_name,
        "procedure_path": args.procedure,
        "input_video": args.video,
        "descriptor_model": args.descriptor_model,
        "reasoner_model": args.reasoner_model,
        "events": events,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(events)}")


if __name__ == "__main__":
    main()
