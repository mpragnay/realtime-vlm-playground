"""
Two-stage descriptor/reasoner routing experiment.

This experimental script consumes descriptor-agent 5-second window descriptions
and asks a text-only reasoning model to emit procedure events over
non-overlapping groups of consecutive descriptor windows. It intentionally does
not look at images; use it to test whether clean visual descriptions are enough
for event detection before wiring this into the streaming runtime.

Example:
    .venv/bin/python src/routing_experiment.py \
        --procedure data/clip_procedures/z045-june-24-22-dslr.json \
        --descriptions output/z045-window-descriptions-mock.json \
        --output output/z045-routing-events.json \
        --reasoner-log output/z045-routing-reasoner.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.step_rubric import load_step_rubrics


def call_openrouter_text(
    *,
    api_key: str,
    prompt: str,
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Routing Experiment",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    resp = requests.post(url, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
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


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def step_lines(steps: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{step['step_id']}. {step['description']}" for step in steps)


def compact_window_description(window: Optional[Dict[str, Any]]) -> str:
    if not isinstance(window, dict):
        return "No descriptor output for this window."
    parts = []
    for key in (
        "start_state",
        "end_state",
        "motion_or_change",
        "student_action",
        "objects",
        "scene_layout",
        "step_relevance",
        "uncertain_inferences",
    ):
        value = window.get(key)
        if value is None or value == []:
            continue
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")
    return "\n".join(parts) if parts else "Descriptor output was empty."


def make_window_groups(windows: List[Dict[str, Any]], windows_per_call: int) -> List[Dict[str, Any]]:
    groups = []
    group_size = max(windows_per_call, 1)
    for start in range(0, len(windows), group_size):
        group_windows = windows[start:start + group_size]
        if not group_windows:
            continue
        start_window = group_windows[0].get("frame_window") or []
        end_window = group_windows[-1].get("frame_window") or []
        if len(start_window) == 2 and len(end_window) == 2:
            frame_window = [float(start_window[0]), float(end_window[1])]
        else:
            frame_window = None
        frame_timestamps = []
        for window in group_windows:
            timestamps = window.get("frame_timestamps")
            if isinstance(timestamps, list):
                frame_timestamps.extend(timestamps)
        midpoint = None
        if frame_window:
            midpoint = round((frame_window[0] + frame_window[1]) / 2, 3)
        groups.append({
            "group_index": len(groups) + 1,
            "frame_window": frame_window,
            "frame_timestamps": frame_timestamps,
            "midpoint_sec": midpoint,
            "windows": group_windows,
        })
    return groups


class RoutingReasoner:
    def __init__(
        self,
        *,
        procedure: Dict[str, Any],
        rubrics: Optional[List[Dict[str, Any]]],
        api_key: str,
        model: str,
        temperature: Optional[float],
        top_p: Optional[float],
        step_confidence_threshold: float,
        error_confidence_threshold: float,
        windows_per_call: int,
        reasoner_log_path: Optional[str],
    ) -> None:
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]
        self.step_by_id = {int(step["step_id"]): step for step in self.steps}
        rubrics = rubrics or []
        self.rubric_by_step_id = {
            int(rubric["step_id"]): rubric for rubric in rubrics if "step_id" in rubric
        }
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.step_confidence_threshold = step_confidence_threshold
        self.error_confidence_threshold = error_confidence_threshold
        self.windows_per_call = windows_per_call
        self.completed_steps: set[int] = set()
        self.current_step_index = 0
        self.step_summaries: Dict[int, str] = {}
        self.events: List[Dict[str, Any]] = []
        self.reasoner_log_path = Path(reasoner_log_path) if reasoner_log_path else None
        if self.reasoner_log_path:
            self.reasoner_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.reasoner_log_path.write_text("")

    def run(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = make_window_groups(windows, self.windows_per_call)
        for index, group in enumerate(groups, start=1):
            if self.current_step_index >= len(self.steps):
                break
            self.process_group(group, index=index)
        return self.events

    def process_group(self, group: Dict[str, Any], index: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.current_step_index >= len(self.steps):
            return []
        prompt = self.build_prompt(group)
        raw_response = call_openrouter_text(
            api_key=self.api_key,
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        parsed = parse_json_response(raw_response)
        emitted = self.handle_response(parsed, group, raw_response)
        self.log_call(index or 0, group, prompt, raw_response, parsed, emitted)
        return emitted

    def build_prompt(self, group: Dict[str, Any]) -> str:
        current_step = self.current_step()
        next_steps = self.steps[self.current_step_index + 1:self.current_step_index + 2]
        relevant_rubrics = self.relevant_rubrics(current_step, next_steps)
        rubric_text = self.rubric_prompt_text(relevant_rubrics)

        current_text = "\n\n".join(
            self.format_window_context(item) for item in group.get("windows", [])
        ) or "No descriptor windows supplied."

        return f"""
You are the text-only reasoning stage in a two-agent visual procedure detector.

The descriptor agent has already looked at the images. You cannot see images.
Use only the textual window descriptions below, the procedure, the current
running step summary, and any provided rubrics.

Task: {self.task_name}

Full procedure:
{step_lines(self.steps)}

Procedure state:
- Completed step_ids: {sorted(self.completed_steps)}
- Current expected step: {self.format_step(current_step)}
- Next step to consider:
{chr(10).join(self.format_step(step) for step in next_steps) or "None"}
- Running summary for current step:
{self.current_step_summary_text(current_step)}

{rubric_text}

Current reasoner interval:
- frame_window: {group.get("frame_window")}
- midpoint_sec: {self.group_midpoint(group)}
- allowed event timestamp_sec values from descriptor windows: {self.allowed_event_timestamps(group)}

Descriptor windows inside this interval:
{current_text}

General reasoning rules:
- The descriptor text is visual evidence, not ground truth. It may overclaim.
- An event can be a step_completion or an error_detected.
- Each descriptor window has its own midpoint. For every event, set
  timestamp_sec to the midpoint of the descriptor window where the visual
  evidence for that event appears.
- Every emitted event must include a concise reason grounded in the descriptor 
  that is based on the provided visual context, or for a catch-up give the visual evidence.
- Maintain the running summary for the current expected step. It is a form of memory for the text reasoner.
- The running summary should combine previous summary and current descriptor
  evidence into a concise visual history of what has happened for the current
  step so far. Preserve uncertainty and do not turn speculation into fact.

Step completion rules:
- A step_completion means the action for the current expected procedure step is
  finished within the current descriptor interval.
- Use only the descriptor window text, procedure order, current step summary,
  and provided rubric if present. Do not invent visual evidence that the
  descriptor did not state.
- If a rubric is provided, use it as guidance, not as a hard checklist. If no
  rubric is provided, reason from the step text: what visible action or stable
  state would normally constitute this step being finished?
- Classify the evidence as matched_phase: state_start_visual,
  state_during_visual, state_end_visual, not_completion, or catch_up.
- Emit step_completion only when the evidence matches a reasonable
  state_end_visual for the step, or when catch_up is strongly justified by
  later-step evidence.
- Do not emit completion for state_start_visual, state_during_visual, or
  not_completion.
- The normal completion target is a stable final state, release, separation,
  closure, placement, or completed control motion that would satisfy the step.
- For steps whose true device/internal state is not visually observable,
  completion may be the visible completion of the control/action itself. For
  example, a finger moves the OFF/ON switch and releases, even if the LCD/power
  state is not visible.
- Do not require visual evidence that cannot plausibly appear in the video, but
  require the descriptor to state a visible completed action or stable end state.
- Do not make hard assumptions that every possible confirmation cue must occur.
  If the step can reasonably be considered complete from the visible action
  described, emit completion and explain the practical visual evidence used.
- Catch-up is allowed only when the current expected step was already underway
  and the current descriptor interval clearly shows a later step
  underway/completed, making it very likely the expected step finished between
  descriptor windows.
- If completion is inferred by catch-up, the description and reason must say
  what later visual evidence implies the missed completion.
- If the descriptor says the action is still being performed, still being held,
  still being adjusted, occluded, ambiguous, or only prepared, do not emit
  completion.
- You may emit multiple step_completion events in one response if the descriptor
  interval clearly shows multiple procedure steps completing. They must be in
  procedure order, starting from the current expected step, and each emitted step
  needs its own visual evidence and reason.
- The event description must quote or paraphrase the visual context that proves
  completion, not merely restate the step name.
- Output the midpoint of the descriptor window containing the completion
  evidence as timestamp_sec. For example, if the evidence is in Window
  [35.0, 39.5] midpoint=37.25s, output timestamp_sec=37.25.

Error detection rules:
- An error_detected means the technician performs an action that has no relation
  to the current expected step or uses the wrong object for the step.
- Emit it when the current descriptor windows show the wrong or contradictory
  action visibly beginning. Use the midpoint of the
  descriptor window where that error action appears.
- Wrong object/action examples: acting on a lens hood when the current step
  requires the lens cover; manipulating a battery/card compartment when the
  current step requires turning the DSLR on; using a toolbox/component unrelated
  to the current expected step.
- Do not report passive searching, looking around, approaching, or brief
  incidental contact as an error. Active manipulation is what matters: picking
  up, attaching, detaching, inserting, removing, opening, closing, pressing,
  switching, seating, or otherwise changing the state of an unrelated object.
- Do not report apparent later-step progress as an error. If the current
  descriptor windows appear to show a later procedure step, treat that as
  possible detector/state drift or missed prior completion, not as an
  error_detected event.
- Often a wrong action is followed by a reverse action where the student undoes
  the mistake. If the descriptor interval shows both the wrong action and its
  reversal, output two error_detected events: one for the wrong action and one
  for the reverse/undo action. If they appear in different descriptor windows,
  use each descriptor window's midpoint. If both appear in the same descriptor
  window, use the same descriptor window midpoint for both.
- Every error_detected event must include a description quoting/paraphrasing
  the visual context and a reason explaining why the action is unrelated or
  wrong for the current expected step.

Return exactly one JSON object and no extra text:
{{
  "events": [
    {{
      "type": "step_completion",
      "step_id": 1,
      "timestamp_sec": {self.allowed_event_timestamps(group)[0]},
      "confidence": 0.0,
      "description": "visual context that led you to believe this step completed",
      "matched_phase": "state_end_visual",
      "reason": "why the descriptor proves the step/action completed"
    }},
    {{
      "type": "error_detected",
      "timestamp_sec": {self.allowed_event_timestamps(group)[0]},
      "confidence": 0.0,
      "error_type": "wrong_action",
      "description": "visual context that led you to believe an error happened",
      "reason": "why this action is unrelated or wrong for the current expected step"
    }}
  ],
  "status": {{
    "type": "step_in_progress",
    "step_id": {current_step.get("step_id") if current_step else "null"},
    "description": "brief reason no event was emitted"
  }},
  "step_summary_updates": [
    {{
      "step_id": {current_step.get("step_id") if current_step else "null"},
      "summary": "updated running visual summary for this step after the current interval"
    }}
  ]
}}
Allowed event types: step_completion, error_detected.
Allowed error_type values: wrong_action, safety_violation, improper_technique, other.
If there are no events, return {{"events": [], "status": {{"type": "...", "description": "..."}}, "step_summary_updates": [...]}}.
""".strip()

    def handle_response(
        self,
        parsed: Optional[Dict[str, Any]],
        group: Dict[str, Any],
        raw_response: str,
    ) -> List[Dict[str, Any]]:
        if not parsed or not isinstance(parsed.get("events"), list):
            if isinstance(parsed, dict):
                self.update_step_summaries(parsed)
            return []

        emitted: List[Dict[str, Any]] = []
        step_events = [
            event for event in parsed["events"]
            if isinstance(event, dict) and event.get("type") == "step_completion"
        ]
        for event in step_events:
            step_id = self.as_int(event.get("step_id"))
            current = self.current_step()
            if not current or step_id != int(current["step_id"]):
                continue
            confidence = self.as_float(event.get("confidence"), 0.0)
            if confidence < self.step_confidence_threshold:
                continue
            timestamp_sec = self.event_timestamp(event, group)
            output = {
                "timestamp_sec": timestamp_sec,
                "type": "step_completion",
                "step_id": step_id,
                "confidence": confidence,
                "description": event.get("description") or current["description"],
                "source": "video",
                "reason": event.get("reason", ""),
                "reasoner_observation": raw_response[:1200],
            }
            matched_phase = event.get("matched_phase")
            if isinstance(matched_phase, str):
                output["matched_phase"] = matched_phase
            self.events.append(output)
            emitted.append(output)
            self.completed_steps.add(step_id)
            while (
                self.current_step_index < len(self.steps)
                and int(self.steps[self.current_step_index]["step_id"]) in self.completed_steps
            ):
                self.current_step_index += 1

        for event in parsed["events"]:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            confidence = self.as_float(event.get("confidence"), 0.0)
            if event_type == "error_detected":
                if confidence < self.error_confidence_threshold:
                    continue
                timestamp_sec = self.event_timestamp(event, group)
                output = {
                    "timestamp_sec": timestamp_sec,
                    "type": "error_detected",
                    "confidence": confidence,
                    "error_type": event.get("error_type") or "wrong_action",
                    "description": event.get("description") or "wrong action detected",
                    "spoken_response": event.get("spoken_response") or "Stop and return to the current step.",
                    "source": "video",
                    "reason": event.get("reason", ""),
                    "reasoner_observation": raw_response[:1200],
                }
                self.events.append(output)
                emitted.append(output)
        self.update_step_summaries(parsed)
        return emitted

    def log_call(
        self,
        index: int,
        group: Dict[str, Any],
        prompt: str,
        raw_response: str,
        parsed: Optional[Dict[str, Any]],
        emitted: List[Dict[str, Any]],
    ) -> None:
        if not self.reasoner_log_path:
            return
        record = {
            "log_index": index,
            "frame_window": group.get("frame_window"),
            "midpoint_sec": self.group_midpoint(group),
            "completed_steps_after": sorted(self.completed_steps),
            "current_step_id_after": self.current_step().get("step_id") if self.current_step() else None,
            "step_summaries_after": self.step_summaries,
            "current_group": group,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "emitted_events": emitted,
        }
        with self.reasoner_log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def current_step(self) -> Optional[Dict[str, Any]]:
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    def relevant_rubrics(
        self,
        current_step: Optional[Dict[str, Any]],
        next_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = [
            self.rubric_by_step_id.get(int(current_step["step_id"])) if current_step else None,
            *[self.rubric_by_step_id.get(int(step["step_id"])) for step in next_steps],
        ]
        return [rubric for rubric in candidates if rubric]

    @staticmethod
    def rubric_prompt_text(relevant_rubrics: List[Dict[str, Any]]) -> str:
        if relevant_rubrics:
            return "Relevant visual rubrics:\n" + json.dumps(relevant_rubrics, indent=2)
        return (
            "Relevant visual rubrics:\n"
            "No precomputed rubric is provided. Infer flexible visual completion cues from the current step text, "
            "the running summary, and the descriptor windows."
        )

    def update_step_summaries(self, parsed: Dict[str, Any]) -> None:
        updates = parsed.get("step_summary_updates")
        if not isinstance(updates, list):
            return
        for update in updates:
            if not isinstance(update, dict):
                continue
            step_id = self.as_int(update.get("step_id"))
            summary = update.get("summary")
            if step_id is None or not isinstance(summary, str) or not summary.strip():
                continue
            self.step_summaries[step_id] = summary.strip()

    def current_step_summary_text(self, current_step: Optional[Dict[str, Any]]) -> str:
        if not current_step:
            return "No current step."
        step_id = int(current_step["step_id"])
        summary = self.step_summaries.get(step_id)
        if not summary:
            return "No running summary for this step yet."
        return f"Step {step_id}: {summary}"

    @staticmethod
    def format_step(step: Optional[Dict[str, Any]]) -> str:
        if not step:
            return "None"
        return f"{step['step_id']}. {step['description']}"

    @staticmethod
    def format_window_context(window: Dict[str, Any]) -> str:
        return (
            f"Window {window.get('frame_window')} midpoint={window.get('midpoint_sec')}s\n"
            f"{compact_window_description(window.get('window_description'))}"
        )

    @staticmethod
    def window_midpoint(window: Dict[str, Any]) -> float:
        midpoint = window.get("midpoint_sec")
        if isinstance(midpoint, (int, float)):
            return float(midpoint)
        frame_window = window.get("frame_window")
        if isinstance(frame_window, list) and len(frame_window) == 2:
            return round((float(frame_window[0]) + float(frame_window[1])) / 2, 3)
        return 0.0

    @staticmethod
    def group_midpoint(group: Dict[str, Any]) -> float:
        midpoint = group.get("midpoint_sec")
        if isinstance(midpoint, (int, float)):
            return float(midpoint)
        frame_window = group.get("frame_window")
        if isinstance(frame_window, list) and len(frame_window) == 2:
            return round((float(frame_window[0]) + float(frame_window[1])) / 2, 3)
        return 0.0

    def allowed_event_timestamps(self, group: Dict[str, Any]) -> List[float]:
        timestamps = []
        for window in group.get("windows", []):
            if isinstance(window, dict):
                timestamps.append(round(self.window_midpoint(window), 3))
        return timestamps or [round(self.group_midpoint(group), 3)]

    def event_timestamp(self, event: Dict[str, Any], group: Dict[str, Any]) -> float:
        requested = self.as_float(event.get("timestamp_sec"), self.group_midpoint(group))
        allowed = self.allowed_event_timestamps(group)
        if not allowed:
            return round(self.group_midpoint(group), 3)
        return min(allowed, key=lambda timestamp: abs(timestamp - requested))

    @staticmethod
    def as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run text-only descriptor/reasoner routing experiment")
    parser.add_argument("--procedure", required=True, help="Procedure JSON path")
    parser.add_argument("--descriptions", required=True, help="Descriptor mock JSON path")
    parser.add_argument("--step-rubric", help="Optional step rubric JSON path")
    parser.add_argument("--output", required=True, help="Output events JSON path")
    parser.add_argument("--reasoner-log", help="Optional reasoner JSONL trace")
    parser.add_argument("--api-key", help="OpenRouter API key, or set OPENROUTER_API_KEY")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview", help="Text reasoner model")
    parser.add_argument("--temperature", type=float, default=0.2, help="Reasoner temperature")
    parser.add_argument("--top-p", type=float, help="Reasoner top_p")
    parser.add_argument("--step-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--error-confidence-threshold", type=float, default=0.60)
    parser.add_argument(
        "--windows-per-call",
        type=int,
        default=2,
        help="Number of consecutive descriptor windows per reasoner call; default gives 0-10, 10-20, ...",
    )
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling the reasoner")
    args = parser.parse_args()

    procedure = load_json(args.procedure)
    descriptions = load_json(args.descriptions)
    rubrics = load_step_rubrics(args.step_rubric) if args.step_rubric else []
    windows = descriptions.get("windows", [])
    if not isinstance(windows, list):
        raise ValueError("descriptions JSON must contain a windows list")

    if args.start_sec is not None:
        windows = [w for w in windows if RoutingReasoner.window_midpoint(w) >= args.start_sec]
    if args.end_sec is not None:
        windows = [w for w in windows if RoutingReasoner.window_midpoint(w) <= args.end_sec]
    if args.max_windows is not None:
        windows = windows[:args.max_windows]

    print("============================================================")
    print("  ROUTING EXPERIMENT")
    print("============================================================")
    print(f"  Procedure:    {procedure.get('task') or procedure.get('task_name', 'Unknown')} ({len(procedure['steps'])} steps)")
    groups = make_window_groups(windows, args.windows_per_call)
    print(f"  Descriptions: {args.descriptions} ({len(windows)} windows selected, {len(groups)} reasoner calls)")
    if args.step_rubric:
        print(f"  Rubric:       {args.step_rubric} ({len(rubrics)} steps)")
    else:
        print("  Rubric:       disabled; reasoner infers completion cues from step text")
    print(f"  Model:        {args.model}")
    print(f"  Output:       {args.output}")
    print()

    if args.dry_run:
        return

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or --api-key is required")

    reasoner = RoutingReasoner(
        procedure=procedure,
        rubrics=rubrics,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        step_confidence_threshold=args.step_confidence_threshold,
        error_confidence_threshold=args.error_confidence_threshold,
        windows_per_call=max(args.windows_per_call, 1),
        reasoner_log_path=args.reasoner_log,
    )
    events = reasoner.run(windows)

    output = {
        "task": procedure.get("task") or procedure.get("task_name", "Unknown"),
        "procedure_path": args.procedure,
        "descriptor_source": args.descriptions,
        "step_rubric": args.step_rubric,
        "reasoner_model": args.model,
        "events": events,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}")
    print(f"Events: {len(events)}")


if __name__ == "__main__":
    main()
