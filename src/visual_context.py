import json
from typing import Any, Dict, List, Optional, Tuple


class VisualContextManager:
    """Builds visual prompts and maintains VLM-produced visual memory."""

    def __init__(
        self,
        steps: List[Dict[str, Any]],
        step_rubrics: Optional[List[Dict[str, Any]]] = None,
        step_rubric_mode: str = "soft",
    ):
        self.steps = steps
        self.step_by_id = {step["step_id"]: step for step in steps}
        self.step_rubric_mode = step_rubric_mode
        self.step_rubrics = {
            int(rubric["step_id"]): rubric
            for rubric in step_rubrics or []
            if isinstance(rubric, dict) and "step_id" in rubric
        }
        self.step_context = {step["step_id"]: "" for step in steps}
        self.window_descriptions: List[Dict[str, Any]] = []
        self.window_context_history: List[Dict[str, Any]] = []

    def build_prompt(
        self,
        task_name: str,
        selected_frames: List[Dict[str, Any]],
        completed_steps: List[int],
        current_step: Optional[Dict[str, Any]],
        next_steps: List[Dict[str, Any]],
    ) -> str:
        frame_lines = [
            f"- image_{idx + 1}: timestamp_sec={frame['timestamp_sec']:.2f}"
            for idx, frame in enumerate(selected_frames)
        ]
        all_steps = "\n".join(
            f"{step['step_id']}. {step['description']}" for step in self.steps
        )
        next_step_text = "\n".join(
            f"{step['step_id']}. {step['description']}" for step in next_steps
        )
        current_step_text = "None; all known procedure steps are complete."
        if current_step:
            current_step_text = f"{current_step['step_id']}. {current_step['description']}"
        input_rubric_guidance, step_rubric_guidance = self.rubric_prompt_guidance()

        return f"""
You are monitoring a procedural task video using visual evidence only.

Task: {task_name}

Camera viewpoint:
- The images are from the student's first-person/egocentric point of view.
- The student's face/body will not be visible. Interpret visible hands,
  camera movement, and manipulated objects as evidence of what the student is
  doing.
- Do not write "no student is visible" as the main action when the first-person
  camera is moving, looking around, approaching an object, or a hand is visible.
  Instead describe the apparent POV action, e.g. "the student looks toward...",
  "the student moves closer to...", or "the student's hand reaches toward...".

Full procedure:
{all_steps}

Procedure state:
- Completed step_ids: {completed_steps}
- Current expected step: {current_step_text}
- Next step to consider:
{next_step_text}

Procedure-only visual completion rubrics:
{self.format_step_rubrics(current_step, next_steps)}

Tentative step-wise visual context from previous calls:
{self.format_step_context()}

Recent visual window context from previous calls:
{self.format_recent_window_context()}

Current frames in chronological order:
{chr(10).join(frame_lines)}

How to use the inputs:
- Use only the attached frames.
- These frames are a short rolling window. Most windows have no event.
{input_rubric_guidance}
- Most windows have no event.
- Use prior visual context as a tentative timeline of what appeared to happen
  before this window. In the current window, describe how that timeline
  continues or changes based on the ordered frames. If an object or action from
  prior context appears in a new state or location, update the description using
  the current frames.
- Use recent visual window context for concrete object continuity: distinguish
  objects by appearance, label/text, size, and location. For example, do not
  merge a small red floor toolbox with a larger red PRO STEEL drawer toolbox.
- Always describe visible objects by appearance first: color, shape, size,
  labels/text, position, and relation to the student. Avoid naming an object as
  "correct" or "wrong" unless the visual evidence and procedure order support it.
- For example, say "a red briefcase-sized toolbox on the floor" instead of
  "the correct toolbox" when correctness is not visually knowable.
- For student_action, describe the visible change across the ordered frames, not
  only the final state. Mention object motion and state changes when visible:
  picked up, carried, lowered, set down, released, opened, closed, moved away
  from, approached, or left in place.
- Do not state hidden state changes as facts. If an object leaves view and later
  reappears, describe only the visible disappearance and reappearance. You may
  speculate about what likely happened, but mark it clearly as uncertain.
- Only report a state change when the visual evidence for that state is visible
  in the frames. If a hand is on a handle, latch, switch, drawer, cover, or
  component but the resulting state is not clearly visible, describe the hand
  contact and say the resulting state is unclear.
- In step context summaries, preserve uncertainty. Do not convert "possibly",
  "appears to", or "unclear" observations into definite facts.
- Maintain context per step: summarize what the student did over time that is
  relevant to that step. If this window completes a step, update that step's
  context, and if the student visibly starts/prepares the next step, add context
  for the next step too.

Step completion rules:
- A step_completion means the action for that procedure step is finished by one
  of the current frames.
- The completed final state must be visible in the current frames.
- Do not report completion when the technician has only started the action, or is still performing the action.
- Prefer the frame timestamp where the completed final state first becomes
  clearly visible.
{step_rubric_guidance}

How to use visual context for step detection:
- Visual context helps you understand continuity: where the student came from,
  which similar-looking objects were previously handled, and what action was
  already underway before this window.
- Do not emit step_completion from prior context alone. A step completion event
  still requires current-frame evidence that the step's final state is visible
  in this window.
- Use visual context to avoid duplicate or premature completions. If context
  says the student only handled a container, searched, approached, or prepared,
  do not treat that as completing a step whose named target object or final
  state is not visible.
- For object acquisition steps, the named target object itself must be visible
  in the student's hand or clearly controlled by the student. Handling a box,
  drawer, bag, panel, or container that may contain the target is preparation,
  not completion.
- If the target object is visually ambiguous, describe the ambiguity in status
  or uncertain_inferences rather than emitting completion.

Error rules:
- An error_detected means the technician starts a wrong action, wrong sequence,
  safety violation, or improper technique.
- Emit it when the current frames show the wrong or contradictory action visibly
  beginning or continuing. Prefer the earliest current-frame timestamp.
- Do not report passive searching, looking around, approaching, or brief
  incidental contact as an error. Use the visual-context rules below to decide
  whether active manipulation is wrong for the current expected step.

How to use visual context for error detection:
- Visual context helps compare the current action against the visual timeline
  and procedure order. Use it to notice when the student returns to a previous
  object, switches to a different similar-looking object, repeats an action, or
  moves away from the expected work area.
- The procedure steps are granular. Treat the current expected step as the main
  task boundary: an object, location, or action from a previous step should only
  remain relevant if it plausibly helps complete the current expected step.
- If the current frames show the student continuing, returning to, or using an
  object/action that belonged to a previous step and is no longer useful for the
  current expected step, infer this as possible wrong_action or wrong_sequence
  evidence.
- Do not emit error_detected from prior context alone. The wrong action must
  visibly begin or continue in the current frames.
- Use visual context to distinguish normal preparation from an error.
- Treat contradictions in the visual timeline as possible error evidence. For
  example, if prior context showed the student apparently preparing to use one
  object or area, but current frames show the student putting it away, abandoning
  it, switching to a different similar-looking object, or doing an incompatible
  action, describe that contradiction.
- If a prior-step object/action is visibly carried forward into the current
  step, ask whether it contributes to the current step's completion. If not,
  and the interaction is more than brief incidental contact, emit error_detected
  with the earliest current-frame timestamp where that irrelevant action begins
  or continues.
- If prior context only speculated that an object would be used, and current
  frames show the student not using it or moving to another object, do not state
  that as a definite error by itself. Mark it as possible_error_not_enough_evidence
  unless the current frames clearly show a wrong object/action/sequence.
- Stronger error evidence comes from visible contradiction plus current action:
  an object previously prepared is set aside while a different object is used;
  a container believed relevant is ignored and another similar container is
  opened; a switch/part is manipulated out of expected order; or the student
  reverses a just-completed action without the procedure calling for it.
- If a possible error depends on an uncertain earlier observation, keep it in
  status as possible_error_not_enough_evidence rather than emitting an error.

Return exactly one JSON object with this schema and no extra text:
{{
  "events": [
    {{
      "type": "step_completion",
      "timestamp_sec": 12.5,
      "step_id": 1,
      "confidence": 0.0,
      "description": "visible completed state"
    }},
    {{
      "type": "error_detected",
      "timestamp_sec": 12.5,
      "confidence": 0.0,
      "error_type": "wrong_action",
      "severity": "warning",
      "description": "what wrong action visibly began",
      "spoken_response": "brief corrective instruction"
    }}
  ],
  "status": {{
    "type": "step_in_progress",
    "step_id": 1,
    "confidence": 0.0,
    "description": "what appears to be happening"
  }},
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
    "step_relevance": "how this visual evidence may relate to the current or next step, without overclaiming"
  }},
  "step_context_updates": [
    {{
      "step_id": current_step_id,
      "summary": "updated tentative visual history for this step, combining prior context and current frames"
    }}
  ],
  "summary": "brief visual observation of this window"
}}

Allowed event types: step_completion, error_detected.
Allowed status type values: step_in_progress, no_action, uncertain, possible_error_not_enough_evidence, idle_or_waiting.
Allowed error_type values: wrong_action, wrong_sequence, safety_violation, improper_technique, other.
Allowed severity values: info, warning, critical.
If there are no events, return {{"events": [], "status": {{"type": "...", "description": "..."}}, "summary": "..."}}.
""".strip()

    def rubric_prompt_guidance(self) -> Tuple[str, str]:
        if self.step_rubric_mode == "strict":
            return (
                "- The current step rubric is the authority for what visual evidence completes\n"
                "  the expected step. Do not invent completion just because a step is expected.",
                "- Before emitting step_completion, explicitly use the current step rubric:\n"
                "  identify the required final visual state, compare it against the current\n"
                "  frames, and emit only if at least one completion_visual_state is visibly true.\n"
                "- If the step is not complete, status.description must say which required visual\n"
                "  evidence from the rubric is missing or still ambiguous.",
            )
        return (
            "- Use the current step rubric as guidance for what visual evidence may complete\n"
            "  the expected step. The rubric is not exhaustive: equivalent visible evidence\n"
            "  can also count if it clearly satisfies the procedure step. Do not invent\n"
            "  completion just because a step is expected.",
            "- For emitting step_completion, use the current step rubric as guidance:\n"
            "  identify the likely required final visual state, compare it against the\n"
            "  current frames, and emit if the frames show that state or an equivalent\n"
            "  visual state that clearly completes the step.\n"
            "- If the step is not complete, status.description must say which expected\n"
            "  visual evidence is missing or still ambiguous.",
        )

    def state_snapshot(
        self,
        completed_steps: List[int],
        current_step_id: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "completed_steps": completed_steps,
            "current_step_id": current_step_id,
            "step_context": {
                step_id: summary
                for step_id, summary in self.step_context.items()
                if summary
            },
            "recent_window_context": list(self.window_context_history[-6:]),
        }

    @staticmethod
    def response_log_context(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return {}
        return {
            "window_description": parsed.get("window_description"),
            "step_context_updates": parsed.get("step_context_updates", []),
            "status": parsed.get("status"),
            "summary": parsed.get("summary"),
            "events": parsed.get("events", []),
        }

    def format_step_rubrics(
        self,
        current_step: Optional[Dict[str, Any]],
        next_steps: List[Dict[str, Any]],
    ) -> str:
        if not self.step_rubrics:
            return "- No procedure-only visual completion rubrics available."

        rubrics = []
        seen = set()
        for step in [current_step] + list(next_steps):
            if not step:
                continue
            step_id = int(step["step_id"])
            if step_id in seen:
                continue
            seen.add(step_id)
            rubric = self.step_rubrics.get(step_id)
            if rubric:
                rubrics.append(rubric)

        if not rubrics:
            return "- No rubric is available for the current/next step."

        return json.dumps(rubrics, indent=2)

    def update_from_response(
        self,
        parsed: Dict[str, Any],
        frame_times: List[float],
    ) -> Tuple[str, List[Tuple[int, str]]]:
        window_description = parsed.get("window_description")
        if isinstance(window_description, dict):
            description_text = self.compact_window_description(window_description)
        elif isinstance(window_description, str):
            description_text = window_description.strip()
        else:
            description_text = (parsed.get("summary") or "").strip()

        updates = parsed.get("step_context_updates", [])
        if not isinstance(updates, list):
            updates = []

        applied_updates = []
        if description_text:
            self.window_descriptions.append({
                "frame_window": [frame_times[0], frame_times[-1]],
                "description": description_text,
            })
            self.window_descriptions = self.window_descriptions[-20:]
            self.window_context_history.append({
                "frame_window": [frame_times[0], frame_times[-1]],
                "window_description": window_description,
            })
            self.window_context_history = self.window_context_history[-8:]

        for update in updates:
            if not isinstance(update, dict):
                continue
            step_id = self._as_int(update.get("step_id"))
            summary = update.get("summary")
            if step_id not in self.step_by_id or not isinstance(summary, str):
                continue
            summary = summary.strip()
            if not summary or self.is_placeholder_summary(summary):
                continue
            self.step_context[step_id] = summary[:900]
            applied_updates.append((step_id, self.step_context[step_id]))

        return description_text, applied_updates

    def format_step_context(self) -> str:
        lines = []
        for step in self.steps:
            step_id = step["step_id"]
            summary = self.step_context.get(step_id, "")
            if summary:
                lines.append(f"- Step {step_id} ({step['description']}): {summary}")
        if not lines:
            return "- No prior visual context yet."
        return "\n".join(lines)

    def format_recent_window_context(self) -> str:
        if not self.window_context_history:
            return "- No prior window context yet."
        lines = []
        for item in self.window_context_history[-6:]:
            description = item.get("window_description", {})
            if isinstance(description, dict):
                action = description.get("student_action") or ""
                start_state = description.get("start_state") or ""
                end_state = description.get("end_state") or ""
                motion = description.get("motion_or_change") or ""
                uncertain = description.get("uncertain_inferences") or []
                if isinstance(uncertain, list):
                    uncertain = "; ".join(str(item) for item in uncertain[:3])
                objects = description.get("objects") or []
                if isinstance(objects, list):
                    objects = "; ".join(str(obj) for obj in objects[:4])
                scene_layout = description.get("scene_layout") or ""
                step_relevance = description.get("step_relevance") or ""
                detail = (
                    f"start={start_state}; end={end_state}; change={motion}; "
                    f"uncertain={uncertain}; "
                    f"action={action}; objects={objects}; "
                    f"scene_layout={scene_layout}; step_relevance={step_relevance}"
                )
            else:
                detail = str(description)
            lines.append(
                f"- {item['frame_window'][0]:.1f}-{item['frame_window'][1]:.1f}s: {detail[:900]}"
            )
        return "\n".join(lines)

    @staticmethod
    def compact_window_description(window_description: Dict[str, Any]) -> str:
        parts = []
        for key in (
            "start_state",
            "end_state",
            "motion_or_change",
            "uncertain_inferences",
            "student_action",
            "objects",
            "scene_layout",
            "step_relevance",
        ):
            value = window_description.get(key)
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value if item)
            if isinstance(value, str) and value.strip():
                parts.append(f"{key}: {value.strip()}")
        return " | ".join(parts)

    @staticmethod
    def is_placeholder_summary(summary: str) -> bool:
        lowered = summary.lower()
        placeholders = {
            "updated tentative visual history",
            "combining prior context and current frames",
            "no prior visual context",
            "unknown",
            "n/a",
        }
        return any(item in lowered for item in placeholders)

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
