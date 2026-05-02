import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def default_step_rubric_path(procedure_path: str, procedure: Dict[str, Any]) -> Path:
    clip = procedure.get("clip") or procedure.get("video_name") or Path(procedure_path).stem
    safe_clip = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(clip)).strip("-")
    return Path("output/step_rubrics") / f"{safe_clip}.json"


def load_step_rubrics(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    rubrics = data.get("rubrics") if isinstance(data, dict) else data
    if not isinstance(rubrics, list):
        raise ValueError("step rubric JSON must contain a rubrics list")
    return [item for item in rubrics if isinstance(item, dict)]


def save_step_rubrics(
    path: str | Path,
    procedure: Dict[str, Any],
    rubrics: List[Dict[str, Any]],
    model: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": procedure.get("task") or procedure.get("task_name", "Unknown"),
        "clip": procedure.get("clip"),
        "source": "procedure_only",
        "model": model,
        "rubrics": rubrics,
    }
    path.write_text(json.dumps(payload, indent=2))


def generate_step_rubrics(
    api_key: str,
    procedure: Dict[str, Any],
    model: str,
) -> List[Dict[str, Any]]:
    prompt = _build_rubric_prompt(procedure)
    text = _call_openrouter_text(api_key=api_key, prompt=prompt, model=model)
    parsed = _parse_json_response(text)
    if not parsed:
        raise ValueError("rubric model did not return parseable JSON")
    rubrics = parsed.get("rubrics")
    if not isinstance(rubrics, list):
        raise ValueError("rubric model JSON missing rubrics list")
    return _normalize_rubrics(procedure["steps"], rubrics)


def resolve_step_rubrics(
    api_key: Optional[str],
    procedure: Dict[str, Any],
    procedure_path: str,
    explicit_path: Optional[str],
    output_path: Optional[str],
    regenerate: bool,
    model: str,
) -> Tuple[List[Dict[str, Any]], Optional[Path], str]:
    rubric_path = Path(explicit_path) if explicit_path else Path(output_path) if output_path else default_step_rubric_path(procedure_path, procedure)

    if explicit_path and not regenerate:
        return load_step_rubrics(rubric_path), rubric_path, "loaded"

    if rubric_path.exists() and not regenerate:
        return load_step_rubrics(rubric_path), rubric_path, "cached"

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or --api-key is required to generate step rubrics")

    rubrics = generate_step_rubrics(api_key=api_key, procedure=procedure, model=model)
    save_step_rubrics(rubric_path, procedure, rubrics, model)
    return rubrics, rubric_path, "generated"


def _build_rubric_prompt(procedure: Dict[str, Any]) -> str:
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    steps = "\n".join(
        f"{step['step_id']}. {step['description']}" for step in procedure["steps"]
    )
    return f"""
You are creating visual-only step completion rubrics for a procedural task.

Task: {task_name}

Procedure steps:
{steps}

Use only the procedure text above. Do not assume any video-specific timing,
ground truth, transcript, or hidden metadata.

For each step, describe how a vision model could decide that the step is
completed from first-person video frames. Focus on final visible state, state
changes, relative object positions, and what does NOT count as completion.

Return exactly one JSON object and no extra text:
{{
  "rubrics": [
    {{
      "step_id": 1,
      "step_description": "original step text",
      "target_objects": ["objects that should be visually involved"],
      "completion_visual_states": [
        "final visual state that proves completion"
      ],
      "state_changes": [
        "visible change from before to after"
      ],
      "not_completion": [
        "actions that may look related but are only preparation"
      ],
      "timestamp_rule": "first frame where the completed final state is visible",
      "ambiguities": [
        "visual ambiguity to be cautious about"
      ]
    }}
  ]
}}
""".strip()


def _call_openrouter_text(api_key: str, prompt: str, model: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
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


def _normalize_rubrics(
    steps: List[Dict[str, Any]],
    rubrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_step_id = {}
    for rubric in rubrics:
        try:
            step_id = int(rubric.get("step_id"))
        except (TypeError, ValueError):
            continue
        by_step_id[step_id] = rubric

    normalized = []
    for step in steps:
        step_id = int(step["step_id"])
        rubric = by_step_id.get(step_id, {})
        normalized.append({
            "step_id": step_id,
            "step_description": step["description"],
            "target_objects": _as_string_list(rubric.get("target_objects")),
            "completion_visual_states": _as_string_list(rubric.get("completion_visual_states")),
            "state_changes": _as_string_list(rubric.get("state_changes")),
            "not_completion": _as_string_list(rubric.get("not_completion")),
            "timestamp_rule": str(rubric.get("timestamp_rule") or "first frame where the completed final state is visible"),
            "ambiguities": _as_string_list(rubric.get("ambiguities")),
        })
    return normalized


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
