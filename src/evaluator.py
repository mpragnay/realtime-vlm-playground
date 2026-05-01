"""
VLM Orchestrator - Evaluation Module
Compares candidate output against ground truth annotations.

Scores:
  - Step completion:  precision, recall, F1 (matched by step_id + timestamp ±tolerance)
  - Error detection:  precision, recall, F1 (matched by timestamp ±tolerance)
  - Idle detection:   precision, recall, F1 (matched by overlap with GT idle periods)
  - Detection latency: mean and max detection_delay_sec across all events
  - Spoken response:  present/missing for error events (quality reviewed manually)

Usage:
    python -m src.evaluator --predicted output/events.json --ground-truth data/ground_truth_sample/R066-15July-Circuit-Breaker-part2.json
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class EvaluationMetrics:
    # Step completion
    step_precision: float
    step_recall: float
    step_f1: float
    total_gt_steps: int
    step_tp: int
    step_fp: int
    step_fn: int

    # Error detection
    error_precision: float
    error_recall: float
    error_f1: float
    total_gt_errors: int
    error_tp: int
    error_fp: int
    error_fn: int

    # Idle detection
    idle_precision: float
    idle_recall: float
    idle_f1: float
    total_gt_idle_periods: int
    idle_tp: int
    idle_fp: int
    idle_fn: int

    # Latency
    mean_detection_delay_sec: float
    max_detection_delay_sec: float
    p50_detection_delay_sec: float
    p90_detection_delay_sec: float


def load_json_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        return json.load(f)


def _min_distance_match(
    pairs: List[Tuple[int, int, float]],
    num_pred: int,
    num_gt: int,
) -> Tuple[int, int, int]:
    """
    Optimal greedy bipartite matching: sort all valid (pred, gt) pairs by
    distance, then assign closest-first. This avoids the order-dependent
    bias of iterating predictions sequentially.

    Args:
        pairs: list of (pred_idx, gt_idx, distance) for all valid matches
        num_pred: total number of predictions
        num_gt: total number of ground truth items

    Returns: (tp, fp, fn)
    """
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    matched_pred = set()
    matched_gt = set()
    for pi, gi, _ in pairs_sorted:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
    tp = len(matched_gt)
    fp = num_pred - tp
    fn = num_gt - tp
    return tp, fp, fn


def _match_steps(pred: List[Dict], gt: List[Dict], tol: float) -> Tuple[int, int, int]:
    """Match step_completion by step_id + timestamp (closest-first)."""
    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.get("step_id") != g.get("step_id"):
                continue
            dist = abs(p.get("timestamp_sec", 0) - g.get("timestamp_sec", 0))
            if dist <= tol:
                pairs.append((pi, gi, dist))
    return _min_distance_match(pairs, len(pred), len(gt))


def _match_errors(pred: List[Dict], gt: List[Dict], tol: float) -> Tuple[int, int, int]:
    """Match error_detected by timestamp (closest-first)."""
    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            dist = abs(p.get("timestamp_sec", 0) - g.get("timestamp_sec", 0))
            if dist <= tol:
                pairs.append((pi, gi, dist))
    return _min_distance_match(pairs, len(pred), len(gt))


def _match_idles(pred: List[Dict], gt_periods: List[Dict]) -> Tuple[int, int, int]:
    """Match idle_detected by overlap with GT idle periods (closest to midpoint first)."""
    if not gt_periods:
        return 0, len(pred), 0
    pairs = []
    for pi, p in enumerate(pred):
        t = p.get("timestamp_sec", 0)
        for gi, g in enumerate(gt_periods):
            start = g.get("start_sec", 0)
            end = g.get("end_sec", 0)
            if start <= t <= end:
                midpoint = (start + end) / 2
                dist = abs(t - midpoint)
                pairs.append((pi, gi, dist))
    return _min_distance_match(pairs, len(pred), len(gt_periods))


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _latency_score(delays: List[float], max_acceptable: float = 10.0) -> float:
    """
    Score latency on 0–1 scale. Lower delay = higher score.
    0s delay = 1.0, max_acceptable delay = 0.0, linear interpolation.
    """
    if not delays:
        return 0.0
    mean_delay = statistics.mean(delays)
    score = max(0.0, 1.0 - mean_delay / max_acceptable)
    return round(score, 3)


def evaluate(
    predicted_log_path: str,
    ground_truth_path: str,
    time_tolerance_sec: float = 5.0,
    verbose: bool = False,
) -> EvaluationMetrics:
    predicted = load_json_file(predicted_log_path)
    ground_truth = load_json_file(ground_truth_path)

    pred_events = predicted.get("events", [])
    gt_events = ground_truth.get("events", [])
    gt_idles = ground_truth.get("idle_periods", [])

    pred_steps = [e for e in pred_events if e.get("type") == "step_completion"]
    pred_errors = [e for e in pred_events if e.get("type") == "error_detected"]
    pred_idles = [e for e in pred_events if e.get("type") == "idle_detected"]

    gt_steps = [e for e in gt_events if e.get("type") == "step_completion"]
    gt_errors = [e for e in gt_events if e.get("type") == "error_detected"]

    # Match events
    s_tp, s_fp, s_fn = _match_steps(pred_steps, gt_steps, time_tolerance_sec)
    e_tp, e_fp, e_fn = _match_errors(pred_errors, gt_errors, time_tolerance_sec)
    i_tp, i_fp, i_fn = _match_idles(pred_idles, gt_idles)

    s_p, s_r, s_f1 = _prf(s_tp, s_fp, s_fn)
    e_p, e_r, e_f1 = _prf(e_tp, e_fp, e_fn)
    i_p, i_r, i_f1 = _prf(i_tp, i_fp, i_fn)

    # Latency
    delays = [e.get("detection_delay_sec", 0) for e in pred_events if "detection_delay_sec" in e]
    delays_sorted = sorted(delays) if delays else []
    mean_delay = statistics.mean(delays) if delays else 0
    max_delay = max(delays) if delays else 0
    p50_delay = delays_sorted[len(delays_sorted) // 2] if delays_sorted else 0
    p90_idx = int(len(delays_sorted) * 0.9)
    p90_delay = delays_sorted[min(p90_idx, len(delays_sorted) - 1)] if delays_sorted else 0

    metrics = EvaluationMetrics(
        step_precision=s_p, step_recall=s_r, step_f1=s_f1,
        total_gt_steps=len(gt_steps), step_tp=s_tp, step_fp=s_fp, step_fn=s_fn,
        error_precision=e_p, error_recall=e_r, error_f1=e_f1,
        total_gt_errors=len(gt_errors), error_tp=e_tp, error_fp=e_fp, error_fn=e_fn,
        idle_precision=i_p, idle_recall=i_r, idle_f1=i_f1,
        total_gt_idle_periods=len(gt_idles), idle_tp=i_tp, idle_fp=i_fp, idle_fn=i_fn,
        mean_detection_delay_sec=round(mean_delay, 3),
        max_detection_delay_sec=round(max_delay, 3),
        p50_detection_delay_sec=round(p50_delay, 3),
        p90_detection_delay_sec=round(p90_delay, 3),
    )

    if verbose:
        print(_format_report(metrics, time_tolerance_sec, pred_steps, gt_steps))

    return metrics


def _closest_step_prediction(step_id: int, pred_steps: List[Dict]) -> Dict[str, Any] | None:
    candidates = [p for p in pred_steps if p.get("step_id") == step_id]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p.get("timestamp_sec", 0)))


def _step_timing_label(delta: float, tol: float, prediction: Dict[str, Any]) -> str:
    is_catchup = "catch-up" in (prediction.get("description") or "").lower()
    abs_delta = abs(delta)
    if abs_delta <= tol:
        return "matched"
    if abs_delta <= tol + 1.0:
        return "just outside tolerance"
    if is_catchup and delta > tol:
        return "late catch-up"
    if is_catchup and delta < -tol:
        return "early catch-up"
    return "late" if delta > 0 else "early"


def _format_step_timing_details(pred_steps: List[Dict], gt_steps: List[Dict], tol: float) -> List[str]:
    if not gt_steps:
        return []

    lines = [
        "",
        "  STEP TIMING DETAILS",
        "  " + "-" * 56,
    ]
    for gt in sorted(gt_steps, key=lambda e: e.get("step_id", 0)):
        step_id = gt.get("step_id")
        gt_ts = gt.get("timestamp_sec", 0)
        pred = _closest_step_prediction(step_id, pred_steps)
        if pred is None:
            lines.append(f"    {step_id}:     n/a   missing prediction")
            continue
        pred_ts = pred.get("timestamp_sec", 0)
        delta = pred_ts - gt_ts
        label = _step_timing_label(delta, tol, pred)
        lines.append(f"    {step_id}: {delta:+7.3f}s   {label}")
    return lines


def _format_report(
    m: EvaluationMetrics,
    tol: float,
    pred_steps: List[Dict] | None = None,
    gt_steps: List[Dict] | None = None,
) -> str:
    lines = [
        "=" * 60,
        "  VLM ORCHESTRATOR — EVALUATION REPORT",
        "=" * 60,
        f"  Tolerance: ±{tol:.0f}s",
        "",
        "  STEP COMPLETION",
        "  " + "-" * 56,
        f"    Precision:  {m.step_precision:.1%}",
        f"    Recall:     {m.step_recall:.1%}",
        f"    F1:         {m.step_f1:.3f}",
        f"    {m.step_tp}/{m.total_gt_steps} matched, {m.step_fp} FP, {m.step_fn} FN",
    ]
    if pred_steps is not None and gt_steps is not None:
        lines.extend(_format_step_timing_details(pred_steps, gt_steps, tol))
    lines.extend([
        "",
        "  ERROR DETECTION",
        "  " + "-" * 56,
        f"    Precision:  {m.error_precision:.1%}",
        f"    Recall:     {m.error_recall:.1%}",
        f"    F1:         {m.error_f1:.3f}",
        f"    {m.error_tp}/{m.total_gt_errors} matched, {m.error_fp} FP, {m.error_fn} FN",
        "",
        "  IDLE DETECTION",
        "  " + "-" * 56,
        f"    Precision:  {m.idle_precision:.1%}",
        f"    Recall:     {m.idle_recall:.1%}",
        f"    F1:         {m.idle_f1:.3f}",
        f"    {m.idle_tp}/{m.total_gt_idle_periods} matched, {m.idle_fp} FP, {m.idle_fn} FN",
        "",
        "  DETECTION LATENCY",
        "  " + "-" * 56,
        f"    Mean:   {m.mean_detection_delay_sec:.2f}s",
        f"    P50:    {m.p50_detection_delay_sec:.2f}s",
        f"    P90:    {m.p90_detection_delay_sec:.2f}s",
        f"    Max:    {m.max_detection_delay_sec:.2f}s",
        "=" * 60,
    ])
    return "\n".join(lines)


def save_metrics_json(metrics: EvaluationMetrics, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Orchestrator output")
    parser.add_argument("--predicted", required=True, help="Candidate output JSON")
    parser.add_argument("--ground-truth", required=True, help="Ground truth JSON")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Timestamp tolerance (default: 5s)")
    parser.add_argument("--output", help="Save metrics JSON to this path")
    args = parser.parse_args()

    metrics = evaluate(args.predicted, args.ground_truth, args.tolerance, verbose=True)

    if args.output:
        save_metrics_json(metrics, args.output)
        print(f"\n  Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
