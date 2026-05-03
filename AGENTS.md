<claude-mem-context>
# Memory Context

# [realtime-vlm-playground-1] recent context, 2026-05-03 9:00am EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 22 obs (8,201t read) | 216,640t work | 96% savings

### Apr 30, 2026
154 4:55p 🔵 Challenge Project: Real-Time Streaming Step Detection
156 " 🔵 realtime-vlm-playground: Full Architecture and Data Format Traced
158 7:42p ⚖️ realtime-vlm-playground: Rolling Frame Buffer Architecture for Step/Error Detection
160 8:00p ⚖️ realtime-vlm-playground: Minimal First Baseline Architecture Decided
162 8:01p 🔵 realtime-vlm-playground: src/run.py Pipeline Template Structure
164 " 🟣 realtime-vlm-playground: Rolling Buffer Pipeline Implemented in src/run.py
166 " 🔵 realtime-vlm-playground: No virtualenv present; system python3 missing requests module
170 8:08p ✅ realtime-vlm-playground: VLM Prompt Near-Future Steps Lookahead Reduced from 4 to 2
171 8:14p 🟣 realtime-vlm-playground: VLM Response Now Includes Structured Status Field
172 8:24p 🔵 realtime-vlm-playground: Primary Experiment Target and Run Command
174 8:25p 🔵 realtime-vlm-playground: R066 Pipeline Run Output — Full Event Sequence
176 8:26p 🔵 R066 Ground Truth Format: Step Timings, Errors, and Idle Periods
177 " 🔴 src/run.py: Step Completion Timestamp Snapped to Window End + Prompt Clarified
180 " 🔴 src/run.py: VLM Error Prompt Enhanced for Wrong Toolbox/Part Detection
182 8:34p 🔵 R066 Pipeline Run 2: Step Ordering Regression and Spurious Errors After Prompt Fix
183 8:41p 🟣 src/run.py: Ordered Step Emission Replaces Per-Event Step Emission
185 8:49p 🟣 src/run.py: Debug JSONL Logging for Event-Bearing VLM Calls
186 8:50p 🔵 src/run.py: File State Reverted — Prior Refactor Changes Lost
187 8:51p 🟣 src/run.py: Console Debug Printing for VLM Event Proposals vs Emissions
### May 2, 2026
192 9:01p 🔵 z045-june-24-22-dslr: Ground Truth vs Integrated Pipeline Output Comparison
193 " 🔵 z045-integrated-events.json Evaluation Results: Step F1=0.25, Error F1=0.60
197 9:02p 🔵 z045 Integrated Pipeline: Detailed GT vs Predicted Event Timing Analysis

Access 217k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>

## Experiment Notes

### 2026-05-01: R066 Current Pipeline Baseline

Command:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-current.json \
  --speed 10.0
```

Evaluation:

```text
STEP COMPLETION
Precision: 45.5%
Recall:    45.5%
F1:        0.455
5/11 matched, 6 FP, 6 FN

ERROR DETECTION
Precision: 0.0%
Recall:    0.0%
F1:        0.000
0/6 matched
```

Step timing detail:

```text
1:  -2.153s   matched
2:  +2.151s   matched
3:  -6.157s   early
4: -27.216s   early
5: -35.845s   early
6: -34.491s   early
7:  -1.349s   matched
8:  -5.265s   just outside tolerance
9: -23.451s   early
10: -4.883s   matched
11: -1.216s   matched
```

Conclusions:

- The VLM can recognize many target actions, but it often emits visually similar switch/door steps much too early.
- Step timing detail is more useful than aggregate F1 alone; it shows that steps 4, 5, 6, and 9 are the major timing failures in this run.
- Error detection remains at zero without audio; R066 errors are mostly wrong-toolbox/wrong-part moments that likely need instructor correction transcripts or stronger multimodal context.
- The current direct event-emission approach is partially viable for step recognition, but needs temporal confirmation/state modeling before it can reliably match completion timestamps.

### 2026-05-01: Hybrid OpenRouter STT + Audio Error Path

Implementation:

- All API calls now use `OPENROUTER_API_KEY`: chat/VLM calls use OpenRouter chat completions, and STT uses OpenRouter `/api/v1/audio/transcriptions`.
- Audio chunks are transcribed asynchronously, then paired with frames from only the exact same 5s audio window for error-only VLM calls.
- The harness now supports `on_complete()` so pending async audio work can finish before results are written.
- Adding recent transcript history to the audio-error prompt was important. Without it, chunks like "to the right" lacked the prior instruction needed to understand the correction.

Best run so far:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-audio-context.json \
  --speed 10.0
```

Evaluation:

```text
STEP COMPLETION
Precision: 40.0%
Recall:    36.4%
F1:        0.381
4/11 matched, 6 FP, 7 FN

ERROR DETECTION
Precision: 20.0%
Recall:    16.7%
F1:        0.182
1/6 matched, 4 FP, 5 FN
```

Step timing detail:

```text
1:  -5.153s   just outside tolerance
2: +20.151s   late
3: +20.843s   late
4:  -3.216s   matched
5:  -5.845s   just outside tolerance
6:  -1.491s   matched
7:  -1.349s   matched
8:  -5.265s   just outside tolerance
9:  -5.451s   just outside tolerance
10: -4.883s   matched
11:    n/a    missing prediction
```

Conclusions:

- Audio integration is viable for surfacing the early wrong-toolbox cluster: the matched error came from the 10-15s chunk after adding recent transcript continuity.
- Raw audio-triggered error calls are still noisy. They need stricter correction detection, not just "instruction changed" detection.
- Step completion did not materially improve with the audio path and remains a separate visual timing/state problem.
- Running `google/gemini-2.5-pro` for every visual and audio VLM call was too slow for this loop. A stronger model should be used selectively, e.g. only for uncertain/error-candidate windows.

### 2026-05-01: Visual Context Prompting Findings

Current direction:

- Runtime is back to a visual-only baseline. Instructor audio/STT is kept out of `src/run.py` because it can leak correction/error labels.
- The visual VLM now receives 5s windows, previous step-wise visual context, and recent window descriptions.
- The prompt asks the VLM to describe the first-person POV action, object appearance, object motion, uncertainty, and relevance before emitting events.

Latest short-window check:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video output/R066-first35s.mp4 \
  --output output/R066-events-visual-context-first35-current.json \
  --speed 10.0 \
  --vlm-log output/R066-visual-context-first35-current.jsonl \
  --vlm-log-start 0 \
  --vlm-log-end 35
```

Observed behavior:

- The VLM no longer treats early red-toolbox handling as step 1 completion.
- It correctly describes the first-person view: hands, camera motion, and manipulated objects are treated as the student's action.
- It can now describe useful temporal changes across consecutive windows, e.g. the small red toolbox is lifted/repositioned in one window, then lowered/resting on the floor in the next.
- Prompting reduced one major hallucination: the model now says the small red toolbox lid does not visibly open, instead of stating that it opened.
- The model distinguishes the small red floor toolbox from the larger red `PRO STEEL` drawer toolbox when the view moves to the larger toolbox.

Current conclusion:

- We are able to prompt the VLM into producing the right kind of visual observations: object identity, state changes, motion, uncertainty, and step relevance are much better than the initial raw event prompt.
- The remaining hard part is not visual description; it is deciding when those observations should become `error_detected`.
- Error detection is still too brittle because many wrong-action cases look like normal searching/preparation without audio or stronger task-level reasoning.
- The next useful architecture is likely a two-layer approach: first collect grounded visual context, then run a stricter event decision layer over that context and current frames. The decision layer should only emit errors when the current frames show a clear contradiction or wrong object/action, not merely because prior context speculated one path and the student moved elsewhere.

### 2026-05-01: R066 Visual Step Definition

Core visual-only rubric:

- A step is completed only when the required final visual state is visible.
- `start_sec` includes searching, preparation, mistakes, corrections, and partial manipulation. `end_sec` is the detection target for completion.
- Prior visual context can explain continuity, but it cannot prove completion by itself.
- Touching, searching, approaching, handling a container, or beginning manipulation should remain `step_in_progress`.
- Object acquisition steps require the target object itself to be visible in the student's hand or clearly controlled by the student.
- Door, switch, and panel steps require the final changed state to be visible after manipulation.
- Errors can occur inside a step without ending that step. For example, wrong-toolbox actions during step 1 are errors, but step 1 continues until the actual circuit breaker is grabbed.

Working mental model:

```text
A step completion is the first frame where the current expected step's required
final state is visually true. Prior context explains how the student got there,
but current-frame evidence must prove the completion.
```

### 2026-05-01: R066 Full Visual Context Run Failure Analysis

Run:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R066-15July-Circuit-Breaker-part2.json \
  --video data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4 \
  --output output/R066-events-visual-context-full.json \
  --speed 10.0 \
  --vlm-log output/R066-visual-context-full.jsonl \
  --vlm-log-start 0 \
  --vlm-log-end 176
```

Evaluation:

```text
STEP COMPLETION
Precision: 0.0%
Recall:    0.0%
F1:        0.000
0/11 matched, 2 FP, 11 FN

Step timing:
1:  -5.153s   just outside tolerance
2: +112.651s  late
3-11: missing prediction
```

Main diagnosis:

- The VLM produced useful visual descriptions, but emitted only two step events.
- Step 1 was visually described and emitted at 44.5s, but GT completion is 49.653s. This is just outside tolerance because the model used first clear object acquisition, while the annotation appears slightly later after moving away with the breaker.
- The 45.0-49.5s VLM response failed JSON parsing because the model put literal newlines inside string values. That call said step 1 was complete and step 2 was being prepared, but it emitted no event.
- Step 2 was never correctly detected near 57.349s. The VLM described internal component manipulation as relevant preparation, but did not know what final visual state completes "turns on the circuit breaker box."
- Because ordered step gating requires step 2 before steps 3+, the pipeline stayed stuck on current expected step 2 for the rest of the video.
- Later windows often described the correct visual facts for later steps, e.g. door closed, main disconnect moved, breaker removed, breaker returned to toolbox, second breaker inserted, but the VLM framed them as still related to step 2 instead of emitting later step completions.

Conclusion:

- The prompt is good enough to produce visual context, but not good enough to map visual context to granular step completion events.
- The step section needs an explicit "active step reasoning" instruction: compare the current frames against the expected step's required final state, decide whether that state is now visually true, and explain why it does or does not complete the step.
- The procedure descriptions are too semantically overlapping for the model without per-step completion criteria. In R066, "turns on the circuit breaker box," "turns on the main power," and "turns on the circuit_breaker" are visually distinct but linguistically easy to collapse.

### 2026-05-02: R142 Strict Rubric Sampling Sweep

Code change:

- Added `--temperature` and `--top-p` to `src/run.py`.
- These values are passed through to the OpenRouter chat-completions payload for multi-frame VLM calls.
- Experiments below used strict/hard step rubrics on `data/clip_procedures/R142-31Aug-RAM.json`.

Experiment command shape:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/R142-31Aug-RAM.json \
  --video data/videos_full/R142-31Aug-RAM/Export_py/Video_pitchshift.mp4 \
  --output output/R142-events-strict-<config>-run<N>.json \
  --speed 10.0 \
  --step-rubric output/step_rubrics/R142-31Aug-RAM.json \
  --step-rubric-mode strict \
  --temperature <temp> \
  --top-p <top_p>
```

Results:

```text
temperature=0.0, top_p=1.0
run 1: F1 0.143, 1/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.143, 1/13 matched
average F1: 0.143

temperature=0.2, top_p=0.5
run 1: F1 0.143, 1/13 matched
run 2: F1 0.538, 7/13 matched
run 3: F1 0.143, 1/13 matched
average F1: 0.275

temperature=0.7, top_p=0.9
run 1: F1 0.500, 5/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.381, 4/13 matched
average F1: 0.341

temperature=1.0, top_p=1.0
run 1: F1 0.462, 6/13 matched
run 2: F1 0.143, 1/13 matched
run 3: F1 0.538, 7/13 matched
average F1: 0.381
```

Main observations:

- `temperature=0.0, top_p=1.0` is reproducible, but reproducibly bad. All three runs emitted step 1 and then missed step 2, so ordered gating blocked the rest of the sequence.
- Moderate/high sampling sometimes helps the model escape the step-2 miss, but it is not reliable. The same config can produce either a full-sequence run or a blocked run.
- `temperature=1.0, top_p=1.0` had the best average in this sweep and resembles the earlier default/provider behavior, but it still collapsed in one of three runs.
- Higher sampling also increases output instability risk, including occasional JSON parse failures seen in the high-sampling runs.
- The dominant failure is not the rubric quality on R142. The VLM often proposes correct later steps even when an earlier step is pending. The state manager discards those proposals because strict ordered gating cannot recover from one missed current step.

Conclusion:

- Sampling controls affect reproducibility, but they do not solve the real bottleneck.
- Deterministic decoding can lock the model into a bad interpretation.
- The next architecture improvement should be pending/catch-up step handling: store blocked future-step proposals instead of discarding them, then replay/commit them when the missing earlier step is recovered or when enough evidence accumulates.

### 2026-05-02: R142 Lifecycle Rubric + Phase-Gate Experiments

Recent changes:

- Rubric generation now asks for lifecycle fields: `state_start_visual`, `state_during_visual`, `state_end_visual`, `not_completion`, `timestamp_target`, and `ambiguities`.
- Runtime prompts now ask every `step_completion` to include `matched_phase`, `rubric_reference`, and `completion_reasoning` so we can inspect why the VLM believed a step completed.
- `matched_phase` is now intentionally limited to `state_end_visual` or `catch_up` for emitted completions. The earlier `mechanical_completion` escape hatch was removed because models misused it for non-mechanical sustained-contact steps.
- Visual wrong-sequence errors were removed from the prompt/runtime framing. Apparent later-step progress should be treated as possible missed prior completion/state drift, not as an error event.

Key findings:

- R142 step 4, "touches the metal of the computer tower," exposed a rubric ambiguity. Models naturally treat first/sustained metal contact as completion because the procedure says "touches"; the benchmark timestamp appears to mark the end/release of sustained contact.
- The manual lifecycle mock should avoid broad end-state wording such as "transitions from touching the metal chassis to the next action or a waiting/silence interval" because waiting while still touching can be misread as completion. Better wording is "after sustained contact, the hand clearly breaks contact and moves away from the metal touch point."
- Adding explanation fields improves debuggability but does not by itself constrain emissions. Models can still cite the wrong rubric field or invent a post-hoc reason unless the prompt forces phase classification before emission.
- The `mechanical_completion` phase improved some RAM insertion matches, but it also gave the VLM a broad escape hatch. In one R142 run it labeled step 4 as `mechanical_completion`, even though sustained-contact steps were explicitly excluded. This confirms that phase values should remain simple unless code-side validation enforces them.
- Gemini 3.1 Flash Image Preview produced mixed results: it sometimes matched more RAM insertion steps than Gemini 2.5 Flash, but it was slower, occasionally emitted invalid event types, and still mishandled step 4 without tighter phase/rubric enforcement.

Current working hypothesis:

```text
Keep the rubric lifecycle strict and auditable first. If RAM insertion becomes
too conservative again, add a narrow, code-validated mechanism for hard-to-see
mechanical steps rather than a broad prompt-only phase.
```

### 2026-05-02: z045 Descriptor/Reasoner Routing Experiment

Purpose:

- Test a two-agent routing architecture before changing the streaming runtime.
- Descriptor stage: a lightweight image VLM describes each 5s frame window in
  `window_description` format only; it does not emit steps/errors.
- Reasoner stage: a text-only stronger model reads non-overlapping pairs of
  descriptor windows (`0-10`, `10-20`, ...), procedure state, step rubrics, and
  a running summary for the current step, then emits events.
- The current experiment uses mock descriptor output extracted from
  `output/z045-3.1-fresh-rubric.jsonl`, so the reasoner can be tested without
  more image calls.

Mock generation command:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

src = Path("output/z045-3.1-fresh-rubric.jsonl")
out = Path("output/z045-window-descriptions-mock.json")
windows = []
with src.open() as f:
    for idx, line in enumerate(f, start=1):
        row = json.loads(line)
        visual_context = row.get("visual_context") if isinstance(row.get("visual_context"), dict) else {}
        parsed = row.get("parsed_response") if isinstance(row.get("parsed_response"), dict) else {}
        frame_window = row.get("frame_window")
        windows.append({
            "source_log_index": row.get("log_index", idx),
            "frame_window": frame_window,
            "frame_timestamps": row.get("frame_timestamps"),
            "midpoint_sec": round((frame_window[0] + frame_window[1]) / 2, 3) if frame_window else None,
            "window_description": visual_context.get("window_description") or parsed.get("window_description"),
        })
out.write_text(json.dumps({
    "source": "descriptor_mock_from_existing_vlm_log",
    "clip": "z045-june-24-22-dslr",
    "input_log": str(src),
    "description": "Mock descriptor-agent output: per-window visual descriptions only, no events or status.",
    "windows": windows,
}, indent=2) + "\n")
print(f"Saved {out} with {len(windows)} windows")
PY
```

Routing run:

```bash
.venv/bin/python src/routing_experiment.py \
  --procedure data/clip_procedures/z045-june-24-22-dslr.json \
  --descriptions output/z045-window-descriptions-mock.json \
  --step-rubric output/step_rubrics/z045-june-24-22-dslr-3.1-fresh-pro.json \
  --output output/z045-routing-events.json \
  --reasoner-log output/z045-routing-reasoner.jsonl \
  --model google/gemini-3.1-pro-preview \
  --temperature 0.2
```

Evaluation:

```bash
.venv/bin/python -m src.evaluator \
  --predicted output/z045-routing-events.json \
  --ground-truth data/ground_truth_sample/z045-june-24-22-dslr.json \
  --tolerance 5
```

Latest result:

```text
STEP COMPLETION
Precision: 20.0%
Recall:    12.5%
F1:        0.154
1/8 matched

ERROR DETECTION
Precision: 33.3%
Recall:    14.3%
F1:        0.200
1/7 matched
```

Emitted steps:

```text
step 1 @ 14.75s  just outside tolerance
step 2 @ 24.75s  early
step 3 @ 54.75s  matched
step 4 @ 54.75s  early
step 5 @ 84.75s  late, catch_up
steps 6-8 missing
```

Key findings:

- The routing reasoner is better at semantic error reasoning than the direct
  VLM. It correctly identified the lens-hood wrong-object action while the
  current step was lens-cover removal.
- The reasoner still struggled with hidden-state control steps. For step 5 it
  saw the power switch move at `70-79.5s`, but refused completion because the
  LCD did not illuminate. It later emitted step 5 as catch-up at `84.75s`.
- The run then got stuck at step 6, "turns off the DSLR," because no explicit
  turn-off visual evidence appeared in the descriptor text. Ordered gating
  blocked steps 7-8 even though the reasoner described battery/SD-card actions.
- The running current-step summary worked mechanically and is logged in
  `step_summaries_after`, but it did not solve the step-6 observability/gating
  issue.

Current conclusion:

```text
Descriptor/reasoner routing is promising for separating visual perception from
event reasoning, especially for wrong-object errors. The next improvement is
not more summary alone; it is better handling of hidden-state control steps and
catch-up/bypass logic for steps like turn-on/turn-off where the video lacks a
clear visual device-state indicator.
```

### 2026-05-02: z045 Routing Without Step Rubric

Change:

- Added no-rubric mode to `src/routing_experiment.py`.
- In no-rubric mode, the reasoner does not receive precomputed lifecycle
  rubrics. It reasons from the step text, running current-step summary, and
  descriptor windows.
- Prompt framing asks the model to infer flexible visual completion cues itself
  and not make hard assumptions that every possible confirmation cue must occur.

Run:

```bash
.venv/bin/python src/routing_experiment.py \
  --procedure data/clip_procedures/z045-june-24-22-dslr.json \
  --descriptions output/z045-window-descriptions-mock.json \
  --output output/z045-routing-no-rubric-events.json \
  --reasoner-log output/z045-routing-no-rubric-reasoner.jsonl \
  --model google/gemini-3.1-pro-preview \
  --temperature 0.2
```

Evaluate:

```bash
.venv/bin/python -m src.evaluator \
  --predicted output/z045-routing-no-rubric-events.json \
  --ground-truth data/ground_truth_sample/z045-june-24-22-dslr.json \
  --tolerance 5
```

Result:

```text
STEP COMPLETION
Precision: 40.0%
Recall:    25.0%
F1:        0.308
2/8 matched

ERROR DETECTION
Precision: 33.3%
Recall:    14.3%
F1:        0.200
1/7 matched
```

Compared to rubric routing:

```text
with rubric:    step F1 0.154, 1/8 matched
without rubric: step F1 0.308, 2/8 matched
```

Step behavior:

```text
1: 14.75s  just outside tolerance
2: 24.75s  early
3: 54.75s  matched
4: 54.75s  early
5: 74.75s  matched
6-8: missing
```

Key finding:

- No-rubric reasoning fixed the hidden-state control issue for step 5. The
  reasoner accepted the visible switch action as completion:

```text
description: student's left index finger applies pressure to the power switch dial
reason: physical OFF/ON dial was actuated; even though LCD does not illuminate,
the switch action is complete
```

- The rubric version had refused step 5 at `70-79.5s` because the LCD did not
  illuminate, then emitted it late as catch-up at `84.75s`.
- This suggests rigid rubrics can hurt when the step has hidden/internal state
  or video lacks a clear final-state indicator.

Remaining failure:

- The reasoner still got stuck at step 6, "turns off the DSLR." No descriptor
  window showed an explicit turn-off action, so ordered gating blocked battery
  and SD-card steps 7-8 even though the text descriptions contained those
  actions.

Current hypothesis:

```text
For routing, we may be able to drop precomputed step rubrics entirely, or at
least stop passing them to the descriptor. Let the descriptor produce grounded
visual context without step-completion rubric pressure, then let the text
reasoner infer practical completion cues from step text + running summary.
Rubrics are useful for audit/debugging, but can overconstrain hidden-state or
domain-specific actions.
```

### 2026-05-02: Integrated Descriptor + Reasoner Pipeline Results

Change:

- Added/reran the integrated descriptor + reasoner pipeline, where the descriptor
  receives the current expected step from reasoner state before each 5s visual
  window.
- Descriptor model: `google/gemini-3.1-flash-image-preview`.
- Reasoner model: `google/gemini-3.1-pro-preview`.
- No step rubric was passed; the reasoner inferred completion cues from step
  text, descriptor windows, procedure state, and running step summaries.
- Updated reasoner timestamping: events now use the midpoint of the specific 5s
  descriptor window containing the evidence, instead of the midpoint of the
  10s reasoner interval.

R142 command:

```bash
.venv/bin/python src/integrated_pipeline.py \
  --procedure data/clip_procedures/R142-31Aug-RAM.json \
  --video data/videos_full/R142-31Aug-RAM/Export_py/Video_pitchshift.mp4 \
  --output output/R142-integrated-events.json \
  --descriptor-log output/R142-integrated-descriptor.jsonl \
  --reasoner-log output/R142-integrated-reasoner.jsonl \
  --descriptor-model google/gemini-3.1-flash-image-preview \
  --reasoner-model google/gemini-3.1-pro-preview
```

R142 result:

```text
STEP COMPLETION
Precision: 46.2%
Recall:    46.2%
F1:        0.462
6/13 matched

ERROR DETECTION
No GT errors and no predicted errors.
```

R142 step timing:

```text
1:  22.25s vs 23.20s   matched
2:  37.25s vs 42.00s   matched
3:  52.25s vs 53.30s   matched
4:  57.25s vs 82.00s   early
5:  87.25s vs 104.30s  early
6:  92.25s vs 113.10s  early
7: 107.25s vs 121.10s  early
8: 117.25s vs 130.30s  early
9: 157.25s vs 157.00s  matched
10: 167.25s vs 167.20s matched
11: 187.25s vs 185.00s matched
12: 192.25s vs 201.20s early
13: 207.25s vs 214.20s early
```

R142 learnings:

- Descriptor-window timestamping helped substantially. R142 improved from
  `3/13` matched (`F1=0.231`) to `6/13` matched (`F1=0.462`).
- Step 2 specifically moved from the 10s group midpoint `34.75s` to the
  descriptor-window midpoint `37.25s`, bringing it within evaluator tolerance.
- RAM insertions improved: steps 9, 10, and 11 matched cleanly. The reasoner
  used practical visual evidence such as "hands move away from RAM slots and
  pick up the next RAM card" as completion evidence for hard-to-see insertion
  states.
- The remaining major problem is lifecycle semantics. Step 4, "touches the
  metal of the computer tower," was emitted at first visible contact
  (`57.25s`), while GT marks the end of the sustained touch/contact period
  (`82.0s`). This early step 4 shifts RAM removals 5-8 early.
- Prompt rule responsible: without rubrics, the reasoner treats "touches metal"
  as a stable final visible state once the hand rests on the chassis. The
  current prompt does not distinguish instantaneous action steps from
  duration/state steps whose completion should be the end/release/transition.

z045 command:

```bash
.venv/bin/python src/integrated_pipeline.py \
  --procedure data/clip_procedures/z045-june-24-22-dslr.json \
  --video data/videos_full/z045-june-24-22-dslr/Export_py/Video_pitchshift.mp4 \
  --output output/z045-integrated-events.json \
  --descriptor-log output/z045-integrated-descriptor.jsonl \
  --reasoner-log output/z045-integrated-reasoner.jsonl \
  --descriptor-model google/gemini-3.1-flash-image-preview \
  --reasoner-model google/gemini-3.1-pro-preview
```

z045 result:

```text
STEP COMPLETION
Precision: 25.0%
Recall:    25.0%
F1:        0.250
2/8 matched

ERROR DETECTION
Precision: 50.0%
Recall:    42.9%
F1:        0.462
3/7 matched
```

z045 step timing:

```text
1:  17.25s vs 20.00s   matched
2:  22.25s vs 31.00s   early
3:  27.25s vs 52.10s   early catch-up
4:  27.25s vs 63.30s   early
5:  82.25s vs 70.20s   late catch-up
6:  82.25s vs 75.20s   late catch-up
7:  92.25s vs 112.73s  early
8: 137.25s vs 139.20s  matched
```

z045 learnings:

- Descriptor-window timestamping is working, but z045 exposed a different
  problem: catch-up is too permissive for visible object-transfer steps.
- In `20.0-29.5s`, the reasoner emitted step 2, then used catch-up for step 3
  because it saw the lens cover being attached for step 4. This pushed current
  state to step 5 too early.
- Because state advanced early, the real lens-cover sequence around `50-63s`
  was interpreted as wrong actions relative to "turn on DSLR" instead of as
  steps 3 and 4.
- Catch-up should probably be limited to hidden/internal/control steps where
  the required visual state may be unobservable, e.g. turn on/off. It should not
  be used for visible object manipulation steps where the target object itself
  should be visible, e.g. withdraw lens cover, remove RAM card, place object.
- Error precision is useful but state-sensitive. Once state drifts, otherwise
  valid step actions become "wrong_action" events.

Integrated takeaway:

```text
The integrated descriptor+reasoner architecture is promising: grounded
descriptions plus text-only reasoning make traces much easier to inspect, and
descriptor-window timestamping improves evaluator alignment. The next work is
not more generic prompt detail; it is explicit step-lifecycle modeling:

1. classify step text as instantaneous action, object-transfer, sustained
   state/duration, hidden control state, or reversible/retry-prone manipulation;
2. restrict catch-up to hidden/control or genuinely skipped visual states;
3. for sustained-state steps, complete on release/transition away rather than
   first contact;
4. keep descriptor output rubric-free and grounded, but give the reasoner a
   lightweight lifecycle policy for each step class.
```
