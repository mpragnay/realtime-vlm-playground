# Technical Report: Realtime VLM Procedure Detector

## R&D Iterations

This challenge was ambiguous enough that the final pipeline came from several R&D iterations rather than a single prompt. The first baseline was a direct visual VLM detector: every rolling frame window received the procedure, current step, previous visual context, and recent window summaries, then emitted step/error events directly. This produced useful scene descriptions, but one missed step could block the ordered state machine, and generic step text such as "turns on the circuit breaker box" or "inserts the RAM card" was not descriptive enough for reliable end-state timing.

The next iteration added pre-generated per-step visual rubrics. These rubrics described likely start, during, end, and not-completion states for each procedure step. They helped the model understand what a step endpoint might look like; for example, a RAM insertion should not complete while the hand is still pressing the card, and a circuit-breaker step should require the target component rather than a toolbox/container. This made the reasoning traces more auditable and helped avoid some premature completions.

However, rigid rubrics also exposed a core limitation: some mechanical states are hard to see from egocentric frames. RAM retaining clips can be hidden by fingers, camera on/off state may have no clear visual indicator, and "touch metal" can mean either first contact or the end of a sustained contact interval. Strict rubrics sometimes caused the model to wait for impossible visual evidence, while loose rubrics caused early detections. This led to the submitted descriptor/reasoner design: keep the image model focused on grounded description, then let a stronger text reasoner handle procedure state, catch-up, ambiguity, and error decisions over the accumulated descriptions.

## Architecture

The submitted pipeline in `src/run.py` uses a two-stage descriptor/reasoner design on top of the provided `StreamingHarness`. The frame callback buffers frames into 5-second visual windows. Each window is sent to a lightweight image descriptor model, which outputs only grounded visual context: beginning state, ending state, motion/change, visible objects, scene layout, uncertainty, and step relevance. It is explicitly instructed not to emit events or decide correctness.

Every two descriptor windows, a text-only reasoner model receives the descriptor text, procedure state, current step summary, and optional rubrics. The reasoner emits `step_completion` and `error_detected` events through `harness.emit_event`. It also maintains completed steps and a running per-step summary so later decisions can use recent visual history without repeatedly sending images to the larger model. After each reasoning call, it can also send a small `descriptor_error_guidance` object back to the descriptor prompt so the next visual window pays attention to concrete error-prone details, while still leaving the final error decision to the reasoner. This keeps image perception and procedure reasoning separate, which made failures easier to inspect in logs, and iterate from there.

Supporting modules:

- `src/descriptor.py`: image-window prompt, OpenRouter VLM call, JSON parsing/normalization.
- `src/routing.py`: text-only event reasoning, procedure state, event filtering, reasoner logs.
- `src/smart_frame_sampler.py`: deterministic model-free frame sampling.

## Run Command

Set `OPENROUTER_API_KEY`, then run any clip by passing the matching procedure JSON, video file, and output paths:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"

.venv/bin/python src/run.py \
  --procedure data/clip_procedures/<procedure_file>.json \
  --video data/videos_full/<clip_folder>/Export_py/Video_pitchshift.mp4 \
  --output output/<clip_name>-events.json \
  --descriptor-log output/<clip_name>-descriptor.jsonl \
  --reasoner-log output/<clip_name>-reasoner.jsonl \
  --speed 1.0
```

Example:

```bash
.venv/bin/python src/run.py \
  --procedure data/clip_procedures/z045-june-24-22-dslr.json \
  --video data/videos_full/z045-june-24-22-dslr/Export_py/Video_pitchshift.mp4 \
  --output output/z045-submission-events.json \
  --descriptor-log output/z045-submission-descriptor.jsonl \
  --reasoner-log output/z045-submission-reasoner.jsonl \
  --speed 1.0
```

## Frame Sampling

The harness delivers frames at 2 FPS by default. The pipeline forms 5-second windows, usually 10 candidate frames. The default `smart` sampler keeps five frames: the first frame, a middle anchor frame, the last frame, one transition frame between the first and middle anchors, and one transition frame between the middle and last anchors. For each transition region, it scores candidate frames by SSIM distance from both neighboring anchors, with a small sharpness adjustment, and selects the frame with the highest combined score.

SSIM works well for these egocentric procedure videos because important evidence often appears as structural visual change: hands entering or leaving view, a tool/object moving, a panel opening, a component separating, or the camera viewpoint shifting toward a new work area. Unlike raw pixel difference, SSIM is less sensitive to small lighting/noise changes and better captures changes in layout, edges, and object structure. This lets the sampler preserve visually informative state changes while reducing image tokens. The sampler can be switched to `uniform` for sending all candidate frames when maximum visual recall is more important than cost.

## Audio Usage

The final pipeline intentionally ignores audio. Earlier experiments showed STT was complicated by pitch-shifted privacy audio and by the instructor speaking corrective instructions, which can leak the answer after an error has already happened. Since the goal is visual error detection before correction, the submitted version keeps the decision path visual-only.

## Model Selection

The default configuration uses `google/gemini-3.1-flash-image-preview` as the descriptor model and `google/gemini-3.1-pro-preview` as the text reasoner. The descriptor is cheaper and sees images frequently; the reasoner is called less often and handles procedure state, ambiguity, catch-up, and error logic. Models are CLI-configurable, I also observed that a descriptor model such as `google/gemini-2.5-flash` works well and can be a cost effective option.

## Cost Breakdown

At default settings, a 60-second clip produces about 12 descriptor calls and 6 reasoner calls. With smart sampling, each descriptor call sends about 5 frames; with uniform sampling, it sends about 10 frames. Cost therefore scales approximately as:

`cost_per_minute = 12 * descriptor_call_cost + 6 * reasoner_call_cost`

The design limits expensive reasoning calls to every 10 seconds and keeps image-heavy calls on a lighter VLM. JSONL descriptor/reasoner logs can be enabled to audit unnecessary calls and tune sampling.

## Latency Analysis

The pipeline is synchronous inside the harness callback path, so detection delay includes descriptor API latency and reasoner API latency. Event timestamps are snapped to the midpoint of the descriptor window where evidence appears. The current detection cadence is 5-second visual windows and 10-second reasoning intervals, so step/error detection typically occurs after the relevant descriptor pair has been processed. A lower-latency production version would run descriptor and reasoner calls asynchronously, allowing the harness to continue receiving frames while model calls are in flight.

## Recent Development Results

I evaluated the integrated descriptor/reasoner approach on four training clips during development. These runs were used to understand generalization issues rather than to tune to a single clip. The strongest recent z039 run showed the routing approach can work well when the descriptor cleanly separates wrong-object actions from step progress; R142 remained the hardest case because visually subtle RAM insertion/removal states are difficult to describe reliably from egocentric frames.

| Clip | Output / Metrics File | Step F1 | Step TP / GT | Error F1 | Error TP / GT |
| --- | --- | ---: | ---: | ---: | ---: |
| z045 DSLR | `output/z045-integrated-smart-eval.json` | 0.250 | 2 / 8 | 0.400 | 2 / 7 |
| z039 DSLR | `output/z039-integrated-smart-metrics-2.json` | 0.556 | 5 / 10 | 0.364 | 2 / 7 |
| R066 Circuit Breaker | `output/R066-integrated-smart-metrics.json` | 0.455 | 5 / 11 | 0.000 | 0 / 6 |
| R142 RAM | `output/R142-integrated-smart-metrics.json` | 0.154 | 2 / 13 | 0.000 | 0 / 0 |

I also ran a submission-speed check at `--speed 1.0` on three clips to estimate the latency component of the challenge score. The formula used here is `latency_score = max(0, 1 - mean_detection_delay / 10)`.

| Clip | Metrics File | Step F1 | Step TP / GT | Error F1 | Error TP / GT | Mean Delay | P90 Delay | Latency Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R066 Circuit Breaker | `output/score_R066-metrics.json` | 0.727 | 8 / 11 | 0.000 | 0 / 6 | 175.903s | 263.122s | 0.000 |
| R073 GoPro | `output/score_R073-metrics.json` | 0.400 | 4 / 11 | 0.000 | 0 / 0 | 185.539s | 321.160s | 0.000 |
| R087 GoPro | `output/score_R087-metrics.json` | 0.500 | 1 / 2 | 0.000 | 0 / 20 | 43.019s | 63.991s | 0.000 |
| R090 ATV | `output/score_R090-metrics.json` | 0.545 | 6 / 11 | 0.000 | 0 / 0 | 175.680s | 276.061s | 0.000 |
| R092 Circuit Breaker | `output/score_R092-metrics.json` | 0.286 | 2 / 12 | 0.000 | 0 / 0 | 146.750s | 242.436s | 0.000 |
| R142 RAM | `output/score_R142-metrics.json` | 0.231 | 3 / 13 | 0.000 | 0 / 0 | 206.321s | 339.134s | 0.000 |
| R190 ATV | `output/score_R190-metrics.json` | 0.400 | 5 / 13 | 0.125 | 1 / 8 | 340.840s | 535.476s | 0.000 |
| R192 Circuit Breaker | `output/score_R192-metrics.json` | 0.667 | 10 / 15 | 0.000 | 0 / 1 | 153.707s | 255.203s | 0.000 |
| R198 Graphics Card | `output/score_R198-metrics.json` | 0.455 | 5 / 11 | 0.000 | 0 / 7 | 231.421s | 397.292s | 0.000 |
| z010 GoPro | `output/score_z010-metrics.json` | 0.250 | 3 / 12 | 0.000 | 0 / 1 | 353.310s | 562.910s | 0.000 |
| z039 DSLR | `output/score_z039-metrics.json` | 0.500 | 5 / 10 | 0.400 | 2 / 7 | 180.040s | 302.640s | 0.000 |
| z045 DSLR | `output/score_z045-metrics.json` | 0.154 | 1 / 8 | 0.222 | 1 / 7 | 79.730s | 144.510s | 0.000 |
| z065 DSLR | `output/score_z065-metrics.json` | 0.375 | 3 / 8 | 0.091 | 1 / 21 | 100.420s | 202.180s | 0.000 |
| z067 GoPro | `output/score_z067-metrics.json` | 0.200 | 3 / 15 | 0.273 | 3 / 8 | 388.550s | 638.900s | 0.000 |
| z108 GoPro | `output/score_z108-metrics.json` | 0.500 | 7 / 14 | 0.133 | 1 / 7 | 338.160s | 639.770s | 0.000 |

These latency numbers are the main weakness of the current implementation. The algorithmic cadence is 5-second descriptor windows and 10-second reasoner intervals, but model calls are synchronous inside the harness callback, so API time accumulates as backlog during a real-time run.

The main takeaway is that the two-stage routing architecture improves debuggability: descriptor logs show what the vision model perceived, and reasoner logs show why an event was or was not emitted. The remaining accuracy bottleneck is mostly visual ambiguity and over/under-claiming in descriptors, especially for mechanical states such as whether a RAM card is fully seated or whether a camera control action actually changed internal state.

The headline F1 scores understate some useful behavior. With verbose timing inspection, several detections were semantically plausible but landed just outside the evaluator's ±5s tolerance window. Examples of near-miss step detections:

| Clip | Step | Predicted | Ground Truth | Delta |
| --- | ---: | ---: | ---: | ---: |
| z045 DSLR | 3 | 57.250s | 52.100s | +5.150s |
| z045 DSLR | 7 | 107.250s | 112.727s | -5.477s |
| z045 DSLR | 8 | 132.250s | 139.200s | -6.950s |
| z039 DSLR | 5 | 77.250s | 84.300s | -7.050s |
| R066 Circuit Breaker | 10 | 157.250s | 163.383s | -6.133s |
| R142 RAM | 3 | 47.250s | 53.300s | -6.050s |
| R142 RAM | 13 | 207.250s | 214.200s | -6.950s |

These near misses suggest the system often identifies the right procedural phase, but timestamping is still coarse. Events are currently assigned to descriptor-window midpoints, so an otherwise correct detection can miss the evaluator window by a few seconds. Some steps also have ambiguous completion semantics: the video may show the action becoming true before the ground truth marks the step as complete.

## Dynamic Descriptor Guidance For Errors

A separate problem was that generic visual descriptions often missed the exact details needed for error detection. For example, saying "the student manipulates the camera" is not enough to decide whether they used the lens cap or lens hood, and saying "the student manipulates the mount" is not enough to know whether a screw failed to seat or a wrong mount was used.

To address this, the reasoner outputs `descriptor_error_guidance` after each reasoning call. This is not an error label. It is a visual-only focus list passed to the next descriptor call, asking it to report concrete possibilities such as wrong object/location, failed or repeated attempt, improper mechanical action, or unnecessary reversal. The descriptor still does not decide correctness; it only describes the relevant visual evidence in more detail. The reasoner remains the only component that emits `error_detected`.

The four error categories are used as an internal guidance taxonomy, not as extra output labels. Final emitted errors still use the challenge schema values such as `wrong_action`, `safety_violation`, `improper_technique`, and `other`.

Examples:

- z039 DSLR: guidance encouraged the descriptor to distinguish lens cap, lens hood, lens body, camera controls, battery compartment, and SD-card compartment. Error detection improved from `2/7` matched in an earlier strong routing run to `5/7` matched in the guided run, although it also introduced extra false positives when procedure state drifted.
- R066 Circuit Breaker: guidance asks the descriptor to distinguish the small red floor toolbox from the larger red `PRO STEEL` toolbox and to report whether the student is actually handling a circuit breaker or only a toolbox/container. This helps wrong-object reasoning, though the run remained sensitive to timing and state drift.
- GoPro clips: guidance asks for details such as whether the SD card or mount screw is repeatedly pushed, released, seated, locked, dropped, or re-aligned. These repeated mechanical patterns are more useful for detecting failed attempts than a generic "the student manipulates the object" description.
- R142 RAM: for hard mechanical states, guidance can ask the descriptor to report whether the hand is still pressing the RAM card, whether the hand releases it, whether clips appear engaged, and whether the student moves on to the next RAM card.

## Bidirectional Streaming Redesign

OpenRouter supports streaming output but not streaming input. With a bidirectional streaming API such as Gemini Live API, I would continuously stream selected frames into a long-lived descriptor session and issue lightweight text queries every 5 seconds for grounded scene updates. A separate reasoner session would consume those updates, maintain procedure state, and emit events as soon as evidence appears. This would avoid creating a fresh VLM request for every window, preserve short-term visual context naturally, reduce repeated prompt overhead, and make the system closer to a real-time assistant.
