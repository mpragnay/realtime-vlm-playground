# Technical Report: Realtime VLM Procedure Detector

## Architecture

The submitted pipeline in `src/run.py` uses a two-stage descriptor/reasoner design on top of the provided `StreamingHarness`. The frame callback buffers frames into 5-second visual windows. Each window is sent to a lightweight image descriptor model, which outputs only grounded visual context: beginning state, ending state, motion/change, visible objects, scene layout, uncertainty, and step relevance. It is explicitly instructed not to emit events or decide correctness.

Every two descriptor windows, a text-only reasoner model receives the descriptor text, procedure state, current step summary, and optional rubrics. The reasoner emits `step_completion` and `error_detected` events through `harness.emit_event`. It also maintains completed steps and a running per-step summary so later decisions can use recent visual history without repeatedly sending images to the larger model. This keeps image perception and procedure reasoning separate, which made failures easier to inspect in logs, and iterate from there.

Supporting modules:

- `src/descriptor.py`: image-window prompt, OpenRouter VLM call, JSON parsing/normalization.
- `src/routing.py`: text-only event reasoning, procedure state, event filtering, reasoner logs.
- `src/smart_frame_sampler.py`: deterministic model-free frame sampling.

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

These near misses suggest the system often identifies the right procedural phase, but timestamping is still coarse because of two reasons, firstly timestamps are assigned to event window midpoints, adding reference times to descriptor descriptions will improve this, secondly there are some steps in many of the videos that have ambiguities regarding their completion. For Ex: In the R066 video's step 1, the description mentions "The student grabs the circuit breaker." which would indicate step completion to be the moment the circuit breaker lands in the students hands, in the video this happens around the 42 second mark, but the GT marks that as step start and the step end is the last frame before the next step begins, which is not in line with the instructions as the step end is marked by the first moment the action is deemed completed/ended. There is a similar case in the z045 video where in step 4 the student is asked to touch the metal surface, the ground truth marks the step end to be when the contact with metal surface is broken, but it is not wrong to interpret such a step to end when the action of making contact with the metal surface is made. A logical way the VLM approaches this is, it reasons that because the step is to make contact with the metal surface, it asks the question have you made contact with the metal surface, if so then we are done with this step. The problem with such steps is they sort of gate and throw off the next steps, causing the VLM to either error out falsely or be off with it's timings.

## Next Step: Dynamic Descriptor Guidance

A concrete next improvement is to make the reasoner produce dynamic guidance for the next descriptor call. Today, the descriptor receives only previous/current/next step names as hints. The reasoner maintains richer state internally, but it does not yet tell the descriptor what visual evidence is missing or ambiguous.

The next version would have the reasoner output a small `descriptor_guidance` object after each reasoning call, for example: what objects to distinguish, what state transition to inspect, what not to overclaim, and what possible wrong action to watch for. For a RAM insertion step, this could tell the descriptor to focus on whether the card is still being pressed, whether the hand released it, whether clips appear engaged, and whether the student has moved on to another RAM card. For DSLR clips, it could ask the descriptor to distinguish lens cap, lens hood, lens body, camera switch, and battery/card compartments. This keeps the descriptor visual-only while making its descriptions more targeted to the current procedural uncertainty.

## Bidirectional Streaming Redesign

OpenRouter supports streaming output but not streaming input. With a bidirectional streaming API such as Gemini Live API, I would continuously stream selected frames into a long-lived descriptor session and issue lightweight text queries every 5 seconds for grounded scene updates. A separate reasoner session would consume those updates, maintain procedure state, and emit events as soon as evidence appears. This would avoid creating a fresh VLM request for every window, preserve short-term visual context naturally, reduce repeated prompt overhead, and make the system closer to a real-time assistant.
