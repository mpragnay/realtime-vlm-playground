# VLM Orchestrator — Take-Home Assignment

**Alcor Labs** | Realtime VLM Engineering

---

## What You're Building

A real-time procedural assistant. Given a video (with audio) of a technician performing a task and a procedure JSON describing the expected steps, your pipeline must detect:

- **Step completions** — when each procedure step is finished
- **Errors** — when the technician does something wrong (before the instructor corrects them)
- **Idle periods** *(optional bonus, not scored)* — when the technician pauses between steps

You call VLMs through the OpenRouter API. We provide a streaming harness that simulates real-time playback and measures how fast you detect events.

---

## What You're Given

- **Procedure JSONs** — step-by-step SOPs in `data/clip_procedures/` (one per video clip)
- **Video clips** — MP4 with video + audio ([download from Google Drive](https://drive.google.com/drive/folders/1SDgpLC154P0nw2jQmknmgH5J9lLEieb5?usp=sharing))
- **Streaming harness** — `src/harness.py` feeds frames and audio to your callbacks in real-time
- **Evaluator** — `src/evaluator.py` scores your output against ground truth
- **Dashboard** — `src/dashboard.py` generates an HTML timeline comparing your output to ground truth
- **Sample ground truth** — `data/ground_truth_sample/` for self-testing
- **OpenRouter API key** — provided separately
- **Starter code** — `src/run.py` with scaffolding and a working VLM call function

The audio is pitch-shifted for privacy. Both the performer and an instructor are audible — the instructor gives verbal corrections that are a strong signal for error detection.

---

## How the Streaming Harness Works

The harness (`src/harness.py`) simulates real-time video playback. It is the **input** side of the pipeline — it delivers data to your code at controlled speed and collects the events you emit as **output**.

### Input delivery (harness → your pipeline)

```
1. STARTUP
   ├─ Opens the video file with OpenCV
   ├─ Reads video metadata: native FPS, total frames, duration
   ├─ Extracts ALL audio upfront via ffmpeg → 16kHz mono 16-bit PCM
   ├─ Chunks audio into segments (default 5s each)
   └─ Records wall-clock start time

2. MAIN LOOP (iterates over video timeline at configured speed)
   For each frame delivery point (every 1/frame_fps seconds of video time):

   a) TIMING — compute when this frame should be delivered:
      target_wall_time = video_timestamp / speed
      If wall clock hasn't reached target yet → sleep(target - elapsed)
      This enforces real-time pacing. At speed=1.0, a frame at t=10s
      is delivered 10s after start. At speed=10.0, it's delivered at 1s.

   b) AUDIO — deliver any audio chunks whose start_sec ≤ current video time:
      Calls: pipeline.on_audio(pcm_bytes, start_sec, end_sec)
      Audio is delivered BEFORE the frame at the same timestamp.

   c) FRAME — decode and deliver the frame:
      - Seeks to the exact frame number: int(video_timestamp × native_fps)
      - Reads the BGR frame via OpenCV
      - Converts to JPEG base64 (quality=80) ready for VLM API
      - Calls: pipeline.on_frame(frame_bgr, timestamp_sec, frame_base64)

   d) ADVANCE — increment video time by 1/frame_fps

3. COMPLETION
   └─ Returns HarnessResults with all emitted events + timing metadata
```

### Output collection (your pipeline → harness)

When your pipeline detects something, it calls `harness.emit_event({...})`. The harness:

1. **Validates** the event against the schema (type, timestamp_sec required; step_id required for step_completion; confidence must be 0-1; etc.). Raises `ValueError` on invalid events.
2. **Records wall-clock time** of emission
3. **Computes detection delay**: `(wall_elapsed × speed) - event_timestamp_sec`
   - This measures how far past the event (in video-time) the real world has advanced by the time you report it
   - Example: at speed=1.0, if a step completes at t=30s and you emit at wall-clock 33s, delay = 3s
   - At speed=10.0, wall-clock 33s = video-time 330s, so delay = 330 - 30 = 300s (speed makes delay worse because your API calls block the timeline)
4. **Thread-safe** — you can call `emit_event()` from any thread

### Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `speed` | 1.0 | Playback multiplier. 1.0 = real-time, 10.0 = 10x faster (for development) |
| `frame_fps` | 2.0 | Frames delivered per second of video time. At 2.0, a 3-min video = 360 frames |
| `audio_chunk_sec` | 5.0 | Audio chunk duration. 5.0 = one chunk every 5 seconds |

### What you need to decide

- **Which frames to send to the VLM** — you receive every frame at `frame_fps` rate, but sending all of them to the API is expensive. Smart sampling is key.
- **Whether/how to use audio** — the instructor's verbal corrections are a strong error signal, but you need speech-to-text to use them.
- **Which models to use** — cheap model for routine frames, expensive model for ambiguous moments.
- **How to track procedure state** — which step you're on, what's been completed, what errors to watch for.

---

## Evaluation Criteria (100 points)

| Category | Points | What we look for |
|----------|--------|-----------------|
| **Architecture & Design** | 25 | Modular pipeline, smart frame sampling, audio usage, procedure state tracking, prompt optimizing |
| **Accuracy** | 40 | Combined score on evaluation clips (we re-run your pipeline at 1x speed) |
| **Detection Latency** | 20 | Mean delay for your response |
| **Code Quality & Cost** | 15 | Clean code, git history, logging, cost efficiency, use of coding agents |

You receive 15 training clips with ground truth for development and self-testing. Your pipeline will also be evaluated on additional test clips not included in this repo — plan for generalization. The provided training clips are representative of the full evaluation set.


### How the evaluator scores your output

The evaluator (`src/evaluator.py`) compares your emitted events against ground truth annotations.

**Step completion matching** — a predicted `step_completion` matches a ground truth `step_completion` if:
- Same `step_id` AND
- `|predicted_timestamp - gt_timestamp| ≤ tolerance` (default ±5 seconds)
- When multiple predictions could match the same GT event, the **closest match wins** (optimal greedy bipartite matching sorted by distance)

**Error detection matching** — a predicted `error_detected` matches a ground truth `error_detected` if:
- `|predicted_timestamp - gt_timestamp| ≤ tolerance` (default ±5 seconds)
- Same closest-first matching as steps (no step_id required — just timestamp proximity)

**For each category (steps, errors), the evaluator computes:**

```
Precision = True Positives / (True Positives + False Positives)
            "Of everything you detected, what fraction was correct?"

Recall    = True Positives / (True Positives + False Negatives)
            "Of everything that actually happened, what fraction did you detect?"

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
            Harmonic mean — balances precision and recall. 0% = nothing correct, 100% = perfect.
```

**Latency scoring:**
```
latency_score = max(0, 1.0 - mean_detection_delay / 10.0)
```
0s mean delay = 1.0 (perfect), 10s+ = 0.0. Linear interpolation.

**Combined score (automated):**
```
combined = 0.40 × step_f1 + 0.40 × error_f1 + 0.20 × latency_score
```

Code quality, cost efficiency, and architecture design are evaluated manually and also contribute to your overall score.

### Ground truth timing conventions

- **Step completions** are timestamped at the **end** of each step (the moment it's considered done). Your model should detect a step completion as soon as possible after the step finishes.
- **Errors** are timestamped at the **start** of the wrong action (the moment the mistake begins). Your model should detect the error before or simultaneously with the instructor's verbal correction, which typically comes 2-5 seconds later.

---

## Streaming Limitations & Bonus

OpenRouter supports **streaming output** (`stream: true` — SSE), but **not streaming input**. Each VLM call is a separate HTTP request; there is no persistent WebSocket connection for continuous frame/audio input.

**Bonus points** in your technical report: describe how you would redesign the pipeline if you had access to a bidirectional streaming API (e.g., Gemini Live API) where you can continuously pipe frames and audio and receive events in real-time.

---

## Quick Start

```bash
# 1. Create your repo from the template (click "Use this template" on GitHub)
#    Then clone your new repo:
git clone https://github.com/YOUR_USERNAME/realtime-vlm-playground.git
cd realtime-vlm-playground

# 2. Download & unzip the training video clips into the repo
#    Download from: https://drive.google.com/drive/folders/1SDgpLC154P0nw2jQmknmgH5J9lLEieb5?usp=sharing
unzip videos.zip      # extracts to data/videos_full/

# 3. Set up environment
export OPENROUTER_API_KEY=your_key_here
make setup
make dry-run  # validate inputs
```

Requires Python 3.11+, [FFmpeg](https://ffmpeg.org/download.html) (system install), OpenCV.

---

## Running Your Pipeline

Implement `Pipeline.on_frame()` and `Pipeline.on_audio()` in `src/run.py`.

```bash
# Run at real-time (evaluation speed)
python src/run.py \
    --procedure data/clip_procedures/CLIP.json \
    --video path/to/video.mp4 \
    --output output/events.json \
    --speed 1.0

# Run at 10x for development
python src/run.py \
    --procedure data/clip_procedures/CLIP.json \
    --video path/to/video.mp4 \
    --output output/events.json \
    --speed 10.0
```

---

## Output Format

Your pipeline produces a JSON event log. See `data/schema/event_log.schema.json` for the full spec and `data/schema/example_output.json` for an example.

Each event has `timestamp_sec` and `type` (required), plus optional fields:

```json
{"timestamp_sec": 49.7, "type": "step_completion", "step_id": 1, "confidence": 0.92, "description": "...", "source": "video"}
{"timestamp_sec": 13.2, "type": "error_detected", "error_type": "wrong_action", "severity": "warning", "description": "...", "spoken_response": "Stop — wrong toolbox.", "source": "video"}
```

`detection_delay_sec` is added automatically by the harness. `source` can be `video`, `audio`, or `both`.

---

## Self-Evaluation

```bash
# Evaluate against ground truth
python -m src.evaluator \
    --predicted output/events.json \
    --ground-truth data/ground_truth_sample/CLIP.json \
    --tolerance 5

# Generate visual dashboard (single clip)
python -m src.dashboard \
    --predicted output/events.json \
    --ground-truth data/ground_truth_sample/CLIP.json \
    --output output/dashboard.html

# Generate dashboard comparing multiple clips (tabs)
python -m src.dashboard \
    --multi output/clip1.json:data/ground_truth_sample/clip1.json \
           output/clip2.json:data/ground_truth_sample/clip2.json \
    --output output/multi_dashboard.html
```

The dashboard shows an interactive SVG timeline with ground truth step bands, error markers, and your predictions aligned on the same time axis. Hover or click any element for full details.

---

## Models

Use **any model on OpenRouter**. Check [OpenRouter models](https://openrouter.ai/models) for the full catalog and current pricing.
**You can use the key to run your CODING AGENTS (Cursor supports OpenRouter)**

---

## Project Structure

```
src/run.py              # YOUR WORK (Pipeline class)
src/harness.py          # Streaming harness (do not modify)
src/evaluator.py        # Scoring
src/dashboard.py        # Visual evaluation dashboard
src/data_loader.py      # Video/JSON utilities
data/clip_procedures/   # Per-clip procedure JSONs (one per training video — use as pipeline input)
data/procedures/        # Canonical SOPs (generic reference for each task type)
data/schema/            # Output schema + example
data/ground_truth_sample/  # Ground truth for self-testing (matches clip_procedures)
```

---

## Deliverables

Submit within **7 days**.

1. **Code** — `src/run.py` implementation + any supporting modules. Clean git history.
2. **Technical report** (1 page) — architecture, frame sampling strategy, audio usage, model selection, cost breakdown, latency analysis. Bonus: bidirectional streaming redesign.
3. **Demo video** (~3 min, optional) — your pipeline running on a clip.

Push to your private repo, invite the evaluation team (**gabe@alcor-labs.com** / **elior@alcor-labs.com**), and email the team with the repo link.

---

## Tips

- Start at `--speed 10`, validate at `--speed 1` before submitting
- Use `stream=True` in VLM calls for lower latency
- The instructor's verbal corrections in audio = strong error signal
- Don't send every frame — the harness delivers at `--frame-fps` but you choose which to send to the API
- Run the evaluator and dashboard early to understand scoring
- Monitor your spend on the [OpenRouter dashboard](https://openrouter.ai/activity)

Questions? **contact@alcor-labs.com**

---
*Alcor Labs — March 2026*
