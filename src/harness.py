"""
VLM Orchestrator — Streaming Harness

Simulates real-time video+audio delivery and measures detection latency.
This is the test harness candidates run their pipeline against.

The harness:
  1. Reads a video file and emits frames + audio chunks at real-time speed
  2. Calls the candidate's pipeline callback for each frame/audio chunk
  3. Collects emitted events and timestamps when they were emitted
  4. Measures detection delay: wall-clock time from when a video moment
     passed to when the pipeline reported a detection for it

Usage:
    from src.harness import StreamingHarness

    harness = StreamingHarness(
        video_path="path/to/video.mp4",
        procedure_path="data/procedures/change_circuit_breaker.json",
        speed=1.0,  # 1.0 = real-time, 2.0 = 2x speed, etc.
    )

    # Your pipeline registers callbacks
    harness.on_frame(my_frame_handler)      # called with (frame, timestamp_sec)
    harness.on_audio(my_audio_handler)      # called with (audio_chunk, timestamp_sec)

    # When your pipeline detects something, it calls:
    harness.emit_event({
        "timestamp_sec": 49.7,
        "type": "step_completion",
        "step_id": 1,
        ...
    })

    # Run the simulation
    results = harness.run()
    # results contains events + detection delays + total time
"""

import json
import time
import io
import base64
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


@dataclass
class EmittedEvent:
    """An event emitted by the candidate's pipeline, with timing metadata."""
    event: Dict[str, Any]          # The event dict the candidate emitted
    wall_time: float               # Wall-clock time (seconds since harness start) when emitted
    video_time_at_emission: float  # What video timestamp the harness had reached when event was emitted
    detection_delay_sec: float     # video_time_at_emission - event["timestamp_sec"]


@dataclass
class HarnessResults:
    """Results from a streaming harness run."""
    task: str
    video_source: str
    procedure_path: str
    speed: float
    start_time: str
    end_time: str
    video_duration_sec: float
    wall_duration_sec: float
    total_frames_delivered: int
    total_audio_chunks_delivered: int
    events: List[Dict[str, Any]]     # Events in output schema format (with detection_delay_sec added)
    mean_detection_delay_sec: float
    max_detection_delay_sec: float


class StreamingHarness:
    """
    Simulates real-time video+audio streaming and measures detection latency.

    The harness plays through a video at a configurable speed, delivering
    frames and audio chunks to registered callbacks. When the candidate's
    pipeline detects an event, it calls emit_event(). The harness records
    the wall-clock time and computes detection delay.
    """

    def __init__(
        self,
        video_path: str,
        procedure_path: str,
        speed: float = 1.0,
        frame_fps: float = 2.0,
        audio_chunk_sec: float = 5.0,
    ):
        """
        Args:
            video_path: Path to MP4 file (with audio)
            procedure_path: Path to procedure JSON
            speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x faster)
            frame_fps: How many frames per second to deliver to the pipeline
            audio_chunk_sec: Audio chunk duration in seconds
        """
        self.video_path = video_path
        self.procedure_path = procedure_path
        self.speed = speed
        self.frame_fps = frame_fps
        self.audio_chunk_sec = audio_chunk_sec

        self._frame_callbacks: List[Callable] = []
        self._audio_callbacks: List[Callable] = []
        self._complete_callbacks: List[Callable] = []
        self._emitted_events: List[EmittedEvent] = []
        self._start_wall_time: float = 0
        self._current_video_time: float = 0
        self._lock = threading.Lock()

        # Load procedure
        with open(procedure_path) as f:
            self.procedure = json.load(f)
        self.task_name = self.procedure.get("task") or self.procedure.get("task_name", "Unknown")

    def on_frame(self, callback: Callable[[np.ndarray, float, str], None]):
        """
        Register a frame callback.

        Your callback receives:
            frame: BGR numpy array
            timestamp_sec: current video timestamp
            frame_base64: JPEG-encoded base64 string (ready for VLM API)
        """
        self._frame_callbacks.append(callback)

    def on_audio(self, callback: Callable[[bytes, float, float], None]):
        """
        Register an audio callback.

        Your callback receives:
            audio_bytes: raw PCM audio bytes for this chunk
            start_sec: chunk start timestamp in the video
            end_sec: chunk end timestamp in the video
        """
        self._audio_callbacks.append(callback)

    def on_complete(self, callback: Callable[[], None]):
        """Register a callback to run after streaming finishes, before results are built."""
        self._complete_callbacks.append(callback)

    VALID_EVENT_TYPES = {"step_completion", "error_detected", "idle_detected"}
    VALID_SOURCES = {"video", "audio", "both"}
    VALID_ERROR_TYPES = {"wrong_action", "wrong_sequence", "safety_violation", "improper_technique", "other"}
    VALID_SEVERITIES = {"info", "warning", "critical"}

    def _validate_event(self, event: Dict[str, Any]) -> List[str]:
        """Validate an event against the schema. Returns list of error messages (empty = valid)."""
        errors = []

        # Required fields
        if "timestamp_sec" not in event:
            errors.append("Missing required field: timestamp_sec")
        elif not isinstance(event["timestamp_sec"], (int, float)):
            errors.append(f"timestamp_sec must be a number, got {type(event['timestamp_sec']).__name__}")

        if "type" not in event:
            errors.append("Missing required field: type")
        elif event["type"] not in self.VALID_EVENT_TYPES:
            errors.append(f"Invalid event type: '{event['type']}'. Must be one of {self.VALID_EVENT_TYPES}")

        # Type-specific validation
        event_type = event.get("type")

        if event_type == "step_completion":
            if "step_id" not in event:
                errors.append("step_completion event missing required field: step_id")
            elif not isinstance(event["step_id"], int):
                errors.append(f"step_id must be an integer, got {type(event['step_id']).__name__}")

        if event_type == "error_detected":
            if "error_type" in event and event["error_type"] not in self.VALID_ERROR_TYPES:
                errors.append(f"Invalid error_type: '{event['error_type']}'. Must be one of {self.VALID_ERROR_TYPES}")
            if "severity" in event and event["severity"] not in self.VALID_SEVERITIES:
                errors.append(f"Invalid severity: '{event['severity']}'. Must be one of {self.VALID_SEVERITIES}")

        # Optional field validation
        if "confidence" in event:
            conf = event["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                errors.append(f"confidence must be a number between 0 and 1, got {conf}")

        if "source" in event and event["source"] not in self.VALID_SOURCES:
            errors.append(f"Invalid source: '{event['source']}'. Must be one of {self.VALID_SOURCES}")

        return errors

    def emit_event(self, event: Dict[str, Any]):
        """
        Call this from your pipeline when you detect an event.

        The harness records the wall-clock time and computes detection delay.
        Detection delay = wall_clock_elapsed * speed - event_timestamp.
        This measures how far past the event (in video-time) the real world
        has advanced by the time the pipeline reports it.

        You can call this from any thread.

        Args:
            event: Dict matching the event schema (must have timestamp_sec and type)

        Raises:
            ValueError: If the event fails schema validation
        """
        validation_errors = self._validate_event(event)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            raise ValueError(f"Invalid event: {error_msg}")

        wall_now = time.monotonic() - self._start_wall_time
        # Convert wall time back to video-time equivalent
        video_time_equivalent = wall_now * self.speed
        event_video_time = event.get("timestamp_sec", 0)
        delay = video_time_equivalent - event_video_time

        with self._lock:
            self._emitted_events.append(EmittedEvent(
                event=event,
                wall_time=wall_now,
                video_time_at_emission=video_time_equivalent,
                detection_delay_sec=max(0, delay),
            ))

    def _extract_audio_chunks(self) -> List[Tuple[bytes, float, float]]:
        """Extract audio as PCM chunks using ffmpeg."""
        chunks = []
        try:
            # Get duration
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps
            cap.release()

            # Extract full audio as WAV PCM
            result = subprocess.run(
                [
                    "ffmpeg", "-i", self.video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-f", "wav", "-"
                ],
                capture_output=True, timeout=60,
            )

            if result.returncode != 0:
                print(f"  [harness] Warning: could not extract audio (ffmpeg returned {result.returncode})")
                return []

            audio_data = result.stdout
            # Skip WAV header (44 bytes)
            pcm_data = audio_data[44:]
            sample_rate = 16000
            bytes_per_sample = 2  # 16-bit

            chunk_samples = int(self.audio_chunk_sec * sample_rate)
            chunk_bytes = chunk_samples * bytes_per_sample

            t = 0.0
            offset = 0
            while offset < len(pcm_data):
                end_offset = min(offset + chunk_bytes, len(pcm_data))
                chunk = pcm_data[offset:end_offset]
                end_t = min(t + self.audio_chunk_sec, duration)
                chunks.append((chunk, t, end_t))
                t = end_t
                offset = end_offset

        except Exception as e:
            print(f"  [harness] Warning: audio extraction failed: {e}")

        return chunks

    @staticmethod
    def frame_to_base64(frame: np.ndarray) -> str:
        """Convert BGR frame to base64 JPEG."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def run(self) -> HarnessResults:
        """
        Run the streaming simulation.

        Delivers frames and audio at real-time speed (adjusted by self.speed).
        Returns results with all emitted events and timing data.
        """
        print(f"{'=' * 60}")
        print(f"  STREAMING HARNESS")
        print(f"{'=' * 60}")
        print(f"  Task:      {self.task_name}")
        print(f"  Video:     {self.video_path}")
        print(f"  Speed:     {self.speed}x real-time")
        print(f"  Frame FPS: {self.frame_fps}")
        print(f"  Audio:     {self.audio_chunk_sec}s chunks")
        print()

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps

        # Compute frame interval (in video-time seconds)
        frame_interval = 1.0 / self.frame_fps

        # Extract audio chunks upfront
        audio_chunks = self._extract_audio_chunks()
        next_audio_idx = 0
        total_audio_delivered = 0

        print(f"  Video:     {video_duration:.1f}s @ {video_fps:.0f}fps ({total_frames} frames)")
        print(f"  Audio:     {len(audio_chunks)} chunks")
        print(f"  Delivering ~{int(video_duration * self.frame_fps)} frames to pipeline")
        print()
        print(f"  Starting simulation...")
        print()

        self._start_wall_time = time.monotonic()
        start_dt = datetime.utcnow().isoformat() + "Z"

        frames_delivered = 0
        next_frame_video_time = 0.0

        while next_frame_video_time < video_duration:
            # Seek to the right frame
            frame_number = int(next_frame_video_time * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            # Wait until real-time catches up (respecting speed multiplier)
            target_wall_time = next_frame_video_time / self.speed
            elapsed = time.monotonic() - self._start_wall_time
            if elapsed < target_wall_time:
                time.sleep(target_wall_time - elapsed)

            # Update current video time
            with self._lock:
                self._current_video_time = next_frame_video_time

            # Deliver audio chunks that fall before this frame
            while next_audio_idx < len(audio_chunks):
                audio_bytes, audio_start, audio_end = audio_chunks[next_audio_idx]
                if audio_start <= next_frame_video_time:
                    for cb in self._audio_callbacks:
                        try:
                            cb(audio_bytes, audio_start, audio_end)
                        except Exception as e:
                            print(f"  [harness] Audio callback error at {audio_start:.1f}s: {e}")
                    next_audio_idx += 1
                    total_audio_delivered += 1
                else:
                    break

            # Deliver frame
            frame_b64 = self.frame_to_base64(frame)
            for cb in self._frame_callbacks:
                try:
                    cb(frame, next_frame_video_time, frame_b64)
                except Exception as e:
                    print(f"  [harness] Frame callback error at {next_frame_video_time:.1f}s: {e}")

            frames_delivered += 1
            if frames_delivered % 10 == 0:
                print(f"  [{next_frame_video_time:.1f}s / {video_duration:.1f}s] "
                      f"{frames_delivered} frames, {len(self._emitted_events)} events detected")

            next_frame_video_time += frame_interval

        cap.release()

        # Final update
        with self._lock:
            self._current_video_time = video_duration

        for cb in self._complete_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"  [harness] Completion callback error: {e}")

        wall_duration = time.monotonic() - self._start_wall_time
        end_dt = datetime.utcnow().isoformat() + "Z"

        # Build output events with detection_delay_sec
        output_events = []
        delays = []
        for ee in self._emitted_events:
            ev = dict(ee.event)
            ev["detection_delay_sec"] = round(ee.detection_delay_sec, 3)
            output_events.append(ev)
            delays.append(ee.detection_delay_sec)

        mean_delay = sum(delays) / len(delays) if delays else 0
        max_delay = max(delays) if delays else 0

        print()
        print(f"  {'=' * 56}")
        print(f"  Simulation complete")
        print(f"  {'=' * 56}")
        print(f"  Frames delivered:  {frames_delivered}")
        print(f"  Audio delivered:   {total_audio_delivered} chunks")
        print(f"  Events detected:   {len(output_events)}")
        print(f"  Wall time:         {wall_duration:.1f}s")
        print(f"  Mean detect delay: {mean_delay:.2f}s")
        print(f"  Max detect delay:  {max_delay:.2f}s")

        return HarnessResults(
            task=self.task_name,
            video_source=self.video_path,
            procedure_path=self.procedure_path,
            speed=self.speed,
            start_time=start_dt,
            end_time=end_dt,
            video_duration_sec=video_duration,
            wall_duration_sec=wall_duration,
            total_frames_delivered=frames_delivered,
            total_audio_chunks_delivered=total_audio_delivered,
            events=output_events,
            mean_detection_delay_sec=round(mean_delay, 3),
            max_detection_delay_sec=round(max_delay, 3),
        )

    def save_results(self, results: HarnessResults, output_path: str):
        """Save results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"  Results saved to: {output_path}")
