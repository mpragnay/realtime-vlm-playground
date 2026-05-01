"""
Audio STT experiment runner.

Use this to iterate on transcription quality without running the VLM pipeline.
It extracts audio chunks from a video, sends each chunk to OpenRouter STT, and
writes a JSON report with chunk timestamps and transcripts.
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests
from dotenv import load_dotenv


def pcm16_mono_to_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_bytes)
    return buffer.getvalue()


def video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(total_frames / fps)


def extract_pcm_audio(video_path: str) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-f", "wav", "-",
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr[:1000]}")
    return result.stdout[44:]


def make_chunks(
    pcm_data: bytes,
    duration_sec: float,
    chunk_sec: float,
    overlap_sec: float,
    start_sec: float,
    end_sec: Optional[float],
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    bytes_per_second = sample_rate * 2
    end_limit = duration_sec if end_sec is None else min(end_sec, duration_sec)
    step_sec = chunk_sec - overlap_sec
    if step_sec <= 0:
        raise ValueError("--overlap-sec must be smaller than --chunk-sec")

    chunks = []
    t = start_sec
    while t < end_limit:
        chunk_end = min(t + chunk_sec, end_limit)
        start_byte = int(t * bytes_per_second)
        end_byte = int(chunk_end * bytes_per_second)
        chunks.append({
            "start_sec": round(t, 3),
            "end_sec": round(chunk_end, 3),
            "pcm": pcm_data[start_byte:end_byte],
        })
        t += step_sec
    return chunks


def call_openrouter_stt(
    api_key: str,
    wav_bytes: bytes,
    model: str,
    language: str,
    temperature: Optional[float],
    prompt: Optional[str],
) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(wav_bytes).decode("utf-8"),
            "format": "wav",
        },
        "language": language,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if prompt:
        # OpenAI's transcribe endpoint supports prompts. OpenRouter's STT docs
        # do not list it for every provider, so keep this optional for testing.
        payload["prompt"] = prompt

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = normalize_text(predicted).split()
    expected_tokens = normalize_text(expected).split()
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0

    remaining = predicted_tokens[:]
    overlap = 0
    for token in expected_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def phrase_score(predicted: str, expected: str) -> Dict[str, Any]:
    predicted_norm = normalize_text(predicted)
    expected_norm = normalize_text(expected)
    return {
        "expected_text": expected,
        "exact_normalized_match": predicted_norm == expected_norm,
        "expected_contained": bool(expected_norm) and expected_norm in predicted_norm,
        "sequence_similarity": round(SequenceMatcher(None, predicted_norm, expected_norm).ratio(), 3),
        "token_f1": round(token_f1(predicted, expected), 3),
    }


def load_expected(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    chunks = payload.get("chunks", payload if isinstance(payload, list) else [])
    if not isinstance(chunks, list):
        raise ValueError("Expected JSON must be a list or an object with a 'chunks' list")
    return chunks


def expected_for_chunk(
    expected_chunks: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
    tolerance_sec: float,
) -> Optional[Dict[str, Any]]:
    for item in expected_chunks:
        item_start = float(item.get("start_sec", -1))
        item_end = float(item.get("end_sec", -1))
        if abs(item_start - start_sec) <= tolerance_sec and abs(item_end - end_sec) <= tolerance_sec:
            return item
    return None


def evaluate_transcript(
    predicted: str,
    expected_item: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not expected_item:
        return {
            "skipped": True,
            "reason": "no expected chunk for this timestamp",
        }

    expected_text = expected_item.get("expected_text")
    expected_phrases = expected_item.get("expected_phrases", [])
    forbidden_phrases = expected_item.get("forbidden_phrases", [])
    if isinstance(expected_phrases, str):
        expected_phrases = [expected_phrases]
    if isinstance(forbidden_phrases, str):
        forbidden_phrases = [forbidden_phrases]

    eval_result: Dict[str, Any] = {
        "notes": expected_item.get("notes"),
    }
    scores = []
    if isinstance(expected_text, str) and expected_text.strip():
        scores.append(phrase_score(predicted, expected_text))
    for phrase in expected_phrases:
        if isinstance(phrase, str) and phrase.strip():
            scores.append(phrase_score(predicted, phrase))

    predicted_norm = normalize_text(predicted)
    forbidden_hits = [
        phrase for phrase in forbidden_phrases
        if isinstance(phrase, str) and normalize_text(phrase) in predicted_norm
    ]

    eval_result["phrase_scores"] = scores
    eval_result["forbidden_hits"] = forbidden_hits
    if scores:
        eval_result["best_token_f1"] = max(score["token_f1"] for score in scores)
        eval_result["best_sequence_similarity"] = max(score["sequence_similarity"] for score in scores)
        eval_result["passed"] = any(
            score["expected_contained"]
            or score["token_f1"] >= 0.75
            or score["sequence_similarity"] >= 0.82
            for score in scores
        ) and not forbidden_hits
    else:
        expected_silence = isinstance(expected_text, str) and not expected_text.strip()
        eval_result["expected_silence"] = expected_silence
        if expected_silence:
            eval_result["passed"] = predicted_norm in {"", "silence", "no audio", "no speech"}
        else:
            eval_result["passed"] = not forbidden_hits
    return eval_result


def summarize_eval(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    evaluated = [
        chunk for chunk in chunks
        if chunk.get("eval") is not None and not chunk["eval"].get("skipped")
    ]
    if not evaluated:
        return None
    passed = [chunk for chunk in evaluated if chunk["eval"].get("passed")]
    return {
        "evaluated_chunks": len(evaluated),
        "passed_chunks": len(passed),
        "failed_chunks": len(evaluated) - len(passed),
        "pass_rate": round(len(passed) / len(evaluated), 3),
        "mean_best_token_f1": round(
            sum(chunk["eval"].get("best_token_f1", 0.0) for chunk in evaluated) / len(evaluated),
            3,
        ),
        "mean_best_sequence_similarity": round(
            sum(chunk["eval"].get("best_sequence_similarity", 0.0) for chunk in evaluated) / len(evaluated),
            3,
        ),
    }


def attach_eval_to_chunks(
    chunks: List[Dict[str, Any]],
    expected_chunks: List[Dict[str, Any]],
    tolerance_sec: float,
) -> None:
    for chunk in chunks:
        expected_item = expected_for_chunk(
            expected_chunks,
            float(chunk["start_sec"]),
            float(chunk["end_sec"]),
            tolerance_sec,
        )
        if expected_item:
            chunk["expected"] = expected_item
        else:
            chunk.pop("expected", None)
        chunk["eval"] = evaluate_transcript(chunk.get("text") or "", expected_item)


def evaluate_existing_report(args: argparse.Namespace) -> None:
    if not args.expected:
        print("ERROR: --expected is required when using --input")
        sys.exit(1)
    report = json.loads(Path(args.input).read_text())
    chunks = report.get("chunks", [])
    if not isinstance(chunks, list):
        print("ERROR: input report does not contain a chunks list")
        sys.exit(1)
    expected_chunks = load_expected(args.expected)
    attach_eval_to_chunks(chunks, expected_chunks, args.expected_tolerance_sec)
    report["expected_file"] = args.expected
    report["eval_summary"] = summarize_eval(chunks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"Evaluated existing STT report: {args.input}")
    print(f"Saved: {output_path}")
    print(json.dumps(report["eval_summary"], indent=2))


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Experiment with STT over video audio chunks")
    parser.add_argument("--input", help="Existing STT JSON report to evaluate without rerunning STT")
    parser.add_argument("--video", help="Path to video MP4")
    parser.add_argument("--output", default="output/audio-stt-experiment.json",
                        help="Output JSON report path")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--model", default="openai/gpt-4o-transcribe",
                        help="OpenRouter STT model")
    parser.add_argument("--chunk-sec", type=float, default=5.0,
                        help="Chunk duration in seconds")
    parser.add_argument("--overlap-sec", type=float, default=0.0,
                        help="Overlap between chunks in seconds")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Start timestamp for experiment")
    parser.add_argument("--end-sec", type=float,
                        help="End timestamp for experiment")
    parser.add_argument("--language", default="en")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", help="Optional STT context prompt")
    parser.add_argument("--prompt-file", help="Read STT context prompt from a file")
    parser.add_argument("--expected", help="JSON file with expected_text/expected_phrases per chunk")
    parser.add_argument("--expected-tolerance-sec", type=float, default=0.25,
                        help="Timestamp tolerance for matching expected chunks")
    parser.add_argument("--save-wav-dir", help="Optional directory to save chunk WAV files")
    args = parser.parse_args()

    if args.input:
        evaluate_existing_report(args)
        return

    if not args.video:
        print("ERROR: --video is required unless --input is used")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    expected_chunks = load_expected(args.expected)
    if not expected_chunks:
        print("Note: no --expected file was provided, so eval entries will be marked skipped.")

    duration = video_duration_sec(args.video)
    pcm_data = extract_pcm_audio(args.video)
    chunks = make_chunks(
        pcm_data,
        duration_sec=duration,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    wav_dir = Path(args.save_wav_dir) if args.save_wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Audio STT experiment: {len(chunks)} chunks, model={args.model}")
    for idx, chunk in enumerate(chunks, start=1):
        start_sec = chunk["start_sec"]
        end_sec = chunk["end_sec"]
        wav_bytes = pcm16_mono_to_wav_bytes(chunk["pcm"])
        if wav_dir:
            wav_path = wav_dir / f"chunk-{idx:03d}-{start_sec:.1f}-{end_sec:.1f}.wav"
            wav_path.write_bytes(wav_bytes)

        record: Dict[str, Any] = {
            "index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        try:
            payload = call_openrouter_stt(
                api_key=api_key,
                wav_bytes=wav_bytes,
                model=args.model,
                language=args.language,
                temperature=args.temperature,
                prompt=prompt,
            )
            record["text"] = (payload.get("text") or "").strip()
            record["usage"] = payload.get("usage")
            expected_item = expected_for_chunk(expected_chunks, start_sec, end_sec, args.expected_tolerance_sec)
            if expected_item:
                record["expected"] = expected_item
            record["eval"] = evaluate_transcript(record["text"], expected_item)
            print(f"[{start_sec:6.1f}-{end_sec:6.1f}s] {record['text'] or '<empty>'}")
            if record.get("eval") is not None and not record["eval"].get("skipped"):
                verdict = "PASS" if record["eval"].get("passed") else "FAIL"
                print(
                    f"    eval={verdict} "
                    f"token_f1={record['eval'].get('best_token_f1', 0):.3f} "
                    f"similarity={record['eval'].get('best_sequence_similarity', 0):.3f}"
                )
        except requests.RequestException as exc:
            record["error"] = str(exc)
            print(f"[{start_sec:6.1f}-{end_sec:6.1f}s] ERROR: {exc}")
        results.append(record)

    report = {
        "video": args.video,
        "duration_sec": duration,
        "model": args.model,
        "chunk_sec": args.chunk_sec,
        "overlap_sec": args.overlap_sec,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "language": args.language,
        "temperature": args.temperature,
        "prompt": prompt,
        "expected_file": args.expected,
        "eval_summary": summarize_eval(results),
        "chunks": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
