from __future__ import annotations

import csv
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

from .io import resolve_ffmpeg


def _moving_mean(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(radius * 2 + 1, dtype=np.float32) / (radius * 2 + 1)
    return np.convolve(padded, kernel, mode="valid")


def _read_mono_pcm(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM WAV")
        return source.getframerate(), np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(
            np.float32
        ) / 32768.0


def analyze_audio(
    input_path: Path, output_csv: Path, plot: Path | None = None, ffmpeg: Path | None = None
) -> dict[str, float | int | str]:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    wav_path = input_path
    if input_path.suffix.lower() != ".wav":
        temporary = tempfile.TemporaryDirectory(prefix="smart-badminton-audio-")
        wav_path = Path(temporary.name) / "audio.wav"
        subprocess.run(
            [
                str(resolve_ffmpeg(ffmpeg)),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )
    sample_rate, samples = _read_mono_pcm(wav_path)
    hop = max(1, round(sample_rate * 0.01))
    frames = samples[: len(samples) // hop * hop].reshape(-1, hop)
    diff = np.diff(frames, axis=1, prepend=frames[:, :1])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    high_rms = np.sqrt(np.mean(diff * diff, axis=1) + 1e-12)
    high_peak = np.max(np.abs(diff), axis=1)
    log_rms = 20.0 * np.log10(rms + 1e-7)
    log_high = 20.0 * np.log10(high_rms + 1e-7)
    crest = high_peak / (high_rms + 1e-7)
    radius = round(1.0 / 0.01)
    score = (log_high - _moving_mean(log_high, radius)) + np.maximum(crest - 3.0, 0.0) * 1.5
    score += np.maximum(log_rms - _moving_mean(log_rms, radius), 0.0) * 0.35
    threshold = float(max(np.percentile(score, 98.8), 8.0))
    minimum_gap = round(0.16 / 0.01)
    candidates: list[int] = []
    for index in range(1, len(score) - 1):
        if score[index] < threshold or score[index] < score[index - 1] or score[index] < score[index + 1]:
            continue
        if candidates and index - candidates[-1] < minimum_gap:
            if score[index] > score[candidates[-1]]:
                candidates[-1] = index
            continue
        candidates.append(index)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["time_seconds", "score", "log_rms_db", "log_high_db"])
        for index in candidates:
            writer.writerow(
                [f"{index * 0.01:.3f}", f"{score[index]:.3f}", f"{log_rms[index]:.3f}", f"{log_high[index]:.3f}"]
            )
    if plot is not None:
        duration = len(samples) / sample_rate
        width, height = max(1200, round(duration * 5)), 620
        canvas = np.full((height, width, 3), 250, dtype=np.uint8)
        top, bottom = 35, height - 55
        clipped = np.clip(score, np.percentile(score, 1), np.percentile(score, 99.9))
        low, high = float(clipped.min()), float(clipped.max())
        y = bottom - ((clipped - low) / max(high - low, 1e-6) * (bottom - top)).astype(np.int32)
        x = np.linspace(0, width - 1, len(score)).astype(np.int32)
        cv2.polylines(canvas, [np.column_stack((x, y)).reshape(-1, 1, 2)], False, (80, 80, 80), 1, cv2.LINE_AA)
        for index in candidates:
            event_x = round(index * 0.01 * 5)
            cv2.line(canvas, (event_x, top), (event_x, bottom), (30, 150, 30), 1)
        plot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(plot), canvas)
    if temporary is not None:
        temporary.cleanup()
    return {
        "duration_seconds": len(samples) / sample_rate,
        "events": len(candidates),
        "threshold": threshold,
        "output": str(output_csv),
    }
