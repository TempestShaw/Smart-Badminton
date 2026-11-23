from __future__ import annotations

import csv
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

from .encoding import has_audio_stream
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
        ffmpeg_path = resolve_ffmpeg(ffmpeg)
        if not has_audio_stream(ffmpeg_path, input_path):
            raise ValueError(f"{input_path.name} has no audio track; rally analysis needs the sound of the hits")
        temporary = tempfile.TemporaryDirectory(prefix="smart-badminton-audio-")
        wav_path = Path(temporary.name) / "audio.wav"
        subprocess.run(
            [
                str(ffmpeg_path),
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


class StreamingHitDetector:
    """The onset detector of :func:`analyze_audio`, fed incrementally from a live microphone or stream.

    Scores use the same 10 ms features against a ±1 s moving baseline, so each hop is final one second after it is
    heard. The adaptive threshold is the 98.8th percentile of the last minute instead of the whole file.
    """

    hop_seconds = 0.01
    radius = 100
    minimum_gap = 16

    def __init__(self, sample_rate: int = 16000, history_seconds: float = 60.0):
        self.sample_rate = sample_rate
        self.hop = max(1, round(sample_rate * self.hop_seconds))
        self.pending = np.zeros(0, dtype=np.float32)
        self.log_high: list[float] = []
        self.log_rms: list[float] = []
        self.crest: list[float] = []
        self.offset = 0  # absolute hop index of element 0 in the feature lists
        self.finalized = 0  # absolute hop index of the next score to compute
        self.scores: list[float] = []  # scores from absolute index ``offset_scores``
        self.offset_scores = 0
        self.history: list[float] = []
        self.history_limit = round(history_seconds / self.hop_seconds)
        self.last_event: tuple[int, float] | None = None
        self.emitted: list[tuple[float, float]] = []

    def push(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """Add mono float samples in [-1, 1]; returns hits ``(time_seconds, score)`` that are now final."""
        self.pending = np.concatenate([self.pending, np.asarray(samples, dtype=np.float32)])
        whole = len(self.pending) // self.hop * self.hop
        if whole:
            frames = self.pending[:whole].reshape(-1, self.hop)
            self.pending = self.pending[whole:]
            diff = np.diff(frames, axis=1, prepend=frames[:, :1])
            rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
            high_rms = np.sqrt(np.mean(diff * diff, axis=1) + 1e-12)
            high_peak = np.max(np.abs(diff), axis=1)
            self.log_rms.extend((20.0 * np.log10(rms + 1e-7)).tolist())
            self.log_high.extend((20.0 * np.log10(high_rms + 1e-7)).tolist())
            self.crest.extend((high_peak / (high_rms + 1e-7)).tolist())
        events: list[tuple[float, float]] = []
        total = self.offset + len(self.log_high)
        log_high, log_rms = np.asarray(self.log_high), np.asarray(self.log_rms)
        while self.finalized + self.radius < total:
            local = self.finalized - self.offset
            low, high = max(0, local - self.radius), local + self.radius + 1
            score = (log_high[local] - float(np.mean(log_high[low:high]))) + max(self.crest[local] - 3.0, 0.0) * 1.5
            score += max(log_rms[local] - float(np.mean(log_rms[low:high])), 0.0) * 0.35
            self.scores.append(score)
            self.history.append(score)
            self.finalized += 1
            events.extend(self._pick_peak())
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        keep = self.radius * 2 + 2
        if len(self.log_high) > keep * 4:
            drop = len(self.log_high) - keep
            del self.log_high[:drop], self.log_rms[:drop], self.crest[:drop]
            self.offset += drop
        if len(self.scores) > 8:
            drop = len(self.scores) - 3
            del self.scores[:drop]
            self.offset_scores += drop
        return events

    def _pick_peak(self) -> list[tuple[float, float]]:
        # The middle of the last three scores is a local maximum candidate.
        if len(self.scores) < 3:
            return []
        index = self.offset_scores + len(self.scores) - 2
        output: list[tuple[float, float]] = []
        if self.last_event is not None and index - self.last_event[0] >= self.minimum_gap:
            output.append((self.last_event[0] * self.hop_seconds, self.last_event[1]))  # merge window has passed
            self.last_event = None
        previous, current, following = self.scores[-3], self.scores[-2], self.scores[-1]
        threshold = max(float(np.percentile(self.history, 98.8)) if len(self.history) >= 500 else 8.0, 8.0)
        if current < threshold or current < previous or current < following:
            return output
        if self.last_event is not None:
            if current > self.last_event[1]:
                self.last_event = (index, current)
            return output
        self.last_event = (index, current)
        return output

    def flush(self) -> list[tuple[float, float]]:
        if self.last_event is None:
            return []
        event = (self.last_event[0] * self.hop_seconds, self.last_event[1])
        self.last_event = None
        return [event]
