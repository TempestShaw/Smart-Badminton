from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from .io import resolve_ffmpeg

H264_ENCODER_PRIORITY = (
    "h264_nvenc",
    "h264_qsv",
    "h264_videotoolbox",
    "libx264",
    "libopenh264",
)


def available_ffmpeg_encoders(ffmpeg: Path | None = None) -> set[str]:
    executable = resolve_ffmpeg(ffmpeg)
    result = subprocess.run(
        [str(executable), "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            encoders.add(parts[1])
    return encoders


def video_encoder_candidates(requested: str, ffmpeg: Path | None = None) -> tuple[list[str], str | None]:
    """Return every compiled H.264 candidate in safe retry order.

    FFmpeg builds often advertise a hardware encoder even when the matching
    driver or device is unavailable. Returning the complete order lets video
    jobs retry another real H.264 implementation instead of assuming that
    libx264 exists.
    """

    requested = requested.strip() or "auto"
    available = available_ffmpeg_encoders(ffmpeg)
    supported = [candidate for candidate in H264_ENCODER_PRIORITY if candidate in available]
    if requested == "auto":
        if supported:
            return supported, None
        raise RuntimeError("FFmpeg does not provide a supported H.264 encoder")

    candidates = []
    if requested in available:
        candidates.append(requested)
    candidates.extend(candidate for candidate in supported if candidate not in candidates)
    if not candidates:
        raise RuntimeError(f"Requested encoder {requested} is unavailable and no supported H.264 encoder is present")
    warning = None if candidates[0] == requested else f"{requested} is unavailable; falling back to {candidates[0]}"
    return candidates, warning


def choose_video_encoder(requested: str, ffmpeg: Path | None = None) -> tuple[str, str | None]:
    candidates, warning = video_encoder_candidates(requested, ffmpeg)
    return candidates[0], warning


def h264_encoding_arguments(encoder: str, quality: int, purpose: str = "final") -> list[str]:
    """Build conservative FFmpeg arguments for supported H.264 encoders."""

    quality = max(0, min(51, int(quality)))
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p5" if purpose == "final" else "p4", "-cq", str(quality), "-b:v", "0"]
    if encoder == "h264_qsv":
        return [
            "-c:v",
            encoder,
            "-preset",
            "medium" if purpose == "final" else "faster",
            "-global_quality",
            str(quality),
        ]
    if encoder == "h264_videotoolbox":
        # VideoToolbox uses a 1-100 quality scale in the opposite direction
        # from CRF/CQ. Keep the mapping bounded and deterministic.
        videotoolbox_quality = max(1, min(100, 100 - quality * 2))
        return ["-c:v", encoder, "-realtime", "true", "-q:v", str(videotoolbox_quality)]
    if encoder == "libx264":
        return [
            "-c:v",
            encoder,
            "-preset",
            "medium" if purpose == "final" else "veryfast",
            "-crf",
            str(quality),
        ]
    if encoder == "libopenh264":
        bitrate = "12M" if purpose == "final" else "4M" if purpose == "proxy" else "6M"
        return ["-c:v", encoder, "-b:v", bitrate]
    return ["-c:v", encoder, "-b:v", "12M"]


def probe_video_encoder(encoder: str, ffmpeg: Path | None = None) -> tuple[bool, str | None]:
    """Encode one synthetic frame so the UI reports a runnable encoder."""

    executable = resolve_ffmpeg(ffmpeg)
    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        # Some current NVENC drivers reject tiny 64x64 probes even though the
        # same encoder is healthy at ordinary video sizes.
        "color=c=black:s=320x180:r=30:d=0.2",
        "-frames:v",
        "1",
        *h264_encoding_arguments(encoder, 30, "preview"),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if result.returncode == 0:
        return True, None
    message = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    return False, message.splitlines()[-1] if message else f"exit code {result.returncode}"


def choose_working_video_encoder(
    requested: str,
    ffmpeg: Path | None = None,
) -> tuple[str, str | None, dict[str, str]]:
    candidates, selection_warning = video_encoder_candidates(requested, ffmpeg)
    failures: dict[str, str] = {}
    for candidate in candidates:
        working, reason = probe_video_encoder(candidate, ffmpeg)
        if working:
            warning_parts = [selection_warning] if selection_warning else []
            if failures:
                warning_parts.append(
                    f"{', '.join(failures)} could not start; using {candidate}"
                )
            return candidate, "; ".join(warning_parts) or None, failures
        failures[candidate] = reason or "encoder probe failed"
    attempted = ", ".join(candidates)
    details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
    raise RuntimeError(f"FFmpeg H.264 encoders were found but none could start ({attempted}). {details}")


def run_ffmpeg_with_encoder_fallback(
    ffmpeg: Path | None,
    requested: str,
    command_for: Callable[[str], list[str]],
    output: Path | None = None,
) -> str:
    """Run an FFmpeg job and retry every compiled H.264 implementation."""

    candidates, _warning = video_encoder_candidates(requested, ffmpeg)
    failures: list[tuple[str, subprocess.CalledProcessError]] = []
    for candidate in candidates:
        if output is not None:
            output.unlink(missing_ok=True)
        try:
            subprocess.run(command_for(candidate), check=True)
            return candidate
        except subprocess.CalledProcessError as error:
            failures.append((candidate, error))
    if output is not None:
        output.unlink(missing_ok=True)
    attempted = ", ".join(candidate for candidate, _error in failures)
    last_error = failures[-1][1]
    raise RuntimeError(
        f"FFmpeg could not encode H.264 with any available encoder ({attempted}); last exit code {last_error.returncode}"
    ) from last_error
