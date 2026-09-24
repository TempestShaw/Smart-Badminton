from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..encoding import (
    choose_working_video_encoder,
    h264_encoding_arguments,
    has_audio_stream,
    run_ffmpeg_with_encoder_fallback,
)
from ..io import resolve_ffmpeg
from .calibration import calibration_ready
from .project import public_segments
from .state import StudioState


def runtime_payload(state: StudioState) -> dict[str, Any]:
    """Probe FFmpeg once; the UI shows the reason when no working H.264 encoder exists."""
    if state.runtime_cache.get("ffmpeg"):
        return state.runtime_cache
    ffmpeg: Path | None = None
    try:
        ffmpeg = resolve_ffmpeg(state.ffmpeg)
        selected, warning, probe_failures = choose_working_video_encoder(state.encoder, ffmpeg)
        state.runtime_cache["ffmpeg"] = {
            "available": True,
            "path": str(ffmpeg),
            "reason": None,
            "requested_encoder": state.encoder,
            "selected_encoder": selected,
            "warning": warning,
            "probe_failures": probe_failures,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        state.runtime_cache["ffmpeg"] = {
            "available": False,
            "path": str(ffmpeg) if ffmpeg else None,
            "reason": str(error),
            "requested_encoder": state.encoder,
            "selected_encoder": None,
            "warning": None,
            "probe_failures": {},
        }
    return state.runtime_cache


def ffmpeg_available(state: StudioState) -> bool:
    return bool(runtime_payload(state)["ffmpeg"]["available"])


NO_AUDIO_MESSAGE = "此视频没有音轨，自动分析需要击球声，无法分析"


@lru_cache(maxsize=64)
def _has_audio(ffmpeg: str, video: str, mtime_ns: int, size: int) -> bool:
    del mtime_ns, size
    return has_audio_stream(Path(ffmpeg), Path(video))


def audio_available(state: StudioState, video: Path | None = None) -> bool:
    """Automatic analysis depends on hit sounds, so a video without audio cannot be analyzed."""
    if not ffmpeg_available(state):
        return False
    video = (video or state.video).resolve()
    stat = video.stat()
    return _has_audio(runtime_payload(state)["ffmpeg"]["path"], str(video), stat.st_mtime_ns, stat.st_size)


def record_encoder(state: StudioState, used_encoder: str, task: str) -> None:
    runtime = runtime_payload(state)["ffmpeg"]
    if runtime.get("selected_encoder") != used_encoder:
        runtime["warning"] = f"{runtime.get('selected_encoder')} could not complete the {task}; using {used_encoder}"
        runtime["selected_encoder"] = used_encoder


def make_proxy(state: StudioState, video: Path) -> Path:
    layout = state.layout(video)
    existing = layout.existing_proxy()
    if existing is not None:
        return existing
    proxy_root = (layout.root if layout.is_match_project else state.library_root / "Analysis" / "Auto" / video.stem)
    proxy = proxy_root / "Proxy" / f"{video.stem}_proxy_720p.mp4"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(state.ffmpeg)

    def command(encoder: str) -> list[str]:
        return [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "scale=-2:720:flags=lanczos,fps=30",
            *h264_encoding_arguments(encoder, 28, "proxy"),
            *["-g", "30", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(proxy)],
        ]

    record_encoder(state, run_ffmpeg_with_encoder_fallback(ffmpeg, state.encoder, command, proxy), "proxy")
    return proxy


def pose_overlay_path(state: StudioState, rally_number: int) -> tuple[Path, dict[str, Any]]:
    segments = public_segments(state.rallies)
    if not 1 <= rally_number <= len(segments):
        raise ValueError(f"Rally {rally_number} does not exist; the timeline has {len(segments)} rallies")
    segment = segments[rally_number - 1]
    start, end = float(segment["start"]), float(segment["end"])
    name = f"pose_v3_rally_{rally_number:03d}_{round(start * 1000):09d}_{round(end * 1000):09d}.mp4"
    return state.layout().analysis.root / "Pose_Overlays" / name, segment


def full_pose_overlay_path(state: StudioState) -> Path:
    return state.layout().analysis.root / "Pose_Overlays" / "pose_v4_full_video.mp4"


def pose_configured(state: StudioState) -> bool:
    return bool(calibration_ready(state) and state.pose_model and state.pose_model.exists() and ffmpeg_available(state))


def pose_status_payload(state: StudioState) -> dict[str, Any]:
    output = full_pose_overlay_path(state)
    stale_sources: list[str] = []
    if output.exists():
        overlay_mtime = output.stat().st_mtime_ns
        stale_sources = [
            reason
            for path, reason in (
                (state.video, "原视频已更新"),
                (state.config, "球场校准已更改"),
                (state.pose_model, "姿态模型已更新"),
            )
            if path is not None and path.exists() and path.stat().st_mtime_ns > overlay_mtime
        ]
    return {
        "configured": pose_configured(state),
        "generated": output.exists(),
        "stale": bool(stale_sources),
        "current": bool(output.exists() and not stale_sources),
        "stale_reason": "；".join(stale_sources) or None,
        "path": str(output),
        "url": f"/media/pose-overlay/full?v={output.stat().st_mtime_ns}" if output.exists() else None,
    }
