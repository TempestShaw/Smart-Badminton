from __future__ import annotations

from pathlib import Path

from .encoding import h264_encoding_arguments, run_ffmpeg_with_encoder_fallback
from .io import read_rows, resolve_ffmpeg


def render_boundary_reviews(
    video: Path,
    rallies_csv: Path,
    output_directory: Path,
    ffmpeg: Path | None = None,
    padding: float = 3.0,
    encoder: str = "auto",
) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in read_rows(rallies_csv):
        if row.get("review_required", "no").lower() not in ("yes", "true", "1"):
            continue
        number = int(row["rally"])
        for label, value in (("start", float(row["start_seconds"])), ("end", float(row["end_seconds"]))):
            start = max(0.0, value - padding)
            duration = padding * 2.0
            target = output_directory / f"rally_{number:03d}_{label}_{value:.2f}.mp4"
            ffmpeg_path = resolve_ffmpeg(ffmpeg)

            def command_for(
                video_encoder: str,
                ffmpeg_path: Path = ffmpeg_path,
                start: float = start,
                duration: float = duration,
                target: Path = target,
            ) -> list[str]:
                return [
                    str(ffmpeg_path),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    str(video),
                    "-t",
                    f"{duration:.3f}",
                    "-vf",
                    "scale=-2:720",
                    *h264_encoding_arguments(video_encoder, 25, "preview"),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "96k",
                    str(target),
                ]

            run_ffmpeg_with_encoder_fallback(ffmpeg_path, encoder, command_for, target)
            count += 1
    return count
