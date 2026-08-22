from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path

from .encoding import h264_encoding_arguments, run_ffmpeg_with_encoder_fallback
from .io import read_rows, resolve_ffmpeg
from .score_overlay import create_score_ass
from .trajectory_overlay import create_trajectory_ass


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix().replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'")


def render_rallies(
    video: Path,
    rallies_csv: Path,
    output: Path,
    ffmpeg: Path | None = None,
    encoder: str = "auto",
    quality: int = 21,
    output_fps: float | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    trajectory_csv: Path | None = None,
    score_csv: Path | None = None,
    winner_filter: str = "all",
    include_score: bool = False,
) -> str:
    rows = read_rows(rallies_csv)
    if not rows:
        raise ValueError("No rally segments found")
    if winner_filter not in {"all", "near", "far"}:
        raise ValueError("Winner filter must be all, near, or far")
    if winner_filter != "all":
        if score_csv is None:
            raise ValueError("Score data is required to filter rallies by winner")
        winners = {int(row["rally"]): str(row.get("winner", "unknown")) for row in read_rows(score_csv)}
        rows = [
            row
            for index, row in enumerate(rows)
            if winners.get(int(row.get("rally", index + 1))) == winner_filter
        ]
        if not rows:
            raise ValueError(f"No {winner_filter}-player winning rallies found")
    if include_score and score_csv is None:
        raise ValueError("Score data is required for the score overlay")
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_overlays = trajectory_csv is not None or include_score
    temporary_context = (
        tempfile.TemporaryDirectory(prefix=".render-overlays-", dir=output.parent)
        if needs_overlays
        else nullcontext(None)
    )
    with temporary_context as temporary_name:
        overlay_filters: list[str] = []
        if trajectory_csv is not None:
            trajectory_ass = Path(str(temporary_name)) / "trajectory.ass"
            create_trajectory_ass(trajectory_csv, trajectory_ass)
            overlay_filters.append(f"ass=filename='{_ffmpeg_filter_path(trajectory_ass)}'")
        if include_score and score_csv is not None:
            score_ass = Path(str(temporary_name)) / "score.ass"
            create_score_ass(rallies_csv, score_csv, score_ass)
            overlay_filters.append(f"ass=filename='{_ffmpeg_filter_path(score_ass)}'")

        filters, inputs = [], []
        if overlay_filters:
            overlay_chain = ",".join(overlay_filters)
            if len(rows) == 1:
                filters.append(f"[0:v:0]{overlay_chain}[marked0]")
            else:
                marked_outputs = "".join(f"[marked{index}]" for index in range(len(rows)))
                filters.append(f"[0:v:0]{overlay_chain},split={len(rows)}{marked_outputs}")
        for index, row in enumerate(rows):
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            if end <= start:
                raise ValueError(f"Invalid rally {row.get('rally', index + 1)}")
            video_input = f"[marked{index}]" if overlay_filters else "[0:v:0]"
            filters.append(f"{video_input}trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]")
            filters.append(f"[0:a:0]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]")
            inputs.append(f"[v{index}][a{index}]")
        video_filters = []
        if output_width is not None or output_height is not None:
            if not output_width or not output_height:
                raise ValueError("Output width and height must be provided together")
            video_filters.append(f"scale={output_width}:{output_height}:flags=lanczos")
            video_filters.append("setsar=1")
        if output_fps is not None:
            if output_fps <= 0:
                raise ValueError("Output FPS must be positive")
            video_filters.append(f"fps={output_fps:.6f}")
        if video_filters:
            filters.append("".join(inputs) + f"concat=n={len(rows)}:v=1:a=1[vcat][aout]")
            filters.append(f"[vcat]{','.join(video_filters)}[vout]")
        else:
            filters.append("".join(inputs) + f"concat=n={len(rows)}:v=1:a=1[vout][aout]")
        ffmpeg_path = resolve_ffmpeg(ffmpeg)

        def command_for(video_encoder: str) -> list[str]:
            command = [
                str(ffmpeg_path),
                "-y",
                "-hide_banner",
                "-i",
                str(video),
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                *h264_encoding_arguments(video_encoder, quality, "final"),
            ]
            return command + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]

        return run_ffmpeg_with_encoder_fallback(ffmpeg_path, encoder, command_for, output)
