from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rallies(path: Path) -> list[tuple[float, float]]:
    rallies = []
    for row in read_rows(path):
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        if end > start:
            rallies.append((start, end))
    return rallies


def time_in_rallies(time_seconds: float, rallies: list[tuple[float, float]]) -> bool:
    return any(start <= time_seconds <= end for start, end in rallies)


def resolve_ffmpeg(explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"FFmpeg not found: {explicit}")
        return explicit
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg)
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or imageio-ffmpeg.") from error
