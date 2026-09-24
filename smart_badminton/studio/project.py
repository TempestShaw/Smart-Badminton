from __future__ import annotations

import csv
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import cv2

from ..io import read_rows
from ..project_layout import ProjectLayout
from .state import StudioState

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
EXCLUDED_VIDEO_DIRECTORIES = {
    ".git",
    ".tools",
    "analysis",
    "archive",
    "documentation",
    "edited",
    "metadata",
    "models",
    "proxy",
    "standard_answers",
    "thirdparty",
}
EXCLUDED_VIDEO_MARKERS = ("_edited", "_final", "_proxy", "_review")
TIMELINE_FIELDS = ["rally", "start_seconds", "end_seconds", "confidence", "review_required", "boundary_reason"]


def discover_videos(library: Path) -> list[Path]:
    videos = []
    for path in library.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(library)
        if any(part.lower() in EXCLUDED_VIDEO_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if any(marker in path.stem.lower() for marker in EXCLUDED_VIDEO_MARKERS):
            continue
        videos.append(path)
    return sorted(videos, key=lambda path: path.relative_to(library).as_posix().lower())


def video_id(library: Path, video: Path) -> str:
    return video.relative_to(library).as_posix()


def timeline_has_segments(path: Path) -> bool:
    return path.exists() and bool(read_rows(path))


def project_has_completed_marker(library: Path, video: Path) -> bool:
    layout = ProjectLayout.for_video(library, video)
    if not layout.is_match_project:
        return layout.ground_truth.exists()
    edited = layout.root / "Edited"
    return (layout.root / "Metadata" / "rallies.csv").exists() or (
        edited.exists() and any(path.suffix.lower() in VIDEO_EXTENSIONS for path in edited.iterdir())
    )


def ensure_working_timeline(library: Path, video: Path) -> Path:
    layout = ProjectLayout.for_video(library, video)
    timeline = layout.working_timeline
    if timeline.exists():
        return timeline
    timeline.parent.mkdir(parents=True, exist_ok=True)
    seed = next((path for path in (layout.latest_auto_cut, layout.ground_truth) if path.exists()), None)
    if seed is None:
        save_segments(timeline, [])
    else:
        shutil.copy2(seed, timeline)
        timeline.chmod(timeline.stat().st_mode | 0o200)
    return timeline


def public_segments(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row.get("rally", index)),
            "start": float(row["start_seconds"]),
            "end": float(row["end_seconds"]),
            "confidence": row.get("confidence", ""),
            "review_required": row.get("review_required", "no"),
            "boundary_reason": row.get("boundary_reason", row.get("notes", "")),
        }
        for index, row in enumerate(read_rows(path), 1)
    ]


def timeline_rows(segments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    ordered = sorted(segments, key=lambda item: float(item["start"]))
    previous_end = 0.0
    for index, item in enumerate(ordered, 1):
        start, end = float(item["start"]), float(item["end"])
        if not (start >= 0 and end - start >= 0.05 and end <= duration + 0.05):  # NaN fails too
            raise ValueError(f"Rally {index} ({start:.3f}–{end:.3f}s) must lie inside the video and last at least 0.05s")
        if start < previous_end - 0.001:
            raise ValueError(f"Rally {index} overlaps the rally before it")
        previous_end = end
    return [
        {
            "rally": index,
            "start_seconds": f"{float(item['start']):.3f}",
            "end_seconds": f"{min(float(item['end']), duration):.3f}",
            "confidence": str(item.get("confidence", "manual")),
            "review_required": str(item.get("review_required", "no")),
            "boundary_reason": str(item.get("boundary_reason", "edited in studio")),
        }
        for index, item in enumerate(ordered, 1)
    ]


def save_segments(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Atomically replace a timeline, keeping the previous version as ``.bak``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_name, path)
    return backup


@lru_cache(maxsize=32)
def _read_video_metadata(video_path: str, mtime_ns: int, size: int) -> tuple[float | int | str, ...]:
    del mtime_ns, size
    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or not fps > 0:
        capture.release()
        raise ValueError(f"Could not read video: {Path(video_path).name}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return Path(video_path).name, frame_count / fps, fps, frame_count, width, height


def video_metadata(video: Path) -> dict[str, float | int | str]:
    resolved = video.resolve()
    stat = resolved.stat()
    name, duration, fps, frame_count, width, height = _read_video_metadata(
        str(resolved), stat.st_mtime_ns, stat.st_size
    )
    return {"name": name, "duration": duration, "fps": fps, "frame_count": frame_count, "width": width, "height": height}


def activate_video(state: StudioState, video: Path) -> None:
    library = state.library_root
    timeline = ensure_working_timeline(library, video)
    layout = ProjectLayout.for_video(library, video)
    with state.project_lock:
        state.video = video
        state.proxy = layout.existing_proxy()
        state.rallies = timeline
        state.output = (
            state.output_directory / layout.default_output.name if state.output_directory else layout.default_output
        )
        state.render_status = {"state": "idle"}


def library_payload(state: StudioState) -> dict[str, Any]:
    library = state.library_root
    active = state.video.resolve()
    videos = []
    for video in discover_videos(library):
        layout = ProjectLayout.for_video(library, video)
        videos.append(
            {
                "id": video_id(library, video),
                "relative_path": video_id(library, video),
                "project_name": layout.root.name,
                "name": video.name,
                "size_bytes": video.stat().st_size,
                "proxy_available": layout.existing_proxy() is not None,
                "timeline_available": timeline_has_segments(layout.working_timeline),
                "latest_auto_cut_available": timeline_has_segments(layout.latest_auto_cut),
                "ground_truth_available": layout.ground_truth.exists(),
                "completed_marker": project_has_completed_marker(library, video),
                "active": video.resolve() == active,
            }
        )
    return {"path": str(library), "videos": videos}


def _filesystem_roots() -> list[Path]:
    if os.name == "nt":
        return [Path(f"{letter}:\\") for letter in ascii_uppercase if Path(f"{letter}:\\").is_dir()]
    return list(dict.fromkeys([Path("/"), Path.home().resolve()]))


def directory_browser_payload(requested: str | None, fallback: Path) -> dict[str, Any]:
    current = (Path(requested) if requested else fallback).expanduser()
    while not current.is_dir() and current != current.parent:
        current = current.parent
    current = current.resolve()
    children = sorted(current.iterdir(), key=lambda path: path.name.casefold())
    return {
        "path": str(current),
        "parent": str(current.parent) if current.parent != current else None,
        "home": str(Path.home().resolve()),
        "roots": [{"name": path.anchor or str(path), "path": str(path)} for path in _filesystem_roots()],
        "directories": [{"name": child.name, "path": str(child.resolve())} for child in children if child.is_dir()],
    }


def output_payload(state: StudioState) -> dict[str, Any]:
    return {
        "directory": str(state.output.parent),
        "filename": state.output.name,
        "path": str(state.output),
        "directory_exists": state.output.parent.is_dir(),
        "file_exists": state.output.exists(),
    }
