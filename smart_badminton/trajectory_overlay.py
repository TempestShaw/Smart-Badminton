from __future__ import annotations

import math
from bisect import bisect_left
from itertools import pairwise
from pathlib import Path
from typing import Any

from .io import read_rows

VISIBLE_TRAJECTORY_STATUSES = {"tracked", "recovered", "manual"}


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _segment_polygon(start: tuple[float, float], end: tuple[float, float], width: float) -> str | None:
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length < 0.5:
        return None
    offset_x = -delta_y * width / (2 * length)
    offset_y = delta_x * width / (2 * length)
    points = [
        (start[0] + offset_x, start[1] + offset_y),
        (end[0] + offset_x, end[1] + offset_y),
        (end[0] - offset_x, end[1] - offset_y),
        (start[0] - offset_x, start[1] - offset_y),
    ]
    first, *remaining = points
    coordinates = " ".join(f"{round(x)} {round(y)}" for x, y in remaining)
    return f"m {round(first[0])} {round(first[1])} l {coordinates}"


def _read_trajectory(path: Path) -> tuple[list[dict[str, Any]], int, int, float]:
    points: list[dict[str, Any]] = []
    widths: list[int] = []
    heights: list[int] = []
    frame_rates: list[float] = []
    previous: dict[str, Any] | None = None
    for row in read_rows(path):
        if str(row.get("status", "")).strip() not in VISIBLE_TRAJECTORY_STATUSES:
            continue
        try:
            time_seconds = float(row["time_seconds"])
            x = float(row["center_x"])
            y = float(row["center_y"])
            width = int(float(row["source_width"]))
            height = int(float(row["source_height"]))
            frame = int(float(row.get("frame", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (time_seconds, x, y)) or width <= 0 or height <= 0:
            continue
        point = {
            "time": time_seconds,
            "frame": frame,
            "x": x / width,
            "y": y / height,
            "flight_id": str(row.get("flight_id", "")).strip(),
        }
        if previous is not None:
            time_delta = time_seconds - float(previous["time"])
            frame_delta = frame - int(previous["frame"])
            if time_delta > 0 and frame_delta > 0:
                frame_rates.append(frame_delta / time_delta)
        previous = point
        widths.append(width)
        heights.append(height)
        points.append(point)
    points.sort(key=lambda point: (float(point["time"]), int(point["frame"])))
    if not points:
        raise ValueError("No usable shuttle trajectory points found")
    width = max(widths)
    height = max(heights)
    frame_rate = sorted(frame_rates)[len(frame_rates) // 2] if frame_rates else 30.0
    return points, width, height, min(240.0, max(1.0, frame_rate))


def create_trajectory_ass(
    trajectory_csv: Path,
    output: Path,
    *,
    tail_seconds: float = 0.85,
) -> int:
    """Write a source-time ASS overlay containing only the clean trajectory trail."""
    points, width, height, frame_rate = _read_trajectory(trajectory_csv)
    maximum_gap = max(0.20, 5 / frame_rate)
    line_width = max(3.0, height * 0.0045)
    events: list[str] = []
    point_times = [float(point["time"]) for point in points]

    for index, current in enumerate(points):
        current_time = float(current["time"])
        flight_id = str(current["flight_id"])
        window_start = bisect_left(point_times, current_time - tail_seconds, hi=index + 1)
        recent = [
            point
            for point in points[window_start : index + 1]
            if str(point["flight_id"]) == flight_id
        ]
        polygons: list[str] = []
        for previous, point in pairwise(recent):
            time_delta = float(point["time"]) - float(previous["time"])
            if time_delta <= 0 or time_delta > maximum_gap:
                continue
            normalized_distance = math.hypot(
                float(point["x"]) - float(previous["x"]),
                float(point["y"]) - float(previous["y"]),
            )
            if normalized_distance / time_delta > 2.6:
                continue
            polygon = _segment_polygon(
                (float(previous["x"]) * width, float(previous["y"]) * height),
                (float(point["x"]) * width, float(point["y"]) * height),
                line_width,
            )
            if polygon:
                polygons.append(polygon)
        if not polygons:
            continue
        next_time = float(points[index + 1]["time"]) if index + 1 < len(points) else current_time + tail_seconds
        end_time = min(current_time + tail_seconds, max(current_time + 0.01, next_time))
        drawing = " ".join(polygons)
        text = r"{\an7\pos(0,0)\p1\bord0\shad0\1c&H00629DFF&\1a&H20&}" + drawing
        events.append(
            f"Dialogue: 0,{_ass_time(current_time)},{_ass_time(end_time)},Trail,,0,0,0,,{text}"
        )

    if not events:
        raise ValueError("No drawable shuttle trajectory segments found")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Trail,Arial,20,&H00629DFF,&H00629DFF,&H00629DFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)
