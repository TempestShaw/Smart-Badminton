from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_rows


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _score_text(near_score: int, far_score: int) -> str:
    return f"近场  {near_score}  :  {far_score}  远场"


def create_score_ass(
    rallies_csv: Path,
    score_csv: Path,
    output: Path,
    *,
    result_seconds: float = 0.65,
) -> int:
    rallies = read_rows(rallies_csv)
    scores = {int(row["rally"]): row for row in read_rows(score_csv)}
    if not rallies or not scores:
        raise ValueError("No score data found")

    events: list[str] = []
    for index, rally in enumerate(rallies):
        rally_number = int(rally.get("rally", index + 1))
        score = scores.get(rally_number)
        if score is None:
            continue
        start = float(rally["start_seconds"])
        end = float(rally["end_seconds"])
        if end <= start:
            continue
        previous: dict[str, Any] | None = scores.get(rally_number - 1)
        reset_after_previous = previous is not None and str(previous.get("game_finished", "")) in {"near", "far"}
        near_before = 0 if previous is None or reset_after_previous else int(previous["near_score"])
        far_before = 0 if previous is None or reset_after_previous else int(previous["far_score"])
        near_after, far_after = int(score["near_score"]), int(score["far_score"])
        result_start = max(start, end - result_seconds)
        if result_start - start >= 0.10:
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(result_start)},Score,,0,0,0,,"
                f"{_score_text(near_before, far_before)}"
            )
        events.append(
            f"Dialogue: 0,{_ass_time(result_start)},{_ass_time(end)},Score,,0,0,0,,"
            f"{_score_text(near_after, far_after)}"
        )

    if not events:
        raise ValueError("No score events match the rally timeline")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Score,Microsoft YaHei,42,&H00FFFFFF,&H00FFFFFF,&H50000000,&H78000000,-1,0,0,0,100,100,0,0,3,3,0,7,42,0,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)
