from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rally_evidence import RallyEvidence
from .start_quality import selected_start_candidates, start_candidate_frame


def _quiet_runs(indices: np.ndarray, times: np.ndarray, minimum_seconds: float) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    return [
        (int(run[0]), int(run[-1]))
        for run in runs
        if len(run) and float(times[run[-1]] - times[run[0]]) >= minimum_seconds
    ]


def split_at_local_starts(
    intervals: list[Any],
    data: pd.DataFrame,
    times: np.ndarray,
    probability: np.ndarray,
    evidence: RallyEvidence,
    model: dict | None,
) -> list[Any]:
    if not model or not intervals:
        return intervals
    start_frame = start_candidate_frame(data, evidence)
    start_candidates = selected_start_candidates(start_frame, model)
    if not start_candidates:
        return intervals

    quiet_ceiling = float(model.get("quiet_probability_ceiling", 0.20))
    quiet_minimum = float(model.get("quiet_min_seconds", 1.0))
    quiet_search = float(model.get("quiet_search_seconds", 8.0))
    terminal_tail = float(model.get("terminal_tail_seconds", 0.70))
    lead_safety = float(model.get("lead_safety_seconds", 0.0))
    coalesced: list[Any] = []
    for interval in intervals:
        if not coalesced:
            coalesced.append(interval)
            continue
        previous = coalesced[-1]
        qualified_boundary = any(
            abs((value - candidate_lead - lead_safety) - interval.start) <= 1.5
            for value, candidate_lead in start_candidates
        )
        if interval.start - previous.end <= 0.05 and not qualified_boundary:
            previous.end = interval.end
            previous.core_end = interval.core_end
            previous.end_locked = interval.end_locked
            previous.confidence = min(previous.confidence, interval.confidence)
            previous.reason += "; local boundary coalesced"
            previous.review = previous.review or interval.review
            continue
        coalesced.append(interval)

    refined: list[Any] = []
    for interval in coalesced:
        candidates = [
            (value, candidate_lead)
            for value, candidate_lead in start_candidates
            if value - candidate_lead - lead_safety >= interval.start + 0.4
            and value <= interval.end - 0.8
        ]
        if not candidates:
            refined.append(interval)
            continue

        interval_type = type(interval)
        cursor = interval.start
        core_start = interval.core_start
        emitted = False
        anchored = any(
            abs((value - candidate_lead - lead_safety) - cursor) <= 2.5
            for value, candidate_lead in start_candidates
        )
        for start_time, candidate_lead in candidates:
            raw_next_start = start_time - candidate_lead - lead_safety
            if not anchored or raw_next_start - cursor <= 2.5:
                next_start = max(cursor, raw_next_start)
                if refined:
                    terminal_mask = (
                        (times >= max(refined[-1].start, refined[-1].end - 1.0))
                        & (times <= refined[-1].end + 1.0)
                        & (times < next_start)
                        & (evidence.trajectory_flight_end | evidence.landing_candidate)
                    )
                    terminal_indices = np.flatnonzero(terminal_mask)
                    if len(terminal_indices):
                        refined[-1].end = max(
                            refined[-1].end,
                            min(next_start, float(times[terminal_indices[-1]]) + terminal_tail),
                        )
                cursor = next_start
                core_start = max(next_start, interval.core_start or next_start)
                anchored = True
                emitted = True
                continue
            next_start = max(cursor + 1.2, raw_next_start)
            search_mask = (
                (times >= max(cursor + 1.0, next_start - quiet_search))
                & (times <= next_start)
                & (probability <= quiet_ceiling)
                & (evidence.player_engagement <= 0.28)
                & ~evidence.protected_flight
            )
            runs = _quiet_runs(np.flatnonzero(search_mask), times, quiet_minimum)
            previous_end = float(times[runs[-1][0]]) if runs else next_start
            if previous_end - cursor < 1.2 or interval.end - next_start < 0.8:
                continue
            refined.append(
                interval_type(
                    cursor,
                    previous_end,
                    interval.confidence,
                    interval.reason + "; local serve split",
                    interval.review,
                    core_start,
                    min(previous_end, interval.core_end or previous_end),
                    interval.start_locked if not emitted else True,
                    False,
                )
            )
            cursor = next_start
            core_start = max(next_start, interval.core_start or next_start)
            emitted = True
        if emitted and interval.end - cursor >= 0.8:
            refined.append(
                interval_type(
                    cursor,
                    interval.end,
                    interval.confidence,
                    interval.reason + "; local serve split",
                    interval.review,
                    core_start,
                    interval.core_end,
                    True,
                    interval.end_locked,
                )
            )
        elif not emitted:
            refined.append(interval)

    gap_ceiling = float(model.get("quiet_gap_probability_ceiling", 0.0))
    if gap_ceiling <= 0:
        return refined
    gap_window = float(model.get("quiet_gap_window_seconds", 2.5))
    gap_minimum = float(model.get("quiet_gap_min_seconds", 0.8))
    fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3)
    rolling_probability = (
        pd.Series(probability)
        .rolling(max(2, round(gap_window * fps)), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    separated: list[Any] = []
    for interval in refined:
        valley_mask = (
            (times >= interval.start + 1.5)
            & (times <= interval.end - 1.0)
            & (rolling_probability <= gap_ceiling)
        )
        runs = _quiet_runs(np.flatnonzero(valley_mask), times, gap_minimum)
        interval_type = type(interval)
        cursor = interval.start
        emitted = False
        for run_start, run_end in runs:
            previous_support = (times >= max(cursor, times[run_start] - 2.5)) & (
                times < times[run_start]
            )
            next_support = (times > times[run_end]) & (
                times <= min(interval.end, times[run_end] + 2.5)
            )
            if (
                not np.any(previous_support)
                or not np.any(next_support)
                or float(np.max(probability[previous_support])) < 0.55
                or float(np.max(probability[next_support])) < 0.55
            ):
                continue
            previous_end = float(times[run_start])
            next_start = float(times[run_end])
            if previous_end - cursor < 1.2 or interval.end - next_start < 0.8:
                continue
            separated.append(
                interval_type(
                    cursor,
                    previous_end,
                    interval.confidence,
                    interval.reason + "; local quiet-gap split",
                    interval.review,
                    interval.core_start if not emitted else cursor,
                    previous_end,
                    interval.start_locked if not emitted else True,
                    False,
                )
            )
            cursor = next_start
            emitted = True
        if emitted:
            separated.append(
                interval_type(
                    cursor,
                    interval.end,
                    interval.confidence,
                    interval.reason + "; local quiet-gap split",
                    interval.review,
                    cursor,
                    interval.core_end,
                    True,
                    interval.end_locked,
                )
            )
        else:
            separated.append(interval)
    return separated
