from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptation import SegmentationAdapter
from .interval_quality import interval_quality_frame, interval_quality_probabilities
from .rally_evidence import RallyEvidence, build_rally_evidence


@dataclass
class Interval:
    start: float
    end: float
    confidence: float
    reason: str
    review: bool = False
    core_start: float | None = None
    core_end: float | None = None
    start_locked: bool = False
    end_locked: bool = False


def _rolling_max(values: pd.Series, samples: int) -> np.ndarray:
    return values.rolling(samples, center=True, min_periods=1).max().to_numpy(dtype=float)


def _gap_prior_value(frame: pd.DataFrame, name: str) -> float:
    if name not in frame or frame.empty:
        return 0.0
    value = pd.to_numeric(frame[name], errors="coerce").dropna()
    return max(0.0, float(value.iloc[0])) if len(value) else 0.0


def _audio_event_count(times: np.ndarray, scores: np.ndarray, start: float, end: float) -> int:
    mask = (times >= start) & (times <= end) & (scores >= 0.12)
    indices = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    events: list[float] = []
    for time_seconds in times[indices]:
        if not events or float(time_seconds) - events[-1] > 0.22:
            events.append(float(time_seconds))
    return len(events)


def _suppress_probable_handoffs(
    intervals: list[Interval],
    times: np.ndarray,
    probability: np.ndarray,
    audio: np.ndarray,
    evidence: RallyEvidence | None = None,
    orphan_action_probability_ceiling: float = 0.75,
    orphan_action_max_duration: float = 5.0,
    orphan_neighbor_gap: float = 3.0,
) -> list[Interval]:
    """Remove weak isolated actions while preserving trajectory-backed serves."""
    if len(intervals) < 2:
        return intervals
    statistics = []
    for interval in intervals:
        core_start = interval.core_start if interval.core_start is not None else interval.start
        core_end = interval.core_end if interval.core_end is not None else interval.end
        mask = (times >= core_start - 1e-6) & (times <= core_end + 1e-6)
        start_mask = mask & (times <= interval.start + 1.80)
        maximum_probability = float(np.max(probability[mask])) if np.any(mask) else 0.0
        formal_serve = bool(evidence is not None and np.any(start_mask & evidence.formal_serve))
        trajectory_serve = bool(evidence is not None and np.any(start_mask & evidence.trajectory_serve_start))
        trajectory_contact = bool(
            evidence is not None and np.any(start_mask & evidence.trajectory_contact_start)
        )
        statistics.append(
            {
                "maximum_probability": maximum_probability,
                "audio_events": _audio_event_count(times, audio, core_start, core_end),
                "duration": interval.end - interval.start,
                "formal_serve": formal_serve,
                "credible_start": formal_serve or (trajectory_serve and trajectory_contact),
                "complete_trajectory": interval.start_locked and interval.end_locked,
                "between_points": bool(evidence is not None and np.any(mask & evidence.between_points)),
            }
        )

    keep = [True] * len(intervals)
    for index, interval in enumerate(intervals):
        stats = statistics[index]
        if stats["credible_start"]:
            continue
        previous_gap = interval.start - intervals[index - 1].end if index else math.inf
        next_gap = intervals[index + 1].start - interval.end if index + 1 < len(intervals) else math.inf
        sandwiched_weak_action = (
            previous_gap <= 3.0
            and next_gap <= 2.05
            and stats["duration"] <= 8.0
            and stats["maximum_probability"] < 0.85
            and stats["audio_events"] <= 1
        )
        weak_before_strong_exchange = False
        if index + 1 < len(intervals):
            following = statistics[index + 1]
            weak_before_strong_exchange = (
                next_gap <= 2.2
                and stats["duration"] <= 8.0
                and stats["maximum_probability"] < 0.75
                and stats["audio_events"] <= 1
                and (following["maximum_probability"] >= 0.95 or following["audio_events"] >= 2)
            )
        opening_non_hit = (
            index == 0
            and interval.start <= 4.0
            and stats["duration"] <= 3.0
            and stats["maximum_probability"] < 0.75
            and stats["audio_events"] == 0
        )
        near_another_point = min(previous_gap, next_gap) <= orphan_neighbor_gap
        orphan_action = (
            near_another_point
            and stats["duration"] <= orphan_action_max_duration
            and stats["maximum_probability"] < orphan_action_probability_ceiling
            and stats["audio_events"] <= 1
        )
        if sandwiched_weak_action or weak_before_strong_exchange or opening_non_hit or orphan_action:
            keep[index] = False
    return [interval for index, interval in enumerate(intervals) if keep[index]]


def _refine_handoff_and_serve_phases(
    intervals: list[Interval],
    times: np.ndarray,
    probability: np.ndarray,
    evidence: RallyEvidence,
    preroll: float,
    postroll: float,
    split_handoff_before_serve: bool = False,
) -> list[Interval]:
    """Split broad intervals at trajectory-backed formal serves."""
    refined: list[Interval] = []
    for interval in intervals:
        serve_indices = np.flatnonzero(
            (evidence.formal_serve | evidence.trajectory_serve_start)
            & (times >= interval.start + 1.0)
            & (times <= interval.end - 0.8)
        )
        if not len(serve_indices):
            refined.append(interval)
            continue

        cursor = interval.start
        emitted_previous = False
        changed = False
        for serve_index in serve_indices:
            serve_time = float(times[serve_index])
            context_mask = (times >= cursor) & (times < serve_time)
            landing_indices = np.flatnonzero(context_mask & evidence.landing_candidate)
            handoff_indices = np.flatnonzero(context_mask & evidence.handoff_candidate)
            flight_end_indices = np.flatnonzero(context_mask & evidence.trajectory_flight_end)
            is_formal = bool(evidence.formal_serve[serve_index])
            marker_candidates = [
                *landing_indices.tolist(),
                *handoff_indices.tolist(),
                *(flight_end_indices.tolist() if is_formal else []),
            ]
            if not marker_candidates:
                continue
            marker_index = int(max(marker_candidates))
            marker_time = float(times[marker_index])
            if serve_time - marker_time < 0.45:
                continue

            next_start = max(cursor, serve_time - preroll)
            has_terminal_marker = bool(
                (len(landing_indices) and int(landing_indices[-1]) == marker_index)
                or (len(flight_end_indices) and int(flight_end_indices[-1]) == marker_index)
            )
            if has_terminal_marker:
                boundary_indices = np.flatnonzero((times >= marker_time) & (times <= next_start))
                if len(boundary_indices):
                    quiet_score = (
                        probability[boundary_indices]
                        + evidence.player_engagement[boundary_indices] * 0.20
                        + evidence.protected_flight[boundary_indices].astype(float) * 0.25
                    )
                    shared_cut = float(times[boundary_indices[int(np.argmin(quiet_score))]])
                else:
                    shared_cut = next_start
                previous_end = shared_cut
                if previous_end - cursor >= 1.5:
                    refined.append(
                        Interval(
                            cursor,
                            previous_end,
                            interval.confidence,
                            interval.reason + "; trajectory landing split before next serve",
                            interval.review,
                            interval.core_start,
                            marker_time,
                            end_locked=True,
                        )
                    )
                    emitted_previous = True
                    changed = True
                cursor = shared_cut
            else:
                if split_handoff_before_serve and marker_time - cursor > 1.5:
                    previous_end = next_start
                    refined.append(
                        Interval(
                            cursor,
                            previous_end,
                            interval.confidence,
                            interval.reason + "; handoff separated before next serve",
                            True,
                            interval.core_start,
                            next_start,
                        )
                    )
                    emitted_previous = True
                    cursor = next_start
                    changed = True
                elif (
                    not interval.start_locked
                    and marker_time - cursor <= 3.0
                    and cursor <= interval.start + 0.05
                    and next_start > cursor
                ):
                    cursor = next_start
                    interval.reason += "; handoff opening trimmed to formal serve"
                    interval.review = True
                    changed = True

        if not changed:
            refined.append(interval)
            continue
        if interval.end - cursor >= 1.5:
            refined.append(
                Interval(
                    cursor,
                    interval.end,
                    interval.confidence,
                    interval.reason,
                    interval.review,
                    max(cursor + preroll, interval.core_start or cursor),
                    interval.core_end,
                    start_locked=True,
                )
            )
        elif not emitted_previous:
            refined.append(interval)
    return refined


def segment_rallies(
    features_csv: Path,
    probabilities_csv: Path,
    output_csv: Path,
    start_threshold: float = 0.56,
    keep_threshold: float = 0.30,
    preroll: float = 0.35,
    postroll: float = 0.55,
    end_pending: float = 0.55,
    maximum_internal_gap: float = 1.2,
    suppress_handoffs: bool = True,
    shuttle_trajectory_csv: Path | None = None,
    adapter: SegmentationAdapter | Path | None = None,
) -> list[Interval]:
    if isinstance(adapter, Path):
        adapter = SegmentationAdapter.from_json(adapter)
    if adapter is not None:
        start_threshold = adapter.start_threshold
        keep_threshold = adapter.keep_threshold
        preroll = adapter.preroll
        postroll = adapter.postroll
        end_pending = adapter.end_pending
        maximum_internal_gap = adapter.maximum_internal_gap
    features = pd.read_csv(features_csv)
    probabilities = pd.read_csv(probabilities_csv)
    gap_soft_min = (
        adapter.soft_gap_seconds
        if adapter is not None and adapter.soft_gap_seconds > 0
        else _gap_prior_value(probabilities, "gap_soft_min_seconds")
    )
    gap_penalty = adapter.gap_penalty if adapter is not None else 0.20
    data = features.merge(probabilities[["time_seconds", "rally_probability"]], on="time_seconds", how="inner")
    times = data.time_seconds.to_numpy(dtype=float)
    if len(times) < 2:
        raise ValueError("Feature timeline is empty")
    fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3)
    probability = data.rally_probability.to_numpy(dtype=float)
    audio = data.audio_hit_score
    fused = build_rally_evidence(data, shuttle_trajectory_csv)
    evidence = fused.activity
    boundary_evidence = np.maximum(evidence, fused.protected_flight.astype(float) * 0.95)
    active_seed = probability >= start_threshold
    model_start = active_seed & (
        (evidence >= 0.22) | (_rolling_max(pd.Series(probability), round(1.0 * fps)) >= 0.78)
    )
    trajectory_start = fused.trajectory_serve_start | fused.trajectory_contact_start
    unqualified_start = (model_start | fused.trajectory_contact_start) & ~fused.between_points
    start_signal = unqualified_start | fused.formal_serve | fused.trajectory_serve_start
    intervals: list[Interval] = []
    active = False
    start = 0.0
    active_preroll = preroll
    last_keep = 0.0
    landing_locked = False
    landing_time = -math.inf
    trajectory_started = False
    collected_probabilities: list[float] = []
    for index, time_seconds in enumerate(times):
        model_keep = (
            probability[index] >= keep_threshold
            or (evidence[index] >= 0.75 and probability[index] >= 0.08)
        )
        keep = model_keep or fused.protected_flight[index]
        may_start = bool(start_signal[index])
        formal_start = bool(fused.formal_serve[index] or fused.trajectory_serve_start[index])
        if not active and may_start and intervals and not formal_start and gap_soft_min > 0:
            gap = float(time_seconds) - intervals[-1].end
            if gap < gap_soft_min:
                support_credit = max(
                    1.0 if fused.trajectory_contact_start[index] else 0.0,
                    0.65 if fused.players_ready[index] else 0.0,
                    min(0.55, float(evidence[index]) * 0.65),
                    0.45 if float(audio.iloc[index]) >= 0.12 else 0.0,
                )
                strength_required = start_threshold + gap_penalty * (1.0 - gap / gap_soft_min) * (
                    1.0 - support_credit
                )
                may_start = bool(
                    fused.trajectory_contact_start[index]
                    or probability[index] >= strength_required
                )
        if not active and may_start:
            active = True
            active_preroll = preroll
            if adapter is not None and adapter.uncertain_preroll > 0 and not formal_start:
                active_preroll = max(preroll, adapter.uncertain_preroll)
            start = max(float(times[0]), time_seconds - active_preroll)
            last_keep = time_seconds
            landing_locked = False
            landing_time = -math.inf
            trajectory_started = bool(trajectory_start[index])
            collected_probabilities = [float(probability[index])]
            continue
        if not active:
            continue
        collected_probabilities.append(float(probability[index]))
        if fused.landing_candidate[index]:
            landing_locked = True
            landing_time = float(time_seconds)
            last_keep = float(time_seconds)
            keep = True
        elif landing_locked:
            false_terminal = (
                float(time_seconds) - landing_time <= 0.35
                and probability[index] >= 0.85
                and fused.player_engagement[index] >= 0.40
            )
            if false_terminal:
                landing_locked = False
            else:
                keep = False
        if keep:
            last_keep = time_seconds
        if time_seconds - last_keep >= end_pending:
            end = min(float(times[-1]), last_keep + postroll)
            minimum_duration = 0.55 if trajectory_started and landing_locked else 1.5
            if end - start >= minimum_duration:
                confidence = float(
                    np.mean(sorted(collected_probabilities, reverse=True)[: max(1, len(collected_probabilities) // 3)])
                )
                recent_start = max(0, index - max(1, round((end_pending + postroll) * fps)))
                if np.any(fused.landing_candidate[recent_start : index + 1]):
                    end_reason = "trajectory landing and player stop confirmed"
                elif np.any(fused.trajectory_occluded[recent_start : index + 1]):
                    end_reason = "trajectory occlusion cleared; visual inactivity confirmed"
                else:
                    end_reason = "visual inactivity confirmed"
                intervals.append(
                    Interval(
                        start,
                        end,
                        confidence,
                        end_reason,
                        confidence < 0.62,
                        start + active_preroll,
                        last_keep,
                        start_locked=trajectory_started,
                        end_locked=landing_locked,
                    )
                )
            active = False
            landing_locked = False
            trajectory_started = False
    if active:
        intervals.append(
            Interval(
                start,
                min(float(times[-1]), last_keep + postroll),
                float(np.mean(collected_probabilities)),
                "end of video",
                True,
                start + active_preroll,
                last_keep,
                start_locked=trajectory_started,
            )
        )

    if suppress_handoffs:
        intervals = _suppress_probable_handoffs(
            intervals,
            times,
            probability,
            audio.to_numpy(dtype=float),
            fused,
            adapter.orphan_action_probability_ceiling if adapter is not None else 0.75,
            adapter.orphan_action_max_duration if adapter is not None else 5.0,
            adapter.orphan_neighbor_gap if adapter is not None else 3.0,
        )
        intervals = _refine_handoff_and_serve_phases(
            intervals,
            times,
            probability,
            fused,
            preroll,
            postroll,
            adapter.split_handoff_before_serve if adapter is not None else False,
        )

    merged: list[Interval] = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        gap = interval.start - previous.end
        gap_mask = (times >= previous.end) & (times <= interval.start)
        gap_evidence = float(np.max(boundary_evidence[gap_mask])) if np.any(gap_mask) else 0.0
        gap_probability = float(np.max(probability[gap_mask])) if np.any(gap_mask) else 0.0
        if (
            not previous.end_locked
            and not interval.start_locked
            and gap <= maximum_internal_gap
            and gap_evidence >= 0.70
            and gap_probability >= 0.08
        ):
            previous.end = interval.end
            previous.core_end = interval.core_end
            previous.end_locked = interval.end_locked
            previous.confidence = min(previous.confidence, interval.confidence)
            previous.reason = f"merged {gap:.2f}s quiet exchange"
            previous.review = previous.review or interval.review or gap > 1.5
        else:
            merged.append(interval)

    # Pre-roll and post-roll can overlap even when the two rally cores do not.
    # Replaying that source range makes players appear to jump backwards. Pick
    # one shared cut at the quietest sampled instant. If the entire overlap is
    # still strongly active, it was probably a false split and is merged.
    deduplicated: list[Interval] = []
    for interval in merged:
        if not deduplicated:
            deduplicated.append(interval)
            continue
        previous = deduplicated[-1]
        if previous.end <= interval.start:
            deduplicated.append(interval)
            continue
        overlap_mask = (times >= interval.start) & (times <= previous.end)
        overlap_indices = np.flatnonzero(overlap_mask)
        if not len(overlap_indices):
            cut = (previous.end + interval.start) / 2.0
        else:
            overlap_probability = probability[overlap_indices]
            overlap_evidence = boundary_evidence[overlap_indices]
            if (
                not previous.end_locked
                and not interval.start_locked
                and float(np.min(overlap_probability)) >= keep_threshold
                and float(np.min(overlap_evidence)) >= 0.55
            ):
                previous.end = max(previous.end, interval.end)
                previous.confidence = min(previous.confidence, interval.confidence)
                previous.reason = "merged active overlap"
                previous.review = True
                continue
            quiet_score = overlap_probability + overlap_evidence * 0.25
            cut = float(times[overlap_indices[int(np.argmin(quiet_score))]])
        previous.end = cut
        previous.reason += "; overlap deduplicated"
        interval.start = cut
        interval.reason += "; overlap deduplicated"
        deduplicated.append(interval)
    merged = deduplicated
    if adapter is not None and adapter.interval_quality:
        quality = interval_quality_frame(
            merged,
            times,
            probability,
            audio.to_numpy(dtype=float),
            fused,
        )
        quality_probability = interval_quality_probabilities(quality, adapter.interval_quality)
        threshold = float(adapter.interval_quality.get("threshold", 0.0))
        merged = [interval for interval, score in zip(merged, quality_probability) if score >= threshold]
    rows = []
    for number, interval in enumerate(merged, 1):
        rows.append(
            {
                "rally": number,
                "start_seconds": f"{interval.start:.3f}",
                "end_seconds": f"{interval.end:.3f}",
                "confidence": f"{interval.confidence:.3f}",
                "review_required": "yes" if interval.review else "no",
                "boundary_reason": interval.reason,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return merged
