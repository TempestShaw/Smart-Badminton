from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MATCH_DIRECTORY_PATTERN = re.compile(r"match[\w-]*", re.IGNORECASE)


def video_key(video: Path) -> str:
    match = re.search(r"_(\d{4})_D$", video.stem, re.IGNORECASE)
    return match.group(1) if match else video.stem


def find_project_root(library: Path, video: Path) -> Path:
    current = video.parent
    while current != library and library in current.parents:
        if MATCH_DIRECTORY_PATTERN.fullmatch(current.name):
            return current
        current = current.parent
    return library


@dataclass(frozen=True)
class AnalysisArtifacts:
    root: Path

    @property
    def features(self) -> Path:
        return self.root / "smart-features.csv"

    @property
    def vision_features(self) -> Path:
        return self.root / "vision-features.csv"

    @property
    def probabilities(self) -> Path:
        return self.root / "rally-probabilities.csv"

    @property
    def automatic_rallies(self) -> Path:
        return self.root / "rallies-auto.csv"

    @property
    def audio_events(self) -> Path:
        return self.root / "audio-events.csv"

    @property
    def shuttle_raw(self) -> Path:
        return self.root / "shuttle-raw.csv"

    @property
    def shuttle_yolo_raw(self) -> Path:
        return self.root / "shuttle-yolo-raw.csv"

    @property
    def shuttle_tracknet_raw(self) -> Path:
        return self.root / "shuttle-tracknet-raw.csv"

    @property
    def shuttle_track(self) -> Path:
        return self.root / "shuttle-track.csv"

    @property
    def shuttle_detection_metadata(self) -> Path:
        return self.root / "shuttle-detection.json"

    @property
    def shuttle_annotations(self) -> Path:
        return self.root / "Shuttle_Annotations" / "user-shuttle-points.csv"

    @property
    def score_events(self) -> Path:
        return self.root / "rally-events.csv"

    @property
    def score_state(self) -> Path:
        return self.root / "score-state.csv"

    @property
    def score_summary(self) -> Path:
        return self.root / "score-summary.json"

    @property
    def score_labeling(self) -> Path:
        return self.root / "Score_Labeling"


@dataclass(frozen=True)
class ProjectLayout:
    library: Path
    video: Path
    root: Path
    is_match_project: bool
    analysis: AnalysisArtifacts

    @classmethod
    def for_video(cls, library: Path, video: Path) -> ProjectLayout:
        root = find_project_root(library, video)
        is_match_project = root != library
        analysis_root = root / "Analysis" if is_match_project else library / "Analysis" / "Auto" / video.stem
        return cls(
            library=library,
            video=video,
            root=root,
            is_match_project=is_match_project,
            analysis=AnalysisArtifacts(analysis_root),
        )

    @property
    def metadata(self) -> Path:
        return self.root / "Metadata" if self.is_match_project else self.library / "Metadata"

    @property
    def working_timeline(self) -> Path:
        if self.is_match_project:
            return self.metadata / "rallies-studio-review.csv"
        return self.library / "Analysis" / "Studio" / f"{self.video.stem}.csv"

    @property
    def latest_auto_cut(self) -> Path:
        name = self.root.name if self.is_match_project else video_key(self.video)
        return self.library / "Latest_Auto_Cut" / f"{name}-rallies.csv"

    @property
    def ground_truth(self) -> Path:
        if self.is_match_project:
            return self.metadata / "rallies-ground-truth.csv"
        return self.library / "Standard_Answers" / f"{video_key(self.video)}_rallies_ground_truth.csv"

    @property
    def default_output(self) -> Path:
        if self.is_match_project:
            return self.root / "Edited" / f"{self.root.name}_final_1080p60.mp4"
        return self.library / "Archive" / "Studio_Exports" / f"{self.video.stem}_edited_1080p60.mp4"

    @property
    def score_corrections(self) -> Path:
        if self.is_match_project:
            return self.metadata / "score-corrections.csv"
        return self.metadata / f"{self.video.stem}-score-corrections.csv"

    @property
    def segmentation_adapter(self) -> Path:
        if self.is_match_project:
            return self.metadata / "segmentation-adapter.json"
        return self.metadata / f"{self.video.stem}-segmentation-adapter.json"

    @property
    def proxy_candidates(self) -> tuple[Path, ...]:
        local_candidates = (self.video.with_suffix(".LRF"), self.video.with_suffix(".lrf"))
        if self.is_match_project:
            managed_candidates = (
                self.root / "Proxy" / f"{self.video.stem}_proxy_720p.mp4",
                self.root / "Proxy" / f"{self.video.stem}.mp4",
            )
        else:
            managed_candidates = (self.analysis.root / "Proxy" / f"{self.video.stem}_proxy_720p.mp4",)
        return local_candidates + managed_candidates

    def existing_proxy(self) -> Path | None:
        return next((candidate for candidate in self.proxy_candidates if candidate.exists()), None)


@dataclass(frozen=True)
class LibraryLayout:
    root: Path

    @property
    def runtime(self) -> Path:
        return self.root / ".smart-badminton"

    @property
    def models(self) -> Path:
        return self.runtime / "models"

    @property
    def jobs(self) -> Path:
        return self.runtime / "jobs"

    @property
    def analysis_status(self) -> Path:
        return self.jobs / "analysis-status.json"
