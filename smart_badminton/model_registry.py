from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    role: str
    engine: str
    source: str
    path: Path | None = None
    model_id: str | None = None
    helper: bool = False

    @property
    def configured(self) -> bool:
        return bool((self.path and self.path.exists()) or self.model_id)

    def public_payload(self) -> dict[str, str | bool | None]:
        return {
            "role": self.role,
            "engine": self.engine,
            "source": self.source,
            "name": self.path.name if self.path else self.model_id,
            "configured": self.configured,
            "helper": self.helper,
        }


@dataclass(frozen=True)
class ModelRegistry:
    rally_state: Path | None = None
    pose: Path | None = None
    shuttle_yolo: Path | None = None
    tracknet: Path | None = None
    inpaint: Path | None = None
    score_evidence: Path | None = None
    vision_models: tuple[str, ...] = ()

    def entries(self) -> tuple[ModelEntry, ...]:
        local = (
            ModelEntry("rally_segmentation", "scikit-learn", "local", self.rally_state),
            ModelEntry("player_pose", "Ultralytics YOLO", "local", self.pose),
            ModelEntry("shuttle_detection", "Ultralytics YOLO", "local", self.shuttle_yolo),
            ModelEntry("shuttle_tracking", "TrackNetV3", "local", self.tracknet),
            ModelEntry("trajectory_repair", "InpaintNet", "local", self.inpaint, helper=True),
            ModelEntry("score_evidence", "rules", "library", self.score_evidence),
        )
        remote = tuple(
            ModelEntry("score_labeling", "vision language model", "remote", model_id=model_id)
            for model_id in self.vision_models
        )
        return local + remote

    def available_shuttle_modes(self) -> list[str]:
        yolo = bool(self.shuttle_yolo and self.shuttle_yolo.exists())
        tracknet = bool(self.tracknet and self.tracknet.exists())
        modes: list[str] = []
        if yolo:
            modes.append("yolo")
        if tracknet:
            modes.append("tracknet")
        if yolo and tracknet:
            modes.append("hybrid")
        return modes

    def public_payload(self) -> dict[str, object]:
        entries = [entry.public_payload() for entry in self.entries()]
        return {
            "entries": entries,
            "configured_count": sum(bool(entry["configured"]) for entry in entries),
            "local_count": sum(entry["source"] in {"local", "library"} and bool(entry["configured"]) for entry in entries),
            "remote_count": sum(entry["source"] == "remote" and bool(entry["configured"]) for entry in entries),
        }
