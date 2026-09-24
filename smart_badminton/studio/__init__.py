"""Local Studio: a FastAPI boundary around the analysis pipeline for the React editor.

Modules, from request to disk:

- ``app``: HTTP routes, host/CORS policy and ``run_studio``;
- ``jobs``: background render and analysis workers;
- ``pipeline``: the automatic analysis hot path for one video;
- ``shuttle``: detector selection, detection and trajectory status;
- ``insights``: analytics, evidence lane and score views;
- ``calibration``: court calibration read/write;
- ``media``: FFmpeg runtime, preview proxy and pose overlays;
- ``project``: video discovery, timelines and library payloads;
- ``state``: shared ``StudioState``, payload cache and job status.
"""

from .app import create_studio_app, run_studio
from .state import StudioState

__all__ = ["StudioState", "create_studio_app", "run_studio"]
