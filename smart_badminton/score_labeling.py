from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .io import load_rallies

LABEL_SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FRAME_OFFSETS = (-1.60, -1.10, -0.75, -0.45, -0.25, -0.10, 0.08, 0.35)
SIDES = {"near", "far", "unknown"}
TERMINAL_EVENTS = {"landing_in", "landing_out", "net", "unreturned", "unknown"}
POST_RALLY_EVENTS = {"handoff", "none", "unknown"}
WINNERS = {"near", "far", "no_point", "unknown"}


def score_labeling_directory(project_root: Path) -> Path:
    return project_root / "Analysis" / "Score_Labeling"


def score_labeling_config() -> dict[str, Any]:
    api_key = os.environ.get("SMART_BADMINTON_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("SMART_BADMINTON_VLM_MODEL", "").strip()
    models = [value.strip() for value in os.environ.get("SMART_BADMINTON_VLM_MODELS", "").split(",") if value.strip()]
    if not models and model:
        models = [model]
    endpoint = os.environ.get("SMART_BADMINTON_VLM_ENDPOINT", DEFAULT_ENDPOINT).strip()
    return {
        "configured": bool(api_key and models and endpoint),
        "model": ", ".join(models) or None,
        "models": models,
        "endpoint": endpoint,
        "api_key": api_key,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _resize_frame(frame: Any, maximum_width: int = 960) -> Any:
    height, width = frame.shape[:2]
    if width <= maximum_width:
        return frame
    scale = maximum_width / width
    return cv2.resize(frame, (maximum_width, round(height * scale)), interpolation=cv2.INTER_AREA)


def prepare_score_evidence(
    video: Path,
    rallies_csv: Path,
    output_directory: Path,
    rally_ids: list[int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    rallies = load_rallies(rallies_csv)
    selected = set(rally_ids or range(1, len(rallies) + 1))
    requested = [(index, rally) for index, rally in enumerate(rallies, 1) if index in selected]
    if not requested:
        raise ValueError("No rallies were selected for score labeling")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / max(fps, 1.0)
    entries: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        for completed, (rally_id, rally) in enumerate(requested, 1):
            rally_directory = output_directory / f"R{rally_id:02d}"
            rally_directory.mkdir(parents=True, exist_ok=True)
            start, end = map(float, rally)
            frame_rows = []
            for frame_index, offset in enumerate(FRAME_OFFSETS, 1):
                timestamp = max(start, min(duration, end + offset))
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = _resize_frame(frame)
                label = f"R{rally_id:02d}  {timestamp:.3f}s  {offset:+.2f}s"
                cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 360), 34), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
                filename = f"{frame_index:02d}_{timestamp:.3f}.jpg"
                path = rally_directory / filename
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise OSError(f"Could not write evidence frame: {path}")
                frame_rows.append({"time": round(timestamp, 3), "offset": offset, "path": str(path.resolve())})
            entries.append(
                {
                    "rally": rally_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "frames": frame_rows,
                }
            )
            if progress_callback:
                progress_callback(completed, len(requested))
    finally:
        capture.release()

    manifest = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "video": str(video.resolve()),
        "timeline": str(rallies_csv.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rallies": entries,
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prompt(rally: dict[str, Any]) -> list[dict[str, Any]]:
    text = (
        "These are chronological frames near the end of one badminton rally. "
        "Near player is closest to the camera; far player is across the net. "
        "Classify only visible evidence. A shuttle hit into the net is net; outside court is landing_out; "
        "inside either court is landing_in; a legal shot the opponent fails to return is unreturned. "
        "A gentle shot after the point that merely returns the shuttle to the next server is handoff. "
        "Return one JSON object with keys: terminal_event, last_hitter, landing_side, winner, "
        "post_rally_event, confidence, evidence_frames, reason. "
        "Use near/far/unknown for sides, landing_in/landing_out/net/unreturned/unknown for terminal_event, "
        "near/far/no_point/unknown for winner, handoff/none/unknown for post_rally_event, "
        "confidence from 0 to 1, and evidence_frames as frame numbers. Keep reason under 20 words."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, frame in enumerate(rally.get("frames", []), 1):
        content.append({"type": "text", "text": f"Frame {index}: {float(frame['time']):.3f}s"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(frame["path"]))}})
    return content


def _default_requester(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Multimodal API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Multimodal API request failed: {error.reason}") from error


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
    if isinstance(response.get("output_text"), str):
        return str(response["output_text"])
    for output in response.get("output", []) if isinstance(response.get("output"), list) else []:
        for item in output.get("content", []) if isinstance(output, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return str(item["text"])
    raise ValueError("Multimodal API response did not contain text")


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Multimodal response was not JSON")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise TypeError("Multimodal response must be a JSON object")
    return value


def validate_machine_label(payload: dict[str, Any]) -> dict[str, Any]:
    terminal_event = str(payload.get("terminal_event", "unknown"))
    last_hitter = str(payload.get("last_hitter", "unknown"))
    landing_side = str(payload.get("landing_side", "unknown"))
    winner = str(payload.get("winner", "unknown"))
    post_rally_event = str(payload.get("post_rally_event", "unknown"))
    if terminal_event not in TERMINAL_EVENTS:
        terminal_event = "unknown"
    if last_hitter not in SIDES:
        last_hitter = "unknown"
    if landing_side not in SIDES:
        landing_side = "unknown"
    if winner not in WINNERS:
        winner = "unknown"
    if post_rally_event not in POST_RALLY_EVENTS:
        post_rally_event = "unknown"
    evidence_frames = []
    for value in payload.get("evidence_frames", []) if isinstance(payload.get("evidence_frames"), list) else []:
        try:
            evidence_frames.append(int(value))
        except (TypeError, ValueError):
            continue
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    expected_winner = "unknown"
    if terminal_event == "landing_in" and landing_side in {"near", "far"}:
        expected_winner = "far" if landing_side == "near" else "near"
    elif terminal_event in {"landing_out", "net"} and last_hitter in {"near", "far"}:
        expected_winner = "far" if last_hitter == "near" else "near"
    elif terminal_event == "unreturned" and last_hitter in {"near", "far"}:
        expected_winner = last_hitter
    if expected_winner != "unknown":
        if winner not in {"unknown", expected_winner}:
            confidence = max(0.0, confidence - 0.12)
        winner = expected_winner
    return {
        "terminal_event": terminal_event,
        "last_hitter": last_hitter,
        "landing_side": landing_side,
        "winner": winner,
        "post_rally_event": post_rally_event,
        "confidence": round(confidence, 3),
        "evidence_frames": sorted(set(evidence_frames)),
        "reason": str(payload.get("reason", ""))[:160],
    }


def consensus_label(labels: list[dict[str, Any]], confidence_threshold: float = 0.82) -> tuple[str, dict[str, Any]]:
    if len(labels) < 2:
        return "review", labels[0] if labels else validate_machine_label({})
    confidence = sum(float(label["confidence"]) for label in labels) / len(labels)
    required = ("winner", "terminal_event")
    agreed = all(len({str(label[field]) for label in labels}) == 1 for field in required)
    merged: dict[str, Any] = {}
    for field in ("winner", "terminal_event", "last_hitter", "landing_side", "post_rally_event"):
        values = {str(label[field]) for label in labels}
        merged[field] = values.pop() if len(values) == 1 else "unknown"
    merged["confidence"] = round(confidence, 3)
    merged["evidence_frames"] = sorted({frame for label in labels for frame in label["evidence_frames"]})
    merged["reason"] = labels[0].get("reason", "")
    ready = agreed and confidence >= confidence_threshold and merged["winner"] in {"near", "far", "no_point"}
    return ("consensus" if ready else "review"), merged


def label_score_evidence(
    manifest_path: Path,
    output_path: Path,
    model: str | list[str],
    endpoint: str,
    api_key: str,
    passes: int = 2,
    timeout: float = 120.0,
    requester: Callable[[str, str, dict[str, Any], float], dict[str, Any]] = _default_requester,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Set SMART_BADMINTON_VLM_API_KEY or OPENAI_API_KEY")
    models = [value.strip() for value in ([model] if isinstance(model, str) else model) if value.strip()]
    if not models:
        raise ValueError("Set SMART_BADMINTON_VLM_MODEL")
    if passes < 1:
        raise ValueError("passes must be at least 1")
    manifest = _read_json(manifest_path, None)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("rallies"), list):
        raise TypeError(f"Invalid evidence manifest: {manifest_path}")
    previous = _read_json(output_path, {"labels": []})
    previous_by_rally = {int(row["rally"]): row for row in previous.get("labels", []) if "rally" in row}
    rows = []
    rallies = manifest["rallies"]
    for completed, rally in enumerate(rallies, 1):
        rally_id = int(rally["rally"])
        responses = []
        request_count = max(passes, len(models))
        for pass_index in range(request_count):
            request_model = models[pass_index % len(models)]
            request_payload = {
                "model": request_model,
                "messages": [{"role": "user", "content": _prompt(rally)}],
                "max_tokens": 600,
                "provider": {"require_parameters": True},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "badminton_terminal_event",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "terminal_event": {"type": "string", "enum": sorted(TERMINAL_EVENTS)},
                                "last_hitter": {"type": "string", "enum": sorted(SIDES)},
                                "landing_side": {"type": "string", "enum": sorted(SIDES)},
                                "winner": {"type": "string", "enum": sorted(WINNERS)},
                                "post_rally_event": {"type": "string", "enum": sorted(POST_RALLY_EVENTS)},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "evidence_frames": {"type": "array", "items": {"type": "integer"}},
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "terminal_event",
                                "last_hitter",
                                "landing_side",
                                "winner",
                                "post_rally_event",
                                "confidence",
                                "evidence_frames",
                                "reason",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            if request_model.startswith("openai/"):
                request_payload["reasoning"] = {"effort": "none", "exclude": True}
            response = requester(endpoint, api_key, request_payload, timeout)
            parsed = validate_machine_label(_parse_json_object(_response_text(response)))
            parsed["model"] = request_model
            responses.append(parsed)
        status, suggestion = consensus_label(responses)
        prior = previous_by_rally.get(rally_id, {})
        rows.append(
            {
                "rally": rally_id,
                "status": prior.get("status") if prior.get("status") in {"accepted", "rejected"} else status,
                "suggestion": suggestion,
                "passes": responses,
                "model": ", ".join(models),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(
            output_path,
            {"schema_version": LABEL_SCHEMA_VERSION, "manifest": str(manifest_path), "labels": rows},
        )
        if progress_callback:
            progress_callback(completed, len(rallies))
    return {"schema_version": LABEL_SCHEMA_VERSION, "manifest": str(manifest_path), "labels": rows}


def load_machine_labels(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path, {"labels": []})
    return payload.get("labels", []) if isinstance(payload, dict) and isinstance(payload.get("labels"), list) else []


def save_machine_label_review(path: Path, rally: int, decision: str) -> list[dict[str, Any]]:
    if decision not in {"accepted", "rejected"}:
        raise ValueError("decision must be accepted or rejected")
    payload = _read_json(path, None)
    if not isinstance(payload, dict) or not isinstance(payload.get("labels"), list):
        raise TypeError("No machine score labels are available")
    found = False
    for row in payload["labels"]:
        if int(row.get("rally", 0)) == rally:
            row["status"] = decision
            row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if not found:
        raise ValueError(f"No machine score label for rally {rally}")
    _atomic_json(path, payload)
    return payload["labels"]
