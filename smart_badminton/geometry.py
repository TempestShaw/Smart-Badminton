from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Point = tuple[float, float]


def _points(value: Iterable[Iterable[float]]) -> list[Point]:
    return [(float(x), float(y)) for x, y in value]


@dataclass
class CourtGeometry:
    reference_width: int
    reference_height: int
    court_ground_polygon: list[Point]
    near_player_zone: list[Point]
    far_player_zone: list[Point]
    net_band: list[Point]
    shuttle_airspace_polygon: list[Point]
    static_false_positive_polygons: list[list[Point]]
    background_court_polygons: list[list[Point]]
    active_court_polygon: list[Point] | None = None
    court_corners: list[Point] | None = None
    shuttle_perspective_axis: list[Point] | None = None
    shuttle_vanishing_extension: float = 1.5

    @classmethod
    def from_json(cls, path: Path) -> CourtGeometry:
        data = json.loads(path.read_text(encoding="utf-8"))
        ref = data["reference_frame"]
        masks = data["masks_normalized"]
        calibration = data.get("calibration", {})
        return cls(
            reference_width=int(ref["width"]),
            reference_height=int(ref["height"]),
            court_ground_polygon=_points(masks["court_ground_polygon"]),
            near_player_zone=_points(masks["near_player_zone"]),
            far_player_zone=_points(masks["far_player_zone"]),
            net_band=_points(masks["net_band"]),
            shuttle_airspace_polygon=_points(masks["shuttle_airspace_polygon"]),
            static_false_positive_polygons=[_points(p) for p in masks.get("static_false_positive_polygons", [])],
            background_court_polygons=[_points(p) for p in masks.get("background_court_polygons", [])],
            active_court_polygon=_points(masks["active_court_polygon"])
            if masks.get("active_court_polygon")
            else None,
            court_corners=_points(calibration["court_corners_normalized"])
            if calibration.get("court_corners_normalized")
            else None,
            shuttle_perspective_axis=_points(calibration["shuttle_perspective_axis_normalized"])
            if calibration.get("shuttle_perspective_axis_normalized")
            else None,
            shuttle_vanishing_extension=float(calibration.get("shuttle_vanishing_extension", 1.5)),
        )

    @staticmethod
    def denormalize(points: list[Point], width: int, height: int) -> np.ndarray:
        return np.asarray([(round(x * width), round(y * height)) for x, y in points], dtype=np.int32)

    def polygon(self, name: str, width: int, height: int) -> np.ndarray:
        return self.denormalize(getattr(self, name), width, height)

    def polygon_mask(self, name: str, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = self.polygon(name, width, height)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
        return mask

    def exclusion_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in self.static_false_positive_polygons + self.background_court_polygons:
            points = self.denormalize(polygon, width, height)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255)
        return mask

    def contains_normalized(self, polygon_name: str, x: float, y: float) -> bool:
        polygon = np.asarray(getattr(self, polygon_name), dtype=np.float32)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def contains_active_court(self, x: float, y: float, margin: float = 0.0) -> bool:
        if not self.active_court_polygon:
            return True
        polygon = np.asarray(self.active_court_polygon, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), True) >= -float(margin)

    def shuttle_vanishing_point(self) -> Point | None:
        if not self.shuttle_perspective_axis or len(self.shuttle_perspective_axis) != 2:
            return None
        top = np.asarray(self.shuttle_perspective_axis[0], dtype=float)
        bottom = np.asarray(self.shuttle_perspective_axis[1], dtype=float)
        if bottom[1] < top[1]:
            top, bottom = bottom, top
        if bottom[1] - top[1] < 0.05:
            return None
        vanishing = top + (top - bottom) * self.shuttle_vanishing_extension
        return float(vanishing[0]), float(vanishing[1])

    def contains_shuttle_volume(self, x: float, y: float) -> bool:
        """Test whether a 2D detection ray can originate above the active court.

        The calibrated perspective axis is extended beyond the image to form a
        vertical vanishing point. A candidate is accepted when the ray from
        that point through the candidate intersects the active ground polygon
        farther down the image. This retains high clears above the legacy 2D
        mask while rejecting most detections whose ray lands on another court.
        """
        vanishing = self.shuttle_vanishing_point()
        if vanishing is None or not self.active_court_polygon:
            return self.contains_normalized("shuttle_airspace_polygon", x, y)
        point = np.asarray([float(x), float(y)], dtype=float)
        origin = np.asarray(vanishing, dtype=float)
        direction = point - origin
        if float(np.linalg.norm(direction)) < 1e-8:
            return True
        polygon = np.asarray(self.active_court_polygon, dtype=float)
        if cv2.pointPolygonTest(polygon.astype(np.float32), (float(x), float(y)), False) >= 0:
            return True
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            edge = second - first
            denominator = _cross_2d(direction, edge)
            if abs(denominator) < 1e-10:
                continue
            offset = first - origin
            ray_scale = _cross_2d(offset, edge) / denominator
            edge_scale = _cross_2d(offset, direction) / denominator
            if ray_scale >= 1.0 - 1e-6 and -1e-6 <= edge_scale <= 1.0 + 1e-6:
                return True
        return False

    def projected_shuttle_volume_polygon(self, width: int, height: int) -> np.ndarray:
        vanishing = self.shuttle_vanishing_point()
        if vanishing is None or not self.active_court_polygon:
            return self.polygon("shuttle_airspace_polygon", width, height)
        origin = np.asarray(vanishing, dtype=float)
        points = [np.asarray(point, dtype=float) for point in self.active_court_polygon]
        projected = list(points)
        top_y = 0.0
        for point in points:
            delta_y = point[1] - origin[1]
            if abs(delta_y) < 1e-8:
                continue
            scale = (top_y - origin[1]) / delta_y
            top = origin + (point - origin) * scale
            projected.append(np.asarray([top[0], top_y]))
        pixels = self.denormalize([(float(point[0]), float(point[1])) for point in projected], width, height)
        return cv2.convexHull(pixels)

    def player_inference_bounds(self, side: str, width: int, height: int) -> tuple[int, int, int, int]:
        if side not in {"near", "far"}:
            raise ValueError("Player inference side must be near or far")
        zone = f"{side}_player_zone"
        x, y, zone_width, zone_height = cv2.boundingRect(self.polygon(zone, width, height))
        padding_x, padding_y = round(width * 0.04), round(height * 0.035)
        x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
        x2, y2 = min(width, x + zone_width + padding_x), min(height, y + zone_height + padding_y)
        headroom = height * (0.50 if side == "near" else 0.24)
        y1 = max(0, round(y - headroom))
        if self.active_court_polygon:
            active_x, _active_y, active_width, _active_height = cv2.boundingRect(
                self.polygon("active_court_polygon", width, height)
            )
            lateral_margin = round(width * 0.06)
            x1 = max(x1, active_x - lateral_margin)
            x2 = min(x2, active_x + active_width + lateral_margin)
        return x1, y1, x2, y2

    def excluded_normalized(self, x: float, y: float) -> bool:
        for polygon in self.static_false_positive_polygons + self.background_court_polygons:
            if cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), (float(x), float(y)), False) >= 0:
                return True
        return False

    def homography_to_court(self, width: int, height: int) -> np.ndarray | None:
        if not self.court_corners or len(self.court_corners) != 4:
            return None
        image = self.denormalize(self.court_corners, width, height).astype(np.float32)
        court = np.asarray([[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]], dtype=np.float32)
        return cv2.getPerspectiveTransform(image, court)

    def image_to_court(self, point: Point, width: int, height: int) -> Point | None:
        matrix = self.homography_to_court(width, height)
        if matrix is None:
            return None
        source = np.asarray([[point]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def homography_quality(self, width: int, height: int, point_uncertainty_pixels: float = 3.0) -> dict[str, float | bool | str]:
        """Estimate sensitivity to approximate four-corner calibration.

        Four points fit a homography exactly, so ordinary reprojection error is
        misleadingly zero. Instead, perturb the clicked corners by the stated
        pixel uncertainty and report how far court-grid points move in meters.
        This quantifies calibration sensitivity, not shuttle height ambiguity.
        """
        matrix = self.homography_to_court(width, height)
        if matrix is None or not self.court_corners:
            return {"available": False, "reason": "four ordered court corners are not calibrated"}
        image = self.denormalize(self.court_corners, width, height).astype(np.float32)
        polygon_area = abs(float(cv2.contourArea(image))) / max(1.0, float(width * height))
        condition_number = float(np.linalg.cond(matrix))
        court_grid = np.asarray(
            [[[x, y] for x in (0.0, 3.05, 6.1) for y in (0.0, 3.35, 6.7, 10.05, 13.4)]],
            dtype=np.float32,
        )
        inverse = np.linalg.inv(matrix)
        image_grid = cv2.perspectiveTransform(court_grid, inverse)
        random = np.random.default_rng(20260821)
        errors = []
        for _ in range(32):
            jitter = random.normal(0.0, point_uncertainty_pixels, size=image.shape).astype(np.float32)
            perturbed = image + jitter
            if not cv2.isContourConvex(perturbed.astype(np.int32)):
                continue
            candidate = cv2.getPerspectiveTransform(
                perturbed,
                np.asarray([[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]], dtype=np.float32),
            )
            mapped = cv2.perspectiveTransform(image_grid, candidate)
            errors.extend(np.linalg.norm(mapped - court_grid, axis=2).ravel().tolist())
        if not errors:
            return {"available": False, "reason": "corner ordering is unstable"}
        median_error = float(np.median(errors))
        p95_error = float(np.percentile(errors, 95))
        confidence = float(
            np.clip((polygon_area / 0.18) * np.exp(-p95_error / 0.45), 0.0, 1.0)
        )
        return {
            "available": True,
            "point_uncertainty_pixels": float(point_uncertainty_pixels),
            "median_uncertainty_meters": round(median_error, 3),
            "p95_uncertainty_meters": round(p95_error, 3),
            "image_area_fraction": round(polygon_area, 4),
            "condition_number": round(condition_number, 2),
            "confidence": round(confidence, 3),
            "score_safe": bool(p95_error <= 0.20 and confidence >= 0.70),
        }


def draw_geometry_preview(frame: np.ndarray, geometry: CourtGeometry) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    layers = [
        ("active_court_polygon", (30, 30, 245), "active court"),
        ("court_ground_polygon", (40, 210, 40), "court"),
        ("near_player_zone", (255, 100, 30), "near"),
        ("far_player_zone", (30, 190, 255), "far"),
        ("net_band", (220, 80, 220), "net"),
    ]
    for name, color, label in layers:
        if getattr(geometry, name) is None:
            continue
        points = geometry.polygon(name, width, height)
        cv2.polylines(output, [points], True, color, 2, cv2.LINE_AA)
        if len(points):
            anchor = tuple(points[0])
            cv2.putText(output, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    volume = geometry.projected_shuttle_volume_polygon(width, height)
    cv2.polylines(output, [volume], True, (40, 230, 230), 2, cv2.LINE_AA)
    if geometry.shuttle_perspective_axis:
        axis = geometry.denormalize(geometry.shuttle_perspective_axis, width, height)
        cv2.arrowedLine(output, tuple(axis[0]), tuple(axis[1]), (20, 245, 245), 2, cv2.LINE_AA)
    for polygon in geometry.static_false_positive_polygons:
        points = geometry.denormalize(polygon, width, height)
        overlay = output.copy()
        cv2.fillPoly(overlay, [points], (20, 20, 210))
        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
    for polygon in geometry.background_court_polygons:
        points = geometry.denormalize(polygon, width, height)
        overlay = output.copy()
        cv2.fillPoly(overlay, [points], (100, 100, 100))
        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
    return output


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])
