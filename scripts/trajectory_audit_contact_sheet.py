from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    rows = pd.read_csv(args.trajectory)
    rows = rows[rows["status"].isin(["tracked", "recovered", "manual"])].copy()
    rows["x_normalized"] = rows["center_x"] / rows["source_width"]
    rows["y_normalized"] = rows["center_y"] / rows["source_height"]
    groups = []
    for flight_id, group in rows.groupby("flight_id"):
        groups.append(
            {
                "flight_id": flight_id,
                "time": float(group["time_seconds"].median()),
                "x": float(group["x_normalized"].mean()),
                "y": float(group["y_normalized"].mean()),
                "count": len(group),
                "priority": max(abs(float(group["x_normalized"].mean()) - 0.42), max(0.0, 0.42 - float(group["y_normalized"].mean()))),
                "evidence": ",".join(sorted(set(group["ownership_evidence"].fillna("unknown").astype(str)))),
            }
        )
    selected = sorted(groups, key=lambda item: item["priority"], reverse=True)[: args.limit]
    capture = cv2.VideoCapture(str(args.video))
    tiles = []
    try:
        for item in selected:
            capture.set(cv2.CAP_PROP_POS_MSEC, item["time"] * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            flight = rows[rows["flight_id"] == item["flight_id"]].sort_values("time_seconds")
            points = np.column_stack(
                [
                    flight["center_x"].to_numpy() * width / flight["source_width"].to_numpy(),
                    flight["center_y"].to_numpy() * height / flight["source_height"].to_numpy(),
                ]
            ).round().astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(frame, [points], False, (0, 185, 255), 3, cv2.LINE_AA)
            for point in points[:: max(1, len(points) // 12)]:
                cv2.circle(frame, tuple(point), 4, (0, 255, 255), -1, cv2.LINE_AA)
            label = f"F{item['flight_id']} {item['time']:.2f}s x={item['x']:.2f} y={item['y']:.2f} n={item['count']}"
            cv2.rectangle(frame, (12, 12), (min(width - 12, 12 + len(label) * 15), 82), (10, 10, 10), -1)
            cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, item["evidence"], (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 120), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError("No audit frames could be read")
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 3:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write {args.output}")


if __name__ == "__main__":
    main()
