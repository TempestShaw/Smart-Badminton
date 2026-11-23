"use client";

import { useTranslation } from "react-i18next";
import { t } from "@/lib/i18n";

import { useMemo, useState } from "react";

import { COURT_LANDMARKS, cornersFromLandmarks, type Point } from "@/lib/court-landmarks";

export type CalibrationMethod = "corners" | "landmarks";
export interface PlacedLandmark { id: string; image: Point }

const CORNER_LABELS = ["近左", "近右", "远右", "远左"];
// Points a side or low camera most often still sees, suggested in this order.
const SUGGESTED = [
  "left-post", "right-post", "near-centre-service", "far-centre-service", "near-left-service", "near-right-service",
  "far-left-service", "far-right-service", "far-left-corner", "far-right-corner", "near-left-corner",
  "near-right-corner", "near-centre-baseline", "far-centre-baseline",
];

/** Corner tapping or landmark placement, shared by the laptop and on-device live pages. */
export function useCourtPoints() {
  const [method, setMethodState] = useState<CalibrationMethod>("corners");
  const [corners, setCorners] = useState<Point[]>([]);
  const [placed, setPlaced] = useState<PlacedLandmark[]>([]);
  const [selected, setSelected] = useState<string>(SUGGESTED[0]);

  const nextSuggestion = (current: PlacedLandmark[], after?: string) => {
    const free = SUGGESTED.filter((id) => !current.some((item) => item.id === id));
    if (!free.length) return after ?? SUGGESTED[0];
    const start = after ? SUGGESTED.indexOf(after) : -1;
    return free.find((id) => SUGGESTED.indexOf(id) > start) ?? free[0];
  };

  const place = (point: Point) => {
    const rounded: Point = [Number(point[0].toFixed(5)), Number(point[1].toFixed(5))];
    if (method === "corners") {
      setCorners((current) => (current.length >= 4 ? current : [...current, rounded]));
      return;
    }
    const next = [...placed.filter((item) => item.id !== selected), { id: selected, image: rounded }];
    setPlaced(next);
    setSelected(nextSuggestion(next, selected));
  };

  const undo = () => {
    if (method === "corners") {
      setCorners((current) => current.slice(0, -1));
      return;
    }
    const last = placed[placed.length - 1];
    if (last) setSelected(last.id);
    setPlaced(placed.slice(0, -1));
  };

  const reset = () => {
    setCorners([]);
    setPlaced([]);
    setSelected(SUGGESTED[0]);
  };

  const markers = useMemo(() => (method === "corners"
    ? corners.map((point, index) => ({ key: `corner-${index}`, point, label: String(index + 1) }))
    : placed.map((item) => ({
      key: item.id,
      point: item.image,
      label: String(COURT_LANDMARKS.findIndex((landmark) => landmark.id === item.id) + 1),
    }))), [corners, method, placed]);

  const prompt = method === "corners"
    ? (corners.length < 4 ? `${t("点击：")}${t(CORNER_LABELS[corners.length])}` : null)
    : `${t("点击：")}${t(COURT_LANDMARKS.find((landmark) => landmark.id === selected)?.label ?? "")}`;

  return {
    method,
    setMethod: (value: CalibrationMethod) => { setMethodState(value); reset(); },
    corners,
    placed,
    selected,
    select: setSelected,
    place,
    undo,
    reset,
    markers,
    prompt,
    count: method === "corners" ? corners.length : placed.length,
    ready: method === "corners" ? corners.length === 4 : placed.length >= 4,
    /** Corners for local use (on-device): tapped, or derived from landmarks; throws a LandmarkError when inconsistent. */
    resolveCorners: (): Point[] => (method === "corners" ? corners : cornersFromLandmarks(placed)),
  };
}

export type CourtPoints = ReturnType<typeof useCourtPoints>;

/** A small court plan (near baseline at the bottom) for choosing which landmark the next tap marks. */
export function CourtLandmarkDiagram({ points }: { points: CourtPoints }) {
  useTranslation();
  const lines: [number, number, number, number][] = [
    [0, 0, 6.1, 0], [0, 13.4, 6.1, 13.4], [0, 0, 0, 13.4], [6.1, 0, 6.1, 13.4], [0.46, 0, 0.46, 13.4], [5.64, 0, 5.64, 13.4],
    [0, 0.76, 6.1, 0.76], [0, 12.64, 6.1, 12.64], [0, 4.72, 6.1, 4.72], [0, 8.68, 6.1, 8.68], [3.05, 0, 3.05, 4.72], [3.05, 8.68, 3.05, 13.4],
  ];
  const y = (value: number) => 13.4 - value;
  return (
    <div className="court-landmarks">
      <svg viewBox="-0.9 -0.9 7.9 15.2" role="group" aria-label={t("选择要标记的场地点")}>
        <rect x={0} y={0} width={6.1} height={13.4} className="court-landmarks-floor" />
        {lines.map(([x1, y1, x2, y2], index) => <line key={index} x1={x1} y1={y(y1)} x2={x2} y2={y(y2)} />)}
        <line x1={-0.4} y1={y(6.7)} x2={6.5} y2={y(6.7)} className="court-landmarks-net" />
        {COURT_LANDMARKS.map((landmark, index) => {
          const done = points.placed.some((item) => item.id === landmark.id);
          const active = points.selected === landmark.id;
          return (
            <g
              key={landmark.id}
              className={`court-landmark${done ? " done" : ""}${active ? " active" : ""}`}
              onClick={() => points.select(landmark.id)}
              role="button"
              tabIndex={0}
              aria-pressed={active}
              aria-label={t(landmark.label)}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") points.select(landmark.id); }}
            >
              <circle cx={landmark.court[0]} cy={y(landmark.court[1])} r={0.52} />
              <text x={landmark.court[0]} y={y(landmark.court[1])}>{index + 1}</text>
            </g>
          );
        })}
      </svg>
      <p>{t("先在小图里选点，再在画面中点它的位置。至少 4 个点，且不要都在同一条线上。")}</p>
    </div>
  );
}
