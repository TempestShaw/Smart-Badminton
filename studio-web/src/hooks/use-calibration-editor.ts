"use client";

import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { CalibrationPayload, CalibrationRegion, Point, ProjectPayload } from "@/types/studio";

interface Options {
  active: boolean;
  project: ProjectPayload | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  notify: (message: string, error?: boolean) => void;
  onDirtyChange: (dirty: boolean) => void;
  onSaved: (payload: CalibrationPayload) => void;
}

function clonePayload(payload: CalibrationPayload): CalibrationPayload {
  return {
    ...payload,
    regions: payload.regions.map((region) => ({
      ...region,
      points: (region.points ?? []).map((point) => [...point] as Point),
    })),
  };
}

export function useCalibrationEditor({ active, project, videoRef, notify, onDirtyChange, onSaved }: Options) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ pointIndex: number; pointerId: number } | null>(null);
  const [payload, setPayload] = useState<CalibrationPayload | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const markDirty = useCallback((value: boolean) => {
    setDirty(value);
    onDirtyChange(value);
  }, [onDirtyChange]);

  useEffect(() => {
    if (!active || !project) return;
    let current = true;
    setLoading(true);
    videoRef.current?.pause();
    apiRequest<CalibrationPayload>("/api/calibration")
      .then((result) => {
        if (!current) return;
        const normalized = clonePayload(result);
        setPayload(normalized);
        const incomplete = normalized.regions.find((region) => region.required && (region.points?.length ?? 0) < (region.minimum_points ?? 3));
        setSelectedId(incomplete?.id ?? normalized.regions[0]?.id ?? "");
        markDirty(false);
      })
      .catch((error: Error) => notify(error.message, true))
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
    };
  }, [active, markDirty, notify, project, videoRef]);

  const selected = useMemo(
    () => payload?.regions.find((region) => region.id === selectedId) ?? null,
    [payload, selectedId],
  );

  const contentRect = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0, width: 1, height: 1 };
    const bounds = canvas.getBoundingClientRect();
    const sourceWidth = project?.preview.width ?? 16;
    const sourceHeight = project?.preview.height ?? 9;
    const sourceAspect = sourceWidth / sourceHeight;
    const boxAspect = bounds.width / Math.max(1, bounds.height);
    if (boxAspect > sourceAspect) {
      const width = bounds.height * sourceAspect;
      return { x: (bounds.width - width) / 2, y: 0, width, height: bounds.height };
    }
    const height = bounds.width / sourceAspect;
    return { x: 0, y: (bounds.height - height) / 2, width: bounds.width, height };
  }, [project?.preview.height, project?.preview.width]);

  const screenPoint = useCallback((point: Point) => {
    const rect = contentRect();
    return { x: rect.x + point[0] * rect.width, y: rect.y + point[1] * rect.height };
  }, [contentRect]);

  const normalizedPoint = useCallback((event: ReactPointerEvent<HTMLCanvasElement>): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    const rect = contentRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (x < rect.x || x > rect.x + rect.width || y < rect.y || y > rect.y + rect.height) return null;
    return [Math.max(0, Math.min(1, (x - rect.x) / rect.width)), Math.max(0, Math.min(1, (y - rect.y) / rect.height))];
  }, [contentRect]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!active || !canvas || !payload) return;
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(bounds.width * ratio));
    const height = Math.max(1, Math.round(bounds.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    for (const region of payload.regions) {
      const normalized = region.points ?? [];
      if (!normalized.length) continue;
      const points = normalized.map(screenPoint);
      const isSelected = region.id === selectedId;
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
      if (points.length >= 3) context.closePath();
      context.fillStyle = `${region.color}24`;
      context.strokeStyle = region.color;
      context.lineWidth = isSelected ? 3 : 1.5;
      context.setLineDash(region.type?.includes("polygon") ? [7, 5] : []);
      if (points.length >= 3) context.fill();
      context.stroke();
      context.setLineDash([]);
      if (isSelected) {
        points.forEach((point, index) => {
          context.beginPath();
          context.arc(point.x, point.y, 6, 0, Math.PI * 2);
          context.fillStyle = "#0b0f0d";
          context.fill();
          context.lineWidth = 3;
          context.strokeStyle = region.color;
          context.stroke();
          context.fillStyle = "#fff";
          context.font = "8px ui-monospace, monospace";
          context.fillText(String(index + 1), point.x + 8, point.y - 8);
        });
      }
      context.fillStyle = region.color;
      context.font = `${isSelected ? 700 : 600} 10px ui-monospace, monospace`;
      context.fillText(region.label, points[0].x + 8, points[0].y + 16);
    }
  }, [active, payload, screenPoint, selectedId]);

  useEffect(() => {
    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [draw]);

  const mutateRegions = useCallback((change: (regions: CalibrationRegion[]) => CalibrationRegion[]) => {
    setPayload((current) => current ? { ...current, regions: change(current.regions.map((region) => ({ ...region, points: [...(region.points ?? [])] }))) } : current);
    markDirty(true);
  }, [markDirty]);

  const pointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = normalizedPoint(event);
    if (!point || !selected) return;
    event.preventDefault();
    const click = screenPoint(point);
    const points = selected.points ?? [];
    let nearest = -1;
    let distanceLimit = 14;
    points.forEach((candidate, index) => {
      const target = screenPoint(candidate);
      const distance = Math.hypot(target.x - click.x, target.y - click.y);
      if (distance < distanceLimit) {
        nearest = index;
        distanceLimit = distance;
      }
    });
    if (nearest < 0) {
      const maximum = selected.maximum_points ?? 64;
      if (points.length >= maximum) return notify(`${selected.label} 最多只能有 ${maximum} 个点`, true);
      nearest = points.length;
      mutateRegions((regions) => regions.map((region) => region.id === selected.id
        ? { ...region, points: [...(region.points ?? []), point.map((value) => Number(value.toFixed(6))) as Point] }
        : region));
    }
    dragRef.current = { pointIndex: nearest, pointerId: event.pointerId };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const pointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    const point = normalizedPoint(event);
    if (!drag || !point || !selected) return;
    mutateRegions((regions) => regions.map((region) => {
      if (region.id !== selected.id) return region;
      const points = [...(region.points ?? [])];
      points[drag.pointIndex] = point.map((value) => Number(value.toFixed(6))) as Point;
      return { ...region, points };
    }));
  };

  const pointerUp = () => {
    dragRef.current = null;
  };

  const undoPoint = () => selected && mutateRegions((regions) => regions.map((region) => region.id === selected.id
    ? { ...region, points: (region.points ?? []).slice(0, -1) }
    : region));
  const clearRegion = () => selected && mutateRegions((regions) => regions.map((region) => region.id === selected.id ? { ...region, points: [] } : region));
  const deleteRegion = () => {
    if (!selected?.type) return;
    mutateRegions((regions) => regions.filter((region) => region.id !== selected.id));
    setSelectedId(payload?.regions[0]?.id ?? "");
  };
  const addRegion = (type: "background_court_polygons" | "static_false_positive_polygons") => {
    if (!payload) return;
    const count = payload.regions.filter((region) => region.type === type).length + 1;
    const background = type === "background_court_polygons";
    const region: CalibrationRegion = {
      id: `${type}:${Date.now()}`,
      type,
      label: `${background ? "背景排除区" : "静态误检区"} ${count}`,
      color: background ? "#909a94" : "#ff9a55",
      required: false,
      points: [],
    };
    mutateRegions((regions) => [...regions, region]);
    setSelectedId(region.id);
  };

  const autoGenerate = (replaceExisting = false) => {
    if (!payload) return false;
    const activeRegion = payload.regions.find((region) => region.id === "active_court_polygon");
    const points = activeRegion?.points ?? [];
    if (points.length < 3) {
      notify("请先在红色有效比赛场地里圈至少 3 个点", true);
      return false;
    }
    const targetIds = ["near_player_zone", "far_player_zone", "net_band", "shuttle_airspace_polygon", "shuttle_perspective_axis"];
    if (!replaceExisting && targetIds.some((id) => (payload.regions.find((region) => region.id === id)?.points?.length ?? 0) > 0)) {
      return false;
    }
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const width = Math.max(0.05, maxX - minX), height = Math.max(0.05, maxY - minY);
    const split = minY + height * 0.43, overlap = height * 0.08, centerX = (minX + maxX) / 2;
    const generated: Record<string, Point[]> = {
      far_player_zone: [[minX, minY], [maxX, minY], [maxX, split + overlap], [minX, split + overlap]],
      near_player_zone: [[minX, split - overlap], [maxX, split - overlap], [maxX, maxY], [minX, maxY]],
      net_band: [[minX, split - height * 0.04], [maxX, split - height * 0.04], [maxX, split + height * 0.04], [minX, split + height * 0.04]],
      shuttle_airspace_polygon: [[minX - width * 0.08, maxY], [minX - width * 0.08, minY - height * 0.75], [maxX + width * 0.08, minY - height * 0.75], [maxX + width * 0.08, maxY]],
      shuttle_perspective_axis: [[centerX, Math.max(0.06, minY - height * 1.05)], [centerX, Math.min(0.95, minY + height * 0.45)]],
    };
    mutateRegions((regions) => regions.map((region) => generated[region.id]
      ? { ...region, points: generated[region.id].map(([x, y]) => [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))]) }
      : region));
    setSelectedId("near_player_zone");
    notify("辅助区已生成");
    return true;
  };

  const save = async () => {
    if (!payload || !project) return;
    try {
      setSaving(true);
      const result = await apiRequest<CalibrationPayload>("/api/calibration", {
        method: "PUT",
        body: JSON.stringify({
          project_id: project.id,
          source_time_seconds: videoRef.current?.currentTime ?? 0,
          regions: payload.regions,
        }),
      });
      setPayload(clonePayload(result));
      markDirty(false);
      onSaved(result);
      notify("球场校准已保存");
    } catch (error) {
      notify((error as Error).message, true);
    } finally {
      setSaving(false);
    }
  };

  return {
    canvasRef,
    payload,
    selected,
    selectedId,
    dirty,
    loading,
    saving,
    setSelectedId,
    draw,
    pointerDown,
    pointerMove,
    pointerUp,
    undoPoint,
    clearRegion,
    deleteRegion,
    addRegion,
    autoGenerate,
    save,
  };
}
