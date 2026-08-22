"use client";

import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { Point, ProjectPayload, ShuttleAnnotation, ShuttleAnnotationPayload } from "@/types/studio";

interface Options {
  visible: boolean;
  editable: boolean;
  displayMode?: "debug" | "trail";
  project: ProjectPayload | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  notify: (message: string, error?: boolean) => void;
  onDirtyChange: (dirty: boolean) => void;
}

export function useShuttleAnnotations({ visible, editable, displayMode = "debug", project, videoRef, notify, onDirtyChange }: Options) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [payload, setPayload] = useState<ShuttleAnnotationPayload | null>(null);
  const [mode, setMode] = useState<"add" | "reject">("add");
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const markDirty = useCallback((value: boolean) => {
    setDirty(value);
    onDirtyChange(value);
  }, [onDirtyChange]);

  useEffect(() => {
    if (!visible || !project) return;
    let current = true;
    setLoading(true);
    if (editable) videoRef.current?.pause();
    apiRequest<ShuttleAnnotationPayload>("/api/shuttle-annotations")
      .then((result) => {
        if (!current || result.project_id !== project.id) return;
        setPayload({ ...result, annotations: result.annotations.map((row) => ({ ...row })) });
        markDirty(false);
      })
      .catch((error: Error) => notify(error.message, true))
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
    };
  }, [editable, markDirty, notify, project, videoRef, visible]);

  const contentRect = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0, width: 1, height: 1 };
    const bounds = canvas.getBoundingClientRect();
    const sourceAspect = (project?.preview.width ?? 16) / (project?.preview.height ?? 9);
    const boxAspect = bounds.width / Math.max(1, bounds.height);
    if (boxAspect > sourceAspect) {
      const width = bounds.height * sourceAspect;
      return { x: (bounds.width - width) / 2, y: 0, width, height: bounds.height };
    }
    const height = bounds.width / sourceAspect;
    return { x: 0, y: (bounds.height - height) / 2, width: bounds.width, height };
  }, [project?.preview.height, project?.preview.width]);

  const screenPoint = useCallback(([x, y]: Point) => {
    const rect = contentRect();
    return { x: rect.x + x * rect.width, y: rect.y + y * rect.height };
  }, [contentRect]);

  const normalizedPoint = (event: ReactPointerEvent<HTMLCanvasElement>): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    const rect = contentRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (x < rect.x || x > rect.x + rect.width || y < rect.y || y > rect.y + rect.height) return null;
    return [(x - rect.x) / rect.width, (y - rect.y) / rect.height];
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!visible || !canvas || !payload) return;
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(bounds.width * ratio));
    canvas.height = Math.max(1, Math.round(bounds.height * ratio));
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    const time = videoRef.current?.currentTime ?? 0;
    const tolerance = Math.max(0.055, 2.5 / Math.max(1, project?.video.fps ?? 30));
    const trailSeconds = 0.85;
    const lowerBound = (target: number) => {
      let low = 0;
      let high = payload.detections.length;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (payload.detections[middle].time < target) low = middle + 1;
        else high = middle;
      }
      return low;
    };
    const recent = payload.detections.slice(lowerBound(time - trailSeconds), lowerBound(time + tolerance + 0.0001));
    const acceptedStatuses = new Set(["tracked", "recovered", "manual"]);
    const latestPrimary = recent.toReversed().find((row) => acceptedStatuses.has(row.status));
    const activeFlight = latestPrimary?.flight_id ?? null;
    const trail = recent.filter((row) => acceptedStatuses.has(row.status) && (!activeFlight || row.flight_id === activeFlight));

    const maximumTrailGap = Math.max(0.20, 5 / Math.max(1, project?.video.fps ?? 30));
    const breaksTrail = (previous: typeof trail[number], current: typeof trail[number]) => {
      const delta = current.time - previous.time;
      if (delta <= 0 || delta > maximumTrailGap) return true;
      return Math.hypot(current.x - previous.x, current.y - previous.y) / delta > 2.6;
    };
    const drawTrail = (color: string, width: number) => {
      if (trail.length < 2) return;
      context.lineCap = "round";
      context.lineJoin = "round";
      for (let index = 1; index < trail.length; index += 1) {
        const previousRow = trail[index - 1];
        const currentRow = trail[index];
        if (breaksTrail(previousRow, currentRow)) continue;
        const previous = screenPoint([previousRow.x, previousRow.y]);
        const current = screenPoint([currentRow.x, currentRow.y]);
        const middle = { x: (previous.x + current.x) / 2, y: (previous.y + current.y) / 2 };
        const age = Math.max(0, time - currentRow.time);
        const alpha = Math.max(0.14, 0.88 * (1 - age / trailSeconds));
        context.beginPath();
        context.moveTo(previous.x, previous.y);
        context.quadraticCurveTo(previous.x, previous.y, middle.x, middle.y);
        context.quadraticCurveTo(current.x, current.y, current.x, current.y);
        context.strokeStyle = color.replace("ALPHA", alpha.toFixed(3));
        context.lineWidth = width;
        context.stroke();
      }
    };

    if (displayMode === "trail") {
      drawTrail("rgba(255, 157, 98, ALPHA)", 3.5);
      return;
    }

    for (let index = 1; index < trail.length; index += 1) {
      const previous = screenPoint([trail[index - 1].x, trail[index - 1].y]);
      const current = screenPoint([trail[index].x, trail[index].y]);
      if (breaksTrail(trail[index - 1], trail[index])) continue;
      const age = Math.max(0, time - trail[index].time);
      const alpha = Math.max(0.12, 0.82 * (1 - age / trailSeconds));
      const manual = trail[index - 1].status === "manual" || trail[index].status === "manual";
      const inpainted = trail[index - 1].detection_status === "inpainted" || trail[index].detection_status === "inpainted";
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(current.x, current.y);
      context.strokeStyle = manual
        ? `rgba(238, 112, 255, ${alpha})`
        : inpainted
          ? `rgba(165, 175, 169, ${alpha})`
          : `rgba(104, 228, 255, ${alpha})`;
      context.lineWidth = manual ? 3 : 2.5;
      context.stroke();
    }
    recent.filter((row) => Math.abs(row.time - time) <= tolerance).forEach((row) => {
      const point = screenPoint([row.x, row.y]);
      context.beginPath();
      if (row.status === "competing") {
        context.moveTo(point.x - 5, point.y - 5); context.lineTo(point.x + 5, point.y + 5);
        context.moveTo(point.x + 5, point.y - 5); context.lineTo(point.x - 5, point.y + 5);
        context.strokeStyle = "#ff9a55";
      } else if (row.status === "stationary" || row.status === "unlinked") {
        context.arc(point.x, point.y, 3, 0, Math.PI * 2);
        context.strokeStyle = "#8d9a93";
      } else {
        context.arc(point.x, point.y, row.status === "recovered" ? 4 : row.status === "manual" ? 6 : 5, 0, Math.PI * 2);
        context.strokeStyle = row.detection_status === "inpainted" || row.status === "recovered"
          ? "#a5afa9"
          : row.status === "manual"
            ? "#ee70ff"
            : "#68e4ff";
      }
      context.lineWidth = 2;
      context.stroke();
      const source = row.source === "optical_flow" ? "OF" : row.source.toUpperCase();
      const status = row.status === "competing" ? "COMP" : row.status.toUpperCase();
      context.font = "700 9px ui-monospace, monospace";
      context.fillStyle = row.status === "competing" ? "#ff9a55" : row.status === "manual" ? "#ee70ff" : "#d7e3dc";
      context.fillText(`${source}/${status}`, point.x + 9, point.y - 8);
    });
    payload.annotations.filter((row) => Math.abs(row.time_seconds - time) <= tolerance).forEach((row) => {
      const point = screenPoint([row.x_normalized, row.y_normalized]);
      context.beginPath();
      if (row.action === "reject") {
        context.moveTo(point.x - 7, point.y - 7); context.lineTo(point.x + 7, point.y + 7);
        context.moveTo(point.x + 7, point.y - 7); context.lineTo(point.x - 7, point.y + 7);
        context.strokeStyle = "#ff5666";
      } else {
        context.arc(point.x, point.y, 7, 0, Math.PI * 2);
        context.strokeStyle = "#ee70ff";
      }
      context.lineWidth = 3;
      context.stroke();
    });
  }, [displayMode, payload, project?.video.fps, screenPoint, videoRef, visible]);

  useEffect(() => {
    const video = videoRef.current;
    draw();
    video?.addEventListener("timeupdate", draw);
    video?.addEventListener("seeked", draw);
    window.addEventListener("resize", draw);
    return () => {
      video?.removeEventListener("timeupdate", draw);
      video?.removeEventListener("seeked", draw);
      window.removeEventListener("resize", draw);
    };
  }, [draw, videoRef]);

  const add = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!editable || !payload || !project) return;
    let point = normalizedPoint(event);
    if (!point) return;
    event.preventDefault();
    const time = videoRef.current?.currentTime ?? 0;
    if (mode === "reject") {
      const tolerance = Math.max(0.055, 2.5 / Math.max(1, project.video.fps));
      const click = screenPoint(point);
      let nearest: typeof payload.detections[number] | null = null;
      let nearestDistance = 18;
      for (const row of payload.detections.filter((candidate) => Math.abs(candidate.time - time) <= tolerance)) {
        const target = screenPoint([row.x, row.y]);
        const distance = Math.hypot(target.x - click.x, target.y - click.y);
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      if (nearest) point = [nearest.x, nearest.y];
    }
    const annotation: ShuttleAnnotation = {
      id: `user-${Date.now()}-${payload.annotations.length + 1}`,
      time_seconds: Number(time.toFixed(6)),
      x_normalized: Number(point[0].toFixed(7)),
      y_normalized: Number(point[1].toFixed(7)),
      action: mode,
      note: "studio",
    };
    setPayload({ ...payload, annotations: [...payload.annotations, annotation] });
    markDirty(true);
  };

  const undo = () => {
    if (!payload?.annotations.length) return;
    setPayload({ ...payload, annotations: payload.annotations.slice(0, -1) });
    markDirty(true);
  };

  const save = async (): Promise<boolean> => {
    if (!payload || !project) return false;
    try {
      setSaving(true);
      const result = await apiRequest<ShuttleAnnotationPayload>("/api/shuttle-annotations", {
        method: "PUT",
        body: JSON.stringify({ project_id: project.id, annotations: payload.annotations }),
      });
      setPayload(result);
      markDirty(false);
      notify("羽球标注已保存");
      return true;
    } catch (error) {
      notify((error as Error).message, true);
      return false;
    } finally {
      setSaving(false);
    }
  };

  return { canvasRef, payload, mode, dirty, loading, saving, setMode, draw, add, undo, save };
}
