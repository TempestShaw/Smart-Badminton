import type { Segment } from "@/types/studio";

export function formatTime(seconds: number, milliseconds = true): string {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(milliseconds ? 3 : 0)
    .padStart(milliseconds ? 6 : 2, "0")}`;
}

export function cloneSegments(segments: Segment[]): Segment[] {
  return segments.map((segment) => ({ ...segment }));
}

export function outputDuration(segments: Segment[]): number {
  return segments.reduce((total, segment) => total + segment.end - segment.start, 0);
}

export function frameSnap(value: number, fps: number): number {
  const step = 1 / Math.max(1, fps || 30);
  return Math.round(value / step) * step;
}

export function clampBoundary(
  segments: Segment[],
  index: number,
  edge: "start" | "end",
  value: number,
  duration: number,
  fps: number,
): Segment[] {
  const next = cloneSegments(segments);
  const segment = next[index];
  if (!segment) return next;
  const step = 1 / Math.max(1, fps || 30);
  if (edge === "start") {
    const minimum = index > 0 ? next[index - 1].end : 0;
    segment.start = frameSnap(Math.max(minimum, Math.min(value, segment.end - step)), fps);
  } else {
    const maximum = index + 1 < next.length ? next[index + 1].start : duration;
    segment.end = frameSnap(Math.min(maximum, Math.max(value, segment.start + step)), fps);
  }
  return next;
}
