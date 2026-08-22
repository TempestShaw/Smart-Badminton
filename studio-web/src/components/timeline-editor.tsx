"use client";

import type { PointerEvent as ReactPointerEvent } from "react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Maximize2, Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { StudioController } from "@/hooks/use-studio-controller";
import { clampBoundary, cloneSegments, formatTime, outputDuration } from "@/lib/time";
import type { EvidencePayload, Segment } from "@/types/studio";

const GUTTER = 92;
const evidenceLabels: Record<string, string> = {
  model_keep: "模型保留阈值",
  model_active: "模型活动阈值",
  activity_high: "高动作证据",
  trajectory_visible: "羽球轨迹可见",
  trajectory_descending: "羽球正在下降",
  trajectory_occluded: "疑似人体遮挡",
  near_ready: "近场已预备",
  far_ready: "远场已预备",
  between_points: "分间阶段",
  handoff: "疑似送球",
};

interface BoundaryDrag {
  index: number;
  edge: "start" | "end";
  startX: number;
  original: Segment[];
  current: Segment[];
  lastSeekAt: number;
}

interface ScrubSeek {
  lastSeekAt: number;
  pendingTime: number;
  timer: number | null;
}

function timelinePercent(time: number, duration: number): string {
  if (duration <= 0) return "0%";
  return `${Math.max(0, Math.min(100, time / duration * 100))}%`;
}

const RulerTicks = memo(function RulerTicks({ duration, major }: { duration: number; major: number }) {
  const minor = major / 5;
  const ticks = [];
  for (let value = 0; value <= duration + 0.001; value += minor) {
    const isMajor = Math.abs(value / major - Math.round(value / major)) < 0.001;
    ticks.push(
      <span key={value} className={isMajor ? "ruler-tick" : "ruler-tick minor"} style={{ left: timelinePercent(value, duration) }}>
        {isMajor ? formatTime(value, false) : "·"}
      </span>,
    );
  }
  return ticks;
});

const EvidenceItems = memo(function EvidenceItems({ evidence, duration }: { evidence: EvidencePayload | undefined; duration: number }) {
  if (!evidence?.available) return <span className="evidence-empty">{evidence?.reason || "运行自动分析后显示逐帧证据"}</span>;
  return (
    <>
      {Object.entries(evidence.signals ?? {}).flatMap(([name, spans]) => name === "landing_candidate" ? [] : spans.map(([start, end], index) => (
        <span
          key={`${name}-${index}`}
          className={`evidence-span ${name.replaceAll("_", "-")}`}
          style={{ left: timelinePercent(start, duration), width: timelinePercent(end - start, duration) }}
          title={`${evidenceLabels[name] || name} ${formatTime(start)}–${formatTime(end)}`}
        />
      )))}
      {(evidence.serves ?? []).map((event, index) => <span key={`serve-${index}`} className={`evidence-marker serve-${event.server}`} style={{ left: timelinePercent(event.time, duration) }} title={`${event.server === "near" ? "近场" : "远场"}正式发球 ${formatTime(event.time)}`} />)}
      {(evidence.contacts ?? []).map((event, index) => <span key={`contact-${index}`} className="evidence-marker contact" style={{ left: timelinePercent(event.time, duration) }} title={`严格球拍接触 ${formatTime(event.time)} · ${Math.round(event.confidence * 100)}%`} />)}
      {(evidence.terminal_events ?? []).map((event, index) => <span key={`terminal-${index}`} className="evidence-marker terminal" style={{ left: timelinePercent(event.time, duration) }} title={`终局 ${event.event} ${formatTime(event.time)} · ${Math.round(event.confidence * 100)}%`} />)}
    </>
  );
});

export function TimelineEditor({ studio }: { studio: StudioController }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<BoundaryDrag | null>(null);
  const segmentsRef = useRef(studio.segments);
  const scrubSeekRef = useRef<ScrubSeek>({ lastSeekAt: 0, pendingTime: 0, timer: null });
  const zoomFrameRef = useRef<number | null>(null);
  const pendingZoomRef = useRef(8);
  const [pixelsPerSecond, setPixelsPerSecond] = useState(8);
  const [containerWidth, setContainerWidth] = useState(900);
  const { finishPreviewChange, previewSegments, project, videoRef } = studio;
  segmentsRef.current = studio.segments;
  const duration = studio.project?.video.duration ?? 0;
  const timelineWidth = Math.max(400, containerWidth - GUTTER, duration * pixelsPerSecond);
  const timelineScale = duration > 0 ? timelineWidth / duration : pixelsPerSecond;
  const rulerMajor = pixelsPerSecond >= 14 ? 5 : pixelsPerSecond >= 6 ? 10 : 30;

  const scheduleZoom = useCallback((value: number) => {
    pendingZoomRef.current = value;
    if (zoomFrameRef.current !== null) return;
    zoomFrameRef.current = window.requestAnimationFrame(() => {
      zoomFrameRef.current = null;
      setPixelsPerSecond(pendingZoomRef.current);
    });
  }, []);

  const commitZoom = useCallback((value: number) => {
    pendingZoomRef.current = value;
    if (zoomFrameRef.current !== null) window.cancelAnimationFrame(zoomFrameRef.current);
    zoomFrameRef.current = null;
    setPixelsPerSecond(value);
  }, []);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    let frame = 0;
    const update = () => {
      if (playheadRef.current && video) {
        playheadRef.current.style.transform = `translateX(${(video.currentTime || 0) * timelineScale}px)`;
      }
      if (video && !video.paused) frame = window.requestAnimationFrame(update);
    };
    const start = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(update);
    };
    video?.addEventListener("play", start);
    video?.addEventListener("seeked", update);
    update();
    return () => {
      window.cancelAnimationFrame(frame);
      video?.removeEventListener("play", start);
      video?.removeEventListener("seeked", update);
    };
  }, [studio.mediaSource, timelineScale, videoRef]);

  const seekPreview = useCallback((time: number, immediate = false) => {
    const video = videoRef.current;
    if (!video) return;
    const now = performance.now();
    const drag = dragRef.current;
    if (!immediate && drag && now - drag.lastSeekAt < 90) return;
    if (drag) drag.lastSeekAt = now;
    video.pause();
    const fastVideo = video as HTMLVideoElement & { fastSeek?: (value: number) => void };
    if (!immediate && typeof fastVideo.fastSeek === "function") fastVideo.fastSeek(time);
    else video.currentTime = time;
  }, [videoRef]);

  const seekScrub = useCallback((time: number, precise = false) => {
    const video = videoRef.current;
    if (!video) return;
    const state = scrubSeekRef.current;
    state.pendingTime = time;
    if (precise) {
      if (state.timer !== null) window.clearTimeout(state.timer);
      state.timer = null;
      state.lastSeekAt = performance.now();
      video.pause();
      video.currentTime = time;
      return;
    }
    if (state.timer !== null) return;
    const delay = Math.max(0, 80 - (performance.now() - state.lastSeekAt));
    state.timer = window.setTimeout(() => {
      state.timer = null;
      state.lastSeekAt = performance.now();
      video.pause();
      const fastVideo = video as HTMLVideoElement & { fastSeek?: (value: number) => void };
      if (typeof fastVideo.fastSeek === "function") fastVideo.fastSeek(state.pendingTime);
      else video.currentTime = state.pendingTime;
    }, delay);
  }, [videoRef]);

  useEffect(() => () => {
    if (scrubSeekRef.current.timer !== null) window.clearTimeout(scrubSeekRef.current.timer);
    if (zoomFrameRef.current !== null) window.cancelAnimationFrame(zoomFrameRef.current);
  }, []);

  const beginBoundaryDrag = (event: ReactPointerEvent, index: number, edge: "start" | "end") => {
    event.preventDefault();
    event.stopPropagation();
    studio.selectSegment(index, false);
    const original = cloneSegments(studio.segments);
    dragRef.current = { index, edge, startX: event.clientX, original, current: original, lastSeekAt: 0 };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  useEffect(() => {
    const move = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || !project) return;
      const originalSegment = drag.original[drag.index];
      const delta = (event.clientX - drag.startX) / timelineScale;
      const value = originalSegment[drag.edge] + delta;
      const next = clampBoundary(
        drag.original,
        drag.index,
        drag.edge,
        value,
        project.video.duration,
        project.video.fps,
      );
      drag.current = next;
      previewSegments(next);
      seekPreview(next[drag.index][drag.edge]);
    };
    const finish = () => {
      const drag = dragRef.current;
      if (!drag) return;
      dragRef.current = null;
      finishPreviewChange(drag.original);
      const segment = (drag.current ?? segmentsRef.current)[drag.index];
      if (segment) seekPreview(segment[drag.edge], true);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
    };
  }, [finishPreviewChange, previewSegments, project, seekPreview, timelineScale]);

  const timelineTime = (event: ReactPointerEvent<HTMLElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.max(0, Math.min(duration, (event.clientX - bounds.left) / Math.max(1, bounds.width) * duration));
  };

  const scrub = (event: ReactPointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest(".timeline-segment")) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    seekScrub(timelineTime(event), true);
    const target = event.currentTarget;
    const move = (native: PointerEvent) => {
      const bounds = target.getBoundingClientRect();
      const time = Math.max(0, Math.min(duration, (native.clientX - bounds.left) / Math.max(1, bounds.width) * duration));
      seekScrub(time);
    };
    const finish = () => {
      seekScrub(scrubSeekRef.current.pendingTime, true);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", finish);
      target.removeEventListener("pointercancel", finish);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", finish);
    target.addEventListener("pointercancel", finish);
  };

  const evidence = studio.project?.evidence;
  return (
    <section className="timeline-section">
      <div className="timeline-toolbar">
        <div className="timeline-summary">
          <span className="eyebrow">SOURCE TIMELINE</span>
          <strong>{studio.segments.length} 个片段</strong>
          <span className="muted">成片约 {formatTime(outputDuration(studio.segments), false)}</span>
        </div>
        <div className="timeline-tools">
          <Button size="sm" variant="outline" onClick={studio.addSegment}><Plus />新增片段</Button>
          <Button size="sm" variant="outline" onClick={() => commitZoom(Math.max(1.5, (containerWidth - GUTTER) / Math.max(1, duration)))}><Maximize2 />适应窗口</Button>
          <label className="zoom-control">缩放<Slider aria-label="时间轴缩放" min={1.5} max={120} step={0.5} value={[pixelsPerSecond]} onValueChange={([value]) => scheduleZoom(value)} onValueCommit={([value]) => commitZoom(value)} /></label>
        </div>
      </div>
      <div className="timeline-scroll" ref={scrollRef}>
        <div className="timeline-canvas" style={{ width: timelineWidth + GUTTER }}>
          <div className="ruler" style={{ width: timelineWidth }} onPointerDown={scrub}>
            <RulerTicks duration={duration} major={rulerMajor} />
          </div>
          <div className="track-label">保留片段</div>
          <div className="segment-track" style={{ width: timelineWidth }} onPointerDown={scrub}>
            {studio.segments.map((segment, index) => (
              <div
                key={segment.id ?? `${segment.start}-${index}`}
                className={`timeline-segment${segment.review_required === "yes" ? " review" : ""}${index === studio.selectedIndex ? " selected" : ""}`}
                style={{ left: timelinePercent(segment.start, duration), width: `max(3px, ${timelinePercent(segment.end - segment.start, duration)})` }}
                onClick={(event) => {
                  event.stopPropagation();
                  studio.selectSegment(index, true);
                }}
              >
                <button className="trim-handle start" type="button" aria-label={`调整 R${index + 1} 开始`} onPointerDown={(event) => beginBoundaryDrag(event, index, "start")} />
                <span className="segment-label">R{String(index + 1).padStart(2, "0")} · {(segment.end - segment.start).toFixed(1)}s</span>
                <button className="trim-handle end" type="button" aria-label={`调整 R${index + 1} 结束`} onPointerDown={(event) => beginBoundaryDrag(event, index, "end")} />
              </div>
            ))}
          </div>
          <div className="evidence-label"><span>模型</span><span>球路</span><span>站位</span><span>事件</span></div>
          <div className="evidence-track" style={{ width: timelineWidth }} onPointerDown={scrub}>
            <EvidenceItems evidence={evidence} duration={duration} />
          </div>
          <div className="playhead" ref={playheadRef}><span /></div>
        </div>
      </div>
      <div className="timeline-legend">
        <span><i className="legend-kept" />最终保留</span><span><i className="legend-model" />模型/动作</span><span><i className="legend-flight" />球路/遮挡</span><span><i className="legend-ready" />双方预备</span>
        <span className="timeline-hint">拖动空白处精确定位；拖动片段边缘调整边界</span>
      </div>
    </section>
  );
}
