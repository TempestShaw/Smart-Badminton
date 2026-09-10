"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Download, FileVideo2, Gauge, Globe2, Pause, Play, Plus, Save, Scissors, Square, Trash2, Upload } from "lucide-react";

import { Brand } from "@/components/brand";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { analyzeVideoInBrowser } from "@/lib/browser-analysis";
import { exportVideoInBrowser } from "@/lib/browser-export";
import { downloadProject, loadBrowserProject, saveBrowserProject, type BrowserProject } from "@/lib/browser-project";
import { formatTime } from "@/lib/time";
import type { Segment } from "@/types/studio";

type TaskState = "idle" | "analyzing" | "exporting";

interface VideoInfo {
  duration: number;
  width: number;
  height: number;
}

function normalizedSegments(segments: Segment[]): Segment[] {
  return segments
    .filter((segment) => segment.end > segment.start)
    .sort((a, b) => a.start - b.start)
    .map((segment, index) => ({ ...segment, id: `Q${String(index + 1).padStart(3, "0")}` }));
}

export function BrowserQuickStudio({ onSwitchNative }: { onSwitchNative: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const taskRef = useRef<AbortController | null>(null);
  const sourceUrlRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [task, setTask] = useState<TaskState>("idle");
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("选择一段比赛视频");
  const [error, setError] = useState("");
  const [dirty, setDirty] = useState(false);

  useEffect(() => () => {
    taskRef.current?.abort();
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
  }, []);

  const selected = segments[selectedIndex] ?? null;
  const keptDuration = useMemo(() => segments.reduce((sum, segment) => sum + segment.end - segment.start, 0), [segments]);

  const updateSegments = useCallback((next: Segment[], nextSelected = selectedIndex) => {
    const normalized = normalizedSegments(next);
    setSegments(normalized);
    setSelectedIndex(Math.max(0, Math.min(normalized.length - 1, nextSelected)));
    setDirty(true);
  }, [selectedIndex]);

  const chooseFile = (nextFile: File) => {
    taskRef.current?.abort();
    if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
    const url = URL.createObjectURL(nextFile);
    sourceUrlRef.current = url;
    setFile(nextFile);
    setSourceUrl(url);
    setVideoInfo(null);
    setCurrentTime(0);
    setPlaying(false);
    setError("");
    const saved = loadBrowserProject(nextFile);
    setSegments(saved?.segments ?? []);
    setSelectedIndex(0);
    setDirty(false);
    setStatus(saved ? "已恢复本地时间轴" : "可以开始快速分析");
  };

  const project = useCallback((): BrowserProject | null => {
    if (!file || !videoInfo) return null;
    return {
      schemaVersion: 1,
      engine: "browser-quick",
      video: {
        name: file.name,
        size: file.size,
        lastModified: file.lastModified,
        duration: videoInfo.duration,
        width: videoInfo.width,
        height: videoInfo.height,
      },
      segments,
      updatedAt: new Date().toISOString(),
    };
  }, [file, segments, videoInfo]);

  const save = () => {
    const payload = project();
    if (!payload) return;
    saveBrowserProject(payload);
    setDirty(false);
    setStatus("时间轴已保存到本机");
  };

  const analyze = async () => {
    if (!file) return;
    taskRef.current?.abort();
    const controller = new AbortController();
    taskRef.current = controller;
    setTask("analyzing");
    setError("");
    setProgress(0);
    try {
      const result = await analyzeVideoInBrowser(file, ({ progress: value, message }) => {
        setProgress(value);
        setStatus(message);
      }, controller.signal);
      setVideoInfo({ duration: result.duration, width: result.width, height: result.height });
      updateSegments(result.segments, 0);
      setStatus(`找到 ${result.segments.length} 个候选回合`);
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") setError((caught as Error).message);
      else setStatus("分析已取消");
    } finally {
      setTask("idle");
      taskRef.current = null;
    }
  };

  const exportVideo = async () => {
    if (!file || !segments.length) return;
    taskRef.current?.abort();
    const controller = new AbortController();
    taskRef.current = controller;
    setTask("exporting");
    setError("");
    setProgress(0);
    try {
      await exportVideoInBrowser(file, segments, (value, message) => {
        setProgress(value);
        setStatus(message);
      }, controller.signal);
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") setError((caught as Error).message);
      else setStatus("输出已取消");
    } finally {
      setTask("idle");
      taskRef.current = null;
    }
  };

  const seekTo = (time: number) => {
    const video = videoRef.current;
    if (!video || !videoInfo) return;
    video.currentTime = Math.max(0, Math.min(videoInfo.duration, time));
  };

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play(); else video.pause();
  };

  const editBoundary = (index: number, edge: "start" | "end", value: number) => {
    if (!videoInfo) return;
    const next = segments.map((segment) => ({ ...segment }));
    const previousEnd = index > 0 ? next[index - 1].end : 0;
    const nextStart = index < next.length - 1 ? next[index + 1].start : videoInfo.duration;
    if (edge === "start") next[index].start = Math.max(previousEnd, Math.min(value, next[index].end - 0.1));
    else next[index].end = Math.min(nextStart, Math.max(value, next[index].start + 0.1));
    updateSegments(next, index);
  };

  const dragBoundary = (event: ReactPointerEvent, index: number, edge: "start" | "end") => {
    if (!videoInfo) return;
    event.preventDefault();
    event.stopPropagation();
    const track = event.currentTarget.closest<HTMLElement>(".quick-ruler");
    if (!track) return;
    const move = (pointer: PointerEvent) => {
      const bounds = track.getBoundingClientRect();
      editBoundary(index, edge, (pointer.clientX - bounds.left) / bounds.width * videoInfo.duration);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  };

  const addSegment = () => {
    if (!videoInfo) return;
    const start = Math.max(0, Math.min(videoInfo.duration - 0.5, currentTime));
    updateSegments([...segments, { start, end: Math.min(videoInfo.duration, start + 3), confidence: "manual" }], segments.length);
  };

  const splitSegment = () => {
    if (!selected || currentTime <= selected.start + 0.1 || currentTime >= selected.end - 0.1) return;
    const next = [...segments];
    next.splice(selectedIndex, 1, { ...selected, end: currentTime }, { ...selected, start: currentTime });
    updateSegments(next, selectedIndex + 1);
  };

  const deleteSegment = () => {
    if (!selected) return;
    updateSegments(segments.filter((_, index) => index !== selectedIndex), Math.max(0, selectedIndex - 1));
  };

  return (
    <div className="browser-quick-app">
      <aside className="quick-sidebar">
        <Brand />
        <Button className="quick-new-project" onClick={() => inputRef.current?.click()}><Plus />{file ? "更换比赛视频" : "新建剪片"}</Button>
        <div className="quick-library">
          <span className="eyebrow">我的工作台</span>
          {file ? <div className="quick-library-file"><FileVideo2 /><span>{file.name}</span></div> : <div className="quick-library-empty"><FileVideo2 /><span>尚未选择视频</span><small>从一场比赛开始</small></div>}
        </div>
        <footer className="quick-sidebar-footer"><span className="quick-privacy-dot" />视频仅在本机处理<a href="https://github.com/TempestShaw/Smart-Badminton/blob/main/THIRD_PARTY_NOTICES.md">开源与许可 ↗</a></footer>
      </aside>
      <header className="quick-topbar">
        <span className="quick-page-label">比赛剪辑 <span>/ {file ? "编辑工作台" : "新建项目"}</span></span>
        <div className="quick-engine-switch" aria-label="执行引擎">
          <button className="selected"><Globe2 />浏览器版</button>
          <button onClick={onSwitchNative}><Gauge />精准版</button>
        </div>
        <div className="quick-top-actions">
          <Badge variant="outline">本地处理</Badge>
          {file ? <Button size="sm" variant="outline" onClick={() => inputRef.current?.click()}><FileVideo2 />更换视频</Button> : null}
          <Button size="sm" disabled={!dirty || !file} onClick={save}><Save />保存</Button>
          <Button size="sm" variant="secondary" disabled={!segments.length || task !== "idle"} onClick={() => void exportVideo()}><Download />输出 MP4</Button>
        </div>
      </header>

      <input ref={inputRef} className="sr-only" type="file" accept="video/mp4,video/webm,video/quicktime,.mov,.m4v" onChange={(event) => {
        const next = event.target.files?.[0];
        if (next) chooseFile(next);
        event.currentTarget.value = "";
      }} />

      {!file ? (
        <main className="quick-empty">
          <div className="quick-empty-icon"><Scissors /></div>
          <p className="eyebrow">SMART BADMINTON STUDIO</p>
          <h1>留下每一个精彩回合</h1>
          <p>让等待退场，让比赛继续。<br />选择视频，整理回合，轻松完成剪辑。</p>
          <section className="quick-upload-card" aria-labelledby="upload-title">
            <div className="quick-card-heading"><h2 id="upload-title">导入比赛视频</h2><span>01 / 开始</span></div>
            <button className="quick-upload-zone" onClick={() => inputRef.current?.click()}>
              <span className="quick-upload-icon"><Upload /></span>
              <strong>点击选择比赛视频</strong>
              <span>MP4、MOV、M4V 或 WebM</span>
            </button>
            <div className="quick-upload-note"><span className="quick-privacy-dot" />无需上传，视频与分析结果都留在这台电脑。</div>
            <div className="quick-workflow-steps"><span><b>01</b> 选择视频</span><span><b>02</b> 分析与微调</span><span><b>03</b> 导出精彩</span></div>
          </section>
          <button className="quick-native-link" onClick={onSwitchNative}>需要完整球路与比分？ <strong>探索精准版 →</strong></button>
        </main>
      ) : (
        <main className="quick-workspace">
          <section className="quick-viewer-panel">
            <div className="quick-video-frame">
              {sourceUrl ? <video
                ref={videoRef}
                src={sourceUrl}
                playsInline
                onLoadedMetadata={(event) => setVideoInfo({
                  duration: event.currentTarget.duration,
                  width: event.currentTarget.videoWidth,
                  height: event.currentTarget.videoHeight,
                })}
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              /> : null}
            </div>
            <div className="quick-player-controls">
              <Button aria-label={playing ? "暂停" : "播放"} size="icon" variant={playing ? "outline" : "secondary"} onClick={togglePlayback}>{playing ? <Pause /> : <Play />}</Button>
              <strong>{formatTime(currentTime, false)} / {formatTime(videoInfo?.duration ?? 0, false)}</strong>
              <input aria-label="视频播放位置" type="range" min={0} max={videoInfo?.duration ?? 1} step={0.01} value={currentTime} onChange={(event) => seekTo(Number(event.target.value))} />
            </div>
          </section>

          <aside className="quick-inspector">
            <div><p className="eyebrow">LOCAL ANALYSIS</p><h2>{file.name}</h2><p>{videoInfo ? `${videoInfo.width}×${videoInfo.height} · ${formatTime(videoInfo.duration, false)}` : "读取视频…"}</p></div>
            <Button disabled={task !== "idle" || !videoInfo} onClick={() => void analyze()}><Scissors />快速分析</Button>
            {task !== "idle" ? <Button variant="outline" onClick={() => taskRef.current?.abort()}><Square />取消任务</Button> : null}
            <div className="quick-status" aria-live="polite"><span>{status}</span>{task !== "idle" ? <Progress value={progress * 100} /> : null}</div>
            {error ? <p className="quick-error">{error}</p> : null}
            <dl className="quick-stats"><div><dt>片段</dt><dd>{segments.length}</dd></div><div><dt>成片</dt><dd>{formatTime(keptDuration, false)}</dd></div></dl>
            <div className="quick-edit-actions">
              <Button size="sm" variant="outline" onClick={addSegment}><Plus />新增</Button>
              <Button size="sm" variant="outline" disabled={!selected} onClick={splitSegment}><Scissors />分割</Button>
              <Button size="sm" variant="destructive" disabled={!selected} onClick={deleteSegment}><Trash2 />删除</Button>
            </div>
            {selected ? <div className="quick-boundaries">
              <label>开始<input type="number" step="0.01" value={selected.start.toFixed(2)} onChange={(event) => editBoundary(selectedIndex, "start", Number(event.target.value))} /></label>
              <label>结束<input type="number" step="0.01" value={selected.end.toFixed(2)} onChange={(event) => editBoundary(selectedIndex, "end", Number(event.target.value))} /></label>
            </div> : null}
            <Button variant="ghost" disabled={!file || !videoInfo} onClick={() => { const payload = project(); if (payload) downloadProject(payload); }}><Download />导出时间轴</Button>
          </aside>

          <section className="quick-timeline-panel">
            <div className="quick-timeline-heading"><div><span className="eyebrow">SOURCE TIMELINE</span><strong>{segments.length} 个片段</strong></div><span>{formatTime(currentTime, false)}</span></div>
            <div className="quick-timeline-scroll">
              <div className="quick-ruler" onPointerDown={(event) => {
                if (!videoInfo) return;
                const bounds = event.currentTarget.getBoundingClientRect();
                seekTo((event.clientX - bounds.left) / bounds.width * videoInfo.duration);
              }}>
                {segments.map((segment, index) => (
                  <button
                    key={segment.id ?? index}
                    className={`quick-segment ${index === selectedIndex ? "selected" : ""}`}
                    style={{ left: `${segment.start / (videoInfo?.duration ?? 1) * 100}%`, width: `${(segment.end - segment.start) / (videoInfo?.duration ?? 1) * 100}%` }}
                    onClick={(event) => { event.stopPropagation(); setSelectedIndex(index); seekTo(segment.start); }}
                  >
                    <span className="quick-trim start" onPointerDown={(event) => dragBoundary(event, index, "start")} />
                    <span>{segment.id}</span>
                    <span className="quick-trim end" onPointerDown={(event) => dragBoundary(event, index, "end")} />
                  </button>
                ))}
                <span className="quick-playhead" style={{ left: `${currentTime / (videoInfo?.duration ?? 1) * 100}%` }} />
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
