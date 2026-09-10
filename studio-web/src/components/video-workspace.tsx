"use client";

import { useTranslation } from "react-i18next";
import { t, translateMessage } from "@/lib/i18n";

import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, LoaderCircle, Pause, Play, Save, ScanSearch, Scissors, SkipBack, SkipForward, Undo2, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import type { StudioController } from "@/hooks/use-studio-controller";
import { useCalibrationEditor } from "@/hooks/use-calibration-editor";
import { useShuttleAnnotations } from "@/hooks/use-shuttle-annotations";
import { mediaUrl } from "@/lib/api";
import { formatTime, outputDuration } from "@/lib/time";
import type { ToolMode } from "@/components/workflow-panel";

export function VideoWorkspace({
  studio,
  toolMode,
  onToolDirty,
  onCloseTool,
}: {
  studio: StudioController;
  toolMode: ToolMode;
  onToolDirty: (dirty: boolean) => void;
  onCloseTool: () => void;
}) {
  useTranslation();
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cutPreview, setCutPreview] = useState(false);
  const [cutIndex, setCutIndex] = useState(0);
  const [volume, setVolume] = useState(1);
  const [trajectoryMode, setTrajectoryMode] = useState<"off" | "debug" | "trail">("off");
  const [poseOverlay, setPoseOverlay] = useState<{ url: string; start: number; end: number } | null>(null);
  const animationRef = useRef(0);
  const lastUiUpdateRef = useRef(0);
  const poseVideoRef = useRef<HTMLVideoElement>(null);
  const handleCalibrationSaved = useCallback((payload: Parameters<StudioController["applyCalibrationResult"]>[0]) => {
    studio.applyCalibrationResult(payload);
    void studio.refreshAnalyticsAndScore();
  }, [studio.applyCalibrationResult, studio.refreshAnalyticsAndScore]);

  const calibration = useCalibrationEditor({
    active: toolMode === "calibration",
    project: studio.project,
    videoRef: studio.videoRef,
    notify: studio.notify,
    onDirtyChange: onToolDirty,
    onSaved: handleCalibrationSaved,
  });
  const shuttle = useShuttleAnnotations({
    visible: toolMode === "shuttle" || trajectoryMode !== "off",
    editable: toolMode === "shuttle",
    displayMode: toolMode === "shuttle" ? "debug" : trajectoryMode === "debug" ? "debug" : "trail",
    project: studio.project,
    videoRef: studio.videoRef,
    notify: studio.notify,
    onDirtyChange: onToolDirty,
  });

  const stopCutPreview = useCallback((pause = false) => {
    setCutPreview(false);
    if (pause) studio.videoRef.current?.pause();
  }, [studio.videoRef]);

  useEffect(() => {
    if (studio.scoreReview.active) stopCutPreview(true);
  }, [stopCutPreview, studio.scoreReview.active]);

  useEffect(() => {
    setCurrentTime(0);
    setPlaying(false);
    setCutPreview(false);
    setTrajectoryMode("off");
    setPoseOverlay(null);
  }, [studio.project?.id]);

  useEffect(() => {
    if (toolMode !== "none") setPoseOverlay(null);
  }, [toolMode]);

  const syncPoseOverlay = useCallback((sourceTime?: number) => {
    const source = studio.videoRef.current;
    const overlay = poseVideoRef.current;
    if (!poseOverlay || !source || !overlay) return;
    const absoluteTime = sourceTime ?? source.currentTime;
    const localTime = Math.max(0, Math.min(poseOverlay.end - poseOverlay.start, absoluteTime - poseOverlay.start));
    if (Number.isFinite(localTime) && Math.abs(overlay.currentTime - localTime) > 0.09) overlay.currentTime = localTime;
    if (source.paused) {
      overlay.pause();
    } else if (overlay.paused && overlay.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      void overlay.play().catch(() => undefined);
    }
  }, [poseOverlay, studio.videoRef]);

  useEffect(() => {
    const video = studio.videoRef.current;
    if (!video) return;
    const tick = () => {
      const now = performance.now();
      if (now - lastUiUpdateRef.current >= 50) {
        lastUiUpdateRef.current = now;
        setCurrentTime(video.currentTime || 0);
        if (toolMode === "shuttle" || trajectoryMode !== "off") shuttle.draw();
        if (poseOverlay) {
          if (video.currentTime < poseOverlay.start - 0.03 || video.currentTime > poseOverlay.end + 0.03) {
            poseVideoRef.current?.pause();
            setPoseOverlay(null);
          } else {
            syncPoseOverlay(video.currentTime);
          }
        }
      }
      if (cutPreview) {
        const segment = studio.segments[cutIndex];
        if (!segment) {
          stopCutPreview(true);
        } else if (video.currentTime >= segment.end - 0.02) {
          const next = cutIndex + 1;
          if (next >= studio.segments.length) {
            video.pause();
            stopCutPreview(false);
          } else {
            setCutIndex(next);
            video.currentTime = studio.segments[next].start;
          }
        }
      }
      if (studio.scoreReview.active) {
        const segment = studio.selectedSegment;
        if (segment && video.currentTime >= segment.end - 0.02) {
          video.pause();
          video.currentTime = segment.end;
        }
      }
      if (!video.paused) animationRef.current = window.requestAnimationFrame(tick);
    };
    if (playing) animationRef.current = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationRef.current);
  }, [cutIndex, cutPreview, playing, poseOverlay, shuttle.draw, stopCutPreview, studio.scoreReview.active, studio.segments, studio.selectedSegment, studio.videoRef, syncPoseOverlay, toolMode, trajectoryMode]);

  const setTrajectory = (mode: "debug" | "trail") => {
    const analysis = studio.project?.shuttle_analysis;
    if (!analysis?.current) {
      studio.notify(t("请先分析整段视觉"), true);
      return;
    }
    setTrajectoryMode((current) => current === mode ? "off" : mode);
  };

  const togglePoseOverlay = async () => {
    if (poseOverlay) {
      poseVideoRef.current?.pause();
      setPoseOverlay(null);
      return;
    }
    const project = studio.project;
    const analysis = project?.pose_analysis;
    if (!project) return;
    if (!analysis?.current || !analysis.url) {
      studio.notify(t("请先分析整段视觉"), true);
      return;
    }
    setPoseOverlay({ url: mediaUrl(analysis.url), start: 0, end: project.video.duration });
  };

  const togglePlayback = async () => {
    const video = studio.videoRef.current;
    if (!video) return;
    if (studio.scoreReview.active && studio.selectedSegment) {
      const segment = studio.selectedSegment;
      if (video.currentTime < segment.start || video.currentTime >= segment.end - 0.02) video.currentTime = segment.start;
    }
    if (video.paused) await video.play(); else video.pause();
  };

  const toggleCutPreview = async () => {
    const video = studio.videoRef.current;
    if (!video || !studio.segments.length) return;
    if (cutPreview) return stopCutPreview(true);
    let index = studio.selectedIndex >= 0 ? studio.selectedIndex : studio.segments.findIndex((segment) => segment.end > video.currentTime);
    if (index < 0) index = 0;
    setCutIndex(index);
    setCutPreview(true);
    video.currentTime = Math.max(studio.segments[index].start, Math.min(video.currentTime, studio.segments[index].end));
    if (video.currentTime >= studio.segments[index].end - 0.04) video.currentTime = studio.segments[index].start;
    await video.play();
  };

  const jumpBoundary = (direction: -1 | 1) => {
    const video = studio.videoRef.current;
    if (!video) return;
    if (studio.scoreReview.active) {
      studio.moveScoreReview(direction);
      return;
    }
    if (cutPreview) {
      const next = Math.max(0, Math.min(studio.segments.length - 1, cutIndex + direction));
      setCutIndex(next);
      video.currentTime = studio.segments[next].start;
      return;
    }
    const boundaries = studio.segments.flatMap((segment) => [segment.start, segment.end]).sort((a, b) => a - b);
    const target = direction < 0
      ? boundaries.toReversed().find((value) => value < video.currentTime - 0.01)
      : boundaries.find((value) => value > video.currentTime + 0.01);
    if (target !== undefined) video.currentTime = target;
  };

  const activeOutputTime = studio.segments.slice(0, cutIndex).reduce((sum, segment) => sum + segment.end - segment.start, 0)
    + Math.max(0, currentTime - (studio.segments[cutIndex]?.start ?? currentTime));

  return (
    <section className="viewer-column">
      {toolMode === "none" ? (
        <div data-overlay-label={t("叠加")} className="viewer-analysis-toolbar" aria-label={t("视频分析叠加")}>
          <Button
            size="sm"
            variant={poseOverlay ? "default" : "secondary"}
            disabled={!studio.project?.pose_analysis.current}
            title={studio.project?.pose_analysis.current ? t("显示整段视频的人物框、骨架和手腕轨迹") : t("先在播放器外运行整段视觉分析")}
            onClick={() => void togglePoseOverlay()}
          >
            <Activity />{poseOverlay ? t("关闭姿态") : t("YOLO 姿态")}
          </Button>
          <Button
            size="sm"
            variant={trajectoryMode === "debug" ? "default" : "secondary"}
            disabled={!studio.project?.shuttle_analysis.current}
            title={studio.project?.shuttle_analysis.current ? t("显示轨迹调试标记") : t("球路未生成")}
            onClick={() => setTrajectory("debug")}
          >
            {studio.analysisStatus.state === "running" && studio.analysisStatus.mode === "shuttle" ? <LoaderCircle className="animate-spin" /> : <ScanSearch />}
            {studio.analysisStatus.state === "running" && studio.analysisStatus.mode === "shuttle" ? t("分析球路…") : trajectoryMode === "debug" ? t("关闭调试") : t("调试标记")}
          </Button>
          <Button
            size="sm"
            variant={trajectoryMode === "trail" ? "default" : "secondary"}
            disabled={!studio.project?.shuttle_analysis.current}
            title={studio.project?.shuttle_analysis.current ? t("只显示橙色球路拖尾") : t("球路未生成")}
            onClick={() => setTrajectory("trail")}
          >
            <ScanSearch />{trajectoryMode === "trail" ? t("关闭拖尾") : t("纯轨迹")}
          </Button>
        </div>
      ) : null}
      <div className="viewer-frame">
        <video
          ref={studio.videoRef}
          src={studio.mediaSource ?? undefined}
          width={studio.project?.preview.width}
          height={studio.project?.preview.height}
          preload="metadata"
          playsInline
          onPlay={() => { setPlaying(true); syncPoseOverlay(); }}
          onPause={() => { setPlaying(false); poseVideoRef.current?.pause(); }}
          onLoadedMetadata={(event) => setCurrentTime(event.currentTarget.currentTime)}
          onSeeked={(event) => { setCurrentTime(event.currentTarget.currentTime); syncPoseOverlay(event.currentTarget.currentTime); }}
        />
        {poseOverlay ? (
          <video
            ref={poseVideoRef}
            className="pose-overlay-video"
            src={poseOverlay.url}
            muted
            playsInline
            aria-label={t("整段视频的 YOLO 人物姿态叠加")}
            onLoadedMetadata={() => syncPoseOverlay()}
          />
        ) : null}
        {toolMode === "calibration" ? (
          <canvas
            ref={calibration.canvasRef}
            className="tool-canvas calibration-canvas"
            aria-label={t("球场区域校准画布")}
            onPointerDown={calibration.pointerDown}
            onPointerMove={calibration.pointerMove}
            onPointerUp={calibration.pointerUp}
            onPointerCancel={calibration.pointerUp}
          />
        ) : null}
        {toolMode === "shuttle" || trajectoryMode !== "off" ? (
          <canvas
            ref={shuttle.canvasRef}
            className={`tool-canvas shuttle-canvas${toolMode === "shuttle" ? "" : " passive"}`}
            aria-label={toolMode === "shuttle" ? t("羽球轨迹标注画布") : t("羽球运动轨迹叠加")}
            onPointerDown={toolMode === "shuttle" ? shuttle.add : undefined}
          />
        ) : null}
        {toolMode !== "none" ? <div className={`tool-mode-badge ${toolMode}`}>{toolMode === "calibration" ? "COURT CALIBRATION" : "SHUTTLE LABEL"}</div> : null}
        {studio.scoreReview.active ? (
          <div className="cut-preview-status"><span>{t("比分复核")}</span><em>R{String(studio.selectedIndex + 1).padStart(2, "0")}</em></div>
        ) : cutPreview ? (
          <div className="cut-preview-status"><span>{t("成片预览")}</span><strong>{formatTime(activeOutputTime)} / {formatTime(outputDuration(studio.segments))}</strong><em>R{String(cutIndex + 1).padStart(2, "0")}</em></div>
        ) : null}
      </div>

      {toolMode === "calibration" ? (
        <CalibrationPanel studio={studio} calibration={calibration} onClose={onCloseTool} />
      ) : null}
      {toolMode === "shuttle" ? (
        <ShuttlePanel studio={studio} shuttle={shuttle} onClose={onCloseTool} />
      ) : null}

      <div className="transport">
        <Button size="icon" variant="outline" aria-label={t("上一个边界")} onClick={() => jumpBoundary(-1)}><SkipBack /></Button>
        <Button size="icon-lg" aria-label={t("播放或暂停")} onClick={() => void togglePlayback()}>{playing ? <Pause /> : <Play />}</Button>
        <Button size="icon" variant="outline" aria-label={t("下一个边界")} onClick={() => jumpBoundary(1)}><SkipForward /></Button>
        <Button variant={cutPreview ? "secondary" : "outline"} disabled={!studio.segments.length || studio.scoreReview.active} onClick={() => void toggleCutPreview()}><Scissors />{cutPreview ? t("停止成片预览") : t("成片预览")}</Button>
        <span className="transport-time">{formatTime(currentTime)} / {formatTime(studio.project?.video.duration ?? 0)}</span>
        <div className="transport-spacer" />
        <Label className="volume-label">{t("音量")}<Slider aria-label={t("音量")} min={0} max={1} step={0.05} value={[volume]} onValueChange={([value]) => {
          setVolume(value);
          if (studio.videoRef.current) studio.videoRef.current.volume = value;
        }} /></Label>
      </div>
    </section>
  );
}

function CalibrationPanel({
  studio,
  calibration,
  onClose,
}: {
  studio: StudioController;
  calibration: ReturnType<typeof useCalibrationEditor>;
  onClose: () => void;
}) {
  useTranslation();
  const [replaceOpen, setReplaceOpen] = useState(false);
  const [predictionOpen, setPredictionOpen] = useState(false);
  const payload = calibration.payload;
  const required = payload?.regions.filter((region) => region.required) ?? [];
  const complete = required.filter((region) => (region.points?.length ?? 0) >= (region.minimum_points ?? 3)).length;
  const corners = payload?.regions.find((region) => region.id === "court_corners");
  const homography = payload?.homography;
  const helperIds = new Set(["near_player_zone", "far_player_zone", "net_band", "shuttle_airspace_polygon", "shuttle_perspective_axis"]);
  const hasGeneratedHelpers = Boolean(payload?.regions.some((region) => helperIds.has(region.id) && (region.points?.length ?? 0) > 0));
  const analysisRunning = studio.analysisStatus.state === "running";
  const visualConfigured = Boolean(studio.project?.pose_analysis.configured || studio.project?.shuttle_analysis.configured);
  const rerunPrediction = () => {
    const settings = studio.project?.automatic_analysis.settings;
    if (!settings) return;
    setPredictionOpen(false);
    void studio.launchAnalysis(false, {
      preroll: settings.preroll,
      postroll: settings.postroll,
      suppress_handoffs: settings.suppress_handoffs,
    });
  };
  return (
    <section className="tool-panel calibration-panel">
      <div className="tool-panel-copy">
        <span className="eyebrow">COURT CALIBRATION</span>
        <strong>{calibration.loading ? t("正在载入…") : t("点击添加顶点，拖动圆点微调")}</strong>
        <span>{t("必需区域")}{complete} / {required.length}{calibration.dirty ? t(" · 未保存") : ""}</span>
        <span>{homography?.available ? t("单应性 95% 约 ±{{value1}} m · {{value2}}", { value1: Number(homography.p95_uncertainty_meters).toFixed(2), value2: homography.score_safe ? t("可用于高置信事件") : t("仅供参考") }) : t("单应性 {{value1}} / 4 个有序角点", { value1: corners?.points?.length ?? 0 })}</span>
      </div>
      <Label>{t("当前区域")}<Select value={calibration.selectedId} onValueChange={calibration.setSelectedId}>
          <SelectTrigger aria-label={t("当前校准区域")}><SelectValue placeholder={t("选择一个校准区域")} /></SelectTrigger>
          <SelectContent>
          {(payload?.regions ?? []).map((region) => {
            const minimum = region.minimum_points ?? 3;
            const count = region.points?.length ?? 0;
            return <SelectItem key={region.id} value={region.id}>{translateMessage(region.label)} · {count >= minimum ? t("✓ {{value1}} 点", { value1: count }) : t("{{value1}} 点", { value1: count })}</SelectItem>;
          })}
          </SelectContent>
        </Select>
      </Label>
      <div className="tool-panel-actions">
        <Button variant="secondary" onClick={() => { if (hasGeneratedHelpers) setReplaceOpen(true); else calibration.autoGenerate(false); }}>{t("从有效场地生成辅助区")}</Button>
        <Button variant="outline" onClick={() => calibration.addRegion("background_court_polygons")}>{t("＋ 背景排除区")}</Button>
        <Button variant="outline" onClick={() => calibration.addRegion("static_false_positive_polygons")}>{t("＋ 静态误检区")}</Button>
        <Button variant="outline" disabled={!calibration.selected?.points?.length} onClick={calibration.undoPoint}><Undo2 />{t("撤销顶点")}</Button>
        <Button variant="destructive" disabled={!calibration.selected?.points?.length} onClick={calibration.clearRegion}>{t("清空当前")}</Button>
        <Button variant="destructive" disabled={!calibration.selected?.type} onClick={calibration.deleteRegion}>{t("删除排除区")}</Button>
        <Button disabled={!calibration.dirty || calibration.saving} onClick={() => void calibration.save()}><Save />{calibration.saving ? t("正在保存…") : t("保存校准")}</Button>
        <Button variant="secondary" disabled={calibration.dirty || calibration.saving || analysisRunning || !studio.project?.automatic_analysis.configured} onClick={() => setPredictionOpen(true)}><ScanSearch />{t("重新跑剪片预测")}</Button>
        <Button variant="outline" disabled={calibration.dirty || calibration.saving || analysisRunning || !visualConfigured} onClick={() => void studio.launchVisualAnalysis(false)}><Activity />{t("按新校准重跑整段视觉")}</Button>
        <Button variant="ghost" onClick={onClose}><X />{t("退出")}</Button>
      </div>
      <AlertDialog open={replaceOpen} onOpenChange={setReplaceOpen}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>{t("重新生成所有辅助区？")}</AlertDialogTitle><AlertDialogDescription>{t("当前辅助区草稿会被覆盖。")}</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>{t("保留当前辅助区")}</AlertDialogCancel><AlertDialogAction onClick={() => { calibration.autoGenerate(true); setReplaceOpen(false); }}>{t("覆盖并生成")}</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog open={predictionOpen} onOpenChange={setPredictionOpen}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>{t("按新校准重新跑剪片预测？")}</AlertDialogTitle><AlertDialogDescription>{t("将备份并替换当前时间轴。")}</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>{t("取消")}</AlertDialogCancel><AlertDialogAction onClick={rerunPrediction}>{t("备份并重新预测")}</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}

function ShuttlePanel({
  studio,
  shuttle,
  onClose,
}: {
  studio: StudioController;
  shuttle: ReturnType<typeof useShuttleAnnotations>;
  onClose: () => void;
}) {
  useTranslation();
  const analysis = studio.project?.shuttle_analysis;
  const running = studio.analysisStatus.state === "running" && studio.analysisStatus.mode === "shuttle";
  const saveAndRebuild = async () => {
    if (!(await shuttle.save())) return;
    if (analysis?.configured) await studio.launchShuttleAnalysis(false);
  };
  return (
    <section className="tool-panel shuttle-panel">
      <div className="tool-panel-copy">
        <span className="eyebrow">{t("羽球轨迹")}</span>
        <strong>{shuttle.loading ? t("正在载入…") : analysis?.stale ? t("需要重新分析") : analysis?.generated ? t("轨迹已生成") : t("尚未生成")}</strong>
        <span>{running ? t("分析中") : shuttle.dirty ? t("未保存") : t("已保存")}</span>
      </div>
      <Label>{t("点击动作")}<Select value={shuttle.mode} onValueChange={(value) => shuttle.setMode(value as "add" | "reject")}><SelectTrigger aria-label={t("羽球标注动作")}><SelectValue /></SelectTrigger><SelectContent><SelectItem value="add">{t("补一个羽球点")}</SelectItem><SelectItem value="reject">{t("排除附近误检")}</SelectItem></SelectContent></Select></Label>
      <div className="tool-panel-actions">
        <Button variant="secondary" disabled={!analysis?.configured || studio.analysisStatus.state === "running"} onClick={() => void studio.launchShuttleAnalysis(Boolean(analysis?.generated))}><ScanSearch />{running ? t("正在分析…") : analysis?.stale ? t("按新校准重新分析") : analysis?.generated ? t("重新分析当前球路") : t("生成当前视频球路")}</Button>
        <Button variant="outline" disabled={!shuttle.payload?.annotations.length} onClick={shuttle.undo}><Undo2 />{t("撤销标注")}</Button>
        <Button disabled={!shuttle.dirty || shuttle.saving || studio.analysisStatus.state === "running"} onClick={() => void saveAndRebuild()}><Save />{shuttle.saving ? t("正在保存…") : analysis?.configured ? t("保存并重建球路") : t("保存标注")}</Button>
        <Button variant="ghost" onClick={onClose}><X />{t("退出")}</Button>
      </div>
    </section>
  );
}
