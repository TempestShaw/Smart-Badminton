"use client";

import { useTranslation } from "react-i18next";
import { t } from "@/lib/i18n";

import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";
import { ArrowLeft, Crosshair, Cpu, Play, RotateCcw, Square, Undo2 } from "lucide-react";

import { Brand } from "@/components/brand";
import { LanguageSwitcher } from "@/components/language-controls";
import { LiveCourt } from "@/components/live-court";
import { PlayerCards, RallyPanel, Segmented } from "@/components/live-studio";
import { ThemeToggle } from "@/components/theme-controls";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { drawOverlay, OnDeviceEngine } from "@/lib/ondevice/engine";
import { validCorners, type Vec2 } from "@/lib/ondevice/court";
import type { LiveAnalysis, MatchFormat } from "@/types/live";

const CORNERS_KEY = "smart-badminton:ondevice-corners:v1";
const CORNER_LABELS = ["近左", "近右", "远右", "远左"];
type Phase = "setup" | "calibrate" | "running";

function describe(failure: unknown): string {
  const message = failure instanceof Error ? failure.message : String(failure);
  if (message === "model-missing") {
    return t("这个网站还没有部署本机分析模型：运行 smart-badminton export-ondevice-models 后重新构建，或设置 ONDEVICE_MODELS_URL。");
  }
  if (/NotAllowedError|Permission/i.test(message)) return t("没有摄像头或麦克风权限，请在浏览器设置里允许后重试。");
  return message;
}

function savedCorners(): Vec2[] | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(CORNERS_KEY) ?? "null") as Vec2[] | null;
    return value && validCorners(value) ? value : null;
  } catch {
    return null;
  }
}

/** Live analysis entirely on this phone or tablet: no computer, nothing uploaded. */
export function OnDeviceLive({ onBack }: { onBack?: () => void }) {
  useTranslation();
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const engine = useRef<OnDeviceEngine | null>(null);
  const [phase, setPhase] = useState<Phase>("setup");
  const [format, setFormat] = useState<MatchFormat>("singles");
  const [corners, setCorners] = useState<Vec2[]>([]);
  const [analysis, setAnalysis] = useState<LiveAnalysis | null>(null);
  const [stats, setStats] = useState<OnDeviceEngine["stats"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const stopAll = useCallback(() => {
    engine.current?.stop();
    engine.current = null;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
  }, []);
  useEffect(() => stopAll, [stopAll]);

  const openCamera = async () => {
    setError(null);
    setBusy(true);
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error(t("摄像头需要 HTTPS 页面。"));
      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const element = video.current!;
      element.srcObject = stream.current;
      await element.play();
      const saved = savedCorners();
      if (saved) {
        setCorners(saved);
        await startEngine(saved);
      } else {
        setCorners([]);
        setPhase("calibrate");
      }
    } catch (failure) {
      setError(describe(failure));
      stopAll();
      setPhase("setup");
    } finally {
      setBusy(false);
    }
  };

  const startEngine = async (points: Vec2[]) => {
    const element = video.current!;
    const next = new OnDeviceEngine({
      video: element,
      corners: points,
      matchFormat: format,
      onUpdate: () => {
        if (overlay.current) drawOverlay(overlay.current, next, element);
      },
    });
    engine.current = next;
    setPhase("running");
    await next.start(stream.current!);
  };

  useEffect(() => {
    if (phase !== "running") return;
    const timer = window.setInterval(() => {
      if (!engine.current) return;
      setAnalysis(engine.current.snapshot());
      setStats({ ...engine.current.stats });
    }, 400);
    return () => window.clearInterval(timer);
  }, [phase]);

  const tap = (event: MouseEvent<HTMLDivElement>) => {
    if (phase !== "calibrate" || corners.length >= 4 || !video.current) return;
    const element = video.current;
    const bounds = element.getBoundingClientRect();
    const scale = Math.min(bounds.width / element.videoWidth, bounds.height / element.videoHeight);
    const x = (event.clientX - bounds.left - (bounds.width - element.videoWidth * scale) / 2) / (element.videoWidth * scale);
    const y = (event.clientY - bounds.top - (bounds.height - element.videoHeight * scale) / 2) / (element.videoHeight * scale);
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) setCorners((current) => [...current, [x, y]]);
  };

  const saveCorners = async () => {
    if (!validCorners(corners)) {
      setError(t("四个角的顺序不对：近左 → 近右 → 远右 → 远左"));
      return;
    }
    window.localStorage.setItem(CORNERS_KEY, JSON.stringify(corners));
    setError(null);
    setBusy(true);
    try {
      await startEngine(corners);
    } catch (failure) {
      setError(describe(failure));
      stopAll();
      setPhase("setup");
    } finally {
      setBusy(false);
    }
  };

  const stop = () => {
    stopAll();
    setPhase("setup");
  };

  const dotStyle = (point: Vec2) => {
    const element = video.current;
    if (!element?.videoWidth) return {};
    const bounds = element.getBoundingClientRect();
    const scale = Math.min(bounds.width / element.videoWidth, bounds.height / element.videoHeight);
    return {
      left: (bounds.width - element.videoWidth * scale) / 2 + point[0] * element.videoWidth * scale,
      top: (bounds.height - element.videoHeight * scale) / 2 + point[1] * element.videoHeight * scale,
    };
  };

  return (
    <div className="live-app">
      <header className="topbar live-topbar">
        <Brand />
        <div className="project-title">
          <span className={`status-dot${phase === "running" ? " live" : ""}`} />
          <span>{t("本机实时分析")}</span>
        </div>
        <div className="top-actions">
          <LanguageSwitcher /><ThemeToggle />
          {onBack ? <Button size="sm" variant="ghost" onClick={() => { stopAll(); onBack(); }}><ArrowLeft />{t("返回")}</Button> : null}
          {phase !== "setup" ? <Button size="sm" variant="destructive" onClick={stop}><Square />{t("结束")}</Button> : null}
        </div>
      </header>
      <main className="live-content">
        {error ? <div className="live-alert" role="alert"><span>{error}</span><button onClick={() => setError(null)}>{t("知道了")}</button></div> : null}
        {phase === "setup" ? (
          <section className="live-card live-setup">
            <div className="live-card-heading">
              <div>
                <p className="eyebrow">{t("本机实时分析")}</p>
                <h2>{t("只用这部手机，边打边看")}</h2>
              </div>
            </div>
            <p className="live-hint">{t("球员跟踪和击球声识别都在这台设备上运行，画面和声音不会上传。把手机横放在底线后方高处，让整片场地入镜。")}</p>
            <div className="live-field">
              <span>{t("比赛类型")}</span>
              <Segmented label={t("比赛类型")} value={format} onChange={setFormat} options={[{ value: "singles", label: t("单打") }, { value: "doubles", label: t("双打") }]} />
            </div>
            <Button size="lg" className="live-start" disabled={busy} onClick={() => void openCamera()}><Play />{busy ? t("正在打开摄像头…") : t("打开摄像头并开始")}</Button>
            {savedCorners() ? <Button size="sm" variant="ghost" onClick={() => { window.localStorage.removeItem(CORNERS_KEY); }}><RotateCcw />{t("下次重新校准场地")}</Button> : null}
          </section>
        ) : null}
        <div className="live-dashboard" hidden={phase === "setup"}>
          <div className="live-main">
            {phase === "calibrate" ? (
              <div className="live-card-heading compact"><h2>{t("依次点击单打场地的四个角")}</h2><small>{corners.length < 4 ? `${t("点击：")}${t(CORNER_LABELS[corners.length])}` : t("可以开始了")}</small></div>
            ) : null}
            <section className="live-stage ondevice-stage" onClick={tap}>
              <video ref={video} playsInline muted />
              <canvas ref={overlay} aria-hidden="true" />
              {phase === "calibrate" ? corners.map((point, index) => (
                <span key={index} className="live-corner-dot" style={dotStyle(point)}>{index + 1}</span>
              )) : null}
            </section>
            {phase === "calibrate" ? (
              <div className="live-calibration-actions">
                <Button size="sm" variant="outline" disabled={!corners.length} onClick={() => setCorners((current) => current.slice(0, -1))}><Undo2 />{t("撤销")}</Button>
                <Button size="sm" disabled={corners.length !== 4 || busy} onClick={() => void saveCorners()}><Crosshair />{t("保存校准")}</Button>
              </div>
            ) : null}
            {phase === "running" && stats ? (
              <div className="live-stage-meta">
                <Badge variant="outline"><Cpu />{stats.provider === "webgpu" ? "WebGPU" : `WASM ×${stats.threads}`}{stats.model.includes("int8") ? " · int8" : ""}</Badge>
                <Badge variant="outline">{t("{{fps}} 帧/秒分析", { fps: stats.fps.toFixed(1) })}</Badge>
                <Badge variant="outline">{t("推理 {{ms}} ms", { ms: stats.inferenceMs.toFixed(0) })}</Badge>
                <Badge variant="outline" title={`ROI ${stats.roiFrames} · scan ${stats.scanFrames}`}>{stats.mode === "roi" ? t("跟踪 {{count}} 人", { count: stats.crops }) : t("全场搜索")}</Badge>
                {analysis?.audio_enabled ? <Badge variant="outline">{t("击球声")}</Badge> : null}
              </div>
            ) : null}
          </div>
          {phase === "running" ? (
            <aside className="live-side">
              <RallyPanel status={{ analysis: analysis ?? undefined }} />
              {analysis ? <section className="live-card live-court-card"><LiveCourt players={analysis.players} label={t("球员站位")} /></section> : null}
              <section className="live-card">
                <div className="live-card-heading compact"><h2>{t("球员")}</h2><small>{format === "doubles" ? t("双打") : t("单打")}</small></div>
                <PlayerCards players={analysis?.players ?? []} metric />
              </section>
            </aside>
          ) : null}
        </div>
      </main>
    </div>
  );
}
