"use client";

import { useTranslation } from "react-i18next";
import { t, translateMessage } from "@/lib/i18n";

import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { ArrowLeft, Camera, Crosshair, Link2, Play, Radio, RotateCcw, Smartphone, Square, Undo2, Video } from "lucide-react";

import { Brand } from "@/components/brand";
import { CourtLandmarkDiagram, useCourtPoints } from "@/components/court-point-picker";
import { LanguageSwitcher } from "@/components/language-controls";
import { LiveCourt } from "@/components/live-court";
import { ThemeToggle } from "@/components/theme-controls";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { RUNNING_STATES, useLiveSession, type LiveController } from "@/hooks/use-live-session";
import { apiUrl } from "@/lib/api";
import type { LivePlayer, LiveStatus, MatchFormat } from "@/types/live";
import type { Point } from "@/types/studio";

type SourceKind = "camera" | "browser" | "url";

function playerName(player: Pick<LivePlayer, "player_id">): string {
  const [side, slot] = player.player_id.split("-");
  return `${side === "near" ? t("近场") : t("远场")} ${slot}`;
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: string; icon?: ReactNode; disabled?: boolean }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="live-segmented" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          className={value === option.value ? "selected" : undefined}
          disabled={option.disabled}
          onClick={() => onChange(option.value)}
        >
          {option.icon}{option.label}
        </button>
      ))}
    </div>
  );
}

function LiveSetup({ live, status }: { live: LiveController; status: LiveStatus | null }) {
  const [kind, setKind] = useState<SourceKind>("camera");
  const [camera, setCamera] = useState("0");
  const [url, setUrl] = useState("");
  const [format, setFormat] = useState<MatchFormat>("singles");
  const [shuttle, setShuttle] = useState(true);
  const [record, setRecord] = useState(false);
  const [secure, setSecure] = useState(true);
  const shuttleAvailable = status?.models.shuttle ?? false;

  useEffect(() => {
    setSecure(window.isSecureContext);
    if (!["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)) setKind("browser");
  }, []);
  const savedFormat = status?.match_format;
  useEffect(() => { if (savedFormat) setFormat(savedFormat); }, [savedFormat]);

  const source = kind === "camera" ? camera.trim() : kind === "browser" ? "browser" : url.trim();
  const start = () => void live.start({ source, match_format: format, shuttle: shuttle && shuttleAvailable, record });

  return (
    <section className="live-card live-setup" aria-labelledby="live-setup-title">
      <div className="live-card-heading">
        <div>
          <p className="eyebrow">{t("实时分析")}</p>
          <h2 id="live-setup-title">{t("连接摄像头，边打边看")}</h2>
        </div>
      </div>
      <div className="live-field">
        <span>{t("画面来源")}</span>
        <Segmented
          label={t("画面来源")}
          value={kind}
          onChange={setKind}
          options={[
            { value: "camera", label: t("本机摄像头"), icon: <Camera /> },
            { value: "browser", label: t("这台设备的摄像头"), icon: <Smartphone /> },
            { value: "url", label: t("视频流或文件"), icon: <Link2 /> },
          ]}
        />
        {kind === "camera" ? (
          <Input aria-label={t("摄像头编号")} inputMode="numeric" value={camera} onChange={(event) => setCamera(event.target.value)} placeholder="0" />
        ) : null}
        {kind === "url" ? (
          <Input aria-label={t("视频流地址或文件路径")} value={url} onChange={(event) => setUrl(event.target.value)} placeholder="rtsp://… · http://192.168.1.20:8080/video · D:\\match.mp4" />
        ) : null}
        {kind === "browser" && !secure ? <p className="live-hint warning">{t("手机摄像头需要 HTTPS：用 --https 启动 Studio，再在手机上打开 https 地址。")}</p> : null}
        {kind === "browser" && secure ? <p className="live-hint">{t("把手机横放在底线后方高处，让整片场地都在画面里。分析在电脑上运行，手机只负责拍摄。")}</p> : null}
      </div>
      <div className="live-field">
        <span>{t("比赛类型")}</span>
        <Segmented
          label={t("比赛类型")}
          value={format}
          onChange={setFormat}
          options={[{ value: "singles", label: t("单打") }, { value: "doubles", label: t("双打") }]}
        />
      </div>
      <div className="live-options">
        <label><Checkbox checked={shuttle && shuttleAvailable} disabled={!shuttleAvailable} onCheckedChange={(checked) => setShuttle(checked === true)} /><span>{shuttleAvailable ? t("跟踪羽球（判断回合与拍数）") : t("未找到羽球模型")}</span></label>
        <label><Checkbox checked={record} onCheckedChange={(checked) => setRecord(checked === true)} /><span>{t("同时录像，结束后可做完整剪辑")}</span></label>
      </div>
      {status && !status.models.pose ? <p className="live-hint warning">{t("未找到姿态模型：只能显示画面，无法跟踪球员。")}</p> : null}
      <Button size="lg" className="live-start" disabled={live.busy || !source} onClick={start}><Play />{live.busy ? t("正在连接…") : t("开始实时分析")}</Button>
    </section>
  );
}

function CornerPicker({ live, status, onDone }: { live: LiveController; status: LiveStatus; onDone: () => void }) {
  const points = useCourtPoints();
  const [format, setFormat] = useState<MatchFormat>(status.match_format);
  const [stamp, setStamp] = useState(() => Date.now());
  const image = useRef<HTMLImageElement>(null);

  const place = (event: MouseEvent<HTMLDivElement>) => {
    if (!image.current) return;
    const box = image.current.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width;
    const y = (event.clientY - box.top) / box.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    points.place([x, y]);
  };
  const save = async () => {
    const target = points.method === "corners" ? { corners: points.corners } : { landmarks: points.placed };
    if (await live.calibrate(target, format)) onDone();
  };

  return (
    <section className="live-card live-calibration" aria-labelledby="live-calibration-title">
      <div className="live-card-heading">
        <div>
          <p className="eyebrow">{t("场地校准")}</p>
          <h2 id="live-calibration-title">{points.method === "corners" ? t("依次点击单打场地的四个角") : t("标出看得到的场地点")}</h2>
        </div>
        <Button size="sm" variant="ghost" onClick={() => { setStamp(Date.now()); points.reset(); }}><RotateCcw />{t("换一帧")}</Button>
      </div>
      <Segmented
        label={t("校准方式")}
        value={points.method}
        onChange={points.setMethod}
        options={[{ value: "corners", label: t("四个角") }, { value: "landmarks", label: t("角不在画面里") }]}
      />
      <p className="live-hint">
        {points.method === "corners"
          ? t("顺序：近左 → 近右 → 远右 → 远左（单打边线与底线的交点）。其余区域会自动推算。")
          : t("手机放在场边或较低时，有的角拍不到：改为标出任意 4 个以上看得清的点，例如网柱底部和前发球线的交点。")}
      </p>
      <div className={points.method === "landmarks" ? "live-landmark-layout" : undefined}>
        <div className="live-corner-stage" onClick={place} role="presentation">
          <img ref={image} src={apiUrl(`/api/live/snapshot.jpg?t=${stamp}`)} alt={t("当前摄像头画面")} draggable={false} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            {points.method === "corners" && points.corners.length > 1 ? <polygon points={points.corners.map(([x, y]) => `${x},${y}`).join(" ")} /> : null}
          </svg>
          {points.markers.map((marker) => (
            <span key={marker.key} className="live-corner-dot" style={{ left: `${marker.point[0] * 100}%`, top: `${marker.point[1] * 100}%` }}>{marker.label}</span>
          ))}
          {points.prompt ? <span className="live-corner-next">{points.prompt}</span> : null}
        </div>
        {points.method === "landmarks" ? <CourtLandmarkDiagram points={points} /> : null}
      </div>
      <div className="live-calibration-actions">
        <Segmented label={t("比赛类型")} value={format} onChange={setFormat} options={[{ value: "singles", label: t("单打") }, { value: "doubles", label: t("双打") }]} />
        <Button size="sm" variant="outline" disabled={!points.count} onClick={points.undo}><Undo2 />{t("撤销")}</Button>
        {status.studio_calibration_available ? (
          <Button size="sm" variant="outline" disabled={live.busy} onClick={async () => { if (await live.calibrate(null, format)) onDone(); }}>{t("使用 Studio 的校准")}</Button>
        ) : null}
        <Button size="sm" disabled={!points.ready || live.busy} onClick={() => void save()}><Crosshair />{t("保存校准")}</Button>
      </div>
    </section>
  );
}

export function RallyPanel({ status }: { status: Pick<LiveStatus, "analysis"> }) {
  const analysis = status.analysis;
  const current = analysis?.rally.current ?? null;
  return (
    <section className={`live-card live-rally${current ? " active" : ""}`} aria-live="polite">
      <div className="live-rally-state">
        <span className="live-rally-dot" />
        <strong>{current ? t("回合 {{index}} 进行中", { index: current.index }) : t("回合间隙")}</strong>
      </div>
      <div className="live-rally-clock">
        <span>{current ? current.duration.toFixed(1) : "0.0"}</span><small>s</small>
        <em>{t("{{count}} 拍", { count: current?.shots ?? 0 })}</em>
      </div>
      <dl className="live-totals">
        <div><dt>{t("已完成")}</dt><dd>{analysis?.totals.rallies ?? 0}</dd></div>
        <div><dt>{t("平均时长")}</dt><dd>{(analysis?.totals.average_duration ?? 0).toFixed(1)}s</dd></div>
        <div><dt>{t("最长回合")}</dt><dd>{(analysis?.totals.longest_duration ?? 0).toFixed(1)}s</dd></div>
        <div><dt>{t("总拍数")}</dt><dd>{analysis?.totals.shots ?? 0}</dd></div>
      </dl>
    </section>
  );
}

export function PlayerCards({ players, metric }: { players: LivePlayer[]; metric: boolean }) {
  if (!players.length) return <p className="live-hint">{t("等待球员进入场地…")}</p>;
  return (
    <div className="live-players">
      {players.map((player) => (
        <article key={player.player_id} className="live-player" data-player={player.player_id} data-visible={player.visible}>
          <header><span className="live-player-swatch" /><strong>{playerName(player)}</strong>{player.visible ? null : <small>{t("暂时不可见")}</small>}</header>
          <dl>
            {metric ? <div><dt>{t("移动距离")}</dt><dd>{player.distance_meters.toFixed(0)}<small>m</small></dd></div> : null}
            {metric ? <div><dt>{t("当前速度")}</dt><dd>{player.speed_meters_per_second === null ? "–" : player.speed_meters_per_second.toFixed(1)}<small>m/s</small></dd></div> : null}
            {metric ? <div><dt>{t("最快")}</dt><dd>{player.max_speed_meters_per_second.toFixed(1)}<small>m/s</small></dd></div> : null}
            <div><dt>{t("挥拍")}</dt><dd>{player.swings}</dd></div>
            <div><dt>{t("击球")}</dt><dd>{player.hits}</dd></div>
          </dl>
        </article>
      ))}
    </div>
  );
}

function LiveDashboard({ live, status }: { live: LiveController; status: LiveStatus }) {
  const [calibrating, setCalibrating] = useState(false);
  const analysis = status.analysis;
  const streamUrl = useMemo(() => apiUrl(`/api/live/stream.mjpg?session=${status.started_at ?? 0}`), [status.started_at]);
  const needsCalibration = !status.calibrated || calibrating;
  const rallies = [...(analysis?.rallies ?? [])].reverse();

  return (
    <div className="live-dashboard">
      <div className="live-main">
        {needsCalibration && status.frame_size ? (
          <CornerPicker live={live} status={status} onDone={() => setCalibrating(false)} />
        ) : (
          <section className="live-stage">
            {/* MJPEG: the browser keeps replacing the image as annotated frames arrive. */}
                <img src={streamUrl} alt={t("实时分析画面")} />
            {!status.frame_size ? <div className="live-stage-wait"><span className="loader" />{t("正在等待第一帧画面…")}</div> : null}
          </section>
        )}
        <div className="live-stage-meta">
          {status.calibrated ? <Badge variant="outline"><Radio />{t("{{fps}} 帧/秒分析", { fps: status.analysis_fps ?? 0 })}</Badge> : <Badge variant="outline">{t("预览：校准后开始分析")}</Badge>}
          {analysis?.shuttle_enabled ? <Badge variant="outline">{t("羽球跟踪")}</Badge> : null}
          {analysis?.audio_enabled ? <Badge variant="outline">{t("击球声")}</Badge> : null}
          {status.recording ? <Badge variant="outline"><Video />{t("录像中")}</Badge> : null}
          <span className="live-source" title={status.source}>{status.source === "browser" ? t("这台设备的摄像头") : status.source}</span>
          {status.calibrated && !calibrating ? <Button size="sm" variant="ghost" onClick={() => setCalibrating(true)}><Crosshair />{t("重新校准")}</Button> : null}
        </div>
        {rallies.length ? (
          <section className="live-card">
            <div className="live-card-heading compact"><h2>{t("回合记录")}</h2></div>
            <ol className="live-rallies">
              {rallies.slice(0, 12).map((rally) => (
                <li key={rally.index}><span>#{rally.index}</span><strong>{rally.duration.toFixed(1)}s</strong><em>{t("{{count}} 拍", { count: rally.shots })}</em></li>
              ))}
            </ol>
          </section>
        ) : null}
      </div>
      <aside className="live-side">
        <RallyPanel status={status} />
        {analysis?.metric ? (
          <section className="live-card live-court-card"><LiveCourt players={analysis.players} label={t("球员站位")} /></section>
        ) : null}
        <section className="live-card">
          <div className="live-card-heading compact"><h2>{t("球员")}</h2><small>{(analysis?.match_format ?? status.match_format) === "doubles" ? t("双打") : t("单打")}</small></div>
          <PlayerCards players={analysis?.players ?? []} metric={analysis?.metric ?? false} />
        </section>
      </aside>
    </div>
  );
}

export function LiveStudio({ onBack }: { onBack?: () => void }) {
  useTranslation();
  const live = useLiveSession();
  const { status } = live;
  const running = status ? RUNNING_STATES.has(status.state) : false;

  return (
    <div className="live-app">
      <header className="topbar live-topbar">
        <Brand />
        <div className="project-title">
          <span className={`status-dot${running ? " live" : ""}`} />
          <span>{running ? t("实时分析进行中") : t("实时分析")}</span>
        </div>
        <div className="top-actions">
          <LanguageSwitcher /><ThemeToggle />
          {onBack ? <Button size="sm" variant="ghost" onClick={onBack}><ArrowLeft />{t("返回剪辑")}</Button> : null}
          {running ? <Button size="sm" variant="destructive" disabled={live.busy} onClick={() => void live.stop()}><Square />{t("结束")}</Button> : null}
        </div>
      </header>
      <main className="live-content">
        {live.error ? <div className="live-alert" role="alert"><span>{translateMessage(live.error)}</span><button onClick={live.clearError}>{t("知道了")}</button></div> : null}
        {status?.state === "error" && status.error ? <div className="live-alert" role="alert"><span>{translateMessage(status.error)}</span></div> : null}
        {live.uploading ? <div className="live-alert info"><span className="loader" />{live.uploading}</div> : null}
        {running && status ? <LiveDashboard live={live} status={status} /> : <LiveSetup live={live} status={status} />}
        {!running && status?.analysis ? (
          <section className="live-card live-summary">
            <div className="live-card-heading compact"><h2>{t("上一节总结")}</h2>{status.recording ? <small title={status.recording}>{t("录像已保存，可在 Studio 中做完整剪辑")}</small> : null}</div>
            <RallyPanel status={status} />
            <PlayerCards players={status.analysis.players} metric={status.analysis.metric} />
          </section>
        ) : null}
      </main>
    </div>
  );
}
