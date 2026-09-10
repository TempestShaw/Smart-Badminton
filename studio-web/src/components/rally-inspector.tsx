"use client";

import { useTranslation } from "react-i18next";
import { t, translateMessage } from "@/lib/i18n";

import { useState } from "react";
import { Calculator, ListChecks, LoaderCircle, Scissors, SkipForward, Trash2, X } from "lucide-react";

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { StudioController } from "@/hooks/use-studio-controller";
import type { ScoreCorrection, ScoreSuggestion } from "@/types/studio";

const eventLabels: Record<string, string> = {
  landing_in_candidate: "界内落地",
  landing_out_candidate: "出界",
  net_candidate: "下网",
  flight_lost: "轨迹丢失",
  unknown: "未知",
};
const sideLabels = { near: "近场", far: "远场", unknown: "" };
const SKIP_DELETE_CONFIRM_KEY = "smart-badminton:skip-delete-confirm:v1";

export function RallyInspector({ studio }: { studio: StudioController }) {
  useTranslation();
  const segment = studio.selectedSegment;
  const rally = studio.selectedIndex + 1;
  const action = !studio.dirty && rally > 0 ? studio.analyticsMap.get(rally) : undefined;
  const score = rally > 0 ? studio.scoreMap.get(rally) : undefined;
  const correction = studio.score.corrections?.find((row) => row.rally === rally);
  const suggestion = studio.scoreSuggestionMap.get(rally);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [skipDeleteConfirm, setSkipDeleteConfirm] = useState(
    () => typeof window !== "undefined" && window.localStorage.getItem(SKIP_DELETE_CONFIRM_KEY) === "1",
  );
  const [rememberDelete, setRememberDelete] = useState(false);
  const step = 1 / Math.max(1, studio.project?.video.fps ?? 30);
  const terminal = action
    ? [t(sideLabels[action.terminal_landing_side]), t(eventLabels[action.terminal_event] ?? "") || t("未知")].filter(Boolean).join(" · ")
    : "—";
  const requestDelete = () => {
    if (skipDeleteConfirm) {
      studio.deleteSegment();
      return;
    }
    setRememberDelete(false);
    setDeleteOpen(true);
  };
  const confirmDelete = () => {
    if (rememberDelete) {
      window.localStorage.setItem(SKIP_DELETE_CONFIRM_KEY, "1");
      setSkipDeleteConfirm(true);
    }
    setDeleteOpen(false);
    studio.deleteSegment();
  };

  return (
    <aside className="inspector" data-guide-action="score" data-guide-surface tabIndex={-1}>
      <div className="panel-heading"><div><span className="eyebrow">{t("CLIP INSPECTOR")}</span><h2>{segment ? t("回合片段 {{value1}}", { value1: rally }) : t("选择一个片段")}</h2></div><span className="clip-chip">{segment ? `R${String(rally).padStart(2, "0")}` : "—"}</span></div>
      <div className="time-fields">
        <Label>{t("开始时间")}<Input type="number" min={0} step={0.001} disabled={!segment} value={segment?.start.toFixed(3) ?? ""} onChange={(event) => studio.editBoundary(studio.selectedIndex, "start", Number(event.target.value))} /></Label>
        <Label>{t("结束时间")}<Input type="number" min={0} step={0.001} disabled={!segment} value={segment?.end.toFixed(3) ?? ""} onChange={(event) => studio.editBoundary(studio.selectedIndex, "end", Number(event.target.value))} /></Label>
      </div>
      <div className="duration-card"><span>{t("片段长度")}</span><strong>{segment ? `${(segment.end - segment.start).toFixed(3)} s` : "—"}</strong></div>

      {segment ? <ScoreCard studio={studio} score={score} correction={correction} suggestion={suggestion} /> : null}

      {studio.analytics.available && segment ? (
        <section className={studio.dirty ? "action-insights stale" : "action-insights"}>
          <div className="action-heading"><span className="eyebrow">{t("球路数据")}</span><span>{studio.dirty ? t("保存后更新") : translateMessage(action?.rally_style)}</span></div>
          {action?.trajectory_available ? (
            <div className="action-grid">
              <Metric label={t("球路覆盖")} value={`${Math.round(action.trajectory_coverage_percent)}%`} />
              <Metric label={t("空中球路")} value={`${action.trajectory_visible_seconds.toFixed(1)} s`} />
              <Metric label={t("连续追踪")} value={`${action.longest_continuous_track_seconds.toFixed(1)} s`} />
              <Metric label={t("轨迹里程")} value={t("{{value1}} 画幅", { value1: action.trajectory_distance_frames.toFixed(1) })} />
              <Metric label={t("网区经过")} value={t("{{value1}} 次", { value1: action.net_zone_transits })} />
              <Metric label={t("终局")} value={terminal} />
            </div>
          ) : <div className="empty-insight">{t("暂无本场球路")}</div>}
        </section>
      ) : null}

      <div className="nudge-grid">
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "start", segment.start - step)}>{t("开始 −1 帧")}</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "start", segment.start + step)}>{t("开始 +1 帧")}</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "end", segment.end - step)}>{t("结束 −1 帧")}</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "end", segment.end + step)}>{t("结束 +1 帧")}</Button>
      </div>
      <div className="edit-actions">
        <Button variant="outline" disabled={!segment} onClick={studio.splitSegment}><Scissors />{t("在播放头分割")}</Button>
        <Button variant="destructive" disabled={!segment} onClick={requestDelete}><Trash2 />{t("删除片段")}</Button>
        <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("删除 R{{rally}}？", { rally: String(rally).padStart(2, "0") })}</AlertDialogTitle>
              <AlertDialogDescription>{t("片段会从时间轴移除，可用“撤销”恢复。")}</AlertDialogDescription>
            </AlertDialogHeader>
            <Label className="delete-confirm-option"><Checkbox checked={rememberDelete} onCheckedChange={(checked) => setRememberDelete(checked === true)} />{t("下次不再提醒")}</Label>
            <AlertDialogFooter><AlertDialogCancel>{t("取消")}</AlertDialogCancel><AlertDialogAction onClick={confirmDelete}>{t("删除片段")}</AlertDialogAction></AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
      <div className="shortcut-help"><span className="eyebrow">{t("快捷键")}</span><p><kbd>Space</kbd> {t("播放")}<kbd>←</kbd><kbd>→</kbd> {t("逐帧")}</p><p><kbd>[</kbd> {t("设为开始")}<kbd>]</kbd> {t("设为结束")}</p></div>

    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  useTranslation();
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ScoreCard({ studio, score, correction, suggestion }: { studio: StudioController; score: ReturnType<StudioController["scoreMap"]["get"]>; correction?: ScoreCorrection; suggestion?: ScoreSuggestion }) {
  useTranslation();
  const rally = score?.rally ?? correction?.rally ?? suggestion?.rally ?? 0;
  const sourceLabels: Record<string, string> = {
    "automatic-next-serve": t("自动"),
    "automatic-terminal": t("自动"),
    manual: t("人工"),
    "manual-no-point": t("无效"),
    unresolved: t("未知"),
  };
  const serverLabels = { near: t("近场"), far: t("远场"), unknown: t("未知") };
  const evidenceCurrent = !studio.dirty && studio.score.available && Boolean(score);
  const button = (winner: ScoreCorrection["winner"], label: string) => <Button disabled={!evidenceCurrent} size="xs" variant={(correction?.winner ?? "auto") === winner ? "default" : "outline"} onClick={() => void studio.updateScore({ winner })}>{label}</Button>;
  const sourceLabel = studio.scoreCalculating
    ? t("计算中")
    : studio.dirty
      ? t("需保存")
      : studio.score.stale
        ? t("需重算")
        : evidenceCurrent
          ? sourceLabels[score?.winner_source ?? "unresolved"]
          : t("未计算");
  return (
    <section className={evidenceCurrent ? "score-card" : "score-card stale"}>
      <div className="score-heading">
        <span className="eyebrow">{t("本局比分")}</span>
        <div className="score-heading-actions">
          <Badge variant={evidenceCurrent && score?.winner_source?.startsWith("automatic") ? "default" : evidenceCurrent && score?.winner_source?.startsWith("manual") ? "secondary" : "outline"}>{sourceLabel}</Badge>
          <Button size="xs" variant="secondary" disabled={studio.scoreCalculating || !studio.segments.length || studio.analysisStatus.state === "running"} onClick={() => void studio.calculateScore()}>
            {studio.scoreCalculating ? <LoaderCircle className="animate-spin" /> : <Calculator />}
            {studio.score.generated ? t("重新计算") : t("计算比分")}
          </Button>
        </div>
      </div>
      {studio.scoreReview.active ? (
        <div className="score-review-bar">
          <strong>{t("待标注 {{count}} 分", { count: studio.scoreReview.remaining })}</strong>
          <Button size="xs" variant="outline" onClick={() => studio.moveScoreReview(1)}><SkipForward />{t("跳过")}</Button>
          <Button size="xs" variant="ghost" onClick={studio.stopScoreReview}><X />{t("退出")}</Button>
        </div>
      ) : (studio.score.unresolved ?? 0) > 0 ? (
        <Button className="score-review-launch" size="sm" variant="outline" disabled={!evidenceCurrent || studio.dirty} onClick={studio.startScoreReview}>
          <ListChecks />{t("标注 {{count}} 分", { count: studio.score.unresolved })}</Button>
      ) : null}
      <div className="score-line"><span>{t("近场")}</span><strong>{evidenceCurrent ? score?.near_score ?? 0 : "—"}</strong><em>:</em><strong>{evidenceCurrent ? score?.far_score ?? 0 : "—"}</strong><span>{t("远场")}</span></div>
      {evidenceCurrent && ((score?.near_games ?? 0) > 0 || (score?.far_games ?? 0) > 0) ? <p>{t("局数")}{score?.near_games ?? 0}:{score?.far_games ?? 0}</p> : null}
      <p>{t("下分发球：")}{evidenceCurrent ? t(serverLabels[score?.server_next ?? "unknown"]) : "—"}</p>
      {suggestion && ["consensus", "review"].includes(suggestion.status) ? (
        <div className="score-suggestion">
          <span>{t("建议：")}{t(winnerLabels[suggestion.suggestion.winner])} · {t(terminalLabels[suggestion.suggestion.terminal_event])}</span>
          <Button size="xs" onClick={() => void studio.reviewScoreLabel(rally, "accepted")}>{t("采用")}</Button>
          <Button size="xs" variant="ghost" onClick={() => void studio.reviewScoreLabel(rally, "rejected")}>{t("忽略")}</Button>
        </div>
      ) : null}
      <div className="score-actions">{button("near", t("近场得分"))}{button("far", t("远场得分"))}{button("no_point", t("本分无效"))}{button("auto", t("恢复自动"))}</div>
      <details className="outcome-editor">
        <summary>{t("标注依据")}</summary>
        <div className="outcome-fields">
          <OutcomeSelect label={t("本分发球")} value={correction?.server ?? "unknown"} disabled={!evidenceCurrent} options={sideOptions} onChange={(server) => void studio.updateScore({ server: server as ScoreCorrection["server"] })} />
          <OutcomeSelect label={t("最后击球")} value={correction?.last_hitter ?? "unknown"} disabled={!evidenceCurrent} options={sideOptions} onChange={(last_hitter) => void studio.updateScore({ last_hitter: last_hitter as ScoreCorrection["last_hitter"] })} />
          <OutcomeSelect label={t("终局")} value={correction?.terminal_event ?? "unknown"} disabled={!evidenceCurrent} options={terminalOptions} onChange={(terminal_event) => void studio.updateScore({ terminal_event: terminal_event as ScoreCorrection["terminal_event"] })} />
          <OutcomeSelect label={t("落点")} value={correction?.landing_side ?? "unknown"} disabled={!evidenceCurrent} options={sideOptions} onChange={(landing_side) => void studio.updateScore({ landing_side: landing_side as ScoreCorrection["landing_side"] })} />
          <OutcomeSelect label={t("回合后")} value={correction?.post_rally_event ?? "unknown"} disabled={!evidenceCurrent} options={postRallyOptions} onChange={(post_rally_event) => void studio.updateScore({ post_rally_event: post_rally_event as ScoreCorrection["post_rally_event"] })} />
          <OutcomeSelect label={t("下分发球")} value={correction?.server_override ?? "unknown"} disabled={!evidenceCurrent} options={sideOptions} onChange={(server_override) => void studio.updateScore({ server_override: server_override as ScoreCorrection["server_override"] })} />
        </div>
      </details>
    </section>
  );
}

const sideOptions = [["unknown", "未知"], ["near", "近场"], ["far", "远场"]] as const;
const terminalOptions = [["unknown", "未知"], ["landing_in", "界内落地"], ["landing_out", "出界"], ["net", "下网"], ["unreturned", "未回击"]] as const;
const postRallyOptions = [["unknown", "未知"], ["none", "无"], ["handoff", "送球"]] as const;
const winnerLabels: Record<string, string> = { near: "近场", far: "远场", no_point: "无效", unknown: "未知" };
const terminalLabels: Record<string, string> = { landing_in: "界内", landing_out: "出界", net: "下网", unreturned: "未回击", unknown: "未知" };

function OutcomeSelect({ label, value, options, disabled, onChange }: { label: string; value: string; options: ReadonlyArray<readonly [string, string]>; disabled: boolean; onChange: (value: string) => void }) {
  useTranslation();
  return <label>{label}<select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>{options.map(([option, text]) => <option key={option} value={option}>{t(text)}</option>)}</select></label>;
}
