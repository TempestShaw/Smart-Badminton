"use client";

import { Calculator, LoaderCircle, Scissors, Trash2 } from "lucide-react";

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { StudioController } from "@/hooks/use-studio-controller";
import type { ScoreCorrection } from "@/types/studio";

const eventLabels: Record<string, string> = {
  landing_in_candidate: "界内落地",
  landing_out_candidate: "出界",
  net_candidate: "下网",
  flight_lost: "轨迹丢失",
  unknown: "未知",
};
const sideLabels = { near: "近场", far: "远场", unknown: "" };

export function RallyInspector({ studio }: { studio: StudioController }) {
  const segment = studio.selectedSegment;
  const rally = studio.selectedIndex + 1;
  const action = !studio.dirty && rally > 0 ? studio.analyticsMap.get(rally) : undefined;
  const score = rally > 0 ? studio.scoreMap.get(rally) : undefined;
  const correction = studio.score.corrections?.find((row) => row.rally === rally);
  const step = 1 / Math.max(1, studio.project?.video.fps ?? 30);
  const terminal = action
    ? [sideLabels[action.terminal_landing_side], eventLabels[action.terminal_event] || "未知"].filter(Boolean).join(" · ")
    : "—";

  return (
    <aside className="inspector">
      <div className="panel-heading"><div><span className="eyebrow">CLIP INSPECTOR</span><h2>{segment ? `回合片段 ${rally}` : "选择一个片段"}</h2></div><span className="clip-chip">{segment ? `R${String(rally).padStart(2, "0")}` : "—"}</span></div>
      <div className="time-fields">
        <Label>开始时间<Input type="number" min={0} step={0.001} disabled={!segment} value={segment?.start.toFixed(3) ?? ""} onChange={(event) => studio.editBoundary(studio.selectedIndex, "start", Number(event.target.value))} /></Label>
        <Label>结束时间<Input type="number" min={0} step={0.001} disabled={!segment} value={segment?.end.toFixed(3) ?? ""} onChange={(event) => studio.editBoundary(studio.selectedIndex, "end", Number(event.target.value))} /></Label>
      </div>
      <div className="duration-card"><span>片段长度</span><strong>{segment ? `${(segment.end - segment.start).toFixed(3)} s` : "—"}</strong></div>

      {segment ? <ScoreCard studio={studio} score={score} correction={correction} /> : null}

      {studio.analytics.available && segment ? (
        <section className={studio.dirty ? "action-insights stale" : "action-insights"}>
          <div className="action-heading"><span className="eyebrow">球路数据</span><span>{studio.dirty ? "保存后更新" : action?.rally_style ?? ""}</span></div>
          {action?.trajectory_available ? (
            <div className="action-grid">
              <Metric label="球路覆盖" value={`${Math.round(action.trajectory_coverage_percent)}%`} />
              <Metric label="空中球路" value={`${action.trajectory_visible_seconds.toFixed(1)} s`} />
              <Metric label="连续追踪" value={`${action.longest_continuous_track_seconds.toFixed(1)} s`} />
              <Metric label="轨迹里程" value={`${action.trajectory_distance_frames.toFixed(1)} 画幅`} />
              <Metric label="网区经过" value={`${action.net_zone_transits} 次`} />
              <Metric label="终局" value={terminal} />
            </div>
          ) : <div className="empty-insight">暂无本场球路</div>}
        </section>
      ) : null}

      <div className="nudge-grid">
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "start", segment.start - step)}>开始 −1 帧</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "start", segment.start + step)}>开始 +1 帧</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "end", segment.end - step)}>结束 −1 帧</Button>
        <Button size="xs" variant="outline" disabled={!segment} onClick={() => segment && studio.editBoundary(studio.selectedIndex, "end", segment.end + step)}>结束 +1 帧</Button>
      </div>
      <div className="edit-actions">
        <Button variant="outline" disabled={!segment} onClick={studio.splitSegment}><Scissors />在播放头分割</Button>
        <AlertDialog><AlertDialogTrigger asChild><Button variant="destructive" disabled={!segment}><Trash2 />删除片段</Button></AlertDialogTrigger><AlertDialogContent><AlertDialogHeader><AlertDialogTitle>删除 R{String(rally).padStart(2, "0")}？</AlertDialogTitle><AlertDialogDescription>这个片段会从当前时间轴移除；之后仍可用“撤销”恢复。</AlertDialogDescription></AlertDialogHeader><AlertDialogFooter><AlertDialogCancel>取消</AlertDialogCancel><AlertDialogAction onClick={studio.deleteSegment}>删除片段</AlertDialogAction></AlertDialogFooter></AlertDialogContent></AlertDialog>
      </div>
      <div className="shortcut-help"><span className="eyebrow">快捷键</span><p><kbd>Space</kbd> 播放　<kbd>←</kbd><kbd>→</kbd> 逐帧</p><p><kbd>[</kbd> 设为开始　<kbd>]</kbd> 设为结束</p></div>

    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ScoreCard({ studio, score, correction }: { studio: StudioController; score: ReturnType<StudioController["scoreMap"]["get"]>; correction?: ScoreCorrection }) {
  const sourceLabels: Record<string, string> = {
    "automatic-next-serve": "自动",
    "automatic-terminal": "自动",
    manual: "人工",
    "manual-no-point": "无效",
    unresolved: "未知",
  };
  const serverLabels = { near: "近场", far: "远场", unknown: "未知" };
  const evidenceCurrent = !studio.dirty && studio.score.available && Boolean(score);
  const button = (winner: ScoreCorrection["winner"], label: string) => <Button disabled={!evidenceCurrent} size="xs" variant={(correction?.winner ?? "auto") === winner ? "default" : "outline"} onClick={() => void studio.updateScore({ winner })}>{label}</Button>;
  const server = (value: ScoreCorrection["server_override"], label: string) => <Button disabled={!evidenceCurrent} size="xs" variant={correction?.server_override === value ? "default" : "outline"} onClick={() => void studio.updateScore({ server_override: value })}>{label}</Button>;
  const sourceLabel = studio.scoreCalculating
    ? "计算中"
    : studio.dirty
      ? "需保存"
      : studio.score.stale
        ? "需重算"
        : evidenceCurrent
          ? sourceLabels[score?.winner_source ?? "unresolved"]
          : "未计算";
  return (
    <section className={evidenceCurrent ? "score-card" : "score-card stale"}>
      <div className="score-heading">
        <span className="eyebrow">比分</span>
        <div className="score-heading-actions">
          <Badge variant={evidenceCurrent && score?.winner_source?.startsWith("automatic") ? "default" : evidenceCurrent && score?.winner_source?.startsWith("manual") ? "secondary" : "outline"}>{sourceLabel}</Badge>
          <Button size="xs" variant="secondary" disabled={studio.scoreCalculating || !studio.segments.length || studio.analysisStatus.state === "running"} onClick={() => void studio.calculateScore()}>
            {studio.scoreCalculating ? <LoaderCircle className="animate-spin" /> : <Calculator />}
            {studio.score.generated ? "重新计算" : "计算比分"}
          </Button>
        </div>
      </div>
      <div className="score-line"><span>近场</span><strong>{evidenceCurrent ? score?.near_score ?? 0 : "—"}</strong><em>:</em><strong>{evidenceCurrent ? score?.far_score ?? 0 : "—"}</strong><span>远场</span></div>
      <p>发球：{evidenceCurrent ? serverLabels[score?.server_next ?? "unknown"] : "—"}</p>
      <div className="score-actions">{button("near", "近场得分")}{button("far", "远场得分")}{button("no_point", "本分无效")}{button("auto", "恢复自动")}{server("near", "近场发球")}{server("far", "远场发球")}{server("unknown", "清除发球方")}</div>
    </section>
  );
}
