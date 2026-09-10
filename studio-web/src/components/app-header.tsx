"use client";

import { Clapperboard, Globe2, Redo2, Save, Undo2 } from "lucide-react";
import { useState } from "react";
import { ThemeToggle } from "@/components/theme-controls";
import { Brand } from "@/components/brand";

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { BackgroundJobCenter } from "@/components/background-job-center";
import { StudioTutorial } from "@/components/studio-tutorial";
import type { StudioController } from "@/hooks/use-studio-controller";

export function AppHeader({ studio, onSwitchEngine }: { studio: StudioController; onSwitchEngine?: () => void }) {
  const [includeTrajectory, setIncludeTrajectory] = useState(false);
  const [winnerFilter, setWinnerFilter] = useState<"all" | "near" | "far">("all");
  const [includeScore, setIncludeScore] = useState(false);
  const outputExists = studio.project?.output.file_exists;
  const ffmpegReady = studio.project?.runtime.ffmpeg.available === true;
  const trajectoryReady = studio.project?.shuttle_analysis.generated === true;
  const scoreReady = studio.score.available && !studio.dirty;
  return (
    <header className="topbar">
      <Brand />
      <div className="project-title">
        <span className="status-dot" />
        <span>{studio.project?.video.name ?? "正在载入项目…"}</span>
      </div>
      <div className="top-actions">
        <ThemeToggle />
        {onSwitchEngine ? <Button size="sm" variant="ghost" onClick={onSwitchEngine}><Globe2 />浏览器版</Button> : null}
        <BackgroundJobCenter studio={studio} />
        <StudioTutorial studio={studio} />
        <Badge variant={studio.dirty ? "destructive" : "outline"}>{studio.dirty ? "有未保存修改" : "已保存"}</Badge>
        <Button size="sm" variant="outline" disabled={!studio.history.length} onClick={studio.undo}><Undo2 />撤销</Button>
        <Button size="sm" variant="outline" disabled={!studio.future.length} onClick={studio.redo}><Redo2 />重做</Button>
        <Button size="sm" disabled={studio.saving || !studio.project} onClick={() => void studio.saveTimeline()}>
          <Save />{studio.saving ? "正在保存…" : "保存时间表"}
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              data-guide-action="export"
              size="sm"
              variant="secondary"
              disabled={!studio.segments.length || studio.renderStatus.state === "running" || !ffmpegReady}
              title={ffmpegReady ? undefined : studio.project?.runtime.ffmpeg.reason ?? "FFmpeg 不可用"}
            >
              <Clapperboard />{studio.renderStatus.state === "running" ? "正在输出…" : outputExists ? "重新输出成片" : "输出成片"}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{outputExists ? "覆盖已有成片？" : "开始输出成片？"}</AlertDialogTitle>
              <AlertDialogDescription className="break-all">
                {outputExists ? "目标文件已经存在，继续会覆盖它。" : "Studio 将读取高清源视频并按当前已保存时间表渲染。"}
                <br />目标：{studio.project?.output.path}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid gap-2">
              <label className="grid gap-2 rounded-md border p-3 text-sm">
                <span>内容范围</span>
                <select
                  aria-label="内容范围"
                  className="h-8 w-full rounded-lg border border-input bg-background px-2 text-sm"
                  value={scoreReady ? winnerFilter : "all"}
                  onChange={(event) => setWinnerFilter(event.target.value as "all" | "near" | "far")}
                >
                  <option value="all">全部回合</option>
                  <option value="near" disabled={!scoreReady}>近场得分</option>
                  <option value="far" disabled={!scoreReady}>远场得分</option>
                </select>
              </label>
              <label className="flex items-center gap-3 rounded-md border p-3 text-sm">
                <Checkbox
                  checked={trajectoryReady && includeTrajectory}
                  disabled={!trajectoryReady}
                  onCheckedChange={(checked) => setIncludeTrajectory(checked === true)}
                />
                <span>{trajectoryReady ? "包含羽球轨迹" : "需先分析球路"}</span>
              </label>
              <label className="flex items-center gap-3 rounded-md border p-3 text-sm">
                <Checkbox
                  checked={scoreReady && includeScore}
                  disabled={!scoreReady}
                  onCheckedChange={(checked) => setIncludeScore(checked === true)}
                />
                <span>{scoreReady ? "显示比分" : "需先计算比分"}</span>
              </label>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void studio.startRender(
                  trajectoryReady && includeTrajectory,
                  scoreReady ? winnerFilter : "all",
                  scoreReady && includeScore,
                )}
              >
                {outputExists ? "确认覆盖并输出" : "开始输出"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
      {studio.analysisStatus.state === "running" ? (
        <div className="topbar-job-progress" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(100, (studio.analysisStatus.progress ?? 0) * 100))}%` }} /></div>
      ) : null}
    </header>
  );
}
