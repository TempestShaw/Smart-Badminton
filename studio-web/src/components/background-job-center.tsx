"use client";

import { CheckCircle2, CircleAlert, Clock3, LoaderCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import type { StudioController } from "@/hooks/use-studio-controller";

const modeLabels = {
  single: "当前视频剪片预测",
  batch: "批量剪片预测",
  shuttle: "整段羽球轨迹",
  visual: "整段视觉分析",
} as const;

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes ? `${minutes} 分 ${remainder} 秒` : `${remainder} 秒`;
}

export function BackgroundJobCenter({ studio }: { studio: StudioController }) {
  const status = studio.analysisStatus;
  const running = status.state === "running";
  const failed = status.state === "error";
  const complete = status.state === "complete";
  const percentage = Math.max(0, Math.min(100, Math.round((status.progress ?? 0) * 100)));
  const mode = status.mode ? modeLabels[status.mode] : "后台任务";
  const duration = status.started_at
    ? formatDuration(((status.state === "running" ? Date.now() / 1000 : status.updated_at) ?? Date.now() / 1000) - status.started_at)
    : null;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          size="sm"
          variant={failed ? "destructive" : running ? "secondary" : "outline"}
          className={`job-center-trigger ${status.state}`}
          aria-label={running ? `${mode}，完成 ${percentage}%` : failed ? "后台任务失败" : "查看后台任务"}
        >
          {running ? <LoaderCircle className="job-spinner" /> : failed ? <CircleAlert /> : complete ? <CheckCircle2 /> : <Clock3 />}
          <span>{running ? `${mode} ${percentage}%` : failed ? "任务失败" : complete ? "任务完成" : "后台任务"}</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="job-center-dialog sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>后台任务中心</DialogTitle>
          <DialogDescription>任务在本地后台运行。</DialogDescription>
        </DialogHeader>

        <div className={`job-detail-card ${status.state}`}>
          <div className="job-detail-heading">
            <div>
              <span className="eyebrow">{mode}</span>
              <strong>{status.label ?? (status.state === "idle" ? "目前没有后台任务" : "正在读取任务状态")}</strong>
            </div>
            <Badge variant={failed ? "destructive" : running ? "default" : "outline"}>
              {running ? "运行中" : failed ? "失败" : complete ? "已完成" : "空闲"}
            </Badge>
          </div>
          {status.state !== "idle" ? (
            <>
              <Progress value={percentage} className="job-detail-progress" aria-label={`任务完成 ${percentage}%`} />
              <div className="job-percentage"><strong>{percentage}%</strong><span>{status.completed ?? 0} / {status.total ?? 1} 个项目</span></div>
              <dl className="job-metadata">
                <div><dt>当前视频</dt><dd>{status.current_name ?? status.project_name ?? studio.project?.video.name ?? "—"}</dd></div>
                <div><dt>当前阶段</dt><dd>{status.stage ?? "—"}</dd></div>
                <div><dt>已运行</dt><dd>{duration ?? "—"}</dd></div>
                <div><dt>任务编号</dt><dd>{status.job_id ?? "本次会话任务"}</dd></div>
              </dl>
            </>
          ) : null}
          {status.message ? <p className="job-error-message">{status.message}</p> : null}
        </div>

        <div className="job-background-note">
          <p>关闭或刷新页面不会中断任务；请保持 Studio 服务运行。</p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
