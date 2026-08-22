"use client";

import { Clapperboard, Redo2, Save, Undo2 } from "lucide-react";

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BackgroundJobCenter } from "@/components/background-job-center";
import { StudioTutorial } from "@/components/studio-tutorial";
import type { StudioController } from "@/hooks/use-studio-controller";

export function AppHeader({ studio }: { studio: StudioController }) {
  const outputExists = studio.project?.output.file_exists;
  const ffmpegReady = studio.project?.runtime.ffmpeg.available === true;
  return (
    <header className="topbar">
      <div className="brand" aria-label="Smart Badminton Studio">
        <span className="brand-mark">SB</span>
        <div><strong>SMART BADMINTON</strong><span>STUDIO</span></div>
      </div>
      <div className="project-title">
        <span className="status-dot" />
        <span>{studio.project?.video.name ?? "正在载入项目…"}</span>
      </div>
      <div className="top-actions">
        <BackgroundJobCenter studio={studio} />
        <StudioTutorial />
        <Badge variant={studio.dirty ? "destructive" : "outline"}>{studio.dirty ? "有未保存修改" : "已保存"}</Badge>
        <Button size="sm" variant="outline" disabled={!studio.history.length} onClick={studio.undo}><Undo2 />撤销</Button>
        <Button size="sm" variant="outline" disabled={!studio.future.length} onClick={studio.redo}><Redo2 />重做</Button>
        <Button size="sm" disabled={studio.saving || !studio.project} onClick={() => void studio.saveTimeline()}>
          <Save />{studio.saving ? "正在保存…" : "保存时间表"}
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
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
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction onClick={() => void studio.startRender()}>{outputExists ? "确认覆盖并输出" : "开始输出"}</AlertDialogAction>
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
