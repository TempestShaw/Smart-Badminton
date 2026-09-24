"use client";

import { useTranslation } from "react-i18next";
import { t, translateMessage } from "@/lib/i18n";

import { Clapperboard, Globe2, Redo2, Save, Undo2 } from "lucide-react";
import { useState } from "react";
import { LanguageSwitcher } from "@/components/language-controls";
import { ThemeToggle } from "@/components/theme-controls";
import { Brand } from "@/components/brand";

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BackgroundJobCenter } from "@/components/background-job-center";
import { StudioTutorial } from "@/components/studio-tutorial";
import type { StudioController } from "@/hooks/use-studio-controller";

export function AppHeader({ studio, onSwitchEngine }: { studio: StudioController; onSwitchEngine?: () => void }) {
  useTranslation();
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
        <span>{studio.project?.video.name ?? t("正在载入项目…")}</span>
      </div>
      <div className="top-actions">
        <LanguageSwitcher /><ThemeToggle />
        {onSwitchEngine ? <Button size="sm" variant="ghost" onClick={onSwitchEngine}><Globe2 />{t("浏览器版")}</Button> : null}
        <BackgroundJobCenter studio={studio} />
        <StudioTutorial studio={studio} />
        <Badge variant={studio.dirty ? "destructive" : "outline"}>{studio.dirty ? t("有未保存修改") : t("已保存")}</Badge>
        <Button size="sm" variant="outline" disabled={!studio.history.length} onClick={studio.undo}><Undo2 />{t("撤销")}</Button>
        <Button size="sm" variant="outline" disabled={!studio.future.length} onClick={studio.redo}><Redo2 />{t("重做")}</Button>
        <Button size="sm" disabled={studio.saving || !studio.project} onClick={() => void studio.saveTimeline()}>
          <Save />{studio.saving ? t("正在保存…") : t("保存时间表")}
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              data-guide-action="export"
              size="sm"
              variant="secondary"
              disabled={!studio.segments.length || studio.renderStatus.state === "running" || !ffmpegReady}
              title={ffmpegReady ? undefined : translateMessage(studio.project?.runtime.ffmpeg.reason) || t("FFmpeg 不可用")}
            >
              <Clapperboard />{studio.renderStatus.state === "running" ? t("正在输出…") : outputExists ? t("重新输出成片") : t("输出成片")}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{outputExists ? t("覆盖已有成片？") : t("开始输出成片？")}</AlertDialogTitle>
              <AlertDialogDescription className="break-all">
                {outputExists ? t("目标文件已经存在，继续会覆盖它。") : t("Studio 将读取高清源视频并按当前已保存时间表渲染。")}
                <br />{t("目标：")}{studio.project?.output.path}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid gap-2">
              <div className="grid gap-2 rounded-[12px] border p-3 text-[13px]">
                <span>{t("内容范围")}</span>
                <Select value={scoreReady ? winnerFilter : "all"} onValueChange={(value) => setWinnerFilter(value as "all" | "near" | "far")}>
                  <SelectTrigger className="w-full" aria-label={t("内容范围")}><SelectValue /></SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="all">{t("全部回合")}</SelectItem>
                    <SelectItem value="near" disabled={!scoreReady}>{t("近场得分")}</SelectItem>
                    <SelectItem value="far" disabled={!scoreReady}>{t("远场得分")}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <label className="flex items-center gap-3 rounded-[12px] border p-3 text-[13px]">
                <Checkbox
                  checked={trajectoryReady && includeTrajectory}
                  disabled={!trajectoryReady}
                  onCheckedChange={(checked) => setIncludeTrajectory(checked === true)}
                />
                <span>{trajectoryReady ? t("包含羽球轨迹") : t("需先分析球路")}</span>
              </label>
              <label className="flex items-center gap-3 rounded-[12px] border p-3 text-[13px]">
                <Checkbox
                  checked={scoreReady && includeScore}
                  disabled={!scoreReady}
                  onCheckedChange={(checked) => setIncludeScore(checked === true)}
                />
                <span>{scoreReady ? t("显示比分") : t("需先计算比分")}</span>
              </label>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("取消")}</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void studio.startRender(
                  trajectoryReady && includeTrajectory,
                  scoreReady ? winnerFilter : "all",
                  scoreReady && includeScore,
                )}
              >
                {outputExists ? t("确认覆盖并输出") : t("开始输出")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
      {studio.analysisStatus.state === "running" ? (
        <div className="topbar-job-progress" aria-hidden="true"><span style={{ transform: `scaleX(${Math.max(0, Math.min(1, studio.analysisStatus.progress ?? 0))})` }} /></div>
      ) : null}
    </header>
  );
}
