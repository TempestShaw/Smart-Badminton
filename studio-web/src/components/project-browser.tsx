"use client";

import { useTranslation } from "react-i18next";
import { t, translateMessage } from "@/lib/i18n";

import { useEffect, useState } from "react";
import { FolderOpen, RefreshCw, Save, Video } from "lucide-react";

import { DirectoryPicker } from "@/components/directory-picker";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { StudioController } from "@/hooks/use-studio-controller";

function formatFileSize(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
}

export function ProjectBrowser({ studio }: { studio: StudioController }) {
  useTranslation();
  const [path, setPath] = useState("");
  const [selected, setSelected] = useState("");
  const [outputDirectory, setOutputDirectory] = useState("");
  const [outputFilename, setOutputFilename] = useState("");
  const [sourcePickerOpen, setSourcePickerOpen] = useState(false);
  const [outputPickerOpen, setOutputPickerOpen] = useState(false);
  const [pendingVideo, setPendingVideo] = useState("");

  useEffect(() => {
    setPath(studio.library?.path ?? "");
    setSelected(studio.library?.videos.find((video) => video.active)?.id ?? studio.library?.videos[0]?.id ?? "");
  }, [studio.library]);

  useEffect(() => {
    setOutputDirectory(studio.project?.output.directory ?? "");
    setOutputFilename(studio.project?.output.filename ?? "");
  }, [studio.project?.id, studio.project?.output.directory, studio.project?.output.filename]);

  const chooseSource = async (directory: string) => {
    setPath(directory);
    return studio.changeLibrary(directory);
  };

  const chooseOutput = async (directory: string) => {
    setOutputDirectory(directory);
    return studio.changeOutput(directory, outputFilename);
  };

  const requestOpen = () => {
    if (!selected) return;
    if (studio.dirty) setPendingVideo(selected);
    else void studio.openVideo(selected);
  };

  return (
    <section className="project-strip" aria-label={t("项目与输出位置")} data-guide-surface>
      <div className="project-path-field source-folder-field">
        <Label htmlFor="source-folder">{t("输入视频文件夹")}</Label>
        <div className="project-field-row">
          <Input id="source-folder" data-guide-action="project" value={path} spellCheck={false} onChange={(event) => setPath(event.target.value)} onKeyDown={(event) => {
            if (event.key === "Enter") void studio.changeLibrary(path);
          }} />
          <Button variant="outline" aria-label={t("浏览输入文件夹")} onClick={() => setSourcePickerOpen(true)}><FolderOpen />{t("选择")}</Button>
          <Button variant="outline" aria-label={t("重新扫描输入文件夹")} disabled={studio.analysisStatus.state === "running"} onClick={() => void studio.changeLibrary(path)}><RefreshCw />{t("扫描")}</Button>
        </div>
      </div>

      <div className="project-path-field video-field">
        <Label>{t("比赛视频")}</Label>
        <div className="project-field-row">
          <Select value={selected} onValueChange={setSelected}>
            <SelectTrigger className="min-w-0 flex-1" aria-label={t("选择视频")}><SelectValue placeholder={t("选择一个视频")} /></SelectTrigger>
            <SelectContent>
              {(studio.library?.videos ?? []).map((item) => {
                const tags = [formatFileSize(item.size_bytes)];
                if (item.proxy_available) tags.push(t("快速预览"));
                if (item.latest_auto_cut_available) tags.push(t("最新自动剪辑"));
                else if (item.ground_truth_available) tags.push(t("标准答案"));
                else if (item.completed_marker) tags.push(t("已剪"));
                else if (item.timeline_available) tags.push(t("待校对"));
                else tags.push(t("新项目"));
                return <SelectItem key={item.id} value={item.id}>{`${item.project_name} / ${item.name} · ${tags.join(" · ")}`}</SelectItem>;
              })}
            </SelectContent>
          </Select>
          <Button disabled={!selected || studio.loading || studio.analysisStatus.state === "running"} onClick={requestOpen}><Video />{t("打开")}</Button>
        </div>
      </div>

      <div className="project-path-field output-folder-field">
        <Label htmlFor="output-folder">{t("成片输出文件夹")}</Label>
        <div className="project-field-row">
          <Input id="output-folder" value={outputDirectory} spellCheck={false} onChange={(event) => setOutputDirectory(event.target.value)} />
          <Button variant="outline" aria-label={t("浏览输出文件夹")} disabled={!studio.project} onClick={() => setOutputPickerOpen(true)}><FolderOpen />{t("选择")}</Button>
        </div>
      </div>

      <div className="project-path-field output-name-field">
        <Label htmlFor="output-filename">{t("输出文件名")}</Label>
        <div className="project-field-row">
          <Input id="output-filename" value={outputFilename} spellCheck={false} onChange={(event) => setOutputFilename(event.target.value)} />
          <Button variant="outline" disabled={!studio.project || !outputDirectory || !outputFilename} onClick={() => void studio.changeOutput(outputDirectory, outputFilename)}><Save />{t("应用")}</Button>
        </div>
      </div>

      <div className="project-output-summary">
        {!studio.project?.runtime.ffmpeg.available ? <Badge variant="destructive">{t("FFmpeg / H.264 不可用")}</Badge> : null}
        <span title={studio.project?.output.path}>{t("最终渲染：")}{studio.project?.output.path ?? t("尚未打开视频")}</span>
        {studio.project?.runtime.ffmpeg.warning ? <small>{translateMessage(studio.project.runtime.ffmpeg.warning)}</small> : null}
        {!studio.project?.runtime.ffmpeg.available ? <small>{translateMessage(studio.project?.runtime.ffmpeg.reason) || t("请配置 FFmpeg")}</small> : null}
      </div>

      <DirectoryPicker open={sourcePickerOpen} onOpenChange={setSourcePickerOpen} initialPath={path || studio.library?.path || ""} title={t("选择输入视频文件夹")} description={t("选择包含比赛视频的文件夹。")} onSelect={chooseSource} />
      <DirectoryPicker open={outputPickerOpen} onOpenChange={setOutputPickerOpen} initialPath={outputDirectory || studio.project?.output.directory || path} title={t("选择成片输出文件夹")} description={t("选择保存成片的文件夹。")} onSelect={chooseOutput} />
      <AlertDialog open={Boolean(pendingVideo)} onOpenChange={(open) => { if (!open) setPendingVideo(""); }}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>{t("放弃未保存的时间轴修改？")}</AlertDialogTitle><AlertDialogDescription>{t("打开其他视频会丢弃未保存的边界调整。")}</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>{t("留在当前视频")}</AlertDialogCancel><AlertDialogAction onClick={() => { const next = pendingVideo; setPendingVideo(""); if (next) void studio.openVideo(next); }}>{t("放弃修改并打开")}</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
