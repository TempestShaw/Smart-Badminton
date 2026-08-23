"use client";

import { useEffect, useState } from "react";
import { Activity, Bot, Boxes, Crosshair, ScanSearch, Sparkles } from "lucide-react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { StudioController } from "@/hooks/use-studio-controller";

export type ToolMode = "none" | "calibration" | "shuttle";

export function WorkflowPanel({
  studio,
  toolMode,
  onToolMode,
}: {
  studio: StudioController;
  toolMode: ToolMode;
  onToolMode: (mode: ToolMode) => void;
}) {
  const settings = studio.project?.automatic_analysis.settings;
  const [preroll, setPreroll] = useState(0.35);
  const [postroll, setPostroll] = useState(0.55);
  const [suppressHandoffs, setSuppressHandoffs] = useState(true);
  const [pendingBatch, setPendingBatch] = useState<boolean | null>(null);
  useEffect(() => {
    setPreroll(settings?.preroll ?? 0.35);
    setPostroll(settings?.postroll ?? 0.55);
    setSuppressHandoffs(settings?.suppress_handoffs !== false);
  }, [settings]);
  const automatic = studio.project?.automatic_analysis;
  const shuttle = studio.project?.shuttle_analysis;
  const pose = studio.project?.pose_analysis;
  const ffmpegReady = studio.project?.runtime.ffmpeg.available === true;
  const running = studio.analysisStatus.state === "running";
  const shuttleRunning = running && studio.analysisStatus.mode === "shuttle";
  const visualRunning = running && studio.analysisStatus.mode === "visual";
  const scoreLabelRunning = running && studio.analysisStatus.mode === "score-labels";
  const requestLaunch = (batch: boolean) => {
    if (![preroll, postroll].every((value) => Number.isFinite(value) && value >= 0 && value <= 2)) {
      return studio.notify("前留和后留必须在 0–2 秒之间", true);
    }
    setPendingBatch(batch);
  };
  const launch = () => {
    if (pendingBatch === null) return;
    const batch = pendingBatch;
    setPendingBatch(null);
    void studio.launchAnalysis(batch, { preroll, postroll, suppress_handoffs: suppressHandoffs });
  };
  return (
    <section className="workflow-panel" aria-label="剪片工作流">
      <Accordion type="single" defaultValue="auto" collapsible className="workflow-accordion">
        <AccordionItem value="auto" className="workflow-section">
          <AccordionTrigger className="workflow-trigger">
            <span><small>01</small><Bot /><strong>自动剪片</strong></span>
            <Badge variant={automatic?.configured ? "default" : ffmpegReady ? "outline" : "destructive"}>
              {automatic?.configured ? "已配置" : ffmpegReady ? "待校准" : "FFmpeg 缺失"}
            </Badge>
          </AccordionTrigger>
          <AccordionContent className="workflow-content auto-cut-content">
            {automatic?.configuration_issue ? <p className="configuration-issue">{automatic.configuration_issue}</p> : null}
            <div className="compact-options">
              <Label>前留<Input type="number" min={0} max={2} step={0.05} value={preroll} onChange={(event) => setPreroll(Number(event.target.value))} />秒</Label>
              <Label>后留<Input type="number" min={0} max={2} step={0.05} value={postroll} onChange={(event) => setPostroll(Number(event.target.value))} />秒</Label>
              <Label className="check-option"><Checkbox checked={suppressHandoffs} onCheckedChange={(checked) => setSuppressHandoffs(checked === true)} />过滤送球</Label>
            </div>
            <div className="workflow-actions">
              <Button disabled={!automatic?.configured || running} onClick={() => requestLaunch(false)}><Sparkles />自动分析当前视频</Button>
              <Button variant="outline" disabled={!automatic?.configured || running} onClick={() => requestLaunch(true)}><Boxes />批量分析新比赛</Button>
              <Button variant="outline" disabled={running || !studio.segments.length} onClick={() => void studio.launchScoreLabeling()}>
                <ScanSearch />{scoreLabelRunning ? "标注中" : studio.project?.score_labeling.configured ? "标注未知比分" : "生成终局素材"}
              </Button>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="court" className="workflow-section">
          <AccordionTrigger className="workflow-trigger">
            <span><small>02</small><Crosshair /><strong>场地与轨迹</strong></span>
            <Badge variant={studio.project?.calibration.ready ? "default" : "outline"}>{studio.project?.calibration.ready ? "场地已校准" : "未校准"}</Badge>
          </AccordionTrigger>
          <AccordionContent className="workflow-content tool-grid">
            {shuttle?.available_modes.length ? (
              <Label className="shuttle-mode-select">
                <span>羽球检测</span>
                <Select
                  value={shuttle.mode ?? undefined}
                  disabled={running}
                  onValueChange={(value) => void studio.changeShuttleMode(value as "yolo" | "tracknet" | "hybrid")}
                >
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {shuttle.available_modes.includes("hybrid") ? <SelectItem value="hybrid">Hybrid</SelectItem> : null}
                    {shuttle.available_modes.includes("tracknet") ? <SelectItem value="tracknet">TrackNet</SelectItem> : null}
                    {shuttle.available_modes.includes("yolo") ? <SelectItem value="yolo">YOLO</SelectItem> : null}
                  </SelectContent>
                </Select>
              </Label>
            ) : null}
            <Button size="sm" variant="outline" className={toolMode === "calibration" ? "tool-card active" : "tool-card"} onClick={() => onToolMode(toolMode === "calibration" ? "none" : "calibration")}>
              <Crosshair /><span><strong>框选球场</strong></span>
            </Button>
            <Button size="sm" variant="outline" className={toolMode === "shuttle" ? "tool-card active" : "tool-card"} disabled={!shuttle?.configured && !shuttle?.generated} onClick={() => onToolMode(toolMode === "shuttle" ? "none" : "shuttle")}>
              <ScanSearch /><span><strong>{shuttle?.stale ? "球路需更新" : shuttle?.generated ? "查看／修正球路" : "羽球轨迹"}</strong></span>
            </Button>
            <div className="tool-note shuttle-status-note">
              <div><Badge variant={shuttle?.stale ? "destructive" : shuttle?.generated ? "default" : shuttle?.configured ? "outline" : "destructive"}>{shuttleRunning ? "分析中" : shuttle?.stale ? "需更新" : shuttle?.generated ? "已生成" : shuttle?.configured ? "未分析" : "不可用"}</Badge></div>
              <Button size="xs" variant="secondary" disabled={!shuttle?.configured || running} onClick={() => void studio.launchShuttleAnalysis(Boolean(shuttle?.generated))}><ScanSearch />{shuttleRunning ? "正在分析球路…" : shuttle?.stale ? "按新校准重新分析" : shuttle?.generated ? "重新分析当前球路" : "分析当前视频球路"}</Button>
            </div>
            <div className="tool-note visual-analysis-card">
              <strong>姿态与球路</strong>
              <Button
                size="xs"
                disabled={running || (!pose?.configured && !shuttle?.configured)}
                onClick={() => void studio.launchVisualAnalysis(Boolean(pose?.current && shuttle?.current))}
              >
                <Activity />
                {visualRunning ? "分析中" : pose?.current && shuttle?.current ? "重新分析" : "分析整段视频"}
              </Button>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
      <AlertDialog open={pendingBatch !== null} onOpenChange={(open) => { if (!open) setPendingBatch(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{pendingBatch ? "批量分析所有新比赛？" : "重新自动分析当前视频？"}</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingBatch
                ? "Studio 会处理输入文件夹中的新项目；已有时间表、标准答案或已剪标记的视频会跳过。"
                : `${studio.dirty ? "当前有未保存修改；" : ""}模型结果会替换当前时间轴，并在已有时间表时创建 .bak 备份。`}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>取消</AlertDialogCancel><AlertDialogAction onClick={launch}>{pendingBatch ? "开始批量分析" : "替换并分析"}</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
