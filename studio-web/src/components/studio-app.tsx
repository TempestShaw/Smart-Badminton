"use client";

import { useCallback, useEffect, useState } from "react";

import { AppHeader } from "@/components/app-header";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { ClipBin } from "@/components/clip-bin";
import { ProjectBrowser } from "@/components/project-browser";
import { RallyInspector } from "@/components/rally-inspector";
import { TimelineEditor } from "@/components/timeline-editor";
import { VideoWorkspace } from "@/components/video-workspace";
import { WorkflowPanel, type ToolMode } from "@/components/workflow-panel";
import { useStudioController } from "@/hooks/use-studio-controller";

export function StudioApp({ onSwitchEngine }: { onSwitchEngine?: () => void }) {
  const studio = useStudioController();
  const [toolMode, setToolMode] = useState<ToolMode>("none");
  const [toolDirty, setToolDirty] = useState(false);
  const [pendingToolMode, setPendingToolMode] = useState<ToolMode | null>(null);
  const { editBoundary, project, selectedIndex, selectedSegment, videoRef } = studio;

  const changeToolMode = useCallback((next: ToolMode) => {
    if (toolDirty && next !== toolMode) {
      setPendingToolMode(next);
      return;
    }
    setToolDirty(false);
    setToolMode(next);
  }, [toolDirty, toolMode]);

  const discardToolChanges = () => {
    if (pendingToolMode === null) return;
    setToolDirty(false);
    setToolMode(pendingToolMode);
    setPendingToolMode(null);
  };

  useEffect(() => {
    setToolMode("none");
    setToolDirty(false);
  }, [studio.project?.id]);

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, select, textarea, button, dialog")) return;
      const video = videoRef.current;
      if (!video || !project) return;
      const step = 1 / Math.max(1, project.video.fps);
      if (event.code === "Space") {
        event.preventDefault();
        if (video.paused) void video.play(); else video.pause();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        video.pause();
        video.currentTime = Math.max(0, Math.min(project.video.duration, video.currentTime + (event.key === "ArrowLeft" ? -step : step)));
      } else if (event.key === "[" && selectedSegment) {
        event.preventDefault();
        editBoundary(selectedIndex, "start", video.currentTime);
      } else if (event.key === "]" && selectedSegment) {
        event.preventDefault();
        editBoundary(selectedIndex, "end", video.currentTime);
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  }, [editBoundary, project, selectedIndex, selectedSegment, videoRef]);

  return (
    <div className="native-studio-app">
      <AppHeader studio={studio} onSwitchEngine={onSwitchEngine} />
      <ProjectBrowser studio={studio} />
      <WorkflowPanel studio={studio} toolMode={toolMode} onToolMode={changeToolMode} />
      <main className="studio-shell" aria-busy={studio.loading}>
        <VideoWorkspace studio={studio} toolMode={toolMode} onToolDirty={setToolDirty} onCloseTool={() => changeToolMode("none")} />
        <RallyInspector studio={studio} />
        <TimelineEditor studio={studio} />
        <ClipBin studio={studio} />
      </main>
      <AlertDialog open={pendingToolMode !== null} onOpenChange={(open) => { if (!open) setPendingToolMode(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>放弃工具中的未保存修改？</AlertDialogTitle><AlertDialogDescription>退出后，这次尚未保存的场地或羽球标注修改会丢失；已保存的数据不受影响。</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>继续编辑</AlertDialogCancel><AlertDialogAction onClick={discardToolChanges}>放弃并退出</AlertDialogAction></AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {studio.loading ? <div className="loading-overlay"><span className="loader" /><strong>正在载入本地项目…</strong></div> : null}
    </div>
  );
}
