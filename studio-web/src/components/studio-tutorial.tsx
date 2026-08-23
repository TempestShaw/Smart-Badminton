"use client";

import { useEffect, useRef, useState } from "react";
import {
  BookOpenText,
  Calculator,
  Check,
  Clapperboard,
  Crosshair,
  FolderOpen,
  Scissors,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import Markdown from "react-markdown";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { StudioController } from "@/hooks/use-studio-controller";
import { apiRequest } from "@/lib/api";

const ONBOARDING_KEY = "smart-badminton:onboarding:v1";

interface GuideStep {
  id: "project" | "court" | "analyze" | "timeline" | "score" | "export";
  title: string;
  description: string;
  outcome: string;
  action: string;
  icon: LucideIcon;
}

const guideSteps: GuideStep[] = [
  { id: "project", title: "选择视频", description: "选择比赛文件夹，再打开一条视频。", outcome: "播放器载入源视频", action: "选择视频", icon: FolderOpen },
  { id: "court", title: "校准球场", description: "沿主球场边线框选有效区域。", outcome: "分析聚焦本场比赛", action: "打开场地校准", icon: Crosshair },
  { id: "analyze", title: "自动剪片", description: "生成每个回合的候选边界。", outcome: "时间轴出现候选片段", action: "查看自动剪片", icon: Sparkles },
  { id: "timeline", title: "校对回合", description: "逐球播放，拖动片段两端修正边界。", outcome: "保存最终时间表", action: "前往时间轴", icon: Scissors },
  { id: "score", title: "计算比分", description: "剪辑完成后一次计算，并修正未知回合。", outcome: "可按得分方筛选输出", action: "前往比分", icon: Calculator },
  { id: "export", title: "输出成片", description: "选择比分与轨迹图层，输出高清成片。", outcome: "获得最终 MP4", action: "打开输出设置", icon: Clapperboard },
];

function focusGuideTarget(step: GuideStep) {
  const focus = (selector: string, click = false) => {
    const target = document.querySelector<HTMLElement>(selector);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    if (click) target.click();
    else target.focus({ preventScroll: true });
    const surface = target.closest<HTMLElement>("[data-guide-surface]") ?? target;
    surface.classList.add("guide-focus");
    window.setTimeout(() => surface.classList.remove("guide-focus"), 1500);
  };
  const openSection = (selector: string) => {
    const trigger = document.querySelector<HTMLElement>(selector);
    if (!trigger) return;
    trigger.scrollIntoView({ behavior: "smooth", block: "center" });
    if (trigger.getAttribute("aria-expanded") !== "true") trigger.click();
  };

  if (step.id === "court") {
    openSection('[data-guide-action="court-section"]');
    window.setTimeout(() => focus('[data-guide-action="calibration"]', true), 180);
    return;
  }
  if (step.id === "analyze") {
    openSection('[data-guide-action="auto-section"]');
    window.setTimeout(() => focus('[data-guide-action="analyze"]'), 180);
    return;
  }
  if (step.id === "export") {
    focus('[data-guide-action="export"]', true);
    return;
  }
  focus(`[data-guide-action="${step.id}"]`);
}

export function StudioTutorial({ studio }: { studio: StudioController }) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<"guide" | "manual">("guide");
  const [selected, setSelected] = useState(0);
  const [markdown, setMarkdown] = useState("");
  const [error, setError] = useState("");
  const autoOpened = useRef(false);
  const projectId = studio.project?.id;
  const libraryPath = studio.library?.path;
  const activeVideo = studio.library?.videos.find((video) => video.active);
  const completed = [
    Boolean(studio.project),
    studio.project?.calibration.ready === true,
    studio.project?.evidence.available === true && studio.segments.length > 0,
    Boolean(activeVideo?.ground_truth_available || activeVideo?.completed_marker),
    studio.score.available === true && !studio.dirty,
    studio.project?.output.file_exists === true,
  ];
  const completeCount = completed.filter(Boolean).length;
  const firstIncomplete = completed.findIndex((value) => !value);
  const currentStep = firstIncomplete === -1 ? guideSteps.length - 1 : firstIncomplete;

  useEffect(() => {
    if (studio.loading || !projectId || !libraryPath || autoOpened.current) return;
    autoOpened.current = true;
    if (window.localStorage.getItem(ONBOARDING_KEY) !== "1") {
      setSelected(currentStep);
      setOpen(true);
    }
  }, [currentStep, libraryPath, projectId, studio.loading]);

  useEffect(() => {
    if (!open || view !== "manual" || markdown || error) return;
    let active = true;
    apiRequest<{ markdown: string }>("/api/tutorial")
      .then((payload) => {
        if (active) setMarkdown(payload.markdown);
      })
      .catch((reason: Error) => {
        if (active) setError(reason.message);
      });
    return () => {
      active = false;
    };
  }, [error, markdown, open, view]);

  const changeOpen = (next: boolean) => {
    if (next) {
      setView("guide");
      setSelected(currentStep);
    }
    setOpen(next);
    if (!next) window.localStorage.setItem(ONBOARDING_KEY, "1");
  };
  const goToStep = () => {
    const targetStep = guideSteps[selected];
    changeOpen(false);
    window.setTimeout(() => focusGuideTarget(targetStep), 160);
  };
  const step = guideSteps[selected];
  const StepIcon = step.icon;

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="ghost" className="tutorial-tab"><BookOpenText />使用指南</Button>
      </DialogTrigger>
      <DialogContent className="tutorial-dialog guide-dialog max-w-5xl">
        <DialogHeader className="guide-header">
          <div>
            <DialogTitle>{view === "guide" ? "六步完成第一条成片" : "Studio 完整手册"}</DialogTitle>
            <DialogDescription>{view === "guide" ? `${completeCount} / 6 已完成` : "按功能查阅详细操作。"}</DialogDescription>
          </div>
          <div className="guide-view-switch" aria-label="指南视图">
            <Button size="sm" variant={view === "guide" ? "default" : "ghost"} onClick={() => setView("guide")}>快速开始</Button>
            <Button size="sm" variant={view === "manual" ? "default" : "ghost"} onClick={() => setView("manual")}>完整手册</Button>
          </div>
        </DialogHeader>

        {view === "guide" ? (
          <div className="guide-content">
            <Progress className="guide-progress" value={completeCount / guideSteps.length * 100} aria-label={`已完成 ${completeCount} 个步骤`} />
            <nav className="guide-steps" aria-label="Studio 使用步骤">
              {guideSteps.map((item, index) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    type="button"
                    className={`guide-step${selected === index ? " selected" : ""}${completed[index] ? " complete" : ""}`}
                    aria-current={selected === index ? "step" : undefined}
                    onClick={() => setSelected(index)}
                  >
                    <span className="guide-step-number">{index + 1}</span>
                    <span><strong>{item.title}</strong></span>
                    {completed[index] ? <Check aria-hidden="true" /> : <Icon aria-hidden="true" />}
                  </button>
                );
              })}
            </nav>

            <section className="guide-detail" aria-live="polite">
              <div className="guide-detail-icon"><StepIcon aria-hidden="true" /></div>
              <div className="guide-detail-copy">
                <span className="eyebrow">STEP {String(selected + 1).padStart(2, "0")}</span>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
              <div className="guide-outcome"><span>完成后</span><strong>{step.outcome}</strong></div>
            </section>

            <footer className="guide-footer">
              <Button variant="ghost" disabled={selected === 0} onClick={() => setSelected((value) => Math.max(0, value - 1))}>上一步</Button>
              <div className="guide-footer-actions">
                <Button variant="outline" onClick={() => setSelected((value) => Math.min(guideSteps.length - 1, value + 1))} disabled={selected === guideSteps.length - 1}>下一步</Button>
                <Button onClick={goToStep}>{step.action}</Button>
              </div>
            </footer>
          </div>
        ) : (
          <ScrollArea className="tutorial-scroll">
            {error ? <div className="tutorial-error"><strong>手册载入失败</strong><span>{error}</span></div> : null}
            {!error && !markdown ? <div className="tutorial-loading"><Skeleton /><Skeleton /><Skeleton /></div> : null}
            {markdown ? <article className="markdown-doc"><Markdown>{markdown}</Markdown></article> : null}
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  );
}
