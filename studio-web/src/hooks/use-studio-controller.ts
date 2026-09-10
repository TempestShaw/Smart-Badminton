"use client";
import { t, translateMessage } from "@/lib/i18n";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast as showToast } from "sonner";

import { apiRequest, mediaUrl } from "@/lib/api";
import { clampBoundary, cloneSegments, frameSnap } from "@/lib/time";
import type {
  AnalysisStatus,
  AnalyticsAction,
  AnalyticsPayload,
  LibraryPayload,
  ProjectPayload,
  RenderStatus,
  ScoreCorrection,
  ScorePayload,
  ScoreRow,
  Segment,
  CalibrationPayload,
  OutputPayload,
} from "@/types/studio";

interface TimelineSaveResponse {
  segments: Segment[];
  backup: string;
  analytics: AnalyticsPayload;
  score: ScorePayload;
}

const API_SCHEMA_VERSION = 8;

function sameTimeline(first: Segment[], second: Segment[]): boolean {
  return first.length === second.length && first.every((segment, index) => {
    const other = second[index];
    return Boolean(other) && Math.abs(segment.start - other.start) < 0.0005 && Math.abs(segment.end - other.end) < 0.0005;
  });
}

export function useStudioController() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const completionKeyRef = useRef("");
  const savedSegmentsRef = useRef<Segment[]>([]);
  const [project, setProject] = useState<ProjectPayload | null>(null);
  const [library, setLibrary] = useState<LibraryPayload | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [dirty, setDirty] = useState(false);
  const [history, setHistory] = useState<Segment[][]>([]);
  const [future, setFuture] = useState<Segment[][]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsPayload>({ available: false });
  const [score, setScore] = useState<ScorePayload>({ available: false });
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>({ state: "idle" });
  const [renderStatus, setRenderStatus] = useState<RenderStatus>({ state: "idle" });
  const [mediaNonce, setMediaNonce] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [scoreCalculating, setScoreCalculating] = useState(false);
  const [scoreReviewActive, setScoreReviewActive] = useState(false);

  const notify = useCallback((text: string, error = false) => {
    if (error) showToast.error(translateMessage(text));
    else showToast.success(translateMessage(text));
  }, []);

  const applyProject = useCallback((payload: ProjectPayload) => {
    const normalized = payload.segments.map((segment) => ({ ...segment, start: Number(segment.start), end: Number(segment.end) }));
    setProject(payload);
    savedSegmentsRef.current = cloneSegments(normalized);
    setSegments(normalized);
    setSelectedIndex(payload.segments.length ? 0 : -1);
    setDirty(false);
    setHistory([]);
    setFuture([]);
    setAnalytics(payload.analytics ?? { available: false });
    setScore(payload.score ?? { available: false });
    setScoreReviewActive(false);
    setMediaNonce(Date.now());
    videoRef.current?.pause();
  }, []);

  useEffect(() => {
    setDirty(!sameTimeline(segments, savedSegmentsRef.current));
  }, [segments]);

  const refreshLibrary = useCallback(async () => {
    const payload = await apiRequest<LibraryPayload>("/api/library");
    setLibrary(payload);
    return payload;
  }, []);

  const reloadProject = useCallback(async () => {
    const payload = await apiRequest<ProjectPayload>("/api/project");
    applyProject(payload);
    return payload;
  }, [applyProject]);

  const refreshProjectMetadata = useCallback(async () => {
    const payload = await apiRequest<ProjectPayload>("/api/project");
    setProject(payload);
    setAnalytics(payload.analytics ?? { available: false });
    setScore(payload.score ?? { available: false });
    setMediaNonce(Date.now());
    return payload;
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([
      apiRequest<{ ok: boolean; api_schema_version: number }>("/api/health"),
      apiRequest<ProjectPayload>("/api/project"),
      apiRequest<LibraryPayload>("/api/library"),
    ])
      .then(([health, projectPayload, libraryPayload]) => {
        if (!active) return;
        if (!health.ok || health.api_schema_version !== API_SCHEMA_VERSION) {
          throw new Error(t("Studio API 版本不兼容：需要 {{value1}}，实际 {{value2}}", { value1: API_SCHEMA_VERSION, value2: health.api_schema_version }));
        }
        applyProject(projectPayload);
        setLibrary(libraryPayload);
      })
      .catch((error: Error) => notify(error.message, true))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [applyProject, notify]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const selectSegment = useCallback(
    (index: number, seek = false) => {
      const safeIndex = Math.max(-1, Math.min(index, segments.length - 1));
      setSelectedIndex(safeIndex);
      if (seek && safeIndex >= 0 && videoRef.current) {
        videoRef.current.currentTime = segments[safeIndex].start;
      }
    },
    [segments],
  );

  const playScoreReviewRally = useCallback((rally: number) => {
    const index = rally - 1;
    const segment = segments[index];
    if (!segment) return;
    setSelectedIndex(index);
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = segment.start;
    window.requestAnimationFrame(() => void video.play().catch(() => undefined));
  }, [segments]);

  const recordChange = useCallback(
    (next: Segment[], nextSelected = selectedIndex) => {
      setHistory((items) => [...items.slice(-79), cloneSegments(segments)]);
      setFuture([]);
      setSegments(next);
      setSelectedIndex(Math.max(-1, Math.min(nextSelected, next.length - 1)));
    },
    [segments, selectedIndex],
  );

  const previewSegments = useCallback((next: Segment[]) => {
    setSegments(next);
  }, []);

  const finishPreviewChange = useCallback((original: Segment[]) => {
    setHistory((items) => [...items.slice(-79), cloneSegments(original)]);
    setFuture([]);
  }, []);

  const editBoundary = useCallback(
    (index: number, edge: "start" | "end", value: number) => {
      if (!project) return;
      recordChange(clampBoundary(segments, index, edge, value, project.video.duration, project.video.fps), index);
    },
    [project, recordChange, segments],
  );

  const undo = useCallback(() => {
    const previous = history.at(-1);
    if (!previous) return;
    setFuture((items) => [...items, cloneSegments(segments)]);
    setSegments(cloneSegments(previous));
    setHistory((items) => items.slice(0, -1));
    setSelectedIndex((index) => Math.min(index, previous.length - 1));
  }, [history, segments]);

  const redo = useCallback(() => {
    const next = future.at(-1);
    if (!next) return;
    setHistory((items) => [...items, cloneSegments(segments)]);
    setSegments(cloneSegments(next));
    setFuture((items) => items.slice(0, -1));
    setSelectedIndex((index) => Math.min(index, next.length - 1));
  }, [future, segments]);

  const addSegment = useCallback(() => {
    if (!project) return;
    const time = frameSnap(videoRef.current?.currentTime ?? 0, project.video.fps);
    let insertion = segments.findIndex((segment) => segment.start > time);
    if (insertion < 0) insertion = segments.length;
    const minimum = insertion > 0 ? segments[insertion - 1].end : 0;
    const maximum = insertion < segments.length ? segments[insertion].start : project.video.duration;
    const start = Math.max(minimum, time);
    const end = Math.min(maximum, start + 2);
    if (end - start < 0.1) return notify(t("播放头附近没有足够的空白区间"), true);
    const next = cloneSegments(segments);
    next.splice(insertion, 0, {
      id: `new-${Date.now()}`,
      start,
      end,
      confidence: "manual",
      review_required: "no",
      boundary_reason: "added in studio",
    });
    recordChange(next, insertion);
  }, [notify, project, recordChange, segments]);

  const splitSegment = useCallback(() => {
    if (!project || selectedIndex < 0) return;
    const segment = segments[selectedIndex];
    const time = frameSnap(videoRef.current?.currentTime ?? 0, project.video.fps);
    if (!segment || time <= segment.start + 0.1 || time >= segment.end - 0.1) {
      return notify(t("请把播放头放在片段内部再分割"), true);
    }
    const next = cloneSegments(segments);
    next[selectedIndex].end = time;
    next.splice(selectedIndex + 1, 0, { ...segment, id: `split-${Date.now()}`, start: time });
    recordChange(next, selectedIndex);
  }, [notify, project, recordChange, segments, selectedIndex]);

  const deleteSegment = useCallback(() => {
    if (selectedIndex < 0) return;
    const next = cloneSegments(segments);
    next.splice(selectedIndex, 1);
    recordChange(next, Math.min(selectedIndex, next.length - 1));
  }, [recordChange, segments, selectedIndex]);

  const saveTimeline = useCallback(async () => {
    if (!project) return false;
    try {
      setSaving(true);
      const payload = await apiRequest<TimelineSaveResponse>("/api/timeline", {
        method: "PUT",
        body: JSON.stringify({ project_id: project.id, segments }),
      });
      savedSegmentsRef.current = cloneSegments(payload.segments);
      setSegments(payload.segments);
      setAnalytics(payload.analytics);
      setScore(payload.score);
      setDirty(false);
      setHistory([]);
      setFuture([]);
      notify(t("时间表已保存"));
      await refreshLibrary();
      return true;
    } catch (error) {
      notify((error as Error).message, true);
      return false;
    } finally {
      setSaving(false);
    }
  }, [notify, project, refreshLibrary, segments]);

  const changeLibrary = useCallback(
    async (path: string) => {
      try {
        const payload = await apiRequest<LibraryPayload>("/api/library", {
          method: "POST",
          body: JSON.stringify({ path }),
        });
        setLibrary(payload);
        notify(t("找到 {{value1}} 个源视频", { value1: payload.videos.length }));
        return true;
      } catch (error) {
        notify((error as Error).message, true);
        return false;
      }
    },
    [notify],
  );

  const changeOutput = useCallback(async (directory: string, filename: string) => {
    if (!project) return false;
    try {
      const payload = await apiRequest<OutputPayload & { ok: boolean }>("/api/output", {
        method: "PUT",
        body: JSON.stringify({ project_id: project.id, directory, filename }),
      });
      setProject((current) => current ? { ...current, output: payload, output_path: payload.path } : current);
      notify(t("输出位置已设置为 {{value1}}", { value1: payload.path }));
      return true;
    } catch (error) {
      notify((error as Error).message, true);
      return false;
    }
  }, [notify, project]);

  const openVideo = useCallback(
    async (id: string) => {
      try {
        setLoading(true);
        const payload = await apiRequest<ProjectPayload>("/api/project/open", {
          method: "POST",
          body: JSON.stringify({ id }),
        });
        applyProject(payload);
        await refreshLibrary();
        notify(t("已打开 {{value1}}", { value1: payload.video.name }));
      } catch (error) {
        notify((error as Error).message, true);
      } finally {
        setLoading(false);
      }
    },
    [applyProject, notify, refreshLibrary],
  );

  const updateScore = useCallback(
    async (values: Partial<Omit<ScoreCorrection, "rally" | "note">>) => {
      if (!project || selectedIndex < 0) return;
      const rally = selectedIndex + 1;
      if (dirty || !(score.rallies ?? []).some((row) => row.rally === rally)) {
        notify(t("请先保存时间表，再修改与当前片段一致的比分"), true);
        return;
      }
      const current = score.corrections?.find((row) => row.rally === rally) ?? {
        rally,
        winner: "auto" as const,
        server_override: "unknown" as const,
        server: "unknown" as const,
        last_hitter: "unknown" as const,
        terminal_event: "unknown" as const,
        landing_side: "unknown" as const,
        post_rally_event: "unknown" as const,
        note: "studio",
      };
      const replacement = { ...current, ...values, rally };
      const corrections = (score.corrections ?? []).filter((row) => row.rally !== rally);
      if (
        replacement.winner !== "auto"
        || replacement.server_override !== "unknown"
        || replacement.server !== "unknown"
        || replacement.last_hitter !== "unknown"
        || replacement.terminal_event !== "unknown"
        || replacement.landing_side !== "unknown"
        || replacement.post_rally_event !== "unknown"
      ) corrections.push(replacement);
      corrections.sort((first, second) => first.rally - second.rally);
      try {
        const payload = await apiRequest<ScorePayload>("/api/score", {
          method: "PUT",
          body: JSON.stringify({ project_id: project.id, corrections }),
        });
        setScore(payload);
        if (scoreReviewActive && replacement.winner !== "auto") {
          const unresolved = (payload.rallies ?? [])
            .filter((row) => row.winner_source === "unresolved")
            .map((row) => row.rally);
          if (!unresolved.length) {
            setScoreReviewActive(false);
            videoRef.current?.pause();
            notify(t("比分标注完成"));
          } else {
            const next = unresolved.find((candidate) => candidate > rally) ?? unresolved[0];
            playScoreReviewRally(next);
          }
        } else {
          notify(t("比分已保存"));
        }
        return true;
      } catch (error) {
        notify((error as Error).message, true);
        return false;
      }
    },
    [dirty, notify, playScoreReviewRally, project, score.corrections, score.rallies, scoreReviewActive, selectedIndex],
  );

  const calculateScore = useCallback(async () => {
    if (!project) return false;
    if (dirty && !(await saveTimeline())) return false;
    try {
      setScoreCalculating(true);
      const payload = await apiRequest<ScorePayload>("/api/score/analyze", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id }),
      });
      setScore(payload);
      notify(t("比分计算完成：{{value1}}/{{value2}} 分已识别", { value1: payload.resolved ?? 0, value2: segments.length }));
      return true;
    } catch (error) {
      notify((error as Error).message, true);
      return false;
    } finally {
      setScoreCalculating(false);
    }
  }, [dirty, notify, project, saveTimeline, segments.length]);

  const launchScoreLabeling = useCallback(async () => {
    if (!project) return;
    if (dirty && !(await saveTimeline())) return;
    try {
      const payload = await apiRequest<AnalysisStatus>("/api/score-labels/analyze", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id }),
      });
      completionKeyRef.current = "";
      setAnalysisStatus(payload);
      notify(payload.state === "complete" ? payload.label ?? t("没有待标注回合") : t("终局标注已开始"));
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [dirty, notify, project, saveTimeline]);

  const reviewScoreLabel = useCallback(async (rally: number, decision: "accepted" | "rejected") => {
    if (!project) return;
    try {
      const payload = await apiRequest<{ score: ScorePayload; score_labeling: ProjectPayload["score_labeling"] }>("/api/score-labels/review", {
        method: "PUT",
        body: JSON.stringify({ project_id: project.id, rally, decision }),
      });
      setScore(payload.score);
      setProject((current) => current ? { ...current, score: payload.score, score_labeling: payload.score_labeling } : current);
      notify(decision === "accepted" ? t("建议已采用") : t("建议已忽略"));
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [notify, project]);

  const refreshAnalyticsAndScore = useCallback(async () => {
    try {
      const [analyticsPayload, scorePayload] = await Promise.all([
        apiRequest<AnalyticsPayload>("/api/analytics"),
        apiRequest<ScorePayload>("/api/score"),
      ]);
      setAnalytics(analyticsPayload);
      setScore(scorePayload);
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [notify]);

  const applyCalibrationResult = useCallback((payload: CalibrationPayload) => {
    setProject((current) => current ? {
      ...current,
      calibration: { ready: payload.ready, path: payload.path },
      automatic_analysis: {
        ...current.automatic_analysis,
        configured: payload.automatic_analysis_configured ?? current.automatic_analysis.configured,
        pose_overlay_configured: payload.pose_overlay_configured ?? current.automatic_analysis.pose_overlay_configured,
        trajectory_boundary_protection: payload.shuttle_analysis?.current
          ?? current.automatic_analysis.trajectory_boundary_protection,
      },
      shuttle_analysis: payload.shuttle_analysis ?? current.shuttle_analysis,
    } : current);
  }, []);

  const launchAnalysis = useCallback(
    async (batch: boolean, options: { preroll: number; postroll: number; suppress_handoffs: boolean }) => {
      if (!project) return;
      try {
        const payload = await apiRequest<AnalysisStatus>(batch ? "/api/analyze/batch" : "/api/analyze", {
          method: "POST",
          body: JSON.stringify(options),
        });
        completionKeyRef.current = "";
        setAnalysisStatus(payload);
        notify(batch ? t("批量自动分析已开始") : t("当前视频已开始自动分析"));
      } catch (error) {
        notify((error as Error).message, true);
      }
    },
    [notify, project],
  );

  const launchShuttleAnalysis = useCallback(async (force = false) => {
    if (!project) return;
    try {
      const payload = await apiRequest<AnalysisStatus>("/api/analyze/shuttle", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id, force }),
      });
      completionKeyRef.current = "";
      setAnalysisStatus(payload);
      notify(payload.state === "complete" ? t("球路已是最新") : force ? t("正在重新分析球路") : t("球路分析已开始"));
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [notify, project]);

  const launchVisualAnalysis = useCallback(async (force = false) => {
    if (!project) return;
    try {
      const payload = await apiRequest<AnalysisStatus>("/api/analyze/visual", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id, force }),
      });
      completionKeyRef.current = "";
      setAnalysisStatus(payload);
      notify(payload.state === "complete" ? t("视觉分析已是最新") : t("视觉分析已开始"));
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [notify, project]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const payload = await apiRequest<AnalysisStatus & { results?: unknown[] }>("/api/analyze");
        if (cancelled) return;
        setAnalysisStatus(payload);
        if (payload.state === "complete") {
          const key = JSON.stringify(payload.results ?? []);
          if (key && key !== completionKeyRef.current) {
            completionKeyRef.current = key;
            const visualOnly = payload.mode === "shuttle" || payload.mode === "visual" || payload.mode === "score-labels";
            await Promise.all([visualOnly ? refreshProjectMetadata() : reloadProject(), refreshLibrary()]);
          }
        } else if (payload.state === "error") {
          const key = `${payload.job_id ?? "analysis"}:${payload.message ?? payload.label ?? "error"}`;
          if (key !== completionKeyRef.current) {
            completionKeyRef.current = key;
            notify(payload.message ?? payload.label ?? t("后台分析失败"), true);
          }
        }
      } catch (error) {
        if (!cancelled) notify((error as Error).message, true);
      }
    };
    void poll();
    const timer = analysisStatus.state === "running" ? window.setInterval(poll, 1200) : null;
    const resumePolling = () => {
      if (!document.hidden) void poll();
    };
    document.addEventListener("visibilitychange", resumePolling);
    return () => {
      cancelled = true;
      if (timer !== null) window.clearInterval(timer);
      document.removeEventListener("visibilitychange", resumePolling);
    };
  }, [analysisStatus.state, notify, refreshLibrary, refreshProjectMetadata, reloadProject]);

  const startRender = useCallback(async (
    includeTrajectory = false,
    winnerFilter: "all" | "near" | "far" = "all",
    includeScore = false,
  ) => {
    if (dirty && !(await saveTimeline())) return;
    try {
      const payload = await apiRequest<RenderStatus>("/api/render", {
        method: "POST",
        body: JSON.stringify({
          include_trajectory: includeTrajectory,
          winner_filter: winnerFilter,
          include_score: includeScore,
        }),
      });
      setRenderStatus(payload);
      notify(t("开始输出成片"));
    } catch (error) {
      notify((error as Error).message, true);
    }
  }, [dirty, notify, saveTimeline]);

  useEffect(() => {
    if (renderStatus.state !== "running") return;
    const timer = window.setInterval(async () => {
      try {
        const payload = await apiRequest<RenderStatus>("/api/render");
        setRenderStatus(payload);
        if (payload.state === "complete") {
          setProject((current) => current ? { ...current, output: { ...current.output, file_exists: true } } : current);
          notify(t("输出完成：{{value1}}", { value1: payload.output }));
        }
        if (payload.state === "error") notify(payload.message || t("输出失败"), true);
      } catch (error) {
        notify((error as Error).message, true);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [notify, renderStatus.state]);

  const analyticsMap = useMemo(
    () => new Map<number, AnalyticsAction>((analytics.rallies ?? []).map((row) => [Number(row.rally), row])),
    [analytics.rallies],
  );
  const scoreMap = useMemo(
    () => new Map<number, ScoreRow>((score.rallies ?? []).map((row) => [Number(row.rally), row])),
    [score.rallies],
  );
  const scoreSuggestionMap = useMemo(
    () => new Map((project?.score_labeling.suggestions ?? []).map((row) => [Number(row.rally), row])),
    [project?.score_labeling.suggestions],
  );
  const scoreReviewRallies = useMemo(
    () => (score.rallies ?? [])
      .filter((row) => row.winner_source === "unresolved")
      .map((row) => Number(row.rally)),
    [score.rallies],
  );
  const startScoreReview = useCallback(() => {
    if (dirty) {
      notify(t("请先保存时间表"), true);
      return;
    }
    if (!scoreReviewRallies.length) {
      notify(t("没有待标注比分"));
      return;
    }
    const currentRally = selectedIndex + 1;
    const first = scoreReviewRallies.includes(currentRally) ? currentRally : scoreReviewRallies[0];
    setScoreReviewActive(true);
    playScoreReviewRally(first);
  }, [dirty, notify, playScoreReviewRally, scoreReviewRallies, selectedIndex]);
  const stopScoreReview = useCallback(() => {
    setScoreReviewActive(false);
    videoRef.current?.pause();
  }, []);
  const moveScoreReview = useCallback((direction: -1 | 1) => {
    if (!scoreReviewRallies.length) return;
    const currentRally = selectedIndex + 1;
    const current = scoreReviewRallies.indexOf(currentRally);
    const next = current < 0
      ? 0
      : (current + direction + scoreReviewRallies.length) % scoreReviewRallies.length;
    playScoreReviewRally(scoreReviewRallies[next]);
  }, [playScoreReviewRally, scoreReviewRallies, selectedIndex]);

  return {
    videoRef,
    project,
    library,
    segments,
    selectedIndex,
    selectedSegment: segments[selectedIndex] ?? null,
    dirty,
    history,
    future,
    analytics,
    analyticsMap,
    score,
    scoreMap,
    scoreSuggestionMap,
    scoreReview: {
      active: scoreReviewActive,
      remaining: scoreReviewRallies.length,
      rallies: scoreReviewRallies,
    },
    analysisStatus,
    renderStatus,
    mediaSource: project ? mediaUrl(`/media/video?v=${mediaNonce}`) : null,
    loading,
    saving,
    scoreCalculating,
    notify,
    selectSegment,
    recordChange,
    previewSegments,
    finishPreviewChange,
    editBoundary,
    undo,
    redo,
    addSegment,
    splitSegment,
    deleteSegment,
    saveTimeline,
    changeLibrary,
    changeOutput,
    openVideo,
    refreshProject: reloadProject,
    updateScore,
    calculateScore,
    startScoreReview,
    stopScoreReview,
    moveScoreReview,
    launchScoreLabeling,
    reviewScoreLabel,
    refreshAnalyticsAndScore,
    applyCalibrationResult,
    launchAnalysis,
    launchShuttleAnalysis,
    launchVisualAnalysis,
    startRender,
  };
}

export type StudioController = ReturnType<typeof useStudioController>;
