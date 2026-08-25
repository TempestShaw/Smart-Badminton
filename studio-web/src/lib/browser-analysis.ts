import type { Segment } from "@/types/studio";

export interface MotionSample {
  time: number;
  motion: number;
}

export interface BrowserAnalysisResult {
  duration: number;
  width: number;
  height: number;
  sampleRate: number;
  threshold: number;
  samples: MotionSample[];
  segments: Segment[];
}

export interface AnalysisProgress {
  progress: number;
  message: string;
}

function percentile(values: number[], ratio: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * ratio)))];
}

export function deriveMotionThreshold(samples: MotionSample[]): number {
  const values = samples.slice(1).map((sample) => sample.motion);
  const noiseFloor = percentile(values, 0.2);
  const median = percentile(values, 0.5);
  const upperQuartile = percentile(values, 0.75);
  const adaptiveOffset = Math.max(0.004, (median - noiseFloor) * 0.5);
  return Math.max(0.014, noiseFloor + adaptiveOffset, upperQuartile * 0.68);
}

export function buildQuickSegments(
  samples: MotionSample[],
  duration: number,
  threshold = deriveMotionThreshold(samples),
): Segment[] {
  if (samples.length < 2 || duration <= 0) return [];
  const smoothed = samples.map((sample, index) => {
    const left = samples[Math.max(0, index - 1)].motion;
    const right = samples[Math.min(samples.length - 1, index + 1)].motion;
    return { time: sample.time, motion: left * 0.25 + sample.motion * 0.5 + right * 0.25 };
  });
  const step = Math.max(0.1, smoothed[1].time - smoothed[0].time);
  const closeAfter = Math.max(1.4, step * 4);
  const raw: Segment[] = [];
  let start: number | null = null;
  let lastActive = 0;

  for (const sample of smoothed) {
    const active = sample.motion >= threshold;
    if (active) {
      start ??= sample.time;
      lastActive = sample.time;
    } else if (start !== null && sample.time - lastActive >= closeAfter) {
      const clipStart = Math.max(0, start - 0.35);
      const clipEnd = Math.min(duration, lastActive + 0.7);
      if (clipEnd - clipStart >= 1.4) raw.push({ start: clipStart, end: clipEnd, confidence: "quick" });
      start = null;
    }
  }
  if (start !== null) {
    const clipStart = Math.max(0, start - 0.35);
    const clipEnd = Math.min(duration, lastActive + 0.7);
    if (clipEnd - clipStart >= 1.4) raw.push({ start: clipStart, end: clipEnd, confidence: "quick" });
  }

  const merged: Segment[] = [];
  for (const segment of raw) {
    const previous = merged.at(-1);
    if (previous && segment.start - previous.end < 0.8) previous.end = Math.max(previous.end, segment.end);
    else merged.push({ ...segment });
  }
  return merged.map((segment, index) => ({ ...segment, id: `Q${String(index + 1).padStart(3, "0")}` }));
}

function waitForMedia(video: HTMLVideoElement, eventName: "loadedmetadata" | "seeked", signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(new DOMException("Analysis cancelled", "AbortError"));
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener(eventName, ready);
      video.removeEventListener("error", failed);
      signal.removeEventListener("abort", cancelled);
    };
    const ready = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error("无法读取这个视频")); };
    const cancelled = () => { cleanup(); reject(new DOMException("Analysis cancelled", "AbortError")); };
    video.addEventListener(eventName, ready, { once: true });
    video.addEventListener("error", failed, { once: true });
    signal.addEventListener("abort", cancelled, { once: true });
  });
}

async function seek(video: HTMLVideoElement, time: number, signal: AbortSignal): Promise<void> {
  if (Math.abs(video.currentTime - time) < 0.001 && video.readyState >= 2) return;
  const ready = waitForMedia(video, "seeked", signal);
  video.currentTime = time;
  await ready;
}

export async function analyzeVideoInBrowser(
  file: File,
  onProgress: (progress: AnalysisProgress) => void,
  signal: AbortSignal,
): Promise<BrowserAnalysisResult> {
  const sourceUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;
  video.src = sourceUrl;
  try {
    if (video.readyState < 1) await waitForMedia(video, "loadedmetadata", signal);
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    if (!duration) throw new Error("无法读取视频时长");
    const sampleRate = duration > 30 * 60 ? 1 : duration > 15 * 60 ? 1.5 : 2.5;
    const sampleCount = Math.max(2, Math.ceil(duration * sampleRate));
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = Math.max(54, Math.round(128 * video.videoHeight / Math.max(1, video.videoWidth)));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("浏览器无法创建视频分析画布");
    const samples: MotionSample[] = [];
    let previous: Uint8Array | null = null;

    for (let index = 0; index < sampleCount; index += 1) {
      if (signal.aborted) throw new DOMException("Analysis cancelled", "AbortError");
      const time = Math.min(duration - 0.001, index / sampleRate);
      await seek(video, time, signal);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const gray = new Uint8Array(canvas.width * canvas.height);
      for (let pixel = 0, source = 0; pixel < gray.length; pixel += 1, source += 4) {
        gray[pixel] = (rgba[source] * 3 + rgba[source + 1] * 6 + rgba[source + 2]) / 10;
      }
      let motion = 0;
      if (previous) {
        for (let pixel = 0; pixel < gray.length; pixel += 2) motion += Math.abs(gray[pixel] - previous[pixel]);
        motion /= Math.ceil(gray.length / 2) * 255;
      }
      samples.push({ time, motion });
      previous = gray;
      if (index % 5 === 0 || index === sampleCount - 1) {
        onProgress({ progress: (index + 1) / sampleCount, message: `分析画面 ${Math.round((index + 1) / sampleCount * 100)}%` });
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    }
    const threshold = deriveMotionThreshold(samples);
    return {
      duration,
      width: video.videoWidth,
      height: video.videoHeight,
      sampleRate,
      threshold,
      samples,
      segments: buildQuickSegments(samples, duration, threshold),
    };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(sourceUrl);
  }
}
