// On-device live engine: camera → newest frame → pose worker → tracker → hits and rallies, entirely in the browser.
//
// Speed comes from inferring only where players are. Each tracked player gets one 192 px crop around their last box
// (≈0.3 GFLOP each with YOLO11n-pose) instead of 320 px half-court crops. A player missed for a moment is looked for
// in a wider crop; only a half with an empty or long-lost slot is scanned whole, and each half is re-scanned every
// few seconds to catch someone the crops cannot see. One frame is in flight at a time, and the next one starts as
// soon as a result arrives, so a slow device drops frames rather than falling behind.

import { Court, type Vec2 } from "./court";
import { HitSoundDetector, RallyTracker } from "./rally";
import { courtDetections, PlayerTracker, type TrackedPlayer } from "./tracker";
import type { LiveAnalysis, LivePlayer, MatchFormat } from "@/types/live";

const ROI_SIZE = 192;
const SCAN_SIZE = 320;
const PERIODIC_SCAN_SECONDS = 3;
const MISSING_SCAN_SECONDS = 0.5;

interface WorkerPerson { box: [number, number, number, number]; score: number; keypoints: Float32Array; appearance: Float32Array | null }
interface Crop { x: number; y: number; w: number; h: number }

export interface EngineStats {
  provider: string;
  model: string;
  threads: number;
  fps: number;
  inferenceMs: number;
  preprocessMs: number;
  mode: "roi" | "scan";
  crops: number;
  roiFrames: number;
  scanFrames: number;
}

export interface EngineOptions {
  video: HTMLVideoElement;
  corners: Vec2[];
  matchFormat: MatchFormat;
  /** Folder holding ondevice.json and the ONNX models (default /models). */
  modelBase?: string;
  ortBase?: string;
  workerUrl?: string;
  onUpdate?: () => void;
}

export class OnDeviceEngine {
  stats: EngineStats = { provider: "…", model: "", threads: 1, fps: 0, inferenceMs: 0, preprocessMs: 0, mode: "scan", crops: 0, roiFrames: 0, scanFrames: 0 };
  players: TrackedPlayer[] = [];
  readonly rallies: RallyTracker;
  court: Court | null = null;
  private worker: Worker | null = null;
  private tracker: PlayerTracker | null = null;
  private running = false;
  private busy = false;
  private started = 0;
  private lastScan: Record<"near" | "far", number> = { near: -Infinity, far: -Infinity };
  private frameCounter = 0;
  private processedFrame = -1;
  private frameTimes: number[] = [];
  private nextId = 0;
  private pending = new Map<number, (value: { results: WorkerPerson[][]; inferenceMs: number; preprocessMs: number }) => void>();
  private audioContext: AudioContext | null = null;
  private audioProcessor: ScriptProcessorNode | null = null;
  private audioFrames = 0;

  constructor(private readonly options: EngineOptions) {
    this.rallies = new RallyTracker(() => this.audioFrames > 0);
  }

  async start(stream: MediaStream): Promise<void> {
    const { video, corners, matchFormat } = this.options;
    this.court = new Court(corners, video.videoWidth, video.videoHeight, matchFormat === "doubles");
    this.tracker = new PlayerTracker(matchFormat === "doubles" ? 2 : 1, Math.hypot(video.videoWidth, video.videoHeight));
    this.worker = new Worker(this.options.workerUrl ?? "/ondevice/pose-worker.mjs", { type: "module" });
    const ready = new Promise<{ provider: string; model: string; threads: number }>((resolve, reject) => {
      this.worker!.onmessage = (event) => {
        const message = event.data;
        if (message.type === "ready") resolve(message);
        else if (message.type === "result") this.pending.get(message.id)?.(message);
        else if (message.type === "error") {
          if (message.id === undefined) reject(new Error(message.message));
          else this.pending.get(message.id)?.({ results: [], inferenceMs: 0, preprocessMs: 0 });
        }
        if (message.id !== undefined) this.pending.delete(message.id);
      };
    });
    this.worker.postMessage({
      type: "init",
      modelBase: this.options.modelBase ?? `${location.origin}/models`,
      ortBase: this.options.ortBase ?? `${location.origin}/ort`,
      preferWebgpu: true,
      // Leave a core for camera decoding and the page; inference threads beyond that only contend (measured).
      threads: Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1)),
    });
    const { provider, model, threads } = await ready;
    this.stats = { ...this.stats, provider, model, threads };
    this.startAudio(stream);
    this.started = performance.now();
    this.running = true;
    this.loop();
  }

  stop(): void {
    this.running = false;
    this.worker?.terminate();
    this.worker = null;
    this.audioProcessor?.disconnect();
    void this.audioContext?.close();
  }

  get audioEnabled(): boolean {
    return this.audioFrames > 0;
  }

  private startAudio(stream: MediaStream): void {
    if (!stream.getAudioTracks().length) return;
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(2048, 1, 1);
    const detector = new HitSoundDetector(context.sampleRate);
    processor.onaudioprocess = (event) => {
      const samples = event.inputBuffer.getChannelData(0);
      const endTime = (performance.now() - this.started) / 1000;
      this.audioFrames += 1;
      this.rallies.addSounds(detector.push(new Float32Array(samples), endTime - samples.length / context.sampleRate));
    };
    source.connect(processor);
    processor.connect(context.destination);
    this.audioContext = context;
    this.audioProcessor = processor;
  }

  private crops(time: number): { crops: Crop[]; size: number; mode: "roi" | "scan" } {
    const { video } = this.options;
    const { tracked, missing } = this.tracker!.regions(time);
    // A half with an empty slot is rescanned every 0.5 s (a doubles partner may stay out of view for a while, and
    // the tracked players still need their crops); every half is rescanned every 3 s regardless.
    const due = (["near", "far"] as const).filter((side) => {
      const since = time - this.lastScan[side];
      return since >= PERIODIC_SCAN_SECONDS || (missing.includes(side) && since >= MISSING_SCAN_SECONDS);
    });
    if (!tracked.length) due.push(...(["near", "far"] as const));
    const scan = due.sort((a, b) => this.lastScan[a] - this.lastScan[b])[0];
    if (scan) {
      // One half per frame: half the cost of scanning both, and the other half is only skipped for one frame.
      this.lastScan[scan] = time;
      return { crops: [this.court!.halfBounds(scan)], size: SCAN_SIZE, mode: "scan" };
    }
    const crops = tracked.map(({ box, age }) => {
      const context = age < 0.15 ? 1.5 : 2.3; // just missed: look wider around where they were
      const side = Math.min(Math.max(64, Math.max(box[2] - box[0], box[3] - box[1]) * context), video.videoWidth, video.videoHeight);
      const cx = (box[0] + box[2]) / 2;
      const cy = (box[1] + box[3]) / 2;
      const x = Math.max(0, Math.min(video.videoWidth - side, cx - side / 2));
      const y = Math.max(0, Math.min(video.videoHeight - side, cy - side / 2));
      return { x, y, w: side, h: side };
    });
    return { crops, size: ROI_SIZE, mode: "roi" };
  }

  private loop(): void {
    if (!this.running) return;
    const { video } = this.options;
    if ("requestVideoFrameCallback" in video) {
      // Count presented frames continuously; a finished step starts on the newest frame instead of waiting for one.
      const watch = () => {
        if (!this.running) return;
        this.frameCounter += 1;
        if (!this.busy) void this.step();
        video.requestVideoFrameCallback(watch);
      };
      video.requestVideoFrameCallback(watch);
      this.schedule = () => {
        if (this.running && this.frameCounter > this.processedFrame) void this.step();
      };
    } else {
      this.schedule = () => {
        if (this.running) requestAnimationFrame(() => void this.step());
      };
      this.schedule();
    }
  }

  private schedule: () => void = () => undefined;

  private async step(): Promise<void> {
    if (!this.running || !this.worker || this.busy) return;
    this.busy = true;
    try {
      const { video } = this.options;
      const time = (performance.now() - this.started) / 1000;
      this.processedFrame = this.frameCounter;
      const bitmap = await createImageBitmap(video);
      const { crops, size, mode } = this.crops(time);
      const id = this.nextId++;
      const reply = new Promise<{ results: WorkerPerson[][]; inferenceMs: number; preprocessMs: number }>((resolve) => this.pending.set(id, resolve));
      this.worker.postMessage({ type: "infer", id, bitmap, crops, size, confidence: mode === "roi" ? 0.2 : 0.25 }, [bitmap]);
      const { results, inferenceMs, preprocessMs } = await reply;
      const people = deduplicate(results.flat());
      this.players = this.tracker!.update(courtDetections(this.court!, people), time);
      this.rallies.update(this.players, time);
      const now = performance.now();
      this.frameTimes.push(now);
      while (this.frameTimes.length && now - this.frameTimes[0] > 2000) this.frameTimes.shift();
      const fps = this.frameTimes.length > 1 ? (this.frameTimes.length - 1) / ((now - this.frameTimes[0]) / 1000) : 0;
      this.stats = { ...this.stats, fps, inferenceMs: smooth(this.stats.inferenceMs, inferenceMs), preprocessMs: smooth(this.stats.preprocessMs, preprocessMs), mode, crops: crops.length,
        roiFrames: this.stats.roiFrames + (mode === "roi" ? 1 : 0), scanFrames: this.stats.scanFrames + (mode === "scan" ? 1 : 0) };
      this.options.onUpdate?.();
    } finally {
      this.busy = false;
      this.schedule();
    }
  }

  snapshot(): LiveAnalysis {
    const time = (performance.now() - this.started) / 1000;
    const finished = this.rallies.rallies;
    const durations = finished.map((rally) => (rally.end ?? rally.start) - rally.start);
    const players: LivePlayer[] = [...this.rallies.players.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([playerId, totals]) => {
      const current = this.players.find((player) => player.playerId === playerId);
      const [side, slot] = playerId.split("-");
      return {
        player_id: playerId,
        label: `${side.toUpperCase()} ${slot}`,
        side: side as "near" | "far",
        visible: Boolean(current?.detected),
        speed_meters_per_second: current ? Number(current.speed.toFixed(2)) : null,
        max_speed_meters_per_second: Number(totals.maxSpeed.toFixed(2)),
        distance_meters: Number(totals.distance.toFixed(1)),
        rally_distance_meters: 0,
        swings: totals.swings,
        hits: totals.hits,
        court: current ? [Number(current.court[0].toFixed(2)), Number(current.court[1].toFixed(2))] : null,
        foot: null,
        trail: totals.trail,
      };
    });
    const current = this.rallies.current;
    return {
      time_seconds: time,
      match_format: this.options.matchFormat,
      metric: true,
      shuttle_enabled: false,
      pose_enabled: true,
      model_enabled: false,
      audio_enabled: this.audioEnabled,
      rally: {
        active: current !== null,
        evidence: 0,
        model_probability: null,
        current: current ? { index: current.index, start: current.start, end: null, duration: Math.max(0, time - current.start), shots: current.shots } : null,
      },
      rallies: finished.slice(-30).map((rally) => ({ index: rally.index, start: rally.start, end: rally.end, duration: (rally.end ?? rally.start) - rally.start, shots: rally.shots })),
      totals: {
        rallies: finished.length,
        average_duration: durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : 0,
        longest_duration: durations.length ? Math.max(...durations) : 0,
        shots: finished.reduce((sum, rally) => sum + rally.shots, 0),
      },
      players,
    };
  }
}

function smooth(previous: number, value: number): number {
  return previous ? previous * 0.8 + value * 0.2 : value;
}

/** Overlapping crops can see the same person twice; keep the stronger box. */
function deduplicate(people: WorkerPerson[]): WorkerPerson[] {
  const kept: WorkerPerson[] = [];
  for (const person of [...people].sort((a, b) => b.score - a.score)) {
    if (kept.every((other) => overlap(person.box, other.box) < 0.6)) kept.push(person);
  }
  return kept;
}

function overlap(a: number[], b: number[]): number {
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const smaller = Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
  return smaller > 0 ? (width * height) / smaller : 0;
}

const SKELETON: [number, number][] = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
export const PLAYER_COLOURS: Record<string, string> = { "near-1": "#3ccf7a", "near-2": "#ff9f2e", "far-1": "#4aa3ff", "far-2": "#e374e8" };

/** Draw the court and tracked players onto an overlay canvas sized to the video's displayed frame. */
export function drawOverlay(canvas: HTMLCanvasElement, engine: OnDeviceEngine, video: HTMLVideoElement): void {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth) return;
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(bounds.width * ratio)) { canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio); }
  const scale = Math.min(bounds.width / video.videoWidth, bounds.height / video.videoHeight);
  const offsetX = (bounds.width - video.videoWidth * scale) / 2;
  const offsetY = (bounds.height - video.videoHeight * scale) / 2;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);
  const map = ([x, y]: Vec2): Vec2 => [offsetX + x * scale, offsetY + y * scale];
  if (engine.court) {
    context.strokeStyle = "rgba(255, 80, 80, 0.8)";
    context.lineWidth = 2;
    context.beginPath();
    engine.court.outline().map(map).forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
    context.closePath();
    context.stroke();
  }
  for (const player of engine.players) {
    const colour = PLAYER_COLOURS[player.playerId] ?? "#ffffff";
    const [x1, y1] = map([player.box[0], player.box[1]]);
    const [x2, y2] = map([player.box[2], player.box[3]]);
    context.strokeStyle = colour;
    context.lineWidth = player.detected ? 2 : 1;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.fillStyle = colour;
    const [side, slot] = player.playerId.split("-");
    const label = `${side === "near" ? "N" : "F"}${slot}`;
    context.fillRect(x1, y1 - 18, 26, 18);
    context.fillStyle = "#0b0f0d";
    context.font = "600 12px system-ui, sans-serif";
    context.fillText(label, x1 + 5, y1 - 5);
    if (!player.keypoints) continue;
    const points = player.keypoints;
    context.strokeStyle = colour;
    context.lineWidth = 2.5;
    for (const [a, b] of SKELETON) {
      if (points[a * 3 + 2] < 0.2 || points[b * 3 + 2] < 0.2) continue;
      const [ax, ay] = map([points[a * 3], points[a * 3 + 1]]);
      const [bx, by] = map([points[b * 3], points[b * 3 + 1]]);
      context.beginPath(); context.moveTo(ax, ay); context.lineTo(bx, by); context.stroke();
    }
  }
}
