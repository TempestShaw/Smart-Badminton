"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest, apiUrl } from "@/lib/api";
import { t } from "@/lib/i18n";
import type { LiveStartOptions, LiveStatus, MatchFormat } from "@/types/live";
import type { Point } from "@/types/studio";

const FRAME_INTERVAL_MS = 125; // ~8 fps upload; the server analyses the newest frame it has
const FRAME_WIDTH = 960;
const AUDIO_RATE = 16000;

export const RUNNING_STATES = new Set(["starting", "running"]);

/** Browser camera: frames and microphone PCM go to Studio; a full-quality recording stays local until Stop. */
class BrowserCamera {
  private stream: MediaStream | null = null;
  private video = document.createElement("video");
  private canvas = document.createElement("canvas");
  private timer: number | null = null;
  private sending = false;
  private started = performance.now();
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private pcm: Int16Array[] = [];
  private pcmLength = 0;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private recordedType = "video/webm";

  async start(record: boolean): Promise<void> {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error(t("手机摄像头需要 HTTPS：用 --https 启动 Studio，再在手机上打开 https 地址。"));
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.srcObject = this.stream;
    await this.video.play();
    this.started = performance.now();
    this.timer = window.setInterval(() => void this.sendFrame(), FRAME_INTERVAL_MS);
    this.startAudio();
    if (record && typeof MediaRecorder !== "undefined") {
      const types = ["video/mp4;codecs=avc1,mp4a.40.2", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
      this.recordedType = types.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
      this.recorder = new MediaRecorder(this.stream, this.recordedType ? { mimeType: this.recordedType, videoBitsPerSecond: 4_000_000 } : undefined);
      this.recorder.ondataavailable = (event) => { if (event.data.size) this.chunks.push(event.data); };
      this.recorder.start(1000);
    }
  }

  private async sendFrame(): Promise<void> {
    if (this.sending || this.video.readyState < 2 || !this.video.videoWidth) return;
    this.sending = true;
    try {
      const scale = Math.min(1, FRAME_WIDTH / this.video.videoWidth);
      this.canvas.width = Math.round(this.video.videoWidth * scale);
      this.canvas.height = Math.round(this.video.videoHeight * scale);
      this.canvas.getContext("2d")?.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => this.canvas.toBlob(resolve, "image/jpeg", 0.72));
      if (blob) {
        await fetch(apiUrl("/api/live/frame"), {
          method: "POST",
          body: blob,
          headers: { "Content-Type": "image/jpeg", "X-Capture-Time": ((performance.now() - this.started) / 1000).toFixed(3) },
        });
      }
      this.flushAudio();
    } catch {
      // A dropped frame is replaced by the next one; the status poll reports a stopped session.
    } finally {
      this.sending = false;
    }
  }

  private startAudio(): void {
    if (!this.stream?.getAudioTracks().length) return;
    const context = new AudioContext();
    const source = context.createMediaStreamSource(this.stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const ratio = context.sampleRate / AUDIO_RATE;
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const output = new Int16Array(Math.floor(input.length / ratio));
      for (let index = 0; index < output.length; index += 1) {
        const sample = input[Math.floor(index * ratio)] ?? 0;
        output[index] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
      }
      this.pcm.push(output);
      this.pcmLength += output.length;
    };
    source.connect(processor);
    processor.connect(context.destination);
    this.audioContext = context;
    this.processor = processor;
  }

  private flushAudio(): void {
    if (this.pcmLength < AUDIO_RATE / 5) return;
    const merged = new Int16Array(this.pcmLength);
    let offset = 0;
    for (const chunk of this.pcm) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.pcm = [];
    this.pcmLength = 0;
    void fetch(apiUrl("/api/live/audio"), { method: "POST", body: merged.buffer, headers: { "Content-Type": "application/octet-stream" } });
  }

  async stop(): Promise<{ blob: Blob; extension: string } | null> {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.processor?.disconnect();
    void this.audioContext?.close();
    let recording: { blob: Blob; extension: string } | null = null;
    if (this.recorder && this.recorder.state !== "inactive") {
      const finished = new Promise<void>((resolve) => { this.recorder!.onstop = () => resolve(); });
      this.recorder.stop();
      await finished;
      const type = this.recorder.mimeType || this.recordedType || "video/webm";
      recording = { blob: new Blob(this.chunks, { type }), extension: type.includes("mp4") ? ".mp4" : ".webm" };
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    return recording;
  }
}

export function useLiveSession() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const camera = useRef<BrowserCamera | null>(null);
  const running = status ? RUNNING_STATES.has(status.state) : false;

  const refresh = useCallback(async () => {
    try {
      setStatus(await apiRequest<LiveStatus>("/api/live"));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), running ? 500 : 2500);
    return () => window.clearInterval(timer);
  }, [refresh, running]);

  const stopCamera = useCallback(async () => {
    const active = camera.current;
    camera.current = null;
    const recording = await active?.stop();
    if (recording) {
      setUploading(t("正在把手机录像保存到素材库…"));
      try {
        const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
        await fetch(apiUrl("/api/live/recording"), {
          method: "POST",
          body: recording.blob,
          headers: { "Content-Type": recording.blob.type || "application/octet-stream", "X-Filename": `phone-${stamp}${recording.extension}` },
        });
      } finally {
        setUploading(null);
      }
    }
  }, []);

  const start = useCallback(async (options: LiveStartOptions) => {
    setBusy(true);
    setError(null);
    try {
      const next = await apiRequest<LiveStatus>("/api/live/start", { method: "POST", body: JSON.stringify(options) });
      setStatus(next);
      if (options.source === "browser") {
        const browserCamera = new BrowserCamera();
        camera.current = browserCamera;
        try {
          await browserCamera.start(options.record);
        } catch (failure) {
          camera.current = null;
          await apiRequest<LiveStatus>("/api/live/stop", { method: "POST" }).catch(() => undefined);
          throw failure;
        }
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }, []);

  const stop = useCallback(async () => {
    setBusy(true);
    try {
      await stopCamera();
      setStatus(await apiRequest<LiveStatus>("/api/live/stop", { method: "POST" }));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }, [stopCamera]);

  /** Four tapped corners, four or more named court landmarks, or ``null`` to copy Studio's calibration. */
  const calibrate = useCallback(async (target: { corners: Point[] } | { landmarks: { id: string; image: Point }[] } | null, matchFormat: MatchFormat) => {
    setBusy(true);
    setError(null);
    try {
      const body = target ? { ...target, match_format: matchFormat } : { copy_from_studio: true, match_format: matchFormat };
      setStatus(await apiRequest<LiveStatus>("/api/live/calibration", { method: "PUT", body: JSON.stringify(body) }));
      return true;
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      return false;
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => () => { void camera.current?.stop(); }, []);

  return { status, error, busy, uploading, running, start, stop, calibrate, clearError: () => setError(null) };
}

export type LiveController = ReturnType<typeof useLiveSession>;
