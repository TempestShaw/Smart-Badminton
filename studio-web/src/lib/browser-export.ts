import { t } from "./i18n";
import type { Segment } from "@/types/studio";

function ffmpegBaseUrl(): string {
  const script = document.querySelector<HTMLScriptElement>('script[src*="/_next/"]');
  const prefix = script ? new URL(script.src).pathname.split("/_next/")[0] : "";
  return `${prefix}/ffmpeg`;
}

function filterGraph(segments: Segment[], includeAudio: boolean): { graph: string; map: string[] } {
  const parts: string[] = [];
  const inputs: string[] = [];
  segments.forEach((segment, index) => {
    parts.push(`[0:v]trim=start=${segment.start.toFixed(3)}:end=${segment.end.toFixed(3)},setpts=PTS-STARTPTS[v${index}]`);
    inputs.push(`[v${index}]`);
    if (includeAudio) {
      parts.push(`[0:a]atrim=start=${segment.start.toFixed(3)}:end=${segment.end.toFixed(3)},asetpts=PTS-STARTPTS[a${index}]`);
      inputs.push(`[a${index}]`);
    }
  });
  parts.push(`${inputs.join("")}concat=n=${segments.length}:v=1:a=${includeAudio ? 1 : 0}[vout]${includeAudio ? "[aout]" : ""}`);
  return { graph: parts.join(";"), map: includeAudio ? ["-map", "[vout]", "-map", "[aout]"] : ["-map", "[vout]"] };
}

export async function exportVideoInBrowser(
  file: File,
  segments: Segment[],
  onProgress: (value: number, message: string) => void,
  signal: AbortSignal,
): Promise<void> {
  if (!segments.length) throw new Error(t("时间轴没有片段"));
  if (file.size > 1.5 * 1024 ** 3) throw new Error(t("浏览器版暂不支持超过 1.5 GB 的输出，请使用精准版"));
  const [{ FFmpeg }, { fetchFile, toBlobURL }] = await Promise.all([import("@ffmpeg/ffmpeg"), import("@ffmpeg/util")]);
  const ffmpeg = new FFmpeg();
  const cancelled = () => ffmpeg.terminate();
  signal.addEventListener("abort", cancelled, { once: true });
  ffmpeg.on("progress", ({ progress }) => onProgress(Math.max(0, Math.min(1, progress)), t("输出成片 {{value1}}%", { value1: Math.round(progress * 100) })));
  const basePath = ffmpegBaseUrl();
  const localBase = new URL(`${basePath}/`, window.location.origin).href.replace(/\/$/, "");
  try {
    onProgress(0, t("载入本地编码器"));
    const embedded = basePath.startsWith("/static");
    const coreURL = embedded
      ? await toBlobURL("https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js", "text/javascript")
      : `${localBase}/ffmpeg-core.js`;
    const wasmURL = embedded
      ? await toBlobURL("https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm", "application/wasm")
      : `${localBase}/ffmpeg-core.wasm`;
    const classWorkerURL = embedded
      ? "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/worker.js"
      : `${localBase}/worker.js`;
    await ffmpeg.load({ coreURL, wasmURL, classWorkerURL });
    if (signal.aborted) throw new DOMException("Export cancelled", "AbortError");
    const extension = file.name.match(/\.[A-Za-z0-9]+$/)?.[0] ?? ".mp4";
    const input = `input${extension}`;
    await ffmpeg.writeFile(input, await fetchFile(file));

    const run = async (includeAudio: boolean) => {
      const filter = filterGraph(segments, includeAudio);
      await ffmpeg.exec([
        "-y",
        "-i", input,
        "-filter_complex", filter.graph,
        ...filter.map,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        ...(includeAudio ? ["-c:a", "aac", "-b:a", "160k"] : ["-an"]),
        "-movflags", "+faststart",
        "output.mp4",
      ]);
    };
    try {
      await run(true);
    } catch (error) {
      if (signal.aborted) throw error;
      await run(false);
    }
    const output = await ffmpeg.readFile("output.mp4");
    const bytes = output instanceof Uint8Array ? output : new TextEncoder().encode(output);
    const buffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(buffer).set(bytes);
    const url = URL.createObjectURL(new Blob([buffer], { type: "video/mp4" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${file.name.replace(/\.[^.]+$/, "")}-quick-cut.mp4`;
    anchor.click();
    URL.revokeObjectURL(url);
    onProgress(1, t("输出完成"));
  } finally {
    signal.removeEventListener("abort", cancelled);
    ffmpeg.terminate();
  }
}
