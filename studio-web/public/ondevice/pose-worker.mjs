// On-device pose inference. Runs YOLO11n-pose with onnxruntime-web (WebGPU when the browser has it, else WASM SIMD)
// on a batch of square crops taken from one camera frame, and returns people in full-frame pixel coordinates.
// Kept as a plain module worker so the app bundler never has to process onnxruntime's WASM loader.

let ort = null;
let session = null;
let provider = "wasm";
const canvases = new Map();

function canvasFor(size) {
  if (!canvases.has(size)) {
    const canvas = new OffscreenCanvas(size, size);
    canvases.set(size, { canvas, context: canvas.getContext("2d", { willReadFrequently: true }) });
  }
  return canvases.get(size);
}

const MODEL_CACHE = "smart-badminton-ondevice-v1";
const FLOAT_MODEL = "yolo11n-pose.onnx";
const INT8_MODEL = "yolo11n-pose-int8.onnx";

/**
 * Cache-first model download, so the next session starts without the network (and without 3–12 MB of data).
 * The URL carries the model's checksum, so a new release replaces the cached copy instead of hiding behind it.
 */
async function modelBytes(url) {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const base = url.split("?")[0];
    for (const request of await cache.keys()) if (request.url.split("?")[0] === base) await cache.delete(request);
  } catch {
    cache = null; // Cache Storage can be unavailable (private browsing); fall back to the network.
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error("model-missing");
  const bytes = new Uint8Array(await response.arrayBuffer());
  try {
    await cache?.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
  } catch {
    // Quota exceeded: keep running from memory.
  }
  return bytes;
}

/** Models this deployment ships, as {name, sha256}; the manifest is written by prepare-ondevice-assets. */
async function availableModels(modelBase) {
  try {
    const response = await fetch(`${modelBase}/ondevice.json`, { cache: "no-cache" });
    if (response.ok) return ((await response.json()).models ?? []).filter((model) => model?.name);
  } catch {
    // No manifest (offline with a cached page, or an older deployment): try the float model.
  }
  return [{ name: FLOAT_MODEL, sha256: "" }];
}

function zeros(batch, size) {
  return new ort.Tensor("float32", new Float32Array(batch * 3 * size * size), [batch, 3, size, size]);
}

/** Median latency of the steady-state workload (two players at 192 px), after compiling kernels at both sizes. */
async function warmUp(candidate) {
  for (const size of [192, 320]) await candidate.run({ [candidate.inputNames[0]]: zeros(1, size) });
  const input = { [candidate.inputNames[0]]: zeros(2, 192) };
  const times = [];
  for (let run = 0; run < 3; run += 1) {
    const started = performance.now();
    await candidate.run(input);
    times.push(performance.now() - started);
  }
  return times.sort((a, b) => a - b)[1];
}

async function createSession(backend, models, modelBase) {
  // WebGPU runs float math natively; on WASM the int8 model is faster and a quarter of the download.
  const find = (name) => models.find((model) => model.name === name);
  const model = (backend === "wasm" && find(INT8_MODEL)) || find(FLOAT_MODEL) || models[0];
  const name = model.name;
  const bytes = await modelBytes(`${modelBase}/${name}${model.sha256 ? `?v=${model.sha256.slice(0, 16)}` : ""}`);
  const created = await ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: "all" });
  return { session: created, backend, model: name, latency: await warmUp(created) };
}

async function init({ modelBase, ortBase, preferWebgpu, threads }) {
  const models = await availableModels(modelBase);
  if (!models.length) throw new Error("model-missing");
  let adapter = null;
  if (preferWebgpu && self.navigator?.gpu) {
    try {
      adapter = await self.navigator.gpu.requestAdapter();
    } catch {
      adapter = null;
    }
    if (adapter?.info?.isFallbackAdapter || adapter?.isFallbackAdapter) adapter = null; // software GPU: slower than WASM
  }
  ort = await import(`${ortBase}/${adapter ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"}`);
  ort.env.wasm.wasmPaths = `${ortBase}/`;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, threads || 4)) : 1;
  ort.env.wasm.simd = true;
  let chosen = null;
  if (adapter) {
    try {
      chosen = await createSession("webgpu", models, modelBase);
    } catch {
      chosen = null;
    }
  }
  // A GPU that is slower than the CPU (old or busy devices) loses to WASM; measure instead of assuming.
  if (!chosen || chosen.latency > 45) {
    const fallback = await createSession("wasm", models, modelBase).catch((error) => {
      if (!chosen) throw error;
      return null;
    });
    if (fallback && (!chosen || fallback.latency < chosen.latency)) {
      await chosen?.session.release?.();
      chosen = fallback;
    } else {
      await fallback?.session.release?.();
    }
  }
  session = chosen.session;
  provider = chosen.backend;
  return {
    provider,
    model: chosen.model,
    latencyMs: chosen.latency,
    threads: provider === "wasm" ? ort.env.wasm.numThreads : 0,
    isolated: self.crossOriginIsolated === true,
  };
}

// Shirt colour: a 12×4 hue/saturation histogram of the torso, the same feature the Python tracker uses.
function torsoHistogram(pixels, size, box) {
  const [x1, y1, x2, y2] = box;
  const width = x2 - x1;
  const height = y2 - y1;
  const left = Math.max(0, Math.floor(x1 + width * 0.25));
  const right = Math.min(size, Math.ceil(x2 - width * 0.25));
  const top = Math.max(0, Math.floor(y1 + height * 0.18));
  const bottom = Math.min(size, Math.ceil(y1 + height * 0.55));
  if (right - left < 3 || bottom - top < 3) return null;
  const histogram = new Float32Array(48);
  let total = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const offset = (y * size + x) * 4;
      const r = pixels[offset] / 255;
      const g = pixels[offset + 1] / 255;
      const b = pixels[offset + 2] / 255;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const delta = max - min;
      let hue = 0;
      if (delta > 0) {
        if (max === r) hue = ((g - b) / delta + 6) % 6;
        else if (max === g) hue = (b - r) / delta + 2;
        else hue = (r - g) / delta + 4;
      }
      const saturation = max > 0 ? delta / max : 0;
      histogram[Math.min(11, Math.floor(hue * 2)) * 4 + Math.min(3, Math.floor(saturation * 4))] += 1;
      total += 1;
    }
  }
  if (!total) return null;
  for (let index = 0; index < 48; index += 1) histogram[index] /= total;
  return histogram;
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  return union > 0 ? intersection / union : 0;
}

async function infer({ bitmap, crops, size, confidence }) {
  const started = performance.now();
  const { context } = canvasFor(size);
  const plane = size * size;
  const input = new Float32Array(crops.length * 3 * plane);
  const letterboxes = [];
  const pixelsByCrop = [];
  for (let index = 0; index < crops.length; index += 1) {
    const crop = crops[index];
    const scale = size / Math.max(crop.w, crop.h);
    const width = crop.w * scale;
    const height = crop.h * scale;
    const padX = (size - width) / 2;
    const padY = (size - height) / 2;
    context.fillStyle = "rgb(114,114,114)";
    context.fillRect(0, 0, size, size);
    context.drawImage(bitmap, crop.x, crop.y, crop.w, crop.h, padX, padY, width, height);
    const pixels = context.getImageData(0, 0, size, size).data;
    const base = index * 3 * plane;
    for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
      input[base + pixel] = pixels[offset] / 255;
      input[base + plane + pixel] = pixels[offset + 1] / 255;
      input[base + 2 * plane + pixel] = pixels[offset + 2] / 255;
    }
    letterboxes.push({ scale, padX, padY, x: crop.x, y: crop.y });
    pixelsByCrop.push(pixels);
  }
  bitmap.close();
  const prepared = performance.now();
  const output = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [crops.length, 3, size, size]) });
  const tensor = output[session.outputNames[0]];
  const data = tensor.data;
  const anchors = tensor.dims[2];
  const results = [];
  for (let index = 0; index < crops.length; index += 1) {
    const base = index * 56 * anchors;
    const candidates = [];
    for (let anchor = 0; anchor < anchors; anchor += 1) {
      const score = data[base + 4 * anchors + anchor];
      if (score < confidence) continue;
      const cx = data[base + anchor];
      const cy = data[base + anchors + anchor];
      const w = data[base + 2 * anchors + anchor];
      const h = data[base + 3 * anchors + anchor];
      candidates.push({ anchor, score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
    }
    candidates.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const candidate of candidates) {
      if (kept.every((other) => intersectionOverUnion(candidate.box, other.box) < 0.5)) kept.push(candidate);
      if (kept.length >= 8) break;
    }
    const { scale, padX, padY, x, y } = letterboxes[index];
    const toFrameX = (value) => (value - padX) / scale + x;
    const toFrameY = (value) => (value - padY) / scale + y;
    results.push(kept.map((candidate) => {
      const keypoints = new Float32Array(51);
      for (let joint = 0; joint < 17; joint += 1) {
        keypoints[joint * 3] = toFrameX(data[base + (5 + joint * 3) * anchors + candidate.anchor]);
        keypoints[joint * 3 + 1] = toFrameY(data[base + (6 + joint * 3) * anchors + candidate.anchor]);
        keypoints[joint * 3 + 2] = data[base + (7 + joint * 3) * anchors + candidate.anchor];
      }
      const [bx1, by1, bx2, by2] = candidate.box;
      return {
        score: candidate.score,
        box: [toFrameX(bx1), toFrameY(by1), toFrameX(bx2), toFrameY(by2)],
        keypoints,
        appearance: torsoHistogram(pixelsByCrop[index], size, candidate.box),
      };
    }));
  }
  return { results, preprocessMs: prepared - started, inferenceMs: performance.now() - prepared };
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      self.postMessage({ type: "ready", ...(await init(message)) });
    } else if (message.type === "infer") {
      self.postMessage({ type: "result", id: message.id, ...(await infer(message)) });
    }
  } catch (error) {
    if (message.bitmap) message.bitmap.close?.();
    self.postMessage({ type: "error", id: message.id, message: String(error?.message ?? error) });
  }
};
