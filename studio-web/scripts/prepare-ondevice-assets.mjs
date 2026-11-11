import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// On-device live analysis: onnxruntime-web (WebGPU and WASM builds) and the YOLO11n-pose ONNX models, served as
// static files so a phone can run inference without any server.
//
// Models come from ../models (made by `smart-badminton export-ondevice-models`). A hosted build without them
// (e.g. Vercel) downloads the release assets instead: set ONDEVICE_MODELS_URL to a release download URL, or leave
// it unset on Vercel to use this repository's latest release. Without models the page still builds and says so.
const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(scriptDirectory, "..");
const publicDirectory = resolve(projectDirectory, "public");
const ortSource = resolve(projectDirectory, "node_modules", "onnxruntime-web", "dist");
const ortTarget = resolve(publicDirectory, "ort");
const modelSource = resolve(projectDirectory, "..", "models");
const modelTarget = resolve(publicDirectory, "models");
const MODELS = ["yolo11n-pose.onnx", "yolo11n-pose-int8.onnx"];
const DEFAULT_RELEASE = "https://github.com/TempestShaw/Smart-Badminton/releases/latest/download";

for (const target of [ortTarget, modelTarget]) {
  if (dirname(target) !== publicDirectory) throw new Error(`Refusing to write outside public/: ${target}`);
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
}
for (const name of [
  "ort.webgpu.min.mjs",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  await copyFile(resolve(ortSource, name), resolve(ortTarget, name));
}

/** An ONNX file is a protobuf whose first field is ir_version (tag 0x08); reject LFS pointers and HTML pages. */
function isOnnx(bytes) {
  return bytes.length > 1_000_000 && bytes[0] === 0x08;
}

async function localModel(name) {
  try {
    const bytes = await readFile(resolve(modelSource, name));
    if (isOnnx(bytes)) return bytes;
    console.warn(`models/${name} is not an ONNX model (a Git LFS pointer?); skipping it`);
  } catch {
    // Not exported locally.
  }
  return null;
}

async function downloadModel(base, name) {
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/${name}`, { redirect: "follow" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!isOnnx(bytes)) throw new Error("the download is not an ONNX model");
    return bytes;
  } catch (error) {
    console.warn(`Could not download ${name} from ${base}: ${error.message}`);
    return null;
  }
}

const releaseBase = process.env.ONDEVICE_MODELS_URL || (process.env.VERCEL ? DEFAULT_RELEASE : "");
const available = [];
for (const name of MODELS) {
  const bytes = (await localModel(name)) ?? (releaseBase ? await downloadModel(releaseBase, name) : null);
  if (!bytes) continue;
  await writeFile(resolve(modelTarget, name), bytes);
  available.push({ name, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") });
}
// The worker reads this to pick a model per backend without probing for missing files.
await writeFile(resolve(modelTarget, "ondevice.json"), `${JSON.stringify({ models: available }, null, 2)}\n`);
if (!available.length) {
  console.warn(
    "No on-device pose model: run `smart-badminton export-ondevice-models` (or set ONDEVICE_MODELS_URL); "
    + "the on-device live page will explain that the model is missing.",
  );
}
console.log(`Prepared on-device assets: onnxruntime-web, models [${available.map((model) => model.name).join(", ") || "none"}]`);
