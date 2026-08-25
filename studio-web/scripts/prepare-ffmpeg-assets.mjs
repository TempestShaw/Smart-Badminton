import { cp, mkdir, rm } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(scriptDirectory, "..");
const sourceDirectory = resolve(projectDirectory, "node_modules", "@ffmpeg", "core", "dist", "esm");
const classWorkerDirectory = resolve(projectDirectory, "node_modules", "@ffmpeg", "ffmpeg", "dist", "esm");
const publicDirectory = resolve(projectDirectory, "public");
const targetDirectory = resolve(publicDirectory, "ffmpeg");

if (dirname(targetDirectory) !== publicDirectory || basename(targetDirectory) !== "ffmpeg") {
  throw new Error(`Refusing to write outside public/ffmpeg: ${targetDirectory}`);
}

await rm(targetDirectory, { recursive: true, force: true });
await mkdir(targetDirectory, { recursive: true });
await cp(resolve(sourceDirectory, "ffmpeg-core.js"), resolve(targetDirectory, "ffmpeg-core.js"));
await cp(resolve(sourceDirectory, "ffmpeg-core.wasm"), resolve(targetDirectory, "ffmpeg-core.wasm"));
await cp(resolve(classWorkerDirectory, "worker.js"), resolve(targetDirectory, "worker.js"));
await cp(resolve(classWorkerDirectory, "const.js"), resolve(targetDirectory, "const.js"));
await cp(resolve(classWorkerDirectory, "errors.js"), resolve(targetDirectory, "errors.js"));
console.log(`Prepared FFmpeg WebAssembly assets in ${targetDirectory}`);
