import { cp, mkdir, rm } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(scriptDirectory, "..");
const exportDirectory = resolve(projectDirectory, "out");
const targetDirectory = resolve(projectDirectory, "..", "smart_badminton", "studio_static");
const targetAssets = resolve(targetDirectory, "_next");
const tutorialSource = resolve(projectDirectory, "..", "docs", "studio-tutorial.zh-CN.md");

const expectedParent = resolve(projectDirectory, "..", "smart_badminton");
if (dirname(targetDirectory) !== expectedParent || basename(targetDirectory) !== "studio_static") {
  throw new Error(`Refusing to publish outside studio_static: ${targetDirectory}`);
}

await mkdir(targetDirectory, { recursive: true });
await cp(resolve(exportDirectory, "index.html"), resolve(targetDirectory, "index.html"));
await cp(resolve(exportDirectory, "icon.svg"), resolve(targetDirectory, "icon.svg"));
await cp(tutorialSource, resolve(targetDirectory, "studio-tutorial.zh-CN.md"));
await rm(targetAssets, { recursive: true, force: true });
await cp(resolve(exportDirectory, "_next"), targetAssets, { recursive: true });
console.log(`Published Next.js static export to ${targetDirectory}`);
