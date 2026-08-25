import type { Segment } from "@/types/studio";

export interface BrowserProject {
  schemaVersion: 1;
  engine: "browser-quick";
  video: {
    name: string;
    size: number;
    lastModified: number;
    duration: number;
    width: number;
    height: number;
  };
  segments: Segment[];
  updatedAt: string;
}

export function projectKey(file: File): string {
  return `smart-badminton:quick:${file.name}:${file.size}:${file.lastModified}`;
}

export function loadBrowserProject(file: File): BrowserProject | null {
  try {
    const value = localStorage.getItem(projectKey(file));
    return value ? JSON.parse(value) as BrowserProject : null;
  } catch {
    return null;
  }
}

export function saveBrowserProject(project: BrowserProject): void {
  localStorage.setItem(`smart-badminton:quick:${project.video.name}:${project.video.size}:${project.video.lastModified}`, JSON.stringify(project));
}

export function downloadProject(project: BrowserProject): void {
  const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${project.video.name.replace(/\.[^.]+$/, "")}.smart-badminton.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
