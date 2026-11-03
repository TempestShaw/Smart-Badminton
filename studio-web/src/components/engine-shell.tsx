"use client";

import { useTranslation } from "react-i18next";
import { t } from "@/lib/i18n";

import { useEffect, useState } from "react";
import { Download, Gauge, Globe2, Laptop, Sparkles } from "lucide-react";

import { LanguageSwitcher } from "@/components/language-controls";
import { ThemeToggle } from "@/components/theme-controls";
import { Brand } from "@/components/brand";
import { BrowserQuickStudio } from "@/components/browser-quick-studio";
import { LiveStudio } from "@/components/live-studio";
import { StudioApp } from "@/components/studio-app";
import { Button } from "@/components/ui/button";
import { apiUrl } from "@/lib/api";

type EngineMode = "detecting" | "browser" | "native" | "live";
type NativeView = "studio" | "live";

function localStudioHost(): boolean {
  return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

/** A page served by the local Studio (also over the LAN, e.g. on a phone) can reach its API on the same origin. */
async function servedBy(): Promise<"studio" | "live" | null> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(apiUrl("/api/health"), { signal: controller.signal });
    if (!response.ok) return null;
    const health = (await response.json()) as { ok?: boolean; mode?: string };
    return health.ok ? (health.mode === "live" ? "live" : "studio") : null;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

export function EngineShell() {
  useTranslation();
  const [mode, setMode] = useState<EngineMode>("detecting");
  const [localHost, setLocalHost] = useState(false);
  const [served, setServed] = useState(false);
  const [view, setView] = useState<NativeView>("studio");

  useEffect(() => {
    const local = localStudioHost();
    const parameters = new URLSearchParams(window.location.search);
    const requested = parameters.get("engine");
    setLocalHost(local);
    void servedBy().then((server) => {
      setServed(server !== null);
      // Phones and tablets reaching Studio over the LAN usually come for the live view.
      setView(parameters.get("view") === "live" || (server !== null && !local && parameters.get("view") !== "studio") ? "live" : "studio");
      if (server === "live") setMode("live");
      else if (requested === "browser") setMode("browser");
      else if (requested === "native" || server === "studio") setMode("native");
      else setMode(local ? "native" : "browser");
    });
  }, []);

  const switchMode = (next: Exclude<EngineMode, "detecting" | "live">) => {
    const url = new URL(window.location.href);
    url.searchParams.set("engine", next);
    window.history.replaceState(null, "", url);
    setMode(next);
  };

  const switchView = (next: NativeView) => {
    const url = new URL(window.location.href);
    url.searchParams.set("view", next);
    window.history.replaceState(null, "", url);
    setView(next);
  };

  if (mode === "detecting") return <div className="engine-loading"><span className="loader" /></div>;
  if (mode === "live") return <LiveStudio />;
  if (mode === "browser") return <BrowserQuickStudio onSwitchNative={() => switchMode("native")} />;
  if (localHost || served) {
    if (view === "live") return <LiveStudio onBack={() => switchView("studio")} />;
    return <StudioApp onSwitchEngine={() => switchMode("browser")} onOpenLive={() => switchView("live")} />;
  }

  return (
    <main className="native-connect">
      <div className="native-connect-header">
        <Brand />
        <div className="native-connect-controls">
          <LanguageSwitcher /><ThemeToggle />
          <div className="quick-engine-switch" aria-label={t("执行引擎")}>
            <button onClick={() => switchMode("browser")}><Globe2 />{t("浏览器版")}</button>
            <button className="selected"><Gauge />{t("精准版")}</button>
          </div>
        </div>
      </div>
      <section className="native-connect-card">
        <div className="native-connect-icon"><Sparkles /></div>
        <p className="eyebrow">{t("NATIVE ACCURATE")}</p>
        <h1>{t("原生精准版")}</h1>
        <p>{t("完整球路、姿态、比分与 CUDA 加速。")}</p>
        <div className="native-connect-actions">
          <Button asChild size="lg"><a href="http://127.0.0.1:8765/?engine=native"><Laptop />{t("打开本地 Studio")}</a></Button>
          <Button asChild size="lg" variant="outline"><a href="https://github.com/TempestShaw/Smart-Badminton/releases/latest"><Download />{t("下载安装包")}</a></Button>
        </div>
      </section>
    </main>
  );
}
