"use client";

import { useEffect, useState } from "react";
import { Download, Gauge, Globe2, Laptop, Sparkles } from "lucide-react";

import { Brand } from "@/components/brand";
import { BrowserQuickStudio } from "@/components/browser-quick-studio";
import { StudioApp } from "@/components/studio-app";
import { Button } from "@/components/ui/button";

type EngineMode = "detecting" | "browser" | "native";

function localStudioHost(): boolean {
  return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

export function EngineShell() {
  const [mode, setMode] = useState<EngineMode>("detecting");
  const [localHost, setLocalHost] = useState(false);

  useEffect(() => {
    const local = localStudioHost();
    const requested = new URLSearchParams(window.location.search).get("engine");
    setLocalHost(local);
    setMode(requested === "browser" ? "browser" : requested === "native" ? "native" : local ? "native" : "browser");
  }, []);

  const switchMode = (next: Exclude<EngineMode, "detecting">) => {
    const url = new URL(window.location.href);
    url.searchParams.set("engine", next);
    window.history.replaceState(null, "", url);
    setMode(next);
  };

  if (mode === "detecting") return <div className="engine-loading"><span className="loader" /></div>;
  if (mode === "browser") return <BrowserQuickStudio onSwitchNative={() => switchMode("native")} />;
  if (localHost) return <StudioApp onSwitchEngine={() => switchMode("browser")} />;

  return (
    <main className="native-connect">
      <div className="native-connect-header">
        <Brand />
        <div className="quick-engine-switch" aria-label="执行引擎">
          <button onClick={() => switchMode("browser")}><Globe2 />浏览器版</button>
          <button className="selected"><Gauge />精准版</button>
        </div>
      </div>
      <section className="native-connect-card">
        <div className="native-connect-icon"><Sparkles /></div>
        <p className="eyebrow">NATIVE ACCURATE</p>
        <h1>原生精准版</h1>
        <p>完整球路、姿态、比分与 CUDA 加速。</p>
        <div className="native-connect-actions">
          <Button asChild size="lg"><a href="http://127.0.0.1:8765/?engine=native"><Laptop />打开本地 Studio</a></Button>
          <Button asChild size="lg" variant="outline"><a href="https://github.com/TempestShaw/Smart-Badminton/releases/latest"><Download />下载安装包</a></Button>
        </div>
      </section>
    </main>
  );
}
