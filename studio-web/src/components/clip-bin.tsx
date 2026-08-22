"use client";

import type { StudioController } from "@/hooks/use-studio-controller";
import { formatTime } from "@/lib/time";
import { Button } from "@/components/ui/button";

export function ClipBin({ studio }: { studio: StudioController }) {
  const highlights = studio.analytics.match?.top_highlights ?? [];
  return (
    <section className="clip-bin">
      <div className="clip-bin-heading"><span className="eyebrow">CLIP BIN</span><span>{highlights.length ? `本场精彩：${highlights.map((number) => `R${String(number).padStart(2, "0")}`).join(" · ")}` : "点击片段可定位"}</span></div>
      <div className="clip-list">
        {studio.segments.map((segment, index) => {
          const action = !studio.dirty ? studio.analyticsMap.get(index + 1) : undefined;
          return (
            <Button key={segment.id ?? `${segment.start}-${index}`} type="button" variant="outline" className={index === studio.selectedIndex ? "clip-card selected" : "clip-card"} onClick={() => studio.selectSegment(index, true)}>
              <strong>R{String(index + 1).padStart(2, "0")}　{(segment.end - segment.start).toFixed(2)}s{action ? ` · 🔥${Math.round(action.highlight_score)}` : ""}{segment.review_required === "yes" ? " · 检查" : ""}</strong>
              <span>{formatTime(segment.start)} → {formatTime(segment.end)}</span>
            </Button>
          );
        })}
      </div>
    </section>
  );
}
