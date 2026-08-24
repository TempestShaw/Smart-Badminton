"use client";

import type { StudioController } from "@/hooks/use-studio-controller";
import { Button } from "@/components/ui/button";

export function ClipBin({ studio }: { studio: StudioController }) {
  const clips = studio.segments
    .map((segment, index) => ({ segment, index }))
    .filter(({ index }) => !studio.scoreReview.active || studio.scoreReview.rallies.includes(index + 1));
  return (
    <section className="clip-bin" aria-label="片段导航">
      <div className="clip-list">
        {clips.map(({ segment, index }) => {
          const label = `R${String(index + 1).padStart(2, "0")}`;
          const selected = index === studio.selectedIndex;
          const review = segment.review_required === "yes";
          return (
            <Button
              key={segment.id ?? `${segment.start}-${index}`}
              type="button"
              size="xs"
              variant="outline"
              className={`clip-card${selected ? " selected" : ""}${review ? " review" : ""}`}
              aria-label={`跳到 ${label}${review ? "，需检查" : ""}`}
              aria-current={selected ? "true" : undefined}
              onClick={() => studio.selectSegment(index, true)}
            >
              {label}
            </Button>
          );
        })}
      </div>
    </section>
  );
}
