import { describe, expect, it } from "vitest";

import { buildQuickSegments, deriveMotionThreshold, type MotionSample } from "./browser-analysis";

function samples(values: number[], step = 0.5): MotionSample[] {
  return values.map((motion, index) => ({ time: index * step, motion }));
}

describe("browser quick segmentation", () => {
  it("returns no rallies for a static recording", () => {
    const motion = samples(Array.from({ length: 30 }, () => 0.004));
    expect(deriveMotionThreshold(motion)).toBeGreaterThan(0.004);
    expect(buildQuickSegments(motion, 15)).toEqual([]);
  });

  it("keeps an active block with conservative boundaries", () => {
    const motion = samples([
      0.004, 0.005, 0.004, 0.006,
      0.08, 0.09, 0.1, 0.09, 0.08, 0.1, 0.09, 0.08,
      0.005, 0.004, 0.006, 0.004, 0.005,
    ]);
    const result = buildQuickSegments(motion, 8.5);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("Q001");
    expect(result[0].start).toBeLessThanOrEqual(1.7);
    expect(result[0].end).toBeGreaterThanOrEqual(6.1);
  });

  it("merges a short inactive gap", () => {
    const motion = samples([
      0.004, 0.004,
      0.09, 0.1, 0.09, 0.08,
      0.004,
      0.08, 0.09, 0.1, 0.09,
      0.004, 0.004, 0.004, 0.004,
    ]);
    expect(buildQuickSegments(motion, 7.5)).toHaveLength(1);
  });
});
