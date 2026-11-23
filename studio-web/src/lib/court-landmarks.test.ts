import { describe, expect, it } from "vitest";

import { COURT_LANDMARKS, cornersFromLandmarks, LandmarkError, type Point } from "./court-landmarks";
import { applyTransform, perspectiveTransform } from "./ondevice/court";

const SINGLES: Point[] = [[0.46, 0], [5.64, 0], [5.64, 13.4], [0.46, 13.4]];

function view(corners: Point[]) {
  const toImage = perspectiveTransform(SINGLES, corners);
  return (id: string) => {
    const landmark = COURT_LANDMARKS.find((candidate) => candidate.id === id)!;
    return { id, image: applyTransform(toImage, landmark.court) };
  };
}

describe("court landmarks", () => {
  it("recovers singles corners that are outside the frame", () => {
    // A low camera beside the net: the near corners fall below and left of the picture.
    const truth: Point[] = [[-0.35, 1.6], [0.55, 1.25], [0.9, 0.42], [0.62, 0.38]];
    const seen = view(truth);
    const corners = cornersFromLandmarks(["left-post", "right-post", "far-left-service", "far-right-service", "far-right-corner", "far-centre-baseline"].map(seen));
    corners.forEach((corner, index) => {
      expect(corner[0]).toBeCloseTo(truth[index][0], 3);
      expect(corner[1]).toBeCloseTo(truth[index][1], 3);
    });
  });

  it("matches the Python ids and positions", () => {
    expect(COURT_LANDMARKS.map((landmark) => landmark.id)).toHaveLength(14);
    expect(COURT_LANDMARKS.find((landmark) => landmark.id === "left-post")?.court).toEqual([0, 6.7]);
  });

  it("explains unusable point sets", () => {
    const seen = view([[0.2, 0.9], [0.8, 0.9], [0.65, 0.4], [0.35, 0.4]]);
    expect(() => cornersFromLandmarks(["left-post", "right-post", "far-centre-service"].map(seen))).toThrow(LandmarkError);
    const onOneLine = ["near-left-corner", "near-centre-baseline", "near-right-corner"].map(seen);
    expect(() => cornersFromLandmarks([...onOneLine, { ...onOneLine[1], id: "near-left-corner" }])).toThrow(/同一条线/);
    const swapped = ["left-post", "right-post", "near-centre-service", "far-centre-service", "far-left-corner"].map(seen);
    swapped[4] = { ...swapped[4], image: [swapped[4].image[0] + 0.3, swapped[4].image[1] + 0.2] };
    expect(() => cornersFromLandmarks(swapped)).toThrow(/对不上/);
  });
});
