import { describe, expect, it } from "vitest";

import { Court, validCorners, type Vec2 } from "./court";
import { HitSoundDetector, RallyTracker } from "./rally";
import { courtDetections, PlayerTracker, type TrackedPlayer } from "./tracker";

// A behind-the-baseline view: the near baseline is wide at the bottom, the far one narrow near the top.
const CORNERS: Vec2[] = [[0.15, 0.95], [0.85, 0.95], [0.66, 0.4], [0.34, 0.4]];
const WIDTH = 1280;
const HEIGHT = 720;

function court(doubles = false): Court {
  return new Court(CORNERS, WIDTH, HEIGHT, doubles);
}

function shirt(hue: number): Float32Array {
  const histogram = new Float32Array(48);
  histogram[hue * 4 + 3] = 1;
  return histogram;
}

/** A standing person whose ankles are at ``foot`` (pixels). */
function person(foot: Vec2, appearance: Float32Array | null, height = 160) {
  const keypoints = new Float32Array(51);
  for (const joint of [15, 16]) {
    keypoints[joint * 3] = foot[0];
    keypoints[joint * 3 + 1] = foot[1];
    keypoints[joint * 3 + 2] = 0.9;
  }
  const box: [number, number, number, number] = [foot[0] - height / 4, foot[1] - height, foot[0] + height / 4, foot[1] + 4];
  return { box, score: 0.9, keypoints, appearance };
}

describe("on-device court", () => {
  it("maps the tapped singles corners onto the court and back", () => {
    const geometry = court();
    const [nearLeft, , farRight] = CORNERS.map(([x, y]) => geometry.court([x * WIDTH, y * HEIGHT]));
    expect(nearLeft[0]).toBeCloseTo(0.46, 6);
    expect(nearLeft[1]).toBeCloseTo(0, 6);
    expect(farRight[0]).toBeCloseTo(5.64, 6);
    expect(farRight[1]).toBeCloseTo(13.4, 6);
    const centre = geometry.image([3.05, 6.7]);
    const back = geometry.court(centre);
    expect(back[0]).toBeCloseTo(3.05, 6);
    expect(back[1]).toBeCloseTo(6.7, 6);
  });

  it("assigns halves and rejects people off this court", () => {
    const geometry = court();
    expect(geometry.sideOf([3, 2])).toBe("near");
    expect(geometry.sideOf([3, 11])).toBe("far");
    expect(geometry.sideOf([3, 6.7])).toBeNull(); // on the net line
    expect(geometry.sideOf([-3, 2])).toBeNull(); // next court over
    expect(geometry.sideOf([3, 17])).toBeNull(); // far behind the baseline
  });

  it("requires the corners in tapping order", () => {
    expect(validCorners(CORNERS)).toBe(true);
    expect(validCorners([CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]])).toBe(false);
    expect(() => new Court([CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]], WIDTH, HEIGHT, false)).toThrow();
  });
});

describe("on-device player tracker", () => {
  it("keeps doubles partners' identities when they cross", () => {
    const geometry = court(true);
    const tracker = new PlayerTracker(2, Math.hypot(WIDTH, HEIGHT));
    const red = shirt(0);
    const blue = shirt(7);
    let ids: Record<string, string> = {};
    for (let step = 0; step <= 20; step += 1) {
      const time = step * 0.1;
      const progress = step / 20;
      // Red walks left → right while blue walks right → left, both 2.5 m from the near baseline.
      const redFoot = geometry.image([1 + 4 * progress, 2.5]);
      const blueFoot = geometry.image([5 - 4 * progress, 2.5 + 0.4 * Math.sin(progress * Math.PI)]);
      const players = tracker.update(courtDetections(geometry, [person(redFoot, red), person(blueFoot, blue)]), time);
      const byShirt = Object.fromEntries(players.map((player) => [player.court[0] < 3.05 === progress < 0.5 ? "red" : "blue", player.playerId]));
      if (step === 0) ids = byShirt;
      else if (step !== 10) expect(byShirt).toEqual(ids); // at the crossing frame positions coincide
    }
    expect(new Set(Object.values(ids)).size).toBe(2);
  });

  it("does not hand a lost player's slot to someone in a different shirt", () => {
    const geometry = court();
    const tracker = new PlayerTracker(1, Math.hypot(WIDTH, HEIGHT));
    const foot = geometry.image([3, 3]);
    tracker.update(courtDetections(geometry, [person(foot, shirt(0))]), 0);
    // A stranger walks on where the player stood; the player has only been gone for two seconds.
    const stranger = tracker.update(courtDetections(geometry, [person(foot, shirt(7))]), 2);
    expect(stranger.filter((player) => player.detected)).toHaveLength(0);
    // After four seconds the slot may be re-filled (clothes look different under new lighting, etc.).
    const later = tracker.update(courtDetections(geometry, [person(foot, shirt(7))]), 5);
    expect(later.filter((player) => player.detected)).toHaveLength(1);
  });

  it("reports where to look next and which half needs a scan", () => {
    const geometry = court();
    const tracker = new PlayerTracker(1, Math.hypot(WIDTH, HEIGHT));
    tracker.update(courtDetections(geometry, [person(geometry.image([3, 3]), shirt(0))]), 0);
    const regions = tracker.regions(0.1);
    expect(regions.tracked.map((region) => region.side)).toEqual(["near"]);
    expect(regions.missing).toEqual(["far"]);
    expect(tracker.regions(1).tracked).toHaveLength(0);
  });
});

function swinging(playerId: string, side: "near" | "far", swing: number): TrackedPlayer {
  return {
    playerId, side, detected: true, box: [0, 0, 1, 1], keypoints: null, court: side === "near" ? [3, 3] : [3, 10],
    speed: 1, wrist: null, wristSpeed: swing * 1.2, swing,
  };
}

/** Feed ~20 fps frames; ``swings`` maps a time to the player who peaks then. */
function play(rallies: RallyTracker, until: number, swings: [number, string, "near" | "far"][]) {
  for (let frame = 0; frame * 0.05 <= until; frame += 1) {
    const time = frame * 0.05;
    const players = [swinging("near-1", "near", 0.05), swinging("far-1", "far", 0.05)];
    for (const [at, playerId, side] of swings) {
      if (Math.abs(time - at) < 0.025) players[side === "near" ? 0 : 1] = swinging(playerId, side, 0.8);
    }
    rallies.update(players, time);
  }
}

describe("on-device hits and rallies", () => {
  it("counts a swing as a hit only when a racket sound accompanies it", () => {
    const rallies = new RallyTracker(() => true);
    rallies.addSounds([1.0, 2.05]);
    play(rallies, 3, [[1.0, "near-1", "near"], [2.0, "far-1", "far"], [2.6, "near-1", "near"]]);
    expect(rallies.totals("near-1").swings).toBe(2);
    expect(rallies.totals("near-1").hits).toBe(1); // the 2.6 s swing had no sound
    expect(rallies.totals("far-1").hits).toBe(1);
  });

  it("drops a second hit by the same side within one exchange", () => {
    const rallies = new RallyTracker(() => true);
    rallies.addSounds([1.0, 1.5, 2.2]);
    play(rallies, 3, [[1.0, "near-1", "near"], [1.5, "near-1", "near"], [2.2, "far-1", "far"]]);
    expect(rallies.totals("near-1").hits).toBe(1);
    expect(rallies.totals("far-1").hits).toBe(1);
  });

  it("opens a rally on the serve exchange and closes it after a quiet spell", () => {
    const rallies = new RallyTracker(() => true);
    rallies.addSounds([1.0, 2.0, 3.0]);
    play(rallies, 3.5, [[1.0, "near-1", "near"], [2.0, "far-1", "far"], [3.0, "near-1", "near"]]);
    expect(rallies.current?.shots).toBe(3);
    expect(rallies.current?.start).toBeCloseTo(0.65, 5);
    play(rallies, 7, [[1.0, "near-1", "near"], [2.0, "far-1", "far"], [3.0, "near-1", "near"]]);
    expect(rallies.current).toBeNull();
    expect(rallies.rallies).toHaveLength(1);
    expect(rallies.rallies[0].end).toBeCloseTo(3.8, 5);
  });
});

describe("on-device hit sounds", () => {
  it("finds sharp transients above a quiet background", () => {
    const sampleRate = 16000;
    const detector = new HitSoundDetector(sampleRate);
    const seconds = 4;
    const samples = new Float32Array(sampleRate * seconds);
    let seed = 1;
    for (let index = 0; index < samples.length; index += 1) {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      samples[index] = (seed / 2147483648 - 0.5) * 0.004;
    }
    for (const at of [1.5, 2.5, 3.5]) {
      const start = Math.round(at * sampleRate);
      for (let index = 0; index < 80; index += 1) samples[start + index] += Math.exp(-index / 12) * (index % 2 ? -0.8 : 0.8);
    }
    const events: number[] = [];
    for (let offset = 0; offset < samples.length; offset += 2048) {
      events.push(...detector.push(samples.subarray(offset, offset + 2048), offset / sampleRate));
    }
    expect(events).toHaveLength(3);
    events.forEach((time, index) => expect(Math.abs(time - [1.5, 2.5, 3.5][index])).toBeLessThan(0.02));
  });
});
