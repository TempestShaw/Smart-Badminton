// Player identity tracking, ported from smart_badminton/players.py: bounded slots per half (1 singles, 2 doubles),
// a capped constant-velocity prediction in court metres, exact assignment and shirt-colour re-identification.

import type { Court, Vec2 } from "./court";

export type Side = "near" | "far";
export const LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_ANKLE = 15, RIGHT_ANKLE = 16;
const KEYPOINT_CONFIDENCE = 0.2;
const MAXIMUM_SPEED = 6.0;

export interface PoseDetection {
  side: Side;
  box: [number, number, number, number];
  score: number;
  keypoints: Float32Array; // 17 × (x, y, confidence) in frame pixels
  foot: Vec2; // pixels
  court: Vec2; // metres
  appearance: Float32Array | null;
}

export interface TrackedPlayer {
  playerId: string;
  side: Side;
  detected: boolean;
  box: [number, number, number, number];
  keypoints: Float32Array | null;
  court: Vec2;
  speed: number; // m/s, smoothed
  wrist: Vec2 | null; // pixels
  wristSpeed: number; // frame diagonals per second
  swing: number; // 0–1, wrist speed / 1.2 diagonals per second
}

interface Track {
  playerId: string;
  side: Side;
  position: Vec2;
  velocity: Vec2;
  lastTime: number;
  appearance: Float32Array | null;
  detection: PoseDetection;
  previousWrists: Map<number, Vec2>;
  previousCourt: Vec2 | null;
  kinematicsTime: number | null;
  speed: number;
  wrist: Vec2 | null;
  wristSpeed: number;
}

export function visible(keypoints: Float32Array, joint: number): boolean {
  return keypoints[joint * 3 + 2] >= KEYPOINT_CONFIDENCE;
}

function footPixel(box: [number, number, number, number], keypoints: Float32Array): Vec2 {
  const ankles = [LEFT_ANKLE, RIGHT_ANKLE].filter((joint) => visible(keypoints, joint));
  if (!ankles.length) return [(box[0] + box[2]) / 2, box[3]];
  return [
    ankles.reduce((sum, joint) => sum + keypoints[joint * 3], 0) / ankles.length,
    ankles.reduce((sum, joint) => sum + keypoints[joint * 3 + 1], 0) / ankles.length,
  ];
}

/** People standing on this court's halves; officials, spectators and neighbouring courts are dropped. */
export function courtDetections(
  court: Court,
  people: { box: [number, number, number, number]; score: number; keypoints: Float32Array; appearance: Float32Array | null }[],
): PoseDetection[] {
  const output: PoseDetection[] = [];
  for (const person of people) {
    const foot = footPixel(person.box, person.keypoints);
    const position = court.court(foot);
    const side = court.sideOf(position);
    if (!side) continue;
    output.push({ side, box: person.box, score: person.score, keypoints: person.keypoints, foot, court: position, appearance: person.appearance });
  }
  return output;
}

function appearanceDistance(a: Float32Array | null, b: Float32Array | null): number | null {
  if (!a || !b) return null;
  let coefficient = 0;
  for (let index = 0; index < a.length; index += 1) coefficient += Math.sqrt(Math.max(0, a[index]) * Math.max(0, b[index]));
  return Math.sqrt(Math.max(0, 1 - Math.min(1, coefficient)));
}

function area(box: [number, number, number, number]): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

export class PlayerTracker {
  private tracks: Record<Side, Track[]> = { near: [], far: [] };

  constructor(private readonly players: 1 | 2, private readonly diagonal: number, private readonly coastSeconds = 1.2) {}

  private predicted(track: Track, time: number): Vec2 {
    const dt = Math.max(0, Math.min(0.3, time - track.lastTime));
    return [track.position[0] + track.velocity[0] * dt, track.position[1] + track.velocity[1] * dt];
  }

  private cost(track: Track, detection: PoseDetection, time: number): number {
    const predicted = this.predicted(track, time);
    const distance = Math.hypot(predicted[0] - detection.court[0], predicted[1] - detection.court[1]);
    const gate = Math.min(6, 1.4 + 5 * Math.max(0.05, time - track.lastTime));
    const lost = time - track.lastTime > this.coastSeconds;
    if (distance > gate && !lost) return Infinity;
    const motion = Math.min(1, distance / gate);
    const appearance = appearanceDistance(track.appearance, detection.appearance);
    if (lost) {
      // A clearly different shirt is someone else (often the partner) until the slot has been empty for 4 s; after
      // that the mismatch only weighs in, so a slot first taken by the wrong person can still pass to the player.
      if (appearance === null) return 0.5 + motion * 0.15;
      if (time - track.lastTime < 4) return appearance > 0.5 ? Infinity : appearance + motion * 0.15;
      return Math.min(appearance, 0.6) + motion * 0.15;
    }
    return appearance === null ? motion : motion * 0.725 + appearance * 0.55;
  }

  private assign(tracks: Track[], detections: PoseDetection[], time: number): Map<number, number> {
    const costs = tracks.map((track) => detections.map((detection) => this.cost(track, detection, time)));
    let best = { total: 0.95 * tracks.length, pairs: new Map<number, number>() };
    const choices = [...detections.keys(), -1];
    const search = (trackIndex: number, used: Set<number>, total: number, pairs: Map<number, number>) => {
      if (trackIndex === tracks.length) {
        if (total < best.total - 1e-9) best = { total, pairs: new Map(pairs) };
        return;
      }
      for (const choice of choices) {
        if (choice >= 0 && used.has(choice)) continue;
        const cost = choice < 0 ? 0.95 : costs[trackIndex][choice];
        if (!Number.isFinite(cost)) continue;
        if (choice >= 0) { used.add(choice); pairs.set(trackIndex, choice); }
        search(trackIndex + 1, used, total + cost, pairs);
        if (choice >= 0) { used.delete(choice); pairs.delete(trackIndex); }
      }
    };
    search(0, new Set(), 0, new Map());
    return best.pairs;
  }

  update(detections: PoseDetection[], time: number): TrackedPlayer[] {
    const output: TrackedPlayer[] = [];
    for (const side of ["near", "far"] as Side[]) {
      const candidates = detections
        .filter((detection) => detection.side === side)
        .sort((a, b) => b.score * area(b.box) - a.score * area(a.box))
        .slice(0, 6);
      const tracks = this.tracks[side];
      const pairs = this.assign(tracks, candidates, time);
      const matched = new Set(pairs.values());
      pairs.forEach((detectionIndex, trackIndex) => this.observe(tracks[trackIndex], candidates[detectionIndex], time));
      candidates.forEach((detection, index) => {
        if (matched.has(index) || tracks.length >= this.players) return;
        const track: Track = {
          playerId: `${side}-${tracks.length + 1}`, side, position: detection.court, velocity: [0, 0], lastTime: time,
          appearance: detection.appearance, detection, previousWrists: new Map(), previousCourt: null,
          kinematicsTime: null, speed: 0, wrist: null, wristSpeed: 0,
        };
        this.kinematics(track, detection, time);
        tracks.push(track);
        matched.add(index);
      });
      for (const track of tracks) {
        const age = time - track.lastTime;
        if (age > 0.5) continue; // absent rather than invented
        output.push({
          playerId: track.playerId,
          side,
          detected: age === 0,
          box: track.detection.box,
          keypoints: age === 0 ? track.detection.keypoints : null,
          court: age === 0 ? track.position : this.predicted(track, time),
          speed: track.speed,
          wrist: age === 0 ? track.wrist : null,
          wristSpeed: age === 0 ? track.wristSpeed : 0,
          swing: age === 0 ? Math.min(1, track.wristSpeed / 1.2) : 0,
        });
      }
    }
    return output;
  }

  private observe(track: Track, detection: PoseDetection, time: number): void {
    const dt = time - track.lastTime;
    if (dt > 0 && dt <= 0.6) {
      let vx = (detection.court[0] - track.position[0]) / dt;
      let vy = (detection.court[1] - track.position[1]) / dt;
      const speed = Math.hypot(vx, vy);
      if (speed > MAXIMUM_SPEED) { vx *= MAXIMUM_SPEED / speed; vy *= MAXIMUM_SPEED / speed; }
      const weight = visible(detection.keypoints, LEFT_ANKLE) && visible(detection.keypoints, RIGHT_ANKLE) ? 0.6 : 0.25;
      track.velocity = [track.velocity[0] * (1 - weight) + vx * weight, track.velocity[1] * (1 - weight) + vy * weight];
    } else {
      track.velocity = [0, 0];
    }
    if (dt > 0.35) { track.previousWrists.clear(); track.previousCourt = null; }
    track.position = detection.court;
    track.lastTime = time;
    track.detection = detection;
    if (detection.appearance) {
      track.appearance = track.appearance
        ? track.appearance.map((value, index) => value * 0.85 + detection.appearance![index] * 0.15)
        : detection.appearance;
    }
    this.kinematics(track, detection, time);
  }

  private kinematics(track: Track, detection: PoseDetection, time: number): void {
    const delta = track.kinematicsTime === null ? 0 : time - track.kinematicsTime;
    track.kinematicsTime = time;
    let fastest: { speed: number; point: Vec2 } | null = null;
    for (const joint of [LEFT_WRIST, RIGHT_WRIST]) {
      if (!visible(detection.keypoints, joint)) { track.previousWrists.delete(joint); continue; }
      const point: Vec2 = [detection.keypoints[joint * 3], detection.keypoints[joint * 3 + 1]];
      const previous = track.previousWrists.get(joint);
      const speed = previous && delta > 0 ? Math.hypot(point[0] - previous[0], point[1] - previous[1]) / this.diagonal / delta : 0;
      if (!fastest || speed > fastest.speed) fastest = { speed, point };
      track.previousWrists.set(joint, point);
    }
    track.wrist = fastest?.point ?? null;
    track.wristSpeed = Math.min(3, fastest?.speed ?? 0);
    if (track.previousCourt && delta > 0) {
      const measured = Math.hypot(detection.court[0] - track.previousCourt[0], detection.court[1] - track.previousCourt[1]) / delta;
      track.speed = track.speed * 0.5 + Math.min(9, measured) * 0.5;
    }
    track.previousCourt = detection.court;
  }

  /**
   * Where to look next frame: the last box of every player seen within ``0.6 s`` (with how long ago), and the
   * halves that need a full scan because a slot is unfilled or its player has been missing longer than that.
   */
  regions(time: number): { tracked: { playerId: string; side: Side; box: [number, number, number, number]; age: number }[]; missing: Side[] } {
    const tracked: { playerId: string; side: Side; box: [number, number, number, number]; age: number }[] = [];
    const missing: Side[] = [];
    for (const side of ["near", "far"] as Side[]) {
      const tracks = this.tracks[side];
      let complete = tracks.length >= this.players;
      for (const track of tracks) {
        const age = time - track.lastTime;
        if (age > 0.6) complete = false;
        else tracked.push({ playerId: track.playerId, side, box: track.detection.box, age });
      }
      if (!complete) missing.push(side);
    }
    return { tracked, missing };
  }
}
