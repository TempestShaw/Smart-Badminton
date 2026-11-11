// Hits and rallies on the device. A racket sound counts as a hit only when a player on this court swings within
// 0.2 s of it, which rejects neighbouring courts, talking and footsteps. Timing follows the offline decoder:
// a reply needs ≥0.22 s and the same side can only hit again after a full exchange (≥0.6 s).

import type { Side, TrackedPlayer } from "./tracker";

export const MINIMUM_REPLY_SECONDS = 0.22;
export const MINIMUM_SAME_SIDE_SECONDS = 0.6;
const HIT_MATCH_SECONDS = 0.2;
const RALLY_END_SECONDS = 2.6;

/** Causal version of the offline onset detector: 10 ms hops against the previous second's baseline. */
export class HitSoundDetector {
  private pending = new Float32Array(0);
  private logHigh: number[] = [];
  private logRms: number[] = [];
  private history: number[] = [];
  private hop: number;
  private hops = 0;
  private lastEvent = -Infinity;

  constructor(private readonly sampleRate = 16000) {
    this.hop = Math.max(1, Math.round(sampleRate * 0.01));
  }

  /** Feed mono samples in [-1, 1] starting at ``startTime``; returns the times (s) of hit-like transients. */
  push(samples: Float32Array, startTime: number): number[] {
    const merged = new Float32Array(this.pending.length + samples.length);
    merged.set(this.pending);
    merged.set(samples, this.pending.length);
    const whole = Math.floor(merged.length / this.hop) * this.hop;
    const firstTime = startTime - this.pending.length / this.sampleRate;
    this.pending = merged.slice(whole);
    const events: number[] = [];
    for (let offset = 0; offset < whole; offset += this.hop) {
      let sumSquares = 0, sumDiff = 0, peakDiff = 0, previous = merged[offset];
      for (let index = offset; index < offset + this.hop; index += 1) {
        const value = merged[index];
        const diff = value - previous;
        previous = value;
        sumSquares += value * value;
        sumDiff += diff * diff;
        peakDiff = Math.max(peakDiff, Math.abs(diff));
      }
      const rms = Math.sqrt(sumSquares / this.hop + 1e-12);
      const highRms = Math.sqrt(sumDiff / this.hop + 1e-12);
      const logRms = 20 * Math.log10(rms + 1e-7);
      const logHigh = 20 * Math.log10(highRms + 1e-7);
      const crest = peakDiff / (highRms + 1e-7);
      const baseline = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0);
      const score = (logHigh - baseline(this.logHigh)) + Math.max(crest - 3, 0) * 1.5 + Math.max(logRms - baseline(this.logRms), 0) * 0.35;
      this.logHigh.push(logHigh); this.logRms.push(logRms);
      if (this.logHigh.length > 100) { this.logHigh.shift(); this.logRms.shift(); }
      this.history.push(score);
      if (this.history.length > 6000) this.history.shift();
      const time = firstTime + offset / this.sampleRate;
      this.hops += 1;
      if (this.hops > 100 && score >= this.threshold() && time - this.lastEvent >= 0.16) {
        this.lastEvent = time;
        events.push(time);
      }
    }
    return events;
  }

  private threshold(): number {
    if (this.history.length < 500) return 8;
    const sorted = [...this.history].sort((a, b) => a - b);
    return Math.max(8, sorted[Math.floor(sorted.length * 0.988)]);
  }
}

export interface Rally { index: number; start: number; end: number | null; shots: number }

export interface PlayerTotals { distance: number; maxSpeed: number; swings: number; hits: number; lastCourt: [number, number] | null; lastTime: number | null; trail: [number, number][] }

export class RallyTracker {
  rallies: Rally[] = [];
  current: Rally | null = null;
  players = new Map<string, PlayerTotals>();
  private sounds: number[] = [];
  private swingPeaks: { time: number; playerId: string; side: Side; strength: number }[] = [];
  private lastShot: { side: Side; time: number } | null = null;
  private pendingHits: number[] = [];
  private lastSwingTime = new Map<string, number>();
  private previousSwing = new Map<string, number>();

  constructor(private readonly audioAvailable: () => boolean) {}

  addSounds(times: number[]): void {
    this.sounds.push(...times);
    if (this.sounds.length > 200) this.sounds.splice(0, this.sounds.length - 200);
  }

  totals(playerId: string): PlayerTotals {
    if (!this.players.has(playerId)) {
      this.players.set(playerId, { distance: 0, maxSpeed: 0, swings: 0, hits: 0, lastCourt: null, lastTime: null, trail: [] });
    }
    return this.players.get(playerId)!;
  }

  update(players: TrackedPlayer[], time: number): void {
    for (const player of players) {
      const totals = this.totals(player.playerId);
      if (!player.detected) continue;
      if (totals.lastCourt && totals.lastTime !== null && time - totals.lastTime > 0 && time - totals.lastTime <= 0.6) {
        // Smooth foot jitter before summing distance (it would otherwise add ~30%).
        const smoothed: [number, number] = [(totals.lastCourt[0] + player.court[0]) / 2, (totals.lastCourt[1] + player.court[1]) / 2];
        const step = Math.hypot(smoothed[0] - totals.lastCourt[0], smoothed[1] - totals.lastCourt[1]);
        if (step <= 9 * (time - totals.lastTime)) totals.distance += step;
        totals.lastCourt = smoothed;
      } else {
        totals.lastCourt = [player.court[0], player.court[1]];
      }
      totals.lastTime = time;
      totals.trail.push(totals.lastCourt);
      if (totals.trail.length > 30) totals.trail.shift();
      totals.maxSpeed = Math.max(totals.maxSpeed, Math.min(9, player.speed));
      // A swing is a wrist-speed peak; on-device frame rates average speed over ~0.1 s, so the bar is moderate.
      const previous = this.previousSwing.get(player.playerId) ?? 0;
      this.previousSwing.set(player.playerId, player.swing);
      const last = this.lastSwingTime.get(player.playerId) ?? -Infinity;
      if (player.swing >= 0.35 && player.swing >= previous && time - last >= 0.45) {
        this.lastSwingTime.set(player.playerId, time);
        totals.swings += 1;
        this.swingPeaks.push({ time, playerId: player.playerId, side: player.side, strength: player.swing });
      }
    }
    this.swingPeaks = this.swingPeaks.filter((peak) => time - peak.time <= 1.5);
    this.confirmHits(time);
    this.updateRally(time);
  }

  /** Pair racket sounds with swings; without audio, only strong swings count. */
  private confirmHits(time: number): void {
    const audio = this.audioAvailable();
    const ready = (peak: { time: number }) => time - peak.time >= HIT_MATCH_SECONDS; // wait for a late sound
    for (const peak of this.swingPeaks.filter(ready)) {
      const sound = this.sounds.find((sound) => Math.abs(sound - peak.time) <= HIT_MATCH_SECONDS);
      const confirmed = audio ? sound !== undefined : peak.strength >= 0.6;
      if (confirmed) this.registerHit(peak.side, peak.playerId, sound ?? peak.time);
    }
    this.swingPeaks = this.swingPeaks.filter((peak) => !ready(peak));
  }

  private registerHit(side: Side, playerId: string, time: number): void {
    if (this.lastShot) {
      const gap = time - this.lastShot.time;
      if (gap < MINIMUM_REPLY_SECONDS || (side === this.lastShot.side && gap < MINIMUM_SAME_SIDE_SECONDS)) return;
    }
    this.lastShot = { side, time };
    this.totals(playerId).hits += 1;
    if (this.current) {
      this.current.shots += 1;
      return;
    }
    // A rally starts at the serve: the first of two hits less than 3 s apart.
    this.pendingHits = this.pendingHits.filter((hit) => time - hit <= 3).concat(time);
    if (this.pendingHits.length >= 2) {
      this.current = { index: this.rallies.length + 1, start: Math.max(0, this.pendingHits[0] - 0.35), end: null, shots: this.pendingHits.length };
      this.pendingHits = [];
    }
  }

  private updateRally(time: number): void {
    if (this.current && this.lastShot && time - this.lastShot.time >= RALLY_END_SECONDS) {
      this.current.end = this.lastShot.time + 0.8;
      this.rallies.push(this.current);
      this.current = null;
    }
  }
}
