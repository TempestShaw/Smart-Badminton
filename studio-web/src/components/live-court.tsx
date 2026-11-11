"use client";

import type { LivePlayer } from "@/types/live";

const WIDTH = 6.1;
const LENGTH = 13.4;
const NET = LENGTH / 2;

/** Court diagram in metres, far court at the top, with each tracked player's position and recent path. */
export function LiveCourt({ players, label }: { players: LivePlayer[]; label: string }) {
  const y = (value: number) => LENGTH - value; // the far baseline is drawn at the top
  return (
    <svg className="live-court" viewBox="-1.1 -1.6 8.3 16.6" role="img" aria-label={label}>
      <rect className="live-court-floor" x={0} y={0} width={WIDTH} height={LENGTH} rx={0.05} />
      <g className="live-court-lines">
        <rect x={0} y={0} width={WIDTH} height={LENGTH} fill="none" />
        <line x1={0.46} y1={0} x2={0.46} y2={LENGTH} />
        <line x1={WIDTH - 0.46} y1={0} x2={WIDTH - 0.46} y2={LENGTH} />
        <line x1={0} y1={y(NET - 1.98)} x2={WIDTH} y2={y(NET - 1.98)} />
        <line x1={0} y1={y(NET + 1.98)} x2={WIDTH} y2={y(NET + 1.98)} />
        <line x1={0} y1={y(0.76)} x2={WIDTH} y2={y(0.76)} />
        <line x1={0} y1={y(LENGTH - 0.76)} x2={WIDTH} y2={y(LENGTH - 0.76)} />
        <line x1={WIDTH / 2} y1={y(0)} x2={WIDTH / 2} y2={y(NET - 1.98)} />
        <line x1={WIDTH / 2} y1={y(NET + 1.98)} x2={WIDTH / 2} y2={y(LENGTH)} />
      </g>
      <line className="live-court-net" x1={-0.35} y1={y(NET)} x2={WIDTH + 0.35} y2={y(NET)} />
      {players.map((player) => {
        if (!player.court) return null;
        const points = player.trail.map(([x, py]) => `${x.toFixed(2)},${y(py).toFixed(2)}`).join(" ");
        return (
          <g key={player.player_id} className="live-court-player" data-player={player.player_id} data-visible={player.visible}>
            {player.trail.length > 1 ? <polyline points={points} /> : null}
            {/* A transform (not cx/cy) moves the marker, so the transition stays on the compositor. */}
            <g className="live-court-marker" style={{ transform: `translate(${player.court[0]}px, ${y(player.court[1])}px)` }}>
              <circle r={0.32} />
              <text y={0.12}>{player.player_id.endsWith("2") ? "2" : "1"}</text>
            </g>
          </g>
        );
      })}
    </svg>
  );
}
