export type MatchFormat = "singles" | "doubles";

export interface LivePlayer {
  player_id: string;
  label: string;
  side: "near" | "far";
  visible: boolean;
  speed_meters_per_second: number | null;
  max_speed_meters_per_second: number;
  distance_meters: number;
  rally_distance_meters: number;
  swings: number;
  hits: number;
  court: [number, number] | null;
  foot: [number, number] | null;
  trail: [number, number][];
}

export interface LiveRally {
  index: number;
  start: number;
  end: number | null;
  duration: number;
  shots: number;
}

export interface LiveAnalysis {
  time_seconds: number;
  match_format: MatchFormat;
  metric: boolean;
  shuttle_enabled: boolean;
  pose_enabled: boolean;
  model_enabled: boolean;
  audio_enabled: boolean;
  rally: { active: boolean; evidence: number; model_probability: number | null; current: LiveRally | null };
  rallies: LiveRally[];
  totals: { rallies: number; average_duration: number; longest_duration: number; shots: number };
  players: LivePlayer[];
}

export interface LiveStatus {
  state: "idle" | "starting" | "running" | "finished" | "stopped" | "error";
  error?: string | null;
  source?: string;
  match_format: MatchFormat;
  calibrated: boolean;
  studio_calibration_available: boolean;
  models: { pose: boolean; shuttle: boolean; rally: boolean };
  recordings: string;
  started_at?: number;
  frames_received?: number;
  frames_analyzed?: number;
  analysis_fps?: number;
  source_fps?: number;
  frame_size?: [number, number] | null;
  recording?: string | null;
  analysis?: LiveAnalysis;
}

export interface LiveStartOptions {
  source: string;
  match_format: MatchFormat;
  shuttle: boolean;
  record: boolean;
}
