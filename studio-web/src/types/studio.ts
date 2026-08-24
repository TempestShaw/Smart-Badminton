export type Point = [number, number];

export interface VideoMetadata {
  name: string;
  duration: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
}

export interface Segment {
  id?: string;
  start: number;
  end: number;
  confidence?: string;
  review_required?: "yes" | "no";
  boundary_reason?: string;
}

export interface AnalyticsAction {
  rally: number;
  trajectory_available: boolean;
  trajectory_points: number;
  trajectory_visible_seconds: number;
  trajectory_coverage_percent: number;
  longest_continuous_track_seconds: number;
  trajectory_distance_frames: number;
  peak_visual_speed_frames_per_second: number;
  net_zone_transits: number;
  rally_style: string;
  highlight_score: number;
  last_hitter: "near" | "far" | "unknown";
  terminal_event: string;
  terminal_event_confidence: number;
  terminal_landing_side: "near" | "far" | "unknown";
  highlight_reasons: string;
  tags: string;
}

export interface HeatmapPoint {
  rally: number;
  x_meters: number;
  y_meters: number;
  event: string;
  confidence: number;
}

export interface AnalyticsPayload {
  available: boolean;
  reason?: string;
  schema_version?: number;
  disclaimer?: string;
  rallies?: AnalyticsAction[];
  match?: {
    rallies: number;
    trajectory_rallies: number;
    trajectory_visible_seconds: number;
    average_coverage_percent: number;
    net_zone_transits: number;
    top_highlights: number[];
    court_heatmap: HeatmapPoint[];
  };
}

export interface ScoreRow {
  rally: number;
  winner: "near" | "far" | "no_point" | "unknown";
  winner_source: string;
  near_score: number;
  far_score: number;
  near_games: number;
  far_games: number;
  server_next: "near" | "far" | "unknown";
  confidence: number;
  note: string;
  score_complete: boolean;
}

export interface ScoreCorrection {
  rally: number;
  winner: "near" | "far" | "no_point" | "auto";
  server_override: "near" | "far" | "unknown";
  server: "near" | "far" | "unknown";
  last_hitter: "near" | "far" | "unknown";
  terminal_event: "landing_in" | "landing_out" | "net" | "unreturned" | "unknown";
  landing_side: "near" | "far" | "unknown";
  post_rally_event: "handoff" | "none" | "unknown";
  note: string;
}

export interface ScorePayload {
  available: boolean;
  generated?: boolean;
  stale?: boolean;
  reason?: string;
  rallies?: ScoreRow[];
  corrections?: ScoreCorrection[];
  resolved?: number;
  no_point?: number;
  unresolved?: number;
  manual?: number;
  automatic?: number;
  complete?: boolean;
  method?: string;
  disclaimer?: string;
  evidence_model?: {
    available: boolean;
    version: number | null;
    manual_examples: number;
    trusted_rules: number;
    disabled_rules: number;
    local_disabled_rules: number;
    project: string | null;
  };
  updated_at?: number;
  serve_observations?: { rally: number; time: number; server: "near" | "far"; confidence: number }[];
}

export interface ScoreSuggestion {
  rally: number;
  status: "consensus" | "review" | "accepted" | "rejected";
  model?: string | null;
  suggestion: {
    winner: "near" | "far" | "no_point" | "unknown";
    terminal_event: "landing_in" | "landing_out" | "net" | "unreturned" | "unknown";
    last_hitter: "near" | "far" | "unknown";
    landing_side: "near" | "far" | "unknown";
    post_rally_event: "handoff" | "none" | "unknown";
    confidence: number;
    evidence_frames?: number[];
    reason?: string;
  };
}

export interface ScoreLabelingPayload {
  configured: boolean;
  model: string | null;
  prepared: boolean;
  directory: string;
  total: number;
  consensus: number;
  review: number;
  accepted: number;
  rejected: number;
  suggestions: ScoreSuggestion[];
}

export interface EvidencePayload {
  available: boolean;
  reason?: string;
  signals?: Record<string, [number, number][]>;
  serves?: { time: number; server: "near" | "far" | "unknown"; confidence: number }[];
  contacts?: { time: number; confidence: number }[];
  terminal_events?: { time: number; event: string; confidence: number }[];
}

export interface ShuttleAnalysisPayload {
  configured: boolean;
  generated: boolean;
  stale: boolean;
  current: boolean;
  stale_reason: string | null;
  raw_generated: boolean;
  point_count: number;
  flight_count: number;
  track_path: string;
  mode: "yolo" | "tracknet" | "hybrid" | null;
  requested_mode: "yolo" | "tracknet" | "hybrid";
  available_modes: ("yolo" | "tracknet" | "hybrid")[];
  tracknet_configured: boolean;
  inpaint_configured: boolean;
}

export interface PoseAnalysisPayload {
  configured: boolean;
  generated: boolean;
  stale: boolean;
  current: boolean;
  stale_reason: string | null;
  path: string;
  url: string | null;
}

export interface ProjectPayload {
  api_schema_version: number;
  id: string;
  video: VideoMetadata;
  preview: VideoMetadata;
  segments: Segment[];
  timeline_path: string;
  output_path: string;
  output: OutputPayload;
  runtime: {
    ffmpeg: {
      available: boolean;
      path: string | null;
      reason: string | null;
      requested_encoder: string;
      selected_encoder: string | null;
      warning: string | null;
    };
  };
  shuttle_analysis: ShuttleAnalysisPayload;
  pose_analysis: PoseAnalysisPayload;
  automatic_analysis: {
    configured: boolean;
    pose_overlay_configured: boolean;
    configuration_issue: string | null;
    model: string | null;
    shuttle_model: string | null;
    tracknet_model: string | null;
    inpaint_model: string | null;
    shuttle_mode: "yolo" | "tracknet" | "hybrid" | null;
    available_shuttle_modes: ("yolo" | "tracknet" | "hybrid")[];
    trajectory_boundary_protection: boolean;
    config: string | null;
    preset: string;
    settings: {
      preroll: number;
      postroll: number;
      end_pending: number;
      maximum_internal_gap: number;
      suppress_handoffs: boolean;
    };
  };
  calibration: { ready: boolean; path: string };
  analytics: AnalyticsPayload;
  score: ScorePayload;
  score_labeling: ScoreLabelingPayload;
  evidence: EvidencePayload;
}

export interface OutputPayload {
  directory: string;
  filename: string;
  path: string;
  directory_exists: boolean;
  file_exists: boolean;
}

export interface DirectoryPayload {
  path: string;
  parent: string | null;
  home: string;
  roots: { name: string; path: string }[];
  directories: { name: string; path: string }[];
}

export interface LibraryVideo {
  id: string;
  name: string;
  relative_path: string;
  project_name: string;
  size_bytes: number;
  proxy_available: boolean;
  timeline_available: boolean;
  latest_auto_cut_available: boolean;
  ground_truth_available: boolean;
  completed_marker: boolean;
  active: boolean;
}

export interface LibraryPayload {
  path: string;
  videos: LibraryVideo[];
}

export interface RenderStatus {
  state: "idle" | "running" | "complete" | "error";
  output?: string;
  message?: string;
  include_trajectory?: boolean;
  winner_filter?: "all" | "near" | "far";
  include_score?: boolean;
}

export interface AnalysisStatus {
  state: "idle" | "running" | "complete" | "error";
  mode?: "single" | "batch" | "shuttle" | "visual" | "score-labels";
  stage?: string;
  label?: string;
  message?: string;
  progress?: number;
  completed?: number;
  total?: number;
  job_id?: string;
  project_id?: string;
  project_name?: string;
  current?: string;
  current_name?: string;
  background?: boolean;
  started_at?: number;
  updated_at?: number;
}

export interface CalibrationRegion {
  id: string;
  type?: string;
  label: string;
  color: string;
  required: boolean;
  minimum_points?: number;
  maximum_points?: number;
  points: Point[] | null;
}

export interface HomographyQuality {
  available: boolean;
  reason?: string;
  p95_uncertainty_meters?: number;
  score_safe?: boolean;
}

export interface CalibrationPayload {
  available: boolean;
  path: string;
  exists: boolean;
  ready: boolean;
  required_completed: number;
  required_total: number;
  reference_frame: { width: number; height: number };
  regions: CalibrationRegion[];
  homography: HomographyQuality;
  backup?: string | null;
  automatic_analysis_configured?: boolean;
  pose_overlay_configured?: boolean;
  shuttle_analysis?: ShuttleAnalysisPayload;
}

export interface ShuttleDetection {
  time: number;
  x: number;
  y: number;
  confidence: number;
  status: "tracked" | "recovered" | "competing" | "manual" | "stationary" | "unlinked";
  source: "yolo" | "tracknet" | "hybrid" | "optical_flow" | "manual" | "unknown";
  detection_status: "detected" | "fused" | "inpainted" | "recovered" | "manual";
  evidence_weight: number;
  flight_id: string | null;
}

export interface ShuttleAnnotation {
  id: string;
  time_seconds: number;
  x_normalized: number;
  y_normalized: number;
  action: "add" | "reject";
  note: string;
}

export interface ShuttleAnnotationPayload {
  project_id: string;
  available: boolean;
  configured: boolean;
  generated: boolean;
  stale: boolean;
  current: boolean;
  stale_reason: string | null;
  raw_generated: boolean;
  point_count: number;
  flight_count: number;
  track_path: string;
  path: string;
  detections: ShuttleDetection[];
  annotations: ShuttleAnnotation[];
  backup?: string | null;
}
