from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .adaptation import fit_segmentation_adapter
from .analytics import analyze_rally_actions
from .audio import analyze_audio
from .contacts import analyze_contacts
from .court_events import analyze_terminal_events
from .doctor import doctor_report, initialize_project
from .evaluate import evaluate_rallies
from .features import extract_features
from .geometry import CourtGeometry, draw_geometry_preview
from .hybrid import fuse_shuttle_detections
from .model import MODEL_FAMILIES, benchmark_multi_models, predict_model, train_model, train_multi_model
from .regression import regression_gate, train_guarded_candidate
from .render import render_rallies
from .review import render_boundary_reviews
from .scoring import analyze_score, evaluate_score
from .segmenter import segment_rallies
from .shuttle import detect_shuttle
from .shuttle_training import audit_shuttle_dataset, train_shuttle_model
from .tracknet import TrackNetRuntimeConfig, detect_tracknet
from .tracknet_training import export_tracknet_labels, train_tracknet
from .trajectory import analyze_shuttle_trajectory, render_shuttle_trajectory
from .trajectory_evaluate import evaluate_shuttle_annotations


def path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smart-badminton", description="Conservative rally segmentation for fixed-camera badminton video"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--config", type=path)
    doctor.add_argument("--model", type=path)
    doctor.add_argument("--pose-model", type=path)
    doctor.add_argument("--shuttle-model", type=path)
    doctor.add_argument("--ffmpeg", type=path)
    doctor.add_argument("--encoder", default="auto")
    doctor.add_argument("--packages", type=path)
    doctor.add_argument("--tracknet-model", type=path)
    doctor.add_argument("--inpaint-model", type=path)
    initialize = commands.add_parser("init-project")
    initialize.add_argument("--directory", type=path, required=True)
    shuttle_audit = commands.add_parser("audit-shuttle-dataset")
    shuttle_audit.add_argument("--manifest", type=path, required=True)
    shuttle_train = commands.add_parser("train-shuttle")
    shuttle_train.add_argument("--manifest", type=path, required=True)
    shuttle_train.add_argument("--base-model", type=path, required=True)
    shuttle_train.add_argument("--output", type=path, required=True)
    shuttle_train.add_argument("--work-directory", type=path, required=True)
    shuttle_train.add_argument("--validation-camera")
    shuttle_train.add_argument("--epochs", type=int, default=80)
    shuttle_train.add_argument("--image-size", type=int, default=1280)
    shuttle_train.add_argument("--device", default="")
    shuttle_train.add_argument("--checkpoint-license", default="AGPL-3.0")
    preview = commands.add_parser("preview-geometry")
    preview.add_argument("--video", type=path, required=True)
    preview.add_argument("--config", type=path, required=True)
    preview.add_argument("--time", type=float, default=300.0)
    preview.add_argument("--output", type=path, required=True)
    audio = commands.add_parser("analyze-audio")
    audio.add_argument("--input", type=path, required=True)
    audio.add_argument("--output", type=path, required=True)
    audio.add_argument("--plot", type=path)
    audio.add_argument("--ffmpeg", type=path)
    feature = commands.add_parser("extract-features")
    feature.add_argument("--video", type=path, required=True)
    feature.add_argument("--config", type=path, required=True)
    feature.add_argument("--output", type=path, required=True)
    feature.add_argument("--audio-events", type=path)
    feature.add_argument("--shuttle-detections", type=path)
    feature.add_argument("--pose-model", type=path)
    feature.add_argument("--packages", type=path)
    feature.add_argument("--fps", type=float)
    feature.add_argument("--start", type=float, default=0.0)
    feature.add_argument("--end", type=float)
    shuttle = commands.add_parser("detect-shuttle")
    shuttle.add_argument("--video", type=path, required=True)
    shuttle.add_argument("--config", type=path, required=True)
    shuttle.add_argument("--model", type=path, required=True)
    shuttle.add_argument("--packages", type=path)
    shuttle.add_argument("--output", type=path, required=True)
    shuttle.add_argument("--fps", type=float, default=15.0)
    shuttle.add_argument("--confidence", type=float, default=0.04)
    shuttle.add_argument("--imgsz", type=int, default=1280)
    shuttle.add_argument("--start", type=float, default=0.0)
    shuttle.add_argument("--end", type=float)
    tracknet = commands.add_parser("detect-tracknet")
    tracknet.add_argument("--video", type=path, required=True)
    tracknet.add_argument("--config", type=path, required=True)
    tracknet.add_argument("--tracknet-model", type=path, required=True)
    tracknet.add_argument("--inpaint-model", type=path)
    tracknet.add_argument("--packages", type=path)
    tracknet.add_argument("--output", type=path, required=True)
    tracknet.add_argument("--batch-size", type=int, default=4)
    tracknet.add_argument("--start", type=float, default=0.0)
    tracknet.add_argument("--end", type=float)
    hybrid = commands.add_parser("fuse-shuttle")
    hybrid.add_argument("--yolo", type=path)
    hybrid.add_argument("--tracknet", type=path)
    hybrid.add_argument("--config", type=path, required=True)
    hybrid.add_argument("--output", type=path, required=True)
    tracknet_export = commands.add_parser("export-tracknet-labels")
    tracknet_export.add_argument("--video", type=path, required=True)
    tracknet_export.add_argument("--trajectory", type=path, required=True)
    tracknet_export.add_argument("--annotations", type=path)
    tracknet_export.add_argument("--output", type=path, required=True)
    tracknet_train = commands.add_parser("train-tracknet")
    tracknet_train.add_argument("--video", type=path, required=True)
    tracknet_train.add_argument("--config", type=path, required=True)
    tracknet_train.add_argument("--labels", type=path, required=True)
    tracknet_train.add_argument("--base-model", type=path, required=True)
    tracknet_train.add_argument("--output", type=path, required=True)
    tracknet_train.add_argument("--epochs", type=int, default=8)
    tracknet_train.add_argument("--batch-size", type=int, default=2)
    tracknet_train.add_argument("--learning-rate", type=float, default=1e-5)
    tracknet_train.add_argument("--maximum-windows", type=int, default=6000)
    trajectory = commands.add_parser("track-shuttle")
    trajectory.add_argument("--video", type=path, required=True)
    trajectory.add_argument("--detections", type=path, required=True)
    trajectory.add_argument("--config", type=path, required=True)
    trajectory.add_argument("--output", type=path, required=True)
    trajectory.add_argument("--preview", type=path)
    trajectory.add_argument("--contact-features", type=path)
    trajectory.add_argument("--annotations", type=path)
    trajectory.add_argument("--ffmpeg", type=path)
    trajectory.add_argument("--encoder", default="auto")
    trajectory.add_argument("--style", choices=("debug", "trail"), default="debug")
    trajectory_evaluate = commands.add_parser("evaluate-trajectory")
    trajectory_evaluate.add_argument("--detections", type=path, required=True)
    trajectory_evaluate.add_argument("--trajectory", type=path, required=True)
    trajectory_evaluate.add_argument("--annotations", type=path, required=True)
    trajectory_evaluate.add_argument("--config", type=path)
    trajectory_evaluate.add_argument("--output", type=path)
    train = commands.add_parser("train")
    train.add_argument("--features", type=path, required=True)
    train.add_argument("--rallies", type=path, required=True)
    train.add_argument("--model", type=path, required=True)
    train.add_argument("--report", type=path)
    train.add_argument("--family", choices=MODEL_FAMILIES, default="hist_gradient_boosting")
    train_multi = commands.add_parser("train-multi")
    train_multi.add_argument("--dataset", type=path, required=True)
    train_multi.add_argument("--model", type=path, required=True)
    train_multi.add_argument("--report", type=path)
    train_multi.add_argument("--family", choices=MODEL_FAMILIES, default="hist_gradient_boosting")
    benchmark = commands.add_parser("benchmark-models")
    benchmark.add_argument("--dataset", type=path, required=True)
    benchmark.add_argument("--report", type=path, required=True)
    benchmark.add_argument("--families", nargs="+", choices=MODEL_FAMILIES)
    gate = commands.add_parser("regression-gate")
    gate.add_argument("--dataset", type=path, required=True)
    gate.add_argument("--baseline-model", type=path, required=True)
    gate.add_argument("--report", type=path, required=True)
    gate.add_argument("--family", choices=MODEL_FAMILIES, default="hist_gradient_boosting")
    guarded_train = commands.add_parser("train-guarded")
    guarded_train.add_argument("--dataset", type=path, required=True)
    guarded_train.add_argument("--baseline-model", type=path, required=True)
    guarded_train.add_argument("--candidate-model", type=path, required=True)
    guarded_train.add_argument("--gate-report", type=path, required=True)
    guarded_train.add_argument("--training-report", type=path)
    guarded_train.add_argument("--family", choices=MODEL_FAMILIES, default="hist_gradient_boosting")
    predict = commands.add_parser("predict")
    predict.add_argument("--features", type=path, required=True)
    predict.add_argument("--model", type=path, required=True)
    predict.add_argument("--output", type=path, required=True)
    segment = commands.add_parser("segment")
    segment.add_argument("--features", type=path, required=True)
    segment.add_argument("--probabilities", type=path, required=True)
    segment.add_argument("--output", type=path, required=True)
    segment.add_argument("--start-threshold", type=float, default=0.56)
    segment.add_argument("--keep-threshold", type=float, default=0.30)
    segment.add_argument("--preroll", type=float, default=0.35)
    segment.add_argument("--postroll", type=float, default=0.55)
    segment.add_argument("--end-pending", type=float, default=0.55)
    segment.add_argument("--maximum-internal-gap", type=float, default=1.2)
    segment.add_argument("--keep-handoffs", action="store_true")
    segment.add_argument("--shuttle-trajectory", type=path)
    segment.add_argument("--adapter", type=path)
    fit_adapter = commands.add_parser("fit-adapter")
    fit_adapter.add_argument("--features", type=path, required=True)
    fit_adapter.add_argument("--probabilities", type=path, required=True)
    fit_adapter.add_argument("--truth", type=path, required=True)
    fit_adapter.add_argument("--output", type=path, required=True)
    fit_adapter.add_argument("--shuttle-trajectory", type=path)
    fit_adapter.add_argument("--base-adapter", type=path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predicted", type=path, required=True)
    evaluate.add_argument("--truth", type=path, required=True)
    evaluate.add_argument("--output", type=path)
    review = commands.add_parser("review")
    review.add_argument("--video", type=path, required=True)
    review.add_argument("--rallies", type=path, required=True)
    review.add_argument("--output-directory", type=path, required=True)
    review.add_argument("--ffmpeg", type=path)
    actions = commands.add_parser("analyze-actions")
    actions.add_argument("--features", type=path, required=True)
    actions.add_argument("--rallies", type=path, required=True)
    actions.add_argument("--output", type=path, required=True)
    actions.add_argument("--summary", type=path)
    actions.add_argument("--audio-events", type=path)
    actions.add_argument("--contacts", type=path)
    actions.add_argument("--events", type=path)
    actions.add_argument("--trajectory", type=path)
    actions.add_argument("--config", type=path)
    contacts = commands.add_parser("analyze-contacts")
    contacts.add_argument("--features", type=path, required=True)
    contacts.add_argument("--rallies", type=path, required=True)
    contacts.add_argument("--output", type=path, required=True)
    contacts.add_argument("--summary", type=path)
    events = commands.add_parser("analyze-events")
    events.add_argument("--video", type=path, required=True)
    events.add_argument("--config", type=path, required=True)
    events.add_argument("--rallies", type=path, required=True)
    events.add_argument("--trajectory", type=path, required=True)
    events.add_argument("--contacts", type=path)
    events.add_argument("--probabilities", type=path)
    events.add_argument("--features", type=path)
    events.add_argument("--output", type=path, required=True)
    events.add_argument("--summary", type=path)
    score = commands.add_parser("analyze-score")
    score.add_argument("--rallies", type=path, required=True)
    score.add_argument("--events", type=path)
    score.add_argument("--corrections", type=path)
    score.add_argument("--output", type=path, required=True)
    score.add_argument("--summary", type=path)
    score.add_argument("--initial-server", choices=["near", "far", "unknown"], default="unknown")
    score_eval = commands.add_parser("evaluate-score")
    score_eval.add_argument("--predicted", type=path, required=True)
    score_eval.add_argument("--truth", type=path, required=True)
    score_eval.add_argument("--output", type=path)
    render = commands.add_parser("render")
    render.add_argument("--video", type=path, required=True)
    render.add_argument("--rallies", type=path, required=True)
    render.add_argument("--output", type=path, required=True)
    render.add_argument("--ffmpeg", type=path)
    render.add_argument("--encoder", default="auto")
    render.add_argument("--quality", type=int, default=21)
    render.add_argument("--fps", type=float)
    render.add_argument("--width", type=int)
    render.add_argument("--height", type=int)
    studio = commands.add_parser("studio")
    studio.add_argument("--video", type=path)
    studio.add_argument("--rallies", type=path)
    studio.add_argument("--library", type=path)
    studio.add_argument("--output", type=path)
    studio.add_argument("--proxy", type=path)
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument("--no-open", action="store_true")
    studio.add_argument("--ffmpeg", type=path)
    studio.add_argument("--encoder", default="auto")
    studio.add_argument("--quality", type=int, default=21)
    studio.add_argument("--config", type=path)
    studio.add_argument("--model", type=path)
    studio.add_argument("--pose-model", type=path)
    studio.add_argument("--shuttle-model", type=path)
    studio.add_argument("--tracknet-model", type=path)
    studio.add_argument("--inpaint-model", type=path)
    studio.add_argument("--shuttle-mode", choices=["yolo", "tracknet", "hybrid"], default="hybrid")
    studio.add_argument("--tracknet-packages", type=path)
    studio.add_argument("--packages", type=path)

    args = parser.parse_args()
    if args.command == "doctor":
        print(
            json.dumps(
                doctor_report(
                    args.config,
                    args.model,
                    args.pose_model,
                    args.shuttle_model,
                    args.ffmpeg,
                    args.encoder,
                    args.packages,
                    args.tracknet_model,
                    args.inpaint_model,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "init-project":
        print(json.dumps(initialize_project(args.directory), ensure_ascii=False, indent=2))
    elif args.command == "audit-shuttle-dataset":
        print(json.dumps(audit_shuttle_dataset(args.manifest), ensure_ascii=False, indent=2))
    elif args.command == "train-shuttle":
        print(
            json.dumps(
                train_shuttle_model(
                    args.manifest,
                    args.output,
                    args.base_model,
                    args.work_directory,
                    args.validation_camera,
                    args.epochs,
                    args.image_size,
                    args.device,
                    args.checkpoint_license,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "preview-geometry":
        capture = cv2.VideoCapture(str(args.video))
        capture.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("Could not read preview frame")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), draw_geometry_preview(frame, CourtGeometry.from_json(args.config)))
    elif args.command == "analyze-audio":
        print(json.dumps(analyze_audio(args.input, args.output, args.plot, args.ffmpeg), indent=2))
    elif args.command == "extract-features":
        extract_features(
            args.video,
            args.config,
            args.output,
            args.audio_events,
            args.shuttle_detections,
            args.pose_model,
            args.packages,
            args.fps,
            args.start,
            args.end,
        )
    elif args.command == "detect-shuttle":
        print(
            json.dumps(
                detect_shuttle(
                    args.video,
                    args.config,
                    args.model,
                    args.packages,
                    args.output,
                    args.fps,
                    args.confidence,
                    args.imgsz,
                    args.start,
                    args.end,
                ),
                indent=2,
            )
        )
    elif args.command == "detect-tracknet":
        print(
            json.dumps(
                detect_tracknet(
                    args.video,
                    args.config,
                    args.tracknet_model,
                    args.output,
                    args.inpaint_model,
                    args.packages,
                    TrackNetRuntimeConfig(batch_size=args.batch_size),
                    args.start,
                    args.end,
                ),
                indent=2,
            )
        )
    elif args.command == "fuse-shuttle":
        print(json.dumps(fuse_shuttle_detections(args.yolo, args.tracknet, args.config, args.output), indent=2))
    elif args.command == "export-tracknet-labels":
        print(
            json.dumps(
                export_tracknet_labels(args.video, args.trajectory, args.output, args.annotations),
                indent=2,
            )
        )
    elif args.command == "train-tracknet":
        print(
            json.dumps(
                train_tracknet(
                    args.video,
                    args.config,
                    args.labels,
                    args.base_model,
                    args.output,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.maximum_windows,
                ),
                indent=2,
            )
        )
    elif args.command == "track-shuttle":
        result = analyze_shuttle_trajectory(
            args.detections,
            args.output,
            args.contact_features,
            args.video,
            args.annotations,
            args.config,
        )
        if args.preview is not None:
            render_shuttle_trajectory(
                args.video,
                args.output,
                args.config,
                args.preview,
                args.ffmpeg,
                args.encoder,
                style=args.style,
            )
            result["preview"] = str(args.preview)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate-trajectory":
        print(
            json.dumps(
                evaluate_shuttle_annotations(
                    args.detections,
                    args.trajectory,
                    args.annotations,
                    args.output,
                    args.config,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        print(json.dumps(train_model(args.features, args.rallies, args.model, args.report, args.family), indent=2))
    elif args.command == "train-multi":
        print(json.dumps(train_multi_model(args.dataset, args.model, args.report, args.family), indent=2))
    elif args.command == "benchmark-models":
        print(json.dumps(benchmark_multi_models(args.dataset, args.report, args.families), indent=2))
    elif args.command == "regression-gate":
        print(json.dumps(regression_gate(args.dataset, args.baseline_model, args.report, args.family), indent=2))
    elif args.command == "train-guarded":
        print(
            json.dumps(
                train_guarded_candidate(
                    args.dataset,
                    args.baseline_model,
                    args.candidate_model,
                    args.gate_report,
                    args.training_report,
                    args.family,
                ),
                indent=2,
            )
        )
    elif args.command == "predict":
        predict_model(args.features, args.model, args.output)
    elif args.command == "segment":
        result = segment_rallies(
            args.features,
            args.probabilities,
            args.output,
            args.start_threshold,
            args.keep_threshold,
            args.preroll,
            args.postroll,
            args.end_pending,
            args.maximum_internal_gap,
            not args.keep_handoffs,
            args.shuttle_trajectory,
            args.adapter,
        )
        print(f"rallies={len(result)} output={args.output}")
    elif args.command == "fit-adapter":
        print(
            json.dumps(
                fit_segmentation_adapter(
                    args.features,
                    args.probabilities,
                    args.truth,
                    args.output,
                    args.shuttle_trajectory,
                    args.base_adapter,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate":
        print(json.dumps(evaluate_rallies(args.predicted, args.truth, args.output), indent=2))
    elif args.command == "review":
        print(f"clips={render_boundary_reviews(args.video, args.rallies, args.output_directory, args.ffmpeg)}")
    elif args.command == "analyze-actions":
        print(
            json.dumps(
                analyze_rally_actions(
                    args.features,
                    args.rallies,
                    args.output,
                    args.summary,
                    args.audio_events,
                    args.contacts,
                    args.events,
                    args.trajectory,
                    args.config,
                ),
                indent=2,
            )
        )
    elif args.command == "analyze-contacts":
        print(json.dumps(analyze_contacts(args.features, args.rallies, args.output, args.summary), indent=2))
    elif args.command == "analyze-events":
        print(
            json.dumps(
                analyze_terminal_events(
                    args.video,
                    args.config,
                    args.rallies,
                    args.trajectory,
                    args.output,
                    args.contacts,
                    args.summary,
                    args.probabilities,
                    args.features,
                ),
                indent=2,
            )
        )
    elif args.command == "analyze-score":
        print(
            json.dumps(
                analyze_score(
                    args.rallies,
                    args.output,
                    args.events,
                    args.corrections,
                    args.summary,
                    args.initial_server,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "evaluate-score":
        print(json.dumps(evaluate_score(args.predicted, args.truth, args.output), indent=2))
    elif args.command == "render":
        render_rallies(
            args.video,
            args.rallies,
            args.output,
            args.ffmpeg,
            args.encoder,
            args.quality,
            args.fps,
            args.width,
            args.height,
        )
    elif args.command == "studio":
        from .studio import run_studio

        if args.video is None and args.library is None:
            parser.error("studio requires --video or --library")

        run_studio(
            args.video,
            args.rallies,
            args.output,
            args.proxy,
            args.host,
            args.port,
            not args.no_open,
            args.ffmpeg,
            args.encoder,
            args.quality,
            args.library,
            args.config,
            args.model,
            args.pose_model,
            args.packages,
            args.shuttle_model,
            args.tracknet_model,
            args.inpaint_model,
            args.shuttle_mode,
            args.tracknet_packages,
        )
