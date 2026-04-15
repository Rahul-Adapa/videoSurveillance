#!/usr/bin/env python3
"""CLI entry point for the Video Surveillance Pipeline.

Usage:
    python run.py --video input.mp4 --zones zones.json --output results/
    python run.py --video input.mp4 --output results/ --model yolov8s.pt
    python run.py --video input.mp4 --output results/ --config config.json
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from src.pipeline import PipelineConfig, SurveillancePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video Surveillance: Detection, Tracking & Event Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video surveillance.mp4 --zones config/zones.json --output results/
  %(prog)s --video input.mp4 --output results/ --model yolov8s.pt --device cuda
  %(prog)s --video input.mp4 --output results/ --confidence 0.5 --tracker botsort
        """,
    )

    parser.add_argument(
        "--video", "-v",
        type=str, required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--zones", "-z",
        type=str, default=None,
        help="Path to zones JSON config (optional, default zones used if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str, default=None,
        help="Path to pipeline config JSON (overrides other CLI args)",
    )

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", "-m",
        type=str, default="yolov8n.pt",
        help="YOLO model name/path (default: yolov8n.pt). Options: yolov8n/s/m/l/x.pt",
    )
    model_group.add_argument(
        "--device",
        type=str, default="",
        help="Device for inference: 'cuda', 'cpu', or '' for auto (default: auto)",
    )
    model_group.add_argument(
        "--img-size",
        type=int, default=640,
        help="Input image size for detection (default: 640)",
    )

    detect_group = parser.add_argument_group("Detection Options")
    detect_group.add_argument(
        "--confidence",
        type=float, default=0.35,
        help="Detection confidence threshold (default: 0.35)",
    )
    detect_group.add_argument(
        "--iou-threshold",
        type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )

    track_group = parser.add_argument_group("Tracking Options")
    track_group.add_argument(
        "--tracker",
        type=str, default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracker algorithm (default: bytetrack)",
    )
    track_group.add_argument(
        "--track-buffer",
        type=int, default=60,
        help="Frames to keep lost tracks (default: 60)",
    )

    event_group = parser.add_argument_group("Event Detection Options")
    event_group.add_argument(
        "--loiter-threshold",
        type=float, default=50.0,
        help="Loitering displacement threshold in pixels (default: 50.0)",
    )
    event_group.add_argument(
        "--zone-method",
        type=str, default="bottom_center",
        choices=["bottom_center", "center", "intersect", "overlap"],
        help="Zone check method (default: bottom_center)",
    )
    event_group.add_argument(
        "--cooldown",
        type=int, default=150,
        help="Event cooldown in frames (default: 150)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video output",
    )
    output_group.add_argument(
        "--frame-skip",
        type=int, default=0,
        help="Process every Nth frame (0=all frames, default: 0)",
    )
    output_group.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable trajectory trails",
    )

    parser.add_argument(
        "--log-level",
        type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        config = PipelineConfig(
            model_name=args.model,
            tracker_type=args.tracker,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            device=args.device,
            img_size=args.img_size,
            loiter_displacement_threshold=args.loiter_threshold,
            zone_check_method=args.zone_method,
            cooldown_frames=args.cooldown,
            track_buffer=args.track_buffer,
            show_trajectories=not args.no_trajectories,
            output_video=not args.no_video,
            frame_skip=args.frame_skip,
        )

    logger.info("=" * 60)
    logger.info("Video Surveillance Pipeline")
    logger.info("=" * 60)
    logger.info(f"Video:   {args.video}")
    logger.info(f"Zones:   {args.zones or 'auto-generated'}")
    logger.info(f"Output:  {args.output}")
    logger.info(f"Model:   {config.model_name}")
    logger.info(f"Tracker: {config.tracker_type}")
    logger.info(f"Device:  {config.device or 'auto'}")
    logger.info("=" * 60)

    pipeline = SurveillancePipeline(config=config)

    summary = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output,
        zones_path=args.zones,
    )

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    stats = summary["stats"]
    logger.info(f"Frames processed: {stats['processed_frames']}/{stats['total_frames']}")
    logger.info(f"Unique persons:   {stats['unique_persons']}")
    logger.info(f"Total events:     {stats['total_events']}")
    logger.info(f"Avg FPS:          {stats['avg_fps']:.1f}")
    logger.info(f"Processing time:  {stats['processing_time_sec']:.1f}s")

    events_summary = summary.get("events_summary", {})
    if events_summary.get("by_type"):
        logger.info("Events by type:")
        for etype, count in events_summary["by_type"].items():
            logger.info(f"  {etype}: {count}")

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
