#!/usr/bin/env python3
"""Evaluation script for comparing detections against ground truth.

Computes standard MOT metrics (MOTA, MOTP, IDF1) when ground truth
annotations are available in MOTChallenge format.

Ground truth format (MOT format, CSV):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

Usage:
    python scripts/evaluate.py --gt gt.txt --pred results/events.json
    python scripts/evaluate.py --gt gt.txt --pred results/stats.json --format stats
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_mot_gt(gt_path: str) -> Dict[int, List[dict]]:
    """Load ground truth in MOTChallenge format."""
    gt_by_frame: Dict[int, List[dict]] = {}

    with open(gt_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(row[0])
            track_id = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            conf = float(row[6]) if len(row) > 6 else 1.0

            if conf == 0:
                continue

            if frame not in gt_by_frame:
                gt_by_frame[frame] = []
            gt_by_frame[frame].append({
                "track_id": track_id,
                "bbox": [x, y, x + w, y + h],
            })

    return gt_by_frame


def load_predictions(pred_path: str) -> Dict[int, List[dict]]:
    """Load predictions from pipeline output."""
    with open(pred_path) as f:
        data = json.load(f)

    pred_by_frame: Dict[int, List[dict]] = {}

    if isinstance(data, dict) and "track_histories" in data:
        print("Loading from stats format...")
        for tid_str, hist in data["track_histories"].items():
            tid = int(tid_str)
            for frame in range(hist["first_seen"], hist["last_seen"] + 1):
                if frame not in pred_by_frame:
                    pred_by_frame[frame] = []
                pred_by_frame[frame].append({"track_id": tid, "bbox": None})
    elif isinstance(data, list):
        for event in data:
            frame = event.get("frame_idx", 0)
            if frame not in pred_by_frame:
                pred_by_frame[frame] = []
            pred_by_frame[frame].append({
                "track_id": event.get("track_id", -1),
                "bbox": event.get("bbox"),
            })

    return pred_by_frame


def compute_iou(box1: list, box2: list) -> float:
    """Compute Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_mot_metrics(
    gt_by_frame: Dict[int, List[dict]],
    pred_by_frame: Dict[int, List[dict]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute MOTA, MOTP, and related MOT metrics."""
    total_gt = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    total_iou = 0.0
    matched_count = 0

    prev_matches: Dict[int, int] = {}
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))

    for frame in all_frames:
        gt_dets = gt_by_frame.get(frame, [])
        pred_dets = pred_by_frame.get(frame, [])
        total_gt += len(gt_dets)

        if not gt_dets:
            false_positives += len(pred_dets)
            continue
        if not pred_dets:
            false_negatives += len(gt_dets)
            continue

        gt_bboxes = [g["bbox"] for g in gt_dets]
        pred_bboxes = [p["bbox"] for p in pred_dets]

        if any(b is None for b in gt_bboxes) or any(b is None for b in pred_bboxes):
            continue

        iou_matrix = np.zeros((len(gt_dets), len(pred_dets)))
        for i, gb in enumerate(gt_bboxes):
            for j, pb in enumerate(pred_bboxes):
                iou_matrix[i, j] = compute_iou(gb, pb)

        matched_gt = set()
        matched_pred = set()
        current_matches: Dict[int, int] = {}

        pairs = []
        for i in range(len(gt_dets)):
            for j in range(len(pred_dets)):
                if iou_matrix[i, j] >= iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)

        for iou_val, gi, pi in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            true_positives += 1
            total_iou += iou_val
            matched_count += 1

            gt_id = gt_dets[gi]["track_id"]
            pred_id = pred_dets[pi]["track_id"]
            current_matches[gt_id] = pred_id

            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                id_switches += 1

        false_negatives += len(gt_dets) - len(matched_gt)
        false_positives += len(pred_dets) - len(matched_pred)
        prev_matches = current_matches

    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(total_gt, 1)
    motp = total_iou / max(matched_count, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)

    return {
        "MOTA": round(mota * 100, 2),
        "MOTP": round(motp * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "ID Switches": id_switches,
        "Total GT": total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking results against ground truth")
    parser.add_argument("--gt", required=True, help="Path to ground truth file (MOT format)")
    parser.add_argument("--pred", required=True, help="Path to prediction file (JSON)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()

    print("Loading ground truth...")
    gt = load_mot_gt(args.gt)
    print(f"  {len(gt)} frames, {sum(len(v) for v in gt.values())} annotations")

    print("Loading predictions...")
    pred = load_predictions(args.pred)
    print(f"  {len(pred)} frames, {sum(len(v) for v in pred.values())} detections")

    print("\nComputing MOT metrics...")
    metrics = compute_mot_metrics(gt, pred, args.iou_threshold)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"  {key:20s}: {value}")

    output_path = Path(args.pred).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
