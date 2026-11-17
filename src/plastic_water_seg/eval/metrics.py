"""Ground-truth evaluation of segmentation predictions on the real test set.

Loads YOLO-seg labels from ``test_images/labels/``, matches predicted masks
to ground-truth masks by mask IoU (greedy, same class, IoU >= threshold),
and reports per-class precision / recall / F1 plus mean matched IoU.

A note on what this measures: dense debris patches are labelled as one
region each (labelling every fragment is not feasible), so debris recall
here measures region detection, not fragment counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from plastic_water_seg.config import CATEGORY_NAMES
from plastic_water_seg.eval.inference import InferenceResult


def load_yolo_labels(label_path: str | Path, image_shape: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    """Parse a YOLO-seg label file into ``(class_id, bool mask)`` pairs."""
    h, w = image_shape
    labels: list[tuple[int, np.ndarray]] = []
    path = Path(label_path)
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 (x, y) points
            continue
        class_id = int(parts[0])
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        labels.append((class_id, mask.astype(bool)))
    return labels


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_ious: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0


def evaluate_predictions(
    gt_labels: list[tuple[int, np.ndarray]],
    result: InferenceResult,
    iou_threshold: float = 0.5,
) -> dict[str, ClassMetrics]:
    """Match one image's predictions to its ground truth; returns per-class metrics."""
    per_class: dict[str, ClassMetrics] = {name: ClassMetrics() for name in CATEGORY_NAMES}
    for class_id, name in enumerate(CATEGORY_NAMES):
        gt_masks = [m for cid, m in gt_labels if cid == class_id]
        pred_idx = [i for i, cid in enumerate(result.class_ids) if cid == class_id]
        # Greedy matching: best IoU pair first.
        pairs: list[tuple[float, int, int]] = []
        for gi, gm in enumerate(gt_masks):
            for pi in pred_idx:
                pairs.append((mask_iou(gm, result.masks[pi]), gi, pi))
        pairs.sort(reverse=True)
        used_gt, used_pred = set(), set()
        for iou, gi, pi in pairs:
            if iou < iou_threshold or gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            per_class[name].tp += 1
            per_class[name].matched_ious.append(iou)
        per_class[name].fp = len(pred_idx) - len(used_pred)
        per_class[name].fn = len(gt_masks) - len(used_gt)
    return per_class


def summarize(per_image: list[dict[str, ClassMetrics]]) -> dict[str, ClassMetrics]:
    """Aggregate per-image metrics into totals."""
    total: dict[str, ClassMetrics] = {name: ClassMetrics() for name in CATEGORY_NAMES}
    for image_metrics in per_image:
        for name, m in image_metrics.items():
            total[name].tp += m.tp
            total[name].fp += m.fp
            total[name].fn += m.fn
            total[name].matched_ious.extend(m.matched_ious)
    return total


def metrics_table(metrics: dict[str, ClassMetrics]) -> str:
    """One-line-per-class summary, plus an overall micro-averaged row."""
    lines = ["class            TP  FP  FN   P      R      F1     mean IoU"]
    tot = ClassMetrics()
    for name in CATEGORY_NAMES:
        m = metrics[name]
        tot.tp += m.tp
        tot.fp += m.fp
        tot.fn += m.fn
        tot.matched_ious.extend(m.matched_ious)
        lines.append(
            f"{name:<16} {m.tp:>2}  {m.fp:>2}  {m.fn:>2}   "
            f"{m.precision:.2f}   {m.recall:.2f}   {m.f1:.2f}   {m.mean_iou:.2f}"
        )
    lines.append(
        f"{'OVERALL':<16} {tot.tp:>2}  {tot.fp:>2}  {tot.fn:>2}   "
        f"{tot.precision:.2f}   {tot.recall:.2f}   {tot.f1:.2f}   {tot.mean_iou:.2f}"
    )
    return "\n".join(lines)
