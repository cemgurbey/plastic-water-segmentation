"""Ground-truth evaluation of segmentation predictions on the real test set.

Loads YOLO-seg labels from ``test_images/labels/``, matches predicted masks
to ground-truth masks by mask IoU (greedy, same class, IoU >= threshold),
and reports per-class precision / recall / F1 plus mean matched IoU.

A note on what this measures: dense debris patches are labelled as one
region each (labelling every fragment is not feasible), so debris recall
here measures region detection, not fragment counting.

Besides the strict per-class score, this module reports:

* a class-agnostic "plastic" score (a detection counts if it covers any
  plastic, whatever class it was given) — the contamination index ignores
  class anyway, and class confusion was the first failure mode on real data;
* per-image macro recall, because two photos (05 and 40) hold 56% of all
  ground-truth bottles and dominate the pooled bottle recall;
* a fixed dev/test split (``test_images/SPLITS.csv``) so decisions are made
  on ``dev`` and ``test`` stays an honest held-out number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from plastic_water_seg.config import CATEGORY_NAMES
from plastic_water_seg.eval.inference import InferenceResult


def load_yolo_labels(
    label_path: str | Path, image_shape: tuple[int, int]
) -> list[tuple[int, np.ndarray]]:
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
        cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
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


AGNOSTIC_NAME = "plastic"


def evaluate_predictions(
    gt_labels: list[tuple[int, np.ndarray]],
    result: InferenceResult,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, ClassMetrics]:
    """Match one image's predictions to its ground truth; returns per-class metrics.

    With ``class_agnostic=True`` every class is merged into one ``"plastic"``
    class, so a debris region predicted as "bag" still counts as a hit.
    """
    if class_agnostic:
        gt_labels = [(0, m) for _, m in gt_labels]
        pred_classes = np.zeros(len(result.class_ids), dtype=int)
        names: tuple[str, ...] = (AGNOSTIC_NAME,)
    else:
        pred_classes = np.asarray(result.class_ids)
        names = CATEGORY_NAMES

    per_class: dict[str, ClassMetrics] = {name: ClassMetrics() for name in names}
    for class_id, name in enumerate(names):
        gt_masks = [m for cid, m in gt_labels if cid == class_id]
        pred_idx = [i for i, cid in enumerate(pred_classes) if cid == class_id]
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


def macro_recall(per_image: list[dict[str, ClassMetrics]]) -> float:
    """Mean over images (with ground truth) of each image's overall recall.

    Unlike pooled recall, every labelled photo weighs the same, so one photo
    with 48 small bottles cannot dominate the score.
    """
    recalls = []
    for image_metrics in per_image:
        tp = sum(m.tp for m in image_metrics.values())
        fn = sum(m.fn for m in image_metrics.values())
        if tp + fn:
            recalls.append(tp / (tp + fn))
    return float(np.mean(recalls)) if recalls else 0.0


def load_splits(splits_csv: str | Path) -> dict[str, str]:
    """``filename -> split`` from ``test_images/SPLITS.csv`` (dev / test)."""
    import csv

    path = Path(splits_csv)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["filename"]: row["split"] for row in csv.DictReader(f)}


def summarize(per_image: list[dict[str, ClassMetrics]]) -> dict[str, ClassMetrics]:
    """Aggregate per-image metrics into totals (works for per-class or agnostic)."""
    total: dict[str, ClassMetrics] = {}
    for image_metrics in per_image:
        for name, m in image_metrics.items():
            t = total.setdefault(name, ClassMetrics())
            t.tp += m.tp
            t.fp += m.fp
            t.fn += m.fn
            t.matched_ious.extend(m.matched_ious)
    return total


def overall(metrics: dict[str, ClassMetrics]) -> ClassMetrics:
    """Micro-averaged totals across classes."""
    tot = ClassMetrics()
    for m in metrics.values():
        tot.tp += m.tp
        tot.fp += m.fp
        tot.fn += m.fn
        tot.matched_ious.extend(m.matched_ious)
    return tot


def metrics_table(metrics: dict[str, ClassMetrics]) -> str:
    """One-line-per-class summary, plus an overall micro-averaged row."""
    lines = ["class            TP   FP   FN    P      R      F1     mean IoU"]
    rows = list(metrics.items())
    if len(rows) > 1:
        rows.append(("OVERALL", overall(metrics)))
    for name, m in rows:
        lines.append(
            f"{name:<16} {m.tp:>3}  {m.fp:>3}  {m.fn:>3}   "
            f"{m.precision:.2f}   {m.recall:.2f}   {m.f1:.2f}   {m.mean_iou:.2f}"
        )
    return "\n".join(lines)
