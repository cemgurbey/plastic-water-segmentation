"""Evaluate a checkpoint on the real test photos (one call, all metrics).

``evaluate_dir`` runs the model over ``test_images/`` (optionally only the
``dev`` or ``test`` split) and writes, under ``out_dir``:

* ``summary.csv`` — detections per class + contamination index per image
* ``metrics.csv`` — per-class P/R/F1 at mask IoU >= 0.5 (+ overall row)
* ``metrics_agnostic.csv`` — the same, with all classes merged into "plastic"
* ``per_image.csv`` — per-image TP/FP/FN (class-agnostic) for spotting outliers
* one ``*_pred.jpg`` visualization per image (``save_vis=True``)

``map_eval`` adds threshold-free mAP50 / mAP50-95 (box and mask) through
Ultralytics' own validator, so progress is visible even when nothing clears
the confidence threshold.
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from plastic_water_seg.config import CATEGORY_NAMES
from plastic_water_seg.eval.inference import contamination_index, load_model, predict, visualize
from plastic_water_seg.eval.metrics import (
    ClassMetrics,
    evaluate_predictions,
    load_splits,
    load_yolo_labels,
    macro_recall,
    metrics_table,
    overall,
    summarize,
)

log = logging.getLogger(__name__)


def _select_images(images_dir: Path, splits_csv: Path | None, split: str | None) -> list[Path]:
    images = sorted(images_dir.glob("*.jpg"))
    if split in (None, "all"):
        return images
    splits = load_splits(splits_csv) if splits_csv else {}
    if not splits:
        raise FileNotFoundError(f"split={split!r} requested but no splits file at {splits_csv}")
    return [p for p in images if splits.get(p.name) == split]


def _write_metrics(path: Path, metrics: dict[str, ClassMetrics]) -> None:
    rows = list(metrics.items())
    if len(rows) > 1:
        rows.append(("overall", overall(metrics)))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "tp", "fp", "fn", "precision", "recall", "f1", "mean_iou"])
        for name, m in rows:
            writer.writerow(
                [
                    name,
                    m.tp,
                    m.fp,
                    m.fn,
                    round(m.precision, 3),
                    round(m.recall, 3),
                    round(m.f1, 3),
                    round(m.mean_iou, 3),
                ]
            )


def evaluate_dir(
    weights,
    images_dir: str | Path = "test_images",
    labels_dir: str | Path | None = None,
    out_dir: str | Path = "results/latest",
    split: str | None = "dev",
    splits_csv: str | Path | None = None,
    conf: float = 0.35,  # same as earlier reports, so numbers stay comparable
    imgsz: int = 1024,
    iou_threshold: float = 0.5,
    save_vis: bool = True,
) -> dict[str, float]:
    """Run ``weights`` on the real photos and write every metric file.

    ``weights`` is a checkpoint path or a loaded Ultralytics model. Defaults to
    the ``dev`` split: iterate on dev, report ``test`` rarely.
    Returns a small dict of headline numbers.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else images_dir / "labels"
    splits_csv = Path(splits_csv) if splits_csv else images_dir / "SPLITS.csv"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(weights)) if isinstance(weights, str | Path) else weights

    images = _select_images(images_dir, splits_csv, split)
    summary_rows, per_class, agnostic, per_image_rows = [], [], [], []
    for img_path in images:
        result = predict(model, img_path, conf=conf, imgsz=imgsz)
        ratio = contamination_index(result)
        counts = {n: int((result.class_ids == i).sum()) for i, n in enumerate(CATEGORY_NAMES)}
        summary_rows.append(
            {"image": img_path.name, **counts, "contamination_pct": round(ratio, 2)}
        )
        if save_vis:
            import matplotlib.pyplot as plt

            visualize(result, save_path=out_dir / f"{img_path.stem}_pred.jpg")
            plt.close("all")
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            gt = load_yolo_labels(label_path, result.image.shape[:2])
            per_class.append(evaluate_predictions(gt, result, iou_threshold))
            ag = evaluate_predictions(gt, result, iou_threshold, class_agnostic=True)
            agnostic.append(ag)
            m = ag["plastic"]
            per_image_rows.append(
                {
                    "image": img_path.name,
                    "gt": m.tp + m.fn,
                    "pred": m.tp + m.fp,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                }
            )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *CATEGORY_NAMES, "contamination_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    headline: dict[str, float] = {"images": len(images)}
    if per_class:
        totals, ag_totals = summarize(per_class), summarize(agnostic)
        _write_metrics(out_dir / "metrics.csv", totals)
        _write_metrics(out_dir / "metrics_agnostic.csv", ag_totals)
        o, a = overall(totals), ag_totals["plastic"]
        headline.update(
            per_class_f1=round(o.f1, 3),
            per_class_recall=round(o.recall, 3),
            agnostic_precision=round(a.precision, 3),
            agnostic_recall=round(a.recall, 3),
            agnostic_f1=round(a.f1, 3),
            macro_recall_agnostic=round(macro_recall(agnostic), 3),
        )
        log.info("Per-class (mask IoU >= %.2f):\n%s", iou_threshold, metrics_table(totals))
        log.info("Class-agnostic:\n%s", metrics_table(ag_totals))
    log.info("Headline (%s split): %s", split or "all", headline)
    return headline


def make_yolo_eval_dataset(
    images_dir: str | Path = "test_images",
    labels_dir: str | Path | None = None,
    out_dir: str | Path = "eval_data",
    splits_csv: str | Path | None = None,
) -> Path:
    """Lay the real photos out as a YOLO dataset (``dev`` -> val, ``test`` -> test).

    Ultralytics finds labels by swapping ``/images/`` for ``/labels/`` in the
    path, which ``test_images/`` does not follow, so files are copied into
    ``out_dir/images/{dev,test}`` and ``out_dir/labels/{dev,test}``.
    Returns the path of the written ``eval.yaml``.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else images_dir / "labels"
    splits = load_splits(splits_csv or images_dir / "SPLITS.csv")
    out_dir = Path(out_dir)
    for img in sorted(images_dir.glob("*.jpg")):
        split = splits.get(img.name, "test")
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_dir / "images" / split / img.name)
        lbl = labels_dir / f"{img.stem}.txt"
        dst = out_dir / "labels" / split / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst)
        else:
            dst.write_text("")
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CATEGORY_NAMES))
    yaml_path = out_dir / "eval.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/dev\nval: images/dev\ntest: images/test\n\n"
        f"names:\n{names_block}\n"
    )
    return yaml_path


def map_eval(
    weights: str | Path, eval_yaml: str | Path, split: str = "val", imgsz: int = 1024
) -> dict[str, float]:
    """Threshold-free mAP50 / mAP50-95 (box and mask) via ``model.val()``.

    ``split="val"`` scores the dev photos, ``split="test"`` the held-out ones.
    """
    model = load_model(str(weights))
    m = model.val(
        data=str(eval_yaml), split=split, imgsz=imgsz, conf=0.001, plots=False, verbose=False
    )
    out = {
        "box_map50": round(float(m.box.map50), 4),
        "box_map50_95": round(float(m.box.map), 4),
        "mask_map50": round(float(m.seg.map50), 4),
        "mask_map50_95": round(float(m.seg.map), 4),
    }
    log.info("mAP on %s: %s", split, out)
    return out
