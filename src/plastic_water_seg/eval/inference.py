"""Inference + water contamination metric for trained YOLO-seg models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    image: np.ndarray  # RGB
    boxes: np.ndarray  # (n, 4) xyxy
    masks: np.ndarray  # (n, h, w) bool
    class_ids: np.ndarray
    class_names: list[str]
    scores: np.ndarray


def predict(
    weights: str | Path,
    image_path: str | Path,
    conf: float = 0.35,
    device: str = "0",
) -> InferenceResult:
    """Run a trained YOLO-seg checkpoint on one image."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    image = np.array(Image.open(image_path).convert("RGB"))
    res = model.predict(image, conf=conf, device=device, verbose=False)[0]

    n = 0 if res.boxes is None else len(res.boxes)
    boxes = res.boxes.xyxy.cpu().numpy() if n else np.zeros((0, 4))
    scores = res.boxes.conf.cpu().numpy() if n else np.zeros(0)
    class_ids = res.boxes.cls.cpu().numpy().astype(int) if n else np.zeros(0, dtype=int)
    masks = (
        (res.masks.data.cpu().numpy() > 0.5)
        if res.masks is not None
        else np.zeros((0, *image.shape[:2]), dtype=bool)
    )
    return InferenceResult(
        image=image,
        boxes=boxes,
        masks=masks,
        class_ids=class_ids,
        class_names=[res.names[i] for i in class_ids],
        scores=scores,
    )


def contamination_index(result: InferenceResult) -> float:
    """Surface Plastic Contamination Index: % of pixels covered by plastic masks."""
    h, w = result.image.shape[:2]
    if len(result.masks) == 0:
        return 0.0
    plastic_pixels = np.any(result.masks, axis=0).sum()
    return float(plastic_pixels / (h * w) * 100.0)


def visualize(result: InferenceResult, save_path: str | Path | None = None) -> None:
    """Plot detections with masks, boxes, labels and the contamination index."""
    ratio = contamination_index(result)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(result.image)
    ax.set_title(
        f"Plastic residue detection — {len(result.class_ids)} objects, coverage {ratio:.1f}%",
        fontsize=12,
    )
    ax.axis("off")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(len(result.class_ids)):
        color = colors[i % len(colors)]
        mask = result.masks[i]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (*color[:3], 0.45)
        ax.imshow(overlay)
        x1, y1, x2, y2 = result.boxes[i]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 6),
            f"{result.class_names[i]} {result.scores[i]:.0%}",
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"),
        )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        log.info("Saved visualization to %s", save_path)
    plt.show()
