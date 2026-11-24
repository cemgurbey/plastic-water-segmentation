"""Look at the data before training on it.

``save_label_grid`` draws the YOLO-seg labels over a sample of images, and
``label_size_stats`` reports how large the labelled objects are relative to
the frame. Comparing that with the real test set (median ~0.2% of the frame)
is the quickest way to catch a scale mismatch between training and testing.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from plastic_water_seg.config import CATEGORY_NAMES

_COLORS = [(255, 51, 102), (51, 204, 102), (51, 153, 255)]  # RGB per class


def _read_polygons(label_path: Path) -> list[tuple[int, np.ndarray]]:
    polys = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
        polys.append((int(parts[0]), coords))
    return polys


def _polygon_area(coords: np.ndarray) -> float:
    """Shoelace area of a normalized polygon (fraction of the image)."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _label_for(image_path: Path, labels_dir: Path | None) -> Path:
    if labels_dir is not None:
        return labels_dir / f"{image_path.stem}.txt"
    # YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt
    parts = list(image_path.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_size_stats(
    images_dir: str | Path, labels_dir: str | Path | None = None
) -> dict[str, dict[str, float]]:
    """Per-class object size (% of image area): count, median, p25, share < 0.5%."""
    images_dir = Path(images_dir)
    lbl = Path(labels_dir) if labels_dir else None
    sizes: dict[str, list[float]] = {n: [] for n in CATEGORY_NAMES}
    for img in sorted(images_dir.glob("*.jpg")):
        for cid, coords in _read_polygons(_label_for(img, lbl)):
            if 0 <= cid < len(CATEGORY_NAMES):
                sizes[CATEGORY_NAMES[cid]].append(100.0 * _polygon_area(coords))
    out = {}
    for name, vals in sizes.items():
        v = np.array(vals) if vals else np.zeros(0)
        out[name] = {
            "count": int(len(v)),
            "median_pct": round(float(np.median(v)), 3) if len(v) else 0.0,
            "p25_pct": round(float(np.percentile(v, 25)), 3) if len(v) else 0.0,
            "share_under_0.5pct": round(float((v < 0.5).mean()), 3) if len(v) else 0.0,
        }
    return out


def save_label_grid(
    images_dir: str | Path,
    out_path: str | Path,
    labels_dir: str | Path | None = None,
    n: int = 24,
    tile: int = 384,
    seed: int = 0,
) -> Path:
    """Save a grid of ``n`` random images with their label polygons drawn on."""
    images_dir = Path(images_dir)
    lbl = Path(labels_dir) if labels_dir else None
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images in {images_dir}")
    random.Random(seed).shuffle(images)
    images = images[:n]
    cols = min(6, len(images))
    rows = math.ceil(len(images) / cols)
    grid = np.full((rows * tile, cols * tile, 3), 255, dtype=np.uint8)
    for k, img_path in enumerate(images):
        img = np.array(Image.open(img_path).convert("RGB").resize((tile, tile)))
        overlay = img.copy()
        for cid, coords in _read_polygons(_label_for(img_path, lbl)):
            pts = np.round(coords * tile).astype(np.int32)
            color = _COLORS[cid % len(_COLORS)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        r, c = divmod(k, cols)
        grid[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = img
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path, quality=90)
    return out_path
