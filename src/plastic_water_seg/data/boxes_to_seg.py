"""Turn a box-labelled real dataset into YOLO-seg labels with SAM 2.

Public floating-plastic datasets (e.g. Kili "Plastic in River", FloW-Img)
mostly ship bounding boxes, not masks. This is the same box-prompted SAM 2
step used to build the test-set labels, applied to a training set.

Input: a YOLO *detection* dataset (``images/`` + ``labels/`` with
``class cx cy w h`` lines). ``class_map`` maps its class ids onto this
project's ids (``0 plastic_bottle, 1 plastic_bag, 2 plastic_debris``);
classes mapped to ``None`` (e.g. "not plastic") are dropped.

Output: the same images (copied) with YOLO-seg polygon labels. Boxes whose
SAM mask fails a sanity check fall back to the box rectangle, so no object
is silently dropped; the count of fallbacks is returned.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from plastic_water_seg.data.masks import (
    mask_to_polygon,
    polygon_to_yolo_line,
    refine_masks_with_sam,
)

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _read_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int(np.ceil((cx + bw / 2) * w)), int(np.ceil((cy + bh / 2) * h))
        boxes.append((cid, (max(0, x1), max(0, y1), min(w, x2), min(h, y2))))
    return boxes


def convert_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    class_map: dict[int, int | None],
    sam_model: str = "sam2.1_b.pt",
) -> dict[str, int]:
    """Convert one split. Writes ``out_dir/images`` and ``out_dir/labels``."""
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "objects": 0, "sam": 0, "box_fallback": 0, "dropped_class": 0}

    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS):
        with Image.open(img_path) as im:
            rgb = np.array(im.convert("RGB"))
        h, w = rgb.shape[:2]
        boxes = []
        for cid, box in _read_boxes(labels_dir / f"{img_path.stem}.txt", w, h):
            mapped = class_map.get(cid)
            if mapped is None:
                stats["dropped_class"] += 1
                continue
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                boxes.append((mapped, box))

        lines = []
        masks = refine_masks_with_sam(rgb, [b for _, b in boxes], sam_model) if boxes else []
        if masks is None or len(masks) != len(boxes):
            masks = [None] * len(boxes)
        for (cid, (x1, y1, x2, y2)), m in zip(boxes, masks, strict=True):
            box_mask = np.zeros((h, w), dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            use = None
            if m is not None:
                clipped = m & box_mask
                # A good object mask fills a reasonable share of its box.
                fill = clipped.sum() / max(box_mask.sum(), 1)
                if 0.15 <= fill <= 0.98:
                    use = clipped
            if use is None:
                use = box_mask
                stats["box_fallback"] += 1
            else:
                stats["sam"] += 1
            if (poly := mask_to_polygon(use, min_area=4.0)) is not None:
                lines.append(polygon_to_yolo_line(cid, poly, w, h))
                stats["objects"] += 1

        shutil.copy2(img_path, out_dir / "images" / img_path.name)
        (out_dir / "labels" / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        stats["images"] += 1

    log.info("Converted %s: %s", images_dir, stats)
    return stats
