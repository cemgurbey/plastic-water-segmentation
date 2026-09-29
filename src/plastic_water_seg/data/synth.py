"""Dataset synthesizer: paint debris into water scenes, export YOLO-seg labels.

Pipeline per image:
  1. Pick a water background, crop/resize to the target size.
  2. Sample non-overlapping elliptical regions.
  3. Paint one debris instance per region with a DebrisPainter
     (diffusion inpainting by default, procedural fallback).
  4. Refine each region into a precise instance mask with SAM 2
     (falls back to the eroded inpaint region when SAM is unavailable).
  5. Export the mask polygon as a YOLO segmentation label.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from plastic_water_seg.config import CATEGORIES, CATEGORY_NAMES, Category, SynthConfig
from plastic_water_seg.data.masks import (
    bbox_of_mask,
    mask_to_polygon,
    polygon_to_yolo_line,
    refine_masks_with_sam,
)
from plastic_water_seg.data.painters import DebrisPainter

log = logging.getLogger(__name__)

_BG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetSynthesizer:
    def __init__(
        self,
        backgrounds_dir: str | Path,
        output_dir: str | Path,
        painter: DebrisPainter,
        config: SynthConfig | None = None,
        categories: tuple[Category, ...] = CATEGORIES,
    ) -> None:
        self.config = config or SynthConfig()
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.painter = painter
        self.categories = categories

        self.backgrounds = sorted(
            p for p in self.backgrounds_dir.iterdir() if p.suffix.lower() in _BG_EXTS
        )
        if not self.backgrounds:
            raise FileNotFoundError(f"No background images in {self.backgrounds_dir}")

    # ------------------------------------------------------------------ #
    # region sampling
    # ------------------------------------------------------------------ #
    def _sample_region_mask(
        self, w: int, h: int, existing: list[tuple[int, int, int, int]], rng: random.Random
    ) -> Image.Image | None:
        cfg = self.config
        for _ in range(25):
            coverage = rng.uniform(cfg.min_coverage, cfg.max_coverage)
            area = coverage * w * h
            aspect = rng.uniform(0.4, 2.5)  # ellipse height / width
            a = min((area / (math.pi * aspect)) ** 0.5, w * 0.30)
            b = min(aspect * a, h * 0.30)
            if a < 8 or b < 8 or 2 * a >= w or 2 * b >= h:
                continue
            cx = rng.uniform(a, w - a)
            cy = rng.uniform(b, h - b)
            box = (int(cx - a), int(cy - b), int(cx + a), int(cy + b))
            if any(self._iou(box, prev) > 0.25 for prev in existing):
                continue
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).ellipse([box[0], box[1], box[2], box[3]], fill=255)
            return mask
        return None

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    # background prep
    # ------------------------------------------------------------------ #
    def _prepare_background(self, path: Path, rng: random.Random) -> Image.Image:
        w, h = self.config.image_size
        with Image.open(path).convert("RGB") as img:
            bw, bh = img.size
            if bw >= w and bh >= h:  # random crop keeps native resolution detail
                x = rng.randint(0, bw - w)
                y = rng.randint(0, bh - h)
                return img.crop((x, y, x + w, y + h))
            return img.resize((w, h), Image.LANCZOS)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _synthesize_one(self, index: int, rng: random.Random) -> tuple[Image.Image, list[str]]:
        cfg = self.config
        w, h = cfg.image_size
        canvas = self._prepare_background(rng.choice(self.backgrounds), rng)

        n_objects = rng.randint(cfg.min_objects, cfg.max_objects)
        instances: list[tuple[int, np.ndarray]] = []  # (class_id, coarse mask)
        boxes: list[tuple[int, int, int, int]] = []

        for _ in range(n_objects):
            region = self._sample_region_mask(w, h, boxes, rng)
            if region is None:
                continue
            category = rng.choice(self.categories)
            class_id = CATEGORY_NAMES.index(category.name)
            canvas = self.painter.paint(canvas, region, category, rng)
            region_np = np.array(region) > 127
            if box := bbox_of_mask(region_np):
                boxes.append(box)
                instances.append((class_id, region_np))

        # Refine coarse inpaint regions into precise masks with SAM 2.
        refined: list[np.ndarray] | None = None
        if cfg.sam_refine and boxes:
            refined = refine_masks_with_sam(np.array(canvas), boxes)
        if refined is None or len(refined) != len(instances):
            # Conservative fallback: slightly erode the inpaint region.
            kernel = np.ones((3, 3), np.uint8)
            refined = [
                cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
                for _, m in instances
            ]

        labels: list[str] = []
        for (class_id, _), mask in zip(instances, refined, strict=True):
            if (polygon := mask_to_polygon(mask)) is not None:
                labels.append(polygon_to_yolo_line(class_id, polygon, w, h))
        return canvas, labels

    def _write_split(self, split: str, count: int, seed_offset: int) -> None:
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(count), desc=f"Synthesizing {split}", unit="img"):
            rng = random.Random(self.config.seed + seed_offset + i)
            image, labels = self._synthesize_one(i, rng)
            stem = f"plastic_water_{i:06d}"
            image.save(img_dir / f"{stem}.jpg", quality=95)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))

    def run(self) -> Path:
        """Generate train/val splits and write ``dataset.yaml``. Returns its path."""
        cfg = self.config
        log.info(
            "Synthesizing %d train + %d val images at %dx%d",
            cfg.train_count,
            cfg.val_count,
            *cfg.image_size,
        )
        self._write_split("train", cfg.train_count, seed_offset=0)
        if cfg.val_count > 0:
            self._write_split("val", cfg.val_count, seed_offset=10_000)

        yaml_path = self.output_dir / "dataset.yaml"
        names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CATEGORY_NAMES))
        yaml_path.write_text(
            f"path: {self.output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n{names_block}\n"
        )
        log.info("Wrote dataset manifest: %s", yaml_path)
        return yaml_path
