"""Dataset synthesizer: paint debris into water scenes, export YOLO-seg labels.

Pipeline per image:
  1. Pick a water background, crop/resize to the target size.
  2. Sample non-overlapping elliptical regions, mostly small (log-uniform
     size), because real floating plastic is mostly small in the frame.
  3. Paint one debris instance per region with a DebrisPainter
     (diffusion inpainting by default, procedural fallback).
  4. Quality control: keep the object only if the painter visibly changed the
     region (and, optionally, CLIP agrees it looks like the class). Failed
     objects are reverted and get no label, so the dataset never says
     "bottle" where only water was painted.
  5. Label = the pixels that actually changed, refined with SAM 2 (prompted
     with the changed area's box, clipped to the region). Debris is labelled
     as one filled region, matching the real test-set convention.
  6. Export the mask polygon as a YOLO segmentation label.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from plastic_water_seg.config import CATEGORIES, CATEGORY_NAMES, Category, SynthConfig
from plastic_water_seg.data.masks import (
    bbox_of_mask,
    changed_mask,
    close_to_region,
    expand_box,
    mask_to_polygon,
    polygon_to_yolo_line,
    refine_masks_with_sam,
)
from plastic_water_seg.data.painters import DebrisPainter

log = logging.getLogger(__name__)

_BG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_DEMO_BG_PREFIX = "water_surface_"  # names written by generate_water_backgrounds
_MIN_REAL_BACKGROUNDS = 20


@dataclass
class SynthStats:
    """Counters written to ``synth_stats.json`` so a bad run is visible early."""

    images: int = 0
    objects_attempted: int = 0
    objects_kept: int = 0
    rejected_no_change: int = 0
    rejected_clip: int = 0
    sam_used: int = 0
    sam_rejected: int = 0
    per_class: dict[str, int] = field(default_factory=lambda: {n: 0 for n in CATEGORY_NAMES})
    coverage_pct: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        cov = np.array(self.coverage_pct) if self.coverage_pct else np.zeros(1)
        d["coverage_pct"] = {
            "median": round(float(np.median(cov)), 4),
            "p10": round(float(np.percentile(cov, 10)), 4),
            "p90": round(float(np.percentile(cov, 90)), 4),
        }
        d["keep_rate"] = round(self.objects_kept / max(self.objects_attempted, 1), 3)
        return d


class DatasetSynthesizer:
    def __init__(
        self,
        backgrounds_dir: str | Path,
        output_dir: str | Path,
        painter: DebrisPainter,
        config: SynthConfig | None = None,
        categories: tuple[Category, ...] = CATEGORIES,
        verifier=None,
    ) -> None:
        self.config = config or SynthConfig()
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.painter = painter
        self.categories = categories
        self.verifier = verifier
        if verifier is None and self.config.clip_verify:
            from plastic_water_seg.data.verify import ClipVerifier

            self.verifier = ClipVerifier(
                self.config.clip_model, self.config.clip_min_prob, self.config.device
            )
        self.stats = SynthStats()

        self.backgrounds = sorted(
            p for p in self.backgrounds_dir.iterdir() if p.suffix.lower() in _BG_EXTS
        )
        if not self.backgrounds:
            raise FileNotFoundError(f"No background images in {self.backgrounds_dir}")
        self._check_backgrounds()

    def _check_backgrounds(self) -> None:
        n = len(self.backgrounds)
        n_demo = sum(p.name.startswith(_DEMO_BG_PREFIX) for p in self.backgrounds)
        if n_demo == n:
            log.warning(
                "All %d backgrounds are procedural demo textures. Fine for a smoke test, "
                "but a model trained on them will not transfer to real photos — put real "
                "water photos (not the test images) in %s.",
                n,
                self.backgrounds_dir,
            )
        elif n < _MIN_REAL_BACKGROUNDS:
            log.warning(
                "Only %d backgrounds; aim for %d+ varied real water photos.",
                n,
                _MIN_REAL_BACKGROUNDS * 10,
            )

    # ------------------------------------------------------------------ #
    # region sampling
    # ------------------------------------------------------------------ #
    def _sample_region_mask(
        self, w: int, h: int, existing: list[tuple[int, int, int, int]], rng: random.Random
    ) -> Image.Image | None:
        """Ellipse whose area is log-uniform in [min_coverage, max_coverage].

        Regions never overlap (boxes kept ``min_gap_px`` apart).
        """
        cfg = self.config
        lo, hi = math.log(cfg.min_coverage), math.log(cfg.max_coverage)
        for _ in range(50):
            coverage = math.exp(rng.uniform(lo, hi))
            area = coverage * w * h
            aspect = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))  # b / a
            a = min((area / (math.pi * aspect)) ** 0.5, w * 0.30)
            b = min(aspect * a, h * 0.30)
            if a < 3 or b < 3 or 2 * a >= w or 2 * b >= h:
                continue
            cx = rng.uniform(a, w - a)
            cy = rng.uniform(b, h - b)
            box = (int(cx - a), int(cy - b), int(cx + a), int(cy + b))
            if any(self._boxes_touch(box, prev, cfg.min_gap_px) for prev in existing):
                continue
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).ellipse([box[0], box[1], box[2], box[3]], fill=255)
            return mask
        return None

    @staticmethod
    def _boxes_touch(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
    ) -> bool:
        return not (
            a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]
        )

    # ------------------------------------------------------------------ #
    # background prep
    # ------------------------------------------------------------------ #
    def _prepare_background(self, path: Path, rng: random.Random) -> Image.Image:
        w, h = self.config.image_size
        with Image.open(path) as src:
            img = src.convert("RGB")
        bw, bh = img.size
        if bw >= w and bh >= h:  # random crop keeps native resolution detail
            x = rng.randint(0, bw - w)
            y = rng.randint(0, bh - h)
            return img.crop((x, y, x + w, y + h))
        # Too small: scale up to cover, then crop (keeps the aspect ratio).
        scale = max(w / bw, h / bh)
        img = img.resize((math.ceil(bw * scale), math.ceil(bh * scale)), Image.LANCZOS)
        x = rng.randint(0, img.width - w)
        y = rng.randint(0, img.height - h)
        return img.crop((x, y, x + w, y + h))

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _synthesize_one(self, index: int, rng: random.Random) -> tuple[Image.Image, list[str]]:
        cfg = self.config
        w, h = cfg.image_size
        canvas = self._prepare_background(rng.choice(self.backgrounds), rng)
        self.stats.images += 1

        n_objects = rng.randint(cfg.min_objects, cfg.max_objects)
        # (class_id, changed-pixel mask, label clip box)
        instances: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
        region_boxes: list[tuple[int, int, int, int]] = []

        for _ in range(n_objects):
            region = self._sample_region_mask(w, h, region_boxes, rng)
            if region is None:
                continue
            region_np = np.array(region) > 127
            rbox = bbox_of_mask(region_np)
            if rbox is None:
                continue
            category = rng.choice(self.categories)
            class_id = CATEGORY_NAMES.index(category.name)
            self.stats.objects_attempted += 1

            before = np.array(canvas)
            painted = self.painter.paint(canvas, region, category, rng)
            after = np.array(painted)

            # Only pixels inside the region's box (+ the paste feather) can be
            # part of this object.
            clip_box = expand_box((rbox[0], rbox[1], rbox[2] + 1, rbox[3] + 1), 3, w, h)
            changed = changed_mask(before, after, clip_box, cfg.change_threshold)
            if changed.sum() < cfg.min_changed_fraction * region_np.sum():
                self.stats.rejected_no_change += 1
                continue  # painter left the region (nearly) as it was: no label
            if self.verifier is not None:
                cbox = bbox_of_mask(changed)
                pad = max(8, (cbox[2] - cbox[0]) // 2)
                crop = painted.crop(
                    expand_box((cbox[0], cbox[1], cbox[2] + 1, cbox[3] + 1), pad, w, h)
                )
                if not self.verifier.accepts(crop, category):
                    self.stats.rejected_clip += 1
                    continue  # painted something, but not this class: revert

            canvas = painted
            region_boxes.append(rbox)
            instances.append((class_id, changed, clip_box))

        masks = self._final_masks(canvas, instances)

        labels: list[str] = []
        for (class_id, _, _), mask in zip(instances, masks, strict=True):
            if CATEGORY_NAMES[class_id] == "plastic_debris":
                mask = close_to_region(mask)
            if (polygon := mask_to_polygon(mask, min_area=9.0)) is not None:
                labels.append(polygon_to_yolo_line(class_id, polygon, w, h))
                self.stats.objects_kept += 1
                self.stats.per_class[CATEGORY_NAMES[class_id]] += 1
                self.stats.coverage_pct.append(100.0 * mask.sum() / (w * h))
        return canvas, labels

    def _final_masks(
        self,
        canvas: Image.Image,
        instances: list[tuple[int, np.ndarray, tuple[int, int, int, int]]],
    ) -> list[np.ndarray]:
        """Changed-pixel masks, refined with SAM 2 where it agrees with them."""
        masks = [changed for _, changed, _ in instances]
        if not (self.config.sam_refine and instances):
            return masks
        prompts = []
        for changed in masks:
            x1, y1, x2, y2 = bbox_of_mask(changed)
            prompts.append((x1, y1, x2 + 1, y2 + 1))
        refined = refine_masks_with_sam(np.array(canvas), prompts)
        if refined is None or len(refined) != len(instances):
            return masks
        out = []
        for (_, changed, (x1, y1, x2, y2)), sam_mask in zip(instances, refined, strict=True):
            clipped = np.zeros_like(changed)
            clipped[y1:y2, x1:x2] = sam_mask[y1:y2, x1:x2]
            # SAM can latch onto a reflection or the whole water patch; trust it
            # only when it roughly agrees with what was actually painted.
            ratio = clipped.sum() / max(changed.sum(), 1)
            if 0.3 <= ratio <= 3.0:
                out.append(clipped)
                self.stats.sam_used += 1
            else:
                out.append(changed)
                self.stats.sam_rejected += 1
        return out

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
        summary = self.stats.summary()
        (self.output_dir / "synth_stats.json").write_text(json.dumps(summary, indent=2))
        log.info(
            "Kept %d/%d objects (keep rate %.0f%%; %d no-change, %d CLIP rejects); "
            "median object size %.2f%% of the image",
            summary["objects_kept"],
            summary["objects_attempted"],
            100 * summary["keep_rate"],
            summary["rejected_no_change"],
            summary["rejected_clip"],
            summary["coverage_pct"]["median"],
        )
        if summary["keep_rate"] < 0.5:
            log.warning(
                "Fewer than half of the painted objects passed QC — inspect the "
                "samples (pws-preview) before training on this data."
            )
        log.info("Wrote dataset manifest: %s", yaml_path)
        return yaml_path
