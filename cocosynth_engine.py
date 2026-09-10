#!/usr/bin/env python3
"""
cocosynth_engine.py

Synthetic COCO Dataset Generation Engine for Instance Segmentation.
Inspired by akTwelve/cocosynth (Adam Kelly / Immersive Limit).

Tailored for:
  - Detecting plastic residue & floating debris on water bodies
  - RGBA foreground compositing with realistic aquatic augmentations
  - Automatic polygon contour extraction, simplification, and COCO JSON formatting
  - Built-in procedural demo asset generator for immediate out-of-the-box testing
"""

import os
import json
import math
import random
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import cv2
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


# =====================================================================
# 1. COCO JSON Builder
# =====================================================================

class CocoJsonBuilder:
    """Helper class to build and serialize Microsoft COCO format annotations."""

    def __init__(self, description: str = "Synthetic Plastic Residue in Water Dataset"):
        now = datetime.datetime.now()
        self.coco_data = {
            "info": {
                "description": description,
                "url": "https://github.com/akTwelve/cocosynth",
                "version": "1.0",
                "year": now.year,
                "contributor": "CocoSynth Plastic Residue Project",
                "date_created": now.strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.category_name_to_id = {}

    def add_category(self, name: str, supercategory: str = "plastic_residue") -> int:
        """Registers a category if not already present, returns integer category_id."""
        if name in self.category_name_to_id:
            return self.category_name_to_id[name]
        
        category_id = len(self.category_name_to_id) + 1
        self.category_name_to_id[name] = category_id
        self.coco_data["categories"].append({
            "id": category_id,
            "name": name,
            "supercategory": supercategory
        })
        return category_id

    def add_image(self, file_name: str, width: int, height: int) -> int:
        """Adds image metadata and returns a unique image_id."""
        image_id = self.image_id_counter
        self.coco_data["images"].append({
            "id": image_id,
            "license": 1,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.image_id_counter += 1
        return image_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        segmentation: List[List[float]],
        bbox: List[float],
        area: float,
        iscrowd: int = 0
    ) -> int:
        """Adds instance annotation for an object in an image."""
        annotation_id = self.annotation_id_counter
        self.coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": float(area),
            "bbox": [float(v) for v in bbox],
            "iscrowd": iscrowd
        })
        self.annotation_id_counter += 1
        return annotation_id

    def save(self, filepath: str):
        """Writes the COCO JSON to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.coco_data, f, indent=2)


# =====================================================================
# 2. Polygon & Mask Utilities
# =====================================================================

def mask_to_polygons(mask_np: np.ndarray, min_area: float = 16.0, tolerance: float = 1.0) -> Tuple[List[List[float]], float, List[float]]:
    """
    Converts a binary instance mask (numpy uint8 array with 0/255 or bool)
    into COCO polygon segmentation lists, area, and bounding box [x, y, w, h].
    """
    if mask_np.dtype != np.uint8:
        mask_np = (mask_np.astype(np.uint8)) * 255

    # Find external contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segmentation = []
    total_area = 0.0

    # Calculate global bounding box
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [], 0.0, [0.0, 0.0, 0.0, 0.0]

    x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
    y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
    bbox = [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0]

    for contour in contours:
        # Contours must have at least 3 points to form a polygon
        if contour.shape[0] < 3:
            continue
        
        pts = contour.reshape(-1, 2)
        poly = Polygon(pts)
        if not poly.is_valid or poly.area < min_area:
            continue

        # Simplify contour to avoid bloated JSON while preserving boundary
        if tolerance > 0:
            poly = poly.simplify(tolerance, preserve_topology=True)

        if poly.is_empty:
            continue

        # Extract coordinates
        if isinstance(poly, MultiPolygon):
            polys = list(poly.geoms)
        else:
            polys = [poly]

        for p in polys:
            coords = np.array(p.exterior.coords)
            if len(coords) < 3:
                continue
            # Flatten to [x1, y1, x2, y2, ...]
            poly_flat = coords.flatten().tolist()
            segmentation.append(poly_flat)
            total_area += p.area

    if total_area == 0.0:
        total_area = float(np.sum(mask_np > 0))

    return segmentation, total_area, bbox


# =====================================================================
# 3. Plastic CocoSynthesizer Engine
# =====================================================================

class PlasticCocoSynthesizer:
    """
    Generates synthetic composite images and COCO JSON annotations.
    Composites transparent foregrounds (plastic residue items) onto background
    water body images with random transformations and aquatic blending.
    """

    def __init__(
        self,
        foregrounds_dir: str,
        backgrounds_dir: str,
        output_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        min_objects_per_image: int = 1,
        max_objects_per_image: int = 5,
        scale_range: Tuple[float, float] = (0.15, 0.45),
        allow_occlusion: bool = True,
        max_overlap_iou: float = 0.35,
        water_blend_transparency: bool = True
    ):
        self.foregrounds_dir = Path(foregrounds_dir)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.min_objects = min_objects_per_image
        self.max_objects = max_objects_per_image
        self.scale_range = scale_range
        self.allow_occlusion = allow_occlusion
        self.max_overlap_iou = max_overlap_iou
        self.water_blend_transparency = water_blend_transparency

        self._load_sources()

    def _load_sources(self):
        """Scans directories for foreground cutouts and background water images."""
        self.background_paths = [
            p for p in self.backgrounds_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

        # Categorized foregrounds: checks for subdirectories or root files
        self.foregrounds_by_cat = {}
        subdirs = [d for d in self.foregrounds_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            for d in subdirs:
                cat_name = d.name
                imgs = [p for p in d.glob("*") if p.suffix.lower() in [".png", ".webp"]]
                if imgs:
                    self.foregrounds_by_cat[cat_name] = imgs
        else:
            # Flat directory, treat root as 'plastic_residue'
            imgs = [p for p in self.foregrounds_dir.glob("*") if p.suffix.lower() in [".png", ".webp"]]
            if imgs:
                self.foregrounds_by_cat["plastic_residue"] = imgs

        print(f"Loaded {len(self.background_paths)} background images.")
        for cat, imgs in self.foregrounds_by_cat.items():
            print(f"  - Category '{cat}': {len(imgs)} foreground cutouts.")

    def _transform_foreground(self, fg_img: Image.Image) -> Image.Image:
        """Applies realistic aquatic transformations to a foreground plastic item."""
        # 1. Random rotation (0 to 360 degrees)
        angle = random.uniform(0, 360)
        fg_img = fg_img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # 2. Random flip
        if random.random() < 0.5:
            fg_img = fg_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            fg_img = fg_img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random scale relative to target canvas
        target_w, target_h = self.image_size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        base_dim = max(target_w, target_h)
        
        orig_w, orig_h = fg_img.size
        aspect = orig_w / max(orig_h, 1)
        if orig_w > orig_h:
            new_w = int(base_dim * scale)
            new_h = max(int(new_w / aspect), 16)
        else:
            new_h = int(base_dim * scale)
            new_w = max(int(new_h * aspect), 16)
            
        fg_img = fg_img.resize((new_w, new_h), Image.LANCZOS)

        # 4. Color / Lighting variations (sunlight glint, water reflection, cloud shadow)
        if random.random() < 0.7:
            # Brightness jitter
            enhancer = ImageEnhance.Brightness(fg_img)
            fg_img = enhancer.enhance(random.uniform(0.75, 1.25))

        if random.random() < 0.6:
            # Contrast jitter
            enhancer = ImageEnhance.Contrast(fg_img)
            fg_img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.5:
            # Color/saturation jitter
            enhancer = ImageEnhance.Color(fg_img)
            fg_img = enhancer.enhance(random.uniform(0.7, 1.3))

        # 5. Aquatic edge softening and slight wet transparency
        r, g, b, a = fg_img.split()
        if self.water_blend_transparency:
            # Slightly modulate alpha to simulate semi-submerged plastic/translucent film
            submerge_factor = random.uniform(0.85, 1.0)
            a_np = np.array(a, dtype=np.float32) * submerge_factor
            a = Image.fromarray(a_np.astype(np.uint8))

        # Soften alpha contour edge slightly to avoid harsh cookie-cutter edges
        a = a.filter(ImageFilter.GaussianBlur(radius=0.7))
        fg_img.putalpha(a)

        return fg_img

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculates IoU between two boxes [x, y, w, h]."""
        x1_min, y1_min, w1, h1 = box1
        x1_max, y1_max = x1_min + w1, y1_min + h1

        x2_min, y2_min, w2, h2 = box2
        x2_max, y2_max = x2_min + w2, y2_min + h2

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_w = max(0.0, xi_max - xi_min)
        inter_h = max(0.0, yi_max - yi_min)
        inter_area = inter_w * inter_h

        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def generate_single_image(
        self,
        image_idx: int,
        coco_builder: CocoJsonBuilder,
        subfolder_name: str = "images"
    ) -> bool:
        """Generates a single synthetic composite image and adds annotations to coco_builder."""
        if not self.background_paths:
            raise RuntimeError("No background images available in backgrounds_dir!")
        if not self.foregrounds_by_cat:
            raise RuntimeError("No foreground cutouts available in foregrounds_dir!")

        target_w, target_h = self.image_size

        # 1. Select and prepare background image
        bg_path = random.choice(self.background_paths)
        with Image.open(bg_path).convert("RGB") as bg:
            # Crop / Resize background to image_size
            bg_w, bg_h = bg.size
            if bg_w >= target_w and bg_h >= target_h:
                # Random crop
                crop_x = random.randint(0, bg_w - target_w)
                crop_y = random.randint(0, bg_h - target_h)
                canvas = bg.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            else:
                canvas = bg.resize((target_w, target_h), Image.LANCZOS)

        # 2. Determine number of objects to paste
        num_objects = random.randint(self.min_objects, self.max_objects)
        
        placed_boxes = []
        placed_annotations = []
        
        # Accumulator for overall instance mask canvas
        composite_canvas = canvas.copy()

        categories = list(self.foregrounds_by_cat.keys())

        for _ in range(num_objects):
            cat_name = random.choice(categories)
            fg_path = random.choice(self.foregrounds_by_cat[cat_name])
            
            with Image.open(fg_path).convert("RGBA") as fg:
                transformed_fg = self._transform_foreground(fg)

            fg_w, fg_h = transformed_fg.size
            if fg_w >= target_w or fg_h >= target_h:
                continue

            # Attempt random placement
            placed = False
            for attempt in range(15):
                pos_x = random.randint(0, target_w - fg_w)
                pos_y = random.randint(0, target_h - fg_h)
                cand_box = [pos_x, pos_y, fg_w, fg_h]

                # Overlap check
                too_much_overlap = False
                for prev_box in placed_boxes:
                    iou = self._calculate_iou(cand_box, prev_box)
                    if iou > self.max_overlap_iou:
                        too_much_overlap = True
                        break

                if not too_much_overlap or self.allow_occlusion:
                    # Place object
                    mask_layer = Image.new("L", (target_w, target_h), 0)
                    fg_alpha = transformed_fg.split()[3]
                    mask_layer.paste(fg_alpha, (pos_x, pos_y))

                    # Convert mask to polygon annotations
                    mask_np = np.array(mask_layer)
                    # Threshold alpha
                    binary_mask = (mask_np > 30).astype(np.uint8) * 255
                    
                    segmentation, area, bbox = mask_to_polygons(binary_mask)
                    if segmentation and area > 25.0:
                        # Composite onto background
                        composite_canvas.paste(transformed_fg, (pos_x, pos_y), transformed_fg)
                        placed_boxes.append(cand_box)
                        
                        cat_id = coco_builder.add_category(cat_name)
                        placed_annotations.append({
                            "category_id": cat_id,
                            "segmentation": segmentation,
                            "bbox": bbox,
                            "area": area
                        })
                        placed = True
                        break

        if not placed_annotations:
            # Fallback retry if no objects could be placed
            return False

        # Save composite image
        file_name = f"plastic_water_{image_idx:06d}.jpg"
        save_img_dir = self.output_dir / subfolder_name
        save_img_dir.mkdir(parents=True, exist_ok=True)
        save_img_path = save_img_dir / file_name

        composite_canvas.save(save_img_path, format="JPEG", quality=95)

        # Register image and annotations in COCO builder
        coco_image_id = coco_builder.add_image(file_name, target_w, target_h)
        for ann in placed_annotations:
            coco_builder.add_annotation(
                image_id=coco_image_id,
                category_id=ann["category_id"],
                segmentation=ann["segmentation"],
                bbox=ann["bbox"],
                area=ann["area"]
            )

        return True

    def generate_dataset(
        self,
        num_train: int = 100,
        num_val: int = 20,
        ann_file_name: str = "coco_instances.json"
    ):
        """Generates both train and val splits with their respective COCO JSONs."""
        print(f"\n=======================================================")
        print(f"Generating Synthetic Plastic Water Dataset")
        print(f"  - Train Images: {num_train}")
        print(f"  - Val Images:   {num_val}")
        print(f"  - Resolution:   {self.image_size[0]}x{self.image_size[1]}")
        print(f"=======================================================")

        splits = [
            ("train", num_train),
            ("val", num_val)
        ]

        for split_name, count in splits:
            if count <= 0:
                continue

            print(f"\n[Processing Split: '{split_name}'] ({count} images)...")
            split_dir = self.output_dir / split_name
            images_dir = split_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            coco_builder = CocoJsonBuilder(description=f"Synthetic Plastic in Water ({split_name})")

            generated = 0
            idx = 1
            pbar = tqdm(total=count, desc=f"Synthesizing {split_name}")
            
            while generated < count:
                success = self.generate_single_image(
                    image_idx=idx,
                    coco_builder=coco_builder,
                    subfolder_name=f"{split_name}/images"
                )
                if success:
                    generated += 1
                    pbar.update(1)
                idx += 1
            pbar.close()

            # Save COCO JSON
            coco_json_path = split_dir / ann_file_name
            coco_builder.save(str(coco_json_path))
            print(f"  -> Saved annotations to: {coco_json_path}")
            print(f"  -> Total annotations: {len(coco_builder.coco_data['annotations'])}")

        print("\nDataset generation completed successfully!")


# =====================================================================
# 4. Procedural Demo Assets Generator (Out-of-the-Box Testing)
# =====================================================================

def generate_procedural_demo_assets(base_dir: str):
    """
    Creates procedural demo water textures and realistic plastic shapes
    with alpha transparency so the pipeline can run immediately without
    needing external uploads.
    """
    base_path = Path(base_dir)
    fg_dir = base_path / "data" / "foregrounds"
    bg_dir = base_path / "data" / "backgrounds"

    fg_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Procedural Water Backgrounds (512x512 with wavelets & water ripples)
    print("Generating procedural water backgrounds...")
    water_palettes = [
        # Deep blue ocean
        ((15, 60, 110), (35, 110, 170), (100, 190, 230)),
        # River / Turbid green-brown
        ((30, 65, 50), (60, 100, 70), (120, 150, 100)),
        # Coastal aqua-teal
        ((10, 80, 90), (25, 130, 140), (80, 200, 210))
    ]

    for idx, (c_low, c_mid, c_high) in enumerate(water_palettes):
        w, h = 600, 600
        # Create sinusoidal wave gradient
        x = np.linspace(0, 8 * np.pi, w)
        y = np.linspace(0, 8 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        waves = np.sin(xx + 0.5 * np.cos(yy)) * np.cos(yy * 0.8 + 0.3 * np.sin(xx))
        waves = (waves - waves.min()) / (waves.max() - waves.min())

        bg_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            # Interpolate low to mid to high
            ch = np.where(
                waves < 0.5,
                c_low[c] + (waves / 0.5) * (c_mid[c] - c_low[c]),
                c_mid[c] + ((waves - 0.5) / 0.5) * (c_high[c] - c_mid[c])
            )
            bg_arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # Add subtle caustic noise
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        bg_arr = np.clip(bg_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        bg_img = Image.fromarray(bg_arr)
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=1.0))
        bg_img.save(bg_dir / f"water_surface_{idx+1}.jpg")

    # 2. Procedural Plastic Residue Cutouts (RGBA)
    print("Generating procedural transparent plastic residue foregrounds...")
    categories = {
        "plastic_bottle": [
            ("clear_bottle", (220, 235, 245, 190), "bottle"),
            ("blue_bottle", (100, 180, 240, 200), "bottle")
        ],
        "plastic_bag": [
            ("white_shopping_bag", (245, 245, 245, 175), "bag"),
            ("yellow_grocery_bag", (250, 230, 90, 180), "bag")
        ],
        "plastic_debris": [
            ("red_plastic_cup", (230, 45, 45, 230), "cup"),
            ("styrofoam_fragment", (240, 240, 240, 255), "fragment")
        ]
    }

    for cat_name, items in categories.items():
        cat_path = fg_dir / cat_name
        cat_path.mkdir(parents=True, exist_ok=True)

        for name, color, shape_type in items:
            size = (200, 200)
            img = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            if shape_type == "bottle":
                # Bottle body, neck, cap
                draw.rounded_rectangle([65, 60, 135, 180], radius=12, fill=color, outline=(255, 255, 255, 220), width=2)
                draw.rectangle([82, 35, 118, 60], fill=color, outline=(255, 255, 255, 200), width=1)
                draw.rectangle([80, 22, 120, 35], fill=(30, 120, 220, 240), outline=(255, 255, 255, 240), width=1) # cap
                # Highlight reflection stripe
                draw.line([75, 70, 75, 170], fill=(255, 255, 255, 150), width=3)

            elif shape_type == "bag":
                # Wrinkly plastic shopping bag polygon
                poly_pts = [
                    (50, 150), (40, 100), (60, 70), (75, 40), (88, 70),
                    (112, 70), (125, 40), (140, 70), (160, 100), (150, 155),
                    (120, 165), (80, 160)
                ]
                draw.polygon(poly_pts, fill=color, outline=(255, 255, 255, 180))
                # Crease lines
                draw.line([60, 70, 100, 140], fill=(255, 255, 255, 90), width=2)
                draw.line([140, 70, 95, 150], fill=(255, 255, 255, 90), width=2)

            elif shape_type == "cup":
                # Tapered cup polygon
                poly_pts = [(55, 50), (145, 50), (130, 165), (70, 165)]
                draw.polygon(poly_pts, fill=color, outline=(255, 255, 255, 230))
                draw.ellipse([50, 42, 150, 58], fill=(255, 255, 255, 210)) # rim

            elif shape_type == "fragment":
                # Jagged jagged fragment
                poly_pts = [(45, 70), (110, 40), (165, 85), (140, 150), (80, 170), (35, 120)]
                draw.polygon(poly_pts, fill=color, outline=(200, 200, 200, 255))

            # Apply slight blur to blend naturally
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            img.save(cat_path / f"{name}.png")

    print(f"Procedural demo assets successfully generated under {base_dir}/data!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CocoSynth Plastic Residue Data Generator")
    parser.add_argument("--demo", action="store_true", help="Generate procedural demo assets")
    parser.add_argument("--base-dir", type=str, default=".", help="Base project directory")
    parser.add_argument("--train-count", type=int, default=20, help="Number of training images")
    parser.add_argument("--val-count", type=int, default=5, help="Number of validation images")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if args.demo:
        generate_procedural_demo_assets(str(base_dir))

    synthesizer = PlasticCocoSynthesizer(
        foregrounds_dir=str(base_dir / "data" / "foregrounds"),
        backgrounds_dir=str(base_dir / "data" / "backgrounds"),
        output_dir=str(base_dir / "dataset"),
        image_size=(512, 512)
    )
    synthesizer.generate_dataset(num_train=args.train_count, num_val=args.val_count)
