"""Central configuration: categories, synthesis and training settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    """A debris class with its diffusion prompt for inpainting."""

    name: str
    prompt: str
    negative_prompt: str = (
        "cartoon, drawing, sketch, painting, blurry, watermark, text, logo, people, animals"
    )


CATEGORIES: tuple[Category, ...] = (
    Category(
        name="plastic_bottle",
        prompt="a floating plastic bottle on a water surface, photorealistic, daylight",
    ),
    Category(
        name="plastic_bag",
        prompt="a floating translucent plastic bag on a water surface, photorealistic, daylight",
    ),
    Category(
        name="plastic_debris",
        prompt="small floating plastic debris fragments on a water surface, photorealistic, daylight",
    ),
)

CATEGORY_NAMES: tuple[str, ...] = tuple(c.name for c in CATEGORIES)


@dataclass
class SynthConfig:
    """Settings for synthetic dataset generation."""

    image_size: tuple[int, int] = (512, 512)  # (width, height)
    min_objects: int = 1
    max_objects: int = 4
    # Inpaint region area as a fraction of the image area.
    min_coverage: float = 0.02
    max_coverage: float = 0.18
    train_count: int = 200
    val_count: int = 40
    # Diffusion inpainting settings (used by DiffusionInpainter).
    diffusion_model: str = "runwayml/stable-diffusion-inpainting"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.99
    # Refine inpaint masks with SAM 2 (falls back to mask contours if unavailable).
    sam_refine: bool = True
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"


@dataclass
class TrainConfig:
    """Settings for YOLO segmentation training."""

    model: str = "yolo11m-seg.pt"  # pretrained checkpoint; auto-downloaded
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "0"  # GPU id, "cpu", or "0,1" for multi-GPU
    project: str = "logs"
    name: str = "yolo11m-seg"
    exist_ok: bool = True
    extra_args: dict = field(default_factory=dict)
