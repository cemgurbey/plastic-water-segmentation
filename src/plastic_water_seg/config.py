"""Central configuration: categories, synthesis and training settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    """A debris class with its diffusion prompts for inpainting.

    ``prompts`` holds several phrasings; one is drawn at random per object so
    the model sees more than one look per class. The phrasings target what the
    real test photos show: clear bottles lying on their side and half
    submerged, crumpled bags floating flat, and clusters of mixed fragments.
    """

    name: str
    prompts: tuple[str, ...]
    negative_prompt: str = (
        "cartoon, drawing, sketch, painting, render, blurry, watermark, text, logo, "
        "people, hands, animals, boat"
    )

    @property
    def prompt(self) -> str:
        """First prompt, kept for backwards compatibility."""
        return self.prompts[0]


CATEGORIES: tuple[Category, ...] = (
    Category(
        name="plastic_bottle",
        prompts=(
            "a discarded clear plastic PET bottle lying on its side, half submerged, "
            "floating on the water surface, reflections, photorealistic photo",
            "an empty transparent plastic water bottle floating horizontally on a river, "
            "photorealistic photo",
            "a crushed plastic soda bottle with a cap floating on murky water, "
            "photorealistic photo",
        ),
    ),
    Category(
        name="plastic_bag",
        prompts=(
            "a crumpled translucent plastic bag floating flat on the water surface, "
            "photorealistic photo",
            "a white plastic shopping bag partly submerged in a river, photorealistic photo",
            "a colored plastic bag drifting on dirty water, wet and wrinkled, photorealistic photo",
        ),
    ),
    Category(
        name="plastic_debris",
        prompts=(
            "a cluster of floating plastic trash, wrappers, caps and foam pieces on the "
            "water surface, photorealistic photo",
            "a small patch of mixed plastic litter floating on a polluted river, "
            "photorealistic photo",
            "plastic food packaging and styrofoam fragments floating together on water, "
            "photorealistic photo",
        ),
    ),
)

CATEGORY_NAMES: tuple[str, ...] = tuple(c.name for c in CATEGORIES)


@dataclass
class SynthConfig:
    """Settings for synthetic dataset generation."""

    image_size: tuple[int, int] = (1024, 1024)  # (width, height)
    min_objects: int = 1
    max_objects: int = 6
    # Inpaint region area as a fraction of the image area, sampled log-uniformly.
    # The real test photos have a median object size of ~0.2% of the frame
    # (104 of 137 bottles are under 0.5%), so most training objects must be
    # small too. The old 2-18% range never showed the model an object that size.
    min_coverage: float = 0.0005
    max_coverage: float = 0.03
    # Minimum gap (px) between object regions. Regions never overlap, so a later
    # object can never paint over an earlier one whose label is already kept.
    min_gap_px: int = 4
    train_count: int = 200
    val_count: int = 40
    # Diffusion inpainting settings (used by DiffusionInpainter).
    diffusion_model: str = "runwayml/stable-diffusion-inpainting"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.99
    # Each object is inpainted on a square crop around its region (resized to
    # ``inpaint_resolution``) and pasted back, so small objects get full
    # diffusion resolution and the rest of the image is never re-encoded.
    inpaint_resolution: int = 512
    inpaint_context: float = 2.5  # crop side = region size x this factor
    # Quality control: an object is kept only if the painter visibly changed its
    # region. ``change_threshold`` is the per-pixel mean abs RGB difference
    # (0-255) that counts as "changed"; ``min_changed_fraction`` is the share of
    # the region that must change. Failed objects are reverted and not labelled.
    change_threshold: float = 20.0
    min_changed_fraction: float = 0.10
    # Optional CLIP check that the changed region actually looks like the class.
    clip_verify: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_min_prob: float = 0.5
    # Refine labels with SAM 2 (prompted with the changed area's box and clipped
    # to the inpaint region). Falls back to the changed-pixel mask.
    sam_refine: bool = True
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"


@dataclass
class TrainConfig:
    """Settings for YOLO segmentation training."""

    model: str = "yolo11m-seg.pt"  # pretrained checkpoint; auto-downloaded
    epochs: int = 50
    # Most real objects are tiny (median ~0.2% of the frame); train large.
    imgsz: int = 1024
    batch: int = 8
    device: str = "0"  # GPU id, "cpu", or "0,1" for multi-GPU
    project: str = "logs"
    name: str = "yolo11m-seg"
    exist_ok: bool = True
    extra_args: dict = field(default_factory=dict)
