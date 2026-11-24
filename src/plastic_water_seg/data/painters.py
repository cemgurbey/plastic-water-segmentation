"""Debris painters: render plastic objects into water backgrounds.

The state-of-the-art path is :class:`DiffusionInpainter`, which uses a
Stable Diffusion inpainting model to hallucinate photorealistic debris
directly into the scene. Each object is generated on a square crop around
its region and pasted back, so nothing outside the region changes. :class:`ProceduralPainter` is a lightweight,
dependency-free fallback that draws stylized debris with PIL.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from plastic_water_seg.config import Category

log = logging.getLogger(__name__)


class DebrisPainter(ABC):
    """Paints one debris instance into a background image within ``region_mask``."""

    @abstractmethod
    def paint(
        self,
        background: Image.Image,
        region_mask: Image.Image,
        category: Category,
        rng: random.Random,
    ) -> Image.Image:
        """Return a new RGB image with the debris painted inside the white region of ``region_mask``."""


class DiffusionInpainter(DebrisPainter):
    """Inpaints photorealistic debris with Stable Diffusion (SOTA synthetic data).

    The model is loaded lazily on first use so importing this module never
    requires torch/diffusers to be installed.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "auto",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.99,
        resolution: int = 512,
        context: float = 2.5,
    ) -> None:
        self.model_id = model_id
        self.resolution = resolution
        self.context = context
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Loading inpainting model %s on %s", self.model_id, self.device)
        # NOTE: runwayml/stable-diffusion-inpainting ships only fp16
        # safetensors weights (no plain .safetensors files), so loading
        # with use_safetensors=True and no variant raises
        # "Could not find the necessary `safetensors` weights". variant="fp16"
        # matches the float16 dtype used on CUDA; on CPU the weights are
        # cast to float32 on load.
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )
        if self.device == "cuda":
            # Keeps VRAM usage low enough for Colab T4 GPUs.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def paint(
        self,
        background: Image.Image,
        region_mask: Image.Image,
        category: Category,
        rng: random.Random,
    ) -> Image.Image:
        """Inpaint one object on a crop around ``region_mask`` and paste it back.

        Working on a crop (instead of the whole image) matters twice over:
        a small region gets the model's full resolution rather than a handful
        of latent pixels, and the rest of the image is never pushed through the
        VAE, so earlier objects are not blurred by later ones.
        """
        box = square_crop_box(region_mask, context=self.context)
        if box is None:
            return background
        res = self.resolution
        crop_img = background.convert("RGB").crop(box).resize((res, res), Image.LANCZOS)
        crop_mask = region_mask.convert("L").crop(box).resize((res, res), Image.NEAREST)
        # Slightly feather the mask so the inpaint blends into the water.
        soft_mask = crop_mask.filter(ImageFilter.GaussianBlur(radius=2))
        prompt = rng.choice(category.prompts)
        generated = self._generate(
            crop_img, soft_mask, prompt, category.negative_prompt, rng.randrange(2**31)
        )
        return paste_into_region(background, generated, region_mask, box)

    def _generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> Image.Image:
        """Run the diffusion pipeline on one square crop (``resolution`` px)."""
        self._load()
        import torch

        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=self.resolution,
            width=self.resolution,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
        )
        return result.images[0].convert("RGB")


def square_crop_box(
    region_mask: Image.Image, context: float = 2.5, min_side: int = 96
) -> tuple[int, int, int, int] | None:
    """Square crop around the mask's bounding box, ``context`` times its size.

    The crop is clipped to the image; near a border it is shifted inward
    rather than shrunk where possible.
    """
    mask_np = np.asarray(region_mask.convert("L")) > 127
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return None
    h, w = mask_np.shape
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    side = int(max(x2 - x1, y2 - y1) * context)
    side = min(max(side, min_side), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    return left, top, left + side, top + side


def paste_into_region(
    background: Image.Image,
    generated_crop: Image.Image,
    region_mask: Image.Image,
    box: tuple[int, int, int, int],
    feather: float = 1.5,
) -> Image.Image:
    """Paste a generated crop back, changing only pixels inside the region.

    A light feather softens the seam; pixels more than a few px outside the
    region are guaranteed to stay identical to ``background``.
    """
    side_w, side_h = box[2] - box[0], box[3] - box[1]
    patch = generated_crop.convert("RGB").resize((side_w, side_h), Image.LANCZOS)
    full_patch = background.convert("RGB").copy()
    full_patch.paste(patch, box[:2])
    alpha = region_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.composite(full_patch, background.convert("RGB"), alpha)


class ProceduralPainter(DebrisPainter):
    """Dependency-free fallback: draws stylized debris directly with PIL.

    Useful for smoke-testing the pipeline on CPU-only machines where
    downloading a diffusion model is impractical.
    """

    _PALETTES: dict[str, list[tuple[int, int, int, int]]] = {
        "plastic_bottle": [(220, 235, 245, 200), (100, 180, 240, 210), (240, 250, 255, 190)],
        "plastic_bag": [(245, 245, 245, 175), (250, 230, 90, 180), (230, 240, 250, 170)],
        "plastic_debris": [(230, 45, 45, 230), (240, 240, 240, 255), (60, 60, 60, 235)],
    }

    def paint(
        self,
        background: Image.Image,
        region_mask: Image.Image,
        category: Category,
        rng: random.Random,
    ) -> Image.Image:
        mask_np = np.array(region_mask.convert("L"))
        ys, xs = np.nonzero(mask_np > 127)
        if len(xs) == 0:
            return background
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1

        color = rng.choice(self._PALETTES.get(category.name, self._PALETTES["plastic_debris"]))
        layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        pad_x, pad_y = int(w * 0.12), int(h * 0.12)

        if category.name == "plastic_bottle":
            radius = max(1, min(8, min(w, h) // 4))
            draw.rounded_rectangle(
                [x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y], radius=radius, fill=color
            )
            if min(w, h) > 24:  # highlight stripe, only where it fits
                draw.line(
                    [x1 + pad_x + 6, y1 + pad_y + 4, x1 + pad_x + 6, y2 - pad_y - 4],
                    fill=(255, 255, 255, 140),
                    width=3,
                )
        elif category.name == "plastic_bag":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pts = [
                (cx + rng.uniform(-0.5, 0.5) * w, cy + rng.uniform(-0.5, 0.5) * h) for _ in range(9)
            ]
            draw.polygon(pts, fill=color)
        else:  # plastic_debris — jagged fragment
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pts = []
            for i in range(7):
                ang = 2 * np.pi * i / 7
                r = rng.uniform(0.3, 0.5)
                pts.append((cx + r * w * np.cos(ang), cy + r * h * np.sin(ang)))
            draw.polygon(pts, fill=color)

        # Soften edges and add a faint water reflection underneath.
        layer = layer.filter(ImageFilter.GaussianBlur(radius=1.2))
        out = background.convert("RGBA")
        out.alpha_composite(layer)
        return out.convert("RGB")
