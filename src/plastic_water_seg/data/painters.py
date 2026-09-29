"""Debris painters: render plastic objects into water backgrounds.

The state-of-the-art path is :class:`DiffusionInpainter`, which uses a
Stable Diffusion inpainting model to hallucinate photorealistic debris
directly into the scene. :class:`ProceduralPainter` is a lightweight,
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
    ) -> None:
        self.model_id = model_id
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
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, use_safetensors=True
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
        self._load()
        import torch

        w, h = background.size
        # Slightly feather the mask so the inpaint blends into the water.
        soft_mask = region_mask.filter(ImageFilter.GaussianBlur(radius=2))
        generator = torch.Generator(device="cpu").manual_seed(rng.randrange(2**31))
        result = self._pipe(
            prompt=category.prompt,
            negative_prompt=category.negative_prompt,
            image=background.convert("RGB").resize((w, h)),
            mask_image=soft_mask.resize((w, h)),
            height=h,
            width=w,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
        )
        return result.images[0].convert("RGB")


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
            draw.rounded_rectangle(
                [x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y], radius=8, fill=color
            )
            # highlight stripe
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
