"""Optional CLIP check that an inpainted object looks like its class.

Diffusion inpainting sometimes paints plain water, foam or an unrelated object
into a region. A pixel-change check catches "nothing happened"; this catches
"something happened, but it is not a plastic bottle".
"""

from __future__ import annotations

import logging

from PIL import Image

from plastic_water_seg.config import Category

log = logging.getLogger(__name__)

_BACKGROUND_PROMPTS = (
    "a photo of a plain water surface",
    "a photo of ripples and reflections on water",
    "a photo of algae and plants in water",
)


class ClipVerifier:
    """Zero-shot CLIP: is this crop more like ``category`` than plain water?"""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        min_prob: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.min_prob = min_prob
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_id)

    def probability(self, crop: Image.Image, category: Category) -> float:
        """Softmax probability of the class prompt against background prompts."""
        self._load()
        import torch

        texts = [f"a photo of a {category.name.replace('_', ' ')} floating on water"]
        texts += list(_BACKGROUND_PROMPTS)
        inputs = self._processor(text=texts, images=crop, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]
        return float(logits.softmax(dim=0)[0])

    def accepts(self, crop: Image.Image, category: Category) -> bool:
        prob = self.probability(crop, category)
        log.debug("CLIP %s p=%.2f", category.name, prob)
        return prob >= self.min_prob
