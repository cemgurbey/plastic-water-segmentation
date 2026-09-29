"""Plastic residue detection on water bodies.

v2 pipeline: diffusion-inpainting synthetic data generation, SAM 2 mask
refinement, and YOLO instance-segmentation training on PyTorch.
"""

from plastic_water_seg.config import CATEGORIES, SynthConfig, TrainConfig

__version__ = "2.0.0"
__all__ = ["CATEGORIES", "SynthConfig", "TrainConfig", "__version__"]
