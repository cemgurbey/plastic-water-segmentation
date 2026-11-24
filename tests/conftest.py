"""Shared fixtures. Tests run without torch/ultralytics/diffusers installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def water_backgrounds(tmp_path: Path) -> Path:
    """A few noisy 'real-looking' backgrounds (not demo-named, so no warning)."""
    rng = np.random.default_rng(0)
    bg = tmp_path / "backgrounds"
    bg.mkdir()
    for i in range(3):
        base = np.array([30 + 20 * i, 90, 120], dtype=np.float32)
        arr = base + rng.normal(0, 4, (300, 300, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(bg / f"river_{i}.png")
    return bg
