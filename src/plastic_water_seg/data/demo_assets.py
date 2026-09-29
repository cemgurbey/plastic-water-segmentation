"""Procedural water-surface backgrounds for bootstrapping the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

_PALETTES: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = [
    ((15, 60, 110), (35, 110, 170), (100, 190, 230)),  # deep ocean blue
    ((30, 65, 50), (60, 100, 70), (120, 150, 100)),  # turbid river green
    ((10, 80, 90), (25, 130, 140), (80, 200, 210)),  # coastal aqua
    ((45, 55, 70), (80, 95, 115), (140, 155, 175)),  # overcast grey-blue
]


def generate_water_backgrounds(
    out_dir: str | Path,
    count: int = 8,
    size: tuple[int, int] = (640, 640),
    seed: int = 0,
) -> list[Path]:
    """Render layered sinusoidal wave textures that read as water surfaces."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    paths: list[Path] = []

    for idx in range(count):
        c_low, c_mid, c_high = _PALETTES[idx % len(_PALETTES)]
        fx, fy = rng.uniform(4, 10, size=2)
        px, py = rng.uniform(0, 2 * np.pi, size=2)

        xs = np.linspace(0, fx * np.pi, w)
        ys = np.linspace(0, fy * np.pi, h)
        xx, yy = np.meshgrid(xs, ys)
        waves = np.sin(xx + px + 0.5 * np.cos(yy + py)) * np.cos(yy * 0.8 + py + 0.3 * np.sin(xx))
        # Second harmonic for finer ripple detail.
        waves += 0.35 * np.sin(2.3 * xx + 1.7 * yy + px)
        waves = (waves - waves.min()) / (waves.max() - waves.min() + 1e-9)

        arr = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            channel = np.where(
                waves < 0.5,
                c_low[c] + (waves / 0.5) * (c_mid[c] - c_low[c]),
                c_mid[c] + ((waves - 0.5) / 0.5) * (c_high[c] - c_mid[c]),
            )
            arr[:, :, c] = channel

        # Caustic-like noise, then a soft blur to sell the water look.
        arr += rng.normal(0, 7, (h, w, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

        path = out_dir / f"water_surface_{idx + 1:02d}.jpg"
        img.save(path, quality=92)
        paths.append(path)

    log.info("Generated %d water backgrounds in %s", len(paths), out_dir)
    return paths
