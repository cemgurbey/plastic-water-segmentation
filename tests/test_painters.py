import random

import numpy as np
from PIL import Image, ImageDraw

from plastic_water_seg.config import CATEGORIES
from plastic_water_seg.data.painters import DiffusionInpainter, paste_into_region, square_crop_box


def _region(w=256, h=256, box=(100, 120, 130, 140)):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m


def test_square_crop_box_is_square_and_inside():
    box = square_crop_box(_region(), context=2.5)
    x1, y1, x2, y2 = box
    assert x2 - x1 == y2 - y1
    assert 0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256
    # near a border the crop is shifted inward, not shrunk
    edge = square_crop_box(_region(box=(0, 0, 40, 40)), context=2.5)
    assert edge[0] == 0 and edge[1] == 0 and edge[2] - edge[0] >= 100


def test_paste_only_changes_region():
    bg = Image.new("RGB", (256, 256), (20, 80, 120))
    region = _region()
    box = square_crop_box(region)
    red = Image.new("RGB", (512, 512), (250, 0, 0))
    out = np.array(paste_into_region(bg, red, region, box)).astype(int)
    diff = np.abs(out - np.array(bg).astype(int)).sum(axis=2)
    reg = np.array(region) > 127
    assert (diff[reg] > 100).mean() > 0.9  # region painted
    # far from the region nothing changes
    far = np.ones_like(reg)
    far[110:150, 90:140] = False
    assert diff[far].max() == 0


class FakeInpainter(DiffusionInpainter):
    """Skips the model: 'generates' a solid white crop."""

    def _generate(self, image, mask, prompt, negative_prompt, seed):
        assert image.size == (self.resolution, self.resolution)
        return Image.new("RGB", image.size, (255, 255, 255))


def test_diffusion_inpainter_crop_and_paste():
    painter = FakeInpainter(device="cpu", resolution=128)
    bg = Image.new("RGB", (256, 256), (20, 80, 120))
    out = np.array(painter.paint(bg, _region(), CATEGORIES[0], random.Random(0)))
    assert (out[130, 115] > 200).all()  # region centre painted
    assert (out[10, 10] == (20, 80, 120)).all()  # elsewhere untouched
