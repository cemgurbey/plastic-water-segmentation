import numpy as np

from plastic_water_seg.data.masks import (
    changed_mask,
    close_to_region,
    expand_box,
    mask_to_polygon,
    polygon_to_yolo_line,
)


def test_polygon_roundtrip_keeps_area():
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 50:150] = True
    poly = mask_to_polygon(mask)
    line = polygon_to_yolo_line(1, poly, 200, 100)
    parts = line.split()
    assert parts[0] == "1"
    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
    assert coords.min() >= 0 and coords.max() <= 1
    x, y = coords[:, 0] * 200, coords[:, 1] * 100
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert abs(area - mask.sum()) / mask.sum() < 0.05


def test_changed_mask_ignores_pixels_outside_box():
    before = np.zeros((50, 50, 3), dtype=np.uint8)
    after = before.copy()
    after[10:20, 10:20] = 200  # inside the box
    after[40:45, 40:45] = 200  # outside the box
    m = changed_mask(before, after, (5, 5, 25, 25), threshold=20)
    assert m[10:20, 10:20].all()
    assert not m[40:45, 40:45].any()


def test_changed_mask_ignores_small_noise():
    before = np.full((40, 40, 3), 100, dtype=np.uint8)
    after = before.copy()
    after[::7, ::7] += 10  # below threshold
    assert not changed_mask(before, after, (0, 0, 40, 40), threshold=20).any()


def test_close_to_region_merges_fragments():
    m = np.zeros((100, 100), dtype=bool)
    m[40:50, 20:30] = True
    m[40:50, 34:44] = True  # 4 px gap
    merged = close_to_region(m)
    assert merged[45, 32]  # gap filled
    assert merged.sum() >= m.sum()


def test_expand_box_clips_to_image():
    assert expand_box((0, 0, 10, 10), 5, 12, 12) == (0, 0, 12, 12)
