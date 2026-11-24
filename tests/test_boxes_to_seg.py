import numpy as np
from PIL import Image

import plastic_water_seg.data.boxes_to_seg as b2s


def _dataset(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 100), (20, 80, 120)).save(tmp_path / "images" / "a.jpg")
    # class 0 box in the middle, class 3 ("not plastic") to be dropped
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n3 0.1 0.1 0.1 0.1\n")
    return tmp_path


def test_box_fallback_without_sam(tmp_path, monkeypatch):
    monkeypatch.setattr(b2s, "refine_masks_with_sam", lambda *a, **k: None)
    src = _dataset(tmp_path / "src")
    stats = b2s.convert_split(src / "images", src / "labels", tmp_path / "out", {0: 1, 3: None})
    assert stats == {"images": 1, "objects": 1, "sam": 0, "box_fallback": 1, "dropped_class": 1}
    line = (tmp_path / "out" / "labels" / "a.txt").read_text().split()
    assert line[0] == "1"  # remapped class


def test_uses_sam_mask_when_plausible(tmp_path, monkeypatch):
    def fake_sam(rgb, boxes, *a, **k):
        out = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(rgb.shape[:2], dtype=bool)
            m[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5] = True  # tighter than the box
            out.append(m)
        return out

    monkeypatch.setattr(b2s, "refine_masks_with_sam", fake_sam)
    src = _dataset(tmp_path / "src")
    stats = b2s.convert_split(src / "images", src / "labels", tmp_path / "out", {0: 0})
    assert stats["sam"] == 1 and stats["box_fallback"] == 0
