import json
import random

import numpy as np

from plastic_water_seg.config import SynthConfig
from plastic_water_seg.data.painters import DebrisPainter, ProceduralPainter
from plastic_water_seg.data.synth import DatasetSynthesizer


def _cfg(**kw):
    base = dict(
        image_size=(256, 256),
        train_count=6,
        val_count=2,
        sam_refine=False,
        min_coverage=0.002,
        max_coverage=0.03,
        max_objects=6,
        seed=1,
    )
    base.update(kw)
    return SynthConfig(**base)


def _read(labels_dir):
    rows = []
    for f in sorted(labels_dir.glob("*.txt")):
        for line in f.read_text().splitlines():
            p = line.split()
            rows.append((int(p[0]), np.array(p[1:], dtype=float).reshape(-1, 2)))
    return rows


def test_procedural_run_writes_valid_small_labels(tmp_path, water_backgrounds):
    synth = DatasetSynthesizer(water_backgrounds, tmp_path / "ds", ProceduralPainter(), _cfg())
    yaml_path = synth.run()
    assert yaml_path.exists()
    rows = _read(tmp_path / "ds" / "labels" / "train")
    assert rows, "expected some labels"
    for cid, coords in rows:
        assert cid in (0, 1, 2)
        assert len(coords) >= 3
        assert coords.min() >= 0 and coords.max() <= 1
    stats = json.loads((tmp_path / "ds" / "synth_stats.json").read_text())
    assert stats["objects_kept"] == len(rows) + len(_read(tmp_path / "ds" / "labels" / "val"))
    assert stats["coverage_pct"]["median"] < 3.0  # small objects, like the real photos


def test_regions_never_overlap(water_backgrounds, tmp_path):
    synth = DatasetSynthesizer(water_backgrounds, tmp_path / "ds", ProceduralPainter(), _cfg())
    rng = random.Random(0)
    boxes = []
    for _ in range(30):
        m = synth._sample_region_mask(256, 256, boxes, rng)
        if m is None:
            continue
        ys, xs = np.nonzero(np.array(m))
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not DatasetSynthesizer._boxes_touch(a, b, 0)


class NoOpPainter(DebrisPainter):
    """Simulates diffusion returning plain water: nothing changes."""

    def paint(self, background, region_mask, category, rng):
        return background.copy()


def test_unchanged_regions_get_no_label(tmp_path, water_backgrounds):
    synth = DatasetSynthesizer(water_backgrounds, tmp_path / "ds", NoOpPainter(), _cfg())
    synth.run()
    assert _read(tmp_path / "ds" / "labels" / "train") == []
    stats = json.loads((tmp_path / "ds" / "synth_stats.json").read_text())
    assert stats["objects_kept"] == 0
    assert stats["rejected_no_change"] == stats["objects_attempted"] > 0


class RejectAll:
    def accepts(self, crop, category):
        return False


def test_verifier_rejection_reverts_the_paint(tmp_path, water_backgrounds):
    synth = DatasetSynthesizer(
        water_backgrounds, tmp_path / "ds", ProceduralPainter(), _cfg(), verifier=RejectAll()
    )
    rng = random.Random(3)
    bg_rng = random.Random(3)
    image, labels = synth._synthesize_one(0, rng)
    expected_bg = synth._prepare_background(bg_rng.choice(synth.backgrounds), bg_rng)
    assert labels == []
    assert np.array_equal(np.array(image), np.array(expected_bg))
