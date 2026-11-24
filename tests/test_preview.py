from pathlib import Path

from plastic_water_seg.data.preview import label_size_stats, save_label_grid

REPO = Path(__file__).resolve().parents[1]


def test_real_test_set_objects_are_small():
    stats = label_size_stats(REPO / "test_images", REPO / "test_images" / "labels")
    assert stats["plastic_bottle"]["count"] == 137
    assert stats["plastic_bottle"]["median_pct"] < 0.5


def test_label_grid_writes_image(tmp_path):
    out = save_label_grid(
        REPO / "test_images", tmp_path / "grid.jpg", REPO / "test_images" / "labels", n=6, tile=64
    )
    assert out.exists() and out.stat().st_size > 0
