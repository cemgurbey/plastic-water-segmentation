import csv
import json
from pathlib import Path
from types import SimpleNamespace

from plastic_water_seg.eval.run import evaluate_dir, make_yolo_eval_dataset

REPO = Path(__file__).resolve().parents[1]


class EmptyModel:
    """Stands in for a YOLO model that detects nothing."""

    def predict(self, image, **kwargs):
        return [SimpleNamespace(boxes=None, masks=None, names={})]


def test_evaluate_dir_on_dev_split(tmp_path):
    headline = evaluate_dir(
        EmptyModel(), REPO / "test_images", out_dir=tmp_path, split="dev", save_vis=False
    )
    assert headline["images"] == 15
    assert headline["agnostic_recall"] == 0.0
    for name in ("summary.csv", "metrics.csv", "metrics_agnostic.csv", "per_image.csv"):
        assert (tmp_path / name).exists()
    rows = list(csv.DictReader((tmp_path / "per_image.csv").open()))
    assert sum(int(r["gt"]) for r in rows) == 140  # dev ground-truth instances
    json.dumps(headline)


def test_make_yolo_eval_dataset(tmp_path):
    yaml_path = make_yolo_eval_dataset(REPO / "test_images", out_dir=tmp_path / "ev")
    assert "test: images/test" in yaml_path.read_text()
    assert len(list((tmp_path / "ev" / "images" / "dev").glob("*.jpg"))) == 15
    assert len(list((tmp_path / "ev" / "labels" / "test").glob("*.txt"))) == 25
