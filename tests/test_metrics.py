from pathlib import Path

import numpy as np

from plastic_water_seg.eval.inference import InferenceResult
from plastic_water_seg.eval.metrics import (
    evaluate_predictions,
    load_splits,
    macro_recall,
    summarize,
)

REPO = Path(__file__).resolve().parents[1]


def _result(masks, class_ids):
    masks = np.array(masks, dtype=bool)
    return InferenceResult(
        image=np.zeros((*masks.shape[1:], 3), dtype=np.uint8),
        boxes=np.zeros((len(class_ids), 4)),
        masks=masks,
        class_ids=np.array(class_ids, dtype=int),
        class_names=[str(c) for c in class_ids],
        scores=np.ones(len(class_ids)),
    )


def _mask(y1, y2, x1, x2, shape=(50, 50)):
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_perfect_prediction():
    gt = [(0, _mask(0, 10, 0, 10)), (2, _mask(20, 40, 20, 40))]
    res = _result([gt[0][1], gt[1][1]], [0, 2])
    m = evaluate_predictions(gt, res)
    assert m["plastic_bottle"].tp == 1 and m["plastic_debris"].tp == 1
    assert sum(x.fp + x.fn for x in m.values()) == 0


def test_wrong_class_scores_zero_strict_but_hits_agnostic():
    gt = [(2, _mask(20, 40, 20, 40))]  # debris
    res = _result([gt[0][1]], [1])  # predicted as bag
    strict = evaluate_predictions(gt, res)
    assert strict["plastic_debris"].fn == 1 and strict["plastic_bag"].fp == 1
    ag = evaluate_predictions(gt, res, class_agnostic=True)
    assert ag["plastic"].tp == 1 and ag["plastic"].fp == 0


def test_macro_recall_weighs_images_equally():
    many = [(0, _mask(i, i + 1, 0, 1)) for i in range(10)]  # 10 GT, none found
    one = [(0, _mask(0, 10, 0, 10))]  # 1 GT, found
    img_a = evaluate_predictions(many, _result(np.zeros((0, 50, 50)), []))
    img_b = evaluate_predictions(one, _result([one[0][1]], [0]))
    pooled = summarize([img_a, img_b])["plastic_bottle"].recall
    assert abs(pooled - 1 / 11) < 1e-9
    assert abs(macro_recall([img_a, img_b]) - 0.5) < 1e-9


def test_repo_split_file_covers_all_photos():
    splits = load_splits(REPO / "test_images" / "SPLITS.csv")
    assert len(splits) == 40
    assert set(splits.values()) == {"dev", "test"}
    assert sum(v == "dev" for v in splits.values()) == 15
