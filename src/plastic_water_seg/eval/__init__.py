"""Package init for evaluation."""

from plastic_water_seg.eval.inference import (
    InferenceResult,
    contamination_index,
    load_model,
    predict,
    visualize,
)
from plastic_water_seg.eval.metrics import (
    evaluate_predictions,
    load_splits,
    load_yolo_labels,
    macro_recall,
    metrics_table,
    overall,
    summarize,
)
from plastic_water_seg.eval.run import evaluate_dir, make_yolo_eval_dataset, map_eval

__all__ = [
    "InferenceResult",
    "contamination_index",
    "evaluate_dir",
    "evaluate_predictions",
    "load_model",
    "load_splits",
    "load_yolo_labels",
    "macro_recall",
    "make_yolo_eval_dataset",
    "map_eval",
    "metrics_table",
    "overall",
    "predict",
    "summarize",
    "visualize",
]
