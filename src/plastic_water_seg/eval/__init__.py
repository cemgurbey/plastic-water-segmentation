"""Package init for evaluation."""

from plastic_water_seg.eval.inference import (
    InferenceResult,
    contamination_index,
    predict,
    visualize,
)

__all__ = ["InferenceResult", "contamination_index", "predict", "visualize"]
