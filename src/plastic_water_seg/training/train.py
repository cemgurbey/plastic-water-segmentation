"""YOLO instance-segmentation training (Ultralytics, PyTorch)."""

from __future__ import annotations

import logging
from pathlib import Path

from plastic_water_seg.config import TrainConfig

log = logging.getLogger(__name__)


def train_yolo(data_yaml: str | Path, config: TrainConfig | None = None):
    """Fine-tune a pretrained YOLO-seg checkpoint on the synthetic dataset.

    Returns the trained Ultralytics YOLO model. Checkpoints and metrics are
    written under ``<project>/<name>/``.
    """
    from ultralytics import YOLO

    cfg = config or TrainConfig()
    data_yaml = str(Path(data_yaml).resolve())
    log.info("Training %s on %s for %d epochs", cfg.model, data_yaml, cfg.epochs)

    model = YOLO(cfg.model)  # auto-downloads the pretrained checkpoint
    model.train(
        data=data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        **cfg.extra_args,
    )
    log.info("Training done. Best weights: %s/%s/weights/best.pt", cfg.project, cfg.name)
    return model
