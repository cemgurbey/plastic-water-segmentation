"""Mask utilities: contour polygons, YOLO-format export, SAM 2 refinement."""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_sam_model = None


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.005,
    min_area: float = 25.0,
) -> np.ndarray | None:
    """Largest external contour of a binary mask as a simplified polygon (pixel coords)."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    poly = approx.reshape(-1, 2)
    return poly if len(poly) >= 3 else None


def polygon_to_yolo_line(class_id: int, polygon_px: np.ndarray, img_w: int, img_h: int) -> str:
    """Encode a pixel-space polygon as one YOLO segmentation label line."""
    norm = polygon_px.astype(np.float64)
    norm[:, 0] /= img_w
    norm[:, 1] /= img_h
    coords = " ".join(f"{v:.6f}" for v in norm.ravel())
    return f"{class_id} {coords}"


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (x1, y1, x2, y2) bounding box of a binary mask, or None if empty."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def refine_masks_with_sam(
    image_rgb: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    model_name: str = "sam2.1_b.pt",
) -> list[np.ndarray] | None:
    """Refine coarse boxes into precise masks with SAM 2 (via Ultralytics).

    Returns a list of boolean masks, or None when SAM is unavailable / fails
    (callers should fall back to the coarse masks).
    """
    global _sam_model
    try:
        from ultralytics import SAM
    except ImportError:
        log.warning("ultralytics SAM not available; skipping SAM refinement")
        return None
    try:
        if _sam_model is None:
            log.info("Loading SAM 2 model (%s) — first run downloads weights", model_name)
            _sam_model = SAM(model_name)
        results = _sam_model(image_rgb, bboxes=[list(b) for b in bboxes], verbose=False)
        masks = results[0].masks
        if masks is None:
            return None
        return [(m.cpu().numpy() > 0.5) for m in masks.data]
    except Exception as exc:  # noqa: BLE001 — refinement is best-effort
        log.warning("SAM refinement failed (%s); using coarse masks", exc)
        return None
