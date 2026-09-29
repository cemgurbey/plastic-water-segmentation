"""Command-line interface: pws-synthesize / pws-train / pws-infer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def synthesize() -> None:
    from plastic_water_seg.config import SynthConfig
    from plastic_water_seg.data.demo_assets import generate_water_backgrounds
    from plastic_water_seg.data.painters import DiffusionInpainter, ProceduralPainter
    from plastic_water_seg.data.synth import DatasetSynthesizer

    ap = argparse.ArgumentParser(description="Generate a synthetic plastic-in-water dataset")
    ap.add_argument("--backgrounds", default="data/backgrounds")
    ap.add_argument("--output", default="dataset")
    ap.add_argument(
        "--method",
        choices=["diffusion", "procedural"],
        default="diffusion",
        help="diffusion = SOTA inpainting (needs GPU + model download)",
    )
    ap.add_argument("--train-count", type=int, default=200)
    ap.add_argument("--val-count", type=int, default=40)
    ap.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    ap.add_argument(
        "--demo-backgrounds",
        action="store_true",
        help="generate procedural water backgrounds first",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-sam", action="store_true", help="skip SAM 2 mask refinement")
    args = ap.parse_args()

    if args.demo_backgrounds or not list(Path(args.backgrounds).glob("*")):
        generate_water_backgrounds(args.backgrounds, seed=args.seed)

    cfg = SynthConfig(
        image_size=tuple(args.image_size),
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        sam_refine=not args.no_sam,
    )
    painter = (
        DiffusionInpainter(
            model_id=cfg.diffusion_model,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
        )
        if args.method == "diffusion"
        else ProceduralPainter()
    )
    DatasetSynthesizer(args.backgrounds, args.output, painter, cfg).run()


def train() -> None:
    from plastic_water_seg.config import TrainConfig
    from plastic_water_seg.training.train import train_yolo

    ap = argparse.ArgumentParser(description="Train YOLO segmentation on the synthetic dataset")
    ap.add_argument("--data", default="dataset/dataset.yaml")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    train_yolo(args.data, cfg)


def infer() -> None:
    from plastic_water_seg.eval.inference import contamination_index, predict, visualize

    ap = argparse.ArgumentParser(description="Run inference + contamination index on an image")
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--save", default=None, help="save visualization to this path")
    args = ap.parse_args()

    result = predict(args.weights, args.image, conf=args.conf)
    print(f"Detected objects : {len(result.class_ids)}")
    for name, score in zip(result.class_names, result.scores, strict=True):
        print(f"  - {name}: {score:.1%}")
    print(f"Contamination    : {contamination_index(result):.2f}% of surface")
    visualize(result, save_path=args.save)
