"""Command-line interface.

pws-synthesize / pws-preview / pws-boxes-to-seg / pws-train / pws-infer / pws-eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def synthesize() -> None:
    from plastic_water_seg.config import SynthConfig
    from plastic_water_seg.data.demo_assets import generate_water_backgrounds
    from plastic_water_seg.data.painters import DiffusionInpainter, ProceduralPainter
    from plastic_water_seg.data.synth import DatasetSynthesizer

    defaults = SynthConfig()
    ap = argparse.ArgumentParser(description="Generate a synthetic plastic-in-water dataset")
    ap.add_argument("--backgrounds", default="data/backgrounds")
    ap.add_argument("--output", default="dataset")
    ap.add_argument(
        "--method",
        choices=["diffusion", "procedural"],
        default="diffusion",
        help="diffusion = SOTA inpainting (needs GPU + model download)",
    )
    ap.add_argument("--train-count", type=int, default=defaults.train_count)
    ap.add_argument("--val-count", type=int, default=defaults.val_count)
    ap.add_argument(
        "--image-size", type=int, nargs=2, default=list(defaults.image_size), metavar=("W", "H")
    )
    ap.add_argument("--min-coverage", type=float, default=defaults.min_coverage)
    ap.add_argument("--max-coverage", type=float, default=defaults.max_coverage)
    ap.add_argument("--max-objects", type=int, default=defaults.max_objects)
    ap.add_argument(
        "--demo-backgrounds",
        action="store_true",
        help="generate procedural water backgrounds first (smoke tests only)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-sam", action="store_true", help="skip SAM 2 mask refinement")
    ap.add_argument("--clip-verify", action="store_true", help="drop objects CLIP rejects")
    args = ap.parse_args()

    bg = Path(args.backgrounds)
    if args.demo_backgrounds or not bg.exists() or not any(bg.iterdir()):
        generate_water_backgrounds(args.backgrounds, seed=args.seed)

    cfg = SynthConfig(
        image_size=tuple(args.image_size),
        train_count=args.train_count,
        val_count=args.val_count,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        max_objects=args.max_objects,
        seed=args.seed,
        sam_refine=not args.no_sam,
        clip_verify=args.clip_verify,
    )
    painter = (
        DiffusionInpainter(
            model_id=cfg.diffusion_model,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
            resolution=cfg.inpaint_resolution,
            context=cfg.inpaint_context,
        )
        if args.method == "diffusion"
        else ProceduralPainter()
    )
    DatasetSynthesizer(args.backgrounds, args.output, painter, cfg).run()


def preview() -> None:
    from plastic_water_seg.data.preview import label_size_stats, save_label_grid

    ap = argparse.ArgumentParser(
        description="Draw labels over sample images and report object sizes"
    )
    ap.add_argument("--images", default="dataset/images/train")
    ap.add_argument("--labels", default=None, help="defaults to the YOLO images->labels path")
    ap.add_argument("--out", default="dataset/preview.jpg")
    ap.add_argument("-n", type=int, default=24)
    args = ap.parse_args()

    path = save_label_grid(args.images, args.out, args.labels, n=args.n)
    print(f"Label grid: {path}")
    print("Object size (% of image area) — real test set median is ~0.2%:")
    print(json.dumps(label_size_stats(args.images, args.labels), indent=2))


def boxes_to_seg() -> None:
    from plastic_water_seg.data.boxes_to_seg import convert_split

    ap = argparse.ArgumentParser(
        description="Convert a YOLO detection dataset (boxes) to YOLO-seg with SAM 2"
    )
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--class-map",
        required=True,
        help='JSON mapping source id -> project id or null, e.g. \'{"0": 1, "1": 0, "2": 2, "3": null}\'',
    )
    args = ap.parse_args()
    cmap = {int(k): v for k, v in json.loads(args.class_map).items()}
    print(convert_split(args.images, args.labels, args.out, cmap))


def train() -> None:
    from plastic_water_seg.config import TrainConfig
    from plastic_water_seg.training.train import train_yolo

    defaults = TrainConfig()
    ap = argparse.ArgumentParser(description="Train YOLO segmentation")
    ap.add_argument("--data", default="dataset/dataset.yaml")
    ap.add_argument("--model", default=defaults.model)
    ap.add_argument("--epochs", type=int, default=defaults.epochs)
    ap.add_argument("--imgsz", type=int, default=defaults.imgsz)
    ap.add_argument("--batch", type=int, default=defaults.batch)
    ap.add_argument("--device", default=defaults.device)
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
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    ap.add_argument("--save", default=None, help="save visualization to this path")
    args = ap.parse_args()

    result = predict(args.weights, args.image, conf=args.conf, imgsz=args.imgsz, device=args.device)
    print(f"Detected objects : {len(result.class_ids)}")
    for name, score in zip(result.class_names, result.scores, strict=True):
        print(f"  - {name}: {score:.1%}")
    print(f"Contamination    : {contamination_index(result):.2f}% of surface")
    visualize(result, save_path=args.save)


def evaluate() -> None:
    from plastic_water_seg.eval.run import evaluate_dir, make_yolo_eval_dataset, map_eval

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the real test photos")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="test_images")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--no-map", action="store_true", help="skip threshold-free mAP")
    args = ap.parse_args()

    split = None if args.split == "all" else args.split
    headline = evaluate_dir(
        args.weights,
        args.images,
        out_dir=args.out,
        split=split,
        conf=args.conf,
        imgsz=args.imgsz,
        save_vis=not args.no_vis,
    )
    if not args.no_map and split is not None:
        yaml_path = make_yolo_eval_dataset(args.images, out_dir=Path(args.out) / "eval_data")
        headline.update(
            map_eval(args.weights, yaml_path, "val" if split == "dev" else "test", args.imgsz)
        )
    (Path(args.out) / "headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))
