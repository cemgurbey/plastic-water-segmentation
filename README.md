# Plastic Residue Detection on Water Bodies

Instance segmentation for detecting plastic pollution — bottles, bags, and debris —
floating on rivers, lakes, canals, and marine surfaces. The pipeline generates its own
photorealistic training data with diffusion models, refines masks with SAM 2, and trains
a YOLO segmentation model on PyTorch.

## How it started

v1 of this project was a cut-and-paste synthetic data pipeline in the
spirit of [akTwelve/cocosynth](https://github.com/akTwelve/cocosynth): transparent PNG
cutouts of plastic items were composited onto water photos with PIL/OpenCV, contours
were extracted into COCO-format JSON, and a Mask R-CNN (TensorFlow 2.x port) was trained
on top. It worked as a proof of concept, but the approach had real limits:

- Pasted cutouts carry telltale edges, flat lighting, and no water interaction —
  models learn the compositing artifacts, not the debris.
- Mask R-CNN on the TF 2.x fork is slow to train, heavy to deploy, and pinned to an
  aging stack (TF < 2.16, Keras < 3, NumPy < 2, imgaug).
- The whole design was derivative of the cocosynth workflow rather than its own thing.

## What changed in v2

The pipeline was rebuilt from scratch:

- **Diffusion-inpainting synthesis instead of cut-and-paste.** A Stable Diffusion
  inpainting model hallucinates debris directly into real water scenes, with correct
  lighting, reflections, and water interaction. Region prompts are class-conditioned
  (`plastic_bottle`, `plastic_bag`, `plastic_debris`).
- **SAM 2 mask refinement.** Each inpainted region is refined into a precise instance
  mask with Segment Anything 2 (via Ultralytics), replacing hand-tuned contour
  extraction. Falls back to the eroded inpaint region when SAM is unavailable.
- **YOLO11 segmentation on PyTorch instead of Mask R-CNN on TensorFlow.** Faster
  training, simpler deployment (ONNX/TensorRT export in one line), and a modern
  dependency stack.
- **Native YOLO-seg labels** instead of COCO JSON — one normalized polygon per line,
  no annotation adapter needed.
- **Modern Python packaging:** `src/` layout, `pyproject.toml` (PEP 621), type hints,
  dataclass configs, `pathlib` throughout, `ruff` linting, and CLI entry points
  (`pws-synthesize`, `pws-train`, `pws-infer`).

## Project structure

```text
plastic-water-segmentation/
├── pyproject.toml                 # packaging, dependencies, ruff config
├── requirements.txt               # pip install -r (Colab friendly)
├── notebooks/
│   └── plastic_water_segmentation.ipynb   # end-to-end walkthrough
└── src/plastic_water_seg/
    ├── config.py                  # Category prompts, SynthConfig, TrainConfig
    ├── cli.py                     # pws-synthesize / pws-train / pws-infer
    ├── data/
    │   ├── synth.py               # DatasetSynthesizer orchestration
    │   ├── painters.py            # DiffusionInpainter (SOTA) + ProceduralPainter (fallback)
    │   ├── masks.py               # polygon export, SAM 2 refinement
    │   └── demo_assets.py         # procedural water backgrounds
    ├── training/train.py          # Ultralytics YOLO training wrapper
    └── eval/inference.py          # inference + contamination metric
```

Generated artifacts (`dataset/`, `logs/`, `data/`) are gitignored.

## Quickstart

### Option A — Google Colab (recommended)

1. Open the notebook in Colab (badge at the top of
   `notebooks/plastic_water_segmentation.ipynb`) with a GPU runtime.
2. Run the cells: install → configure → generate demo backgrounds → synthesize →
   inspect → train → infer.

### Option B — Local

```bash
git clone https://github.com/cemgurbey/plastic-water-segmentation.git
cd plastic-water-segmentation
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
```

Then either use the notebook or the CLI:

```bash
# 1. Synthesize a dataset (diffusion inpainting; needs a GPU)
pws-synthesize --backgrounds data/backgrounds --output dataset \
    --method diffusion --train-count 500 --val-count 100 --demo-backgrounds

#    ...or the lightweight procedural fallback (CPU-friendly smoke test)
pws-synthesize --method procedural --train-count 60 --val-count 15 --demo-backgrounds

# 2. Train
pws-train --data dataset/dataset.yaml --model yolo11m-seg.pt --epochs 100

# 3. Inference + contamination index
pws-infer --weights logs/yolo11m-seg/weights/best.pt --image test_images/river.jpg
```

## Using your own images

- **Backgrounds** (`data/backgrounds/`): real water photos ie. calm water, ripples, waves,
  sediment, sun glints, overcast. The more varied, the better the model generalizes.
- **Test images** (`test_images/`): unseen polluted water photos for evaluation.

## Evaluation metric: Surface Plastic Contamination Index

$$\text{Contamination} = \frac{\sum \text{plastic mask pixels}}{\text{total image pixels}} \times 100\%$$

Computed at inference time from the predicted instance masks. Useful for tracking
pollution density over time or across river sectors.

## Requirements

- Python 3.10+
- GPU strongly recommended for diffusion synthesis and training (Colab T4 works);
  the procedural painter and CPU inference run anywhere.
- Key dependencies: `torch`, `ultralytics`, `diffusers`, `transformers`, `accelerate`,
  `opencv-python`, `pillow`, `numpy`, `matplotlib`, `pyyaml`, `tqdm`, `rich`.

## License

MIT
