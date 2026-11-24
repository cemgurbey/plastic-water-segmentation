# Plastic Residue Detection on Water Bodies

Instance segmentation for detecting plastic pollution — bottles, bags, and debris —
floating on rivers, lakes, canals, and marine surfaces. The pipeline generates its own
photorealistic training data with diffusion models, refines masks with SAM 2, and trains
a YOLO segmentation model on PyTorch.

## How it started

v1 of this project (August 2026) was a cut-and-paste synthetic data pipeline in the
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

The pipeline was rebuilt from scratch — no code, config, or design remains from the
old approach:

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
  (`pws-synthesize`, `pws-preview`, `pws-boxes-to-seg`, `pws-train`, `pws-infer`,
  `pws-eval`).

## Project structure

```text
plastic-water-segmentation/
├── pyproject.toml                 # packaging, dependencies, ruff config
├── requirements.txt               # pip install -r (Colab friendly)
├── notebooks/
│   └── plastic_water_segmentation.ipynb   # end-to-end walkthrough
└── src/plastic_water_seg/
    ├── config.py                  # Category prompts, SynthConfig, TrainConfig
    ├── cli.py                     # pws-* command-line entry points
    ├── data/
    │   ├── synth.py               # DatasetSynthesizer orchestration
    │   ├── painters.py            # DiffusionInpainter (SOTA) + ProceduralPainter (fallback)
    │   ├── masks.py               # polygon export, change masks, SAM 2 refinement
    │   ├── verify.py              # optional CLIP check of each inpainted object
    │   ├── preview.py             # label overlay grid + object-size stats
    │   ├── boxes_to_seg.py        # box-labelled real data -> YOLO-seg via SAM 2
    │   └── demo_assets.py         # procedural water backgrounds
    ├── training/train.py          # Ultralytics YOLO training wrapper
    └── eval/
        ├── inference.py           # inference + contamination metric
        ├── metrics.py             # P/R/F1 @ mask IoU, class-agnostic, macro recall
        └── run.py                 # one-call evaluation on test_images + mAP
tests/                             # pytest suite (no GPU / model downloads needed)
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
# 1. Synthesize a dataset (diffusion inpainting; needs a GPU and REAL water
#    photos in data/backgrounds/ — not the test images)
pws-synthesize --backgrounds data/backgrounds --output dataset \
    --method diffusion --train-count 500 --val-count 100 --clip-verify

#    ...or the lightweight procedural fallback (CPU-friendly smoke test)
pws-synthesize --method procedural --train-count 60 --val-count 15 --demo-backgrounds

# 2. Look at the data before training: label overlay grid + object sizes
#    (real test photos: median object ~0.2% of the frame)
pws-preview --images dataset/images/train --out dataset/preview.jpg

# 3. Train (imgsz 1024: most real plastic is tiny in the frame)
pws-train --data dataset/dataset.yaml --model yolo11m-seg.pt --epochs 100 --imgsz 1024

# 4. Evaluate on the real photos (dev split; P/R/F1, class-agnostic, mAP)
pws-eval --weights runs/segment/logs/yolo11m-seg/weights/best.pt --split dev

# 5. Single-image inference + contamination index
pws-infer --weights runs/segment/logs/yolo11m-seg/weights/best.pt --image test_images/12_bottle_danube.jpg
```

Run the tests with `pip install -e ".[dev]" && pytest` (no GPU needed).

## How synthesis keeps labels honest

The first diffusion run scored mask mAP50 0.0135 on its *own* validation set,
which means the labels did not match the images. The synthesizer now:

- **Samples small objects.** Region area is log-uniform in 0.05%-3% of the
  frame (it was 2-18%); the real test objects have a median of ~0.2%.
- **Inpaints on a crop and pastes back.** Each object is generated on a square
  crop around its region at 512 px, then pasted into the region only — small
  objects get full diffusion resolution and nothing else in the image changes.
- **Labels what actually changed.** The label starts from the pixels the painter
  changed, refined with SAM 2 when SAM agrees with them. If the region barely
  changed (diffusion painted water), or CLIP says it is not the class
  (`clip_verify`), the object is reverted and gets no label.
- **Never overlaps objects**, so a later object cannot paint over an earlier
  object's label.
- **Labels debris as one region**, the same convention as the test labels.
- Writes `dataset/synth_stats.json` (keep rate, rejects, object sizes) and warns
  when every background is a procedural demo texture.

## Real training data

Box-labelled public datasets of floating plastic — e.g. Kili "Plastic in River"
(bottle / bag / other plastic / not plastic) and FloW-Img (small floating
bottles) — can be converted to this project's YOLO-seg format with the same
box-prompted SAM 2 step used for the test labels (check each dataset's
license first):

```bash
# source ids -> project ids (0 bottle, 1 bag, 2 debris); null drops a class
pws-boxes-to-seg --images kili/images/train --labels kili/labels/train \
    --out data/real/train --class-map '{"0": 1, "1": 0, "2": 2, "3": null}'
```

Check the source dataset's class order before writing the map.

## Using your own images

- **Backgrounds** (`data/backgrounds/`): real water photos — calm water, ripples, waves,
  sediment, sun glints, overcast. The more varied, the better the model generalizes.
- **Test images** (`test_images/`): unseen polluted water photos for evaluation.
- No foreground cutouts needed anymore — the diffusion model generates the debris.

## Real test images (`test_images/`)

40 real photos of plastic on rivers, lakes, canals, and the sea — floating
bottles, bags, and dense debris patches — for evaluating a trained model on
data it never saw (training is synthetic). All are copyright-free (CC0,
CC BY, CC BY-SA, or public domain), sourced via Openverse and Wikimedia
Commons, resized to max 1280 px, with full attribution in
`test_images/SOURCES.csv`. See `test_images/README.md` for how the set was
built. The `visual_tag` column there is a rough human tag of what plastic is
visible — not ground-truth labels.

The photos are split into **dev (15)** and **test (25)** in
`test_images/SPLITS.csv` (stratified by visual tag). Compare runs on `dev`;
report `test` only when a change is final, so it stays a held-out number.

Run the evaluation from the notebook (§8) or the CLI:

```bash
pws-eval --weights runs/segment/logs/yolo11m-seg/weights/best.pt --split dev
```

It writes per-class P/R/F1 at mask IoU ≥ 0.5, a class-agnostic "plastic"
score, per-image counts and threshold-free mAP50 / mAP50-95. Two photos
(05 and 40) hold 56% of all ground-truth bottles, so it also reports
per-image macro recall, where every photo weighs the same.

Latest results live in `results/` — one visualization per image plus
`summary.csv` (detections per class and contamination index), for both the
repo-trained model (`results/repo_model/`) and a COCO-pretrained
`yolo11n-seg` baseline (`results/coco_baseline/`, bottle detections only).
See [`results/README.md`](results/README.md) for the numbers and what they mean.
Expect the honest gap: a model trained on the small procedural demo set
misses a lot on real photos, and the COCO baseline found no bottles at all
(0/40) — floating, half-submerged bottles are out of distribution for COCO.
Closing that gap is what real training data and the diffusion-synthesis path
(GPU, Colab) are for.

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
