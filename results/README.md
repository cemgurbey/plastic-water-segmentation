# Results — first evaluation on the 40 real test photos

Run locally on CPU, 2026-11-16. These are honest first numbers from the
smallest, cheapest version of the pipeline — not the production recipe.

## How the model was trained

- **Data:** procedural synthetic set only — 60 train / 15 val images
  (`ProceduralPainter` on procedural water backgrounds, SAM refinement off).
  No diffusion inpainting, no real backgrounds.
- **Model:** `yolo11n-seg` (the small checkpoint; the notebook default is
  `yolo11m-seg` on a Colab GPU), 10 epochs, imgsz 640, batch 8, CPU.
- **Inference:** confidence threshold 0.35 on all 40 images in `test_images/`.

On the synthetic validation set it looked good (mask mAP50 0.799 overall —
bottle 0.986, bag 0.486, debris 0.926). That did not transfer to real photos,
which is the point of this folder.

## `repo_model/` — the repo's own trained model

- **18/40** images with at least one detection; mean contamination index
  **8.58%** (corrected — the first published figure, 2.58%, understated it;
  see the mask note below).
- 26 objects total: 4× `plastic_bottle`, 20× `plastic_bag`, 2× `plastic_debris`.
- What it got right: the clean isolated bottle in `01_bottle_blue_water`
  (63% confidence, good mask). What it got wrong: class confusion everywhere
  — debris patches and even algae get called `plastic_bag`, and most real
  bottles are missed entirely, including the large Danube bottle in `12`
  and every dense Marilao River patch (`13`–`16`). There are false positives
  too, e.g. a "bottle" box on a person's head in `32`.
- Per-image numbers: [`repo_model/summary.csv`](repo_model/summary.csv);
  one visualization per image in the same folder.

### Ground-truth metrics (mask IoU ≥ 0.5) — `metrics.csv`

The 40 photos now have hand-verified ground-truth labels
(`test_images/labels/`, 213 instances: 137 bottle, 31 bag, 45 debris;
method and known limitations in `test_images/README.md`), so the model is
scored properly instead of by detection counts:

| Class | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| plastic_bottle | 0 | 4 | 137 | 0.00 | 0.00 | 0.00 | 0.00 |
| plastic_bag | 0 | 20 | 31 | 0.00 | 0.00 | 0.00 | 0.00 |
| plastic_debris | 1 | 1 | 44 | 0.50 | 0.02 | 0.04 | 0.64 |
| **Overall** | **1** | **25** | **212** | **0.04** | **0.00** | **0.01** | **0.64** |

The single true positive is the plate in image 28 (IoU 0.64). The nearest
miss is the bottle in image 01 (IoU 0.37 — the model's mask is 2.7× the
true mask, so it fails the 0.5 bar despite finding the bottle). The
picture is a little less bleak class-agnostically: 8 of the 26 detections
overlap real plastic at IoU ≥ 0.25, and 3 at ≥ 0.5 — the model often
senses plastic but labels it wrong, calling debris regions "bag" (images
38 and 30 both overlap real debris at IoU ~0.6, with the wrong class).
The 8.58% mean contamination is itself mostly false positives: giant
"bag" masks on algae and vegetation — 35% of the frame in image 25,
29% in images 03 and 09.

### Mask note — why the contamination figure changed

The first `summary.csv` was computed with a bug in `predict()`: it used
`res.masks.data`, which Ultralytics returns at the model's processed
resolution (e.g. 448×640), not the photo's resolution — so mask areas were
counted against the wrong pixel total and mask IoU could not be computed
at all. `predict()` now rasterizes the `masks.xy` polygons at the photo's
own resolution (fixed in `src/plastic_water_seg/eval/inference.py`),
which is also what makes the ground-truth metrics above possible. The
detection counts are unaffected; only the contamination values changed
(2.58% → 8.58%).

## `coco_baseline/` — COCO-pretrained `yolo11n-seg`, bottle class only

- **0/40** images with a bottle detection at confidence 0.35 — so
  against the 137 ground-truth bottles: 0 TP, 0 FP, 137 FN, recall 0.00
  (`metrics_coco.csv`).
- This was checked, not assumed: the model loads and detects other things
  fine (at 0.10 it calls the bottle in `33` an "airplane" at 0.26 and a
  "boat" at 0.24). Re-running all 40 images at 0.25 still finds 0 bottles;
  at 0.10 it finds exactly 1. Transparent, horizontal, half-submerged
  bottles on water are effectively out-of-distribution for COCO's `bottle`
  class (upright, opaque bottles on tables). That is the strongest argument
  for this project's own training data.
- Per-image numbers: [`coco_baseline/summary.csv`](coco_baseline/summary.csv);
  visualizations (Ultralytics plots) in the same folder.

## What would close the gap

1. Train on the diffusion path (`DiffusionInpainter` + SAM 2) on a GPU
   with real water backgrounds — that is the pipeline's intended recipe,
   and the notebook does exactly this on Colab.
2. More synthetic data (500+ images) and more epochs, `yolo11m-seg`.
3. Add real photos as backgrounds in `data/backgrounds/` so the synthetic
   distribution gets closer to these test images.

## Reproduce

- **Colab (GPU):** open the notebook and run all cells — section 8
  regenerates this folder.
- **CLI:** `pws-infer --weights runs/segment/logs/yolo11n-seg/weights/best.pt --image test_images/12_bottle_danube.jpg`
