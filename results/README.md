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

- **18/40** images with at least one detection; mean contamination index **2.58%**.
- 26 objects total: 4× `plastic_bottle`, 20× `plastic_bag`, 2× `plastic_debris`.
- What it got right: the clean isolated bottle in `01_bottle_blue_water`
  (63% confidence, good mask). What it got wrong: class confusion everywhere
  — debris patches and even algae get called `plastic_bag`, and most real
  bottles are missed entirely, including the large Danube bottle in `12`
  and every dense Marilao River patch (`13`–`16`). There are false positives
  too, e.g. a "bottle" box on a person's head in `32`.
- Per-image numbers: [`repo_model/summary.csv`](repo_model/summary.csv);
  one visualization per image in the same folder.

## `coco_baseline/` — COCO-pretrained `yolo11n-seg`, bottle class only

- **0/40** images with a bottle detection at confidence 0.35.
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
