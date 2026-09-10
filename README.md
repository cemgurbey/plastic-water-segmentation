# 🌊 Plastic Residue Detection on Water Bodies using CocoSynth & Mask R-CNN

A computer vision and deep learning project inspired by [akTwelve/cocosynth](https://github.com/akTwelve/cocosynth) (Adam Kelly / Immersive Limit), tailored specifically for detecting and segmenting plastic pollution and floating debris across rivers, lakes, canals, and marine environments.

---

## 🎯 Project Features

- **Synthetic COCO Dataset Generation (CocoSynth)**: Automatically generates thousands of photorealistic training images with instance-level polygon segmentations by compositing transparent plastic cutouts onto diverse water body backgrounds.
- **Aquatic Augmentations**: Implements random 360° rotation, scale variation, color/brightness jitter (simulating sunlight glints and cloud shadows), and alpha edge-softening for natural aquatic blending.
- **COCO Format Export**: Computes external contours with OpenCV, simplifies polygon vertices with Shapely, and outputs valid Microsoft COCO JSON (`coco_instances.json`).
- **Procedural Demo Generator**: Generates procedural water textures and transparent plastic shapes out of the box, allowing immediate execution and verification without needing initial image uploads.
- **Mask R-CNN on TensorFlow 2.x & Keras**: Employs `akTwelve/Mask_RCNN` with transfer learning from pretrained COCO weights (`mask_rcnn_coco.h5`).
- **Water Surface Contamination Metric**: Quantifies the percentage of the water surface area covered by detected plastic litter during real-world inference.

---

## 📂 Project Structure

```text
plastic-water-segmentation/
├── plastic_water_segmentation_cocosynth.ipynb   # Master Jupyter Notebook
├── cocosynth_engine.py                         # Reusable Python module for synthetic data generation
├── requirements.txt                            # Python environment dependencies
├── README.md                                   # Documentation and quickstart guide
├── data/
│   ├── foregrounds/                            # Transparent PNG cutouts of plastic items
│   │   ├── plastic_bottle/
│   │   ├── plastic_bag/
│   │   └── plastic_debris/
│   └── backgrounds/                            # Water surface photos (rivers, lakes, oceans)
├── dataset/
│   ├── train/                                  # Generated synthetic training images & annotations
│   │   ├── images/
│   │   └── coco_instances.json
│   └── val/                                    # Generated synthetic validation images & annotations
│       ├── images/
│       └── coco_instances.json
├── test_images/                                # Real-world water photos for inference
└── logs/                                       # Trained model checkpoints & TensorBoard logs
```

---

## 🚀 Quickstart Guide

### Option A: Running in Google Colab (Recommended for GPU Training)
1. Open [Google Colab](https://colab.research.google.com/) and upload `plastic_water_segmentation_cocosynth.ipynb` and `cocosynth_engine.py`.
2. Select a GPU runtime: **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook cells sequentially.
4. If you don't upload images, the notebook will automatically generate procedural demo assets so you can see the entire synthetic generation, training, and inference pipeline run immediately!

### Option B: Running Locally with Jupyter

1. Clone or open this repository directory:
   ```bash
   cd /Users/cemgurbey/.gemini/antigravity/scratch/plastic-water-segmentation
   ```
2. Create and activate a Python 3.8 - 3.10 virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook plastic_water_segmentation_cocosynth.ipynb
   ```

---

## 📸 Preparing Your Own Images

When you are ready to train on your custom images:

### 1. Transparent Plastic Cutouts (`data/foregrounds/<category>/`)
- Save plastic items as **PNG with transparency (RGBA)**.
- Isolate the foreground using tools like Photoshop, GIMP, [remove.bg](https://www.remove.bg/), or [Segment Anything (SAM)](https://segment-anything.com/).
- Organize items into category subfolders:
  - `data/foregrounds/plastic_bottle/`
  - `data/foregrounds/plastic_bag/`
  - `data/foregrounds/plastic_debris/`

### 2. Water Body Backgrounds (`data/backgrounds/`)
- Save photos of water bodies as `.jpg` or `.png`.
- Collect varied conditions: calm water, ripples, waves, river sediment, sun reflections, and overcast lighting.

### 3. Real-World Test Images (`test_images/`)
- Place unseen photos of polluted water bodies into `test_images/` to evaluate model generalization.

---

## 💻 Standalone Synthetic Generation via CLI

You can also run the synthetic data generator directly from the terminal:

```bash
# Generate procedural demo assets first (if needed)
python3 cocosynth_engine.py --demo --train-count 100 --val-count 20
```

---

## 📊 Evaluation Metric: Water Contamination Ratio

The inference module computes the **Surface Plastic Contamination Index (%)**:
$$\text{Contamination Ratio (\%)} = \frac{\sum \text{Plastic Mask Pixels}}{\text{Total Water Surface Pixels}} \times 100$$

This metric enables environmental researchers and municipal river monitors to track pollution density over time or across different river sectors.

---

## 🔗 References & Credits

- [akTwelve/cocosynth](https://github.com/akTwelve/cocosynth) - Adam Kelly (Immersive Limit)
- [akTwelve/Mask_RCNN](https://github.com/akTwelve/Mask_RCNN) - TensorFlow 2.x port of Matterport Mask R-CNN
- [COCO Dataset](https://cocodataset.org/) - Common Objects in Context Format
