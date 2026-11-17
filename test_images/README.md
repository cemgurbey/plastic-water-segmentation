# Real test images — plastic on water

40 real photos of plastic pollution on rivers, lakes, canals, and the sea,
for evaluating a trained model on data it never saw during training
(the training set is synthetic — see the main README).

- **Filenames:** `01_….jpg` … `40_….jpg`, resized to max 1280 px on the long
  edge (JPEG quality 85) to keep the repo small. Originals are linked in
  `SOURCES.csv`.
- **Provenance:** every image is freely licensed — CC0, CC BY, CC BY-SA, or
  public domain — sourced via Openverse and Wikimedia Commons. Full
  attribution (title, creator, license, license URL, source page, original
  image URL) is in [`SOURCES.csv`](SOURCES.csv). If you reuse these images,
  keep the attribution: CC BY / CC BY-SA images require credit, and CC BY-SA
  images require derivatives to be shared under the same license.
- **Visual tags:** `SOURCES.csv` also carries a rough `visual_tag`
  (`plastic_bottle` / `plastic_bag` / `plastic_debris`) from a human look at
  each photo. These are *not* ground-truth labels — they say what plastic is
  visible, not where it is. Do not use them as training labels.
- **What to expect:** these are hard, real-world images — small and
  translucent bottles, dense debris patches, algae, glare, and clutter. A
  model trained only on the small procedural demo set will miss a lot here;
  that gap is the point of this folder. See `results/` for the latest
  evaluation run and `notebooks/plastic_water_segmentation.ipynb` §8 to
  reproduce it in Colab.

## How this set was built

1. Searched Openverse (`license_type=commercial`, i.e. CC0 / BY / BY-SA /
   public domain only) and Wikimedia Commons for plastic bottles, bags, and
   debris floating on water.
2. Visually inspected thumbnails and kept only photos where plastic is
   clearly visible on or at the edge of a water surface. Near-duplicates,
   charts, drawings, beach-only shots, and tabletop close-ups were dropped.
3. Downloaded the originals and resized them to max 1280 px.
