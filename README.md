# Loop closure with DINOv3 on TUM RGB-D

Can a frozen DINOv3 backbone do the loop-closure job of an RGB-D SLAM system better than ORB (as in ORB-SLAM)? A loop closure has to recognise a revisited place, verify it geometrically, and provide the relative pose that goes into the pose graph.

Data: TUM RGB-D `fr3_long_office_household`, keyframes at 6 Hz (497). The protocol is causal: a query may only match frames at least 20 s in its past. A ground-truth loop is a past frame within 0.5 m and 30° of the query, which gives 55 queries with a true loop.

## Pipeline

| Step | | |
|---|---|---|
| 1. Backbone | frozen DINOv3 ViT-S/16, one pass per keyframe | 40 × 30 grid of patch features |
| 2. Retrieval | GeM pooling of **block-9** patch tokens | keep the 5 most similar past keyframes |
| 3. Verification | mutual-NN patch matches, lifted to 3D with depth | PnP-RANSAC fits one camera motion |
| 4. Acceptance | accept the best candidate if ≥ 25 % of its matches are inliers (and ≥ 15 inliers) | pose refined with Lucas-Kanade |

## Results

| Method | AP | R@100P |
|---|---|---|
| ORB bag of words | 0.731 | 0.345 |
| DINOv3 block 12 (final layer), GeM | 0.733 | 0.255 |
| DINOv3 block 9, GeM (steps 1–2) | 0.894 | 0.527 |
| DINOv3 + verification (steps 1–3) | **0.957** | **0.764** |

After step 4: **53/55 true loops accepted, 0 harmful constraints**. A constraint is harmful if its pose is off by more than 10 cm or 5°. The 16 other accepted loops lie outside the 0.5 m / 30° radius but still have a correct pose.

Why block 9 rather than the last block? It is better on both AP and R@100P. One hypothesis, not tested here, is that the last blocks encode *what* is in the image while block 9 still keeps *where* things are.

## Files

| File | Purpose |
|---|---|
| `loop_closure.py` | The whole pipeline, the ORB baseline, the tables and figures |
| `slam_utils.py` | TUM loader, DINOv3 wrapper, ORB BoW, patch matching, PnP, evaluation |
| `results/retrieval.csv` | Table above |
| `results/accepted_loops.csv` | Every accepted loop with its pose error and class |
| `results/summary.json` | All numbers above |
| `results/fig_ground_truth.png` | Trajectory with the 55 true-loop queries |
| `results/fig_examples.png` | Retrieved frames: DINOv3 retrieval only, ORB BoW, DINOv3 + verification |

<!-- ## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # the HF DINOv3 weights are gated; without a token the ungated timm copy is used
.venv/bin/python loop_closure.py
```

The TUM sequences are downloaded into `data/` (not on git  due to size) on first use: `fr3_long_office_household` is about 1.6 GB, and `fr1_desk` (0.4 GB) is used to train the ORB vocabulary. Features are cached in `results/cache/`. The first run takes about 15 minutes on CPU; later runs take about 4 minutes. -->
