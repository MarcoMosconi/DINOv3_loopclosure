# Loop closure with DINOv3 on TUM RGB-D

This repo tests whether frozen DINOv3 features can serve as the loop-closure module of an RGB-D SLAM system. A loop-closure module has three jobs: recognise a revisited place, verify the match geometrically, and estimate the relative pose that goes into the pose graph. Each step is compared against an ORB baseline (bag of words + ORB feature matching). The study is then repeated with DINOv2 to see how the two backbones differ.

All experiments use TUM RGB-D `fr3_long_office_household`. The protocol is causal (a query can only match frames at least 20 s in its past), and a ground-truth loop is a frame within 0.5 m and 30° of the query.

## Main results

| | ORB | DINOv3 ViT-S/16 |
|---|---|---|
| Appearance-only retrieval, AP / R@1 | 0.731 / 0.836 (BoW) | **0.894 / 0.945** (GeM of block-9 patch tokens) |
| After geometric verification, AP / R@100P | 0.828 / 0.400 | **0.957 / 0.764** |
| True loops recalled with zero bad constraints | 92.7 % | **96.4 %** |
| Median relative-pose error of accepted loops | 0.97 cm / 0.39° | 1.03 cm / 0.42° (with LK refinement) |

Recommended pipeline: block-9 GeM retrieval → top-5 candidates → PnP-RANSAC verification on DINOv3 patch matches + depth → accept if inlier ratio ≥ 0.25 → pose refined with Lucas-Kanade. It needs one backbone forward pass per keyframe.

Full write-ups:
- [LOOP_CLOSURE_REPORT.md](LOOP_CLOSURE_REPORT.md): protocol, DINOv3 vs ORB, ablations and limitations.
- [DINOV2_VS_DINOV3_REPORT.md](DINOV2_VS_DINOV3_REPORT.md): the same study with DINOv2 ViT-S/14 (4 registers).

## Contents

| File | Purpose |
|---|---|
| `01_loop_closure.ipynb` | Interactive retrieval study: CLS vs pooled/GeM/VLAD patch tokens vs ORB BoW, plus geometric verification with patch matches. Reports Recall@N, PR-AUC and recall at 100 % precision |
| `loop_closure_experiments.py` | Headless, extended version of the notebook: retrieval, temporal consistency, geometric verification, loop acceptance, loop-pose accuracy, layer × scale ablation |
| `loop_closure_followup.py` | Judges accepted loops by the pose they would add to the graph and sweeps the acceptance threshold |
| `loop_closure_final_pipeline.py` | End-to-end run of the recommended pipeline |
| `loop_closure_layer_sweep.py` | Retrieval quality of all 12 transformer blocks, for DINOv3 and DINOv2 |
| `loop_closure_compare.py` | Side-by-side DINOv3 vs DINOv2 tables and figures |
| `slam_utils.py` | Shared code: TUM loader, DINOv3/DINOv2 wrapper, matchers, PnP, loop-closure evaluation |
| `results/` | Tables, figures and logs: `loop_closure/` (DINOv3), `loop_closure_dinov2/`, `loop_closure_compare/` |

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
hf auth login          # optional: the HF DINOv3 weights are gated; without them the wrapper falls back to the ungated timm checkpoint
.venv/bin/jupyter lab
```

The TUM sequence is downloaded into `data/` on first use (fr3_long_office_household ≈ 1.6 GB; `fr1_desk` ≈ 0.4 GB is also fetched to train the VLAD and BoW vocabularies). A GPU is recommended. On CPU, ViT-S/16 takes about 1 s per 640×480 frame.

## Reproduce

```bash
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # avoids an invalid stored HF token blocking public timm downloads
.venv/bin/python loop_closure_experiments.py     # ~15–20 min CPU → results/loop_closure/
.venv/bin/python loop_closure_followup.py        # reuses the feature cache from the step above
.venv/bin/python loop_closure_final_pipeline.py

# DINOv2: same scripts, selected with LC_BACKBONE → results/loop_closure_dinov2/
LC_BACKBONE=dinov2 .venv/bin/python loop_closure_experiments.py
LC_BACKBONE=dinov2 .venv/bin/python loop_closure_followup.py
LC_BACKBONE=dinov2 .venv/bin/python loop_closure_final_pipeline.py

.venv/bin/python loop_closure_layer_sweep.py dinov3 dinov2
.venv/bin/python loop_closure_compare.py         # → results/loop_closure_compare/
```

`LC_RET_LAYER` changes the retrieval block used by the follow-up and final-pipeline scripts (default 9).
