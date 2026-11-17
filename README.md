# DINOv3 for SLAM on TUM RGB-D

Three notebooks probe whether frozen DINOv3 features help with the main parts of an RGB-D SLAM front end. Each compares DINOv3 against ORB.

| Notebook | SLAM question | Data | Metrics |
|---|---|---|---|
| `01_loop_closure.ipynb` | Can it recognise revisited places? CLS vs pooled/GeM/VLAD patch tokens vs ORB BoW, plus geometric verification with patch matches | `fr3_long_office_household` | Recall@N, PR-AUC, recall @ 100% precision |
| `02_tracking.ipynb` | Are 16-px patches precise enough for pose? Patch matches + depth + PnP-RANSAC; coarse vs softargmax vs ×2 input vs LK refinement vs ORB | `fr1_desk` | GT match precision, inlier rate and pose error vs frame gap; VO ATE/RPE |
| `03_robustness.ipynb` | Where does it hold up or fail? Brightness, gamma, blur, noise, wide baselines, moving people (incl. a DINOv3 linear-probe "person" mask) | `fr1_desk`, `fr3_long_office_household`, `fr3_walking_xyz` | how the above degrade; inlier share on people; VO with and without masks |

Shared code is in `slam_utils.py`: the TUM loader, DINOv3 wrapper, matchers, PnP, ATE/RPE, loop-closure evaluation and perturbations.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
hf auth login          # DINOv3 weights on Hugging Face are gated: accept the licence on the model page first
.venv/bin/jupyter lab
```

TUM sequences are downloaded into `data/` on first use: fr1_desk ≈ 0.4 GB, long_office ≈ 1.6 GB, walking_xyz ≈ 0.9 GB.
A GPU is recommended. On CPU, ViT-S/16 takes about 0.5–1 s per 640×480 frame, and full visual-odometry runs take tens of minutes. Each notebook has stride and sample-count knobs at the top of its cells.
