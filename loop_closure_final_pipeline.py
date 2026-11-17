"""(setup copied from loop_closure_followup.py) Follow-up to loop_closure_experiments.py (run that first: this reuses its feature cache).

1. Uses the best retrieval descriptor from the ablation (GeM of block-9 patch tokens) as front end.
2. Judges accepted loops the way a pose graph sees them: a loop is *harmful* only if its estimated
   relative pose is wrong. Every accepted top-1 is compared with the motion-capture relative pose:
       true loop          - within D_THR / ANG_THR of the query (the label used for R@N / AP)
       valid overlap      - outside that radius, but the estimated pose is within 10 cm / 5 deg of
                            ground truth: a correct, usable loop constraint
       harmful            - estimated pose is wrong (> 10 cm or > 5 deg): this corrupts the graph
3. Sweeps the acceptance threshold on #inliers and on inlier ratio, and refines accepted poses
   with DINOv3 + Lucas-Kanade.

Outputs: results/loop_closure/followup_*.csv, fig8_*.png, followup_summary.json
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import slam_utils as su

SEQ, VOCAB_SEQ, STRIDE, EXCLUDE_S, D_THR, ANG_THR, K_RR = "fr3_long_office_household", "fr1_desk", 5, 20.0, 0.5, 30.0, 5
OK_T_CM, OK_R_DEG = 10.0, 5.0
BACKBONE, _BB = su.lc_backbone()
TAG = _BB["label"]
RET_LAYER = int(__import__("os").environ.get("LC_RET_LAYER", 9))  # block for the GeM retrieval descriptor
OUT = Path(__file__).resolve().parent / "results" / _BB["out"]
CACHE = OUT / "cache"

seq = su.TUMSequence(SEQ, su.DATA_ROOT, download=False)
frames = np.arange(0, len(seq), STRIDE)
poses, ts = seq.poses[frames], seq.timestamps[frames]
cand, pos = su.loop_ground_truth(poses, ts, D_THR, ANG_THR, EXCLUDE_S, causal=True)
has_loop = pos.any(1)
qs_all = np.nonzero(cand.any(1))[0]

z = np.load(CACHE / "main_s1.0.npz")
stride = tuple(z["stride"])
PAT12 = z["pat_12"]
desc = {
    f"L{RET_LAYER} GeM": np.stack([su.pool_descriptors(c, p.astype(np.float32))["gem"] for c, p in zip(z[f"cls_{RET_LAYER}"], z[f"pat_{RET_LAYER}"])]),
    "L12 mean": np.stack([su.pool_descriptors(c, p.astype(np.float32))["mean"] for c, p in zip(z["cls_12"], PAT12)]),
    "ORB BoW": np.load(CACHE / "orb_bow.npy"),
}
del z
h, w = PAT12.shape[1:3]
centers = su.patch_centers(h, w, stride)
gray = lru_cache(maxsize=None)(lambda i: cv2.cvtColor(seq.rgb(frames[i]), cv2.COLOR_RGB2GRAY))
depth = lru_cache(maxsize=96)(lambda i: seq.depth(frames[i]))
dfeat = lru_cache(maxsize=128)(lambda i: {"grid": su.l2n(PAT12[i].astype(np.float32)).astype(np.float16), "h": h, "w": w,
                                         "stride": stride, "centers": centers, "gray": gray(i)})


class _NoExt:
    scale = 1.0


dmatch = {None: su.DinoMatcher(_NoExt(), None, min_sim=0.3), "lk": su.DinoMatcher(_NoExt(), "lk", min_sim=0.3)}
orb = su.OrbMatcher(2000, 0.8)
orb_feat = lru_cache(maxsize=None)(lambda i: orb.features(seq.rgb(frames[i])))


def verify(q, j, how):
    if how == "ORB":
        m, thr = orb.match(orb_feat(j), orb_feat(q)), 3.0
    elif how == TAG:
        m, thr = dmatch[None].match(dfeat(j), dfeat(q)), 8.0
    else:  # f"{TAG}+LK"
        m, thr = dmatch["lk"].match(dfeat(j), dfeat(q)), 3.0
    r = su.estimate_pose_pnp(m["pts_a"], m["pts_b"], depth(j), seq.K, seq.dist, reproj_thr=thr, iters=500)
    n = int(r["inliers"].sum())
    return n, n / max(1, len(m["pts_a"])), r["T_ba"]




# ---------------------------------------------------------------- final pipeline
"""Recommended pipeline, judged on the pose that would enter the pose graph:
   L9 GeM retrieval -> top-5 DINOv3 patch verification (choose best) -> accept rule -> pose from DINOv3 + LK
   (fallback: coarse DINOv3 pose if LK-PnP fails). For every accepted loop we also report the GT visual
   overlap so that 'wrong place' (overlap ~ 0) can be told apart from 'right place, imprecise pose'."""
top = pd.read_csv(OUT / f"followup_top1_L{RET_LAYER}_GeM_{TAG}.csv")
rows = []
for _, r in tqdm(top.iterrows(), total=len(top), desc="final", mininterval=20):
    q, j = int(r.q), int(r.j)
    if r.inliers < 15:
        continue
    T_gt = seq.rel_pose(frames[j], frames[q])
    n_lk, _, T_lk = verify(q, j, f"{TAG}+LK")
    _, _, T_c = verify(q, j, TAG)
    T = T_lk if T_lk is not None else T_c
    er, et = su.pose_error(T, T_gt)
    ov = su.overlap_ratio(depth(j), T_gt, seq.K, seq.dist, depth(j).shape)
    rows.append({**r.to_dict(), "lk_inliers": n_lk, "lk_solved": T_lk is not None, "final_rot_deg": er,
                 "final_trans_cm": 100 * et, "gt_overlap": ov})
f = pd.DataFrame(rows)
good = (f.final_trans_cm < OK_T_CM) & (f.final_rot_deg < OK_R_DEG)
f["final_class"] = np.where(f.true_loop & good, "true loop", np.where(good, "valid overlap",
                   np.where(f.gt_overlap < 0.05, "wrong place", "right place, bad pose")))
f.to_csv(OUT / "final_pipeline_top1.csv", index=False, float_format="%.4f")
rows = []
for crit, grid in [("inlier_ratio", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]), ("inliers", [15, 20, 30, 50, 75, 100])]:
    for t in grid:
        a = f[f[crit] >= t]
        c = a.final_class.value_counts()
        rows.append({"criterion": crit, "tau": t, "accepted": len(a), **{k: int(c.get(k, 0)) for k in
                     ["true loop", "valid overlap", "right place, bad pose", "wrong place"]},
                     "recall_true_loops": c.get("true loop", 0) / has_loop.sum()})
fp = pd.DataFrame(rows)
fp.to_csv(OUT / "final_pipeline_acceptance.csv", index=False, float_format="%.4f")
print(fp.round(3).to_string(index=False))
acc = f[f.inlier_ratio >= 0.3]
print("accepted at ratio>=0.3: pose error of the final constraint")
print(acc.groupby("final_class")[["final_trans_cm", "final_rot_deg", "gt_dist_m", "gt_rot_deg", "gt_overlap"]].describe().T.round(2).to_string())
wp = f[f.final_class == "wrong place"]
print("all wrong-place top-1s (any score):", len(wp),
      *(("| max inlier ratio among them:", round(float(wp.inlier_ratio.max()), 3),
         "| max inliers:", int(wp.inliers.max())) if len(wp) else ()))
print(f[f.final_class == "right place, bad pose"].sort_values("inlier_ratio", ascending=False).head(6)[
    ["q", "j", "inliers", "inlier_ratio", "lk_solved", "final_trans_cm", "final_rot_deg", "gt_dist_m", "gt_rot_deg", "gt_overlap"]].round(2).to_string())
