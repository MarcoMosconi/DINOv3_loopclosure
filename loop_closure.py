"""Loop closure with frozen DINOv3 on TUM RGB-D fr3_long_office_household (the pipeline in the slides).

  Step 1  Backbone      frozen DINOv3 ViT-S/16, one pass per keyframe -> 40 x 30 patch grid
  Step 2  Retrieval     GeM of block-9 patch tokens; keep the 5 most similar past keyframes
  Step 3  Verification  mutual-NN patch matches (final-layer tokens), lifted to 3D with depth, PnP-RANSAC
  Step 4  Acceptance    accept the best candidate only if >= 25 % of its matches are PnP inliers

Baseline: ORB bag of words (as in ORB-SLAM). Block 12 (final layer) is reported to justify block 9.
An accepted loop is *harmful* if the relative pose it would add to the pose graph (refined with
Lucas-Kanade) is off by more than 10 cm or 5 deg from motion-capture ground truth.

Usage:  python loop_closure.py      (first run ~15 min on CPU for features; cached in results/cache/)
Outputs (results/): retrieval.csv, accepted_loops.csv, summary.json, fig_ground_truth.png, fig_examples.png
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

SEQ, VOCAB_SEQ = "fr3_long_office_household", "fr1_desk"
STRIDE = 5  # 30 Hz -> 6 Hz keyframes
EXCLUDE_S = 20.0  # a query may only match frames at least 20 s in its past
D_THR, ANG_THR = 0.5, 30.0  # ground-truth loop: within 0.5 m and 30 deg
RET_LAYER = 9  # step 2: block used for the retrieval descriptor
TOP_K = 5  # step 2: candidates verified per query
MIN_SIM, PNP_THR_PX = 0.3, 8.0  # step 3: patch-match similarity, RANSAC threshold for 16 px patch centres
MIN_INLIERS, MIN_RATIO = 15, 0.25  # step 4: acceptance
OK_T_CM, OK_R_DEG = 10.0, 5.0  # a loop constraint with a larger pose error is harmful
OUT = Path(__file__).resolve().parent / "results"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
cv2.setRNGSeed(0)

seq = su.TUMSequence(SEQ)
frames = np.arange(0, len(seq), STRIDE)
poses, ts = seq.poses[frames], seq.timestamps[frames]
P = poses[:, :3, 3]
cand, pos = su.loop_ground_truth(poses, ts, D_THR, ANG_THR, EXCLUDE_S)
has_loop = pos.any(1)
qs = np.nonzero(cand.any(1))[0]
print(seq, "|", len(frames), "keyframes |", int(has_loop.sum()), "queries with a true loop")

# ---------------------------------------------------------------- step 1: DINOv3 features (cached)
f_feat = CACHE / "main_s1.0.npz"
if not f_feat.exists():
    import torch

    torch.set_num_threads(8)
    dino = su.DinoV3()
    print("backbone:", dino.name, "| device", dino.device)
    acc = {l: ([], []) for l in (RET_LAYER, 12)}
    for i in tqdm(range(0, len(frames), 8), desc="DINOv3", mininterval=10):
        o, stride = dino.blocks([seq.rgb(j) for j in frames[i : i + 8]], layers=(RET_LAYER,))
        for l, (c, p) in o.items():
            acc[l][0].append(c)
            acc[l][1].append(p.astype(np.float16))
    np.savez(f_feat, stride=np.array(stride), **{f"cls_{l}": np.concatenate(c) for l, (c, _) in acc.items()},
             **{f"pat_{l}": np.concatenate(p) for l, (_, p) in acc.items()})
z = np.load(f_feat)
stride = tuple(z["stride"])
PAT12 = z["pat_12"]  # final-layer patch tokens, used for matching (step 3)
desc = {
    "ORB bag of words": None,
    "DINOv3 block 12 (final), GeM": np.stack([su.gem(p) for p in PAT12]),
    f"DINOv3 block {RET_LAYER}, GeM (steps 1-2)": np.stack([su.gem(p) for p in z[f"pat_{RET_LAYER}"]]),
}
del z

# ---------------------------------------------------------------- baseline: ORB bag of words (cached)
f_orb = CACHE / "orb_bow.npy"
if f_orb.exists():
    desc["ORB bag of words"] = np.load(f_orb)
else:
    vocab_seq = su.TUMSequence(VOCAB_SEQ)  # vocabulary trained on another sequence
    bow = su.OrbBoW(k=1000, n_features=1000).fit_vocab([vocab_seq.rgb(f) for f in range(0, len(vocab_seq), 5)])
    TF = np.stack([bow.tf(seq.rgb(f)) for f in tqdm(frames, desc="ORB BoW", mininterval=10)])
    desc["ORB bag of words"] = bow.fit_idf(TF).encode(TF)
    np.save(f_orb, desc["ORB bag of words"])

# ---------------------------------------------------------------- step 2: appearance-only retrieval
results = {n: su.evaluate_retrieval(D @ D.T, cand, pos) for n, D in desc.items()}

# ---------------------------------------------------------------- step 3: geometric verification of the top 5
h, w = PAT12.shape[1:3]
centers = su.patch_centers(h, w, stride)
depth = lru_cache(maxsize=96)(lambda i: seq.depth(frames[i]))


@lru_cache(maxsize=128)
def feat(i):
    return {"grid": su.l2n(PAT12[i].astype(np.float32)).astype(np.float16), "h": h, "w": w, "stride": stride,
            "centers": centers, "gray": cv2.cvtColor(seq.rgb(frames[i]), cv2.COLOR_RGB2GRAY)}


def verify(q, j, lk=False):
    """Pose of query q relative to past keyframe j (3D from j's depth): (T_qj, #inliers, inlier ratio)."""
    pa, pb = su.match_patches(feat(j), feat(q), MIN_SIM, lk)
    T, n = su.estimate_pose_pnp(pa, pb, depth(j), seq.K, seq.dist, reproj_thr=3.0 if lk else PNP_THR_PX)
    return T, n, n / max(1, len(pa))


ret = f"DINOv3 block {RET_LAYER}, GeM (steps 1-2)"
S0 = np.where(cand, desc[ret] @ desc[ret].T, -np.inf)
S_rr = np.full_like(S0, -np.inf)  # score after verification: #inliers (ties broken by appearance)
info = {}
for q in tqdm(qs, desc="verification", mininterval=15):
    for j in np.argsort(-S0[q])[:TOP_K]:
        if np.isfinite(S0[q, j]):
            T, n, ratio = verify(q, j)
            S_rr[q, j] = n + 1e-3 * S0[q, j]
            info[(q, j)] = (T, n, ratio)
verified = "DINOv3 + verification (steps 1-3)"
results[verified] = su.evaluate_retrieval(S_rr, cand, pos)

tab = pd.DataFrame({n: {k: r[k] for k in ["AP", "R@100P", "R@1", "R@5"]} for n, r in results.items()}).T
tab.to_csv(OUT / "retrieval.csv", float_format="%.4f")
print(tab.round(3).to_string())

# ---------------------------------------------------------------- step 4: acceptance, judged by the pose it adds
top1 = results[verified]["top1"]
rows = []
for q in tqdm(qs, desc="acceptance", mininterval=15):
    j = top1[q]
    T_c, n, ratio = info.get((q, j), (None, 0, 0.0))
    if n < MIN_INLIERS or ratio < MIN_RATIO:
        continue
    T_lk, _, _ = verify(q, j, lk=True)  # pose that goes into the graph: LK-refined, coarse as fallback
    T_gt = seq.rel_pose(frames[j], frames[q])
    er, et = su.pose_error(T_lk if T_lk is not None else T_c, T_gt)
    good = et * 100 < OK_T_CM and er < OK_R_DEG
    rows.append({"q": q, "j": j, "t_query_s": ts[q] - ts[0], "t_match_s": ts[j] - ts[0], "inliers": n,
                 "inlier_ratio": ratio, "rot_err_deg": er, "trans_err_cm": 100 * et,
                 "gt_dist_m": float(np.linalg.norm(P[q] - P[j])), "true_loop": bool(pos[q, j]),
                 "class": ("true loop" if pos[q, j] else "valid overlap") if good else "harmful"})
acc = pd.DataFrame(rows)
acc.to_csv(OUT / "accepted_loops.csv", index=False, float_format="%.4f")
counts = acc["class"].value_counts()
n_true, n_harm = int(counts.get("true loop", 0)), int(counts.get("harmful", 0))
print(f"accepted {len(acc)} loops (inliers >= {MIN_INLIERS}, ratio >= {MIN_RATIO}): "
      f"{n_true}/{int(has_loop.sum())} true loops, {int(counts.get('valid overlap', 0))} valid overlaps, {n_harm} harmful")
med = acc[acc["class"] != "harmful"][["trans_err_cm", "rot_err_deg"]].median()
print(f"median pose error of accepted loops: {med.trans_err_cm:.2f} cm / {med.rot_err_deg:.2f} deg")

json.dump({"keyframes": int(len(frames)), "queries_with_true_loop": int(has_loop.sum()),
           "retrieval": tab.to_dict(orient="index"),
           "acceptance": {"min_inliers": MIN_INLIERS, "min_inlier_ratio": MIN_RATIO, "accepted": len(acc),
                          "true_loops": n_true, "valid_overlaps": int(counts.get("valid overlap", 0)), "harmful": n_harm,
                          "median_trans_err_cm": float(med.trans_err_cm), "median_rot_err_deg": float(med.rot_err_deg)}},
          open(OUT / "summary.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(P[:, 0], P[:, 1], c=ts - ts[0], s=6, cmap="viridis")
ax.scatter(P[has_loop, 0], P[has_loop, 1], facecolors="none", edgecolors="r", s=30, lw=0.6, label="query with a true loop")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "fig_ground_truth.png", dpi=110); plt.close()

q_ids = np.nonzero(has_loop)[0]
q_ids = q_ids[np.linspace(0, len(q_ids) - 1, 4).astype(int)]
cols = {"DINOv3 retrieval only": ret, "ORB BoW": "ORB bag of words", "DINOv3 + verification": verified}
fig, ax = plt.subplots(len(q_ids), 1 + len(cols), figsize=(4 * (1 + len(cols)), 3.2 * len(q_ids)))
for row, q in enumerate(q_ids):
    ax[row, 0].imshow(seq.rgb(frames[q])); ax[row, 0].set_title(f"Query, t = {ts[q] - ts[0]:.0f} s")
    for col, (label, n) in enumerate(cols.items(), 1):
        j = results[n]["top1"][q]
        ax[row, col].imshow(seq.rgb(frames[j]))
        ax[row, col].set_title(f"{label}: {'true loop' if pos[q, j] else 'wrong'}, {np.linalg.norm(P[q] - P[j]):.2f} m",
                               color="g" if pos[q, j] else "r", fontsize=9)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "fig_examples.png", dpi=90); plt.close()
print("done ->", OUT)
