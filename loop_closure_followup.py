"""Follow-up to loop_closure_experiments.py (run that first: this reuses its feature cache).

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


def run(appearance, how):
    S0 = np.where(cand, desc[appearance] @ desc[appearance].T, -np.inf)
    S_rr = np.full_like(S0, -np.inf)
    info = {}
    for q in tqdm(qs_all, desc=f"{appearance} + {how}", mininterval=20):
        for j in np.argsort(-S0[q])[:K_RR]:
            if np.isfinite(S0[q, j]):
                n, ratio, T = verify(q, j, how)
                S_rr[q, j] = n + 1e-3 * S0[q, j]
                info[(q, j)] = (n, ratio, T)
    top1 = S_rr.argmax(1)
    rows = []
    for q in qs_all:
        j = top1[q]
        n, ratio, T = info.get((q, j), (0, 0.0, None))
        T_gt = seq.rel_pose(frames[j], frames[q])
        er, et = su.pose_error(T, T_gt)
        rows.append({"q": q, "j": j, "inliers": n, "inlier_ratio": ratio, "rot_err_deg": er, "trans_err_cm": 100 * et,
                     "true_loop": bool(pos[q, j]), "query_has_loop": bool(has_loop[q]),
                     "gt_dist_m": float(np.linalg.norm(T_gt[:3, 3])), "gt_rot_deg": float(su.rotation_angle_deg(T_gt[:3, :3]))})
    df = pd.DataFrame(rows)
    good_pose = (df.trans_err_cm < OK_T_CM) & (df.rot_err_deg < OK_R_DEG)
    df["class"] = np.where(df.true_loop, "true loop", np.where(good_pose, "valid overlap", "harmful"))
    df["true_loop_bad_pose"] = df.true_loop & ~good_pose
    return df, su.evaluate_retrieval(S_rr, cand, pos)


pipelines = [(f"L{RET_LAYER} GeM", TAG), (f"L{RET_LAYER} GeM", "ORB"), ("L12 mean", TAG), ("ORB BoW", "ORB")]
per_q, retr, sweep = {}, {}, []
for app, how in pipelines:
    name = f"{app} + {how} verif."
    df, r = run(app, how)
    per_q[name], retr[name] = df, r
    df.to_csv(OUT / f"followup_top1_{app.replace(' ', '_')}_{how}.csv", index=False, float_format="%.4f")
    for tau in [10, 15, 20, 30, 40, 50, 75, 100, 150]:
        a = df[df.inliers >= tau]
        sweep.append({"pipeline": name, "criterion": "inliers", "tau": tau, "accepted": len(a),
                      "true_loops": int((a["class"] == "true loop").sum()),
                      "valid_overlap": int((a["class"] == "valid overlap").sum()),
                      "harmful": int((a["class"] == "harmful").sum()),
                      "recall_true_loops": (a["class"] == "true loop").sum() / has_loop.sum(),
                      "precision_label": (a["class"] == "true loop").mean() if len(a) else np.nan,
                      "precision_metric": (a["class"] != "harmful").mean() if len(a) else np.nan})
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        a = df[(df.inlier_ratio >= tau) & (df.inliers >= 15)]
        sweep.append({"pipeline": name, "criterion": "ratio (>=15 inl.)", "tau": tau, "accepted": len(a),
                      "true_loops": int((a["class"] == "true loop").sum()),
                      "valid_overlap": int((a["class"] == "valid overlap").sum()),
                      "harmful": int((a["class"] == "harmful").sum()),
                      "recall_true_loops": (a["class"] == "true loop").sum() / has_loop.sum(),
                      "precision_label": (a["class"] == "true loop").mean() if len(a) else np.nan,
                      "precision_metric": (a["class"] != "harmful").mean() if len(a) else np.nan})

sw = pd.DataFrame(sweep)
sw.to_csv(OUT / "followup_acceptance_metric.csv", index=False, float_format="%.4f")
rt = pd.DataFrame({n: {k: r[k] for k in ["R@1", "R@5", "AP", "R@100P"]} for n, r in retr.items()}).T
rt.to_csv(OUT / "followup_retrieval.csv", float_format="%.4f")
print(rt.round(3).to_string())
print(sw.round(3).to_string(index=False))


# safest threshold = smallest tau with zero harmful loops; report what it keeps
def safe(df, col="inliers", grid=range(5, 400)):
    for t in grid:
        a = df[df[col] >= t]
        if (a["class"] == "harmful").sum() == 0:
            return t, int((a["class"] == "true loop").sum()), int((a["class"] == "valid overlap").sum())
    return None, 0, 0


safe_rows = []
for n, df in per_q.items():
    t, tl, vo = safe(df)
    safe_rows.append({"pipeline": n, "min_inliers_zero_harmful": t, "true_loops_kept": tl,
                      "recall_true_loops": tl / has_loop.sum(), "valid_overlaps_kept": vo,
                      "harmful_examples": df[df["class"] == "harmful"].sort_values("inliers", ascending=False)
                      .head(3)[["q", "j", "inliers", "trans_err_cm", "rot_err_deg", "gt_dist_m"]].round(2).to_dict("records")})
sf = pd.DataFrame(safe_rows)
print(sf.drop(columns="harmful_examples").to_string(index=False))
for r in safe_rows:
    print(r["pipeline"], "top harmful:", r["harmful_examples"])

# pose refinement of the accepted (verified) loops of the best DINOv3 pipeline with LK
best = f"L{RET_LAYER} GeM + {TAG} verif."
d = per_q[best]
acc = d[d["class"] != "harmful"]
rows = []
for _, r in tqdm(acc.iterrows(), total=len(acc), desc="refine", mininterval=20):
    q, j = int(r.q), int(r.j)
    T_gt = seq.rel_pose(frames[j], frames[q])
    for how in [TAG, f"{TAG}+LK", "ORB"]:
        n, _, T = verify(q, j, how)
        er, et = su.pose_error(T, T_gt)
        rows.append({"q": q, "j": j, "class": r["class"], "method": how, "inliers": n, "rot_deg": er, "trans_cm": 100 * et})
pr = pd.DataFrame(rows)
pr.to_csv(OUT / "followup_pose_refinement.csv", index=False, float_format="%.4f")
prs = pr.groupby(["class", "method"]).agg(pairs=("q", "size"), solved=("rot_deg", lambda v: v.notna().mean()),
                                          med_trans_cm=("trans_cm", "median"), p90_trans_cm=("trans_cm", lambda v: v.quantile(0.9)),
                                          med_rot_deg=("rot_deg", "median"), p90_rot_deg=("rot_deg", lambda v: v.quantile(0.9)))
print(prs.round(2).to_string())

# figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for n in per_q:
    s = sw[(sw.pipeline == n) & (sw.criterion == "inliers")]
    ax[0].plot(s.tau, s.harmful, "o-", label=n)
    ax[1].plot(s.tau, s.true_loops + s.valid_overlap, "o-", label=n)
ax[0].set_ylabel("# harmful loops accepted (pose error > 10 cm or 5°)"); ax[1].set_ylabel("# correct loop constraints accepted")
for a in ax:
    a.set_xlabel("min #PnP inliers τ"); a.set_xscale("log"); a.grid(alpha=0.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig8_acceptance_metric.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"true loop": "tab:green", "valid overlap": "tab:blue", "harmful": "tab:red"}
for c, g in d.groupby("class"):
    ax.scatter(g.gt_dist_m, g.inliers, s=12, c=colors[c], label=f"{c} ({len(g)})")
ax.set_xlabel("ground-truth distance query ↔ retrieved frame [m]"); ax.set_ylabel("# PnP inliers"); ax.set_yscale("symlog")
ax.set_title(f"{best}: every query's top-1 after verification"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "fig9_top1_classes.png", dpi=110); plt.close()

json.dump({"retrieval": rt.to_dict(orient="index"), "safe_thresholds": safe_rows,
           "pose_refinement": {f"{a}|{b}": v for (a, b), v in prs.to_dict(orient="index").items()}},
          open(OUT / "followup_summary.json", "w"), indent=1, default=float)
print("done")
