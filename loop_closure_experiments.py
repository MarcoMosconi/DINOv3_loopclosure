"""Loop-closure experiments with DINOv3 (or DINOv2, LC_BACKBONE=dinov2) on TUM RGB-D fr3_long_office_household.

Headless, extended version of 01_loop_closure.ipynb. It uses the same protocol and settings
and adds the parts a real loop-closure module needs:

  1. Retrieval: CLS / mean / GeM / CLS+mean / VLAD vs ORB bag of words (as in the notebook)
  2. Temporal consistency: average the similarity along the diagonal over L consecutive keyframes
  3. Geometric verification: top-K candidates -> patch matches + depth + PnP-RANSAC, using DINOv3
     patch tokens and, as a baseline, ORB features on the same candidates
  4. Loop acceptance: precision/recall of the full detector when a loop is accepted if the
     number of PnP inliers is at least tau
  5. Loop-constraint accuracy: relative-pose error of accepted loops vs motion-capture ground truth
     (coarse DINOv3 patches, soft-argmax, DINOv3 + Lucas-Kanade, ORB)
  6. Ablations: transformer layer (6 / 9 / 12) x input scale (0.5 / 1.0), and ground-truth radius

Usage:  .venv/bin/python loop_closure_experiments.py      (about 15-20 min on CPU the first time)
Outputs go to results/loop_closure/ (tables as CSV + JSON summary, figures as PNG);
extracted features are cached in results/loop_closure/cache/.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import slam_utils as su

# ---------------------------------------------------------------- settings (= notebook defaults)
SEQ = "fr3_long_office_household"
VOCAB_SEQ = "fr1_desk"
BACKBONE, _BB = su.lc_backbone()
MODEL, TAG = _BB["model"], _BB["label"]
STRIDE = 5  # 30 Hz -> 6 Hz keyframes
EXCLUDE_S = 20.0
D_THR, ANG_THR = 0.5, 30.0
LAYERS = (6, 9)  # intermediate blocks for the ablation; "12" below = final normed output
K_RR = 5  # candidates verified per query
SEQ_LENS = (1, 3, 5, 9)
OUT = Path(__file__).resolve().parent / "results" / _BB["out"]
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)
torch.set_num_threads(8)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


seq = su.TUMSequence(SEQ, su.DATA_ROOT)
vocab_seq = su.TUMSequence(VOCAB_SEQ, su.DATA_ROOT)
frames = np.arange(0, len(seq), STRIDE)
vfr = np.arange(0, len(vocab_seq), 5)
poses, ts = seq.poses[frames], seq.timestamps[frames]
P = poses[:, :3, 3]
cand, pos = su.loop_ground_truth(poses, ts, D_THR, ANG_THR, EXCLUDE_S, causal=True)
has_loop = pos.any(1)
log(seq, "|", len(frames), "keyframes |", has_loop.sum(), "queries with a true loop")

_dino = None


def dino():
    global _dino
    if _dino is None:
        _dino = su.DinoV3(MODEL)
        log("backbone:", _dino.name, "| backend", _dino.backend, "| device", _dino.device)
    return _dino


def forward_multi(imgs, scale, layers):
    """One forward pass -> {layer: (cls [B,D], patches [B,h,w,D])}; key 12 = final normed output."""
    d = dino()
    x, h, w, stride = d._prep(imgs, scale)
    x = x.to(d.device)
    out = {}
    with torch.inference_mode():
        if d.backend == "timm" and not layers:
            toks = {12: d.model.forward_features(x)}
        elif d.backend == "timm":
            final, inter = d.model.forward_intermediates(
                x, indices=[l - 1 for l in layers], output_fmt="NLC", return_prefix_tokens=True, norm=False
            )
            toks = {12: final} | {l: torch.cat([pre, pat], 1) for l, (pat, pre) in zip(layers, inter)}
        else:
            o = d.model(pixel_values=x, output_hidden_states=True)
            toks = {12: o.last_hidden_state} | {l: o.hidden_states[l] for l in layers}
    for l, t in toks.items():
        t = t.float().cpu().numpy()
        out[l] = (t[:, 0], t[:, 1 + d.n_reg :].reshape(len(imgs), h, w, -1))
    return out, stride


def extract(sq, idx, scale, layers, tag, bs=8):
    """Cached multi-layer extraction. Returns {layer: (CLS, PATCHES fp16)}, stride."""
    f = CACHE / f"{tag}.npz"
    if f.exists():
        z = np.load(f)
        keys = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("cls_")})
        return {l: (z[f"cls_{l}"], z[f"pat_{l}"]) for l in keys}, tuple(z["stride"])
    acc, t = {}, time.time()
    for i in tqdm(range(0, len(idx), bs), desc=f"{TAG} {tag}", mininterval=10):
        o, stride = forward_multi([sq.rgb(j) for j in idx[i : i + bs]], scale, layers)
        for l, (c, p) in o.items():
            acc.setdefault(l, ([], []))
            acc[l][0].append(c)
            acc[l][1].append(p.astype(np.float16))
    res = {l: (np.concatenate(c), np.concatenate(p)) for l, (c, p) in acc.items()}
    log(f"{tag}: {1000 * (time.time() - t) / len(idx):.0f} ms/frame")
    np.savez(f, stride=np.array(stride), **{f"cls_{l}": c for l, (c, _) in res.items()},
             **{f"pat_{l}": p for l, (_, p) in res.items()})
    return res, stride


def global_descs(CLS, PAT, centers):
    pooled = [su.pool_descriptors(c, p.astype(np.float32)) for c, p in zip(CLS, PAT)]
    D = {k: np.stack([d[k] for d in pooled]) for k in pooled[0]}
    D["vlad"] = np.stack([su.vlad(p.astype(np.float32), centers) for p in PAT])
    return D


def vlad_vocab(VPAT, k=32):
    X = su.l2n(VPAT.reshape(-1, VPAT.shape[-1]).astype(np.float32))
    X = X[np.random.default_rng(0).choice(len(X), min(len(X), 100_000), replace=False)]
    return su.fit_kmeans(X, k).cluster_centers_


def summarise(r):
    return {k: float(r[k]) for k in ["R@1", "R@5", "R@10", "AP", "R@100P"] if k in r}


def seq_smooth(S, L):
    """Temporal consistency: mean of S[i-k, j-k] for k < L (causal; both runs move forward)."""
    if L == 1:
        return S
    acc, n = np.zeros_like(S), np.zeros_like(S)
    for k in range(L):
        acc[k:, k:] += S[: len(S) - k, : len(S) - k]
        n[k:, k:] += 1
    return acc / n


summary = {"config": dict(backbone_key=BACKBONE, seq=SEQ, vocab_seq=VOCAB_SEQ, model=MODEL, stride=STRIDE, exclude_s=EXCLUDE_S,
                          d_thr=D_THR, ang_thr=ANG_THR, k_rr=K_RR, n_keyframes=int(len(frames)),
                          n_queries_with_loop=int(has_loop.sum()), n_queries_with_candidates=int(cand.any(1).sum()))}

# ================================================================ 1. features + descriptors
V1, _ = extract(vocab_seq, vfr, 1.0, LAYERS, "vocab_s1.0")
M1, stride = extract(seq, frames, 1.0, LAYERS, "main_s1.0")
V05, _ = extract(vocab_seq, vfr, 0.5, (), "vocab_s0.5")
M05, _ = extract(seq, frames, 0.5, (), "main_s0.5")
summary["config"]["backbone"] = dino().name if _dino else "(cached)"

CLS, PAT = M1[12]
h, w = PAT.shape[1:3]
log("descriptors ...")
ablation_descs = {}
for scale, V, M in [(1.0, V1, M1), (0.5, V05, M05)]:
    for l in M:
        ablation_descs[(l, scale)] = global_descs(*M[l], vlad_vocab(V[l][1]))
desc = {f"{TAG} {k}": v for k, v in ablation_descs[(12, 1.0)].items()}

f_orb = CACHE / "orb_bow.npy"
if f_orb.exists():
    desc["ORB BoW"] = np.load(f_orb)
else:
    bow = su.OrbBoW(k=1000, n_features=1000).fit_vocab([vocab_seq.rgb(f) for f in vfr])
    TF = np.stack([bow.tf(seq.rgb(f)) for f in tqdm(frames, desc="ORB BoW", mininterval=10)])
    bow.fit_idf(TF)
    desc["ORB BoW"] = bow.encode(TF)
    np.save(f_orb, desc["ORB BoW"])

# ================================================================ 2. retrieval results
results = {n: su.evaluate_retrieval(D @ D.T, cand, pos) for n, D in desc.items()}
tab = pd.DataFrame({n: summarise(r) for n, r in results.items()}).T
tab.to_csv(OUT / "retrieval.csv", float_format="%.4f")
log("retrieval\n" + tab.round(3).to_string())
summary["retrieval"] = tab.to_dict(orient="index")
dino_names = [n for n in desc if n.startswith(f"{TAG}")]
best = tab.loc[dino_names, "AP"].idxmax()
summary["best_dino_descriptor"] = best
log(f"best {TAG} descriptor (by AP):", best)

# ---- ablation: layer x scale
rows = []
for (l, scale), D in ablation_descs.items():
    for k, d in D.items():
        r = su.evaluate_retrieval(d @ d.T, cand, pos)
        rows.append({"layer": l, "scale": scale, "desc": k, **summarise(r)})
abl = pd.DataFrame(rows)
abl.to_csv(OUT / "ablation_layer_scale.csv", index=False, float_format="%.4f")
abl_p = abl.pivot_table(index=["layer", "scale"], columns="desc", values=["R@1", "AP"]).round(3)
log("layer/scale ablation\n" + abl_p.to_string())
summary["ablation_best"] = abl.sort_values("AP", ascending=False).head(5).to_dict(orient="records")

# ---- ground-truth radius sensitivity
rows = []
for d_thr in [0.25, 0.5, 1.0, 1.5]:
    c_, p_ = su.loop_ground_truth(poses, ts, d_thr, ANG_THR, EXCLUDE_S)
    for n in dict.fromkeys([best, f"{TAG} cls", f"{TAG} vlad", "ORB BoW"]):
        r = su.evaluate_retrieval(desc[n] @ desc[n].T, c_, p_)
        rows.append({"D_THR": d_thr, "descriptor": n, "#loops": r["n_queries_with_loop"], **summarise(r)})
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "sensitivity_dthr.csv", index=False, float_format="%.4f")
log("D_THR sensitivity\n" + sens.pivot(index="D_THR", columns="descriptor", values=["R@1", "AP"]).round(3).to_string())

# ---- temporal consistency
rows = []
for n in dict.fromkeys([best, f"{TAG} cls", f"{TAG} vlad", "ORB BoW"]):
    S = desc[n] @ desc[n].T
    for L in SEQ_LENS:
        rows.append({"descriptor": n, "L": L, **summarise(su.evaluate_retrieval(seq_smooth(S, L), cand, pos))})
tc = pd.DataFrame(rows)
tc.to_csv(OUT / "temporal_consistency.csv", index=False, float_format="%.4f")
log("temporal consistency\n" + tc.pivot(index="L", columns="descriptor", values=["R@1", "AP", "R@100P"]).round(3).to_string())
summary["temporal_consistency"] = tc.to_dict(orient="records")

# ================================================================ 3-5. geometric verification
log("geometric verification ...")
centers = su.patch_centers(h, w, stride)
gray = lru_cache(maxsize=None)(lambda i: cv2.cvtColor(seq.rgb(frames[i]), cv2.COLOR_RGB2GRAY))
depth = lru_cache(maxsize=64)(lambda i: seq.depth(frames[i]))


def dino_feat(i):
    return {"grid": su.l2n(PAT[i].astype(np.float32)).astype(np.float16), "h": h, "w": w, "stride": stride,
            "centers": centers, "gray": gray(i)}


class _NoExt:
    scale = 1.0


matchers = {
    f"{TAG} coarse": (su.DinoMatcher(_NoExt(), None, min_sim=0.3), 8.0),
    f"{TAG} softargmax": (su.DinoMatcher(_NoExt(), "softargmax", min_sim=0.3), 4.0),
    f"{TAG} + LK": (su.DinoMatcher(_NoExt(), "lk", min_sim=0.3), 3.0),
}
orb = su.OrbMatcher(2000, 0.8)
orb_feat = lru_cache(maxsize=None)(lambda i: orb.features(seq.rgb(frames[i])))
dfeat = lru_cache(maxsize=128)(dino_feat)


def verify(q, j, name):
    """PnP between db frame j (3D from its depth) and query q. Returns (#inliers, T_qj or None)."""
    if name == "ORB":
        m, thr = orb.match(orb_feat(j), orb_feat(q)), 3.0
    else:
        matcher, thr = matchers[name]
        m = matcher.match(dfeat(j), dfeat(q))
    r = su.estimate_pose_pnp(m["pts_a"], m["pts_b"], depth(j), seq.K, seq.dist, reproj_thr=thr, iters=500)
    return int(r["inliers"].sum()), r["T_ba"]


def rerank(S_app, name, k=K_RR):
    """Verify the top-k appearance candidates; return score matrix (#inliers) and pose per pair."""
    S0 = np.where(cand, S_app, -np.inf)
    S_rr = np.full_like(S0, -np.inf)
    T = {}
    for q in tqdm(np.nonzero(cand.any(1))[0], desc=f"verify {name}", mininterval=15):
        for j in np.argsort(-S0[q])[:k]:
            if not np.isfinite(S0[q, j]):
                continue
            n, Tq = verify(q, j, name)
            S_rr[q, j] = n + 1e-3 * S0[q, j]
            T[(q, j)] = Tq
    return S_rr, T


f_v = CACHE / "verification.npz"
S_best = desc[best] @ desc[best].T
S_orb = desc["ORB BoW"] @ desc["ORB BoW"].T
pipelines = {  # name -> (appearance similarity, verifier)
    f"{best} + {TAG} verif.": (S_best, f"{TAG} coarse"),
    "ORB BoW + ORB verif.": (S_orb, "ORB"),
    f"{best} + ORB verif.": (S_best, "ORB"),
}
ver = {}
t = time.time()
for pname, (S_app, vname) in pipelines.items():
    S_rr, T = rerank(S_app, vname)
    ver[pname] = (S_rr, T)
log(f"verification done in {time.time() - t:.0f}s")

for pname, (S_rr, _) in ver.items():
    results[pname] = su.evaluate_retrieval(S_rr, cand, pos)
tab2 = pd.DataFrame({n: summarise(r) for n, r in results.items()}).T
tab2.to_csv(OUT / "retrieval_with_verification.csv", float_format="%.4f")
log("with verification\n" + tab2.round(3).to_string())
summary["with_verification"] = tab2.to_dict(orient="index")


# ---- loop acceptance by inlier threshold
def acceptance(S_rr):
    Sv = np.where(cand, S_rr, -np.inf)
    top1 = Sv.argmax(1)
    q = np.nonzero(cand.any(1))[0]
    n_in = np.floor(Sv[q, top1[q]]).clip(0)
    ok = pos[q, top1[q]]
    rows = []
    for tau in [10, 15, 20, 25, 30, 40, 50, 75, 100]:
        acc = n_in >= tau
        tp = int((acc & ok).sum())
        rows.append({"tau": tau, "accepted": int(acc.sum()), "TP": tp, "FP": int((acc & ~ok).sum()),
                     "precision": tp / max(1, acc.sum()), "recall": tp / max(1, has_loop.sum())})
    fp_scores = n_in[~ok]
    return pd.DataFrame(rows), (float(fp_scores.max()) if len(fp_scores) else 0.0), q, top1, n_in, ok


acc_rows = []
summary["acceptance"] = {}
for pname, (S_rr, _) in ver.items():
    df, max_fp, *_ = acceptance(S_rr)
    df.insert(0, "pipeline", pname)
    acc_rows.append(df)
    summary["acceptance"][pname] = {"max_inliers_of_a_false_top1": max_fp, "table": df.drop(columns="pipeline").to_dict(orient="records")}
acc_df = pd.concat(acc_rows)
acc_df.to_csv(OUT / "acceptance_by_inliers.csv", index=False, float_format="%.4f")
log("acceptance\n" + acc_df.round(3).to_string(index=False))

# ---- loop-constraint accuracy: relative pose of the verified true loops
log("relative-pose accuracy ...")
S_rr_best, _ = ver[f"{best} + {TAG} verif."]
_, _, qs, top1, n_in, ok = acceptance(S_rr_best)
pairs = [(q, top1[q]) for q, o in zip(qs, ok) if o]
rows = []
for q, j in tqdm(pairs, desc="pose accuracy", mininterval=15):
    T_gt = seq.rel_pose(frames[j], frames[q])
    for name in [*matchers, "ORB"]:
        n, T = verify(q, j, name)
        er, et = su.pose_error(T, T_gt)
        rows.append({"q": q, "j": j, "method": name, "inliers": n, "rot_deg": er, "trans_cm": 100 * et,
                     "gt_dist_m": float(np.linalg.norm(T_gt[:3, 3])),
                     "gt_rot_deg": float(su.rotation_angle_deg(T_gt[:3, :3]))})
pe = pd.DataFrame(rows)
pe.to_csv(OUT / "loop_pose_errors.csv", index=False, float_format="%.4f")


def pe_stats(g):
    ok_ = g["rot_deg"].notna()
    return pd.Series({"pairs": len(g), "solved": ok_.mean(), "median_rot_deg": g["rot_deg"].median(),
                      "median_trans_cm": g["trans_cm"].median(), "p90_trans_cm": g["trans_cm"].quantile(0.9),
                      "within_5cm_5deg": ((g["trans_cm"] < 5) & (g["rot_deg"] < 5)).sum() / len(g),
                      "within_10cm_10deg": ((g["trans_cm"] < 10) & (g["rot_deg"] < 10)).sum() / len(g),
                      "median_inliers": g["inliers"].median()})


pe_tab = pe.groupby("method").apply(pe_stats, include_groups=False).loc[[*matchers, "ORB"]]
pe_tab.to_csv(OUT / "loop_pose_accuracy.csv", float_format="%.4f")
log(f"relative pose of {len(pairs)} true loops (gt baseline median "
    f"{pe.drop_duplicates(['q', 'j'])['gt_dist_m'].median():.2f} m, "
    f"{pe.drop_duplicates(['q', 'j'])['gt_rot_deg'].median():.1f} deg)\n" + pe_tab.round(3).to_string())
summary["pose_accuracy"] = pe_tab.to_dict(orient="index")
summary["pose_accuracy_gt_baseline"] = {
    "median_dist_m": float(pe.drop_duplicates(["q", "j"])["gt_dist_m"].median()),
    "median_rot_deg": float(pe.drop_duplicates(["q", "j"])["gt_rot_deg"].median())}

# ================================================================ figures
log("figures ...")
ln = lambda n: "--" if "ORB BoW" == n or n.startswith("ORB BoW") else "-"

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=ts - ts[0], s=6, cmap="viridis")
ax[0].scatter(P[has_loop, 0], P[has_loop, 1], facecolors="none", edgecolors="r", s=30, lw=0.6, label="query with a true loop")
ax[0].set_aspect("equal"); ax[0].set_title("Ground-truth trajectory (top view)"); ax[0].legend()
plt.colorbar(sc, ax=ax[0], label="time [s]")
ax[1].imshow(pos, cmap="gray_r", interpolation="nearest")
ax[1].set_title("True loops: query (row) vs past keyframe (col)"); ax[1].set_xlabel("database"); ax[1].set_ylabel("query")
plt.tight_layout(); plt.savefig(OUT / "fig1_ground_truth.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for n in desc:
    p, rc = results[n]["curve"]
    ax[0].plot(rc, p, ln(n), label=f"{n} (AP={results[n]['AP']:.2f})")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_ylim(0, 1.02)
ax[0].set_title("Top-1 loop proposals (appearance only)"); ax[0].legend(fontsize=8)
ks = np.arange(1, 26)
for n, D in desc.items():
    r = su.evaluate_retrieval(D @ D.T, cand, pos, ks=tuple(ks))
    ax[1].plot(ks, [r[f"R@{k}"] for k in ks], ln(n), label=n)
ax[1].set_xlabel("N"); ax[1].set_ylabel("Recall@N"); ax[1].set_title("Recall@N"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig2_retrieval_pr.png", dpi=110); plt.close()

names = [f"{TAG} cls", best, f"{TAG} vlad", "ORB BoW"]
names = list(dict.fromkeys(names))
fig, ax = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.4))
for k, n in enumerate(names):
    S = desc[n] @ desc[n].T
    ax[0, k].imshow(S, cmap="magma", vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    ax[0, k].set_title(n); ax[0, k].set_xticks([]); ax[0, k].set_yticks([])
    r = results[n]; okk = np.isfinite(r["top1_score"])
    ax[1, k].hist(r["top1_score"][okk & r["top1_correct"]], 30, alpha=0.6, label="correct top-1", density=True)
    ax[1, k].hist(r["top1_score"][okk & ~r["top1_correct"]], 30, alpha=0.6, label="wrong top-1", density=True)
    ax[1, k].set_xlabel("top-1 cosine similarity")
ax[1, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig3_similarity_and_scores.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for n in [best, "ORB BoW", *ver]:
    p, rc = results[n]["curve"]
    ax[0].plot(rc, p, ln(n), label=f"{n} (AP={results[n]['AP']:.2f})")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)
ax[0].set_title("Effect of geometric verification (top-1, score = #inliers)")
for pname in ver:
    d = acc_df[acc_df.pipeline == pname]
    ax[1].plot(d["recall"], d["precision"], "o-", label=pname)
    for _, rr in d.iterrows():
        if rr["tau"] in (15, 30, 50):
            ax[1].annotate(f"τ={int(rr['tau'])}", (rr["recall"], rr["precision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Accept loop if #inliers ≥ τ"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig4_verification.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for m in [*matchers, "ORB"]:
    g = pe[pe.method == m]
    for a, col, lim in [(ax[0], "trans_cm", 50), (ax[1], "rot_deg", 20)]:
        v = np.sort(g[col].fillna(np.inf).values)
        a.plot(v, np.arange(1, len(v) + 1) / len(v), label=m)
        a.set_xlim(0, lim)
ax[0].set_xlabel("translation error [cm]"); ax[1].set_xlabel("rotation error [deg]")
for a in ax:
    a.set_ylabel("fraction of true loops"); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle(f"Relative-pose accuracy of the {len(pairs)} verified true loops (CDF; unsolved = ∞)")
plt.tight_layout(); plt.savefig(OUT / "fig5_loop_pose_accuracy.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for metric, a in zip(["AP", "R@100P"], ax):
    pv = abl.pivot_table(index=["layer", "scale"], columns="desc", values=metric)
    im = a.imshow(pv.values, cmap="Greens", vmin=0, vmax=max(1e-3, pv.values.max()))
    a.set_xticks(range(len(pv.columns)), pv.columns); a.set_yticks(range(len(pv.index)), [f"L{l} ×{s:g}" for l, s in pv.index])
    for (i, j), v in np.ndenumerate(pv.values):
        a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    a.set_title(f"{metric}: layer × input scale")
plt.tight_layout(); plt.savefig(OUT / "fig6_layer_scale_ablation.png", dpi=110); plt.close()

q_ids = np.nonzero(has_loop)[0]
q_ids = q_ids[np.linspace(0, len(q_ids) - 1, 4).astype(int)]
cols = [best, "ORB BoW", f"{best} + {TAG} verif."]
fig, ax = plt.subplots(len(q_ids), 1 + len(cols), figsize=(4 * (1 + len(cols)), 3.2 * len(q_ids)))
for row, q in enumerate(q_ids):
    ax[row, 0].imshow(seq.rgb(frames[q])); ax[row, 0].set_title(f"query t={ts[q] - ts[0]:.0f}s")
    for col, n in enumerate(cols, 1):
        j = results[n]["top1"][q]
        ax[row, col].imshow(seq.rgb(frames[j]))
        ax[row, col].set_title(f"{n}\nt={ts[j] - ts[0]:.0f}s, {np.linalg.norm(P[q] - P[j]):.2f} m",
                               color="g" if pos[q, j] else "r", fontsize=8)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "fig7_examples.png", dpi=90); plt.close()

json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
log("done ->", OUT)
