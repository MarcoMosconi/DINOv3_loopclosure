"""Side-by-side comparison of the DINOv3 and DINOv2 loop-closure results.

Run after both backbones have been through loop_closure_experiments.py, loop_closure_followup.py
and loop_closure_final_pipeline.py (and optionally loop_closure_layer_sweep.py).
Reads results/loop_closure/ (DINOv3) and results/loop_closure_dinov2/ (DINOv2); writes tables and
figures to results/loop_closure_compare/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slam_utils as su

ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "loop_closure_compare"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"DINOv3": ROOT / su.LC_BACKBONES["dinov3"]["out"], "DINOv2": ROOT / su.LC_BACKBONES["dinov2"]["out"]}
COL = {"DINOv3": "tab:blue", "DINOv2": "tab:orange", "ORB": "tab:gray"}
pd.set_option("display.width", 200)
md = []  # markdown dump of every table


def to_md(df, fmt):
    df = df.reset_index()
    cell = lambda v: (fmt % v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    head = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def show(title, df, fmt="%.3f"):
    print(f"\n### {title}\n" + df.to_string(float_format=lambda v: fmt % v))
    md.append(f"### {title}\n\n" + to_md(df, fmt) + "\n")
    df.to_csv(OUT / (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv"), float_format="%.4f")


def strip(name, tag):  # "DINOv3 gem" -> "gem"; "DINOv2 mean + DINOv2 verif." -> "mean + verif."
    return name.replace(f"{tag} ", "").replace(f" {tag} ", " ").strip()


# ---------------------------------------------------------------- 1. appearance-only retrieval
r = {t: pd.read_csv(d / "retrieval.csv", index_col=0) for t, d in MODELS.items()}
rows = []
for t, df in r.items():
    for n, row in df.iterrows():
        if n.startswith("ORB") and t != "DINOv3":
            continue
        rows.append({"model": "ORB" if n.startswith("ORB") else t, "descriptor": strip(n, t), **row[["R@1", "R@5", "AP", "R@100P"]]})
ret = pd.DataFrame(rows).set_index(["descriptor", "model"]).sort_index()
show("Retrieval, final layer, appearance only", ret)

# ---------------------------------------------------------------- 2. layer x scale ablation (+ full sweep)
ab = pd.concat([pd.read_csv(d / "ablation_layer_scale.csv").assign(model=t) for t, d in MODELS.items()])
ab = ab[ab.desc.isin(["mean", "gem", "vlad", "cls"])]
show("Layer x scale ablation, AP", ab.pivot_table(index=["layer", "scale"], columns=["desc", "model"], values="AP"))
show("Layer x scale ablation, R@100P", ab.pivot_table(index=["layer", "scale"], columns=["desc", "model"], values="R@100P"))
best_cfg = ab.sort_values("AP", ascending=False).groupby("model").head(1).set_index("model")
show("Best retrieval configuration per model", best_cfg[["layer", "scale", "desc", "R@1", "R@5", "AP", "R@100P"]])

sweep_f = OUT / "layer_sweep.csv"
sweep = pd.read_csv(sweep_f, dtype={"layer": str}) if sweep_f.exists() else None
if sweep is not None:
    sweep["model"] = sweep.backbone.map({"dinov3": "DINOv3", "dinov2": "DINOv2", "dinov2_noreg": "DINOv2 (no reg)"})
    s = sweep[sweep.desc.isin(["gem", "mean", "cls"])]
    show("All-block sweep, AP", s.pivot_table(index="layer", columns=["desc", "model"], values="AP", sort=False))
    show("All-block sweep, R@100P", s.pivot_table(index="layer", columns=["desc", "model"], values="R@100P", sort=False))

# ---------------------------------------------------------------- 3. with geometric verification
rows = []
for t, d in MODELS.items():
    for n, row in pd.read_csv(d / "followup_retrieval.csv", index_col=0).iterrows():
        if n.startswith("ORB BoW") and t != "DINOv3":
            continue
        model = "ORB" if n.startswith("ORB BoW") else t
        label = n if model == "ORB" else n.replace(f"+ {t} verif.", "+ patch verif.")
        rows.append({"pipeline": label, "model": model,
                     **row[["R@1", "R@5", "AP", "R@100P"]]})
ver = pd.DataFrame(rows).set_index(["pipeline", "model"]).sort_index()
show("Retrieval + top-5 geometric verification", ver)

# ---------------------------------------------------------------- 4. loop-constraint accuracy
pe = pd.concat([pd.read_csv(d / "loop_pose_accuracy.csv", index_col=0).assign(model=t) for t, d in MODELS.items()])
pe.index = [i.replace(f"{m} ", "patch ").replace("patch + LK", "patch + LK") if i != "ORB" else "ORB" for i, m in zip(pe.index, pe.model)]
pe = pe.reset_index(names="matcher").set_index(["matcher", "model"]).sort_index()
show("Relative-pose accuracy of verified true loops",
     pe[["pairs", "median_trans_cm", "p90_trans_cm", "median_rot_deg", "within_5cm_5deg", "median_inliers"]])

# ---------------------------------------------------------------- 5. metric acceptance (final pipeline)
fp = pd.concat([pd.read_csv(d / "final_pipeline_acceptance.csv").assign(model=t) for t, d in MODELS.items()])
show("Final pipeline: accepted loops by class (inlier ratio)",
     fp[fp.criterion == "inlier_ratio"].set_index(["tau", "model"]).drop(columns="criterion").sort_index(), fmt="%.3g")
show("Final pipeline: accepted loops by class (#inliers)",
     fp[fp.criterion == "inliers"].set_index(["tau", "model"]).drop(columns="criterion").sort_index(), fmt="%.3g")


def zero_bad(df):
    """Loosest ratio threshold (on the grid) with no 'right place, bad pose' or 'wrong place' accepted."""
    g = df[(df["right place, bad pose"] == 0) & (df["wrong place"] == 0)].sort_values("tau")
    return g.iloc[0] if len(g) else None


rows = []
for t, d in MODELS.items():
    f = pd.read_csv(d / "final_pipeline_top1.csv")
    bad = f[f.final_class.isin(["wrong place", "right place, bad pose"])]
    z = zero_bad(fp[(fp.model == t) & (fp.criterion == "inlier_ratio")])
    rows.append({"model": t,
                 "loosest safe ratio": z["tau"] if z is not None else np.nan,
                 "true loops kept": z["true loop"] if z is not None else np.nan,
                 "extra valid overlaps kept": z["valid overlap"] if z is not None else np.nan,
                 "max ratio of a wrong place": f[f.final_class == "wrong place"].inlier_ratio.max(),
                 "max ratio of a bad pose": f[f.final_class == "right place, bad pose"].inlier_ratio.max(),
                 "min ratio of a true loop": f[f.final_class == "true loop"].inlier_ratio.min(),
                 "median ratio of a true loop": f[f.final_class == "true loop"].inlier_ratio.median(),
                 "#bad top-1 (>=15 inl.)": len(bad)})
show("Final pipeline (LK pose): separation by inlier ratio", pd.DataFrame(rows).set_index("model"))
rows = []
for t, d in MODELS.items():
    for s_ in json.load(open(d / "followup_summary.json"))["safe_thresholds"]:
        if s_["pipeline"].startswith("ORB BoW") and t != "DINOv3":
            continue
        rows.append({"pipeline": s_["pipeline"].replace(f"+ {t} verif.", "+ patch verif."),
                     "model": "ORB" if s_["pipeline"].startswith("ORB BoW") else t,
                     **{k: s_[k] for k in ["min_inliers_zero_harmful", "true_loops_kept", "valid_overlaps_kept"]}})
show("Follow-up (pose from verifier): min #inliers with zero harmful loops", pd.DataFrame(rows).set_index(["pipeline", "model"]).sort_index(), fmt="%.0f")

# ---------------------------------------------------------------- 6. compute
rows = []
for t, d in MODELS.items():
    log = (d / "log_experiments.txt").read_text()
    for tag, ms in re.findall(r"(main_s[\d.]+): (\d+) ms/frame", log):
        rows.append({"model": t, "input": tag.replace("main_s", "scale "), "ms/frame (CPU)": int(ms)})
sl = OUT / "log_layer_sweep.txt"
if sl.exists():
    for key, ms in re.findall(r"^(\w+): (\d+) ms/frame", sl.read_text(), re.M):
        rows.append({"model": {"dinov3": "DINOv3", "dinov2": "DINOv2"}.get(key, key), "input": "scale 1.0, all 12 blocks", "ms/frame (CPU)": int(ms)})
cfg = {t: json.load(open(d / "summary.json"))["config"] for t, d in MODELS.items()}
if rows:
    show("Feature extraction time (multi-layer pass, 8 CPU threads)", pd.DataFrame(rows).set_index(["input", "model"]), fmt="%.0f")

# ================================================================ figures
# (a) retrieval bars
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
descs = ["cls", "mean", "gem", "vlad"]
for a, m in zip(ax, ["AP", "R@100P"]):
    x = np.arange(len(descs))
    for k, t in enumerate(MODELS):
        a.bar(x + (k - 0.5) * 0.38, [ret.loc[(d, t), m] for d in descs], 0.38, label=t, color=COL[t])
    a.axhline(ret.loc[("ORB BoW", "ORB"), m], color="k", ls="--", lw=1, label="ORB BoW")
    a.set_xticks(x, descs); a.set_title(f"{m}, final layer, appearance only"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "cmp1_retrieval.png", dpi=110); plt.close()

# (b) layer sweep
if sweep is not None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, m in zip(ax, ["AP", "R@100P"]):
        for t in s.model.unique():
            for d_, ls in [("gem", "-"), ("mean", "--"), ("cls", ":")]:
                g = s[(s.model == t) & (s.desc == d_)]
                g = g[g.layer != "final"]
                a.plot(g.layer.astype(int), g[m], ls, color=COL.get(t, "tab:green"), label=f"{t} {d_}", marker="o", ms=3)
        a.axhline(ret.loc[("ORB BoW", "ORB"), m], color="k", ls="-.", lw=1, label="ORB BoW")
        a.set_xlabel("transformer block"); a.set_ylabel(m); a.set_xticks(range(1, 13)); a.grid(alpha=0.3)
        a.set_title(f"Retrieval {m} per block (un-normed tokens)")
    ax[0].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(OUT / "cmp2_layer_sweep.png", dpi=110); plt.close()

# (c) pose-accuracy CDFs
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for t, d in MODELS.items():
    e = pd.read_csv(d / "loop_pose_errors.csv")
    for meth, ls in [(f"{t} coarse", ":"), (f"{t} + LK", "-")]:
        g = e[e.method == meth]
        for a, c, lim in [(ax[0], "trans_cm", 20), (ax[1], "rot_deg", 6)]:
            v = np.sort(g[c].fillna(np.inf).values)
            a.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=COL[t], label=f"{meth} (n={len(g)})")
            a.set_xlim(0, lim)
    if t == "DINOv3":
        g = e[e.method == "ORB"]
        for a, c in [(ax[0], "trans_cm"), (ax[1], "rot_deg")]:
            v = np.sort(g[c].fillna(np.inf).values)
            a.plot(v, np.arange(1, len(v) + 1) / len(v), "-", color="k", lw=1, label="ORB (DINOv3 pairs)")
ax[0].set_xlabel("translation error [cm]"); ax[1].set_xlabel("rotation error [deg]")
for a in ax:
    a.set_ylabel("fraction of verified true loops"); a.grid(alpha=0.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "cmp3_pose_accuracy.png", dpi=110); plt.close()

# (d) final pipeline: inlier ratio by class
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
cc = {"true loop": "tab:green", "valid overlap": "tab:blue", "right place, bad pose": "tab:orange", "wrong place": "tab:red"}
for a, (t, d) in zip(ax, MODELS.items()):
    f = pd.read_csv(d / "final_pipeline_top1.csv")
    for c, g in f.groupby("final_class"):
        a.scatter(g.gt_dist_m, g.inlier_ratio, s=12, c=cc[c], label=f"{c} ({len(g)})")
    a.axhline(0.25, color="k", lw=0.8, ls="--")
    a.set_title(f"{t}: top-1 after verification (≥15 inliers)"); a.set_xlabel("GT distance query ↔ match [m]")
    a.legend(fontsize=7); a.grid(alpha=0.3)
ax[0].set_ylabel("PnP inlier ratio")
plt.tight_layout(); plt.savefig(OUT / "cmp4_acceptance.png", dpi=110); plt.close()

(OUT / "tables.md").write_text("\n".join(md))
json.dump(cfg, open(OUT / "configs.json", "w"), indent=1)
print("\ndone ->", OUT)
