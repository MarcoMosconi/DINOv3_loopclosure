"""Retrieval quality of every transformer block, for DINOv3 and DINOv2 (same protocol as
loop_closure_experiments.py, appearance-only).

block 9 was chosen for DINOv3 from {6, 9, 12}; this checks all 12 blocks for both backbones so the
comparison does not favour whichever model the layer was tuned on. Only pooled descriptors
(CLS, mean, GeM) are kept, so no patch tokens are cached.

Usage:  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python loop_closure_layer_sweep.py [dinov3 dinov2 ...]
Output: results/loop_closure_compare/layer_sweep.csv (+ per-backbone descriptor cache)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import slam_utils as su

SEQ, STRIDE, EXCLUDE_S, D_THR, ANG_THR = "fr3_long_office_household", 5, 20.0, 0.5, 30.0
OUT = Path(__file__).resolve().parent / "results" / "loop_closure_compare"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)

seq = su.TUMSequence(SEQ, su.DATA_ROOT, download=False)
frames = np.arange(0, len(seq), STRIDE)
poses, ts = seq.poses[frames], seq.timestamps[frames]
cand, pos = su.loop_ground_truth(poses, ts, D_THR, ANG_THR, EXCLUDE_S, causal=True)


def extract_all_blocks(key, bs=8):
    f = OUT / f"layer_descs_{key}.npz"
    if f.exists():
        return dict(np.load(f))
    d = su.DinoV3(su.LC_BACKBONES[key]["model"])
    L = d.n_layers
    acc = {}
    t = time.time()
    for i in tqdm(range(0, len(frames), bs), desc=f"{key} all blocks", mininterval=15):
        x, h, w, _ = d._prep([seq.rgb(j) for j in frames[i : i + bs]], 1.0)
        with torch.inference_mode():
            final, inter = d.model.forward_intermediates(
                x.to(d.device), indices=list(range(L)), output_fmt="NLC", return_prefix_tokens=True, norm=False)
        toks = {l + 1: torch.cat([pre, pat], 1) for l, (pat, pre) in enumerate(inter)} | {"final": final}
        for l, tk in toks.items():
            tk = tk.float().cpu().numpy()
            for c, p in zip(tk[:, 0], tk[:, 1 + d.n_reg :]):
                for name, v in su.pool_descriptors(c, p).items():
                    acc.setdefault(f"{l}|{name}", []).append(v)
    print(f"{key}: {1000 * (time.time() - t) / len(frames):.0f} ms/frame (all blocks, one pass)")
    res = {k: np.stack(v) for k, v in acc.items()}
    np.savez(f, **res)
    return res


rows = []
for key in sys.argv[1:] or ["dinov3", "dinov2"]:
    D = extract_all_blocks(key)
    for k, v in D.items():
        layer, name = k.split("|")
        r = su.evaluate_retrieval(v @ v.T, cand, pos)
        rows.append({"backbone": key, "layer": layer, "desc": name,
                     **{m: float(r[m]) for m in ["R@1", "R@5", "AP", "R@100P"]}})
df = pd.DataFrame(rows)
f = OUT / "layer_sweep.csv"
if f.exists():  # merge with earlier runs of other backbones
    old = pd.read_csv(f, dtype={"layer": str})
    df = pd.concat([old[~old.backbone.isin(df.backbone.unique())], df])
df.to_csv(f, index=False, float_format="%.4f")
for m in ["AP", "R@100P"]:
    print(m)
    print(df[df.desc != "cls+mean"].pivot_table(index="layer", columns=["backbone", "desc"], values=m,
                                                 sort=False).round(3).to_string())
