"""Shared utilities for the DINOv3-for-SLAM experiments on TUM RGB-D.

Used by the three notebooks:
    01_loop_closure.ipynb   place recognition / loop closure
    02_tracking.ipynb       patch matching + depth + PnP visual odometry
    03_robustness.ipynb     photometric changes, wide baselines, moving people

Conventions
-----------
* Images are HxWx3 uint8 RGB numpy arrays; depth is HxW float32 in metres (0 = invalid).
* Poses are 4x4 numpy arrays. ``T_wc`` maps camera coordinates to world coordinates
  (this is what TUM ground truth stores). ``T_ba`` maps points from camera A to camera B.
* Pixel coordinates are (x, y) with the centre of the top-left pixel at (0, 0).
"""
from __future__ import annotations

import tarfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# --------------------------------------------------------------------------------------
# TUM RGB-D dataset
# --------------------------------------------------------------------------------------
TUM_BASE = "https://cvg.cit.tum.de/rgbd/dataset"
SEQUENCES = {
    "fr1_desk": ("freiburg1", "rgbd_dataset_freiburg1_desk"),
    "fr1_room": ("freiburg1", "rgbd_dataset_freiburg1_room"),
    "fr3_long_office_household": ("freiburg3", "rgbd_dataset_freiburg3_long_office_household"),
    "fr3_walking_xyz": ("freiburg3", "rgbd_dataset_freiburg3_walking_xyz"),
    "fr3_sitting_xyz": ("freiburg3", "rgbd_dataset_freiburg3_sitting_xyz"),
}
# fx, fy, cx, cy, distortion (k1, k2, p1, p2, k3) from the TUM website.
# freiburg3 images are already undistorted.
INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3, (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "freiburg2": (520.9, 521.0, 325.1, 249.7, (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, (0.0, 0.0, 0.0, 0.0, 0.0)),
}
DEPTH_SCALE = 5000.0
DATA_ROOT = Path(__file__).resolve().parent / "data"


def download_tum(name: str, root: str | Path = DATA_ROOT) -> Path:
    """Download and extract a TUM RGB-D sequence if it is not already present."""
    cam, folder = SEQUENCES[name]
    root = Path(root)
    out = root / folder
    if (out / "groundtruth.txt").exists():
        return out
    root.mkdir(parents=True, exist_ok=True)
    url = f"{TUM_BASE}/{cam}/{folder}.tgz"
    tgz = root / f"{folder}.tgz"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(root, filter="data")
    tgz.unlink()
    return out


def _read_list(path: Path):
    stamps, rows = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        stamps.append(float(parts[0]))
        rows.append(parts[1:])
    return np.array(stamps), rows


def pose_from_tq(t, q_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


class TUMSequence:
    """RGB, depth and interpolated ground-truth pose for every RGB frame.

    Only RGB frames that have a depth frame within ``max_dt`` seconds and that lie inside
    the ground-truth time range are kept.
    """

    def __init__(self, name: str, root: str | Path = DATA_ROOT, max_dt: float = 0.02, download: bool = True):
        self.name = name
        cam, folder = SEQUENCES[name]
        self.path = download_tum(name, root) if download else Path(root) / folder
        fx, fy, cx, cy, dist = INTRINSICS[cam]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)

        t_rgb, r_rgb = _read_list(self.path / "rgb.txt")
        t_dep, r_dep = _read_list(self.path / "depth.txt")
        t_gt, r_gt = _read_list(self.path / "groundtruth.txt")
        gt = np.array(r_gt, dtype=np.float64)  # tx ty tz qx qy qz qw

        # nearest depth frame for each rgb frame
        j = np.clip(np.searchsorted(t_dep, t_rgb), 1, len(t_dep) - 1)
        j = np.where(np.abs(t_dep[j - 1] - t_rgb) < np.abs(t_dep[j] - t_rgb), j - 1, j)
        ok = (np.abs(t_dep[j] - t_rgb) < max_dt) & (t_rgb > t_gt[0]) & (t_rgb < t_gt[-1])
        idx = np.nonzero(ok)[0]

        # interpolate ground truth (linear translation, slerp rotation)
        slerp = Slerp(t_gt, Rotation.from_quat(gt[:, 3:]))
        ts = t_rgb[idx]
        R = slerp(ts).as_matrix()
        tr = np.stack([np.interp(ts, t_gt, gt[:, k]) for k in range(3)], 1)
        self.poses = np.tile(np.eye(4), (len(idx), 1, 1))
        self.poses[:, :3, :3] = R
        self.poses[:, :3, 3] = tr

        self.timestamps = ts
        self.rgb_files = [self.path / r_rgb[i][0] for i in idx]
        self.depth_files = [self.path / r_dep[j[i]][0] for i in idx]

    def __len__(self):
        return len(self.timestamps)

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i]

    def rel_pose(self, a: int, b: int) -> np.ndarray:
        """Ground-truth T_ba (camera A coordinates -> camera B coordinates)."""
        return np.linalg.inv(self.poses[b]) @ self.poses[a]

    @property
    def positions(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    def __repr__(self):
        dur = self.timestamps[-1] - self.timestamps[0]
        length = np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum()
        return f"TUMSequence({self.name}: {len(self)} frames, {dur:.1f}s, path length {length:.2f} m)"


# --------------------------------------------------------------------------------------
# DINOv3 backbone
# --------------------------------------------------------------------------------------
DINOV3_MODELS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
# Same LVD-1689M checkpoints re-hosted (ungated) by timm. Used with model="timm:vits16", or as an
# automatic fallback when the gated Hugging Face checkpoint cannot be loaded (no/invalid token).
DINOV3_TIMM = {
    "vits16": "vit_small_patch16_dinov3.lvd1689m",
    "vits16plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "vitl16": "vit_large_patch16_dinov3.lvd1689m",
}
# DINOv2 (LVD-142M) via timm, ungated. Use as DinoV3("timm:dinov2_vits14_reg"); the wrapper is
# architecture-agnostic (patch size and register count are read from the model).
DINOV2_TIMM = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
}
# Backbones the loop-closure scripts can run with (select with env LC_BACKBONE).
#   model: argument to DinoV3(); label: name used in tables; out: results sub-folder
LC_BACKBONES = {
    "dinov3": {"model": "vits16", "label": "DINOv3", "out": "loop_closure"},
    "dinov2": {"model": "timm:dinov2_vits14_reg", "label": "DINOv2", "out": "loop_closure_dinov2"},
    "dinov2_noreg": {"model": "timm:dinov2_vits14", "label": "DINOv2", "out": "loop_closure_dinov2_noreg"},
}


def lc_backbone():
    """Backbone config for the loop-closure scripts, from env LC_BACKBONE (default dinov3)."""
    import os

    key = os.environ.get("LC_BACKBONE", "dinov3")
    return key, LC_BACKBONES[key]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DinoV3:
    """Thin wrapper around a Hugging Face DINOv3 ViT.

    The HF checkpoints are gated: accept the licence on the model page and run
    ``huggingface-cli login`` once.
    """

    def __init__(self, model: str = "vits16", device: str | None = None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or pick_device()
        self.backend = "hf"
        if model.startswith("timm:"):
            self._init_timm(model[5:])
        else:
            name = DINOV3_MODELS.get(model, model)
            try:
                self.model = AutoModel.from_pretrained(name).to(self.device).eval()
            except OSError as e:  # gated repo without a valid token
                if model not in DINOV3_TIMM:
                    raise
                print(f"[DinoV3] could not load {name} ({type(e).__name__}); using the timm copy of the same weights")
                self._init_timm(model)
            else:
                self.name = name
                cfg = self.model.config
                self.patch = cfg.patch_size
                self.n_reg = getattr(cfg, "num_register_tokens", 0)
                self.dim = cfg.hidden_size
                self.n_layers = cfg.num_hidden_layers
        self.use_amp = self.device == "cuda"

    def _init_timm(self, model):
        import timm

        name = DINOV3_TIMM.get(model) or DINOV2_TIMM.get(model, model)
        self.backend, self.name = "timm", "timm/" + name
        self.model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.model = self.model.to(self.device).eval()
        self.patch = self.model.patch_embed.patch_size[0]
        self.n_reg = self.model.num_prefix_tokens - 1
        self.dim = self.model.embed_dim
        self.n_layers = len(self.model.blocks)

    def _prep(self, imgs, scale):
        H, W = imgs[0].shape[:2]
        h = max(1, round(H * scale / self.patch))
        w = max(1, round(W * scale / self.patch))
        Hr, Wr = h * self.patch, w * self.patch
        batch = []
        for im in imgs:
            if (Hr, Wr) != (H, W):
                im = cv2.resize(im, (Wr, Hr), interpolation=cv2.INTER_AREA if Hr < H else cv2.INTER_LINEAR)
            batch.append((im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
        x = self.torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).contiguous()
        return x, h, w, (W / w, H / h)

    def __call__(self, imgs, scale: float = 1.0, layer: int | None = None):
        """Run the backbone.

        Args:
            imgs: one image or a list of same-size images (HxWx3 uint8 RGB).
            scale: resize factor before patchifying. 1.0 on 640x480 gives a 40x30 grid
                (16 px patches); 2.0 gives 80x60 (8 px in the original image).
            layer: transformer block to read tokens from (1..n_layers); None = final normed output.

        Returns:
            cls      [B, D] float32
            patches  [B, h, w, D] float32
            stride   (sx, sy): original-image pixels per patch
        """
        single = isinstance(imgs, np.ndarray)
        if single:
            imgs = [imgs]
        x, h, w, stride = self._prep(imgs, scale)
        x = x.to(self.device)
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16, enabled=self.use_amp
        ):
            if self.backend == "timm":
                if layer is None:
                    tok = self.model.forward_features(x)
                else:  # un-normed output of block `layer` (1-based), as HF hidden_states[layer]
                    pat, pre = self.model.forward_intermediates(
                        x, indices=[layer - 1], output_fmt="NLC", intermediates_only=True, return_prefix_tokens=True
                    )[0]
                    tok = self.torch.cat([pre, pat], 1)
            else:
                out = self.model(pixel_values=x, output_hidden_states=layer is not None)
                tok = out.last_hidden_state if layer is None else out.hidden_states[layer]
        tok = tok.float()
        cls = tok[:, 0].cpu().numpy()
        patches = tok[:, 1 + self.n_reg :].reshape(len(imgs), h, w, -1).cpu().numpy()
        if single:
            return cls[0], patches[0], stride
        return cls, patches, stride


def l2n(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def patch_centers(h: int, w: int, stride) -> np.ndarray:
    """Pixel coordinates (x, y) of every patch centre, in row-major order, shape [h*w, 2]."""
    sx, sy = stride
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx.ravel() + 0.5) * sx - 0.5, (yy.ravel() + 0.5) * sy - 0.5], 1)


def pca_rgb(patches: np.ndarray, basis=None):
    """Project an [h, w, D] patch grid onto 3 principal components for display."""
    h, w, D = patches.shape
    X = patches.reshape(-1, D)
    if basis is None:
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        P = (X - mu) @ Vt[:3].T
        lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
        basis = (mu, Vt[:3], lo, hi)
    mu, V, lo, hi = basis
    P = (X - mu) @ V.T
    return np.clip((P - lo) / (hi - lo + 1e-8), 0, 1).reshape(h, w, 3), basis


# --------------------------------------------------------------------------------------
# Global descriptors for place recognition
# --------------------------------------------------------------------------------------
def pool_descriptors(cls: np.ndarray, patches: np.ndarray, gem_p: float = 3.0) -> dict:
    """Training-free global descriptors from one image's tokens. All are L2-normalised."""
    P = patches.reshape(-1, patches.shape[-1])
    gem = (np.clip(P, 1e-6, None) ** gem_p).mean(0) ** (1.0 / gem_p)
    return {
        "cls": l2n(cls),
        "mean": l2n(P.mean(0)),
        "gem": l2n(gem),
        "cls+mean": l2n(np.concatenate([l2n(cls), l2n(P.mean(0))])),
    }


def fit_kmeans(X: np.ndarray, k: int, seed: int = 0):
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3).fit(X)


def vlad(patches: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Hard-assignment VLAD over L2-normalised patch tokens (AnyLoc-style), intra + power + L2 norm."""
    P = l2n(patches.reshape(-1, patches.shape[-1]))
    C = l2n(centers)
    a = (P @ C.T).argmax(1)
    V = np.zeros_like(C)
    np.add.at(V, a, P - C[a])
    V = l2n(V)
    V = V.ravel()
    V = np.sign(V) * np.sqrt(np.abs(V))
    return l2n(V)


class OrbBoW:
    """Flat ORB bag-of-visual-words with TF-IDF (a simple stand-in for DBoW2).

    Binary ORB descriptors are unpacked to {0,1}^256 and clustered with k-means; words are
    assigned by nearest centre. IDF is computed on the database images passed to ``fit_idf``.
    """

    def __init__(self, k: int = 1000, n_features: int = 1000):
        self.k = k
        self.orb = cv2.ORB_create(n_features)
        self.km = None
        self.idf = None

    def descriptors(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, d = self.orb.detectAndCompute(gray, None)
        if d is None:
            return np.zeros((0, 256), np.float32)
        return np.unpackbits(d, axis=1).astype(np.float32)

    def fit_vocab(self, imgs, max_desc: int = 200_000, seed: int = 0):
        D = np.concatenate([self.descriptors(im) for im in imgs])
        rng = np.random.default_rng(seed)
        if len(D) > max_desc:
            D = D[rng.choice(len(D), max_desc, replace=False)]
        self.km = fit_kmeans(D, self.k, seed)
        return self

    def tf(self, img):
        d = self.descriptors(img)
        h = np.zeros(self.k, np.float32)
        if len(d):
            np.add.at(h, self.km.predict(d), 1.0)
            h /= len(d)
        return h

    def fit_idf(self, tfs: np.ndarray):
        n = len(tfs)
        df = (tfs > 0).sum(0)
        self.idf = np.log((n + 1) / (df + 1)).astype(np.float32)
        return self

    def encode(self, tfs: np.ndarray) -> np.ndarray:
        return l2n(tfs * self.idf[None])


# --------------------------------------------------------------------------------------
# Loop-closure evaluation
# --------------------------------------------------------------------------------------
def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def loop_ground_truth(poses, timestamps, d_thr=0.5, ang_thr=30.0, exclude_s=20.0, causal=True):
    """Boolean matrices (candidate[i, j], positive[i, j]) for query i vs database frame j.

    A database frame is a *candidate* for query i if it is at least ``exclude_s`` seconds
    away (and, if causal, in the past). It is a *positive* if it is also within ``d_thr``
    metres and ``ang_thr`` degrees of the query.
    """
    t = np.asarray(timestamps)
    dt = t[:, None] - t[None, :]
    cand = dt > exclude_s if causal else np.abs(dt) > exclude_s
    P = poses[:, :3, 3]
    dist = np.linalg.norm(P[:, None] - P[None], axis=-1)
    R = poses[:, :3, :3]
    Rrel = np.einsum("iba,jbc->ijac", R, R)  # R_i^T R_j
    ang = rotation_angle_deg(Rrel)
    pos = cand & (dist < d_thr) & (ang < ang_thr)
    return cand, pos


def evaluate_retrieval(S: np.ndarray, cand: np.ndarray, pos: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Recall@k and top-1 precision-recall for a similarity matrix S[query, db]."""
    S = np.where(cand, S, -np.inf)
    has_cand = cand.any(1)
    has_pos = pos.any(1)
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = np.take_along_axis(pos, order[:, :k], 1).any(1)
        out[f"R@{k}"] = hit[has_pos].mean() if has_pos.any() else np.nan
    top1 = order[:, 0]
    score = S[np.arange(len(S)), top1]
    correct = pos[np.arange(len(S)), top1]
    q = np.nonzero(has_cand)[0]
    o = q[np.argsort(-score[q])]
    tp = np.cumsum(correct[o])
    precision = tp / np.arange(1, len(o) + 1)
    recall = tp / max(1, has_pos.sum())
    ap = np.sum(np.diff(np.concatenate([[0], recall])) * precision)
    full = precision >= 1.0 - 1e-9
    out["AP"] = ap
    out["R@100P"] = recall[full].max() if full.any() else 0.0
    out["n_queries_with_loop"] = int(has_pos.sum())
    out["curve"] = (precision, recall)
    out["top1"] = top1
    out["top1_correct"] = correct
    out["top1_score"] = score
    return out


# --------------------------------------------------------------------------------------
# Local matching
# --------------------------------------------------------------------------------------
class FeatureCache:
    """Small LRU cache so several matchers can share one DINOv3 forward pass per frame."""

    def __init__(self, max_items: int = 64):
        self.max_items = max_items
        self.d = OrderedDict()

    def get(self, key, fn):
        if key is not None and key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = fn()
        if key is not None:
            self.d[key] = v
            if len(self.d) > self.max_items:
                self.d.popitem(last=False)
        return v

    def clear(self):
        self.d.clear()


class DinoExtractor:
    """DINOv3 dense features for matching: L2-normalised patch grid + patch centres."""

    def __init__(self, dino: DinoV3, scale: float = 1.0, layer: int | None = None, cache_size: int = 64):
        self.dino, self.scale, self.layer = dino, scale, layer
        self.cache = FeatureCache(cache_size)

    def __call__(self, img, key=None):
        def run():
            _, P, stride = self.dino(img, scale=self.scale, layer=self.layer)
            h, w, _ = P.shape
            return {
                "grid": l2n(P).astype(np.float16),
                "h": h,
                "w": w,
                "stride": stride,
                "centers": patch_centers(h, w, stride),
                "gray": cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            }

        return self.cache.get(key, run)


def mutual_nn(fa: np.ndarray, fb: np.ndarray, min_sim: float = 0.0, ratio: float | None = None):
    """Mutual nearest neighbours under cosine similarity. fa [Na, D], fb [Nb, D] L2-normalised."""
    S = fa.astype(np.float32) @ fb.astype(np.float32).T
    ab = S.argmax(1)
    ba = S.argmax(0)
    ia = np.arange(len(fa))
    sim = S[ia, ab]
    keep = (ba[ab] == ia) & (sim > min_sim)
    if ratio is not None:
        part = np.partition(S, -2, axis=1)[:, -2]
        keep &= (1 - sim) < ratio * (1 - part)  # ratio on cosine distances
    return ia[keep], ab[keep], sim[keep], S


def softargmax_refine(S: np.ndarray, ia, ib, h, w, temperature: float = 0.02, radius: int = 1):
    """Sub-patch refinement: softmax-weighted offset over the (2r+1)^2 neighbourhood of each match.

    Returns refined (col, row) grid coordinates in B (floating point).
    """
    rb, cb = np.divmod(ib, w)
    offs = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    rr = rb[:, None] + dy[None]
    cc = cb[:, None] + dx[None]
    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    flat = np.clip(rr, 0, h - 1) * w + np.clip(cc, 0, w - 1)
    s = np.take_along_axis(S[ia], flat, 1)
    s = np.where(valid, s, -np.inf) / temperature
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    return np.stack([cb + (p * dx).sum(1), rb + (p * dy).sum(1)], 1)


def strongest_corner_per_patch(gray, h, w, stride, max_corners=3000, quality=0.001, min_dist=3):
    """For each patch, the strongest Shi-Tomasi corner inside it (or NaN if none)."""
    sx, sy = stride
    out = np.full((h * w, 2), np.nan, np.float32)
    c = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
    if c is None:
        return out
    c = c.reshape(-1, 2)  # sorted by quality, strongest first
    col = np.clip((c[:, 0] / sx).astype(int), 0, w - 1)
    row = np.clip((c[:, 1] / sy).astype(int), 0, h - 1)
    idx = row * w + col
    _, first = np.unique(idx, return_index=True)
    out[idx[first]] = c[first]
    return out


def lk_refine(gray_a, gray_b, pts_a, pts_b_init, win=21, levels=2, fb_thr=1.0):
    """Lucas-Kanade refinement seeded at the coarse match, with a forward-backward check."""
    pa = pts_a.reshape(-1, 1, 2).astype(np.float32)
    pb0 = pts_b_init.reshape(-1, 1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    pb, st, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, pa, pb0.copy(), winSize=(win, win), maxLevel=levels,
        criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    pa2, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray_b, gray_a, pb, pa.copy(), winSize=(win, win), maxLevel=levels,
        criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    fb = np.linalg.norm((pa2 - pa).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thr)
    return pb.reshape(-1, 2), ok


class DinoMatcher:
    """Dense DINOv3 patch matching with optional refinement.

    refine:
        None          patch centre -> patch centre (quantised to the patch stride)
        "softargmax"  sub-patch position from the local similarity distribution
        "lk"          anchor each matched patch at its strongest corner in A, transfer the
                      offset to B and refine with Lucas-Kanade (DINO = coarse, KLT = fine)
    """

    def __init__(self, extractor: DinoExtractor, refine: str | None = None, min_sim: float = 0.0,
                 ratio: float | None = None, temperature: float = 0.02, name: str | None = None):
        self.ext, self.refine, self.min_sim, self.ratio, self.temperature = extractor, refine, min_sim, ratio, temperature
        self.name = name or f"DINO x{extractor.scale:g}" + (f" + {refine}" if refine else "")

    def features(self, img, key=None):
        return self.ext(img, key)

    def match(self, fa, fb) -> dict:
        h, w = fb["h"], fb["w"]
        ia, ib, sim, S = mutual_nn(fa["grid"].reshape(h * w, -1), fb["grid"].reshape(h * w, -1),
                                   self.min_sim, self.ratio)
        pa = fa["centers"][ia]
        pb = fb["centers"][ib]
        if self.refine == "softargmax":
            g = softargmax_refine(S, ia, ib, h, w, self.temperature)
            sx, sy = fb["stride"]
            pb = np.stack([(g[:, 0] + 0.5) * sx - 0.5, (g[:, 1] + 0.5) * sy - 0.5], 1)
        elif self.refine == "lk":
            if "corners" not in fa:
                fa["corners"] = strongest_corner_per_patch(fa["gray"], fa["h"], fa["w"], fa["stride"])
            corner = fa["corners"][ia]
            has = ~np.isnan(corner[:, 0])
            pa = np.where(has[:, None], corner, pa)
            pb_init = pb + (pa - fa["centers"][ia])
            pb, ok = lk_refine(fa["gray"], fb["gray"], pa, pb_init)
            ia, ib, sim, pa, pb = ia[ok], ib[ok], sim[ok], pa[ok], pb[ok]
        return {"pts_a": pa, "pts_b": pb, "idx_a": ia, "idx_b": ib, "score": sim}


class OrbMatcher:
    """ORB + brute-force Hamming matching with Lowe's ratio test (the classical baseline)."""

    def __init__(self, n_features: int = 2000, ratio: float = 0.8, name: str = "ORB"):
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio, self.name = ratio, name
        self.cache = FeatureCache(64)

    def features(self, img, key=None):
        def run():
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, d = self.orb.detectAndCompute(gray, None)
            pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
            return {"pts": pts, "desc": d}

        return self.cache.get(key, run)

    def match(self, fa, fb) -> dict:
        empty = {"pts_a": np.zeros((0, 2)), "pts_b": np.zeros((0, 2)), "idx_a": np.zeros(0, int),
                 "idx_b": np.zeros(0, int), "score": np.zeros(0)}
        if fa["desc"] is None or fb["desc"] is None or len(fa["desc"]) < 2 or len(fb["desc"]) < 2:
            return empty
        ia, ib, sc = [], [], []
        for m in self.bf.knnMatch(fa["desc"], fb["desc"], k=2):
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance:
                ia.append(m[0].queryIdx)
                ib.append(m[0].trainIdx)
                sc.append(-m[0].distance)
        if not ia:
            return empty
        ia, ib = np.array(ia), np.array(ib)
        return {"pts_a": fa["pts"][ia], "pts_b": fb["pts"][ib], "idx_a": ia, "idx_b": ib, "score": np.array(sc)}


# --------------------------------------------------------------------------------------
# Geometry: back-projection, PnP, match verification
# --------------------------------------------------------------------------------------
def sample_depth(depth, pts, zmin=0.1, zmax=8.0):
    H, W = depth.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    z = np.zeros(len(pts), np.float32)
    z[inside] = depth[y[inside], x[inside]]
    return z, inside & (z > zmin) & (z < zmax)


def normalized_coords(pts, K, dist):
    if np.any(dist):
        return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    return (pts - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]


def backproject(pts, depth, K, dist):
    """3D points in the camera frame for pixel coordinates ``pts`` (N x 2)."""
    z, valid = sample_depth(depth, pts)
    xy = normalized_coords(pts, K, dist) if len(pts) else np.zeros((0, 2))
    return np.concatenate([xy * z[:, None], z[:, None]], 1), valid


def project(X, K, dist):
    if len(X) == 0:
        return np.zeros((0, 2))
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def transform(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def estimate_pose_pnp(pts_a, pts_b, depth_a, K, dist, reproj_thr=3.0, iters=2000, min_inliers=10, refine=True):
    """T_ba from 2D-2D matches + depth in A, via PnP-RANSAC (+ LM refinement on inliers).

    Returns dict with T_ba (or None on failure), the inlier mask over *all* input matches,
    and the number of matches that had valid depth.
    """
    n = len(pts_a)
    res = {"T_ba": None, "inliers": np.zeros(n, bool), "n_valid": 0}
    if n < 6:
        return res
    X, valid = backproject(pts_a, depth_a, K, dist)
    res["n_valid"] = int(valid.sum())
    if valid.sum() < 6:
        return res
    vi = np.nonzero(valid)[0]
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            X[vi].astype(np.float64), pts_b[vi].astype(np.float64), K, dist,
            iterationsCount=iters, reprojectionError=reproj_thr, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:  # degenerate configuration (e.g. all points nearly identical)
        return res
    if not ok or inl is None or len(inl) < min_inliers:
        return res
    inl = inl.ravel()
    if refine:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(X[vi[inl]], pts_b[vi[inl]].astype(np.float64), K, dist, rvec, tvec)
        except cv2.error:
            pass
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    res["T_ba"] = T
    res["inliers"][vi[inl]] = True
    return res


def gt_reprojection_error(pts_a, pts_b, depth_a, T_ba, K, dist):
    """Pixel distance between each match in B and where ground truth says it should be (NaN if no depth)."""
    X, valid = backproject(pts_a, depth_a, K, dist)
    err = np.full(len(pts_a), np.nan)
    if valid.any():
        Xb = transform(T_ba, X[valid])
        front = Xb[:, 2] > 0.05
        uv = project(Xb, K, dist)
        e = np.linalg.norm(uv - pts_b[valid], axis=1)
        e[~front] = np.nan
        err[valid] = e
    return err


def pose_error(T_est, T_gt):
    """(rotation error in degrees, translation error in metres)."""
    if T_est is None:
        return np.nan, np.nan
    E = np.linalg.inv(T_gt) @ T_est
    return float(rotation_angle_deg(E[:3, :3])), float(np.linalg.norm(E[:3, 3]))


def overlap_ratio(depth_a, T_ba, K, dist, shape, step=8):
    """Fraction of A's valid depth pixels that project inside B's image (ignores occlusion)."""
    H, W = shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    pts = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
    X, valid = backproject(pts, depth_a, K, dist)
    if not valid.any():
        return 0.0
    Xb = transform(T_ba, X[valid])
    uv = project(Xb, K, dist)
    inside = (Xb[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return float(inside.mean())


# --------------------------------------------------------------------------------------
# Visual odometry and trajectory metrics
# --------------------------------------------------------------------------------------
def run_odometry(seq: TUMSequence, frames, matcher, reproj_thr=3.0, mask_fn=None, progress=True):
    """Frame-to-frame PnP odometry over ``frames`` (a list of sequence indices).

    The first pose is initialised from ground truth. On failure the previous relative motion
    is reused (constant velocity). ``mask_fn(match, frame_a, frame_b)`` may return a boolean array to
    drop matches before PnP (e.g. features on people).

    Returns (poses [N,4,4], per-step stats list).
    """
    from tqdm.auto import tqdm

    poses = [seq.pose(frames[0]).copy()]
    stats = []
    last_T = np.eye(4)
    prev_img = seq.rgb(frames[0])
    prev_f = matcher.features(prev_img, key=(seq.name, frames[0]))
    prev_depth = seq.depth(frames[0])
    it = tqdm(frames[1:], desc=matcher.name, leave=False) if progress else frames[1:]
    for a, b in zip(frames[:-1], it):
        img = seq.rgb(b)
        f = matcher.features(img, key=(seq.name, b))
        m = matcher.match(prev_f, f)
        pa, pb = m["pts_a"], m["pts_b"]
        if mask_fn is not None and len(pa):
            keep = mask_fn(m, a, b)
            pa, pb = pa[keep], pb[keep]
        r = estimate_pose_pnp(pa, pb, prev_depth, seq.K, seq.dist, reproj_thr=reproj_thr)
        ok = r["T_ba"] is not None
        T_ba = r["T_ba"] if ok else last_T
        last_T = T_ba
        poses.append(poses[-1] @ np.linalg.inv(T_ba))
        stats.append({"a": a, "b": b, "n_matches": len(pa), "n_inliers": int(r["inliers"].sum()), "ok": ok})
        prev_f, prev_depth = f, seq.depth(b)
    return np.stack(poses), stats


def umeyama(src, dst, with_scale=False):
    """Least-squares similarity/rigid transform with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(est, gt, with_scale=False):
    """Absolute trajectory error (RMSE of positions after Umeyama alignment). Returns (rmse, aligned_xyz, errors)."""
    R, t, s = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale)
    aligned = (s * (R @ est[:, :3, 3].T)).T + t
    e = np.linalg.norm(aligned - gt[:, :3, 3], axis=1)
    return float(np.sqrt((e**2).mean())), aligned, e


def rpe(est, gt, delta=1):
    """Relative pose error over ``delta`` frames. Returns (trans RMSE [m], rot RMSE [deg])."""
    delta = max(1, min(delta, len(est) - 1))
    te, re = [], []
    for i in range(len(est) - delta):
        dg = np.linalg.inv(gt[i]) @ gt[i + delta]
        de = np.linalg.inv(est[i]) @ est[i + delta]
        E = np.linalg.inv(dg) @ de
        te.append(np.linalg.norm(E[:3, 3]))
        re.append(rotation_angle_deg(E[:3, :3]))
    te, re = np.array(te), np.array(re)
    return float(np.sqrt((te**2).mean())), float(np.sqrt((re**2).mean()))


# --------------------------------------------------------------------------------------
# Image perturbations (robustness)
# --------------------------------------------------------------------------------------
def adjust_brightness(img, gain=1.0, gamma=1.0):
    x = img.astype(np.float32) / 255.0
    x = np.clip(gain * x, 0, 1) ** gamma
    return (x * 255 + 0.5).astype(np.uint8)


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur(img, length, angle_deg=0.0):
    if length <= 1:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def add_noise(img, std, seed=0):
    if std <= 0:
        return img
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# People masks (for the dynamic-scene test)
# --------------------------------------------------------------------------------------
class PersonSegmenter:
    """DeepLabV3 (torchvision, Pascal-VOC classes) used as a pseudo ground-truth person mask."""

    PERSON = 15

    def __init__(self, device: str | None = None, arch: str = "resnet50"):
        import torch
        from torchvision.models import segmentation as seg

        self.torch = torch
        self.device = device or pick_device()
        if arch == "resnet50":
            w = seg.DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = seg.deeplabv3_resnet50(weights=w)
        else:
            w = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model = seg.deeplabv3_mobilenet_v3_large(weights=w)
        self.model = self.model.to(self.device).eval()
        self.tf = w.transforms()

    def __call__(self, img) -> np.ndarray:
        x = self.torch.from_numpy(img).permute(2, 0, 1)
        x = self.tf(x).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            out = self.model(x)["out"][0].argmax(0).cpu().numpy()
        return cv2.resize((out == self.PERSON).astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)


def mask_lookup(mask, pts):
    H, W = mask.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    return mask[y, x]


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def draw_matches(ax, img_a, img_b, pts_a, pts_b, colors=None, max_lines=300, seed=0, lw=0.6):
    """Side-by-side images with match lines; ``colors`` is an array of matplotlib colours."""
    H, W = img_a.shape[:2]
    ax.imshow(np.concatenate([img_a, img_b], 1))
    n = len(pts_a)
    sel = np.arange(n)
    if n > max_lines:
        sel = np.random.default_rng(seed).choice(n, max_lines, replace=False)
    for i in sel:
        c = "lime" if colors is None else colors[i]
        ax.plot([pts_a[i, 0], pts_b[i, 0] + W], [pts_a[i, 1], pts_b[i, 1]], color=c, lw=lw)
    ax.set_axis_off()
