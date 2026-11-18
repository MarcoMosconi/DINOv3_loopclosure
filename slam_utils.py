"""Utilities for the DINOv3 loop-closure pipeline on TUM RGB-D (see loop_closure.py).

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
    "fr3_long_office_household": ("freiburg3", "rgbd_dataset_freiburg3_long_office_household"),
}
# fx, fy, cx, cy, distortion (k1, k2, p1, p2, k3) from the TUM website.
# freiburg3 images are already undistorted.
INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3, (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
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


class TUMSequence:
    """RGB, depth and interpolated ground-truth pose for every RGB frame.

    Only RGB frames that have a depth frame within ``max_dt`` seconds and that lie inside
    the ground-truth time range are kept.
    """

    def __init__(self, name: str, root: str | Path = DATA_ROOT, max_dt: float = 0.02):
        self.name = name
        cam, _ = SEQUENCES[name]
        self.path = download_tum(name, root)
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
        self.poses = np.tile(np.eye(4), (len(idx), 1, 1))
        self.poses[:, :3, :3] = slerp(ts).as_matrix()
        self.poses[:, :3, 3] = np.stack([np.interp(ts, t_gt, gt[:, k]) for k in range(3)], 1)

        self.timestamps = ts
        self.rgb_files = [self.path / r_rgb[i][0] for i in idx]
        self.depth_files = [self.path / r_dep[j[i]][0] for i in idx]

    def __len__(self):
        return len(self.timestamps)

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE

    def rel_pose(self, a: int, b: int) -> np.ndarray:
        """Ground-truth T_ba (camera A coordinates -> camera B coordinates)."""
        return np.linalg.inv(self.poses[b]) @ self.poses[a]

    def __repr__(self):
        dur = self.timestamps[-1] - self.timestamps[0]
        length = np.linalg.norm(np.diff(self.poses[:, :3, 3], axis=0), axis=1).sum()
        return f"TUMSequence({self.name}: {len(self)} frames, {dur:.1f}s, path length {length:.2f} m)"


# --------------------------------------------------------------------------------------
# DINOv3 backbone (step 1)
# --------------------------------------------------------------------------------------
DINOV3_HF = "facebook/dinov3-vits16-pretrain-lvd1689m"
# Same LVD-1689M weights re-hosted (ungated) by timm; used when the gated HF checkpoint cannot be loaded.
DINOV3_TIMM = "vit_small_patch16_dinov3.lvd1689m"
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
    """Frozen DINOv3 ViT-S/16. One forward pass returns tokens of several transformer blocks."""

    def __init__(self, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or pick_device()
        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(DINOV3_HF).to(self.device).eval()
            self.backend, self.name = "hf", DINOV3_HF
            self.patch = self.model.config.patch_size
            self.n_reg = getattr(self.model.config, "num_register_tokens", 0)
        except OSError as e:  # gated repo without a valid token
            import timm

            print(f"[DinoV3] could not load {DINOV3_HF} ({type(e).__name__}); using the timm copy of the same weights")
            self.model = timm.create_model(DINOV3_TIMM, pretrained=True, num_classes=0, dynamic_img_size=True)
            self.model = self.model.to(self.device).eval()
            self.backend, self.name = "timm", "timm/" + DINOV3_TIMM
            self.patch = self.model.patch_embed.patch_size[0]
            self.n_reg = self.model.num_prefix_tokens - 1

    def _prep(self, imgs):
        H, W = imgs[0].shape[:2]
        h, w = round(H / self.patch), round(W / self.patch)
        batch = []
        for im in imgs:
            if (h * self.patch, w * self.patch) != (H, W):
                interp = cv2.INTER_AREA if h * self.patch < H else cv2.INTER_LINEAR
                im = cv2.resize(im, (w * self.patch, h * self.patch), interpolation=interp)
            batch.append((im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
        x = self.torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).contiguous()
        return x, h, w, (W / w, H / h)

    def blocks(self, imgs, layers=(9,)):
        """One forward pass -> {block: (cls [B,D], patches [B,h,w,D])}, stride.

        Intermediate blocks are un-normed (as HF ``hidden_states[l]``); key 12 is the final normed output.
        """
        x, h, w, stride = self._prep(imgs)
        x = x.to(self.device)
        with self.torch.inference_mode():
            if self.backend == "timm":
                final, inter = self.model.forward_intermediates(
                    x, indices=[l - 1 for l in layers], output_fmt="NLC", return_prefix_tokens=True, norm=False
                )
                toks = {12: final} | {l: self.torch.cat([pre, pat], 1) for l, (pat, pre) in zip(layers, inter)}
            else:
                o = self.model(pixel_values=x, output_hidden_states=True)
                toks = {12: o.last_hidden_state} | {l: o.hidden_states[l] for l in layers}
        out = {}
        for l, t in toks.items():
            t = t.float().cpu().numpy()
            out[l] = (t[:, 0], t[:, 1 + self.n_reg :].reshape(len(imgs), h, w, -1))
        return out, stride


def l2n(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def patch_centers(h: int, w: int, stride) -> np.ndarray:
    """Pixel coordinates (x, y) of every patch centre, in row-major order, shape [h*w, 2]."""
    sx, sy = stride
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx.ravel() + 0.5) * sx - 0.5, (yy.ravel() + 0.5) * sy - 0.5], 1)


# --------------------------------------------------------------------------------------
# Global descriptors for retrieval (step 2) and the ORB bag-of-words baseline
# --------------------------------------------------------------------------------------
def gem(patches: np.ndarray, p: float = 3.0) -> np.ndarray:
    """Generalised-mean pooling of one image's patch tokens [h, w, D] -> L2-normalised [D]."""
    P = patches.reshape(-1, patches.shape[-1]).astype(np.float32)
    return l2n((np.clip(P, 1e-6, None) ** p).mean(0) ** (1.0 / p))


class OrbBoW:
    """Flat ORB bag-of-visual-words with TF-IDF (a simple stand-in for DBoW2 in ORB-SLAM).

    Binary ORB descriptors are unpacked to {0,1}^256 and clustered with k-means; words are
    assigned by nearest centre. IDF is computed on the database images passed to ``fit_idf``.
    """

    def __init__(self, k: int = 1000, n_features: int = 1000):
        self.k = k
        self.orb = cv2.ORB_create(n_features)
        self.km = None
        self.idf = None

    def descriptors(self, img):
        _, d = self.orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
        if d is None:
            return np.zeros((0, 256), np.float32)
        return np.unpackbits(d, axis=1).astype(np.float32)

    def fit_vocab(self, imgs, max_desc: int = 200_000, seed: int = 0):
        from sklearn.cluster import MiniBatchKMeans

        D = np.concatenate([self.descriptors(im) for im in imgs])
        if len(D) > max_desc:
            D = D[np.random.default_rng(seed).choice(len(D), max_desc, replace=False)]
        self.km = MiniBatchKMeans(n_clusters=self.k, random_state=seed, batch_size=4096, n_init=3).fit(D)
        return self

    def tf(self, img):
        d = self.descriptors(img)
        h = np.zeros(self.k, np.float32)
        if len(d):
            np.add.at(h, self.km.predict(d), 1.0)
            h /= len(d)
        return h

    def fit_idf(self, tfs: np.ndarray):
        df = (tfs > 0).sum(0)
        self.idf = np.log((len(tfs) + 1) / (df + 1)).astype(np.float32)
        return self

    def encode(self, tfs: np.ndarray) -> np.ndarray:
        return l2n(tfs * self.idf[None])


# --------------------------------------------------------------------------------------
# Loop ground truth and evaluation
# --------------------------------------------------------------------------------------
def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def loop_ground_truth(poses, timestamps, d_thr=0.5, ang_thr=30.0, exclude_s=20.0):
    """Boolean matrices (candidate[i, j], positive[i, j]) for query i vs database frame j.

    A database frame is a *candidate* for query i if it is at least ``exclude_s`` seconds in
    the past (causal). It is a *positive* (true loop) if it is also within ``d_thr`` metres and
    ``ang_thr`` degrees of the query.
    """
    t = np.asarray(timestamps)
    cand = (t[:, None] - t[None, :]) > exclude_s
    P = poses[:, :3, 3]
    dist = np.linalg.norm(P[:, None] - P[None], axis=-1)
    R = poses[:, :3, :3]
    ang = rotation_angle_deg(np.einsum("iba,jbc->ijac", R, R))  # angle of R_i^T R_j
    return cand, cand & (dist < d_thr) & (ang < ang_thr)


def evaluate_retrieval(S: np.ndarray, cand: np.ndarray, pos: np.ndarray, ks=(1, 5)) -> dict:
    """Recall@k and top-1 precision-recall (AP, recall at 100 % precision) for a score matrix S[query, db]."""
    S = np.where(cand, S, -np.inf)
    has_pos = pos.any(1)
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        out[f"R@{k}"] = np.take_along_axis(pos, order[:, :k], 1).any(1)[has_pos].mean()
    top1 = order[:, 0]
    score = S[np.arange(len(S)), top1]
    correct = pos[np.arange(len(S)), top1]
    q = np.nonzero(cand.any(1))[0]
    o = q[np.argsort(-score[q])]
    tp = np.cumsum(correct[o])
    precision = tp / np.arange(1, len(o) + 1)
    recall = tp / max(1, has_pos.sum())
    full = precision >= 1.0 - 1e-9
    out["AP"] = np.sum(np.diff(np.concatenate([[0], recall])) * precision)
    out["R@100P"] = recall[full].max() if full.any() else 0.0
    out["top1"] = top1
    return out


# --------------------------------------------------------------------------------------
# Patch matching (step 3)
# --------------------------------------------------------------------------------------
def mutual_nn(fa: np.ndarray, fb: np.ndarray, min_sim: float = 0.0):
    """Mutual nearest neighbours under cosine similarity. fa [Na, D], fb [Nb, D] L2-normalised."""
    S = fa.astype(np.float32) @ fb.astype(np.float32).T
    ab, ba = S.argmax(1), S.argmax(0)
    ia = np.arange(len(fa))
    keep = (ba[ab] == ia) & (S[ia, ab] > min_sim)
    return ia[keep], ab[keep]


def strongest_corner_per_patch(gray, h, w, stride, max_corners=3000, quality=0.001, min_dist=3):
    """For each patch, the strongest Shi-Tomasi corner inside it (or NaN if none)."""
    sx, sy = stride
    out = np.full((h * w, 2), np.nan, np.float32)
    c = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist)
    if c is None:
        return out
    c = c.reshape(-1, 2)  # sorted by quality, strongest first
    idx = np.clip((c[:, 1] / sy).astype(int), 0, h - 1) * w + np.clip((c[:, 0] / sx).astype(int), 0, w - 1)
    _, first = np.unique(idx, return_index=True)
    out[idx[first]] = c[first]
    return out


def lk_refine(gray_a, gray_b, pts_a, pts_b_init, win=21, levels=2, fb_thr=1.0):
    """Lucas-Kanade refinement seeded at the coarse match, with a forward-backward check."""
    pa = pts_a.reshape(-1, 1, 2).astype(np.float32)
    pb0 = pts_b_init.reshape(-1, 1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    pb, st, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pa, pb0.copy(), winSize=(win, win), maxLevel=levels,
                                         criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    pa2, st2, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, pb, pa.copy(), winSize=(win, win), maxLevel=levels,
                                           criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    fb = np.linalg.norm((pa2 - pa).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thr)
    return pb.reshape(-1, 2), ok


def match_patches(fa: dict, fb: dict, min_sim: float = 0.3, lk: bool = False):
    """Mutual-NN matching of two DINOv3 patch grids ("both ways").

    fa / fb: {"grid" [h,w,D] L2-normalised, "h", "w", "stride", "centers", "gray"}.
    Without ``lk`` the matches are patch centre -> patch centre (16 px quantisation). With ``lk``
    each matched patch is anchored at its strongest corner in A and refined in B with Lucas-Kanade.
    Returns (pts_a, pts_b) pixel coordinates.
    """
    h, w = fb["h"], fb["w"]
    ia, ib = mutual_nn(fa["grid"].reshape(h * w, -1), fb["grid"].reshape(h * w, -1), min_sim)
    pa, pb = fa["centers"][ia], fb["centers"][ib]
    if lk:
        if "corners" not in fa:
            fa["corners"] = strongest_corner_per_patch(fa["gray"], fa["h"], fa["w"], fa["stride"])
        corner = fa["corners"][ia]
        pa = np.where(~np.isnan(corner[:, :1]), corner, pa)
        pb, ok = lk_refine(fa["gray"], fb["gray"], pa, pb + (pa - fa["centers"][ia]))
        pa, pb = pa[ok], pb[ok]
    return pa, pb


# --------------------------------------------------------------------------------------
# Geometry: back-projection and PnP-RANSAC (step 3), pose error (step 4)
# --------------------------------------------------------------------------------------
def backproject(pts, depth, K, dist, zmin=0.1, zmax=8.0):
    """3D points in the camera frame for pixel coordinates ``pts`` (N x 2), and a validity mask."""
    H, W = depth.shape
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    z = np.zeros(len(pts), np.float32)
    z[inside] = depth[y[inside], x[inside]]
    valid = inside & (z > zmin) & (z < zmax)
    if not len(pts):
        xy = np.zeros((0, 2))
    elif np.any(dist):
        xy = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    else:
        xy = (pts - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]
    return np.concatenate([xy * z[:, None], z[:, None]], 1), valid


def estimate_pose_pnp(pts_a, pts_b, depth_a, K, dist, reproj_thr=3.0, iters=500, min_inliers=10):
    """T_ba from 2D-2D matches + depth in A, via PnP-RANSAC + LM refinement on the inliers.

    Returns (T_ba or None on failure, number of inliers).
    """
    if len(pts_a) < 6:
        return None, 0
    X, valid = backproject(pts_a, depth_a, K, dist)
    if valid.sum() < 6:
        return None, 0
    vi = np.nonzero(valid)[0]
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            X[vi].astype(np.float64), pts_b[vi].astype(np.float64), K, dist,
            iterationsCount=iters, reprojectionError=reproj_thr, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:  # degenerate configuration (e.g. all points nearly identical)
        return None, 0
    if not ok or inl is None or len(inl) < min_inliers:
        return None, 0
    inl = inl.ravel()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(X[vi[inl]], pts_b[vi[inl]].astype(np.float64), K, dist, rvec, tvec)
    except cv2.error:
        pass
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(inl)


def pose_error(T_est, T_gt):
    """(rotation error in degrees, translation error in metres); NaN if no estimate."""
    if T_est is None:
        return np.nan, np.nan
    E = np.linalg.inv(T_gt) @ T_est
    return float(rotation_angle_deg(E[:3, :3])), float(np.linalg.norm(E[:3, 3]))
