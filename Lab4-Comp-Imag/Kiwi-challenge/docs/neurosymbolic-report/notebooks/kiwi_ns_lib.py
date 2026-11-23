"""kiwi_ns_lib.py — Shared PyTorch & Quantitative Imaging Library for the BMED365
Differentiable Neurosymbolic AI Technical Report ('Thesis + Code' Package).

All 5 companion notebooks in `docs/neurosymbolic-report/notebooks/` import this
module (or download it automatically from GitHub `main` when run in Google Colab),
ensuring zero code drift between experiments, figures, tables, and LaTeX prose.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import filters, measure, segmentation
from sklearn.cluster import KMeans

GITHUB_RAW_KIWI = "https://raw.githubusercontent.com/arvidl/BMED365-2027/main/Lab4-Comp-Imag/Kiwi-challenge"


# =============================================================================
# 1. Cross-Platform Runtime & Portable Path Resolution
# =============================================================================

def get_best_device() -> torch.device:
    """Select fastest available accelerator at runtime: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(dev: torch.device) -> None:
    """Synchronize asynchronous GPU streams before wall-clock timing."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def resolve_kiwi_file(rel_path: str) -> Path:
    """Resolve a Kiwi-challenge data/asset file locally or auto-download on Colab."""
    here = Path.cwd()
    candidates = [
        here / rel_path,
        here / ".." / rel_path,
        here / ".." / ".." / rel_path,
        here / ".." / ".." / ".." / rel_path,
        here / "Lab4-Comp-Imag/Kiwi-challenge" / rel_path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    dst = Path(rel_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {rel_path} from GitHub main for Colab session...")
    urllib.request.urlretrieve(f"{GITHUB_RAW_KIWI}/{rel_path}", dst)
    return dst.resolve()


def resolve_report_dirs() -> tuple[Path, Path]:
    """Return (figures_dir, results_dir) for the neurosymbolic-report package."""
    here = Path.cwd()
    for base in [here / "..", here, here / "Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report"]:
        if (base / "notebooks").exists() or base.name == "neurosymbolic-report":
            figs = (base / "figures").resolve()
            res = (base / "results").resolve()
            figs.mkdir(parents=True, exist_ok=True)
            res.mkdir(parents=True, exist_ok=True)
            return figs, res
    figs = (here / "figures").resolve()
    res = (here / "results").resolve()
    figs.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return figs, res


# =============================================================================
# 2. Data Loading, Whole-Organ Coordinate Frame & Biophysical Predicates
# =============================================================================

@dataclass
class KiwiDataset:
    vol: np.ndarray
    spacing: tuple[float, float, float]
    vox_vol_mm3: float
    gt_full: np.ndarray
    roi_gt: np.ndarray
    train_gt: np.ndarray
    test_gt: np.ndarray
    rx0: int
    rx1: int
    ry0: int
    ry1: int
    rz0: int
    rz1: int
    split_y: int
    fruit_mask: np.ndarray
    r_norm: np.ndarray
    z_norm: np.ndarray
    pole_dist: np.ndarray
    soft_ring: np.ndarray
    vol_norm: np.ndarray
    env_local: np.ndarray
    env_smooth: np.ndarray
    t2_ratio: np.ndarray
    non_dark: np.ndarray
    cx0: int = 112
    cx1: int = 144
    cy0: int = 112
    cy1: int = 144
    affine: np.ndarray | None = None
    gt_affine_max_diff: float = 0.0


def build_whole_organ_geometry(
    vol_3d: np.ndarray,
    sp: tuple[float, float, float],
    center_shift_vox: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radial_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract intact berry mask, normalized radial coordinate r/R_fruit(z),
    axial coordinate z/Z, and smooth differentiable inner-pericarp membership R_ring.

    Optional `center_shift_vox` (dx, dy, dz) and `radial_scale` allow synthetic
    pathological distortion / mass-effect stress testing (Notebook 05).
    """
    v_sm = ndi.gaussian_filter(vol_3d, sigma=(1.2 / sp[0], 1.2 / sp[1], 1.2 / sp[2]))
    fg_raw = v_sm > (filters.threshold_otsu(v_sm[::2, ::2, ::2]) * 0.35)
    fg_fill = np.zeros_like(fg_raw)
    for z in range(vol_3d.shape[2]):
        fg_fill[:, :, z] = ndi.binary_fill_holes(fg_raw[:, :, z])
    fg_fill = ndi.binary_fill_holes(fg_fill)
    lbl_fg, _ = ndi.label(fg_fill)
    fruit_mask = lbl_fg == (1 + int(np.argmax(np.bincount(lbl_fg.ravel())[1:])))

    dx, dy, dz = center_shift_vox
    r_norm_3d = np.full(vol_3d.shape, 1.5, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(vol_3d.shape[1]), np.arange(vol_3d.shape[0]))
    for z in range(vol_3d.shape[2]):
        m = fruit_mask[:, :, z]
        if m.sum() >= 80:
            cx, cy = float(xx[m].mean()) + dx, float(yy[m].mean()) + dy
            r_fruit = max(np.sqrt(m.sum() / np.pi) * radial_scale, 1.0)
            r_norm_3d[:, :, z] = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r_fruit

    z_idx = np.arange(vol_3d.shape[2], dtype=np.float32) + dz
    z_norm_1d = np.clip(z_idx / float(vol_3d.shape[2] - 1), 0.0, 1.0)
    z_norm_3d = np.broadcast_to(z_norm_1d[None, None, :], vol_3d.shape).copy()

    z_gate = np.clip((z_idx[None, None, :] - 16.0) / 5.0, 0.0, 1.0) * np.clip(
        (111.0 - z_idx[None, None, :]) / 5.0, 0.0, 1.0
    )
    r_in = 1.0 / (1.0 + np.exp(-np.clip((r_norm_3d - 0.22) / 0.02, -20.0, 20.0)))
    r_out = 1.0 / (1.0 + np.exp(-np.clip((0.65 - r_norm_3d) / 0.02, -20.0, 20.0)))
    soft_ring = (r_in * r_out * z_gate * fruit_mask.astype(np.float32)).astype(np.float32)
    return fruit_mask, r_norm_3d, z_norm_3d, soft_ring


def load_kiwi_dataset(gt_nii_override: str | Path | None = None) -> KiwiDataset:
    """Load 7T RARE kiwi MRI, curated 3D Micro-GT, organ coordinates & biophysical predicates.

    Pass `gt_nii_override` (e.g. 'MyStuff/kiwi-gt/kiwi_gt_manual_v1.nii.gz') to audit,
    evaluate, or retrain against a manually refined 3D instance mask without modifying
    the shared silver standard in `data/kiwi_micro_gt_labels.nii.gz`.
    """
    nii_path = resolve_kiwi_file("data/kiwi_20050217191129_131073_RARE_high_res_axial.nii.gz")
    gt_nii_path = (
        Path(gt_nii_override).expanduser().resolve()
        if gt_nii_override is not None
        else resolve_kiwi_file("data/kiwi_micro_gt_labels.nii.gz")
    )
    gt_json_path = resolve_kiwi_file("data/kiwi_micro_gt_roi.json")

    nii_img = nib.load(str(nii_path))
    vol = nii_img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in nii_img.header.get_zooms()[:3])
    vox_vol_mm3 = float(np.prod(spacing))

    gt_img = nib.load(str(gt_nii_path))
    gt_full = np.rint(np.asarray(gt_img.dataobj)).astype(np.uint16)
    gt_affine_max_diff = float(np.max(np.abs(nii_img.affine - gt_img.affine)))
    gt_meta = json.loads(gt_json_path.read_text(encoding="utf-8"))
    rx0, rx1 = gt_meta["roi_bounds_voxel_xyz"]["x"]
    ry0, ry1 = gt_meta["roi_bounds_voxel_xyz"]["y"]
    rz0, rz1 = gt_meta["roi_bounds_voxel_xyz"]["z"]
    roi_gt = gt_full[rx0:rx1, ry0:ry1, rz0:rz1]

    split_y = 40
    train_gt = roi_gt[:, :split_y, :].copy()
    test_gt = roi_gt[:, split_y:, :].copy()

    fruit_mask, r_norm, z_norm, soft_ring = build_whole_organ_geometry(vol, spacing)
    pole_dist = (np.abs(z_norm - 0.5) * 2.0).astype(np.float32)

    p99 = float(np.percentile(vol[fruit_mask], 99.0))
    vol_norm = np.clip(vol / (p99 + 1e-6), 0.0, 1.25).astype(np.float32)
    env_close = ndi.grey_closing(vol_norm, size=(7, 7, 3))
    env_local = ndi.gaussian_filter(
        env_close, sigma=(0.8 / spacing[0], 0.8 / spacing[1], 0.8 / spacing[2])
    ).astype(np.float32)
    env_smooth = ndi.gaussian_filter(
        vol_norm, sigma=(2.0 / spacing[0], 2.0 / spacing[1], 1.6 / spacing[2])
    ).astype(np.float32)

    t2_ratio = np.clip(vol_norm / (env_local + 0.03), 0.0, 1.25).astype(np.float32)
    non_dark = np.maximum(
        np.clip((t2_ratio - 0.82) / 0.15, 0.0, 1.0),
        np.clip((0.32 - env_local) / 0.10, 0.0, 1.0),
    ).astype(np.float32)

    return KiwiDataset(
        vol=vol, spacing=spacing, vox_vol_mm3=vox_vol_mm3,
        gt_full=gt_full, roi_gt=roi_gt, train_gt=train_gt, test_gt=test_gt,
        rx0=rx0, rx1=rx1, ry0=ry0, ry1=ry1, rz0=rz0, rz1=rz1, split_y=split_y,
        fruit_mask=fruit_mask, r_norm=r_norm, z_norm=z_norm, pole_dist=pole_dist,
        soft_ring=soft_ring, vol_norm=vol_norm, env_local=env_local,
        env_smooth=env_smooth, t2_ratio=t2_ratio, non_dark=non_dark,
        affine=nii_img.affine.copy(), gt_affine_max_diff=gt_affine_max_diff,
    )


def save_aligned_label_nifti(lbl_3d: np.ndarray, ref_affine: np.ndarray, out_path: Path) -> Path:
    """Save a uint16 3D instance label mask bit-aligned to the original 7T MRI affine."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(lbl_3d.astype(np.uint16), ref_affine)
    img.set_sform(ref_affine, code=1)
    img.set_qform(ref_affine, code=1)
    nib.save(img, str(out_path))
    return out_path


def make_3ch_tensor(ds: KiwiDataset, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> torch.Tensor:
    """Stack [vol_norm, env_local, env_smooth] -> (1, 3, X, Y, Z)."""
    arr = np.stack(
        [ds.vol_norm[x0:x1, y0:y1, z0:z1], ds.env_local[x0:x1, y0:y1, z0:z1], ds.env_smooth[x0:x1, y0:y1, z0:z1]],
        axis=0,
    )[None, ...]
    return torch.from_numpy(arr)


def make_3ch_and_gps(
    ds: KiwiDataset,
    x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
    r_override: np.ndarray | None = None,
    pole_override: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x3, gps) where gps = [r/R, (r/R - 0.43)^2, |2z/Z - 1|]."""
    x3 = make_3ch_tensor(ds, x0, x1, y0, y1, z0, z1)
    r_src = ds.r_norm if r_override is None else r_override
    p_src = ds.pole_dist if pole_override is None else pole_override
    rn = np.clip(r_src[x0:x1, y0:y1, z0:z1], 0.0, 1.5)
    zn = p_src[x0:x1, y0:y1, z0:z1]
    gps = np.stack([rn, (rn - 0.43) ** 2, zn], axis=0)[None, ...].astype(np.float32)
    return x3, torch.from_numpy(gps)


def build_distance_targets(inst_lbl: np.ndarray, sp: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary semantic foreground + per-instance normalized 3D EDT in [0, 1]."""
    sem_fg = (inst_lbl > 0).astype(np.float32)
    norm_dist = np.zeros(inst_lbl.shape, dtype=np.float32)
    for prop in measure.regionprops(inst_lbl):
        s0, s1, s2 = prop.slice
        sub = np.pad(inst_lbl[s0, s1, s2] == prop.label, pad_width=1, mode="constant", constant_values=False)
        d_local = ndi.distance_transform_edt(sub, sampling=sp)[1:-1, 1:-1, 1:-1]
        m_local = sub[1:-1, 1:-1, 1:-1]
        norm_dist[s0, s1, s2][m_local] = (d_local[m_local] / max(float(d_local.max()), 1e-6)).astype(np.float32)
    return sem_fg, norm_dist


# =============================================================================
# 3. Neural & Differentiable Neurosymbolic 3D Architectures
# =============================================================================

class ConvBlock3D(nn.Module):
    """Standard 3D medical CNN block (Conv3d -> InstanceNorm3d -> SiLU, x2)."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Backbone3DUNet(nn.Module):
    """Compact 3-level 3D U-Net (~340k params) shared across all benchmarks."""
    def __init__(self, in_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.bot = ConvBlock3D(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        return self.dec1(torch.cat([self.up1(d2), e1], dim=1))


class MultiTaskUNet3D(nn.Module):
    """Pure neural (in_ch=3) or naive CoordConv3D+InstanceNorm3d (in_ch=6) 3D U-Net."""
    def __init__(self, in_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.backbone = Backbone3DUNet(in_ch, base_ch)
        self.head_sem = nn.Conv3d(base_ch, 1, kernel_size=1)
        self.head_dist = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.head_sem(feat), torch.sigmoid(self.head_dist(feat))


class AdditiveBayesOrganNet3D(nn.Module):
    """Explicit Bayes Log-Odds Neurosymbolic 3D Network:
      z_total(x) = z_app(f_theta(X)) + z_gps(h_psi(gps)).
    Directly auditable spatial log-prior (+353 parameters over MultiTaskUNet3D).
    """
    def __init__(self, in_ch: int = 3, gps_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.backbone = Backbone3DUNet(in_ch, base_ch)
        self.head_app = nn.Conv3d(base_ch, 1, kernel_size=1)
        self.gps_mlp = nn.Sequential(
            nn.Conv3d(gps_ch, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(16, 16, kernel_size=1),
            nn.SiLU(),
        )
        self.head_gps = nn.Conv3d(16, 1, kernel_size=1)
        self.head_dist = nn.Conv3d(base_ch, 1, kernel_size=1)

    def gps_log_prior(self, gps: torch.Tensor) -> torch.Tensor:
        return self.head_gps(self.gps_mlp(gps))

    def forward(self, x3: torch.Tensor, gps: torch.Tensor):
        f = self.backbone(x3)
        return self.head_app(f) + self.gps_log_prior(gps), torch.sigmoid(self.head_dist(f))


class CoupledOrganNet3D(nn.Module):
    """Coupled feature-fusion Organ-GPS 3D Network (NB 06 architecture):
      z_total(x) = head_sem([f_theta(X), h_psi(gps)]).
    Allows nonlinear interaction between appearance and organ position (+864 params).
    """
    def __init__(self, in_ch: int = 3, gps_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.backbone = Backbone3DUNet(in_ch, base_ch)
        self.gps_mlp = nn.Sequential(
            nn.Conv3d(gps_ch, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(16, 16, kernel_size=1),
            nn.SiLU(),
        )
        self.head_sem = nn.Sequential(
            nn.Conv3d(base_ch + 16, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        self.head_dist = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x3: torch.Tensor, gps: torch.Tensor):
        f = self.backbone(x3)
        g = self.gps_mlp(gps)
        return self.head_sem(torch.cat([f, g], dim=1)), torch.sigmoid(self.head_dist(f))


# =============================================================================
# 4. Differentiable Real Logic (LTN) & Supervised Loss Primitives
# =============================================================================

def implication_violation(premise: torch.Tensor, conclusion: torch.Tensor, family: str = "product") -> torch.Tensor:
    """Compute fuzzy implication violation v = 1 - I(premise -> conclusion) in [0, 1].
    Families:
      - 'product' (Reichenbach): v = premise * (1 - conclusion)   [smooth everywhere]
      - 'lukasiewicz'          : v = max(0, premise - conclusion) [piecewise linear, dead zone]
      - 'godel'                : v = (1 - conclusion) if premise > conclusion else 0 [discontinuous]
    """
    a = premise.clamp(0.0, 1.0)
    b = conclusion.clamp(0.0, 1.0)
    if family == "product":
        return a * (1.0 - b)
    if family == "lukasiewicz":
        return torch.clamp(a - b, min=0.0, max=1.0)
    if family == "godel":
        return torch.where(a > b, 1.0 - b, torch.zeros_like(a))
    raise ValueError(f"Unknown fuzzy implication family: {family}")


def ltn_universal_pmean(violation: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    """LTN generalized p-mean universal error quantifier (Badreddine et al., AIJ 2022)."""
    return (torch.mean(violation.clamp(0.0, 1.0) ** p) + 1e-8) ** (1.0 / p)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    return 1.0 - (2.0 * inter + 1.0) / (probs.sum() + targets.sum() + 1.0)


def supervised_multitask_loss(s_log: torch.Tensor, d_hat: torch.Tensor, y_s: torch.Tensor, y_d: torch.Tensor) -> torch.Tensor:
    pos_w = torch.tensor([6.0], device=s_log.device)
    bce = F.binary_cross_entropy_with_logits(s_log, y_s, pos_weight=pos_w)
    dl = soft_dice_loss(s_log, y_s)
    mse = ((1.0 + 6.0 * y_s) * (d_hat - y_d) ** 2).mean()
    return 0.45 * bce + 0.55 * dl + 2.5 * mse


# =============================================================================
# 5. Unlabeled Sampling, 3D SSL Warm-Start & Unified Training Harness
# =============================================================================

def pretrain_ssl_backbone(ds: KiwiDataset, device: torch.device, seed: int = 2027, epochs: int = 100) -> tuple[Backbone3DUNet, float]:
    """3D Masked Context Restoration SSL warm-start on 36 unlabeled test-disjoint cubes."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    log_blob = np.clip(
        -ndi.gaussian_laplace(ds.vol_norm, sigma=(0.48 / ds.spacing[0], 0.48 / ds.spacing[1], 0.48 / ds.spacing[2])),
        0.0, 1.5,
    ).astype(np.float32)
    log_blob = np.clip(log_blob / (float(np.percentile(log_blob[ds.fruit_mask], 99.5)) + 1e-6), 0.0, 1.0)

    ssl_in, ssl_tgt = [], []
    rng = np.random.default_rng(seed)
    for _ in range(36):
        while True:
            x0 = int(rng.integers(24, 256 - 64 - 24))
            y0 = int(rng.integers(24, 256 - 40 - 24))
            z0 = int(rng.integers(16, 128 - 16 - 16))
            if (x0 < ds.rx1 and x0 + 64 > ds.rx0) and (y0 < ds.ry1 and y0 + 40 > ds.ry0 + ds.split_y) and (z0 < ds.rz1 and z0 + 16 > ds.rz0):
                continue
            if ds.fruit_mask[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16].mean() > 0.65:
                ssl_in.append(make_3ch_tensor(ds, x0, x0 + 64, y0, y0 + 40, z0, z0 + 16))
                c_drop = np.clip(ds.env_local[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16] - ds.vol_norm[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16], 0.0, 1.0)
                lg = log_blob[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16]
                ssl_tgt.append(torch.from_numpy(np.stack([ds.vol_norm[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16], c_drop, lg], axis=0)[None]))
                break

    x_bank = torch.cat(ssl_in, dim=0).to(device)
    y_bank = torch.cat(ssl_tgt, dim=0).to(device)
    backbone = Backbone3DUNet(in_ch=3, base_ch=16).to(device)
    head = nn.Conv3d(16, 3, kernel_size=1).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=4e-3, weight_decay=1e-4)

    for _ in range(epochs):
        idx = torch.randint(0, x_bank.shape[0], (4,), device=device)
        xb, yb = x_bank[idx], y_bank[idx]
        bmask = F.interpolate((torch.rand((4, 1, 16, 10, 8), device=device) < 0.35).float(), size=(64, 40, 16), mode="nearest")
        gamma = torch.empty((4, 1, 1, 1, 1), device=device).uniform_(0.80, 1.25)
        xb_cor = torch.clamp(xb.clamp_min(0.0) ** gamma + torch.randn_like(xb) * 0.05, 0.0, 1.3)
        xb_cor = xb_cor * (1.0 - bmask) + xb[:, 2:3].expand_as(xb) * bmask
        pred = head(backbone(xb_cor))
        loss = ((1.0 + 2.0 * bmask) * (1.0 + 3.0 * yb[:, 2:3]) * (pred - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    sync_device(device)
    return backbone, time.perf_counter() - t0


def sample_unlabeled_blocks(ds: KiwiDataset, device: torch.device, n_blocks: int = 40, seed: int = 2027):
    """Sample stratified unlabeled 3D whole-fruit blocks strictly disjoint from the held-out Test Wedge."""
    ux, ug, ur, un = [], [], [], []
    rng = np.random.default_rng(seed)
    z_strata = [0, 8, 16, 28, 44, 60, 76, 92, 104, 112]
    for i in range(n_blocks):
        while True:
            x0 = int(rng.integers(8, 256 - 64 - 8))
            y0 = int(rng.integers(8, 256 - 40 - 8))
            z0 = z_strata[i % len(z_strata)]
            if (x0 < ds.rx1 and x0 + 64 > ds.rx0) and (y0 < ds.ry1 and y0 + 40 > ds.ry0 + ds.split_y) and (z0 < ds.rz1 and z0 + 16 > ds.rz0):
                continue
            if ds.fruit_mask[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16].mean() > 0.12:
                xa, ga = make_3ch_and_gps(ds, x0, x0 + 64, y0, y0 + 40, z0, z0 + 16)
                ux.append(xa)
                ug.append(ga)
                ur.append(torch.from_numpy(ds.soft_ring[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16][None, None]))
                un.append(torch.from_numpy(ds.non_dark[x0 : x0 + 64, y0 : y0 + 40, z0 : z0 + 16][None, None]))
                break
    return (
        torch.cat(ux, dim=0).to(device),
        torch.cat(ug, dim=0).to(device),
        torch.cat(ur, dim=0).to(device),
        torch.cat(un, dim=0).to(device),
    )


def train_neurosymbolic_variant(
    ds: KiwiDataset,
    device: torch.device,
    mode: str,
    ssl_backbone: Backbone3DUNet | None = None,
    unlabeled_bank: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    p_quant: float = 3.0,
    w_comp: float = 2.0,
    w_t2: float = 1.2,
    gps_lr: float = 1.5e-2,
    epochs: int = 100,
    seed: int = 2027,
    use_columella_neg: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    """Train one controlled variant with CosineAnnealingLR and exact seed determinism."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    train_sem_np, train_dist_np = build_distance_targets(ds.train_gt, ds.spacing)
    x_tr, g_tr = [t.to(device) for t in make_3ch_and_gps(ds, ds.rx0, ds.rx1, ds.ry0, ds.ry0 + ds.split_y, ds.rz0, ds.rz1)]
    x_ng, g_ng = [t.to(device) for t in make_3ch_and_gps(ds, ds.cx0, ds.cx1, ds.cy0, ds.cy1, ds.rz0, ds.rz1)]
    y_sem = torch.from_numpy(train_sem_np[None, None]).to(device)
    y_dist = torch.from_numpy(train_dist_np[None, None]).to(device)

    if mode == "baseline":
        model = MultiTaskUNet3D(in_ch=3, base_ch=16).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3.2e-3, weight_decay=1e-4)
    elif mode == "coordconv_inorm":
        model = MultiTaskUNet3D(in_ch=6, base_ch=16).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3.2e-3, weight_decay=1e-4)
    elif mode == "additive_gps":
        model = AdditiveBayesOrganNet3D(in_ch=3, gps_ch=3, base_ch=16).to(device)
        if ssl_backbone is not None:
            model.backbone.load_state_dict(ssl_backbone.state_dict())
        gps_params = list(model.gps_mlp.parameters()) + list(model.head_gps.parameters())
        gps_ids = {id(p) for p in gps_params}
        other_params = [p for p in model.parameters() if id(p) not in gps_ids]
        opt = torch.optim.AdamW(
            [{"params": other_params, "lr": 3.2e-3}, {"params": gps_params, "lr": gps_lr}],
            weight_decay=1e-4,
        )
    elif mode == "coupled_gps":
        model = CoupledOrganNet3D(in_ch=3, gps_ch=3, base_ch=16).to(device)
        if ssl_backbone is not None:
            model.backbone.load_state_dict(ssl_backbone.state_dict())
        gps_params = list(model.gps_mlp.parameters())
        gps_ids = {id(p) for p in gps_params}
        other_params = [p for p in model.parameters() if id(p) not in gps_ids]
        opt = torch.optim.AdamW(
            [{"params": other_params, "lr": 3.2e-3}, {"params": gps_params, "lr": gps_lr}],
            weight_decay=1e-4,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-4)
    hist: dict[str, list[float]] = {"sup": [], "comp": [], "t2": []}

    for _ in range(epochs):
        dims = [d for d in (2, 3, 4) if np.random.rand() < 0.5]
        xp = torch.flip(x_tr, dims) if dims else x_tr
        gp = torch.flip(g_tr, dims) if dims else g_tr
        ys = torch.flip(y_sem, dims) if dims else y_sem
        yd = torch.flip(y_dist, dims) if dims else y_dist

        if mode == "baseline":
            sl, dh = model(xp)
            loss_sup = supervised_multitask_loss(sl, dh, ys, yd)
            if use_columella_neg:
                sl_n, _ = model(x_ng)
                loss_sup = loss_sup + 0.35 * F.binary_cross_entropy_with_logits(sl_n, torch.zeros_like(sl_n))
            loss_comp = torch.tensor(0.0, device=device)
            loss_t2 = torch.tensor(0.0, device=device)
            loss = loss_sup
        elif mode == "coordconv_inorm":
            sl, dh = model(torch.cat([xp, gp], dim=1))
            loss_sup = supervised_multitask_loss(sl, dh, ys, yd)
            if use_columella_neg:
                sl_n, _ = model(torch.cat([x_ng, g_ng], dim=1))
                loss_sup = loss_sup + 0.35 * F.binary_cross_entropy_with_logits(sl_n, torch.zeros_like(sl_n))
            ux, ug, ur, un = unlabeled_bank
            uidx = torch.randint(0, ux.shape[0], (4,), device=device)
            p_u = torch.sigmoid(model(torch.cat([ux[uidx], ug[uidx]], dim=1))[0])
            loss_comp = ltn_universal_pmean((1.0 - ur[uidx]) * p_u, p=p_quant)
            loss_t2 = ltn_universal_pmean(p_u * un[uidx], p=p_quant)
            loss = loss_sup + w_comp * loss_comp + w_t2 * loss_t2
        else:
            sl, dh = model(xp, gp)
            loss_sup = supervised_multitask_loss(sl, dh, ys, yd)
            if use_columella_neg:
                sl_n, _ = model(x_ng, g_ng)
                loss_sup = loss_sup + 0.35 * F.binary_cross_entropy_with_logits(sl_n, torch.zeros_like(sl_n))
            ux, ug, ur, un = unlabeled_bank
            uidx = torch.randint(0, ux.shape[0], (4,), device=device)
            p_u = torch.sigmoid(model(ux[uidx], ug[uidx])[0])
            loss_comp = ltn_universal_pmean((1.0 - ur[uidx]) * p_u, p=p_quant)
            loss_t2 = ltn_universal_pmean(p_u * un[uidx], p=p_quant)
            loss = loss_sup + w_comp * loss_comp + w_t2 * loss_t2

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist["sup"].append(float(loss_sup.item()))
        hist["comp"].append(float(loss_comp.item()))
        hist["t2"].append(float(loss_t2.item()))

    model.eval()
    sync_device(device)
    return model, hist, time.perf_counter() - t0


# =============================================================================
# 6. Decoding, Declarative Hard-Core Splitting, Evaluation & Inference
# =============================================================================

def build_ellip(rx_mm: float, rz_mm: float, sp: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = int(np.ceil(rx_mm / sp[0])), int(np.ceil(rx_mm / sp[1])), int(np.ceil(rz_mm / sp[2]))
    xx, yy, zz = np.ogrid[-rx : rx + 1, -ry : ry + 1, -rz : rz + 1]
    return ((xx * sp[0] / rx_mm) ** 2 + (yy * sp[1] / rx_mm) ** 2 + (zz * sp[2] / rz_mm) ** 2) <= 1.0


def decode_deep_watershed_3d(fg_bin: np.ndarray, dist_hat: np.ndarray, sp: tuple[float, float, float], min_mm3: float = 0.16) -> np.ndarray:
    """Marker-controlled 3D watershed seeded by anisotropic maxima of predicted distance transform."""
    d_sm = ndi.gaussian_filter(dist_hat * fg_bin, sigma=(0.65, 0.65, 0.45))
    maxima = (d_sm == ndi.maximum_filter(d_sm, footprint=build_ellip(0.68, 0.95, sp=sp))) & fg_bin & (d_sm >= 0.22)
    markers, _ = ndi.label(maxima)
    ws = segmentation.watershed(-d_sm, markers=markers, mask=fg_bin)
    unlabeled_islands, n_un = ndi.label(fg_bin & (ws == 0))
    vv = float(np.prod(sp))
    if n_un > 0:
        nxt = int(ws.max()) + 1
        for uid in range(1, n_un + 1):
            m = unlabeled_islands == uid
            if m.sum() * vv >= min_mm3:
                ws[m] = nxt
                nxt += 1
    out = np.zeros_like(ws, dtype=np.uint16)
    nid = 0
    for prop in measure.regionprops(ws):
        if prop.area * vv >= min_mm3:
            nid += 1
            out[ws == prop.label] = nid
    return out


def symbolic_hardcore_clump_split(
    lbl_3d: np.ndarray,
    sp: tuple[float, float, float],
    split_mm3: float = 2.55,
    d_min_mm: float = 0.88,
) -> np.ndarray:
    """Declarative Lab-5 Matérn hard-core clump disentanglement."""
    out = np.zeros_like(lbl_3d, dtype=np.uint16)
    sp_arr = np.asarray(sp, dtype=np.float32)
    vv = float(np.prod(sp))
    nid = 0
    for prop in measure.regionprops(lbl_3d):
        coords = prop.coords
        vol_mm3 = len(coords) * vv
        if vol_mm3 < 0.16:
            continue
        k_est = int(np.clip(round(vol_mm3 / 1.60), 2, 4)) if vol_mm3 >= split_mm3 else 1
        if k_est >= 2 and len(coords) >= k_est * 12:
            km = KMeans(n_clusters=k_est, n_init=4, random_state=2027).fit(coords.astype(np.float32) * sp_arr[None, :])
            dists = np.linalg.norm(km.cluster_centers_[:, None, :] - km.cluster_centers_[None, :, :], axis=-1)
            np.fill_diagonal(dists, 99.0)
            if dists.min() >= d_min_mm and (np.bincount(km.labels_, minlength=k_est) * vv).min() >= 0.35:
                for cid in range(k_est):
                    cp = coords[km.labels_ == cid]
                    nid += 1
                    out[cp[:, 0], cp[:, 1], cp[:, 2]] = nid
                continue
        nid += 1
        out[coords[:, 0], coords[:, 1], coords[:, 2]] = nid
    return out


def evaluate_semantic_and_instance_3d(pred_lbl: np.ndarray, gt_lbl: np.ndarray, min_iou: float = 0.25) -> dict:
    """Exact 3D Hungarian bipartite matching and Panoptic Quality (PQ = SQ x RQ)."""
    b_p, b_g = (pred_lbl > 0), (gt_lbl > 0)
    sem_dice = (2.0 * int((b_p & b_g).sum())) / max(int(b_p.sum() + b_g.sum()), 1)
    ids_p = [int(v) for v in np.unique(pred_lbl) if v > 0]
    ids_g = [int(v) for v in np.unique(gt_lbl) if v > 0]
    n_p, n_g = len(ids_p), len(ids_g)
    if n_p == 0 or n_g == 0:
        return {
            "Pred": n_p, "GT": n_g, "TP": 0, "FP": n_p, "FN": n_g,
            "Dice": round(sem_dice, 4), "Merges": 0, "Splits": 0,
            "Precision": 0.0, "Recall": 0.0, "SQ": 0.0, "RQ": 0.0, "PQ": 0.0,
        }
    p_map, g_map = {v: i for i, v in enumerate(ids_p)}, {v: j for j, v in enumerate(ids_g)}
    area_p = np.array([(pred_lbl == v).sum() for v in ids_p], dtype=np.float64)
    area_g = np.array([(gt_lbl == v).sum() for v in ids_g], dtype=np.float64)
    inter_m = np.zeros((n_p, n_g), dtype=np.float64)
    ov_fg = b_p & b_g
    if ov_fg.any():
        pairs, cnts = np.unique(np.stack([pred_lbl[ov_fg], gt_lbl[ov_fg]], axis=1), axis=0, return_counts=True)
        for (pv, gv), c in zip(pairs, cnts):
            inter_m[p_map[int(pv)], g_map[int(gv)]] = float(c)
    iou_m = inter_m / np.maximum(area_p[:, None] + area_g[None, :] - inter_m, 1.0)
    ri, ci = linear_sum_assignment(-iou_m)
    tp_pairs = [(r, c) for r, c in zip(ri, ci) if iou_m[r, c] >= min_iou]
    tp = len(tp_pairs)
    fp, fn = n_p - tp, n_g - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    sq = float(np.mean([iou_m[r, c] for r, c in tp_pairs])) if tp > 0 else 0.0
    rq = (2.0 * tp) / float(2 * tp + fp + fn)
    merges = int(np.sum(np.sum(inter_m / np.maximum(area_g[None, :], 1.0) >= 0.25, axis=1) >= 2))
    splits = int(np.sum(np.sum(inter_m / np.maximum(area_p[:, None], 1.0) >= 0.25, axis=0) >= 2))
    return {
        "Pred": n_p, "GT": n_g, "TP": tp, "FP": fp, "FN": fn,
        "Dice": round(sem_dice, 4), "Merges": merges, "Splits": splits,
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "SQ": round(sq, 4), "RQ": round(rq, 4), "PQ": round(sq * rq, 4),
    }


def match_instance_sets(pred_lbl: np.ndarray, gt_lbl: np.ndarray, min_iou: float = 0.25):
    """Return (tp_pred_ids, tp_gt_ids, fn_gt_ids, fp_pred_ids) for physical error anatomy."""
    ids_p = [int(v) for v in np.unique(pred_lbl) if v > 0]
    ids_g = [int(v) for v in np.unique(gt_lbl) if v > 0]
    p_map, g_map = {v: i for i, v in enumerate(ids_p)}, {v: j for j, v in enumerate(ids_g)}
    area_p = np.array([(pred_lbl == v).sum() for v in ids_p], dtype=np.float64)
    area_g = np.array([(gt_lbl == v).sum() for v in ids_g], dtype=np.float64)
    inter_m = np.zeros((len(ids_p), len(ids_g)), dtype=np.float64)
    ov = (pred_lbl > 0) & (gt_lbl > 0)
    if ov.any():
        pairs, cnts = np.unique(np.stack([pred_lbl[ov], gt_lbl[ov]], axis=1), axis=0, return_counts=True)
        for (pv, gv), c in zip(pairs, cnts):
            inter_m[p_map[int(pv)], g_map[int(gv)]] = float(c)
    iou_m = inter_m / np.maximum(area_p[:, None] + area_g[None, :] - inter_m, 1.0)
    ri, ci = linear_sum_assignment(-iou_m)
    tp_p = {ids_p[r] for r, c in zip(ri, ci) if iou_m[r, c] >= min_iou}
    tp_g = {ids_g[c] for r, c in zip(ri, ci) if iou_m[r, c] >= min_iou}
    return tp_p, tp_g, set(ids_g) - tp_g, set(ids_p) - tp_p


def forward_patch(model: nn.Module, mode: str, x3: torch.Tensor, gps: torch.Tensor):
    if mode == "baseline":
        return model(x3)
    if mode == "coordconv_inorm":
        return model(torch.cat([x3, gps], dim=1))
    return model(x3, gps)


def calibrate_threshold_on_train(
    model: nn.Module,
    mode: str,
    ds: KiwiDataset,
    device: torch.device,
    use_split: bool = True,
) -> float:
    """Leakage-free operating-point calibration strictly on the labeled TRAIN wedge."""
    x_tr, g_tr = [t.to(device) for t in make_3ch_and_gps(ds, ds.rx0, ds.rx1, ds.ry0, ds.ry0 + ds.split_y, ds.rz0, ds.rz1)]
    with torch.no_grad():
        sl, dh = forward_patch(model, mode, x_tr, g_tr)
        p_tr = torch.sigmoid(sl)[0, 0].cpu().numpy()
        d_tr = dh[0, 0].cpu().numpy()
    best_th, best_pq = 0.42, -1.0
    for th in [0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50]:
        lbl = decode_deep_watershed_3d(p_tr >= th, d_tr, ds.spacing)
        if use_split:
            lbl = symbolic_hardcore_clump_split(lbl, ds.spacing)
        ev = evaluate_semantic_and_instance_3d(lbl, ds.train_gt)
        if ev["PQ"] > best_pq:
            best_pq = ev["PQ"]
            best_th = th
    return float(best_th)


def run_test_and_whole_fruit(
    model: nn.Module,
    mode: str,
    ds: KiwiDataset,
    device: torch.device,
    thresh: float,
    use_split: bool = True,
    r_override: np.ndarray | None = None,
    pole_override: np.ndarray | None = None,
    do_whole_fruit: bool = True,
) -> dict:
    """Evaluate strictly on held-out Test Wedge (unmasked) and optionally full 128-slice berry."""
    x_te, g_te = [
        t.to(device)
        for t in make_3ch_and_gps(ds, ds.rx0, ds.rx1, ds.ry0 + ds.split_y, ds.ry1, ds.rz0, ds.rz1, r_override, pole_override)
    ]
    with torch.no_grad():
        sl_t, dh_t = forward_patch(model, mode, x_te, g_te)
        prob_te = torch.sigmoid(sl_t)[0, 0].cpu().numpy()
        dist_te = dh_t[0, 0].cpu().numpy()

    pred_te_raw = decode_deep_watershed_3d(prob_te >= thresh, dist_te, ds.spacing)
    pred_te = symbolic_hardcore_clump_split(pred_te_raw, ds.spacing) if use_split else pred_te_raw
    ev_te_raw = evaluate_semantic_and_instance_3d(pred_te_raw, ds.test_gt)
    ev_te = evaluate_semantic_and_instance_3d(pred_te, ds.test_gt)

    res = {
        "thresh": thresh, "prob_te": prob_te, "dist_te": dist_te,
        "pred_te_raw": pred_te_raw, "pred_te": pred_te,
        "ev_te_raw": ev_te_raw, "ev_te": ev_te,
    }
    if not do_whole_fruit:
        return res

    prob_wf = np.zeros(ds.vol.shape, dtype=np.float32)
    dist_wf = np.zeros(ds.vol.shape, dtype=np.float32)
    with torch.no_grad():
        for x0 in [16, 96, 176]:
            for y0 in [16, 56, 96, 136, 176]:
                for z0 in range(0, 128, 16):
                    if ds.fruit_mask[x0 : x0 + 80, y0 : y0 + 40, z0 : z0 + 16].sum() < 100:
                        continue
                    xa, ga = [
                        t.to(device)
                        for t in make_3ch_and_gps(ds, x0, x0 + 80, y0, y0 + 40, z0, z0 + 16, r_override, pole_override)
                    ]
                    s_o, d_o = forward_patch(model, mode, xa, ga)
                    prob_wf[x0 : x0 + 80, y0 : y0 + 40, z0 : z0 + 16] = torch.sigmoid(s_o)[0, 0].cpu().numpy()
                    dist_wf[x0 : x0 + 80, y0 : y0 + 40, z0 : z0 + 16] = d_o[0, 0].cpu().numpy()

    fg_wf = (prob_wf >= thresh) & ds.fruit_mask
    ood_vox = int((fg_wf & (ds.soft_ring < 0.05)).sum())
    polar_vox = int(fg_wf[:, :, :16].sum() + fg_wf[:, :, 112:].sum())
    ws_wf = decode_deep_watershed_3d(fg_wf, dist_wf, ds.spacing)
    if use_split:
        ws_wf = symbolic_hardcore_clump_split(ws_wf, ds.spacing)
    n_wf = len(np.unique(ws_wf)) - (1 if 0 in ws_wf else 0)
    res.update({
        "fg_wf": fg_wf, "ws_wf": ws_wf, "n_wf": n_wf,
        "ood_vox": ood_vox, "polar_vox": polar_vox,
    })
    return res
