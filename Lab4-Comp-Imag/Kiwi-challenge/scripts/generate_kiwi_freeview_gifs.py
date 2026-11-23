#!/usr/bin/env python3
"""Generate annotated animated GIF illustrations modeled on `view_kiwi_freeview.sh`.

Reads the exact NIfTI volumes and custom categorical ColorLUT (`kiwi_seeds_ColorLUT.txt`)
used by `Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh` and produces four
publication-grade, dark-theme Freeview-inspired animated GIFs saved by default to
`Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/`:

  1. `kiwi_freeview_microgt_triplanar.gif`
     Matches `./view_kiwi_freeview.sh` (4-pane `-layout 2` synchronized `-cc` flythrough
     across Micro-GT slices z=56..71 with `kiwi_seeds_ColorLUT.txt`, ROI box, y_split=168,
     and high-magnification Test-Wedge seed-ID inspector).
  2. `kiwi_freeview_outline_layers.gif`
     Matches `./view_kiwi_freeview.sh --outline` and interactive `Alt+V` layer toggling at
     the default equatorial slice z=64 (Raw MRI -> ColorLUT fill -> ColorLUT outline ->
     companion yellow outline RGB(255,235,59) -> Neurosymbolic green RGB(0,230,118) ->
     combined audit overlay), complete with an animated Freeview Volumes sidebar.
  3. `kiwi_freeview_wholefruit_pred.gif`
     Matches `./view_kiwi_freeview.sh --with-pred` (full 128-slice axial flythrough of all
     749 unmasked Neurosymbolic 3D seed predictions colored by `kiwi_seeds_ColorLUT.txt`,
     paired with coronal profile in true 2.13x mm aspect, live 128-slice axial seed count
     curve, polar-cap zero-FP zones, and yellow Micro-GT outlines inside z=56..71).
  4. `kiwi_freeview_3d_isosurfaces.gif`
     Matches `./view_kiwi_freeview.sh --with-pred --3d` (360-degree synchronized turntable
     of smoothed 3D marching-cubes seed isosurfaces in true physical millimeter space:
     98 Silver Micro-GT seeds in `kiwi_seeds_ColorLUT.txt` + y_split=168 plane on left;
     749 whole-fruit Neurosymbolic 3D seeds in RGB(0,230,118) inside a Fresnel-shaded
     translucent kiwifruit shell with embedded 1.22% ROI wireframe on right).

Usage (from repository root):
    uv run python Lab4-Comp-Imag/Kiwi-challenge/scripts/generate_kiwi_freeview_gifs.py
    uv run python Lab4-Comp-Imag/Kiwi-challenge/scripts/generate_kiwi_freeview_gifs.py --only triplanar outline
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure


# ---------------------------------------------------------------------------
# Color palette & Freeview visual constants (matching view_kiwi_freeview.sh)
# ---------------------------------------------------------------------------
BG_DARK = "#0d1117"
PANEL_BG = "#131820"
BORDER_COL = "#30363d"
TEXT_MAIN = "#f0f6fc"
TEXT_MUTED = "#9ba7b4"
ACCENT_BLUE = "#58a6ff"
ROI_RED = "#ff5252"
SPLIT_CYAN = "#00e5ff"
YELLOW_OUTLINE_RGB = np.array([255, 235, 59], dtype=np.float32) / 255.0   # 255,235,59
PRED_GREEN_RGB = np.array([0, 230, 118], dtype=np.float32) / 255.0        # 0,230,118
YELLOW_HEX = "#ffeb3b"
PRED_GREEN_HEX = "#00e676"

GT_OPACITY = 0.58
OUTLINE_OPACITY = 0.95
PRED_OPACITY = 0.48

RX0, RX1 = 128, 208
RY0, RY1 = 128, 208
RZ0, RZ1 = 56, 72
Y_SPLIT = 168
CURSOR_X, CURSOR_Y, CURSOR_Z = 168, 168, 64


# ---------------------------------------------------------------------------
# Data loading & ColorLUT utilities
# ---------------------------------------------------------------------------

def load_color_lut(lut_path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """Parse FreeSurfer-format kiwi_seeds_ColorLUT.txt -> (rgb_table[id, 3] in [0,1], names)."""
    entries: dict[int, tuple[str, int, int, int]] = {}
    max_id = 0
    for line in lut_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        idx = int(parts[0])
        name = parts[1]
        r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
        entries[idx] = (name, r, g, b)
        if idx > max_id:
            max_id = idx

    rgb_table = np.zeros((max_id + 1, 3), dtype=np.float32)
    names: dict[int, str] = {}
    for idx, (name, r, g, b) in entries.items():
        rgb_table[idx] = (r / 255.0, g / 255.0, b / 255.0)
        names[idx] = name
    return rgb_table, names


def normalize_mri_percentile(vol: np.ndarray, p_low: float = 0.0, p_high: float = 99.5) -> np.ndarray:
    """Match Freeview `percentile=1:grayscale=0,99.5` contrast windowing."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    return np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def compute_instance_outline_2d(lbl_2d: np.ndarray) -> np.ndarray:
    """Return 1-voxel inner boundary mask preserving boundaries between distinct adjacent IDs."""
    if not np.any(lbl_2d > 0):
        return np.zeros_like(lbl_2d, dtype=bool)
    up = np.zeros_like(lbl_2d)
    dn = np.zeros_like(lbl_2d)
    lt = np.zeros_like(lbl_2d)
    rt = np.zeros_like(lbl_2d)
    up[1:, :] = lbl_2d[:-1, :]
    dn[:-1, :] = lbl_2d[1:, :]
    lt[:, 1:] = lbl_2d[:, :-1]
    rt[:, :-1] = lbl_2d[:, 1:]
    return (lbl_2d > 0) & ((lbl_2d != up) | (lbl_2d != dn) | (lbl_2d != lt) | (lbl_2d != rt))


def composite_slice(
    gray_2d: np.ndarray,
    *,
    lut_fill_lbl: np.ndarray | None = None,
    lut_fill_alpha: float = GT_OPACITY,
    lut_outline_lbl: np.ndarray | None = None,
    lut_outline_alpha: float = OUTLINE_OPACITY,
    binary_fill_mask: np.ndarray | None = None,
    binary_fill_rgb: np.ndarray = PRED_GREEN_RGB,
    binary_fill_alpha: float = PRED_OPACITY,
    yellow_outline_lbl: np.ndarray | None = None,
    yellow_outline_alpha: float = OUTLINE_OPACITY,
    lut_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-composite Freeview-style layers onto a grayscale 2D slice (all in [0, 1])."""
    rgb = np.stack([gray_2d, gray_2d, gray_2d], axis=-1).astype(np.float32)

    if binary_fill_mask is not None and np.any(binary_fill_mask):
        m = binary_fill_mask > 0
        rgb[m] = (1.0 - binary_fill_alpha) * rgb[m] + binary_fill_alpha * binary_fill_rgb

    if lut_fill_lbl is not None and lut_rgb is not None and np.any(lut_fill_lbl > 0):
        m = lut_fill_lbl > 0
        ids = np.clip(lut_fill_lbl[m], 0, len(lut_rgb) - 1)
        rgb[m] = (1.0 - lut_fill_alpha) * rgb[m] + lut_fill_alpha * lut_rgb[ids]

    if lut_outline_lbl is not None and lut_rgb is not None:
        edge = compute_instance_outline_2d(lut_outline_lbl)
        if np.any(edge):
            ids = np.clip(lut_outline_lbl[edge], 0, len(lut_rgb) - 1)
            rgb[edge] = (1.0 - lut_outline_alpha) * rgb[edge] + lut_outline_alpha * lut_rgb[ids]

    if yellow_outline_lbl is not None:
        edge = compute_instance_outline_2d(yellow_outline_lbl)
        if np.any(edge):
            rgb[edge] = (1.0 - yellow_outline_alpha) * rgb[edge] + yellow_outline_alpha * YELLOW_OUTLINE_RGB

    return np.clip(rgb, 0.0, 1.0)


def extract_fruit_mask(vol: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Reproduce `kiwi_ns_lib.build_whole_organ_geometry` intact kiwifruit mask."""
    v_sm = ndi.gaussian_filter(vol, sigma=(1.2 / spacing[0], 1.2 / spacing[1], 1.2 / spacing[2]))
    fg_raw = v_sm > (filters.threshold_otsu(v_sm[::2, ::2, ::2]) * 0.35)
    fg_fill = np.zeros_like(fg_raw)
    for z in range(vol.shape[2]):
        fg_fill[:, :, z] = ndi.binary_fill_holes(fg_raw[:, :, z])
    fg_fill = ndi.binary_fill_holes(fg_fill)
    lbl_fg, _ = ndi.label(fg_fill)
    counts = np.bincount(lbl_fg.ravel())
    counts[0] = 0
    return lbl_fg == int(np.argmax(counts))


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Render a Matplotlib figure cleanly to an RGB PIL Image."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy(), mode="RGB")


def save_optimized_gif(
    frames: list[Image.Image],
    durations_ms: list[int],
    out_path: Path,
    lut_rgb: np.ndarray | None = None,
    colors: int = 256,
) -> Path:
    """Save crisp animated GIF preserving exact ColorLUT hues and neutral MRI grayscales."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, len(frames) // 8)
    sample = frames[::step][:8]
    w, h = sample[0].size
    sw, sh = min(w, 600), min(h, 400)

    # Calibration strips force MEDIANCUT to preserve neutral MRI grays AND exact ColorLUT/UI RGBs
    gray_bar_h = 28
    lut_bar_h = 36
    mosaic = Image.new("RGB", (sw, sh * len(sample) + gray_bar_h + lut_bar_h))
    for i, fr in enumerate(sample):
        mosaic.paste(fr.resize((sw, sh), Image.Resampling.NEAREST), (0, i * sh))

    ramp = np.linspace(0, 255, sw, dtype=np.uint8)
    gray_strip = np.broadcast_to(ramp[None, :, None], (gray_bar_h, sw, 3)).copy()
    mosaic.paste(Image.fromarray(gray_strip, mode="RGB"), (0, sh * len(sample)))

    # Key saturated colors (18 LUT swatches + UI accents) repeated in wide tiles
    key_cols: list[tuple[int, int, int]] = [
        (255, 235, 59),   # Yellow outline
        (0, 230, 118),    # Neurosymbolic green
        (255, 82, 82),    # ROI red
        (0, 229, 255),    # y_split cyan
        (88, 166, 255),   # Accent blue
        (255, 171, 64),   # Polar amber
    ]
    if lut_rgb is not None:
        for sid in range(1, min(19, len(lut_rgb))):
            r, g, b = (np.clip(lut_rgb[sid] * 255.0, 0, 255)).astype(int)
            key_cols.append((int(r), int(g), int(b)))
    lut_strip = np.zeros((lut_bar_h, sw, 3), dtype=np.uint8)
    tile_w = max(1, sw // len(key_cols))
    for idx_c, rgb_t in enumerate(key_cols):
        x_a = idx_c * tile_w
        x_b = sw if idx_c == len(key_cols) - 1 else (idx_c + 1) * tile_w
        lut_strip[:, x_a:x_b] = rgb_t
    mosaic.paste(Image.fromarray(lut_strip, mode="RGB"), (0, sh * len(sample) + gray_bar_h))

    pal_img = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    quantized = [
        fr.quantize(palette=pal_img, dither=Image.Dither.NONE)
        for fr in frames
    ]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {out_path.name} ({len(frames)} frames, {w}x{h} px, {size_mb:.2f} MB)")
    return out_path


def draw_colorlut_strip(
    ax: plt.Axes,
    lut_rgb: np.ndarray,
    extra_legend: list[tuple[str, str, str]] | None = None,
    title_prefix: str = "kiwi_seeds_ColorLUT.txt (18-color cyclic palette for IDs 1..2000):",
) -> None:
    """Draw a compact, legible 18-swatch ColorLUT strip and cleanly spaced layer key."""
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER_COL)
        sp.set_linewidth(0.8)

    y_top = 0.70 if extra_legend else 0.54
    ax.text(
        0.010, y_top, title_prefix,
        color=TEXT_MAIN, fontsize=7.9, fontweight="bold", va="center", ha="left",
    )

    x_start = 0.455
    x_end = 0.990
    n_swatches = 18
    sw_w = (x_end - x_start) / n_swatches
    box_h = 0.36 if extra_legend else 0.50
    box_y = y_top - box_h / 2.0
    for i in range(n_swatches):
        sid = i + 1
        col = lut_rgb[sid]
        rx = x_start + i * sw_w
        rect = mpatches.Rectangle(
            (rx + 0.0018, box_y), sw_w - 0.0036, box_h,
            facecolor=col, edgecolor="#0d1117", linewidth=0.6,
        )
        ax.add_patch(rect)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        tcol = "#0d1117" if lum > 0.52 else "#ffffff"
        ax.text(
            rx + sw_w / 2.0, box_y + box_h / 2.0, f"{sid}",
            color=tcol, fontsize=6.7, fontweight="bold", ha="center", va="center",
        )

    if extra_legend:
        # Row 2: proportionally distributed across [0.012, 0.985] so items never collide
        y_row2 = 0.23
        weights = [max(len(lbl), 10) + 7.0 for _, _, lbl in extra_legend]
        total_w = sum(weights)
        usable = 0.970
        cursor_x = 0.012
        for (kind, col, label), wt in zip(extra_legend, weights):
            if kind == "box":
                ax.add_patch(mpatches.Rectangle(
                    (cursor_x, y_row2 - 0.10), 0.013, 0.20,
                    facecolor=col, edgecolor=TEXT_MAIN, linewidth=0.5,
                ))
            elif kind == "outline":
                ax.add_patch(mpatches.Rectangle(
                    (cursor_x, y_row2 - 0.10), 0.013, 0.20,
                    facecolor="none", edgecolor=col, linewidth=1.6,
                ))
            elif kind == "dash":
                ax.plot(
                    [cursor_x, cursor_x + 0.016], [y_row2, y_row2],
                    color=col, lw=1.9, ls="--",
                )
            ax.text(
                cursor_x + 0.020, y_row2, label,
                color=TEXT_MUTED, fontsize=7.4, va="center", ha="left",
            )
            cursor_x += usable * (wt / total_w)


def draw_scalebar(
    ax: plt.Axes,
    x_right: float,
    y_bottom: float,
    length_mm: float,
    spacing_x_mm: float,
    label: str | None = None,
) -> None:
    """Draw a clean physical-millimeter scale bar in voxel coordinates."""
    length_vox = length_mm / spacing_x_mm
    x0 = x_right - length_vox
    ax.plot([x0, x_right], [y_bottom, y_bottom], color="#ffffff", lw=2.6, solid_capstyle="butt")
    ax.plot([x0, x0], [y_bottom - 0.8, y_bottom + 0.8], color="#ffffff", lw=1.4)
    ax.plot([x_right, x_right], [y_bottom - 0.8, y_bottom + 0.8], color="#ffffff", lw=1.4)
    lbl = label or f"{length_mm:g} mm"
    ax.text(
        (x0 + x_right) / 2.0, y_bottom + 2.2, lbl,
        color="#ffffff", fontsize=7.6, fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="#0d1117", edgecolor="none", alpha=0.72),
    )


def annotate_seed_ids_on_crop(
    ax: plt.Axes,
    lbl_2d: np.ndarray,
    x0: int,
    y0: int,
    lut_rgb: np.ndarray,
    min_pixels: int = 4,
    fontsize: float = 6.8,
) -> int:
    """Place legible, non-overlapping `#ID` annotations at 2D centroids of seeds inside a crop."""
    # lbl_2d has shape (ny, nx) corresponding to vol[x0:x1, y0:y1, z].T
    uniq = [int(u) for u in np.unique(lbl_2d) if u > 0]
    pts: list[tuple[int, float, float, int]] = []
    for sid in uniq:
        ys, xs = np.where(lbl_2d == sid)
        if len(xs) < min_pixels:
            continue
        cx = x0 + float(xs.mean())
        cy = y0 + float(ys.mean())
        pts.append((sid, cx, cy, len(xs)))

    # Sort larger seeds first so they claim clean spots
    pts.sort(key=lambda t: -t[3])
    placed: list[tuple[float, float]] = []
    for sid, cx, cy, _ in pts:
        # Light greedy vertical offset if very close to an already placed label
        tx, ty = cx, cy
        for attempt in range(5):
            if all((tx - px) ** 2 + (ty - py) ** 2 > 14.0 for px, py in placed):
                break
            dy = (2.4 if attempt % 2 == 0 else -2.4) * (1 + attempt // 2)
            ty = cy + dy
        placed.append((tx, ty))
        col = lut_rgb[min(sid, len(lut_rgb) - 1)]
        ax.text(
            tx, ty, f"#{sid}",
            color=col, fontsize=fontsize, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#090c10", edgecolor=col, lw=0.65, alpha=0.84),
        )
    return len(uniq)


def style_image_ax(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_facecolor("#000000")
    ax.tick_params(colors=TEXT_MUTED, labelsize=7.2, length=2.5, pad=2)
    for sp in ax.spines.values():
        sp.set_color(BORDER_COL)
        sp.set_linewidth(0.9)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, color=TEXT_MAIN, fontsize=8.8, fontweight="bold", pad=5, loc="left")


# ---------------------------------------------------------------------------
# GIF 1: 4-Pane Freeview Tri-Planar Micro-GT Flythrough (default script mode)
# ---------------------------------------------------------------------------

def build_gif_triplanar(
    mri_norm: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    affine: np.ndarray,
    lut_rgb: np.ndarray,
    out_path: Path,
) -> Path:
    print("Rendering GIF 1: 4-Pane Freeview Micro-GT Tri-Planar Flythrough...")
    asp_z = spacing[2] / spacing[0]  # 2.1333

    # Ping-pong sweep through the 16 labeled Micro-GT slices z=56..71
    z_seq = list(range(RZ0, RZ1)) + list(range(RZ1 - 2, RZ0, -1))
    durations: list[int] = []
    frames: list[Image.Image] = []

    # Pre-render static sagittal (x=168) and coronal (y=168) RGB composites once
    sag_rgb = composite_slice(
        mri_norm[CURSOR_X, :, :].T,
        lut_fill_lbl=gt[CURSOR_X, :, :].T,
        lut_fill_alpha=GT_OPACITY,
        yellow_outline_lbl=gt[CURSOR_X, :, :].T,
        yellow_outline_alpha=0.55,
        lut_rgb=lut_rgb,
    )
    cor_rgb = composite_slice(
        mri_norm[:, CURSOR_Y, :].T,
        lut_fill_lbl=gt[:, CURSOR_Y, :].T,
        lut_fill_alpha=GT_OPACITY,
        yellow_outline_lbl=gt[:, CURSOR_Y, :].T,
        yellow_outline_alpha=0.55,
        lut_rgb=lut_rgb,
    )

    # Zoom bounds for High-Mag Test-Wedge inspector (x=126..196, y=164..202)
    zx0, zx1 = 126, 196
    zy0, zy1 = 164, 202

    fig = plt.figure(figsize=(11.6, 8.2), dpi=100, facecolor=BG_DARK)

    for idx_f, z in enumerate(z_seq):
        fig.clf()
        fig.patch.set_facecolor(BG_DARK)

        z_mm = float((affine @ np.array([CURSOR_X, CURSOR_Y, z, 1.0]))[2])
        gt_slice = gt[:, :, z]
        active_ids = [int(u) for u in np.unique(gt_slice) if u > 0]
        n_active = len(active_ids)
        n_train_s = len([u for u in np.unique(gt[:, :Y_SPLIT, z]) if u > 0])
        n_test_s = len([u for u in np.unique(gt[:, Y_SPLIT:, z]) if u > 0])

        # --- Top Header Banner ---
        fig.text(
            0.018, 0.970,
            "BMED365 7T Kiwi MRI  |  Silver Micro-GT 4-Pane Freeview Flythrough",
            color=TEXT_MAIN, fontsize=12.0, fontweight="bold", va="center",
        )
        fig.text(
            0.018, 0.941,
            f"ROI x=128..207, y=128..207, z=56..71; y_split=168   •   "
            f"Cursor ({CURSOR_X}, {CURSOR_Y}, {z})  |  z = {z_mm:+.1f} mm  |  "
            f"Slice seeds: {n_active} (Train {n_train_s} / Test {n_test_s})",
            color=TEXT_MUTED, fontsize=8.1, va="center",
        )
        fig.text(
            0.982, 0.968,
            " ./view_kiwi_freeview.sh ",
            color="#0d1117", fontsize=8.4, fontweight="bold", family="monospace",
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=ACCENT_BLUE, edgecolor="none"),
        )
        if z == CURSOR_Z:
            fig.text(
                0.982, 0.939,
                "★ Default Equator (z=64)",
                color=YELLOW_HEX, fontsize=7.9, fontweight="bold", va="center", ha="right",
            )

        # Generous vertical separation between header, Row 1, Row 2, and bottom ColorLUT strip
        ax_ax = fig.add_axes([0.052, 0.545, 0.420, 0.315])
        ax_sg = fig.add_axes([0.550, 0.545, 0.428, 0.315])
        ax_co = fig.add_axes([0.052, 0.132, 0.420, 0.305])
        ax_zm = fig.add_axes([0.550, 0.132, 0.428, 0.305])
        ax_lut = fig.add_axes([0.018, 0.012, 0.964, 0.078])

        # (1) Axial pane (zoomed 2.2x around Micro-GT sector)
        ax_rgb = composite_slice(
            mri_norm[:, :, z].T,
            lut_fill_lbl=gt_slice.T,
            lut_fill_alpha=GT_OPACITY,
            lut_outline_lbl=gt_slice.T,
            lut_outline_alpha=0.85,
            lut_rgb=lut_rgb,
        )
        ax_ax.imshow(ax_rgb, origin="lower", interpolation="nearest")
        ax_ax.plot([RX0, RX1, RX1, RX0, RX0], [RY0, RY0, RY1, RY1, RY0], color=ROI_RED, lw=1.3, ls="--")
        ax_ax.plot([RX0, RX1], [Y_SPLIT, Y_SPLIT], color=SPLIT_CYAN, lw=1.4, ls="--")
        # Freeview -cc crosshair at (168, 168)
        ax_ax.axvline(CURSOR_X, color="#ffffff", lw=0.75, alpha=0.55)
        ax_ax.axhline(CURSOR_Y, color="#ffffff", lw=0.75, alpha=0.55)
        ax_ax.plot(CURSOR_X, CURSOR_Y, "+", color=YELLOW_HEX, ms=7, mew=1.3)
        # Zoom inset box indicator
        ax_ax.plot([zx0, zx1, zx1, zx0, zx0], [zy0, zy0, zy1, zy1, zy0], color=YELLOW_HEX, lw=0.95, ls=":")
        ax_ax.text(RX1 - 2, RY0 + 3, "Train (y<168)", color="#7ee787", fontsize=7.4, fontweight="bold", ha="right",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none", alpha=0.76))
        ax_ax.text(RX1 - 2, RY1 - 7, "Held-Out Test (y≥168)", color=SPLIT_CYAN, fontsize=7.4, fontweight="bold", ha="right",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none", alpha=0.76))
        ax_ax.set_xlim(96, 232)
        ax_ax.set_ylim(96, 232)
        ax_ax.set_xlabel("X voxel (L→)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_ax.set_ylabel("Y voxel (A↑)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        draw_scalebar(ax_ax, x_right=225, y_bottom=103, length_mm=5.0, spacing_x_mm=spacing[0])
        style_image_ax(ax_ax, f"Axial Viewport  (z = {z} / 127)", "Zoom 2.2× • ColorLUT α=0.58 + ROI & y_split")

        # (2) Sagittal pane (x = 168, true 2.13x mm aspect)
        ax_sg.imshow(sag_rgb, origin="lower", aspect=asp_z, interpolation="nearest")
        ax_sg.plot([RY0, RY1, RY1, RY0, RY0], [RZ0, RZ0, RZ1, RZ1, RZ0], color=ROI_RED, lw=1.2, ls="--")
        ax_sg.plot([Y_SPLIT, Y_SPLIT], [RZ0, RZ1], color=SPLIT_CYAN, lw=1.3, ls="--")
        ax_sg.axhline(z, color=YELLOW_HEX, lw=1.1, alpha=0.90)
        ax_sg.axvline(CURSOR_Y, color="#ffffff", lw=0.7, alpha=0.50)
        ax_sg.set_xlim(28, 228)
        ax_sg.set_ylim(16, 112)
        ax_sg.set_xlabel("Y voxel (A→)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_sg.set_ylabel("Z slice (S↑, true 2.13× mm)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_sg.text(34, 104, f"z = {z}", color=YELLOW_HEX, fontsize=8.0, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.16", facecolor="#0d1117", edgecolor=YELLOW_HEX, lw=0.6, alpha=0.84))
        style_image_ax(ax_sg, f"Sagittal Viewport  (x = {CURSOR_X})", "Crosshair synced to axial slice z; Δz = 0.50 mm")

        # (3) Coronal pane (y = 168, true 2.13x mm aspect)
        ax_co.imshow(cor_rgb, origin="lower", aspect=asp_z, interpolation="nearest")
        ax_co.plot([RX0, RX1, RX1, RX0, RX0], [RZ0, RZ0, RZ1, RZ1, RZ0], color=ROI_RED, lw=1.2, ls="--")
        ax_co.axhline(z, color=YELLOW_HEX, lw=1.1, alpha=0.90)
        ax_co.axvline(CURSOR_X, color="#ffffff", lw=0.7, alpha=0.50)
        ax_co.set_xlim(28, 228)
        ax_co.set_ylim(16, 112)
        ax_co.set_xlabel("X voxel (L→)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_co.set_ylabel("Z slice (S↑, true 2.13× mm)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_co.text(34, 104, "1.22% labeled slab", color=ROI_RED, fontsize=7.5, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none", alpha=0.78))
        style_image_ax(ax_co, f"Coronal Viewport  (y = {CURSOR_Y}, y_split plane)", "Anisotropic spacing: 0.234 × 0.234 × 0.500 mm")

        # (4) High-Mag Test-Wedge Seed-ID Inspector (vivid ColorLUT fill + razor-sharp contour + IDs)
        crop_gray = mri_norm[zx0:zx1, zy0:zy1, z].T
        crop_lbl = gt[zx0:zx1, zy0:zy1, z].T
        crop_rgb = composite_slice(
            crop_gray,
            lut_fill_lbl=crop_lbl,
            lut_fill_alpha=0.62,
            lut_rgb=lut_rgb,
        )
        ax_zm.imshow(
            crop_rgb, origin="lower", interpolation="nearest",
            extent=[zx0 - 0.5, zx1 - 0.5, zy0 - 0.5, zy1 - 0.5],
        )
        if np.any(crop_lbl > 0):
            ax_zm.contour(
                np.arange(zx0, zx1), np.arange(zy0, zy1), (crop_lbl > 0).astype(float),
                levels=[0.5], colors=[YELLOW_HEX], linewidths=0.85,
            )
        ax_zm.axhline(Y_SPLIT, color=SPLIT_CYAN, lw=1.3, ls="--")
        n_vis_zoom = annotate_seed_ids_on_crop(ax_zm, crop_lbl, zx0, zy0, lut_rgb, min_pixels=3, fontsize=6.7)
        ax_zm.set_xlim(zx0 - 0.5, zx1 - 0.5)
        ax_zm.set_ylim(zy0 - 0.5, zy1 - 0.5)
        ax_zm.set_xlabel("X voxel (High-Mag dotted yellow box)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_zm.set_ylabel("Y voxel", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        draw_scalebar(ax_zm, x_right=zx1 - 3, y_bottom=zy0 + 2.5, length_mm=2.0, spacing_x_mm=spacing[0])
        style_image_ax(
            ax_zm,
            f"High-Mag Seed-ID Inspector  ({n_vis_zoom} IDs in Test crop)",
            "Bit-aligned uint16 IDs (#1..#98, Δaffine = 0.0) colored by LUT",
        )

        # Progress bar tucked cleanly at top of LUT strip
        draw_colorlut_strip(
            ax_lut, lut_rgb,
            extra_legend=[
                ("dash", ROI_RED, "Micro-GT ROI (80×80×16)"),
                ("dash", SPLIT_CYAN, "y_split = 168 (Train / Test)"),
                ("outline", YELLOW_HEX, "Per-instance boundary"),
                ("box", "#ffffff", f"Z-slab progress: z={z} ({z - RZ0 + 1}/16)"),
            ],
        )

        frames.append(fig_to_pil(fig))
        # Hold slightly longer on the default equator slice z=64 on the forward sweep
        durations.append(760 if (z == CURSOR_Z and idx_f < (RZ1 - RZ0)) else 250)

    plt.close(fig)
    return save_optimized_gif(frames, durations, out_path, lut_rgb=lut_rgb)


# ---------------------------------------------------------------------------
# GIF 2: Interactive Layer & Outline Toggle Inspector (--outline & Alt+V)
# ---------------------------------------------------------------------------

def build_gif_outline_layers(
    mri_norm: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    lut_rgb: np.ndarray,
    out_path: Path,
) -> Path:
    print("Rendering GIF 2: Freeview Outline & Layer-Toggle Inspector (z=64)...")
    z = CURSOR_Z
    gray_full = mri_norm[:, :, z].T
    gt_full = gt[:, :, z].T
    pred_full = pred[:, :, z].T

    # High-mag Test-Wedge crop (showing clear seed cores + split boundaries)
    zx0, zx1 = 128, 192
    zy0, zy1 = 164, 202
    gray_zm = mri_norm[zx0:zx1, zy0:zy1, z].T
    gt_zm = gt[zx0:zx1, zy0:zy1, z].T
    pred_zm = pred[zx0:zx1, zy0:zy1, z].T

    # 6 inspection modes matching view_kiwi_freeview.sh options
    states = [
        {
            "step": "1 / 6",
            "name": "Raw 7T RARE MRI",
            "cli": "Raw MRI (Alt+V all overlays off)",
            "desc": "Percentile-windowed grayscale (0–99.5%).\nJet-black hypointense seed cores sit inside\nbright hydrated T2 mucilage gel.",
            "checks": [True, False, False, False],
            "active_row": 0,
            "sector": composite_slice(gray_full),
            "zoom": composite_slice(gray_zm),
            "show_ids": False,
        },
        {
            "step": "2 / 6",
            "name": "Silver Micro-GT (ColorLUT Fill)",
            "cli": "./view_kiwi_freeview.sh",
            "desc": "Default launch: bit-aligned uint16 IDs 1..98\nrendered with kiwi_seeds_ColorLUT.txt at\nopacity 0.58 so adjacent twins separate.",
            "checks": [True, True, False, False],
            "active_row": 1,
            "sector": composite_slice(gray_full, lut_fill_lbl=gt_full, lut_fill_alpha=GT_OPACITY, lut_rgb=lut_rgb),
            "zoom": composite_slice(gray_zm, lut_fill_lbl=gt_zm, lut_fill_alpha=GT_OPACITY, lut_rgb=lut_rgb),
            "show_ids": True,
        },
        {
            "step": "3 / 6",
            "name": "Silver Micro-GT (--outline Mode)",
            "cli": "./view_kiwi_freeview.sh --outline",
            "desc": "Crisp per-instance ColorLUT outlines\n(outline=1, opacity=0.95) leave dark 7T\nseed cores 100% unobstructed for audit.",
            "checks": [True, True, False, False],
            "active_row": 1,
            "sector": composite_slice(gray_full, lut_outline_lbl=gt_full, lut_outline_alpha=OUTLINE_OPACITY, lut_rgb=lut_rgb),
            "zoom": composite_slice(gray_zm, lut_outline_lbl=gt_zm, lut_outline_alpha=OUTLINE_OPACITY, lut_rgb=lut_rgb),
            "show_ids": True,
        },
        {
            "step": "4 / 6",
            "name": "Companion Yellow Outline",
            "cli": "Toggle Silver_MicroGT_Outline",
            "desc": "High-contrast binary yellow boundary\nRGB(255,235,59) at opacity 0.95—ideal for\nVoxel Edit brushstroke refinement.",
            "checks": [True, False, True, False],
            "active_row": 2,
            "sector": composite_slice(gray_full, yellow_outline_lbl=gt_full, yellow_outline_alpha=OUTLINE_OPACITY),
            "zoom": composite_slice(gray_zm, yellow_outline_lbl=gt_zm, yellow_outline_alpha=OUTLINE_OPACITY),
            "show_ids": False,
        },
        {
            "step": "5 / 6",
            "name": "Neurosymbolic 3D Prediction",
            "cli": "./view_kiwi_freeview.sh --with-pred",
            "desc": "Unmasked whole-fruit Neurosymbolic 3D\nprediction RGB(0,230,118) at opacity 0.48\n(749 raw 3D seeds; trained on y<168 only).",
            "checks": [True, False, False, True],
            "active_row": 3,
            "sector": composite_slice(gray_full, binary_fill_mask=pred_full > 0, binary_fill_rgb=PRED_GREEN_RGB, binary_fill_alpha=PRED_OPACITY),
            "zoom": composite_slice(gray_zm, binary_fill_mask=pred_zm > 0, binary_fill_rgb=PRED_GREEN_RGB, binary_fill_alpha=PRED_OPACITY),
            "show_ids": False,
        },
        {
            "step": "6 / 6",
            "name": "Direct Audit: Pred + GT Outline",
            "cli": "--with-pred + Yellow GT Outline",
            "desc": "Green Neurosymbolic prediction inside crisp\nyellow Silver Micro-GT outlines: immediate\nvisual verification (Test PQ=0.473, Prec=0.875).",
            "checks": [True, False, True, True],
            "active_row": 3,
            "sector": composite_slice(
                gray_full,
                binary_fill_mask=pred_full > 0, binary_fill_rgb=PRED_GREEN_RGB, binary_fill_alpha=0.52,
                yellow_outline_lbl=gt_full, yellow_outline_alpha=OUTLINE_OPACITY,
            ),
            "zoom": composite_slice(
                gray_zm,
                binary_fill_mask=pred_zm > 0, binary_fill_rgb=PRED_GREEN_RGB, binary_fill_alpha=0.52,
                yellow_outline_lbl=gt_zm, yellow_outline_alpha=OUTLINE_OPACITY,
            ),
            "show_ids": True,
        },
    ]

    volume_rows = [
        ("7T_Kiwi_RARE", "Grayscale 0–99.5%", "#c9d1d9"),
        ("Silver_MicroGT_98seeds", "ColorLUT (1..98)", "#ff5252"),
        ("Silver_MicroGT_Outline", "Yellow (255,235,59)", YELLOW_HEX),
        ("Neurosymbolic_3D_WholeFruit", "Green (0,230,118)", PRED_GREEN_HEX),
    ]

    fig = plt.figure(figsize=(11.8, 6.5), dpi=100, facecolor=BG_DARK)
    frames: list[Image.Image] = []
    durations: list[int] = []

    def render_state_frame(
        st: dict,
        sector_img: np.ndarray,
        zoom_img: np.ndarray,
        show_ids: bool,
    ) -> Image.Image:
        fig.clf()
        fig.patch.set_facecolor(BG_DARK)

        # Header
        fig.text(
            0.018, 0.958,
            "BMED365 7T Kiwi MRI  |  Freeview Interactive Layer & Outline Modes (Equator z = 64)",
            color=TEXT_MAIN, fontsize=11.8, fontweight="bold", va="center",
        )
        fig.text(
            0.018, 0.922,
            f"State {st['step']}:  {st['name']}   •   Press Alt+V in freeview to toggle selected overlay",
            color=TEXT_MUTED, fontsize=8.5, va="center",
        )
        fig.text(
            0.982, 0.946,
            f" {st['cli']} ",
            color="#0d1117", fontsize=8.2, fontweight="bold", family="monospace",
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=ACCENT_BLUE, edgecolor="none"),
        )

        # Layout: Left Freeview Volumes panel | Center Sector Axial | Right High-Mag Test Zoom
        ax_side = fig.add_axes([0.018, 0.145, 0.250, 0.725])
        ax_sec = fig.add_axes([0.305, 0.145, 0.335, 0.725])
        ax_zm = fig.add_axes([0.675, 0.145, 0.307, 0.725])
        ax_lut = fig.add_axes([0.018, 0.016, 0.964, 0.090])

        # --- Left sidebar: Freeview Volumes panel ---
        ax_side.set_facecolor(PANEL_BG)
        ax_side.set_xlim(0, 1)
        ax_side.set_ylim(0, 1)
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        for sp in ax_side.spines.values():
            sp.set_color(BORDER_COL)
            sp.set_linewidth(1.0)

        ax_side.text(0.05, 0.945, "FREEVIEW VOLUMES PANEL", color=ACCENT_BLUE, fontsize=8.3, fontweight="bold", va="center")
        ax_side.plot([0.04, 0.96], [0.905, 0.905], color=BORDER_COL, lw=0.8)

        for r_idx, (vname, vmeta, vcol) in enumerate(volume_rows):
            y_c = 0.815 - r_idx * 0.125
            checked = st["checks"][r_idx]
            is_act = r_idx == st["active_row"]
            row_bg = "#1f2937" if is_act else "#161b22"
            edge_c = ACCENT_BLUE if is_act else BORDER_COL
            ax_side.add_patch(mpatches.FancyBboxPatch(
                (0.04, y_c - 0.048), 0.92, 0.098,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=row_bg, edgecolor=edge_c, linewidth=1.0 if is_act else 0.6,
            ))
            # Checkbox
            cb_face = ACCENT_BLUE if checked else "#0d1117"
            ax_side.add_patch(mpatches.Rectangle(
                (0.07, y_c - 0.016), 0.055, 0.036,
                facecolor=cb_face, edgecolor=TEXT_MAIN if checked else TEXT_MUTED, linewidth=0.9,
            ))
            if checked:
                ax_side.text(0.098, y_c + 0.002, "✓", color="#0d1117", fontsize=8.0, fontweight="bold", ha="center", va="center")
            # Aspect-corrected color swatch square
            ax_side.add_patch(mpatches.Rectangle(
                (0.148, y_c - 0.015), 0.036, 0.030,
                facecolor=vcol, edgecolor="#0d1117", lw=0.6,
            ))
            sub_meta = vmeta
            if r_idx == 1:
                sub_meta = "ColorLUT (outline=1, α=0.95)" if "outline" in st["cli"].lower() and st["active_row"] == 1 else "ColorLUT (fill, α=0.58)"
            ax_side.text(0.205, y_c + 0.018, vname, color=TEXT_MAIN if checked else TEXT_MUTED, fontsize=7.5, fontweight="bold", va="center")
            ax_side.text(0.205, y_c - 0.022, sub_meta, color=TEXT_MUTED, fontsize=6.7, va="center")

        # Pedagogical note box inside sidebar
        ax_side.add_patch(mpatches.FancyBboxPatch(
            (0.04, 0.145), 0.92, 0.215,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#0d1117", edgecolor=BORDER_COL, linewidth=0.8,
        ))
        ax_side.text(0.075, 0.328, "WHAT TO INSPECT:", color=YELLOW_HEX, fontsize=7.6, fontweight="bold", va="center")
        ax_side.text(0.075, 0.285, st["desc"], color=TEXT_MAIN, fontsize=7.4, va="top", linespacing=1.35)

        # Step indicator pills at bottom of sidebar
        for s_i in range(len(states)):
            px = 0.12 + s_i * 0.15
            active_dot = (s_i + 1) == int(st["step"].split("/")[0].strip())
            ax_side.add_patch(mpatches.Circle(
                (px, 0.068), radius=0.024 if active_dot else 0.017,
                facecolor=ACCENT_BLUE if active_dot else "#262c36",
                edgecolor=TEXT_MAIN if active_dot else BORDER_COL, lw=0.7,
            ))
            ax_side.text(px, 0.068, f"{s_i+1}", color="#0d1117" if active_dot else TEXT_MUTED,
                         fontsize=6.5, fontweight="bold", ha="center", va="center")

        # --- Center panel: Equatorial sector ---
        ax_sec.imshow(sector_img, origin="lower", interpolation="nearest")
        ax_sec.plot([RX0, RX1, RX1, RX0, RX0], [RY0, RY0, RY1, RY1, RY0], color=ROI_RED, lw=1.2, ls="--")
        ax_sec.plot([RX0, RX1], [Y_SPLIT, Y_SPLIT], color=SPLIT_CYAN, lw=1.3, ls="--")
        ax_sec.plot([zx0, zx1, zx1, zx0, zx0], [zy0, zy0, zy1, zy1, zy0], color=YELLOW_HEX, lw=1.0, ls=":")
        ax_sec.text(RX1 - 2, RY0 + 3, "Train (y<168)", color="#7ee787", fontsize=7.3, fontweight="bold", ha="right",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#0d1117", edgecolor="none", alpha=0.72))
        ax_sec.text(RX1 - 2, RY1 - 7, "Test (y≥168)", color=SPLIT_CYAN, fontsize=7.3, fontweight="bold", ha="right",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#0d1117", edgecolor="none", alpha=0.72))
        ax_sec.set_xlim(96, 228)
        ax_sec.set_ylim(96, 228)
        ax_sec.set_xlabel("X voxel", color=TEXT_MUTED, fontsize=7.5)
        ax_sec.set_ylabel("Y voxel", color=TEXT_MUTED, fontsize=7.5)
        draw_scalebar(ax_sec, x_right=221, y_bottom=103, length_mm=5.0, spacing_x_mm=spacing[0])
        style_image_ax(ax_sec, "Equatorial Sector  (z = 64)", "80×80 ROI (red) • High-Mag crop (dotted yellow)")

        # --- Right panel: High-Mag Test-Wedge Zoom ---
        ax_zm.imshow(
            zoom_img, origin="lower", interpolation="nearest",
            extent=[zx0 - 0.5, zx1 - 0.5, zy0 - 0.5, zy1 - 0.5],
        )
        ax_zm.axhline(Y_SPLIT, color=SPLIT_CYAN, lw=1.3, ls="--")
        if show_ids:
            annotate_seed_ids_on_crop(ax_zm, gt_zm, zx0, zy0, lut_rgb, min_pixels=3, fontsize=6.6)
        ax_zm.set_xlim(zx0 - 0.5, zx1 - 0.5)
        ax_zm.set_ylim(zy0 - 0.5, zy1 - 0.5)
        ax_zm.set_xlabel("X voxel (Held-Out Test Wedge)", color=TEXT_MUTED, fontsize=7.5)
        ax_zm.set_ylabel("Y voxel", color=TEXT_MUTED, fontsize=7.5)
        draw_scalebar(ax_zm, x_right=zx1 - 2.5, y_bottom=zy0 + 2.2, length_mm=2.0, spacing_x_mm=spacing[0])
        style_image_ax(ax_zm, "High-Mag Voxel Alignment (Test)", "Unobstructed cores & pixel-exact LUT IDs")

        # Bottom ColorLUT strip
        draw_colorlut_strip(
            ax_lut, lut_rgb,
            extra_legend=[
                ("box", PRED_GREEN_HEX, "Neurosymbolic 3D Pred RGB(0,230,118) α=0.48"),
                ("outline", YELLOW_HEX, "Silver Micro-GT Outline RGB(255,235,59) α=0.95"),
                ("dash", ROI_RED, "Micro-GT ROI (80×80)"),
                ("dash", SPLIT_CYAN, "y_split = 168"),
            ],
        )
        return fig_to_pil(fig)

    for i, st in enumerate(states):
        # Main held frame
        frames.append(render_state_frame(st, st["sector"], st["zoom"], st["show_ids"]))
        durations.append(2100)
        # Smooth 1-step cross-fade transition to next state for pleasant pacing
        nxt = states[(i + 1) % len(states)]
        mid_sec = np.clip(0.5 * st["sector"] + 0.5 * nxt["sector"], 0.0, 1.0)
        mid_zm = np.clip(0.5 * st["zoom"] + 0.5 * nxt["zoom"], 0.0, 1.0)
        frames.append(render_state_frame(nxt, mid_sec, mid_zm, False))
        durations.append(160)

    plt.close(fig)
    return save_optimized_gif(frames, durations, out_path, lut_rgb=lut_rgb)


# ---------------------------------------------------------------------------
# GIF 3: Full 128-Slice Whole-Fruit Neurosymbolic 3D Prediction (--with-pred)
# ---------------------------------------------------------------------------

def build_gif_wholefruit_pred(
    mri_norm: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    affine: np.ndarray,
    lut_rgb: np.ndarray,
    out_path: Path,
) -> Path:
    print("Rendering GIF 3: Whole-Fruit 128-Slice Neurosymbolic 3D Flythrough...")
    asp_z = spacing[2] / spacing[0]

    # Precompute active 2D predicted seed count per slice z = 0..127
    z_all = np.arange(pred.shape[2])
    pred_counts = np.array([len(np.unique(pred[:, :, z])) - 1 for z in range(pred.shape[2])], dtype=np.int32)
    gt_counts = np.array([len(np.unique(gt[:, :, z])) - 1 for z in range(gt.shape[2])], dtype=np.int32)
    total_pred_3d = int(len(np.unique(pred)) - 1)

    # Pre-render static coronal cross-section at y=168 with Freeview --with-pred green RGB(0,230,118)
    cor_rgb = composite_slice(
        mri_norm[:, CURSOR_Y, :].T,
        binary_fill_mask=pred[:, CURSOR_Y, :].T > 0,
        binary_fill_rgb=PRED_GREEN_RGB,
        binary_fill_alpha=0.55,
        yellow_outline_lbl=gt[:, CURSOR_Y, :].T,
        yellow_outline_alpha=0.85,
    )

    # Sweep z = 12 .. 116 in steps of 2 (53 frames: covers both polar caps + equator!)
    z_seq = list(range(12, 117, 2))
    frames: list[Image.Image] = []
    durations: list[int] = []

    fig = plt.figure(figsize=(11.4, 6.8), dpi=100, facecolor=BG_DARK)

    for z in z_seq:
        fig.clf()
        fig.patch.set_facecolor(BG_DARK)

        z_mm = float((affine @ np.array([128, 128, z, 1.0]))[2])
        n_pred_z = int(pred_counts[z])
        n_gt_z = int(gt_counts[z])
        in_polar = (z < 16) or (z > 111)
        in_slab = (RZ0 <= z < RZ1)

        if in_polar:
            zone_txt, zone_col = "POLAR CAP  (0 FPs)", "#ffab40"
        elif in_slab:
            zone_txt, zone_col = f"EQUATORIAL MICRO-GT SLAB  (GT={n_gt_z})", YELLOW_HEX
        else:
            zone_txt, zone_col = "UNLABELED FRUIT  (Zero-Shot 360°)", PRED_GREEN_HEX

        # Header
        fig.text(
            0.018, 0.966,
            f"BMED365 7T Kiwi MRI  |  Unmasked Whole-Fruit Neurosymbolic 3D Prediction ({total_pred_3d} Seeds)",
            color=TEXT_MAIN, fontsize=11.4, fontweight="bold", va="center",
        )
        fig.text(
            0.018, 0.935,
            f"Slice z = {z:03d} / 127  ({z_mm:+5.1f} mm)   •   Active 2D Pred Seeds: {n_pred_z:2d}   •   "
            f"Trained on 0.083% of berry (59 seeds) → Calibrated Total ≈ 1,223 seeds",
            color=TEXT_MUTED, fontsize=8.2, va="center",
        )
        fig.text(
            0.982, 0.960,
            " ./view_kiwi_freeview.sh --with-pred ",
            color="#0d1117", fontsize=8.2, fontweight="bold", family="monospace",
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PRED_GREEN_HEX, edgecolor="none"),
        )

        # Generous margins above LUT strip and below header banner
        ax_ax = fig.add_axes([0.048, 0.165, 0.475, 0.675])
        ax_co = fig.add_axes([0.585, 0.500, 0.395, 0.340])
        ax_pr = fig.add_axes([0.585, 0.165, 0.395, 0.245])
        ax_lut = fig.add_axes([0.018, 0.014, 0.964, 0.088])

        # --- Left panel: Full axial slice with all 749 Pred seeds colored via kiwi_seeds_ColorLUT.txt ---
        pred_z = pred[:, :, z].T
        gt_z = gt[:, :, z].T if in_slab else None
        ax_rgb = composite_slice(
            mri_norm[:, :, z].T,
            lut_fill_lbl=pred_z,
            lut_fill_alpha=0.62,
            lut_outline_lbl=pred_z,
            lut_outline_alpha=0.75,
            yellow_outline_lbl=gt_z,
            yellow_outline_alpha=0.95,
            lut_rgb=lut_rgb,
        )
        ax_ax.imshow(ax_rgb, origin="lower", interpolation="nearest")
        if in_slab:
            ax_ax.plot([RX0, RX1, RX1, RX0, RX0], [RY0, RY0, RY1, RY1, RY0], color=ROI_RED, lw=1.35, ls="--")
            ax_ax.plot([RX0, RX1], [Y_SPLIT, Y_SPLIT], color=SPLIT_CYAN, lw=1.35, ls="--")
            ax_ax.text(RX1 - 2, RY1 + 4, "Micro-GT ROI (Yellow = Silver GT)", color=YELLOW_HEX,
                       fontsize=7.3, fontweight="bold", ha="right",
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none", alpha=0.80))

        ax_ax.set_xlim(18, 238)
        ax_ax.set_ylim(18, 238)
        ax_ax.set_xlabel("X voxel (Full Axial Cross-Section)", color=TEXT_MUTED, fontsize=7.6)
        ax_ax.set_ylabel("Y voxel", color=TEXT_MUTED, fontsize=7.6)
        ax_ax.text(
            24, 228, zone_txt, color=zone_col, fontsize=8.0, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0d1117", edgecolor=zone_col, lw=0.8, alpha=0.88),
        )
        draw_scalebar(ax_ax, x_right=230, y_bottom=26, length_mm=10.0, spacing_x_mm=spacing[0])
        style_image_ax(
            ax_ax,
            f"Whole-Fruit Axial Slice  (z = {z:03d})",
            "Pred IDs 1..749 colored by kiwi_seeds_ColorLUT.txt (Yellow = Silver GT in z=56..71)",
        )

        # --- Top-Right panel: Coronal whole-fruit profile in Freeview green RGB(0,230,118) ---
        ax_co.imshow(cor_rgb, origin="lower", aspect=asp_z, interpolation="nearest")
        ax_co.axhspan(RZ0, RZ1 - 1, color=ACCENT_BLUE, alpha=0.14)
        ax_co.axhline(16, color=ROI_RED, lw=1.0, ls="--", alpha=0.85)
        ax_co.axhline(111, color=ROI_RED, lw=1.0, ls="--", alpha=0.85)
        ax_co.plot([RX0, RX1, RX1, RX0, RX0], [RZ0, RZ0, RZ1, RZ1, RZ0], color=ROI_RED, lw=1.1, ls=":")
        ax_co.axhline(z, color="#ffffff", lw=1.35, alpha=0.95)
        ax_co.set_xlim(24, 232)
        ax_co.set_ylim(6, 122)
        ax_co.set_xlabel("X voxel", color=TEXT_MUTED, fontsize=7.3, labelpad=2)
        ax_co.set_ylabel("Z slice (2.13× mm)", color=TEXT_MUTED, fontsize=7.3, labelpad=2)
        ax_co.text(28, 114, "Polar caps (z<16, z>111): 0 FPs", color=ROI_RED, fontsize=7.1, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.14", facecolor="#0d1117", edgecolor="none", alpha=0.78))
        style_image_ax(ax_co, f"Coronal Profile  (y = {CURSOR_Y})", "Green: --with-pred RGB(0,230,118) | White: slice z")

        # --- Bottom-Right panel: Live 128-slice axial seed count curve ---
        ax_pr.set_facecolor(PANEL_BG)
        ax_pr.axvspan(0, 16, color=ROI_RED, alpha=0.12)
        ax_pr.axvspan(111, 127, color=ROI_RED, alpha=0.12)
        ax_pr.axvspan(RZ0, RZ1 - 1, color=ACCENT_BLUE, alpha=0.18, label="Labeled slab (z=56..71)")
        ax_pr.fill_between(z_all, pred_counts, color=PRED_GREEN_HEX, alpha=0.22)
        ax_pr.plot(z_all, pred_counts, color=PRED_GREEN_HEX, lw=1.8, label="Neurosymbolic 3D (749 raw)")
        ax_pr.axvline(z, color="#ffffff", lw=1.2, ls="--", alpha=0.9)
        ax_pr.scatter([z], [n_pred_z], color=YELLOW_HEX, edgecolor="#0d1117", s=46, zorder=6)
        ax_pr.set_xlim(0, 127)
        ax_pr.set_ylim(0, 58)
        ax_pr.set_xlabel("Axial Slice Index z (0..127)", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_pr.set_ylabel("2D Seeds / Slice", color=TEXT_MUTED, fontsize=7.4, labelpad=2)
        ax_pr.grid(color=BORDER_COL, lw=0.5, alpha=0.55)
        ax_pr.tick_params(colors=TEXT_MUTED, labelsize=7.2)
        for sp in ax_pr.spines.values():
            sp.set_color(BORDER_COL)
        ax_pr.text(
            6, 49, "0 polar FPs", color="#ff8a80", fontsize=7.0, fontweight="bold",
        )
        ax_pr.text(
            122, 49, "0 polar FPs", color="#ff8a80", fontsize=7.0, fontweight="bold", ha="right",
        )
        ax_pr.legend(loc="upper center", fontsize=6.9, facecolor="#0d1117", edgecolor=BORDER_COL, labelcolor=TEXT_MAIN, framealpha=0.85)
        ax_pr.set_title("Unmasked 128-Slice Organ Profile (-99.92% OOD vs Baseline)", color=TEXT_MAIN, fontsize=8.3, fontweight="bold", loc="left", pad=4)

        # Bottom ColorLUT strip
        draw_colorlut_strip(
            ax_lut, lut_rgb,
            extra_legend=[
                ("box", PRED_GREEN_HEX, "Coronal --with-pred RGB(0,230,118)"),
                ("outline", YELLOW_HEX, "Silver Micro-GT Outline (1..98, z=56..71)"),
                ("dash", ROI_RED, "Polar limits (z=16, 111) & ROI"),
                ("dash", SPLIT_CYAN, "y_split = 168"),
            ],
        )

        frames.append(fig_to_pil(fig))
        durations.append(550 if z == CURSOR_Z else (380 if z in (z_seq[0], z_seq[-1]) else 145))

    plt.close(fig)
    return save_optimized_gif(frames, durations, out_path, lut_rgb=lut_rgb)


# ---------------------------------------------------------------------------
# GIF 4: 3D Smoothed Seed Isosurfaces Turntable (--with-pred --3d)
# ---------------------------------------------------------------------------

def build_gt_3d_meshes(
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    lut_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-seed smoothed 3D isosurface triangles (physical mm) colored by kiwi_seeds_ColorLUT.txt."""
    pad = 4
    x0, x1 = RX0 - pad, RX1 + pad
    y0, y1 = RY0 - pad, RY1 + pad
    z0, z1 = RZ0 - pad, RZ1 + pad
    sub = gt[x0:x1, y0:y1, z0:z1]

    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    fcols_list: list[np.ndarray] = []
    v_offset = 0

    for sid in np.unique(sub):
        if sid == 0:
            continue
        m = (sub == sid).astype(np.float32)
        # Mild anisotropic Gaussian smoothing matching Freeview isosurface_smooth=2
        m_sm = ndi.gaussian_filter(m, sigma=(0.85, 0.85, 0.55))
        if m_sm.max() < 0.34:
            continue
        v, f, _, _ = measure.marching_cubes(m_sm, level=0.34, spacing=spacing)
        v[:, 0] += x0 * spacing[0]
        v[:, 1] += y0 * spacing[1]
        v[:, 2] += z0 * spacing[2]
        verts_list.append(v.astype(np.float32))
        faces_list.append((f + v_offset).astype(np.int32))
        col = lut_rgb[min(int(sid), len(lut_rgb) - 1)]
        fcols_list.append(np.broadcast_to(col[None, :], (len(f), 3)).copy())
        v_offset += len(v)

    return np.vstack(verts_list), np.vstack(faces_list), np.vstack(fcols_list)


def build_wholefruit_3d_meshes(
    vol: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Extract smoothed whole-fruit Neurosymbolic 3D seed isosurfaces + translucent fruit shell."""
    # Downsample (x, y) by 2x after mild smoothing -> nearly isotropic (0.469, 0.469, 0.500) mm
    # Preserves all 749 seeds cleanly across every z-slice while keeping face count crisp & fast!
    pm = ndi.gaussian_filter((pred > 0).astype(np.float32), sigma=(0.85, 0.85, 0.45))
    pm_ds = pm[::2, ::2, :]
    sp_ds = (spacing[0] * 2.0, spacing[1] * 2.0, spacing[2])
    pv, pf, _, _ = measure.marching_cubes(pm_ds, level=0.32, spacing=sp_ds)

    # Intact kiwifruit outer ghost shell (smooth 4x4x2 downsample -> ~1 mm isotropic)
    fmask = extract_fruit_mask(vol, spacing).astype(np.float32)
    fm_sm = ndi.gaussian_filter(fmask, sigma=(2.2, 2.2, 1.4))
    fm_ds = fm_sm[::4, ::4, ::2]
    sp_shell = (spacing[0] * 4.0, spacing[1] * 4.0, spacing[2] * 2.0)
    sv, sf, _, _ = measure.marching_cubes(fm_ds, level=0.45, spacing=sp_shell)
    return (pv.astype(np.float32), pf.astype(np.int32)), (sv.astype(np.float32), sf.astype(np.int32))


def rotation_matrix_3d(az_deg: float, el_deg: float) -> np.ndarray:
    """Return 3x3 rotation matrix mapping world (X, Y, Z) -> camera (right, up, depth)."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    # Rotate around Z by -az, then tilt by elevation around horizontal axis
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    # Camera right (u), camera up (v), camera view depth (w, pointing toward viewer)
    u = np.array([-sa, ca, 0.0], dtype=np.float32)
    v = np.array([-ca * se, -sa * se, ce], dtype=np.float32)
    w = np.array([ca * ce, sa * ce, se], dtype=np.float32)
    return np.stack([u, v, w], axis=0)  # (3, 3)


def render_shaded_mesh_2d(
    ax: plt.Axes,
    verts: np.ndarray,
    faces: np.ndarray,
    face_rgb: np.ndarray,
    center: np.ndarray,
    R: np.ndarray,
    *,
    alpha: float = 1.0,
    fresnel_ghost: bool = False,
    light_dir: np.ndarray = np.array([0.38, 0.55, 0.74], dtype=np.float32),
    zorder: float = 2.0,
) -> None:
    """Project, backface-cull, Blinn-Phong shade, depth-sort, and render triangles into a 2D axes."""
    vc = (verts - center[None, :]) @ R.T  # (V, 3) -> (u, v, depth)
    tri = vc[faces]                        # (F, 3, 3)

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(e1, e2)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = n_len[:, 0] > 1e-8
    normals = normals[valid] / n_len[valid]
    tri = tri[valid]
    fcols = face_rgb[valid] if face_rgb.ndim == 2 else face_rgb

    if not fresnel_ghost:
        # Backface cull for solid opaque seeds (facing viewer means normal_w > -0.05)
        front = normals[:, 2] > -0.05
        tri = tri[front]
        normals = normals[front]
        if face_rgb.ndim == 2:
            fcols = fcols[front]

    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth)  # painter's algorithm: far -> near
    tri = tri[order]
    normals = normals[order]
    if face_rgb.ndim == 2:
        fcols = fcols[order]

    ld = light_dir / np.linalg.norm(light_dir)
    if fresnel_ghost:
        # Medical glass/ghost shell: bright on grazing silhouette rim, near-invisible face-on
        ndotv = np.abs(normals[:, 2])
        rim = np.power(1.0 - ndotv, 1.9)
        rgba = np.empty((len(tri), 4), dtype=np.float32)
        rgba[:, :3] = (0.55 + 0.45 * rim[:, None]) * np.array([0.76, 0.85, 0.96], dtype=np.float32)
        rgba[:, 3] = np.clip(0.015 + 0.24 * rim, 0.01, 0.30)
    else:
        # Blinn-Phong directional + rim shading
        ndotl = np.clip(np.abs(normals @ ld), 0.0, 1.0)
        # Secondary fill light from opposite lower-left for clean dark-side readability
        ld2 = np.array([-0.55, -0.30, 0.78], dtype=np.float32)
        ld2 /= np.linalg.norm(ld2)
        fill = np.clip(np.abs(normals @ ld2), 0.0, 1.0)
        # Soft specular highlight
        hvec = ld + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        hvec /= np.linalg.norm(hvec)
        spec = np.power(np.clip(np.abs(normals @ hvec), 0.0, 1.0), 22.0) * 0.26
        shade = 0.24 + 0.64 * ndotl + 0.16 * fill
        shaded_rgb = np.clip(fcols * shade[:, None] + spec[:, None], 0.0, 1.0)
        rgba = np.empty((len(tri), 4), dtype=np.float32)
        rgba[:, :3] = shaded_rgb
        rgba[:, 3] = alpha

    poly = PolyCollection(
        tri[:, :, :2],
        facecolors=rgba,
        edgecolors="none",
        antialiaseds=False,
        zorder=zorder,
    )
    ax.add_collection(poly)


def draw_3d_box_wireframe(
    ax: plt.Axes,
    bounds_mm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    center: np.ndarray,
    R: np.ndarray,
    color: str = ROI_RED,
    lw: float = 1.1,
    ls: str = "--",
    alpha: float = 0.85,
    zorder: float = 5.0,
) -> None:
    """Project and draw a 3D bounding box wireframe."""
    (x0, x1), (y0, y1), (z0, z1) = bounds_mm
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    proj = (corners - center[None, :]) @ R.T
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [proj[a, 0], proj[b, 0]], [proj[a, 1], proj[b, 1]],
            color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
        )


def draw_3d_plane(
    ax: plt.Axes,
    pts_mm: np.ndarray,
    center: np.ndarray,
    R: np.ndarray,
    facecolor: str = SPLIT_CYAN,
    edgecolor: str = SPLIT_CYAN,
    alpha: float = 0.14,
    zorder: float = 1.5,
) -> None:
    proj = (pts_mm - center[None, :]) @ R.T
    ax.add_patch(mpatches.Polygon(
        proj[:, :2], closed=True, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.1, linestyle="--", alpha=alpha, zorder=zorder,
    ))
    # Crisp dashed border around the dividing plane
    p_closed = np.vstack([proj[:, :2], proj[0:1, :2]])
    ax.plot(p_closed[:, 0], p_closed[:, 1], color=edgecolor, lw=1.1, ls="--", alpha=0.80, zorder=zorder + 4)


def draw_3d_triad(ax: plt.Axes, R: np.ndarray, origin_xy: tuple[float, float], length: float) -> None:
    """Draw a rotating 3D XYZ axis orientation triad in the corner of a 3D viewport."""
    ox, oy = origin_xy
    axes_spec = [
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), "#ff5252", "X"),
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), "#00e676", "Y"),
        (np.array([0.0, 0.0, 1.0], dtype=np.float32), "#40c4ff", "Z"),
    ]
    # Sort by depth so nearer arrows draw on top
    items = []
    for vec, col, lbl in axes_spec:
        p = vec @ R.T
        items.append((float(p[2]), p, col, lbl))
    items.sort(key=lambda t: t[0])
    for _, p, col, lbl in items:
        ax.plot([ox, ox + p[0] * length], [oy, oy + p[1] * length], color=col, lw=1.8, zorder=10)
        ax.text(
            ox + p[0] * length * 1.28, oy + p[1] * length * 1.28, lbl,
            color=col, fontsize=7.6, fontweight="bold", ha="center", va="center", zorder=11,
        )


def build_gif_3d_isosurfaces(
    vol: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    lut_rgb: np.ndarray,
    out_path: Path,
) -> Path:
    print("Rendering GIF 4: 3D Smoothed Seed Isosurfaces Turntable (--with-pred --3d)...")
    t0 = time.time()
    gt_v, gt_f, gt_c = build_gt_3d_meshes(gt, spacing, lut_rgb)
    (pr_v, pr_f), (sh_v, sh_f) = build_wholefruit_3d_meshes(vol, pred, spacing)
    print(f"  Meshes built in {time.time() - t0:.2f}s | GT: {len(gt_f)} faces | Pred: {len(pr_f)} faces | Shell: {len(sh_f)} faces")

    # Physical mm ROI bounds & centers
    roi_bounds_mm = (
        (RX0 * spacing[0], RX1 * spacing[0]),
        (RY0 * spacing[1], RY1 * spacing[1]),
        (RZ0 * spacing[2], RZ1 * spacing[2]),
    )
    gt_center = np.array([
        0.5 * (roi_bounds_mm[0][0] + roi_bounds_mm[0][1]),
        0.5 * (roi_bounds_mm[1][0] + roi_bounds_mm[1][1]),
        0.5 * (roi_bounds_mm[2][0] + roi_bounds_mm[2][1]),
    ], dtype=np.float32)
    fruit_center = 0.5 * (sh_v.min(axis=0) + sh_v.max(axis=0))

    # y_split = 168 dividing plane inside ROI (physical mm)
    y_sp_mm = Y_SPLIT * spacing[1]
    split_plane_mm = np.array([
        [roi_bounds_mm[0][0], y_sp_mm, roi_bounds_mm[2][0]],
        [roi_bounds_mm[0][1], y_sp_mm, roi_bounds_mm[2][0]],
        [roi_bounds_mm[0][1], y_sp_mm, roi_bounds_mm[2][1]],
        [roi_bounds_mm[0][0], y_sp_mm, roi_bounds_mm[2][1]],
    ], dtype=np.float32)

    # Polar-cap marker rings (z=16 and z=111) projected onto fruit equatorial radius
    theta_ring = np.linspace(0, 2 * np.pi, 72)
    r_polar = 15.5
    polar_lo = np.stack([
        fruit_center[0] + r_polar * np.cos(theta_ring),
        fruit_center[1] + r_polar * np.sin(theta_ring),
        np.full_like(theta_ring, 16.0 * spacing[2]),
    ], axis=1).astype(np.float32)
    polar_hi = np.stack([
        fruit_center[0] + r_polar * np.cos(theta_ring),
        fruit_center[1] + r_polar * np.sin(theta_ring),
        np.full_like(theta_ring, 111.0 * spacing[2]),
    ], axis=1).astype(np.float32)

    # 48 smooth 360° turntable frames (7.5° steps)
    n_frames = 48
    el_deg = 22.0
    frames: list[Image.Image] = []
    durations: list[int] = []

    fig = plt.figure(figsize=(11.4, 6.6), dpi=100, facecolor=BG_DARK)

    for k in range(n_frames):
        az_deg = (k * 360.0 / n_frames + 35.0) % 360.0
        R = rotation_matrix_3d(az_deg, el_deg)

        fig.clf()
        fig.patch.set_facecolor(BG_DARK)

        # Header
        fig.text(
            0.018, 0.960,
            "BMED365 7T Kiwi MRI  |  3D Smoothed Seed Isosurfaces (Physical mm)",
            color=TEXT_MAIN, fontsize=11.4, fontweight="bold", va="center",
        )
        fig.text(
            0.018, 0.925,
            f"Freeview isosurface_smooth=2 equivalent   •   Azimuth: {az_deg:5.1f}°  |  Elevation: {el_deg:.0f}°   •   "
            f"Voxel spacing: 0.234 × 0.234 × 0.500 mm",
            color=TEXT_MUTED, fontsize=8.3, va="center",
        )
        fig.text(
            0.982, 0.955,
            " ./view_kiwi_freeview.sh --with-pred --3d ",
            color="#0d1117", fontsize=8.1, fontweight="bold", family="monospace",
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PRED_GREEN_HEX, edgecolor="none"),
        )

        ax_l = fig.add_axes([0.024, 0.135, 0.462, 0.735])
        ax_r = fig.add_axes([0.514, 0.135, 0.462, 0.735])
        ax_lut = fig.add_axes([0.018, 0.014, 0.964, 0.092])

        # --- Left 3D viewport: 98 Silver Micro-GT Seeds in kiwi_seeds_ColorLUT.txt ---
        ax_l.set_facecolor("#090d13")
        ax_l.set_aspect("equal")
        ax_l.set_xlim(-15.0, 15.0)
        ax_l.set_ylim(-12.2, 14.2)
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        for sp in ax_l.spines.values():
            sp.set_color(BORDER_COL)

        draw_3d_plane(ax_l, split_plane_mm, gt_center, R, facecolor=SPLIT_CYAN, edgecolor=SPLIT_CYAN, alpha=0.14, zorder=1.5)
        render_shaded_mesh_2d(ax_l, gt_v, gt_f, gt_c, gt_center, R, alpha=1.0, zorder=3.0)
        draw_3d_box_wireframe(ax_l, roi_bounds_mm, gt_center, R, color=ROI_RED, lw=1.05, ls="--", alpha=0.80, zorder=4.5)
        draw_3d_triad(ax_l, R, origin_xy=(-11.6, -8.8), length=2.4)

        # Physical 5 mm scale bar (mm coords directly!)
        ax_l.plot([8.2, 13.2], [-10.4, -10.4], color="#ffffff", lw=2.4, zorder=10)
        ax_l.text(10.7, -9.8, "5 mm", color="#ffffff", fontsize=7.6, fontweight="bold", ha="center", va="bottom", zorder=10)
        ax_l.text(
            -14.2, 13.3,
            "98 Silver Micro-GT Seeds (IDs 1..98)  •  ROI: 18.75 × 18.75 × 8.00 mm\nTrain y<168 (59 seeds)  |  Held-Out Test y≥168 (52 seeds; 13 straddle)",
            color=TEXT_MAIN, fontsize=7.3, va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#0d1117", edgecolor=BORDER_COL, alpha=0.86),
            zorder=10,
        )
        ax_l.set_title("Silver Micro-GT Sector  (kiwi_seeds_ColorLUT.txt)", color=TEXT_MAIN, fontsize=9.2, fontweight="bold", loc="left", pad=5)

        # --- Right 3D viewport: 749 Whole-Fruit Neurosymbolic 3D Seeds + Translucent Shell ---
        ax_r.set_facecolor("#090d13")
        ax_r.set_aspect("equal")
        ax_r.set_xlim(-39.5, 39.5)
        ax_r.set_ylim(-33.5, 38.5)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_color(BORDER_COL)

        # Render 749 Neurosymbolic seeds in Freeview green RGB(0,230,118), then translucent ghost shell on top
        render_shaded_mesh_2d(ax_r, pr_v, pr_f, PRED_GREEN_RGB, fruit_center, R, alpha=1.0, zorder=2.5)
        render_shaded_mesh_2d(ax_r, sh_v, sh_f, np.array([0.8, 0.9, 1.0], dtype=np.float32), fruit_center, R, fresnel_ghost=True, zorder=4.0)
        # Embed the 1.22% red ROI wireframe box inside the whole fruit for dramatic scale context!
        draw_3d_box_wireframe(ax_r, roi_bounds_mm, fruit_center, R, color=ROI_RED, lw=1.2, ls="-", alpha=0.92, zorder=5.0)
        # Polar-cap marker rings (z=16 and z=111)
        for ring in (polar_lo, polar_hi):
            pr_ring = (ring - fruit_center[None, :]) @ R.T
            ax_r.plot(pr_ring[:, 0], pr_ring[:, 1], color="#ffab40", lw=0.85, ls=":", alpha=0.72, zorder=4.8)

        draw_3d_triad(ax_r, R, origin_xy=(-31.0, -24.5), length=6.2)
        ax_r.plot([24.0, 34.0], [-29.0, -29.0], color="#ffffff", lw=2.4, zorder=10)
        ax_r.text(29.0, -27.2, "10 mm", color="#ffffff", fontsize=7.6, fontweight="bold", ha="center", va="bottom", zorder=10)
        ax_r.text(
            -37.5, 36.5,
            "749 Raw 3D Seeds (~1,223 Calibrated)  •  Berry vol: 85.97 cm³\nSeed load: 0.73%  •  0 polar FPs (amber dotted z=16, 111)",
            color=TEXT_MAIN, fontsize=7.4, va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#0d1117", edgecolor=BORDER_COL, alpha=0.86),
            zorder=10,
        )
        ax_r.set_title("Whole-Fruit Neurosymbolic 3D  (RGB 0,230,118 + Ghost Shell)", color=TEXT_MAIN, fontsize=9.2, fontweight="bold", loc="left", pad=5)

        # Bottom ColorLUT strip
        draw_colorlut_strip(
            ax_lut, lut_rgb,
            extra_legend=[
                ("box", PRED_GREEN_HEX, "Neurosymbolic 3D Isosurfaces RGB(0,230,118)"),
                ("outline", ROI_RED, "1.22% Micro-GT ROI Box"),
                ("dash", SPLIT_CYAN, "y_split = 168 Plane"),
                ("dash", "#ffab40", "Polar Caps (z=16, 111)"),
            ],
        )

        frames.append(fig_to_pil(fig))
        durations.append(115)

    plt.close(fig)
    return save_optimized_gif(frames, durations, out_path, lut_rgb=lut_rgb)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    kiwi_dir = script_dir.parent
    default_out = kiwi_dir / "docs" / "neurosymbolic-report"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gt",
        type=Path,
        default=kiwi_dir / "data" / "kiwi_micro_gt_labels.nii.gz",
        help="Path to 3D Micro-GT uint16 NIfTI mask (default: data/kiwi_micro_gt_labels.nii.gz)",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=default_out / "results" / "kiwi_neurosymbolic_pred_labels.nii.gz",
        help="Path to whole-fruit Neurosymbolic 3D prediction NIfTI mask",
    )
    parser.add_argument(
        "--lut",
        type=Path,
        default=script_dir / "kiwi_seeds_ColorLUT.txt",
        help="Path to FreeSurfer-format ColorLUT (default: scripts/kiwi_seeds_ColorLUT.txt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for animated .gif files (default: docs/neurosymbolic-report)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["triplanar", "outline", "wholefruit", "3d"],
        default=["triplanar", "outline", "wholefruit", "3d"],
        help="Subset of GIFs to render (default: all four)",
    )
    args = parser.parse_args()

    mri_path = kiwi_dir / "data" / "kiwi_20050217191129_131073_RARE_high_res_axial.nii.gz"
    for req in (mri_path, args.gt, args.pred, args.lut):
        if not req.exists():
            raise FileNotFoundError(f"Required input not found: {req}")

    t_start = time.time()
    print(f"Loading NIfTI volumes & ColorLUT from {kiwi_dir} ...")
    mri_img = nib.load(str(mri_path))
    mri_raw = mri_img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in mri_img.header.get_zooms()[:3])
    affine = mri_img.affine.copy()

    gt_img = nib.load(str(args.gt))
    gt = np.rint(np.asarray(gt_img.dataobj)).astype(np.uint16)
    pred_img = nib.load(str(args.pred))
    pred = np.rint(np.asarray(pred_img.dataobj)).astype(np.uint16)

    max_diff_gt = float(np.max(np.abs(affine - gt_img.affine)))
    max_diff_pr = float(np.max(np.abs(affine - pred_img.affine)))
    print(f"  Shape: {mri_raw.shape} | Spacing: {spacing} mm | Affine diff: GT={max_diff_gt:.1e}, Pred={max_diff_pr:.1e}")

    lut_rgb, _ = load_color_lut(args.lut)
    mri_norm = normalize_mri_percentile(mri_raw, 0.0, 99.5)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if "triplanar" in args.only:
        build_gif_triplanar(mri_norm, gt, spacing, affine, lut_rgb, out_dir / "kiwi_freeview_microgt_triplanar.gif")
    if "outline" in args.only:
        build_gif_outline_layers(mri_norm, gt, pred, spacing, lut_rgb, out_dir / "kiwi_freeview_outline_layers.gif")
    if "wholefruit" in args.only:
        build_gif_wholefruit_pred(mri_norm, gt, pred, spacing, affine, lut_rgb, out_dir / "kiwi_freeview_wholefruit_pred.gif")
    if "3d" in args.only:
        build_gif_3d_isosurfaces(mri_raw, gt, pred, spacing, lut_rgb, out_dir / "kiwi_freeview_3d_isosurfaces.gif")

    print(f"Done in {time.time() - t_start:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
