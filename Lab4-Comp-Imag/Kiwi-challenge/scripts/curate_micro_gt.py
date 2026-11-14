#!/usr/bin/env python3
"""Generate the semi-automated ("silver-standard") 3D Micro-Ground-Truth sector for the Kiwi Challenge.

Usage (from repository root):
    uv run python Lab4-Comp-Imag/Kiwi-challenge/scripts/curate_micro_gt.py
    uv run python Lab4-Comp-Imag/Kiwi-challenge/scripts/curate_micro_gt.py --audit-png /tmp/micro_gt_audit.png

Provenance & Pedagogical Note:
    This script builds a reproducible 3D seed-instance reference mask (`data/kiwi_micro_gt_labels.nii.gz`)
    and companion metadata file (`data/kiwi_micro_gt_roi.json`) inside an 80 x 80 x 16 voxel equatorial
    sector (`x=[128, 208), y=[128, 208), z=[56, 72)`, 18.75 x 18.75 x 8.0 mm^3).

    Because the mask is produced by a multi-stage 3D physical-space pipeline (padded multi-scale 3D LoG +
    anisotropic ellipsoidal closing + bright-locule envelope + 3D marker-controlled watershed + physical-mm
    K-means pair splitting) refined across three 2D slice visual-audit passes rather than painted manually
    voxel-by-voxel, it represents a **semi-automated "silver-standard" reference**. Its NIfTI header and
    affine match the raw 7 T MRI scan exactly so it can be opened directly in ITK-SNAP or 3D Slicer for
    3D visual inspection or manual brushstroke refinement into a human-certified gold standard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation
from sklearn.cluster import KMeans


def build_ellipsoid_footprint(rx_mm: float, rz_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """Create a 3D binary ellipsoidal structuring element in physical millimeter coordinates."""
    rx = int(np.ceil(rx_mm / spacing[0]))
    ry = int(np.ceil(rx_mm / spacing[1]))
    rz = int(np.ceil(rz_mm / spacing[2]))
    xx, yy, zz = np.ogrid[-rx : rx + 1, -ry : ry + 1, -rz : rz + 1]
    return ((xx * spacing[0] / rx_mm) ** 2 + (yy * spacing[1] / rx_mm) ** 2 + (zz * spacing[2] / rz_mm) ** 2) <= 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--audit-png",
        type=Path,
        default=None,
        help="Optional path to save an 8-slice 2-row visual audit PNG (Raw vs. Micro-GT overlay).",
    )
    args = parser.parse_args()

    kiwi_root = Path(__file__).resolve().parents[1]
    nii_path = kiwi_root / "data" / "kiwi_20050217191129_131073_RARE_high_res_axial.nii.gz"
    gt_nii_path = kiwi_root / "data" / "kiwi_micro_gt_labels.nii.gz"
    gt_json_path = kiwi_root / "data" / "kiwi_micro_gt_roi.json"

    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])  # (0.234375, 0.234375, 0.5) mm
    vox_vol = float(np.prod(spacing))

    # 1. Whole-fruit mask & slice-by-slice normalized radial coordinate r / R(z)
    vol_sm = ndi.gaussian_filter(vol, sigma=(1.2 / spacing[0], 1.2 / spacing[1], 1.2 / spacing[2]))
    fg_raw = vol_sm > (filters.threshold_otsu(vol_sm[::2, ::2, ::2]) * 0.35)
    fg_fill = np.zeros_like(fg_raw)
    for z in range(vol.shape[2]):
        fg_fill[:, :, z] = ndi.binary_fill_holes(fg_raw[:, :, z])
    fg_fill = ndi.binary_fill_holes(fg_fill)
    lbl_fg, _ = ndi.label(fg_fill)
    fruit_mask = lbl_fg == (1 + int(np.argmax(np.bincount(lbl_fg.ravel())[1:])))

    r_norm = np.full(vol.shape, 1.5, dtype=np.float32)
    ix, iy = np.indices(vol.shape[:2], dtype=np.float32)
    for z in range(vol.shape[2]):
        m = fruit_mask[:, :, z]
        if m.sum() < 50:
            continue
        cx, cy = ndi.center_of_mass(m)
        r_eq = np.sqrt(m.sum() / np.pi)
        r_norm[:, :, z] = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2) / max(r_eq, 1.0)

    # 2. Extract padded 3D sub-volume around ROI x=[128, 208), y=[128, 208), z=[56, 72)
    x0, x1 = 128, 208
    y0, y1 = 128, 208
    z0, z1 = 56, 72
    pad_xy, pad_z = 8, 5

    sub = vol[x0 - pad_xy : x1 + pad_xy, y0 - pad_xy : y1 + pad_xy, z0 - pad_z : z1 + pad_z]
    sub_r = r_norm[x0 - pad_xy : x1 + pad_xy, y0 - pad_xy : y1 + pad_xy, z0 - pad_z : z1 + pad_z]

    # 3. Hybrid multi-scale 3D Laplacian-of-Gaussian + anisotropic ellipsoidal black top-hat
    log_stack = []
    for s_mm in (0.34, 0.44, 0.56):
        sig = (s_mm / spacing[0], s_mm / spacing[1], (s_mm * 1.30) / spacing[2])
        log_stack.append((s_mm**2) * ndi.gaussian_laplace(sub, sigma=sig))
    log_max = np.max(log_stack, axis=0)

    fp_close = build_ellipsoid_footprint(rx_mm=1.15, rz_mm=1.50, spacing=spacing)
    v_den = ndi.gaussian_filter(sub, sigma=(0.16 / spacing[0], 0.16 / spacing[1], 0.20 / spacing[2]))
    closed = ndi.grey_closing(v_den, footprint=fp_close)
    bright_env = ndi.gaussian_filter(closed, sigma=(0.95 / spacing[0], 0.95 / spacing[1], 0.95 / spacing[2]))

    rel_drop = (closed - v_den) / (bright_env + 800.0)
    norm_log = np.maximum(log_max, 0.0) / (bright_env + 800.0)
    combined_score = 0.55 * rel_drop + 1.9 * norm_log

    cand_fg = (
        (combined_score > 0.30)
        & (v_den < 0.44 * bright_env)
        & (bright_env > 11500.0)
        & (sub_r >= 0.24)
        & (sub_r <= 0.60)
    )

    # 4. Physical-distance 3D marker-controlled watershed
    dist_mm = ndi.distance_transform_edt(cand_fg, sampling=spacing)
    peak_field = ndi.gaussian_filter(
        dist_mm * (0.45 + combined_score),
        sigma=(0.22 / spacing[0], 0.22 / spacing[1], 0.28 / spacing[2]),
    )
    fp_pk = build_ellipsoid_footprint(rx_mm=0.62, rz_mm=0.80, spacing=spacing)
    loc_max = (peak_field == ndi.maximum_filter(peak_field, footprint=fp_pk)) & cand_fg & (dist_mm >= 0.20)
    markers, _ = ndi.label(loc_max)
    ws_pad = segmentation.watershed(-peak_field, markers=markers, mask=cand_fg)

    # 5. Reject grey radial septa and split any fused multi-seed pair (> 3.2 mm^3) via 3D physical K-means
    valid_pad = np.zeros_like(ws_pad)
    keep_id = 0
    for prop in measure.regionprops(ws_pad, intensity_image=sub):
        full_vol_mm3 = prop.area * vox_vol
        if full_vol_mm3 < 0.28 or full_vol_mm3 > 10.5:
            continue
        icx, icy, icz = [int(round(c)) for c in prop.centroid]
        loc_env = bright_env[icx, icy, icz]
        if prop.intensity_min > min(2200.0, 0.22 * loc_env):
            continue
        if prop.intensity_mean > 0.40 * loc_env:  # rejects grey radial septum creases between chambers
            continue
        if full_vol_mm3 <= 3.2:
            if prop.extent < 0.27:
                continue
            keep_id += 1
            coords = prop.coords
            valid_pad[coords[:, 0], coords[:, 1], coords[:, 2]] = keep_id
        else:
            n_split = max(2, int(round(full_vol_mm3 / 1.55)))
            coords = prop.coords
            phys_pts = coords * np.array(spacing)
            km = KMeans(n_clusters=n_split, random_state=42, n_init=10).fit(phys_pts)
            for s_idx in range(n_split):
                sub_coords = coords[km.labels_ == s_idx]
                if len(sub_coords) * vox_vol >= 0.28:
                    keep_id += 1
                    valid_pad[sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2]] = keep_id

    # 6. Crop to exact ROI [x0:x1, y0:y1, z0:z1] keeping instances whose 3D centroid lies inside the ROI
    roi_crop = valid_pad[pad_xy:-pad_xy, pad_xy:-pad_xy, pad_z:-pad_z]
    clean_roi = np.zeros_like(roi_crop, dtype=np.uint16)
    records = []
    final_id = 0
    for prop_pad in measure.regionprops(valid_pad, intensity_image=sub):
        gx = (x0 - pad_xy) + prop_pad.centroid[0]
        gy = (y0 - pad_xy) + prop_pad.centroid[1]
        gz = (z0 - pad_z) + prop_pad.centroid[2]
        if not (x0 <= gx < x1 and y0 <= gy < y1 and z0 <= gz < z1):
            continue
        mask_in_roi = roi_crop == prop_pad.label
        if mask_in_roi.sum() < 6:
            continue
        final_id += 1
        clean_roi[mask_in_roi] = final_id
        c_abs = prop_pad.coords + np.array([x0 - pad_xy, y0 - pad_xy, z0 - pad_z])
        fully_interior = bool(
            (c_abs[:, 0].min() >= x0)
            and (c_abs[:, 0].max() < x1)
            and (c_abs[:, 1].min() >= y0)
            and (c_abs[:, 1].max() < y1)
            and (c_abs[:, 2].min() >= z0)
            and (c_abs[:, 2].max() < z1)
        )
        records.append(
            {
                "id": final_id,
                "centroid_voxel": [round(float(gx), 2), round(float(gy), 2), round(float(gz), 2)],
                "full_3d_vol_mm3": round(float(prop_pad.area * vox_vol), 3),
                "roi_voxels": int(mask_in_roi.sum()),
                "min_intensity": round(float(prop_pad.intensity_min), 0),
                "mean_intensity": round(float(prop_pad.intensity_mean), 0),
                "fully_interior": fully_interior,
            }
        )

    # Save full-grid uint16 NIfTI aligned to raw MRI affine
    full_gt_vol = np.zeros(vol.shape, dtype=np.uint16)
    full_gt_vol[x0:x1, y0:y1, z0:z1] = clean_roi
    gt_hdr = img.header.copy()
    gt_hdr.set_data_dtype(np.uint16)
    nib.save(nib.Nifti1Image(full_gt_vol, img.affine, gt_hdr), str(gt_nii_path))

    # Bright-locule stereological extrapolation diagnostics
    seed_zone_mask = fruit_mask & (r_norm >= 0.24) & (r_norm <= 0.60)
    roi_box_mask = np.zeros(vol.shape, dtype=bool)
    roi_box_mask[x0:x1, y0:y1, z0:z1] = True
    closed_full_approx = ndi.maximum_filter(vol, size=(7, 7, 3))
    bright_locule_mask = seed_zone_mask & (ndi.gaussian_filter(closed_full_approx, sigma=(3.0, 3.0, 1.5)) > 11500.0)
    locule_vox_roi = int((bright_locule_mask & roi_box_mask).sum())
    locule_vox_tot = int(bright_locule_mask.sum())
    extrap_factor = locule_vox_tot / max(locule_vox_roi, 1)

    rng = np.random.default_rng(2027)
    angles = [np.arctan2(r["centroid_voxel"][1] - 128.0, r["centroid_voxel"][0] - 128.0) for r in records]
    bins_ang = np.linspace(0.0, np.pi / 2, 9)
    sector_counts = np.histogram(angles, bins=bins_ang)[0]
    boot_totals = [
        rng.choice(sector_counts, size=len(sector_counts), replace=True).sum() * extrap_factor for _ in range(4000)
    ]
    ci_lo, ci_hi = np.percentile(boot_totals, [2.5, 97.5])
    est_total_seeds = len(records) * extrap_factor

    meta = {
        "description": (
            "Semi-automated ('silver-standard') 3D Micro-Ground-Truth seed instance mask "
            "for the BMED365 Kiwi Challenge (upper-right equatorial sector)"
        ),
        "curation_method": {
            "standard": "silver-standard (semi-automated algorithmic curation + multi-slice visual audit)",
            "generator_script": "Lab4-Comp-Imag/Kiwi-challenge/scripts/curate_micro_gt.py",
            "pipeline_summary": (
                "Padded 3D sub-volume (x=120..216, y=120..216, z=51..77) processed with multi-scale physical-mm "
                "3D Laplacian-of-Gaussian (0.34, 0.44, 0.56 mm) + anisotropic ellipsoidal grey closing "
                "(1.15 x 1.15 x 1.50 mm) normalized by local bright-locule envelope (> 11,500) inside "
                "0.24 <= r/R <= 0.60, segmented via 3D marker-controlled watershed, filtered for jet-black "
                "core/mean intensity to reject grey radial septa, and split on fused pairs (> 3.2 mm^3) via "
                "3D physical-mm K-means."
            ),
            "manual_refinement_ready": (
                "Saved on the full 256 x 256 x 128 grid with identical NIfTI affine so users can inspect or "
                "manually refine voxel boundaries in ITK-SNAP or 3D Slicer."
            ),
        },
        "source_scan": nii_path.name,
        "label_nifti": gt_nii_path.name,
        "roi_bounds_voxel_xyz": {"x": [x0, x1], "y": [y0, y1], "z": [z0, z1]},
        "roi_physical_size_mm": [
            round((x1 - x0) * spacing[0], 2),
            round((y1 - y0) * spacing[1], 2),
            round((z1 - z0) * spacing[2], 2),
        ],
        "voxel_spacing_mm": list(spacing),
        "n_seeds_in_roi": len(records),
        "n_strictly_interior": sum(r["fully_interior"] for r in records),
        "n_boundary_crossing_centroid_in_roi": sum(not r["fully_interior"] for r in records),
        "median_seed_vol_mm3": round(float(np.median([r["full_3d_vol_mm3"] for r in records])), 3),
        "stereological_extrapolation": {
            "note": (
                "Equatorial seed density exceeds polar density; prefer ratio calibration N_raw * P / R in "
                "Notebook 03 (~1,440 seeds) over unadjusted equatorial volume extrapolation."
            ),
            "bright_locule_voxels_in_roi": locule_vox_roi,
            "bright_locule_voxels_whole_kiwi": locule_vox_tot,
            "volume_extrapolation_factor": round(float(extrap_factor), 3),
            "estimated_total_kiwi_seeds": int(round(est_total_seeds)),
            "ci_95_spatial_bootstrap": [int(round(ci_lo)), int(round(ci_hi))],
        },
        "seeds": records,
    }
    gt_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Saved NIfTI ({gt_nii_path.stat().st_size:,} bytes) & JSON ({gt_json_path.stat().st_size:,} bytes)")
    print(
        f"ROI seeds: {len(records)} ({meta['n_strictly_interior']} strictly interior, "
        f"{meta['n_boundary_crossing_centroid_in_roi']} face-crossing)"
    )
    print(f"Median 3D seed vol: {meta['median_seed_vol_mm3']} mm^3")

    if args.audit_png is not None:
        show_z = [57, 59, 61, 63, 64, 65, 67, 69]
        fig, axes = plt.subplots(2, len(show_z), figsize=(22, 6.2), dpi=140)
        roi_raw = vol[x0:x1, y0:y1, z0:z1]
        for col, z_abs in enumerate(show_z):
            iz = z_abs - z0
            sl_raw = roi_raw[:, :, iz].T
            sl_lbl = clean_roi[:, :, iz].T
            axes[0, col].imshow(sl_raw, cmap="gray", origin="lower", extent=[x0, x1, y0, y1])
            axes[0, col].set_title(f"Raw z={z_abs}", fontsize=9.5)
            axes[1, col].imshow(sl_raw, cmap="gray", origin="lower", extent=[x0, x1, y0, y1])
            ov = np.ma.masked_where(sl_lbl == 0, ((sl_lbl * 7) % 23) + 1)
            axes[1, col].imshow(ov, cmap="turbo", alpha=0.72, origin="lower", extent=[x0, x1, y0, y1], vmin=1, vmax=24)
            n_sl = len(np.unique(sl_lbl)) - (1 if 0 in sl_lbl else 0)
            axes[1, col].set_title(f"Micro-GT z={z_abs} ({n_sl})", fontsize=9.5)
        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        args.audit_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.audit_png)
        plt.close(fig)
        print(f"Saved audit sheet: {args.audit_png}")


if __name__ == "__main__":
    main()
