#!/usr/bin/env bash
# =============================================================================
# BMED365 (Spring 2027) — Lab 4: 7T Kiwi MRI Interactive 3D Viewer (FreeSurfer)
#
# Launches FreeSurfer 8.2.0 `freeview` on macOS with:
#   1. Raw 7T RARE 3D MRI (percentile-windowed grayscale)
#   2. Silver-Standard 3D Micro-GT mask (bit-aligned uint16, IDs 1..98) using
#      a custom high-contrast ColorLUT so every seed ID lights up cleanly
#   3. Optional toggleable yellow Micro-GT outline layer (unobstructed cores)
#   4. Optional toggleable whole-fruit Neurosymbolic 3D prediction mask
#      (`docs/neurosymbolic-report/results/kiwi_neurosymbolic_pred_labels.nii.gz`)
#
# Usage:
#   ./Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh [OPTIONS]
#
# Options:
#   --gt PATH          Overlay a custom/manually edited GT NIfTI (e.g. MyStuff/kiwi-gt/kiwi_gt_manual_v1.nii.gz)
#   --outline          Start Micro-GT in crisp outline-only mode (default: semi-transparent fill)
#   --with-pred        Show whole-fruit Neurosymbolic 3D prediction layer visible on launch
#   --3d               Enable 3D smoothed seed isosurfaces in the 3D viewport
#   --slice X Y Z      Initial voxel cursor (default: 168 168 64 = Micro-GT sector center)
#   --opacity VAL      Micro-GT overlay opacity in [0, 1] (default: 0.58)
#   --dry-run          Print the resolved freeview command without launching the GUI
#   -h, --help         Show this help message
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIWI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MRI="${KIWI_DIR}/data/kiwi_20050217191129_131073_RARE_high_res_axial.nii.gz"
GT="${KIWI_DIR}/data/kiwi_micro_gt_labels.nii.gz"
GT_LABEL="Silver_MicroGT_98seeds"
PRED="${KIWI_DIR}/docs/neurosymbolic-report/results/kiwi_neurosymbolic_pred_labels.nii.gz"
LUT="${SCRIPT_DIR}/kiwi_seeds_ColorLUT.txt"

OUTLINE=0
SHOW_PRED=0
ENABLE_3D=0
DRY_RUN=0
OPACITY="0.58"
SLICE_X=168
SLICE_Y=168
SLICE_Z=64

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt)
      GT="$2"
      GT_LABEL="Custom_Manual_GT"
      shift 2
      ;;
    --outline)
      OUTLINE=1
      shift
      ;;
    --with-pred)
      SHOW_PRED=1
      shift
      ;;
    --3d)
      ENABLE_3D=1
      shift
      ;;
    --slice)
      SLICE_X="$2"
      SLICE_Y="$3"
      SLICE_Z="$4"
      shift 4
      ;;
    --opacity)
      OPACITY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,27p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (use --help for usage)" >&2
      exit 1
      ;;
  esac
done

# 1. Locate FreeSurfer 8.2.0 freeview binary on macOS
FREEVIEW_BIN=""
if [[ -n "${FREESURFER_HOME:-}" && -x "${FREESURFER_HOME}/bin/freeview" ]]; then
  FREEVIEW_BIN="${FREESURFER_HOME}/bin/freeview"
elif [[ -x "/Applications/freesurfer/8.2.0/bin/freeview" ]]; then
  export FREESURFER_HOME="/Applications/freesurfer/8.2.0"
  FREEVIEW_BIN="${FREESURFER_HOME}/bin/freeview"
elif command -v freeview >/dev/null 2>&1; then
  FREEVIEW_BIN="$(command -v freeview)"
else
  echo "ERROR: Could not find FreeSurfer freeview. Please check FREESURFER_HOME (expected /Applications/freesurfer/8.2.0)." >&2
  exit 1
fi

# 2. Verify required NIfTI files and custom ColorLUT
for req in "${MRI}" "${GT}" "${LUT}"; do
  if [[ ! -f "${req}" ]]; then
    echo "ERROR: Required file not found: ${req}" >&2
    exit 1
  fi
done

# 3. Assemble freeview volume arguments
MRI_SPEC="${MRI}:name=7T_Kiwi_RARE:colormap=grayscale:percentile=1:grayscale=0,99.5"

GT_SPEC="${GT}:name=${GT_LABEL}:colormap=lut:lut=${LUT}:opacity=${OPACITY}"
if [[ "${OUTLINE}" -eq 1 ]]; then
  GT_SPEC="${GT_SPEC}:outline=1:opacity=0.95"
fi
if [[ "${ENABLE_3D}" -eq 1 ]]; then
  GT_SPEC="${GT_SPEC}:isosurface=on:isosurface_smooth=2"
fi

# Companion toggleable crisp yellow outline layer (hidden by default unless --outline=0)
GT_OUTLINE_SPEC="${GT}:name=${GT_LABEL}_Outline:colormap=binary:binary_color=255,235,59:outline=1:opacity=0.95:visible=0"

CMD=(
  "${FREEVIEW_BIN}"
  -v "${MRI_SPEC}"
  -v "${GT_SPEC}"
  -v "${GT_OUTLINE_SPEC}"
)

# Attach bit-aligned Neurosymbolic 3D whole-fruit prediction mask if present
if [[ -f "${PRED}" ]]; then
  PRED_SPEC="${PRED}:name=Neurosymbolic_3D_WholeFruit:colormap=binary:binary_color=0,230,118:opacity=0.48:visible=${SHOW_PRED}"
  if [[ "${ENABLE_3D}" -eq 1 && "${SHOW_PRED}" -eq 1 ]]; then
    PRED_SPEC="${PRED_SPEC}:isosurface=on:isosurface_color=0,230,118:isosurface_smooth=2"
  fi
  CMD+=(-v "${PRED_SPEC}")
fi

CMD+=(
  -slice "${SLICE_X}" "${SLICE_Y}" "${SLICE_Z}"
  -cc
  -layout 2
  -viewport axial
  -zoom 2.2
  -subtitle "BMED365 7T Kiwi MRI | Silver Micro-GT (x=128..207, y=128..207, z=56..71; y_split=168)"
)

echo "FreeSurfer freeview : ${FREEVIEW_BIN}"
echo "Raw 7T MRI          : ${MRI}"
echo "GT Overlay          : ${GT} (${GT_LABEL}, custom high-contrast LUT)"
if [[ -f "${PRED}" ]]; then
  echo "Neurosymbolic 3D    : ${PRED} (toggleable layer, initial visible=${SHOW_PRED})"
fi
echo "Initial cursor      : (${SLICE_X}, ${SLICE_Y}, ${SLICE_Z}) [Micro-GT center; Train y<168 | Test y>=168]"
echo "Tip inside freeview : Press Alt+V to toggle the selected overlay, or click the checkboxes on the left panel."

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '\n[dry-run command]\n'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
