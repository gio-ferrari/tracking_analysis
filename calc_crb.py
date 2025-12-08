import numpy as np
import copy
from natsort import natsorted
import matplotlib.pyplot as plt

from loc_tools import indexToSpace, crb_minflux
from configvar import (
    NUM_PULSES,
    STEP_NM,
    PSF_DIR_BASE,
    COLOR_LIST
)

date = '20250225'
psf_dir = PSF_DIR_BASE / date

ph_perloc = 1000
sbr = 5

psf_fit_list = []
pos_min_nm_list = []
for filepath in natsorted(psf_dir.iterdir()):
    if filepath.is_file():
        if filepath.suffix == '.npy':
            psf_fit = np.load(filepath)
            psf_fit_list.append(psf_fit)
            psf_size = np.shape(psf_fit)[1]
            size_nm = psf_size * STEP_NM
            pos_min_nm = indexToSpace(
                np.unravel_index(np.argmin(psf_fit, axis=None),psf_fit.shape),
                size_nm,
                STEP_NM
            )
            pos_min_nm_list.append(pos_min_nm)

psf_fit_arr = np.array(psf_fit_list)
pos_min_nm_arr = np.array(pos_min_nm_list)
pos_min_nm_centered_arr = copy.deepcopy(pos_min_nm_arr)
for min_idx in range(len(pos_min_nm_centered_arr)):
    pos_min_nm_centered_arr[min_idx][0] -= pos_min_nm_arr[0][0]
    pos_min_nm_centered_arr[min_idx][1] -= pos_min_nm_arr[0][1]

print(f"Computing CRB with {ph_perloc} photons and {sbr} of SBR")
σ_CRB = crb_minflux(NUM_PULSES, psf_fit_arr, sbr, STEP_NM, size_nm, ph_perloc, method='1')

# Create the CRB plot with the same extent as the scatter plots
plt.figure('CRB_map')
# Shift CRB map by the same amount as the EBP
plt.imshow(
    σ_CRB, cmap='viridis', vmin=0, vmax=5,
    extent=(
        - pos_min_nm_arr[0][0] - 0.5, σ_CRB.shape[1] - pos_min_nm_arr[0][0] - 0.5,
        - pos_min_nm_arr[0][1] - 0.5, σ_CRB.shape[0] - pos_min_nm_arr[0][1] - 0.5
    ),
    origin='lower'
)
plt.colorbar(label='σ_CRB Value')

# Plot PSF minima positions with the same color mapping as before
for beam_idx, min_pos in enumerate(pos_min_nm_centered_arr):
    plt.scatter(*min_pos, color=COLOR_LIST[beam_idx], s=100)

# Ensure the axes and aspect ratio are the same as in scatter plots
plt.gca().set_aspect('equal')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title('σ_CRB with Aligned Reference Frame')
plt.tight_layout()

plt.show()