from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path

from tracking_analysis.tools.loc_tools import indexToSpace
from tracking_analysis.config.configvar import (
    STEP_NM,
    COLOR_LIST
)

class EBP():
    def __init__(self, psf_dir: Path):
        self.psf_dir = psf_dir
        self.psf_colors = COLOR_LIST
        self.psf_fits, self.pos_mins_nm, self.pos_mins_centered_nm = self.open_psf()
        #self.plot_psf()

    def open_psf(self):
        """
        This function opens the specified data folder, reads in natural alphabetic order all the .npy
        files, which should contain the numpy arrays of the experimental PSFs, and loads them
        """
        psf_fit_list = []
        pos_min_nm_list = []
        print(self.psf_dir)
        for filepath in natsorted(self.psf_dir.iterdir()):
            if filepath.is_file():
                if filepath.suffix == '.npy':
                    print(filepath.name)
                    psf_fit = np.load(filepath)
                    psf_fit_list.append(psf_fit)
                    psf_size = np.shape(psf_fit)[1]
                    self.size_nm = psf_size * STEP_NM
                    pos_min_nm = indexToSpace(
                        np.unravel_index(np.argmin(psf_fit, axis=None),psf_fit.shape),
                        self.size_nm,
                        STEP_NM
                    )
                    pos_min_nm_list.append(pos_min_nm)

        psf_fit_arr = np.array(psf_fit_list)
        pos_min_nm_arr = np.array(pos_min_nm_list)
        pos_min_nm_centered_arr = deepcopy(pos_min_nm_arr)
        for min_idx in range(len(pos_min_nm_centered_arr)):
            pos_min_nm_centered_arr[min_idx][0] -= pos_min_nm_arr[0][0]
            pos_min_nm_centered_arr[min_idx][1] -= pos_min_nm_arr[0][1]  
            
        return psf_fit_arr, pos_min_nm_arr, pos_min_nm_centered_arr
    
    def plot_psf(self):
        """
        This function plots the individual experimental PSFs and the EBP
        """
        # Plot PSFs with minima positions
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.set(xlabel = 'x (nm)', ylabel= 'y (nm)')
            ax.imshow(self.psf_fits[i], cmap='viridis', origin='lower')
            ax.scatter(*self.pos_mins_nm[i],
                        color=self.psf_colors[i], s=100)
            ax.set_title(f'Fitted PSF {i + 1}', fontsize=10)
        plt.tight_layout()

        #Plot EBP
        x_min, y_min = self.pos_mins_centered_nm[:, 0], self.pos_mins_centered_nm[:, 1]
        plt.figure('EBP')
        for i in range(len(self.pos_mins_centered_nm)):
            plt.scatter(x_min[i]- self.pos_mins_centered_nm[0][0], y_min[i] - self.pos_mins_centered_nm[0][1], c=self.psf_colors[i], label=f'{i+1}')
        plt.title('EBP')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()