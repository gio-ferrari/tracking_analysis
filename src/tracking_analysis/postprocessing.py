from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from scipy.ndimage import gaussian_filter1d

from tracking_analysis.tools.loc_tools import crb_minflux
from tracking_analysis.config.configvar import (
    NUM_PULSES,
    STEP_NM,
    SIGMA_TOL_OUTLIERS,
    CENTRAL_DONA_IDX,
    MAX_DIST_FROMEBP_CENT_NM
)
from tracking_analysis.ebp import EBP

class DataPostProcessor():
    def __init__(self, locs_filepath: Path, drift_filepath_list: List[Path], ebp: EBP, bin_size: int, use_drift_data_choice: bool, use_lifetime: bool):
        self.locs_filepath = locs_filepath
        print(self.locs_filepath)
        self.drift_filepath_list = drift_filepath_list
        self.ebp = ebp
        self.bin_size = bin_size
        self.use_drift_data_choice = use_drift_data_choice
        self.use_lifetime = use_lifetime
        
        # open file containing localization results
        self.locs = self.load_locs(self.locs_filepath)
        # eliminate spatial outliers from localizations
        self.locs = self.eliminate_outliers(self.locs)
        if self.use_drift_data_choice:
            # open file with drift data for a posteriori drift correction
            self.drift_data = self.prepare_drift_data(self.drift_filepath_list)
            # correct a posteriori localizations with drift data
            self.locs = self.apost_drift_locs_corr(self.locs, self.drift_data)

        # now filter localizations based on photons numbers
        self.locs_filt = self.filter_locs_forph(self.locs)
        # compute average photon number and SBR based on filtered localizations
        self.avg_ph_perloc, self.avg_sbr = self.get_avg_loc_param(self.locs_filt)
        # compute and plot CRB map, superimposed with EBP
        self.calc_crb(self.avg_ph_perloc, self.avg_sbr)
        #self.plot_crb_andebp()
        # recenter localizations
        self.locs_centered = self.center_locs(self.locs_filt)
        # get range for future plots
        self.x_plot_range, self.y_plot_range = self.get_glob_plots_limits(self.locs_centered)
        # all plots
        self.plot_locs_timecoded_withebp(self.locs_centered)
        self.plot_loc_density_withebp(self.locs_centered)
        self.plot_locs_withcrb(self.locs_centered)
        
        if self.use_lifetime:
            self.plot_lt_trace(self.locs_centered)
            self.plot_locs_wlt(self.locs_centered)
        
    def load_locs(self, locs_filepath):
        """
        This function loads the localization array saved as a .npy
        """
        locs = np.load(locs_filepath)
        self.avg_ph_perloc = np.mean(locs[:,3])
        self.sigma_ph_perloc = np.std(locs[:,3])
        print(f"Average number of photons per localization: {self.avg_ph_perloc} \u00B1 {self.sigma_ph_perloc}")
        return locs
        
    def load_takyaq_data(self, filename):
        """
        This function reads the output data of the takyaq stabilization system
        """
        with open(filename, 'rb') as fd:
            data = []
            n_batches = 0
            try:
                while True:
                    data.append(np.load(fd))
                    n_batches += 1
            except EOFError:
                print(f"Loaded {n_batches} batches of lengths {[len(x) for x in data]} data points")
        rv = np.concatenate(data) if data else None
        return rv
        
    def prepare_drift_data(self, drift_filepath_list):
        """
        This function loads and prepare drift data for a posteriori correction
        """
        should_fill_t_axis = True
        with open(drift_filepath_list[0], 'r') as part_coord_file:
                drift_num_pts = len(part_coord_file.readlines()[1:])
        t_axis = np.empty(drift_num_pts, dtype=float)
        allpart_coord_arr = np.empty((len(drift_filepath_list), drift_num_pts, 2), dtype=float)
        
        for particle_idx in range(len(drift_filepath_list)):
            with open(drift_filepath_list[particle_idx], 'r') as part_coord_file:
                coords = part_coord_file.readlines()[1:]
                for coord_idx in range(len(coords)):
                    if should_fill_t_axis:
                        t_axis[coord_idx] = coords[coord_idx].split(' ')[0]
                    allpart_coord_arr[particle_idx, coord_idx, 0] = coords[coord_idx].split(' ')[1]
                    allpart_coord_arr[particle_idx, coord_idx, 1] = coords[coord_idx].split(' ')[2]
                should_fill_t_axis = False

        avg_coord_arr = np.mean(allpart_coord_arr, axis=0)
        avg_coord_arr[:, 0] = gaussian_filter1d(avg_coord_arr[:, 0], sigma=3)
        avg_coord_arr[:, 1] = gaussian_filter1d(avg_coord_arr[:, 1], sigma=3)
        
        for particle_idx in range(len(drift_filepath_list)):
            plt.plot(t_axis, -allpart_coord_arr[particle_idx, :, 0], linestyle=':')
        plt.plot(t_axis, -avg_coord_arr[:,0], linestyle='-')
        plt.show()
        for particle_idx in range(len(drift_filepath_list)):
            plt.plot(t_axis, allpart_coord_arr[particle_idx, :, 1], linestyle=':')
        plt.plot(t_axis, avg_coord_arr[:,1], linestyle='-')
        plt.show()
        return np.column_stack((t_axis, avg_coord_arr))
        
    def apost_drift_locs_corr(self, locs, drift_data):
        """
        This function uses the data of the xy drift to correct a posteriori the MINFLUX localizations. For each localization
        it uses the closest (in time) datapoint of the drift
        """
        plt.plot(drift_data[:,0], -drift_data[:,1] + np.mean(drift_data[:,1]), label='x drift')
        plt.plot(locs[:,0], locs[:,1] - np.mean(locs[:,1]), label='x pos')
        plt.legend()
        plt.show()
        
        plt.plot(drift_data[:,0], drift_data[:,2] - np.mean(drift_data[:,2]), label='y drift')
        plt.plot(locs[:,0], locs[:,2] - np.mean(locs[:,2]), label='x pos')
        plt.legend()
        plt.show()
        
        last_closest_t_idx = 0
        for loc_idx in range(len(locs[:, 0])):
            for t_drift_idx in range(last_closest_t_idx, len(drift_data[:, 0])):
                if drift_data[t_drift_idx, 0] > locs[loc_idx, 0]:
                    last_closest_t_idx = t_drift_idx
                    locs[loc_idx, 1] += drift_data[t_drift_idx, 1]
                    locs[loc_idx, 2] -= drift_data[t_drift_idx, 2]
                    break
        self.locs_driftcorr_results_filename = self.locs_filepath.stem + '_driftcorr_'  + '.npy'
        self.locs_driftcorr_results_filepath = self.locs_filepath.parent / self.locs_driftcorr_results_filename
        np.save(self.locs_driftcorr_results_filepath, self.locs)
        return locs
        
    def eliminate_outliers(self, locs):
        """
        This function eliminates all localization exceeding a certain number of sigma from the center of mass,
        or too far from the EBP center
        """
        dists_loc_from_ebp_center = np.sqrt(
            (locs[:, 1] - self.ebp.pos_mins_nm[CENTRAL_DONA_IDX, 0])**2 + (locs[:, 2] - self.ebp.pos_mins_nm[CENTRAL_DONA_IDX, 1])**2
        )   
        locs = locs[dists_loc_from_ebp_center < MAX_DIST_FROMEBP_CENT_NM]
        
        self.average_coords_locs = (
            np.mean(locs[:, 1]),
            np.mean(locs[:, 2])
        )
        self.average_sigma_locs = (
            np.std(locs[:, 1]),
            np.std(locs[:, 2])
        )
        dists_loc_from_center = np.sqrt(
            (locs[:, 1] - self.average_coords_locs[0])**2 + (locs[:, 2] - self.average_coords_locs[1])**2
        )  
        locs = locs[dists_loc_from_center < SIGMA_TOL_OUTLIERS * np.sqrt(self.average_sigma_locs[0]**2 + self.average_sigma_locs[1]**2)]
        return locs
        
    def filter_locs_forph(self, locs):
        """This function filters out localizations obtained with less photons than a chosen threshold"""
        min_ph_perloc_input = input("Minimum number of photons required for a single localization (no input for no filtering): ")
        if min_ph_perloc_input:
            self.min_ph_perloc = float(min_ph_perloc_input)
        else:
            self.min_ph_perloc = 0
        locs_filtered = locs[locs[:, 3] >= self.min_ph_perloc]
        return locs_filtered
        
    def get_avg_loc_param(self, locs):
        """
        This function computes the average number of photons and SBR for the remaining localizations
        """
        avg_ph_perloc = np.mean(locs[:, 3])
        avg_sbr = np.average(locs[:, 4], weights=locs[:, 3])
        return avg_ph_perloc, avg_sbr
        
    def center_locs(self, locs):
        """
        This function changes system of reference for the localizations, cenetering their values in the minimum of the first beam
        """
        locs_centered = deepcopy(locs)
        locs_centered[:, 1] -= self.ebp.pos_mins_nm[0][0]
        locs_centered[:, 2] -= self.ebp.pos_mins_nm[0][1]
        return locs_centered
        
    def get_glob_plots_limits(self, locs):
        """
        This function computes the limit in x and y for plots to keep the size of the plotted area consistent
        """
        x_ebp_range = (
            min(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[0])[0],
            max(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[0])[0]
        )
        y_ebp_range = (
            min(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[1])[1],
            max(self.ebp.pos_mins_centered_nm, key=lambda elem: elem[1])[1]
        )
        x_filt_locs_range = (np.min(locs[:, 1]), np.max(locs[:, 1]))
        y_filt_locs_range = (np.min(locs[:, 2]), np.max(locs[:, 2]))
        x_global_range = (np.min((x_ebp_range[0], x_filt_locs_range[0])), np.max((x_ebp_range[1], x_filt_locs_range[1])))
        y_global_range = (np.min((y_ebp_range[0], y_filt_locs_range[0])), np.max((y_ebp_range[1], y_filt_locs_range[1])))
        # add a 10% padding
        x_plot_range = (x_global_range[0] - 0.1 * (x_global_range[1] - x_global_range[0]), x_global_range[1] + 0.1 * (x_global_range[1] - x_global_range[0]))
        y_plot_range = (y_global_range[0] - 0.1 * (y_global_range[1] - y_global_range[0]), y_global_range[1] + 0.1 * (y_global_range[1] - y_global_range[0]))
        return x_plot_range, y_plot_range
                    
    def calc_crb(self, ph_perloc, sbr):
        """
        This function computes the crb based on the experimental SBR and average photon number (averaging x and y errors)
        """
        print(f"Computing CRB with {ph_perloc} photons and {sbr} of SBR")
        self.σ_CRB = crb_minflux(NUM_PULSES, self.ebp.psf_fits, sbr, STEP_NM, self.ebp.size_nm, ph_perloc, method='1')
     
    def plot_crb_andebp(self):
        """
        This function plots the CRB map alone
        """
        # Create the CRB plot with the same extent as the scatter plots
        plt.figure('CRB_map')
        # Shift CRB map by the same amount as the EBP
        plt.imshow(
            self.σ_CRB, cmap='viridis', vmin=0, vmax=5,
            extent=(
                - self.ebp.pos_mins_nm[0][0] - 0.5, self.σ_CRB.shape[1] - self.ebp.pos_mins_nm[0][0] - 0.5,
                - self.ebp.pos_mins_nm[0][1] - 0.5, self.σ_CRB.shape[0] - self.ebp.pos_mins_nm[0][1] - 0.5
            ),
            origin='lower'
        )
        plt.colorbar(label='σ_CRB Value')

        # Plot PSF minima positions with the same color mapping as before
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)

        # Ensure the axes and aspect ratio are the same as in scatter plots
        plt.gca().set_aspect('equal')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title('σ_CRB with Aligned Reference Frame')
        plt.tight_layout()
        plt.show()
                        
    def plot_locs_timecoded_withebp(self, locs):
        """
        This function plots all (filtered) localizations, encoding with time, superposed with the EBP
        """
        plt.figure('Time-encoded localizations')
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.scatter(locs[:, 1], locs[:, 2], c=(locs[:, 0] - locs[0, 0]), cmap='rainbow', s=20, alpha=0.05)
        color_bar = plt.colorbar(label="Time [s]", orientation="vertical")
        color_bar.solids.set(alpha=1)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()
        
    def plot_loc_density_withebp(self, locs):
        """
        This function produces the 2D histogram of the localization density, superposed with the EBP
        """
        bin_x_edges = np.arange(int(self.x_plot_range[0]), int(self.x_plot_range[1]) + self.bin_size, self.bin_size)
        bin_y_edges = np.arange(int(self.y_plot_range[0]), int(self.y_plot_range[1]) + self.bin_size, self.bin_size)
        loc_dens_hist, _, _ = np.histogram2d(locs[:, 1], locs[:, 2], bins=(bin_x_edges, bin_y_edges))
        max_loc_dens = np.max(loc_dens_hist)
        plt.figure('Localization density')
        plt.hist2d(locs[:, 1], locs[:, 2], bins=(bin_x_edges, bin_y_edges), cmap='magma', vmin=max_loc_dens * 0.1)
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()
        
    def plot_locs_withcrb(self, locs):
        """
        This function plots the localizations superimposed with the CRB map
        """
        # Create the CRB plot with the same extent as the scatter plots
        plt.figure('CRB_map with loalizations')
        # Shift CRB map by the same amount as the EBP
        plt.imshow(
            self.σ_CRB, cmap='viridis', vmin=0, vmax=5,
            extent=(
                - self.ebp.pos_mins_nm[0][0] - 0.5, self.σ_CRB.shape[1] - self.ebp.pos_mins_nm[0][0] - 0.5,
                - self.ebp.pos_mins_nm[0][1] - 0.5, self.σ_CRB.shape[0] - self.ebp.pos_mins_nm[0][1] - 0.5
            ),
            origin='lower'
        )
        plt.colorbar(label='σ_CRB Value')

        # Plot PSF minima positions with the same color mapping as before
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)

        # add localizations to plot
        plt.scatter(locs[:, 1], locs[:, 2], c='gray', s=20, alpha=0.05)
        
        # Ensure the axes and aspect ratio are the same as in scatter plots
        plt.gca().set_aspect('equal')
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.title('σ_CRB with Aligned Reference Frame')
        plt.tight_layout()
        plt.show()
    
    def plot_lt_trace(self, locs):
        plt.plot(locs[:, 0], locs[:, 5])
        plt.show()
        
    def plot_locs_wlt(self, locs):
        """
        This function plots all (filtered) localizations, encoding with lifetime, superposed with the EBP
        """
        plt.figure('Lifetime-encoded localizations')
        for beam_idx, min_pos in enumerate(self.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.ebp.psf_colors[beam_idx], s=100)
        plt.scatter(locs[:, 1], locs[:, 2], c=(locs[:, 5]), cmap='rainbow', vmin=0.8, vmax=1.4, s=20, alpha=0.05)
        color_bar = plt.colorbar(label="Lifetime [ns]", orientation="vertical")
        color_bar.solids.set(alpha=1)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()       