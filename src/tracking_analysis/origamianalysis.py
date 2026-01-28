from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

from tracking_analysis.postprocessing import DataPostProcessor

class SMOrigamiAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path):
        self.post_proc_data = post_proc_data
        self.locs_filepath = locs_filepath
        self.tcspc_data_dir = tcspc_data_dir
        self.fit_cloud(self.post_proc_data.locs_centered)
        
    def fit_cloud(self, locs):
        """
        Simple 2D gaussian fit of a cloud of points, it prints center, sigma and CRB in the center
        """
        gmm = GaussianMixture(n_components=1, covariance_type='full')
        gmm.fit(locs[:, 1:3])  # Fit on (x, y) coordinates

        # Extract means and covariances
        means = gmm.means_  # Shape (2, 2), centers of the Gaussians
        covariances = gmm.covariances_  # Shape (2, 2, 2), full covariance matrices

        # Compute standard deviations (σ) from covariance matrix
        sigmas = np.sqrt(np.array([np.diag(cov) for cov in covariances]))
        print(f"Center: {means[0]}")
        print(f"Sigmas: {sigmas[0]}")
        print(f"Average sigma: {np.sqrt((sigmas[0][0]**2 + sigmas[0][1]**2)/2)}")
        print(f"CRB in cloud center: {self.post_proc_data.σ_CRB[
            int(self.post_proc_data.ebp.pos_mins_nm[0][1] + means[0][1]),
            int(self.post_proc_data.ebp.pos_mins_nm[0][0] + means[0][0])
            ]}")
        
class ClockOrigamiAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path):
        self.post_proc_data = post_proc_data
        self.locs_filepath = locs_filepath
        self.tcspc_data_dir = tcspc_data_dir
        self.x_plot_range, self.y_plot_range = self.post_proc_data.get_glob_plots_limits(self.post_proc_data.locs_centered)
        self.center_of_locs, self.locs_zeroavg = self.recenter_locs(self.post_proc_data.locs_centered)
        self.fit_andplot_clockaxis(self.post_proc_data.locs_centered)
        self.locs_zeroavg_rotated = self.rotate_locs(self.locs_zeroavg, self.axis_slope)
        self.plot_loc_trace(self.locs_zeroavg_rotated)
        self.hidden_states, self.hidden_states_rescaled = self.hmm_fit(self.locs_zeroavg_rotated)
        self.calc_clock_times(self.locs_zeroavg_rotated, self.hidden_states)
        self.locs_hmmfilt = self.hmm_filter(self.post_proc_data.locs_centered, self.hidden_states)
        # repeat fits with HMM-filtered data
        self.post_proc_data.plot_locs_withcrb(self.locs_hmmfilt)
        self.post_proc_data.plot_loc_density_withebp(self.locs_hmmfilt)
        # fit clouds of points
        self.fit_clock_clouds(self.locs_hmmfilt)

    def recenter_locs(self, locs):
        """
        This function recenter localizations with respect their center of mass
        """
        locs_recenter = deepcopy(locs)
        center_of_locs = (np.mean(locs[:, 1]), np.mean(locs[:, 2]))
        locs_recenter[:, 1] -= center_of_locs[0]
        locs_recenter[:, 2] -= center_of_locs[1]
        return center_of_locs, locs_recenter
        
    def fit_andplot_clockaxis(self, locs):
        """
        This function performs gets the axis of the clock origami
        """
        # now perform Singular Value Decomposition analysis to find axis
        _, _, Vt = np.linalg.svd(self.locs_zeroavg[:, 1:3])
        main_axis = Vt[0]
        self.axis_slope = main_axis[1] / main_axis[0]
        axis_x = np.linspace(self.x_plot_range[0], self.x_plot_range[1], 100)
        axis_y = (axis_x - self.center_of_locs[0]) * self.axis_slope + self.center_of_locs[1]
        plt.figure('Clock axis plot')
        for beam_idx, min_pos in enumerate(self.post_proc_data.ebp.pos_mins_centered_nm):
            plt.scatter(*min_pos, color=self.post_proc_data.ebp.psf_colors[beam_idx], s=100)
        plt.scatter(locs[:, 1], locs[:, 2], c='gray', s=20, alpha=0.05)
        plt.plot(axis_x, axis_y, c='red', linestyle='--', linewidth=2)
        plt.xlim(self.x_plot_range)
        plt.ylim(self.y_plot_range)
        # Annotations
        plt.gca().set_aspect('equal'), plt.xlabel('x (nm)'), plt.ylabel('y (nm)'), plt.tight_layout()
        plt.show()
        
    def rotate_locs(self, locs, axis_slope):
        """
        This function returns the localizations in a new system of reference rotated accoridng to a given new x axis
        """
        self.rot_angle = np.arctan(axis_slope)
        rotated_locs = deepcopy(locs)
        rotated_locs[:, 1] = locs[:, 1] * np.cos(self.rot_angle) + locs[:, 2] * np.sin(self.rot_angle)
        rotated_locs[:, 2] = - locs[:, 1] * np.sin(self.rot_angle) + locs[:, 2] * np.cos(self.rot_angle)
        return rotated_locs
    
    def plot_loc_trace(self, locs):
        """
        This function 
        """
        plt.figure("Localizations x time trace")
        plt.plot(locs[:, 0], locs[:, 1], label='x (nm)')
        plt.xlabel('Time (s)')
        plt.ylabel('Localizations (nm)')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure("Localizations y time trace")
        plt.plot(locs[:, 0], locs[:, 2], label='y (nm)')
        plt.xlabel('Time (s)')
        plt.ylabel('Localizations (nm)')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def calc_clock_times(self, locs, hidden_states):
        """
        This function extracts the average binding times from the HMM analysis
        """
        jumps_bins = locs[:, 0][:-1][np.logical_or(
            np.isclose(np.diff(hidden_states), 1, atol=1e-2),
            np.isclose(np.diff(hidden_states), -1, atol=1e-2)
        )]
        print(f"Average clocking time: {np.mean(np.diff(jumps_bins))} s")
        
    def hmm_fit(self, locs):
        """
        This function implements a basic HMM fit of the time trace and plots the rescaled prediciton for the hidden states
        """
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
        model.fit(locs[:, 1].reshape(-1, 1))
        hidden_states = model.predict(locs[:, 1].reshape(-1, 1))
        locs_statezero = locs[hidden_states == 0]
        locs_stateone = locs[hidden_states == 1]
        self.avg_statezero = np.mean(locs_statezero[:, 1])
        self.avg_stateone = np.mean(locs_stateone[:, 1])
        downstate_idx = np.argmin((self.avg_statezero, self.avg_stateone))
        if downstate_idx == 0:    
            hidden_states_rescaled = hidden_states * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
        else:
            hidden_states_rescaled = (1 - hidden_states) * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
        plt.figure(figsize=(12, 6))
        plt.plot(locs[:, 1], label="Localization Trace")
        plt.plot(hidden_states_rescaled, label="Predicted Hidden States", linestyle='--', color='red')
        plt.legend()
        plt.show()
        # prepare array with rotated localizations and HMM result (rescaled)
        self.rotated_clock_trace = np.empty((len(locs[:, 0]), 3), dtype=np.float64)
        self.rotated_clock_trace[:, 0] = locs[:, 0]
        self.rotated_clock_trace[:, 1] = locs[:, 1]
        self.rotated_clock_trace[:, 2] = hidden_states_rescaled
        self.trace_hmm_filename = self.locs_filepath.stem + '_timetracewHMM.npy'
        self.trace_hmm_filepath = self.tcspc_data_dir / self.trace_hmm_filename
        np.save(self.trace_hmm_filepath, self.rotated_clock_trace)
        return hidden_states, hidden_states_rescaled
        
        
    def hmm_filter(self, locs, hidden_states):
        """
        This function finds all hidden state transitions and filters data based on this, throwing all
        localizations right before and right after a jump
        """
        locs_beforefilt = len(locs)
        locs_hmmfilt = locs[:-1][
            np.logical_and(
                np.isclose(np.diff(hidden_states), 0, atol=1e-2),
                np.isclose(np.diff(np.concatenate([hidden_states[:1], hidden_states[:-1]])), 0, atol=1e-6)
            )
        ] 
        locs_afterfilt = len(locs_hmmfilt)
        print(f"HMM filtering discarded {locs_beforefilt - locs_afterfilt} localizations")
        print(f"{locs_afterfilt} localizations remaining ({int(locs_afterfilt / locs_beforefilt * 100)}%)")
        
        self.locs_hmmfilts_filename = self.locs_filepath.stem + '_HMMfilt.npy'
        self.locs_hmmfilts_filepath = self.tcspc_data_dir / self.locs_hmmfilts_filename
        np.save(self.locs_hmmfilts_filepath, locs_hmmfilt)
        return locs_hmmfilt
    
    def fit_clock_clouds(self, locs):
        """
        This function uses a clustering algorithm to fit the two clouds of points of the clock
        """
        gmm = GaussianMixture(n_components=2, covariance_type='full')
        gmm.fit(locs[:, 1:3])  # Fit on (x, y) coordinates

        # Extract means and covariances
        means = gmm.means_  # Shape (2, 2), centers of the Gaussians
        covariances = gmm.covariances_  # Shape (2, 2, 2), full covariance matrices

        # Compute standard deviations (σ) from covariance matrix
        sigmas = np.sqrt(np.array([np.diag(cov) for cov in covariances]))
        for gauss_idx in range(2):
            print(f"Cloud number {gauss_idx}:")
            print(f"Center: {means[gauss_idx]}")
            print(f"Sigma: {sigmas[gauss_idx]}")
            print(f"CRB in cloud center: {self.post_proc_data.σ_CRB[
                int(self.post_proc_data.ebp.pos_mins_nm[0][1] + means[gauss_idx][1]),
                int(self.post_proc_data.ebp.pos_mins_nm[0][0] + means[gauss_idx][0])
                ]}")
            
        print(f"Estimated clock origami size: {np.sqrt((means[0][0] - means[1][0])**2 + (means[0][1] - means[1][1])**2)}")
            