from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


from tracking_analysis.postprocessing import DataPostProcessor

class EndoIVAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path):
        self.post_proc_data = post_proc_data
        self.locs_filepath = locs_filepath
        self.tcspc_data_dir = tcspc_data_dir
        self.x_plot_range, self.y_plot_range = self.post_proc_data.get_glob_plots_limits(self.post_proc_data.locs_centered)
        self.center_of_locs, self.locs_zeroavg = self.recenter_locs(self.post_proc_data.locs_centered)
        self.fit_andplot_locaxis(self.post_proc_data.locs_centered)
        self.locs_zeroavg_rotated = self.rotate_locs(self.locs_zeroavg, self.axis_slope)
        self.plot_loc_trace(self.locs_zeroavg_rotated)
        self.hidden_states, self.hidden_states_rescaled = self.hmm_fit_lt(self.locs_zeroavg_rotated)
        self.get_jump_bins(self.locs_zeroavg_rotated, self.hidden_states)
        self.plot_locs_bystate(self.locs_zeroavg_rotated, self.hidden_states)
        self.locs_hmmfilt = self.hmm_filter(self.post_proc_data.locs_centered, self.hidden_states)
        self.calc_state_avg_pos()

    def recenter_locs(self, locs):
        """
        This function recenter localizations with respect their center of mass
        """
        locs_recenter = deepcopy(locs)
        center_of_locs = (np.mean(locs[:, 1]), np.mean(locs[:, 2]))
        locs_recenter[:, 1] -= center_of_locs[0]
        locs_recenter[:, 2] -= center_of_locs[1]
        return center_of_locs, locs_recenter

    def fit_andplot_locaxis(self, locs):
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
        
    def get_jump_bins(self, locs, hidden_states):
        """
        This function extracts the average binding times from the HMM analysis
        """
        self.jumps_bins = locs[:, 0][:-1][np.logical_or(
            np.isclose(np.diff(hidden_states), 1, atol=1e-2),
            np.isclose(np.diff(hidden_states), -1, atol=1e-2)
        )]
        print(f"Average time between jumps: {np.mean(np.diff(self.jumps_bins))} s")

    def hmm_fit_lt(self, locs):
        """
        This function implements a basic HMM fit of the lifetime trace and plots the rescaled prediciton for the hidden states
        """
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
        model.fit(locs[:, 5].reshape(-1, 1))
        hidden_states = model.predict(locs[:, 5].reshape(-1, 1))
        locs_statezero = locs[hidden_states == 0]
        locs_stateone = locs[hidden_states == 1]
        self.avg_statezero = np.mean(locs_statezero[:, 5])
        self.avg_stateone = np.mean(locs_stateone[:, 5])
        lowtau_idx = np.argmin((self.avg_statezero, self.avg_stateone))
        if lowtau_idx == 0:    
            hidden_states_rescaled = hidden_states * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
        else:
            hidden_states_rescaled = (1 - hidden_states) * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
        print("Lower lifetime: ", np.min((self.avg_statezero, self.avg_stateone)))
        print("Higher lifetime: ", np.max((self.avg_statezero, self.avg_stateone)))
        plt.figure(figsize=(12, 6))
        plt.plot(locs[:, 5], label="Lifetime Trace HMM")
        plt.plot(hidden_states_rescaled, label="Predicted Hidden States", linestyle='--', color='red')
        plt.legend()
        plt.show()
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
        
        self.locs_hmmfilts_filename = self.locs_filepath.stem + '_HMMfilt_withlifetime.npy'
        self.locs_hmmfilts_filepath = self.tcspc_data_dir / self.locs_hmmfilts_filename
        np.save(self.locs_hmmfilts_filepath, locs_hmmfilt)
        return locs_hmmfilt
        
    def plot_locs_bystate(self, locs, hidden_states):
        state_zero_mask = np.isclose(hidden_states, 0)
        state_one_mask = ~state_zero_mask
        plt.scatter(locs[state_zero_mask, 0], locs[state_zero_mask, 1], color="blue")
        plt.scatter(locs[state_one_mask, 0], locs[state_one_mask, 1], color="red")
        plt.show()
        
    def calc_state_avg_pos(self):
        """
        This function fits the positions between jumps
        """
        pass
        
        