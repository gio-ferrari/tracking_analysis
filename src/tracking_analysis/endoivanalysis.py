from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from tracking_analysis.postprocessing import DataPostProcessor

D0_ATTO643_NM = 18.5
TAU0_ATTO643_NS = 3.9

class EndoIVAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path):
        self.post_proc_data = post_proc_data
        self.locs_filepath = locs_filepath
        self.tcspc_data_dir = tcspc_data_dir
        self.how_many_states = self.ask_how_many_states()
        self.x_plot_range, self.y_plot_range = self.post_proc_data.get_glob_plots_limits(self.post_proc_data.locs_centered)
        self.hidden_states, self.hidden_states_rescaled = self.hmm_fit_lt(self.post_proc_data.locs_centered)
        self.get_jump_bins(self.post_proc_data.locs_centered, self.hidden_states)
        #self.locs_hmmfilt = self.hmm_filter(self.post_proc_data.locs_centered, self.hidden_states)
        self.locs_3d = self.calc_height(self.post_proc_data.locs_centered)
        if self.how_many_states==2:
            self.plot_3d_2states(self.locs_3d, self.hidden_states)
        elif self.how_many_states==3:
            self.plot_3d_3states(self.locs_3d, self.hidden_states)

    def ask_how_many_states(self):
        
        how_many_states_str = input("How many lifetime states are present?")
        match how_many_states_str:
            case "1":
                print("No dynamics observed")
                how_many_states = 1
            case "2":
                how_many_states = 2
            case "3":
                how_many_states = 3
        return how_many_states

    def recenter_locs(self, locs):
        """
        This function recenter localizations with respect their center of mass
        """
        locs_recenter = deepcopy(locs)
        center_of_locs = (np.mean(locs[:, 1]), np.mean(locs[:, 2]))
        locs_recenter[:, 1] -= center_of_locs[0]
        locs_recenter[:, 2] -= center_of_locs[1]
        return center_of_locs, locs_recenter
        
    def get_jump_bins(self, locs, hidden_states):
        """
        This function extracts the average binding times from the HMM analysis
        """
        self.jumps_bins = locs[:, 0][:-1][np.logical_or(
            np.isclose(np.abs(np.diff(hidden_states)), 1, atol=1e-2),
            np.isclose(np.abs(np.diff(hidden_states)), 2, atol=1e-2),
        )]
        print(f"Average time between jumps: {np.mean(np.diff(self.jumps_bins))} s")

    def hmm_fit_lt(self, locs):
        """
        This function implements a basic HMM fit of the lifetime trace and plots the rescaled prediciton for the hidden states
        """
        model = GaussianHMM(n_components=self.how_many_states, covariance_type="diag", n_iter=1000)
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

    def height_fromget(self, tau, dist_halfquench, tau_unquench):
        return dist_halfquench*(tau_unquench/tau - 1)**(-1/4)  

    def calc_height(self, locs):
        height_arr = self.height_fromget(locs[:,5], D0_ATTO643_NM, TAU0_ATTO643_NS)
        return np.concatenate((
                locs,
                height_arr.reshape(-1,1),
            ), axis=1)
        
    def plot_locs_bystate(self, locs, hidden_states):
        state_zero_mask = np.isclose(hidden_states, 0)
        state_one_mask = ~state_zero_mask
        plt.scatter(locs[state_zero_mask, 0], locs[state_zero_mask, 1], color="blue")
        plt.scatter(locs[state_one_mask, 0], locs[state_one_mask, 1], color="red")
        plt.show()
        
    def plot_3d_2states(self, locs, hidden_states):
        pass
    
    def plot_3d_3states(self, locs, hidden_states):
        state_zero_mask = np.isclose(hidden_states, 0)
        state_one_mask = np.isclose(hidden_states, 1)
        state_two_mask = np.isclose(hidden_states, 2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(locs[state_zero_mask, 1], locs[state_zero_mask, 2], locs[state_zero_mask, 6], color='blue', s=50, alpha=0.4)
        ax.scatter(locs[state_one_mask, 1], locs[state_one_mask, 2], locs[state_one_mask, 6], color='green', s=50, alpha=0.4)
        ax.scatter(locs[state_two_mask, 1], locs[state_two_mask, 2], locs[state_two_mask, 6], color='red', s=50, alpha=0.4)
        ax.set_box_aspect([
            np.ptp(np.concatenate((locs[state_zero_mask, 1], locs[state_one_mask, 1], locs[state_two_mask, 1]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 2], locs[state_one_mask, 2], locs[state_two_mask, 2]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 6], locs[state_one_mask, 6], locs[state_two_mask, 6])))
        ])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        
        
        
        plt.show()
        
        