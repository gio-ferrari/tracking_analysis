from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from iminuit import Minuit

from functools import partial
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from matplotlib.animation import FuncAnimation

from tracking_analysis.postprocessing import DataPostProcessor
from tracking_analysis.config.configvar import HMM_WLT_SUFFIX

D0_ATTO643_NM = 18.5
TAU0_ATTO643_NS = 3.9
ARM_LENGTH_PRIOR_NM = 4.3
KINK_Z_PRIOR_NM = 15
KINK_Z_SIGMA_NM = 0.05
TOTAL_DNA_LENGTH_PRIOR_NM = 19.3
TOTAL_DNA_LENGTH_SIGMA_NM = 0.04

state_zero_col = "#4dac26"
state_one_col = "#e66101"
state_two_col = '#225ea8'
dye_col = '#d01c8b'

def cost_single_point(x, y, z, x0, y0, z0, cov_xx, cov_yy, cov_xy, cov_zz, det_cov):
    delta_x = x - x0
    delta_y = y - y0
    delta_z = z - z0
    
    lateral_term = (1/2)*np.log(det_cov) + (1/2) * (1/det_cov) * (cov_yy*delta_x**2 - 2*cov_xy*delta_x*delta_y + cov_xx*delta_y**2)
    axial_term = (1/2)*np.log(cov_zz) + (1/2)*delta_z**2/cov_zz
    
    return lateral_term + axial_term

def kink_localizer(coord_down, coord_up, z_kink, arm_length):
    
    #check whether points are not too far from kink plane
    if ((np.abs(coord_down[2] - z_kink) > arm_length) or (np.abs(coord_up[2] - z_kink) > arm_length)):
        x_kink = None
        y_kink = None
        
    else:
        #compute projections of radii on kink plane
        radius_down_proj = np.sqrt(arm_length**2 - (coord_down[2] - z_kink)**2)
        radius_up_proj = np.sqrt(arm_length**2 - (coord_up[2] - z_kink)**2)
        #project coords on kink plane
        coord_down_proj = coord_down[:2]
        coord_up_proj = coord_up[:2]
        #compute distance of localizations on kink plane
        loc_dist_proj = np.linalg.norm(coord_down_proj - coord_up_proj)
        
        #check whether circumferences intersect on kink plane
        if (((radius_down_proj + radius_up_proj) < loc_dist_proj)
                        or (loc_dist_proj + min(radius_down_proj, radius_up_proj) < max(radius_down_proj, radius_up_proj))):
            x_kink = None
            y_kink = None
            
        else:
            #define versors
            parall_versor = (coord_up_proj - coord_down_proj)
            parall_versor = parall_versor/np.linalg.norm(parall_versor)
            perp_versor = np.array([parall_versor[1], -parall_versor[0]])
            
            #distance between down localization and axis of intersection
            dist_parall = (loc_dist_proj**2 + radius_down_proj**2 - radius_up_proj**2)/(2*loc_dist_proj)
            #half of the chord between intersections
            dist_perp = np.sqrt(radius_down_proj**2 - dist_parall**2)
            
            if perp_versor[0]>0:
                coord_kink_proj = coord_down_proj + dist_parall*parall_versor + dist_perp*perp_versor
            else:
                coord_kink_proj = coord_down_proj + dist_parall*parall_versor - dist_perp*perp_versor
            
            x_kink = coord_kink_proj[0]
            y_kink = coord_kink_proj[1]
        
        
    return x_kink, y_kink             

def anim_update_func(
    frame,
    locs_3d,
    coords_kink,
    coords_c0,
    coords_c1,
    coords_c2,
    mask_state_zero,
    mask_state_one,
    mask_state_two,
    scat_state_zero,
    scat_state_one,
    scat_state_two,
    dye_scat,
    dna_arm_line
):
    scat_state_zero._offsets3d = (
        locs_3d[:frame, 1][mask_state_zero[:frame]],
        locs_3d[:frame, 2][mask_state_zero[:frame]],
        locs_3d[:frame, 7][mask_state_zero[:frame]],
    )
    scat_state_one._offsets3d = (
        locs_3d[:frame, 1][mask_state_one[:frame]],
        locs_3d[:frame, 2][mask_state_one[:frame]],
        locs_3d[:frame, 7][mask_state_one[:frame]],
    )   
    scat_state_two._offsets3d = (
        locs_3d[:frame, 1][mask_state_two[:frame]],
        locs_3d[:frame, 2][mask_state_two[:frame]],
        locs_3d[:frame, 7][mask_state_two[:frame]],
    )
    if np.isclose(locs_3d[frame, 6], 0):
        dye_scat._offsets3d = coords_c0
        dna_arm_line.set_data([coords_kink[0], coords_c0[0][0]], [coords_kink[1], coords_c0[1][0]])
        dna_arm_line.set_3d_properties([coords_kink[2], coords_c0[2][0]])
    elif np.isclose(locs_3d[frame, 6], 1):
        dye_scat._offsets3d = coords_c1
        dna_arm_line.set_data([coords_kink[0], coords_c1[0][0]], [coords_kink[1], coords_c1[1][0]])
        dna_arm_line.set_3d_properties([coords_kink[2], coords_c1[2][0]])
    elif np.isclose(locs_3d[frame, 6], 2):
        dye_scat._offsets3d = coords_c2
        dna_arm_line.set_data([coords_kink[0], coords_c2[0][0]], [coords_kink[1], coords_c2[1][0]])
        dna_arm_line.set_3d_properties([coords_kink[2], coords_c2[2][0]])
    return [scat_state_zero, scat_state_one, scat_state_two, dye_scat, dna_arm_line]

class EndoIVAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path, hmm_filt_done: bool):
        self.post_proc_data = post_proc_data
        self.locs_filepath = locs_filepath
        self.tcspc_data_dir = tcspc_data_dir
        self.hmm_filt_done = hmm_filt_done
        self.how_many_states = self.ask_how_many_states()
        if hmm_filt_done:
            self.locs_hmmfilt = self.load_locs(self.locs_filepath)
        else:
            self.hidden_states, self.hidden_states_rescaled = self.hmm_fit_lt(self.post_proc_data.locs_centered)
            #self.get_jump_bins(self.post_proc_data.locs_centered, self.hidden_states)
            self.locs_hmmfilt = self.hmm_filter(self.post_proc_data.locs_centered, self.hidden_states)
        self.locs_3d = self.calc_height(self.locs_hmmfilt)
        self.num_locs = len(self.locs_3d[:,0])
        print(f'Total number of localizations after filtering: {self.num_locs}')
        if self.how_many_states==2:
            self.plot_3d_2states(self.locs_3d)
        elif self.how_many_states==3:
            self.plot_3d_3states(self.locs_3d)
            self.part_analysis_3states_w2states(self.locs_3d)
            self.analysis_3states(self.locs_3d)

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

    def load_locs(self, locs_filepath):
        """
        This function loads the localization array saved as a .npy
        """
        locs = np.load(locs_filepath)
        self.avg_ph_perloc = np.mean(locs[:,3])
        self.sigma_ph_perloc = np.std(locs[:,3])
        print(f"Average number of photons per localization: {self.avg_ph_perloc} \u00B1 {self.sigma_ph_perloc}")
        return locs

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
        locs_hmmfilt = np.concatenate((
                locs,
                hidden_states.reshape(-1,1),
            ), axis=1)
        locs_hmmfilt = locs_hmmfilt[:-1][
            np.logical_and(
                np.isclose(np.diff(hidden_states), 0, atol=1e-2),
                np.isclose(np.diff(np.concatenate([hidden_states[:1], hidden_states[:-1]])), 0, atol=1e-6)
            )
        ] 
        locs_afterfilt = len(locs_hmmfilt)
        print(f"HMM filtering discarded {locs_beforefilt - locs_afterfilt} localizations")
        print(f"{locs_afterfilt} localizations remaining ({int(locs_afterfilt / locs_beforefilt * 100)}%)")
        
        self.locs_hmmfilts_filename = self.locs_filepath.stem + '_' + HMM_WLT_SUFFIX +'.npy'
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
        plt.scatter(locs[state_zero_mask, 0], locs[state_zero_mask, 1], color=state_zero_col)
        plt.scatter(locs[state_one_mask, 0], locs[state_one_mask, 1], color=state_one_col)
        plt.show()
        
    def plot_3d_2states(self, locs):
        state_zero_mask = np.isclose(locs[:, 6], 0)
        state_one_mask = np.isclose(locs[:, 6], 1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(locs[state_zero_mask, 1], locs[state_zero_mask, 2], locs[state_zero_mask, 7], color=state_zero_col, s=50, alpha=0.4)
        ax.scatter(locs[state_one_mask, 1], locs[state_one_mask, 2], locs[state_one_mask, 7], color=state_one_col, s=50, alpha=0.4)
        ax.set_box_aspect([
            np.ptp(np.concatenate((locs[state_zero_mask, 1], locs[state_one_mask, 1]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 2], locs[state_one_mask, 2]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 7], locs[state_one_mask, 7])))
        ])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
    
    def plot_3d_3states(self, locs):
        state_zero_mask = np.isclose(locs[:, 6], 0)
        state_one_mask = np.isclose(locs[:, 6], 1)
        state_two_mask = np.isclose(locs[:, 6], 2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(locs[state_zero_mask, 1], locs[state_zero_mask, 2], locs[state_zero_mask, 7], color=state_zero_col, s=50, alpha=0.4)
        ax.scatter(locs[state_one_mask, 1], locs[state_one_mask, 2], locs[state_one_mask, 7], color=state_one_col, s=50, alpha=0.4)
        ax.scatter(locs[state_two_mask, 1], locs[state_two_mask, 2], locs[state_two_mask, 7], color=state_two_col, s=50, alpha=0.4)
        ax.set_box_aspect([
            np.ptp(np.concatenate((locs[state_zero_mask, 1], locs[state_one_mask, 1], locs[state_two_mask, 1]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 2], locs[state_one_mask, 2], locs[state_two_mask, 2]))),
            np.ptp(np.concatenate((locs[state_zero_mask, 7], locs[state_one_mask, 7], locs[state_two_mask, 7])))
        ])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        
    def part_analysis_3states_w2states(self, locs):
        self.state_zero_mask = np.isclose(locs[:, 6], 0)
        self.state_one_mask = np.isclose(locs[:, 6], 1)
        self.state_two_mask = np.isclose(locs[:, 6], 2)
        
        gmm_state0 = GaussianMixture(n_components=1, covariance_type='full')
        gmm_state0.fit(locs[self.state_zero_mask][:, [1, 2, 7]])  # Fit on (x, y, z) coordinates

        # Extract means and covariances
        self.means_0 = gmm_state0.means_
        self.covariances_0 = gmm_state0.covariances_
        
        gmm_state1 = GaussianMixture(n_components=1, covariance_type='full')
        gmm_state1.fit(locs[self.state_one_mask][:, [1, 2, 7]])  # Fit on (x, y, z) coordinates

        # Extract means and covariances
        self.means_1 = gmm_state1.means_
        self.covariances_1 = gmm_state1.covariances_

        gmm_state2 = GaussianMixture(n_components=1, covariance_type='full')
        gmm_state2.fit(locs[self.state_two_mask][:, [1, 2, 7]])  # Fit on (x, y, z) coordinates

        # Extract means and covariances
        self.means_2 = gmm_state2.means_
        self.covariances_2 = gmm_state2.covariances_
        
        self.x_kink_01, self.y_kink_01 = kink_localizer(self.means_0[0], self.means_1[0], KINK_Z_PRIOR_NM, ARM_LENGTH_PRIOR_NM)
        self.x_kink_02, self.y_kink_02 = kink_localizer(self.means_0[0], self.means_2[0], KINK_Z_PRIOR_NM, ARM_LENGTH_PRIOR_NM)
        self.x_kink_12, self.y_kink_12 = kink_localizer(self.means_1[0], self.means_2[0], KINK_Z_PRIOR_NM, ARM_LENGTH_PRIOR_NM)
        
        sigmas_0 = np.sqrt(np.array(np.diag(self.covariances_0[0])))
        sigmas_1 = np.sqrt(np.array(np.diag(self.covariances_1[0])))
        sigmas_2 = np.sqrt(np.array(np.diag(self.covariances_2[0])))
        print(sigmas_0)
        print(sigmas_1)
        print(sigmas_2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.means_0[0][0], self.means_0[0][1], self.means_0[0][2], color=state_zero_col, s=50, alpha=0.4)
        ax.scatter(self.means_1[0][0], self.means_1[0][1], self.means_1[0][2], color=state_one_col, s=50, alpha=0.4)
        ax.scatter(self.means_2[0][0], self.means_2[0][1], self.means_2[0][2], color=state_two_col, s=50, alpha=0.4)
        ax.scatter(self.x_kink_01, self.y_kink_01, KINK_Z_PRIOR_NM, color='black', s=50, alpha=0.4, marker='o')
        ax.scatter(self.x_kink_02, self.y_kink_02, KINK_Z_PRIOR_NM, color='black', s=50, alpha=0.4, marker='s')
        ax.scatter(self.x_kink_12, self.y_kink_12, KINK_Z_PRIOR_NM, color='black', s=50, alpha=0.4, marker='^')
        
        ax.set_box_aspect([
            np.ptp(np.concatenate((locs[self.state_zero_mask, 1], locs[self.state_one_mask, 1], locs[self.state_two_mask, 1]))),
            np.ptp(np.concatenate((locs[self.state_zero_mask, 2], locs[self.state_one_mask, 2], locs[self.state_two_mask, 2]))),
            np.ptp(np.concatenate((locs[self.state_zero_mask, 7], locs[self.state_one_mask, 7], locs[self.state_two_mask, 7])))
        ])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        
    def calc_guess_3states_analysis(self):
        '''
        This function calculates guesses for the maximum likelihood analysis in the case of a 3 state trace
        '''
        guess_dict = {
            'z_kink': KINK_Z_PRIOR_NM,
            'dna_length': TOTAL_DNA_LENGTH_PRIOR_NM,
            'x_c0': self.means_0[0][0],
            'y_c0': self.means_0[0][1],
            'z_c0': self.means_0[0][2],
            'cov_c0_xx': self.covariances_0[0][0,0],
            'cov_c0_yy': self.covariances_0[0][1,1],
            'cov_c0_xy': self.covariances_0[0][0,1],
            'cov_c0_zz': self.covariances_0[0][2,2],
            'x_c1': self.means_1[0][0],
            'y_c1': self.means_1[0][1],
            'z_c1': self.means_1[0][2],
            'cov_c1_xx': self.covariances_1[0][0,0],
            'cov_c1_yy': self.covariances_1[0][1,1],
            'cov_c1_xy': self.covariances_1[0][0,1],
            'cov_c1_zz': self.covariances_1[0][2,2],
            'x_c2': self.means_2[0][0],
            'y_c2': self.means_2[0][1],
            'cov_c2_xx': self.covariances_2[0][0,0],
            'cov_c2_yy': self.covariances_2[0][1,1],
            'cov_c2_xy': self.covariances_2[0][0,1],
            'cov_c2_zz': self.covariances_2[0][2,2],
        }
        return guess_dict
    
    def calc_cost_3states_analysis(
        self,
        z_kink,
        dna_length,
        x_c0,
        y_c0,
        z_c0,
        cov_c0_xx,
        cov_c0_yy,
        cov_c0_xy,
        cov_c0_zz,
        x_c1,
        y_c1,
        z_c1,
        cov_c1_xx,
        cov_c1_yy,
        cov_c1_xy,
        cov_c1_zz,
        x_c2,
        y_c2,
        cov_c2_xx,
        cov_c2_yy,
        cov_c2_xy,
        cov_c2_zz,
    ):
        # computing determinant of covariance matrices
        det_cov_c0 = cov_c0_xx*cov_c0_yy - cov_c0_xy**2
        det_cov_c1 = cov_c1_xx*cov_c1_yy - cov_c1_xy**2
        det_cov_c2 = cov_c2_xx*cov_c2_yy - cov_c2_xy**2
            
        # length of the upper arm (i.e. radius of the circumference)
        arm_length = dna_length - z_kink
            
        # fixing kink position based on first two states
        x_kink, y_kink = kink_localizer(
            np.array((x_c0, y_c0, z_c0)),
            np.array((x_c1, y_c1, z_c1)),
            z_kink,
            arm_length
        )
        if (x_kink is None) or (y_kink is None):
            return np.inf
        
        #computing z of the third state
        proj_dist_state2 = np.sqrt(
            (x_c2 - x_kink)**2 + (y_c2 - y_kink)**2
        )
        # state 2 is too far for this arm length: discard
        if proj_dist_state2>arm_length:
            return np.inf
        else:
            z_c2 = z_kink + np.sqrt(arm_length**2 - proj_dist_state2**2)
            
        cost = 0

        cost += np.sum(cost_single_point(
            self.locs_3d[self.state_zero_mask, 1],
            self.locs_3d[self.state_zero_mask, 2],
            self.locs_3d[self.state_zero_mask, 7],
            x_c0,
            y_c0,
            z_c0,
            cov_c0_xx,
            cov_c0_yy,
            cov_c0_xy,
            cov_c0_zz,
            det_cov_c0
        ))

        cost += np.sum(cost_single_point(
            self.locs_3d[self.state_one_mask, 1],
            self.locs_3d[self.state_one_mask, 2],
            self.locs_3d[self.state_one_mask, 7],
            x_c1,
            y_c1,
            z_c1,
            cov_c1_xx,
            cov_c1_yy,
            cov_c1_xy,
            cov_c1_zz,
            det_cov_c1
        ))

        cost += np.sum(cost_single_point(
            self.locs_3d[self.state_two_mask, 1],
            self.locs_3d[self.state_two_mask, 2],
            self.locs_3d[self.state_two_mask, 7],
            x_c2,
            y_c2,
            z_c2,
            cov_c2_xx,
            cov_c2_yy,
            cov_c2_xy,
            cov_c2_zz,
            det_cov_c2
        ))
        # now add the prior cost contribution
        cost += (z_kink - KINK_Z_PRIOR_NM)**2/(2*KINK_Z_SIGMA_NM**2)
        cost += (dna_length - TOTAL_DNA_LENGTH_PRIOR_NM)**2/(2*TOTAL_DNA_LENGTH_SIGMA_NM**2)
        
        return cost
    
    def analysis_3states(self, locs_3d):
        '''
        This function finds the most likely position of the kink in 3d for a trace with 3 states, using the maximum likelihood method.
        '''
        fit_params_w_guess = self.calc_guess_3states_analysis()
        minuit = Minuit(self.calc_cost_3states_analysis, grad=None, **fit_params_w_guess)
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.migrad()
        
        result = minuit.values.to_dict()
        print(fit_params_w_guess)
        print(result, minuit.valid)
        
        x_kink, y_kink = kink_localizer(
            np.array((result['x_c0'], result['y_c0'], result['z_c0'])),
            np.array((result['x_c1'], result['y_c1'], result['z_c1'])),
            result['z_kink'],
            result['dna_length'] - result['z_kink']
        )
        proj_dist_state2 = np.sqrt(
            (result['x_c2'] - x_kink)**2 + (result['y_c2'] - y_kink)**2
        )
        z_c2 = result['z_kink'] + np.sqrt((result['dna_length'] - result['z_kink'])**2 - proj_dist_state2**2)
        
        print('Lateral sigma state 0:', np.sqrt(np.sum(np.linalg.eig([[result['cov_c0_xx'], result['cov_c0_xy']], [result['cov_c0_xy'], result['cov_c0_yy']]])[0])/2))
        print('Lateral sigma state 0 (simplified):', np.sqrt((result['cov_c0_xx'] + result['cov_c0_yy'])/2))
        print('Axial sigma state 0:', np.sqrt(result['cov_c0_zz']))
        print('Lateral sigma state 1:', np.sqrt(np.sum(np.linalg.eig([[result['cov_c1_xx'], result['cov_c1_xy']], [result['cov_c1_xy'], result['cov_c1_yy']]])[0])/2))
        print('Lateral sigma state 1 (simplified):', np.sqrt((result['cov_c1_xx'] + result['cov_c1_yy'])/2))
        print('Axial sigma state 1:', np.sqrt(result['cov_c1_zz']))
        print('Lateral sigma state 2:', np.sqrt(np.sum(np.linalg.eig([[result['cov_c2_xx'], result['cov_c2_xy']], [result['cov_c2_xy'], result['cov_c2_yy']]])[0])/2))
        print('Lateral sigma state 2 (simplified):', np.sqrt((result['cov_c2_xx'] + result['cov_c2_yy'])/2))
        print('Axial sigma state 2:', np.sqrt(result['cov_c2_zz']))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        state_zero_mask = np.isclose(locs_3d[:, 6], 0)
        state_one_mask = np.isclose(locs_3d[:, 6], 1)
        state_two_mask = np.isclose(locs_3d[:, 6], 2)

        ax.scatter(locs_3d[state_zero_mask, 1], locs_3d[state_zero_mask, 2], locs_3d[state_zero_mask, 7], color=state_zero_col, s=50, alpha=0.1)
        ax.scatter(locs_3d[state_one_mask, 1], locs_3d[state_one_mask, 2], locs_3d[state_one_mask, 7], color=state_one_col, s=50, alpha=0.1)
        ax.scatter(locs_3d[state_two_mask, 1], locs_3d[state_two_mask, 2], locs_3d[state_two_mask, 7], color=state_two_col, s=50, alpha=0.1)

        ax.scatter(result['x_c0'], result['y_c0'], result['z_c0'], color=state_zero_col, s=400, alpha=1, marker='*')
        ax.scatter(result['x_c1'], result['y_c1'], result['z_c1'], color=state_one_col, s=400, alpha=1, marker='*')
        ax.scatter(result['x_c2'], result['y_c2'], z_c2, color=state_two_col, s=400, alpha=1, marker='*')
        ax.scatter(x_kink, y_kink, result['z_kink'], color='black', s=100, alpha=1, marker='o')
        
        ax.plot([x_kink, x_kink], [y_kink, y_kink], [0, result['z_kink']], lw='10', alpha=0.7, color='gray')
        ax.plot([x_kink, result['x_c0']], [y_kink, result['y_c0']], [result['z_kink'], result['z_c0']], lw='10', alpha=0.7, color='gray')
        ax.plot([x_kink, result['x_c1']], [y_kink, result['y_c1']], [result['z_kink'], result['z_c1']], lw='10', alpha=0.7, color='gray')
        ax.plot([x_kink, result['x_c2']], [y_kink, result['y_c2']], [result['z_kink'], z_c2], lw='10', alpha=0.7, color='gray')
        
        ax.set_zlim([0,20])
        ax.set_box_aspect([
            np.ptp(locs_3d[:, 1]),
            np.ptp(locs_3d[:, 2]),
            20
        ])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        
        fig_foranim = plt.figure()
        ax = fig_foranim.add_subplot(111, projection='3d') 
        
        scat_state_zero = ax.scatter(locs_3d[state_zero_mask, 1], locs_3d[state_zero_mask, 2], locs_3d[state_zero_mask, 7], color=state_zero_col, s=50, alpha=0.1)
        scat_state_one = ax.scatter(locs_3d[state_one_mask, 1], locs_3d[state_one_mask, 2], locs_3d[state_one_mask, 7], color=state_one_col, s=50, alpha=0.1)
        scat_state_two = ax.scatter(locs_3d[state_two_mask, 1], locs_3d[state_two_mask, 2], locs_3d[state_two_mask, 7], color=state_two_col, s=50, alpha=0.1)

        dye_scat = ax.scatter([result['x_c0']], [result['y_c0']], [result['z_c0']], color=dye_col, s=400, alpha=1, marker='*')
        ax.scatter(x_kink, y_kink, result['z_kink'], color='black', s=80, alpha=1, marker='o')
        
        ax.plot([x_kink, x_kink], [y_kink, y_kink], [0, result['z_kink']], lw='10', alpha=0.7, color='gray')
        dna_arm_line, = ax.plot([x_kink, result['x_c0']], [y_kink, result['y_c0']], [result['z_kink'], result['z_c0']], lw='10', alpha=0.7, color='gray')
        
        ax.set_zlim([0,20])
        ax.set_box_aspect([
            np.ptp(locs_3d[:, 1]),
            np.ptp(locs_3d[:, 2]),
            20
        ])
        
        anim_3d = FuncAnimation(
            fig_foranim,
            partial(anim_update_func,
                locs_3d=locs_3d,
                coords_kink=(x_kink, y_kink, result['z_kink']),
                coords_c0=([result['x_c0']], [result['y_c0']], [result['z_c0']]),
                coords_c1=([result['x_c1']], [result['y_c1']], [result['z_c1']]),
                coords_c2=([result['x_c2']], [result['y_c2']], [z_c2]),
                mask_state_zero=self.state_zero_mask,
                mask_state_one=self.state_one_mask,
                mask_state_two=self.state_two_mask,
                scat_state_zero=scat_state_zero,
                scat_state_one=scat_state_one,
                scat_state_two=scat_state_two,
                dye_scat=dye_scat,
                dna_arm_line=dna_arm_line
                ),
            frames=self.num_locs,
            interval=50,
            blit=False
        )
        
        plt.show()
        
