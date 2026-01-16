from copy import deepcopy
import numpy as np
import numpy.typing as npt
from pathlib import Path
import rocket_fft
from scipy import fft
from scipy.ndimage import gaussian_filter1d
from iminuit import Minuit
from tqdm import tqdm
import matplotlib.pyplot as plt

from tracking_analysis.tools.pqreader_multiharp import load_ptu
from tracking_analysis.tools.loc_tools import loc_trace_minflux, loc_trace_minflux_lt
from tracking_analysis.config.configvar import (
    NUM_PULSES,
    LASER_PERIOD_NS,
    PULSES_POS_NS,
    LIFETIME_WIN_END_NS,
    IRF_WIN_END_NS,
    TCSPC_NANOT_RES_PS,
    OFFSET_FORNANOT_HIST_NS,
    STEP_NM,
    LOCS_FILE_SUFFIX_NOLT,
    LOCS_FILE_SUFFIX_LT,
    MIN_PH_MINFLUX_LOC_LT
)
from tracking_analysis.ebp import EBP
from tracking_analysis.tcspcdata import TCSPCData
from tracking_analysis.numba_func import nb_trunc_shift_exp_conv_eval_fullfs_wobg, calc_delta_t, calc_cost, calc_exp_coeff, calc_exp_coeff_inv, update_exp_coeff

class MINFLUXAnalysis():
    def __init__(self, ebp: EBP, tcspc_data: TCSPCData, irf_file: Path, target_n_ph: int, do_lifetime_fit: bool):
        self.ebp = ebp
        self.tcspc_data = tcspc_data
        self.irf_file = irf_file
        self.target_n_ph = target_n_ph
        self.do_lifetime_fit = do_lifetime_fit
        if self.do_lifetime_fit:
            self.choose_locs_t_binning_lt()
            self.lifetime_analysis = LifetimeFitAnalysis(self.tcspc_data, irf_file, self.locs_t_binning_s)
            self.ph_perloc_perpulse, self.ph_perloc_allpulses, self.sbr_perloc, self.lifetime_trace = self.lifetime_analysis.get_n_ph_perpulse_from_lt_fit()
            self.localizations_wlt = self.minflux_localize_wlt()
            self.localizations_wlt = self.localizations_wlt[(~np.isnan(self.localizations_wlt[:,1])) & (~np.isnan(self.localizations_wlt[:,2]))]
            self.save_locs()
        else:
            self.choose_locs_t_binning()
            self.calc_ph_perloc_perpulse()
            self.localizations = self.minflux_localize()
            self.localizations_wlt = self.localizations_wlt[(~np.isnan(self.localizations_wlt[:,1])) & (~np.isnan(self.localizations_wlt[:,2]))]
            self.save_locs()
        
    def choose_locs_t_binning(self):
        """
        this function allows the user to choose the time binning used for MINFLUX localizations
        """
        target_locs_t_binning_s = float(self.target_n_ph) / self.tcspc_data.tot_counts_timegated
        print(f"Localization time binning (in s) to obtain {self.target_n_ph} photons per bin: {target_locs_t_binning_s}")
        self.locs_t_binning_s = float(input("Choose localization time binning (in s): "))
        self.avg_n_ph_perloc = self.tcspc_data.tot_counts_timegated * self.locs_t_binning_s
        print(f"Average number of photons per localization: {self.avg_n_ph_perloc}")
        # compute background photons per localization bin, per pulse
        self.bckg_ph_perloc_perpulse = np.empty(NUM_PULSES, dtype=np.float64)
        for pulse_idx in range(NUM_PULSES):
            if not self.tcspc_data.use_dark_cnts_choice:
                self.bckg_ph_perloc_perpulse[pulse_idx] = self.tcspc_data.bckg_counts_timegated_perpulse[pulse_idx] * self.locs_t_binning_s
                print(f"Expected background photons per localization for pulse {pulse_idx + 1}: {self.bckg_ph_perloc_perpulse[pulse_idx]}")
            else:
                self.bckg_dark_cnts_ph_perloc_perpulse = (self.tcspc_data.bckg_dark_cnts_timegated / NUM_PULSES) * self.locs_t_binning_s
                self.bckg_ph_perloc_perpulse[pulse_idx] = ((self.tcspc_data.baseline_bckg_cnts_timegated_perpulse[pulse_idx] /
                                                           (self.tcspc_data.sgnl_cnts_forbaseline_sbr_timegated - self.tcspc_data.bckg_dark_cnts_timegated) *
                                                           (self.tcspc_data.tot_counts_timegated - self.tcspc_data.bckg_dark_cnts_timegated)) *
                                                           self.locs_t_binning_s +
                                                           self.bckg_dark_cnts_ph_perloc_perpulse) 
                print(f"Expected background photons per localization for pulse {pulse_idx + 1}: {self.bckg_ph_perloc_perpulse[pulse_idx]}")
                print(f"Of which from dark counts: {self.bckg_dark_cnts_ph_perloc_perpulse}")
        # compute histograms of photons for each pulse using the chosen time binning
        self.locs_bin_edges = np.arange(self.tcspc_data.start_t_s, np.min((self.tcspc_data.end_t_s, self.tcspc_data.emitter_stop_t_s)), self.locs_t_binning_s)
        
    def choose_locs_t_binning_lt(self):
        """
        this function allows the user to choose the time binning used for MINFLUX localizations
        """
        target_locs_t_binning_s = float(self.target_n_ph) / self.tcspc_data.avg_tot_counts
        print(f"Localization time binning (in s) to obtain {self.target_n_ph} photons per bin: {target_locs_t_binning_s}")
        self.locs_t_binning_s = float(input("Choose localization time binning (in s): "))
        self.avg_n_ph_perloc = self.tcspc_data.avg_tot_counts * self.locs_t_binning_s
        print(f"Average number of photons per localization: {self.avg_n_ph_perloc}")
        # compute histograms of photons for each pulse using the chosen time binning
        self.locs_bin_edges = np.arange(self.tcspc_data.start_t_s, np.min((self.tcspc_data.end_t_s, self.tcspc_data.emitter_stop_t_s)), self.locs_t_binning_s)
        
    def calc_ph_perloc_perpulse(self):
        """
        This function computes the histograms of the counts for each localization bin and for all pulses separately and
        alltogether, and the SBR for each localization
        """
        print(f"Total number of localizations: {len(self.locs_bin_edges) - 1}")
        self.ph_perloc_perpulse = np.empty((NUM_PULSES, len(self.locs_bin_edges) - 1), dtype=np.int64)
        for pulse_idx in range(NUM_PULSES):
            self.ph_perloc_perpulse[pulse_idx, :], _ = np.histogram(self.tcspc_data.abs_time_s_foranalysis_perpulse[pulse_idx], self.locs_bin_edges)
        self.ph_perloc_allpulses = np.sum(self.ph_perloc_perpulse, axis=0)
        # Now compute the SBR for each individual localization by dividing the number of photons of each bin for the expected number of background
        # photons in the same time interval
        self.sbr_perloc = deepcopy(self.ph_perloc_allpulses) / np.sum(self.bckg_ph_perloc_perpulse) - 1
        
    def minflux_localize(self):
        """
        This function computes MINFLUX localizations for the selected part of the trace
        """
        locs_coords = loc_trace_minflux(self.ph_perloc_perpulse, self.bckg_ph_perloc_perpulse, self.sbr_perloc, self.ebp.psf_fits, STEP_NM)
        return np.concatenate((
                self.locs_bin_edges[:-1].reshape(-1,1),
                locs_coords,
                self.ph_perloc_allpulses.reshape(-1,1), 
                self.sbr_perloc.reshape(-1,1)
            ), axis=1)
        
    def minflux_localize_wlt(self):
        """
        This function computes MINFLUX localizations for the selected part of the trace using lifetime fitting
        """
        locs_coords = loc_trace_minflux_lt(self.ph_perloc_perpulse, self.ebp.psf_fits, STEP_NM)
        return np.concatenate((
                self.locs_bin_edges[:-1].reshape(-1,1),
                locs_coords,
                self.ph_perloc_allpulses.reshape(-1,1), 
                self.sbr_perloc.reshape(-1,1),
                self.lifetime_trace.reshape(-1,1)
            ), axis=1)
       
    def save_locs(self):
        """
        This function save the localizations as a .npy file, to be accessed and used for post-processing
        """
        if self.do_lifetime_fit:
            self.locs_results_filename = (self.tcspc_data.tcspc_data_filename + LOCS_FILE_SUFFIX_LT + 'from' + str(int(self.tcspc_data.start_t_s)) +
                                        'to' + str(int(self.tcspc_data.end_t_s)) + 's_' + str(int(self.locs_t_binning_s * 1e3)) + 'ms_binning.npy')
            self.locs_results_filepath = self.tcspc_data.tcspc_data_dir / self.locs_results_filename
            np.save(self.locs_results_filepath, self.localizations_wlt)
        else:
            self.locs_results_filename = (self.tcspc_data.tcspc_data_filename + LOCS_FILE_SUFFIX_NOLT + 'from' + str(int(self.tcspc_data.start_t_s)) +
                                        'to' + str(int(self.tcspc_data.end_t_s)) + 's_' + str(int(self.locs_t_binning_s * 1e3)) + 'ms_binning.npy')
            self.locs_results_filepath = self.tcspc_data.tcspc_data_dir / self.locs_results_filename
            np.save(self.locs_results_filepath, self.localizations)
        
class LifetimeFitAnalysis():
    def __init__(self, tcspc_data: TCSPCData, irf_file: Path, locs_t_binning_s: float):
        self.irf_raw = load_ptu(irf_file)
        self.tcspc_data = tcspc_data
        self.locs_t_binning_s = locs_t_binning_s
        self.locs_bin_edges = np.arange(self.tcspc_data.start_t_s, np.min((self.tcspc_data.end_t_s, self.tcspc_data.emitter_stop_t_s)), self.locs_t_binning_s)
        
        # get number of nanotime points based on axis length and time resolution, and the optimal value for fft
        self.nanot_ax_len = LASER_PERIOD_NS*1000 // TCSPC_NANOT_RES_PS
        self.nanot_ax_len_opt = fft.next_fast_len(self.nanot_ax_len, real=True)
        
        # get histogram of measured background
        self.bckg_hist = np.histogram(self.tcspc_data.bckg_rel_time_ns, bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))[0]
        self.n_bckg_ph = np.sum(self.bckg_hist)
        
        self.exp_n_bckg_ph_perloc = self.n_bckg_ph/self.tcspc_data.tot_t_measuring_bckg_s*self.locs_t_binning_s
        
        # get smoothed background histogram
        self.bckg_smooth_hist = gaussian_filter1d(self.bckg_hist.astype(np.float64), sigma=5)
        self.bckg_smooth_norm_hist = self.bckg_smooth_hist / np.sum(self.bckg_smooth_hist)
        
        # get unnormed, normed and normed fourier transformed histogram of irf pulses (separately!)
        self.irf_pulses_hist = self.get_irf_hists(self.irf_raw[2])
        self.irf_pulses_hist_norm = [pulse_hist/np.sum(pulse_hist)*(1/NUM_PULSES) for pulse_hist in self.irf_pulses_hist]
        self.irf_pulses_hist_norm_ft = [fft.rfft(pulse_hist, n=self.nanot_ax_len_opt, workers=-1, norm="backward") for pulse_hist in self.irf_pulses_hist_norm]
        self.fs_len = len(self.irf_pulses_hist_norm_ft[0])
        
    def get_irf_hists(self, irf_raw_nt):
        irf_pulses_hists = []
        for pulse_idx in range(NUM_PULSES):
            restricted_irf_raw = irf_raw_nt[np.logical_and(self.irf_raw[2] >= PULSES_POS_NS[pulse_idx], self.irf_raw[2] < PULSES_POS_NS[pulse_idx] + IRF_WIN_END_NS)]
            irf_pulses_hists.append(np.histogram(restricted_irf_raw, bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))[0])
        return irf_pulses_hists
        
    def get_n_ph_perpulse_from_lt_fit(self):
        """
        This function gets the number of photons per pulse per localization using the lifetime fit method
        """
        n_loc = len(self.locs_bin_edges) - 1
        ph_perloc_perpulse = np.empty((NUM_PULSES, n_loc), dtype=np.float64)
        lifetime_trace = np.empty(n_loc, dtype=np.float64)
        tot_ph_perloc = np.empty(n_loc, dtype=np.int64)
        sbr_perloc = np.empty(n_loc, dtype=np.float64)
        for loc_idx, loc_start_s in tqdm(enumerate(self.locs_bin_edges[:-1]), total=n_loc):
            # reset helpers arrays used to fit
            self.last_closest_idx = 0
            self.exp_coeff = calc_exp_coeff(self.nanot_ax_len_opt, self.fs_len)
            self.exp_coeff_inv = calc_exp_coeff_inv(self.nanot_ax_len_opt, self.fs_len)
            self.exp_coeff_closest_idx = np.ones(self.fs_len, dtype=np.complex128)
            # get relative times inside localization time range
            start_loc_idx, end_loc_idx = np.searchsorted(self.tcspc_data.filt_abs_time_s, [loc_start_s, loc_start_s + self.locs_t_binning_s])
            rel_times_inloc = self.tcspc_data.filt_rel_time_ns[start_loc_idx:end_loc_idx]
            if len(rel_times_inloc)>MIN_PH_MINFLUX_LOC_LT:
                # get histogram of measurement data, nanotime axis and total number of photons
                self.data_hist_nt, self.nanot_ax_ns = np.histogram(rel_times_inloc, bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))
                self.nanot_ax_ns = self.nanot_ax_ns[:-1]
                self.n_ph = np.sum(self.data_hist_nt)
                tot_ph_perloc[loc_idx] = self.n_ph
                sbr_perloc[loc_idx] = self.n_ph / self.exp_n_bckg_ph_perloc
                self.data_hist_normed_nt = self.data_hist_nt / self.n_ph
                
                self.signal_contr = (self.n_ph - self.exp_n_bckg_ph_perloc)/self.n_ph
                self.bckg_contr = self.exp_n_bckg_ph_perloc/self.n_ph
                
                guess_dict = self.calc_guess()
                
                fit_fval, results_dict, fit_is_valid = self.eval_minuit_fit_monoexp(guess_dict)
                if fit_is_valid:
                    ph_perloc_perpulse[0, loc_idx] = results_dict['a1']*self.n_ph
                    ph_perloc_perpulse[1, loc_idx] = results_dict['a2']*self.n_ph
                    ph_perloc_perpulse[2, loc_idx] = results_dict['a3']*self.n_ph
                    ph_perloc_perpulse[3, loc_idx] = (1 - (results_dict['a1'] + results_dict['a2'] + results_dict['a3']))*self.n_ph
                    lifetime_trace[loc_idx] = results_dict['tau']
                else:
                    ph_perloc_perpulse[0, loc_idx] = np.nan
                    ph_perloc_perpulse[1, loc_idx] = np.nan
                    ph_perloc_perpulse[2, loc_idx] = np.nan
                    ph_perloc_perpulse[3, loc_idx] = np.nan
                    lifetime_trace[loc_idx] = np.nan
            else:
                ph_perloc_perpulse[0, loc_idx] = np.nan
                ph_perloc_perpulse[1, loc_idx] = np.nan
                ph_perloc_perpulse[2, loc_idx] = np.nan
                ph_perloc_perpulse[3, loc_idx] = np.nan
                lifetime_trace[loc_idx] = np.nan
                
        return ph_perloc_perpulse, tot_ph_perloc, sbr_perloc, lifetime_trace 
            
    def calc_guess(self):
        """
        This function computes reasonable guesses for the MLE fit
        """
        tau_guess_arr = np.empty(NUM_PULSES)
        shift_guess_arr = np.empty(NUM_PULSES)
        n_ph_guess_arr = np.empty(NUM_PULSES)
        
        for pulse_idx in range(NUM_PULSES):
            # start by getting the timegated histogram for each window
            start_idx = int(PULSES_POS_NS[pulse_idx] / (TCSPC_NANOT_RES_PS*1e-3))
            end_idx = int((PULSES_POS_NS[pulse_idx] + LIFETIME_WIN_END_NS) / (TCSPC_NANOT_RES_PS*1e-3))
            timegated_ax = self.nanot_ax_ns[start_idx:end_idx]
            timegated_data = self.data_hist_nt[start_idx:end_idx]
            
            # now get the number of photons in the timegated window
            n_ph_guess_arr[pulse_idx] = np.sum(timegated_data)
            
            # get the time position of the maximum of the pulse
            data_max_pos_ns = np.argmax(timegated_data)*TCSPC_NANOT_RES_PS*1e-3 + PULSES_POS_NS[pulse_idx]
            
            # now get the time position of the maximum of the corresponding IRF pulse
            irf_max_pos_ns = np.argmax(self.irf_pulses_hist[pulse_idx])*TCSPC_NANOT_RES_PS*1e-3
            
            # get the shift guess from their difference
            shift_guess_arr[pulse_idx] = data_max_pos_ns - irf_max_pos_ns
            
            # now get the lifetime guess
            tau_guess_arr[pulse_idx] = np.sum(timegated_ax*timegated_data)/n_ph_guess_arr[pulse_idx] - irf_max_pos_ns
            
        guess_dict = {
            'tau': np.mean(tau_guess_arr),
            'shift': np.mean(shift_guess_arr),
            'a1': n_ph_guess_arr[0]/np.sum(n_ph_guess_arr),
            'a2': n_ph_guess_arr[1]/np.sum(n_ph_guess_arr),
            'a3': n_ph_guess_arr[2]/np.sum(n_ph_guess_arr),
        }
        return guess_dict
        
    def calc_monoexp(self, tau: np.float64, shift: np.float64, a1: np.float64, a2: np.float64, a3: np.float64) -> npt.NDArray[np.float64]:
        """
        This function computes the full fitting function, consisting of a shifted monoexponential convoluted with the IRF, where each IRF peak is
        rescaled with a factor, and the smoothed background is then summed. Free parameters are: lifetime, exponential time offset and 3 out of 4 multiplicative
        factors for the IRF peaks (the last one is computed to keep normalization correct).
        """
        fitting_fn = np.zeros(self.nanot_ax_len)
        weights = [a1, a2, a3, 1 - (a1 + a2 + a3)]
        closest_idx = self.nanot_ax_ns.searchsorted(shift, side="right")
        delta_t = calc_delta_t(self.nanot_ax_ns, closest_idx, shift)
        
        if closest_idx != self.last_closest_idx:
            self.exp_coeff_last_closest_idx = update_exp_coeff(closest_idx,
                                                                     self.last_closest_idx,
                                                                     self.nanot_ax_len_opt,
                                                                     self.fs_len,
                                                                     self.exp_coeff,
                                                                     self.exp_coeff_inv,
                                                                     self.exp_coeff_closest_idx,)
            self.last_closest_idx = closest_idx
        
        complete_irf = np.zeros(self.nanot_ax_len)
        complete_irf_ft = np.zeros(len(self.irf_pulses_hist_norm_ft[0]), dtype=np.complex128)
        for pulse_idx in range(NUM_PULSES):
            complete_irf += weights[pulse_idx]*self.irf_pulses_hist_norm[pulse_idx]
            complete_irf_ft += weights[pulse_idx]*self.irf_pulses_hist_norm_ft[pulse_idx]
        fitting_fn += nb_trunc_shift_exp_conv_eval_fullfs_wobg(
            self.nanot_ax_ns,
            complete_irf,  # rs = real space
            complete_irf_ft,  # fs = fourier space
            tau,
            closest_idx,
            delta_t,
            TCSPC_NANOT_RES_PS*1e-3,
            self.nanot_ax_len,
            self.nanot_ax_len_opt,
            self.fs_len,
            self.exp_coeff,
            self.exp_coeff_closest_idx,
        )
        fitting_fn *= self.signal_contr
        fitting_fn += self.bckg_smooth_norm_hist*self.bckg_contr
        return fitting_fn
    
    def calc_cost_monoexp(self, tau: np.float64, shift: np.float64, a1: np.float64, a2: np.float64, a3: np.float64) -> np.float64:
        """
        Function to be called from Minuit for minimization.
        Computes the cost function for a MLE fit using a convoluted shifted monoexponential decay with time-correlated background added.
        """
        fitting_fn = self.calc_monoexp(tau, shift, a1, a2, a3)
        res = calc_cost(fitting_fn, self.data_hist_normed_nt)
        return res
        
    def eval_minuit_fit_monoexp(
        self,
        fit_params_w_guess: dict[str, float]
    ) -> tuple[float | None, dict[str, float], bool]:
        """
        Run the fit with Minuit
        """
        minuit = Minuit(self.calc_cost_monoexp, grad=None, **fit_params_w_guess)
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.migrad()

        return minuit.fval, minuit.values.to_dict(), minuit.valid
        

        