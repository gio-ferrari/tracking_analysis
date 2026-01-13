import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
import rocket_fft
from scipy import fft
from scipy.ndimage import gaussian_filter1d
from iminuit import Minuit

from tracking_analysis.tools.ptu_tools import load_tcspc_data
from tracking_analysis.tools.pqreader_multiharp import load_ptu
from tracking_analysis.config.configvar import NUM_PULSES, LASER_PERIOD_NS, PULSES_POS_NS, TCSPC_NANOT_RES_PS, IRF_WIN_END_NS, LIFETIME_WIN_END_NS, OFFSET_FORNANOT_HIST_NS
from tracking_analysis.numba_func import nb_trunc_shift_exp_conv_eval_fullfs_wobg, calc_delta_t, calc_cost, calc_exp_coeff, calc_exp_coeff_inv, update_exp_coeff

irf_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\IRFs\red\IRF_red_20251217_60kHz1.ptu")
data_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX\20251217\nc_sixsites_atto643_red_14_20251217_arrays.ptu")
bckg_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX\20251217\bckg_sixsites_red_20251217_arrays.ptu")

class LifetimeFit():
    def __init__(self, irf_file: Path, data_file: Path, bckg_file: Path):
        self.irf_file = irf_file
        self.data_file = data_file
        self.bckg_file = bckg_file
        
        # get data from files
        self.irf_raw = load_ptu(irf_file)
        self.data_raw = load_tcspc_data(data_file)
        self.tot_t_meas_data_s = (self.data_raw[0].max() - self.data_raw[0].min())
        self.bckg_raw = load_tcspc_data(bckg_file)
        self.tot_t_meas_bckg_s = (self.bckg_raw[0].max() - self.bckg_raw[0].min())
        
        # get number of nanotime points based on axis length and time resolution, and the optimal value for fft
        self.nanot_ax_len = LASER_PERIOD_NS*1000 // TCSPC_NANOT_RES_PS
        self.nanot_ax_len_opt = fft.next_fast_len(self.nanot_ax_len, real=True)
        
        # get histogram of measurement data, nanotime axis and total number of photons
        self.data_hist_nt, self.nanot_ax_ns = np.histogram(self.data_raw[1], bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))
        self.nanot_ax_ns = self.nanot_ax_ns[:-1]
        self.n_ph = np.sum(self.data_hist_nt)
        self.data_hist_normed_nt = self.data_hist_nt / self.n_ph
        
        # get histogram of measured background
        self.bckg_hist = np.histogram(self.bckg_raw[1], bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))[0]
        self.n_bckg_ph = np.sum(self.bckg_hist)
        
        self.exp_n_bckg_ph = self.n_bckg_ph/self.tot_t_meas_bckg_s*self.tot_t_meas_data_s
        self.signal_contr = (self.n_ph - self.exp_n_bckg_ph)/self.n_ph
        self.bckg_contr = self.exp_n_bckg_ph/self.n_ph
        
        # get smoothed background histogram
        self.bckg_smooth_hist = gaussian_filter1d(self.bckg_hist, sigma=5)
        self.bckg_smooth_norm_hist = self.bckg_smooth_hist / np.sum(self.bckg_smooth_hist)
        
        # get unnormed, normed and normed fourier transformed histogram of irf pulses (separately!)
        self.irf_pulses_hist = self.get_irf_hists(self.irf_raw[2])
        self.irf_pulses_hist_norm = [pulse_hist/np.sum(pulse_hist)*(1/NUM_PULSES) for pulse_hist in self.irf_pulses_hist]
        self.irf_pulses_hist_norm_ft = [fft.rfft(pulse_hist, n=self.nanot_ax_len_opt, workers=-1, norm="backward") for pulse_hist in self.irf_pulses_hist_norm]
        self.fs_len = len(self.irf_pulses_hist_norm_ft[0])
        
        self.last_closest_idx = 0
        self.exp_coeff = calc_exp_coeff(self.nanot_ax_len_opt, self.fs_len)
        self.exp_coeff_inv = calc_exp_coeff_inv(self.nanot_ax_len_opt, self.fs_len)
        self.exp_coeff_closest_idx = np.ones(self.fs_len, dtype=np.complex128)
        
        self.plot_hists()
    
    def get_irf_hists(self, irf_raw_nt):
        irf_pulses_hists = []
        for pulse_idx in range(NUM_PULSES):
            restricted_irf_raw = irf_raw_nt[np.logical_and(self.irf_raw[2] >= PULSES_POS_NS[pulse_idx], self.irf_raw[2] < PULSES_POS_NS[pulse_idx] + IRF_WIN_END_NS)]
            irf_pulses_hists.append(np.histogram(restricted_irf_raw, bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))[0])
        return irf_pulses_hists
        
    def plot_hists(self):
        #for pulse_idx in range(NUM_PULSES):
        #    plt.plot(self.nanot_ax_ns, self.irf_pulses_hist[pulse_idx], alpha=0.5)
        plt.plot(self.nanot_ax_ns, self.data_hist_nt, alpha=0.5)
        self.guess_dict = self.calc_guess()
        
        _, results_dict, _ = self.eval_minuit_fit_monoexp(self.guess_dict)
        
        print(results_dict)
        fitting_fn = self.calc_monoexp(results_dict['tau'], results_dict['shift'], results_dict['a1'], results_dict['a2'], results_dict['a3'])*self.n_ph
        plt.plot(self.nanot_ax_ns, fitting_fn, color='black')
        plt.yscale('log')
        plt.show()
        
    def calc_guess(self):
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
        
        for pulse_idx in range(NUM_PULSES):
            fitting_fn += weights[pulse_idx]*nb_trunc_shift_exp_conv_eval_fullfs_wobg(
                self.nanot_ax_ns,
                self.irf_pulses_hist_norm[pulse_idx],  # rs = real space
                self.irf_pulses_hist_norm_ft[pulse_idx],  # fs = fourier space
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
        Function to be called from Minuit for minimization
        -> has to follow the cost_fn_spec: fn(arg1, arg2, ...)
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
        
basicfit = LifetimeFit(irf_file, data_file, bckg_file)

        