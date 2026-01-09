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
from tracking_analysis.numba_func import _njit, nb_trunc_shift_exp_conv_eval_fullfs_wobg, calc_delta_t, calc_cost
from tracking_analysis.tools.trunc_shift_cont_conv_mono_fs import Cost_TruncShiftContConvMonoExpFS

irf_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\IRFs\red\IRF_red_20251217_60kHz1.ptu")
data_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX\20251217\nc_sixsites_atto643_red_14_20251217_arrays.ptu")
bckg_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX\20251217\bckg_sixsites_red_20251217_arrays.ptu")

class BasicFit():
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
        
        # get smoothed background histogram
        self.bckg_smooth_hist = gaussian_filter1d(self.bckg_hist, sigma=5)
        
        # get unnormed, normed and normed fourier transformed histogram of irf pulses (separately!)
        self.irf_pulses_hist = self.get_irf_hists(self.irf_raw[2])
        self.irf_pulses_hist_norm = [pulse_hist/np.sum(pulse_hist)*(1/NUM_PULSES) for pulse_hist in self.irf_pulses_hist]
        self.irf_pulses_hist_norm_ft = [fft.rfft(pulse_hist, n=self.nanot_ax_len_opt, workers=-1, norm="backward") for pulse_hist in self.irf_pulses_hist_norm]
        self.fs_len = len(self.irf_pulses_hist_norm_ft[0])
        
        self.last_closest_idx = 0
        self.exp_coeff = self._calc_exp_coeff(self.nanot_ax_len_opt, self.fs_len)
        self.exp_coeff_inv = self._calc_exp_coeff_inv(self.nanot_ax_len_opt, self.fs_len)
        self.exp_coeff_closest_idx = np.ones(self.fs_len, dtype=np.complex128)
        
        self.cost_fn = Cost_TruncShiftContConvMonoExpFS(
            self.nanot_ax_ns,
            self.irf_pulses_hist_norm,
            self.irf_pulses_hist_norm_ft,
            self.data_hist_nt,
            TCSPC_NANOT_RES_PS*1e-3,
            self.n_ph
        )
        
        self.plot_hists()
    
    @staticmethod
    @_njit("c16[:](i8, i8)")
    def _calc_exp_coeff(rs_len_opt: np.int64,
                        fs_len: np.int64):
        '''
        This function computes, at the object creation, the vector of exponentials used later on
        to compute the modulated Lorentzian in Fourier space
        '''
        const_fourier_pref = 2 * np.pi * 1.0j / float(rs_len_opt)
        exp_coeff = np.empty(fs_len, dtype=np.complex128)
        for i in range(fs_len):
            exp_coeff[i] = np.exp(-const_fourier_pref * i)
        return exp_coeff
    
    @staticmethod
    @_njit("c16[:](i8, i8)")
    def _calc_exp_coeff_inv(rs_len_opt: np.int64,
                            fs_len: np.int64):
        '''
        This function computes, at the object creation, the vector of inverse exponentials used later on
        to compute the modulated Lorentzian in Fourier space
        '''
        const_fourier_pref = 2 * np.pi * 1.0j / float(rs_len_opt)
        exp_coeff_inv = np.empty(fs_len, dtype=np.complex128)
        for i in range(fs_len):
            exp_coeff_inv[i] = np.exp(const_fourier_pref * i)
        return exp_coeff_inv
    
    @staticmethod
    @_njit("c16[:](i8, i8, i8, i8, c16[:], c16[:], c16[:])")
    def _update_exp_coeff(closest_idx: np.int64,
                          last_closest_idx: np.int64,
                          rs_len: np.int64,
                          fs_len: np.int64,
                          exp_coeff: npt.NDArray[np.complex128],
                          exp_coeff_inv: npt.NDArray[np.complex128],
                          exp_coeff_last_closest_idx: npt.NDArray[np.complex128],):
        '''
        This function updates the vector of exponentials elevated to the closest_idx power
        '''
        const_fourier_pref = 2 * np.pi * 1.0j / float(rs_len)
        closest_idx_diff = closest_idx - last_closest_idx
        #print('necessary update, difference:', closest_idx_diff)
        if ((closest_idx_diff > 0) and (closest_idx_diff < 21)):
            for i in range(fs_len):
                exp_coeff_last_closest_idx[i] = (exp_coeff_last_closest_idx[i] *
                                                 exp_coeff[i]**closest_idx_diff)
        elif ((closest_idx_diff < 0) and (closest_idx_diff > -21)):
            for i in range(fs_len):
                exp_coeff_last_closest_idx[i] = (exp_coeff_last_closest_idx[i] *
                                                 exp_coeff_inv[i]**(-closest_idx_diff))
        else:
            const_fourier_pref = 2 * np.pi * 1.0j / float(rs_len)
            for i in range(fs_len):
                exp_coeff_last_closest_idx[i] = np.exp(-closest_idx * const_fourier_pref * i)
        return exp_coeff_last_closest_idx
    
    def get_irf_hists(self, irf_raw_nt):
        irf_pulses_hists = []
        for pulse_idx in range(NUM_PULSES):
            restricted_irf_raw = irf_raw_nt[np.logical_and(self.irf_raw[2] >= PULSES_POS_NS[pulse_idx], self.irf_raw[2] < PULSES_POS_NS[pulse_idx] + IRF_WIN_END_NS)]
            irf_pulses_hists.append(np.histogram(restricted_irf_raw, bins = self.nanot_ax_len, range=(0 + OFFSET_FORNANOT_HIST_NS, LASER_PERIOD_NS + OFFSET_FORNANOT_HIST_NS))[0])
        return irf_pulses_hists
        
    def plot_hists(self):
        for pulse_idx in range(NUM_PULSES):
            plt.plot(self.nanot_ax_ns, self.irf_pulses_hist[pulse_idx], alpha=0.5)
        plt.plot(self.nanot_ax_ns, self.data_hist_nt, alpha=0.5)
        self.guess_dict = self.calc_guess()
        example_conv_data = self.example_conv(
            self.guess_dict['tau'],
            self.guess_dict['shift'],
            self.guess_dict['a1'],
            self.guess_dict['a2'],
            self.guess_dict['a3'],
        )
        plt.plot(self.nanot_ax_ns, example_conv_data, color='black')
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
            shift_guess_arr[pulse_idx] = 0#data_max_pos_ns - irf_max_pos_ns
            
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
        
    def example_conv(self, tau, shift, a1, a2, a3):
        convoluted_data = np.zeros(self.nanot_ax_len)
        
        closest_idx = self.nanot_ax_ns.searchsorted(shift, side="right")
        delta_t = calc_delta_t(self.nanot_ax_ns, closest_idx, shift)
        
        self.exp_coeff_last_closest_idx = self._update_exp_coeff(closest_idx,
                                                            self.last_closest_idx,
                                                            self.nanot_ax_len_opt,
                                                            self.fs_len,
                                                            self.exp_coeff,
                                                            self.exp_coeff_inv,
                                                            self.exp_coeff_closest_idx,)
        
        weights = [a1, a2, a3, 1 - (a1 + a2 + a3)]
        
        for pulse_idx in range(NUM_PULSES):
            convoluted_data += weights[pulse_idx]*nb_trunc_shift_exp_conv_eval_fullfs_wobg(
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
        convoluted_data *= self.n_ph - self.n_bckg_ph/self.tot_t_meas_bckg_s*self.tot_t_meas_data_s
        convoluted_data += self.bckg_smooth_hist/self.tot_t_meas_bckg_s*self.tot_t_meas_data_s
        return convoluted_data
        
basicfit = BasicFit(irf_file, data_file, bckg_file)

        