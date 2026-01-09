from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import fft

from tracking_analysis.numba_func import _njit, nb_trunc_shift_exp_conv_eval_fullfs, safe_log, calc_delta_t

@dataclass(slots=True)
class Cost_TruncShiftContConvMonoExpFS():
    t_ax: npt.NDArray[np.float64]
    irf_normed_vals_rs: npt.NDArray[np.float64]
    irf_normed_vals_fs: npt.NDArray[np.complex128]
    cnts: npt.NDArray[np.uint32]
    bin_sz_ns: float
    n_phot: np.uint32
    cnt: int = 0
    
    def __post_init__(self):
        self.rs_len_original: np.int64 = len(self.t_ax)
        self.rs_len_opt: np.int64 = fft.next_fast_len(self.rs_len_original, real=True)
        self.fs_len: np.int64 = self.rs_len_opt // 2 + 1
        self.cnts_norm = self.cnts / np.sum(self.cnts)
        self.cnts_norm_log = self._calc_log_cnts(self.cnts_norm.size, self.cnts_norm)
        self.last_closest_idx: np.int64 = 0
        self.exp_coeff: npt.NDArray[np.complex128] = self._calc_exp_coeff(self.rs_len_opt, self.fs_len)
        self.exp_coeff_inv: npt.NDArray[np.complex128] = self._calc_exp_coeff_inv(self.rs_len_opt, self.fs_len)
        self.exp_coeff_closest_idx: npt.NDArray[np.complex128] = np.ones(self.fs_len, dtype=np.complex128)

    @staticmethod
    @_njit("f8[:](i8, f8[:])")
    def _calc_log_cnts(n_bins, cnts):
        log_cnts = np.empty(n_bins, dtype=np.float64)
        for i in range(len(cnts)):
            log_cnts[i] = safe_log(cnts[i])
        return log_cnts

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
        
    @staticmethod
    @_njit("f8(f8, i8, f8, f8, f8[:], f8[:], f8[:], f8[:], c16[:], f8, u4, i8, i8, i8, c16[:], c16[:])")
    def _calc_cost(
        tau: float,
        closest_idx: np.int64,
        delta_t: float,
        c_bg: float,
        t_ax: npt.NDArray[np.float64],
        normed_cnts: npt.NDArray[np.float64],
        log_normed_cnts: npt.NDArray[np.float64],
        irf_normed_vals_rs: npt.NDArray[np.float64],
        irf_normed_vals_fs: npt.NDArray[np.complex128],
        bin_sz_ns: float,
        n_phot: np.uint32,
        rs_len_original: np.int64,
        rs_len_opt: np.int64,
        fs_len: np.int64,
        exp_coeff: npt.NDArray[np.complex128],
        exp_coeff_closest_idx: npt.NDArray[np.complex128]
    ) -> float:
        """
        JIT compiled numba function that is used internally
        to evaluate the cost function
        """
        model_fn_eval = nb_trunc_shift_exp_conv_eval_fullfs(
            t_ax, irf_normed_vals_rs, irf_normed_vals_fs,
            tau, closest_idx, delta_t, c_bg,
            bin_sz_ns, rs_len_original, rs_len_opt, fs_len,
            exp_coeff, exp_coeff_closest_idx
        )
        # model_fn_eval *= n_phot
        #oo_n_phot = np.float64(1.0 / n_phot)
        res = 0.0
        for i in range(t_ax.shape[0]):
            # both work:
            # faster, but not chi^2
            # normed_cnt = cnts[i] * oo_n_phot
            # res += model_fn_eval[i] - normed_cnt * safe_log(model_fn_eval[i])

            # ted lawrence chi^2
            #normed_cnt = cnts[i] * oo_n_phot
            # normed_cnt = cnts[i]
            #res += model_fn_eval[i] - normed_cnt
            if normed_cnts[i] != 0:
                res -= normed_cnts[i] * (safe_log(model_fn_eval[i]) - log_normed_cnts[i])
        res *= 2.0
        # res /= n_phot - 3  # 3 is the number of parameters(tau, shift, c_bg)
        #print('res = ', res)
        return res

    def calc(self, tau: float, shift: float, c_bg: float) -> float:
        """
        Function to be called from Minuit for minimization
        -> has to follow the cost_fn_spec: fn(arg1, arg2, ...)
        """
        closest_idx = self.t_ax.searchsorted(shift, side="right")
        delta_t = calc_delta_t(self.t_ax, closest_idx, shift)
        #print(closest_idx, self.last_closest_idx)
        if closest_idx != self.last_closest_idx:
            self.exp_coeff_last_closest_idx = self._update_exp_coeff(closest_idx,
                                                                     self.last_closest_idx,
                                                                     self.rs_len_opt,
                                                                     self.fs_len,
                                                                     self.exp_coeff,
                                                                     self.exp_coeff_inv,
                                                                     self.exp_coeff_closest_idx,)
            self.last_closest_idx = closest_idx
        return self._calc_cost(
            tau,
            closest_idx,
            delta_t,
            c_bg,
            self.t_ax,
            self.normed_cnts,
            self.log_normed_cnts,
            self.irf_normed_vals_rs,
            self.irf_normed_vals_fs,
            self.bin_sz_ns,
            self.n_phot,
            self.rs_len_original,
            self.rs_len_opt,
            self.fs_len,
            self.exp_coeff,
            self.exp_coeff_closest_idx,
        )

    def num_eval(self, tau: float, shift: float, c_bg: float) -> npt.NDArray[np.float64]:
        closest_idx = self.t_ax.searchsorted(shift, side="right")
        delta_t = calc_delta_t(self.t_ax, closest_idx, shift)
        if closest_idx != self.last_closest_idx:
            self.exp_coeff_last_closest_idx = self._update_exp_coeff(closest_idx,
                                                                     self.last_closest_idx,
                                                                     self.rs_len_opt,
                                                                     self.fs_len,
                                                                     self.exp_coeff,
                                                                     self.exp_coeff_inv,
                                                                     self.exp_coeff_closest_idx,)
            self.last_closest_idx = closest_idx
        return self.n_phot * nb_trunc_shift_exp_conv_eval_fullfs(
            self.t_ax,
            self.irf_normed_vals_rs,
            self.irf_normed_vals_fs,
            tau,
            closest_idx,
            delta_t,
            c_bg,
            self.bin_sz_ns,
            self.rs_len_original,
            self.rs_len_opt,
            self.fs_len,
            self.exp_coeff,
            self.exp_coeff_closest_idx
        )
