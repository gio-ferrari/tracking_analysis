
import numba as nb
import numpy as np
import numpy.typing as npt
import rocket_fft
from scipy import fft

def _njit(
    signature: str | list[str],
    *,
    cache=True,
    inline="always",
    fastmath=True,
    nogil=True,
    parallel=False,
):
    """
    Wrap numba.njit to reduce boilerplate code.
    SOURCE: https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/_util.py

    We want to build jitted functions with explicit signatures to restrict the argument
    types which are used in the implemetation to float32 or float64. We also want to
    pass specific options consistently: error_model='numpy' and inline='always'. The
    latter is important to profit from auto-parallelization of surrounding code.

    Parameters
    ----------
    signature : str
        The numba signature of the function to be jitted.
    """
    return nb.njit(
        signature,
        cache=cache,
        inline=inline,
        fastmath=fastmath,
        nogil=nogil,
        parallel=parallel,
        error_model="numpy",
    )


# SAFE LOG (from the `iminuit` package)
_TINY_FLOAT = np.finfo(float).tiny


@nb.vectorize("f8(f8)", cache=True, fastmath=True)
def safe_log(x: np.float64):
    # guard against x = 0
    return np.log(np.maximum(_TINY_FLOAT, x))

@_njit("c16(c16, c16)", cache=False)
def safe_compl_divide(num: np.complex128, denom: np.complex128):
    if np.isfinite(num) and np.isfinite(denom) and (np.abs(denom) > _TINY_FLOAT):
        return num / denom
    return np.nan+1j*np.nan

# @nb.njit(cache=True, nogil=True, inline="always", fastmath=True, parallel=False)
# def compute_histo_vals(
#     histogrammable_data: npt.NDArray[np.float64], n_bins: int, histo_range: tuple[float, float]
# ) -> npt.NDArray[np.uint64]:
#     # cnts, bin_edges = np.histogram(histogrammable_data, bin_times)
#     # half_bin_size: float = (bin_edges[1] - bin_edges[0]) * 0.5
#     # start bins at middle of bin; bin_edges are 1 longer than input
#     # -> shorten
#     cnts = histogram1d(histogrammable_data, n_bins, range=histo_range)
#     return cnts.astype(np.uint64)


@_njit("f8[:](f8[:])")
def get_two_smallest_uniq(inp_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # sort array to have the smallest two vals at front
    partitioned = np.partition(inp_arr, 2)
    lowest_val = partitioned[0]
    sec_val = partitioned[1]
    # now get the second smallest unique elem
    # (does not have to be the second elem)
    for i in range(inp_arr.shape[0]):
        if partitioned[i] != partitioned[i + 1]:
            sec_val = partitioned[i + 1]
            break
    return np.array([lowest_val, sec_val])

@_njit("f8(f8[:], f8[:])")
def calc_cost(model_fn_eval: npt.NDArray[np.float64], normed_cnts: npt.NDArray[np.float64]) -> np.float64:
    res = 0.0
    for i in range(len(normed_cnts)):
        if normed_cnts[i] != 0:
            res -= normed_cnts[i] * (np.log(model_fn_eval[i]) - np.log(normed_cnts[i]))
    res *= 2.0
    return res

@_njit("f8[:](f8[:], f8[:])")
def rtfft_convolve(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    `rtfft` stands for `rocketfft` based convolve
    and is the custom implementation of the (cut) FFT based convolution,
    which is used in the `smPyFLIM` package.

    Fast frequency domain convolution:
    Uses the `rocketfft` package to speed up fft and uses numba
    for the convolution.
    """
    out_len = np.maximum(len(x), len(y))
    n: int = fft.next_fast_len(out_len, real=True)
    fft_x = fft.rfft(x, n=n)
    fft_y = fft.rfft(y, n=n)
    return fft.irfft(fft_x * fft_y)[:out_len]


@_njit("f8[:](c16[:], f8[:])")
def half_transform_convolve(
    x_fs: npt.NDArray[np.complex128], y_rs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    x_fs is the fourier space version of x (usually the IRF in FS)
    y_rs is the real space version of y (usually the exp. decay)

    Only y_rs has to be transform for the convolution
    """
    out_len = len(y_rs)
    n: int = fft.next_fast_len(len(y_rs), real=True)
    return fft.irfft(x_fs * fft.rfft(y_rs, n=n))[:out_len]


@_njit("f8[:](f8[:], f8, f8, intp)")
def cust_trunc_expon(
    x: npt.NDArray[np.float64], oo_tau: float, shift: float, closest_idx: np.intp
) -> npt.NDArray[np.float64]:
    """Custom version of the truncated mono exponential decay function
    with a shift paramter"""
    result = np.empty_like(x)  # only "malloc" and dont iterate over the array
    for i in range(0, closest_idx):
        result[i] = 0.0
    oo_tau_shift = oo_tau * shift
    for i in range(closest_idx, x.size):
        result[i] = np.exp(-oo_tau * x[i] + oo_tau_shift)
    return result


@_njit("c16[:](i8, i8, c16[:], c16[:], f8, f8, i8)")
def modulated_half_lorentz(
    rs_len_opt: np.int64,
    fs_len: np.int64,
    exp_coeff: npt.NDArray[np.complex128],
    exp_coeff_closest_idx: npt.NDArray[np.complex128],
    oo_tau: float,
    bin_sz_ns: float,
    closest_idx: np.int64,
) -> npt.NDArray[np.complex128]:
    """This function produces a modulated Lorentzian in Fourier space,
    which is the transform of a shifted exponential"""
    result = np.empty(fs_len, dtype=np.complex128)  # only "malloc" and dont iterate over the array
    real_exp_fact = np.exp(-bin_sz_ns * oo_tau)
    real_exp_fact_rs_len = np.exp(-(rs_len_opt - closest_idx) * bin_sz_ns * oo_tau)
    # real_exp_fact_closest_idx = np.exp(- closest_idx * bin_sz_ns * oo_tau)
    for i in range(0, fs_len):
        result[i] = safe_compl_divide(exp_coeff_closest_idx[i] - real_exp_fact_rs_len, 1 - real_exp_fact * exp_coeff[i])
    # print(result)
    return result

@_njit("f8(f8[:], intp, f8)")
def calc_delta_t(t: npt.NDArray[np.float64], closest_idx: np.intp, shift: float) -> float:
    if np.round(t[closest_idx], 8) == np.round(shift, 8):
        return 0.0
    else:
        return shift - t[closest_idx]  # negative val! -> intended

@_njit("f8[:](f8[:], i8)")
def pad_right_withzero(
    arr_topad: npt.NDArray[np.float64],
    ax_len: np.int64
) -> npt.NDArray[np.float64]:
    if arr_topad.size >= ax_len:
        return arr_topad
    else:
        arr_padded = np.zeros(ax_len)
        arr_padded[:arr_topad.size] = arr_topad
        return arr_padded

@_njit("f8[:](f8[:], f8[:], c16[:], f8, i8, f8, f8, i8, i8, i8, c16[:], c16[:])")
def nb_trunc_shift_exp_conv_eval_fullfs_wobg(
    x: npt.NDArray[np.float64],
    norm_irf_rs: npt.NDArray[np.float64],  # rs = real space
    norm_irf_fs: npt.NDArray[np.complex128],  # fs = fourier space
    tau: float,
    closest_idx: np.int64,
    delta_t: float,
    bin_sz_ns: float,
    rs_len_original: np.int64,
    rs_len_opt: np.int64,
    fs_len: np.int64,
    exp_coeff: npt.NDArray[np.complex128],
    exp_coeff_closest_idx: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    oo_tau = 1.0 / tau  # reduce no. of divisions
    """
    This function computes and return the fitting function
    using the continuous convolution with shifted exponential model
    and performing all computations in Fourier space.
    The transformed IRF is multiplied with a Lorentzian
    and then transformed back
    """
    lorentz_eval = np.exp(delta_t * oo_tau) * modulated_half_lorentz(
        rs_len_opt, fs_len, exp_coeff, exp_coeff_closest_idx, oo_tau, bin_sz_ns, closest_idx
    )

    conv_exp = np.empty(x.size, dtype=np.float64)
    conv_exp = (
        -np.expm1(-bin_sz_ns * oo_tau)
        * fft.irfft(np.multiply(lorentz_eval, norm_irf_fs))[:rs_len_original]
    )
    conv_exp = pad_right_withzero(conv_exp, rs_len_original)
    
    # compute correction
    corr_const_fac = -np.expm1(delta_t * oo_tau) if delta_t != 0.0 else 0.0
    corr_const = corr_const_fac * norm_irf_rs
    end_idx = np.minimum(closest_idx + norm_irf_rs.size, x.size)
    slice_len = end_idx - closest_idx

    # add correction to the convolution
    for i in range(0, slice_len):
        conv_exp[i + closest_idx - 1] += corr_const[i]

    sum_conv_exp = np.sum(conv_exp)
    # Protect against zero-division error
    if np.round(sum_conv_exp, 8) == np.round(0.0, 8):
        return conv_exp
    else:
        inv_norm_fac = 1.0 / np.sum(conv_exp)
        # return no_phot * inv_norm_fac * conv_exp
        # MRJD test:
        return inv_norm_fac * conv_exp

@_njit("f8[:](f8[:], f8[:], c16[:], f8, f8, f8, i8, f8, f8, f8, i8, i8, i8, c16[:], c16[:])")
def nb_trunc_shift_biexp_conv_eval_fullfs(
    x: npt.NDArray[np.float64],
    norm_irf_rs: npt.NDArray[np.float64],  # rs = real space
    norm_irf_fs: npt.NDArray[np.complex128],  # fs = fourier space
    tau1: float,
    tau2: float,
    rel_amp1: float,
    closest_idx: np.int64,
    delta_t: float,
    c_bg: float,
    bin_sz_ns: float,
    rs_len_original: np.int64,
    rs_len_opt: np.int64,
    fs_len: np.int64,
    exp_coeff: npt.NDArray[np.complex128],
    exp_coeff_closest_idx: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    oo_tau1 = 1.0 / tau1  # reduce no. of divisions
    oo_tau2 = 1.0 / tau2  # reduce no. of divisions
    """
    This function computes and return the fitting function
    using the continuous convolution with shifted exponential model
    and performing all computations in Fourier space.
    The transformed IRF is multiplied with a Lorentzian
    and then transformed back
    """
    # print('using FS version')
    lorentz_eval_1 = np.exp(delta_t * oo_tau1) * modulated_half_lorentz(
        rs_len_opt, fs_len, exp_coeff, exp_coeff_closest_idx, oo_tau1, bin_sz_ns, closest_idx
    )

    lorentz_eval_2 = np.exp(delta_t * oo_tau2) * modulated_half_lorentz(
        rs_len_opt, fs_len, exp_coeff, exp_coeff_closest_idx, oo_tau2, bin_sz_ns, closest_idx
    )

    conv_exp = np.empty(x.size, dtype=np.float64)
    conv_exp = (
        fft.irfft(np.multiply(-np.expm1(-bin_sz_ns * oo_tau1) 
                            * lorentz_eval_1
                            - np.expm1(-bin_sz_ns * oo_tau2) 
                            * lorentz_eval_2,
                            norm_irf_fs))[:rs_len_original]
    )

    # compute correction
    corr_const_fac = (
        (-np.expm1(delta_t * oo_tau1) * rel_amp1 - np.expm1(delta_t * oo_tau2) * (1 - rel_amp1))
        if delta_t != 0.0
        else 0.0
    )
    corr_const = corr_const_fac * norm_irf_rs
    end_idx = np.minimum(closest_idx + norm_irf_rs.size, x.size)
    slice_len = end_idx - closest_idx

    # add correction to the convolution
    for i in range(0, slice_len):
        conv_exp[i + closest_idx - 1] += corr_const[i]

    # add background
    conv_exp += c_bg

    sum_conv_exp = np.sum(conv_exp)
    # Protect against zero-division error
    if np.round(sum_conv_exp, 8) == np.round(0.0, 8):
        return conv_exp
    else:
        inv_norm_fac = 1.0 / np.sum(conv_exp)
        # return no_phot * inv_norm_fac * conv_exp
        # MRJD test:
        return inv_norm_fac * conv_exp


@_njit("f8[:](f8[:], f8[:], f8, f8, f8)")
def nb_amp_fit_exp(
    x: npt.NDArray[np.float64],
    irf_normed_vals: npt.NDArray[np.float64],
    amp: float,
    tau: float,
    bg: float,
) -> npt.NDArray[np.float64]:
    oo_tau = 1.0 / tau
    exp_model = amp * np.exp(-x * oo_tau)
    return rtfft_convolve(exp_model, irf_normed_vals) + bg


@_njit("f8(f8, f8, f8)")
def nb_norm_fit_calc_norm_const(bin_sz_ns: float, oo_tau: float, tot_time_ns: np.float64) -> float:
    return (-np.expm1(-bin_sz_ns * oo_tau)) / (-np.expm1(-(tot_time_ns + bin_sz_ns) * oo_tau))


@_njit("f8[:](f8[:], f8[:], f8, f8, f8, f8, i8, u4)")
def nb_norm_fit(
    x: npt.NDArray[np.float64],
    irf_normed_vals: npt.NDArray[np.float64],
    tau: float,
    c_bg: float,
    bin_sz_ns: float,
    tot_time_ns: np.float64,
    n_bins: int,
    n_phot: np.uint64,
) -> npt.NDArray[np.float64]:
    oo_tau = 1.0 / tau
    norm_const = nb_norm_fit_calc_norm_const(bin_sz_ns, oo_tau, tot_time_ns)
    exp_eval = norm_const * np.exp(-x * oo_tau)
    inv_fac = 1.0 / (1.0 + c_bg * n_bins)
    return n_phot * (rtfft_convolve(exp_eval, irf_normed_vals) + c_bg) * inv_fac

@_njit("c16[:](i8, i8)")
def calc_exp_coeff(rs_len_opt: np.int64,
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

@_njit("c16[:](i8, i8)")
def calc_exp_coeff_inv(rs_len_opt: np.int64,
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

@_njit("c16[:](i8, i8, i8, i8, c16[:], c16[:], c16[:])")
def update_exp_coeff(closest_idx: np.int64,
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