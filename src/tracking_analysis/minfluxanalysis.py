from copy import deepcopy
import numpy as np
from pathlib import Path

from tracking_analysis.tools.pqreader_multiharp import load_ptu
from tracking_analysis.tools.loc_tools import loc_trace_minflux
from tracking_analysis.config.configvar import (
    NUM_PULSES,
    STEP_NM,
    LOCS_FILE_SUFFIX,
)
from tracking_analysis.ebp import EBP
from tracking_analysis.tcspcdata import TCSPCData

class MINFLUXAnalysis():
    def __init__(self, ebp: EBP, tcspc_data: TCSPCData, irf_file: Path, target_n_ph: int, do_lifetime_fit: bool):
        self.ebp = ebp
        self.tcspc_data = tcspc_data
        self.irf_file = irf_file
        self.target_n_ph = target_n_ph
        self.do_lifetime_fit = do_lifetime_fit
        if self.do_lifetime_fit:
            self.irf_raw = load_ptu(irf_file)
        else:
            self.choose_locs_t_binning()
            self.calc_ph_perloc_perpulse()
            self.localizations = self.minflux_localize()
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
        
    def calc_ph_perloc_perpulse(self):
        """
        This function computes the histograms of the counts for each localization bin and for all pulses separately and
        alltogether, and the SBR for each localization
        """
        # compute histograms of photons for each pulse using the chosen time binning
        self.locs_bin_edges = np.arange(self.tcspc_data.start_t_s, np.min((self.tcspc_data.end_t_s, self.tcspc_data.emitter_stop_t_s)), self.locs_t_binning_s)
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
        This function calls numba optimized functions to compute MINFLUX localizations for the selected part of the trace
        """
        locs_coords = loc_trace_minflux(self.ph_perloc_perpulse, self.bckg_ph_perloc_perpulse, self.sbr_perloc, self.ebp.psf_fits, STEP_NM)
        return np.concatenate((
                self.locs_bin_edges[:-1].reshape(-1,1),
                locs_coords,
                self.ph_perloc_allpulses.reshape(-1,1), 
                self.sbr_perloc.reshape(-1,1)
            ), axis=1)
       
    def save_locs(self):
        """
        This function save the localizations as a .npy file, to be accessed and used for post-processing
        """
        self.locs_results_filename = (self.tcspc_data.tcspc_data_filename + LOCS_FILE_SUFFIX + 'from' + str(int(self.tcspc_data.start_t_s)) +
                                      'to' + str(int(self.tcspc_data.end_t_s)) + 's_' + str(int(self.locs_t_binning_s * 1e3)) + 'ms_binning.npy')
        self.locs_results_filepath = self.tcspc_data.tcspc_data_dir / self.locs_results_filename
        np.save(self.locs_results_filepath, self.localizations)