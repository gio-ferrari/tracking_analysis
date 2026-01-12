import sys
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tracking_analysis.config.configvar import (
    TCSPC_TIME_OFFSET_NS,
    LASER_PERIOD_NS,
    NUM_PULSES,
    LIFETIME_WIN_BEG_NS,
    LIFETIME_WIN_END_NS
)
from tracking_analysis.tools.ptu_tools import load_tcspc_data

class TCSPCData():
    def __init__(self, tcspc_data_path: Path, bckg_data_path: Path, bckg_dark_cnts_data_path: Path, timetrace_bin_width_s: float, τ: list):
        self.tcspc_data_path = tcspc_data_path
        self.bckg_data_path = bckg_data_path
        self.bckg_dark_cnts_data_path = bckg_dark_cnts_data_path
        self.timetrace_bin_width_s = timetrace_bin_width_s
        self.τ = τ
        # get folder and filename of the tcspc data file
        self.tcspc_data_dir = self.tcspc_data_path.parent
        self.tcspc_data_filename = self.tcspc_data_path.stem
        # load TCSPC data, plot timetrace and ask if there is a photobleaching step
        self.abs_time_s, self.rel_time_ns = load_tcspc_data(self.tcspc_data_path)
        self.tot_t_measuring_s = (self.abs_time_s.max() - self.abs_time_s.min())
        self.plot_timetrace()
        self.ask_analysis_type()
        # if there is no photobleaching step, gets background data from separate file
        if not self.is_single_mol:
            self.bckg_abs_time_s, self.bckg_rel_time_ns = load_tcspc_data(self.bckg_data_path)
            self.tot_t_measuring_bckg_s = (self.bckg_abs_time_s.max() - self.bckg_abs_time_s.min())
        if self.use_dark_cnts_choice:
            self.bckg_dark_cnts_abs_time_s, self.bckg_dark_cnts_rel_time_ns = load_tcspc_data(self.bckg_dark_cnts_data_path)
            self.tot_t_measuring_bckg_dark_cnts_s = (self.bckg_dark_cnts_abs_time_s.max() - self.bckg_dark_cnts_abs_time_s.min())
        # process and show data
        self.filter_time_data()
        self.shift_tcspc_data()
        #self.plot_tcspc_data()
        self.prep_ph_foranalysis()

    def plot_timetrace(self):
        """
        This function plots the intensity time trace of the measurement
        """
        plt.figure("Histogram abs_time")
        self.raw_timetrace_bin_edges = np.arange(0, self.tot_t_measuring_s + self.timetrace_bin_width_s, self.timetrace_bin_width_s)
        self.raw_timetrace_counts_hz, _, _ = plt.hist(self.abs_time_s, bins=self.raw_timetrace_bin_edges, alpha=0.7, color='blue', weights=np.ones_like(self.abs_time_s) / self.timetrace_bin_width_s)
        plt.xlabel("Time [s]")
        plt.ylabel("Counts [Hz]")
        plt.title("Time trace")
        plt.tight_layout()
        plt.show()
        
    def ask_analysis_type(self):
        """
        This function asks the user whether the measurement presents a photobleaching step or not. If yes, the background will be extracted
        from the measurement itself, after the photobleaching steps. Otherwise, a separate background file will be loaded.
        """
        is_there_photobleach = input("Does the measurement present a photobleaching step? (y/n) ")
        if is_there_photobleach == 'y':
            self.is_single_mol = True
        else:
            self.is_single_mol = False
        #use_dark_cnts_choice_input = input("Do you want to use a dark counts measurement? (y/n) ")
        use_dark_cnts_choice_input = 'n'
        if use_dark_cnts_choice_input == 'y':
            self.use_dark_cnts_choice = True
        else:
            self.use_dark_cnts_choice = False
        
    def filter_time_data(self):
        """
        This function asks the user the start and end time to select the relevant part of the measurement
        to be analysed, and filter all the data using these parameters and the relative time windows.
        If it is a single molecule, it also asks for an intensity threshold to identify the photobleaching step
        as the last moment the molecule goes below the intensity threshold and never recovers.
        Finally, it computes the emitter and background counts and average SBR.
        """
        self.start_t_input = input("Start time for analysis in s (input nothing for 0): ")
        if not self.start_t_input:
            self.start_t_s = 0.0
        else:
            self.start_t_s = float( self.start_t_input)
        self.end_t_input = input("End time for analysis in s (input nothing for end of the trace): ")
        if not  self.end_t_input:
            self.end_t_s = self.tot_t_measuring_s
        else:
            self.end_t_s = float( self.end_t_input)
        # if it is a single molecule, asks for intensity threshold to identify the photobleaching step and filter data based on that
        if self.is_single_mol:
            int_threshold = float(input("Intensity threshold for signal in Hz: "))
            # find all the bin (left) edges where the molecule intensity is below/above the threshold
            dark_bin_edges = self.raw_timetrace_bin_edges[:-1][self.raw_timetrace_counts_hz < int_threshold]
            bright_bin_edges = self.raw_timetrace_bin_edges[:-1][self.raw_timetrace_counts_hz >= int_threshold]
            # compute times until two nearest dark/bright bins. The n-th element is the time between the (n-1)-th and the n-th dark/bright bin
            t_tonext_dark_bin = np.concatenate(([dark_bin_edges[0]], np.diff(dark_bin_edges)))
            t_tonext_bright_bin = np.concatenate(([bright_bin_edges[0]], np.diff(bright_bin_edges)))
            # find the first bin of each dark/bright period of the molecule
            start_dark_time = dark_bin_edges[t_tonext_dark_bin > (self.timetrace_bin_width_s * 1.5)]
            start_bright_time = bright_bin_edges[t_tonext_bright_bin > (self.timetrace_bin_width_s * 1.5)]
            # check whether the trace starts already in a dark state
            if self.raw_timetrace_counts_hz[0] < int_threshold:
                start_dark_time = np.concatenate(([self.raw_timetrace_bin_edges[0]], start_dark_time))
            else:
                start_bright_time = np.concatenate(([self.raw_timetrace_bin_edges[0]], start_bright_time))
            # if the trace ends in a bright state, add an imaginary dark state at the end, so every bright state has an end (useful whe looping through bright states to filter data)
            if self.raw_timetrace_counts_hz[-1] >= int_threshold:
                start_dark_time = np.concatenate((start_dark_time, [self.raw_timetrace_bin_edges[-1]]))
            # filter photons from off states
            self.filt_rel_time_ns = np.array([])
            self.filt_abs_time_s = np.array([])
            for bright_state_idx in range(len(start_bright_time)):
                self.filt_rel_time_ns = np.concatenate(
                    self.filt_rel_time_ns,
                    self.rel_time_ns[np.logical_and(
                        self.abs_time_s > start_bright_time[bright_state_idx],
                        self.abs_time_s < start_dark_time[bright_state_idx])
                    ])
                self.filt_abs_time_s = np.concatenate(
                    self.filt_abs_time_s,
                    self.rel_time_ns[np.logical_and(
                        self.abs_time_s > start_bright_time[bright_state_idx],
                        self.abs_time_s < start_dark_time[bright_state_idx])
                    ])
            # the bleaching step is selected as the last time the molecule goes dark and never recovers
            self.bleach_t_s = start_dark_time[-1]
            print(f"Molecule photobleached after {self.bleach_t_s} s")
            self.emitter_stop_t_s = self.bleach_t_s - self.timetrace_bin_width_s
            self.bckg_start_t_s = self.bleach_t_s + self.timetrace_bin_width_s
            self.tot_t_measuring_bckg_s = self.tot_t_measuring_s - self.bckg_start_t_s
        else: 
            # if there is not photobleaching step, counts are considered until the end of the measurement
            self.emitter_stop_t_s = self.tot_t_measuring_s
        if self.use_dark_cnts_choice:
            self.start_t_s_forbaseline_bckg = float(input("Start time to estimate baseline SBR in s: "))
            if not self.start_t_s_forbaseline_bckg:
                self.start_t_s = 0.0
            
        self.tot_t_on = np.min((self.end_t_s, self.emitter_stop_t_s)) - self.start_t_s
            
        # here filter the data based on the start and end time selected by the user
        self.filt_rel_time_ns = self.rel_time_ns[np.logical_and(
            self.abs_time_s > self.start_t_s,
            self.abs_time_s < np.min((self.end_t_s, self.emitter_stop_t_s))
        )]
        self.filt_abs_time_s = self.abs_time_s[np.logical_and(
            self.abs_time_s > self.start_t_s,
            self.abs_time_s < np.min((self.end_t_s, self.emitter_stop_t_s))
        )]
        # only if it is a single molecule, gets the background counts from the measurement itself, after photobleaching
        if self.is_single_mol:
            print(f"Background starts at {self.bckg_start_t_s}")
            self.bckg_rel_time_ns = self.rel_time_ns[self.abs_time_s > self.bckg_start_t_s]
            self.bckg_abs_time_s = self.abs_time_s[self.abs_time_s > self.bckg_start_t_s]
        self.photons_tokeep = len(self.filt_abs_time_s)
        
        if self.use_dark_cnts_choice:
            self.sgnl_for_baseline_sbr_rel_time_ns = self.rel_time_ns[
                np.logical_and(
                    self.abs_time_s > self.start_t_s_forbaseline_bckg,
                    self.abs_time_s < self.emitter_stop_t_s
                )
            ]
            self.sgnl_for_baseline_sbr_abs_time_s = self.rel_time_ns[
                np.logical_and(
                    self.abs_time_s > self.start_t_s_forbaseline_bckg,
                    self.abs_time_s < self.emitter_stop_t_s
                )
            ]
        
        print(f"Total number of detected photons in the relevant part of the measurement: {self.photons_tokeep}")
        self.avg_bckg_counts = len(self.bckg_abs_time_s) / self.tot_t_measuring_bckg_s
        self.avg_tot_counts = self.photons_tokeep / self.tot_t_on
        self.avg_emitter_counts = self.avg_tot_counts - self.avg_bckg_counts
        self.avg_sbr_notimegating = self.avg_emitter_counts / self.avg_bckg_counts
        print("*****************************")
        print("Measure parameters before TCSPC timegating:")
        print(f"Average signal counts: {self.avg_emitter_counts} Hz")
        print(f"Average background counts: {self.avg_bckg_counts} Hz")
        print(f"Average SBR: {self.avg_sbr_notimegating}")
        print("*****************************")
        
    def shift_tcspc_data(self):
        """
        This function plots the decay curves of the TCSPC data, and the time windows used for analysis.
        """
        # shift TCSPC data (filtered, unfiltered and background) to put first pulse close to 0
        self.rel_time_shift_ns = (self.rel_time_ns - TCSPC_TIME_OFFSET_NS) % LASER_PERIOD_NS
        self.filt_rel_time_shift_ns = (self.filt_rel_time_ns - TCSPC_TIME_OFFSET_NS) % LASER_PERIOD_NS
        self.bckg_rel_time_shift_ns = (self.bckg_rel_time_ns - TCSPC_TIME_OFFSET_NS) % LASER_PERIOD_NS
        if self.use_dark_cnts_choice:
            self.bckg_dark_cnts_rel_time_shift_ns = (self.bckg_dark_cnts_rel_time_ns - TCSPC_TIME_OFFSET_NS) % LASER_PERIOD_NS
        
    def plot_tcspc_data(self):
        plt.figure('Emitter TCSPC Histogram')
        plt.hist(self.filt_rel_time_shift_ns, bins = 300, range=(0,50), label='arrival time (shifted)', alpha=0.7)
        for tau in self.τ:
            plt.axvline(tau, color='red', linestyle='--')
            plt.axvspan(tau + LIFETIME_WIN_BEG_NS, tau + LIFETIME_WIN_END_NS, color='red', alpha=0.2)
        plt.xlabel('Time [ns]')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure('Background TCSPC Histogram')
        plt.hist(self.bckg_rel_time_shift_ns, bins = 300, range=(0,50), label= 'arrival time (shifted)', alpha=0.7)
        for tau in self.τ:
            plt.axvline(tau, color='red', linestyle='--')
            plt.axvspan(tau + LIFETIME_WIN_BEG_NS, tau + LIFETIME_WIN_END_NS, color='red', alpha=0.2)
        plt.xlabel('Time [ns]')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def prep_ph_foranalysis(self):
        """
        This function prepares, for each pulse, an array of absolute times keeping only the photons in the TCSPC timegating window
        which will be used for analysis. Also, it computes the corrected background counts and sbr using timegating
        """
        self.abs_time_s_foranalysis_perpulse = []
        self.bckg_counts_timegated_perpulse = np.empty(NUM_PULSES, dtype=np.float64)
        if self.use_dark_cnts_choice:
            self.sgnl_cnts_forbaseline_sbr_timegated_perpulse = np.empty(NUM_PULSES, dtype=np.float64)
            self.bckg_dark_counts_timegated_perpulse = np.empty(NUM_PULSES, dtype=np.float64)
        self.tot_counts_timegated = 0

        for pulse_idx in range(NUM_PULSES):
            # computing start and end of the time window used for timegating for the current pulse
            start_win = self.τ[pulse_idx] + LIFETIME_WIN_BEG_NS
            end_win = self.τ[pulse_idx] + LIFETIME_WIN_END_NS
            # selecting all the photons used for analysis for the current pulse
            abs_time_s_foranalysis = self.filt_abs_time_s[
                np.logical_and(
                    self.filt_rel_time_shift_ns > start_win,
                    self.filt_rel_time_shift_ns < end_win,
                )
            ]
            self.tot_counts_timegated += len(abs_time_s_foranalysis) / (np.min((self.end_t_s, self.emitter_stop_t_s)) - self.start_t_s)
            # appending (important: a deepcopy!) of the obtained array to the list
            self.abs_time_s_foranalysis_perpulse.append(deepcopy(abs_time_s_foranalysis))
            # now doing the same with the background to get the correct background counts
            self.bckg_counts_timegated_perpulse[pulse_idx] = len(self.bckg_abs_time_s[
                np.logical_and(
                    self.bckg_rel_time_shift_ns > start_win,
                    self.bckg_rel_time_shift_ns < end_win,
                )
            ]) / self.tot_t_measuring_bckg_s
            if self.use_dark_cnts_choice:
                self.sgnl_cnts_forbaseline_sbr_timegated_perpulse[pulse_idx] = len(self.sgnl_for_baseline_sbr_abs_time_s[
                    np.logical_and(
                        self.sgnl_for_baseline_sbr_rel_time_ns > start_win,
                        self.sgnl_for_baseline_sbr_rel_time_ns < end_win,
                    )
                ]) / (self.emitter_stop_t_s - self.start_t_s_forbaseline_bckg)
                self.bckg_dark_counts_timegated_perpulse[pulse_idx] = len(self.bckg_dark_cnts_abs_time_s[
                    np.logical_and(
                        self.bckg_dark_cnts_rel_time_shift_ns > start_win,
                        self.bckg_dark_cnts_rel_time_shift_ns < end_win,
                    )
                ]) / self.tot_t_measuring_bckg_dark_cnts_s
        self.bckg_counts_timegated = np.sum(self.bckg_counts_timegated_perpulse)
        if self.use_dark_cnts_choice:
            self.sgnl_cnts_forbaseline_sbr_timegated = np.sum(self.sgnl_cnts_forbaseline_sbr_timegated_perpulse)
            self.bckg_dark_cnts_timegated = np.sum(self.bckg_dark_counts_timegated_perpulse)
            self.baseline_bckg_cnts_timegated_perpulse = self.bckg_counts_timegated_perpulse - self.bckg_dark_cnts_timegated / NUM_PULSES
            self.baseline_bckg_cnts_timegated = np.sum(self.baseline_bckg_cnts_timegated_perpulse)
            self.baseline_sbr = (self.sgnl_cnts_forbaseline_sbr_timegated - self.bckg_dark_cnts_timegated) / self.baseline_bckg_cnts_timegated - 1
        # total emitter photons used for analysis
        self.emitter_counts_timegated = self.tot_counts_timegated - self.bckg_counts_timegated
        # real SBR
        self.sbr_timegated = self.emitter_counts_timegated / self.bckg_counts_timegated
            
        print("*****************************")
        print("Measure parameters after TCSPC timegating:")
        print(f"Signal counts used for analysis: {self.emitter_counts_timegated}")
        print(f"Background counts for all pulses: {self.bckg_counts_timegated}")
        if self.use_dark_cnts_choice:
            print(f"Of which from dark counts: {self.bckg_dark_cnts_timegated}")
            print(f"Baseline SBR (without dark counts): {self.baseline_sbr}")
            print(f"Average signal counts for baseline SBR calculation: {self.sgnl_cnts_forbaseline_sbr_timegated}")
        print(f"Average SBR with timegating: {self.sbr_timegated}")
        print("*****************************")