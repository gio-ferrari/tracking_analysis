import numpy as np
from pathlib import Path

TCSPC_TIME_OFFSET_NS = 0
LASER_PERIOD_NS = 50
TCSPC_NANOT_RES_PS = 16
OFFSET_FORNANOT_HIST_NS = 5e-4
NUM_PULSES = 4
STEP_NM = 1
LIFETIME_WIN_BEG_NS = 0
LIFETIME_WIN_END_NS = 10
IRF_WIN_END_NS = 2
DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\EBPs")
PSF_DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\EBPs")
DATA_DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX")
IRF_DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\IRFs")
LOCS_FILE_SUFFIX_NOLT = "_locs_nolt"
LOCS_FILE_SUFFIX_LT = "_locs_wlt_"
SIGMA_TOL_OUTLIERS = 3
COLOR_LIST = ['blue', 'orange', 'gray', 'yellow']
PULSES_POS_NS = np.array([1.0, 13.7, 26.2 , 38.8])  # [ns] 
CENTRAL_DONA_IDX = 3
MAX_DIST_FROMEBP_CENT_NM = 800
MIN_PH_MINFLUX_LOC = 100
MIN_PH_MINFLUX_LOC_LT = 100

EBP_DIR_SUFFIX = 'EBP'
PSF_FIT_DIR_NAME = 'full_dona_fit'
TCSPC_SUFFIX = 'arrays'
TCSPC_EXT = '.ptu'
DRIFT_SUFFIX = '1_xydata_particle'
DRIFT_EXT = '.txt'
HMM_WLT_SUFFIX = 'HMMfilt_withlifetime'

LIFETIME_CUTOFF_NS = 4
