import numpy as np
from pathlib import Path

TCSPC_TIME_OFFSET_NS = 0
LASER_PERIOD_NS = 50
NUM_PULSES = 4
STEP_NM = 1
LIFETIME_WIN_BEG_NS = 0
LIFETIME_WIN_END_NS = 5
DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\EBPs")
PSF_DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\EBPs")
DATA_DIR_BASE = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX")
LOCS_FILE_SUFFIX = "_locs_"
SIGMA_TOL_OUTLIERS = 3
COLOR_LIST = ['blue', 'orange', 'gray', 'yellow']
PULSES_POS_NS = np.array([1.5, 14.3, 26.6 , 39.2])  # [ns] 
CENTRAL_DONA_IDX = 3
MAX_DIST_FROMEBP_CENT_NM = 800

EBP_DIR_SUFFIX = 'EBP'
PSF_FIT_DIR_NAME = 'full_dona_fit'
TCSPC_SUFFIX = 'arrays'
TCSPC_EXT = '.ptu'
DRIFT_SUFFIX = '1_xydata_particle'
DRIFT_EXT = '.txt'
