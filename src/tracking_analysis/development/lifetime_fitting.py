import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tracking_analysis.tools.ptu_tools import load_tcspc_data

irf_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\IRFs\red\IRF_red_20251217_60kHz1.ptu")
data_file = Path(r"Y:\messdaten\Giovanni_B\MINFLUX\Data_PyFLUX\20251217\nc_sixsites_atto643_red_14_20251217_arrays.ptu")

class BasicFit():
    def __init__(self, irf_file: Path, data_file: Path):
        self.irf_file = irf_file
        self.data_file = data_file
        
        self.irf = load_tcspc_data(irf_file)
        self.data = load_tcspc_data(data_file)
        
basicfit = BasicFit(irf_file, data_file)

        