from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


from tracking_analysis.postprocessing import DataPostProcessor

class EndoIVAnalysis():
    def __init__(self, post_proc_data: DataPostProcessor, locs_filepath: Path, tcspc_data_dir: Path):
        print("WIP")