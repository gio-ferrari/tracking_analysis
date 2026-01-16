import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

def hmm_fit(self, locs):
    """
    This function implements a basic HMM fit of the time trace and plots the rescaled prediciton for the hidden states
    """
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
    model.fit(locs[:, 1].reshape(-1, 1))
    hidden_states = model.predict(locs[:, 1].reshape(-1, 1))
    locs_statezero = locs[hidden_states == 0]
    locs_stateone = locs[hidden_states == 1]
    self.avg_statezero = np.mean(locs_statezero[:, 1])
    self.avg_stateone = np.mean(locs_stateone[:, 1])
    downstate_idx = np.argmin((self.avg_statezero, self.avg_stateone))
    if downstate_idx == 0:    
        hidden_states_rescaled = hidden_states * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
    else:
        hidden_states_rescaled = (1 - hidden_states) * abs(self.avg_stateone - self.avg_statezero) + np.min((self.avg_stateone, self.avg_statezero))
    plt.figure(figsize=(12, 6))
    plt.plot(locs[:, 1], label="Localization Trace")
    plt.plot(hidden_states_rescaled, label="Predicted Hidden States", linestyle='--', color='red')
    plt.legend()
    plt.show()
    # prepare array with rotated localizations and HMM result (rescaled)
    self.rotated_clock_trace = np.empty((len(locs[:, 0]), 3), dtype=np.float64)
    self.rotated_clock_trace[:, 0] = locs[:, 0]
    self.rotated_clock_trace[:, 1] = locs[:, 1]
    self.rotated_clock_trace[:, 2] = hidden_states_rescaled
    self.trace_hmm_filename = self.locs_filepath.stem + '_timetracewHMM.npy'
    self.trace_hmm_filepath = self.tcspc_data_dir / self.trace_hmm_filename
    np.save(self.trace_hmm_filepath, self.rotated_clock_trace)
    return hidden_states, hidden_states_rescaled
    