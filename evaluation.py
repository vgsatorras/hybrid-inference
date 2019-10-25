import numpy as np

def mse(state, est_state, normalize=True):
    if normalize:
        value = np.mean(np.mean(np.square((state) - (est_state)), axis=1))
    else:
        value = np.sum(np.mean(np.square((state) - (est_state)), axis=1))
    return value
