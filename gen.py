import numpy as np

def generate_vector(cnt=1, dim=16, mx=1, save_path=None):
    vec = np.random.uniform(-mx, mx, (cnt, dim))
    if save_path:
        np.save(save_path, vec)
    return vec
