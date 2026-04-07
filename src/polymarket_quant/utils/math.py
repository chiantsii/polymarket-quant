import numpy as np

def logit(p: float, epsilon: float = 1e-6) -> float:
    """Clamped logit transform to handle 0/1 prices."""
    p = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p / (1 - p))

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))