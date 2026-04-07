import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LinearRegression


def calculate_brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the Brier score for probabilistic binary forecasts."""
    return float(brier_score_loss(outcomes, probs))

def calibration_diagnostics(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """
    Computes slope, intercept, and reliability table.
    Perfect calibration: Intercept=0, Slope=1.
    """
    # 1. Calibration Slope and Intercept (via Cox Calibration)
    # logit(y) ~ A + B * logit(probs)
    from polymarket_quant.utils.math import logit
    eps = 1e-6
    x = logit(probs).reshape(-1, 1)
    
    # We use a linear proxy for the slope/intercept diagnostics
    lr = LinearRegression().fit(x, outcomes)
    
    # 2. Reliability Table
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    
    table = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            mean_pred = np.mean(probs[mask])
            obs_fraction = np.mean(outcomes[mask])
            count = np.sum(mask)
            table.append({
                "bin": i,
                "mean_pred": mean_pred,
                "obs_fraction": obs_fraction,
                "count": count
            })
            
    return {
        "brier": brier_score_loss(outcomes, probs),
        "log_loss": log_loss(outcomes, probs),
        "slope": lr.coef_[0],
        "intercept": lr.intercept_,
        "reliability_table": table
    }
