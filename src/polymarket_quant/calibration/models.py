import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Optional
from polymarket_quant.utils.math import logit

class LogisticCalibrator:
    """
    Platt Scaling variant.
    Fits: P(y=1) = sigmoid(A * logit(p) + B)
    """
    def __init__(self):
        self.model = LogisticRegression(penalty=None) # Pure Platt
        self.is_fitted = False

    def fit(self, prices: np.ndarray, outcomes: np.ndarray):
        X = logit(prices).reshape(-1, 1)
        self.model.fit(X, outcomes)
        self.is_fitted = True

    def calibrate(self, prices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return prices
        X = logit(prices).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

class SegmentedCalibrator:
    """
    Segments calibration by Category and Time-to-Resolution (TTR) bins.
    Uses Logistic regression within each segment.
    """
    def __init__(self, ttr_bins_days: list = [1, 7, 30]):
        self.ttr_bins = ttr_bins_days
        self.models: Dict[str, LogisticCalibrator] = {}

    def _get_segment_key(self, category: str, ttr_days: float) -> str:
        # Binning TTR: 'Politics_0-1d', 'Politics_1-7d', etc.
        bin_idx = np.digitize(ttr_days, self.ttr_bins)
        return f"{category}_bin{bin_idx}"

    def fit(self, prices: np.ndarray, outcomes: np.ndarray, categories: np.ndarray, ttrs: np.ndarray):
        segments = [self._get_segment_key(c, t) for c, t in zip(categories, ttrs)]
        unique_segments = set(segments)
        
        for seg in unique_segments:
            mask = [s == seg for s in segments]
            model = LogisticCalibrator()
            model.fit(prices[mask], outcomes[mask])
            self.models[seg] = model

    def calibrate(self, price: float, category: str, ttr_days: float) -> float:
        key = self._get_segment_key(category, ttr_days)
        model = self.models.get(key)
        if not model:
            # Fallback to a global model or raw price if segment is unseen
            return price
        return model.calibrate(np.array([price]))[0]