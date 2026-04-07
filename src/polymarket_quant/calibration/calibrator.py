import abc
import numpy as np

class BaseCalibrator(abc.ABC):
    """Calibrates raw market prices into true probability estimates."""
    
    @abc.abstractmethod
    def fit(self, features: np.ndarray, outcomes: np.ndarray):
        """Fit the calibration model (e.g., Platt Scaling, Beta Calibration)."""
        pass

    @abc.abstractmethod
    def calibrate(self, raw_prices: np.ndarray, features: np.ndarray = None) -> np.ndarray:
        """Return calibrated probabilities conditioned on market state."""
        pass