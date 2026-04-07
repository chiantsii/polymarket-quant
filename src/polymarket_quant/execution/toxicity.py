import numpy as np

class ToxicityMonitor:
    """
    Detects 'toxic' regimes where market making is dangerous.
    """
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold

    def calculate_score(self, jump_z: float, spread_widening: float, vol_surge: float) -> float:
        """
        Normalizes various 'scary' signals into a 0.0 - 1.0 toxicity score.
        """
        # High Z-score = Jump risk
        # Spread widening = Informed traders or liquidity pull
        # Vol surge = Sudden news
        raw_score = (abs(jump_z) * 0.5) + (spread_widening * 10) + (vol_surge * 2)
        return min(1.0, raw_score / 10.0)

    def is_risky(self, score: float) -> bool:
        return score > 0.7