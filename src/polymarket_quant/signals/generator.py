import abc
from typing import Dict
from polymarket_quant.schemas.market import OrderBook

class BaseSignal(abc.ABC):
    """Generates trading signals based on calibrated probabilities and jumps."""
    
    @abc.abstractmethod
    def generate(self, orderbook: OrderBook, calibrated_prob: float, jump_state: bool) -> Dict:
        """
        Returns a signal dictionary (e.g., target_position, confidence).
        """
        pass