from dataclasses import dataclass

@dataclass
class RiskState:
    current_inventory: float
    realized_pnl: float
    unrealized_pnl: float

class RiskManager:
    """Enforces event-time risk controls and bounds inventory."""
    
    def __init__(self, max_inventory: float, toxicity_threshold: float):
        self.max_inventory = max_inventory
        self.toxicity_threshold = toxicity_threshold

    def check_trade(self, risk_state: RiskState, intended_size: float, toxicity_score: float) -> float:
        """Returns the approved trade size after applying constraints."""
        if toxicity_score > self.toxicity_threshold:
            return 0.0 # Halt on toxic flow
        
        # Calculate available capacity
        available = self.max_inventory - abs(risk_state.current_inventory)
        return min(intended_size, available)