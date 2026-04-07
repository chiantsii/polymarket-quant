import random

class PassiveExecutionSim:
    """
    An intellectually honest simulator for passive fills.
    """
    def simulate_fill(self, 
                      bid: float, 
                      ask: float, 
                      mkt_low: float, 
                      mkt_high: float, 
                      mkt_vol: float) -> str:
        """
        Determines if a passive quote was 'touched' and filled.
        Returns: 'BUY', 'SELL', or None
        """
        # Simple Logic: If the market traded at or through our price, 
        # we assume a fill probability based on volume/intensity.
        
        fill_prob = min(0.9, mkt_vol / 1000.0) # More volume = higher fill likelihood
        
        if mkt_low <= bid and random.random() < fill_prob:
            return "BUY"
        if mkt_high >= ask and random.random() < fill_prob:
            return "SELL"
            
        return None