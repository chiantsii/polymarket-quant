import numpy as np
from typing import Tuple

class BinaryMarketMaker:
    """
    Computes bid/ask quotes based on fair value, inventory, and risk.
    """
    def __init__(self, 
                 gamma: float = 0.1,      # Risk aversion parameter
                 kappa: float = 1.5,      # Order book liquidity parameter
                 max_inv: int = 500,      # Max contracts held
                 base_spread: float = 0.02):
        self.gamma = gamma
        self.kappa = kappa
        self.max_inv = max_inv
        self.base_spread = base_spread

    def calculate_reservation_price(self, p_fair: float, inventory: int, sigma: float, ttr: float) -> float:
        """
        Adjusts the fair price based on inventory risk.
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        """
        # Linear approximation for binary space
        # TTR is normalized (0 to 1) where 1 is far from resolution
        risk_adjustment = inventory * self.gamma * (sigma ** 2) * ttr
        res_price = p_fair - risk_adjustment
        return np.clip(res_price, 0.01, 0.99)

    def get_quotes(self, 
                   p_fair: float, 
                   inventory: int, 
                   sigma: float, 
                   ttr: float, 
                   toxicity_score: float) -> Tuple[float, float]:
        """
        Returns (bid, ask) quotes.
        """
        # 1. Calculate Reservation Price (where we want the center of our spread)
        res_price = self.calculate_reservation_price(p_fair, inventory, sigma, ttr)
        
        # 2. Dynamic Spread based on toxicity and distance to boundary
        # Spread widens as toxicity increases or as price approaches 0/1 (liquidity thins)
        boundary_width = 1.0 - (4 * (p_fair - 0.5)**2) # Narrower at the edges
        dynamic_spread = self.base_spread * (1 + toxicity_score * 5) / (boundary_width + 0.1)
        
        bid = res_price - (dynamic_spread / 2)
        ask = res_price + (dynamic_spread / 2)
        
        # 3. Inventory Bounds (Quote suppression)
        if inventory >= self.max_inv:
            ask = res_price # Only try to sell
            bid = 0.0       # Pull bid
        elif inventory <= -self.max_inv:
            bid = res_price # Only try to buy
            ask = 1.0       # Pull ask
            
        return np.clip(bid, 0.001, 0.999), np.clip(ask, 0.001, 0.999)