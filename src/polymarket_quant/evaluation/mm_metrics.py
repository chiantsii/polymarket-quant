import pandas as pd
import numpy as np
from typing import List, Dict

class MMPerformanceEvaluator:
    """
    針對做市策略的專用評估器，專注於庫存風險與價差捕獲。
    """
    def __init__(self):
        self.trades: List[Dict] = []

    def record_trade(self, timestamp, side, price, size, inventory_before, mkt_price_now):
        self.trades.append({
            "timestamp": timestamp,
            "side": side,  # 'BUY' 或 'SELL'
            "price": price,
            "size": size,
            "inv_before": inventory_before,
            "mkt_price_now": mkt_price_now
        })

    def calculate_metrics(self, final_mkt_price: float) -> Dict:
        if not self.trades:
            return {"error": "No trades recorded"}

        df = pd.DataFrame(self.trades)
        
        # 1. Edge Captured (與市場中間價的偏離度)
        df['edge'] = np.where(
            df['side'] == 'BUY',
            df['mkt_price_now'] - df['price'],
            df['price'] - df['mkt_price_now']
        )

        # 2. PnL 分解
        realized_pnl = (df['edge'] * df['size']).sum()
        
        # 3. 庫存路徑分析
        df['inv_after'] = np.where(df['side'] == 'BUY', df['inv_before'] + 1, df['inv_before'] - 1)
        max_inv = df['inv_after'].abs().max()

        # 4. Adverse Selection (假設性的 Mark-out 邏輯)
        # 在正式系統中，這裡會與未來 5min 的價格做對比
        
        return {
            "total_trades": len(df),
            "realized_pnl_raw": realized_pnl,
            "max_inventory_exposure": max_inv,
            "avg_edge_per_trade": df['edge'].mean(),
            "win_rate_vs_mid": (df['edge'] > 0).mean()
        }