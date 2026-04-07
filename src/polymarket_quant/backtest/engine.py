import pandas as pd
from typing import List
from polymarket_quant.signals.generator import BaseSignal
from polymarket_quant.risk.limits import RiskManager
from polymarket_quant.execution.simulator import ExecutionSimulator

class EventDrivenBacktester:
    """Core loop for stepping through historical tick/orderbook data."""
    
    def __init__(self, signal: BaseSignal, risk: RiskManager, execution: ExecutionSimulator):
        self.signal = signal
        self.risk = risk
        self.execution = execution

    def run(self, historical_events: List[pd.DataFrame]):
        """
        Main event loop.
        1. Fetch next state
        2. Generate Signal
        3. Pass through Risk Manager
        4. Simulate Execution
        5. Update Portfolio/PnL
        """
        pass