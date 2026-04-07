import pandas as pd
from typing import Dict

class FeaturePipeline:
    """Standardized pipeline for generating alpha features."""
    
    def __init__(self, config: Dict):
        self.config = config

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers/imputers and return engineered features."""
        # TODO: Implement bid-ask spread, orderbook imbalance, time-to-expiry features
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to out-of-sample data."""
        return data