import yaml
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Initializing Backtest Engine...")
    
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config for project: {config['project']['name']}")
    # TODO: Wire up engine, signals, and execution simulator here
    logger.info("Backtest complete.")

if __name__ == "__main__":
    main()