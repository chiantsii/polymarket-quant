# Polymarket Quant Research Environment

A robust, event-driven research and backtesting framework for Polymarket, emphasizing miscalibrated probability signals, jump-aware modeling, and strict execution simulation.

## Core Principles
1. **Calibration Over Direction:** Prices are treated as conditionally miscalibrated signals.
2. **Jump-Aware:** Built to model event-driven, discontinuous price evolution.
3. **Execution-First:** Adverse selection and inventory risk are modeled natively.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"