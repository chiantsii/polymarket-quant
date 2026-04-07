# Polymarket Quant Codebase Overview

本文檔整理目前 `polymarket_quant` 專案中每個主要檔案的用途、資料流、已完成能力與目前限制。內容依照現有架構說明，不重新設計專案。

## 專案定位

這是一個 Polymarket 量化研究與回測框架。核心方向是：

- 從 Polymarket Gamma API 與 CLOB API 收集市場資料、歷史價格、即時 orderbook。
- 針對 BTC/ETH Up or Down 5m 市場建立時間序資料。
- 用 pricing 模型估計 binary contract fair probability / fair price。
- 用 calibration 模型校準市場或模型機率。
- 之後透過 signals、risk、execution、backtest 串成策略研究流程。
- 另有 market-making simulation 元件，用於報價、毒性偵測、被動成交模擬與 MM 評估。

## 目前主要資料流

```text
Polymarket API
  -> ingestion/client.py
  -> ingestion/pipeline.py
  -> data/raw/*.json
  -> data/processed/*.parquet
  -> preprocessing/features.py
  -> pricing/*
  -> calibration/*
  -> signals/generator.py
  -> risk/limits.py
  -> execution/*
  -> backtest/engine.py
  -> evaluation/*
```

目前 ingestion 與 pricing 測試較完整。preprocessing、signals、backtest、jump model 仍以介面或 skeleton 為主。

## 專案根目錄檔案

### `pyproject.toml`

定義 Python package metadata、build backend、依賴套件與 dev dependencies。

主要依賴：

- `pandas`, `numpy`: 資料處理與數值計算。
- `scikit-learn`: calibration、metrics、linear/logistic models。
- `scipy`: stratified Monte Carlo 使用 `scipy.stats.norm`。
- `pydantic`: schema validation。
- `pyyaml`: 讀取 YAML config。
- `pyarrow`: parquet 讀寫。
- `requests`: API client。
- `pytest`: 測試。

### `configs/base.yaml`

專案基本設定。

目前包含：

- `project.name`: 專案名稱。
- `data.raw_dir`: raw JSON 輸出位置，預設 `data/raw`。
- `data.processed_dir`: parquet 輸出位置，預設 `data/processed`。
- `api.gamma_url`: Polymarket Gamma API URL。
- `api.clob_url`: Polymarket CLOB API URL。

### `.env.example`

環境變數範例。

目前包含：

- `POLYMARKET_API_BASE_URL`
- `POLYMARKET_SUBGRAPH_URL`
- `LOG_LEVEL`

目前程式主要透過 `configs/base.yaml` 使用 API URL，`LOG_LEVEL` 會被 `utils/logger.py` 使用。

### `.gitignore`

忽略：

- local venv
- Python cache
- `.env`
- `data/*`
- `artifacts/*`
- `logs/*`
- `reports/*`

並保留 `.gitkeep`，避免空資料夾消失。

### `README.md`

專案簡介與 setup 指令。描述專案目標是 Polymarket research/backtesting framework，核心概念包含 calibration、jump-aware modeling、execution-first。

## `src/polymarket_quant/__init__.py`

Package root。定義：

- `__version__ = "0.1.0"`

## Schemas

### `src/polymarket_quant/schemas/market.py`

定義 Pydantic data models。

主要 class：

- `Contract`: 表示一個 outcome asset，例如 Up/Down 或 Yes/No token。
- `MarketMetadata`: 標準市場 metadata，包含 `market_id`, `condition_id`, `title`, `category`, `resolution_date`, `contracts`, `is_active`。
- `MarketMetadata.time_to_resolution_seconds`: computed field，計算距離 resolution 還有幾秒。
- `Trade`: 單筆成交資料，包含 market、timestamp、price、size、side、maker/taker address。
- `Order`: orderbook 中一層 price/size。
- `OrderbookSnapshot`: L2 orderbook snapshot，包含 bids/asks。
- `OrderBook`: `OrderbookSnapshot` 的相容別名。
- `MarketStateSnapshot`: 市場狀態摘要，例如 last price、best bid/ask、volume、open interest、liquidity。
- `Resolution`: 最終 resolved market 狀態。

### `src/polymarket_quant/schemas/__init__.py`

匯出 schema 類別，讓其他模組可以從 `polymarket_quant.schemas` import。

## Ingestion

### `src/polymarket_quant/ingestion/client.py`

負責與 Polymarket public APIs 溝通。

`BasePolymarketClient` 是抽象邊界，定義或保留以下方法：

- `fetch_active_markets`
- `fetch_series`
- `fetch_event_by_slug`
- `fetch_orderbook`
- `fetch_price_history`

`PolymarketRESTClient` 是實作：

- `fetch_active_markets(limit)`: 呼叫 Gamma `/markets`，抓 active/open markets。
- `fetch_series(slug)`: 呼叫 Gamma `/series?slug=...`，用於 BTC/ETH Up or Down 5m series discovery。
- `fetch_event_by_slug(slug)`: 呼叫 Gamma `/events/slug/{slug}`，取得 event detail 與 markets/token ids。
- `fetch_orderbook(token_id)`: 呼叫 CLOB `/book`，取得目前 L2 orderbook。
- `fetch_price_history(token_id, interval, fidelity)`: 呼叫 CLOB `/prices-history`，取得歷史價格序列。

錯誤處理方式：捕捉 `requests.exceptions.RequestException`，記錄 error，回傳空 list 或 empty dict。

### `src/polymarket_quant/ingestion/pipeline.py`

負責 orchestrate fetching、normalization、儲存 raw/processed data。

主要公開方法：

- `run_market_metadata_ingestion()`
- `run_orderbook_snapshot_ingestion(market_limit=20)`
- `run_crypto_5m_history_ingestion(...)`
- `collect_crypto_5m_orderbooks_once(...)`
- `save_crypto_5m_orderbook_collection(...)`

`run_market_metadata_ingestion` 做：

- 抓 active markets。
- 儲存 raw JSON 到 `markets_raw_*.json` 與 `markets_raw_latest.json`。
- 轉成 `MarketMetadata`。
- 儲存 normalized parquet 到 `markets_normalized_*.parquet` 與 `markets_normalized_latest.parquet`。

`run_orderbook_snapshot_ingestion` 做：

- 抓 active markets。
- 從 markets 抽出 outcome token ids。
- 對每個 token 呼叫 CLOB `/book`。
- 儲存 raw orderbook JSON。
- 產生 summary parquet，例如 best bid/ask、spread、depth、levels。

`run_crypto_5m_history_ingestion` 做：

- 針對 `btc-up-or-down-5m`、`eth-up-or-down-5m` 這類 series 抓 event list。
- 預設只抓 closed events，避免 unresolved market 沒有 label。
- 用 event slug 取得 event detail。
- 抽出 Up/Down token ids。
- 對 token 抓 CLOB price history。
- 產生 `crypto_5m_price_history_*.parquet`，欄位包含 asset、event、market、token、timestamp、price、market start/end、outcome label。

`collect_crypto_5m_orderbooks_once` 做：

- 針對 open BTC/ETH 5m events 抓一次 live orderbook snapshot。
- 回傳三種資料：
  - raw snapshots
  - orderbook level rows
  - orderbook summary rows

`save_crypto_5m_orderbook_collection` 做：

- 把 live collector 累積的 orderbook 資料寫出：
  - `crypto_5m_orderbooks_raw_*.json`
  - `crypto_5m_orderbook_levels_*.parquet`
  - `crypto_5m_orderbook_summary_*.parquet`

重要 helper：

- `_extract_contracts`: 從 `tokens` 或 `clobTokenIds/outcomes` 解析 token ids。
- `_get_series_events`: 從 Gamma series 取得最近 event。
- `_price_history_rows`: 把 CLOB history 轉成 tabular rows。
- `_orderbook_level_rows`: 把 bids/asks 每一層攤平成 rows。
- `_summarize_orderbook`: 計算 best bid/ask、spread、mid price、depth、top 5 depth、imbalance。
- `_parse_clob_timestamp`: 把 CLOB timestamp 轉成 ISO timestamp。

## Pricing

`src/polymarket_quant/pricing/` 是定價與機率估計層。它不直接做 calibration，也不直接下單。

### `pricing/common.py`

共用工具。

- `PricingResult`: 統一 pricing estimator output，包含 `probability`, `standard_error`, `n_samples`, `diagnostics`。
- `PricingResult.fair_price`: binary contract 的 fair price，等於 probability。
- `make_rng`: 建立 NumPy random generator。
- `validate_binary_pricing_inputs`: 驗證 GBM binary pricing input。
- `simulate_gbm_terminal_values`: 幾何布朗運動 terminal value 模擬。
- `bernoulli_standard_error`: Bernoulli estimator standard error。

### `pricing/monte_carlo.py`

`estimate_monte_carlo_probability` 用 GBM Monte Carlo 估計：

```text
P(S_T >= threshold)
```

用於 binary event contract，例如 BTC/ETH 5m Up market。

輸入包含：

- `initial_value`
- `threshold`
- `drift`
- `volatility`
- `horizon`
- `n_samples`
- `seed`

回傳 `PricingResult`。

### `pricing/importance_sampling.py`

`estimate_importance_sampled_probability` 用 shifted-normal proposal 做 importance sampling，適合 rare event 機率估計。

核心概念：

- 從偏移後的 normal proposal 取樣。
- 用 likelihood ratio 修正回原始分布。
- 回傳 probability、standard error、effective sample size 等 diagnostics。

### `pricing/stratified.py`

`estimate_stratified_probability` 用 equal-probability strata 做 stratified Monte Carlo，以降低 variance。

流程：

- 把 `[0, 1]` uniform 空間切成 `n_strata`。
- 每個 stratum 取樣。
- 用 `norm.ppf` 轉 normal shocks。
- 估計每個 stratum 的 event probability，再平均。

### `pricing/particle_filter.py`

`ParticleFilter` 用 sequential observations 更新 latent probability。

用途：

- 吃入一段市場觀察值，例如 mid price 或 market-implied probability。
- 在 logit space 中維護 particles。
- 用 observation likelihood 更新 weights。
- ESS 太低時 resample。

`ParticleFilterResult` 包含：

- `probabilities`: 每一步 filtered probability。
- `effective_sample_sizes`: 每一步 ESS。
- `final_particles`
- `final_weights`

### `pricing/abm.py`

`BinaryMarketABM` 是 binary prediction market 的 agent-based price formation simulator。

用途：

- 研究交易者 beliefs、order flow 與 price impact 如何形成價格路徑。
- 目前不接 production trading engine。

`simulate` 回傳 `ABMResult`：

- `prices`
- `net_order_flow`
- `mean_beliefs`

### `pricing/__init__.py`

匯出 pricing 層的 public API：

- `PricingResult`
- `estimate_monte_carlo_probability`
- `estimate_importance_sampled_probability`
- `estimate_stratified_probability`
- `ParticleFilter`
- `BinaryMarketABM`

## Calibration

### `calibration/calibrator.py`

定義 `BaseCalibrator` 抽象介面。

方法：

- `fit(features, outcomes)`
- `calibrate(raw_prices, features=None)`

目前主要是 interface，供更具體的 calibrator 實作遵循。

### `calibration/models.py`

定義 calibration model。

`LogisticCalibrator`：

- Platt scaling 變體。
- 對 price 做 logit transform。
- 用 `LogisticRegression` fit outcomes。
- 如果尚未 fitted，`calibrate` 會直接回傳原始 prices。

`SegmentedCalibrator`：

- 依 category 和 time-to-resolution bins 分段。
- 每個 segment 訓練一個 `LogisticCalibrator`。
- 如果 unseen segment 沒有 model，就 fallback 回原始 price。

目前注意事項：

- `IsotonicRegression` 有 import 但尚未使用。
- 當某個 segment 樣本太少或 outcomes 只有單一類別時，`LogisticRegression` 可能無法 fit。

## Preprocessing

### `preprocessing/features.py`

`FeaturePipeline` 是特徵工程 pipeline skeleton。

目前：

- `fit_transform` 直接回傳 input DataFrame。
- `transform` 直接回傳 input DataFrame。

未來應實作：

- spread
- orderbook imbalance
- depth features
- time-to-expiry
- volatility
- momentum
- liquidity dynamics

## Signals

### `signals/generator.py`

`BaseSignal` 是 trading signal 抽象介面。

方法：

- `generate(orderbook, calibrated_prob, jump_state) -> Dict`

目前尚未有具體 signal implementation。

預期用途：

- 比較 calibrated probability / fair price 與 market price。
- 產生 target position、confidence、edge。

## Jump Models

### `jump_models/detector.py`

`JumpDetector` 是 jump/regime detection skeleton。

目前：

- 保存 `window_size` 與 `threshold_sigma`。
- `detect_regime` 永遠回傳 `(False, 0.0)`。

未來可以實作：

- bipower variation
- HMM regime switching
- z-score jump detection
- orderbook microstructure jump signals

## Risk

### `risk/limits.py`

定義：

- `RiskState`: current inventory、realized PnL、unrealized PnL。
- `RiskManager`: inventory 與 toxicity 風控。

`check_trade` 做：

- 如果 toxicity score 超過 threshold，回傳 `0.0`，代表禁止交易。
- 否則根據 `max_inventory - abs(current_inventory)` 限制 intended trade size。

目前注意事項：

- 未區分 buy/sell direction，因此 capacity 對降低 inventory 的交易也可能過度限制。
- 沒有處理 negative intended size。

## Execution

### `execution/simulator.py`

`ExecutionSimulator` 是 general execution simulator skeleton。

目前：

- 保存 latency。
- `simulate_fill` 直接回傳 requested size。

未來應實作：

- queue position
- partial fills
- slippage
- adverse selection
- latency effects

### `execution/market_maker.py`

`BinaryMarketMaker` 是 market-making quote logic。

主要方法：

- `calculate_reservation_price`: 根據 fair probability、inventory、sigma、time-to-resolution 產生 reservation price。
- `get_quotes`: 產生 bid/ask quote。

邏輯：

- inventory 為正時，reservation price 下移，鼓勵賣出、降低繼續買入誘因。
- toxicity 越高，spread 越寬。
- 接近 0/1 邊界時，spread 也會變寬。
- 到達 inventory bound 時嘗試 suppress 某一側報價。

目前注意事項：

- `kappa` 被保存但未使用。
- 到 inventory bound 時用 `0.0` 或 `1.0` pull quote，但最後又被 `np.clip` 成 `0.001` 或 `0.999`，因此不是完全取消報價。

### `execution/toxicity.py`

`ToxicityMonitor` 把市場危險訊號轉成 0 到 1 toxicity score。

輸入：

- `jump_z`
- `spread_widening`
- `vol_surge`

目前 `calculate_score` 使用線性加權：

```text
abs(jump_z) * 0.5 + spread_widening * 10 + vol_surge * 2
```

再除以 10 並 cap 到 1。

目前注意事項：

- `threshold` constructor parameter 沒有被 `is_risky` 使用。
- `is_risky` hardcode `score > 0.7`。

### `execution/sim.py`

`PassiveExecutionSim` 是被動掛單成交模擬器。

`simulate_fill` 輸入：

- bid
- ask
- market low
- market high
- market volume

邏輯：

- `mkt_low <= bid` 時代表 bid 被打到，有機率 BUY fill。
- `mkt_high >= ask` 時代表 ask 被打到，有機率 SELL fill。
- fill probability = `min(0.9, mkt_vol / 1000.0)`。

目前注意事項：

- 使用 global `random.random()`，所以模擬結果不可重現，除非外部設定 global seed。

## Backtest

### `backtest/engine.py`

`EventDrivenBacktester` 是 event-driven backtest skeleton。

建構時需要：

- `BaseSignal`
- `RiskManager`
- `ExecutionSimulator`

`run` 目前是 pass，docstring 描述未來流程：

1. 讀下一個 historical state。
2. 產生 signal。
3. 通過 risk manager。
4. execution simulation。
5. 更新 portfolio/PnL。

## Evaluation

### `evaluation/metrics.py`

一般機率模型評估工具。

`calculate_brier_score`：

- 對 probabilistic binary forecasts 計算 Brier score。

`calibration_diagnostics`：

- 計算 Brier score。
- 計算 log loss。
- 用 linear regression 在 logit probability 上估 calibration slope/intercept proxy。
- 產生 reliability table。

目前注意事項：

- `eps` 變數宣告但未使用。
- `log_loss` 對全 0 或全 1 outcome 的小樣本可能需要額外 labels 處理。

### `evaluation/mm_metrics.py`

Market making 專用 metrics。

`MMPerformanceEvaluator`：

- `record_trade`: 儲存 timestamp、side、price、size、inventory_before、market mid price。
- `calculate_metrics`: 回傳 total trades、raw realized PnL proxy、max inventory exposure、average edge、win rate vs mid。

目前注意事項：

- `final_mkt_price` 目前未使用。
- inventory after 用 `+1/-1`，未乘上 trade size。
- PnL 是 edge capture proxy，不是完整 mark-to-market PnL。

## Utils

### `utils/math.py`

數學工具。

- `logit(p, epsilon=1e-6)`: 對 p 做 clipping 後取 logit。
- `sigmoid(x)`: sigmoid transform。

### `utils/logger.py`

標準 logger helper。

`get_logger(name)`：

- 若 logger 沒有 handlers，新增 stream handler。
- formatter: timestamp、name、level、message。
- log level 從 `LOG_LEVEL` 環境變數讀取，預設 `INFO`。

## Scripts

### `scripts/run_ingestion.py`

CLI ingestion entry point。

參數：

- `--config`
- `--pipeline`: `markets`, `orderbooks`, `crypto-5m-history`
- `--market-limit`
- `--event-limit`
- `--fidelity`
- `--interval`
- `--include-open`

用途：

- `markets`: 抓 market metadata。
- `orderbooks`: 抓 active market current orderbook snapshot。
- `crypto-5m-history`: 抓 BTC/ETH Up or Down 5m 的歷史 price series。

### `scripts/collect_orderbooks.py`

Live BTC/ETH 5m orderbook collector。

用途：

- 每隔 N 秒抓一次 open BTC/ETH 5m Up/Down token orderbook。
- 預設 `duration` 模式會依照 `--duration-seconds` 收集。
- `--mode full-window` 會等待下一個尚未開始的完整 5 分鐘窗口，從該窗口開始收集並在窗口結束時停止。
- `full-window` 會用 UTC 5 分鐘邊界直接產生 `btc-updown-5m-{timestamp}` / `eth-updown-5m-{timestamp}`，不依賴 Gamma series metadata 等待。
- 累積 raw snapshots、level rows、summary rows。
- 最後寫到 `data/raw` 與 `data/processed`。
- 只負責 Polymarket orderbook，不計算 mispricing signal。

參數：

- `--config`
- `--mode`: `duration` 或 `full-window`
- `--interval-seconds`
- `--duration-seconds`
- `--event-duration-seconds`
- `--window-start`
- `--event-limit`
- `--event-slug-prefixes`
- `--series-slugs`

這是目前最接近真正 orderbook time-series 訓練資料的收集工具。

### `scripts/collect_spot_prices.py`

Live BTC/ETH spot price collector。

用途：

- 每隔 N 秒抓一次 Coinbase BTC/ETH ticker。
- 只負責 underlying spot price，不抓 Polymarket orderbook，不計算 signal。
- 輸出 `crypto_spot_ticks_*.parquet` 與 `crypto_spot_ticks_latest.parquet`。

參數：

- `--config`
- `--interval-seconds`
- `--duration-seconds`

### `scripts/backfill_resolutions.py`

BTC/ETH 5m resolution label backfill tool。

用途：

- 讀取已收集的 orderbook summary parquet，例如 `crypto_5m_orderbook_summary_*.parquet`。
- 從資料中取出 unique `event_slug`。
- 只對實際收過的 `btc-updown-5m*` / `eth-updown-5m*` events 回查 Gamma API。
- 預設跳過尚未超過 `event_start + 300s + 60s` 的 slugs，避免查到還沒 resolved 的市場。
- 解析 Up/Down token 的 `outcome_price` 與 `is_winner`。
- 預設只儲存已能推論 winner 的 rows；可用 `--include-unresolved` 保留 pending rows。
- 輸出 `crypto_5m_resolutions_*.parquet` 與 `crypto_5m_resolutions_latest.parquet`。

參數：

- `--config`
- `--input-glob`
- `--event-limit`
- `--event-duration-seconds`
- `--settlement-delay-seconds`
- `--include-unresolved`
- `--event-slug-prefixes`

### `scripts/train_calibration.py`

Calibration training skeleton。

目前流程：

- 讀取 `data/processed/markets_normalized_latest.parquet`。
- 以 2025-01-01 切 train/test。
- 建立 `SegmentedCalibrator`。
- 尚未真正 fit，因為缺少 joined price/outcome dataset。
- 儲存空的 calibrator artifact 到 `artifacts/calibration/segmented_calibrator.pkl`。

目前注意事項：

- 目前 markets metadata 並不足以訓練 calibration。
- 實際需要 price/outcome joined dataset。

### `scripts/run_backtest.py`

Backtest entry point skeleton。

目前：

- 讀取 `configs/base.yaml`。
- log project name。
- 尚未串接 engine、signals、execution。

### `scripts/run_mm_sim.py`

Market-making simulation entry point skeleton。

目前：

- 讀 `data/processed/backtest_input.parquet`。
- 對每列資料計算 toxicity。
- 用 `BinaryMarketMaker` 產生 bid/ask。
- 用 `PassiveExecutionSim` 模擬 fill。
- 更新 inventory 與 cash PnL。

目前注意事項：

- `backtest_input.parquet` 需要外部先產生。
- `t_score` 的 spread/vol 參數目前是 placeholder。
- PnL 未包含 final inventory mark-to-market。

## Tests

### `tests/test_schemas.py`

測試 `Order` 與 `OrderBook` schema 可以建立，且 bids 正確保存。

### `tests/test_ingestion.py`

使用 mock client 測試 ingestion pipeline，不打真實 Polymarket API。

覆蓋：

- market metadata ingestion 能寫 raw/latest JSON 與 parquet。
- orderbook snapshot ingestion 能抽出 Yes/No token，寫 summary parquet。
- crypto 5m history ingestion 能產生 Up/Down price history rows 與 winner label。
- live crypto 5m orderbook collector 能產生 raw snapshots、orderbook level rows、summary rows。
- crypto 5m resolution collector 能產生 Up/Down winner label rows。

### `tests/test_pricing_models.py`

測試 pricing 模型。

覆蓋：

- Monte Carlo probability 在 `[0, 1]`，且 seed deterministic。
- Importance sampling 回傳有效 probability、standard error、ESS。
- Stratified Monte Carlo 回傳 bounded probability 與 stratum diagnostics。
- Particle filter 回傳與 observations 同長度的 probability path，weights normalized。
- ABM 產生 bounded price path。
- Brier score 符合手算值。

### `tests/test_execution.py`

測試 market maker quote logic。

覆蓋：

- 正 inventory 時 bid/ask 會下移。
- toxicity 高時 spread 會變寬。

### `src/polymarket_quant/ingestion/spot.py`

新增的 underlying spot price adapter。

目前支援 Coinbase Exchange public REST ticker：

- `CoinbaseSpotPriceClient.fetch_spot_ticker()` 抓 BTC/ETH spot last price、bid、ask、volume、exchange time。
- `CoinbaseSpotPriceClient.fetch_reference_price()` 用 event slug 的 Unix start timestamp 抓 Coinbase 1-minute candle open，作為 BTC/ETH 5m Up/Down event 的近似 reference price。

### `src/polymarket_quant/ingestion/storage.py`

小型存檔 helper。

- `save_json_and_parquet_rows()` 將 normalized rows 同時寫成 raw JSON 與 parquet，並維護 `latest` 檔案。

### `src/polymarket_quant/signals/mispricing.py`

新增即時 mispricing detector。

它會把 Polymarket orderbook summary、Coinbase spot ticker、event reference price 串起來：

- 用 Monte Carlo / importance sampling / stratified MC 估 `fair_up_probability`。
- 可選用 particle filter 平滑即時 probability path。
- 用 spot jump、spread widening、volatility surge 產生 `toxicity_score`。
- 對 Up/Down token 分別計算 `buy_edge`、`sell_edge`、`fair_token_price`。
- 依照 edge、toxicity、depth 輸出 `BUY`、`SELL` 或 `HOLD`。

### `scripts/run_mispricing_detector.py`

新增 live mispricing detector 入口。

每次 poll 會：

- 抓 BTC/ETH Coinbase spot ticker。
- 抓目前 BTC/ETH 5m event 的 Up/Down orderbook。
- 用 event slug 末尾 timestamp 抓 reference candle open。
- 產出 orderbook parquet、spot ticker parquet、mispricing signal parquet。

主要輸出：

- `data/processed/crypto_spot_ticks_latest.parquet`
- `data/processed/crypto_5m_mispricing_signals_latest.parquet`

### `tests/test_mispricing.py`

測試即時 mispricing detector。

覆蓋：

- 當 Up token ask 明顯低於 fair price 時會產生 `BUY`。
- reference source 可以使用 `coinbase_1m_candle_open`。
- toxicity 超過設定上限時會強制 `HOLD`。

## 目前已通過的驗證

最近一次完整測試結果：

```text
venv/bin/pytest
22 passed
```

語法編譯：

```text
python -m compileall src scripts tests
passed
```

## 目前專案完成度摘要

較完整：

- Pydantic schemas。
- Polymarket REST client。
- market metadata ingestion。
- BTC/ETH 5m price history ingestion。
- BTC/ETH 5m live orderbook collector。
- BTC/ETH spot ticker/reference price ingestion。
- BTC/ETH 5m resolution label ingestion。
- 即時 BTC/ETH 5m mispricing detector。
- pricing models: Monte Carlo、importance sampling、stratified MC、particle filter、ABM。
- Brier score 與 calibration diagnostics。
- mock-based ingestion/pricing/schema/MM tests。

仍是 skeleton 或需要補強：

- preprocessing feature engineering。
- signal generator 實作。
- jump detector。
- general execution simulator。
- event-driven backtest engine。
- calibration training 真正的 price/outcome dataset。
- market-making PnL 與 inventory metrics。
- deterministic passive fill simulator。

## 建議下一步

1. 先修正 market-making simulation 的小風險：
   - inventory bound 時不要用 clip 後的 0.001/0.999 假裝取消報價。
   - `MMPerformanceEvaluator` 的 inventory update 使用 trade size。
   - `ToxicityMonitor.is_risky` 使用 `self.threshold`。
   - `PassiveExecutionSim` 支援 seed/RNG。

2. 建立真正的 feature table：
   - 從 `crypto_5m_orderbook_summary_*.parquet` 與 `crypto_5m_orderbook_levels_*.parquet` 產生 model-ready features。

3. 建立 pricing output table：
   - 對每個 timestamp/market/token 產生 `fair_probability`、`fair_price`、`model_name`。

4. 建立 calibration dataset：
   - join pricing output、market outcome、time-to-resolution、asset、orderbook features。

5. 再把 signal、risk、execution、backtest 串起來。
