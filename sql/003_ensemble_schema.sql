-- CRANE-X Ensemble Scoring and Calibration Schema

-- Landing table: per-article ensemble scores across all 5 assets
CREATE TABLE IF NOT EXISTS cranex_ensemble_scores (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id BIGINT NOT NULL COMMENT 'FK to eodhd_news.id',
    scored_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),

    -- Per-asset ensemble scores (-1 to +1 scale)
    es_score DECIMAL(6,4) COMMENT 'Ensemble sentiment for ES',
    nq_score DECIMAL(6,4) COMMENT 'Ensemble sentiment for NQ',
    cl_score DECIMAL(6,4) COMMENT 'Ensemble sentiment for CL',
    btc_score DECIMAL(6,4) COMMENT 'Ensemble sentiment for BTC',
    eth_score DECIMAL(6,4) COMMENT 'Ensemble sentiment for ETH',

    -- Composite: volatility-normalized weighted average of all 5
    composite_score DECIMAL(6,4) COMMENT 'Vol-normalized composite sentiment',

    -- Raw signals (for audit / debugging)
    eodhd_signal DECIMAL(6,4) COMMENT 'EODHD pre-baked polarity',
    stat_cluster_id INT COMMENT 'Cluster assigned by stat_scorer_x',
    stat_signal_es DECIMAL(8,4) COMMENT 'Cluster avg pc_esf',
    stat_signal_nq DECIMAL(8,4) COMMENT 'Cluster avg pc_nqf',
    stat_signal_cl DECIMAL(8,4) COMMENT 'Cluster avg pc_clf',
    stat_signal_btc DECIMAL(8,4) COMMENT 'Cluster avg pc_btc',
    stat_signal_eth DECIMAL(8,4) COMMENT 'Cluster avg pc_eth',
    llm_signal DECIMAL(6,4) DEFAULT NULL COMMENT 'LLM score (optional)',

    -- Weights used for this scoring run
    weights_used JSON COMMENT '{es:{w_eodhd,w_stat,w_llm}, nq:{...}, ...}',

    -- Realized 24h impacts (filled by calibration)
    realized_es DECIMAL(8,4) COMMENT 'Actual 24h ES move after article',
    realized_nq DECIMAL(8,4) COMMENT 'Actual 24h NQ move',
    realized_cl DECIMAL(8,4) COMMENT 'Actual 24h CL move',
    realized_btc DECIMAL(8,4) COMMENT 'Actual 24h BTC move',
    realized_eth DECIMAL(8,4) COMMENT 'Actual 24h ETH move',

    -- Prediction errors (filled by calibration)
    pred_error_es DECIMAL(8,4),
    pred_error_nq DECIMAL(8,4),
    pred_error_cl DECIMAL(8,4),
    pred_error_btc DECIMAL(8,4),
    pred_error_eth DECIMAL(8,4),

    UNIQUE KEY uk_article (article_id),
    INDEX idx_scored (scored_at),
    INDEX idx_realized (realized_es)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Calibration history: per-asset weight snapshots every 6h
-- We optimize weights per asset independently because each asset
-- responds differently to the same news signals.
CREATE TABLE IF NOT EXISTS cranex_calibration (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    calibrated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
    window_start DATETIME(3) NOT NULL,
    window_end DATETIME(3) NOT NULL,

    -- Asset identifier
    asset VARCHAR(10) NOT NULL COMMENT 'es, nq, cl, btc, eth',

    -- Optimal ensemble weights for this asset
    w_eodhd DECIMAL(6,4) COMMENT 'Weight for EODHD pre-baked sentiment',
    w_stat DECIMAL(6,4) COMMENT 'Weight for statistical cluster signal',
    w_llm DECIMAL(6,4) DEFAULT 0 COMMENT 'Weight for LLM signal',

    -- Performance metrics
    spearman_rho DECIMAL(6,4) COMMENT 'Correlation of ensemble vs realized',
    eodhd_rho DECIMAL(6,4) COMMENT 'EODHD signal predictive power',
    stat_rho DECIMAL(6,4) COMMENT 'Stat cluster signal predictive power',
    llm_rho DECIMAL(6,4) COMMENT 'LLM signal predictive power',

    -- Volatility (for normalization)
    realized_vol DECIMAL(8,4) COMMENT 'Asset realized volatility in window',
    n_articles INT COMMENT 'Articles in this calibration window',

    -- Regime detection
    regime VARCHAR(20) COMMENT 'bullish/bearish/neutral/volatile',

    INDEX idx_asset_time (asset, calibrated_at),
    INDEX idx_window (window_start, window_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Running volatility table for normalization weights
CREATE TABLE IF NOT EXISTS cranex_volatility (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset VARCHAR(10) NOT NULL COMMENT 'es, nq, cl, btc, eth',
    computed_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
    daily_vol DECIMAL(8,4) COMMENT 'Daily realized volatility',
    annualized_vol DECIMAL(8,4) COMMENT 'Annualized vol (daily * sqrt(252))',
    lookback_days INT DEFAULT 20,
    UNIQUE KEY uk_asset_time (asset, computed_at),
    INDEX idx_asset (asset, computed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Seed initial default weights (equal 50/50 EODHD/Stat until LLM kicks in)
INSERT IGNORE INTO cranex_calibration
    (calibrated_at, window_start, window_end, asset,
     w_eodhd, w_stat, w_llm, spearman_rho, realized_vol, n_articles, regime)
VALUES
    (NOW(), NOW() - INTERVAL 7 DAY, NOW(), 'es',  0.50, 0.50, 0.00, 0.00, 15.0, 0, 'neutral'),
    (NOW(), NOW() - INTERVAL 7 DAY, NOW(), 'nq',  0.50, 0.50, 0.00, 0.00, 18.0, 0, 'neutral'),
    (NOW(), NOW() - INTERVAL 7 DAY, NOW(), 'cl',  0.50, 0.50, 0.00, 0.00, 28.0, 0, 'neutral'),
    (NOW(), NOW() - INTERVAL 7 DAY, NOW(), 'btc', 0.50, 0.50, 0.00, 0.00, 55.0, 0, 'neutral'),
    (NOW(), NOW() - INTERVAL 7 DAY, NOW(), 'eth', 0.50, 0.50, 0.00, 0.00, 60.0, 0, 'neutral');
