-- ==========================================================================
-- CRANE-X Schema: EODHD News Ingestion Layer
-- Replaces old TradeFlags NewsFeed poller + FinBERT lexicon scoring
-- 
-- EODHD news API provides sentiment inline (polarity, neg, neu, pos)
-- so we store it directly — no separate FinBERT scoring step needed.
-- ==========================================================================

-- Topics to poll. CRANE-X loops over active topics, calling the EODHD news API
-- (endpoint configured in config.yaml).
CREATE TABLE IF NOT EXISTS eodhd_topics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    topic_name VARCHAR(100) NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT TRUE,
    poll_interval_min INT DEFAULT 15 COMMENT 'How often to poll this topic (minutes)',
    max_articles INT DEFAULT 25 COMMENT 'Max articles to fetch per poll call',
    last_polled_at DATETIME(3) DEFAULT NULL,
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_active (is_active),
    INDEX idx_poll (is_active, last_polled_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Raw news from EODHD with pre-baked sentiment.
-- The API returns VADER-style scores: polarity (-1..1), neg/neu/pos (sum=1.0)
-- This replaces the old news_events + FinBERT scoring pipeline.
CREATE TABLE IF NOT EXISTS eodhd_news (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ingested_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) COMMENT 'When CRANE-X saved it',
    topic VARCHAR(100) NOT NULL COMMENT 'Which topic tag fetched this article',
    
    -- Article fields (from EODHD API response)
    date_utc DATETIME(3) NOT NULL COMMENT 'UTC publish timestamp from API',
    title TEXT NOT NULL,
    content LONGTEXT COMMENT 'Full article body',
    httplink VARCHAR(1024) COMMENT 'Original article URL',
    
    -- Structured data from API
    symbols JSON COMMENT '["AAPL.US", "MSFT.US", ...] — related tickers',
    tags JSON COMMENT '["TECHNOLOGY", "EARNINGS", ...] — topic tags',
    
    -- Pre-baked sentiment scores (EODHD provides these)
    sentiment_polarity DECIMAL(6,4) COMMENT '-1.0000 to +1.0000',
    sentiment_neg DECIMAL(6,4) COMMENT '0-1, negative weight',
    sentiment_neu DECIMAL(6,4) COMMENT '0-1, neutral weight',
    sentiment_pos DECIMAL(6,4) COMMENT '0-1, positive weight',

    -- Price snapshots (from external price API at poll time)
    esf DECIMAL(12,4) COMMENT 'S&P 500 E-mini Futures level',
    pc_esf DECIMAL(8,4) COMMENT 'ES change percent',
    nqf DECIMAL(12,4) COMMENT 'Nasdaq 100 E-mini Futures level',
    pc_nqf DECIMAL(8,4) COMMENT 'NQ change percent',
    clf DECIMAL(12,4) COMMENT 'Brent Crude Futures level',
    pc_clf DECIMAL(8,4) COMMENT 'CL change percent',
    btc DECIMAL(14,4) COMMENT 'Bitcoin USD',
    pc_btc DECIMAL(8,4) COMMENT 'BTC change percent',
    eth DECIMAL(14,4) COMMENT 'Ethereum USD',
    pc_eth DECIMAL(8,4) COMMENT 'ETH change percent',

    -- Dedup hash
    headline_hash VARCHAR(64) NOT NULL COMMENT 'SHA256 of normalized title',
    UNIQUE KEY uk_headline_hash (headline_hash),
    
    INDEX idx_date (date_utc),
    INDEX idx_topic (topic),
    INDEX idx_topic_date (topic, date_utc),
    INDEX idx_symbols ((CAST(symbols AS CHAR(128) ARRAY))) 
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Seed initial topics (all poll every 15 min by default)
INSERT IGNORE INTO eodhd_topics (topic_name, is_active, poll_interval_min, max_articles) VALUES
    ('markets',     TRUE, 15, 25),
    ('stocks',      TRUE, 15, 25),
    ('economy',     TRUE, 15, 25),
    ('fed',         TRUE, 15, 25),
    ('earnings',    TRUE, 15, 25),
    ('technology',  TRUE, 15, 25),
    ('oil',         TRUE, 15, 25),
    ('energy',      TRUE, 15, 25),
    ('crypto',      TRUE, 15, 25),
    ('mergers',     TRUE, 15, 25),
    ('inflation',   TRUE, 15, 25),
    ('recession',   TRUE, 15, 25),
    ('bonds',       TRUE, 15, 25),
    ('commodities', TRUE, 15, 25),
    ('regulation',  TRUE, 15, 25)
ON DUPLICATE KEY UPDATE topic_name = topic_name;
