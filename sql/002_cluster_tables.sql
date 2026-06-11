-- CRANE-X Cluster Tables
-- Separate namespace from the original CRANE's cluster_dictionary/cluster_vocab

-- Cluster definitions: semantic neighborhoods with multi-asset price reactions
CREATE TABLE IF NOT EXISTS cranex_cluster_dictionary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cluster_label VARCHAR(100),
    keywords TEXT,
    centroid_data JSON,
    avg_pc_esf DECIMAL(8,4),
    avg_pc_nqf DECIMAL(8,4),
    avg_pc_btc DECIMAL(8,4),
    avg_pc_clf DECIMAL(8,4),
    avg_pc_eth DECIMAL(8,4) DEFAULT 0,
    sample_count INT DEFAULT 0,
    last_updated DATETIME(3),
    sharpe_ratio DECIMAL(8,4),
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Vocabulary: global TF-IDF term list for vector consistency
CREATE TABLE IF NOT EXISTS cranex_cluster_vocab (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vocab_json LONGTEXT NOT NULL,
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Migrate existing old clusters as starting point
INSERT IGNORE INTO cranex_cluster_dictionary
  (id, cluster_label, keywords, centroid_data,
   avg_pc_esf, avg_pc_nqf, avg_pc_btc, avg_pc_clf, avg_pc_eth,
   sample_count, last_updated, sharpe_ratio, is_active)
SELECT id, cluster_label, keywords, centroid_data,
       avg_pc_esf, avg_pc_nqf, avg_pc_btc, avg_pc_clf, 0 AS avg_pc_eth,
       sample_count, last_updated, sharpe_ratio, is_active
FROM cluster_dictionary
WHERE is_active = TRUE;

INSERT IGNORE INTO cranex_cluster_vocab (id, vocab_json, created_at)
SELECT id, vocab_json, created_at FROM cluster_vocab
ORDER BY id DESC LIMIT 1;
