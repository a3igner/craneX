"""
CRANE-X Ensemble Scorer.

Computes per-asset sentiment scores for each eodhd_news article by combining:
1. EODHD pre-baked sentiment (polarity)
2. Statistical cluster signal (from stat_scorer_x cluster assignment)
3. LLM signal (optional, placeholder)

Results stored in cranex_ensemble_scores with per-asset breakdown (ES, NQ, CL, BTC, ETH)
and a volatility-normalized composite score.

Calibration runs separately (calibrate.py) to optimize per-asset weights.
"""

import sys
import os
import json
import time
import math
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

from db import get_connection, load_config
from stat_scorer_x import StatClusterX, tokenize_article


ASSETS = ['es', 'nq', 'cl', 'btc', 'eth']
PRICE_COLS = ['esf', 'nqf', 'clf', 'btc', 'eth']
PCT_COLS = ['pc_esf', 'pc_nqf', 'pc_clf', 'pc_btc', 'pc_eth']
SIGNAL_FIELDS = ['eodhd', 'stat', 'llm']

# Default weights (used when no calibration exists)
DEFAULT_WEIGHTS = {'eodhd': 0.50, 'stat': 0.50, 'llm': 0.00}

# Diversity floor: no signal drops below this weight
DIVERSITY_FLOOR = 0.05


def get_latest_weights(conn):
    """Fetch latest calibration weights per asset. 
    
    Returns dict like {'es': {'eodhd': 0.5, 'stat': 0.5, 'llm': 0}, ...}
    Falls back to defaults if no calibration exists.
    """
    cursor = conn.cursor(dictionary=True)
    weights = {}
    for asset in ASSETS:
        cursor.execute("""
            SELECT w_eodhd, w_stat, w_llm
            FROM cranex_calibration
            WHERE asset = %s
            ORDER BY calibrated_at DESC
            LIMIT 1
        """, (asset,))
        row = cursor.fetchone()
        if row:
            weights[asset] = {
                'eodhd': float(row['w_eodhd'] or DEFAULT_WEIGHTS['eodhd']),
                'stat': float(row['w_stat'] or DEFAULT_WEIGHTS['stat']),
                'llm': float(row['w_llm'] or DEFAULT_WEIGHTS['llm']),
            }
        else:
            weights[asset] = dict(DEFAULT_WEIGHTS)
    cursor.close()
    return weights


def get_volatility_normalization(conn):
    """Fetch latest annualized vol per asset for composite normalization.
    
    Returns dict like {'es': 15.0, 'nq': 18.0, ...}
    Falls back to defaults.
    """
    defaults = {'es': 15.0, 'nq': 18.0, 'cl': 28.0, 'btc': 55.0, 'eth': 60.0}
    cursor = conn.cursor(dictionary=True)
    vols = {}
    for asset in ASSETS:
        cursor.execute("""
            SELECT annualized_vol FROM cranex_volatility
            WHERE asset = %s
            ORDER BY computed_at DESC LIMIT 1
        """, (asset,))
        row = cursor.fetchone()
        vols[asset] = float(row['annualized_vol']) if row else defaults[asset]
    cursor.close()
    return vols


def compute_composite(scores, vols):
    """Volatility-normalized composite sentiment.
    
    Each asset score is weighted by 1/vol, so high-vol assets (BTC, ETH)
    contribute less noise to the composite than low-vol assets (ES, NQ).
    
    composite = Σ(score_i / vol_i) / Σ(1/vol_i)
    """
    numerator = 0.0
    denominator = 0.0
    for asset in ASSETS:
        s = scores.get(asset)
        if s is None:
            continue
        vol = vols.get(asset, 15.0)
        if vol > 0:
            w = 1.0 / vol
            numerator += s * w
            denominator += w

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def compute_ensemble(eodhd_polarity, stat_signals, llm_signal, weights):
    """Compute per-asset ensemble scores from signals + weights.
    
    eodhd_polarity: float -1 to 1 (same for all assets)
    stat_signals: dict {es: val, nq: val, ...} from cluster assignment
    llm_signal: float -1 to 1 or None
    weights: dict {asset: {eodhd: w, stat: w, llm: w}}
    
    Returns dict {es: score, nq: score, ..., composite: score}
    """
    scores = {}
    for asset in ASSETS:
        w = weights.get(asset, DEFAULT_WEIGHTS)
        stat_val = stat_signals.get(asset, 0) if stat_signals else 0

        # Normalize stat signal from percentage to -1..1 scale
        stat_norm = max(-1.0, min(1.0, stat_val * 2.0)) if stat_val else 0.0

        # EODHD polarity is already -1 to 1
        eodhd_val = max(-1.0, min(1.0, eodhd_polarity))

        # LLM signal (may be None if not yet scored)
        llm_val = max(-1.0, min(1.0, llm_signal)) if llm_signal is not None else 0.0

        # Weighted sum
        score = (
            w.get('eodhd', 0) * eodhd_val +
            w.get('stat', 0) * stat_norm +
            w.get('llm', 0) * llm_val
        )
        scores[asset] = round(max(-1.0, min(1.0, score)), 4)

    return scores


def get_stat_signals(conn, clusterer, article_id, title, content, tags_str, local_idf, similarity_threshold=0.12):
    """Get statistical cluster signals for a single article.
    
    Returns (cluster_id, signals_dict) where signals_dict is {es: val, nq: val, ...}
    """
    cluster_id, sim, avg_esf, sharpe = clusterer.score_article(
        title, content, tags_str, local_idf, similarity_threshold
    )

    if cluster_id is None:
        return None, {}

    # Look up cluster for multi-asset reactions
    cluster = next((c for c in clusterer.clusters if c['id'] == cluster_id), None)
    if not cluster:
        return cluster_id, {}

    signals = {
        'es': float(cluster.get('avg_pc_esf') or 0),
        'nq': float(cluster.get('avg_pc_nqf') or 0),
        'cl': float(cluster.get('avg_pc_clf') or 0),
        'btc': float(cluster.get('avg_pc_btc') or 0),
        'eth': float(cluster.get('avg_pc_eth') or 0),
    }
    return cluster_id, signals


def score_unscored(conn, limit=200):
    """Score all unscored eodhd_news articles.
    
    Returns (scored_count, error_count).
    """
    cursor = conn.cursor(dictionary=True)

    # Find articles needing ensemble scoring
    # 1. Never scored (no row, or es_score is NULL)
    # 2. Has LLM signal but es_score was computed with old weights (0% LLM)
    cursor.execute("""
        (SELECT n.id, n.title, n.content, n.tags, n.sentiment_polarity,
               e.llm_signal
        FROM eodhd_news n
        JOIN cranex_ensemble_scores e ON e.article_id = n.id
        WHERE (e.es_score IS NULL
           OR (e.llm_signal IS NOT NULL AND e.llm_signal != 0
               AND (e.weights_used IS NULL OR e.weights_used NOT LIKE '%llm%')))
          AND n.title IS NOT NULL AND n.title != ''
          AND n.ingested_at >= NOW() - INTERVAL 7 DAY)
        UNION
        (SELECT n.id, n.title, n.content, n.tags, n.sentiment_polarity,
               NULL AS llm_signal
        FROM eodhd_news n
        LEFT JOIN cranex_ensemble_scores e ON e.article_id = n.id
        WHERE e.article_id IS NULL
          AND n.title IS NOT NULL AND n.title != ''
          AND n.ingested_at >= NOW() - INTERVAL 7 DAY)
        ORDER BY id DESC
        LIMIT %s
    """, (limit,))
    unscored = cursor.fetchall()
    cursor.close()

    if not unscored:
        print("[Scorer] No unscored articles found")
        return 0, 0

    print(f"[Scorer] Scoring {len(unscored)} articles...")

    # Load clusterer
    clusterer = StatClusterX(conn)
    if not clusterer.clusters:
        print("[Scorer] WARNING: No clusters loaded — falling back to EODHD-only scoring")
        have_clusters = False
    else:
        have_clusters = True

    # Build local IDF for batch scoring
    if have_clusters:
        all_vec_tokens = []
        for r in unscored:
            vt, _ = tokenize_article(r['title'], r.get('content') or '', r.get('tags') or '[]')
            all_vec_tokens.append(vt)

        from collections import Counter
        from stat_scorer_x import compute_idf
        local_idf = compute_idf(all_vec_tokens)
        for term in clusterer.vocab:
            if term not in local_idf:
                local_idf[term] = 1.0
    else:
        local_idf = {}

    # Get latest weights and volatility
    weights = get_latest_weights(conn)
    vols = get_volatility_normalization(conn)

    insert_sql = """
        INSERT INTO cranex_ensemble_scores
        (article_id, scored_at,
         es_score, nq_score, cl_score, btc_score, eth_score, composite_score,
         eodhd_signal, stat_cluster_id,
         stat_signal_es, stat_signal_nq, stat_signal_cl, stat_signal_btc, stat_signal_eth,
         weights_used)
        VALUES (%s, NOW(),
                %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s,
                %s)
        ON DUPLICATE KEY UPDATE
            es_score = VALUES(es_score),
            nq_score = VALUES(nq_score),
            cl_score = VALUES(cl_score),
            btc_score = VALUES(btc_score),
            eth_score = VALUES(eth_score),
            composite_score = VALUES(composite_score),
            eodhd_signal = VALUES(eodhd_signal),
            stat_cluster_id = VALUES(stat_cluster_id),
            stat_signal_es = VALUES(stat_signal_es),
            stat_signal_nq = VALUES(stat_signal_nq),
            stat_signal_cl = VALUES(stat_signal_cl),
            stat_signal_btc = VALUES(stat_signal_btc),
            stat_signal_eth = VALUES(stat_signal_eth),
            weights_used = VALUES(weights_used),
            scored_at = NOW()
    """

    insert_cursor = conn.cursor()
    scored = 0
    errors = 0

    for r in unscored:
        try:
            article_id = r['id']
            title = r['title'] or ''
            content = r.get('content') or ''
            tags_str = r.get('tags') or '[]'
            eodhd_polarity = float(r.get('sentiment_polarity') or 0)

            # Signal 1: EODHD pre-baked
            eodhd_signal = max(-1.0, min(1.0, eodhd_polarity))

            # Signal 2: Statistical clusters
            if have_clusters:
                cluster_id, stat_signals = get_stat_signals(
                    conn, clusterer, article_id, title, content, tags_str,
                    local_idf
                )
            else:
                cluster_id = None
                stat_signals = {}

            # Signal 3: LLM (from llm_scorer, may be NULL)
            llm_signal = r.get('llm_signal')
            if llm_signal is not None:
                llm_signal = float(llm_signal)

            # Ensemble: combine signals per asset
            scores = compute_ensemble(eodhd_signal, stat_signals, llm_signal, weights)

            # Composite: volatility-normalized
            composite = compute_composite(scores, vols)

            # Weights JSON for audit trail
            weights_json = json.dumps(weights)

            insert_cursor.execute(insert_sql, (
                article_id,
                scores['es'], scores['nq'], scores['cl'],
                scores['btc'], scores['eth'], composite,
                eodhd_signal,
                cluster_id,
                stat_signals.get('es'),
                stat_signals.get('nq'),
                stat_signals.get('cl'),
                stat_signals.get('btc'),
                stat_signals.get('eth'),
                weights_json,
            ))
            scored += 1

        except Exception as e:
            print(f"[Scorer] Error scoring article {r.get('id')}: {e}")
            errors += 1

    conn.commit()
    insert_cursor.close()
    print(f"[Scorer] Scored: {scored}, Errors: {errors}")
    return scored, errors


def main():
    print(f"[Scorer] CRANE-X Ensemble Scorer starting at {datetime.now(timezone.utc).isoformat()}")

    config = load_config()
    conn = get_connection(config)
    if not conn:
        print("[Scorer] FATAL: No DB connection")
        sys.exit(1)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    score_unscored(conn, limit=limit)
    conn.close()
    print(f"[Scorer] Done at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
