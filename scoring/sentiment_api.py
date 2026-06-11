"""
CRANE-X Sentiment API — JSON endpoint for the sentiment gauge.

Returns:
{
  "composite_score": 0.4363,
  "n_articles": 402,
  "scores": {"es": 0.61, "nq": 0.65, "cl": 0.11, "btc": 0.11, "eth": 0.11},
  "weights": {
    "es": {"eodhd": 0.5, "stat": 0.5},
    "nq": {"eodhd": 0.5, "stat": 0.5},
    ...
  },
  "vols": {"es": 15.0, "nq": 18.0, ...},
  "regimes": {"es": "bullish", ...},
  "articles_since": 42
}
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

from db import get_connection, load_config

ASSETS = ['es', 'nq', 'cl', 'btc', 'eth']


def get_current_scores(conn):
    """Get aggregate scores per asset from the last N scored articles."""
    cursor = conn.cursor(dictionary=True)

    # Latest ensemble scores
    cursor.execute("""
        SELECT 
            COUNT(*) as n_articles,
            ROUND(AVG(es_score), 4) as avg_es,
            ROUND(AVG(nq_score), 4) as avg_nq,
            ROUND(AVG(cl_score), 4) as avg_cl,
            ROUND(AVG(btc_score), 4) as avg_btc,
            ROUND(AVG(eth_score), 4) as avg_eth,
            ROUND(AVG(composite_score), 4) as avg_composite
        FROM cranex_ensemble_scores
        WHERE scored_at >= NOW() - INTERVAL 2 HOUR
    """)
    agg = cursor.fetchone()

    # If no recent articles, fall back to last 100
    if not agg or (agg['n_articles'] or 0) == 0:
        cursor.execute("""
            SELECT 
                COUNT(*) as n_articles,
                ROUND(AVG(es_score), 4) as avg_es,
                ROUND(AVG(nq_score), 4) as avg_nq,
                ROUND(AVG(cl_score), 4) as avg_cl,
                ROUND(AVG(btc_score), 4) as avg_btc,
                ROUND(AVG(eth_score), 4) as avg_eth,
                ROUND(AVG(composite_score), 4) as avg_composite
            FROM (
                SELECT es_score, nq_score, cl_score, btc_score, eth_score, composite_score
                FROM cranex_ensemble_scores
                ORDER BY scored_at DESC LIMIT 100
            ) latest
        """)
        agg = cursor.fetchone()

    cursor.close()
    return agg


def get_latest_weights(conn):
    """Fetch latest calibration weights per asset."""
    cursor = conn.cursor(dictionary=True)
    weights = {}
    for asset in ASSETS:
        cursor.execute("""
            SELECT w_eodhd, w_stat, w_llm, spearman_rho, regime
            FROM cranex_calibration
            WHERE asset = %s
            ORDER BY calibrated_at DESC LIMIT 1
        """, (asset,))
        row = cursor.fetchone()
        if row:
            weights[asset] = {
                'eodhd': float(row['w_eodhd'] or 0.5),
                'stat': float(row['w_stat'] or 0.5),
                'llm': float(row['w_llm'] or 0),
            }
    cursor.close()
    return weights


def get_volatilities(conn):
    """Fetch latest annualized volatility per asset."""
    cursor = conn.cursor(dictionary=True)
    vols = {}
    for asset in ASSETS:
        cursor.execute("""
            SELECT annualized_vol FROM cranex_volatility
            WHERE asset = %s
            ORDER BY computed_at DESC LIMIT 1
        """, (asset,))
        row = cursor.fetchone()
        vols[asset] = float(row['annualized_vol']) if row else 15.0
    cursor.close()
    return vols


def get_regimes(conn):
    """Fetch latest regime per asset."""
    cursor = conn.cursor(dictionary=True)
    regimes = {}
    for asset in ASSETS:
        cursor.execute("""
            SELECT regime FROM cranex_calibration
            WHERE asset = %s AND regime IS NOT NULL
            ORDER BY calibrated_at DESC LIMIT 1
        """, (asset,))
        row = cursor.fetchone()
        regimes[asset] = row['regime'] if row else 'neutral'
    cursor.close()
    return regimes


def get_recent_count(conn):
    """Count articles scored in last 30 min."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM cranex_ensemble_scores
        WHERE scored_at >= NOW() - INTERVAL 30 MINUTE
    """)
    cnt = cursor.fetchone()[0]
    cursor.close()
    return cnt


def build_response(conn):
    """Build the full API response."""
    agg = get_current_scores(conn)
    if not agg:
        return {'error': 'No data yet'}

    scores = {
        'es': float(agg['avg_es'] or 0),
        'nq': float(agg['avg_nq'] or 0),
        'cl': float(agg['avg_cl'] or 0),
        'btc': float(agg['avg_btc'] or 0),
        'eth': float(agg['avg_eth'] or 0),
    }

    response = {
        'composite_score': float(agg['avg_composite'] or 0),
        'n_articles': int(agg['n_articles'] or 0),
        'scores': scores,
        'weights': get_latest_weights(conn),
        'vols': get_volatilities(conn),
        'regimes': get_regimes(conn),
        'articles_since': get_recent_count(conn),
    }
    return response


def main():
    config = load_config()
    conn = get_connection(config)
    if not conn:
        print(json.dumps({'error': 'DB connection failed'}))
        return

    # If called with --pretty, print formatted JSON for CLI testing
    pretty = '--pretty' in sys.argv

    response = build_response(conn)
    conn.close()

    if pretty:
        print(json.dumps(response, indent=2))
    else:
        # For web serving: output as single-line JSON (used by a Flask/FastAPI wrapper)
        print(json.dumps(response))


if __name__ == "__main__":
    main()
