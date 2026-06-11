"""
CRANE-X Calibration Engine.

Runs every 6 hours to:
1. Compute realized 24h price impacts for each article per asset
2. Compute Spearman rho between each signal and realized moves
3. Optimize per-asset ensemble weights (softmax + diversity floor)
4. Detect market regime per asset
5. Store volatility metrics

Each asset has its OWN calibration because the same news affects 
ES, NQ, CL, BTC, and ETH differently.
"""

import sys
import os
import json
import math
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

import numpy as np
from scipy.stats import spearmanr
from db import get_connection, load_config


ASSETS = ['es', 'nq', 'cl', 'btc', 'eth']
PCT_COLS = ['pc_esf', 'pc_nqf', 'pc_clf', 'pc_btc', 'pc_eth']
PRICE_COLS = ['esf', 'nqf', 'clf', 'btc', 'eth']
SIGNAL_NAMES = ['eodhd', 'stat', 'llm']

DIVERSITY_FLOOR = 0.05  # No signal weight drops below this
DEFAULT_LOOKBACK_HOURS = 168  # 7 days for calibration window
VOL_LOOKBACK_DAYS = 20  # Days for volatility computation


def compute_realized_impacts(conn, window_start, window_end):
    """Compute 24h forward price impacts for articles in a time window.
    
    For each article with a price snapshot at poll time, finds the nearest
    future price snapshot 24h later and computes the % change.
    
    Returns list of dicts: {article_id, published_at, asset, realized_pc}
    """
    cursor = conn.cursor(dictionary=True)

    # Get articles with price snapshots in the window
    cursor.execute("""
        SELECT id, date_utc, esf, pc_esf, nqf, pc_nqf, clf, pc_clf,
               btc, pc_btc, eth, pc_eth
        FROM eodhd_news
        WHERE date_utc >= %s AND date_utc <= %s
          AND esf IS NOT NULL
        ORDER BY date_utc ASC
    """, (window_start, window_end))
    articles = cursor.fetchall()
    cursor.close()

    if len(articles) < 5:
        print(f"[Calibrate] Only {len(articles)} articles with prices — need 5+")
        return []

    print(f"[Calibrate] Computing 24h impacts for {len(articles)} articles...")

    # Build a price time series for each asset
    price_series = {a: [] for a in ASSETS}
    price_dates = {a: [] for a in ASSETS}

    for art in articles:
        for asset, col in zip(ASSETS, PRICE_COLS):
            val = art.get(col)
            if val is not None:
                price_series[asset].append(float(val))
                price_dates[asset].append(art['date_utc'])

    # For each article, find: snapshot price BEFORE, future price AFTER +24h
    results = []
    for art in articles:
        pub_ts = art['date_utc']
        target_ts = pub_ts + timedelta(hours=24)

        for asset, col in zip(ASSETS, PCT_COLS):
            dates = price_dates[asset]
            prices = price_series[asset]

            if len(dates) < 2:
                continue

            # Snapshot: price at or before article time
            snap_idx = None
            for i in range(len(dates) - 1, -1, -1):
                if dates[i] <= pub_ts:
                    snap_idx = i
                    break
            if snap_idx is None:
                continue

            snapshot = prices[snap_idx]
            if snapshot == 0:
                continue

            # Future: price at or after article time + 24h
            fut_idx = None
            for i in range(len(dates)):
                if dates[i] >= target_ts:
                    fut_idx = i
                    break
            if fut_idx is None or fut_idx <= snap_idx:
                continue

            future = prices[fut_idx]
            realized_pc = round((future - snapshot) / snapshot * 100, 4)

            results.append({
                'article_id': art['id'],
                'published_at': pub_ts,
                'asset': asset,
                'snapshot': snapshot,
                'future': future,
                'realized_pc': realized_pc,
            })

    return results


def compute_spearman_rho(signal_values, realized_values):
    """Compute Spearman rank correlation between signal and realized moves."""
    if len(signal_values) < 5 or len(realized_values) < 5:
        return 0.0
    if len(signal_values) != len(realized_values):
        return 0.0
    try:
        rho, _ = spearmanr(signal_values, realized_values)
        return round(float(rho), 4) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def optimize_weights(rhos, diversity_floor=DIVERSITY_FLOOR):
    """Optimize ensemble weights from Spearman correlations.
    
    Uses softmax on correlations with a diversity floor.
    Each signal gets weight proportional to max(0, rho^2) but never below floor.
    """
    signal_names = ['eodhd', 'stat', 'llm']
    raw_weights = {}

    for name in signal_names:
        rho = rhos.get(name, 0)
        # Square the rho (ignoring direction) — we care about predictiveness
        w = max(0, rho * rho) if rho > 0 else 0.01
        raw_weights[name] = w

    # If all are near-zero, use equal weights
    total_raw = sum(raw_weights.values())
    if total_raw < 0.01:
        return {n: 1.0 / len(signal_names) for n in signal_names}

    # Softmax with temperature
    # Apply diversity floor by redistributing
    n_signals = len(signal_names)
    floor_total = diversity_floor * n_signals

    if floor_total >= 1.0:
        return {n: 1.0 / n_signals for n in signal_names}

    # Proportional allocation above floor
    remaining = 1.0 - floor_total
    raw_total = sum(raw_weights.values())

    weights = {}
    for name in signal_names:
        if raw_total > 0:
            prop = raw_weights[name] / raw_total
        else:
            prop = 1.0 / n_signals
        weights[name] = round(diversity_floor + prop * remaining, 4)

    # Normalize to ensure sum = 1
    total = sum(weights.values())
    weights = {n: round(w / total, 4) for n, w in weights.items()}

    return weights


def detect_regime(realized_values):
    """Detect market regime from recent realized price moves.
    
    Returns: 'bullish', 'bearish', 'neutral', or 'volatile'
    """
    if len(realized_values) < 5:
        return 'neutral'

    vals = np.array([v for v in realized_values if v is not None])
    if len(vals) < 5:
        return 'neutral'

    mean = np.mean(vals)
    std = np.std(vals)

    if std > np.mean(np.abs(vals)) * 2:
        return 'volatile'
    elif mean > 0.1:
        return 'bullish'
    elif mean < -0.1:
        return 'bearish'
    else:
        return 'neutral'


def compute_volatility(conn, asset):
    """Compute rolling realized volatility for an asset from eodhd_news.
    
    Uses pc_ values to estimate daily volatility.
    """
    col = f"pc_{asset}f" if asset != 'btc' and asset != 'eth' else f"pc_{asset}"
    if asset == 'btc':
        col = 'pc_btc'
    elif asset == 'eth':
        col = 'pc_eth'

    # Actually use the standard column names
    pct_col = {
        'es': 'pc_esf', 'nq': 'pc_nqf', 'cl': 'pc_clf',
        'btc': 'pc_btc', 'eth': 'pc_eth'
    }[asset]

    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT {pct_col} as pc FROM eodhd_news
        WHERE {pct_col} IS NOT NULL
          AND date_utc >= NOW() - INTERVAL {VOL_LOOKBACK_DAYS} DAY
        ORDER BY date_utc DESC
        LIMIT 100
    """)
    rows = cursor.fetchall()
    cursor.close()

    if len(rows) < 5:
        return None, None

    vals = [abs(float(r['pc'])) for r in rows if r['pc'] is not None]
    if not vals:
        return None, None

    daily_vol = float(np.std(vals))
    annualized_vol = round(daily_vol * math.sqrt(252), 4)
    return round(daily_vol, 4), annualized_vol


def store_volatility(conn, asset, daily_vol, annualized_vol):
    """Store volatility measurement."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cranex_volatility
        (asset, computed_at, daily_vol, annualized_vol, lookback_days)
        VALUES (%s, NOW(), %s, %s, %s)
    """, (asset, daily_vol, annualized_vol, VOL_LOOKBACK_DAYS))
    conn.commit()
    cursor.close()


def calibrate_asset(conn, asset, realized_list, window_start, window_end):
    """Run calibration for a single asset.
    
    1. Get scored articles with realized impacts
    2. Compute correlations per signal
    3. Optimize weights
    4. Store calibration row
    """
    cursor = conn.cursor(dictionary=True)

    # Build asset-specific pct column
    pct_col = {
        'es': 'pc_esf', 'nq': 'pc_nqf', 'cl': 'pc_clf',
        'btc': 'pc_btc', 'eth': 'pc_eth'
    }[asset]

    signal_col = f'realized_{asset}'

    # Update realized impacts in ensemble_scores
    for r_data in realized_list:
        if r_data['asset'] == asset:
            cursor.execute(f"""
                UPDATE cranex_ensemble_scores
                SET realized_{asset} = %s
                WHERE article_id = %s
            """, (r_data['realized_pc'], r_data['article_id']))
    conn.commit()

    # Now fetch scored articles WITH both signals and realized moves
    stat_col = f'stat_signal_{asset}'
    cursor.execute(f"""
        SELECT eodhd_signal, {stat_col} as stat_signal,
               llm_signal,
               realized_{asset} as realized
        FROM cranex_ensemble_scores
        WHERE realized_{asset} IS NOT NULL
          AND scored_at >= %s AND scored_at <= %s
        ORDER BY scored_at ASC
    """, (window_start, window_end))
    scored = cursor.fetchall()
    cursor.close()

    if len(scored) < 5:
        print(f"[Calibrate] {asset}: only {len(scored)} scored with impacts — need 5+")
        return

    # Extract signal arrays
    eodhd_vals = [float(r['eodhd_signal']) for r in scored if r['eodhd_signal'] is not None]
    stat_vals = [float(r['stat_signal']) for r in scored if r['stat_signal'] is not None]
    llm_vals = [float(r['llm_signal']) for r in scored if r['llm_signal'] is not None]
    realized_vals = [float(r['realized']) for r in scored if r['realized'] is not None]

    # Align lengths
    min_len = min(len(eodhd_vals), len(stat_vals), len(llm_vals), len(realized_vals))
    if min_len < 5:
        print(f"[Calibrate] {asset}: only {min_len} aligned pairs (need 5+)")
        return

    eodhd_vals = eodhd_vals[:min_len]
    stat_vals = stat_vals[:min_len]
    llm_vals = llm_vals[:min_len]
    realized_vals = realized_vals[:min_len]

    # Compute correlations
    rho_eodhd = compute_spearman_rho(eodhd_vals, realized_vals)
    rho_stat = compute_spearman_rho(stat_vals, realized_vals)
    rho_llm = compute_spearman_rho(llm_vals, realized_vals)

    # Optimize weights with 3 signals
    rhos = {'eodhd': rho_eodhd, 'stat': rho_stat, 'llm': rho_llm}
    weights = optimize_weights(rhos)

    # Compute ensemble scores for correlation check
    ensemble_vals = []
    for i in range(min_len):
        ev = (
            weights['eodhd'] * eodhd_vals[i] +
            weights['stat'] * stat_vals[i] +
            weights['llm'] * llm_vals[i]
        )
        ensemble_vals.append(max(-1.0, min(1.0, ev)))

    rho_ensemble = compute_spearman_rho(ensemble_vals, realized_vals)

    # Regime detection
    regime = detect_regime(realized_vals)

    # Store calibration
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cranex_calibration
        (calibrated_at, window_start, window_end, asset,
         w_eodhd, w_stat, w_llm,
         spearman_rho, eodhd_rho, stat_rho, llm_rho,
         realized_vol, n_articles, regime)
        VALUES (NOW(), %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s)
    """, (
        window_start, window_end, asset,
        weights['eodhd'], weights['stat'], weights['llm'],
        rho_ensemble, rho_eodhd, rho_stat, rho_llm,
        0.0, min_len, regime,
    ))
    conn.commit()
    cursor.close()

    # Compute and store volatility
    daily_vol, ann_vol = compute_volatility(conn, asset)
    if ann_vol:
        store_volatility(conn, asset, daily_vol, ann_vol)

    print(f"[Calibrate] {asset}: {min_len} articles, "
          f"ρ_eodhd={rho_eodhd:.3f}, ρ_stat={rho_stat:.3f}, ρ_llm={rho_llm:.3f}, "
          f"ρ_ensemble={rho_ensemble:.3f}, "
          f"weights: eodhd={weights['eodhd']:.3f}, stat={weights['stat']:.3f}, llm={weights['llm']:.3f}, "
          f"regime={regime}")

    # Update ensemble scores with new pred_error
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE cranex_ensemble_scores e
        JOIN (
            SELECT id, es_score as ens FROM cranex_ensemble_scores
            WHERE realized_{asset} IS NOT NULL
              AND scored_at >= %s AND scored_at <= %s
        ) sq ON e.id = sq.id
        SET e.pred_error_{asset} = e.realized_{asset} - e.es_score
        WHERE e.realized_{asset} IS NOT NULL
    """, (window_start, window_end))
    conn.commit()
    cursor.close()


def main():
    print(f"[Calibrate] CRANE-X Calibration starting at {datetime.now(timezone.utc).isoformat()}")

    config = load_config()
    conn = get_connection(config)
    if not conn:
        print("[Calibrate] FATAL: No DB connection")
        sys.exit(1)

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=DEFAULT_LOOKBACK_HOURS)

    print(f"[Calibrate] Window: {window_start.isoformat()} → {window_end.isoformat()}")

    # Compute realized impacts across all assets
    realized = compute_realized_impacts(conn, window_start, window_end)
    print(f"[Calibrate] {len(realized)} realized impact datapoints")

    if not realized:
        print("[Calibrate] No realized impacts to calibrate against")
        conn.close()
        return

    # Calibrate each asset independently
    for asset in ASSETS:
        asset_realized = [r for r in realized if r['asset'] == asset]
        if len(asset_realized) < 3:
            print(f"[Calibrate] {asset}: only {len(asset_realized)} impacts — skipping")
            continue
        calibrate_asset(conn, asset, realized, window_start, window_end)

    conn.close()
    print(f"[Calibrate] Done at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
