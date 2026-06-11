"""
CRANE-X Hourly Telegram Status Report.

Generates a concise status summary and sends it via Telegram.
Call every hour via cron.

Usage:
    python3 hourly_status.py
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

from db import get_connection, load_config


def get_status(conn):
    """Collect all status metrics into a dict."""
    cursor = conn.cursor(dictionary=True)

    # Article counts
    cursor.execute("SELECT COUNT(*) as cnt FROM eodhd_news")
    total_articles = cursor.fetchone()['cnt']

    cursor.execute("SELECT COUNT(*) as cnt FROM eodhd_news WHERE esf IS NOT NULL")
    with_prices = cursor.fetchone()['cnt']

    cursor.execute("""
        SELECT TIMESTAMPDIFF(MINUTE, MAX(date_utc), NOW()) as min_ago
        FROM eodhd_news
    """)
    freshest = cursor.fetchone()['min_ago']

    cursor.execute("""
        SELECT COUNT(*) as cnt FROM eodhd_news
        WHERE ingested_at >= NOW() - INTERVAL 1 HOUR
    """)
    last_hour = cursor.fetchone()['cnt']

    cursor.execute("""
        SELECT COUNT(*) as cnt FROM cranex_ensemble_scores
        WHERE scored_at >= NOW() - INTERVAL 1 HOUR
    """)
    scored_hour = cursor.fetchone()['cnt']

    # Sentiment averages
    cursor.execute("""
        SELECT 
            ROUND(AVG(es_score),4) as es,
            ROUND(AVG(nq_score),4) as nq,
            ROUND(AVG(cl_score),4) as cl,
            ROUND(AVG(btc_score),4) as btc,
            ROUND(AVG(eth_score),4) as eth,
            ROUND(AVG(composite_score),4) as composite,
            COUNT(*) as n
        FROM cranex_ensemble_scores
        WHERE scored_at >= NOW() - INTERVAL 2 HOUR
    """)
    sentiment = cursor.fetchone()

    # Topics polled recently
    cursor.execute("""
        SELECT topic_name, TIMESTAMPDIFF(MINUTE, last_polled_at, NOW()) as min_ago
        FROM eodhd_topics WHERE is_active=TRUE
        ORDER BY last_polled_at DESC LIMIT 5
    """)
    recent_topics = cursor.fetchall()

    # Regimes
    cursor.execute("""
        SELECT DISTINCT asset, regime
        FROM cranex_calibration c1
        WHERE calibrated_at = (
            SELECT MAX(calibrated_at) FROM cranex_calibration c2
            WHERE c2.asset = c1.asset
        )
    """)
    regimes = cursor.fetchall()

    # Price snapshots (latest from daemon / db)
    cursor.execute("""
        SELECT esf, nqf, clf, btc, eth,
               pc_esf, pc_nqf, pc_clf, pc_btc, pc_eth
        FROM eodhd_news
        WHERE esf IS NOT NULL
        ORDER BY date_utc DESC LIMIT 1
    """)
    prices = cursor.fetchone()

    cursor.close()

    return {
        'total_articles': total_articles,
        'with_prices': with_prices,
        'freshest_min_ago': freshest,
        'last_hour_ingested': last_hour,
        'scored_hour': scored_hour,
        'sentiment': sentiment,
        'recent_topics': recent_topics,
        'regimes': regimes,
        'prices': prices,
    }


def format_status(s):
    """Format the status dict into a Telegram message."""
    lines = []
    lines.append("📊 *CRANE-X Hourly Report*")
    lines.append("")

    # Pipeline health
    lines.append("🔧 *Pipeline*")
    lines.append(f"Articles: {s['total_articles']} total, +{s['last_hour_ingested']}/h")
    lines.append(f"With prices: {s['with_prices']} | Scored: {s['scored_hour']}/h")
    lines.append(f"Freshest article: {s['freshest_min_ago']} min ago")
    lines.append("")

    # Recent topics
    if s['recent_topics']:
        lines.append("🔄 *Topics polled*")
        for t in s['recent_topics']:
            ago = t['min_ago']
            label = "just now" if ago < 2 else f"{ago}min ago"
            lines.append(f"  {t['topic_name']}: {label}")
        lines.append("")

    # Current prices
    if s['prices']:
        p = s['prices']
        lines.append("💰 *Prices*")
        lines.append(f"ES {p['esf']} ({p['pc_esf']:+.2f}%)")
        lines.append(f"NQ {p['nqf']} ({p['pc_nqf']:+.2f}%)")
        lines.append(f"CL {p['clf']} ({p['pc_clf']:+.2f}%)")
        lines.append(f"BTC {p['btc']} ({p['pc_btc']:+.2f}%)")
        lines.append(f"ETH {p['eth']} ({p['pc_eth']:+.2f}%)")
        lines.append("")

    # Sentiment
    sen = s['sentiment']
    if sen and sen['n'] and sen['n'] > 0:
        def fmt(v):
            if v is None: return "---"
            return f"{v:+.3f}"
        lines.append("📈 *Sentiment (2h avg)*")
        lines.append(f"ES:  {fmt(sen['es'])}   NQ:  {fmt(sen['nq'])}")
        lines.append(f"CL:  {fmt(sen['cl'])}   BTC: {fmt(sen['btc'])}")
        lines.append(f"ETH: {fmt(sen['eth'])}")
        lines.append(f"Composite: {fmt(sen['composite'])}")
        lines.append("")

    # Regimes
    if s['regimes']:
        emoji_map = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡', 'volatile': '🟣'}
        lines.append("🏷 *Regimes*")
        for r in s['regimes']:
            e = emoji_map.get(r['regime'], '⚪')
            lines.append(f"  {r['asset'].upper()}: {e} {r['regime']}")
        lines.append("")

    lines.append(f"⏱ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("[cranex-gauge.html](https://tradeflags.com/cranex-gauge.html)")

    return '\n'.join(lines)


def main():
    config = load_config()
    conn = get_connection(config)
    if not conn:
        print("[Status] DB connection failed")
        sys.exit(1)

    status = get_status(conn)
    message = format_status(status)
    conn.close()

    # Print to stdout — cron job will capture and deliver
    print(message)


if __name__ == "__main__":
    main()
