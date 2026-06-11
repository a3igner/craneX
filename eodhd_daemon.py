"""
CRANE-X EODHD News Daemon.

Runs continuously, checking the eodhd_topics table every 30 seconds
and polling any topics whose last_polled_at + poll_interval_min has elapsed.

Fetches price snapshots once per cycle, filters headlines
to only those ≤ fresh so prices stay tightly correlated.

Designed to run as a systemd service for reboot survival and
continuous journal logging.

Usage:
    python3 eodhd_daemon.py              # foreground (continuous)
    python3 eodhd_daemon.py --once       # single pass (for cron)
"""

import sys
import os
import signal
import time
import argparse
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eodhd_ingest import (
    get_api_key,
    get_connection,
    get_topics_to_poll,
    fetch_topic_news,
    fetch_dylan_prices,
    filter_fresh_articles,
    insert_articles,
    update_topic_poll_time,
    ensure_tables,
    load_config,
    MAX_HEADLINE_AGE_MINUTES,
)

CHECK_INTERVAL = 30  # seconds between topic-reap checks
HEARTBEAT_INTERVAL = 60  # minutes between heartbeat log lines
SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    if not SHUTDOWN:
        print(f"[Daemon] Received signal {sig}, shutting down gracefully...")
        SHUTDOWN = True


def poll_topic(conn, topic, api_key, dylan_prices):
    """Poll a single topic with price snapshot attached, filter fresh, store."""
    topic_name = topic['topic_name']
    topic_id = topic['id']
    max_articles = topic.get('max_articles') or 25

    print(f"[Daemon] Polling '{topic_name}' (limit={max_articles})...")

    articles = fetch_topic_news(topic_name, api_key, lookback_days=1, limit=max_articles)
    fresh, discarded = filter_fresh_articles(articles)

    if fresh:
        inserted, skipped, errors = insert_articles(conn, topic_name, fresh, prices=dylan_prices)
        status = f"+{inserted} new, {skipped} dup, {errors} err ({discarded} aged out)"
    else:
        status = f"0 fresh (0 new, {discarded} aged out)"

    update_topic_poll_time(conn, topic_id)
    print(f"[Daemon] Done '{topic_name}' — {status}")


def do_poll_cycle(conn, api_key):
    """Fetch prices, find due topics, poll them. Returns True if any were polled."""
    # Fetch prices ONCE — shared across all topics this cycle
    dylan_prices = fetch_dylan_prices()
    if dylan_prices:
        price_str = ", ".join(
            f"{k}={v}" for k, v in sorted(dylan_prices.items()) if v is not None
        )
        print(f"[Daemon] Prices: {price_str}")
    else:
        print("[Daemon] Warning: no Dylan price snapshot")

    topics = get_topics_to_poll(conn)
    if not topics:
        return False

    print(f"[Daemon] {len(topics)} topic(s) due for polling")
    for topic in topics:
        if SHUTDOWN:
            break
        poll_topic(conn, topic, api_key, dylan_prices)
        time.sleep(0.5)  # rate limit between topics

    return True


def main_loop():
    global SHUTDOWN

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    api_key = get_api_key()
    if not api_key:
        print("[Daemon] FATAL: No EODHD_API_KEY found")
        sys.exit(1)

    config = load_config()
    conn = get_connection(config['db'])
    if not conn:
        print("[Daemon] FATAL: Cannot connect to MySQL")
        sys.exit(1)

    ensure_tables(conn)

    print(f"[Daemon] CRANE-X EODHD Daemon started at {datetime.now(timezone.utc).isoformat()}")
    print(f"[Daemon] Checking topics every {CHECK_INTERVAL}s")
    print(f"[Daemon] Headline freshness filter: ≤{MAX_HEADLINE_AGE_MINUTES}min old")
    print(f"[Daemon] All topics: 15-min interval, with price snapshots")

    last_heartbeat = time.time()
    consecutive_db_errors = 0

    while not SHUTDOWN:
        try:
            any_polled = do_poll_cycle(conn, api_key)

            if not any_polled:
                # Nothing due — log heartbeat every 60 minutes
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL * 60:
                    print(f"[Daemon] Heartbeat — all topics current at {datetime.now(timezone.utc).isoformat()}")
                    last_heartbeat = now

            consecutive_db_errors = 0

        except Exception as e:
            consecutive_db_errors += 1
            print(f"[Daemon] Error: {e}")

            if consecutive_db_errors >= 3:
                print(f"[Daemon] {consecutive_db_errors} consecutive errors — reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_connection(config)
                if conn:
                    print("[Daemon] Reconnected successfully")
                    consecutive_db_errors = 0
                else:
                    print("[Daemon] Reconnect failed, will retry...")

        # Sleep with 1s granularity (wake early on SIGTERM)
        for _ in range(CHECK_INTERVAL):
            if SHUTDOWN:
                break
            time.sleep(1)

    # Graceful shutdown
    try:
        conn.close()
    except Exception:
        pass
    print(f"[Daemon] Stopped at {datetime.now(timezone.utc).isoformat()}")


def main():
    parser = argparse.ArgumentParser(description='CRANE-X EODHD Daemon')
    parser.add_argument('--once', action='store_true',
                        help='Run a single pass and exit (for cron)')
    args = parser.parse_args()

    if args.once:
        api_key = get_api_key()
        if not api_key:
            print("[Daemon] FATAL: No EODHD_API_KEY found")
            sys.exit(1)

        config = load_config()
        conn = get_connection(config['db'])
        if not conn:
            print("[Daemon] FATAL: Cannot connect to MySQL")
            sys.exit(1)

        ensure_tables(conn)
        do_poll_cycle(conn, api_key)
        conn.close()
        return

    main_loop()


if __name__ == "__main__":
    main()
