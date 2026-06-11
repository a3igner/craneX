"""
EODHD news ingestion for CRANE-X.

Polls EODHD News API by topic tag (t=TOPICNAME), deduplicates by
headline hash, attaches price snapshots from external price API, and stores
results in eodhd_news with pre-baked sentiment scores.

Replaces the old TradeFlags NewsFeed poller + FinBERT lexicon step.

API endpoints are configured in config.yaml (see config.yaml.example).

Only stores headlines fresher than MAX_HEADLINE_AGE_MINUTES (default 15 min)
so price snapshots are tightly correlated with the news.
"""

import sys
import os
import hashlib
import json
import time
import re
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import requests
from db import get_connection
from config import load_config

cfg = load_config()

EODHD_BASE = cfg['api']['eodhd']['base_url']

# External price API — real-time futures/crypto prices
WSJ_TOKEN = cfg['secrets']['wsj_entitlement_token']
WSJ_CKEY = cfg['secrets']['wsj_ckey']
WSJ_PRICE_URL = cfg['api']['wsj_dylan']['base_url']
WSJ_TICKERS = [
    tuple(t) for t in cfg.get('wsj_tickers', [
        ("Future-US-ES00", "esf", "pc_esf"),       # S&P 500 E-mini
        ("Future-US-NQ00", "nqf", "pc_nqf"),        # Nasdaq 100 E-mini
        ("Future-UK-BRN00", "clf", "pc_clf"),       # Brent Crude
        ("CryptoCurrency-US-BTCUSD", "btc", "pc_btc"),
        ("CryptoCurrency-US-ETHUSD", "eth", "pc_eth"),
    ])
]

MAX_HEADLINE_AGE_MINUTES = cfg.get('pipeline', {}).get('max_headline_age_minutes', 60)


def get_api_key():
    """Get EODHD API key from environment."""
    key = os.environ.get('EODHD_API_KEY')
    if not key:
        for dotenv_path in [
            '/home/a3/crane-x/.env',
            '/home/a3/.env',
        ]:
            if os.path.exists(dotenv_path):
                with open(dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            if k.strip() == 'EODHD_API_KEY':
                                key = v.strip().strip('"').strip("'")
                                break
                if key:
                    break
    return key


def get_topics_to_poll(conn):
    """Return list of active topics that are due for polling."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, topic_name, poll_interval_min, max_articles, last_polled_at
        FROM eodhd_topics
        WHERE is_active = TRUE
          AND (last_polled_at IS NULL
               OR last_polled_at <= NOW() - INTERVAL poll_interval_min MINUTE)
        ORDER BY last_polled_at ASC
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def fetch_topic_news(topic_name, api_key, lookback_days=1, limit=25):
    """Fetch news articles for a topic from EODHD."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    yesterday = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    url = f"{EODHD_BASE}?t={topic_name}&limit={limit}&api_token={api_key}&fmt=json"
    if lookback_days > 0:
        url += f"&from={yesterday}&to={today}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, dict) and 'articles' in data:
            return data['articles']
        else:
            print(f"[EODHD] Unexpected response type for '{topic_name}': {type(data)}")
            return []
    except requests.exceptions.Timeout:
        print(f"[EODHD] Timeout fetching '{topic_name}'")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[EODHD] Error fetching '{topic_name}': {e}")
        return []


def fetch_dylan_prices():
    """Fetch current prices from external price API.

    Returns dict like {esf: 5432.10, pc_esf: 0.32, nqf: 19500.00, ...}
    or empty dict on failure.
    """
    prices = {}
    try:
        ids = "%2C".join(t[0] for t in WSJ_TICKERS)
        url = (
            f"{WSJ_PRICE_URL}?dialect=official&needed=CompositeTrading|BluegrassChannels"
            f"&MaxInstrumentMatches=1&accept=application/json"
            f"&EntitlementToken={WSJ_TOKEN}&ckey={WSJ_CKEY}&dialects=Charting"
            f"&id={ids}"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        instruments = data["InstrumentResponses"]

        for idx, inst in enumerate(instruments):
            if idx >= len(WSJ_TICKERS):
                break
            field, price_col, pct_col = WSJ_TICKERS[idx]
            try:
                ct = inst["Matches"][0]["CompositeTrading"]
                price = ct["Last"]["Price"]["Value"]
                change_pct = ct.get("ChangePercent", 0) or 0
                prices[price_col] = round(float(price), 2)
                prices[pct_col] = round(float(change_pct), 2)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                print(f"[Prices] Failed to parse {field}: {e}")
                prices[price_col] = None
                prices[pct_col] = None

    except Exception as e:
        print(f"[Prices] Fetch error: {e}")

    return prices


def filter_fresh_articles(articles, max_age_minutes=MAX_HEADLINE_AGE_MINUTES):
    """Filter articles to only those published within the last max_age_minutes."""
    now = datetime.now(timezone.utc)
    kept = []
    discarded = 0
    for art in articles:
        pub_dt = parse_eodhd_datetime(art.get('date'))
        if pub_dt is None:
            discarded += 1
            continue
        age = (now - pub_dt).total_seconds() / 60
        if age <= max_age_minutes:
            kept.append(art)
        else:
            discarded += 1
    return kept, discarded


def make_headline_hash(title):
    """SHA256 of normalized title for dedup."""
    h = title.lower().strip()
    h = re.sub(r'\s+', ' ', h)
    h = h.rstrip('.,;:!?')
    return hashlib.sha256(h.encode('utf-8')).hexdigest()


def parse_eodhd_datetime(dt_str):
    """Parse EODHD datetime string to Python datetime."""
    if not dt_str:
        return None
    for fmt in [
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
    ]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def insert_articles(conn, topic, articles, prices=None):
    """Batch-insert articles into eodhd_news, skipping dupes.

    prices: dict from fetch_dylan_prices() to attach price snapshots.
    """
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    errors = 0

    if prices is None:
        prices = {}

    insert_sql = """
        INSERT IGNORE INTO eodhd_news
        (topic, date_utc, title, content, httplink, symbols, tags,
         esf, pc_esf, nqf, pc_nqf, clf, pc_clf, btc, pc_btc, eth, pc_eth,
         sentiment_polarity, sentiment_neg, sentiment_neu, sentiment_pos,
         headline_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s)
    """

    for art in articles:
        title = (art.get('title') or '').strip()
        if not title:
            skipped += 1
            continue

        headline_hash = make_headline_hash(title)
        sentiment = art.get('sentiment') or {}

        try:
            cursor.execute(insert_sql, (
                topic,
                parse_eodhd_datetime(art.get('date')),
                title,
                art.get('content') or '',
                art.get('link') or '',
                json.dumps(art.get('symbols') or []),
                json.dumps(art.get('tags') or []),
                # Price snapshots from external API
                safe_float(prices.get('esf')),
                safe_float(prices.get('pc_esf')),
                safe_float(prices.get('nqf')),
                safe_float(prices.get('pc_nqf')),
                safe_float(prices.get('clf')),
                safe_float(prices.get('pc_clf')),
                safe_float(prices.get('btc')),
                safe_float(prices.get('pc_btc')),
                safe_float(prices.get('eth')),
                safe_float(prices.get('pc_eth')),
                # Sentiment
                safe_float(sentiment.get('polarity')),
                safe_float(sentiment.get('neg')),
                safe_float(sentiment.get('neu')),
                safe_float(sentiment.get('pos')),
                headline_hash,
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[EODHD] Insert error: {e}")
            errors += 1

    conn.commit()
    cursor.close()
    return inserted, skipped, errors


def update_topic_poll_time(conn, topic_id):
    """Set last_polled_at for a topic."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE eodhd_topics SET last_polled_at = NOW() WHERE id = %s",
        (topic_id,)
    )
    conn.commit()
    cursor.close()


def ensure_tables(conn):
    """Create schema tables if they don't exist."""
    cursor = conn.cursor()

    schema_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sql', '001_eodhd_schema.sql'
    )
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            sql = f.read()
        for statement in sql.split(';'):
            stmt = statement.strip()
            if stmt and not stmt.startswith('--'):
                try:
                    cursor.execute(stmt)
                except Exception as e:
                    print(f"[Schema] Skipping statement: {e}")
        conn.commit()
        print("[Schema] Tables ensured from 001_eodhd_schema.sql")
    else:
        print(f"[Schema] Warning: {schema_path} not found")

    cursor.close()


def main():
    api_key = get_api_key()
    if not api_key:
        print("[EODHD] FATAL: No EODHD_API_KEY found in environment or .env")
        sys.exit(1)

    print(f"[EODHD] CRANE-X News Ingestion starting at {datetime.now(timezone.utc).isoformat()}")

    conn = get_connection(cfg['db'])
    if not conn:
        print("[EODHD] FATAL: Could not connect to MySQL")
        sys.exit(1)

    ensure_tables(conn)

    # Fetch prices ONCE per run — all topics share the same snapshot
    dylan_prices = fetch_dylan_prices()
    if dylan_prices:
        price_str = ", ".join(
            f"{k}={v}" for k, v in sorted(dylan_prices.items()) if v is not None
        )
        print(f"[Prices] Price snapshot: {price_str}")
    else:
        print("[Prices] Warning: no price snapshot (prices will be NULL)")

    topics = get_topics_to_poll(conn)
    print(f"[EODHD] Found {len(topics)} topics to poll")

    total_inserted = 0
    total_skipped = 0

    for topic in topics:
        topic_name = topic['topic_name']
        topic_id = topic['id']
        max_articles = topic.get('max_articles') or 25

        print(f"[EODHD] Polling topic '{topic_name}' (limit={max_articles})...")

        articles = fetch_topic_news(topic_name, api_key, lookback_days=1, limit=max_articles)
        print(f"[EODHD]   -> {len(articles)} articles from API")

        # Only keep headlines fresher than MAX_HEADLINE_AGE_MINUTES
        fresh_articles, discarded = filter_fresh_articles(articles)
        print(f"[EODHD]   -> {len(fresh_articles)} fresh (≤{MAX_HEADLINE_AGE_MINUTES}min), {discarded} too old")

        if fresh_articles:
            inserted, skipped, errors = insert_articles(
                conn, topic_name, fresh_articles, prices=dylan_prices
            )
            total_inserted += inserted
            total_skipped += skipped
            print(f"[EODHD]   -> Inserted: {inserted}, Skipped: {skipped}, Errors: {errors}")
        else:
            print(f"[EODHD]   -> No fresh articles to insert")

        update_topic_poll_time(conn, topic_id)
        time.sleep(1.0)

    conn.close()

    print(f"[EODHD] Done. Total: {total_inserted} new, {total_skipped} skipped")
    print(f"[EODHD] Finished at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
