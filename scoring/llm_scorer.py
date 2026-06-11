"""
CRANE-X LLM Scorer.

Scores eodhd_news article content (title + content) using DeepSeek API.
Batches articles to minimize API calls — small batches because content
is ~700 words per article.

Stores results in cranex_ensemble_scores.llm_signal.

The ensemble scorer (scorer.py) then incorporates llm_signal as a 3rd
weighted signal, alongside EODHD pre-baked sentiment and statistical clusters.
"""

import sys
import os
import json
import time
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

import requests
from db import get_connection
from config import load_config

cfg = load_config()

# DeepSeek API
API_URL = cfg['api']['deepseek']['base_url']
MODEL = cfg['api']['deepseek']['model']
BATCH_SIZE = cfg.get('pipeline', {}).get('batch_size', 4)
MAX_PER_RUN = cfg.get('pipeline', {}).get('max_per_run', 200)

SYSTEM_PROMPT = """You are a financial sentiment analyst. Score each news article on a scale from -1.0 (very bearish) to +1.0 (very bullish), where 0 = neutral.

Consider the article's likely market impact: earnings results, macroeconomic data, central bank decisions, geopolitical events, mergers/acquisitions, product launches, regulatory news, analyst actions, and market trends.

Each article is formatted as:
--- Article N ---
TITLE: <headline>
CONTENT: <full article text>

Return a JSON array only. Each element must have:
- "score": float -1.0 to +1.0
- "confidence": float 0.0 to 1.0
- "themes": comma-separated keywords (max 3)

Examples:
Input: --- Article 1 ---
TITLE: Fed cuts rates by 50bps
CONTENT: The Federal Reserve today announced...

--- Article 2 ---
TITLE: Tech stocks plunge on AI fears  
CONTENT: Technology shares fell sharply...

Output:
[
  {"score": 0.8, "confidence": 0.9, "themes": "monetary policy, rate cut"},
  {"score": -0.7, "confidence": 0.85, "themes": "tech selloff, AI"}
]"""


def get_api_key():
    """Get DeepSeek API key from environment or .env."""
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        for dotenv_path in [
            '/home/a3/crane-x/.env',
            '/home/a3/.env',
            '/home/a3/claudecode/.env',
        ]:
            if os.path.exists(dotenv_path):
                with open(dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            if k.strip() == 'DEEPSEEK_API_KEY':
                                key = v.strip().strip('"').strip("'")
                                break
                if key:
                    break
    return key


def get_unscored(conn, limit=MAX_PER_RUN):
    """Fetch eodhd_news articles that don't have an LLM score yet."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT n.id, n.title, n.content
        FROM eodhd_news n
        LEFT JOIN cranex_ensemble_scores e ON e.article_id = n.id
        WHERE (e.llm_signal IS NULL OR e.id IS NULL)
          AND n.title IS NOT NULL AND n.title != ''
        ORDER BY n.date_utc DESC
        LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def format_batch(articles):
    """Format a batch of articles for the LLM prompt."""
    parts = []
    for i, art in enumerate(articles, 1):
        title = (art['title'] or '').strip()
        content = (art.get('content') or '')
        # Truncate content to ~1500 chars to keep prompt size manageable
        if len(content) > 1500:
            content = content[:1500] + "..."
        parts.append(f"--- Article {i} ---\nTITLE: {title}\nCONTENT: {content}")
    return "\n\n".join(parts)


def call_deepseek(articles):
    """Send a batch of articles to DeepSeek for scoring."""
    api_key = get_api_key()
    if not api_key:
        print("[LLM] FATAL: No DEEPSEEK_API_KEY found")
        return None

    formatted = format_batch(articles)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON (handle markdown-wrapped)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        results = json.loads(content)
        if not isinstance(results, list):
            results = [results]

        return results

    except requests.exceptions.Timeout:
        print(f"[LLM] Timeout on batch of {len(articles)}")
        return None
    except Exception as e:
        print(f"[LLM] API error: {e}")
        return None


def store_results(conn, articles, results):
    """Update cranex_ensemble_scores with LLM scores.

    If an ensemble row doesn't exist yet, creates one with just the LLM signal.
    """
    cursor = conn.cursor()
    updated = 0

    for art, result in zip(articles, results):
        if not isinstance(result, dict):
            continue

        score = result.get("score")
        if score is None:
            continue

        # Clamp to -1..+1
        score = max(-1.0, min(1.0, float(score)))

        # Upsert: if ensemble row exists, update llm_signal; if not, insert minimal row
        cursor.execute("""
            INSERT INTO cranex_ensemble_scores
            (article_id, scored_at, llm_signal)
            VALUES (%s, NOW(), %s)
            ON DUPLICATE KEY UPDATE
                llm_signal = VALUES(llm_signal),
                scored_at = NOW()
        """, (art['id'], score))

        updated += cursor.rowcount

    conn.commit()
    cursor.close()
    return updated


def score_pending(conn, limit=MAX_PER_RUN):
    """Main entry point: score all unscored articles."""
    unscored = get_unscored(conn, limit)
    if not unscored:
        print("[LLM] No unscored articles")
        return 0

    print(f"[LLM] Scoring {len(unscored)} articles in batches of {BATCH_SIZE}...")

    total_updated = 0
    total_cost = 0.0

    for i in range(0, len(unscored), BATCH_SIZE):
        batch = unscored[i:i + BATCH_SIZE]

        print(f"[LLM] Batch {i // BATCH_SIZE + 1}/{(len(unscored) + BATCH_SIZE - 1) // BATCH_SIZE}: "
              f"{len(batch)} articles...")

        results = call_deepseek(batch)
        if results is None:
            print(f"[LLM]   -> Batch failed, skipping")
            continue

        # Results may not match 1:1
        n = min(len(batch), len(results))
        if n > 0:
            updated = store_results(conn, batch[:n], results[:n])
            total_updated += updated
            # Estimate cost: DeepSeek ~$0.0005 per 1K tokens, ~500 tokens per article
            batch_cost = len(batch) * 500 * 0.0005 / 1000
            total_cost += batch_cost
            print(f"[LLM]   -> {updated} scored (~${batch_cost:.4f})")

        # Rate limit: 0.5s between batches
        if i + BATCH_SIZE < len(unscored):
            time.sleep(0.5)

    print(f"[LLM] Total: {total_updated} scored, ~${total_cost:.4f}")
    return total_updated


def main():
    print(f"[LLM] CRANE-X LLM Scorer starting at {datetime.now(timezone.utc).isoformat()}")

    api_key = get_api_key()
    if not api_key:
        print("[LLM] FATAL: No DEEPSEEK_API_KEY")
        sys.exit(1)
    print(f"[LLM] API key found: {api_key[:8]}...{api_key[-4:]}")

    conn = get_connection(cfg['db'])
    if not conn:
        print("[LLM] FATAL: No DB connection")
        sys.exit(1)

    score_pending(conn)
    conn.close()
    print(f"[LLM] Done at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
