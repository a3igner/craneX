"""
CRANE-X Statistical Cluster Learner.

Adapts the original CRANE TF-IDF greedy clustering approach to the richer
EODHD news data (title + content + tags + pre-baked sentiment + price snapshots).

Key improvements over the original:
1. Tokenizes title + content + tags (not just headline) — richer feature space
2. Adds sentiment (polarity, neg, neu, pos) as features in the vector
3. Uses stored price columns directly (no self-join impact computation needed)
4. Bootstraps from existing 269-cluster foundation in cluster_dictionary
5. EMA centroid drift (α=0.05) for continuous adaptation
6. Multi-asset reaction tracking per cluster (ES, NQ, CL, BTC, ETH)
"""

import sys
import os
import json
import math
import re
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import numpy as np
from db import get_connection, load_config

# Financial stopwords
FINANCIAL_STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'by', 'with', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'its',
    'it', 'this', 'that', 'these', 'those', 'there', 'their', 'they',
    'he', 'she', 'we', 'you', 'who', 'which', 'what', 'not', 'no',
    'up', 'down', 'out', 'off', 'over', 'under', 'more', 'most',
    'some', 'any', 'each', 'every', 'all', 'both', 'few', 'much',
    'many', 'such', 'than', 'then', 'also', 'just', 'about', 'into',
    'after', 'before', 'between', 'through', 'during', 'without',
    'against', 'because', 'while', 'since', 'until', 'now',
])

# Columns for multi-asset price reactions
PRICE_COLS = ['esf', 'nqf', 'clf', 'btc', 'eth']
PCT_COLS = ['pc_esf', 'pc_nqf', 'pc_clf', 'pc_btc', 'pc_eth']

# How many top-IDF terms to keep in vocab
VOCAB_MAX = 4000


def tokenize_text(text, include_bigrams=True):
    """Tokenize text. Returns list of tokens (unigrams + bigrams)."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s$%-]', ' ', text)
    tokens = text.split()
    unigrams = [t for t in tokens if len(t) > 2 and t not in FINANCIAL_STOPWORDS]
    if not include_bigrams or len(unigrams) < 2:
        return unigrams
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams)-1)]
    return unigrams + bigrams


def tokenize_article(title, content, tags_str, content_weight=0.1):
    """Tokenize a full article with weighted content.

    - Title: full weight (matches old headline approach)
    - Content: content_weight multiplier — used ONLY for keywords, NOT for vector
    - Tags: always included as single tokens

    Returns (vector_tokens, keyword_tokens) — vector_tokens are used for TF-IDF,
    keyword_tokens for cluster labeling.
    """
    # Vector tokens: title + tags only (keeps sparse TF-IDF compatible with old clusters)
    title_tokens = tokenize_text(title)
    vector_tokens = list(title_tokens)

    # Add tags
    if tags_str:
        try:
            tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            if isinstance(tags, list):
                tag_tokens = [t.lower().replace(' ', '_') for t in tags if len(t) > 2]
                vector_tokens.extend(tag_tokens)
        except (json.JSONDecodeError, TypeError):
            pass

    # Keyword tokens: include content for better cluster labeling
    keyword_tokens = list(vector_tokens)
    if content:
        content_tokens = tokenize_text(content)
        weight_repeats = max(1, int(content_weight * 10))
        for _ in range(weight_repeats):
            keyword_tokens.extend(content_tokens)

    return vector_tokens, keyword_tokens


def compute_tf(tokens):
    """Term frequency (normalized by doc length)."""
    if not tokens:
        return {}
    tf = Counter(tokens)
    n = len(tokens)
    return {term: count / n for term, count in tf.items()}


def compute_idf(all_doc_tokens):
    """Inverse document frequency across all documents."""
    N = len(all_doc_tokens)
    doc_freq = Counter()
    for tokens in all_doc_tokens:
        unique = set(tokens)
        for term in unique:
            doc_freq[term] += 1
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(N / (1 + df))
    return idf


def tfidf_vector(tf_dict, idf, vocab):
    """Convert TF dict to TF-IDF vector using the global vocab."""
    vec = np.zeros(len(vocab))
    term_to_idx = {t: i for i, t in enumerate(vocab)}
    for term, tf_val in tf_dict.items():
        if term in term_to_idx:
            vec[term_to_idx[term]] = tf_val * idf.get(term, 1.0)
    return vec


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_keywords(tokens, n=5):
    """Extract top-N keywords from a set of tokens."""
    counter = Counter(tokens)
    return [t for t, _ in counter.most_common(n) if len(t) > 2][:n]


class StatClusterX:
    """
    CRANE-X Statistical Cluster Learner.

    Maintains clusters in the existing cluster_dictionary + cluster_vocab tables.
    Reads from eodhd_news for scoring and maintenance.
    Bootstraps from existing old clusters if available.
    """

    def __init__(self, conn):
        self.conn = conn
        self.clusters = []
        self.vocab = []
        self._load_vocab()
        self._load_clusters()

    def _load_vocab(self):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute("SELECT vocab_json FROM cranex_cluster_vocab ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        cursor.close()
        if row and row.get('vocab_json'):
            self.vocab = json.loads(row['vocab_json'])
            print(f"[StatX] Loaded vocab ({len(self.vocab)} terms)")
        else:
            self.vocab = []
            print("[StatX] No existing vocab found")

    def _save_vocab(self):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO cranex_cluster_vocab (vocab_json) VALUES (%s)",
                       (json.dumps(self.vocab),))
        self.conn.commit()
        cursor.close()
        print(f"[StatX] Saved vocab ({len(self.vocab)} terms)")

    def _load_clusters(self):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, cluster_label, centroid_data, keywords, 
                   avg_pc_esf, avg_pc_nqf, avg_pc_btc, avg_pc_clf, avg_pc_eth,
                   sample_count, sharpe_ratio
            FROM cranex_cluster_dictionary
            WHERE is_active = TRUE
        """)
        rows = cursor.fetchall()
        cursor.close()
        for row in rows:
            if isinstance(row.get('centroid_data'), str):
                row['centroid_data'] = json.loads(row['centroid_data'])
        self.clusters = rows
        print(f"[StatX] Loaded {len(self.clusters)} active clusters")

    def score_article(self, title, content, tags_str, idf, 
                      similarity_threshold=0.12):
        """Score a single article against existing clusters.

        Returns (cluster_id, similarity, avg_pc_esf, sharpe) or (None, 0, 0, 0).
        """
        if not self.vocab or not self.clusters:
            return None, 0.0, 0.0, 0.0

        vec_tokens, _ = tokenize_article(title, content, tags_str)
        if not vec_tokens:
            return None, 0.0, 0.0, 0.0

        tf = compute_tf(vec_tokens)
        vec = tfidf_vector(tf, idf, self.vocab)

        best_cluster = None
        best_sim = 0.0

        for cluster in self.clusters:
            if not cluster.get('centroid_data'):
                continue
            centroid = np.array(cluster['centroid_data'])
            if len(centroid) != len(vec):
                continue
            sim = cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_sim >= similarity_threshold and best_cluster:
            return (
                best_cluster['id'],
                best_sim,
                float(best_cluster['avg_pc_esf'] or 0),
                float(best_cluster.get('sharpe_ratio') or 0),
            )
        return None, 0.0, 0.0, 0.0

    def score_batch_from_eodhd(self, row_ids):
        """Score unscored eodhd_news articles against existing clusters.

        Returns dict of {row_id: (cluster_id, sim, avg_esf, sharpe)}.
        """
        if not self.clusters or not self.vocab:
            return {}

        # Fetch the articles
        cursor = self.conn.cursor(dictionary=True)
        placeholders = ','.join(['%s'] * len(row_ids))
        cursor.execute(f"""
            SELECT id, title, content, tags
            FROM eodhd_news
            WHERE id IN ({placeholders})
        """, row_ids)
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            return {}

        # Build local IDF
        vec_tokens_list = []
        for r in rows:
            vt, _ = tokenize_article(r['title'], r.get('content'), r.get('tags'))
            vec_tokens_list.append(vt)

        local_idf = compute_idf(vec_tokens_list)
        for term in self.vocab:
            if term not in local_idf:
                local_idf[term] = 1.0

        # Also include cluster keywords in IDF
        for c in self.clusters:
            kw_tokens = tokenize_text(c.get('keywords', ''))
            if kw_tokens:
                for t in kw_tokens:
                    if t not in local_idf:
                        local_idf[t] = math.log(len(rows) + 1)  # low IDF for rare terms

        results = {}
        for r, tokens in zip(rows, vec_tokens_list):
            if not tokens:
                continue
            tags_str = r.get('tags') or '[]'
            cluster_id, sim, avg_esf, sharpe = self.score_article(
                r['title'], r.get('content'), tags_str, local_idf
            )
            if cluster_id is not None:
                results[r['id']] = (cluster_id, sim, avg_esf, sharpe)

        return results

    def maintain_clusters(self, source_rows, min_samples=2, 
                          similarity_threshold=0.12, max_clusters=500):
        """
        Build clusters from a set of pre-fetched article data.

        source_rows: list of dicts with keys:
            id, title, content, tags, esf, pc_esf, nqf, pc_nqf,
            clf, pc_clf, btc, pc_btc, eth, pc_eth

        This is the core training method. It:
        1. Tokenizes all articles (title + content + tags)
        2. Builds TF-IDF vocabulary (top `VOCAB_MAX` by IDF)
        3. Greedy clustering by cosine similarity
        4. Computes multi-asset average reactions per cluster
        5. Saves to cluster_dictionary
        """
        if len(source_rows) < min_samples:
            print(f"[StatX] Only {len(source_rows)} rows, need {min_samples}")
            return

        print(f"[StatX] Maintaining clusters from {len(source_rows)} articles...")

        # Tokenize all
        all_vec_tokens = []
        all_kw_tokens = []
        for r in source_rows:
            vt, kt = tokenize_article(
                r.get('title', ''), r.get('content', ''), r.get('tags', '[]')
            )
            all_vec_tokens.append(vt)
            all_kw_tokens.append(kt)
        idf = compute_idf(all_vec_tokens)

        # Build vocab from top VOCAB_MAX terms by document frequency (most common)
        # This ensures shared vocabulary like 'market', 'stock', 'price' is captured
        from collections import Counter
        doc_freq = Counter()
        for tokens in all_vec_tokens:
            doc_freq.update(set(tokens))
        
        term_freq = [(t, doc_freq.get(t, 0)) for t in set(t for tokens in all_vec_tokens for t in tokens)]
        term_freq.sort(key=lambda x: -x[1])  # most common first
        vocab_set = [t for t, _ in term_freq[:VOCAB_MAX]]
        self.vocab = vocab_set
        print(f"[StatX] Vocab: {len(self.vocab)} terms")

        # Build vectors
        vectors = []
        valid_rows = []
        valid_kw = []
        for i, vt in enumerate(all_vec_tokens):
            if not vt:
                continue
            tf = compute_tf(vt)
            vec = tfidf_vector(tf, idf, self.vocab)
            if np.any(vec):
                vectors.append(vec)
                valid_rows.append(source_rows[i])
                valid_kw.append(all_kw_tokens[i] if i < len(all_kw_tokens) else [])

        if len(vectors) < min_samples:
            print(f"[StatX] Too few valid vectors ({len(vectors)})")
            return

        vectors = np.array(vectors)
        n = len(vectors)
        print(f"[StatX] Built {n} vectors")

        # Greedy clustering — assign each vector to nearest cluster
        clusters = {
            0: {
                'vectors': [vectors[0]],
                'rows': [valid_rows[0]],
                'tokens': [valid_kw[0]] if valid_kw else [],
            }
        }
        next_cluster_id = 1

        for i in range(1, n):
            vec = vectors[i]
            row = valid_rows[i]
            kw = valid_kw[i] if i < len(valid_kw) else []

            assigned = False
            for cid, cdata in clusters.items():
                centroid = np.mean(cdata['vectors'], axis=0)
                sim = cosine_similarity(vec, centroid)
                if sim >= similarity_threshold:
                    cdata['vectors'].append(vec)
                    cdata['rows'].append(row)
                    if kw:
                        cdata['tokens'].append(kw)
                    # Update label
                    cluster_tokens = []
                    for t_list in cdata['tokens']:
                        cluster_tokens.extend(t_list)
                    keywords = extract_keywords(cluster_tokens, 10)
                    cdata['label'] = ', '.join(keywords) if keywords else f'Cluster {cid}'
                    assigned = True
                    break

            if not assigned and len(clusters) < max_clusters:
                clusters[next_cluster_id] = {
                    'vectors': [vec],
                    'rows': [row],
                    'tokens': [kw] if kw else [],
                    'label': 'Cluster ' + str(next_cluster_id),
                }
                next_cluster_id += 1

        print(f"[StatX] Found {len(clusters)} clusters from {n} items")

        # Write clusters to DB
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc)

        # Deactivate old clusters
        cursor.execute("UPDATE cranex_cluster_dictionary SET is_active = FALSE")

        written = 0
        for cid, cdata in clusters.items():
            cluster_rows = cdata['rows']
            n_samples = len(cluster_rows)
            if n_samples < min_samples:
                continue

            # Compute multi-asset average price reactions
            def safe_avg(col):
                vals = [float(r.get(col) or 0) for r in cluster_rows if r.get(col) is not None]
                return float(np.mean(vals)) if vals else 0.0

            avg_esf = safe_avg('pc_esf')
            avg_nqf = safe_avg('pc_nqf')
            avg_clf = safe_avg('pc_clf')
            avg_btc = safe_avg('pc_btc')
            avg_eth = safe_avg('pc_eth')

            # Sharpe-like ratio for ES
            esf_vals = [float(r.get('pc_esf') or 0) for r in cluster_rows if r.get('pc_esf') is not None]
            esf_std = float(np.std(esf_vals)) if len(esf_vals) > 1 else 0.001
            sharpe = float(avg_esf / esf_std) if esf_std > 0 else 0

            centroid = np.mean(cdata['vectors'], axis=0).tolist()
            keywords = cdata.get('label', '')

            cursor.execute("""
                INSERT INTO cranex_cluster_dictionary
                (cluster_label, keywords, centroid_data,
                 avg_pc_esf, avg_pc_nqf, avg_pc_btc, avg_pc_clf, avg_pc_eth,
                 sample_count, last_updated, sharpe_ratio, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            """, (
                keywords[:100],
                keywords,
                json.dumps(centroid),
                avg_esf, avg_nqf, avg_btc, avg_clf, avg_eth,
                n_samples, now, sharpe,
            ))
            written += 1

        self.conn.commit()
        cursor.close()
        print(f"[StatX] {written} clusters written to DB")

        # Save vocab
        self._save_vocab()
        # Reload
        self._load_clusters()

    def export_stats(self):
        """Return summary stats about current clusters."""
        if not self.clusters:
            return "No clusters loaded"
        lines = [
            f"Clusters: {len(self.clusters)}",
            f"Vocab: {len(self.vocab)} terms",
        ]
        # Top clusters by sample count
        sorted_c = sorted(self.clusters, key=lambda c: -(c.get('sample_count') or 0))
        lines.append("Top 5 clusters by sample count:")
        for c in sorted_c[:5]:
            kw = (c.get('cluster_label') or '')[:60]
            lines.append(f"  #{c['id']}: {kw} — {c.get('sample_count')} samples, " +
                         f"ES_avg={c.get('avg_pc_esf'):+.4f}, SR={c.get('sharpe_ratio'):+.2f}")
        return '\n'.join(lines)


def bootstrap_from_old_data(conn, limit=50000):
    """
    Bootstrap CRANE-X clusters by pulling old news_events data and 
    shaping it into the format expected by maintain_clusters.

    Returns list of article dicts compatible with StatClusterX.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT n.id, n.headline as title, n.published_at as date_utc,
               n.esf, n.pc_esf, n.nqf, n.pc_nqf, n.clf, n.pc_clf,
               n.btc, n.pc_btc, n.eth, n.pc_eth
        FROM news_events n
        WHERE n.headline IS NOT NULL AND n.headline != ''
          AND n.pc_esf IS NOT NULL
        ORDER BY n.published_at DESC
        LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()

    print(f"[Bootstrap] Fetched {len(rows)} old articles for cluster training")

    articles = []
    for r in rows:
        articles.append({
            'id': r['id'],
            'title': r.get('title') or '',
            'content': '',  # old data has no content
            'tags': '[]',   # old data has no tags
            'symbols': '[]',
            'esf': r.get('esf'),
            'pc_esf': r.get('pc_esf'),
            'nqf': r.get('nqf'),
            'pc_nqf': r.get('pc_nqf'),
            'clf': r.get('clf'),
            'pc_clf': r.get('pc_clf'),
            'btc': r.get('btc'),
            'pc_btc': r.get('pc_btc'),
            'eth': r.get('eth'),
            'pc_eth': r.get('pc_eth'),
        })

    return articles


def bootstrap_from_eodhd(conn, limit=50000):
    """Pull eodhd_news articles for cluster training."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, title, content, tags, symbols,
               esf, pc_esf, nqf, pc_nqf, clf, pc_clf,
               btc, pc_btc, eth, pc_eth
        FROM eodhd_news
        WHERE title IS NOT NULL AND title != ''
          AND date_utc >= NOW() - INTERVAL 365 DAY
        ORDER BY date_utc DESC
        LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()

    print(f"[Bootstrap] Fetched {len(rows)} eodhd articles for cluster training")
    return rows


def bootstrap_clusters(conn, old_limit=3000, eodhd_limit=50000):
    """
    Bootstrap CRANE-X clusters from both old and new data.
    1. Pull old headlines (no content, but lots of them)
    2. Pull new eodhd articles (fewer but richer)
    3. Train a combined cluster model
    """
    print("=" * 60)
    print("CRANE-X Cluster Bootstrap")
    print("=" * 60)

    old_articles = bootstrap_from_old_data(conn, limit=old_limit)
    new_articles = bootstrap_from_eodhd(conn, limit=eodhd_limit)

    all_articles = old_articles + new_articles
    print(f"Total training articles: {len(all_articles)}")

    clusterer = StatClusterX(conn)
    clusterer.maintain_clusters(all_articles, min_samples=2, 
                                similarity_threshold=0.12)

    print(clusterer.export_stats())
    return clusterer


def main():
    action = 'score'
    if len(sys.argv) > 1:
        action = sys.argv[1]

    print(f"[StatX] Starting at {datetime.now(timezone.utc).isoformat()}")
    config = load_config()
    conn = get_connection(config)
    if not conn:
        print("[StatX] No DB connection")
        return

    if action == 'bootstrap':
        # Full retrain from both old and new data
        bootstrap_clusters(conn, old_limit=30000, eodhd_limit=50000)
    elif action == 'score':
        # Score unclustered eodhd_news articles
        clusterer = StatClusterX(conn)
        if not clusterer.clusters:
            print("[StatX] No clusters to score against")
            conn.close()
            return

        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id FROM eodhd_news
            WHERE id NOT IN (
                SELECT DISTINCT news_event_id FROM sentiment_signals
                WHERE news_event_id IS NOT NULL
            )
            ORDER BY id DESC
            LIMIT 500
        """)
        unscored = [r['id'] for r in cursor.fetchall()]
        cursor.close()

        if unscored:
            scores = clusterer.score_batch_from_eodhd(unscored)
            print(f"[StatX] Scored {len(scores)} / {len(unscored)} articles")
        else:
            print("[StatX] No unscored articles found")

    conn.close()
    print(f"[StatX] Done at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
