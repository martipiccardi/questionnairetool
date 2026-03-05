import os
import re
import threading
from functools import lru_cache

import numpy as np

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{MODEL_NAME}/pipeline/feature-extraction"

_STOPWORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their', 'he', 'she',
    'not', 'no', 'if', 'so', 'as', 'up', 'out', 'all', 'also', 'any',
    'some', 'than', 'too', 'very', 'just', 'only', 'each', 'other', 'own',
    'about', 'after', 'before', 'between', 'both', 'few', 'more', 'most',
    'much', 'many', 'such', 'into', 'over', 'there', 'here', 'then',
    'now', 'when', 'where', 'what', 'which', 'who', 'why', 'how',
    'les', 'des', 'une', 'par', 'que', 'qui', 'dans', 'sur', 'pour',
    'avec', 'est', 'sont', 'pas', 'vous', 'nous', 'der', 'die', 'das',
    'und', 'ein', 'eine', 'ist', 'sind', 'nicht', 'von', 'mit',
    # Survey-specific generic terms (appear in almost every question, not informative)
    'read', 'show', 'screen', 'answer', 'answers', 'following', 'please',
    'know', 'don', 'think', 'say', 'agree', 'disagree', 'tend', 'rather',
    'strongly', 'totally', 'fairly', 'none', 'rotate', 'new', 'per',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'yes', 'item', 'items', 'scale', 'option', 'options',
})
# Words that must NOT appear as the last word of a multi-word related term.
# These are adjectives (not nouns) — they produce meaningless fragments like
# "european central", "matters european", "policy national".
_TERMINAL_ADJECTIVES = frozenset({
    # Geo/scope adjectives
    'central', 'national', 'federal', 'regional', 'local', 'global',
    'international', 'bilateral', 'multilateral', 'unilateral', 'transnational',
    'eastern', 'western', 'northern', 'southern',
    # Nationality adjectives (when last word they don't form a noun phrase)
    'european', 'american', 'german', 'french', 'italian', 'spanish',
    'british', 'russian', 'chinese', 'turkish', 'ukrainian',
    # Institutional/political adjectives
    'political', 'electoral', 'parliamentary', 'governmental', 'constitutional',
    'diplomatic', 'military', 'judicial', 'legislative', 'executive',
    'administrative', 'institutional', 'democratic', 'authoritarian',
    # Economic/social adjectives
    'economic', 'financial', 'fiscal', 'monetary', 'budgetary',
    'commercial', 'industrial', 'social', 'cultural', 'environmental',
    # Other common adjectives
    'public', 'private', 'civil', 'official', 'formal', 'informal',
    'collective', 'individual', 'personal', 'professional',
    'digital', 'technological', 'scientific', 'academic',
    'legal', 'illegal', 'criminal', 'domestic', 'foreign',
    'internal', 'external', 'general', 'specific', 'overall',
    'major', 'minor', 'primary', 'secondary', 'additional',
})

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
INDEX_DIR = os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR)


_model_ready = False
_model = None
_model_lock = threading.Lock()


def _load_model():
    """Load local model on first call; thread-safe via explicit lock (no lru_cache).
    Loads lazily on the first request rather than in a background thread,
    avoiding the Windows deadlock that occurs when SentenceTransformer is
    initialised from a ThreadPoolExecutor worker."""
    global _model, _model_ready
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:          # another thread finished while we waited
            return _model
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _model_ready = True
    return _model


def is_model_ready():
    """Return True once the model and index are fully loaded."""
    if HF_API_TOKEN:
        # Using HF Inference API — no local model to load; ready once index is loaded
        return _INDEX_CACHE is not None
    return _model_ready and _INDEX_CACHE is not None


def _hf_api_encode(texts):
    """Call HuggingFace Inference API to get embeddings. Returns normalized np.float32 array."""
    import requests
    resp = requests.post(
        _HF_API_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=15,
    )
    resp.raise_for_status()
    vecs = np.array(resp.json(), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


_QUERY_ALIASES = {
    'eu': 'European Union',
    'farmer': 'farmer farming agriculture agricultural',
    'farmers': 'farmers farming agriculture agricultural',
    'farming': 'farming agriculture farmer agricultural',
    'agriculture': 'agriculture farming farmer agricultural',
    'agricultural': 'agricultural agriculture farming farmer',
}


def _expand_query(text):
    """Expand known abbreviations/synonyms at the word level (case-insensitive)."""
    words = text.strip().split()
    expanded = []
    for word in words:
        alias = _QUERY_ALIASES.get(word.lower())
        if alias:
            expanded.append(alias)
        else:
            expanded.append(word)
    return ' '.join(expanded)


def _encode_texts(texts):
    """Encode a list of texts — uses HF API if token is set, local model otherwise."""
    if HF_API_TOKEN:
        return _hf_api_encode(texts)
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)


@lru_cache(maxsize=512)
def _encode_query(query_text):
    """Encode a single query string, cached permanently until restart."""
    return _encode_texts([query_text])


@lru_cache(maxsize=1)
def _load_term_cache():
    """Load pre-computed term embeddings for fast related-terms ranking."""
    path = os.path.join(INDEX_DIR, "term_embeddings.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        terms = data["terms"].tolist()
        vecs = data["embeddings"].astype(np.float32)
        return {t: vecs[i] for i, t in enumerate(terms)}
    return {}


def _encode_texts_cached(texts):
    """Encode texts using pre-computed cache, falling back to model for cache misses."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    cache = _load_term_cache()
    results = [None] * len(texts)
    uncached = []
    uncached_idx = []
    for i, t in enumerate(texts):
        if t in cache:
            results[i] = cache[t]
        else:
            uncached.append(t)
            uncached_idx.append(i)
    if uncached:
        vecs = _encode_texts(uncached)
        for j, idx in enumerate(uncached_idx):
            results[idx] = vecs[j]
    return np.vstack(results)


_INDEX_CACHE = None
_INDEX_LOCK = threading.Lock()
_INDEX_BUILD_FAILED = False  # True after a failed build — stops per-request retries


def _build_index(_conn_factory):
    """Build or load a pre-computed embedding matrix for every row in the DB.
    Thread-safe: only one rebuild runs at a time; all other callers wait."""
    global _INDEX_CACHE, _INDEX_BUILD_FAILED
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    # If a previous build attempt failed, don't retry on every request —
    # raise immediately so the caller can return empty results fast.
    if _INDEX_BUILD_FAILED:
        raise RuntimeError("Semantic index unavailable (previous build failed)")
    with _INDEX_LOCK:
        if _INDEX_CACHE is not None:  # another thread finished while we waited
            return _INDEX_CACHE
        if _INDEX_BUILD_FAILED:
            raise RuntimeError("Semantic index unavailable (previous build failed)")

        index_path = os.path.join(INDEX_DIR, "semantic_index.npz")

        # Fast path: try loading pre-computed index from disk first
        if os.path.exists(index_path):
            try:
                data = np.load(index_path)
                rids = data["row_ids"]
                embs = data["embeddings"].astype(np.float32)
                if embs.shape[0] > 0 and embs.ndim == 2:
                    print(f"[semantic] Loaded index from file: {embs.shape[0]} rows, {embs.shape[1]} dims", flush=True)
                    result = rids, embs
                    _INDEX_CACHE = result
                    return result
                else:
                    print(f"[semantic] npz has unexpected shape: {embs.shape}", flush=True)
            except Exception as e:
                print(f"[semantic] Failed to load npz ({index_path}): {e}", flush=True)

        # Rebuild: query DuckDB and re-encode
        try:
            from data_store import get_conn, ensure_table

            con = _conn_factory()
            try:
                ensure_table(con)
                df = con.execute("""
                    SELECT rowid AS rid,
                           COALESCE(CAST("Question(s)" AS VARCHAR), '') AS q,
                           COALESCE(CAST("Answer(s)" AS VARCHAR), '') AS a
                    FROM enes
                """).fetchdf()
            finally:
                con.close()

            row_ids = df["rid"].values.astype(np.int64)
            texts = (df["q"].str.strip() + " " + df["a"].str.strip()).tolist()

            print(f"[semantic] Rebuilding index for {len(texts)} rows...", flush=True)

            if HF_API_TOKEN:
                all_vecs = []
                for i in range(0, len(texts), 64):
                    all_vecs.append(_hf_api_encode(texts[i:i + 64]))
                embeddings = np.vstack(all_vecs)
            else:
                model = _load_model()
                embeddings = model.encode(
                    texts,
                    show_progress_bar=True,
                    batch_size=128,
                    normalize_embeddings=True,
                ).astype(np.float32)

            np.savez(index_path, row_ids=row_ids, embeddings=embeddings)
            result = row_ids, embeddings
            _INDEX_CACHE = result
            return result
        except Exception:
            _INDEX_BUILD_FAILED = True
            raise


@lru_cache(maxsize=256)
def semantic_search(query_text, top_n=500, threshold=0.30):
    """Return (list_of_rowids, {rowid: score}) for rows similar to *query_text*.

    Combines semantic scoring with a keyword fallback: rows where any expanded
    query term appears literally in Question(s) or Answer(s) are always included,
    even if their semantic score falls below the threshold.
    """
    expanded = _expand_query(query_text)
    from data_store import get_conn
    row_ids, embeddings = _build_index(get_conn)

    query_vec = _encode_query(expanded)
    scores = (embeddings @ query_vec.T).flatten()

    # Semantic pass
    mask = scores >= threshold
    if mask.any():
        indices = np.where(mask)[0]
        top = np.argsort(scores[indices])[::-1][:top_n]
        selected = indices[top]
        result_ids = row_ids[selected].tolist()
        score_map = {int(row_ids[i]): float(scores[i]) for i in selected}
    else:
        result_ids = []
        score_map = {}

    # Keyword fallback: include rows where the query appears literally,
    # even if their semantic score falls below the threshold.
    # For multi-word queries use the whole phrase to avoid noise from common
    # short words (e.g. "of" in "quality of life" matching every question).
    # For single-word queries use all expanded synonym terms individually.
    try:
        from data_store import get_conn as _gc, ensure_table
        original_words = query_text.strip().lower().split()
        if len(original_words) > 1:
            phrase = query_text.strip().lower()
            conditions = (
                '(LOWER(CAST("Question(s)" AS VARCHAR)) LIKE ? '
                'OR LOWER(CAST("Answer(s)" AS VARCHAR)) LIKE ?)'
            )
            params = [f'%{phrase}%', f'%{phrase}%']
        else:
            terms = list(dict.fromkeys(expanded.lower().split()))  # dedup, preserve order
            conditions = " OR ".join(
                '(LOWER(CAST("Question(s)" AS VARCHAR)) LIKE ? OR LOWER(CAST("Answer(s)" AS VARCHAR)) LIKE ?)'
                for _ in terms
            )
            params = [p for t in terms for p in (f"%{t}%", f"%{t}%")]
        con = _gc()
        try:
            ensure_table(con)
            kw_df = con.execute(f"SELECT rowid FROM enes WHERE {conditions}", params).fetchdf()
        finally:
            con.close()
        # O(1) reverse lookup: rowid → index in embeddings array
        id_to_idx = {int(rid): i for i, rid in enumerate(row_ids)}
        in_results = set(result_ids)
        for rid in kw_df["rowid"].tolist():
            rid = int(rid)
            if rid not in in_results and rid in id_to_idx:
                result_ids.append(rid)
                score_map[rid] = float(scores[id_to_idx[rid]])
    except Exception:
        pass

    return result_ids, score_map


# ---------------------------------------------------------------------------
# Supplementary vocabulary — terms that may not appear in the dataset but
# are semantically adjacent to common survey topics.  Their embeddings are
# computed once at first call and then cached for the process lifetime.
# ---------------------------------------------------------------------------
_SUPPLEMENTARY_VOCAB = (
    # Technology / Social media
    "social media", "social networks", "online platforms", "digital media",
    "facebook", "instagram", "twitter", "youtube", "tiktok", "whatsapp",
    "artificial intelligence", "machine learning", "automation", "cybersecurity",
    "data privacy", "digital transformation", "algorithms", "big data",
    "internet of things", "blockchain", "virtual reality",
    # Climate / Environment
    "climate change", "global warming", "carbon emissions", "greenhouse gases",
    "paris agreement", "net zero", "carbon neutral",
    "renewable energy", "solar power", "wind energy", "electric vehicles",
    "biodiversity loss", "deforestation", "ocean pollution", "plastic waste",
    "extreme weather", "floods", "droughts", "wildfires",
    # Economy
    "economic crisis", "recession", "inflation", "cost of living",
    "minimum wage", "universal basic income", "wealth inequality", "tax evasion",
    "supply chain", "trade war", "austerity", "gig economy", "remote work",
    # Migration / Society
    "migration crisis", "illegal immigration", "border management",
    "social cohesion", "multiculturalism", "integration policy",
    "urbanization", "rural depopulation",
    # Health
    "public health", "universal healthcare", "mental health",
    "pandemic", "vaccination", "antibiotic resistance", "aging population",
    # Democracy / Governance
    "rule of law", "press freedom", "judicial independence",
    "disinformation", "fake news", "information warfare", "propaganda",
    "populism", "authoritarianism", "democratic backsliding",
    "civil society", "voter turnout", "electoral fraud",
    # European affairs
    "european union", "EU", "eurozone", "euro area",
    "eu enlargement", "eu budget", "euroscepticism", "eu reform",
    "eu sanctions", "cohesion policy", "structural funds", "recovery fund",
    # Quality of life / wellbeing
    "wellbeing", "well-being", "life satisfaction", "standard of living",
    "work-life balance", "living standards", "lifestyle", "happiness",
    "social welfare", "life quality",
    # Social values
    "lgbtq rights", "gender equality", "reproductive rights",
    "religious freedom", "hate speech", "social justice",
    # Security / Defence
    "military spending", "arms race", "nuclear deterrence",
    "cyberattack", "hybrid warfare", "organized crime",
    "drug trafficking", "human trafficking",
)

# Lowercase set for fast membership checks
_SUPPLEMENTARY_VOCAB_LOWER = frozenset(v.lower() for v in _SUPPLEMENTARY_VOCAB)


def _is_clean_ngram(term):
    """Return True if no word in the term is a stopword.
    Used to reject dataset n-grams that are sentence fragments
    (e.g. 'within the european') while keeping curated supplementary terms."""
    return all(w.lower() not in _STOPWORDS for w in term.split())


@lru_cache(maxsize=1)
def _get_supplementary_embeddings():
    """Encode the supplementary vocabulary once; cached for the process lifetime."""
    return _encode_texts(list(_SUPPLEMENTARY_VOCAB))  # shape (n, 768)


@lru_cache(maxsize=1)
def _get_full_vocab():
    """Build combined (dataset vocab + supplementary) term list and embedding matrix.
    Built once and cached — avoids rebuilding on every query."""
    term_cache = _load_term_cache()
    dataset_terms = list(term_cache.keys())
    dataset_vecs = (
        np.array(list(term_cache.values()), dtype=np.float32)
        if dataset_terms else np.empty((0, 768), dtype=np.float32)
    )
    try:
        supp_terms = list(_SUPPLEMENTARY_VOCAB)
        supp_vecs = _get_supplementary_embeddings()
        all_terms = dataset_terms + supp_terms
        all_vecs = np.vstack([dataset_vecs, supp_vecs]) if len(dataset_vecs) > 0 else supp_vecs
    except Exception:
        all_terms = dataset_terms
        all_vecs = dataset_vecs
    return all_terms, all_vecs


def _is_meaningful_term(term):
    """Return False for terms whose first or last word is a stopword."""
    words = term.split()
    if not words:
        return False
    if len(words) == 1:
        return words[0].lower() not in _STOPWORDS
    # First and last word must not be stopwords, and last word must not be
    # a standalone adjective (e.g. "european central", "matters european")
    return (words[0].lower() not in _STOPWORDS and
            words[-1].lower() not in _STOPWORDS and
            words[-1].lower() not in _TERMINAL_ADJECTIVES)


def _count_terms_in_dataset(terms, row_ids=None):
    """Return {term: count} — rows containing each term, optionally limited to row_ids."""
    if not terms:
        return {}
    from data_store import get_conn, ensure_table
    col_expr = (
        'LOWER(CAST("Question(s)" AS VARCHAR) || \' \' || CAST("Answer(s)" AS VARCHAR))'
    )
    con = get_conn()
    try:
        ensure_table(con)
        where = ""
        if row_ids:
            id_list = ",".join(str(int(r)) for r in row_ids)
            where = f"WHERE rowid IN ({id_list})"
        cases = ", ".join(
            f"SUM(CASE WHEN {col_expr} LIKE ? THEN 1 ELSE 0 END)"
            for _ in terms
        )
        params = [f"%{t.lower()}%" for t in terms]
        row = con.execute(f"SELECT {cases} FROM enes {where}", params).fetchone()
        return {terms[i]: int(row[i] or 0) for i in range(len(terms))}
    finally:
        con.close()


@lru_cache(maxsize=256)
def get_related_terms(query_text, top_n_terms=15, search_col="both"):
    """Find terms semantically related to *query_text*.

    Searches the full dataset vocabulary (term_embeddings.npz) PLUS a built-in
    supplementary vocabulary of ~80 terms not necessarily present in the dataset.

    Returns [(term, similarity_score, dataset_count), ...] sorted by score.
    dataset_count=0 means the term is semantically related but absent from the data.
    """
    query_expanded = _expand_query(query_text)
    query_vec = _encode_query(query_expanded)          # shape (1, 768)
    # Use the ORIGINAL (unexpanded) query for the "already in query" filter so
    # that expansion synonyms (e.g. "agriculture" added for "farmer") are NOT
    # incorrectly excluded from related terms and thus from highlighting.
    query_words = set(re.findall(r'\b\w{3,}\b', query_text.lower()))

    # Pre-built combined vocabulary matrix (cached, built only once) ----------
    all_terms, all_vecs = _get_full_vocab()

    # Score all terms against the query ----------------------------------------
    sims = (all_vecs @ query_vec.T).flatten()

    # Filter: threshold + no stopword boundaries + no query-word overlap -------
    threshold = 0.28
    candidates = []
    for i, (term, sim) in enumerate(zip(all_terms, sims)):
        if float(sim) < threshold:
            continue
        if not _is_meaningful_term(term):
            continue
        # For dataset n-grams (not curated supplementary vocab):
        # - limit to max 2 words (longer n-grams are almost always sentence fragments)
        # - reject if any word is a stopword
        if term.lower() not in _SUPPLEMENTARY_VOCAB_LOWER:
            if len(term.split()) > 2:
                continue
            if len(term.split()) > 1 and not _is_clean_ngram(term):
                continue
        term_words = set(re.findall(r'\b\w{3,}\b', term.lower()))
        # For short terms like "EU" (2 chars), treat the whole term as its word set
        if not term_words:
            stripped = term.strip().lower()
            if len(stripped) >= 2 and stripped not in _STOPWORDS:
                term_words = {stripped}
            else:
                continue
        if term_words.issubset(query_words):
            continue
        # For single-word queries only: skip multi-word terms containing the query word
        # (blocks reversed n-grams like "union europe" for query "europe").
        # Not applied for multi-word queries — "social networks" must appear for "social media".
        if len(query_words) == 1 and len(term.split()) > 1 and term_words & query_words:
            continue
        # For multi-word queries: exclude noisy dataset terms.
        # Supplementary-vocab terms are always exempt (they are curated).
        if len(query_words) > 1 and term.lower() not in _SUPPLEMENTARY_VOCAB_LOWER:
            # Single-word dataset inflections ("lives", "living" for "quality of life"):
            # require a high similarity bar so only genuinely on-concept singles survive.
            if len(term.split()) == 1 and float(sim) < 0.65:
                continue
            # Dataset bigrams that just attach a word to a query word ("human life",
            # "daily life") — they add no new concept beyond what's already in the query.
            if len(term.split()) > 1 and term_words & query_words:
                continue
            # Dataset bigrams with low similarity are usually noisy fragments
            # ("marriage living", "matters people") — apply a stricter floor.
            if len(term.split()) > 1 and float(sim) < 0.50:
                continue
        candidates.append((term, float(sim)))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate exact phrases only (so "european union" isn't blocked by "european")
    deduped = []
    seen_exact = set()
    for term, sim in candidates:
        term_lower = term.lower()
        if term_lower in seen_exact:
            continue
        seen_exact.add(term_lower)
        deduped.append((term, sim))
        if len(deduped) >= top_n_terms * 3:
            break

    # Remove terms identical to a query word (e.g. "europe" when searching "europe")
    deduped = [(t, s) for t, s in deduped if t.lower() not in query_words]

    # Guarantee supplementary vocab synonyms appear even if dataset terms crowd them out.
    # Reserve up to 5 slots for supplementary terms, fill the rest with dataset terms.
    supp = [(t, s) for t, s in deduped if t.lower() in _SUPPLEMENTARY_VOCAB_LOWER]
    data = [(t, s) for t, s in deduped if t.lower() not in _SUPPLEMENTARY_VOCAB_LOWER]
    n_supp = min(len(supp), 8)
    top_candidates = supp[:n_supp] + data[:top_n_terms * 2]
    top_candidates.sort(key=lambda x: x[1], reverse=True)

    if not top_candidates:
        return []

    # Count within semantic results so the number matches what clicking shows --
    sem_row_ids, _ = semantic_search(query_text)
    candidate_terms = tuple(t for t, _ in top_candidates)
    counts = _count_terms_in_dataset(candidate_terms, sem_row_ids)

    result = [
        (term, round(sim, 2), counts.get(term, 0))
        for term, sim in top_candidates
    ]
    return result[:top_n_terms]
