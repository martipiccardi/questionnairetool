import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from queries import (
    get_distinct_values, run_query, run_query_all,
    run_query_semantic, run_query_all_semantic,
    get_wave_rows, get_waves_for_question, get_waves_in_period,
    _parse_period, _wave_sort_key, df_to_rows, _DROP_COLUMNS,
)
from semantic_search import _expand_query

app = FastAPI(title="ENES Question Bank API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warmup():
    # Kept for compatibility but no longer called directly.
    pass


def _warmup_vola():
    """Pre-build Volume A sheet map, text-match cache, and HTML cache (runs in background thread).

    HTML prerender strategy:
    - If the disk HTML cache already exists (previous run), load all sheets from disk into
      memory — this is fast (gzip reads) and doesn't starve user request threads.
    - If the disk cache is empty (first deployment), skip the prerender entirely so the
      background thread doesn't hog the GIL while openpyxl parses large xlsx files.
      The disk cache will be built on-demand as users open Vol A pages.
    """
    import os
    try:
        from vol_a import (
            get_wave_sheet_map, _batch_load_match_for_file,
            prerender_all_sheets, _HTML_CACHE_DIR,
        )
        m = get_wave_sheet_map()
        print(f"[vol_a] Sheet map ready: {len(m)} waves — pre-warming match cache…", flush=True)
        total_files = 0
        for file_sheets in m.values():
            for fpath, sheets in file_sheets.items():
                try:
                    _batch_load_match_for_file(fpath, sheets)
                    total_files += 1
                except Exception:
                    pass
        print(f"[vol_a] Match cache ready: {total_files} files scanned", flush=True)

        # Only prerender if the disk cache already has entries — on first deployment
        # the disk cache is empty and prerender would block user requests (GIL + CPU).
        disk_cache_populated = (
            os.path.isdir(_HTML_CACHE_DIR)
            and any(True for _ in __import__('os').walk(_HTML_CACHE_DIR)
                    if _[2])  # any file exists
        )
        if disk_cache_populated:
            print("[vol_a] Disk HTML cache found — loading into memory…", flush=True)
            rendered, disk_hits, skipped = prerender_all_sheets()
            print(
                f"[vol_a] HTML cache ready: {rendered} rendered from Excel, "
                f"{disk_hits} loaded from disk, {skipped} skipped",
                flush=True,
            )
        else:
            print(
                "[vol_a] No disk HTML cache yet — skipping prerender "
                "(cache will build on-demand as pages are opened).",
                flush=True,
            )
    except Exception as e:
        print(f"[vol_a] Warmup failed: {e}", flush=True)


def _warmup_semantic():
    """Pre-load the embedding index from disk and pre-warm the encoding pipeline."""
    import semantic_search as _ss
    from semantic_search import _load_term_cache, _build_index, _encode_texts, _get_full_vocab, HF_API_TOKEN
    from data_store import get_conn
    try:
        # Reset failed flag so startup always gets a fresh attempt
        _ss._INDEX_BUILD_FAILED = False
        _load_term_cache()
        _build_index(get_conn)
        # Pre-warm the encoding pipeline (tests HF API or loads local model)
        _encode_texts(["warmup"])
        # Pre-build vocab matrix used by get_related_terms (encodes ~80 supplementary terms)
        _get_full_vocab()
        print("[semantic] Index and model ready.", flush=True)
    except Exception as e:
        print(f"[semantic] Warmup error: {type(e).__name__}: {e}", flush=True)


@app.on_event("startup")
async def startup_event():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _warmup_semantic)
    loop.run_in_executor(None, _warmup_vola)


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    semantic: bool = True
    wave: str = ""
    question_number: str = ""
    period_from: str = ""
    period_to: str = ""
    text_contains: str = ""
    search_scope: str = "both"
    sem_filter: str = ""
    page: int = 1
    per_page: int = 100


class DownloadRequest(BaseModel):
    query: str = ""
    semantic: bool = True
    wave: str = ""
    question_number: str = ""
    period_from: str = ""
    period_to: str = ""
    text_contains: str = ""
    search_scope: str = "both"
    sem_filter: str = ""
    fmt: str = "csv"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _build_search_state(req):
    filters = {}
    if req.wave:
        filters["Wave"] = req.wave
    if req.question_number:
        filters["Question Number"] = req.question_number

    text_contains = req.text_contains.strip() if req.text_contains else ""

    parsed_from = _parse_period(req.period_from)
    parsed_to = _parse_period(req.period_to)
    date_range = (parsed_from, parsed_to) if (parsed_from or parsed_to) else None

    has_text = bool(text_contains)
    sem_row_ids = []
    sem_score_map = {}
    semantic_count = 0
    related_terms = []

    if req.semantic and has_text:
        from semantic_search import semantic_search, get_related_terms

        try:
            sem_row_ids, sem_score_map = semantic_search(text_contains)
        except Exception as e:
            import traceback
            print(f"[search] semantic_search error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sem_row_ids, sem_score_map = [], {}
        semantic_count = len(sem_row_ids)

        try:
            terms = get_related_terms(text_contains, search_col=req.search_scope)
            related_terms = [{"term": t, "score": s, "count": c} for t, s, c in terms]
        except Exception as e:
            print(f"[search] get_related_terms error: {type(e).__name__}: {e}", flush=True)
            related_terms = []

    sem_filter = req.sem_filter.strip() if req.sem_filter else None

    return (
        filters, text_contains, date_range,
        sem_row_ids, sem_score_map, semantic_count,
        related_terms, sem_filter, has_text,
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/api/model-ready")
def model_ready():
    from semantic_search import is_model_ready
    return {"ready": is_model_ready()}


@app.get("/api/distinct-values/{column}")
def distinct_values(column: str):
    values = get_distinct_values(column)
    if column == "Wave":
        values = sorted(values, key=_wave_sort_key, reverse=True)
    return {"values": values}


@app.post("/api/search")
def search(req: SearchRequest):
    (filters, text_contains, date_range,
     sem_row_ids, sem_score_map, semantic_count,
     related_terms, sem_filter, has_text) = _build_search_state(req)

    offset = (req.page - 1) * req.per_page

    if req.semantic and has_text and sem_row_ids:
        must_contain = [text_contains] + [rt["term"] for rt in related_terms]
        total, df = run_query_semantic(
            filters, sem_row_ids, sem_score_map,
            req.per_page, offset, date_range,
            text_filter=sem_filter,
            text_contains=text_contains if req.search_scope != "both" else None,
            search_scope=req.search_scope,
            must_contain_terms=must_contain,
        )
    else:
        total, df = run_query(filters, text_contains, req.search_scope, req.per_page, offset, date_range)

    rows = df_to_rows(df)

    waves_in_period = []
    if date_range:
        waves_in_period = get_waves_in_period(date_range, filters)

    if req.semantic and has_text:
        if len(text_contains.strip().split()) > 1:
            # Multi-word phrase: the phrase itself is already green-highlighted via
            # qExact; individual tokens like "of" would highlight noise everywhere.
            expanded_query_terms = []
        else:
            expanded_query_terms = _expand_query(text_contains).lower().split()
    else:
        expanded_query_terms = []

    return {
        "rows": rows,
        "total": total,
        "page": req.page,
        "per_page": req.per_page,
        "semantic_count": semantic_count,
        "related_terms": related_terms,
        "waves_in_period": waves_in_period,
        "expanded_query_terms": expanded_query_terms,
    }


@app.post("/api/download")
def download(req: DownloadRequest):
    (filters, text_contains, date_range,
     sem_row_ids, sem_score_map, _sc,
     _rt, sem_filter, has_text) = _build_search_state(req)

    if req.semantic and has_text:
        must_contain = [text_contains] + [rt["term"] for rt in _rt]
        df = run_query_all_semantic(
            filters, sem_row_ids, sem_score_map, date_range,
            text_filter=sem_filter,
            text_contains=text_contains if req.search_scope != "both" else None,
            search_scope=req.search_scope,
            must_contain_terms=must_contain,
        )
    else:
        df = run_query_all(filters, text_contains, req.search_scope, date_range)

    df = df.drop(columns=[c for c in _DROP_COLUMNS if c in df.columns])

    if req.fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=question_bank_results.xlsx"},
        )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=question_bank_results.csv"},
    )


@app.get("/api/volume-a")
def volume_a(wave: str = Query(...), question: str = Query(...)):
    from fastapi.responses import HTMLResponse
    from vol_a import render_sheet_as_html
    html = render_sheet_as_html(wave, question)
    return HTMLResponse(content=html)


@app.get("/api/vol-a-status")
def vol_a_status():
    """Return current Vol A coverage without rebuilding the cache."""
    import os as _os
    from vol_a import get_wave_sheet_map
    file_map = get_wave_sheet_map()
    detail = {
        wave: [_os.path.basename(fp) for fp in files]
        for wave, files in file_map.items()
    }
    from vol_a import _GATE_VERSION
    return {
        "gate_version": _GATE_VERSION,
        "waves": sorted(file_map.keys()),
        "total_waves": len(file_map),
        "total_files": sum(len(v) for v in file_map.values()),
        "detail": detail,
    }


@app.post("/api/reload-vol-a")
def reload_vol_a():
    """Force-rebuild the Volume A file map (call after uploading new files via Kudu)."""
    from vol_a import reload_wave_file_map
    file_map = reload_wave_file_map()
    return {"status": "ok", "waves": sorted(file_map.keys()), "total_files": sum(len(v) for v in file_map.values())}


@app.get("/api/vol-a-coverage")
def vol_a_coverage():
    """Check how many DB wave/question rows have a Vol A match."""
    from vol_a import get_wave_sheet_map, _find_sheets_for_question
    from data_store import get_conn
    from queries import ensure_table

    con = get_conn()
    try:
        ensure_table(con)
        rows = con.execute(
            'SELECT DISTINCT "Wave", "Question Number" FROM enes WHERE "Wave" IS NOT NULL AND "Question Number" IS NOT NULL'
        ).fetchall()
    finally:
        con.close()

    wave_sheet_map = get_wave_sheet_map()
    total = len(rows)
    no_file = 0
    matched = 0
    unmatched = []

    for wave, question in rows:
        wave_str = str(wave).strip()
        q_str = str(question).strip()
        from vol_a import _normalize_wave
        key = _normalize_wave(wave_str)
        if key not in wave_sheet_map:
            no_file += 1
            continue
        hits = _find_sheets_for_question(wave_str, q_str)
        if hits:
            matched += 1
        else:
            unmatched.append({"wave": wave_str, "question": q_str})

    with_file = total - no_file
    return {
        "total_rows": total,
        "no_vol_a_file": no_file,
        "with_vol_a_file": with_file,
        "matched": matched,
        "unmatched_count": len(unmatched),
        "unmatched_pct": round(100 * len(unmatched) / with_file, 1) if with_file else 0,
        "unmatched": unmatched,
    }


@app.get("/api/waves/{wave:path}")
def wave_rows(wave: str):
    df = get_wave_rows(wave)
    rows = df_to_rows(df)
    return {"rows": rows, "total": len(rows)}


@app.get("/api/waves-for-question")
def waves_for_question(q: str = Query(...), mnemo: str = Query("")):
    waves = get_waves_for_question(q, mnemo)
    return {"waves": waves}


# ─────────────────────────────────────────────
# Serve React build (production)
# ─────────────────────────────────────────────

_FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend", "dist"
)

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
