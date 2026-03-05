#!/bin/bash
export DATA_DIR=/home/site/wwwroot/data
export XLSX_PATH=/home/site/wwwroot/data/SUPERDATASETCLEANED.xlsx
# HTML cache lives at /home (outside wwwroot) so it survives deployments
export VOL_A_HTML_CACHE_DIR=/home/vol_a_html_cache

# Persistent virtualenv at /home (survives restarts)
VENV=/home/site/venv
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating persistent virtualenv..."
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install core deps (always needed)
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing core dependencies..."
    pip install --no-cache-dir pandas==2.2.2 openpyxl==3.1.5 xlrd==2.0.1 duckdb==1.0.0 pyarrow==17.0.0 fastapi==0.115.0 "uvicorn[standard]==0.30.6" numpy
fi

# Only install torch + sentence-transformers if NOT using HF API
# (HF_API_TOKEN means we call the remote API, no local model needed)
if [ -z "$HF_API_TOKEN" ] && ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "Installing sentence-transformers (local model, may take ~7 min)..."
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    pip install --no-cache-dir sentence-transformers
fi

# xlrd needed for .xls Volume A files — install if missing
if ! python -c "import xlrd" 2>/dev/null; then
    echo "Installing xlrd for .xls support..."
    pip install --no-cache-dir xlrd==2.0.1
fi

# uvicorn needed to serve — install if binary is missing
if [ ! -f "$VENV/bin/uvicorn" ]; then
    echo "Installing uvicorn..."
    pip install --no-cache-dir "uvicorn[standard]==0.30.6" fastapi==0.115.0
fi

cd /home/site/wwwroot
exec "$VENV/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000
