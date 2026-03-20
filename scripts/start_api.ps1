# Launch the RAG Production System API
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=" * 60
Write-Host "  RAG Production System — API Launcher"
Write-Host "=" * 60

# Activate virtual environment
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "`n[1/3] Activating virtual environment..."
    & $VenvActivate
} else {
    Write-Warning "Virtual environment not found at $VenvActivate"
    Write-Host "      Run: python -m venv venv && pip install -r requirements.txt"
    exit 1
}

# Check Qdrant storage exists
$QdrantPath = Join-Path $ProjectRoot "local_qdrant_storage"
if (-not (Test-Path $QdrantPath)) {
    Write-Warning "`n[!] local_qdrant_storage not found — no documents ingested yet."
    Write-Host "    Run: python scripts/ingest_documents.py --directory data/raw"
}

# Start API
Write-Host "`n[2/3] Starting FastAPI server..."
Write-Host "      URL: http://localhost:8000"
Write-Host "      Docs: http://localhost:8000/docs"
Write-Host "      Health: http://localhost:8000/health"
Write-Host "      Stats: http://localhost:8000/api/v1/stats"
Write-Host "`n[3/3] Press Ctrl+C to stop.`n"

Set-Location $ProjectRoot
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
