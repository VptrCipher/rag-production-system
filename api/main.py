"""
FastAPI application — entry point.

Configures:
- CORS middleware
- Structured logging
- Health check
- Startup / shutdown events
"""

from __future__ import annotations

import os
import sys
import nest_asyncio
from contextlib import asynccontextmanager

nest_asyncio.apply()

# Add parent directory to sys.path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.getenv("PHOENIX_ENABLE") == "true":
    try:
        from scripts.phoenix_setup import setup_phoenix
        setup_phoenix()
    except ImportError:
        print("Phoenix setup failed: dependencies missing. Run pip install arize-phoenix")

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import get_settings

logger = structlog.get_logger(__name__)


import time
from fastapi import Request
from fastapi.responses import JSONResponse

# Simple in-memory rate limiter
class RateLimitMiddleware:
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_history = {} # {ip: [timestamps]}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract client IP
        client_ip = scope.get("client", ["unknown"])[0]
        now = time.time()
        
        # Clean up old timestamps
        history = [ts for ts in self.request_history.get(client_ip, []) if now - ts < self.window_seconds]
        
        if len(history) >= self.max_requests:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."}
            )
            await response(scope, receive, send)
            return

        history.append(now)
        self.request_history[client_ip] = history
        await self.app(scope, receive, send)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()
    
    # ── Startup: Validate configuration ───────────────
    critical_keys = ["openai_api_key", "qdrant_collection"]
    missing = [key for key in critical_keys if not getattr(settings, key)]
    if missing:
        logger.error("missing_required_config", missing=missing)
        # In a real production system, we might want to sys.exit(1) here
        # but for this environment we'll just log loudly.
    
    settings.configure_llama_index() # Initialize LlamaIndex models
    logger.info(
        "app_starting",
        qdrant_storage="local_qdrant_storage",
        model=settings.llm_model,
    )

    # ── Startup: verify Qdrant storage is accessible ──────────
    try:
        from config.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection in collections:
            info = client.get_collection(settings.qdrant_collection)
            points = info.points_count or 0
            logger.info("qdrant_ready", collection=settings.qdrant_collection, points=points)
        else:
            logger.warning("collection_not_found", collection=settings.qdrant_collection)
    except Exception as exc:
        logger.warning("qdrant_check_failed", error=str(exc))

    yield
    logger.info("app_shutting_down")

app = FastAPI(
    title="RAG Production System",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Apply rate limiting
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error_msg = traceback.format_exc()
    logger.error("global_error", error=str(exc), traceback=error_msg)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_type": type(exc).__name__}
    )


# ── Serve the Chat UI ────────────────────────────────────────
_static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "static"))
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/", include_in_schema=False)
async def chat_ui():
    """Serve the premium chat web UI."""
    return FileResponse(os.path.join(_static_dir, "index.html"))

@app.get("/health")
async def health():
    """Enhanced health check with dependency verification."""
    health_status = {
        "status": "healthy",
        "service": "rag-production-system",
        "dependencies": {
            "qdrant": "unknown",
            "llm_api": "unknown"
        }
    }
    
    # Check Qdrant
    try:
        from config.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        health_status["dependencies"]["qdrant"] = "online"
    except Exception:
        health_status["dependencies"]["qdrant"] = "offline"
        health_status["status"] = "degraded"

    # Check LLM (Shallow check)
    settings = get_settings()
    if settings.openai_api_key or settings.groq_api_key:
        health_status["dependencies"]["llm_api"] = "configured"
    else:
        health_status["dependencies"]["llm_api"] = "missing_key"
        health_status["status"] = "degraded"

    return health_status


if __name__ == "__main__":
    settings = get_settings()
    # In production, reload should be False. Use an env var or default to False.
    is_dev = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=is_dev,
        log_level=settings.log_level.lower(),
    )
