from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.limiter import limiter
from app.api.chat import router as chat_router

# Create app instance
app = FastAPI(
    title="ASOIAF Self-Corrective RAG Core API",
    description="Asynchronous Backend serving the Self-Corrective RAG agent graph.",
    version="0.1.0"
)

# Set up slowapi rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for browser-based UI calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])


@app.get("/health")
def health_check() -> dict:
    """
    Service health check endpoint.
    """
    return {"status": "healthy"}
