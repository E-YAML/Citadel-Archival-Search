from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router

# Create app instance
app = FastAPI(
    title="ASOIAF Self-Corrective RAG Core API",
    description="Asynchronous Backend serving the Self-Corrective RAG agent graph.",
    version="0.1.0"
)

# Enable CORS for browser-based UI calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to trusted domains in production settings
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
