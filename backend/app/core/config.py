from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings validated via Pydantic.
    Reads from environment variables or a .env file.
    On Streamlit Cloud, secrets are injected into os.environ by streamlit_app.py
    before this class is instantiated.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core Project Settings
    PROJECT_NAME: str = "ASOIAF Self-Corrective RAG"

    # API Keys & Third-party integrations
    GROQ_API_KEY: str

    # Qdrant Config
    QDRANT_URL: str
    QDRANT_API_KEY: str

    # Neo4j Config
    NEO4J_URI: str
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str

    # LangChain / LangSmith Tracing Config
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "citadel-archival-search-evals"

    # Production Architecture Configuration
    # Default None → workflow.py resolves to tempfile.gettempdir() at runtime
    CHECKPOINT_DB_PATH: Optional[str] = None
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]
    RATE_LIMIT_LIMIT: int = 10
    RATE_LIMIT_PERIOD: int = 60
    LLM_TIMEOUT: float = 30.0


# Instantiate single global settings object
settings = Settings()

