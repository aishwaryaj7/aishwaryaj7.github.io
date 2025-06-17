"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "RAG Chatbot"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = "gpt-4-1106-preview"
    openai_temperature: float = 0.1
    
    # Weaviate Configuration
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    weaviate_index_name: str = "MultiModalDocuments"
    
    # Pinecone Configuration (optional fallback)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "gcp-starter"
    pinecone_index_name: str = "rag-chatbot"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    
    # Data Directories (centralized)
    data_dir: Path = Path("../../../data/datasets")
    upload_dir: Path = Path("../../../data/datasets/uploads")
    processed_dir: Path = Path("../../../data/datasets/processed")
    cache_dir: Path = Path("../../../data/datasets/cache")
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Monitoring
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Security
    allowed_file_types: list[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    )
    max_query_length: int = 1000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.upload_dir, self.processed_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings() 