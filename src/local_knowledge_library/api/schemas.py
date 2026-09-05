from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LibraryCreateRequest(BaseModel):
    library_id: str
    name: str
    description: str = ""
    chunk_size: int = 300
    chunk_overlap: int = 60
    top_k: int = 8
    llm_model: str = "qwen2:1.5b"
    embedding_model: Optional[str] = "nomic-embed-text"


class LibraryUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None


class LibraryResponse(BaseModel):
    library_id: str
    name: str
    description: str
    source_count: int
    document_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_model: str
    embedding_model: Optional[str]


class SourceAddRequest(BaseModel):
    source_path: str


class SourceResponse(BaseModel):
    source_id: str
    source_path: str
    filename: str
    file_type: str


class IngestResponse(BaseModel):
    processed: List[str]
    skipped: List[str]
    removed: List[str]


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    query: str
    plan: str
    answer: str
    citations: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    name: Optional[str]
    size: Optional[int]
    quantization: Optional[str]


class ModelsResponse(BaseModel):
    available: bool
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    data_dir: str
    ollama_available: bool
