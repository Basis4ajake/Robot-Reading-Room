from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LibraryCreateRequest(BaseModel):
    library_id: str
    name: str
    description: str = ""
    # chunk_size <= 0 used to be accepted and silently disabled sub-splitting
    # (ParagraphChunker._split_to_size treats it as "unset") rather than
    # erroring - the GUI's min="1" caught this for GUI users, but a direct
    # API call had no such guard. Reject at the boundary instead, matching
    # the GUI form's own min attributes exactly.
    chunk_size: int = Field(default=300, gt=0)
    chunk_overlap: int = Field(default=60, ge=0)
    top_k: int = Field(default=8, gt=0)
    llm_model: str = "qwen2:1.5b"
    embedding_model: Optional[str] = "nomic-embed-text"
    enable_recipe_extraction: bool = False


class LibraryUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, gt=0)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)
    top_k: Optional[int] = Field(default=None, gt=0)
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    enable_recipe_extraction: Optional[bool] = None


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
    enable_recipe_extraction: bool


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
    top_k: Optional[int] = Field(default=None, gt=0)


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
