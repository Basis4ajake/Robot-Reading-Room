from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .models import LibraryConfig, LibraryMetadata
from .storage import KnowledgeLibrary


class LibraryNotFoundError(KeyError):
    pass


class LibraryRegistry:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_libraries(self) -> List[LibraryMetadata]:
        results: List[LibraryMetadata] = []
        for child in sorted(self.data_dir.iterdir()):
            meta_path = child / "meta.json"
            if child.is_dir() and meta_path.exists():
                library = KnowledgeLibrary.open_by_id(child.name, str(self.data_dir))
                results.append(library.metadata)
        return results

    def create_library(
        self,
        library_id: str,
        name: str,
        description: str = "",
        chunk_size: int = 300,
        chunk_overlap: int = 60,
        top_k: int = 8,
        llm_model: str = "qwen2:1.5b",
        embedding_model: Optional[str] = "nomic-embed-text",
        enable_recipe_extraction: bool = False,
    ) -> KnowledgeLibrary:
        config = LibraryConfig(
            library_id=library_id,
            name=name,
            description=description,
            data_dir=str(self.data_dir),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            llm_model=llm_model,
            embedding_model=embedding_model,
            enable_recipe_extraction=enable_recipe_extraction,
        )
        library = KnowledgeLibrary.create(config)
        library.persist()
        return library

    def get_library(self, library_id: str) -> KnowledgeLibrary:
        try:
            return KnowledgeLibrary.open_by_id(library_id, str(self.data_dir))
        except FileNotFoundError as exc:
            raise LibraryNotFoundError(library_id) from exc

    def update_config(self, library_id: str, **fields) -> KnowledgeLibrary:
        library = self.get_library(library_id)
        for key, value in fields.items():
            if value is not None:
                setattr(library.config, key, value)
        # name/description are duplicated on LibraryMetadata for list_libraries();
        # keep it in sync so a rename is actually visible through the API.
        library.metadata.name = library.config.name
        library.metadata.description = library.config.description
        library.persist()
        return library

    def delete_library(self, library_id: str) -> None:
        library_dir = self.data_dir / library_id
        if not library_dir.exists():
            raise LibraryNotFoundError(library_id)
        shutil.rmtree(library_dir)
