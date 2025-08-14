"""
RAG (Retrieval-Augmented Generation) lightweight module.

Goals:
- Prevent context overflow by retrieving only the most relevant chunks from large files.
- Work standalone via CLI or as a library.
- Integrate smoothly with KMGR (uses KMGR/scratch for index storage by default).

Backend:
- Default: BM25 lexical retrieval (no heavy deps), designed for local/offline use.
- Optional: Plug-in interface for other embedding/vector backends later.
"""

from .index import (
    build_index,
    load_index,
    retrieve,
    DEFAULT_INDEX_PATH,
)
from .compose import compose_context, compose_ultra_compact

__all__ = [
    "build_index",
    "load_index",
    "retrieve",
    "compose_context",
    "compose_ultra_compact",
    "DEFAULT_INDEX_PATH",
]
