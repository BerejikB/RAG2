# Goose RAG (lightweight)

A minimal Retrieval-Augmented Generation helper to avoid context overflow by only supplying the most relevant chunks from large files.

- Default BM25 lexical retrieval (no embeddings required)
- Works standalone via CLI or library
- Stores index under `KMGR/scratch/rag_index.pkl` by default for easy KMGR use

## Install (editable)

```bash
pip install -e ./RAG
```

## CLI

Build an index over the current repo (skips huge/binary files):

```bash
goose-rag build .
```

Query:

```bash
goose-rag query "how do we start the filestream server?"
```

JSONL output (for programmatic use):

```bash
goose-rag query --jsonl "KMGR bootstrap protocol"
```

Options:
- `--include-exts` e.g. `--include-exts .py .md .txt`
- `--ignore-dirs` to add more directories
- `--chunk-lines` and `--chunk-overlap` control chunking granularity

## Library usage

```python
from rag.index import build_index, retrieve

# build once
index_path = build_index(["."], index_path="KMGR/scratch/rag_index.pkl")

# query many times
hits = retrieve("context overflow protocol", index_path=index_path, k=5)
for h in hits:
    print(h["path"], h["start_line"], h["end_line"]) 
    print(h["text"]) 
```

## Notes
- This BM25 approach is often strong for code/config/doc searches.
- Later we can plug in embeddings if needed, but this keeps dependencies light and local.
- Max file size default is 50MB; binary files are skipped by a quick null-byte heuristic.

## KMGR integration idea
- KMGR recipes can call `goose-rag query` to pull only relevant snippets before assembling prompts.
- Index file lives under `KMGR/scratch` so it is easy to cache/clean with KMGR utilities.
