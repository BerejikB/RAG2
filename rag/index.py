import os
# NOTE: This module avoids reading files at query-time by caching truncated snippets in the index.

import re
import io
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# Default location: use KMGR/scratch so KMGR can find/manage it easily
DEFAULT_INDEX_PATH = str(Path("KMGR/scratch/rag_index.pkl"))

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# BM25 parameters
K1 = 1.5
B = 0.75


@dataclass
class ChunkRef:
    doc_id: str
    path: str
    start_line: int
    end_line: int


class RAGIndex:
    def __init__(self):
        # term -> list[(doc_id, tf)]
        self.postings: Dict[str, List[Tuple[str, int]]] = {}
        # term -> document frequency
        self.df: Dict[str, int] = {}
        # doc_id -> doc length (tokens)
        self.doc_len: Dict[str, int] = {}
        # doc_id -> ChunkRef
        self.docs: Dict[str, ChunkRef] = {}
        # Optional in-index snippet cache to avoid reading files at query time
        # doc_id -> snippet (truncated)
        self.snippets: Dict[str, str] = {}
        # global stats
        self.N: int = 0
        self.avgdl: float = 0.0
        # parameters
        self.params: Dict[str, object] = {
            "chunk_lines": 200,
            "chunk_overlap": 40,
            "min_token_len": 1,
            "max_bytes": 50 * 1024 * 1024,  # 50 MB per file cap
            "include_exts": [
                ".py",
                ".txt",
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".sh",
                ".ps1",
                ".bat",
                ".mjs",
                ".js",
                ".ts",
                ".ini",
                ".cfg",
                ".csv",
                ".sql",
                ".log",
            ],
            "ignore_dirs": [
                ".git",
                "__pycache__",
                "node_modules",
                "dist",
                "build",
                "venv",
                ".venv",
                ".mypy_cache",
                ".pytest_cache",
                "KMGR/scratch",  # avoid indexing the index output
            ],
            # Max characters to cache per chunk for retrieval (controls index size)
            "snippet_max_chars": 2000,
        }

    def to_bytes(self) -> bytes:
        payload = {
            "postings": self.postings,
            "df": self.df,
            "doc_len": self.doc_len,
            "docs": {k: vars(v) for k, v in self.docs.items()},
            "snippets": self.snippets,
            "N": self.N,
            "avgdl": self.avgdl,
            "params": self.params,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_bytes(b: bytes) -> "RAGIndex":
        payload = pickle.loads(b)
        idx = RAGIndex()
        idx.postings = payload.get("postings", {})
        idx.df = payload.get("df", {})
        idx.doc_len = payload.get("doc_len", {})
        idx.docs = {k: ChunkRef(**v) for k, v in payload.get("docs", {}).items()}
        idx.snippets = payload.get("snippets", {})
        idx.N = payload.get("N", 0)
        idx.avgdl = payload.get("avgdl", 0.0)
        idx.params = payload.get("params", idx.params)
        # ensure new params have defaults if loading older index
        if "snippet_max_chars" not in idx.params:
            idx.params["snippet_max_chars"] = 2000
        return idx


# ---------------- Tokenization -----------------

def tokenize(text: str, min_token_len: int = 1) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    if min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= min_token_len]
    return tokens


# --------------- File Iteration ----------------

def _should_skip_dir(path: Path, ignore_dirs: List[str]) -> bool:
    ps = str(path).replace("\\", "/")
    for pat in ignore_dirs:
        if pat in ps.split("/") or ps.endswith(pat):
            return True
    return False


def _is_text_file(p: Path) -> bool:
    try:
        # Heuristic: try opening a small chunk in text mode
        with open(p, "rb") as f:
            chunk = f.read(2048)
        if b"\0" in chunk:
            return False
        return True
    except Exception:
        return False


def _iter_files(paths: List[str], include_exts: List[str], ignore_dirs: List[str], max_bytes: int) -> Iterable[Path]:
    for root in paths:
        root_path = Path(root)
        if root_path.is_file():
            if _is_text_file(root_path) and (root_path.suffix.lower() in include_exts or not include_exts):
                try:
                    if root_path.stat().st_size <= max_bytes:
                        yield root_path
                except Exception:
                    continue
            continue
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            # filter directories in-place to avoid descending into them
            dpath = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(dpath / d, ignore_dirs)]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    if p.stat().st_size > max_bytes:
                        continue
                except Exception:
                    continue
                if not _is_text_file(p):
                    continue
                if include_exts and p.suffix.lower() not in include_exts:
                    continue
                yield p


# --------------- Chunking ----------------------

def _chunk_lines(text: str, chunk_lines: int, overlap: int) -> Iterable[Tuple[int, int, str]]:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(n, start + chunk_lines)
        chunk_txt = "\n".join(lines[start:end])
        yield start + 1, end, chunk_txt
        if end == n:
            break
        start = end - overlap if overlap < end else end


# --------------- Build Index -------------------

def build_index(
    paths: List[str],
    index_path: Optional[str] = None,
    *,
    chunk_lines: int = 200,
    chunk_overlap: int = 40,
    min_token_len: int = 1,
    include_exts: Optional[List[str]] = None,
    ignore_dirs: Optional[List[str]] = None,
    max_bytes: int = 50 * 1024 * 1024,
    snippet_max_chars: Optional[int] = None,
) -> str:
    """
    Build a BM25 inverted index over the provided paths.

    Returns the path to the saved index.
    """
    idx = RAGIndex()
    idx.params["chunk_lines"] = chunk_lines
    idx.params["chunk_overlap"] = chunk_overlap
    idx.params["min_token_len"] = min_token_len
    idx.params["max_bytes"] = max_bytes
    if include_exts is not None:
        idx.params["include_exts"] = [e.lower() for e in include_exts]
    if ignore_dirs is not None:
        idx.params["ignore_dirs"] = ignore_dirs
    if snippet_max_chars is not None:
        idx.params["snippet_max_chars"] = int(snippet_max_chars)

    index_path = index_path or DEFAULT_INDEX_PATH
    out_path = Path(index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_len = 0

    files = list(_iter_files(paths, idx.params["include_exts"], idx.params["ignore_dirs"], max_bytes))

    for fpath in files:
        try:
            with io.open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        for start_line, end_line, chunk_txt in _chunk_lines(content, chunk_lines, chunk_overlap):
            tokens = tokenize(chunk_txt, min_token_len)
            if not tokens:
                continue
            doc_id = f"{fpath}:{start_line}-{end_line}"
            dl = len(tokens)
            idx.doc_len[doc_id] = dl
            total_len += dl
            idx.docs[doc_id] = ChunkRef(
                doc_id=doc_id,
                path=str(fpath),
                start_line=start_line,
                end_line=end_line,
            )
            # term frequencies for this chunk
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, cnt in tf.items():
                if term not in idx.postings:
                    idx.postings[term] = []
                idx.postings[term].append((doc_id, cnt))
            # store truncated snippet to avoid reading files in retrieval
            smc = int(idx.params.get("snippet_max_chars", 2000))
            snippet = chunk_txt if len(chunk_txt) <= smc else (chunk_txt[:smc] + "\n... [truncated]")
            idx.snippets[doc_id] = snippet

    # compute df and stats
    idx.N = len(idx.doc_len)
    if idx.N == 0:
        idx.avgdl = 0.0
    else:
        idx.avgdl = total_len / idx.N

    for term, plist in idx.postings.items():
        # df = number of unique docs for this term
        idx.df[term] = len(plist)

    # save index
    with open(out_path, "wb") as f:
        f.write(idx.to_bytes())

    return str(out_path)


# --------------- Load Index --------------------

def load_index(index_path: Optional[str] = None) -> RAGIndex:
    ipath = Path(index_path or DEFAULT_INDEX_PATH)
    with open(ipath, "rb") as f:
        return RAGIndex.from_bytes(f.read())


# --------------- Retrieval ---------------------

def _bm25_idf(N: int, df: int) -> float:
    # Using Robertson/Sparck Jones BM25 IDF variant with +1 smoothing
    return math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0


def _bm25_score(tf: int, dl: int, avgdl: float, idf: float, k1: float = K1, b: float = B) -> float:
    denom = tf + k1 * (1 - b + b * (dl / (avgdl if avgdl > 0 else 1.0)))
    return idf * (tf * (k1 + 1)) / (denom if denom > 0 else 1e-9)


def retrieve(
    query: str,
    *,
    index: Optional[RAGIndex] = None,
    index_path: Optional[str] = None,
    k: int = 5,
    max_chars: int = 8000,
) -> List[Dict[str, object]]:
    """
    Retrieve top-k chunks for the query.

    Returns a list of dicts: { path, start_line, end_line, score, text }
    Text is sourced from in-index snippet cache and truncated to max_chars per chunk.
    """
    if index is None:
        index = load_index(index_path)

    q_terms = tokenize(query, index.params.get("min_token_len", 1))
    if not q_terms:
        return []

    scores: Dict[str, float] = {}
    seen_terms = set()
    for term in q_terms:
        if term in seen_terms:
            continue
        seen_terms.add(term)
        plist = index.postings.get(term)
        if not plist:
            continue
        idf = _bm25_idf(index.N, index.df.get(term, 0))
        for doc_id, tf in plist:
            dl = index.doc_len.get(doc_id, 0)
            s = _bm25_score(tf, dl, index.avgdl, idf)
            scores[doc_id] = scores.get(doc_id, 0.0) + s

    # rank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results: List[Dict[str, object]] = []
    for doc_id, score in ranked:
        ref = index.docs[doc_id]
        snippet = index.snippets.get(doc_id, "")
        # Fallback to disk read only if cache missing (legacy indexes)
        if not snippet:
            try:
                with io.open(ref.path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
                snippet = "\n".join(lines[ref.start_line - 1 : ref.end_line])
            except Exception:
                snippet = ""
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n... [truncated]"
        results.append(
            {
                "path": ref.path,
                "start_line": ref.start_line,
                "end_line": ref.end_line,
                "score": score,
                "text": snippet,
            }
        )
    return results
