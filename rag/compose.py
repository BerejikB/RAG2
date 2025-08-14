from typing import Dict, List, Optional, Tuple
from .index import retrieve, load_index, RAGIndex


def compose_context(
    query: str,
    *,
    index: Optional[RAGIndex] = None,
    index_path: Optional[str] = None,
    k: int = 5,
    max_total_chars: int = 12000,
    snippet_sep: str = "\n\n-----\n\n",
    header: Optional[str] = None,
    footer: Optional[str] = None,
) -> Tuple[str, List[Dict[str, object]]]:
    """
    Retrieve and compose a bounded-size context string suitable for passing to an LLM.

    - Enforces a hard cap (max_total_chars) to prevent context overflow.
    - Returns (context_string, results_metadata)
    """
    idx = index if index is not None else load_index(index_path)
    results = retrieve(query, index=idx, k=k)

    parts: List[str] = []
    total = 0

    if header:
        parts.append(header)
        total += len(header)

    for i, r in enumerate(results, start=1):
        citation = f"[{i}] {r['path']}:{r['start_line']}-{r['end_line']} (score={r['score']:.3f})"
        snippet = r.get("text", "")
        block = f"{citation}\n{snippet}"
        extra = (len(snippet_sep) if parts else 0) + len(block)
        if total + extra > max_total_chars:
            # Try to truncate the snippet to fit remaining budget
            remaining = max_total_chars - total - (len(snippet_sep) if parts else 0)
            if remaining <= 0:
                break
            if remaining < len(citation) + 1:
                break
            # allocate remaining to snippet
            take = remaining - len(citation) - 1
            block = f"{citation}\n{snippet[:max(0, take)]}\n... [truncated]"
            parts.append(snippet_sep + block if parts else block)
            total = max_total_chars
            break
        parts.append(snippet_sep + block if parts else block)
        total += extra

    if footer and total < max_total_chars:
        sep = snippet_sep if parts else ""
        need = max_total_chars - total - len(sep)
        if need > 0:
            parts.append(sep + footer[:need])

    return "".join(parts), results


def compose_ultra_compact(
    query: str,
    *,
    index: Optional[RAGIndex] = None,
    index_path: Optional[str] = None,
    k: int = 1,
    snippet_max_chars: int = 500,
    max_total_chars: int = 1200,
    snippet_sep: str = "\n\n-----\n\n",
    header: Optional[str] = None,
    footer: Optional[str] = None,
) -> Tuple[str, List[Dict[str, object]]]:
    """
    Convenience helper for the ultra-compact mode:
    - At most one snippet (k=1 by default)
    - Snippet trimmed to <= snippet_max_chars (default 500 chars)
    - Total context capped to ~max_total_chars (default 1200)
    """
    idx = index if index is not None else load_index(index_path)
    results = retrieve(query, index=idx, k=k, max_chars=snippet_max_chars)

    parts: List[str] = []
    total = 0

    if header:
        parts.append(header)
        total += len(header)

    for i, r in enumerate(results, start=1):
        citation = f"[{i}] {r['path']}:{r['start_line']}-{r['end_line']} (score={r['score']:.3f})"
        snippet = r.get("text", "")
        block = f"{citation}\n{snippet}"
        extra = (len(snippet_sep) if parts else 0) + len(block)
        if total + extra > max_total_chars:
            remaining = max_total_chars - total - (len(snippet_sep) if parts else 0)
            if remaining <= 0:
                break
            if remaining < len(citation) + 1:
                break
            take = remaining - len(citation) - 1
            block = f"{citation}\n{snippet[:max(0, take)]}\n... [truncated]"
            parts.append(snippet_sep + block if parts else block)
            total = max_total_chars
            break
        parts.append(snippet_sep + block if parts else block)
        total += extra

    if footer and total < max_total_chars:
        sep = snippet_sep if parts else ""
        need = max_total_chars - total - len(sep)
        if need > 0:
            parts.append(sep + footer[:need])

    return "".join(parts), results
