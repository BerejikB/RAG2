import argparse
import json
from pathlib import Path
from .index import build_index, load_index, retrieve, DEFAULT_INDEX_PATH


def cmd_build(args):
    out = build_index(
        paths=args.paths,
        index_path=args.index,
        chunk_lines=args.chunk_lines,
        chunk_overlap=args.chunk_overlap,
        min_token_len=args.min_token_len,
        include_exts=args.include_exts,
        ignore_dirs=args.ignore_dirs,
        max_bytes=args.max_bytes,
    )
    print(out)


def cmd_query(args):
    idx = None
    if args.index:
        idx = load_index(args.index)
    results = retrieve(
        args.query,
        index=idx,
        index_path=args.index if not idx else None,
        k=args.k,
        max_chars=args.max_chars,
    )
    if args.jsonl:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['path']}:{r['start_line']}-{r['end_line']} score={r['score']:.4f}")
            print(r["text"]) 
            print("-" * 80)


def main():
    p = argparse.ArgumentParser(description="Lightweight RAG index/query tool (BM25)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build index")
    pb.add_argument("paths", nargs="+", help="Files or directories to index")
    pb.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Path to save index")
    pb.add_argument("--chunk-lines", type=int, default=200)
    pb.add_argument("--chunk-overlap", type=int, default=40)
    pb.add_argument("--min-token-len", type=int, default=1)
    pb.add_argument("--include-exts", nargs="*", default=None)
    pb.add_argument("--ignore-dirs", nargs="*", default=None)
    pb.add_argument("--max-bytes", type=int, default=50*1024*1024)
    pb.set_defaults(func=cmd_build)

    pq = sub.add_parser("query", help="Query index")
    pq.add_argument("query", help="Search query")
    pq.add_argument("--index", default=DEFAULT_INDEX_PATH)
    pq.add_argument("--k", type=int, default=5)
    pq.add_argument("--max-chars", type=int, default=8000)
    pq.add_argument("--jsonl", action="store_true")
    pq.set_defaults(func=cmd_query)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
