#!/usr/bin/env python3
"""
Explore an AideML journal.json file, summarizing runs and highlighting buggy ones.

Default path points to selected_logs/2-crafty-mamba-of-skill/journal.json from project root.

Usage examples:
  python explore_journal.py                       # use default path and show buggy runs
  python explore_journal.py path/to/journal.json  # custom file
  python explore_journal.py --all                 # include non-buggy runs
  python explore_journal.py --limit 5             # show only first N entries per group

The script prints:
  - Summary counts (total, buggy, non-buggy)
  - Exception type histogram (for buggy runs)
  - Detailed list of buggy runs with analysis and brief error context
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PATH = Path("selected_logs/2-merry-onyx-lemming/journal.json")


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(nodes)
    buggy = [n for n in nodes if n.get("is_buggy") is True]
    ok = [n for n in nodes if not n.get("is_buggy")]
    exc_types = [n.get("exc_type") for n in buggy if n.get("exc_type")]
    exc_hist = Counter(exc_types)
    return {
        "total": total,
        "buggy": len(buggy),
        "ok": len(ok),
        "exc_hist": exc_hist,
        "buggy_nodes": buggy,
        "ok_nodes": ok,
    }


def brief_error(n: Dict[str, Any]) -> str:
    # Try exc_info.args[0], then first line of _term_out
    msg = None
    args0 = safe_get(n, "exc_info", "args")
    if isinstance(args0, (list, tuple)) and args0:
        msg = str(args0[0])
    if not msg:
        term = n.get("_term_out")
        if isinstance(term, list) and term:
            # take first line of first entry that's not empty
            for entry in term:
                if entry:
                    msg = entry.strip().splitlines()[0]
                    break
    return msg or "(no error message available)"


def print_summary(summary: Dict[str, Any]) -> None:
    print("Journal summary")
    print("---------------")
    print(f"Total runs:     {summary['total']}")
    print(f"Buggy runs:     {summary['buggy']}")
    print(f"Non-buggy runs: {summary['ok']}")
    if summary["exc_hist"]:
        print("\nException types (buggy runs):")
        for exc, cnt in summary["exc_hist"].most_common():
            print(f"  {exc}: {cnt}")
    print()


def print_nodes(nodes: List[Dict[str, Any]], *, title: str, limit: int | None = None) -> None:
    print(title)
    print("=" * len(title))
    shown = 0
    for n in nodes:
        if limit is not None and shown >= limit:
            remaining = len(nodes) - shown
            if remaining > 0:
                print(f"... ({remaining} more not shown)\n")
            break
        step = n.get("step")
        node_id = n.get("id")
        exc_type = n.get("exc_type")
        exec_time = n.get("exec_time")
        analysis = n.get("analysis")
        metric_val = safe_get(n, "metric", "value")
        print(f"- step={step} id={node_id}")
        if exc_type:
            print(f"  exc_type:   {exc_type}")
            print(f"  error:      {brief_error(n)}")
        if exec_time is not None:
            print(f"  exec_time:  {exec_time:.3f}s")
        if metric_val is not None:
            print(f"  metric:     {metric_val}")
        if analysis:
            print("  analysis:")
            for line in str(analysis).splitlines():
                print(f"    {line}")
        print()
        shown += 1


def load_journal(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError("Invalid journal format: expecting an object with a 'nodes' list")
    return data


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Explore an AideML journal.json file")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="Path to journal.json")
    parser.add_argument("--all", action="store_true", help="Also list non-buggy runs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items shown per section")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    try:
        journal = load_journal(path)
    except Exception as e:
        print(f"Failed to load journal: {e}", file=sys.stderr)
        return 1

    nodes = journal["nodes"]
    summary = summarize_nodes(nodes)
    print_summary(summary)

    # Buggy runs first
    buggy_nodes = summary["buggy_nodes"]
    if buggy_nodes:
        print_nodes(buggy_nodes, title="Buggy runs", limit=args.limit)
    else:
        print("No buggy runs found.\n")

    if args.all:
        ok_nodes = summary["ok_nodes"]
        if ok_nodes:
            print_nodes(ok_nodes, title="Non-buggy runs", limit=args.limit)
        else:
            print("No non-buggy runs found.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
