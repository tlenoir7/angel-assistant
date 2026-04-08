#!/usr/bin/env python3
"""
Remove the last N appended local memories (tyler_memories.json).

Railway (web volume): run from app root, default path /app/data/tyler_memories.json

  cd /app && python scripts/delete_last_n_memories.py 20

Env:
  ANGEL_MEMORY_PATH — default /app/data/tyler_memories.json
  ANGEL_USER_ID     — user bucket to trim (default railway-user; use tyler if that is your bucket)
"""
from __future__ import annotations

import os
import sys

# Repo root = parent of scripts/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import angel  # noqa: E402


def main() -> int:
    n = 20
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python scripts/delete_last_n_memories.py [N]", file=sys.stderr)
            return 2
    uid = (os.environ.get("ANGEL_USER_ID") or "railway-user").strip() or "railway-user"
    path = os.environ.get("ANGEL_MEMORY_PATH") or str(angel.LOCAL_MEMORY_FILE)
    print(f"memory_file={path!r} user_id={uid!r} delete_last={n}", flush=True)
    r = angel.delete_last_n_local_memories(uid, n)
    print(r, flush=True)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
