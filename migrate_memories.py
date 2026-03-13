import json
import os
import sys
import time
from pathlib import Path

import requests


MEM0_API_BASE_URL = "https://api.mem0.ai"


def get_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing environment variable: {name}")
        sys.exit(1)
    return val


def mem0_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def upload_memory(api_key: str, user_id: str, memory_text: str, metadata: dict | None = None):
    url = f"{MEM0_API_BASE_URL}/v1/memories/"
    payload = {
        "user_id": user_id,
        "messages": [{"role": "user", "content": memory_text}],
        "metadata": metadata or {},
        "version": "v2",
        "output_format": "v1.1",
        "async_mode": True,
        # These local JSON entries are already "memories"; store as-is.
        "infer": False,
    }
    resp = requests.post(url, headers=mem0_headers(api_key), json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    api_key = get_env("MEM0_API_KEY")

    base_dir = Path(__file__).resolve().parent
    src = base_dir / "tyler_memories.json"
    if not src.exists():
        print(f"Could not find {src}")
        sys.exit(1)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    users = data.get("users", {})
    if not isinstance(users, dict) or not users:
        print("No users/memories found in tyler_memories.json")
        return

    total_uploaded = 0
    for user_id, memories in users.items():
        if not isinstance(memories, list):
            continue
        print(f"Uploading {len(memories)} memories for user_id='{user_id}' ...")

        for idx, item in enumerate(memories, start=1):
            if isinstance(item, str):
                memory_text = item
                metadata = {}
            elif isinstance(item, dict):
                memory_text = item.get("memory") or item.get("data") or ""
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                created_at = item.get("created_at") or item.get("createdAt")
                if created_at:
                    metadata = dict(metadata)
                    metadata.setdefault("created_at", created_at)
            else:
                continue

            memory_text = (memory_text or "").strip()
            if not memory_text:
                continue

            try:
                upload_memory(api_key, user_id, memory_text, metadata=metadata)
                total_uploaded += 1
                if idx % 25 == 0:
                    print(f"  Uploaded {idx}/{len(memories)}...")
                # Be gentle to the API
                time.sleep(0.1)
            except Exception as e:
                print(f"  Failed to upload memory {idx}/{len(memories)}: {e}")

    print(f"Done. Uploaded {total_uploaded} memories to Mem0 Cloud.")


if __name__ == "__main__":
    main()

