import os
import json
import time
import requests
from pathlib import Path

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
MEM0_API_BASE = "https://api.mem0.ai"
USER_ID = os.environ.get("ANGEL_USER_ID", "tyler")
LOCAL_FILE = Path(os.environ.get(
    "ANGEL_MEMORY_PATH",
    "tyler_memories.json"
))


def get_all_cloud_memories():
    headers = {
        "Authorization": f"Token {MEM0_API_KEY}",
        "Content-Type": "application/json"
    }
    all_memories = []
    page = 1
    max_pages = 20
    while True:
        if page > max_pages:
            print(f"Reached max_pages={max_pages}; stopping to prevent infinite loop.")
            break
        try:
            results = fetch_page(headers, page, retries=3)

            if not results:
                break
            all_memories.extend(results)
            print(f"Page {page}: {len(results)} memories fetched, total so far: {len(all_memories)}")
            if page == 1 and len(results) > 1000:
                print("Page 1 returned >1000 records; assuming Mem0 returned everything at once.")
                break
            if len(results) < 200:
                break
            page += 1
        except Exception as e:
            print(f"Timeout/error on page {page}: {e}")
            print(f"Saving {len(all_memories)} memories fetched so far...")
            break
    return all_memories


def fetch_page(headers, page, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{MEM0_API_BASE}/v2/memories/",
                headers=headers,
                json={
                    "filters": {"user_id": USER_ID},
                    "page": page,
                    "page_size": 200
                },
                timeout=90
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("results", data.get("memories", []))
            return []
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print("  Retrying in 5 seconds...")
                time.sleep(5)
    return []


def load_local():
    if LOCAL_FILE.exists():
        with open(LOCAL_FILE) as f:
            return json.load(f)
    return []


def merge_and_save(cloud, local):
    local_dicts = [m for m in local if isinstance(m, dict)]
    local_ids = {m.get("id") for m in local_dicts if m.get("id")}
    added = 0
    for m in cloud:
        if isinstance(m, dict) and m.get("id") not in local_ids:
            local_dicts.append(m)
            added += 1
    with open(LOCAL_FILE, "w") as f:
        json.dump(local_dicts, f, indent=2)
    return added


if __name__ == "__main__":
    print(f"Fetching from Mem0 cloud for user: {USER_ID}")
    cloud = get_all_cloud_memories()
    print(f"Total cloud memories: {len(cloud)}")

    local = load_local()
    print(f"Total local memories: {len(local)}")

    added = merge_and_save(cloud, local)
    local_after = load_local()
    print(f"Added: {added} new memories")
    print(f"Total after merge: {len(local_after)}")
    print(f"Saved to: {LOCAL_FILE}")
