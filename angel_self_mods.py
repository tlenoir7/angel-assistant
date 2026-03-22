"""
Stage 6 — Tyler-approved self-modifications layered into Angel's system prompt.

Mutable state lives in ``angel_self_mods_data.json`` (same directory as this module).
Approved instructions are merged by ``get_self_modification_additions()``; do not remove
capabilities or safety rules here — the approval pipeline enforces that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parent / "angel_self_mods_data.json"


def _default_data() -> dict[str, Any]:
    return {"approved": [], "reverted_log": []}


def _load_data() -> dict[str, Any]:
    if not _DATA_PATH.exists():
        return _default_data()
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return _default_data()
        raw.setdefault("approved", [])
        raw.setdefault("reverted_log", [])
        return raw
    except Exception:
        return _default_data()


def _save_data(data: dict[str, Any]) -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _DATA_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(_DATA_PATH)


def get_self_modification_additions() -> str:
    """
    Additional system-prompt text from approved modifications.
    Called from ``build_system_prompt`` in ``angel.py``.
    """
    data = _load_data()
    approved = data.get("approved") or []
    if not isinstance(approved, list) or not approved:
        return ""
    lines: list[str] = []
    for item in approved:
        if not isinstance(item, dict):
            continue
        mid = (item.get("modification_id") or "").strip()
        title = (item.get("title") or "").strip()
        proc = (item.get("applied_instruction") or item.get("proposed_change") or "").strip()
        if not proc:
            continue
        head = f"[{mid}] {title}".strip() if title else f"[{mid}]"
        lines.append(f"{head}\n{proc}")
    if not lines:
        return ""
    return (
        "\n\n--- Tyler-approved self-modifications (Stage 6; permanent until reverted) ---\n"
        "These instructions were explicitly approved by Tyler. They do not override your core "
        "values, safety rules, or the requirement that Tyler approve future changes.\n\n"
        + "\n\n---\n\n".join(lines)
    )


def append_approved_modification(record: dict[str, Any]) -> None:
    """Append one approved modification to on-disk state (idempotent by modification_id)."""
    data = _load_data()
    approved = data.get("approved")
    if not isinstance(approved, list):
        approved = []
    mid = (record.get("modification_id") or "").strip()
    if not mid:
        return
    for i, ex in enumerate(approved):
        if isinstance(ex, dict) and (ex.get("modification_id") or "").strip() == mid:
            approved[i] = record
            data["approved"] = approved
            _save_data(data)
            return
    approved.append(dict(record))
    data["approved"] = approved
    _save_data(data)


def remove_approved_modification(modification_id: str) -> bool:
    """Remove an applied modification by id (revert). Returns True if something was removed."""
    data = _load_data()
    approved = data.get("approved")
    if not isinstance(approved, list):
        return False
    mid = modification_id.strip()
    new = [x for x in approved if not (isinstance(x, dict) and (x.get("modification_id") or "").strip() == mid)]
    if len(new) == len(approved):
        return False
    data["approved"] = new
    log = data.get("reverted_log")
    if not isinstance(log, list):
        log = []
    log.append({"modification_id": mid, "reverted": True})
    data["reverted_log"] = log[-200:]
    _save_data(data)
    return True


def list_applied_records() -> list[dict[str, Any]]:
    data = _load_data()
    a = data.get("approved") or []
    return [x for x in a if isinstance(x, dict)]
