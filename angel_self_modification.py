"""
Stage 6 — Self-modification: observe interactions, propose changes, apply with Tyler's approval.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import UTC, datetime
from typing import Any

from angel import CATEGORY_SELF_OBSERVATION, CATEGORY_SELF_MODIFICATION

SELF_MOD_FOLDER = "Self Modifications"

_FORBIDDEN_SUBSTRINGS = (
    "remove capability",
    "remove a capability",
    "disable safety",
    "ignore safety",
    "no approval",
    "without approval",
    "bypass approval",
    "core values",
    "override your values",
)

def _memory_category_from_item(m: dict) -> str | None:
    """
    Category string for a memory row — same logical field as angel.add_structured_memory uses.
    Handles Mem0 shapes: metadata dict, metadata as JSON string, or top-level category.
    """
    if not isinstance(m, dict):
        return None
    meta = m.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    c = meta.get("category")
    if c is None:
        c = meta.get("Category")
    if c is None:
        c = m.get("category")
    if c is not None and str(c).strip():
        return str(c).strip()
    return None


_STOPWORDS = frozenset(
    """
    a an the to of and or for in on at by is it as if be are was were been being
    this that these those i you we he she they them me my your our their his her
    not no yes so do does did just very can could would should about into from
    with have has had what which who when where how why than then there here
    """.split()
)


def _ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _state_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "angel_self_mod_state.json")


def set_pending_proposal_notification(message: str) -> None:
    try:
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump({"pending_intro": (message or "").strip()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def consume_pending_proposal_notification() -> str:
    p = _state_path()
    if not os.path.isfile(p):
        return ""
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        intro = (data.get("pending_intro") or "").strip()
        if intro:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"pending_intro": ""}, f)
            except Exception:
                pass
        return intro
    except Exception:
        return ""


def _topic_hints(text: str, max_terms: int = 6) -> list[str]:
    if not text:
        return []
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    out: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_terms:
            break
    return out


def _signals_from_turn(user_message: str, assistant_reply: str) -> dict[str, Any]:
    u = user_message or ""
    low = u.lower()
    a_low = (assistant_reply or "").lower()
    return {
        "user_chars": len(u),
        "assistant_chars": len(assistant_reply or ""),
        "actually_or_correction": bool(
            re.search(r"\bactually\b|correction:|you('re| are) wrong|not quite|misunderstood", low)
        ),
        "strong_agreement": bool(
            re.search(r"\b(exactly|yes\b|that'?s right|perfect|spot on|precisely|agreed)\b", low)
        ),
        "wants_shorter": bool(re.search(r"\b(shorter|too long|less detail|brief|tl;dr|concise)\b", low)),
        "wants_longer": bool(re.search(r"\b(longer|more detail|expand|elaborate|go deeper)\b", low)),
        "follow_up_why": bool(re.search(r"\b(why|how come|what about|and what|clarify)\b", low)),
        "assistant_substantive": len((assistant_reply or "").strip()) > 80,
    }


def record_turn_observation(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    user_message: str,
    assistant_reply: str,
    *,
    files_cabinet: Any | None = None,
) -> None:
    """Heuristic observation for this turn; stored as category self_observation (compact)."""
    from angel import add_structured_memory

    print(
        "[selfmod] add_structured_memory (observation) category_passed="
        f"{CATEGORY_SELF_OBSERVATION!r} — must equal angel.CATEGORY_SELF_OBSERVATION",
        flush=True,
    )
    u = (user_message or "").strip()
    if not u:
        return
    sig = _signals_from_turn(u, assistant_reply)
    obs_id = f"obs-{secrets.token_hex(6)}"
    hints = _topic_hints(u)
    payload = {
        "observation_id": obs_id,
        "recorded_at": _ts(),
        "topic_hints": hints,
        "signals": sig,
        "user_preview": u[:320],
    }
    body = json.dumps(payload, ensure_ascii=False)
    if len(body) > 1800:
        body = body[:1800] + "…"
    meta_note = f"self_observation:{obs_id}"
    text = f"[{meta_note}]\n{body}"
    try:
        add_structured_memory(
            memory_client,
            user_id,
            text,
            CATEGORY_SELF_OBSERVATION,
            person_name=None,
            use_mem0_cloud=use_mem0_cloud,
        )
    except Exception:
        pass


def record_turn_observation_background(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    user_message: str,
    assistant_reply: str,
    files_cabinet: Any | None = None,
) -> None:
    """Non-blocking observation write (daemon thread)."""

    def _run() -> None:
        try:
            record_turn_observation(
                memory_client,
                user_id,
                use_mem0_cloud,
                user_message,
                assistant_reply,
                files_cabinet=files_cabinet,
            )
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _normalize_memories(memories: Any) -> list[dict]:
    from angel import _normalize_memories_list  # type: ignore

    return _normalize_memories_list(memories)


def iter_self_modifications(memories: list) -> list[dict[str, Any]]:
    """Latest record per modification_id (Mem0 may store multiple versions)."""
    n_in = len(memories) if isinstance(memories, list) else 0
    norm = _normalize_memories(memories)
    sample_cats: list[str | None] = []
    for m in norm[:24]:
        if isinstance(m, dict):
            sample_cats.append(_memory_category_from_item(m))
        else:
            sample_cats.append(None)
    print(
        "[selfmod] iter_self_modifications: filter_category="
        f"{CATEGORY_SELF_MODIFICATION!r} (angel.CATEGORY_SELF_MODIFICATION) "
        f"sample_categories_first_rows={sample_cats!r}",
        flush=True,
    )

    by_id: dict[str, dict[str, Any]] = {}
    n_cat = 0
    n_parse_fail = 0
    for m in norm:
        if not isinstance(m, dict):
            continue
        cat = _memory_category_from_item(m)
        if cat != CATEGORY_SELF_MODIFICATION:
            continue
        n_cat += 1
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                continue
            mid = (rec.get("modification_id") or "").strip()
            if not mid:
                continue
            prev = by_id.get(mid)
            u_new = str(rec.get("updated_at") or rec.get("created_at") or "")
            u_old = str((prev or {}).get("updated_at") or (prev or {}).get("created_at") or "")
            if not prev or u_new >= u_old:
                by_id[mid] = rec
        except Exception:
            n_parse_fail += 1
            continue
    print(
        f"[selfmod] iter_self_modifications: raw_memories={n_in} "
        f"rows_matching_self_modification_category={n_cat} parse_fail={n_parse_fail} unique_ids={len(by_id)}",
        flush=True,
    )
    return list(by_id.values())


def _save_modification_memory(
    memory_client,
    user_id: str,
    use_mem0_cloud: bool,
    record: dict[str, Any],
) -> None:
    from angel import (
        CATEGORY_SELF_MODIFICATION,
        _load_local_memory_entries,
        add_structured_memory,
    )

    mid = (record.get("modification_id") or "").strip()
    title = (record.get("title") or "").strip()
    print(f"[selfmod] saving proposal: {mid!r} {title!r}", flush=True)
    print(
        "[selfmod] add_structured_memory (proposal) category_passed="
        f"{CATEGORY_SELF_MODIFICATION!r} — must equal angel.CATEGORY_SELF_MODIFICATION",
        flush=True,
    )

    def _count_self_mod_rows() -> int:
        try:
            return sum(
                1
                for e in _load_local_memory_entries(user_id)
                if isinstance(e, dict) and _memory_category_from_item(e) == CATEGORY_SELF_MODIFICATION
            )
        except Exception:
            return -1

    before = _count_self_mod_rows()
    text = json.dumps(record, ensure_ascii=False)
    ok_local = add_structured_memory(
        memory_client,
        user_id,
        text,
        CATEGORY_SELF_MODIFICATION,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )
    after = _count_self_mod_rows()
    result = {
        "local_append_ok": ok_local,
        "local_self_mod_rows_before": before,
        "local_self_mod_rows_after": after,
        "use_mem0_cloud": use_mem0_cloud,
    }
    print(f"[selfmod] save result: {result}", flush=True)


def mirror_mod_file(files_cabinet: Any, record: dict[str, Any]) -> None:
    mid = (record.get("modification_id") or "").strip()
    if not mid:
        return
    fn = f"MOD-{mid}"
    body = json.dumps(record, ensure_ascii=False, indent=2)
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(
                SELF_MOD_FOLDER,
                fn,
                body,
                tags=["self_modification", f"status:{record.get('status', '')}"],
            )
    except Exception:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass


def _safety_check_proposal(text: str) -> tuple[bool, str]:
    t = (text or "").lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in t:
            return False, f"Proposal blocked (safety): contains disallowed theme near {bad!r}."
    return True, ""


def _find_mod(records: list[dict], needle: str) -> dict[str, Any] | None:
    n = (needle or "").strip().lower()
    if not n:
        return None
    # exact id
    for r in records:
        if (r.get("modification_id") or "").lower() == n:
            return r
    # prefix / substring on id
    for r in records:
        mid = (r.get("modification_id") or "").lower()
        if len(n) >= 6 and (mid.startswith(n) or n in mid):
            return r
    # title
    for r in records:
        if n in (r.get("title") or "").lower():
            return r
    return None


def revert_self_modification(core: Any, needle: str) -> dict[str, Any]:
    """
    Revert an applied modification (same logic as chat: "revert [modification]").

    Returns ``{"ok": True, "modification_id", "message"}`` or ``{"ok": False, "error"}``.
    """
    from angel import fetch_combined_memories
    import angel_self_mods as mods

    needle = (needle or "").strip()
    if not needle:
        return {"ok": False, "error": "missing modification id or title"}

    mem = fetch_combined_memories(core.memory_client, core.user_id, core._use_mem0_cloud)
    records = iter_self_modifications(mem)

    disk = mods.list_applied_records()
    match = _find_mod(disk, needle)  # type: ignore[arg-type]
    if not match:
        match = _find_mod([r for r in records if r.get("status") == "applied"], needle)
    if not match:
        return {
            "ok": False,
            "error": f"I couldn’t find an applied modification matching {needle!r}.",
        }

    mid = (match.get("modification_id") or "").strip()
    if not mid:
        return {"ok": False, "error": "invalid modification record"}

    if not mods.remove_approved_modification(mid):
        return {"ok": False, "error": "Revert failed (record not found on disk)."}

    for r in records:
        if (r.get("modification_id") or "").strip() == mid:
            r["status"] = "reverted"
            r["updated_at"] = _ts()
            _save_modification_memory(core.memory_client, core.user_id, core._use_mem0_cloud, r)
            mirror_mod_file(core.files_cabinet, r)
            break

    msg = (
        f"Reverted `{mid}`. That instruction will no longer be layered into my system prompt."
    )
    return {"ok": True, "modification_id": mid, "message": msg}


def detect_self_mod_intent(user_message: str) -> tuple[str, str] | None:
    """Return (command, arg) if this message is a self-mod command."""
    raw = (user_message or "").strip()
    if not raw:
        return None
    low = raw.lower()

    if re.search(
        r"\b(show me |list )?(your )?proposed modifications\b|\bwhat (are |)(your )?proposed modifications\b",
        low,
    ):
        return "list_proposed", ""

    if "what have you changed about yourself" in low or "changed about yourself" in low:
        return "list_applied", ""

    if re.search(r"\bgenerate (new )?modification proposals\b|\bnew modification proposals\b", low):
        return "generate", ""

    m = re.match(
        r"^\s*approve(?:\s+modification)?\s+(.+?)\s*$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return "approve", m.group(1).strip()

    m = re.match(
        r"^\s*reject(?:\s+modification)?\s+(.+?)\s*$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return "reject", m.group(1).strip()

    m = re.match(
        r"^\s*revert(?:\s+modification)?\s+(.+?)\s*$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return "revert", m.group(1).strip()

    return None


def handle_self_mod_intent(core: Any, command: str, arg: str, user_message: str) -> str:
    from angel import fetch_combined_memories
    import angel_self_mods as mods

    mem = fetch_combined_memories(core.memory_client, core.user_id, core._use_mem0_cloud)
    records = iter_self_modifications(mem)

    if command == "list_proposed":
        props = [r for r in records if (r.get("status") or "") == "proposed"]
        if not props:
            return (
                "There are no pending self-modification proposals right now. "
                "Say “generate new modification proposals” if you want me to run an analysis pass."
            )
        lines = []
        for r in props:
            lines.append(
                f"- **{r.get('title', '')}** (`{r.get('modification_id')}`) — "
                f"{r.get('confidence', '?')} confidence. {r.get('observation', '')[:200]}"
            )
        return (
            "Here are proposed modifications (Tyler approval required before anything is applied):\n\n"
            + "\n".join(lines)
            + "\n\nThis is a suggestion only. You have full control."
        )

    if command == "list_applied":
        applied_disk = mods.list_applied_records()
        applied_mem = [r for r in records if (r.get("status") or "") == "applied"]
        if not applied_disk and not applied_mem:
            return "I haven’t applied any approved self-modifications yet—your baseline behavior is unchanged."
        lines = []
        for r in applied_disk:
            lines.append(
                f"- {r.get('title', '')} (`{r.get('modification_id')}`): "
                f"{(r.get('applied_instruction') or r.get('proposed_change') or '')[:400]}"
            )
        return "What I’ve changed about myself (Tyler-approved):\n\n" + "\n".join(lines)

    if command == "generate":
        r = run_self_modification_analysis(
            core.anthropic_client,
            core.memory_client,
            core.user_id,
            core.files_cabinet,
            core._use_mem0_cloud,
        )
        if r.get("ok"):
            return (
                f"Analysis complete. New proposals: {r.get('count', 0)}. "
                "Ask me to show proposed modifications when you want to review them."
            )
        return f"I couldn’t complete modification analysis: {r.get('error', 'unknown')}"

    if command == "approve":
        needle = arg.strip()
        rec = _find_mod([r for r in records if r.get("status") == "proposed"], needle)
        if not rec:
            return f"I couldn’t find a **proposed** modification matching {needle!r}. Try the exact id or title."
        ok, err = _safety_check_proposal(rec.get("proposed_change", "") + " " + rec.get("title", ""))
        if not ok:
            return err
        now = _ts()
        applied_instruction = (rec.get("proposed_change") or "").strip()
        rec["status"] = "applied"
        rec["updated_at"] = now
        rec["applied_at"] = now
        rec["applied_instruction"] = applied_instruction
        _save_modification_memory(core.memory_client, core.user_id, core._use_mem0_cloud, rec)
        mirror_mod_file(core.files_cabinet, rec)
        from angel_self_mods import append_approved_modification

        append_approved_modification(
            {
                "modification_id": rec.get("modification_id"),
                "title": rec.get("title"),
                "applied_instruction": applied_instruction,
                "change_type": rec.get("change_type"),
                "approved_at": now,
            }
        )
        desc = (rec.get("title") or "this change")[:200]
        return (
            f"Modification applied. Starting now I will reflect this: {desc}\n\n"
            f"Details: {applied_instruction[:600]}"
        )

    if command == "reject":
        needle = arg.strip()
        rec = _find_mod([r for r in records if r.get("status") == "proposed"], needle)
        if not rec:
            return f"I couldn’t find a **proposed** modification matching {needle!r}."
        rec["status"] = "rejected"
        rec["updated_at"] = _ts()
        _save_modification_memory(core.memory_client, core.user_id, core._use_mem0_cloud, rec)
        mirror_mod_file(core.files_cabinet, rec)
        return f"Rejected modification `{rec.get('modification_id')}` — I won’t apply it."

    if command == "revert":
        r = revert_self_modification(core, arg.strip())
        if r.get("ok"):
            return str(r.get("message") or "")
        return str(r.get("error") or "Revert failed.")

    return ""


def run_self_modification_analysis(
    anthropic_client,
    memory_client,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    max_proposals: int = 3,
) -> dict[str, Any]:
    """Analyze observations + memory context; create up to ``max_proposals`` proposals."""
    from angel import (
        CATEGORY_REFLECTION,
        CATEGORY_SELF_MODIFICATION,
        CATEGORY_SELF_OBSERVATION,
        build_memory_summary_with_sections,
        call_claude,
        fetch_combined_memories,
    )

    memories = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    filtered: list = []
    for m in _normalize_memories(memories):
        if not isinstance(m, dict):
            continue
        cat = _memory_category_from_item(m)
        if cat in (CATEGORY_SELF_OBSERVATION, CATEGORY_SELF_MODIFICATION, CATEGORY_REFLECTION):
            continue
        filtered.append(m)

    summary = build_memory_summary_with_sections(filtered, user_message=None)
    obs_lines: list[str] = []
    for m in _normalize_memories(memories):
        if not isinstance(m, dict):
            continue
        if _memory_category_from_item(m) != CATEGORY_SELF_OBSERVATION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if raw:
            obs_lines.append(raw[:500])
    obs_blob = "\n".join(obs_lines[-40:])

    system = """You are Angel's Stage 6 self-modification planner.

You receive (1) a summary of Tyler-relevant memories and (2) recent self-observation JSON snippets from conversation turns.

Task: propose 1-3 concrete improvements to how Angel should operate for Tyler. Each must be safe:
- Do NOT propose removing capabilities, bypassing approval, weakening safety, or changing core values.
- Do NOT propose removing the requirement that Tyler approves modifications.
- Prefer small, testable prompt-level or communication-style adjustments.

Respond with JSON ONLY:
{
  "proposals": [
    {
      "title": "short string",
      "observation": "what pattern you infer",
      "proposed_change": "exact instruction text to add to Angel's system behavior (plain English)",
      "change_type": "system_prompt|behavior|capability_emphasis|communication_style",
      "confidence": "LOW|MEDIUM|HIGH"
    }
  ],
  "safety_note": "string"
}
If nothing is worth changing, return {"proposals": [], "safety_note": "..."}."""

    user_blob = f"--- Memory summary ---\n{summary[:12000]}\n\n--- Recent observations ---\n{obs_blob[:8000]}"
    try:
        raw = call_claude(
            anthropic_client,
            system,
            user_blob,
            model="claude-sonnet-4-5",
            prior_turns=None,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "count": 0}

    try:
        # strip markdown fences
        t = (raw or "").strip()
        if "```" in t:
            t = re.sub(r"^```[a-z]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        data = json.loads(t)
    except Exception as e:
        return {"ok": False, "error": f"invalid JSON from model: {e}", "count": 0}

    proposals = data.get("proposals") or []
    if not isinstance(proposals, list):
        proposals = []

    print(f"[selfmod] generated {len(proposals)} proposals", flush=True)

    n = 0
    for p in proposals[:max_proposals]:
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        prop = (p.get("proposed_change") or "").strip()
        ok, err = _safety_check_proposal(title + " " + prop)
        if not ok or not title or not prop:
            continue
        mid = secrets.token_hex(8)
        now = _ts()
        ctype = (p.get("change_type") or "behavior").strip()
        if ctype not in ("system_prompt", "behavior", "capability_emphasis", "communication_style"):
            ctype = "behavior"
        conf = (p.get("confidence") or "MEDIUM").upper()
        if conf not in ("LOW", "MEDIUM", "HIGH"):
            conf = "MEDIUM"
        record = {
            "modification_id": mid,
            "title": title,
            "observation": (p.get("observation") or "").strip(),
            "proposed_change": prop,
            "change_type": ctype,
            "confidence": conf,
            "status": "proposed",
            "created_at": now,
            "updated_at": now,
            "safety_disclaimer": "This is a suggestion only. You have full control.",
        }
        _save_modification_memory(memory_client, user_id, use_mem0_cloud, record)
        mirror_mod_file(files_cabinet, record)
        n += 1

    if n:
        print(f"[selfmod] proposals saved successfully (n={n})", flush=True)
        set_pending_proposal_notification(
            "I've been observing our interactions and I have a proposed modification to how I operate. Want to review it?"
        )
    else:
        print("[selfmod] no proposals saved (model empty, safety filter, or save failure)", flush=True)
    return {"ok": True, "count": n, "safety_note": data.get("safety_note")}


def seed_initial_self_modification(
    anthropic_client,
    memory_client,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """If no proposals exist, seed 2-3 insightful starter proposals from memory."""
    from angel import (
        CATEGORY_SELF_MODIFICATION,
        CATEGORY_SELF_OBSERVATION,
        build_memory_summary_with_sections,
        call_claude,
        fetch_combined_memories,
    )

    mem = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    existing = iter_self_modifications(mem)
    if any(r.get("status") in ("proposed", "applied", "approved") for r in existing):
        return {"ok": True, "seeded": False, "reason": "already present"}

    filtered: list = []
    for m in _normalize_memories(mem):
        if not isinstance(m, dict):
            continue
        cat = _memory_category_from_item(m)
        if cat in (CATEGORY_SELF_OBSERVATION, CATEGORY_SELF_MODIFICATION):
            continue
        filtered.append(m)
    summary = build_memory_summary_with_sections(filtered, user_message=None)
    if len(summary) < 80:
        return {"ok": True, "seeded": False, "reason": "insufficient memory context"}

    system = """You know Tyler only from the memory summary below.

Generate exactly 2 or 3 JSON objects for Angel's Stage 6 self-modification *seed* proposals. They should feel specific to Tyler—not generic chatbot advice.

Rules:
- JSON ONLY: {"proposals": [ {...}, ... ]}
- Each proposal: title, observation (what Tyler seems to need), proposed_change (plain instruction for Angel), change_type (behavior|communication_style|capability_emphasis|system_prompt), confidence (MEDIUM or HIGH).
- Never suggest removing capabilities, bypassing approval, or weakening safety."""

    try:
        raw = call_claude(
            anthropic_client,
            system,
            summary[:10000],
            model="claude-sonnet-4-5",
            prior_turns=None,
        )
        t = (raw or "").strip()
        if "```" in t:
            t = re.sub(r"^```[a-z]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        data = json.loads(t)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    proposals = data.get("proposals") or []
    if not isinstance(proposals, list):
        proposals = []
    n = 0
    for p in proposals[:3]:
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        prop = (p.get("proposed_change") or "").strip()
        ok, _ = _safety_check_proposal(title + " " + prop)
        if not ok or not title or not prop:
            continue
        mid = secrets.token_hex(8)
        now = _ts()
        record = {
            "modification_id": mid,
            "title": title,
            "observation": (p.get("observation") or "").strip(),
            "proposed_change": prop,
            "change_type": (p.get("change_type") or "behavior").strip(),
            "confidence": (p.get("confidence") or "MEDIUM").upper(),
            "status": "proposed",
            "created_at": now,
            "updated_at": now,
            "safety_disclaimer": "This is a suggestion only. You have full control.",
            "seed": True,
        }
        if record["change_type"] not in ("system_prompt", "behavior", "capability_emphasis", "communication_style"):
            record["change_type"] = "behavior"
        _save_modification_memory(memory_client, user_id, use_mem0_cloud, record)
        mirror_mod_file(files_cabinet, record)
        n += 1
    if n:
        set_pending_proposal_notification(
            "I've been observing our interactions and I have a proposed modification to how I operate. Want to review it?"
        )
    return {"ok": True, "seeded": n > 0, "count": n}


def api_list_observations(memory_client, user_id: str, use_mem0_cloud: bool, limit: int = 50) -> list[dict]:
    from angel import fetch_combined_memories

    mem = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    out: list[dict] = []
    for m in _normalize_memories(mem):
        if not isinstance(m, dict):
            continue
        if _memory_category_from_item(m) != CATEGORY_SELF_OBSERVATION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        out.append(
            {
                "created_at": m.get("created_at"),
                "text": raw[:2000],
            }
        )
    try:
        out.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    except Exception:
        pass
    return out[:limit]


def api_list_proposals(memory_client, user_id: str, use_mem0_cloud: bool) -> list[dict]:
    from angel import fetch_combined_memories

    mem = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    nraw = len(mem) if isinstance(mem, list) else 0
    print(f"[selfmod] api_list_proposals: fetch_combined_memories -> {nraw} raw rows", flush=True)
    out = iter_self_modifications(mem)
    print(f"[selfmod] api_list_proposals: returning {len(out)} proposal record(s)", flush=True)
    return out
