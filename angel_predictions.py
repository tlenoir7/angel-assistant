"""
Item 15 — Predictive modeling: forecasts, Mem0 + Predictions folder storage, accuracy tracking.
Imports from `angel` are deferred to function bodies to avoid circular import at startup.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic

PREDICTIONS_FOLDER = "Predictions"
PREDICTION_FILE_PREFIX = "PRED-"

PREDICTION_CATEGORIES = frozenset(
    {
        "UAP_disclosure",
        "political",
        "institutional",
        "personal_mission",
        "geopolitical",
        "technology",
        "wildcard",
    }
)
CONFIDENCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "VERY_HIGH"})
TIMEFRAMES = frozenset(
    {
        "within 30 days",
        "within 90 days",
        "within 1 year",
        "2-5 years",
    }
)
STATUSES = frozenset({"active", "confirmed", "denied", "expired"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prediction_upsert_structured_memory(
    memory_client: Any,
    user_id: str,
    *,
    prediction_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_PREDICTION
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-predictions",
        "prediction_id": prediction_id,
    }
    try:
        entries = ang._load_local_memory_entries(user_id)
        if not isinstance(entries, list):
            entries = []
        filtered = [
            e
            for e in entries
            if not (
                isinstance(e, dict)
                and isinstance(e.get("metadata"), dict)
                and e["metadata"].get("category") == cat
                and e["metadata"].get("prediction_id") == prediction_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel prediction {prediction_id}] {text[:1200]}"},
                {"role": "assistant", "content": "Stored."},
            ]
            try:
                memory_client.add(
                    messages,
                    user_id=user_id,
                    metadata=dict(meta),
                    infer=False,
                    async_mode=True,
                )
            except TypeError:
                memory_client.add(messages, user_id=user_id, metadata=dict(meta))
        except Exception:
            pass


def _sync_prediction_intel_file(files_cabinet: Any, pred: dict) -> None:
    import angel as ang

    pid = (pred.get("prediction_id") or "").strip()
    if not pid:
        return
    fn = f"{PREDICTION_FILE_PREFIX}{pid}"
    body = json.dumps(pred, ensure_ascii=False, indent=2)
    tags = [
        "prediction",
        f"category:{pred.get('category', 'wildcard')}",
        f"status:{pred.get('status', 'active')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(PREDICTIONS_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass


def _normalize_prediction(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    pid = str(raw.get("prediction_id") or "").strip()
    if not pid:
        return None
    cat = str(raw.get("category") or "wildcard").strip()
    if cat not in PREDICTION_CATEGORIES:
        cat = "wildcard"
    conf = str(raw.get("confidence") or "MEDIUM").strip().upper()
    if conf not in CONFIDENCE_LEVELS:
        conf = "MEDIUM"
    tf = str(raw.get("timeframe") or "within 90 days").strip()
    if tf not in TIMEFRAMES:
        # allow close matches
        tl = tf.lower()
        if "30" in tl:
            tf = "within 30 days"
        elif "90" in tl:
            tf = "within 90 days"
        elif "year" in tl and "2" not in tl:
            tf = "within 1 year"
        elif "2" in tl and "5" in tl:
            tf = "2-5 years"
        else:
            tf = "within 90 days"
    st = str(raw.get("status") or "active").strip().lower()
    if st not in STATUSES:
        st = "active"
    ev = raw.get("supporting_evidence")
    if not isinstance(ev, list):
        ev = [str(ev)] if ev else []
    ev = [str(x).strip() for x in ev if str(x).strip()][:40]
    acc = raw.get("accuracy_score")
    if acc is not None:
        try:
            acc = max(0, min(100, int(acc)))
        except (TypeError, ValueError):
            acc = None
    else:
        acc = None
    return {
        "prediction_id": pid,
        "title": str(raw.get("title") or "Untitled forecast").strip()[:500],
        "full_prediction": str(raw.get("full_prediction") or "").strip()[:12000],
        "category": cat,
        "confidence": conf,
        "timeframe": tf,
        "supporting_evidence": ev,
        "created_at": str(raw.get("created_at") or _now_iso())[:32],
        "status": st,
        "outcome_notes": str(raw.get("outcome_notes") or "").strip()[:8000],
        "accuracy_score": acc,
        "updated_at": str(raw.get("updated_at") or raw.get("created_at") or _now_iso())[:32],
    }


def _merge_prediction_dict(into: dict[str, dict], p: dict) -> None:
    np = _normalize_prediction(p)
    if not np:
        return
    pid = np["prediction_id"]
    prev = into.get(pid)
    if prev is None:
        into[pid] = np
        return
    tu = (np.get("updated_at") or np.get("created_at") or "")
    pu = (prev.get("updated_at") or prev.get("created_at") or "")
    if tu >= pu:
        into[pid] = np
    else:
        into[pid] = prev


def fetch_all_predictions(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict]:
    import angel as ang

    out: dict[str, dict] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PREDICTION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _merge_prediction_dict(out, obj)

    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PREDICTION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _merge_prediction_dict(out, obj)
    return out


def _save_prediction(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    pred: dict,
) -> dict:
    p = _normalize_prediction(pred)
    if not p:
        raise ValueError("invalid prediction")
    p["updated_at"] = _now_iso()
    text = json.dumps(p, ensure_ascii=False)
    _prediction_upsert_structured_memory(
        memory_client,
        user_id,
        prediction_id=p["prediction_id"],
        text=text,
        use_mem0_cloud=use_mem0_cloud,
    )
    _sync_prediction_intel_file(files_cabinet, p)
    return p


def _new_prediction_id(seed: str) -> str:
    h = hashlib.sha256(f"{seed}|{uuid.uuid4().hex}|{time.time()}".encode()).hexdigest()
    return h[:16]


def _gather_intelligence_context(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
    anthropic_client: anthropic.Anthropic,
) -> str:
    import angel as ang

    parts: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    try:
        recs = files_cabinet.list_files(folder=ang.THREAT_INTEL_FOLDER)
    except Exception:
        recs = []
    threat_bits: list[str] = []
    for rec in recs[:40]:
        name = (rec.get("name") or "").strip()
        if not name:
            continue
        try:
            full = files_cabinet.get_file(name)
        except Exception:
            full = None
        if not full:
            continue
        ua = (full.get("updated_at") or full.get("created_at") or "").strip()
        if ua:
            try:
                dt = datetime.fromisoformat(ua.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    continue
            except Exception:
                pass
        body = (full.get("content") or "")[:2000]
        threat_bits.append(f"- File {name}: {body[:900]}...")
    if threat_bits:
        parts.append("=== Recent threat intelligence (last ~30 days, excerpts) ===\n" + "\n".join(threat_bits[:25]))

    try:
        osrecs = files_cabinet.list_files(folder=ang.OSINT_DOSSIERS_FOLDER)
    except Exception:
        osrecs = []
    os_bits: list[str] = []
    for rec in osrecs[:25]:
        name = (rec.get("name") or "").strip()
        if not name:
            continue
        full = files_cabinet.get_file(name)
        if not full:
            continue
        os_bits.append(
            f"- {name}: {(full.get('content') or '')[:700]}..."
        )
    if os_bits:
        parts.append("=== OSINT dossiers (excerpts) ===\n" + "\n".join(os_bits))

    try:
        summ = ang.get_network_summary(
            memory_client, user_id, use_mem0_cloud, files_cabinet
        )
        parts.append("=== Mission network summary ===\n" + json.dumps(summ, indent=2, ensure_ascii=False)[:6000])
    except Exception:
        pass

    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    bh = ang.get_recent_briefing_history_for_prompt(memories, days=30)
    if bh:
        parts.append("=== Recent briefing history ===\n" + bh[:6000])

    api_key = __import__("os").getenv("TAVILY_API_KEY") or ""
    if api_key:
        queries = [
            "UAP disclosure alien.gov Trump executive order 2025 2026",
            "FBI budget staffing federal law enforcement news",
            "AI regulation US legislation 2025 2026",
        ]
        for q in queries:
            hits = ang._tavily_search_one(q, api_key, max_results=4, search_depth="basic")
            lines = [f"Query: {q}"]
            for h in hits[:4]:
                if isinstance(h, dict):
                    lines.append(
                        f"  - {(h.get('title') or '')[:120]} | {(h.get('content') or h.get('snippet') or '')[:400]}"
                    )
            parts.append("=== Tavily (geopolitical / news context) ===\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "(No structured intelligence context available yet.)"


def _active_titles_for_dedupe(by_id: dict[str, dict]) -> str:
    act = [p for p in by_id.values() if (p.get("status") or "") == "active"]
    lines = [f"- {p.get('title', '')} [{p.get('category', '')}]" for p in act[:30]]
    return "\n".join(lines) if lines else "(none)"


def _claude_parse_predictions_json(
    anthropic_client: anthropic.Anthropic,
    user_block: str,
    *,
    model: str = "claude-sonnet-4-5",
) -> list[dict]:
    system = """You are Angel's predictive modeling core. Output ONLY valid JSON: an array of 3 to 5 objects.
Each object MUST have exactly these string fields:
- title (short headline)
- full_prediction (detailed forecast, 2-6 sentences)
- category: one of UAP_disclosure | political | institutional | personal_mission | geopolitical | technology | wildcard
- confidence: one of LOW | MEDIUM | HIGH | VERY_HIGH
- timeframe: exactly one of "within 30 days" | "within 90 days" | "within 1 year" | "2-5 years"
- supporting_evidence: array of short strings citing what drives the forecast (e.g. threat file theme, network pattern, OSINT, briefing, news)

Do not include prediction_id, status, or timestamps — the server will assign them.
No markdown fences. No commentary outside the JSON array."""

    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0.35,
        system=system,
        messages=[{"role": "user", "content": user_block[:100_000]}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict)]


def _title_similarity(a: str, b: str) -> float:
    a_set = set(re.findall(r"[a-z0-9]{4,}", (a or "").lower()))
    b_set = set(re.findall(r"[a-z0-9]{4,}", (b or "").lower()))
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    return inter / max(len(a_set), len(b_set))


def generate_predictions(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    context: str | None = None,
    focus_topic: str | None = None,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """
    Synthesize 3–5 predictions from intelligence + Tavily. Skips duplicates vs active predictions (fuzzy title).
    """
    by_id = fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
    block = context or _gather_intelligence_context(
        memory_client, user_id, use_mem0_cloud, files_cabinet, anthropic_client
    )
    dedupe = _active_titles_for_dedupe(by_id)
    extra = ""
    if focus_topic:
        extra = f"\n\nTyler asked specifically for a forecast about: {focus_topic.strip()}\nMake at least one object centered on that topic; others may stay mission-wide.\n"
    user_block = f"""Using the intelligence context below, produce 3-5 forward-looking predictions about what is likely to happen next across Tyler's mission space (UAP/disclosure, institutions, politics, tech, whistleblowers, network dynamics).

Already ACTIVE predictions (do NOT duplicate or lightly rephrase the same thesis — pick distinct angles):
{dedupe}
{extra}
=== CONTEXT ===
{block}
"""
    raw_list = _claude_parse_predictions_json(anthropic_client, user_block, model=model)
    saved: list[dict] = []
    skipped = 0
    for item in raw_list:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        dup = False
        for p in by_id.values():
            if (p.get("status") or "") != "active":
                continue
            if _title_similarity(title, p.get("title", "")) > 0.45:
                dup = True
                break
        if dup:
            skipped += 1
            continue
        pid = _new_prediction_id(title)
        item["prediction_id"] = pid
        item["status"] = "active"
        item["outcome_notes"] = ""
        item["accuracy_score"] = None
        item["created_at"] = _now_iso()
        item["updated_at"] = item["created_at"]
        try:
            sp = _save_prediction(
                memory_client, user_id, files_cabinet, use_mem0_cloud, item
            )
            saved.append(sp)
            by_id[pid] = sp
        except Exception:
            continue
    return {"ok": True, "saved": saved, "skipped_duplicates": skipped, "count": len(saved)}


def resolve_prediction(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    prediction_id: str,
    outcome: str,
    accurate: bool,
) -> dict | None:
    pid = (prediction_id or "").strip()
    if not pid:
        return None
    by_id = fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
    prev = by_id.get(pid)
    if not prev:
        return None
    prev["status"] = "confirmed" if accurate else "denied"
    prev["outcome_notes"] = (outcome or "").strip()[:8000]
    prev["accuracy_score"] = 100 if accurate else 0
    prev["updated_at"] = _now_iso()
    return _save_prediction(
        memory_client, user_id, files_cabinet, use_mem0_cloud, prev
    )


def get_active_predictions(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict]:
    return [
        p
        for p in fetch_all_predictions(memory_client, user_id, use_mem0_cloud).values()
        if (p.get("status") or "") == "active"
    ]


def get_prediction_accuracy(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    all_p = list(fetch_all_predictions(memory_client, user_id, use_mem0_cloud).values())
    resolved = [
        p
        for p in all_p
        if (p.get("status") or "") in ("confirmed", "denied")
        and p.get("accuracy_score") is not None
    ]
    if not resolved:
        return {
            "overall_avg": None,
            "resolved_count": 0,
            "confirmed": 0,
            "denied": 0,
            "by_category": {},
            "recent_resolved": [],
        }
    scores = [int(p["accuracy_score"]) for p in resolved]
    overall = round(sum(scores) / len(scores), 1)
    by_cat: dict[str, list[int]] = {}
    for p in resolved:
        c = p.get("category") or "wildcard"
        by_cat.setdefault(c, []).append(int(p["accuracy_score"]))
    by_cat_avg = {k: round(sum(v) / len(v), 1) for k, v in by_cat.items()}
    recent = sorted(
        resolved,
        key=lambda x: (x.get("updated_at") or x.get("created_at") or ""),
        reverse=True,
    )[:12]
    recent_resolved = [
        {
            "prediction_id": p.get("prediction_id"),
            "title": p.get("title"),
            "status": p.get("status"),
            "accuracy_score": p.get("accuracy_score"),
            "updated_at": p.get("updated_at"),
        }
        for p in recent
    ]
    return {
        "overall_avg": overall,
        "resolved_count": len(resolved),
        "confirmed": sum(1 for p in resolved if p.get("status") == "confirmed"),
        "denied": sum(1 for p in resolved if p.get("status") == "denied"),
        "by_category": by_cat_avg,
        "recent_resolved": recent_resolved,
    }


def check_predictions_against_reality(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    max_checks: int = 10,
) -> dict[str, Any]:
    import angel as ang
    import os

    api_key = os.getenv("TAVILY_API_KEY") or ""
    active = get_active_predictions(memory_client, user_id, use_mem0_cloud)[
        :max_checks
    ]
    results: list[dict] = []
    auto_resolved = 0
    for p in active:
        pid = p.get("prediction_id")
        title = (p.get("title") or "")[:200]
        fp = (p.get("full_prediction") or "")[:1200]
        snippets = ""
        if api_key:
            hits = ang._tavily_search_one(
                f"{title} news developments", api_key, max_results=5, search_depth="basic"
            )
            for h in hits:
                if isinstance(h, dict):
                    snippets += (h.get("content") or h.get("snippet") or "")[:500] + "\n---\n"
        if not snippets.strip():
            results.append(
                {"prediction_id": pid, "action": "skipped", "reason": "no_tavily_results"}
            )
            continue
        try:
            resp = anthropic_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=400,
                temperature=0.1,
                system='Reply with JSON only: {"verdict":"confirmed"|"denied"|"unclear","confidence":0.0-1.0,"notes":"short"}. '
                "confirmed = clear real-world evidence the prediction direction occurred; denied = clear contrary evidence; unclear = mixed or insufficient.",
                messages=[
                    {
                        "role": "user",
                        "content": f"PREDICTION TITLE: {title}\nDETAIL: {fp}\n\nRECENT WEB SNIPPETS:\n{snippets[:8000]}",
                    }
                ],
            )
            txt = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    txt += block.text
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                raise ValueError("no json")
            ev = json.loads(m.group(0))
        except Exception as e:
            results.append(
                {
                    "prediction_id": pid,
                    "action": "error",
                    "error": str(e),
                }
            )
            continue
        verdict = str(ev.get("verdict") or "unclear").lower()
        conf = float(ev.get("confidence") or 0)
        notes = str(ev.get("notes") or "")[:2000]
        results.append(
            {
                "prediction_id": pid,
                "verdict": verdict,
                "confidence": conf,
                "notes": notes,
            }
        )
        if verdict in ("confirmed", "denied") and conf >= 0.85:
            resolve_prediction(
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                prediction_id=str(pid),
                outcome=f"Auto-resolved (reality check): {notes}",
                accurate=(verdict == "confirmed"),
            )
            auto_resolved += 1
    return {"ok": True, "checked": len(results), "auto_resolved": auto_resolved, "results": results}


def seed_initial_predictions_if_needed(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    existing = fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
    if existing:
        return {"ok": True, "seeded": False, "reason": "predictions_already_exist"}
    ctx = _gather_intelligence_context(
        memory_client, user_id, use_mem0_cloud, files_cabinet, anthropic_client
    )
    seed_note = """

Additional seed focus (first deploy — ensure coverage):
- UAP disclosure trajectory: alien.gov, Trump disclosure-related orders, what likely happens next in public / policy.
- FBI institutional trajectory: budget cuts, staffing, morale.
- Whistleblower / retaliation environment for national-security disclosure themes.
- AI regulation consolidation in the US / West.
- Mission network: who may surface next in the disclosure-adjacent community (new voices, hearings, media).
"""
    res = generate_predictions(
        anthropic_client,
        memory_client,
        user_id,
        files_cabinet,
        use_mem0_cloud,
        context=ctx + seed_note,
        model="claude-sonnet-4-5",
    )
    if isinstance(res, dict):
        res["seeded"] = bool(res.get("count", 0) > 0)
        return res
    return {"ok": False, "seeded": False, "error": "generate_failed"}


def detect_prediction_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw:
        return None, {}
    lower = raw.lower()

    if re.search(
        r"(?i)\b(what\s+do\s+you\s+predict|what'?s\s+coming|your\s+predictions|forecast\s+for\s+the\s+mission)\b",
        lower,
    ):
        return "list_active", {}

    m = re.search(r"(?i)\bwere\s+you\s+right\s+about\s+(.+)$", raw)
    if m:
        return "status_topic", {"topic": m.group(1).strip().rstrip("?.!")}

    if re.search(
        r"(?i)\b(how\s+accurate\s+are\s+you|prediction\s+accuracy|your\s+track\s+record\s+on\s+predictions)\b",
        lower,
    ):
        return "accuracy", {}

    m = re.search(
        r"(?i)\b(?:angel\s+)?make\s+a\s+prediction\s+(?:about|on|regarding)\s+(.+)$",
        raw,
    )
    if m:
        return "targeted", {"topic": m.group(1).strip().rstrip("?.!")}

    return None, {}


def _find_predictions_matching_topic(
    by_id: dict[str, dict], topic: str
) -> list[dict]:
    t = (topic or "").strip().lower()
    if not t:
        return []
    out: list[dict] = []
    for p in by_id.values():
        blob = f"{p.get('title','')} {p.get('full_prediction','')}".lower()
        if t in blob or any(t in str(x).lower() for x in (p.get("supporting_evidence") or [])):
            out.append(p)
    return sorted(
        out,
        key=lambda x: (x.get("updated_at") or x.get("created_at") or ""),
        reverse=True,
    )


def format_prediction_reply_for_prompt(
    intent: str,
    payload: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    anthropic_client: anthropic.Anthropic | None,
    files_cabinet: Any,
) -> str:
    by_id = fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
    if intent == "list_active":
        act = [p for p in by_id.values() if (p.get("status") or "") == "active"]
        act.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
        if not act:
            return "[Angel predictions — active forecasts]\n(none on file yet; you can run prediction generation from the server schedule or API.)"
        lines = []
        for p in act[:20]:
            lines.append(
                f"- [{p.get('category')}] {p.get('title')} | confidence={p.get('confidence')} | timeframe={p.get('timeframe')} | id={p.get('prediction_id')}\n  Summary: {(p.get('full_prediction') or '')[:400]}"
            )
        return "[Angel predictions — active forecasts]\n" + "\n".join(lines)

    if intent == "accuracy":
        acc = get_prediction_accuracy(memory_client, user_id, use_mem0_cloud)
        return (
            "[Angel predictions — accuracy scorecard]\n"
            + json.dumps(acc, indent=2, ensure_ascii=False)[:8000]
        )

    if intent == "targeted":
        topic = (payload.get("topic") or "").strip()
        if not topic or not anthropic_client:
            return f"[Angel predictions — targeted request]\n(topic missing or model unavailable: {topic!r})"
        r = generate_predictions(
            anthropic_client,
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            focus_topic=topic,
        )
        return (
            "[Angel predictions — generated targeted forecasts]\n"
            + json.dumps(r, indent=2, ensure_ascii=False)[:8000]
        )

    if intent == "status_topic":
        topic = (payload.get("topic") or "").strip()
        hits = _find_predictions_matching_topic(by_id, topic)
        if not hits:
            return f"[Angel predictions — status lookup for {topic!r}]\nNo matching predictions found."
        lines = []
        for p in hits[:8]:
            lines.append(
                json.dumps(
                    {
                        "prediction_id": p.get("prediction_id"),
                        "title": p.get("title"),
                        "status": p.get("status"),
                        "accuracy_score": p.get("accuracy_score"),
                        "outcome_notes": (p.get("outcome_notes") or "")[:600],
                        "timeframe": p.get("timeframe"),
                    },
                    ensure_ascii=False,
                )
            )
        return (
            f"[Angel predictions — matches for {topic!r}]\n" + "\n".join(lines)
        )

    return ""
