"""
Item 16 — Proactive background intelligence: watch list, scheduled Tavily research,
cross-links to threats, OSINT, predictions, and network graph.
Imports from `angel` / `angel_predictions` are deferred inside functions to avoid cycles.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

PROACTIVE_INTEL_FOLDER = "Proactive Intelligence"
WATCH_FILE_PREFIX = "WATCH-"

PRIORITIES = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
WATCH_TYPES = frozenset({"person", "topic", "situation", "organization"})
FREQUENCIES = frozenset({"daily", "every_3_days", "weekly"})

_LAST_PROACTIVE_RUN: dict[str, Any] = {"at": None, "summary": {}}
_RECENT_FINDINGS: deque[dict] = deque(maxlen=80)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_watch_id(seed: str) -> str:
    return hashlib.sha256(f"{seed}|{uuid.uuid4().hex}|{time.time()}".encode()).hexdigest()[:14]


def _watch_upsert_structured_memory(
    memory_client: Any,
    user_id: str,
    *,
    watch_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_PROACTIVE_WATCH
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-proactive-watch",
        "watch_id": watch_id,
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
                and e["metadata"].get("watch_id") == watch_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel proactive watch {watch_id}] {text[:1200]}"},
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


def _finding_upsert(
    memory_client: Any,
    user_id: str,
    finding: dict,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    fid = (finding.get("finding_id") or "").strip()
    if not fid:
        return
    cat = ang.CATEGORY_PROACTIVE_FINDING
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-proactive-finding",
        "finding_id": fid,
    }
    text = json.dumps(finding, ensure_ascii=False)
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
                and e["metadata"].get("finding_id") == fid
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel proactive finding {fid}] {text[:1200]}"},
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


def _normalize_watch(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    wid = str(raw.get("watch_id") or "").strip()
    if not wid:
        return None
    wt = str(raw.get("watch_type") or "topic").strip().lower()
    if wt not in WATCH_TYPES:
        wt = "topic"
    pr = str(raw.get("priority") or "MEDIUM").strip().upper()
    if pr not in PRIORITIES:
        pr = "MEDIUM"
    cf = str(raw.get("check_frequency") or "weekly").strip().lower().replace(" ", "_")
    if cf == "every3days":
        cf = "every_3_days"
    if cf not in FREQUENCIES:
        cf = "weekly"
    return {
        "watch_id": wid,
        "label": str(raw.get("label") or "Untitled watch").strip()[:500],
        "watch_type": wt,
        "priority": pr,
        "last_checked": str(raw.get("last_checked") or "").strip()[:32] or None,
        "check_frequency": cf,
        "auto_added": bool(raw.get("auto_added", False)),
        "active": bool(raw.get("active", True)),
        "updated_at": str(raw.get("updated_at") or _now_iso())[:32],
    }


def _merge_watch_dict(into: dict[str, dict], obj: dict) -> None:
    w = _normalize_watch(obj)
    if not w:
        return
    wid = w["watch_id"]
    prev = into.get(wid)
    if prev is None:
        into[wid] = w
        return
    tu = w.get("updated_at") or ""
    pu = prev.get("updated_at") or ""
    into[wid] = w if tu >= pu else prev


def fetch_all_watches(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict]:
    import angel as ang

    out: dict[str, dict] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PROACTIVE_WATCH:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _merge_watch_dict(out, obj)
    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PROACTIVE_WATCH:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _merge_watch_dict(out, obj)
    return out


def _save_watch(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    w: dict,
) -> dict:
    import angel as ang

    obj = _normalize_watch(w)
    if not obj:
        raise ValueError("invalid watch")
    obj["updated_at"] = _now_iso()
    text = json.dumps(obj, ensure_ascii=False)
    _watch_upsert_structured_memory(
        memory_client,
        user_id,
        watch_id=obj["watch_id"],
        text=text,
        use_mem0_cloud=use_mem0_cloud,
    )
    fn = f"{WATCH_FILE_PREFIX}{obj['watch_id']}"
    body = text
    tags = ["proactive_watch", f"priority:{obj['priority']}", f"type:{obj['watch_type']}"]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(PROACTIVE_INTEL_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass
    return obj


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def watch_is_due(w: dict, now: datetime) -> bool:
    if not w.get("active", True):
        return False
    pri = (w.get("priority") or "MEDIUM").upper()
    last = _parse_iso(w.get("last_checked"))
    delta = (now - last) if last else timedelta(days=999)
    if pri == "CRITICAL":
        return True
    if pri == "HIGH":
        return delta >= timedelta(days=3)
    return delta >= timedelta(days=7)


def _build_tavily_queries(w: dict) -> list[str]:
    label = (w.get("label") or "").strip()
    wt = w.get("watch_type") or "topic"
    q = [
        f"{label} latest news developments",
        f"{label} updates 2025 2026",
    ]
    if wt == "person":
        q.append(f"{label} interview statement UAP")
    elif wt == "organization":
        q.append(f"{label} federal agency news")
    else:
        q.append(f"{label} congressional policy")
    return q[:3]


def _record_finding(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    rec: dict,
) -> None:
    fid = rec.get("finding_id") or _new_watch_id(str(rec.get("watch_id", "")))
    rec["finding_id"] = fid
    rec["at"] = _now_iso()
    _finding_upsert(memory_client, user_id, rec, use_mem0_cloud)
    _RECENT_FINDINGS.appendleft(dict(rec))


def run_proactive_intelligence(
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    import angel as ang
    import angel_predictions as apred

    global _LAST_PROACTIVE_RUN
    now = datetime.now(timezone.utc)
    by_id = fetch_all_watches(memory_client, user_id, use_mem0_cloud)
    active = [w for w in by_id.values() if w.get("active", True)]
    due = [w for w in active if watch_is_due(w, now)]
    api_key = __import__("os").getenv("TAVILY_API_KEY") or ""

    out: dict[str, Any] = {
        "ok": True,
        "due_count": len(due),
        "checked": 0,
        "significant": 0,
        "filed_proactive": 0,
        "filed_threat": 0,
        "details": [],
    }

    preds = apred.fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
    active_preds = [p for p in preds.values() if (p.get("status") or "") == "active"]

    nodes, _edges = ang.network_load_graph(
        memory_client, user_id, use_mem0_cloud, files_cabinet
    )

    for w in due:
        wid = w.get("watch_id")
        label = w.get("label") or ""
        out["checked"] += 1
        snippets = ""
        if api_key:
            for q in _build_tavily_queries(w):
                for r in ang._tavily_search_one(q, api_key, max_results=4, search_depth="basic"):
                    if isinstance(r, dict):
                        snippets += f"{r.get('title','')}\n{(r.get('content') or r.get('snippet') or '')[:600]}\n---\n"
                if len(snippets) > 12000:
                    break
        if not snippets.strip():
            w["last_checked"] = _now_iso()
            w["updated_at"] = w["last_checked"]
            try:
                _save_watch(memory_client, user_id, files_cabinet, use_mem0_cloud, w)
            except Exception:
                pass
            out["details"].append({"watch_id": wid, "result": "no_search_results"})
            continue

        try:
            resp = anthropic_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=700,
                temperature=0.15,
                system='Reply JSON only: {"significant":true/false,"summary":"2-4 sentences","threat_level":"NONE|LOW|MEDIUM|HIGH|CRITICAL","matches_prediction_id":"","suggests_osint_refresh":true/false,"network_note":""}',
                messages=[
                    {
                        "role": "user",
                        "content": f"WATCH: {label} (type={w.get('watch_type')})\n\nWEB SNIPPETS:\n{snippets[:10000]}",
                    }
                ],
            )
            txt = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    txt += block.text
            m = re.search(r"\{[\s\S]*\}", txt)
            ev = json.loads(m.group(0)) if m else {}
        except Exception as ex:
            ev = {"significant": False, "error": str(ex)}

        w["last_checked"] = _now_iso()
        w["updated_at"] = w["last_checked"]
        try:
            _save_watch(memory_client, user_id, files_cabinet, use_mem0_cloud, w)
        except Exception:
            pass

        if not ev.get("significant"):
            out["details"].append({"watch_id": wid, "result": "not_significant"})
            continue

        out["significant"] += 1
        summary = str(ev.get("summary") or "").strip() or "Update detected."
        tl = str(ev.get("threat_level") or "NONE").upper()
        safe_slug = hashlib.sha256(f"{wid}|{summary[:80]}".encode()).hexdigest()[:10]
        fname = f"PI-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{safe_slug}"
        body = "\n".join(
            [
                f"proactive_watch_id: {wid}",
                f"watch_label: {label}",
                f"threat_level_signal: {tl}",
                "",
                summary,
                "",
                f"Sources condensed from Tavily (watch run {_now_iso()}).",
            ]
        )
        try:
            files_cabinet.create_file(
                PROACTIVE_INTEL_FOLDER,
                fname,
                body,
                tags=["proactive_intel", f"watch:{wid}", f"priority:{w.get('priority','')}"],
            )
            out["filed_proactive"] += 1
        except ValueError:
            try:
                files_cabinet.update_file(fname, body)
                out["filed_proactive"] += 1
            except Exception:
                pass
        except Exception:
            pass

        cross: dict[str, Any] = {}
        if tl in ("HIGH", "CRITICAL"):
            try:
                tb = "\n".join(
                    [
                        f"watch_category: {label[:200]}",
                        f"threat_level: {tl}",
                        f"source_url: (proactive watch)",
                        f"event_date:",
                        "",
                        f"Proactive monitoring surfaced this regarding «{label}»:\n\n{summary}",
                    ]
                )
                tf = f"TI-PRO-{safe_slug}"
                files_cabinet.create_file(ang.THREAT_INTEL_FOLDER, tf, tb, tags=[f"threat_level:{tl}", "proactive_intel"])
                out["filed_threat"] += 1
                cross["threat_file"] = tf
            except Exception:
                pass

        pid_match = ""
        for p in active_preds:
            pt = (p.get("title") or "") + " " + (p.get("full_prediction") or "")
            if label.lower() in pt.lower() or (len(label) > 12 and label.lower()[:20] in pt.lower()):
                pid_match = str(p.get("prediction_id") or "")
                break
        if pid_match:
            cross["related_prediction_id"] = pid_match

        if ev.get("suggests_osint_refresh"):
            try:
                note = f"Proactive intel suggests refreshing OSINT context for: {label}\n"
                files_cabinet.create_file(
                    PROACTIVE_INTEL_FOLDER,
                    f"OSINT-REFRESH-{safe_slug}",
                    note,
                    tags=["osint_refresh_hint"],
                )
            except Exception:
                pass

        nn = str(ev.get("network_note") or "").strip()
        if nn:
            cross["network_context"] = nn[:800]
        try:
            nid = ang.network_resolve_name_to_id(label, nodes)
            if nid:
                cross["network_node_match"] = nid
        except Exception:
            pass

        finding_rec = {
            "finding_id": _new_watch_id(wid),
            "watch_id": wid,
            "label": label,
            "summary": summary,
            "filed_as": fname,
            "threat_level": tl,
            "cross_system": cross,
        }
        _record_finding(memory_client, user_id, use_mem0_cloud, finding_rec)
        out["details"].append({"watch_id": wid, "result": "filed", "file": fname})

    _LAST_PROACTIVE_RUN = {"at": _now_iso(), "summary": out}
    return out


def format_proactive_intelligence_for_briefing(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    last_run_summary: dict[str, Any] | None = None,
) -> str:
    """Plain-text block for morning briefing."""
    lines: list[str] = []
    if last_run_summary and last_run_summary.get("significant", 0) > 0:
        lines.append(
            f"Overnight proactive scan: {last_run_summary.get('checked', 0)} watch(es) checked, "
            f"{last_run_summary.get('significant', 0)} significant update(s), "
            f"{last_run_summary.get('filed_proactive', 0)} new file(s) in Proactive Intelligence."
        )
        for d in (last_run_summary.get("details") or [])[:8]:
            if d.get("result") == "filed":
                lines.append(f"- Watch {d.get('watch_id', '')[:8]}… → filed {d.get('file', '')}")
    recent = fetch_recent_findings(memory_client, user_id, use_mem0_cloud, limit=6)
    if recent:
        lines.append("Recent proactive findings (high level):")
        for f in recent:
            lines.append(
                f"- {f.get('label', '')}: {(f.get('summary') or '')[:220]}…"
            )
    if not lines:
        return ""
    return "PROACTIVE INTELLIGENCE\n" + "\n".join(lines)


def fetch_recent_findings(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    limit: int = 20,
) -> list[dict]:
    import angel as ang

    out: list[tuple[str, dict]] = []
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PROACTIVE_FINDING:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("finding_id"):
            ca = ang._memory_created_at(m)
            out.append((ca, obj))
    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_PROACTIVE_FINDING:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("finding_id"):
            ca = m.get("created_at") or ""
            out.append((ca, obj))
    try:
        out.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        pass
    return [x[1] for x in out[:limit]]


def add_watch_item(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    label: str,
    watch_type: str = "topic",
    priority: str = "MEDIUM",
    check_frequency: str = "weekly",
    auto_added: bool = False,
) -> dict:
    lab_key = (label or "").strip().lower()
    if lab_key:
        for w in fetch_all_watches(memory_client, user_id, use_mem0_cloud).values():
            if w.get("active", True) and (w.get("label") or "").strip().lower() == lab_key:
                return w
    wt = watch_type.strip().lower()
    if wt not in WATCH_TYPES:
        wt = "topic"
    pr = priority.strip().upper()
    if pr not in PRIORITIES:
        pr = "MEDIUM"
    cf = check_frequency.strip().lower().replace(" ", "_")
    if cf not in FREQUENCIES:
        cf = "weekly"
    wid = _new_watch_id(label)
    w = {
        "watch_id": wid,
        "label": label.strip()[:500],
        "watch_type": wt,
        "priority": pr,
        "last_checked": None,
        "check_frequency": cf,
        "auto_added": auto_added,
        "active": True,
    }
    return _save_watch(memory_client, user_id, files_cabinet, use_mem0_cloud, w)


def deactivate_watch(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    watch_id: str,
) -> dict | None:
    by_id = fetch_all_watches(memory_client, user_id, use_mem0_cloud)
    w = by_id.get(watch_id.strip())
    if not w:
        return None
    w["active"] = False
    w["updated_at"] = _now_iso()
    return _save_watch(memory_client, user_id, files_cabinet, use_mem0_cloud, w)


def seed_proactive_watch_if_empty(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    if fetch_all_watches(memory_client, user_id, use_mem0_cloud):
        return {"ok": True, "seeded": False}
    seeds: list[tuple[str, str, str, str]] = [
        ("David Grusch", "person", "CRITICAL", "daily"),
        ("Luis Elizondo", "person", "HIGH", "every_3_days"),
        ("Christopher Mellon", "person", "HIGH", "every_3_days"),
        ("Ross Coulthart", "person", "HIGH", "every_3_days"),
        ("alien.gov and UAP disclosure progress", "situation", "CRITICAL", "daily"),
        ("FBI budget and staffing crisis", "situation", "HIGH", "every_3_days"),
        ("Whistleblower crackdown escalation", "situation", "HIGH", "every_3_days"),
        ("Congressional UAP hearings", "topic", "HIGH", "every_3_days"),
        ("AI regulation federal vs state", "topic", "MEDIUM", "weekly"),
        ("Black-eyed people phenomenon", "topic", "MEDIUM", "weekly"),
    ]
    n = 0
    for label, wt, pr, cf in seeds:
        try:
            add_watch_item(
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                label=label,
                watch_type=wt,
                priority=pr,
                check_frequency=cf,
                auto_added=True,
            )
            n += 1
        except Exception:
            pass
    return {"ok": True, "seeded": n > 0, "count": n}


# --- Auto hooks ---

def maybe_auto_watch_from_osint(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    target: str,
    target_type: str,
) -> None:
    t = (target or "").strip()
    if len(t) < 2:
        return
    wt = "person" if (target_type or "").lower() == "person" else "organization"
    try:
        add_watch_item(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            label=t,
            watch_type=wt,
            priority="HIGH",
            check_frequency="every_3_days",
            auto_added=True,
        )
    except Exception:
        pass


def maybe_auto_watch_from_threat(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    category: str,
    threat_level: str,
) -> None:
    if threat_level not in ("HIGH", "CRITICAL"):
        return
    cat = (category or "").strip()[:400]
    if not cat:
        return
    try:
        add_watch_item(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            label=cat,
            watch_type="topic",
            priority="HIGH",
            check_frequency="every_3_days",
            auto_added=True,
        )
    except Exception:
        pass


def maybe_auto_watch_from_network(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    display_name: str,
    relevance: str,
) -> None:
    if relevance not in ("HIGH", "CRITICAL"):
        return
    label = (display_name or "").strip()
    if len(label) < 2:
        return
    try:
        add_watch_item(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            label=label,
            watch_type="person",
            priority="HIGH",
            check_frequency="every_3_days",
            auto_added=True,
        )
    except Exception:
        pass


def maybe_auto_watch_from_prediction(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    prediction_title: str,
) -> None:
    t = (prediction_title or "").strip()
    if len(t) < 4:
        return
    try:
        add_watch_item(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            label=t[:500],
            watch_type="situation",
            priority="MEDIUM",
            check_frequency="weekly",
            auto_added=True,
        )
    except Exception:
        pass


def track_user_mentions_for_watch(
    core: Any,
    user_message: str,
) -> None:
    """Increment mention counts for capitalized phrases; add watch after 3 mentions."""
    text = user_message or ""
    found: set[str] = set()
    for m in re.finditer(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b|\b[A-Z][a-z]{3,}\b",
        text,
    ):
        phrase = m.group(0).strip()
        if len(phrase) < 4 or phrase.lower() in {"what", "when", "where", "angel", "tyler"}:
            continue
        found.add(phrase)
    if not hasattr(core, "_topic_mention_counts"):
        core._topic_mention_counts = {}
    counts: dict[str, int] = core._topic_mention_counts
    for ph in found:
        k = ph.lower()
        counts[k] = counts.get(k, 0) + 1
        if counts[k] == 3:
            try:
                add_watch_item(
                    core.memory_client,
                    core.user_id,
                    core.files_cabinet,
                    core._use_mem0_cloud,
                    label=ph,
                    watch_type="topic",
                    priority="MEDIUM",
                    check_frequency="weekly",
                    auto_added=True,
                )
            except Exception:
                pass


# --- Conversational ---

def detect_proactive_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw:
        return None, {}
    lower = raw.lower()

    if re.search(
        r"(?i)\bwhat\s+are\s+you\s+watching\b|\bwhat'?s\s+on\s+your\s+watch\s+list\b",
        lower,
    ):
        return "list", {}

    m = re.search(
        r"(?i)\b(?:angel\s+)?watch\s+(.+?)\s+closely\b",
        raw,
    )
    if m:
        return "add_close", {"label": m.group(1).strip().rstrip("?.!")}

    m = re.search(r"(?i)\bstop\s+watching\s+(.+)$", raw)
    if m:
        return "stop", {"label": m.group(1).strip().rstrip("?.!")}

    m = re.search(
        r"(?i)\bwhat\s+did\s+you\s+find\s+(?:on|about)\s+(.+)$",
        raw,
    )
    if m:
        return "findings_topic", {"topic": m.group(1).strip().rstrip("?.!")}

    return None, {}


def format_proactive_reply_for_prompt(
    intent: str,
    payload: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> str:
    if intent == "list":
        by_id = fetch_all_watches(memory_client, user_id, use_mem0_cloud)
        rows = [w for w in by_id.values() if w.get("active", True)]
        rows.sort(key=lambda x: (x.get("priority", ""), x.get("label", "")), reverse=True)
        if not rows:
            return "[Angel proactive watch list]\n(none yet — the server can seed defaults on deploy.)"
        lines = [
            f"- [{w.get('priority')}] {w.get('label')} ({w.get('watch_type')}) id={w.get('watch_id')} freq={w.get('check_frequency')}"
            for w in rows[:40]
        ]
        return "[Angel proactive watch list — active items]\n" + "\n".join(lines)

    if intent == "add_close":
        lab = (payload.get("label") or "").strip()
        if not lab:
            return "[Proactive watch] Missing label."
        try:
            w = add_watch_item(
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                label=lab,
                watch_type="topic",
                priority="HIGH",
                check_frequency="daily",
                auto_added=False,
            )
            return (
                f"[Proactive watch] Added HIGH priority daily watch for «{lab}» (watch_id={w.get('watch_id')}). "
                "Tell Tyler you added it to your standing watch list."
            )
        except Exception as e:
            return f"[Proactive watch] Could not add: {e}"

    if intent == "stop":
        lab = (payload.get("label") or "").strip().lower()
        by_id = fetch_all_watches(memory_client, user_id, use_mem0_cloud)
        hit = None
        for w in by_id.values():
            if not w.get("active", True):
                continue
            if lab in (w.get("label") or "").lower():
                hit = w
                break
        if not hit:
            return f"[Proactive watch] No active watch matched {lab!r}."
        deactivate_watch(
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            hit["watch_id"],
        )
        return f"[Proactive watch] Deactivated watch matching «{hit.get('label')}»."

    if intent == "findings_topic":
        topic = (payload.get("topic") or "").strip().lower()
        recent = fetch_recent_findings(memory_client, user_id, use_mem0_cloud, limit=40)
        hits = [
            f
            for f in recent
            if topic in (f.get("label") or "").lower()
            or topic in (f.get("summary") or "").lower()
        ]
        if not hits:
            return f"[Proactive findings] Nothing recent on file for {topic!r}."
        lines = [f"- {(h.get('summary') or '')[:500]}" for h in hits[:6]]
        return "[Proactive findings]\n" + "\n".join(lines)

    return ""
