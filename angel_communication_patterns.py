"""
Batcomputer Layer — Communication pattern analysis (when/how key figures communicate).
Mem0 category comm_pattern + files under Communication Intelligence (CI-*).
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

COMM_INTEL_FOLDER = "Communication Intelligence"
CI_PATTERN_PREFIX = "CI-"
# Coordination filings: CI-{YYYYMMDD}-{hash}
_LAST_COMM_PATTERN_RUN: dict[str, Any] = {"at": None, "result": None}

PLATFORMS = frozenset(
    {
        "twitter",
        "interview",
        "congressional",
        "press_release",
        "documentary",
        "book",
        "podcast",
        "unknown",
    }
)
BASELINE_FREQUENCIES = frozenset({"daily", "weekly", "monthly", "sporadic"})
CURRENT_STATUSES = frozenset(
    {"active", "silent", "escalating", "de-escalating", "unknown"}
)
SIGNIFICANCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

ANOMALY_SILENCE = "SILENCE"
ANOMALY_ESCALATION = "ESCALATION"
ANOMALY_COORDINATION = "COORDINATION"
ANOMALY_POSITION_SHIFT = "POSITION_SHIFT"
ANOMALY_UNUSUAL_VENUE = "UNUSUAL_VENUE"
ANOMALY_COORDINATED_SIGNAL = "COORDINATED_SIGNAL"

DEFAULT_WATCH_ENTITIES: list[str] = [
    "David Grusch",
    "Luis Elizondo",
    "Christopher Mellon",
    "Ross Coulthart",
    "Tim Burchett",
    "Marco Rubio",
    "Avi Loeb",
]


def _safe_int(v: Any, default: int = 7) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def entity_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip())[:72].strip("-").lower()
    return s or f"ent-{hashlib.sha256((name or '').encode()).hexdigest()[:10]}"


def _parse_json_obj(txt: str) -> dict[str, Any] | None:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_pattern(raw: dict[str, Any], *, record_kind: str = "entity") -> dict[str, Any]:
    rk = raw.get("record_kind")
    if isinstance(rk, str) and rk in ("entity", "coordination_signal"):
        record_kind = rk
    pid = (raw.get("pattern_id") or "").strip() or entity_slug(str(raw.get("entity_name") or ""))
    plat = str(raw.get("platform") or "unknown").strip().lower()
    if plat not in PLATFORMS:
        plat = "unknown"
    bf = str(raw.get("baseline_frequency") or "sporadic").strip().lower()
    if bf not in BASELINE_FREQUENCIES:
        bf = "sporadic"
    st = str(raw.get("current_status") or "unknown").strip().lower()
    if st not in CURRENT_STATUSES:
        st = "unknown"
    sig = str(raw.get("significance") or "MEDIUM").strip().upper()
    if sig not in SIGNIFICANCE_LEVELS:
        sig = "MEDIUM"
    anoms = raw.get("anomalies")
    if not isinstance(anoms, list):
        anoms = []
    anoms = [str(a).strip().upper() for a in anoms if str(a).strip()][:20]
    ent_id = raw.get("entity_id")
    if ent_id is not None:
        ent_id = str(ent_id).strip()[:120] or None
    coord_ents = raw.get("entities_involved")
    if not isinstance(coord_ents, list):
        coord_ents = []
    coord_ents = [str(x).strip() for x in coord_ents if str(x).strip()][:30]
    pred_xr = raw.get("prediction_crossrefs")
    if not isinstance(pred_xr, list):
        pred_xr = []
    pred_xr = [str(x).strip() for x in pred_xr if str(x).strip()][:15]
    return {
        "pattern_id": pid[:200],
        "record_kind": record_kind if record_kind in ("entity", "coordination_signal") else "entity",
        "entity_name": str(raw.get("entity_name") or pid).strip()[:500],
        "entity_id": ent_id,
        "platform": plat,
        "baseline_frequency": bf,
        "last_communication_date": str(raw.get("last_communication_date") or "")[:32],
        "current_status": st,
        "pattern_notes": str(raw.get("pattern_notes") or "").strip()[:8000],
        "anomalies": anoms,
        "last_analyzed": str(raw.get("last_analyzed") or _today())[:32],
        "significance": sig,
        "entities_involved": coord_ents,
        "coordination_window_days": _safe_int(raw.get("coordination_window_days"), 7),
        "prediction_crossrefs": pred_xr,
    }


def _pattern_upsert_memory(
    memory_client: Any,
    user_id: str,
    pattern_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_COMM_PATTERN
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-communication-patterns",
        "comm_pattern_id": pattern_id,
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
                and e["metadata"].get("comm_pattern_id") == pattern_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel comm pattern {pattern_id}] {text[:1200]}"},
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


def _sync_entity_pattern_file(files_cabinet: Any, pat: dict[str, Any]) -> None:
    slug = entity_slug(pat.get("entity_name") or pat.get("pattern_id") or "")
    fn = f"{CI_PATTERN_PREFIX}{slug}-pattern"
    body = json.dumps(pat, ensure_ascii=False, indent=2)
    tags = [
        "communication_pattern",
        f"significance:{pat.get('significance', 'MEDIUM')}",
        f"status:{pat.get('current_status', 'unknown')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(COMM_INTEL_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass


def _sync_coordination_file(files_cabinet: Any, fname: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    tags = ["communication_pattern", "coordination_signal", "CI"]
    try:
        if files_cabinet.get_file(fname):
            files_cabinet.update_file(fname, body)
        else:
            files_cabinet.create_file(COMM_INTEL_FOLDER, fname, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fname, body)
        except Exception:
            pass
    except Exception:
        pass


def _load_all_patterns(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict[str, Any]]:
    import angel as ang

    by_id: dict[str, dict[str, Any]] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_COMM_PATTERN:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("pattern_id"):
            by_id[str(obj["pattern_id"])] = _normalize_pattern(obj, record_kind=obj.get("record_kind") or "entity")

    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_COMM_PATTERN:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("pattern_id"):
            by_id[str(obj["pattern_id"])] = _normalize_pattern(obj, record_kind=obj.get("record_kind") or "entity")
    return by_id


def collect_watched_entities(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
    *,
    extra: list[str] | None = None,
) -> list[str]:
    """Default list + HIGH/CRITICAL person nodes from mission network."""
    import angel as ang

    seen: set[str] = set()
    out: list[str] = []
    for n in DEFAULT_WATCH_ENTITIES:
        k = n.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(n.strip())
    try:
        nodes, _edges = ang.network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
        for _nid, node in (nodes or {}).items():
            if not isinstance(node, dict):
                continue
            rel = str(node.get("relevance") or "").upper()
            if rel not in ("HIGH", "CRITICAL"):
                continue
            nt = str(node.get("node_type") or "person").strip().lower()
            if nt != "person":
                continue
            nm = (node.get("name") or "").strip()
            if not nm:
                continue
            lk = nm.lower()
            if lk not in seen:
                seen.add(lk)
                out.append(nm)
    except Exception:
        pass
    if extra:
        for x in extra:
            nm = (x or "").strip()
            if not nm:
                continue
            lk = nm.lower()
            if lk not in seen:
                seen.add(lk)
                out.append(nm)
    return out


def _parse_iso_date(s: str) -> datetime | None:
    s = (s or "").strip()[:10]
    if len(s) < 8:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _within_days(last_date_str: str, days: int = 7) -> bool:
    d = _parse_iso_date(last_date_str)
    if d is None:
        return False
    return datetime.now(timezone.utc) - d <= timedelta(days=days)


def _gather_prediction_snippets(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> str:
    try:
        import angel_predictions as ap

        preds = ap.fetch_all_predictions(memory_client, user_id, use_mem0_cloud)
        lines: list[str] = []
        for p in (preds or {}).values():
            if not isinstance(p, dict):
                continue
            if str(p.get("status") or "active").lower() != "active":
                continue
            pid = (p.get("prediction_id") or "")[:16]
            title = (p.get("title") or "")[:200]
            fp = (p.get("full_prediction") or "")[:400]
            if title or fp:
                lines.append(f"- [{pid}] {title}\n  {fp}")
        return "\n".join(lines[:25])
    except Exception:
        return ""


def analyze_communication_patterns(
    entity_list: list[str] | None,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """
    Tavily (2 queries/entity) + Claude — update CommunicationPattern records, detect coordination.
    """
    import angel as ang

    api_key = (__import__("os").getenv("TAVILY_API_KEY") or "").strip()
    out: dict[str, Any] = {
        "ok": True,
        "at": _now_iso(),
        "entities_analyzed": [],
        "patterns": [],
        "coordination_signals": [],
        "errors": [],
        "prediction_crossrefs": [],
    }
    if not api_key:
        out["ok"] = False
        out["error"] = "TAVILY_API_KEY not set"
        return out

    year = datetime.now(timezone.utc).year
    try:
        year = int((__import__("os").getenv("COMM_PATTERN_SEARCH_YEAR") or str(year)).strip())
    except ValueError:
        pass

    entities = collect_watched_entities(
        memory_client, user_id, use_mem0_cloud, files_cabinet, extra=entity_list
    )
    nodes, _edges = ang.network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
    name_to_id: dict[str, str] = {}
    for nid, node in (nodes or {}).items():
        if isinstance(node, dict) and (node.get("name") or "").strip():
            name_to_id[(node.get("name") or "").strip().lower()] = str(nid)

    pred_ctx = _gather_prediction_snippets(memory_client, user_id, use_mem0_cloud)

    for name in entities:
        slug = entity_slug(name)
        eid = name_to_id.get(name.lower())
        lines: list[str] = []
        seen_u: set[str] = set()
        for q in (
            f"{name} statement interview {year}",
            f"{name} latest news appearance",
        ):
            try:
                for r in ang._tavily_search_one(q, api_key, max_results=5, search_depth="basic"):
                    if not isinstance(r, dict):
                        continue
                    url = (r.get("url") or "").strip()
                    if not url or url in seen_u:
                        continue
                    seen_u.add(url)
                    lines.append(
                        f"{(r.get('title') or '')[:220]}\n{(r.get('content') or r.get('snippet') or '')[:900]}\n{url}"
                    )
                    if len(lines) >= 10:
                        break
            except Exception as e:
                out["errors"].append(f"{name} tavily: {e}")
            if len(lines) >= 10:
                break

        bundle = "\n\n".join(lines)[:45_000]
        sys_eval = """You analyze PUBLIC communication patterns for a named figure (open sources only).
Output ONE JSON object only (no markdown):
{
  "entity_name": "display name",
  "platform": "twitter|interview|congressional|press_release|documentary|book|podcast|unknown",
  "baseline_frequency": "daily|weekly|monthly|sporadic",
  "last_communication_date": "YYYY-MM-DD if inferable from snippets, else empty string",
  "current_status": "active|silent|escalating|de-escalating|unknown",
  "pattern_notes": "2-6 sentences: cadence, tone shift, venue — open sources only",
  "anomalies": ["SILENCE","ESCALATION","POSITION_SHIFT","UNUSUAL_VENUE"],
  "significance": "LOW|MEDIUM|HIGH|CRITICAL"
}
Rules:
- SILENCE: no credible recent public statements vs expected cadence for this person.
- ESCALATION: clearly more interviews/posts/statements than their typical baseline.
- POSITION_SHIFT: stance or framing changed vs prior public positioning (say so carefully).
- UNUSUAL_VENUE: unexpected channel (e.g. only podcast vs usual congressional).
Use only the search excerpts; mark uncertainty explicitly."""

        user_eval = f"FIGURE: {name}\nNETWORK_NODE_ID (if any): {eid or 'none'}\n\nEXCERPTS:\n{bundle or '(no results)'}"

        try:
            resp = anthropic_client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.15,
                system=sys_eval,
                messages=[{"role": "user", "content": user_eval}],
            )
            txt = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    txt += block.text
            prof = _parse_json_obj(txt)
            if not prof:
                out["errors"].append(f"{name}: no JSON from model")
                continue
            prof["pattern_id"] = slug
            prof["entity_id"] = eid
            prof["last_analyzed"] = _today()
            if prof.get("entity_name") is None or str(prof.get("entity_name")).strip() == "":
                prof["entity_name"] = name
            pat = _normalize_pattern(prof, record_kind="entity")
            text = json.dumps(pat, ensure_ascii=False)
            _pattern_upsert_memory(memory_client, user_id, pat["pattern_id"], text, use_mem0_cloud)
            _sync_entity_pattern_file(files_cabinet, pat)
            out["entities_analyzed"].append(name)
            out["patterns"].append(pat)
        except Exception as e:
            out["errors"].append(f"{name}: {e}")
        time.sleep(0.15)

    # Coordination: 2+ figures with activity in 7d window + related narrative
    recent_active: list[dict[str, Any]] = []
    for pat in out["patterns"]:
        if pat.get("record_kind") != "entity":
            continue
        last_d = pat.get("last_communication_date") or ""
        st = (pat.get("current_status") or "").lower()
        if _within_days(last_d, 7) or st in ("active", "escalating"):
            recent_active.append(pat)

    if len(recent_active) >= 2:
        brief = json.dumps(
            [
                {
                    "name": p.get("entity_name"),
                    "last": p.get("last_communication_date"),
                    "status": p.get("current_status"),
                    "notes": (p.get("pattern_notes") or "")[:500],
                    "anomalies": p.get("anomalies") or [],
                }
                for p in recent_active[:14]
            ],
            ensure_ascii=False,
        )[:12_000]
        csys = """You output JSON only:
{"coordinated_clusters":[{"entities":["Name A","Name B"],"related":true/false,"rationale":"short","theme":"UAP/disclosure/etc"}],"prediction_crossrefs":["short labels matching any active prediction themes — or empty"]}
related=true only if open-source timing/content suggests coordinated messaging (same news cycle, joint narrative, not coincidence). Max 3 clusters."""
        cuser = f"RECENT_ACTIVE (7d context):\n{brief}\n\nACTIVE_PREDICTIONS (titles):\n{pred_ctx or '(none)'}"
        try:
            cr = anthropic_client.messages.create(
                model=model,
                max_tokens=1200,
                temperature=0.1,
                system=csys,
                messages=[{"role": "user", "content": cuser}],
            )
            ct = ""
            for block in cr.content:
                if getattr(block, "type", None) == "text":
                    ct += block.text
            cobj = _parse_json_obj(ct)
            clusters = (cobj or {}).get("coordinated_clusters") if isinstance(cobj, dict) else None
            pxr = (cobj or {}).get("prediction_crossrefs") if isinstance(cobj, dict) else []
            if not isinstance(clusters, list):
                clusters = []
            if not isinstance(pxr, list):
                pxr = []
            for cl in clusters:
                if not isinstance(cl, dict):
                    continue
                if not cl.get("related"):
                    continue
                ents = cl.get("entities")
                if not isinstance(ents, list) or len(ents) < 2:
                    continue
                ents = [str(x).strip() for x in ents if str(x).strip()][:12]
                h = hashlib.sha256(
                    json.dumps(ents, sort_keys=True).encode()
                ).hexdigest()[:8]
                day = datetime.now(timezone.utc).strftime("%Y%m%d")
                fname = f"{CI_PATTERN_PREFIX}{day}-{h}"
                pid = f"coord-{day}-{h}"
                coord_rec = _normalize_pattern(
                    {
                        "pattern_id": pid,
                        "entity_name": f"COORDINATED_SIGNAL: {', '.join(ents[:6])}",
                        "entity_id": None,
                        "platform": "unknown",
                        "baseline_frequency": "sporadic",
                        "last_communication_date": _today(),
                        "current_status": "active",
                        "pattern_notes": (cl.get("rationale") or "")[:4000],
                        "anomalies": [ANOMALY_COORDINATED_SIGNAL],
                        "last_analyzed": _today(),
                        "significance": "HIGH",
                        "entities_involved": ents,
                        "coordination_window_days": 7,
                        "prediction_crossrefs": [str(x) for x in pxr[:10] if str(x).strip()],
                    },
                    record_kind="coordination_signal",
                )
                tjson = json.dumps(coord_rec, ensure_ascii=False)
                _pattern_upsert_memory(memory_client, user_id, coord_rec["pattern_id"], tjson, use_mem0_cloud)
                _sync_coordination_file(files_cabinet, fname, coord_rec)
                out["coordination_signals"].append(coord_rec)
                out["prediction_crossrefs"].extend(coord_rec.get("prediction_crossrefs") or [])
        except Exception as e:
            out["errors"].append(f"coordination: {e}")

    global _LAST_COMM_PATTERN_RUN
    _LAST_COMM_PATTERN_RUN = {"at": out.get("at"), "result": dict(out)}
    return out


def list_entity_patterns(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    rows = []
    for p in _load_all_patterns(memory_client, user_id, use_mem0_cloud).values():
        if p.get("record_kind") == "coordination_signal":
            continue
        rows.append(p)
    try:
        rows.sort(key=lambda x: (x.get("last_analyzed") or "", x.get("entity_name") or ""))
    except Exception:
        pass
    return rows


def list_coordination_signals(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    rows = [p for p in _load_all_patterns(memory_client, user_id, use_mem0_cloud).values() if p.get("record_kind") == "coordination_signal"]
    try:
        rows.sort(key=lambda x: -(x.get("last_analyzed") or ""))
    except Exception:
        pass
    return rows


def get_pattern_for_entity_name(
    name: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    if not (name or "").strip():
        return None
    by_id = _load_all_patterns(memory_client, user_id, use_mem0_cloud)
    slug = entity_slug(name)
    if slug in by_id and by_id[slug].get("record_kind") != "coordination_signal":
        return by_id[slug]
    nlow = name.strip().lower()
    best: dict[str, Any] | None = None
    for p in by_id.values():
        if p.get("record_kind") == "coordination_signal":
            continue
        en = (p.get("entity_name") or "").lower()
        pid = (p.get("pattern_id") or "").lower()
        if nlow == en or nlow in en or en in nlow or nlow in pid or pid.startswith(nlow):
            best = p
            if nlow == en:
                return p
    return best


def list_current_anomalies(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in list_entity_patterns(memory_client, user_id, use_mem0_cloud):
        an = p.get("anomalies") or []
        if an:
            out.append(
                {
                    "entity_name": p.get("entity_name"),
                    "pattern_id": p.get("pattern_id"),
                    "significance": p.get("significance"),
                    "anomalies": an,
                    "pattern_notes": (p.get("pattern_notes") or "")[:1200],
                }
            )
    return out


def format_communication_patterns_for_briefing(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    last_run: dict[str, Any] | None = None,
) -> str:
    """
    SILENCE+CRITICAL always; coordination always; ESCALATION if 2+ entities same window.
    """
    if last_run is None:
        lr = _LAST_COMM_PATTERN_RUN.get("result")
        last_run = lr if isinstance(lr, dict) else None

    lines: list[str] = []
    if last_run and last_run.get("ok"):
        lines.append(
            f"Communication pattern scan: {len(last_run.get('entities_analyzed') or [])} figure(s), "
            f"{len(last_run.get('coordination_signals') or [])} coordination signal(s)."
        )

    patterns = list_entity_patterns(memory_client, user_id, use_mem0_cloud)
    coords = list_coordination_signals(memory_client, user_id, use_mem0_cloud)

    for c in coords[:5]:
        ents = ", ".join((c.get("entities_involved") or [])[:8])
        lines.append(f"*** COORDINATED SIGNAL: {ents} — {(c.get('pattern_notes') or '')[:280]}")
        px = c.get("prediction_crossrefs") or []
        if px:
            lines.append(f"   Prediction cross-reference: {'; '.join(px[:4])}")

    silence_crit = [
        p
        for p in patterns
        if ANOMALY_SILENCE in (p.get("anomalies") or [])
        and (p.get("significance") or "").upper() == "CRITICAL"
    ]
    for p in silence_crit[:6]:
        lines.append(
            f"CRITICAL silence watch: {p.get('entity_name')} — {(p.get('pattern_notes') or '')[:220]}"
        )

    esc = [p for p in patterns if ANOMALY_ESCALATION in (p.get("anomalies") or [])]
    if len(esc) >= 2:
        lines.append("ESCALATION cluster (2+ figures spiking activity vs baseline):")
        for p in esc[:8]:
            lines.append(f"- {p.get('entity_name')}: {(p.get('pattern_notes') or '')[:160]}")

    if len(lines) <= (1 if last_run and last_run.get("ok") else 0):
        return ""
    return "COMMUNICATION INTELLIGENCE (pattern / cadence — open sources)\n" + "\n".join(lines[:24])


def detect_comms_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(
        r"(?i)\b(?:communication\s+)?landscape\b|pattern\s+summary|comm(?:unication)?\s+patterns?\b",
        msg,
    ):
        return "landscape", {}
    if re.search(r"(?i)\bcoordinated\s+activit", msg):
        return "coordination", {}
    if re.search(r"(?i)\bgoing\s+silent\b|\bsilence\b.*\b(anomaly|anyone|figures?)", msg):
        return "silence", {}
    m = re.search(
        r"(?i)\bwhat(?:'s| is)\s+([\w\s\-]+?)\s+saying\s+lately\b",
        msg,
    )
    if m:
        return "entity_lately", {"name": m.group(1).strip()}
    m_is = re.search(r"(?i)\bis\s+([\w\s\-]+?)\s+saying\s+lately\b", msg)
    if m_is:
        return "entity_lately", {"name": m_is.group(1).strip()}
    return None, {}


def format_comms_chat_block(
    intent: str,
    payload: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> str:
    try:
        if intent == "landscape":
            patterns = list_entity_patterns(memory_client, user_id, use_mem0_cloud)
            coords = list_coordination_signals(memory_client, user_id, use_mem0_cloud)
            block = {
                "entity_patterns": patterns[:20],
                "coordination_signals": coords[:10],
                "note": "Open-source cadence/timing analysis — not private communications.",
            }
            return "[Angel communication patterns]\n" + json.dumps(block, ensure_ascii=False, indent=2)[:14000]
        if intent == "coordination":
            coords = list_coordination_signals(memory_client, user_id, use_mem0_cloud)
            return "[Angel communication patterns — coordination]\n" + json.dumps(
                {"coordination_signals": coords[:15]}, ensure_ascii=False, indent=2
            )[:14000]
        if intent == "silence":
            silent = [
                p
                for p in list_entity_patterns(memory_client, user_id, use_mem0_cloud)
                if ANOMALY_SILENCE in (p.get("anomalies") or [])
            ]
            return "[Angel communication patterns — silence]\n" + json.dumps(
                {"figures_with_silence_anomaly": silent[:20]}, ensure_ascii=False, indent=2
            )[:14000]
        if intent == "entity_lately":
            name = (payload.get("name") or "").strip()
            p = get_pattern_for_entity_name(name, memory_client, user_id, use_mem0_cloud)
            return "[Angel communication patterns — entity]\n" + json.dumps(
                p or {"notice": f"No saved pattern for {name!r}. Run GET /api/comms/run or wait for scheduled scan."},
                ensure_ascii=False,
                indent=2,
            )[:12000]
    except Exception as e:
        return f"[Angel communication patterns error]\n{str(e)[:500]}"
    return ""


def run_scheduled_analysis(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Wrapper for scheduler / API (uses default + network entity list)."""
    return analyze_communication_patterns(
        None,
        anthropic_client,
        memory_client,
        user_id,
        files_cabinet,
        use_mem0_cloud,
    )
