"""
Batcomputer Layer — Threat Actor Database (opposition / anti-disclosure network).
Mem0 category threat_actor + mirrored files TA-{actor_id} under Threat Actors.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import anthropic

THREAT_ACTORS_FOLDER = "Threat Actors"
TA_FILE_PREFIX = "TA-"

ACTOR_TYPES = frozenset({"person", "organization", "program", "faction"})
THREAT_TYPES = frozenset(
    {
        "suppression",
        "disinformation",
        "retaliation",
        "classification",
        "obstruction",
        "surveillance",
        "unknown",
    }
)
THREAT_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
STATUSES = frozenset({"active", "inactive", "unknown"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def actor_id_from_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip())[:72].strip("-").lower()
    return s or f"actor-{hashlib.sha256((name or '').encode()).hexdigest()[:10]}"


def _normalize_actor(raw: dict[str, Any]) -> dict[str, Any]:
    aid = (raw.get("actor_id") or "").strip() or actor_id_from_name(str(raw.get("name") or ""))
    tl = str(raw.get("threat_level") or "MEDIUM").strip().upper()
    if tl not in THREAT_LEVELS:
        tl = "MEDIUM"
    tt = str(raw.get("threat_type") or "unknown").strip().lower()
    if tt not in THREAT_TYPES:
        tt = "unknown"
    at = str(raw.get("actor_type") or "organization").strip().lower()
    if at not in ACTOR_TYPES:
        at = "organization"
    st = str(raw.get("status") or "active").strip().lower()
    if st not in STATUSES:
        st = "unknown"
    ka = raw.get("known_actions")
    if not isinstance(ka, list):
        ka = [str(ka)] if ka else []
    ka = [str(x).strip() for x in ka if str(x).strip()][:80]
    ev = raw.get("evidence")
    if not isinstance(ev, list):
        ev = [str(ev)] if ev else []
    ev = [str(x).strip() for x in ev if str(x).strip()][:80]
    return {
        "actor_id": aid,
        "name": str(raw.get("name") or aid).strip()[:500],
        "actor_type": at,
        "role": str(raw.get("role") or "").strip()[:2000],
        "affiliation": str(raw.get("affiliation") or "").strip()[:2000],
        "threat_type": tt,
        "threat_level": tl,
        "known_actions": ka,
        "evidence": ev,
        "status": st,
        "first_identified": str(raw.get("first_identified") or _today_date())[:32],
        "last_updated": str(raw.get("last_updated") or _today_date())[:32],
        "notes": str(raw.get("notes") or "").strip()[:8000],
    }


def _threat_actor_upsert_memory(
    memory_client: Any,
    user_id: str,
    actor_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_THREAT_ACTOR
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-threat-actors",
        "threat_actor_id": actor_id,
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
                and e["metadata"].get("threat_actor_id") == actor_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel threat actor {actor_id}] {text[:1200]}"},
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


def _sync_ta_intel_file(files_cabinet: Any, actor: dict[str, Any]) -> None:
    aid = (actor.get("actor_id") or "").strip()
    if not aid:
        return
    fn = f"{TA_FILE_PREFIX}{aid}"
    body = json.dumps(actor, ensure_ascii=False, indent=2)
    tags = [
        "threat_actor",
        f"threat_level:{actor.get('threat_level', 'MEDIUM')}",
        f"threat_type:{actor.get('threat_type', 'unknown')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(THREAT_ACTORS_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass


def _load_all_actors(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict[str, Any]]:
    import angel as ang

    by_id: dict[str, dict[str, Any]] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_THREAT_ACTOR:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("actor_id"):
            by_id[str(obj["actor_id"])] = _normalize_actor(obj)

    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_THREAT_ACTOR:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("actor_id"):
            by_id[str(obj["actor_id"])] = _normalize_actor(obj)

    return by_id


def sync_threat_actor_to_network(
    actor: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> None:
    """Mirror threat actor as a network node; optional edges (e.g. retaliation → target)."""
    import angel as ang

    aid = (actor.get("actor_id") or "").strip()
    if not aid:
        return
    atype = actor.get("actor_type") or "organization"
    # Network uses person|organization|program|event|faction
    nt = atype if atype in ang.NETWORK_NODE_TYPES else "organization"
    desc = "\n".join(
        [
            f"Threat actor ({actor.get('threat_type', 'unknown')}) — {actor.get('role', '')}"[:2000],
            f"Affiliation: {actor.get('affiliation', '')}"[:500],
        ]
    ).strip()
    tags = [
        "threat_actor",
        f"threat_level:{actor.get('threat_level', 'MEDIUM')}",
        f"threat_type:{actor.get('threat_type', 'unknown')}",
    ]
    try:
        ang.add_network_node(
            actor.get("name") or aid,
            nt,
            desc,
            str(actor.get("threat_level") or "MEDIUM").upper(),
            tags,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            node_id_override=aid,
        )
    except Exception:
        pass


def add_retaliation_edge_to_grusch(
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> None:
    """Seed edge: grusch retaliation network —retaliates_against→ David Grusch."""
    import angel as ang

    src = "grusch-retaliation-network"
    desc = "Alleged retaliation against whistleblower David Grusch (open-source / testimony context)."
    try:
        nodes, edges = ang.network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
        sid = ang._network_normalize_node_id(src)
        tid = ang._network_normalize_node_id("David Grusch")
        for e in edges:
            if (
                e.get("source_id") == sid
                and e.get("target_id") == tid
                and (e.get("relationship_type") or "") == "retaliates_against"
            ):
                return
    except Exception:
        pass
    try:
        ang.add_network_edge(
            src,
            "David Grusch",
            "retaliates_against",
            desc,
            "STRONG",
            "Public testimony and reporting; not classified sourcing.",
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    except Exception:
        pass


def add_threat_actor(
    name: str,
    actor_type: str,
    role: str,
    affiliation: str,
    threat_type: str,
    threat_level: str,
    known_actions: list[str] | None,
    evidence: list[str] | None,
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    notes: str = "",
    status: str = "active",
    actor_id: str | None = None,
    sync_network: bool = True,
) -> dict[str, Any]:
    raw = {
        "actor_id": actor_id or actor_id_from_name(name),
        "name": name,
        "actor_type": actor_type,
        "role": role,
        "affiliation": affiliation,
        "threat_type": threat_type,
        "threat_level": threat_level,
        "known_actions": known_actions or [],
        "evidence": evidence or [],
        "status": status,
        "first_identified": _today_date(),
        "last_updated": _today_date(),
        "notes": notes,
    }
    actor = _normalize_actor(raw)
    text = json.dumps(actor, ensure_ascii=False)
    _threat_actor_upsert_memory(memory_client, user_id, actor["actor_id"], text, use_mem0_cloud)
    _sync_ta_intel_file(files_cabinet, actor)
    if sync_network:
        sync_threat_actor_to_network(
            actor,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    return actor


def get_threat_actor(
    actor_id: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    aid = (actor_id or "").strip()
    if not aid:
        return None
    all_a = _load_all_actors(memory_client, user_id, use_mem0_cloud)
    return all_a.get(aid)


def update_threat_actor(
    actor_id: str,
    updates: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    sync_network: bool = True,
) -> dict[str, Any] | None:
    cur = get_threat_actor(actor_id, memory_client, user_id, use_mem0_cloud)
    if not cur:
        return None
    merged = dict(cur)
    merged.update(updates)
    merged["last_updated"] = _today_date()
    actor = _normalize_actor(merged)
    text = json.dumps(actor, ensure_ascii=False)
    _threat_actor_upsert_memory(memory_client, user_id, actor["actor_id"], text, use_mem0_cloud)
    _sync_ta_intel_file(files_cabinet, actor)
    if sync_network:
        sync_threat_actor_to_network(
            actor,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    return actor


def list_threat_actors(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    threat_level: str | None = None,
) -> list[dict[str, Any]]:
    all_a = _load_all_actors(memory_client, user_id, use_mem0_cloud)
    rows = list(all_a.values())
    if threat_level:
        tl = threat_level.strip().upper()
        rows = [a for a in rows if (a.get("threat_level") or "").upper() == tl]
    try:
        rows.sort(key=lambda x: (x.get("name") or "").lower())
    except Exception:
        pass
    return rows


def search_threat_actors(
    query: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    out: list[dict[str, Any]] = []
    for a in _load_all_actors(memory_client, user_id, use_mem0_cloud).values():
        blob = " ".join(
            [
                str(a.get("name", "")),
                str(a.get("role", "")),
                str(a.get("affiliation", "")),
                str(a.get("notes", "")),
                " ".join(a.get("known_actions") or []),
            ]
        ).lower()
        if q in blob:
            out.append(a)
    return out


def get_threat_actor_summary(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    rows = list_threat_actors(memory_client, user_id, use_mem0_cloud)
    by_level: dict[str, int] = {}
    for a in rows:
        lv = (a.get("threat_level") or "MEDIUM").upper()
        by_level[lv] = by_level.get(lv, 0) + 1
    return {
        "total": len(rows),
        "by_threat_level": by_level,
        "actors": [{"actor_id": x["actor_id"], "name": x["name"], "threat_level": x.get("threat_level")} for x in rows],
    }


def assess_new_threat_actor(
    anthropic_client: anthropic.Anthropic,
    name: str,
    context: str,
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Tavily + Claude → structured ThreatActor; saved if ok."""
    import angel as ang

    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "name required"}
    api_key = (__import__("os").getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    q1 = f"{name} UAP disclosure obstruction retaliation disinformation"
    q2 = f"{name} government transparency FOIA classification"
    bundle_lines: list[str] = []
    seen: set[str] = set()
    for q in (q1, q2):
        for r in ang._tavily_search_one(q, api_key, max_results=5, search_depth="basic"):
            if not isinstance(r, dict):
                continue
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            bundle_lines.append(
                f"{(r.get('title') or '')}\nURL: {url}\n{(r.get('content') or '')[:900]}"
            )
        if len(bundle_lines) >= 8:
            break
    bundle = "\n\n---\n\n".join(bundle_lines)[:60_000]
    ctx = (context or "").strip() or "Assess as potential opposition to UAP disclosure / Tyler's transparency mission (open sources only)."

    sys = """You output ONLY valid JSON (no markdown) for a threat actor profile:
{
  "name": "display name",
  "actor_type": "person" | "organization" | "program" | "faction",
  "role": "short",
  "affiliation": "string",
  "threat_type": "suppression" | "disinformation" | "retaliation" | "classification" | "obstruction" | "surveillance" | "unknown",
  "threat_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "known_actions": ["documented open-source actions re: disclosure/transparency"],
  "evidence": ["short source lines with URLs if present in bundle"],
  "status": "active" | "inactive" | "unknown",
  "notes": "careful caveats — open source only"
}
Be conservative; if evidence is thin, use LOW/MEDIUM and say so in notes."""

    user = f"SUBJECT: {name}\nCONTEXT: {ctx[:4000]}\n\nOPEN WEB SNIPPETS:\n{bundle or '(no results)'}"

    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=sys,
            messages=[{"role": "user", "content": user}],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                txt += block.get("text", "")
        txt = (txt or "").strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```\s*$", "", txt)
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return {"ok": False, "error": "no JSON in model response", "raw": txt[:1500]}
        prof = json.loads(m.group(0))
        if not isinstance(prof, dict):
            return {"ok": False, "error": "invalid JSON shape"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    prof["actor_id"] = actor_id_from_name(prof.get("name") or name)
    actor = _normalize_actor(prof)
    text = json.dumps(actor, ensure_ascii=False)
    _threat_actor_upsert_memory(memory_client, user_id, actor["actor_id"], text, use_mem0_cloud)
    _sync_ta_intel_file(files_cabinet, actor)
    sync_threat_actor_to_network(
        actor,
        memory_client=memory_client,
        user_id=user_id,
        files_cabinet=files_cabinet,
        use_mem0_cloud=use_mem0_cloud,
    )
    return {"ok": True, "actor": actor}


# --- Seeds ---

_SEED_ACTORS: list[dict[str, Any]] = [
    {
        "actor_id": "aaro",
        "name": "AARO (All-domain Anomaly Resolution Office)",
        "actor_type": "organization",
        "role": "DoD office tasked with UAP data collection and public reporting",
        "affiliation": "U.S. Department of Defense",
        "threat_type": "classification",
        "threat_level": "HIGH",
        "known_actions": [
            "Public messaging has often attributed UAP to conventional explanations; critics allege insufficient engagement with whistleblower and pilot testimony.",
            "Per open debate, perceived as downplaying or dismissing some disclosure-aligned claims without full public evidence.",
        ],
        "evidence": [
            "Open-source policy reporting and public hearings on AARO's role (verify against primary sources).",
        ],
        "status": "active",
        "notes": "Assessed in opposition-network framing: institutional posture toward disclosure is contested in public discourse—not a legal finding of wrongdoing.",
    },
    {
        "actor_id": "pentagon-public-affairs",
        "name": "Pentagon Public Affairs",
        "actor_type": "organization",
        "role": "Coordinated public communications for the Department of Defense",
        "affiliation": "U.S. Department of Defense",
        "threat_type": "disinformation",
        "threat_level": "HIGH",
        "known_actions": [
            "Historical and ongoing press narratives around UAP that critics describe as denial or minimization campaigns.",
            "Public statements framing witnesses or leaks in ways transparency advocates dispute.",
        ],
        "evidence": ["Media analysis and hearing transcripts (open sources)."],
        "status": "active",
        "notes": "Framed as narrative opposition to full transparency in public debate.",
    },
    {
        "actor_id": "cia-classification-authority",
        "name": "CIA Classification Authority (institutional)",
        "actor_type": "organization",
        "role": "Classification and control of sensitive intelligence holdings",
        "affiliation": "U.S. Intelligence Community",
        "threat_type": "classification",
        "threat_level": "HIGH",
        "known_actions": [
            "FOIA delays and redactions affecting UAP-related requests (as reported in open advocacy and journalism).",
            "Institutional control over evidence that disclosure advocates want declassified.",
        ],
        "evidence": ["FOIA reporting and oversight letters (open sources)."],
        "status": "active",
        "notes": "Institutional role—not an accusation against named individuals without sourcing.",
    },
    {
        "actor_id": "unnamed-ic-contractors",
        "name": "Unnamed IC / defense contractors (alleged programs)",
        "actor_type": "faction",
        "role": "Alleged compartmented programs and reverse-engineering efforts",
        "affiliation": "Contractor ecosystem / alleged special access programs",
        "threat_type": "suppression",
        "threat_level": "CRITICAL",
        "known_actions": [
            "Per David Grusch and related open testimony/allegations: illegal or non-disclosed UAP programs outside normal oversight (alleged; not adjudicated here).",
        ],
        "evidence": [
            "Congressional hearing testimony and investigative journalism (verify primary sources).",
        ],
        "status": "unknown",
        "notes": "Highly contested; track as allegation cluster, not established fact.",
    },
    {
        "actor_id": "grusch-retaliation-network",
        "name": "David Grusch retaliation network (alleged)",
        "actor_type": "faction",
        "role": "Actors alleged to have retaliated against whistleblowing on UAP programs",
        "affiliation": "Unknown / institutional",
        "threat_type": "retaliation",
        "threat_level": "HIGH",
        "known_actions": [
            "Grusch publicly described retaliation as 'brutal' in open testimony and interviews (verify exact quotes from primary video/transcripts).",
        ],
        "evidence": ["Public hearing and interview record (open sources)."],
        "status": "active",
        "notes": "Opposition edge to disclosure advocates; ties to David Grusch as alleged target.",
    },
]


def maybe_ensure_threat_actor_seeds(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Idempotent: seed default opposition actors if database is empty."""
    if _load_all_actors(memory_client, user_id, use_mem0_cloud):
        return {"ok": True, "skipped": True, "reason": "already_populated"}
    return ensure_threat_actor_seeds(memory_client, user_id, files_cabinet, use_mem0_cloud)


def ensure_threat_actor_seeds(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    existing = _load_all_actors(memory_client, user_id, use_mem0_cloud)
    added = 0
    for seed in _SEED_ACTORS:
        aid = seed["actor_id"]
        if aid in existing:
            continue
        add_threat_actor(
            seed["name"],
            seed["actor_type"],
            seed["role"],
            seed["affiliation"],
            seed["threat_type"],
            seed["threat_level"],
            seed.get("known_actions") or [],
            seed.get("evidence") or [],
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            notes=seed.get("notes") or "",
            status=seed.get("status") or "active",
            actor_id=aid,
            sync_network=True,
        )
        added += 1
    # One-time edge for Grusch after seeds exist
    try:
        add_retaliation_edge_to_grusch(
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    except Exception:
        pass
    return {"ok": True, "added": added, "total": len(_load_all_actors(memory_client, user_id, use_mem0_cloud))}


# --- OSINT / threat scan hooks ---

def osint_hint_for_threat_actor(
    target: str,
    dossier_excerpt: str,
    mission_relevance: str,
    red_flags: list[str],
) -> str | None:
    """If OSINT suggests obstruction, return a hint line for Tyler."""
    rel = (mission_relevance or "").strip().upper()
    blob = " ".join([target, dossier_excerpt or "", " ".join(red_flags or [])]).lower()
    keywords = (
        "obstruct",
        "suppress",
        "retaliat",
        "classif",
        "deny",
        "disinformation",
        "foia",
        "whistle",
    )
    if rel in ("HIGH", "CRITICAL") and any(k in blob for k in keywords):
        return (
            f"Potential Threat Actor candidate: {target.strip()!r} — OSINT suggests obstruction/suppression themes. "
            f"Consider `POST /api/threat-actors/assess` or ask Angel to assess as a threat actor."
        )
    return None


def append_threat_scan_actor_hint(body: str, threat_level: str) -> str:
    if (threat_level or "").upper() not in ("HIGH", "CRITICAL"):
        return body
    hint = "\n---\nthreat_actor_db_hint: Named parties in this scan may warrant review for the Threat Actors (opposition) database.\n"
    if "threat_actor_db_hint" in body:
        return body
    return body + hint


# --- Chat ---

def detect_threat_actor_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw:
        return None, {}

    if re.search(r"(?i)\bwho(?:'s| is) working against disclosure\b", raw):
        return "list_opposition", {}

    if re.search(
        r"(?i)\b(?:list|show)\s+(?:me\s+)?(?:the\s+)?(?:high|critical)?\s*threat\s+actors\b", raw
    ):
        return "list_opposition", {}

    m = re.search(r"(?i)\bangel\s+assess\s+(.+?)\s+as\s+(?:a\s+)?threat\s+actor\b", raw)
    if m:
        return "assess", {"name": m.group(1).strip().rstrip("?.!")}

    m = re.search(r"(?i)\bassess\s+(.+?)\s+as\s+(?:a\s+)?threat\s+actor\b", raw)
    if m:
        return "assess", {"name": m.group(1).strip().rstrip("?.!")}

    m = re.search(r"(?i)\badd\s+(.+?)\s+to\s+(?:the\s+)?threat\s+actors?\b", raw)
    if m:
        return "add_named", {"name": m.group(1).strip().rstrip("?.!")}

    m = re.search(r"(?i)\bwho\s+is\s+opposing\s+([^\?\n\.]+)", raw)
    if m:
        return "opposing", {"target": m.group(1).strip().rstrip("?.!")}

    return None, {}


def format_threat_actor_chat_block(
    intent: str,
    payload: dict[str, Any],
    *,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> str:
    try:
        if intent == "list_opposition":
            rows = [
                a
                for a in list_threat_actors(memory_client, user_id, use_mem0_cloud)
                if (a.get("threat_level") or "").upper() in ("HIGH", "CRITICAL")
            ]
            block = {
                "threat_actors_high_critical": [
                    {"actor_id": x["actor_id"], "name": x["name"], "threat_level": x.get("threat_level"), "threat_type": x.get("threat_type")}
                    for x in rows
                ],
                "note": "Opposition network (Batcomputer Threat Actors) — open-source assessments only.",
            }
            return "[Angel Threat Actors — opposition / anti-disclosure]\n" + json.dumps(block, ensure_ascii=False, indent=2)[:12000]

        if intent == "assess":
            name = (payload.get("name") or "").strip()
            if not name:
                return ""
            r = assess_new_threat_actor(
                anthropic_client,
                name,
                "Tyler asked in chat to profile this subject as a threat actor.",
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
            return "[Angel Threat Actor assessment]\n" + json.dumps(r, ensure_ascii=False, indent=2)[:12000]

        if intent == "add_named":
            name = (payload.get("name") or "").strip()
            if not name:
                return ""
            return (
                "[Angel Threat Actors]\n"
                + json.dumps(
                    {
                        "notice": f"To add {name!r}, use POST /api/threat-actors/add with role, threat_type, and evidence, "
                        f"or say 'Angel assess {name} as threat actor' for an auto-researched profile."
                    },
                    indent=2,
                )
            )

        if intent == "opposing":
            tgt = (payload.get("target") or "").strip()
            if not tgt:
                return ""
            import angel as ang

            slug = ang.network_slug_from_display_name(tgt)
            conn = ang.get_node_connections(
                slug,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
                files_cabinet=files_cabinet,
            )
            edges = (conn or {}).get("edges") or []
            ta_rows = list_threat_actors(memory_client, user_id, use_mem0_cloud)
            blob = tgt.lower()
            related_ta = []
            for a in ta_rows:
                pack = " ".join(
                    [a.get("name", ""), a.get("notes", ""), " ".join(a.get("known_actions") or [])]
                ).lower()
                if blob in pack or slug in pack:
                    related_ta.append({"actor_id": a["actor_id"], "name": a["name"]})
            out = {
                "target": tgt,
                "network_edges_to_from_target": edges[:30],
                "threat_actors_mentioning_target": related_ta,
            }
            return "[Angel opposition lookup]\n" + json.dumps(out, ensure_ascii=False, indent=2)[:12000]
    except Exception as e:
        return f"[Angel Threat Actors error]\n{str(e)[:800]}"
    return ""
