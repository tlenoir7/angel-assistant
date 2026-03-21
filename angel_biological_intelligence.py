"""
Batcomputer Layer — Biological & medical intelligence (UAP-adjacent physiology, exposure, witness health).
Mem0 category bio_medical + files BIO-* under Biological Intelligence.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic

BIO_INTEL_FOLDER = "Biological Intelligence"
BIO_PREFIX = "BIO-"

EXPOSURE_TYPES = frozenset(
    {"radiation", "electromagnetic", "psychological", "unknown", "multiple"}
)
SEVERITY_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
CONFIDENCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
RECOMMENDED_ACTIONS = frozenset({"file", "investigate", "monitor", "none"})

# --- Seed: documented / publicly discussed UAP-adjacent medical patterns (open sources only) ---
SEED_UAP_MEDICAL_CASES: list[dict[str, Any]] = [
    {
        "case_id": "cash-landrum-1980",
        "title": "Cash-Landrum incident (1980)",
        "incident_year": 1980,
        "region": "Texas, USA",
        "symptoms": [
            "radiation-like burns",
            "hair loss",
            "blisters",
            "long-term health complaints reported in open literature",
        ],
        "exposure_type": "radiation",
        "summary": "Witnesses reported health effects after close encounter with diamond-shaped craft; "
        "long-running debate in open sources about cause — document as pattern only, not adjudicated fact.",
        "pattern_tags": ["burns", "radiation_sickness_symptoms", "hair_loss"],
    },
    {
        "case_id": "rendlesham-penniston-1980",
        "title": "Rendlesham Forest — reported health sequelae (open reporting)",
        "incident_year": 1980,
        "region": "Suffolk, UK",
        "symptoms": [
            "reported radiation exposure concerns",
            "long-term neurological complaints described in public interviews",
        ],
        "exposure_type": "multiple",
        "summary": "Public accounts (e.g. Sgt. Jim Penniston in open media) reference lasting effects; "
        "treat as witness-reported pattern, not medical diagnosis.",
        "pattern_tags": ["neurological", "radiation_concern"],
    },
    {
        "case_id": "hessdalen-em-sensitivity",
        "title": "Hessdalen lights — electromagnetic sensitivity reports",
        "incident_year": None,
        "region": "Hessdalen, Norway",
        "symptoms": ["electromagnetic sensitivity reports in witness literature", "headaches (anecdotal)"],
        "exposure_type": "electromagnetic",
        "summary": "Witnesses in open Hessdalen literature sometimes report EM-related discomfort; "
        "scientific monitoring of lights is separate from symptom claims.",
        "pattern_tags": ["em", "witness_reports"],
    },
    {
        "case_id": "aatip-program-witness-health",
        "title": "AATIP-era reporting — physiological effects on military witnesses (public record)",
        "incident_year": None,
        "region": "United States",
        "symptoms": [
            "brain injury (reported in congressional/media discourse)",
            "nerve damage (reported)",
            "cardiac symptoms (reported)",
        ],
        "exposure_type": "unknown",
        "summary": "Congressional testimony and media summarized alleged physiological effects on "
        "personnel associated with UAP encounters — pattern only; verify claims against primary sources.",
        "pattern_tags": ["military_witness", "physiological", "congressional_discourse"],
    },
    {
        "case_id": "grusch-testimony-injuries",
        "title": "Grusch testimony — colleagues injured (public testimony context)",
        "incident_year": None,
        "region": "United States",
        "symptoms": ["physical injury (reported in open testimony)", "occupational harm (alleged)"],
        "exposure_type": "unknown",
        "summary": "Open-source reporting on whistleblower testimony referencing injuries; "
        "do not infer classified details beyond public statements.",
        "pattern_tags": ["testimony", "workplace_injury_allegations"],
    },
    {
        "case_id": "general-uap-medical-pattern",
        "title": "Recurring open-source symptom pattern in UAP encounter literature",
        "incident_year": None,
        "region": "Various",
        "symptoms": [
            "burns",
            "radiation sickness-like symptoms (reported)",
            "eye irritation or vision changes (reported)",
            "neurological symptoms (reported)",
            "PTSD and anxiety",
            "unusual dreams or vivid imagery (reported)",
        ],
        "exposure_type": "multiple",
        "summary": "Aggregated pattern from public UAP literature — not a single case; "
        "use for similarity matching only.",
        "pattern_tags": ["burns", "radiation", "ptsd", "neuro", "dreams"],
    },
]

BLACK_EYED_PROFILE: dict[str, Any] = {
    "profile_id": "black-eyed-profile",
    "title": "Black-eyed people (BEK) phenomenon — medical / psychological profile (open literature)",
    "tyler_reference": (
        "Tyler reports a childhood encounter consistent with public 'black-eyed people' narratives; "
        "treat as personally significant lived experience and as a research anchor — not as "
        "clinically verified diagnosis."
    ),
    "documented_psychological_effects": [
        "Intense fear / dread response",
        "Anxiety and hypervigilance after encounter (anecdotal reports)",
        "Sleep disturbance in some narratives",
    ],
    "physiological_responses_reported": [
        "Acute stress response (fight/flight)",
        "Paralysis or 'frozen' feeling (subjective reports)",
        "Pupil appearance described as anomalous black — optical/perceptual interpretation varies",
    ],
    "pattern_across_cases": [
        "Unexpected visitors at door or vehicle",
        "Request for entry (narrative trope)",
        "Compulsion or persuasion attempts (reported)",
    ],
    "theories_mechanism": [
        "Psychological: sleep paralysis, hypnagogia, social contagion of narrative",
        "Cultural: folklore / urban legend dynamics",
        "Unknown: retained as epistemic bucket when evidence does not resolve mechanism",
    ],
    "connections_to_other_profiles": [
        "Overlaps with high-strangeness encounter narratives (stress, altered perception)",
        "May be compared cautiously to UAP witness stress responses — distinct phenomenology",
    ],
    "scientific_rigor_note": (
        "Angel summarizes open-source and witness-reported patterns. She does not assert "
        "paranormal etiology; she distinguishes observation, interpretation, and uncertainty."
    ),
    "last_updated": None,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def case_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip())[:72].strip("-").lower()
    return s or f"bio-{hashlib.sha256((name or '').encode()).hexdigest()[:10]}"


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


def _normalize_analysis_result(raw: dict[str, Any]) -> dict[str, Any]:
    sym = raw.get("symptoms_identified")
    if not isinstance(sym, list):
        sym = []
    sym = [str(x).strip() for x in sym if str(x).strip()][:40]
    timeline = str(raw.get("symptom_timeline_notes") or "").strip()[:4000]
    anom = raw.get("anomalous_biological_markers")
    if not isinstance(anom, list):
        anom = []
    anom = [str(x).strip() for x in anom if str(x).strip()][:20]
    et = str(raw.get("exposure_type") or "unknown").strip().lower()
    if et not in EXPOSURE_TYPES:
        et = "unknown"
    sev = str(raw.get("severity") or "MEDIUM").strip().upper()
    if sev not in SEVERITY_LEVELS:
        sev = "MEDIUM"
    conf = str(raw.get("confidence") or "MEDIUM").strip().upper()
    if conf not in CONFIDENCE_LEVELS:
        conf = "MEDIUM"
    act = str(raw.get("recommended_action") or "monitor").strip().lower()
    if act not in RECOMMENDED_ACTIONS:
        act = "monitor"
    kcs = raw.get("known_case_similarities")
    if not isinstance(kcs, list):
        kcs = []
    kcs = [str(x).strip() for x in kcs if str(x).strip()][:20]
    return {
        "symptoms_identified": sym,
        "symptom_timeline_notes": timeline,
        "anomalous_biological_markers": anom,
        "exposure_type": et,
        "severity": sev,
        "pattern_match": bool(raw.get("pattern_match")),
        "known_case_similarities": kcs,
        "mission_relevance": str(raw.get("mission_relevance") or "").strip()[:4000],
        "recommended_action": act,
        "confidence": conf,
    }


def _bio_case_upsert_memory(
    memory_client: Any,
    user_id: str,
    case_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_BIO_MEDICAL
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-biological-intelligence",
        "bio_case_id": case_id,
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
                and e["metadata"].get("bio_case_id") == case_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel bio case {case_id}] {text[:1200]}"},
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


def _sync_bio_file(files_cabinet: Any, case_id: str, body: str | dict, *, tags: list[str] | None = None) -> None:
    fn = f"{BIO_PREFIX}{case_id}"
    if isinstance(body, dict):
        text = json.dumps(body, ensure_ascii=False, indent=2)
    else:
        text = str(body)
    tagl = tags or ["biological_intelligence", f"case:{case_id}"]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, text)
        else:
            files_cabinet.create_file(BIO_INTEL_FOLDER, fn, text, tags=tagl)
    except ValueError:
        try:
            files_cabinet.update_file(fn, text)
        except Exception:
            pass
    except Exception:
        pass


def _load_all_bio_cases(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict[str, Any]]:
    import angel as ang

    by_id: dict[str, dict[str, Any]] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_BIO_MEDICAL:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("case_id"):
            by_id[str(obj["case_id"])] = obj

    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_BIO_MEDICAL:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("case_id"):
            by_id[str(obj["case_id"])] = obj
    return by_id


def ensure_seed_bio_cases(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Idempotent seed of known UAP medical cases + black-eyed profile file."""
    existing = _load_all_bio_cases(memory_client, user_id, use_mem0_cloud)
    added = 0
    for row in SEED_UAP_MEDICAL_CASES:
        cid = row["case_id"]
        if cid in existing:
            continue
        rec = dict(row)
        rec["case_type"] = "uap_medical_reference"
        rec["source"] = "angel-seed-open-literature"
        rec["last_updated"] = _today()
        text = json.dumps(rec, ensure_ascii=False)
        _bio_case_upsert_memory(memory_client, user_id, cid, text, use_mem0_cloud)
        if files_cabinet is not None:
            _sync_bio_file(
                files_cabinet,
                cid,
                rec,
                tags=["biological_intelligence", "seed", "uap_medical"],
            )
        added += 1

    prof = dict(BLACK_EYED_PROFILE)
    prof["last_updated"] = _today()
    if files_cabinet is not None:
        _sync_bio_file(
            files_cabinet,
            "black-eyed-profile",
            prof,
            tags=["biological_intelligence", "black_eyed", "profile"],
        )
    return {
        "ok": True,
        "added": added,
        "total_cases": len(_load_all_bio_cases(memory_client, user_id, use_mem0_cloud)),
    }


def list_known_cases(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    rows = list(_load_all_bio_cases(memory_client, user_id, use_mem0_cloud).values())
    try:
        rows.sort(key=lambda x: (x.get("title") or x.get("case_id") or "").lower())
    except Exception:
        pass
    return rows


def aggregate_medical_patterns(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Summarize recurring tags/symptoms across stored cases."""
    cases = list_known_cases(memory_client, user_id, use_mem0_cloud)
    tag_counts: dict[str, int] = {}
    sym_counts: dict[str, int] = {}
    exposure_counts: dict[str, int] = {}
    for c in cases:
        for t in c.get("pattern_tags") or []:
            tt = str(t).strip().lower()
            if tt:
                tag_counts[tt] = tag_counts.get(tt, 0) + 1
        for s in c.get("symptoms") or []:
            ss = str(s).strip()[:120]
            if ss:
                sym_counts[ss] = sym_counts.get(ss, 0) + 1
        et = str(c.get("exposure_type") or "unknown").lower()
        exposure_counts[et] = exposure_counts.get(et, 0) + 1
    return {
        "case_count": len(cases),
        "top_pattern_tags": sorted(tag_counts.items(), key=lambda x: -x[1])[:12],
        "symptom_mentions": sorted(sym_counts.items(), key=lambda x: -x[1])[:15],
        "exposure_types": exposure_counts,
    }


def get_black_eyed_profile() -> dict[str, Any]:
    p = dict(BLACK_EYED_PROFILE)
    p["last_updated"] = _today()
    return p


def _seed_reference_block() -> str:
    lines = []
    for c in SEED_UAP_MEDICAL_CASES:
        lines.append(
            f"- {c.get('case_id')}: {c.get('title')} | symptoms: {', '.join((c.get('symptoms') or [])[:6])}"
        )
    return "\n".join(lines)


def analyze_biological_report(
    content: str,
    context: str,
    source: str,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    model: str = "claude-sonnet-4-5",
    persist: bool = True,
) -> dict[str, Any]:
    """
    Claude structured analysis of medical/biological text; no substitute for clinicians.
    """
    ensure_seed_bio_cases(memory_client, user_id, files_cabinet, use_mem0_cloud)
    c = (content or "").strip()
    if len(c) < 20:
        return {"ok": False, "error": "content too short or empty"}

    sys = """You are Angel's biological/medical intelligence analyst for Tyler's UAP/disclosure mission.
You output ONE JSON object only (no markdown) with this shape:
{
  "symptoms_identified": ["..."],
  "symptom_timeline_notes": "if text describes timing, summarize onset vs exposure event; else empty string",
  "anomalous_biological_markers": ["reported lab or physical findings if any — else empty"],
  "exposure_type": "radiation|electromagnetic|psychological|unknown|multiple",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "pattern_match": true/false,
  "known_case_similarities": ["short case refs — Cash-Landrum, Rendlesham, etc. only when similarity is plausible"],
  "mission_relevance": "short",
  "recommended_action": "file|investigate|monitor|none",
  "confidence": "LOW|MEDIUM|HIGH"
}
Rules:
- This is open-source / user-supplied narrative analysis — NOT a clinical diagnosis.
- Note temporal ordering if described (symptom onset vs exposure).
- Flag radiation/EM/psychological cues explicitly when present.
- Be conservative; uncertainty belongs in confidence and wording.
- Never invent classified medical facts."""

    user = f"""SOURCE: {source or 'unknown'}
MISSION CONTEXT: {context or '(none)'}

REFERENCE PATTERNS (for similarity only — not exhaustive):
{_seed_reference_block()}

CONTENT TO ANALYZE:
{c[:60_000]}"""

    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.15,
            system=sys,
            messages=[{"role": "user", "content": user}],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        o = _parse_json_obj(txt)
        if not o:
            return {"ok": False, "error": "no JSON from model", "raw": txt[:2000]}
        result = _normalize_analysis_result(o)
        result["analysis_id"] = hashlib.sha256(f"{c[:200]}|{uuid.uuid4().hex}".encode()).hexdigest()[:14]
        result["source"] = (source or "")[:500]
        result["context"] = (context or "")[:4000]
        result["analyzed_at"] = _now_iso()

        # Environmental map cross-reference (textual)
        try:
            import angel_environmental_map as aem

            nm = aem.match_text_to_locations(
                f"{c} {context}",
                memory_client,
                user_id,
                use_mem0_cloud,
                top_n=6,
            )
            if nm:
                result["near_mapped_locations"] = nm
        except Exception:
            pass

        if persist and result.get("severity") in ("HIGH", "CRITICAL"):
            aid = result["analysis_id"]
            _sync_bio_file(
                files_cabinet,
                f"analysis-{aid}",
                result,
                tags=["biological_intelligence", "analysis", result["severity"].lower()],
            )

        return {"ok": True, "analysis": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def add_bio_case(
    case: dict[str, Any],
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Add or update a case in the database."""
    cid = (case.get("case_id") or "").strip() or case_slug(str(case.get("title") or "case"))
    if not cid:
        return {"ok": False, "error": "case_id or title required"}
    rec = dict(case)
    rec["case_id"] = cid
    rec["last_updated"] = _today()
    if "case_type" not in rec:
        rec["case_type"] = "user_added"
    text = json.dumps(rec, ensure_ascii=False)
    _bio_case_upsert_memory(memory_client, user_id, cid, text, use_mem0_cloud)
    _sync_bio_file(files_cabinet, cid, rec, tags=["biological_intelligence", "user_case"])
    return {"ok": True, "case": rec}


def keyword_pattern_match(
    text: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    """Lightweight overlap with known UAP medical cases (for surveillance / triage)."""
    ensure_seed_bio_cases(memory_client, user_id, None, use_mem0_cloud)
    blob = (text or "").lower()
    if len(blob) < 12:
        return []
    hits: list[dict[str, Any]] = []
    for c in SEED_UAP_MEDICAL_CASES:
        score = 0
        for sym in c.get("symptoms") or []:
            s = str(sym).lower()
            if len(s) > 8 and s in blob:
                score += 3
            for w in re.findall(r"[a-z]{5,}", s):
                if w in blob:
                    score += 1
        for t in c.get("pattern_tags") or []:
            if str(t).lower() in blob:
                score += 2
        if score >= 4:
            hits.append(
                {
                    "case_id": c.get("case_id"),
                    "title": c.get("title"),
                    "score": score,
                }
            )
    try:
        hits.sort(key=lambda x: -x["score"])
    except Exception:
        pass
    return hits[:8]


def detect_bio_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(
        r"(?i)\banalyze\s+(?:this\s+)?(?:medical|health|biological)\s+report\b",
        msg,
    ):
        return "analyze_report", {}
    if re.search(
        r"(?i)\bmedical\s+effects\s+of\s+UAP|UAP\s+.*\bmedical\b.*\b(effects|health|injuries)\b",
        msg,
    ):
        return "uap_medical_summary", {}
    if re.search(
        r"(?i)\bblack[-\s]?eyed\s+people\b.*\b(medical|health|psychological|physiological)\b|"
        r"\bmedically\b.*\bblack[-\s]?eyed",
        msg,
    ):
        return "black_eyed", {}
    if re.search(
        r"(?i)\bmedical\s+patterns\b.*\bUAP|UAP\s+.*\bmedical\s+patterns\b",
        msg,
    ):
        return "patterns", {}
    return None, {}


def format_bio_chat_block(
    intent: str,
    payload: dict[str, Any],
    *,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    user_message: str,
) -> str:
    try:
        ensure_seed_bio_cases(memory_client, user_id, files_cabinet, use_mem0_cloud)
        if intent == "uap_medical_summary":
            cases = list_known_cases(memory_client, user_id, use_mem0_cloud)
            return (
                "[Angel biological intelligence — UAP medical cases (open literature)]\n"
                + json.dumps(
                    {
                        "known_cases": cases[:25],
                        "note": "Reference patterns only — not clinical advice.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )[:14000]
            )
        if intent == "black_eyed":
            p = get_black_eyed_profile()
            return "[Angel biological intelligence — black-eyed people profile]\n" + json.dumps(
                p, ensure_ascii=False, indent=2
            )[:12000]
        if intent == "patterns":
            agg = aggregate_medical_patterns(memory_client, user_id, use_mem0_cloud)
            return "[Angel biological intelligence — pattern summary]\n" + json.dumps(
                agg, ensure_ascii=False, indent=2
            )[:12000]
        if intent == "analyze_report":
            body = (user_message or "").strip()
            r = analyze_biological_report(
                body,
                "Tyler asked in chat for biological/medical pattern analysis.",
                "chat_message",
                anthropic_client,
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
                persist=False,
            )
            return "[Angel biological intelligence — analysis]\n" + json.dumps(r, ensure_ascii=False, indent=2)[
                :14000
            ]
    except Exception as e:
        return f"[Angel biological intelligence error]\n{str(e)[:500]}"
    return ""


def cluster_bio_signals_from_run(signals: list[dict[str, Any]]) -> list[str]:
    """
    If multiple bio_medical_watch signals share a region hint and strength, flag clustering.
    """
    bio_rows = [
        s
        for s in signals or []
        if isinstance(s, dict)
        and (s.get("category_id") or "") == "bio_medical_watch"
        and (s.get("signal_strength") or "") in ("MODERATE", "STRONG")
    ]
    if len(bio_rows) < 2:
        return []
    by_region: dict[str, list[str]] = {}
    for s in bio_rows:
        rh = (s.get("region_hint") or "unknown").strip().lower()[:80]
        hl = (s.get("headline") or "")[:120]
        by_region.setdefault(rh, []).append(hl)
    notes: list[str] = []
    for reg, hls in by_region.items():
        if len(hls) >= 2 and reg != "unknown":
            notes.append(
                f"Multiple bio/medical-adjacent signals in similar region ({reg}): "
                + "; ".join(hls[:4])
            )
    return notes[:5]
