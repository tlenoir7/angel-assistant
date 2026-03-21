"""
Batcomputer Layer — Open-source surveillance monitoring (legal public OSINT only).
Tavily multi-category scans, Claude evaluation, Mem0 + Surveillance Intelligence files.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic

SURVEILLANCE_FOLDER = "Surveillance Intelligence"
SI_PREFIX = "SI-"

SIGNAL_LEVELS = frozenset({"NOISE", "WEAK", "MODERATE", "STRONG"})

# Two Tavily queries per category (12 searches per full run)
_LAST_SURVEILLANCE_RUN: dict[str, Any] = {"at": None, "result": None}

SURVEILLANCE_CATEGORY_QUERIES: dict[str, dict[str, Any]] = {
    "aerial": {
        "label": "Aerial monitoring",
        "queries": [
            "military flight activity United States region news",
            "restricted airspace activation NOTAM unusual",
        ],
    },
    "ground": {
        "label": "Ground activity",
        "queries": [
            "military base activity news unusual",
            "government facility lockdown emergency news",
        ],
    },
    "maritime": {
        "label": "Maritime",
        "queries": [
            "naval vessel unusual activity news",
            "coast guard emergency operation news",
        ],
    },
    "public_records": {
        "label": "Public records",
        "queries": [
            "FOIA UAP classified government request news",
            "court filing government secrecy national security news",
        ],
    },
    "anomalous_events": {
        "label": "Anomalous events",
        "queries": [
            "unexplained explosion sonic boom news",
            "power grid anomaly unexplained news",
        ],
    },
    "social_signals": {
        "label": "Social signals",
        "queries": [
            "UAP sighting cluster report news",
            "military personnel UAP incident report news",
        ],
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_sid(seed: str) -> str:
    return hashlib.sha256(f"{seed}|{uuid.uuid4().hex}".encode()).hexdigest()[:14]


def _finding_upsert(
    memory_client: Any,
    user_id: str,
    finding: dict[str, Any],
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    fid = (finding.get("finding_id") or "").strip()
    if not fid:
        return
    cat = ang.CATEGORY_SURVEILLANCE_INTEL
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-surveillance",
        "surveillance_finding_id": fid,
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
                and e["metadata"].get("surveillance_finding_id") == fid
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel surveillance {fid}] {text[:1200]}"},
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


def _parse_json_obj(txt: str) -> dict[str, Any] | None:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\s*```\s*$", "", txt)
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_json_array(txt: str) -> list | None:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    m = re.search(r"\[[\s\S]*\]", txt)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, list) else None
    except json.JSONDecodeError:
        return None


def _file_strong_signal(
    files_cabinet: Any,
    body_obj: dict[str, Any],
    *,
    cross_note: str = "",
) -> str | None:
    h = hashlib.sha256(json.dumps(body_obj, sort_keys=True).encode()).hexdigest()[:10]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    fname = f"{SI_PREFIX}{day}-{h}"
    if cross_note:
        body_obj["cross_reference_note"] = cross_note
    content = json.dumps(body_obj, ensure_ascii=False, indent=2)
    tags = [
        "surveillance_intelligence",
        f"signal:{body_obj.get('signal_strength', 'STRONG')}",
        f"category:{(body_obj.get('category') or 'unknown')[:40]}",
    ]
    try:
        if files_cabinet.get_file(fname):
            files_cabinet.update_file(fname, content)
        else:
            files_cabinet.create_file(SURVEILLANCE_FOLDER, fname, content, tags=tags)
        return fname
    except ValueError:
        try:
            files_cabinet.update_file(fname, content)
            return fname
        except Exception:
            return None
    except Exception:
        return None


def _threat_headlines_for_crossref(files_cabinet: Any, limit: int = 12) -> list[str]:
    import angel as ang

    out: list[str] = []
    try:
        for meta in files_cabinet.list_files(folder=ang.THREAT_INTEL_FOLDER)[:limit]:
            name = (meta.get("name") or "").strip()
            if not name:
                continue
            rec = files_cabinet.get_file(name)
            if not rec:
                continue
            hl = ang._parse_threat_headline_from_body(rec.get("content") or "") or name
            out.append(hl[:300])
    except Exception:
        pass
    return out


def _keyword_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]{5,}", (a or "").lower()))
    wb = set(re.findall(r"[a-zA-Z]{5,}", (b or "").lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)


def _aligns_with_predictions(summaries: list[str], memory_client: Any, user_id: str, use_mem0_cloud: bool) -> list[str]:
    try:
        import angel_predictions as ap

        preds = ap.get_active_predictions(memory_client, user_id, use_mem0_cloud)[:15]
        hits: list[str] = []
        blob = " ".join(summaries).lower()
        for p in preds:
            t = (p.get("title") or "").strip()
            if len(t) > 8 and t.lower() in blob:
                hits.append(t[:200])
                continue
            for w in re.findall(r"[A-Za-z]{6,}", t):
                if w.lower() in blob:
                    hits.append(f"Topic overlap: {t[:120]}")
                    break
        return list(dict.fromkeys(hits))[:8]
    except Exception:
        return []


def run_osint_surveillance(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    model_eval: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    """
    Run 2 Tavily queries per category, evaluate with Claude, store findings, file STRONG to SI-*.
    """
    import angel as ang

    api_key = (__import__("os").getenv("TAVILY_API_KEY") or "").strip()
    out: dict[str, Any] = {
        "ok": True,
        "at": _now_iso(),
        "categories_scanned": 0,
        "signals": [],
        "filed_si": [],
        "correlated": None,
        "errors": [],
        "prediction_alignment": [],
    }
    if not api_key:
        return {**out, "ok": False, "error": "TAVILY_API_KEY not set"}

    bundles: dict[str, list[dict[str, Any]]] = {}
    raw_for_eval: list[str] = []

    for cat_id, spec in SURVEILLANCE_CATEGORY_QUERIES.items():
        out["categories_scanned"] += 1
        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for q in spec["queries"][:3]:
            try:
                chunk = ang._tavily_search_one(q, api_key, max_results=4, search_depth="basic")
                for r in chunk:
                    if not isinstance(r, dict):
                        continue
                    url = (r.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append(r)
                    if len(merged) >= 6:
                        break
            except Exception as e:
                out["errors"].append(f"{cat_id} tavily: {e}")
            if len(merged) >= 6:
                break
        bundles[cat_id] = merged[:6]
        lines = []
        for i, r in enumerate(merged[:5], start=1):
            lines.append(
                f"[{i}] {(r.get('title') or '')[:200]}\nURL: {(r.get('url') or '')}\n"
                f"{(r.get('content') or r.get('snippet') or '')[:900]}"
            )
        raw_for_eval.append(f"=== CATEGORY {cat_id} ({spec['label']}) ===\n" + "\n\n".join(lines))

    eval_blob = "\n\n".join(raw_for_eval)[:100_000]
    if not eval_blob.strip():
        return {**out, "ok": False, "error": "no search results"}

    sys_eval = """You evaluate open-source news snippets for surveillance monitoring (legal OSINT only).
Output a single JSON object ONLY (no markdown):
{
  "signals": [
    {
      "category_id": "aerial|ground|maritime|public_records|anomalous_events|social_signals",
      "signal_strength": "NOISE|WEAK|MODERATE|STRONG",
      "headline": "short",
      "summary": "2-4 sentences — is this a genuine anomaly vs routine noise?",
      "region_hint": "geographic hint or unknown",
      "timeframe_hint": "recent / unknown",
      "source_urls": ["url1"],
      "notes": "why this strength level"
    }
  ]
}
Rules: Be conservative. STRONG = clearly unusual, well-sourced pattern relevant to defense/government transparency/mission. NOISE = routine news. Do not invent URLs not in the bundle."""

    try:
        resp = anthropic_client.messages.create(
            model=model_eval,
            max_tokens=4096,
            temperature=0.15,
            system=sys_eval,
            messages=[
                {
                    "role": "user",
                    "content": f"Evaluate these category bundles:\n\n{eval_blob}\n\nReturn JSON only.",
                }
            ],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                txt += block.get("text", "")
        parsed = _parse_json_obj(txt)
        signals = (parsed or {}).get("signals") if isinstance(parsed, dict) else None
        if not isinstance(signals, list):
            signals = []
    except Exception as e:
        return {**out, "ok": False, "error": f"evaluation failed: {e}"}

    threat_hls = _threat_headlines_for_crossref(files_cabinet)
    summaries_for_pred: list[str] = []

    for sig in signals:
        if not isinstance(sig, dict):
            continue
        cid = str(sig.get("category_id") or "").strip()
        if cid not in SURVEILLANCE_CATEGORY_QUERIES:
            continue
        ss = str(sig.get("signal_strength") or "NOISE").strip().upper()
        if ss not in SIGNAL_LEVELS:
            ss = "NOISE"
        fid = _new_sid(f"{cid}|{sig.get('headline')}|{ss}")
        row = {
            "finding_id": fid,
            "category_id": cid,
            "category_label": SURVEILLANCE_CATEGORY_QUERIES[cid]["label"],
            "signal_strength": ss,
            "headline": (sig.get("headline") or "")[:500],
            "summary": (sig.get("summary") or "")[:4000],
            "region_hint": (sig.get("region_hint") or "")[:200],
            "timeframe_hint": (sig.get("timeframe_hint") or "")[:200],
            "source_urls": [str(u)[:500] for u in (sig.get("source_urls") or []) if u][:12],
            "notes": (sig.get("notes") or "")[:2000],
            "at": _now_iso(),
        }
        cross_bits: list[str] = []
        comb = (row["headline"] + " " + row["summary"]).lower()
        for th in threat_hls:
            if _keyword_overlap(comb, th) > 0.08:
                cross_bits.append(f"Possible theme overlap with Threat Intelligence: {th[:160]}")
        if cross_bits:
            row["threat_intel_crossref"] = cross_bits[:5]
        _finding_upsert(memory_client, user_id, row, use_mem0_cloud)
        out["signals"].append(row)
        summaries_for_pred.append(row.get("summary") or row.get("headline") or "")

        if ss == "STRONG":
            cr = "\n".join(cross_bits) if cross_bits else ""
            fn = _file_strong_signal(
                files_cabinet,
                {
                    **row,
                    "uap_incidents_note": "Review UAP Incidents folder for related case files if topic matches.",
                },
                cross_note=cr,
            )
            if fn:
                out["filed_si"].append(fn)

    # Correlation pass
    try:
        mod_strong = [s for s in out["signals"] if s.get("signal_strength") in ("MODERATE", "STRONG")]
        regions = [s.get("region_hint") or "" for s in mod_strong if s.get("region_hint")]
        uniq_regions = {r.strip().lower() for r in regions if r and r.lower() != "unknown"}
        corr_sys = """You output JSON only: {"correlated": true/false, "priority": "NORMAL|HIGH", "reason": "short", "related_categories": ["..."]}
Correlated = multiple independent open-source signals in the same rough timeframe/geography suggesting a pattern worth Tyler's attention (not proof of causation)."""
        corr_user = json.dumps(
            {
                "signals": [
                    {
                        "category": s.get("category_id"),
                        "strength": s.get("signal_strength"),
                        "region": s.get("region_hint"),
                        "summary": (s.get("summary") or "")[:400],
                    }
                    for s in mod_strong[:12]
                ],
                "distinct_region_hints": list(uniq_regions)[:10],
            },
            ensure_ascii=False,
        )[:12000]
        cresp = anthropic_client.messages.create(
            model=model_eval,
            max_tokens=800,
            temperature=0.1,
            system=corr_sys,
            messages=[{"role": "user", "content": corr_user}],
        )
        ct = ""
        for block in cresp.content:
            if getattr(block, "type", None) == "text":
                ct += block.text
        cobj = _parse_json_obj(ct)
        if isinstance(cobj, dict) and cobj.get("correlated"):
            out["correlated"] = {
                "priority": cobj.get("priority") or "HIGH",
                "reason": (cobj.get("reason") or "")[:2000],
                "related_categories": cobj.get("related_categories") or [],
            }
            cfid = _new_sid("correlated|" + _now_iso())
            crow = {
                "finding_id": cfid,
                "category_id": "correlated_cluster",
                "category_label": "Correlated cluster",
                "signal_strength": "STRONG" if cobj.get("priority") == "HIGH" else "MODERATE",
                "headline": "CORRELATED SURVEILLANCE SIGNAL",
                "summary": out["correlated"]["reason"],
                "region_hint": ",".join(list(uniq_regions)[:5]) or "unknown",
                "timeframe_hint": "recent",
                "source_urls": [],
                "notes": "Automated correlation pass across categories.",
                "at": _now_iso(),
                "correlated_cluster": True,
            }
            _finding_upsert(memory_client, user_id, crow, use_mem0_cloud)
            out["signals"].append(crow)
            if out["correlated"]["priority"] == "HIGH":
                cfn = _file_strong_signal(
                    files_cabinet,
                    crow,
                    cross_note="CORRELATED CLUSTER — elevated priority.",
                )
                if cfn:
                    out["filed_si"].append(cfn)
    except Exception as e:
        out["errors"].append(f"correlation: {e}")

    out["prediction_alignment"] = _aligns_with_predictions(
        summaries_for_pred, memory_client, user_id, use_mem0_cloud
    )
    if out["prediction_alignment"]:
        out["notes"] = (
            "Surveillance run may intersect active prediction topics: "
            + "; ".join(out["prediction_alignment"][:5])
        )

    global _LAST_SURVEILLANCE_RUN
    _LAST_SURVEILLANCE_RUN = {"at": out.get("at"), "result": dict(out)}

    return out


def fetch_recent_surveillance_findings(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    import angel as ang

    out: list[tuple[str, dict[str, Any]]] = []
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_SURVEILLANCE_INTEL:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("finding_id"):
            out.append((ang._memory_created_at(m), obj))
    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_SURVEILLANCE_INTEL:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("finding_id"):
            out.append((m.get("created_at") or "", obj))
    try:
        out.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        pass
    return [x[1] for x in out[:limit]]


def get_signals_by_category(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, list[dict[str, Any]]]:
    by_cat: dict[str, list[dict[str, Any]]] = {k: [] for k in SURVEILLANCE_CATEGORY_QUERIES}
    by_cat["correlated_cluster"] = []
    for f in fetch_recent_surveillance_findings(memory_client, user_id, use_mem0_cloud, limit=80):
        cid = f.get("category_id") or "unknown"
        if cid not in by_cat:
            by_cat[cid] = []
        by_cat[cid].append(f)
    return by_cat


def format_surveillance_for_briefing(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    last_run: dict[str, Any] | None = None,
) -> str:
    """STRONG + correlated + MODERATE snippets for morning briefing."""
    if last_run is None:
        lr = _LAST_SURVEILLANCE_RUN.get("result")
        last_run = lr if isinstance(lr, dict) else None
    lines: list[str] = []
    if last_run and last_run.get("ok"):
        lines.append(
            f"Open-source surveillance scan: {last_run.get('categories_scanned', 0)} categories, "
            f"{len(last_run.get('signals') or [])} signal(s) evaluated, "
            f"{len(last_run.get('filed_si') or [])} STRONG filing(s) to Surveillance Intelligence."
        )
        if last_run.get("correlated"):
            c = last_run["correlated"]
            lines.append(
                f"*** CORRELATED SIGNAL ({c.get('priority', 'HIGH')} priority): {c.get('reason', '')[:400]}"
            )
        for p in (last_run.get("prediction_alignment") or [])[:4]:
            lines.append(f"Alignment note (predictions): {p}")
    recent = fetch_recent_surveillance_findings(memory_client, user_id, use_mem0_cloud, limit=12)
    strong = [f for f in recent if (f.get("signal_strength") or "") == "STRONG"]
    corr = [f for f in recent if f.get("correlated_cluster")]
    mod = [f for f in recent if (f.get("signal_strength") or "") == "MODERATE"]
    if strong:
        lines.append("STRONG surveillance signals (review SI-* files for detail):")
        for f in strong[:5]:
            lines.append(
                f"- [{f.get('category_label') or f.get('category_id')}] {f.get('headline', '')[:180]}"
            )
    if corr:
        lines.append("Correlated clusters:")
        for f in corr[:3]:
            lines.append(f"- {f.get('summary', '')[:280]}")
    elif not lines and mod:
        lines.append("Moderate surveillance signals (nothing STRONG in recent window):")
        for f in mod[:4]:
            lines.append(f"- {f.get('headline', '')[:160]}")
    if not lines:
        return ""
    return "SURVEILLANCE INTELLIGENCE (open-source, legal OSINT)\n" + "\n".join(lines)


def detect_surveillance_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(r"(?i)\bany\s+unusual\s+activity\b", msg):
        return "recent_all", {}
    if re.search(r"(?i)\bwhat(?:'s| is)\s+happening\s+in\s+the\s+sk(?:y|ies)\b", msg):
        return "aerial", {}
    if re.search(r"(?i)\bany\s+anomalous\s+events?\b", msg):
        return "anomalous_events", {}
    if re.search(r"(?i)\bsurveillance\s+(?:intel|monitoring|scan)\b", msg):
        return "recent_all", {}
    return None, {}


def format_surveillance_chat_block(
    intent: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> str:
    try:
        if intent == "aerial":
            rows = [
                f
                for f in fetch_recent_surveillance_findings(memory_client, user_id, use_mem0_cloud, limit=25)
                if (f.get("category_id") or "") == "aerial"
            ]
        elif intent == "anomalous_events":
            rows = [
                f
                for f in fetch_recent_surveillance_findings(memory_client, user_id, use_mem0_cloud, limit=25)
                if (f.get("category_id") or "") == "anomalous_events"
            ]
        else:
            rows = fetch_recent_surveillance_findings(memory_client, user_id, use_mem0_cloud, limit=20)
        block = {
            "intent": intent,
            "recent_findings": rows[:15],
            "note": "Open-source legal OSINT only; not classified collection.",
        }
        return "[Angel surveillance monitoring]\n" + json.dumps(block, ensure_ascii=False, indent=2)[:14000]
    except Exception as e:
        return f"[Angel surveillance error]\n{str(e)[:500]}"
