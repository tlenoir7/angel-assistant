"""
Item 18 — Real-time translation & foreign-source monitoring for mission intelligence.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic

FOREIGN_INTEL_FOLDER = "Foreign Intelligence"
FOREIGN_FILE_PREFIX = "FI-"

# Fixed multilingual UAP queries for proactive / monitor pass (Item 18)
FOREIGN_UAP_TAVILY_QUERIES: list[tuple[str, str]] = [
    ("es", "fenómenos aéreos no identificados"),
    ("fr", "phénomènes aériens non identifiés"),
    ("de", "unidentifizierte Flugobjekte Regierung"),
    ("ru", "НЛО раскрытие информации"),
    ("zh", "不明飞行物 政府"),
    ("ja", "未確認飛行物体 政府開示"),
]

FOREIGN_WATCH_SEEDS: list[tuple[str, str, str, str]] = [
    ("UAP disclosure foreign governments", "situation", "HIGH", "weekly"),
    ("Russian UAP intelligence", "topic", "MEDIUM", "weekly"),
    ("Chinese UAP programs", "topic", "MEDIUM", "weekly"),
    ("European UAP acknowledgments", "topic", "MEDIUM", "weekly"),
    ("Latin American UAP incidents", "topic", "MEDIUM", "weekly"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_fid(seed: str) -> str:
    return hashlib.sha256(f"{seed}|{uuid.uuid4().hex}".encode()).hexdigest()[:14]


def _foreign_finding_upsert(
    memory_client: Any,
    user_id: str,
    finding: dict,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    fid = (finding.get("finding_id") or "").strip()
    if not fid:
        return
    cat = ang.CATEGORY_FOREIGN_INTEL
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-foreign-intel",
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
                {"role": "user", "content": f"[Angel foreign intel {fid}] {text[:1200]}"},
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


def translate_and_analyze(
    anthropic_client: anthropic.Anthropic,
    content: str,
    context: str = "",
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """
    Detect language, translate to English, and return structured mission analysis as JSON dict.
    """
    raw = (content or "").strip()
    if not raw:
        return {"ok": False, "error": "empty content"}
    ctx = (context or "").strip() or "General mission context: UAP/disclosure, Tyler's federal law enforcement and intelligence work."

    system = """You are Angel's multilingual intelligence analyst for Tyler's mission.
You MUST respond with a single JSON object only (no markdown fences), with these keys:
- "detected_language": BCP-47 or English name of the source language (best guess).
- "translation": clean, faithful English translation of the full input (or the same text if already English).
- "summary": 2-3 sentences on key points.
- "mission_relevance": how this connects to UAP/disclosure, government transparency, Tyler's mission (plain text).
- "key_terms": array of strings — important people, places, agencies, programs, dates.
- "red_flags": array of strings — anything unusual, contradictory, coercive, propaganda-like, or strategically significant.
- "linguistic_notes": 1-3 sentences on tone: diplomatic vs direct, euphemism, what may be avoided or implied.
- "threat_signal": one of "NONE"|"LOW"|"MEDIUM"|"HIGH"|"CRITICAL" — only if content suggests career/safety/mission risk; else "NONE".

If the input is already English, still populate all fields; translation can echo the original with light cleanup."""

    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.2,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": f"CONTEXT FOR TYLER:\n{ctx[:6000]}\n\n---\nTEXT TO ANALYZE:\n{raw[:120_000]}",
                }
            ],
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
            return {"ok": False, "error": "model did not return JSON", "raw": txt[:2000]}
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return {"ok": False, "error": "invalid JSON shape"}
        obj["ok"] = True
        return obj
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _is_mostly_english(text: str) -> bool:
    """Heuristic: skip heavy translation pass for obviously Latin/English web snippets."""
    t = text or ""
    if len(t) < 30:
        return True
    non_latin = sum(1 for c in t if ord(c) > 127)
    if non_latin / max(len(t), 1) > 0.08:
        return False
    if re.search(r"[\u0400-\u04FF\u4e00-\u9fff\u3040-\u30ff]", t):
        return False
    return True


def file_foreign_intelligence(
    files_cabinet: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    original_excerpt: str,
    analysis: dict[str, Any],
    source_label: str,
    tavily_url: str = "",
) -> dict[str, Any]:
    """Save original + translation + analysis to Foreign Intelligence folder and Mem0."""
    import angel as ang

    fid = _new_fid(original_excerpt[:120])
    slug = hashlib.sha256(f"{fid}|{source_label}".encode()).hexdigest()[:10]
    fname = f"{FOREIGN_FILE_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d')}-{slug}"
    trans = (analysis.get("translation") or "").strip()
    body = "\n".join(
        [
            f"foreign_intel_id: {fid}",
            f"source: {source_label}",
            f"source_url: {tavily_url}",
            f"detected_language: {analysis.get('detected_language', '')}",
            "",
            "=== ORIGINAL (excerpt) ===",
            (original_excerpt or "")[:80_000],
            "",
            "=== ENGLISH TRANSLATION ===",
            trans[:80_000],
            "",
            "=== STRUCTURED ANALYSIS (JSON) ===",
            json.dumps({k: v for k, v in analysis.items() if k != "ok"}, ensure_ascii=False, indent=2),
        ]
    )
    tags = [
        "foreign_intelligence",
        f"lang:{str(analysis.get('detected_language', 'unknown'))[:40]}",
        f"threat:{analysis.get('threat_signal', 'NONE')}",
    ]
    try:
        if files_cabinet.get_file(fname):
            files_cabinet.update_file(fname, body)
        else:
            files_cabinet.create_file(FOREIGN_INTEL_FOLDER, fname, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fname, body)
        except Exception:
            pass
    except Exception:
        pass

    rec = {
        "finding_id": fid,
        "file_name": fname,
        "source_label": source_label,
        "source_url": tavily_url,
        "detected_language": analysis.get("detected_language"),
        "summary": analysis.get("summary"),
        "mission_relevance": analysis.get("mission_relevance"),
        "threat_signal": analysis.get("threat_signal", "NONE"),
        "at": _now_iso(),
    }
    _foreign_finding_upsert(memory_client, user_id, rec, use_mem0_cloud)

    tsig = str(analysis.get("threat_signal") or "NONE").upper()
    if tsig in ("HIGH", "CRITICAL"):
        try:
            tb = "\n".join(
                [
                    f"watch_category: foreign source — {source_label[:160]}",
                    f"threat_level: {tsig}",
                    f"source_url: {tavily_url}",
                    "",
                    f"Foreign intelligence (translated):\n{(analysis.get('summary') or '')[:2000]}",
                ]
            )
            tf = f"TI-FOR-{slug}"
            files_cabinet.create_file(
                ang.THREAT_INTEL_FOLDER,
                tf,
                tb,
                tags=[f"threat_level:{tsig}", "foreign_intel"],
            )
        except Exception:
            pass

    return {"ok": True, "finding_id": fid, "file_name": fname, "record": rec}


def run_foreign_uap_monitor_pass(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    max_items: int = 4,
) -> dict[str, Any]:
    """
    Run fixed foreign-language UAP Tavily queries; translate & file non-English or high-signal hits.
    """
    import angel as ang

    api_key = __import__("os").getenv("TAVILY_API_KEY") or ""
    out: dict[str, Any] = {
        "ok": True,
        "queries_run": 0,
        "filed": 0,
        "skipped": 0,
        "items": [],
    }
    if not api_key:
        return {**out, "error": "TAVILY_API_KEY not set"}

    filed = 0
    for lang, q in FOREIGN_UAP_TAVILY_QUERIES:
        if filed >= max_items:
            break
        out["queries_run"] += 1
        hits = ang._tavily_search_one(q, api_key, max_results=3, search_depth="basic")
        for r in hits:
            if filed >= max_items:
                break
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            snippet = (r.get("content") or r.get("snippet") or "").strip()
            url = (r.get("url") or "").strip()
            blob = f"{title}\n{snippet}"
            if len(blob) < 80:
                continue
            must_translate = not _is_mostly_english(blob)
            ctx = f"Foreign-language UAP monitoring query ({lang}). Assess for new government acknowledgments, policy shifts, or credible reporting."
            analysis = translate_and_analyze(
                anthropic_client,
                blob,
                ctx,
            )
            if not analysis.get("ok"):
                out["skipped"] += 1
                continue
            if not must_translate and (analysis.get("mission_relevance") or "").strip() == "":
                out["skipped"] += 1
                continue
            if not must_translate and "UAP" not in (analysis.get("summary") or "").upper() and "UAP" not in (
                analysis.get("mission_relevance") or ""
            ).upper():
                if not re.search(r"(?i)uap|ufo|unidentified|аномал|不明|未確認", blob):
                    out["skipped"] += 1
                    continue

            try:
                file_foreign_intelligence(
                    files_cabinet,
                    memory_client,
                    user_id,
                    use_mem0_cloud,
                    original_excerpt=blob[:50_000],
                    analysis=analysis,
                    source_label=f"tavily:{lang}:{q[:80]}",
                    tavily_url=url,
                )
                filed += 1
                out["items"].append({"lang": lang, "url": url, "filed": True})
            except Exception as ex:
                out["items"].append({"lang": lang, "error": str(ex)})

    out["filed"] = filed
    return out


def search_foreign_sources_and_translate(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    topic: str,
    languages: list[str] | None,
    context: str,
) -> dict[str, Any]:
    """Tavily search per requested language code, merge snippets, translate & analyze once."""
    import angel as ang

    api_key = __import__("os").getenv("TAVILY_API_KEY") or ""
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "error": "topic required"}
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    langs = languages or ["es", "fr", "de"]
    lang_queries = {
        "es": f"{topic} fenómenos aéreos noticias",
        "fr": f"{topic} phénomènes aériens actualités",
        "de": f"{topic} UFO Regierung",
        "ru": f"{topic} НЛО новости",
        "zh": f"{topic} 不明飞行物",
        "ja": f"{topic} 未確認飛行物体",
    }

    bundle: list[str] = []
    for code in langs:
        code = str(code).strip().lower()[:8]
        q = lang_queries.get(code) or f"{topic} news"
        for r in ang._tavily_search_one(q, api_key, max_results=3, search_depth="basic"):
            if isinstance(r, dict):
                bundle.append(
                    f"[{code}] {(r.get('title') or '')}\n{(r.get('content') or r.get('snippet') or '')[:900]}\nURL: {(r.get('url') or '')}\n"
                )

    combined = "\n\n".join(bundle)[:100_000]
    if not combined.strip():
        return {"ok": False, "error": "no search results"}

    analysis = translate_and_analyze(
        anthropic_client,
        combined,
        context or f"Tyler asked for foreign-source perspective on: {topic}",
    )
    if not analysis.get("ok"):
        return analysis

    try:
        file_foreign_intelligence(
            files_cabinet,
            memory_client,
            user_id,
            use_mem0_cloud,
            original_excerpt=combined[:80_000],
            analysis=analysis,
            source_label=f"search:{topic[:80]}",
            tavily_url="",
        )
    except Exception:
        pass

    return {"ok": True, "analysis": analysis, "snippets_chars": len(combined)}


def translate_document_and_file(
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    content: str,
    context: str = "",
    *,
    source_label: str = "document_upload",
) -> dict[str, Any]:
    """
    Full-document path: translate/analyze, file original + translation under Foreign Intelligence,
    extract key terms / red flags (via structured analysis). Proper nouns surface in key_terms.
    """
    raw = (content or "").strip()
    if not raw:
        return {"ok": False, "error": "empty content"}
    ctx = (context or "").strip() or (
        "Tyler uploaded or pasted a foreign-language document for full translation and mission analysis."
    )
    analysis = translate_and_analyze(anthropic_client, raw, ctx)
    if not analysis.get("ok"):
        return analysis
    try:
        filed = file_foreign_intelligence(
            files_cabinet,
            memory_client,
            user_id,
            use_mem0_cloud,
            original_excerpt=raw[:120_000],
            analysis=analysis,
            source_label=source_label[:200],
            tavily_url="",
        )
    except Exception as ex:
        return {"ok": False, "error": str(ex), "analysis": analysis}
    # Cross-link named entities into mission network (Item 18 integration)
    try:
        from angel import map_osint_to_network

        primary = (source_label or "Foreign document")[:120]
        body = (
            f"PRIMARY (foreign document): {primary}\n\n"
            f"ENGLISH TRANSLATION (excerpt):\n{(analysis.get('translation') or '')[:12000]}\n\n"
            f"SUMMARY:\n{analysis.get('summary') or ''}\n\n"
            f"KEY TERMS:\n{json.dumps(analysis.get('key_terms') or [], ensure_ascii=False)}"
        )
        map_osint_to_network(
            body,
            primary,
            primary_target_type="event",
            anthropic_client=anthropic_client,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    except Exception:
        pass
    return {"ok": True, "analysis": analysis, "filed": filed}


def fetch_recent_foreign_findings(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    limit: int = 30,
) -> list[dict]:
    import angel as ang

    out: list[tuple[str, dict]] = []
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_FOREIGN_INTEL:
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
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_FOREIGN_INTEL:
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


def ensure_foreign_watch_items(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    import angel_proactive as ap

    added = 0
    for label, wt, pr, cf in FOREIGN_WATCH_SEEDS:
        try:
            ap.add_watch_item(
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
            added += 1
        except Exception:
            pass
    return {"ok": True, "ensured": added}


def looks_like_foreign_paste(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 80:
        return False
    if re.search(r"[\u0400-\u04FF\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t):
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    return non_ascii / max(len(t), 1) > 0.12


def detect_translation_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw:
        return None, {}

    m = re.search(
        r"(?i)\b(?:search|find)\s+(?:for\s+)?(?:UAP\s+)?news\s+in\s+([a-zA-Z\s]+?)(?:\s+about|\s+on\s+|$)",
        raw,
    )
    if m:
        lang = m.group(1).strip().rstrip("?.!")
        topic_m = re.search(r"(?i)(?:about|on)\s+(.+)$", raw)
        topic = (topic_m.group(1) if topic_m else "UAP disclosure").strip().rstrip("?.!")
        return "foreign_search", {"languages_hint": lang, "topic": topic}

    if re.search(
        r"(?i)\b(translate\s+this|what\s+does\s+this\s+say|translate\s+the\s+following)\b",
        raw,
    ):
        rest = re.split(r"(?i)translate\s+this|what\s+does\s+this\s+say", raw, maxsplit=1)
        extra = rest[-1].strip() if len(rest) > 1 else ""
        return "translate_explicit", {"inline": extra}

    if looks_like_foreign_paste(raw) and len(raw) < 50_000:
        return "translate_paste", {"text": raw}

    return None, {}


def format_translation_for_prompt(
    intent: str,
    payload: dict[str, Any],
    *,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
    user_message: str,
) -> str:
    ctx = "Tyler is asking in chat; provide concise translated intelligence he can act on."

    if intent == "translate_paste":
        text = payload.get("text") or user_message
        a = translate_and_analyze(anthropic_client, text, ctx)
        if not a.get("ok"):
            return f"[Translation]\nError: {a.get('error', 'unknown')}"
        try:
            file_foreign_intelligence(
                files_cabinet,
                memory_client,
                user_id,
                use_mem0_cloud,
                original_excerpt=text[:100_000],
                analysis=a,
                source_label="user_paste_chat",
            )
        except Exception:
            pass
        return (
            "[Angel translation — pasted foreign text]\n"
            + json.dumps(
                {
                    "detected_language": a.get("detected_language"),
                    "translation": a.get("translation"),
                    "summary": a.get("summary"),
                    "mission_relevance": a.get("mission_relevance"),
                    "key_terms": a.get("key_terms"),
                    "red_flags": a.get("red_flags"),
                    "linguistic_notes": a.get("linguistic_notes"),
                },
                ensure_ascii=False,
                indent=2,
            )[:12000]
        )

    if intent == "translate_explicit":
        inline = (payload.get("inline") or "").strip()
        text = inline if len(inline) > 40 else user_message
        a = translate_and_analyze(anthropic_client, text, ctx)
        if not a.get("ok"):
            return f"[Translation]\nError: {a.get('error')}"
        return "[Angel translation]\n" + json.dumps(
            {k: a.get(k) for k in ("detected_language", "translation", "summary", "mission_relevance", "key_terms", "red_flags", "linguistic_notes")},
            ensure_ascii=False,
            indent=2,
        )[:12000]

    if intent == "foreign_search":
        topic = (payload.get("topic") or "UAP").strip()
        hint = (payload.get("languages_hint") or "").lower()
        lang_map = {
            "spanish": ["es"],
            "french": ["fr"],
            "german": ["de"],
            "russian": ["ru"],
            "chinese": ["zh"],
            "japanese": ["ja"],
            "europe": ["fr", "de", "es"],
            "latin": ["es"],
        }
        langs = None
        for k, v in lang_map.items():
            if k in hint:
                langs = v
                break
        if not langs:
            langs = ["es", "fr", "de"]
        r = search_foreign_sources_and_translate(
            anthropic_client,
            memory_client,
            user_id,
            files_cabinet,
            use_mem0_cloud,
            topic=topic,
            languages=langs,
            context=ctx,
        )
        if not r.get("ok"):
            return f"[Foreign search] {r.get('error', 'failed')}"
        an = r.get("analysis") or {}
        return "[Angel foreign-source search + translation]\n" + json.dumps(
            {
                "topic": topic,
                "languages": langs,
                "summary": an.get("summary"),
                "mission_relevance": an.get("mission_relevance"),
                "translation_excerpt": (an.get("translation") or "")[:4000],
                "key_terms": an.get("key_terms"),
            },
            ensure_ascii=False,
            indent=2,
        )[:12000]

    return ""
