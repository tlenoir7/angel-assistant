"""
Computer vision on demand — single-call classification + routed forensic pipelines.
Auto-files HIGH/CRITICAL to Intelligence folder `Visual Intelligence` (VI-*).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import anthropic

from angel_forensic import (
    _crossref_mission,
    _detect_media_type,
    _parse_json_from_response,
    decode_image_b64,
    extract_exif_summary,
)

VISUAL_INTEL_FOLDER = "Visual Intelligence"
VI_PREFIX = "VI-"

_FORENSIC_VISION_SYSTEM = """You are Angel's forensic computer-vision analyst. Tyler submits one image.

TASK (single pass):
1) Classify the PRIMARY content type (choose exactly one primary_type):
   - document — paper, screen, text, forms, IDs, letters, contracts, screenshots of text
   - scene — location, room, outdoor environment, wide context
   - person — face, individual, crowd (focus on people)
   - object — device, equipment, vehicle, weapon, tool, single salient object
   - media — photo-of-photo, screenshot UI, digital content, meme, unclear capture chain
   - unknown — cannot determine

2) Run the matching pipeline in depth. Populate ONLY the pipeline block that matches primary_type; set all other pipeline_* keys to null.

3) Set universal fields. Be honest about limits: you are not a certified digital forensics lab; no actual EXIF beyond what CONTEXT provides; no facial recognition (only general visible traits).

4) For mission_relevance use Tyler's MEMORY CONTEXT and INTELLIGENCE FILE CABINET SUMMARY when judging connections.

5) network_updates: propose graph updates ONLY for clearly identified people or organizations (not guesses). node_type must be one of: person, organization, program, event, faction. relationship_type must be one of: works_with, testified_with, employed_by, investigated_by, connected_to, corroborates, contradicts, funded_by, member_of, retaliates_against, opposes, suppresses. strength: WEAK, MODERATE, STRONG, CONFIRMED. relevance on nodes: LOW, MEDIUM, HIGH, CRITICAL.

Output ONLY valid JSON (no markdown), exactly this shape:

{
  "classification": {
    "primary_type": "document|scene|person|object|media|unknown",
    "secondary_types": [],
    "classification_confidence": "HIGH|MEDIUM|LOW",
    "rationale": "one sentence"
  },
  "analysis_type": "same string as primary_type",
  "confidence": "HIGH|MEDIUM|LOW",
  "summary": "2-3 sentences plain English",
  "key_findings": ["most important first"],
  "anomalies": ["things that do not fit or warrant attention"],
  "mission_relevance": "LOW|MEDIUM|HIGH|CRITICAL",
  "recommended_action": "what Tyler should do next",
  "file_to_intelligence": true or false,

  "pipeline_document": null or {
    "visible_text_transcription": "OCR-style, reading order; note illegible bits",
    "document_subtype": "ID|form|letter|contract|screenshot|other",
    "formatting_anomalies": ["fonts, alignment, suspicious layout"],
    "tampering_or_manipulation_signs": [],
    "key_data_points": {"names": [], "dates": [], "numbers": [], "addresses": []},
    "authenticity_assessment": "genuine|likely_altered|unclear",
    "authenticity_notes": "",
    "intel_file_connections": ["names or topics that match cabinet summary if any"]
  },
  "pipeline_scene": null or {
    "location_environment": "",
    "people_present_general": [],
    "objects_of_interest": [],
    "anomalies_tactical": [],
    "threat_indicators": {"exits": "", "cover": "", "chokepoints": ""},
    "time_and_lighting": "",
    "tyler_location_correlation": "compare to TYLER REPORTED LOCATION if provided; else state unknown"
  },
  "pipeline_person": null or {
    "general_description": "",
    "notable_identifiers": [],
    "body_language": "",
    "context_activity": "",
    "network_match_candidates": [{"name": "possible match", "rationale": ""}],
    "threat_actor_match_notes": "flag if traits align with opposition/threat profiles mentioned in context"
  },
  "pipeline_object": null or {
    "identification": "",
    "make_model_manufacturer": "",
    "condition_and_anomalies": "",
    "likely_function": "",
    "visible_identifiers": [],
    "threat_assessment": ""
  },
  "pipeline_media": null or {
    "authenticity_verdict": "AUTHENTIC|SUSPICIOUS|LIKELY_MANIPULATED|CANNOT_DETERMINE",
    "ai_generation_signs": [],
    "editing_signs": [],
    "metadata_indicators_inferred": "what EXIF might show if present — do not invent specific values",
    "source_assessment": ""
  },

  "network_updates": {
    "nodes": [{"name": "", "node_type": "person", "description": "", "relevance": "MEDIUM"}],
    "edges": [{"source_name": "", "target_name": "", "relationship_type": "connected_to", "description": "", "strength": "MODERATE", "evidence": "visual intel"}]
  }
}

If no network updates, use "nodes": [] and "edges": []."""


def _mission_autofile_threshold(mr: str) -> bool:
    return (mr or "").strip().upper() in ("HIGH", "CRITICAL")


def _autofile_visual_intel(
    parsed: dict[str, Any],
    raw: bytes,
    file_name: str,
    files_cabinet: Any,
    *,
    exif_block: dict[str, Any],
    auto_filed: bool,
    force: bool = False,
) -> dict[str, Any]:
    """HIGH/CRITICAL → VI-* in Visual Intelligence (or force=True for manual client filing)."""
    mr = str(parsed.get("mission_relevance") or "LOW")
    if not force and not _mission_autofile_threshold(mr):
        return {"filed": False, "reason": "mission_relevance below HIGH/CRITICAL auto-file threshold"}

    h = hashlib.sha256(raw[:50_000] + (file_name or "").encode()).hexdigest()[:10]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    fname = f"{VI_PREFIX}{day}-{h}"
    ptype = (parsed.get("classification") or {}).get("primary_type") or parsed.get("analysis_type") or "unknown"
    tags = [
        "visual_intelligence",
        f"type:{str(ptype).lower()}",
        f"mission:{str(mr).upper()}",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    ]
    for n in (parsed.get("network_updates") or {}).get("nodes") or []:
        if isinstance(n, dict) and n.get("name"):
            tags.append(f"entity:{str(n.get('name'))[:40]}")
    tags = [t for t in tags if t][:35]

    body = json.dumps(
        {
            "source": "angel_vision_forensic",
            "file_name": file_name,
            "classification": parsed.get("classification"),
            "confidence": parsed.get("confidence"),
            "summary": parsed.get("summary"),
            "key_findings": parsed.get("key_findings"),
            "anomalies": parsed.get("anomalies"),
            "mission_relevance": parsed.get("mission_relevance"),
            "recommended_action": parsed.get("recommended_action"),
            "pipelines": {
                "document": parsed.get("pipeline_document"),
                "scene": parsed.get("pipeline_scene"),
                "person": parsed.get("pipeline_person"),
                "object": parsed.get("pipeline_object"),
                "media": parsed.get("pipeline_media"),
            },
            "metadata_exif": exif_block,
            "auto_filed_by_server": auto_filed,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        files_cabinet.create_file(VISUAL_INTEL_FOLDER, fname, body, tags=tags)
        return {"filed": True, "cabinet_file": fname, "folder": VISUAL_INTEL_FOLDER}
    except ValueError:
        fname2 = f"{VI_PREFIX}{day}-{h}-b"
        try:
            files_cabinet.create_file(VISUAL_INTEL_FOLDER, fname2, body, tags=tags)
            return {"filed": True, "cabinet_file": fname2, "folder": VISUAL_INTEL_FOLDER}
        except Exception as e:
            return {"filed": False, "error": str(e)}
    except Exception as e:
        return {"filed": False, "error": str(e)}


def _apply_network_updates(
    parsed: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    import angel as ang

    nu = parsed.get("network_updates")
    if not isinstance(nu, dict):
        return {"applied_nodes": 0, "applied_edges": 0, "errors": []}
    nodes_raw = nu.get("nodes") if isinstance(nu.get("nodes"), list) else []
    edges_raw = nu.get("edges") if isinstance(nu.get("edges"), list) else []
    err: list[str] = []
    added_n = 0
    name_to_id: dict[str, str] = {}

    for n in nodes_raw[:20]:
        if not isinstance(n, dict):
            continue
        name = str(n.get("name") or "").strip()
        if len(name) < 2:
            continue
        nt = str(n.get("node_type") or "person").strip().lower()
        if nt not in ang.NETWORK_NODE_TYPES:
            nt = "person"
        rel = str(n.get("relevance") or "MEDIUM").strip().upper()
        if rel not in ang.NETWORK_NODE_RELEVANCE:
            rel = "MEDIUM"
        desc = str(n.get("description") or "From visual intelligence analysis.").strip()
        try:
            node = ang.add_network_node(
                name,
                nt,
                desc,
                rel,
                ["visual_intel"],
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
            nid = (node or {}).get("id") or ang.network_slug_from_display_name(name)
            name_to_id[name.lower()] = nid
            added_n += 1
        except Exception as e:
            err.append(f"node {name}: {e}")

    def _resolve_id(display: str) -> str | None:
        d = (display or "").strip()
        if not d:
            return None
        low = d.lower()
        if low in name_to_id:
            return name_to_id[low]
        return ang.network_slug_from_display_name(d)

    added_e = 0
    for e in edges_raw[:30]:
        if not isinstance(e, dict):
            continue
        sa = str(e.get("source_name") or "").strip()
        ta = str(e.get("target_name") or "").strip()
        sid = _resolve_id(sa)
        tid = _resolve_id(ta)
        if not sid or not tid:
            continue
        rt = str(e.get("relationship_type") or "connected_to").strip().lower()
        if rt not in ang.NETWORK_RELATIONSHIP_TYPES:
            rt = "connected_to"
        st = str(e.get("strength") or "MODERATE").strip().upper()
        if st not in ang.NETWORK_EDGE_STRENGTHS:
            st = "MODERATE"
        try:
            ang.add_network_edge(
                sid,
                tid,
                rt,
                str(e.get("description") or "Visual intelligence").strip(),
                st,
                str(e.get("evidence") or "Angel vision forensic").strip(),
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
            added_e += 1
        except Exception as ex:
            err.append(f"edge {sa}->{ta}: {ex}")

    return {"applied_nodes": added_n, "applied_edges": added_e, "errors": err[:12]}


def _run_forensic_vision_call(
    anthropic_client: anthropic.Anthropic,
    raw: bytes,
    media_type: str,
    file_name: str,
    user_payload: str,
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    b64 = base64.standard_b64encode(raw).decode("ascii")
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=6144,
        temperature=0.2,
        system=_FORENSIC_VISION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": user_payload},
                ],
            }
        ],
    )
    txt = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            txt += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            txt += block.get("text", "")
    parsed = _parse_json_from_response(txt)
    if not parsed:
        return {"ok": False, "error": "model did not return valid JSON", "raw_excerpt": (txt or "")[:2000]}
    parsed["ok"] = True
    return parsed


def analyze_image_forensic(
    image_base64: str,
    context: str,
    tyler_location: str | None,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: Any,
    *,
    memory_client: Any | None = None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
    file_name: str = "photo.jpg",
    intelligence_files_summary: str | None = None,
    memory_summary: str | None = None,
    skip_autofile: bool = False,
    skip_network_apply: bool = False,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """
    Single Claude vision call: classification + routed pipeline + universal fields.
    Auto-files to Visual Intelligence when mission_relevance is HIGH or CRITICAL.
    """
    fn = (file_name or "photo.jpg").strip() or "photo.jpg"
    try:
        raw, _ = decode_image_b64(image_base64)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    mt = _detect_media_type(raw, fn)
    exif_block = extract_exif_summary(raw)
    ctx = (context or "").strip() or "Tyler submitted this image for forensic visual analysis."
    loc = (tyler_location or "").strip()
    intel_sum = (intelligence_files_summary or "").strip() or "(no cabinet summary)"
    mem_sum = (memory_summary or "").strip() or "(no memory excerpt)"

    user_payload = f"""FILE_NAME: {fn}
CONTEXT FROM TYLER:
{ctx[:6000]}

TYLER REPORTED LOCATION (may be empty):
{loc[:500]}

INTELLIGENCE FILE CABINET SUMMARY:
{intel_sum[:8000]}

MEMORY CONTEXT (excerpt for mission relevance):
{mem_sum[:6000]}

EXIF / image facts (may be partial):
{json.dumps(exif_block, ensure_ascii=False)[:3500]}

Analyze and output the required JSON only."""

    result = _run_forensic_vision_call(
        anthropic_client, raw, mt, fn, user_payload, model=model
    )
    if not result.get("ok"):
        return result

    result["metadata_exif"] = exif_block
    result["file_name"] = fn
    result["media_type"] = mt

    narrative = json.dumps(result, ensure_ascii=False)[:100_000]
    if memory_client and user_id and files_cabinet:
        result["mission_cross_reference"] = _crossref_mission(
            narrative,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )
    else:
        result["mission_cross_reference"] = {"network_matches": []}

    mr = str(result.get("mission_relevance") or "LOW").upper()
    should_autofile = mr in ("HIGH", "CRITICAL")

    filed_meta: dict[str, Any] = {"filed": False}
    if not skip_autofile and should_autofile:
        filed_meta = _autofile_visual_intel(
            result, raw, fn, files_cabinet, exif_block=exif_block, auto_filed=True
        )
        result["auto_file"] = filed_meta
    elif not skip_autofile:
        result["auto_file"] = {"filed": False, "reason": "below auto-file threshold"}
    else:
        result["auto_file"] = {"filed": False, "skipped": True}

    result["auto_filed"] = bool((result.get("auto_file") or {}).get("filed"))
    result["show_manual_file_button"] = bool(result.get("file_to_intelligence")) and not result["auto_filed"]

    net_meta: dict[str, Any] = {"skipped": True}
    if (
        not skip_network_apply
        and memory_client
        and user_id
        and files_cabinet
        and should_autofile
    ):
        net_meta = _apply_network_updates(
            result,
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
        )
    elif not skip_network_apply and should_autofile:
        net_meta = {"skipped": True, "reason": "missing memory client or user_id"}
    result["network_updates_applied"] = net_meta

    # Progressive / mobile-friendly top-level mirrors
    result["summary_for_progressive_ui"] = result.get("summary") or ""
    return result


def file_visual_intel_manual(
    forensic_json: dict[str, Any],
    raw_image_base64: str,
    files_cabinet: Any,
    *,
    file_name: str = "photo.jpg",
) -> dict[str, Any]:
    """Client-triggered save to Visual Intelligence (e.g. iOS 'File to Intelligence' button)."""
    try:
        raw, _ = decode_image_b64(raw_image_base64)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    exif_block = extract_exif_summary(raw)
    fn = (file_name or "photo.jpg").strip() or "photo.jpg"
    meta = _autofile_visual_intel(
        forensic_json,
        raw,
        fn,
        files_cabinet,
        exif_block=exif_block,
        auto_filed=False,
        force=True,
    )
    return {"ok": bool(meta.get("filed")), **meta}
