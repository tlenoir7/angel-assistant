"""
Batcomputer Layer — Forensic visual analysis (authenticity, UAP, document images).
Claude vision + optional EXIF; auto-file to Forensic Analysis / UAP Incidents cross-ref.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from typing import Any

import anthropic

FORENSIC_FOLDER = "Forensic Analysis"
UAP_INCIDENTS_FOLDER = "UAP Incidents"
FA_PREFIX = "FA-"

# --- Image decode / EXIF ---


def _detect_media_type(raw: bytes, file_name: str) -> str:
    n = (file_name or "").lower()
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".gif"):
        return "image/gif"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and b"WEBP" in raw[:12]:
        return "image/webp"
    return "image/jpeg"


def decode_image_b64(image_b64: str) -> tuple[bytes, str]:
    raw = b""
    s = (image_b64 or "").strip()
    if s.startswith("data:"):
        m = re.match(r"data:image/[^;]+;base64,(.+)", s, re.I | re.DOTALL)
        if m:
            s = m.group(1).strip()
    try:
        raw = base64.b64decode(s, validate=False)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}") from e
    if not raw:
        raise ValueError("empty image")
    return raw, s


def extract_exif_summary(raw: bytes) -> dict[str, Any]:
    """Best-effort EXIF / basic image facts (no GPS precision in output if sensitive)."""
    out: dict[str, Any] = {"available": False}
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        out["format"] = (getattr(img, "format", None) or "")[:16]
        out["size_pixels"] = {"width": img.width, "height": img.height}
        out["mode"] = (getattr(img, "mode", None) or "")[:16]
        exif = getattr(img, "getexif", lambda: None)()
        if exif:
            out["available"] = True
            from PIL.ExifTags import TAGS

            flat: dict[str, str] = {}
            for k, v in exif.items():
                tag = TAGS.get(k, k)
                if tag in ("Make", "Model", "DateTime", "DateTimeOriginal", "Software", "LensModel"):
                    flat[str(tag)] = str(v)[:200]
            out["tags"] = flat
    except Exception as ex:
        out["error"] = str(ex)[:200]
    return out


def _parse_json_from_response(txt: str) -> dict[str, Any] | None:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\s*```\s*$", "", txt)
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _crossref_mission(
    analysis_text: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    """Lightweight network / OSINT string match on forensic narrative."""
    try:
        import angel as ang

        blob = (analysis_text or "")[:100_000].lower()
        hits: list[dict[str, Any]] = []
        try:
            nodes, _e = ang.network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)
            for nid, node in (nodes or {}).items():
                name = (node.get("name") or nid or "").strip()
                if len(name) >= 4 and name.lower() in blob:
                    hits.append({"type": "network_node", "id": nid, "name": name})
        except Exception:
            pass
        return {"network_matches": hits[:25]}
    except Exception:
        return {}


def _system_prompt_full() -> str:
    return """You are a forensic imagery analyst assisting Angel (Tyler's UAP/disclosure mission, law-enforcement-adjacent context).
You MUST respond with a single JSON object only (no markdown). Use this exact top-level structure:

{
  "layer_1_content": {
    "description": "what is shown",
    "objects_identified": ["..."],
    "people": ["..."],
    "locations_or_setting": "indoor/outdoor/unknown and cues",
    "visible_text_snippets": ["OCR-style lines if any"],
    "environmental_context": {"time_of_day_guess": "", "weather_or_lighting": "", "setting": ""},
    "visible_anomalies": ["anything unusual in the scene"]
  },
  "layer_2_authenticity": {
    "manipulation_signs": ["cloning, compositing, AI, inconsistent lighting, edge artifacts, compression, etc. — or none noted"],
    "lighting_consistency": "assessment",
    "shadow_consistency": "assessment",
    "compression_or_resampling_artifacts": "assessment",
    "ai_generation_indicators": ["hands, text, symmetry, backgrounds, or none"],
    "overall_confidence": "LIKELY_AUTHENTIC | UNCERTAIN | LIKELY_MANIPULATED | CONFIRMED_MANIPULATION",
    "rationale": "short"
  },
  "layer_3_intelligence": {
    "ocr_text": ["ordered text blocks visible"],
    "landmarks_or_locations": ["..."],
    "people_or_uniforms": ["..."],
    "vehicles": ["plates/markings if visible — redact if sensitive"],
    "equipment_or_technology": ["..."],
    "timestamps_or_dates_visible": ["..."],
    "classification_or_control_markings": ["if any — describe, do not invent"]
  },
  "layer_4_mission": {
    "uap_relevance": "none | contextual | primary_subject",
    "network_entity_mentions": ["names that may match Tyler's graph — from image text/description only"],
    "intelligence_value": "LOW | MEDIUM | HIGH | CRITICAL",
    "recommended_action": "file | investigate_further | discard | monitor",
    "mission_notes": "1-3 sentences, cautious language"
  },
  "scene_type": "general | aerial_phenomenon | document_photo | screenshot | other",
  "uap_analysis": null,
  "document_forensics": null
}

If the image clearly shows a DOCUMENT (paper/screen), set scene_type to document_photo and fill document_forensics with:
{ "visible_full_text": "best-effort transcription", "classification_markings": [], "agency_markings": [], "authenticity_notes": "", "redactions_or_alterations": "", "dates_names_case_numbers": [] }.

If the image may show AERIAL / UAP phenomena, set scene_type to aerial_phenomenon and set uap_analysis to:
{
  "size_relative_to_reference": "",
  "movement_indicators": "",
  "atmospheric_interaction": "",
  "comparison_to_known_reports": "",
  "assessment": "CONVENTIONAL | UNKNOWN | ANOMALOUS",
  "confidence": "LOW | MEDIUM | HIGH"
}

Otherwise leave uap_analysis and document_forensics as null unless clearly applicable.

Be explicit about uncertainty. Do not claim classified access or chain-of-custody."""


def _system_prompt_uap() -> str:
    return _system_prompt_full() + """

Prioritize aerial_phenomenon: always populate uap_analysis when anything in the sky / unexplained object is visible; still complete all four layers."""


def _system_prompt_document() -> str:
    return """You are a forensic document-image analyst for Tyler's mission.
Respond with ONE JSON object only (no markdown), structure:

{
  "layer_1_content": { ... same as full forensic schema ... },
  "layer_2_authenticity": { ... },
  "layer_3_intelligence": {
    "ocr_text": ["full visible text in reading order"],
    "classification_or_control_markings": [],
    "agency_or_organization_markings": [],
    "dates_names_case_numbers": [],
    "equipment_or_technology": []
  },
  "layer_4_mission": { ... },
  "scene_type": "document_photo",
  "document_forensics": {
    "visible_full_text": "concatenated transcription",
    "classification_markings": [],
    "agency_markings": [],
    "authenticity_notes": "signs of tampering on the document image",
    "redactions_or_alterations": "",
    "structured_fields": {}
  },
  "uap_analysis": null
}

Extract maximum visible text. Note uncertainty where illegible."""


def _run_vision_json(
    anthropic_client: anthropic.Anthropic,
    raw: bytes,
    media_type: str,
    file_name: str,
    context: str,
    system: str,
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    b64 = base64.standard_b64encode(raw).decode("ascii")
    ctx = (context or "").strip() or "Tyler submitted this image for forensic visual analysis."
    user_txt = f"""FILE_NAME: {file_name}
CONTEXT FROM TYLER:
{ctx[:8000]}

Optional technical context (EXIF / file — may be empty):
{json.dumps(extract_exif_summary(raw), ensure_ascii=False)[:4000]}

Analyze the image and output the required JSON only."""
    resp = anthropic_client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0.15,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": user_txt},
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


def _intel_value_from_result(result: dict[str, Any]) -> str:
    try:
        m = result.get("layer_4_mission") or {}
        v = (m.get("intelligence_value") or "LOW").strip().upper()
        if v in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            return v
    except Exception:
        pass
    return "LOW"


def _autofile_forensic(
    result: dict[str, Any],
    raw: bytes,
    file_name: str,
    files_cabinet: Any,
    *,
    exif_block: dict[str, Any],
) -> dict[str, Any]:
    """HIGH/CRITICAL → FA-{date}-{hash} in Forensic Analysis."""
    iv = _intel_value_from_result(result)
    if iv not in ("HIGH", "CRITICAL"):
        return {"filed": False, "reason": "intelligence_value below threshold"}

    h = hashlib.sha256(raw[:50_000] + (file_name or "").encode()).hexdigest()[:10]
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    fname = f"{FA_PREFIX}{day}-{h}"
    body = json.dumps(
        {
            "file_name": file_name,
            "exif_summary": exif_block,
            "analysis": result,
        },
        ensure_ascii=False,
        indent=2,
    )
    tags = ["forensic_analysis", f"intelligence_value:{iv}"]
    try:
        if files_cabinet.get_file(fname):
            files_cabinet.update_file(fname, body)
        else:
            files_cabinet.create_file(FORENSIC_FOLDER, fname, body, tags=tags)
        return {"filed": True, "cabinet_file": fname, "folder": FORENSIC_FOLDER}
    except ValueError:
        try:
            files_cabinet.update_file(fname, body)
            return {"filed": True, "cabinet_file": fname, "folder": FORENSIC_FOLDER}
        except Exception as e:
            return {"filed": False, "error": str(e)}
    except Exception as e:
        return {"filed": False, "error": str(e)}


def _maybe_uap_incident_crossref(
    result: dict[str, Any],
    forensic_fname: str | None,
    files_cabinet: Any,
) -> dict[str, Any]:
    """Link anomalous/unknown UAP-relevant images to UAP Incidents folder."""
    ua = result.get("uap_analysis")
    if not isinstance(ua, dict):
        return {"cross_filed": False}
    assess = (ua.get("assessment") or "").upper()
    if assess not in ("UNKNOWN", "ANOMALOUS"):
        return {"cross_filed": False}
    if (result.get("analysis_mode") or "") == "uap":
        pass  # dedicated UAP pass — always allow cross-ref when assessment matches
    else:
        # Only create cross-ref when scene suggests aerial/UAP or layer 4 flags UAP relevance
        st = (result.get("scene_type") or "").lower()
        l4 = result.get("layer_4_mission") or {}
        uap_rel = (l4.get("uap_relevance") or "").lower() if isinstance(l4, dict) else ""
        if "aerial" not in st and "uap" not in uap_rel and "phenomenon" not in st:
            desc = str((result.get("layer_1_content") or {}).get("description", "")).lower()
            if not any(x in desc for x in ("sky", "aerial", "object", "light", "craft", "uap", "ufo")):
                return {"cross_filed": False, "reason": "not_uap_scene"}
    h = hashlib.sha256(json.dumps(ua, sort_keys=True).encode()).hexdigest()[:12]
    ref_name = f"REF-FORENSIC-{h}"
    body = "\n".join(
        [
            f"source_forensic_file: {forensic_fname or '(inline analysis)'}",
            f"assessment: {ua.get('assessment')}",
            f"confidence: {ua.get('confidence')}",
            "",
            "Cross-reference: Forensic visual analysis flagged UNKNOWN/ANOMALOUS UAP-relevant imagery.",
            "Review primary FA-* file in Forensic Analysis for full JSON.",
        ]
    )
    try:
        if files_cabinet.get_file(ref_name):
            files_cabinet.update_file(ref_name, body)
        else:
            files_cabinet.create_file(UAP_INCIDENTS_FOLDER, ref_name, body, tags=["forensic_crossref", "uap"])
        return {"cross_filed": True, "uap_ref_file": ref_name, "folder": UAP_INCIDENTS_FOLDER}
    except Exception as e:
        return {"cross_filed": False, "error": str(e)}


def forensic_analyze_image(
    image_b64: str,
    file_name: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: Any,
    *,
    memory_client: Any | None = None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
    model: str = "claude-sonnet-4-5",
    skip_autofile: bool = False,
) -> dict[str, Any]:
    """
    Full four-layer forensic analysis + optional mission cross-reference + auto-file.
    """
    fn = (file_name or "image.jpg").strip() or "image.jpg"
    try:
        raw, _ = decode_image_b64(image_b64)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    mt = _detect_media_type(raw, fn)
    exif_block = extract_exif_summary(raw)

    result = _run_vision_json(
        anthropic_client,
        raw,
        mt,
        fn,
        context,
        _system_prompt_full(),
        model=model,
    )
    if not result.get("ok"):
        return result

    result["metadata_exif"] = exif_block
    result["file_name"] = fn
    result["media_type"] = mt

    # Mission cross-reference from narrative
    narrative = json.dumps(
        {
            "l1": result.get("layer_1_content"),
            "l3": result.get("layer_3_intelligence"),
            "l4": result.get("layer_4_mission"),
        },
        ensure_ascii=False,
    )
    if memory_client and user_id and files_cabinet:
        result["mission_cross_reference"] = _crossref_mission(
            narrative,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )

    filed_meta: dict[str, Any] = {}
    if not skip_autofile:
        filed_meta = _autofile_forensic(result, raw, fn, files_cabinet, exif_block=exif_block)
        result["auto_file"] = filed_meta
        fname = filed_meta.get("cabinet_file") if filed_meta.get("filed") else None
        result["uap_incidents_crossref"] = _maybe_uap_incident_crossref(result, fname, files_cabinet)
    else:
        result["auto_file"] = {"filed": False, "skipped": True}

    iv = _intel_value_from_result(result)
    if iv in ("HIGH", "CRITICAL"):
        result["document_read_suggestion"] = (
            "If this is a document photograph, you may also run full document pipeline via POST /api/files/read with a scan/PDF export, "
            "or paste transcribed text for deeper analysis."
        )
    return result


def forensic_analyze_uap(
    image_b64: str,
    file_name: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """UAP-focused forensic pass (all layers + mandatory uap_analysis when applicable)."""
    fn = (file_name or "uap.jpg").strip() or "uap.jpg"
    try:
        raw, _ = decode_image_b64(image_b64)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    mt = _detect_media_type(raw, fn)
    exif_block = extract_exif_summary(raw)
    ctx = (context or "").strip() or "UAP / aerial phenomenon — forensic authenticity assessment requested."

    result = _run_vision_json(
        anthropic_client,
        raw,
        mt,
        fn,
        ctx,
        _system_prompt_uap(),
        model=kwargs.get("model", "claude-sonnet-4-5"),
    )
    if not result.get("ok"):
        return result
    result["metadata_exif"] = exif_block
    result["file_name"] = fn
    result["media_type"] = mt
    result["analysis_mode"] = "uap"

    memory_client = kwargs.get("memory_client")
    uid = kwargs.get("user_id")
    umc = kwargs.get("use_mem0_cloud", False)
    if memory_client and uid and files_cabinet:
        narrative = json.dumps(result, ensure_ascii=False)[:100_000]
        result["mission_cross_reference"] = _crossref_mission(
            narrative,
            memory_client=memory_client,
            user_id=uid,
            use_mem0_cloud=umc,
            files_cabinet=files_cabinet,
        )

    skip_autofile = bool(kwargs.get("skip_autofile", False))
    if not skip_autofile:
        filed_meta = _autofile_forensic(result, raw, fn, files_cabinet, exif_block=exif_block)
        result["auto_file"] = filed_meta
        fname = filed_meta.get("cabinet_file") if filed_meta.get("filed") else None
        result["uap_incidents_crossref"] = _maybe_uap_incident_crossref(result, fname, files_cabinet)
    else:
        result["auto_file"] = {"filed": False, "skipped": True}
        result["uap_incidents_crossref"] = _maybe_uap_incident_crossref(result, None, files_cabinet)
    return result


def forensic_analyze_document(
    image_b64: str,
    file_name: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Document-centric forensic pass."""
    fn = (file_name or "document.jpg").strip() or "document.jpg"
    try:
        raw, _ = decode_image_b64(image_b64)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    mt = _detect_media_type(raw, fn)
    exif_block = extract_exif_summary(raw)
    ctx = (context or "").strip() or "Document image — extract text and forensic authenticity."

    result = _run_vision_json(
        anthropic_client,
        raw,
        mt,
        fn,
        ctx,
        _system_prompt_document(),
        model=kwargs.get("model", "claude-sonnet-4-5"),
    )
    if not result.get("ok"):
        return result
    result["metadata_exif"] = exif_block
    result["file_name"] = fn
    result["media_type"] = mt
    result["analysis_mode"] = "document"
    result["full_file_reading_note"] = (
        "For multi-page PDFs or native files, use POST /api/files/read with the original file for full text extraction."
    )

    memory_client = kwargs.get("memory_client")
    uid = kwargs.get("user_id")
    umc = kwargs.get("use_mem0_cloud", False)
    if memory_client and uid and files_cabinet:
        narrative = json.dumps(result, ensure_ascii=False)[:100_000]
        result["mission_cross_reference"] = _crossref_mission(
            narrative,
            memory_client=memory_client,
            user_id=uid,
            use_mem0_cloud=umc,
            files_cabinet=files_cabinet,
        )

    skip_autofile = bool(kwargs.get("skip_autofile", False))
    if not skip_autofile:
        filed_meta = _autofile_forensic(result, raw, fn, files_cabinet, exif_block=exif_block)
        result["auto_file"] = filed_meta
    else:
        result["auto_file"] = {"filed": False, "skipped": True}
    return result


# --- Chat ---

_FORENSIC_TRIGGERS = re.compile(
    r"(?i)\b(?:"
    r"forensic\s+analysis|analyze\s+this\s+image\s+for\s+authenticity|"
    r"is\s+this\s+UAP\s+photo\s+real|is\s+this\s+real|is\s+this\s+fake|"
    r"authenticity|manipulated|photoshopped|deepfake|ai\s+generated|"
    r"examine\s+this\s+image|forensic\s+mode"
    r")\b"
)


def detect_forensic_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(r"(?i)\bwhat\s+does\s+this\s+document\s+say", msg) and re.search(
        r"(?i)\b(photo|image|picture|screenshot|scan|pic)\b", msg
    ):
        return "forensic_document", {}
    if _FORENSIC_TRIGGERS.search(msg):
        if re.search(r"(?i)\bUAP\b|ufo|flying\s+saucer", msg):
            return "forensic_uap", {}
        if re.search(r"(?i)document|memo|classified|paper|screenshot\s+of\s+(?:a\s+)?doc", msg):
            return "forensic_document", {}
        return "forensic_full", {}
    if re.search(r"(?i)\b(?:real|fake|authentic|manipulated|leaked)\b", msg) and re.search(
        r"(?i)\b(photo|image|picture|pic|screenshot)\b", msg
    ):
        return "forensic_full", {}
    return None, {}


def extract_inline_data_uri_image(user_message: str) -> tuple[str | None, str]:
    """If message contains embedded data:image/...;base64,..., return (b64_payload, rest_context)."""
    msg = user_message or ""
    m = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", msg, re.I)
    if not m:
        return None, msg
    return m.group(1).strip(), msg[: m.start()] + msg[m.end() :]


def format_forensic_chat_block(
    intent: str,
    *,
    anthropic_client: anthropic.Anthropic,
    files_cabinet: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    user_message: str,
) -> str:
    """Run forensic on inline image if present; else return API instructions."""
    b64, rest = extract_inline_data_uri_image(user_message)
    ctx = (rest or user_message or "").strip()[:4000]
    if not b64:
        return (
            "[Angel forensic visual analysis]\n"
            + json.dumps(
                {
                    "notice": "No image bytes in this chat message. "
                    "Use POST /api/forensic/analyze (or /api/forensic/uap, /api/forensic/document) with image_b64, "
                    "or enable forensic capture on the client. iPhone: use forensic endpoint instead of /api/vision when Tyler wants authenticity analysis.",
                    "intent": intent,
                },
                indent=2,
            )
        )
    fn = "chat-inline.jpg"
    try:
        if intent == "forensic_uap":
            r = forensic_analyze_uap(
                b64,
                fn,
                ctx,
                anthropic_client,
                files_cabinet,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
            )
        elif intent == "forensic_document":
            r = forensic_analyze_document(
                b64,
                fn,
                ctx,
                anthropic_client,
                files_cabinet,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
            )
        else:
            r = forensic_analyze_image(
                b64,
                fn,
                ctx,
                anthropic_client,
                files_cabinet,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
            )
        return "[Angel forensic visual analysis — structured result]\n" + json.dumps(r, ensure_ascii=False, indent=2)[
            :18000
        ]
    except Exception as e:
        return f"[Angel forensic analysis error]\n{str(e)[:800]}"
