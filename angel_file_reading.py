"""
File reading & document intelligence for Angel — extract text, analyze with Claude,
cross-reference OSINT dossiers and mission network.
"""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import io
import json
import logging
import re
import traceback
from typing import Any

import anthropic

# --- Limits ---
_MAX_TEXT_FOR_MODEL = 100_000
_MAX_TEXT_STORE = 120_000
_TRUNC_NOTE = "\n\n[… truncated for length …]"
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB hard guard

_log = logging.getLogger("angel.file_reading")

# Per-request cap so Railway logs show failures instead of hanging on stalled API calls.
_ANTHROPIC_MESSAGES_TIMEOUT_SEC = 60.0
# Hard ceiling: SDK timeout may not abort blocked reads; outer thread join enforces a real limit.
_ANTHROPIC_HTTP_TIMEOUT_SEC = 30.0
_MESSAGES_FUTURE_TIMEOUT_SEC = 45.0
# cross_reference_intel: cap Mem0/graph and cabinet list_files so xref never blocks the request.
_CROSS_REF_EXTERNAL_TIMEOUT_SEC = 10.0


def _truncate(s: str, n: int = _MAX_TEXT_FOR_MODEL) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - len(_TRUNC_NOTE)] + _TRUNC_NOTE


def _guess_kind_from_name(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".pdf"):
        return "pdf"
    if n.endswith(".docx"):
        return "docx"
    if n.endswith(".doc"):
        return "doc"
    if n.endswith(".csv"):
        return "csv"
    if n.endswith((".xlsx", ".xlsm")):
        return "xlsx"
    if n.endswith(".xls"):
        return "xls"
    if n.endswith((".txt", ".md", ".markdown")):
        return "text"
    if n.endswith((".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".htm")):
        return "markup"
    if n.endswith(
        (
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".rs",
            ".go",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".cs",
            ".rb",
            ".sh",
            ".ps1",
            ".sql",
        )
    ):
        return "code"
    if n.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
        return "image"
    return "unknown"


def _mime_to_kind(mime: str) -> str | None:
    m = (mime or "").strip().lower().split(";")[0]
    mapping = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "doc",
        "text/csv": "csv",
        "text/plain": "text",
        "text/markdown": "text",
        "application/json": "markup",
        "text/html": "markup",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
        "image/png": "image",
        "image/jpeg": "image",
        "image/gif": "image",
        "image/webp": "image",
    }
    return mapping.get(m)


def classify_file(file_name: str, file_type: str, raw: bytes | None) -> str:
    """Return logical kind: pdf, docx, doc, csv, xlsx, text, code, markup, image, unknown."""
    fk = _mime_to_kind(file_type or "")
    if fk:
        return fk
    k = _guess_kind_from_name(file_name)
    if k != "unknown":
        return k
    if raw and len(raw) >= 4:
        if raw[:4] == b"%PDF":
            return "pdf"
        if raw[:2] == b"\xff\xd8":
            return "image"
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image"
        if raw[:4] in (b"GIF8",) or (raw[:4] == b"RIFF" and b"WEBP" in raw[:12]):
            return "image"
    return "unknown"


def _extract_pdf(raw: bytes) -> tuple[str, str]:
    """Returns (text, method_label)."""
    bio = io.BytesIO(raw)
    try:
        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(bio) as pdf:
            for page in pdf.pages[:400]:
                try:
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                except Exception:
                    continue
        txt = "\n\n".join(parts).strip()
        if len(txt) >= 40:
            return txt, "pdfplumber"
        _log.warning(
            "pdfplumber extracted too little text (len=%s) for PDF bytes=%s",
            len(txt),
            len(raw),
        )
    except Exception:
        _log.exception("PDF extraction failed with pdfplumber")
    try:
        import fitz  # pymupdf

        doc = fitz.open(stream=raw, filetype="pdf")
        parts = []
        for i, page in enumerate(doc):
            if i >= 400:
                break
            try:
                t = page.get_text("text") or ""
                if t.strip():
                    parts.append(t)
            except Exception:
                continue
        txt = "\n\n".join(parts).strip()
        if txt:
            return txt, "pymupdf"
        _log.warning(
            "pymupdf extracted no text for PDF bytes=%s",
            len(raw),
        )
    except Exception:
        _log.exception("PDF extraction failed with pymupdf")
    try:
        from PyPDF2 import PdfReader

        bio2 = io.BytesIO(raw)
        r = PdfReader(bio2)
        parts = []
        for p in r.pages[:400]:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                continue
        txt = "\n\n".join(parts).strip()
        if txt:
            return txt, "PyPDF2"
        _log.warning(
            "PyPDF2 extracted no text for PDF bytes=%s",
            len(raw),
        )
    except Exception:
        _log.exception("PDF extraction failed with PyPDF2")
    return "", "failed"


def _decode_file_b64(file_content_b64: str) -> tuple[bytes, str]:
    """
    Decode incoming base64 payload safely.
    Handles raw base64 and data-URI forms like:
    data:application/pdf;base64,JVBERi0x...
    """
    payload = (file_content_b64 or "").strip()
    if not payload:
        return b"", "empty"
    note = "raw_base64"
    if payload.lower().startswith("data:"):
        comma = payload.find(",")
        if comma > 0:
            hdr = payload[:comma].lower()
            payload = payload[comma + 1 :].strip()
            note = f"data_uri:{hdr[:120]}"
    payload = re.sub(r"\s+", "", payload)
    try:
        return base64.b64decode(payload, validate=True), note
    except binascii.Error:
        # Fallback for imperfect senders; still log for diagnosis.
        _log.warning("Strict base64 decode failed; retrying permissive decode.")
        try:
            return base64.b64decode(payload, validate=False), note + ";permissive"
        except Exception:
            _log.exception("Base64 decode failed (strict+permissive)")
            raise


def _extract_docx(raw: bytes) -> str:
    try:
        import docx

        d = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in d.paragraphs if p.text).strip()
    except Exception:
        return ""


def _extract_csv(raw: bytes) -> str:
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(raw), nrows=5000)
        return df.describe(include="all").to_string() + "\n\n--- head ---\n" + df.head(50).to_string()
    except Exception:
        try:
            import pandas as pd

            df = pd.read_csv(io.BytesIO(raw), header=None, nrows=200)
            return df.to_string()
        except Exception:
            return ""


def _extract_xlsx(raw: bytes) -> str:
    try:
        import pandas as pd

        xl = pd.ExcelFile(io.BytesIO(raw))
        chunks = [f"Sheets: {xl.sheet_names}"]
        for sheet in xl.sheet_names[:8]:
            df = pd.read_excel(xl, sheet_name=sheet, nrows=500)
            chunks.append(f"\n=== {sheet} ===\n" + df.head(80).to_string())
        return "\n".join(chunks)
    except Exception:
        return ""


def _extract_text_plain(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _analyze_with_claude_pdf_document(
    anthropic_client: anthropic.Anthropic,
    raw_pdf: bytes,
    file_name: str,
    context: str,
    model: str,
) -> dict[str, Any] | None:
    """Fallback when text extraction yields little — Claude reads PDF natively."""
    try:
        b64 = base64.standard_b64encode(raw_pdf).decode("ascii")
    except Exception:
        return None
    ctx = (context or "").strip() or "Tyler shared this PDF for intelligence review."
    user_text = (
        f"FILE_NAME: {file_name}\nCONTEXT FROM TYLER:\n{ctx[:8000]}\n\n"
        "This PDF could not be fully text-extracted. Read the document and produce the JSON analysis described in the system prompt."
    )
    def _call_pdf():
        return anthropic_client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.2,
            system=_analysis_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
            timeout=_ANTHROPIC_HTTP_TIMEOUT_SEC,
        )

    try:
        _log.info(
            "Claude PDF document fallback: messages.create model=%s file=%s pdf_bytes=%s http_timeout_s=%s future_timeout_s=%s",
            model,
            file_name,
            len(raw_pdf),
            _ANTHROPIC_HTTP_TIMEOUT_SEC,
            _MESSAGES_FUTURE_TIMEOUT_SEC,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call_pdf)
            try:
                resp = fut.result(timeout=_MESSAGES_FUTURE_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                _log.error(
                    "Claude PDF document fallback timed out after %ss file=%s",
                    _MESSAGES_FUTURE_TIMEOUT_SEC,
                    file_name,
                )
                return None
            except Exception as e:
                _log.exception("Claude PDF document fallback failed for file=%s err=%s", file_name, e)
                return None
        _log.info("Claude PDF document fallback: response received file=%s", file_name)
        return _parse_json_response(resp)
    except Exception:
        _log.exception("Claude PDF document fallback failed for file=%s", file_name)
        return None


def _vision_analyze_image(
    anthropic_client: anthropic.Anthropic,
    raw: bytes,
    image_media_type: str,
    file_name: str,
    context: str,
    model: str,
) -> dict[str, Any]:
    """Route images to Claude vision; returns partial analysis dict merged later."""
    try:
        b64 = base64.standard_b64encode(raw).decode("ascii")
    except Exception:
        return {"ok": False, "error": "base64 encode failed"}
    ctx = (context or "").strip() or "Tyler shared this image for review."
    q = (
        f"File name: {file_name}. Context: {ctx[:4000]}\n\n"
        "Describe everything visible (text, people, logos, documents in frame). "
        "Then output ONLY valid JSON with keys: summary (string), key_findings (array of strings), "
        "mission_relevance (string), intelligence_value (LOW|MEDIUM|HIGH|CRITICAL), "
        "entities_found (object with keys people, organizations, dates, locations — each array of strings), "
        "suggested_filing (string folder name under Intelligence)."
    )
    try:
        _log.info(
            "Claude vision (file read): messages.create model=%s file=%s media=%s raw_bytes=%s timeout_s=%s",
            model,
            file_name,
            image_media_type,
            len(raw),
            _ANTHROPIC_MESSAGES_TIMEOUT_SEC,
        )
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": image_media_type, "data": b64},
                        },
                        {"type": "text", "text": q},
                    ],
                }
            ],
            timeout=_ANTHROPIC_MESSAGES_TIMEOUT_SEC,
        )
        _log.info("Claude vision (file read): response received file=%s", file_name)
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                txt += block.get("text", "")
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                obj["ok"] = True
                obj["extracted_text"] = "[Image analysis] " + (obj.get("summary") or txt[:2000])
                obj["file_type_detected"] = f"image ({image_media_type})"
                return obj
        return {
            "ok": True,
            "extracted_text": "[Image] " + txt[:8000],
            "summary": txt[:1500],
            "file_type_detected": f"image ({image_media_type})",
        }
    except Exception as e:
        _log.error(
            "Claude vision (file read) failed file=%s err=%s\n%s",
            file_name,
            e,
            traceback.format_exc(),
        )
        return {"ok": False, "error": str(e)}


def _image_media_type(kind: str, file_name: str) -> str:
    n = file_name.lower()
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _analysis_system_prompt() -> str:
    return """You are Angel's document intelligence analyst for Tyler (UAP/disclosure, federal law enforcement mission).
Respond with a single JSON object ONLY (no markdown fences), keys:
- "file_type_detected": short string describing what you infer the document is
- "extracted_text": echo or summarize the supplied text excerpt (if huge, summarize faithfully); for code, note structure
- "summary": what the document is about (2-5 sentences)
- "key_findings": array of important bullet strings
- "mission_relevance": how this connects to Tyler's mission
- "intelligence_value": one of LOW, MEDIUM, HIGH, CRITICAL
- "suggested_filing": which Intelligence File Cabinet folder to use (e.g. "OSINT Dossiers", "Threat Intelligence", "Proactive Intelligence", "Foreign Intelligence", "Network Intelligence", or "General Intelligence")
- "entities_found": object with keys "people", "organizations", "dates", "locations" — each an array of strings (best effort from the text)

Be factual; do not invent content not supported by the input."""


def _parse_json_response(resp: Any) -> dict[str, Any] | None:
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
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _run_text_analysis(
    anthropic_client: anthropic.Anthropic,
    extracted_text: str,
    file_name: str,
    kind: str,
    context: str,
    *,
    model: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    ctx = (context or "").strip() or "General document review for Tyler's mission."
    user_block = f"""FILE_NAME: {file_name}
DETECTED_KIND: {kind}
CONTEXT FROM TYLER:
{ctx[:12000]}

--- DOCUMENT TEXT ---
{_truncate(extracted_text, _MAX_TEXT_FOR_MODEL)}
"""
    system_prompt = _analysis_system_prompt()
    mt = 8192

    def _call():
        return anthropic_client.messages.create(
            model=model,
            max_tokens=mt,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_block}],
            timeout=_ANTHROPIC_HTTP_TIMEOUT_SEC,
        )

    _log.info(
        "Claude text analysis: messages.create model=%s file=%s kind=%s user_block_chars=%s http_timeout_s=%s future_timeout_s=%s",
        model,
        file_name,
        kind,
        len(user_block),
        _ANTHROPIC_HTTP_TIMEOUT_SEC,
        _MESSAGES_FUTURE_TIMEOUT_SEC,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            resp = fut.result(timeout=_MESSAGES_FUTURE_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            _log.error(
                "Claude text analysis timed out after %ss file=%s kind=%s",
                _MESSAGES_FUTURE_TIMEOUT_SEC,
                file_name,
                kind,
            )
            return {
                "ok": False,
                "error": f"File analysis timed out after {_MESSAGES_FUTURE_TIMEOUT_SEC:.0f} seconds",
                "file_type_detected": kind,
                "extracted_text": _truncate(extracted_text, 8000),
            }
        except Exception as e:
            _log.error(
                "Claude text analysis API failed file=%s kind=%s err=%s\n%s",
                file_name,
                kind,
                e,
                traceback.format_exc(),
            )
            return {
                "ok": False,
                "error": f"Claude API error: {e}",
                "file_type_detected": kind,
                "extracted_text": _truncate(extracted_text, 8000),
            }
    _log.info("Claude text analysis: response received file=%s", file_name)

    parsed = _parse_json_response(resp)
    if not parsed:
        return {
            "ok": False,
            "error": "model did not return valid JSON",
            "file_type_detected": kind,
            "extracted_text": _truncate(extracted_text, 8000),
        }
    parsed["ok"] = True
    if "extracted_text" not in parsed or not (parsed.get("extracted_text") or "").strip():
        parsed["extracted_text"] = _truncate(extracted_text, _MAX_TEXT_STORE)
    return parsed


def cross_reference_intel(
    extracted_text: str,
    entities_found: dict[str, Any] | None,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    """Match document text/entities to mission network nodes and OSINT dossier filenames."""
    import angel as ang

    blob = (extracted_text or "")[:200_000].lower()
    hits: list[dict[str, Any]] = []

    def _load_graph():
        return ang.network_load_graph(memory_client, user_id, use_mem0_cloud, files_cabinet)

    nodes: dict[str, Any] | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex_g:
            _fut_g = _ex_g.submit(_load_graph)
            nodes, _edges = _fut_g.result(timeout=_CROSS_REF_EXTERNAL_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        _log.warning(
            "cross_reference_intel: network_load_graph timed out after %ss — skipping network matches",
            _CROSS_REF_EXTERNAL_TIMEOUT_SEC,
        )
    except Exception:
        pass

    if nodes is not None:
        try:
            for nid, node in (nodes or {}).items():
                name = (node.get("name") or nid or "").strip()
                if len(name) < 3:
                    continue
                if name.lower() in blob:
                    hits.append(
                        {
                            "type": "network_node",
                            "id": nid,
                            "name": name,
                            "relevance": node.get("relevance"),
                        }
                    )
        except Exception:
            pass

    # Entity strings (local only — no Mem0)
    ent = entities_found if isinstance(entities_found, dict) else {}
    for cat in ("people", "organizations"):
        for x in ent.get(cat) or []:
            s = str(x).strip()
            if len(s) >= 4 and s.lower() in blob and s not in [h.get("name") for h in hits]:
                hits.append({"type": "entity_mention", "name": s, "category": cat})

    osint_matches: list[str] = []

    def _list_osint_metas():
        return list(files_cabinet.list_files(folder=ang.OSINT_DOSSIERS_FOLDER))

    metas: list[Any] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex_o:
            _fut_o = _ex_o.submit(_list_osint_metas)
            metas = _fut_o.result(timeout=_CROSS_REF_EXTERNAL_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        _log.warning(
            "cross_reference_intel: OSINT list_files timed out after %ss — skipping dossier hits",
            _CROSS_REF_EXTERNAL_TIMEOUT_SEC,
        )
    except Exception:
        pass

    try:
        for meta in metas:
            fname = (meta.get("name") or "").strip()
            if not fname:
                continue
            stem = fname.replace("-OSINT", "").replace("_", " ").lower()
            if len(stem) > 3 and stem in blob:
                osint_matches.append(fname)
    except Exception:
        pass

    return {"network_matches": hits[:40], "osint_dossier_hits": osint_matches[:20]}


def _merge_cross_reference_intel(
    out: dict[str, Any],
    extracted_text: str,
    entities_found: dict[str, Any] | None,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> None:
    # Skipped — causes deadlock with local memory lock when xref runs in a worker thread.
    # Cross-referencing is non-critical for file reading; re-enable when lock contention is resolved.
    return


def read_and_analyze_file(
    file_content_b64: str,
    file_name: str,
    file_type: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    *,
    memory_client: Any | None = None,
    user_id: str | None = None,
    files_cabinet: Any | None = None,
    use_mem0_cloud: bool = False,
    model: str = "claude-haiku-4-5",
    skip_analysis: bool = False,
) -> dict[str, Any]:
    """
    Decode base64 file, extract text by type, analyze with Claude.
    Optionally cross-reference when memory_client + files_cabinet provided.
    """
    fn = (file_name or "upload.bin").strip() or "upload.bin"
    _log.info(
        "read_and_analyze_file: start file=%s type=%s model=%s b64_len=%s context_len=%s",
        fn,
        (file_type or "").strip() or "(none)",
        model,
        len(file_content_b64 or ""),
        len(context or ""),
    )
    raw = b""
    try:
        raw, decode_note = _decode_file_b64(file_content_b64 or "")
        _log.info(
            "read_and_analyze_file decode ok file=%s type=%s bytes=%s decode=%s",
            fn,
            (file_type or "").strip() or "(none)",
            len(raw),
            decode_note,
        )
    except Exception as e:
        _log.error(
            "read_and_analyze_file base64 decode failed file=%s type=%s err=%s\n%s",
            fn,
            (file_type or "").strip() or "(none)",
            e,
            traceback.format_exc(),
        )
        return {"ok": False, "error": f"invalid base64: {e}"}

    if not raw:
        return {"ok": False, "error": "empty file"}
    if len(raw) > _MAX_UPLOAD_BYTES:
        _log.warning(
            "read_and_analyze_file rejected oversized file=%s bytes=%s max=%s",
            fn,
            len(raw),
            _MAX_UPLOAD_BYTES,
        )
        return {
            "ok": False,
            "error": f"File too large ({len(raw)} bytes). Max supported is {_MAX_UPLOAD_BYTES} bytes.",
        }

    kind = classify_file(fn, file_type, raw)
    _log.info(
        "read_and_analyze_file classified file=%s type=%s kind=%s bytes=%s",
        fn,
        (file_type or "").strip() or "(none)",
        kind,
        len(raw),
    )
    extraction_method = ""
    extracted = ""
    vision_used = False
    pdf_doc_fallback = False

    if kind == "image":
        vision_used = True
        mt = _mime_to_kind(file_type or "") or _image_media_type(kind, fn)
        if mt == "image":
            mt = _image_media_type(kind, fn)
        _log.info(
            "read_and_analyze_file: image branch file=%s media_type=%s raw_bytes=%s — calling Claude vision",
            fn,
            mt,
            len(raw),
        )
        vis = _vision_analyze_image(anthropic_client, raw, mt, fn, context, model)
        _log.info(
            "read_and_analyze_file: image branch done file=%s ok=%s err=%s",
            fn,
            vis.get("ok"),
            vis.get("error"),
        )
        if not vis.get("ok"):
            return vis
        vis["vision_used"] = True
        vis["extraction_method"] = "claude_vision"
        if memory_client and files_cabinet and user_id:
            _log.info("read_and_analyze_file: cross_reference_intel (image) file=%s", fn)
            _merge_cross_reference_intel(
                vis,
                vis.get("extracted_text") or "",
                vis.get("entities_found") if isinstance(vis.get("entities_found"), dict) else {},
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
                files_cabinet=files_cabinet,
            )
        return vis

    if kind == "pdf":
        _log.info("read_and_analyze_file: pdf branch file=%s bytes=%s — extracting text", fn, len(raw))
        extracted, extraction_method = _extract_pdf(raw)
        if len(extracted) < 80:
            pdf_doc_fallback = True
            _log.info(
                "PDF text extraction low output file=%s method=%s len=%s; trying Claude document fallback.",
                fn,
                extraction_method or "failed",
                len(extracted),
            )
            doc_json = _analyze_with_claude_pdf_document(anthropic_client, raw, fn, context, model)
            if doc_json and doc_json.get("summary"):
                doc_json["ok"] = True
                doc_json["file_type_detected"] = doc_json.get("file_type_detected") or "pdf"
                doc_json["extracted_text"] = _truncate(
                    doc_json.get("extracted_text") or extracted or "[PDF read via Claude document]",
                    _MAX_TEXT_STORE,
                )
                doc_json["extraction_method"] = "claude_pdf_document"
                doc_json["vision_used"] = False
                if memory_client and files_cabinet and user_id:
                    _log.info("read_and_analyze_file: cross_reference_intel (pdf fallback) file=%s", fn)
                    ent = doc_json.get("entities_found") if isinstance(doc_json.get("entities_found"), dict) else {}
                    _merge_cross_reference_intel(
                        doc_json,
                        doc_json.get("extracted_text") or "",
                        ent,
                        memory_client=memory_client,
                        user_id=user_id,
                        use_mem0_cloud=use_mem0_cloud,
                        files_cabinet=files_cabinet,
                    )
                return doc_json
            _log.warning(
                "Claude PDF fallback did not return usable analysis file=%s",
                fn,
            )
        if not extracted:
            _log.error(
                "PDF read failed file=%s type=%s bytes=%s extraction_method=%s",
                fn,
                (file_type or "").strip() or "(none)",
                len(raw),
                extraction_method or "failed",
            )
            return {
                "ok": False,
                "error": "Could not extract text from PDF; try a different format or image export.",
                "file_type_detected": "pdf",
            }

    elif kind == "docx":
        extracted = _extract_docx(raw)
        extraction_method = "python-docx"
    elif kind == "doc":
        extracted = _extract_docx(raw)
        extraction_method = "python-docx"
        if len(extracted) < 20:
            extracted = ""
            extraction_method = "unsupported_legacy_doc"

    elif kind in ("csv",):
        extracted = _extract_csv(raw)
        extraction_method = "pandas_csv"

    elif kind in ("xlsx", "xls"):
        extracted = _extract_xlsx(raw)
        extraction_method = "pandas_excel"

    elif kind in ("text", "markup", "code"):
        extracted = _extract_text_plain(raw)
        extraction_method = "utf-8_text"

    else:
        # Unknown binary — try as text
        extracted = _extract_text_plain(raw)
        extraction_method = "utf-8_fallback"
        kind = "text"

    _log.info(
        "read_and_analyze_file: extraction done file=%s kind=%s method=%s extracted_chars=%s",
        fn,
        kind,
        extraction_method,
        len(extracted or ""),
    )

    if skip_analysis or not (extracted or "").strip():
        if not skip_analysis:
            _log.error(
                "No text extracted file=%s kind=%s method=%s bytes=%s",
                fn,
                kind,
                extraction_method,
                len(raw),
            )
            return {
                "ok": False,
                "error": f"No text extracted ({extraction_method}); file kind was {kind}.",
                "file_type_detected": kind,
                "extraction_method": extraction_method,
            }
        _log.info(
            "read_and_analyze_file: skip_analysis=True — skipping Claude text analysis file=%s kind=%s",
            fn,
            kind,
        )
        return {
            "ok": True,
            "summary": (extracted or "")[:500],
            "extracted_text": extracted or "",
            "file_type_detected": kind,
            "extraction_method": extraction_method,
            "intelligence_value": "MEDIUM",
            "vision_used": False,
        }

    _log.info(
        "read_and_analyze_file: invoking Claude text analysis model=%s file=%s kind=%s",
        model,
        fn,
        kind,
    )
    result = _run_text_analysis(anthropic_client, extracted, fn, kind, context, model=model)
    _log.info(
        "read_and_analyze_file: Claude text analysis returned ok=%s err=%s file=%s",
        result.get("ok"),
        result.get("error"),
        fn,
    )
    result["extraction_method"] = extraction_method + (";pdf_claude_fallback" if pdf_doc_fallback else "")
    result["vision_used"] = vision_used
    if not result.get("file_type_detected"):
        result["file_type_detected"] = kind

    if memory_client and files_cabinet and user_id:
        _log.info("read_and_analyze_file: cross_reference_intel (text) file=%s", fn)
        ent = result.get("entities_found") if isinstance(result.get("entities_found"), dict) else {}
        _merge_cross_reference_intel(
            result,
            extracted,
            ent,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )

    _log.info("read_and_analyze_file: complete ok=%s file=%s", result.get("ok"), fn)
    return result


def analyze_pasted_text(
    text: str,
    inferred_name: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    *,
    memory_client: Any | None = None,
    user_id: str | None = None,
    files_cabinet: Any | None = None,
    use_mem0_cloud: bool = False,
    model: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    """Direct analysis of pasted document text (no base64)."""
    result = _run_text_analysis(
        anthropic_client,
        text,
        inferred_name,
        "pasted_text",
        context,
        model=model,
    )
    result["extraction_method"] = "paste"
    result["vision_used"] = False
    if memory_client and files_cabinet and user_id:
        ent = result.get("entities_found") if isinstance(result.get("entities_found"), dict) else {}
        result.update(
            cross_reference_intel(
                text,
                ent,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
                files_cabinet=files_cabinet,
            )
        )
    return result


def looks_like_document_paste(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 800:
        return False
    lines = t.count("\n") + 1
    words = len(t.split())
    if lines >= 12 and words >= 120:
        return True
    if words >= 250:
        return True
    # Structured cues
    if re.search(r"(?i)(executive\s+summary|table\s+of\s+contents|confidential|appendix\s+[a-z])", t):
        return len(t) >= 500
    return False


def detect_file_read_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw:
        return None, {}

    if re.search(
        r"(?i)\b(?:read|analyze|summar(?:e|ise)|parse|review)\s+(?:this\s+)?(?:document|file|attachment|pdf|report)\b",
        raw,
    ):
        return "explicit_doc", {"inline": _strip_leading_command(raw)}

    if re.search(r"(?i)\bwhat\s+(?:does\s+)?this\s+(?:document|file|report)\s+say\b", raw) and len(raw) > 200:
        return "explicit_doc", {"inline": _strip_leading_command(raw)}

    if looks_like_document_paste(raw) and not re.match(
        r"(?i)^\s*(?:hi|hey|hello|thanks|thank you|ok|okay)[,!\s]",
        raw[:40],
    ):
        return "document_paste", {"text": raw}

    return None, {}


def _strip_leading_command(msg: str) -> str:
    """Remove first line if it's only a short command; keep body."""
    lines = msg.splitlines()
    if not lines:
        return msg
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    if len(rest) > 200:
        return rest
    if re.match(r"(?i).{0,80}\b(?:read|analyze|summarize|parse|review)\b.{0,80}$", first) and len(first) < 120:
        return rest if rest else msg
    return msg


def filing_suggestion_line(intel: str) -> str:
    v = (intel or "MEDIUM").strip().upper()
    if v in ("HIGH", "CRITICAL"):
        return (
            "Intelligence value is elevated — offer to file this in the suggested Intelligence folder immediately "
            "and offer to create or update related OSINT dossiers for any named principals."
        )
    return "Offer to file a summary in the File Cabinet under the suggested folder if Tyler wants it retained."


def format_file_read_for_prompt(
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
    """Build block for augmented user message."""
    ctx = "Tyler asked about this in chat."
    try:
        if intent == "document_paste":
            text = payload.get("text") or user_message
            r = analyze_pasted_text(
                text,
                "pasted-document.txt",
                ctx,
                anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
        elif intent == "explicit_doc":
            body = (payload.get("inline") or "").strip() or user_message
            if len(body.strip()) < 120 and not looks_like_document_paste(body):
                return (
                    "[Angel document / file analysis]\n"
                    + json.dumps(
                        {
                            "notice": "No substantial document text in this message. "
                            "Paste the document text here, or upload the file via POST /api/files/read — "
                            "I can analyze PDFs, Word, spreadsheets, text, code, and images.",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            r = analyze_pasted_text(
                body,
                "user-document.txt",
                ctx,
                anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
            )
        else:
            return ""
        if not r.get("ok") and r.get("error"):
            return f"[Angel file analysis — error]\n{r.get('error')}"

        nm = r.get("network_matches") or []
        od = r.get("osint_dossier_hits") or []
        filing = filing_suggestion_line(str(r.get("intelligence_value") or "MEDIUM"))

        block = {
            "summary": r.get("summary"),
            "key_findings": r.get("key_findings"),
            "mission_relevance": r.get("mission_relevance"),
            "intelligence_value": r.get("intelligence_value"),
            "suggested_filing": r.get("suggested_filing"),
            "entities_found": r.get("entities_found"),
            "network_matches": nm,
            "osint_dossier_hits": od,
            "filing_guidance": filing,
        }
        return "[Angel document / file analysis]\n" + json.dumps(block, ensure_ascii=False, indent=2)[:14000]
    except Exception as e:
        return f"[Angel file analysis — error]\n{str(e)[:500]}"


def suggest_filing_folder(analysis: dict[str, Any]) -> str:
    return str(analysis.get("suggested_filing") or "General Intelligence").strip() or "General Intelligence"
