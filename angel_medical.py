"""
Medical Intelligence Core (Build 5) — PubMed, openFDA, MedlinePlus, ClinicalTrials.gov.
Open-source APIs only; not a substitute for licensed clinical care.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree as ET

import requests

_log = logging.getLogger(__name__)

MEDICAL_INTEL_FOLDER = "Medical Intelligence"
BIO_INTEL_FOLDER = "Biological Intelligence"
MED_PREFIX = "MED-"

PUBMED_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OPENFDA_BASE = "https://api.fda.gov/drug/"
MEDLINEPLUS_WS = "https://wsearch.nlm.nih.gov/ws/query"
CLINICALTRIALS_V2 = "https://clinicaltrials.gov/api/v2/studies"

CACHE_TTL_DAYS = 7
_HTTP_TIMEOUT = 45.0
_HTTP_HEADERS = {"User-Agent": "AngelMedicalAssistant/1.0 (research; +https://github.com)"}

EVIDENCE_LEVELS = frozenset({"ESTABLISHED", "EMERGING", "EXPERIMENTAL", "THEORETICAL"})
AGENT_CLASS = frozenset({"NATURAL", "ENGINEERED", "UNKNOWN"})
MISSION_RELEVANCE = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _et_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join((el.text or "").split())


def _find_first(root: ET.Element, *names: str) -> ET.Element | None:
    for el in root.iter():
        if _localname(el.tag) in names:
            return el
    return None


def _findall_local(root: ET.Element, name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _localname(el.tag) == name]


def _cache_key(kind: str, query: str) -> str:
    h = hashlib.sha256(f"{kind}:{query.strip().lower()}".encode()).hexdigest()[:20]
    return f"medical:{kind}:{h}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _medical_cache_get(
    cache_key: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    from angel import CATEGORY_MEDICAL_CACHE, _load_local_memory_entries

    cutoff = _now_utc() - timedelta(days=CACHE_TTL_DAYS)
    for entry in _load_local_memory_entries(user_id):
        meta = entry.get("metadata") if isinstance(entry, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_MEDICAL_CACHE:
            continue
        raw = (entry.get("memory") or entry.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("key") != cache_key:
            continue
        try:
            ts = datetime.fromisoformat((obj.get("created") or "").replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        data = obj.get("data")
        if isinstance(data, dict):
            return data
    return None


def _medical_cache_set(
    cache_key: str,
    data: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> None:
    from angel import CATEGORY_MEDICAL_CACHE, add_structured_memory

    payload = {
        "key": cache_key,
        "created": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": data,
    }
    add_structured_memory(
        memory_client,
        user_id,
        json.dumps(payload, ensure_ascii=False),
        CATEGORY_MEDICAL_CACHE,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )


def _http_get(url: str, *, timeout: float = _HTTP_TIMEOUT) -> tuple[int, str]:
    try:
        r = requests.get(url, timeout=timeout, headers=_HTTP_HEADERS)
        return r.status_code, r.text
    except Exception as e:
        return -1, str(e)


# --- PubMed ---


def _parse_pubmed_article(medline: ET.Element) -> dict[str, Any]:
    pmid_el = _find_first(medline, "PMID", "pmid")
    pmid = _et_text(pmid_el)
    article = _find_first(medline, "Article")
    title = ""
    abstract_parts: list[str] = []
    authors: list[str] = []
    journal_name = ""
    pub_date = ""
    mesh_terms: list[str] = []
    pub_types: list[str] = []

    if article is not None:
        t_el = _find_first(article, "ArticleTitle")
        title = _et_text(t_el)
        abs_block = _find_first(article, "Abstract")
        if abs_block is not None:
            for at in _findall_local(abs_block, "AbstractText"):
                label = at.get("Label", "")
                chunk = _et_text(at)
                if label and chunk:
                    abstract_parts.append(f"{label}: {chunk}")
                elif chunk:
                    abstract_parts.append(chunk)
        auth_list = _find_first(article, "AuthorList")
        if auth_list is not None:
            for au in _findall_local(auth_list, "Author"):
                fn = _et_text(_find_first(au, "ForeName"))
                ln = _et_text(_find_first(au, "LastName"))
                coll = _et_text(_find_first(au, "CollectiveName"))
                if coll:
                    authors.append(coll)
                elif fn or ln:
                    authors.append(f"{fn} {ln}".strip())
        journal = _find_first(article, "Journal")
        if journal is not None:
            journal_name = _et_text(_find_first(journal, "Title"))
            pub_el = _find_first(journal, "JournalIssue", "PubDate")
            if pub_el is not None:
                y = _et_text(_find_first(pub_el, "Year"))
                m = _et_text(_find_first(pub_el, "Month"))
                d = _et_text(_find_first(pub_el, "Day"))
                pub_date = "-".join(x for x in (y, m, d) if x)

    mesh_list = _find_first(medline, "MeshHeadingList")
    if mesh_list is not None:
        for mh in _findall_local(mesh_list, "MeshHeading"):
            desc = _find_first(mh, "DescriptorName")
            if desc is not None and desc.text:
                mesh_terms.append(desc.text.strip())

    pub_hist = _find_first(medline, "PublicationTypeList")
    if pub_hist is not None:
        for pt in _findall_local(pub_hist, "PublicationType"):
            if pt.text:
                pub_types.append(pt.text.strip())

    return {
        "pmid": pmid,
        "title": title,
        "abstract": "\n\n".join(abstract_parts).strip(),
        "authors": authors[:40],
        "journal": journal_name,
        "publication_date": pub_date,
        "mesh_terms": mesh_terms[:40],
        "publication_types": pub_types,
    }


def _infer_study_quality(pub_types: list[str]) -> str:
    pt = " ".join(pub_types).lower()
    if "meta-analysis" in pt or "systematic review" in pt:
        return "meta-analysis_or_systematic_review"
    if "randomized controlled trial" in pt or "clinical trial" in pt or "controlled clinical trial" in pt:
        return "rct_or_clinical_trial"
    if "case reports" in pt:
        return "case_report"
    if "review" in pt:
        return "review"
    return "other_or_unknown"


def search_pubmed(
    query: str,
    max_results: int = 10,
    *,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    """esearch + efetch PubMed; returns articles with PMID, title, abstract, journal, MeSH, etc."""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty_query", "articles": []}

    ck = _cache_key("pubmed_search", f"{q}|{max_results}")
    if memory_client and user_id:
        hit = _medical_cache_get(ck, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit is not None:
            return {"ok": True, "articles": hit.get("articles", []), "cached": True, "query": q}

    term = urllib.parse.quote(q)
    es_url = (
        f"{PUBMED_EUTILS}esearch.fcgi?db=pubmed&term={term}&retmax={max(1, min(max_results, 50))}"
        "&retmode=json&sort=relevance"
    )
    code, body = _http_get(es_url)
    if code != 200:
        return {"ok": False, "error": f"esearch_http_{code}", "articles": [], "query": q}
    try:
        data = json.loads(body)
        id_list = (data.get("esearchresult") or {}).get("idlist") or []
    except Exception:
        return {"ok": False, "error": "esearch_parse", "articles": [], "query": q}

    if not id_list:
        out = {"ok": True, "articles": [], "query": q, "cached": False}
        if memory_client and user_id:
            _medical_cache_set(ck, {"articles": []}, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        return out

    ids = ",".join(id_list)
    ef_url = f"{PUBMED_EUTILS}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
    code2, xml_body = _http_get(ef_url)
    if code2 != 200:
        return {"ok": False, "error": f"efetch_http_{code2}", "articles": [], "query": q}

    articles: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_body.encode("utf-8", errors="replace"))
    except ET.ParseError:
        return {"ok": False, "error": "efetch_xml_parse", "articles": [], "query": q}

    for medline in root.iter():
        if _localname(medline.tag) == "MedlineCitation":
            art = _parse_pubmed_article(medline)
            if art.get("pmid"):
                art["study_quality_hint"] = _infer_study_quality(art.get("publication_types") or [])
                articles.append(art)

    result = {"ok": True, "articles": articles, "query": q, "cached": False}
    if memory_client and user_id:
        _medical_cache_set(ck, {"articles": articles}, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    return result


def get_pubmed_article(
    pmid: str,
    *,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    pid = re.sub(r"[^\d]", "", (pmid or "").strip())
    if not pid:
        return {"ok": False, "error": "invalid_pmid"}
    ck = _cache_key("pubmed_article", pid)
    if memory_client and user_id:
        hit = _medical_cache_get(ck, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit is not None and hit.get("article"):
            return {"ok": True, "article": hit["article"], "cached": True}

    ef_url = f"{PUBMED_EUTILS}efetch.fcgi?db=pubmed&id={pid}&retmode=xml"
    code, xml_body = _http_get(ef_url)
    if code != 200:
        return {"ok": False, "error": f"efetch_http_{code}"}
    try:
        root = ET.fromstring(xml_body.encode("utf-8", errors="replace"))
    except ET.ParseError:
        return {"ok": False, "error": "xml_parse"}
    for medline in root.iter():
        if _localname(medline.tag) == "MedlineCitation":
            art = _parse_pubmed_article(medline)
            art["study_quality_hint"] = _infer_study_quality(art.get("publication_types") or [])
            out = {"ok": True, "article": art, "cached": False}
            if memory_client and user_id:
                _medical_cache_set(ck, {"article": art}, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
            return out
    return {"ok": False, "error": "not_found"}


def search_clinical_trials(
    condition: str,
    status: str = "recruiting",
    *,
    max_results: int = 10,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    """
    Trial registry via ClinicalTrials.gov API v2 (NIH); satisfies structured trial fields
    requested alongside PubMed integration.
    """
    return search_trials(
        condition,
        intervention=None,
        status=status.upper() if status else "RECRUITING",
        max_results=max_results,
        memory_client=memory_client,
        user_id=user_id,
        use_mem0_cloud=use_mem0_cloud,
    )


# --- FDA openFDA ---


def search_drug(drug_name: str, *, max_results: int = 5) -> dict[str, Any]:
    name = (drug_name or "").strip()
    if not name:
        return {"ok": False, "error": "empty_name", "results": []}
    enc = urllib.parse.quote(name)
    search = urllib.parse.quote(f'openfda.generic_name:"{name}"+OR+openfda.brand_name:"{name}"')
    url = f"{OPENFDA_BASE}ndc.json?search={search}&limit={max_results}"
    code, body = _http_get(url)
    if code != 200:
        url2 = f"{OPENFDA_BASE}ndc.json?search=brand_name:{enc}&limit={max_results}"
        code, body = _http_get(url2)
    if code != 200:
        return {"ok": False, "error": f"openfda_http_{code}", "results": []}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json_parse", "results": []}
    raw_results = data.get("results") or []
    out: list[dict[str, Any]] = []
    for row in raw_results[:max_results]:
        if not isinstance(row, dict):
            continue
        of = row.get("openfda") if isinstance(row.get("openfda"), dict) else {}
        out.append(
            {
                "product_ndc": row.get("product_ndc"),
                "generic_name": row.get("generic_name"),
                "brand_name": row.get("brand_name"),
                "labeler_name": row.get("labeler_name"),
                "active_ingredients": row.get("active_ingredients"),
                "dosage_form": row.get("dosage_form"),
                "route": row.get("route"),
                "openfda_brand": of.get("brand_name"),
                "openfda_generic": of.get("generic_name"),
                "manufacturer_name": (of.get("manufacturer_name") or [None])[0]
                if isinstance(of.get("manufacturer_name"), list)
                else of.get("manufacturer_name"),
            }
        )
    return {"ok": True, "results": out, "query": name}


def get_drug_label(drug_name: str) -> dict[str, Any]:
    name = (drug_name or "").strip()
    if not name:
        return {"ok": False, "error": "empty_name"}
    enc = urllib.parse.quote(name)
    search = urllib.parse.quote(f'openfda.generic_name:"{name}"+OR+openfda.brand_name:"{name}"')
    url = f"{OPENFDA_BASE}label.json?search={search}&limit=1"
    code, body = _http_get(url)
    if code != 200:
        url2 = f"{OPENFDA_BASE}label.json?search=openfda.brand_name:\"{enc}\"&limit=1"
        code, body = _http_get(url2)
    if code != 200:
        return {"ok": False, "error": f"openfda_http_{code}"}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json_parse"}
    results = data.get("results") or []
    if not results or not isinstance(results[0], dict):
        return {"ok": False, "error": "not_found"}
    lab = results[0]
    of = lab.get("openfda") if isinstance(lab.get("openfda"), dict) else {}

    def _join_field(key: str) -> str:
        v = lab.get(key)
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if x)[:12000]
        return str(v or "")[:12000]

    text_blob = _join_field("warnings_and_cautions") + "\n" + _join_field("boxed_warning")
    black_box = bool(lab.get("boxed_warning")) or ("boxed" in text_blob.lower() and "warning" in text_blob.lower())

    return {
        "ok": True,
        "indications_and_usage": _join_field("indications_and_usage"),
        "contraindications": _join_field("contraindications"),
        "warnings": _join_field("warnings_and_cautions") or _join_field("warnings"),
        "boxed_warning": _join_field("boxed_warning"),
        "adverse_reactions": _join_field("adverse_reactions"),
        "dosage_and_administration": _join_field("dosage_and_administration"),
        "drug_interactions": _join_field("drug_interactions"),
        "black_box_warning_flag": black_box,
        "generic_name": (of.get("generic_name") or [None])[0] if isinstance(of.get("generic_name"), list) else of.get("generic_name"),
        "brand_name": (of.get("brand_name") or [None])[0] if isinstance(of.get("brand_name"), list) else of.get("brand_name"),
    }


def get_drug_adverse_events(drug_name: str, max_results: int = 10) -> dict[str, Any]:
    name = (drug_name or "").strip()
    if not name:
        return {"ok": False, "error": "empty_name", "events": []}
    search = urllib.parse.quote(
        f'patient.drug.openfda.generic_name:"{name}"+OR+patient.drug.openfda.brand_name:"{name}"'
    )
    url = f"{OPENFDA_BASE}event.json?search={search}&limit={max_results}"
    code, body = _http_get(url)
    if code != 200:
        return {"ok": False, "error": f"openfda_http_{code}", "events": []}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json_parse", "events": []}
    events_out: list[dict[str, Any]] = []
    for ev in (data.get("results") or [])[:max_results]:
        if not isinstance(ev, dict):
            continue
        patient = ev.get("patient") if isinstance(ev.get("patient"), dict) else {}
        drugs = patient.get("drug") if isinstance(patient.get("drug"), list) else []
        reaction = patient.get("reaction") if isinstance(patient.get("reaction"), list) else []
        reactions_txt = ", ".join(
            str((r or {}).get("reactionmeddrapt") or (r or {}).get("reactionmeddraversionpt") or "")
            for r in reaction[:5]
            if isinstance(r, dict)
        )
        serious = ev.get("serious")
        outcomes = []
        seq = patient.get("summary") if isinstance(patient.get("summary"), dict) else {}
        if isinstance(seq, dict):
            outcomes.append(str(seq.get("narrativeincludeclinical") or "")[:500])
        events_out.append(
            {
                "safety_report_id": ev.get("safetyreportid"),
                "serious": serious,
                "reactions": reactions_txt,
                "outcome_notes": " ".join(outcomes)[:800],
                "drug_count": len(drugs),
            }
        )
    return {"ok": True, "events": events_out, "query": name}


# --- MedlinePlus ---


def search_medical_topic(topic: str) -> dict[str, Any]:
    t = (topic or "").strip()
    if not t:
        return {"ok": False, "error": "empty_topic"}
    params = urllib.parse.urlencode({"db": "healthTopics", "term": t, "retmax": "5"})
    url = f"{MEDLINEPLUS_WS}?{params}"
    code, body = _http_get(url)
    if code != 200:
        return {"ok": False, "error": f"medlineplus_http_{code}", "raw_excerpt": body[:500]}
    try:
        root = ET.fromstring(body.encode("utf-8", errors="replace"))
    except ET.ParseError:
        return {"ok": False, "error": "xml_parse", "raw_excerpt": body[:800]}

    summaries: list[dict[str, str]] = []
    for doc in root.iter():
        if _localname(doc.tag) != "document":
            continue
        title = ""
        url_s = ""
        snippet = ""
        for ch in doc:
            ln = _localname(ch.tag)
            if ln == "content" and ch.get("name") == "title":
                title = _et_text(ch)
            if ln == "content" and ch.get("name") == "FullSummary":
                snippet = _et_text(ch)[:4000]
            if ln == "content" and ch.get("name") == "organizationName":
                pass
            if ln == "url":
                url_s = _et_text(ch)
        if title or snippet:
            summaries.append({"title": title, "url": url_s, "summary_excerpt": snippet[:2000]})

    return {"ok": True, "topic": t, "summaries": summaries[:5]}


def get_condition_summary(condition: str) -> dict[str, Any]:
    r = search_medical_topic(condition)
    if not r.get("ok"):
        return r
    parts = []
    for s in r.get("summaries") or []:
        if isinstance(s, dict):
            parts.append(f"## {s.get('title', '')}\n{s.get('summary_excerpt', '')}\nURL: {s.get('url', '')}")
    return {
        "ok": True,
        "condition": condition,
        "structured_summary": "\n\n".join(parts)[:12000],
        "summaries": r.get("summaries"),
    }


# --- ClinicalTrials.gov v2 ---


def search_trials(
    condition: str,
    intervention: str | None = None,
    status: str = "RECRUITING",
    max_results: int = 10,
    *,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    cond = (condition or "").strip()
    if not cond:
        return {"ok": False, "error": "empty_condition", "trials": []}

    iv = (intervention or "").strip()
    st = (status or "RECRUITING").strip().upper()
    ck = _cache_key("ctgov", f"{cond}|{iv}|{st}|{max_results}")
    if memory_client and user_id:
        hit = _medical_cache_get(ck, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit is not None:
            return {"ok": True, "trials": hit.get("trials", []), "cached": True}

    params: dict[str, str] = {
        "query.cond": cond,
        "pageSize": str(max(1, min(max_results, 50))),
        "format": "json",
    }
    if iv:
        params["query.intr"] = iv
    if st:
        params["filter.overallStatus"] = st
    q = urllib.parse.urlencode(params, doseq=True)
    url = f"{CLINICALTRIALS_V2}?{q}"
    code, body = _http_get(url)
    if code != 200:
        return {"ok": False, "error": f"ctgov_http_{code}", "trials": []}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json_parse", "trials": []}

    trials_out: list[dict[str, Any]] = []
    for study in (data.get("studies") or [])[:max_results]:
        if not isinstance(study, dict):
            continue
        proto = study.get("protocolSection") if isinstance(study.get("protocolSection"), dict) else {}
        ids = proto.get("identificationModule") if isinstance(proto.get("identificationModule"), dict) else {}
        status_m = proto.get("statusModule") if isinstance(proto.get("statusModule"), dict) else {}
        design = proto.get("designModule") if isinstance(proto.get("designModule"), dict) else {}
        desc = proto.get("descriptionModule") if isinstance(proto.get("descriptionModule"), dict) else {}
        cond_m = proto.get("conditionsModule") if isinstance(proto.get("conditionsModule"), dict) else {}
        arms = proto.get("armsInterventionsModule") if isinstance(proto.get("armsInterventionsModule"), dict) else {}
        elig = proto.get("eligibilityModule") if isinstance(proto.get("eligibilityModule"), dict) else {}
        contacts = proto.get("contactsLocationsModule") if isinstance(proto.get("contactsLocationsModule"), dict) else {}

        nct = ids.get("nctId") or study.get("nctId")
        title = ids.get("briefTitle") or ids.get("officialTitle") or ""
        phases = design.get("phases") if isinstance(design.get("phases"), list) else []
        overall = status_m.get("overallStatus") or ""
        enrollment = (design.get("enrollmentInfo") or {}).get("count") if isinstance(design.get("enrollmentInfo"), dict) else None
        conditions = cond_m.get("conditions") if isinstance(cond_m.get("conditions"), list) else []
        interventions = arms.get("interventions") if isinstance(arms.get("interventions"), list) else []
        iv_text = []
        for it in interventions[:8]:
            if isinstance(it, dict):
                iv_text.append(f"{it.get('type', '')}: {it.get('name', '')}")

        locs = contacts.get("locations") if isinstance(contacts.get("locations"), list) else []
        loc_names = []
        for loc in locs[:6]:
            if isinstance(loc, dict):
                fac = loc.get("facility") if isinstance(loc.get("facility"), dict) else {}
                loc_names.append(fac.get("name") or loc.get("city") or "")

        central_contacts = contacts.get("centralContacts") if isinstance(contacts.get("centralContacts"), list) else []
        contact_lines = []
        for c in central_contacts[:3]:
            if isinstance(c, dict):
                contact_lines.append(f"{c.get('name', '')} {c.get('phone', '')} {c.get('email', '')}")

        trials_out.append(
            {
                "nct_id": nct,
                "title": title,
                "status": overall,
                "phase": phases,
                "conditions": conditions[:12],
                "interventions": iv_text,
                "enrollment": enrollment,
                "eligibility_criteria": (elig.get("eligibilityCriteria") or "")[:4000],
                "brief_summary": (desc.get("briefSummary") or "")[:2000],
                "locations": loc_names,
                "contacts": contact_lines,
            }
        )

    if memory_client and user_id:
        _medical_cache_set(ck, {"trials": trials_out}, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    return {"ok": True, "trials": trials_out, "cached": False}


def get_trial_detail(nct_id: str) -> dict[str, Any]:
    raw = (nct_id or "").strip().upper()
    m = re.match(r"NCT\d+", raw)
    nid = m.group(0) if m else ""
    if not nid:
        return {"ok": False, "error": "invalid_nct_id"}
    url = f"{CLINICALTRIALS_V2}/{urllib.parse.quote(nid)}?format=json"
    code, body = _http_get(url)
    if code != 200:
        return {"ok": False, "error": f"ctgov_http_{code}"}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json_parse"}
    return {"ok": True, "study": data}


# --- Filing ---


def _med_file_slug(query: str) -> str:
    h = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:10]
    d = _now_utc().strftime("%Y%m%d")
    return f"{MED_PREFIX}{d}-{h}"


def _maybe_file_medical_intel(
    files_cabinet: Any,
    *,
    folder: str,
    query: str,
    body: dict[str, Any],
    tags: list[str],
) -> str | None:
    try:
        fn = _med_file_slug(query)
        text = json.dumps(body, ensure_ascii=False, indent=2)
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, text, skip_mem0=True)
        else:
            files_cabinet.create_file(folder, fn, text, tags=tags, skip_mem0=True)
        return fn
    except Exception as e:
        _log.debug("medical auto-file skipped: %s", e)
        return None


# --- Claude-backed analysis ---


def _claude_medical_json(
    anthropic_client: Any,
    system: str,
    user_blob: str,
) -> dict[str, Any]:
    from angel import call_claude

    try:
        raw = call_claude(anthropic_client, system, user_blob[:100000], model="claude-sonnet-4-5", prior_turns=None)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    t = (raw or "").strip()
    if t.startswith("(Angel encountered"):
        return {"ok": False, "error": t}
    if "```" in t:
        t = re.sub(r"^```[a-z]*\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    try:
        return {"ok": True, "data": json.loads(t)}
    except json.JSONDecodeError:
        return {"ok": True, "data": {"narrative_brief": t[:12000], "parse_error": True}}


def analyze_condition(
    condition_name: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    cond = (condition_name or "").strip()
    ctx = (context or "").strip()
    pub = search_pubmed(f"{cond} review OR guideline OR pathophysiology", max_results=8, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    mpl = get_condition_summary(cond)
    trials = search_trials(cond, status="RECRUITING", max_results=6, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)

    blob = json.dumps(
        {"pubmed": pub, "medlineplus": mpl, "recruiting_trials": trials, "user_context": ctx},
        ensure_ascii=False,
        indent=2,
    )[:95000]

    system = """You are a medical intelligence analyst. Using ONLY the JSON data provided (PubMed, MedlinePlus, trials), produce a single JSON object with keys:
- pathophysiology_overview (string, concise)
- standard_of_care (string)
- recent_research_developments (string)
- experimental_options (string, from trials + literature)
- genetic_factors_biomarkers (string, or "not highlighted in sources")
- prognosis_notes (string, cautious)
- evidence_quality (one of: ESTABLISHED, EMERGING, EXPERIMENTAL, THEORETICAL)
- mission_relevance (one of: LOW, MEDIUM, HIGH, CRITICAL) for Tyler's disclosure/UAP-adjacent mission context when applicable; else LOW/MEDIUM
- data_sources_used (array of short strings)
- limitations (string: not medical advice; open sources only)
JSON only, no markdown."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    out: dict[str, Any] = {
        "ok": parsed.get("ok", False),
        "condition": cond,
        "raw_pubmed": pub,
        "raw_medlineplus": mpl,
        "raw_trials": trials,
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
    }
    if parsed.get("error"):
        out["error"] = parsed["error"]

    mr = str((out.get("analysis") or {}).get("mission_relevance") or "").upper()
    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        fn = _maybe_file_medical_intel(
            files_cabinet,
            folder=MEDICAL_INTEL_FOLDER,
            query=cond,
            body=out,
            tags=["medical_intelligence", f"condition:{cond[:40]}", f"mission:{mr}", "sources:pubmed,medlineplus,ctgov"],
        )
        out["filed_as"] = fn

    return out


def analyze_drug(
    drug_name: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    d = (drug_name or "").strip()
    ctx = (context or "").strip()
    ndc = search_drug(d)
    label = get_drug_label(d)
    adv = get_drug_adverse_events(d, max_results=8)
    pub = search_pubmed(f"{d} mechanism OR pharmacology OR clinical trial", max_results=6, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)

    blob = json.dumps({"openfda_ndc": ndc, "label": label, "adverse_events": adv, "pubmed": pub, "context": ctx}, ensure_ascii=False, indent=2)[
        :95000
    ]

    system = """You are a medical intelligence analyst. From the JSON (FDA label snippets, adverse event summaries, PubMed), output ONE JSON object with keys:
- mechanism_of_action (string, best effort from sources)
- indications (string)
- contraindications (string)
- interactions (string)
- adverse_events_summary (string)
- black_box_warnings (string; say "none flagged" if not present)
- research_new_applications (string)
- evidence_quality (ESTABLISHED|EMERGING|EXPERIMENTAL|THEORETICAL)
- mission_relevance (LOW|MEDIUM|HIGH|CRITICAL)
- data_sources_used (array of strings)
- limitations (string)
JSON only."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    out = {
        "ok": parsed.get("ok", False),
        "drug": d,
        "raw_openfda_ndc": ndc,
        "raw_label": label,
        "raw_adverse_events": adv,
        "raw_pubmed": pub,
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
    }
    if parsed.get("error"):
        out["error"] = parsed["error"]
    if label.get("black_box_warning_flag"):
        out["black_box_warning_flag"] = True

    mr = str((out.get("analysis") or {}).get("mission_relevance") or "").upper()
    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        fn = _maybe_file_medical_intel(
            files_cabinet,
            folder=MEDICAL_INTEL_FOLDER,
            query=d,
            body=out,
            tags=["medical_intelligence", f"drug:{d[:40]}", f"mission:{mr}", "sources:openfda,pubmed"],
        )
        out["filed_as"] = fn
    return out


def analyze_treatment_options(
    condition: str,
    patient_context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    c = (condition or "").strip()
    pc = (patient_context or "").strip()
    pub = search_pubmed(f"{c} treatment efficacy randomized OR meta-analysis OR guideline", max_results=10, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    trials = search_trials(c, status="RECRUITING", max_results=8, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    mpl = get_condition_summary(c)

    blob = json.dumps({"pubmed": pub, "trials": trials, "medlineplus": mpl, "patient_context": pc}, ensure_ascii=False, indent=2)[:95000]
    system = """From the evidence JSON, output ONE JSON object:
- ranked_treatments (array of {name, rationale, evidence_snippet, evidence_tier})
- experimental_trials (array of short strings from trials data)
- comparison_notes (string)
- evidence_quality (ESTABLISHED|EMERGING|EXPERIMENTAL|THEORETICAL)
- mission_relevance (LOW|MEDIUM|HIGH|CRITICAL)
- limitations (string)
JSON only."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    out = {
        "ok": parsed.get("ok", False),
        "condition": c,
        "raw_pubmed": pub,
        "raw_trials": trials,
        "raw_medlineplus": mpl,
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
    }
    if parsed.get("error"):
        out["error"] = parsed["error"]
    mr = str((out.get("analysis") or {}).get("mission_relevance") or "").upper()
    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        fn = _maybe_file_medical_intel(
            files_cabinet,
            folder=MEDICAL_INTEL_FOLDER,
            query=f"treatments:{c}",
            body=out,
            tags=["medical_intelligence", "treatment_landscape", f"mission:{mr}"],
        )
        out["filed_as"] = fn
    return out


def search_medical_literature(
    query: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    q = (query or "").strip()
    pub = search_pubmed(q, max_results=12, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    arts = pub.get("articles") if isinstance(pub, dict) else []
    lines = []
    for a in arts or []:
        if not isinstance(a, dict):
            continue
        lines.append(
            f"PMID {a.get('pmid')}: {a.get('title')} [{a.get('study_quality_hint')}]\n{a.get('abstract', '')[:800]}"
        )
    blob = "CONTEXT:\n" + (context or "") + "\n\nARTICLES:\n" + "\n---\n".join(lines)[:90000]
    system = """Summarize the literature list into ONE JSON object:
- key_findings (string)
- study_quality_notes (string; call out RCT, meta-analysis, case series where visible)
- gaps (string)
- evidence_quality (ESTABLISHED|EMERGING|EXPERIMENTAL|THEORETICAL)
- mission_relevance (LOW|MEDIUM|HIGH|CRITICAL)
- limitations (string)
JSON only."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    out = {
        "ok": parsed.get("ok", False),
        "query": q,
        "raw_pubmed": pub,
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
    }
    if parsed.get("error"):
        out["error"] = parsed["error"]
    mr = str((out.get("analysis") or {}).get("mission_relevance") or "").upper()
    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        fn = _maybe_file_medical_intel(
            files_cabinet,
            folder=MEDICAL_INTEL_FOLDER,
            query=q,
            body=out,
            tags=["medical_intelligence", "literature", f"mission:{mr}"],
        )
        out["filed_as"] = fn
    return out


def analyze_biological_agent(
    agent_name: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    ag = (agent_name or "").strip()
    pub = search_pubmed(f"{ag} biosecurity OR pathogen OR CDC OR WHO", max_results=8, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    mpl = search_medical_topic(f"{ag} infection")

    blob = json.dumps({"pubmed": pub, "medlineplus_topic": mpl, "context": context or ""}, ensure_ascii=False, indent=2)[:95000]
    system = """You assess biological agents for open-source biosecurity-style intelligence. Output ONE JSON:
- agent_summary (string)
- transmission (string)
- lethality_morbidity_notes (string, from sources only)
- treatment_countermeasures (string)
- detection_diagnostics (string)
- historical_incidents (string)
- classification (NATURAL|ENGINEERED|UNKNOWN)
- mission_relevance (LOW|MEDIUM|HIGH|CRITICAL)
- evidence_quality (ESTABLISHED|EMERGING|EXPERIMENTAL|THEORETICAL)
- limitations (string; open sources; not classified data)
JSON only."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    out = {
        "ok": parsed.get("ok", False),
        "agent": ag,
        "raw_pubmed": pub,
        "raw_medlineplus": mpl,
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
    }
    if parsed.get("error"):
        out["error"] = parsed["error"]
    mr = str((out.get("analysis") or {}).get("mission_relevance") or "").upper()
    if mr in ("HIGH", "CRITICAL") and files_cabinet is not None:
        fn = _maybe_file_medical_intel(
            files_cabinet,
            folder=BIO_INTEL_FOLDER,
            query=f"bioagent:{ag}",
            body=out,
            tags=["biological_intelligence", "biothreat_assessment", f"agent:{ag[:40]}", f"mission:{mr}"],
        )
        out["filed_as"] = fn
    return out


# --- Personal health (Build 8 preview) ---


def update_health_profile(
    health_data: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    from angel import CATEGORY_PERSONAL_HEALTH, add_structured_memory

    if not isinstance(health_data, dict):
        return {"ok": False, "error": "health_data must be an object"}
    payload = {
        "updated_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age": health_data.get("age"),
        "weight": health_data.get("weight"),
        "conditions": health_data.get("conditions"),
        "medications": health_data.get("medications"),
        "allergies": health_data.get("allergies"),
        "fitness_metrics": health_data.get("fitness_metrics"),
        "sleep_data": health_data.get("sleep_data"),
        "notes": health_data.get("notes"),
    }
    text = json.dumps({"personal_health_profile": payload}, ensure_ascii=False)
    ok = add_structured_memory(
        memory_client,
        user_id,
        text,
        CATEGORY_PERSONAL_HEALTH,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )
    return {"ok": bool(ok), "stored_fields": list(payload.keys())}


def _load_latest_health_profile(memory_client: Any, user_id: str, use_mem0_cloud: bool) -> dict[str, Any] | None:
    from angel import CATEGORY_PERSONAL_HEALTH, fetch_combined_memories

    mem = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    best_ts = ""
    best: dict[str, Any] | None = None
    for m in mem or []:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        if meta.get("category") != CATEGORY_PERSONAL_HEALTH:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        prof = obj.get("personal_health_profile")
        if not isinstance(prof, dict):
            continue
        ts = str(prof.get("updated_at") or "")
        if ts >= best_ts:
            best_ts = ts
            best = prof
    return best


def get_health_recommendations(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    prof = _load_latest_health_profile(memory_client, user_id, use_mem0_cloud)
    if not prof:
        return {"ok": False, "error": "no_health_profile", "hint": "POST /api/medical/health-profile first"}

    conditions = prof.get("conditions")
    cond_q = ""
    if isinstance(conditions, list) and conditions:
        cond_q = str(conditions[0])
    elif isinstance(conditions, str):
        cond_q = conditions

    pub = {"articles": []}
    if cond_q:
        pub = search_pubmed(f"{cond_q} patient education OR guideline", max_results=5, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)

    blob = json.dumps({"profile": prof, "context": context, "pubmed": pub}, ensure_ascii=False, indent=2)[:80000]
    system = """You are Angel. Using the stored health profile and optional PubMed snippets, produce ONE JSON:
- personalized_recommendations (string: lifestyle, questions for their clinician, monitoring ideas — not prescriptions)
- research_highlights (string)
- evidence_quality (ESTABLISHED|EMERGING|EXPERIMENTAL|THEORETICAL)
- urgent_seek_care_if (string: red-flag symptoms to watch, generic)
- limitations (string: not a doctor; Tyler must consult licensed professionals)
JSON only."""

    parsed = _claude_medical_json(anthropic_client, system, blob)
    return {
        "ok": parsed.get("ok", False),
        "profile_snapshot": {k: prof.get(k) for k in ("age", "conditions", "medications", "allergies")},
        "analysis": parsed.get("data") if isinstance(parsed.get("data"), dict) else {},
        "error": parsed.get("error"),
    }


# --- Intent + prompt block ---


_MEDICAL_TRIGGERS = re.compile(
    r"(?i)\b("
    r"symptom|diagnosis|condition|disease|disorder|syndrome|medication|drug|pill|prescription|"
    r"treatment|therapy|clinical trial|biosecurity|pathogen|anthrax|smallpox|plague|ebola|"
    r"what does the research say about|pubmed|fda label|side effect|adverse event|contraindication|"
    r"medical study|health question|biological agent|bioweapon|vaccine trial|"
    r"headache|fever|nausea|chronic pain|acute pain|infection|cancer|diabetes|hypertension"
    r")\b"
)


def detect_medical_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if len(msg) < 12:
        return None, {}
    if not _MEDICAL_TRIGGERS.search(msg):
        return None, {}

    low = msg.lower()
    payload: dict[str, Any] = {"original": msg[:500]}

    if re.search(r"(?i)\b(clinical trial|recruiting trial|nct\d)", msg):
        m = re.search(r"(?i)for\s+([^?.!\n]{3,80})", msg)
        payload["condition"] = (m.group(1).strip() if m else msg[:120]).strip()
        return "trials", payload

    if re.search(r"(?i)\b(biological agent|pathogen|bioweapon|biosecurity)\b", msg):
        payload["agent"] = msg[:200]
        return "biological_threat", payload

    if re.search(r"(?i)\b(drug|medication|pill|metformin|ibuprofen|adverse event|side effect|fda)\b", msg):
        payload["drug"] = msg[:200]
        return "drug", payload

    if re.search(r"(?i)\b(treatment options|standard of care|what treats)\b", msg):
        payload["condition"] = msg[:200]
        return "treatment", payload

    if re.search(r"(?i)\b(literature|pubmed|papers on|studies on|research on)\b", msg):
        payload["query"] = msg[:300]
        return "literature", payload

    payload["condition"] = msg[:200]
    return "condition", payload


def run_medical_intent_for_chat(
    intent: str,
    payload: dict[str, Any],
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None,
) -> dict[str, Any]:
    ctx = str(payload.get("original") or "")
    try:
        if intent == "condition":
            return analyze_condition(str(payload.get("condition") or ctx), ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        if intent == "drug":
            line = str(payload.get("drug") or ctx)
            drug_guess = line
            m = re.search(r"(?i)\b(metformin|ibuprofen|aspirin|lisinopril|atorvastatin|omeprazole|sertraline|prednisone)\b", line)
            if m:
                drug_guess = m.group(1)
            return analyze_drug(drug_guess, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        if intent == "treatment":
            return analyze_treatment_options(str(payload.get("condition") or ctx), ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        if intent == "literature":
            return search_medical_literature(str(payload.get("query") or ctx), ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        if intent == "trials":
            cond = str(payload.get("condition") or ctx)
            return {
                "ok": True,
                "mode": "trials",
                "result": search_trials(cond, intervention=None, status="RECRUITING", max_results=10, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            }
        if intent == "biological_threat":
            ag = str(payload.get("agent") or ctx)[:200]
            return analyze_biological_agent(ag, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "unknown_intent"}


def format_medical_block_for_prompt(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"\n\n[Medical intelligence — tool error: {result.get('error', 'unknown')}]\n"
    if result.get("mode") == "trials":
        inner = result.get("result") or {}
        return (
            "\n\n[Medical intelligence — clinical trials (ClinicalTrials.gov; open sources only; not medical advice)]\n"
            + json.dumps(inner, ensure_ascii=False, indent=2)[:14000]
        )
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        ev = analysis.get("evidence_quality") or analysis.get("limitations")
        header = f"Evidence quality (per synthesis): {ev}\n" if ev else ""
        body = json.dumps(analysis, ensure_ascii=False, indent=2)[:12000]
        raw_note = ""
        if result.get("filed_as"):
            raw_note = f"\nAuto-filed intelligence file: {result.get('filed_as')}\n"
        return (
            "\n\n[Medical intelligence appendix — synthesized from PubMed / FDA / NIH / ClinicalTrials.gov where applicable. "
            "Not medical advice; Tyler should consult licensed professionals.]\n"
            + header
            + body
            + raw_note
        )
    return "\n\n[Medical intelligence — partial result]\n" + json.dumps(result, ensure_ascii=False, indent=2)[:8000]


def medical_databases_status() -> dict[str, Any]:
    """Lightweight reachability check (no keys required)."""
    checks: dict[str, Any] = {}
    c1, _ = _http_get(f"{PUBMED_EUTILS}einfo.fcgi?db=pubmed&retmode=json", timeout=12.0)
    checks["pubmed_eutils"] = c1 == 200
    c2, _ = _http_get(f"{OPENFDA_BASE}label.json?limit=1", timeout=12.0)
    checks["openfda_drug"] = c2 == 200
    c3, _ = _http_get(f"{MEDLINEPLUS_WS}?db=healthTopics&term=asthma&retmax=1", timeout=12.0)
    checks["medlineplus_ws"] = c3 == 200
    c4, _ = _http_get(f"{CLINICALTRIALS_V2}?pageSize=1&format=json", timeout=12.0)
    checks["clinicaltrials_gov_v2"] = c4 == 200
    checks["all_reachable"] = all(checks.values())
    return checks
