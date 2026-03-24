"""
Theoretical Research Agent — ArXiv, NASA NTRS, DARPA/DTIC (Tavily), USPTO/Patents (PatentsView or Tavily).
Mem0 category research_cache (7-day TTL) for ArXiv query results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import requests

RESEARCH_INTEL_FOLDER = "Research Intelligence"
RES_PREFIX = "RES-"
ARXIV_API = "http://export.arxiv.org/api/query"
NASA_SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
NASA_CITATION_URL = "https://ntrs.nasa.gov/api/citations"
PATENTSVIEW_SEARCH_URL = "https://search.patentsview.org/api/v1/patent/"

NASA_QUERY_FOCUS = (
    "aerospace propulsion materials science energy systems advanced concepts"
)

CACHE_TTL_DAYS = 7

_THEORETICAL_TRIGGER = re.compile(
    r"(?i)\b(?:"
    r"research|papers?|peer[- ]?reviewed|studies|literature|arxiv|technical\s+reports?|"
    r"what\s+does\s+nasa\s+say|nasa\s+(?:say|report|paper)|\bnasa\b.*\b(?:about|on)\b|"
    r"\bdarpa\b|defense\s+advanced|patents?\s+for|uspto|state\s+of\s+the\s+art|\bsota\b|"
    r"theoretical\s+basis|science\s+behind|who'?s\s+working\s+on|who\s+is\s+working\s+on|"
    r"latest\s+(?:paper|research|results)|journal\s+article|doi\b|cite\s+sources?"
    r")\b"
)

_TECH_CONTEXT = re.compile(
    r"(?i)\b(?:"
    r"engineering|aerospace|propulsion|materials?|energy\s+systems?|physics|plasma|"
    r"semiconductor|algorithm|simulation|hardware|sensor|thermal|composite|quantum|"
    r"hypersonic|turbine|rocket|satellite|orbit|navigation|metamaterial"
    r")\b"
)

_LITERATURE_HINT = re.compile(
    r"(?i)\b(?:papers?|literature|arxiv|studies|research|peer|journal|citations?|doi)\b"
)

_HTTP_HEADERS = {"User-Agent": "AngelAssistant/1.0 (theoretical-research; +https://github.com)"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _cache_key(kind: str, query: str) -> str:
    h = hashlib.sha256(f"{kind}:{query.strip().lower()}".encode()).hexdigest()[:20]
    return f"{kind}:{h}"


def _research_cache_get(
    cache_key: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    from angel import CATEGORY_RESEARCH_CACHE, _load_local_memory_entries

    cutoff = _now_utc() - timedelta(days=CACHE_TTL_DAYS)
    for entry in _load_local_memory_entries(user_id):
        meta = entry.get("metadata") if isinstance(entry, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_RESEARCH_CACHE:
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


def _research_cache_set(
    cache_key: str,
    data: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> None:
    from angel import CATEGORY_RESEARCH_CACHE, add_structured_memory

    payload = {
        "key": cache_key,
        "created": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": data,
    }
    add_structured_memory(
        memory_client,
        user_id,
        json.dumps(payload, ensure_ascii=False),
        CATEGORY_RESEARCH_CACHE,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )


def _http_get(url: str, *, timeout: float = 35.0) -> tuple[int, str]:
    try:
        r = requests.get(url, timeout=timeout, headers=_HTTP_HEADERS)
        return r.status_code, r.text
    except Exception as e:
        return -1, str(e)


def _arxiv_ns() -> dict[str, str]:
    return {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }


def _arxiv_parse_entry(entry: ET.Element, ns: dict[str, str]) -> dict[str, Any]:
    def txt(tag: str) -> str:
        el = entry.find(tag, ns)
        if el is None or el.text is None:
            return ""
        return " ".join(el.text.split())

    aid = txt("atom:id")
    m = re.search(r"arxiv\.org/abs/([^?#]+)", aid)
    paper_id = m.group(1).rstrip("/") if m else aid

    authors: list[str] = []
    for a in entry.findall("atom:author", ns):
        name_el = a.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    cats: list[str] = []
    pc = entry.find("arxiv:primary_category", ns)
    if pc is not None:
        term = pc.get("term")
        if term:
            cats.append(term)
    for c in entry.findall("atom:category", ns):
        term = c.get("term")
        if term and term not in cats:
            cats.append(term)

    pdf_link = ""
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf" or (
            link.get("type") == "application/pdf" and link.get("href")
        ):
            pdf_link = link.get("href") or ""
            break
    if not pdf_link and paper_id:
        pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"

    return {
        "paper_id": paper_id,
        "title": txt("atom:title"),
        "authors": authors,
        "abstract": txt("atom:summary"),
        "published": txt("atom:published"),
        "updated": txt("atom:updated"),
        "pdf_link": pdf_link,
        "categories": cats,
    }


def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
    *,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    q = (query or "").strip()
    out: dict[str, Any] = {"ok": False, "query": q, "papers": [], "error": None}
    if not q:
        out["error"] = "empty query"
        return out

    parts = [f"all:{quote(q)}"]
    if categories:
        for c in categories:
            c = (c or "").strip()
            if c:
                parts.append(f"cat:{quote(c)}")
    search_query = "+AND+".join(parts) if len(parts) > 1 else parts[0]
    url = (
        f"{ARXIV_API}?search_query={search_query}&start=0&max_results={max(1, min(max_results, 50))}"
    )

    cache_key = _cache_key("arxiv_search", url)
    if memory_client and user_id:
        hit = _research_cache_get(cache_key, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit and hit.get("papers"):
            hit = dict(hit)
            hit["ok"] = True
            hit["cached"] = True
            return hit

    code, text = _http_get(url)
    if code != 200:
        out["error"] = f"http {code}: {text[:200]}"
        return out
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        out["error"] = f"xml parse: {e}"
        return out

    ns = _arxiv_ns()
    papers: list[dict[str, Any]] = []
    for ent in root.findall("atom:entry", ns):
        papers.append(_arxiv_parse_entry(ent, ns))

    out["ok"] = True
    out["papers"] = papers
    if memory_client and user_id and papers:
        _research_cache_set(
            cache_key,
            {"query": q, "papers": papers, "categories": categories or []},
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
        )
    return out


def get_arxiv_paper(
    arxiv_id: str,
    *,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    aid = (arxiv_id or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    out: dict[str, Any] = {"ok": False, "arxiv_id": aid, "paper": None, "error": None}
    if not aid:
        out["error"] = "empty arxiv_id"
        return out

    url = f"{ARXIV_API}?id_list={quote(aid)}"
    cache_key = _cache_key("arxiv_id", aid)
    if memory_client and user_id:
        hit = _research_cache_get(cache_key, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit and hit.get("paper"):
            hit = dict(hit)
            hit["ok"] = True
            hit["cached"] = True
            return hit

    code, text = _http_get(url)
    if code != 200:
        out["error"] = f"http {code}"
        return out
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        out["error"] = str(e)
        return out
    ns = _arxiv_ns()
    ent = root.find("atom:entry", ns)
    if ent is None:
        out["error"] = "not found"
        return out
    paper = _arxiv_parse_entry(ent, ns)
    out["ok"] = True
    out["paper"] = paper
    if memory_client and user_id:
        _research_cache_set(
            cache_key,
            {"arxiv_id": aid, "paper": paper},
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
        )
    return out


def _nasa_authors(affs: list[Any] | None) -> list[str]:
    names: list[str] = []
    for a in affs or []:
        if not isinstance(a, dict):
            continue
        meta = a.get("meta") or {}
        auth = (meta.get("author") or {}) if isinstance(meta, dict) else {}
        n = (auth.get("name") or "").strip()
        if n:
            names.append(n)
    return names


def _nasa_record(rec: dict[str, Any]) -> dict[str, Any]:
    rid = str(rec.get("id") or "")
    pubs = rec.get("publications") or []
    pub_date = ""
    if isinstance(pubs, list) and pubs:
        p0 = pubs[0] if isinstance(pubs[0], dict) else {}
        pub_date = (p0.get("publicationDate") or p0.get("issuePublicationDate") or "")[:10]

    downloads = rec.get("downloads") or []
    doc_url = ""
    if isinstance(downloads, list):
        for d in downloads:
            if isinstance(d, dict) and d.get("url"):
                doc_url = str(d["url"])
                break
    if not doc_url and rid:
        doc_url = f"https://ntrs.nasa.gov/citations/{rid}"

    kw = rec.get("keywords") or []
    subjects: list[str] = []
    if isinstance(kw, list):
        subjects = [str(x) for x in kw if x]
    idx = rec.get("index")
    if idx:
        subjects.append(f"index:{idx}")

    abstract = (rec.get("abstract") or rec.get("summary") or "").strip()

    return {
        "report_id": rid,
        "title": (rec.get("title") or "").strip(),
        "authors": _nasa_authors(rec.get("authorAffiliations")),
        "abstract": abstract,
        "publication_date": pub_date,
        "document_url": doc_url,
        "subject_categories": subjects,
        "sti_type": (rec.get("stiType") or "").strip(),
    }


def search_nasa_reports(query: str, max_results: int = 10) -> dict[str, Any]:
    q = (query or "").strip()
    out: dict[str, Any] = {"ok": False, "query": q, "reports": [], "error": None}
    if not q:
        out["error"] = "empty query"
        return out
    enriched = f"{q} {NASA_QUERY_FOCUS}".strip()
    body = {"query": enriched, "page": {"size": max(1, min(max_results, 50)), "from": 0}}
    try:
        r = requests.post(
            NASA_SEARCH_URL,
            json=body,
            headers={**_HTTP_HEADERS, "Content-Type": "application/json"},
            timeout=45,
        )
        if r.status_code != 200:
            out["error"] = f"http {r.status_code}: {r.text[:300]}"
            return out
        data = r.json()
    except Exception as e:
        out["error"] = str(e)
        return out

    results = data.get("results") or []
    reports = [_nasa_record(x) for x in results if isinstance(x, dict)]
    out["ok"] = True
    out["reports"] = reports
    return out


def get_nasa_report(report_id: str) -> dict[str, Any]:
    rid = (report_id or "").strip()
    out: dict[str, Any] = {"ok": False, "report_id": rid, "report": None, "error": None}
    if not rid:
        out["error"] = "empty report_id"
        return out
    try:
        r = requests.get(
            f"{NASA_CITATION_URL}/{quote(rid)}",
            headers=_HTTP_HEADERS,
            timeout=35,
        )
        if r.status_code != 200:
            out["error"] = f"http {r.status_code}"
            return out
        rec = r.json()
        if not isinstance(rec, dict):
            out["error"] = "invalid json"
            return out
    except Exception as e:
        out["error"] = str(e)
        return out
    out["ok"] = True
    out["report"] = _nasa_record(rec)
    return out


def _tavily_rows(query: str, *, max_results: int = 6) -> list[dict[str, Any]]:
    from angel import TAVILY_API_URL

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            TAVILY_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "topic": "general",
                "include_answer": False,
            },
            timeout=28,
        )
        resp.raise_for_status()
        return list(resp.json().get("results") or [])
    except Exception:
        return []


def search_darpa_programs(query: str) -> dict[str, Any]:
    q = (query or "").strip()
    out: dict[str, Any] = {"ok": False, "query": q, "programs": [], "error": None}
    if not q:
        out["error"] = "empty query"
        return out
    if not os.getenv("TAVILY_API_KEY"):
        out["error"] = "TAVILY_API_KEY not set"
        return out

    programs: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    def consume(rows: list[dict[str, Any]], source: str) -> None:
        for r in rows:
            url = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            snippet = (r.get("content") or r.get("snippet") or "").strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            office = ""
            mo = re.search(
                r"(?i)\b(TTO|STO|I2O|DSO|BTO)\b|Defense Sciences|Information Innovation|"
                r"Tactical Technology|Biological Technologies",
                title + " " + snippet,
            )
            if mo:
                office = mo.group(0)
            status = ""
            for token in ("Ended", "Active", "Archived", "completed"):
                if re.search(rf"(?i)\b{re.escape(token)}\b", snippet):
                    status = token
                    break
            pm = ""
            mpm = re.search(
                r"(?i)(?:program\s+manager|pm)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
                snippet,
            )
            if mpm:
                pm = mpm.group(1).strip()

            programs.append(
                {
                    "program_name": title or url,
                    "office": office,
                    "description": snippet[:1200],
                    "status": status,
                    "program_manager": pm,
                    "relevant_links": [url] if url else [],
                    "source": source,
                }
            )

    consume(_tavily_rows(f"site:darpa.mil {q}", max_results=6), "darpa.mil")
    consume(_tavily_rows(f"site:dtic.mil {q} technical report", max_results=4), "dtic.mil")

    out["ok"] = bool(programs)
    out["programs"] = programs[:15]
    if not programs:
        out["error"] = out["error"] or "no results"
    return out


def get_darpa_program_detail(program_name: str) -> dict[str, Any]:
    name = (program_name or "").strip()
    out: dict[str, Any] = {
        "ok": False,
        "program_name": name,
        "detail": None,
        "error": None,
    }
    if not name:
        out["error"] = "empty program_name"
        return out
    if not os.getenv("TAVILY_API_KEY"):
        out["error"] = "TAVILY_API_KEY not set"
        return out

    rows = _tavily_rows(f'site:darpa.mil "{name}" program', max_results=8)
    rows += _tavily_rows(f"site:dtic.mil {name}", max_results=4)
    blobs = []
    links: list[str] = []
    for r in rows:
        t = (r.get("title") or "").strip()
        c = (r.get("content") or "").strip()
        u = (r.get("url") or "").strip()
        if u and u not in links:
            links.append(u)
        blobs.append(f"{t}\n{u}\n{c}")

    out["ok"] = bool(blobs)
    out["detail"] = {
        "program_name": name,
        "aggregated_snippets": "\n\n---\n\n".join(blobs[:12])[:25_000],
        "goals_technical_areas": "",
        "performers_public": "",
        "funding_notes": "",
        "links": links[:20],
    }
    return out


def _patentsview_new_search(
    text_query: str,
    max_results: int,
) -> dict[str, Any]:
    key = (os.getenv("PATENTSVIEW_API_KEY") or "").strip()
    out: dict[str, Any] = {"ok": False, "patents": [], "error": None, "via": "patentsview_search"}
    if not key:
        out["error"] = "PATENTSVIEW_API_KEY not set"
        return out
    words = (text_query or "").strip()
    if not words:
        out["error"] = "empty query"
        return out

    q_obj: dict[str, Any] = {
        "_or": [
            {"_text_any": {"patent_title": words}},
            {"_text_any": {"patent_abstract": words}},
        ]
    }
    fields = [
        "patent_number",
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_date",
        "app_date",
        "assignees.assignee_organization",
        "inventors.inventor_name_first",
        "inventors.inventor_name_last",
        "cpc_current.cpc_subsection_id",
    ]
    params = {
        "q": json.dumps(q_obj),
        "f": json.dumps(fields),
        "o": json.dumps({"size": max(1, min(max_results, 25))}),
    }
    try:
        r = requests.get(
            PATENTSVIEW_SEARCH_URL,
            params=params,
            headers={**_HTTP_HEADERS, "X-Api-Key": key},
            timeout=40,
        )
        if r.status_code != 200:
            out["error"] = f"http {r.status_code}: {r.text[:400]}"
            return out
        data = r.json()
    except Exception as e:
        out["error"] = str(e)
        return out

    if data.get("error"):
        out["error"] = str(data.get("error"))
        return out

    raw_list = data.get("patents") or data.get("data") or []
    patents: list[dict[str, Any]] = []
    for p in raw_list:
        if not isinstance(p, dict):
            continue
        invs = p.get("inventors") or []
        inv_names: list[str] = []
        if isinstance(invs, list):
            for inv in invs:
                if not isinstance(inv, dict):
                    continue
                fn = (inv.get("inventor_name_first") or "").strip()
                ln = (inv.get("inventor_name_last") or "").strip()
                inv_names.append(f"{fn} {ln}".strip())
        asg = p.get("assignees") or []
        assignee = ""
        if isinstance(asg, list) and asg and isinstance(asg[0], dict):
            assignee = (asg[0].get("assignee_organization") or "").strip()

        cpcs = p.get("cpc_current") or []
        cpc_codes: list[str] = []
        if isinstance(cpcs, list):
            for c in cpcs:
                if isinstance(c, dict) and c.get("cpc_subsection_id"):
                    cpc_codes.append(str(c["cpc_subsection_id"]))

        patents.append(
            {
                "patent_number": (p.get("patent_number") or p.get("patent_id") or "").strip(),
                "title": (p.get("patent_title") or "").strip(),
                "abstract": (p.get("patent_abstract") or "").strip(),
                "inventors": inv_names,
                "assignee": assignee,
                "filing_date": (p.get("app_date") or "")[:10],
                "grant_date": (p.get("patent_date") or "")[:10],
                "cpc_codes": list(dict.fromkeys(cpc_codes)),
            }
        )

    out["ok"] = True
    out["patents"] = patents
    return out


def _patents_tavily(text_query: str, max_results: int) -> dict[str, Any]:
    q = (text_query or "").strip()
    out: dict[str, Any] = {"ok": False, "patents": [], "error": None, "via": "tavily_patents_google"}
    if not q:
        out["error"] = "empty query"
        return out
    rows = _tavily_rows(f"site:patents.google.com {q}", max_results=max_results)
    patents: list[dict[str, Any]] = []
    for r in rows:
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        url = (r.get("url") or "").strip()
        pnum = ""
        m = re.search(
            r"(?:patent|patents)\/(?:US)?([0-9]{6,})(?:[A-Z]\d+)?|US\s*([0-9]{6,})",
            url + " " + title,
            re.I,
        )
        if m:
            pnum = (m.group(1) or m.group(2) or "").strip()
        patents.append(
            {
                "patent_number": pnum,
                "title": title,
                "abstract": snippet[:2000],
                "inventors": [],
                "assignee": "",
                "filing_date": "",
                "grant_date": "",
                "cpc_codes": [],
                "source_url": url,
            }
        )
    out["ok"] = bool(patents)
    out["patents"] = patents
    if not patents:
        out["error"] = "no patent search results (Tavily)"
    return out


def search_patents(query: str, max_results: int = 10) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"ok": False, "query": q, "patents": [], "error": "empty query"}

    pv = _patentsview_new_search(q, max_results)
    if pv.get("ok"):
        pv["query"] = q
        return pv

    tv = _patents_tavily(q, max_results)
    tv["query"] = q
    tv.setdefault("note", "PatentsView legacy API discontinued; used Tavily on patents.google.com")
    return tv


def get_patent_detail(patent_number: str) -> dict[str, Any]:
    raw = (patent_number or "").strip().upper().replace("US", "").replace(",", "")
    out: dict[str, Any] = {"ok": False, "patent_number": raw, "patent": None, "error": None}
    if not raw:
        out["error"] = "empty patent_number"
        return out

    key = (os.getenv("PATENTSVIEW_API_KEY") or "").strip()
    if key:
        q_obj = {"patent_number": raw}
        params = {
            "q": json.dumps(q_obj),
            "f": json.dumps(
                [
                    "patent_number",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "app_date",
                    "assignees.assignee_organization",
                    "inventors.inventor_name_first",
                    "inventors.inventor_name_last",
                    "claims.claim_text",
                    "cpc_current.cpc_subsection_id",
                ]
            ),
            "o": json.dumps({"size": 1}),
        }
        try:
            r = requests.get(
                PATENTSVIEW_SEARCH_URL,
                params=params,
                headers={**_HTTP_HEADERS, "X-Api-Key": key},
                timeout=35,
            )
            if r.status_code == 200:
                data = r.json()
                plist = data.get("patents") or data.get("data") or []
                if plist and isinstance(plist[0], dict):
                    p0 = plist[0]
                    claims = p0.get("claims") or []
                    claim_texts: list[str] = []
                    if isinstance(claims, list):
                        for c in claims[:40]:
                            if isinstance(c, dict) and c.get("claim_text"):
                                claim_texts.append(str(c["claim_text"])[:500])
                    invs = p0.get("inventors") or []
                    inv_names: list[str] = []
                    if isinstance(invs, list):
                        for inv in invs:
                            if not isinstance(inv, dict):
                                continue
                            fn = (inv.get("inventor_name_first") or "").strip()
                            ln = (inv.get("inventor_name_last") or "").strip()
                            inv_names.append(f"{fn} {ln}".strip())
                    asg = p0.get("assignees") or []
                    assignee = ""
                    if isinstance(asg, list) and asg and isinstance(asg[0], dict):
                        assignee = (asg[0].get("assignee_organization") or "").strip()
                    cpcs = p0.get("cpc_current") or []
                    cpc_codes: list[str] = []
                    if isinstance(cpcs, list):
                        for c in cpcs:
                            if isinstance(c, dict) and c.get("cpc_subsection_id"):
                                cpc_codes.append(str(c["cpc_subsection_id"]))
                    out["ok"] = True
                    out["patent"] = {
                        "patent_number": (p0.get("patent_number") or raw).strip(),
                        "title": (p0.get("patent_title") or "").strip(),
                        "abstract": (p0.get("patent_abstract") or "").strip(),
                        "inventors": inv_names,
                        "assignee": assignee,
                        "filing_date": (p0.get("app_date") or "")[:10],
                        "grant_date": (p0.get("patent_date") or "")[:10],
                        "cpc_codes": cpc_codes,
                        "claims_summary": claim_texts[:15],
                        "via": "patentsview_search",
                    }
                    return out
        except Exception as e:
            out["error"] = str(e)

    tv = _patents_tavily(f"US{raw}", max_results=4)
    if tv.get("ok") and tv.get("patents"):
        p = tv["patents"][0]
        p["claims_summary"] = []
        p["via"] = "tavily_patents_google"
        out["ok"] = True
        out["patent"] = p
        return out

    if not out.get("error"):
        out["error"] = "patent not found"
    return out


def research_status() -> dict[str, Any]:
    return {
        "arxiv": {"available": True, "note": "public Atom API"},
        "nasa_ntrs": {"available": True, "note": "public REST API"},
        "darpa_dtic": {
            "available": bool(os.getenv("TAVILY_API_KEY")),
            "note": "Tavily site search (no public DARPA API)",
        },
        "patents": {
            "available": bool(os.getenv("PATENTSVIEW_API_KEY")) or bool(os.getenv("TAVILY_API_KEY")),
            "patentsview_search_api": bool(os.getenv("PATENTSVIEW_API_KEY")),
            "tavily_fallback": bool(os.getenv("TAVILY_API_KEY")),
            "note": "Prefer PATENTSVIEW_API_KEY for structured fields; else Tavily on patents.google.com",
        },
    }


def _gather_cross_references(files_cabinet: Any | None, query: str) -> dict[str, Any]:
    refs: dict[str, Any] = {"chemistry_intelligence": [], "osint_dossiers": []}
    if files_cabinet is None or not query:
        return refs
    terms = list(dict.fromkeys([w for w in re.findall(r"[A-Za-z]{5,}", query.lower())][:6]))
    seen: set[tuple[str, str]] = set()
    for t in terms:
        try:
            hits = files_cabinet.search_files(t)
        except Exception:
            continue
        for h in hits[:8]:
            if not isinstance(h, dict):
                continue
            folder = (h.get("folder") or "").strip()
            name = (h.get("name") or "").strip()
            key = (folder, name)
            if key in seen:
                continue
            seen.add(key)
            if folder == "Chemistry Intelligence":
                refs["chemistry_intelligence"].append({"file": name, "folder": folder})
            elif folder == "OSINT Dossiers":
                refs["osint_dossiers"].append({"file": name, "folder": folder})
    return refs


def maybe_file_research_intelligence(
    full_output: dict[str, Any],
    files_cabinet: Any | None,
    *,
    query: str,
    research_types_used: list[str],
) -> dict[str, Any]:
    if files_cabinet is None:
        return {"filed": False, "reason": "no files_cabinet"}
    mr = (full_output.get("mission_relevance") or "LOW").strip().upper()
    if mr not in ("HIGH", "CRITICAL"):
        return {"filed": False, "reason": "below filing threshold"}

    day = _now_utc().strftime("%Y%m%d")
    qh = hashlib.sha256((query or "").encode()).hexdigest()[:10]
    fname = f"{RES_PREFIX}{day}-{qh}"
    trl = full_output.get("technology_readiness")
    tags = [
        "research_intelligence",
        f"mission_relevance:{mr}",
        f"trl:{trl}",
        "sources:" + ",".join(research_types_used),
    ]
    tech_guess = full_output.get("technology_area_guess") or ""
    if tech_guess:
        tags.append(f"technology:{str(tech_guess)[:80]}")

    body_obj = dict(full_output)
    body_obj["cross_references"] = _gather_cross_references(files_cabinet, query)
    body = json.dumps(body_obj, ensure_ascii=False, indent=2)

    try:
        files_cabinet.create_file(RESEARCH_INTEL_FOLDER, fname, body, tags=tags)
        return {"filed": True, "cabinet_file": fname, "folder": RESEARCH_INTEL_FOLDER}
    except ValueError:
        fname2 = f"{RES_PREFIX}{day}-{qh}-b"
        try:
            files_cabinet.create_file(RESEARCH_INTEL_FOLDER, fname2, body, tags=tags)
            return {"filed": True, "cabinet_file": fname2, "folder": RESEARCH_INTEL_FOLDER}
        except Exception as e:
            return {"filed": False, "error": str(e)}
    except Exception as e:
        return {"filed": False, "error": str(e)}


def _synthesize_with_claude(
    query: str,
    context: str,
    sources_bundle: dict[str, Any],
    anthropic_client: Any,
) -> dict[str, Any]:
    compact = json.dumps(sources_bundle, ensure_ascii=False, indent=2)[:70_000]
    system = (
        "You are Angel's theoretical research synthesizer for Tyler's engineering and scientific questions. "
        "Output ONLY valid JSON (no markdown fences) with exactly these keys:\n"
        "- synthesis (string): unified technical narrative across sources; cite source types in prose (e.g. NASA report, arXiv paper).\n"
        "- key_discoveries (array of strings)\n"
        "- technology_readiness (integer 1-9): TRL estimate for the core technology in the query.\n"
        "- gaps_identified (array of strings): what is unknown or under-researched.\n"
        "- mission_relevance (string): one of LOW, MEDIUM, HIGH, CRITICAL for Tyler's UAP/disclosure/mission context.\n"
        "- technology_area_guess (short string): e.g. propulsion, materials, sensors.\n"
        "- recommended_followup (array of strings): specific papers, programs, or patent lines to dig into.\n"
        "- file_to_intelligence (boolean): true only if mission_relevance is HIGH or CRITICAL.\n"
        "Be conservative: do not invent citations not implied by the payload."
    )
    user = (
        f"Research query: {query}\n\nMission / user context: {context}\n\n"
        f"Structured source payload (ArXiv / NASA / DARPA+DTIC / Patents):\n{compact}"
    )
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        return {
            "synthesis": f"Synthesis failed: {e}",
            "key_discoveries": [],
            "technology_readiness": 3,
            "gaps_identified": [],
            "mission_relevance": "LOW",
            "technology_area_guess": "",
            "recommended_followup": [],
            "file_to_intelligence": False,
        }

    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    text = text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "synthesis": text[:8000] if text else "No synthesis produced.",
        "key_discoveries": [],
        "technology_readiness": 3,
        "gaps_identified": [],
        "mission_relevance": "MEDIUM",
        "technology_area_guess": "",
        "recommended_followup": [],
        "file_to_intelligence": False,
    }


def run_research_agent(
    query: str,
    context: str,
    research_types: list[str] | None = None,
    *,
    anthropic_client: Any,
    memory_client: Any | None = None,
    user_id: str = "",
    use_mem0_cloud: bool = False,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    q = (query or "").strip()
    ctx = (context or "").strip()
    types = research_types or ["arxiv", "nasa", "darpa", "patents"]
    types = [t.strip().lower() for t in types if t]

    sources: dict[str, Any] = {"arxiv": [], "nasa": [], "darpa": [], "patents": []}
    errors: dict[str, str] = {}

    uid = user_id or "default-user"

    def job_arxiv() -> None:
        if "arxiv" not in types:
            return
        try:
            r = search_arxiv(
                q,
                max_results=10,
                categories=None,
                memory_client=memory_client,
                user_id=uid,
                use_mem0_cloud=use_mem0_cloud,
            )
            if r.get("ok"):
                sources["arxiv"] = r.get("papers") or []
            else:
                errors["arxiv"] = str(r.get("error") or "unknown")
        except Exception as e:
            errors["arxiv"] = str(e)

    def job_nasa() -> None:
        if "nasa" not in types:
            return
        try:
            r = search_nasa_reports(q, max_results=10)
            if r.get("ok"):
                sources["nasa"] = r.get("reports") or []
            else:
                errors["nasa"] = str(r.get("error") or "unknown")
        except Exception as e:
            errors["nasa"] = str(e)

    def job_darpa() -> None:
        if "darpa" not in types:
            return
        try:
            r = search_darpa_programs(q)
            if r.get("ok"):
                sources["darpa"] = r.get("programs") or []
            else:
                errors["darpa"] = str(r.get("error") or "unknown")
        except Exception as e:
            errors["darpa"] = str(e)

    def job_patents() -> None:
        if "patents" not in types:
            return
        try:
            r = search_patents(q, max_results=10)
            if r.get("ok"):
                sources["patents"] = r.get("patents") or []
            else:
                errors["patents"] = str(r.get("error") or "unknown")
        except Exception as e:
            errors["patents"] = str(e)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(job_arxiv),
            ex.submit(job_nasa),
            ex.submit(job_darpa),
            ex.submit(job_patents),
        ]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:
                pass

    syn = _synthesize_with_claude(
        q,
        ctx,
        {"sources": sources, "errors": errors, "research_types": types},
        anthropic_client,
    )

    mr = (syn.get("mission_relevance") or "LOW").strip().upper()
    if mr not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        mr = "MEDIUM"
    try:
        trl = int(syn.get("technology_readiness", 3))
        trl = max(1, min(9, trl))
    except Exception:
        trl = 3
    syn["technology_readiness"] = trl
    syn["mission_relevance"] = mr

    file_intel = mr in ("HIGH", "CRITICAL")

    out: dict[str, Any] = {
        "query": q,
        "synthesis": syn.get("synthesis") or "",
        "key_discoveries": syn.get("key_discoveries") or [],
        "technology_readiness": trl,
        "gaps_identified": syn.get("gaps_identified") or [],
        "mission_relevance": mr,
        "sources": sources,
        "recommended_followup": syn.get("recommended_followup") or [],
        "file_to_intelligence": file_intel,
        "technology_area_guess": syn.get("technology_area_guess") or "",
        "errors": errors,
        "intelligence_filing": {"filed": False},
    }

    if file_intel:
        out["intelligence_filing"] = maybe_file_research_intelligence(
            out,
            files_cabinet,
            query=q,
            research_types_used=[t for t in types if t in ("arxiv", "nasa", "darpa", "patents")],
        )

    return out


def detect_theoretical_research_intent(
    user_message: str,
    *,
    skip_if_pure_chemistry: bool = False,
) -> tuple[bool, str]:
    raw = (user_message or "").strip()
    if len(raw) < 12:
        return False, raw
    if skip_if_pure_chemistry and not _LITERATURE_HINT.search(raw):
        return False, raw
    if _THEORETICAL_TRIGGER.search(raw):
        return True, raw
    if _TECH_CONTEXT.search(raw) and _LITERATURE_HINT.search(raw):
        return True, raw
    return False, raw


def format_research_agent_block_for_prompt(result: dict[str, Any]) -> str:
    top_sources: list[str] = []

    for p in (result.get("sources") or {}).get("arxiv") or []:
        if isinstance(p, dict) and p.get("title"):
            top_sources.append(f"arXiv: {p.get('title')} — {p.get('paper_id')}")
    for r in (result.get("sources") or {}).get("nasa") or []:
        if isinstance(r, dict) and r.get("title"):
            top_sources.append(f"NASA NTRS: {r.get('title')} — id {r.get('report_id')}")
    for d in (result.get("sources") or {}).get("darpa") or []:
        if isinstance(d, dict) and d.get("program_name"):
            top_sources.append(f"DARPA/DTIC: {d.get('program_name')}")
    for pat in (result.get("sources") or {}).get("patents") or []:
        if isinstance(pat, dict) and pat.get("title"):
            top_sources.append(f"Patent: {pat.get('title')} — {pat.get('patent_number')}")

    instructions = (
        "Use the JSON below. In your visible reply: (1) give a clear synthesis in natural language; "
        "(2) list roughly 3–5 of the strongest sources with one-line descriptions (choose from arXiv, NASA, DARPA/DTIC, patents); "
        "(3) offer to go deeper on any specific paper, NASA report, DARPA program, or patent. "
        "Do not dump the entire JSON as the answer."
    )
    payload = {
        "instructions": instructions,
        "structured": result,
        "top_source_lines": top_sources[:12],
    }
    return (
        "[Angel theoretical research — structured data: ArXiv, NASA NTRS, DARPA/DTIC (Tavily), "
        "Patents (PatentsView or patents.google.com via Tavily)]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def format_research_chat_block(
    user_message: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any | None,
) -> str:
    tri, tq = detect_theoretical_research_intent(user_message)
    if not tri:
        return ""
    try:
        res = run_research_agent(
            tq,
            user_message[:4000],
            None,
            anthropic_client=anthropic_client,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
            files_cabinet=files_cabinet,
        )
        return format_research_agent_block_for_prompt(res)
    except Exception as e:
        return (
            "[Angel theoretical research — error]\n"
            + json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        )
