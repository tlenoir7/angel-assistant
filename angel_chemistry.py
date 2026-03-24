"""
Chemistry & materials intelligence — PubChem, NIST WebBook, Materials Project,
synthesis briefs (Tavily + Claude), Mem0 cache (chemistry_cache, 30-day TTL).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import quote, urlencode

import requests

CHEM_INTEL_FOLDER = "Chemistry Intelligence"
CHEM_PREFIX = "CHEM-"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
NIST_CBOOK = "https://webbook.nist.gov/cgi/cbook.cgi"
CACHE_TTL_DAYS = 30

_CHEM_KEYWORDS = re.compile(
    r"(?i)\b(?:synthesis|synthesize|synthetic route|precursor|reaction|compound|chemical|"
    r"material|materials|alloy|polymer|ceramic|molecule|formula|element|properties of|"
    r"band gap|bandgap|crystal structure|enthalpy|entropy|thermodynamic|NIST|PubChem|"
    r"melting point|boiling point|solvent|catalyst|stoichiometry|reactant|reagent|"
    r"coating|composite|tensile|hardness|elastic|polymerization|hydrate|oxide|nitride|"
    r"carbide|allotropic|phase diagram|spectral|IR spectrum|mass spec|UV-Vis|"
    r"GHS|hazard|MSDS|SDS|bioactivity|assay|ligand|metallocene)\b"
)

_SYNTHESIS_TRIGGERS = re.compile(
    r"(?i)\b(?:synthesis|synthetic route|synthesize|how to make|prepare\s+\w+|"
    r"laboratory preparation|multi-?step synthesis)\b"
)

_DESIGN_TRIGGERS = re.compile(
    r"(?i)\b(?:design a material|material requirements|select a material|"
    r"materials selection|tensile strength|yield strength|requirements\s*:\s*\{)\b"
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _cache_key(kind: str, query: str) -> str:
    h = hashlib.sha256(f"{kind}:{query.strip().lower()}".encode()).hexdigest()[:20]
    return f"{kind}:{h}"


def _cache_get(
    cache_key: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    from angel import CATEGORY_CHEMISTRY_CACHE, _load_local_memory_entries

    cutoff = _now_utc() - timedelta(days=CACHE_TTL_DAYS)
    for entry in _load_local_memory_entries(user_id):
        meta = entry.get("metadata") if isinstance(entry, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != CATEGORY_CHEMISTRY_CACHE:
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


def _cache_set(
    cache_key: str,
    data: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> None:
    from angel import CATEGORY_CHEMISTRY_CACHE, add_structured_memory

    payload = {
        "key": cache_key,
        "created": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": data,
    }
    add_structured_memory(
        memory_client,
        user_id,
        json.dumps(payload, ensure_ascii=False),
        CATEGORY_CHEMISTRY_CACHE,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )


def _http_get(url: str, *, timeout: float = 25.0) -> tuple[int, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "AngelAssistant/1.0 (research)"})
        return r.status_code, r.text
    except Exception as e:
        return -1, str(e)


def _pubchem_json(url: str) -> dict[str, Any] | None:
    code, text = _http_get(url)
    if code != 200 or not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def search_compound(name_or_formula: str) -> dict[str, Any]:
    """
    Resolve name or formula to CID and identifiers (PubChem PUG REST).
    """
    q = (name_or_formula or "").strip()
    out: dict[str, Any] = {"ok": False, "query": q, "cids": [], "primary_cid": None}
    if not q:
        out["error"] = "empty query"
        return out

    url = f"{PUBCHEM_BASE}/compound/name/{quote(q, safe='')}/cids/JSON?MaxRecords=10"
    data = _pubchem_json(url)
    if not data or "IdentifierList" not in data:
        url2 = f"{PUBCHEM_BASE}/compound/fastformula/{quote(q, safe='')}/cids/JSON?MaxRecords=10"
        data = _pubchem_json(url2)
    if not data or "IdentifierList" not in data:
        out["error"] = "no PubChem match (try another spelling or CID)"
        return out
    cids = data["IdentifierList"].get("CID") or []
    if isinstance(cids, int):
        cids = [cids]
    if not cids:
        out["error"] = "no CIDs returned"
        return out
    cid = int(cids[0])
    out["ok"] = True
    out["cids"] = [int(x) for x in cids]
    out["primary_cid"] = cid

    prop_names = (
        "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,"
        "InChIKey,IUPACName,MonoisotopicMass,XLogP,ExactMass,TPSA,Complexity,"
        "Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount"
    )
    purl = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{prop_names}/JSON"
    props = _pubchem_json(purl)
    if props and props.get("PropertyTable", {}).get("Properties"):
        out["properties"] = props["PropertyTable"]["Properties"][0]
    else:
        out["properties"] = {}

    surl = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    sdata = _pubchem_json(surl)
    if sdata and sdata.get("InformationList", {}).get("Information"):
        syn = sdata["InformationList"]["Information"][0].get("Synonym") or []
        if isinstance(syn, list):
            out["synonyms"] = [str(s) for s in syn[:40]]
        else:
            out["synonyms"] = [str(syn)]
    else:
        out["synonyms"] = []

    return out


def get_compound_properties(cid: int) -> dict[str, Any]:
    """Extended property sheet + experimental annotations from PUG View (subset)."""
    cid = int(cid)
    prop_names = (
        "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,"
        "InChIKey,IUPACName,MonoisotopicMass,XLogP,ExactMass,TPSA,Complexity,"
        "Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount"
    )
    purl = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{prop_names}/JSON"
    props = _pubchem_json(purl)
    base_props: dict[str, Any] = {
        "ok": True,
        "primary_cid": cid,
        "cids": [cid],
        "properties": {},
        "synonyms": [],
    }
    if props and props.get("PropertyTable", {}).get("Properties"):
        base_props["properties"] = props["PropertyTable"]["Properties"][0]
    surl = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    sdata = _pubchem_json(surl)
    if sdata and sdata.get("InformationList", {}).get("Information"):
        syn = sdata["InformationList"]["Information"][0].get("Synonym") or []
        base_props["synonyms"] = [str(s) for s in (syn if isinstance(syn, list) else [syn])[:40]]

    experimental: dict[str, Any] = {}
    for heading in ("Melting Point", "Boiling Point", "Density", "Solubility", "Vapor Pressure", "Flash Point"):
        u = f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON?heading={quote(heading)}"
        j = _pubchem_json(u)
        if j:
            experimental[heading.lower().replace(" ", "_")] = _extract_pug_view_text(j)[:4000]

    base_props["experimental_notes"] = experimental
    return base_props


def _props_iupac_name(props: dict[str, Any]) -> str:
    p = props.get("properties") or {}
    return str(p.get("IUPACName") or "").strip()


def _extract_pug_view_text(node: Any, depth: int = 0) -> str:
    """Flatten PUG View JSON to readable lines (bounded)."""
    if depth > 14:
        return ""
    chunks: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("String", "Value", "Name", "Heading"):
                if isinstance(v, str) and len(v) < 800:
                    chunks.append(v)
            else:
                chunks.append(_extract_pug_view_text(v, depth + 1))
    elif isinstance(node, list):
        for item in node[:80]:
            chunks.append(_extract_pug_view_text(item, depth + 1))
    return "\n".join(x for x in chunks if x).strip()[:12000]


def get_compound_safety(cid: int) -> dict[str, Any]:
    cid = int(cid)
    out: dict[str, Any] = {"cid": cid, "ghs": {}, "raw_excerpt": ""}
    u = f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON?heading={quote('GHS Classification')}"
    j = _pubchem_json(u)
    if not j:
        return out
    text = _extract_pug_view_text(j)
    out["raw_excerpt"] = text[:8000]
    # Light structure extraction
    for label, pat in (
        ("signal_word", r"(?i)signal\s*word[:\s]+(\w+)"),
        ("p_codes", r"(?i)(P\d{3}(?:\+P\d{3})*)"),
        ("h_codes", r"\b(H\d{3}(?:\s*\+\s*H\d{3})*)\b"),
    ):
        m = re.search(pat, text)
        if m:
            out["ghs"][label] = m.group(1) if m.lastindex else m.group(0)
    return out


def get_compound_bioactivity(cid: int) -> dict[str, Any]:
    cid = int(cid)
    out: dict[str, Any] = {"cid": cid, "chembl_targets": [], "pubmed_count": None, "notes": ""}
    xu = f"{PUBCHEM_BASE}/compound/cid/{cid}/xrefs/ChEMBL/JSON"
    xd = _pubchem_json(xu)
    try:
        if xd and xd.get("InformationList", {}).get("Information"):
            info = xd["InformationList"]["Information"][0]
            out["chembl_targets"] = info.get("ChEMBL_ID") or info.get("URL") or []
            if isinstance(out["chembl_targets"], str):
                out["chembl_targets"] = [out["chembl_targets"]]
    except Exception:
        pass
    pu = f"{PUBCHEM_BASE}/compound/cid/{cid}/xrefs/PubMedID/JSON"
    pd = _pubchem_json(pu)
    try:
        if pd and pd.get("InformationList", {}).get("Information"):
            inf = pd["InformationList"]["Information"][0]
            pms = inf.get("PubMedID") or []
            if isinstance(pms, list):
                out["pubmed_count"] = len(pms)
            elif pms:
                out["pubmed_count"] = 1
    except Exception:
        pass
    u = f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON?heading={quote('Pharmacology and Biochemistry')}"
    j = _pubchem_json(u)
    if j:
        out["notes"] = _extract_pug_view_text(j)[:6000]
    return out


def _nist_find_id(name: str) -> str | None:
    q = (name or "").strip()
    if not q:
        return None
    url = f"{NIST_CBOOK}?{urlencode({'Name': q, 'Units': 'SI'})}"
    code, html = _http_get(url)
    if code != 200:
        return None
    m = re.search(r"cgi\?ID=([A-Z0-9]+)", html)
    return m.group(1) if m else None


def _nist_mask_page(nist_id: str, mask: str) -> str:
    url = f"{NIST_CBOOK}?{urlencode({'ID': nist_id, 'Units': 'SI', 'Mask': mask})}"
    _, html = _http_get(url)
    return _strip_html(html)[:12000]


def _strip_html(html: str) -> str:
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def get_thermodynamic_data(compound_name: str) -> dict[str, Any]:
    nid = _nist_find_id(compound_name)
    if not nid:
        return {"ok": False, "error": "NIST WebBook name not found", "query": compound_name}
    gas = _nist_mask_page(nid, "1")
    condensed = _nist_mask_page(nid, "2")
    return {
        "ok": True,
        "nist_id": nid,
        "gas_phase_thermo_text": gas,
        "condensed_phase_thermo_text": condensed,
    }


def get_spectral_data(compound_name: str) -> dict[str, Any]:
    nid = _nist_find_id(compound_name)
    if not nid:
        return {"ok": False, "error": "NIST WebBook name not found", "query": compound_name}
    # IR=80, Mass=90, UV=100 (legacy mask bits — NIST uses Mask hex; 1F is common multi)
    ir = _nist_mask_page(nid, "80")
    ms = _nist_mask_page(nid, "90")
    uv = _nist_mask_page(nid, "100")
    return {"ok": True, "nist_id": nid, "ir_excerpt": ir, "mass_spec_excerpt": ms, "uv_vis_excerpt": uv}


def get_phase_change_data(compound_name: str) -> dict[str, Any]:
    nid = _nist_find_id(compound_name)
    if not nid:
        return {"ok": False, "error": "NIST WebBook name not found", "query": compound_name}
    phase = _nist_mask_page(nid, "4")
    return {"ok": True, "nist_id": nid, "phase_change_text": phase}


def _mp_client():
    key = (os.getenv("MATERIALS_PROJECT_API_KEY") or "").strip()
    if not key:
        return None, None, "MATERIALS_PROJECT_API_KEY not set"
    try:
        from mp_api.client import MPRester
    except ImportError:
        return None, None, "mp-api package not installed"
    return MPRester, key, None


def _doc_to_dict(doc: Any) -> dict[str, Any]:
    for meth in ("model_dump", "dict"):
        if hasattr(doc, meth):
            try:
                d = getattr(doc, meth)()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    if hasattr(doc, "__dict__"):
        return {k: v for k, v in doc.__dict__.items() if not k.startswith("_")}
    return {"repr": repr(doc)[:500]}


def search_material(formula_or_name: str) -> dict[str, Any]:
    MPRester, key, err = _mp_client()
    if not MPRester:
        return {"ok": False, "error": err, "results": []}
    q = (formula_or_name or "").strip()
    if not q:
        return {"ok": False, "error": "empty query", "results": []}

    def run(mpr: Any) -> list[dict[str, Any]]:
        # Try formula first, then elements text search via reduced formula
        docs = mpr.materials.summary.search(formula=q, _limit=15)
        rows: list[dict[str, Any]] = []
        for d in docs:
            dd = _doc_to_dict(d)
            mid = dd.get("material_id") or dd.get("material_ids")
            rows.append(
                {
                    "material_id": mid if isinstance(mid, str) else (mid[0] if isinstance(mid, list) and mid else str(mid)),
                    "formula_pretty": dd.get("formula_pretty") or dd.get("pretty_formula"),
                    "space_group": dd.get("spacegroup_symbol") or dd.get("symmetry", {}).get("symbol")
                    if isinstance(dd.get("symmetry"), dict)
                    else dd.get("spacegroup_symbol"),
                    "density": dd.get("density"),
                    "band_gap": dd.get("band_gap"),
                    "formation_energy_per_atom": dd.get("formation_energy_per_atom"),
                    "energy_above_hull": dd.get("energy_above_hull"),
                    "is_stable": dd.get("is_stable"),
                }
            )
            if len(rows) >= 15:
                break
        return rows

    try:
        with MPRester(key) as mpr:
            rows = run(mpr)
        return {"ok": True, "query": q, "results": rows}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def get_material_properties(material_id: str) -> dict[str, Any]:
    MPRester, key, err = _mp_client()
    if not MPRester:
        return {"ok": False, "error": err}
    mid = (material_id or "").strip()
    if not mid:
        return {"ok": False, "error": "empty material_id"}
    try:
        with MPRester(key) as mpr:
            docs = list(mpr.materials.summary.search(material_ids=[mid], _limit=1))
            if not docs:
                return {"ok": False, "error": f"material_id {mid!r} not found"}
            summary = _doc_to_dict(docs[0])
            el_out: dict[str, Any] = {}
            try:
                el = mpr.elasticity.search(material_ids=[mid])
                el_list = list(el)
                if el_list:
                    el_out = _doc_to_dict(el_list[0])
            except Exception:
                pass
        return {"ok": True, "material_id": mid, "summary": summary, "elasticity": el_out or None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_material_stability(material_id: str) -> dict[str, Any]:
    base = get_material_properties(material_id)
    if not base.get("ok"):
        return base
    s = base.get("summary") or {}
    return {
        "ok": True,
        "material_id": material_id,
        "energy_above_hull": s.get("energy_above_hull"),
        "is_stable": s.get("is_stable"),
        "equilibrium_reaction": s.get("equilibrium_reaction"),
        "decomposition_enthalpy": s.get("decomposition_enthalpy"),
        "warnings": s.get("warnings"),
    }


def search_by_property(
    property_name: str,
    min_val: float | None,
    max_val: float | None,
    *,
    elements: list[str] | None = None,
    _limit: int = 25,
) -> dict[str, Any]:
    MPRester, key, err = _mp_client()
    if not MPRester:
        return {"ok": False, "error": err, "results": []}
    pname = (property_name or "").strip().lower().replace(" ", "_")
    kwargs: dict[str, Any] = {"_limit": _limit}
    if elements:
        kwargs["elements"] = elements
    lo, hi = min_val, max_val
    if pname in ("band_gap", "bandgap"):
        if lo is not None and hi is not None:
            kwargs["band_gap"] = (float(lo), float(hi))
        elif lo is not None:
            kwargs["band_gap"] = (float(lo), None)
        elif hi is not None:
            kwargs["band_gap"] = (None, float(hi))
    elif pname in ("density",):
        if lo is not None and hi is not None:
            kwargs["density"] = (float(lo), float(hi))
    elif pname in ("formation_energy_per_atom", "formation_energy"):
        if lo is not None and hi is not None:
            kwargs["formation_energy_per_atom"] = (float(lo), float(hi))
    elif pname in ("energy_above_hull", "e_above_hull"):
        if lo is not None and hi is not None:
            kwargs["energy_above_hull"] = (float(lo), float(hi))
    else:
        return {
            "ok": False,
            "error": f"unsupported property {property_name!r}; try band_gap, density, formation_energy_per_atom, energy_above_hull",
            "results": [],
        }

    filters_only = {k: v for k, v in kwargs.items() if k != "_limit"}
    meaningful = False
    for k, v in filters_only.items():
        if k == "elements" and isinstance(v, list) and len(v) > 0:
            meaningful = True
        elif isinstance(v, tuple) and (v[0] is not None or v[1] is not None):
            meaningful = True
    if not meaningful:
        return {"ok": False, "error": "no property range or elements for Materials Project search", "results": []}

    try:
        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(**kwargs)
            rows = []
            for d in docs:
                dd = _doc_to_dict(d)
                rows.append(
                    {
                        "material_id": dd.get("material_id"),
                        "formula_pretty": dd.get("formula_pretty"),
                        "band_gap": dd.get("band_gap"),
                        "density": dd.get("density"),
                        "formation_energy_per_atom": dd.get("formation_energy_per_atom"),
                        "energy_above_hull": dd.get("energy_above_hull"),
                        "is_stable": dd.get("is_stable"),
                    }
                )
        return {"ok": True, "property": pname, "results": rows}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def chemistry_status() -> dict[str, Any]:
    MPRester, key, err = _mp_client()
    return {
        "pubchem": {"available": True, "auth": "none"},
        "nist_webbook": {"available": True, "auth": "none"},
        "materials_project": {
            "available": MPRester is not None,
            "detail": None if MPRester else err,
        },
    }


def fetch_compound_bundle(
    query: str,
    *,
    use_cache: bool = True,
    memory_client: Any = None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    q = (query or "").strip()
    ck = _cache_key("compound_bundle", q)
    if use_cache and memory_client and user_id:
        hit = _cache_get(ck, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
        if hit:
            hit = dict(hit)
            hit["cache"] = "hit"
            return hit

    pc = search_compound(q)
    cid = pc.get("primary_cid") if pc.get("ok") else None
    safety: dict[str, Any] = {}
    bio: dict[str, Any] = {}
    props: dict[str, Any] = pc
    if cid:
        props = get_compound_properties(int(cid))
        safety = get_compound_safety(int(cid))
        bio = get_compound_bioactivity(int(cid))
    nist_name = q
    if pc.get("ok") and (pc.get("properties") or {}).get("IUPACName"):
        nist_name = str(pc["properties"]["IUPACName"])
    elif cid and props.get("properties"):
        iupac = _props_iupac_name(props)
        if iupac:
            nist_name = iupac
    thermo = get_thermodynamic_data(nist_name)
    spectral = get_spectral_data(nist_name)
    phase = get_phase_change_data(nist_name)

    out = {
        "ok": True,
        "query": q,
        "pubchem": props,
        "safety": safety,
        "bioactivity": bio,
        "nist": {"thermodynamic": thermo, "spectral": spectral, "phase_change": phase},
        "cache": "miss",
    }
    if use_cache and memory_client and user_id:
        _cache_set(ck, out, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    return out


def analyze_synthesis_route(
    target_compound: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any | None = None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    import angel as ang

    target = (target_compound or "").strip()
    ctx = (context or "").strip()
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    web_blob = ""
    if api_key:
        queries = [
            f"{target} synthesis organic chemistry route",
            f"{target} preparation laboratory procedure OR patent",
            f"{target} precursors reagents conditions",
        ]
        lines: list[str] = []
        for q in queries[:3]:
            for r in ang._tavily_search_one(q, api_key, max_results=4, search_depth="basic"):
                lines.append(f"- {r.get('title', '')}: {r.get('content', '')[:400]}")
        web_blob = "\n".join(lines)[:12000]

    base = fetch_compound_bundle(
        target,
        memory_client=memory_client,
        user_id=user_id,
        use_mem0_cloud=use_mem0_cloud,
    )

    sys = """You are a chemistry intelligence analyst. Output ONLY valid JSON (no markdown) with keys:
{
  "mission_relevance": "LOW|MEDIUM|HIGH|CRITICAL",
  "unusual_or_significant": "short note or empty",
  "precursors_suggested": ["names"],
  "routes": [{"name": "", "steps_summary": "", "conditions": "", "yield_notes": "", "safety": "", "equipment": ""}],
  "alternatives": ["brief"],
  "purity_and_workup": "",
  "summary": "2-4 sentences"
}
Use web snippets and PubChem context; flag uncertainty; no classified or proprietary lab claims."""

    user_txt = f"""TARGET: {target}
CONTEXT: {ctx[:4000]}

Open-web snippets (may be incomplete):
{web_blob[:10000]}

PubChem / NIST bundle (truncated):
{json.dumps(base, ensure_ascii=False)[:8000]}"""

    brief: dict[str, Any] = {}
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.2,
            system=sys,
            messages=[{"role": "user", "content": user_txt}],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            brief = json.loads(m.group(0))
    except Exception as e:
        brief = {"error": str(e), "summary": "Structured synthesis brief unavailable."}

    out = {
        "ok": True,
        "target": target,
        "context": ctx,
        "literature_snippets_chars": len(web_blob),
        "compound_data": base,
        "synthesis_brief": brief,
    }
    if files_cabinet:
        out["auto_file"] = maybe_file_chemistry_intel(
            f"Synthesis: {target}",
            out,
            files_cabinet,
            mission_relevance=str(brief.get("mission_relevance") or "MEDIUM"),
            unusual=bool((brief.get("unusual_or_significant") or "").strip()),
        )
    return out


def analyze_material_properties(
    material_description: str,
    use_case: str,
    *,
    anthropic_client: Any,
    memory_client: Any | None = None,
    user_id: str | None = None,
    use_mem0_cloud: bool = False,
    files_cabinet: Any | None = None,
) -> dict[str, Any]:
    import angel as ang

    desc = (material_description or "").strip()
    uc = (use_case or "").strip()
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    web_blob = ""
    if api_key:
        q = f"{desc} {uc} mechanical thermal electrical properties engineering datasheet"
        for r in ang._tavily_search_one(q, api_key, max_results=6, search_depth="basic"):
            web_blob += f"- {r.get('title', '')}: {r.get('content', '')[:450]}\n"
    mp = search_material(desc)
    mp_detail: dict[str, Any] | None = None
    if mp.get("ok") and mp.get("results"):
        mid = mp["results"][0].get("material_id")
        if mid:
            mp_detail = get_material_properties(str(mid))

    sys = """Output ONLY valid JSON (no markdown):
{
  "mission_relevance": "LOW|MEDIUM|HIGH|CRITICAL",
  "unusual_or_significant": "",
  "mechanical": "",
  "thermal": "",
  "electrical": "",
  "chemical_stability": "",
  "comparables": [{"material": "", "tradeoffs": ""}],
  "commercial": "",
  "processing": "",
  "summary": ""
}
Ground answers in MP data when present; otherwise literature snippets — cite uncertainty."""

    user_txt = f"""MATERIAL: {desc}
USE CASE: {uc}

Materials Project search (may be empty):
{json.dumps(mp, ensure_ascii=False)[:6000]}

MP detail (first hit):
{json.dumps(mp_detail or {}, ensure_ascii=False)[:6000]}

Web snippets:
{web_blob[:9000]}"""

    brief: dict[str, Any] = {}
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0.2,
            system=sys,
            messages=[{"role": "user", "content": user_txt}],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            brief = json.loads(m.group(0))
    except Exception as e:
        brief = {"error": str(e)}

    out = {
        "ok": True,
        "material_description": desc,
        "use_case": uc,
        "materials_project": {"search": mp, "primary_detail": mp_detail},
        "property_brief": brief,
    }
    if files_cabinet:
        out["auto_file"] = maybe_file_chemistry_intel(
            f"Material: {desc[:60]}",
            out,
            files_cabinet,
            mission_relevance=str(brief.get("mission_relevance") or "LOW"),
            unusual=bool((brief.get("unusual_or_significant") or "").strip()),
        )
    return out


def design_material_for_requirements(
    requirements_dict: dict[str, Any],
    *,
    anthropic_client: Any,
) -> dict[str, Any]:
    """Use Claude to map requirements to MP queries, then rank results."""
    MPRester, key, err = _mp_client()
    req = requirements_dict or {}
    sys = """Given material requirements JSON, output ONLY valid JSON:
{
  "mp_searches": [
    {"property": "band_gap|density|formation_energy_per_atom|energy_above_hull", "min": number|null, "max": number|null, "elements": ["optional"]}
  ],
  "requirement_conflicts": ["human-readable conflicts"],
  "notes": ""
}
Use null when bounds unknown. Keep at most 3 searches."""

    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            temperature=0.1,
            system=sys,
            messages=[
                {
                    "role": "user",
                    "content": f"Requirements:\n{json.dumps(req, ensure_ascii=False)[:6000]}",
                }
            ],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        m = re.search(r"\{[\s\S]*\}", txt)
        plan = json.loads(m.group(0)) if m else {}
    except Exception as e:
        plan = {"error": str(e), "mp_searches": [], "requirement_conflicts": [], "notes": ""}

    merged: list[dict[str, Any]] = []
    if MPRester:
        for s in (plan.get("mp_searches") or [])[:3]:
            if not isinstance(s, dict):
                continue
            prop = str(s.get("property") or "band_gap")
            lo, hi = s.get("min"), s.get("max")
            els = s.get("elements") if isinstance(s.get("elements"), list) else None
            if lo is None and hi is None and not els:
                continue
            merged.append(
                search_by_property(
                    prop,
                    lo,
                    hi,
                    elements=els,
                    _limit=20,
                )
            )
    else:
        merged = [{"ok": False, "error": err}]

    rank_sys = """Rank MP candidates against user requirements. Output ONLY JSON:
{
  "ranked": [{"material_id": "", "formula": "", "score": 0-100, "rationale": "", "tradeoffs": ""}],
  "summary": ""
}"""

    rank_user = f"""Requirements:\n{json.dumps(req, ensure_ascii=False)[:4000]}\n\nMP batches:\n{json.dumps(merged, ensure_ascii=False)[:10000]}"""
    ranked: dict[str, Any] = {}
    try:
        r2 = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2048,
            temperature=0.15,
            system=rank_sys,
            messages=[{"role": "user", "content": rank_user}],
        )
        t2 = ""
        for block in r2.content:
            if getattr(block, "type", None) == "text":
                t2 += block.text
        m2 = re.search(r"\{[\s\S]*\}", t2)
        if m2:
            ranked = json.loads(m2.group(0))
    except Exception as e:
        ranked = {"error": str(e)}

    return {
        "ok": True,
        "requirements": req,
        "plan": plan,
        "mp_batches": merged,
        "ranked": ranked,
        "materials_project_available": MPRester is not None,
    }


def maybe_file_chemistry_intel(
    title: str,
    payload: dict[str, Any],
    files_cabinet: Any | None,
    *,
    mission_relevance: str = "MEDIUM",
    unusual: bool = False,
) -> dict[str, Any]:
    if files_cabinet is None:
        return {"filed": False, "reason": "no files_cabinet"}
    mr = (mission_relevance or "LOW").strip().upper()
    if mr not in ("HIGH", "CRITICAL") and not unusual:
        return {"filed": False, "reason": "below filing threshold"}
    day = _now_utc().strftime("%Y%m%d")
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    fname = f"{CHEM_PREFIX}{day}-{h}"
    body = json.dumps(
        {"title": title, "mission_relevance": mr, "payload": payload},
        ensure_ascii=False,
        indent=2,
    )
    tags = ["chemistry_intelligence", f"relevance:{mr}"]
    try:
        files_cabinet.create_file(CHEM_INTEL_FOLDER, fname, body, tags=tags)
        return {"filed": True, "cabinet_file": fname, "folder": CHEM_INTEL_FOLDER}
    except ValueError:
        fname2 = f"{CHEM_PREFIX}{day}-{h}-b"
        try:
            files_cabinet.create_file(CHEM_INTEL_FOLDER, fname2, body, tags=tags)
            return {"filed": True, "cabinet_file": fname2, "folder": CHEM_INTEL_FOLDER}
        except Exception as e:
            return {"filed": False, "error": str(e)}
    except Exception as e:
        return {"filed": False, "error": str(e)}


def detect_chemistry_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg or not _CHEM_KEYWORDS.search(msg):
        return None, {}
    if _SYNTHESIS_TRIGGERS.search(msg):
        return "synthesis", {"target": msg[:500], "context": msg[:2000]}
    if _DESIGN_TRIGGERS.search(msg) or re.search(r"\{[^}]*tensile", msg, re.I):
        return "design", {"raw": msg[:3000]}
    if re.search(
        r"(?i)\b(alloy|ceramic|polymer|composite|band\s*gap|crystal structure|"
        r"material(s)?\s+for|properties of\s+\w+)\b",
        msg,
    ):
        return "material", {"description": msg[:800], "use_case": msg[:2000]}
    return "compound", {"query": msg[:500]}


def format_chemistry_chat_block(
    user_message: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> str:
    kind, payload = detect_chemistry_chat_intent(user_message)
    if not kind:
        return ""
    try:
        if kind == "synthesis":
            r = analyze_synthesis_route(
                payload.get("target") or user_message[:400],
                payload.get("context") or user_message,
                anthropic_client=anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
                files_cabinet=files_cabinet,
            )
        elif kind == "design":
            req: dict[str, Any] = {}
            m = re.search(r"\{[\s\S]{0,8000}\}", user_message)
            if m:
                try:
                    req = json.loads(m.group(0))
                except json.JSONDecodeError:
                    req = {"raw_text": user_message[:2000]}
            else:
                req = {"raw_text": user_message[:2000]}
            r = design_material_for_requirements(req, anthropic_client=anthropic_client)
        elif kind == "material":
            r = analyze_material_properties(
                payload.get("description") or user_message[:500],
                payload.get("use_case") or "",
                anthropic_client=anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
                files_cabinet=files_cabinet,
            )
        else:
            q = payload.get("query") or user_message[:400]
            r = fetch_compound_bundle(
                q,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
            )
        return "[Angel chemistry & materials intelligence — structured data]\n" + json.dumps(
            r, ensure_ascii=False, indent=2
        )[:28000]
    except Exception as e:
        return f"[Angel chemistry intelligence — error]\n{str(e)[:800]}"


def compound_api_payload(
    query: str,
    context: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    ctx = (context or "").strip()
    bundle = fetch_compound_bundle(
        (query or "").strip(),
        memory_client=memory_client,
        user_id=user_id,
        use_mem0_cloud=use_mem0_cloud,
    )
    return {"ok": True, "query": query, "context": ctx, "data": bundle}


def material_api_payload(
    query: str,
    use_case: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    return analyze_material_properties(
        query,
        use_case,
        anthropic_client=anthropic_client,
        memory_client=memory_client,
        user_id=user_id,
        use_mem0_cloud=use_mem0_cloud,
        files_cabinet=files_cabinet,
    )
