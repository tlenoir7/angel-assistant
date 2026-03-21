"""
Batcomputer Layer — Environmental map (mission-relevant locations).
Mem0 category env_location + files LOC-* under Environmental Map.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic

ENV_MAP_FOLDER = "Environmental Map"
LOC_PREFIX = "LOC-"

LOCATION_TYPES = frozenset(
    {
        "uap_hotspot",
        "military_installation",
        "restricted_airspace",
        "incident_site",
        "research_facility",
        "government_facility",
        "anomalous_zone",
        "person_associated",
    }
)
SIGNIFICANCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def location_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip())[:72].strip("-").lower()
    return s or f"loc-{hashlib.sha256((name or '').encode()).hexdigest()[:10]}"


def _normalize_location(raw: dict[str, Any]) -> dict[str, Any]:
    lid = (raw.get("location_id") or "").strip() or location_slug(str(raw.get("name") or ""))
    lt = str(raw.get("location_type") or "anomalous_zone").strip().lower()
    if lt not in LOCATION_TYPES:
        lt = "anomalous_zone"
    sig = str(raw.get("significance") or "MEDIUM").strip().upper()
    if sig not in SIGNIFICANCE_LEVELS:
        sig = "MEDIUM"
    coords = raw.get("coordinates")
    if not isinstance(coords, dict):
        coords = {}
    lat = coords.get("lat")
    lon = coords.get("lon")
    try:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat_f, lon_f = None, None
    coord_out: dict[str, Any] = {}
    if lat_f is not None and lon_f is not None:
        coord_out = {"lat": lat_f, "lon": lon_f}
    ce = raw.get("connected_entities")
    if not isinstance(ce, list):
        ce = []
    ce = [str(x).strip() for x in ce if str(x).strip()][:40]
    ki = raw.get("known_incidents")
    if not isinstance(ki, list):
        ki = []
    ki = [str(x).strip() for x in ki if str(x).strip()][:40]
    tags = raw.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:30]
    return {
        "location_id": lid,
        "name": str(raw.get("name") or lid).strip()[:500],
        "location_type": lt,
        "coordinates": coord_out,
        "region": str(raw.get("region") or "").strip()[:300],
        "description": str(raw.get("description") or "").strip()[:8000],
        "significance": sig,
        "connected_entities": ce,
        "known_incidents": ki,
        "first_recorded": str(raw.get("first_recorded") or _today())[:32],
        "last_updated": str(raw.get("last_updated") or _today())[:32],
        "active_monitoring": bool(raw.get("active_monitoring", True)),
        "tags": tags,
    }


def _location_upsert_memory(
    memory_client: Any,
    user_id: str,
    location_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_ENV_LOCATION
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-environmental-map",
        "env_location_id": location_id,
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
                and e["metadata"].get("env_location_id") == location_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel env location {location_id}] {text[:1200]}"},
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


def _sync_loc_file(files_cabinet: Any, loc: dict[str, Any]) -> None:
    lid = (loc.get("location_id") or "").strip()
    if not lid:
        return
    fn = f"{LOC_PREFIX}{lid}"
    body = json.dumps(loc, ensure_ascii=False, indent=2)
    tags = [
        "environmental_map",
        f"type:{loc.get('location_type', '')}",
        f"significance:{loc.get('significance', 'MEDIUM')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(ENV_MAP_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass


def _load_all_locations(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict[str, Any]]:
    import angel as ang

    by_id: dict[str, dict[str, Any]] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_ENV_LOCATION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("location_id"):
            by_id[str(obj["location_id"])] = _normalize_location(obj)
    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_ENV_LOCATION:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("location_id"):
            by_id[str(obj["location_id"])] = _normalize_location(obj)
    return by_id


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    r = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return r * c


def add_location(
    name: str,
    location_type: str,
    coordinates: dict[str, float] | None,
    region: str,
    description: str,
    significance: str,
    connected_entities: list[str] | None,
    known_incidents: list[str] | None,
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    location_id: str | None = None,
    tags: list[str] | None = None,
    active_monitoring: bool = True,
) -> dict[str, Any]:
    raw = {
        "location_id": location_id or location_slug(name),
        "name": name,
        "location_type": location_type,
        "coordinates": coordinates or {},
        "region": region,
        "description": description,
        "significance": significance,
        "connected_entities": connected_entities or [],
        "known_incidents": known_incidents or [],
        "tags": tags or [],
        "active_monitoring": active_monitoring,
        "first_recorded": _today(),
        "last_updated": _today(),
    }
    loc = _normalize_location(raw)
    text = json.dumps(loc, ensure_ascii=False)
    _location_upsert_memory(memory_client, user_id, loc["location_id"], text, use_mem0_cloud)
    _sync_loc_file(files_cabinet, loc)
    return loc


def get_location(
    location_id: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    lid = (location_id or "").strip()
    if not lid:
        return None
    return _load_all_locations(memory_client, user_id, use_mem0_cloud).get(lid)


def list_locations(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    location_type: str | None = None,
    significance: str | None = None,
) -> list[dict[str, Any]]:
    rows = list(_load_all_locations(memory_client, user_id, use_mem0_cloud).values())
    if location_type:
        lt = location_type.strip().lower()
        rows = [r for r in rows if (r.get("location_type") or "") == lt]
    if significance:
        sg = significance.strip().upper()
        rows = [r for r in rows if (r.get("significance") or "").upper() == sg]
    try:
        rows.sort(key=lambda x: ((x.get("significance") or "MEDIUM"), (x.get("name") or "").lower()))
    except Exception:
        pass
    return rows


def get_locations_by_region(
    region: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    q = (region or "").strip().lower()
    if not q:
        return []
    return [
        loc
        for loc in _load_all_locations(memory_client, user_id, use_mem0_cloud).values()
        if q in (loc.get("region") or "").lower() or q in (loc.get("name") or "").lower()
    ]


def get_locations_near_coordinates(
    lat: float,
    lon: float,
    radius_miles: float,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    out: list[tuple[float, dict[str, Any]]] = []
    for loc in _load_all_locations(memory_client, user_id, use_mem0_cloud).values():
        c = loc.get("coordinates") or {}
        try:
            la = float(c.get("lat"))
            lo = float(c.get("lon"))
        except (TypeError, ValueError):
            continue
        d = haversine_miles(lat, lon, la, lo)
        if d <= radius_miles:
            out.append((d, loc))
    try:
        out.sort(key=lambda x: x[0])
    except Exception:
        pass
    return [x[1] for x in out]


def search_locations(
    query: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    found: list[dict[str, Any]] = []
    for loc in _load_all_locations(memory_client, user_id, use_mem0_cloud).values():
        blob = " ".join(
            [
                str(loc.get("name", "")),
                str(loc.get("region", "")),
                str(loc.get("description", "")),
                " ".join(loc.get("tags") or []),
                " ".join(loc.get("known_incidents") or []),
            ]
        ).lower()
        if q in blob:
            found.append(loc)
    return found


def get_location_summary(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    rows = list(_load_all_locations(memory_client, user_id, use_mem0_cloud).values())
    by_t: dict[str, int] = {}
    by_s: dict[str, int] = {}
    for r in rows:
        by_t[r.get("location_type") or "unknown"] = by_t.get(r.get("location_type") or "unknown", 0) + 1
        by_s[r.get("significance") or "MEDIUM"] = by_s.get(r.get("significance") or "MEDIUM", 0) + 1
    return {
        "total": len(rows),
        "by_type": by_t,
        "by_significance": by_s,
        "locations": [{"location_id": x["location_id"], "name": x["name"], "type": x.get("location_type")} for x in rows],
    }


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


def research_location(
    name: str,
    context: str,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Tavily + Claude → EnvironmentalLocation, saved to map."""
    import angel as ang

    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "name required"}
    api_key = (__import__("os").getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    q1 = f"{name} UAP military government site geography"
    q2 = f"{name} location coordinates significance"
    lines: list[str] = []
    seen: set[str] = set()
    for q in (q1, q2):
        for r in ang._tavily_search_one(q, api_key, max_results=5, search_depth="basic"):
            if not isinstance(r, dict):
                continue
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            lines.append(f"{(r.get('title') or '')}\n{(r.get('content') or '')[:900]}\n{url}")
        if len(lines) >= 8:
            break
    bundle = "\n\n".join(lines)[:50_000]
    ctx = (context or "").strip() or "Tyler's UAP/disclosure mission — map this place if mission-relevant."

    sys = """Output ONE JSON object only (no markdown) for an environmental map entry:
{
  "name": "display name",
  "location_type": "uap_hotspot|military_installation|restricted_airspace|incident_site|research_facility|government_facility|anomalous_zone|person_associated",
  "coordinates": {"lat": number or omit if unknown, "lon": number or omit},
  "region": "state/country",
  "description": "why it matters to UAP/disclosure/mission",
  "significance": "LOW|MEDIUM|HIGH|CRITICAL",
  "connected_entities": ["people/orgs from open sources"],
  "known_incidents": ["short incident refs"],
  "tags": ["short"],
  "active_monitoring": true
}
Use only open sources; mark uncertainty in description."""

    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=sys,
            messages=[
                {
                    "role": "user",
                    "content": f"PLACE: {name}\nCONTEXT: {ctx[:4000]}\n\nSOURCES:\n{bundle or 'no results'}",
                }
            ],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                txt += block.get("text", "")
        prof = _parse_json_obj(txt)
        if not prof:
            return {"ok": False, "error": "no JSON from model", "raw": txt[:1500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    prof["location_id"] = location_slug(prof.get("name") or name)
    loc = _normalize_location(prof)
    text = json.dumps(loc, ensure_ascii=False)
    _location_upsert_memory(memory_client, user_id, loc["location_id"], text, use_mem0_cloud)
    _sync_loc_file(files_cabinet, loc)
    return {"ok": True, "location": loc}


def match_text_to_locations(
    text: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """Return mapped locations that plausibly relate to headline/summary/region text."""
    blob = (text or "").lower()
    if len(blob) < 8:
        return []
    hits: list[tuple[int, dict[str, Any]]] = []
    for loc in _load_all_locations(memory_client, user_id, use_mem0_cloud).values():
        name = (loc.get("name") or "").lower()
        region = (loc.get("region") or "").lower()
        score = 0
        if name and name in blob:
            score += 10
        for part in re.split(r"[\s,]+", name):
            if len(part) >= 5 and part in blob:
                score += 4
        if region and region in blob:
            score += 5
        for tok in re.findall(r"[a-z]{5,}", region):
            if tok in blob:
                score += 1
        if score > 0:
            hits.append((score, loc))
    try:
        hits.sort(key=lambda x: -x[0])
    except Exception:
        pass
    out = []
    for _s, loc in hits[:top_n]:
        out.append(
            {
                "location_id": loc.get("location_id"),
                "name": loc.get("name"),
                "significance": loc.get("significance"),
                "region": loc.get("region"),
            }
        )
    return out


def format_proximity_alert_for_prompt(
    location: dict[str, Any] | None,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    radius_miles: float = 75.0,
    min_significance: str = "HIGH",
) -> str:
    """When Tyler's device GPS is available — note nearby CRITICAL/HIGH map points."""
    if not location or not isinstance(location, dict):
        return ""
    try:
        lat = float(location.get("latitude"))
        lon = float(location.get("longitude"))
    except (TypeError, ValueError):
        return ""
    rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_r = rank.get(min_significance.upper(), 3)
    near = get_locations_near_coordinates(lat, lon, radius_miles, memory_client, user_id, use_mem0_cloud)
    sig_near = [
        loc
        for loc in near
        if rank.get((loc.get("significance") or "MEDIUM").upper(), 0) >= min_r
    ]
    if not sig_near:
        return ""
    place = (location.get("place_name") or "").strip()
    lines = [
        f"[Environmental map — proximity] Tyler's reported position ({place or 'coordinates'}): "
        f"near {len(sig_near)} significant mapped location(s) within ~{radius_miles:.0f} mi:"
    ]
    for loc in sig_near[:6]:
        c = loc.get("coordinates") or {}
        lines.append(
            f"- {loc.get('name')} ({loc.get('significance')}) — {loc.get('location_type')} — {loc.get('region', '')[:80]}"
        )
    lines.append(
        "Mention naturally if relevant; do not dump coordinates unless Tyler asked. Open-source map context only."
    )
    return "\n".join(lines)


def format_surveillance_map_lines(
    signals: list[dict[str, Any]],
) -> str:
    """Extra briefing lines when surveillance signals align with mapped locations."""
    lines: list[str] = []
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        nm = s.get("near_mapped_locations") or []
        if not nm:
            continue
        if (s.get("signal_strength") or "") not in ("STRONG", "MODERATE"):
            continue
        hl = (s.get("headline") or "")[:120]
        names = ", ".join(x.get("name") or "" for x in nm if isinstance(x, dict))[:200]
        if names:
            lines.append(f"Surveillance signal near mapped location(s) [{names}]: {hl}")
    if not lines:
        return ""
    return "ENVIRONMENTAL MAP (surveillance cross-reference)\n" + "\n".join(lines[:8])


def ensure_seed_locations(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Idempotent seeds for core mission map."""
    existing = _load_all_locations(memory_client, user_id, use_mem0_cloud)
    seeds: list[dict[str, Any]] = [
        {
            "location_id": "skinwalker-ranch",
            "name": "Skinwalker Ranch, Utah",
            "location_type": "uap_hotspot",
            "coordinates": {"lat": 40.245, "lon": -109.887},
            "region": "Utah, USA",
            "description": "Privately owned ranch with long-running reports of anomalous phenomena; open-source reporting on government-affiliated research interest (BAASS/DIA context in public discourse).",
            "significance": "CRITICAL",
            "connected_entities": ["BAASS", "DIA", "NIDS (historical)"],
            "known_incidents": ["Ongoing phenomena reports", "Media investigations"],
            "tags": ["ranch", "Utah", "phenomena"],
        },
        {
            "location_id": "area-51",
            "name": "Area 51, Nevada",
            "location_type": "restricted_airspace",
            "coordinates": {"lat": 37.235, "lon": -115.811},
            "region": "Nevada, USA",
            "description": "Highly restricted test range; persistent public speculation on classified aerospace programs and UAP-related theories (open sources only).",
            "significance": "CRITICAL",
            "connected_entities": ["USAF", "DoD"],
            "known_incidents": ["Public UAP test-flight theories"],
            "tags": ["Nevada", "classified", "airspace"],
        },
        {
            "location_id": "hessdalen-valley",
            "name": "Hessdalen Valley, Norway",
            "location_type": "uap_hotspot",
            "coordinates": {"lat": 62.817, "lon": 11.233},
            "region": "Norway",
            "description": "Long-documented light phenomena; scientific monitoring efforts in open literature.",
            "significance": "HIGH",
            "connected_entities": [],
            "known_incidents": ["Hessdalen lights (documented)"],
            "tags": ["Europe", "lights"],
        },
        {
            "location_id": "rendlesham-forest",
            "name": "Rendlesham Forest, UK",
            "location_type": "incident_site",
            "coordinates": {"lat": 52.087, "lon": 1.407},
            "region": "United Kingdom",
            "description": "1980 RAF Bentwaters / Woodbridge incident — multiple military witnesses in public record.",
            "significance": "HIGH",
            "connected_entities": ["RAF Bentwaters", "USAF (historical)"],
            "known_incidents": ["Rendlesham Forest incident (1980)"],
            "tags": ["UK", "military"],
        },
        {
            "location_id": "gulf-mexico-uap-corridor",
            "name": "Gulf of Mexico UAP corridor",
            "location_type": "anomalous_zone",
            "coordinates": {"lat": 25.5, "lon": -90.0},
            "region": "Gulf of Mexico",
            "description": "Open-source reporting on Navy UAP encounters including Nimitz (2004) and Roosevelt strike group incidents (publicly discussed).",
            "significance": "HIGH",
            "connected_entities": ["US Navy"],
            "known_incidents": ["Nimitz 2004", "Roosevelt group encounters (reported)"],
            "tags": ["maritime", "Navy"],
        },
        {
            "location_id": "wright-patterson-afb",
            "name": "Wright-Patterson AFB, Ohio",
            "location_type": "military_installation",
            "coordinates": {"lat": 39.826, "lon": -84.045},
            "region": "Ohio, USA",
            "description": "Historic USAF base; public lore and congressional interest re alleged material storage — verify only from open sources.",
            "significance": "CRITICAL",
            "connected_entities": ["USAF", "Project Blue Book (historical)"],
            "known_incidents": ["Blue Book headquarters (historical)"],
            "tags": ["Ohio", "USAF"],
        },
        {
            "location_id": "dugway-proving-ground",
            "name": "Dugway Proving Ground, Utah",
            "location_type": "military_installation",
            "coordinates": {"lat": 40.178, "lon": -112.937},
            "region": "Utah, USA",
            "description": "Remote chemical/biological defense test range; classified testing in open defense reporting.",
            "significance": "HIGH",
            "connected_entities": ["US Army"],
            "known_incidents": [],
            "tags": ["Utah", "testing"],
        },
        {
            "location_id": "nas-oceana",
            "name": "Naval Air Station Oceana, Virginia",
            "location_type": "military_installation",
            "coordinates": {"lat": 36.821, "lon": -76.033},
            "region": "Virginia, USA",
            "description": "Major Navy master jet base; proximity to East Coast UAP reporting clusters in public discourse.",
            "significance": "HIGH",
            "connected_entities": ["US Navy"],
            "known_incidents": ["East Coast encounter reporting (open sources)"],
            "tags": ["Navy", "Virginia"],
        },
        {
            "location_id": "the-pentagon",
            "name": "The Pentagon, Virginia",
            "location_type": "government_facility",
            "coordinates": {"lat": 38.871, "lon": -77.056},
            "region": "Virginia, USA",
            "description": "DoD headquarters; AARO and UAP program oversight in public record.",
            "significance": "CRITICAL",
            "connected_entities": ["DoD", "AARO"],
            "known_incidents": [],
            "tags": ["DoD", "policy"],
        },
        {
            "location_id": "cia-headquarters-langley",
            "name": "CIA Headquarters, Langley, Virginia",
            "location_type": "government_facility",
            "coordinates": {"lat": 38.952, "lon": -77.146},
            "region": "Virginia, USA",
            "description": "IC classification and policy; UAP-related oversight themes in open debate.",
            "significance": "HIGH",
            "connected_entities": ["CIA"],
            "known_incidents": [],
            "tags": ["IC", "classification"],
        },
        {
            "location_id": "nsa-fort-meade",
            "name": "NSA Fort Meade, Maryland",
            "location_type": "government_facility",
            "coordinates": {"lat": 39.108, "lon": -76.771},
            "region": "Maryland, USA",
            "description": "Signals intelligence hub; relevant to surveillance and classification themes in open mission context.",
            "significance": "HIGH",
            "connected_entities": ["NSA"],
            "known_incidents": [],
            "tags": ["SIGINT", "Maryland"],
        },
        {
            "location_id": "lexington-south-carolina",
            "name": "Lexington, South Carolina",
            "location_type": "person_associated",
            "coordinates": {"lat": 33.981, "lon": -81.236},
            "region": "South Carolina, USA",
            "description": "Tyler's home base (as configured); proximity to regional FBI field resources and mission travel hub.",
            "significance": "MEDIUM",
            "connected_entities": ["Tyler"],
            "known_incidents": [],
            "tags": ["home", "FBI context"],
            "active_monitoring": True,
        },
    ]
    added = 0
    for s in seeds:
        lid = s["location_id"]
        if lid in existing:
            continue
        add_location(
            s["name"],
            s["location_type"],
            s.get("coordinates"),
            s.get("region") or "",
            s.get("description") or "",
            s.get("significance") or "MEDIUM",
            s.get("connected_entities"),
            s.get("known_incidents"),
            memory_client=memory_client,
            user_id=user_id,
            files_cabinet=files_cabinet,
            use_mem0_cloud=use_mem0_cloud,
            location_id=lid,
            tags=s.get("tags"),
            active_monitoring=s.get("active_monitoring", True),
        )
        added += 1
    return {"ok": True, "added": added, "total": len(_load_all_locations(memory_client, user_id, use_mem0_cloud))}


def maybe_ingest_locations_from_osint_text(
    dossier_text: str,
    target_label: str,
    anthropic_client: anthropic.Anthropic,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    """Extract place names from OSINT dossier; add only if not already on map (conservative)."""
    t = (dossier_text or "").strip()
    if len(t) < 400:
        return {"ok": False, "skipped": True}
    existing = _load_all_locations(memory_client, user_id, use_mem0_cloud)
    sys = """Reply JSON only: {"places": [{"name": "...", "region": "...", "why": "short"}]} max 4 places tied to UAP/government/military geography from the text."""
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            temperature=0.1,
            system=sys,
            messages=[
                {
                    "role": "user",
                    "content": f"TARGET SUBJECT: {target_label}\n\nTEXT:\n{t[:12000]}",
                }
            ],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        o = _parse_json_obj(txt)
        places = (o or {}).get("places") if isinstance(o, dict) else None
        if not isinstance(places, list):
            return {"ok": True, "added": 0}
        n = 0
        for p in places[:4]:
            if not isinstance(p, dict):
                continue
            nm = (p.get("name") or "").strip()
            if len(nm) < 4:
                continue
            lid = location_slug(nm)
            if lid in existing or any(nm.lower() == (v.get("name") or "").lower() for v in existing.values()):
                continue
            add_location(
                nm,
                "person_associated" if target_label.lower() in nm.lower() else "government_facility",
                {},
                (p.get("region") or "")[:200],
                f"Mentioned in OSINT dossier on {target_label}: {(p.get('why') or '')[:500]}",
                "LOW",
                [target_label[:120]],
                [],
                memory_client=memory_client,
                user_id=user_id,
                files_cabinet=files_cabinet,
                use_mem0_cloud=use_mem0_cloud,
                tags=["osint_ingest"],
            )
            n += 1
        return {"ok": True, "added": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def detect_map_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(r"(?i)\bshow\s+me\s+the\s+map\b", msg):
        return "map_summary", {}
    m = re.search(r"(?i)\bwhat(?:'s| is)\s+significant\s+about\s+(.+?)(?:\?|$)", msg)
    if m:
        return "location_profile", {"name": m.group(1).strip().rstrip("?.!")}
    m = re.search(r"(?i)\bwhat(?:'s| is)\s+near\s+(.+?)(?:\?|$)", msg)
    if m:
        return "near_place", {"place": m.group(1).strip().rstrip("?.!")}
    m = re.search(
        r"(?i)\b(?:research|map)\s+(?:location\s+)?(.+?)(?:\s+for\s+Tyler)?(?:\?|$)",
        msg,
    )
    if m and len(m.group(1).strip()) > 3:
        return "research", {"name": m.group(1).strip().rstrip("?.!")}
    return None, {}


def format_map_chat_block(
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
        if intent == "map_summary":
            s = get_location_summary(memory_client, user_id, use_mem0_cloud)
            return "[Angel environmental map]\n" + json.dumps(s, ensure_ascii=False, indent=2)[:12000]
        if intent == "location_profile":
            name = (payload.get("name") or "").strip()
            hits = search_locations(name, memory_client, user_id, use_mem0_cloud)
            if not hits:
                return (
                    "[Angel environmental map]\n"
                    + json.dumps({"notice": f"No exact match for {name!r}. Try POST /api/map/research or refine the name."}, indent=2)
                )
            return "[Angel environmental map]\n" + json.dumps(hits[0], ensure_ascii=False, indent=2)[:12000]
        if intent == "near_place":
            place = (payload.get("place") or "").strip()
            hits = search_locations(place, memory_client, user_id, use_mem0_cloud)
            if not hits:
                return "[Angel environmental map]\n" + json.dumps({"near": place, "matches": []}, indent=2)
            loc = hits[0]
            c = loc.get("coordinates") or {}
            lat, lon = c.get("lat"), c.get("lon")
            if lat is None or lon is None:
                return "[Angel environmental map]\n" + json.dumps({"anchor": loc, "nearby": "no coordinates for distance"}, indent=2)
            near = get_locations_near_coordinates(
                float(lat), float(lon), 500.0, memory_client, user_id, use_mem0_cloud
            )
            return "[Angel environmental map — nearby]\n" + json.dumps(
                {"anchor": loc.get("name"), "within_500mi": [x.get("name") for x in near[:15]]},
                indent=2,
            )[:12000]
        if intent == "research":
            res = research_location(
                (payload.get("name") or "").strip(),
                "Tyler asked in chat to research and map this location.",
                anthropic_client,
                memory_client,
                user_id,
                files_cabinet,
                use_mem0_cloud,
            )
            return "[Angel environmental map — research]\n" + json.dumps(res, indent=2)[:12000]
    except Exception as e:
        return f"[Angel environmental map error]\n{str(e)[:600]}"
    return ""


def format_briefing_travel_note_if_env(
    lat: float | None,
    lon: float | None,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> str:
    """Optional BRIEFING_LOCATION_LAT/LON in server env for travel context."""
    if lat is None or lon is None:
        return ""
    loc = {"latitude": lat, "longitude": lon, "place_name": "briefing travel context"}
    near = get_locations_near_coordinates(lat, lon, 200.0, memory_client, user_id, use_mem0_cloud)
    sig = [x for x in near if (x.get("significance") or "") in ("CRITICAL", "HIGH")]
    if not sig:
        return ""
    return (
        "TRAVEL / AREA NOTE (environmental map)\n"
        + "Significant mapped locations within ~200 mi of briefing coordinates:\n"
        + "\n".join(f"- {x.get('name')} ({x.get('significance')})" for x in sig[:6])
    )
