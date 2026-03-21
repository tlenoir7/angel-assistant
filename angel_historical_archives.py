"""
Batcomputer Layer — Historical Intelligence Archives (UAP timeline, programs, documents).
Mem0 category historical_record + files HIST-* under Historical Archives.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic

ARCHIVES_FOLDER = "Historical Archives"
HIST_PREFIX = "HIST-"

RECORD_TYPES = frozenset(
    {
        "incident",
        "program",
        "document",
        "testimony",
        "disclosure",
        "cover_up",
        "turning_point",
    }
)
DATE_PRECISION_LEVELS = frozenset({"exact", "approximate", "decade"})
EVIDENCE_LEVELS = frozenset({"anecdotal", "witness", "documented", "official", "declassified"})
SIGNIFICANCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> tuple[int, int, int]:
    now = datetime.now(timezone.utc)
    return now.year, now.month, now.day


def record_slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (title or "").strip())[:72].strip("-").lower()
    return s or f"hist-{hashlib.sha256((title or '').encode()).hexdigest()[:10]}"


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


def _extract_year(date_str: str) -> int | None:
    s = (date_str or "").strip()
    m = re.match(r"^(\d{4})", s)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _parse_month_day(date_str: str) -> tuple[int, int] | None:
    """Return (month, day) if YYYY-MM-DD or ...-MM-DD present."""
    s = (date_str or "").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return int(m.group(2)), int(m.group(3))
        except ValueError:
            return None
    m2 = re.search(r"-(\d{2})-(\d{2})\s*$", s)
    if m2:
        try:
            return int(m2.group(1)), int(m2.group(2))
        except ValueError:
            return None
    return None


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    rid = (raw.get("record_id") or "").strip() or record_slug(str(raw.get("title") or ""))
    rt = str(raw.get("record_type") or "incident").strip().lower()
    if rt not in RECORD_TYPES:
        rt = "incident"
    dp = str(raw.get("date_precision") or "approximate").strip().lower()
    if dp not in DATE_PRECISION_LEVELS:
        dp = "approximate"
    ev = str(raw.get("evidence_quality") or "documented").strip().lower()
    if ev not in EVIDENCE_LEVELS:
        ev = "documented"
    sig = str(raw.get("significance") or "MEDIUM").strip().upper()
    if sig not in SIGNIFICANCE_LEVELS:
        sig = "MEDIUM"
    for key in ("connected_people", "connected_programs", "connected_locations", "sources", "tags"):
        v = raw.get(key)
        if not isinstance(v, list):
            v = []
        raw[key] = [str(x).strip() for x in v if str(x).strip()][:60]
    return {
        "record_id": rid[:200],
        "title": str(raw.get("title") or rid).strip()[:500],
        "record_type": rt,
        "date": str(raw.get("date") or "").strip()[:32],
        "date_precision": dp,
        "location": str(raw.get("location") or "").strip()[:500],
        "description": str(raw.get("description") or "").strip()[:16000],
        "significance": sig,
        "connected_people": raw["connected_people"],
        "connected_programs": raw["connected_programs"],
        "connected_locations": raw["connected_locations"],
        "evidence_quality": ev,
        "current_relevance": str(raw.get("current_relevance") or "").strip()[:4000],
        "sources": raw["sources"],
        "tags": raw["tags"][:40],
        "last_updated": str(raw.get("last_updated") or "")[:32],
    }


def _historical_upsert_memory(
    memory_client: Any,
    user_id: str,
    record_id: str,
    text: str,
    use_mem0_cloud: bool,
) -> None:
    import angel as ang

    cat = ang.CATEGORY_HISTORICAL_RECORD
    ts = _now_iso()
    meta = {
        "category": cat,
        "timestamp": ts,
        "source": "angel-historical-archives",
        "historical_record_id": record_id,
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
                and e["metadata"].get("historical_record_id") == record_id
            )
        ]
        filtered.append({"memory": text, "metadata": dict(meta), "created_at": ts})
        ang._save_local_memory_entries(user_id, filtered)
    except Exception:
        pass
    if use_mem0_cloud and hasattr(memory_client, "add"):
        try:
            messages = [
                {"role": "user", "content": f"[Angel historical record {record_id}] {text[:1200]}"},
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


def _sync_hist_file(files_cabinet: Any, rec: dict[str, Any]) -> None:
    rid = (rec.get("record_id") or "").strip()
    if not rid:
        return
    fn = f"{HIST_PREFIX}{rid}"
    body = json.dumps(rec, ensure_ascii=False, indent=2)
    tags = [
        "historical_archives",
        f"type:{rec.get('record_type', '')}",
        f"significance:{rec.get('significance', 'MEDIUM')}",
    ]
    try:
        if files_cabinet.get_file(fn):
            files_cabinet.update_file(fn, body)
        else:
            files_cabinet.create_file(ARCHIVES_FOLDER, fn, body, tags=tags)
    except ValueError:
        try:
            files_cabinet.update_file(fn, body)
        except Exception:
            pass
    except Exception:
        pass


def _load_all_records(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, dict[str, Any]]:
    import angel as ang

    by_id: dict[str, dict[str, Any]] = {}
    memories = ang.fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
    for m in ang._normalize_memories_list(memories):
        meta = m.get("metadata") if isinstance(m, dict) else {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_HISTORICAL_RECORD:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("record_id"):
            by_id[str(obj["record_id"])] = _normalize_record(obj)

    for m in ang._load_local_memory_entries(user_id):
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("category") != ang.CATEGORY_HISTORICAL_RECORD:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("record_id"):
            by_id[str(obj["record_id"])] = _normalize_record(obj)
    return by_id


def _sort_key_date(rec: dict[str, Any]) -> tuple[int, int, int, str]:
    y = _extract_year(rec.get("date") or "") or 0
    md = _parse_month_day(rec.get("date") or "")
    if md:
        return (y, md[0], md[1], rec.get("title") or "")
    return (y, 0, 0, rec.get("title") or "")


SEED_HISTORICAL_RECORDS: list[dict[str, Any]] = [
    # INCIDENTS
    {
        "record_id": "roswell-1947",
        "title": "Roswell incident",
        "record_type": "incident",
        "date": "1947-07-08",
        "date_precision": "approximate",
        "location": "Roswell, New Mexico, USA",
        "description": "Alleged crash retrieval and debris recovery; USAF later attributed debris to Project Mogul balloons; "
        "persistent public debate over material evidence and cover-up narratives in open literature.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["Project Mogul (cover story)", "USAF public statements"],
        "connected_locations": ["Roswell Army Air Field"],
        "evidence_quality": "witness",
        "current_relevance": "Foundational crash-retrieval narrative in disclosure discourse; referenced by media and witnesses.",
        "sources": [],
        "tags": ["crash", "1947", "new_mexico", "cover_story"],
    },
    {
        "record_id": "kenneth-arnold-1947",
        "title": "Kenneth Arnold sighting",
        "record_type": "incident",
        "date": "1947-06-24",
        "date_precision": "exact",
        "location": "Mount Rainier area, Washington, USA",
        "description": "Pilot reported nine crescent-shaped objects; press popularized 'flying saucer' framing.",
        "significance": "HIGH",
        "connected_people": ["Kenneth Arnold"],
        "connected_programs": [],
        "connected_locations": [],
        "evidence_quality": "witness",
        "current_relevance": "Often cited as start of modern UFO era in public consciousness.",
        "sources": [],
        "tags": ["first_wave", "flying_saucer", "civilian_witness"],
    },
    {
        "record_id": "washington-dc-flyover-1952",
        "title": "Washington, D.C. UFO flyovers",
        "record_type": "incident",
        "date": "1952-07",
        "date_precision": "approximate",
        "location": "Washington, D.C., USA",
        "description": "Multiple radar/visual reports over restricted airspace; USAF intercepts; major press coverage.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["USAF interception operations"],
        "connected_locations": ["Washington National Airport"],
        "evidence_quality": "official",
        "current_relevance": "Template for radar-confirmed cases over sensitive airspace.",
        "sources": [],
        "tags": ["radar", "restricted_airspace", "1952"],
    },
    {
        "record_id": "levelland-texas-1957",
        "title": "Levelland, Texas encounters",
        "record_type": "incident",
        "date": "1957-11",
        "date_precision": "approximate",
        "location": "Levelland, Texas, USA",
        "description": "Multiple independent witnesses reported EM effects on vehicles and glowing object; Blue Book investigation.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["Project Blue Book"],
        "connected_locations": ["Levelland"],
        "evidence_quality": "witness",
        "current_relevance": "Early EM-interference cluster case in Blue Book files.",
        "sources": [],
        "tags": ["em_effects", "vehicle_stall", "blue_book"],
    },
    {
        "record_id": "betty-barney-hill-1961",
        "title": "Betty and Barney Hill abduction account",
        "record_type": "incident",
        "date": "1961-09-19",
        "date_precision": "approximate",
        "location": "New Hampshire, USA",
        "description": "First widely publicized abduction narrative with medical-examination elements under hypnosis; debated in literature.",
        "significance": "HIGH",
        "connected_people": ["Betty Hill", "Barney Hill"],
        "connected_programs": [],
        "connected_locations": [],
        "evidence_quality": "witness",
        "current_relevance": "Influential template for abduction discourse.",
        "sources": [],
        "tags": ["abduction", "hypnosis", "medical_exam"],
    },
    {
        "record_id": "kecksburg-1965",
        "title": "Kecksburg object",
        "record_type": "incident",
        "date": "1965-12-09",
        "date_precision": "approximate",
        "location": "Kecksburg, Pennsylvania, USA",
        "description": "Acorn-shaped object reported; military presence; multiple witness accounts; disputed identification.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": [],
        "connected_locations": ["Kecksburg"],
        "evidence_quality": "witness",
        "current_relevance": "Recovery/retrieval narrative often compared to other crash folklore.",
        "sources": [],
        "tags": ["recovery", "military", "1965"],
    },
    {
        "record_id": "rendlesham-forest-1980",
        "title": "Rendlesham Forest incident",
        "record_type": "incident",
        "date": "1980-12-26",
        "date_precision": "approximate",
        "location": "Rendlesham Forest, Suffolk, UK",
        "description": "RAF Bentwaters/Woodbridge; military witnesses; Halt memo; radiation readings debated in open sources.",
        "significance": "HIGH",
        "connected_people": ["Charles Halt"],
        "connected_programs": ["RAF Bentwaters / USAF"],
        "connected_locations": ["Rendlesham Forest"],
        "evidence_quality": "declassified",
        "current_relevance": "Key UK military-associated case cited in disclosure debates.",
        "sources": [],
        "tags": ["military", "memo", "radiation_reports"],
    },
    {
        "record_id": "cash-landrum-1980",
        "title": "Cash-Landrum incident",
        "record_type": "incident",
        "date": "1980-12-29",
        "date_precision": "approximate",
        "location": "Texas, USA",
        "description": "Witness health effects alleged after diamond-shaped craft encounter; litigation and medical documentation in public record.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": [],
        "connected_locations": [],
        "evidence_quality": "documented",
        "current_relevance": "Medical injury narrative tied to UAP encounter in open literature.",
        "sources": [],
        "tags": ["health_effects", "lawsuit", "radiation"],
    },
    {
        "record_id": "belgium-ufo-wave-1989-1990",
        "title": "Belgium UFO wave",
        "record_type": "incident",
        "date": "1989-11",
        "date_precision": "approximate",
        "location": "Belgium",
        "description": "Wave of reports; F-16 radar data discussed; Belgian authorities engaged publicly at times.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["Belgian Air Force (public engagement)"],
        "connected_locations": ["Belgium"],
        "evidence_quality": "official",
        "current_relevance": "Often cited for radar + military chase narratives in Europe.",
        "sources": [],
        "tags": ["radar", "F-16", "wave"],
    },
    {
        "record_id": "phoenix-lights-1997",
        "title": "Phoenix Lights",
        "record_type": "incident",
        "date": "1997-03-13",
        "date_precision": "approximate",
        "location": "Arizona, USA",
        "description": "Mass witness event; governor later discussed public stance; multiple explanatory hypotheses in open sources.",
        "significance": "HIGH",
        "connected_people": ["Fife Symington"],
        "connected_programs": [],
        "connected_locations": ["Phoenix"],
        "evidence_quality": "witness",
        "current_relevance": "Major mass-sighting reference point in U.S. discourse.",
        "sources": [],
        "tags": ["mass_sighting", "1997"],
    },
    {
        "record_id": "chilean-navy-uap-2014",
        "title": "Chilean Navy UAP video release",
        "record_type": "incident",
        "date": "2014-11-11",
        "date_precision": "approximate",
        "location": "Chile",
        "description": "Government-associated release and analysis discussed in open media; official interest framed as unidentified.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["Chilean Navy (context)"],
        "connected_locations": [],
        "evidence_quality": "official",
        "current_relevance": "International official-release precedent.",
        "sources": [],
        "tags": ["infrared", "government_release", "2014"],
    },
    {
        "record_id": "uss-nimitz-2004",
        "title": "USS Nimitz 'Tic Tac' encounter",
        "record_type": "incident",
        "date": "2004-11-14",
        "date_precision": "approximate",
        "location": "Pacific Ocean / off Southern California",
        "description": "Carrier group encounters; radar, FLIR, and witness testimony in public record; foundational to modern DoD UAP discussion.",
        "significance": "CRITICAL",
        "connected_people": ["David Fravor"],
        "connected_programs": ["Navy UAP reporting (context)"],
        "connected_locations": [],
        "evidence_quality": "official",
        "current_relevance": "Central case in congressional and media framing of UAP.",
        "sources": [],
        "tags": ["tic_tac", "radar", "FLIR", "navy"],
    },
    {
        "record_id": "uss-roosevelt-2015",
        "title": "USS Roosevelt encounters / Gimbal & GoFast",
        "record_type": "incident",
        "date": "2015",
        "date_precision": "approximate",
        "location": "East Coast operating areas, Atlantic",
        "description": "Navy pilot reports and released videos (Gimbal, GoFast) tied to Roosevelt strike group era in public discourse.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["UAPTF / AARO lineage (policy context)"],
        "connected_locations": [],
        "evidence_quality": "declassified",
        "current_relevance": "Catalyst for Pentagon task force and wider release debates.",
        "sources": [],
        "tags": ["gimbal", "gofast", "navy", "videos"],
    },
    # PROGRAMS
    {
        "record_id": "project-sign-1947",
        "title": "Project Sign",
        "record_type": "program",
        "date": "1947",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Early USAF UAP investigation; internal Estimate of the Situation reportedly considered non-human hypothesis (disputed).",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["USAF"],
        "connected_locations": ["Wright-Patterson AFB (context)"],
        "evidence_quality": "documented",
        "current_relevance": "Institutional origin of USAF UAP study.",
        "sources": [],
        "tags": ["USAF", "first_program"],
    },
    {
        "record_id": "project-grudge-1949",
        "title": "Project Grudge",
        "record_type": "program",
        "date": "1949",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Successor to Sign; often characterized in literature as debunking-oriented.",
        "significance": "MEDIUM",
        "connected_people": [],
        "connected_programs": ["USAF"],
        "connected_locations": [],
        "evidence_quality": "documented",
        "current_relevance": "Historical contrast to later Blue Book era.",
        "sources": [],
        "tags": ["USAF", "debunking_frame"],
    },
    {
        "record_id": "project-blue-book-1952-1969",
        "title": "Project Blue Book",
        "record_type": "program",
        "date": "1952-1969",
        "date_precision": "approximate",
        "location": "United States",
        "description": "12,618 cases investigated; 701 unexplained in official summary framing; HQ Wright-Patterson.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["USAF"],
        "connected_locations": ["Wright-Patterson AFB"],
        "evidence_quality": "official",
        "current_relevance": "Baseline for 'official unexplained' statistics in public debate.",
        "sources": [],
        "tags": ["blue_book", "701", "USAF"],
    },
    {
        "record_id": "aatip-2007-2012",
        "title": "AATIP (Advanced Aerospace Threat Identification Program)",
        "record_type": "program",
        "date": "2007-2012",
        "date_precision": "approximate",
        "location": "United States",
        "description": "DIA-associated program; Elizondo affiliation in public record; ~$22M budget figure in media reporting.",
        "significance": "CRITICAL",
        "connected_people": ["Luis Elizondo"],
        "connected_programs": ["DIA"],
        "connected_locations": ["Pentagon (oversight context)"],
        "evidence_quality": "official",
        "current_relevance": "Bridge from classified-adjacent study to public disclosure era.",
        "sources": [],
        "tags": ["AATIP", "DIA", "pentagon"],
    },
    {
        "record_id": "baass-skinwalker-2008-2012",
        "title": "BAASS / Skinwalker Ranch studies",
        "record_type": "program",
        "date": "2008-2012",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Bigelow Aerospace Advanced Space Studies under DIA contract; Skinwalker Ranch fieldwork in public reporting.",
        "significance": "HIGH",
        "connected_people": ["Robert Bigelow"],
        "connected_programs": ["DIA", "BAASS"],
        "connected_locations": ["Skinwalker Ranch"],
        "evidence_quality": "documented",
        "current_relevance": "Links defense funding to anomalous phenomena field studies in public narrative.",
        "sources": [],
        "tags": ["BAASS", "Skinwalker", "DIA"],
    },
    {
        "record_id": "aaro-2022",
        "title": "AARO (All-domain Anomaly Resolution Office)",
        "record_type": "program",
        "date": "2022",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Current official DoD UAP analysis office; reports to Congress in public process.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["DoD", "ODNI (context)"],
        "connected_locations": ["Pentagon"],
        "evidence_quality": "official",
        "current_relevance": "Primary official U.S. channel for UAP reporting today.",
        "sources": [],
        "tags": ["AARO", "current", "DoD"],
    },
    # DOCUMENTS
    {
        "record_id": "twining-memo-1947",
        "title": "Twining memo (September 23, 1947)",
        "record_type": "document",
        "date": "1947-09-23",
        "date_precision": "exact",
        "location": "United States",
        "description": "General Twining memo on discs being real and not U.S. secret weapons in open document discussion.",
        "significance": "HIGH",
        "connected_people": ["Nathan Twining"],
        "connected_programs": ["USAF"],
        "connected_locations": [],
        "evidence_quality": "declassified",
        "current_relevance": "Early official-text anchor for 'real unknowns' narrative.",
        "sources": [],
        "tags": ["memo", "1947", "USAF"],
    },
    {
        "record_id": "robertson-panel-1953",
        "title": "Robertson Panel",
        "record_type": "document",
        "date": "1953-01",
        "date_precision": "approximate",
        "location": "United States",
        "description": "CIA-sponsored panel; debunking and public education recommendations in open historical accounts.",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["CIA"],
        "connected_locations": [],
        "evidence_quality": "documented",
        "current_relevance": "Policy framing of public messaging vs UAP interest.",
        "sources": [],
        "tags": ["CIA", "debunking", "policy"],
    },
    {
        "record_id": "bolender-memo-1969",
        "title": "Bolender memo",
        "record_type": "document",
        "date": "1969-10",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Air Force memo framing: cases affecting national security not solved by Blue Book (open literature interpretation).",
        "significance": "HIGH",
        "connected_people": [],
        "connected_programs": ["USAF", "Project Blue Book"],
        "connected_locations": [],
        "evidence_quality": "documented",
        "current_relevance": "Supports 'hidden lane' narrative in FOIA discourse.",
        "sources": [],
        "tags": ["blue_book", "national_security"],
    },
    {
        "record_id": "wilson-davis-memo-2019",
        "title": "Wilson-Davis notes (alleged)",
        "record_type": "document",
        "date": "2019",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Leaked-note controversy: alleged meeting between Admiral Wilson and Eric Davis re SAP programs; authenticity debated.",
        "significance": "CRITICAL",
        "connected_people": ["Thomas R. Wilson", "Eric W. Davis"],
        "connected_programs": [],
        "connected_locations": [],
        "evidence_quality": "anecdotal",
        "current_relevance": "High-impact document in insider-program speculation.",
        "sources": [],
        "tags": ["leak", "SAP", "contested"],
    },
    {
        "record_id": "pentagon-uap-videos-2020",
        "title": "Pentagon UAP videos official release",
        "record_type": "document",
        "date": "2020-04-27",
        "date_precision": "approximate",
        "location": "United States",
        "description": "DoD confirmation of videos as 'unidentified' in public statements; includes Tic Tac / Gimbal / GoFast context.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["DoD"],
        "connected_locations": ["Pentagon"],
        "evidence_quality": "official",
        "current_relevance": "Official acknowledgment of military UAP media.",
        "sources": [],
        "tags": ["videos", "DoD", "release"],
    },
    # TURNING POINTS
    {
        "record_id": "nyt-aatip-2017",
        "title": "New York Times AATIP exposé",
        "record_type": "turning_point",
        "date": "2017-12-16",
        "date_precision": "approximate",
        "location": "United States / global media",
        "description": "Mainstream article on AATIP and Pentagon UAP interest; widely treated as inflection in public discourse.",
        "significance": "CRITICAL",
        "connected_people": ["Luis Elizondo", "Harry Reid"],
        "connected_programs": ["AATIP"],
        "connected_locations": [],
        "evidence_quality": "documented",
        "current_relevance": "Normalized Pentagon UAP topic in prestige press.",
        "sources": [],
        "tags": ["media", "AATIP", "2017"],
    },
    {
        "record_id": "grusch-congressional-2023",
        "title": "David Grusch congressional testimony",
        "record_type": "testimony",
        "date": "2023-07-26",
        "date_precision": "approximate",
        "location": "Washington, D.C., USA",
        "description": "Under-oath claims regarding retrieval programs and non-human biologics in public hearing framing.",
        "significance": "CRITICAL",
        "connected_people": ["David Grusch"],
        "connected_programs": ["Congressional oversight"],
        "connected_locations": [],
        "evidence_quality": "official",
        "current_relevance": "Central whistleblower testimony node in current mission.",
        "sources": [],
        "tags": ["congress", "whistleblower", "2023"],
    },
    {
        "record_id": "trump-uap-disclosure-order-2026",
        "title": "Executive order on UAP records / disclosure process (2026)",
        "record_type": "disclosure",
        "date": "2026",
        "date_precision": "approximate",
        "location": "United States",
        "description": "Public reporting on executive action directing release/review of UAP-related records; alien.gov portal referenced in discourse.",
        "significance": "CRITICAL",
        "connected_people": [],
        "connected_programs": ["White House", "Federal archives (context)"],
        "connected_locations": [],
        "evidence_quality": "official",
        "current_relevance": "Current policy layer for Tyler's mission timing.",
        "sources": [],
        "tags": ["EO", "disclosure", "2026"],
    },
]


def ensure_seed_historical_records(
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    existing = _load_all_records(memory_client, user_id, use_mem0_cloud)
    added = 0
    for row in SEED_HISTORICAL_RECORDS:
        rid = row["record_id"]
        if rid in existing:
            continue
        rec = _normalize_record(row)
        rec["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        text = json.dumps(rec, ensure_ascii=False)
        _historical_upsert_memory(memory_client, user_id, rid, text, use_mem0_cloud)
        if files_cabinet is not None:
            _sync_hist_file(files_cabinet, rec)
        added += 1
    return {"ok": True, "added": added, "total": len(_load_all_records(memory_client, user_id, use_mem0_cloud))}


def add_historical_record(
    title: str,
    record_type: str,
    date: str,
    location: str,
    description: str,
    significance: str,
    connected_people: list[str] | None,
    evidence_quality: str,
    current_relevance: str,
    sources: list[str] | None,
    tags: list[str] | None,
    *,
    memory_client: Any,
    user_id: str,
    files_cabinet: Any,
    use_mem0_cloud: bool,
    record_id: str | None = None,
    date_precision: str = "approximate",
    connected_programs: list[str] | None = None,
    connected_locations: list[str] | None = None,
) -> dict[str, Any]:
    raw = {
        "record_id": record_id or record_slug(title),
        "title": title,
        "record_type": record_type,
        "date": date,
        "date_precision": date_precision,
        "location": location,
        "description": description,
        "significance": significance,
        "connected_people": connected_people or [],
        "connected_programs": connected_programs or [],
        "connected_locations": connected_locations or [],
        "evidence_quality": evidence_quality,
        "current_relevance": current_relevance,
        "sources": sources or [],
        "tags": tags or [],
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    rec = _normalize_record(raw)
    text = json.dumps(rec, ensure_ascii=False)
    _historical_upsert_memory(memory_client, user_id, rec["record_id"], text, use_mem0_cloud)
    _sync_hist_file(files_cabinet, rec)
    return rec


def get_record(
    record_id: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    rid = (record_id or "").strip()
    if not rid:
        return None
    return _load_all_records(memory_client, user_id, use_mem0_cloud).get(rid)


def search_archives(
    query: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    out: list[dict[str, Any]] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        blob = " ".join(
            [
                str(rec.get("title", "")),
                str(rec.get("description", "")),
                str(rec.get("location", "")),
                " ".join(rec.get("tags") or []),
                " ".join(rec.get("connected_people") or []),
                " ".join(rec.get("connected_programs") or []),
            ]
        ).lower()
        if q in blob:
            out.append(rec)
    return out


def get_records_by_type(
    record_type: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    rt = (record_type or "").strip().lower()
    return [
        r
        for r in _load_all_records(memory_client, user_id, use_mem0_cloud).values()
        if (r.get("record_type") or "").lower() == rt
    ]


def get_records_by_timeframe(
    start_year: int,
    end_year: int,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        y = _extract_year(rec.get("date") or "")
        if y is None:
            continue
        if start_year <= y <= end_year:
            out.append(rec)
    try:
        out.sort(key=_sort_key_date)
    except Exception:
        pass
    return out


def get_records_by_person(
    person_name: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    needle = (person_name or "").strip().lower()
    if len(needle) < 2:
        return []
    out: list[dict[str, Any]] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        for p in rec.get("connected_people") or []:
            if needle in str(p).lower():
                out.append(rec)
                break
    return out


def get_records_by_location(
    location_name: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    needle = (location_name or "").strip().lower()
    if len(needle) < 2:
        return []
    out: list[dict[str, Any]] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        loc = (rec.get("location") or "").lower()
        cl = " ".join(rec.get("connected_locations") or []).lower()
        if needle in loc or needle in cl:
            out.append(rec)
    return out


def list_all_records(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    return list(_load_all_records(memory_client, user_id, use_mem0_cloud).values())


def get_timeline(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> list[dict[str, Any]]:
    rows = list_all_records(memory_client, user_id, use_mem0_cloud)
    try:
        rows.sort(key=_sort_key_date)
    except Exception:
        pass
    return rows


def get_archive_summary(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    rows = list(_load_all_records(memory_client, user_id, use_mem0_cloud).values())
    by_t: dict[str, int] = {}
    by_s: dict[str, int] = {}
    for r in rows:
        by_t[r.get("record_type") or "unknown"] = by_t.get(r.get("record_type") or "unknown", 0) + 1
        by_s[r.get("significance") or "MEDIUM"] = by_s.get(r.get("significance") or "MEDIUM", 0) + 1
    years = [_extract_year(r.get("date") or "") for r in rows]
    years = [y for y in years if y]
    return {
        "total_records": len(rows),
        "by_type": by_t,
        "by_significance": by_s,
        "year_span": (min(years), max(years)) if years else None,
    }


def research_historical_event(
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
    """Tavily + Claude → HistoricalRecord, saved to archives."""
    import angel as ang

    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "name required"}
    api_key = (__import__("os").getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    lines: list[str] = []
    seen: set[str] = set()
    for q in (
        f"{name} UAP history incident program document",
        f"{name} UFO disclosure timeline",
    ):
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
    ctx = (context or "").strip() or "Tyler UAP/disclosure mission — add to historical archives if accurate in open sources."

    sys = """Output ONE JSON object only (no markdown) for a historical intelligence record:
{
  "title": "display title",
  "record_type": "incident|program|document|testimony|disclosure|cover_up|turning_point",
  "date": "YYYY or YYYY-MM or YYYY-MM-DD",
  "date_precision": "exact|approximate|decade",
  "location": "where",
  "description": "factual summary from sources — mark uncertainty",
  "significance": "LOW|MEDIUM|HIGH|CRITICAL",
  "connected_people": ["names"],
  "connected_programs": ["programs"],
  "connected_locations": ["places"],
  "evidence_quality": "anecdotal|witness|documented|official|declassified",
  "current_relevance": "why it matters now",
  "sources": ["urls from bundle if any"],
  "tags": ["short"]
}
Use only open sources; no classified claims."""

    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=sys,
            messages=[
                {
                    "role": "user",
                    "content": f"EVENT/TOPIC: {name}\nCONTEXT: {ctx[:3000]}\n\nSOURCES:\n{bundle or 'no results'}",
                }
            ],
        )
        txt = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt += block.text
        prof = _parse_json_obj(txt)
        if not prof:
            return {"ok": False, "error": "no JSON from model", "raw": txt[:1500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    prof["record_id"] = record_slug(prof.get("title") or name)
    rec = _normalize_record(prof)
    rec["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = json.dumps(rec, ensure_ascii=False)
    _historical_upsert_memory(memory_client, user_id, rec["record_id"], text, use_mem0_cloud)
    _sync_hist_file(files_cabinet, rec)
    return {"ok": True, "record": rec}


def maybe_link_historical_from_text(
    text: str,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Match free text to archive records (for OSINT / threat cross-reference)."""
    blob = (text or "").lower()
    if len(blob) < 30:
        return []
    hits: list[tuple[int, dict[str, Any]]] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        title = (rec.get("title") or "").lower()
        rid = (rec.get("record_id") or "").lower()
        score = 0
        if title and len(title) > 5 and title in blob:
            score += 12
        for part in re.split(r"[\s,]+", title):
            if len(part) >= 6 and part in blob:
                score += 4
        if rid and rid.replace("-", " ") in blob:
            score += 6
        for tag in rec.get("tags") or []:
            if len(str(tag)) > 4 and str(tag).lower() in blob:
                score += 2
        if score > 0:
            hits.append((score, rec))
    try:
        hits.sort(key=lambda x: -x[0])
    except Exception:
        pass
    return [h[1] for h in hits[:top_n]]


def format_on_this_day_and_anniversaries(
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    *,
    month: int | None = None,
    day: int | None = None,
) -> str:
    """Briefing lines: 'on this day' + rolling anniversaries for CRITICAL/HIGH items."""
    ensure_seed_historical_records(memory_client, user_id, None, use_mem0_cloud)
    y, m, d = _today_utc()
    if month is not None:
        m = month
    if day is not None:
        d = day
    lines: list[str] = []
    on_day: list[str] = []
    anniv: list[str] = []
    for rec in _load_all_records(memory_client, user_id, use_mem0_cloud).values():
        md = _parse_month_day(rec.get("date") or "")
        if not md or md[0] != m or md[1] != d:
            continue
        sig = rec.get("significance") or "MEDIUM"
        if sig not in ("CRITICAL", "HIGH"):
            continue
        title = rec.get("title") or rec.get("record_id")
        on_day.append(f"- {title} ({rec.get('date')}) — {sig}")
        yr = _extract_year(rec.get("date") or "")
        if yr and yr < y:
            anniv.append(f"- Anniversary: {title} (noted ~{yr}; {y - yr} years ago in timeline)")
    if on_day:
        lines.append("ON THIS DAY (UAP history — open-source framing)")
        lines.extend(on_day[:6])
    if anniv and len(lines) < 14:
        lines.append("Anniversary notes:")
        lines.extend(anniv[:5])
    if not lines:
        return ""
    return "HISTORICAL ARCHIVES (timeline context)\n" + "\n".join(lines[:16])


def detect_hist_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if not msg:
        return None, {}
    if re.search(r"(?i)\bUAP\s+history\s+timeline\b|\btimeline\s+of\s+UAP\b|\bUAP\s+timeline\b", msg):
        return "timeline", {}
    if re.search(r"(?i)\bprograms?\s+.*\s+before\s+AARO\b|\bbefore\s+AARO\b.*\bprogram", msg):
        return "programs_before_aaro", {}
    if re.search(r"(?i)\bhistorical\s+context\s+(?:on|for|about)\s+(.+?)(?:\?|$)", msg):
        m = re.search(r"(?i)\bhistorical\s+context\s+(?:on|for|about)\s+(.+?)(?:\?|$)", msg)
        if m:
            return "context", {"topic": m.group(1).strip().rstrip("?.!")}
    if re.search(r"(?i)\bhistory\s+of\s+([\w\s\-\.]+?)\s+in\s+UAP\b", msg):
        m = re.search(r"(?i)\bhistory\s+of\s+([\w\s\-\.]+?)\s+in\s+UAP\b", msg)
        if m:
            return "person_history", {"person": m.group(1).strip()}
    m = re.search(
        r"(?i)\bwhat\s+happened\s+(?:at|in)\s+([\w\s\-]+?)(?:\?|$)",
        msg,
    )
    if m and len(m.group(1).strip()) > 2:
        return "what_happened", {"place": m.group(1).strip()}
    return None, {}


def format_hist_chat_block(
    intent: str,
    payload: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> str:
    try:
        ensure_seed_historical_records(memory_client, user_id, None, use_mem0_cloud)
        if intent == "timeline":
            tl = get_timeline(memory_client, user_id, use_mem0_cloud)
            return "[Angel historical archives — timeline]\n" + json.dumps(
                [{"date": x.get("date"), "title": x.get("title"), "type": x.get("record_type"), "id": x.get("record_id")} for x in tl[:80]],
                ensure_ascii=False,
                indent=2,
            )[:14000]
        if intent == "programs_before_aaro":
            progs = [
                r
                for r in get_records_by_type("program", memory_client, user_id, use_mem0_cloud)
                if (_extract_year(r.get("date") or "") or 9999) < 2022
            ]
            progs.sort(key=_sort_key_date)
            return "[Angel historical archives — programs before AARO]\n" + json.dumps(
                progs[:40], ensure_ascii=False, indent=2
            )[:14000]
        if intent == "context":
            q = (payload.get("topic") or "").strip()
            hits = search_archives(q, memory_client, user_id, use_mem0_cloud)
            return "[Angel historical archives — search]\n" + json.dumps(
                {"query": q, "records": hits[:20]}, ensure_ascii=False, indent=2
            )[:14000]
        if intent == "person_history":
            pn = (payload.get("person") or "").strip()
            rows = get_records_by_person(pn, memory_client, user_id, use_mem0_cloud)
            return "[Angel historical archives — person]\n" + json.dumps(
                {"person": pn, "records": rows[:25]}, ensure_ascii=False, indent=2
            )[:14000]
        if intent == "what_happened":
            place = (payload.get("place") or "").strip()
            hits = search_archives(place, memory_client, user_id, use_mem0_cloud)
            if not hits:
                hits = search_archives(re.sub(r"(?i)^the\s+", "", place), memory_client, user_id, use_mem0_cloud)
            return "[Angel historical archives — place/topic]\n" + json.dumps(
                {"query": place, "records": hits[:15]}, ensure_ascii=False, indent=2
            )[:12000]
    except Exception as e:
        return f"[Angel historical archives error]\n{str(e)[:500]}"
    return ""

