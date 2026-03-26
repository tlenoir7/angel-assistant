"""
Theoretical Suit Design Targets (Build X)

Maintains a living engineering roadmap for two suit philosophies:
- Design A: "iron_man" (maximum power and capability)
- Design B: "batman_beyond" (stealth, agility, augmentation)

This module stores per-domain "target files" in Mem0 as structured JSON under
category `suit_targets` (excluded from routine memory summaries/digests).
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any, Literal

# --- Design philosophies ---------------------------------------------------------

DesignPhilosophy = Literal["iron_man", "batman_beyond"]

# Iron Man domains
DOMAIN_POWER = "im_power_core"
DOMAIN_PROPULSION = "im_propulsion"
DOMAIN_STRUCTURE = "im_structural_materials"
DOMAIN_FLIGHT = "im_flight_systems"
DOMAIN_AI = "im_ai_integration"
DOMAIN_BIOMEDICAL = "im_biomedical_interface"

# Batman Beyond domains
DOMAIN_BB_STEALTH = "bb_stealth_systems"
DOMAIN_BB_STRUCTURE = "bb_adaptive_structure"
DOMAIN_BB_AUGMENTATION = "bb_physical_augmentation"
DOMAIN_BB_AI = "bb_ai_interface"
DOMAIN_BB_SENSORS = "bb_sensor_suite"
DOMAIN_BB_MOBILITY = "bb_mobility_systems"

IRON_MAN_DOMAINS = [
    DOMAIN_POWER,
    DOMAIN_PROPULSION,
    DOMAIN_STRUCTURE,
    DOMAIN_FLIGHT,
    DOMAIN_AI,
    DOMAIN_BIOMEDICAL,
]

BATMAN_BEYOND_DOMAINS = [
    DOMAIN_BB_STEALTH,
    DOMAIN_BB_STRUCTURE,
    DOMAIN_BB_AUGMENTATION,
    DOMAIN_BB_AI,
    DOMAIN_BB_SENSORS,
    DOMAIN_BB_MOBILITY,
]

ALL_DOMAINS = IRON_MAN_DOMAINS + BATMAN_BEYOND_DOMAINS


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_suit_status() -> dict[str, Any]:
    return {
        "ok": True,
        "domains": sorted(ALL_DOMAINS),
        "iron_man_domains": IRON_MAN_DOMAINS,
        "batman_beyond_domains": BATMAN_BEYOND_DOMAINS,
    }


IRONMAN_ENGINEERING_FOLDER = "Iron Man Engineering"
BATMAN_BEYOND_ENGINEERING_FOLDER = "Batman Beyond Engineering"
UAP_INTELLIGENCE_FOLDER = "UAP Intelligence"


def _iso_parse(ts: str) -> datetime | None:
    s = (ts or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_stale(ts: str | None, *, days: int = 7) -> bool:
    dt = _iso_parse(ts or "")
    if dt is None:
        return True
    return dt < (_now_utc() - timedelta(days=int(days)))


def _deep_merge(dst: Any, src: Any) -> Any:
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
                continue
            dst[k] = _deep_merge(dst.get(k), v)
        return dst
    if isinstance(dst, list):
        if src is None:
            return dst
        if isinstance(src, list):
            out: list[Any] = []
            seen: set[str] = set()
            for item in dst + src:
                if item is None:
                    continue
                s = str(item).strip()
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            return out
        if isinstance(src, str) and src.strip():
            return _deep_merge(dst, [src.strip()])
        return dst
    return src if src is not None else dst


def _schema_base(
    domain: str,
    philosophy: DesignPhilosophy,
    target_name: str,
    mission_statement: str,
    performance_requirements: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": domain,
        "design_philosophy": philosophy,
        "target_name": target_name,
        "mission_statement": mission_statement,
        "performance_requirements": performance_requirements,
        "current_best": {
            "technology": "",
            "performance": {},
            "trl": None,
            "gap_ratio": None,
        },
        "research_vectors": [],
        "uap_connection": "",
        "limiting_physics": "",
        "breakthrough_required": "",
        "last_researched": None,
        "mission_relevance": "MEDIUM",
    }


def _default_target_for_domain(domain: str) -> dict[str, Any]:
    d = (domain or "").strip()
    if d == DOMAIN_POWER:
        return _schema_base(
            DOMAIN_POWER,
            "iron_man",
            "Compact high-power energy core",
            "Deliver sustained high electrical power in a compact, wearable-safe package.",
            {
                "energy_density_MJ_per_kg": 1.0,
                "power_output_kW_continuous": 100.0,
                "system_mass_kg_max": 10.0,
                "max_diameter_cm": 30.0,
                "runtime_hr_min": 1.0,
            },
        )
    if d == DOMAIN_PROPULSION:
        return _schema_base(
            DOMAIN_PROPULSION,
            "iron_man",
            "Wearable high-thrust propulsion",
            "Enable human-scale hovering and rapid flight with controllable thrust vectors.",
            {
                "thrust_N_min": 2000.0,
                "specific_impulse_s_min": 500.0,
                "throttle_range": "hover_to_mach3_plus",
                "form_factor": "palms_and_boots",
            },
        )
    if d == DOMAIN_STRUCTURE:
        return _schema_base(
            DOMAIN_STRUCTURE,
            "iron_man",
            "High-strength lightweight armored structure",
            "Provide ballistic/thermal protection while allowing full-body mobility at low mass.",
            {
                "tensile_strength_MPa_min": 2000.0,
                "suit_mass_kg_max": 25.0,
                "ballistic_resistance": "NIJ_Level_IV_equivalent",
                "temp_range_C": {"min": -50, "max": 800},
                "full_range_of_motion": True,
            },
        )
    if d == DOMAIN_FLIGHT:
        return _schema_base(
            DOMAIN_FLIGHT,
            "iron_man",
            "High-altitude, high-speed flight systems",
            "Stabilize and control a human in powered flight over a wide envelope safely.",
            {
                "altitude_ceiling_m_min": 15000.0,
                "top_speed": "Mach_3_plus",
                "sustained_g_limit": 3.0,
                "autonomous_stability": "all_conditions",
            },
        )
    if d == DOMAIN_AI:
        return _schema_base(
            DOMAIN_AI,
            "iron_man",
            "Low-latency AI integration for suit control",
            "Provide real-time sensor fusion, control assistance, and operator UX under constraints.",
            {
                "decision_latency_ms_max": 50.0,
                "power_budget_W_max": 50.0,
                "interface": ["natural_language", "hands_free", "neural_optional"],
            },
        )
    if d == DOMAIN_BIOMEDICAL:
        return _schema_base(
            DOMAIN_BIOMEDICAL,
            "iron_man",
            "Biomedical interface and human factors",
            "Keep the operator safe and effective under acceleration, heat, fatigue, and stress.",
            {
                "sustained_g_tolerance_target": 9.0,
                "core_temp_C_target": 37.0,
                "continuous_biometrics": True,
                "cognition_support": "stress_resilience",
            },
        )
    if d == DOMAIN_BB_STEALTH:
        return _schema_base(
            DOMAIN_BB_STEALTH,
            "batman_beyond",
            "Multi-spectral stealth envelope",
            "Minimize detectability across visual, acoustic, thermal, radar, and EM domains.",
            {
                "visual": "near_invisible_low_light",
                "acoustic_noise_dB_max": 20.0,
                "thermal": "IR_suppressed_to_ambient",
                "radar": "minimal_RCS",
                "electronic": "shielded_low_emissions",
            },
        )
    if d == DOMAIN_BB_STRUCTURE:
        return _schema_base(
            DOMAIN_BB_STRUCTURE,
            "batman_beyond",
            "Adaptive skin-tight protective structure",
            "Second-skin protective layer that stays flexible, quiet, and low-signature.",
            {
                "suit_mass_kg_max": 5.0,
                "ballistic_min": "handgun_rounds",
                "full_range_of_motion": True,
                "self_healing": "minor_damage_autonomous",
                "hardening": "impact_or_threat_triggered",
            },
        )
    if d == DOMAIN_BB_AUGMENTATION:
        return _schema_base(
            DOMAIN_BB_AUGMENTATION,
            "batman_beyond",
            "Physical augmentation without bulk",
            "Enhance strength, endurance, and injury resilience while remaining stealthy and agile.",
            {
                "assistive_torque": "meaningful_for_sprinting_climbing",
                "endurance": "hours",
                "noise": "sub_stealth_threshold",
                "injury_mitigation": "joint_and_spine_support",
            },
        )
    if d == DOMAIN_BB_AI:
        return _schema_base(
            DOMAIN_BB_AI,
            "batman_beyond",
            "Human-directed AI interface",
            "Assist navigation, perception, and planning without automation overriding the operator.",
            {
                "latency_ms_max": 80.0,
                "power_budget_W_max": 20.0,
                "controls": ["silent_gestures", "haptics", "natural_language_optional"],
            },
        )
    if d == DOMAIN_BB_SENSORS:
        return _schema_base(
            DOMAIN_BB_SENSORS,
            "batman_beyond",
            "Low-signature sensor suite",
            "Provide situational awareness while minimizing emissions and capture risk.",
            {
                "passive_primary": True,
                "multispectral": ["low_light", "thermal", "RF_detection_optional"],
                "recording": "secure_local_encrypted",
            },
        )
    if d == DOMAIN_BB_MOBILITY:
        return _schema_base(
            DOMAIN_BB_MOBILITY,
            "batman_beyond",
            "Stealth mobility systems",
            "Enable silent traversal (parkour, climbing, short boosts) with minimal signature.",
            {
                "sprint_speed_gain": "meaningful",
                "vertical_climb_assist": True,
                "silent_operation": True,
                "no_visible_weapons": True,
            },
        )
    # Unknown domain: return a generic placeholder.
    return _schema_base(
        d or "unknown_domain",
        "iron_man",
        "Unknown target",
        "Define mission statement and measurable requirements for this domain.",
        {},
    )


def _extract_numeric(v: Any) -> float | None:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        # Strip common units suffixes, keep numbers and exponent.
        s2 = re.sub(r"[^0-9eE\.\-\+]+", "", s)
        return float(s2)
    except Exception:
        return None


def _calc_gap_ratio(required: Any, current: Any) -> float | None:
    req = _extract_numeric(required)
    cur = _extract_numeric(current)
    if req is None or cur is None:
        return None
    if cur <= 0:
        return None
    # ratio > 1 means current is below requirement (gap exists)
    return round(req / cur, 2)


def _claude_json(anthropic_client: Any, system: str, user: str) -> dict[str, Any]:
    """
    Returns {ok: bool, data: dict|list|str, error: str|None}.
    """
    try:
        resp = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1400,
            temperature=0.25,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        txt = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                txt += str(getattr(block, "text", "") or "")
        raw = (txt or "").strip()
        m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
        payload = m.group(0).strip() if m else raw
        try:
            return {"ok": True, "data": json.loads(payload), "error": None}
        except Exception:
            return {"ok": True, "data": raw, "error": "non_json_response"}
    except Exception as e:
        return {"ok": False, "data": {}, "error": str(e)}


# --- Mem0 storage ----------------------------------------------------------------


def _load_latest_domain_target(
    domain: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any] | None:
    from angel import CATEGORY_SUIT_TARGETS, fetch_combined_memories

    dom = (domain or "").strip()
    if not dom:
        return None
    mem = fetch_combined_memories(memory_client, user_id, use_mem0_cloud) or []
    best_ts = ""
    best: dict[str, Any] | None = None
    for m in mem:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        if meta.get("category") != CATEGORY_SUIT_TARGETS:
            continue
        raw = (m.get("memory") or m.get("data") or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        tgt = obj.get("suit_domain_target")
        if not isinstance(tgt, dict):
            continue
        if str(tgt.get("domain") or "").strip() != dom:
            continue
        ts = str(tgt.get("last_researched") or tgt.get("last_updated") or "")
        if ts >= best_ts:
            best_ts = ts
            best = tgt
    return best


def get_domain_target(
    domain: str,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    base = _default_target_for_domain(domain)
    current = _load_latest_domain_target(domain, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud) or {}
    merged = _deep_merge(base, current)
    if not merged.get("last_researched"):
        merged["last_researched"] = current.get("last_researched") or current.get("last_updated")
    return {"ok": True, "target": merged}


def update_domain_target(
    domain: str,
    updates: dict[str, Any],
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    from angel import CATEGORY_SUIT_TARGETS, add_structured_memory

    if not isinstance(updates, dict):
        return {"ok": False, "error": "updates must be an object"}
    base = get_domain_target(domain, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(domain)
    merged = _deep_merge(base, updates)
    merged["domain"] = (domain or merged.get("domain") or "").strip() or domain
    merged["last_researched"] = _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = json.dumps({"suit_domain_target": merged}, ensure_ascii=False)
    ok = add_structured_memory(
        memory_client,
        user_id,
        payload,
        CATEGORY_SUIT_TARGETS,
        person_name=None,
        use_mem0_cloud=use_mem0_cloud,
    )
    return {"ok": bool(ok), "target": merged}


def list_domain_targets(
    philosophy: DesignPhilosophy | None,
    *,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    ph = philosophy
    domains = IRON_MAN_DOMAINS if ph == "iron_man" else BATMAN_BEYOND_DOMAINS if ph == "batman_beyond" else ALL_DOMAINS
    out: list[dict[str, Any]] = []
    for d in domains:
        out.append(get_domain_target(d, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(d))
    return {"ok": True, "targets": out}


# --- Domain research helpers ------------------------------------------------------


def _research_bundle_for_domain(domain: str, context: str, *, memory_client: Any, user_id: str, use_mem0_cloud: bool) -> dict[str, Any]:
    """
    Pulls multi-source research snippets (ArXiv/NASA/DARPA/Patents) for the domain.
    """
    from angel_research import search_arxiv, search_nasa_reports, search_darpa_programs, search_patents

    dom = (domain or "").strip()
    ctx = (context or "").strip()
    query_map = {
        DOMAIN_POWER: "compact fusion microreactor high energy density battery supercapacitor wearable power",
        DOMAIN_PROPULSION: "personal VTOL propulsion high thrust ducted fan plasma thruster electric jetpack control",
        DOMAIN_STRUCTURE: "ballistic composite ceramic armor shear thickening fluid aerogel high temperature lightweight",
        DOMAIN_FLIGHT: "autonomous flight control VTOL stabilization human wearable flight envelope hypersonic",
        DOMAIN_AI: "edge AI neuromorphic sensor fusion low power real time control wearable",
        DOMAIN_BIOMEDICAL: "G-suit acceleration tolerance exoskeleton human factors thermoregulation wearable biometrics",
        DOMAIN_BB_STEALTH: "metamaterial cloaking active camouflage electrochromic polymer thermal masking acoustic stealth",
        DOMAIN_BB_STRUCTURE: "adaptive armor shear thickening fluid self healing polymer textile composite",
        DOMAIN_BB_AUGMENTATION: "soft exosuit textile actuator silent wearable augmentation",
        DOMAIN_BB_AI: "haptic interface silent control wearable AR sensor fusion low power",
        DOMAIN_BB_SENSORS: "passive sensing thermal low light event camera RF detection low emission",
        DOMAIN_BB_MOBILITY: "silent mobility climbing assist soft robotics microthruster pneumatic actuator",
    }
    q = (query_map.get(dom) or dom).strip()
    if ctx:
        q = f"{q} {ctx[:220]}"
    mem_kw = {"memory_client": memory_client, "user_id": user_id, "use_mem0_cloud": use_mem0_cloud}
    with ThreadPoolExecutor(max_workers=6) as ex:
        f_arxiv = ex.submit(search_arxiv, q, 8, **mem_kw)
        f_nasa = ex.submit(search_nasa_reports, q, 8)
        f_darpa = ex.submit(search_darpa_programs, q)
        f_pat = ex.submit(search_patents, q, 6)
        return {
            "query": q,
            "arxiv": f_arxiv.result(timeout=70),
            "nasa": f_nasa.result(timeout=70),
            "darpa": f_darpa.result(timeout=70),
            "patents": f_pat.result(timeout=70),
        }


def _file_assessment(
    folder: str,
    name: str,
    body: dict[str, Any],
    *,
    files_cabinet: Any,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    try:
        content = json.dumps(body, ensure_ascii=False, indent=2, default=str)
    except Exception:
        content = str(body)
    try:
        rec = files_cabinet.create_file(folder, name, content, tags=tags or [])
        return {"ok": True, "file": rec}
    except Exception as e:
        return {"ok": False, "error": str(e), "folder": folder, "name": name}


def _synthesize_target_update(
    domain_target: dict[str, Any],
    research_bundle: dict[str, Any],
    *,
    anthropic_client: Any,
    extra_notes: str = "",
) -> dict[str, Any]:
    """
    Claude synthesis: produce an updated target record (domain file structure).
    """
    blob = json.dumps(
        {"target": domain_target, "research": research_bundle, "notes": extra_notes},
        ensure_ascii=False,
        indent=2,
    )[:120_000]
    system = """You are building a "domain target file" for a theoretical suit roadmap.
Using ONLY the JSON bundle, output ONE JSON object with exactly these keys:
- current_best {technology, performance, trl, gap_ratio}
- research_vectors [ {approach, lead_organizations, trl, timeline, key_papers, gap_closes} ... ] (max 6)
- uap_connection
- limiting_physics
- breakthrough_required
- mission_relevance (LOW|MEDIUM|HIGH|CRITICAL)
Constraints:
- Be concrete and measurable where possible.
- If you cannot quantify a value, set it to null and explain in limiting_physics/breakthrough_required.
JSON only."""
    parsed = _claude_json(anthropic_client, system, blob)
    data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
    return {"ok": bool(parsed.get("ok")), "update": data, "error": parsed.get("error")}


# --- Iron Man research functions --------------------------------------------------


def research_im_power_core(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_POWER, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_POWER)
    bundle = _research_bundle_for_domain(DOMAIN_POWER, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)

    # Quick physics sanity: 100 kW for 1 hr = 360 MJ energy. At 1 MJ/kg => 360 kg, so requirement is extremely aggressive.
    req = tgt.get("performance_requirements") if isinstance(tgt.get("performance_requirements"), dict) else {}
    p_kw = _extract_numeric(req.get("power_output_kW_continuous")) or 100.0
    rt_hr = _extract_numeric(req.get("runtime_hr_min")) or 1.0
    e_mj = p_kw * rt_hr * 3.6  # (kW * hr) -> MJ
    notes = f"Sanity check: {p_kw:.0f} kW for {rt_hr:.1f} hr => ~{e_mj:.0f} MJ electrical energy (not counting conversion losses)."

    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes=notes)
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_POWER, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_im_propulsion(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_PROPULSION, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_PROPULSION)
    bundle = _research_bundle_for_domain(DOMAIN_PROPULSION, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)

    # Hover thrust sanity: need >= weight. For 100 kg system, ~981 N.
    notes = "Sanity check: hover thrust must exceed total weight (operator + suit + fuel). 2000 N implies ~200 kg weight-equivalent at 1 g ignoring control margins."
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes=notes)
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_PROPULSION, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_im_structural_materials(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_STRUCTURE, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_STRUCTURE)
    bundle = _research_bundle_for_domain(DOMAIN_STRUCTURE, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Focus: strength-to-weight, ballistic ceramics/composites, high-temp insulation, mobility joints.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_STRUCTURE, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_im_flight_systems(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_FLIGHT, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_FLIGHT)
    bundle = _research_bundle_for_domain(DOMAIN_FLIGHT, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Include stability augmentation, control laws, redundancy, and safe flight envelopes for a human operator.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_FLIGHT, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_im_ai_integration(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_AI, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_AI)
    bundle = _research_bundle_for_domain(DOMAIN_AI, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Note: Angel is the AI layer; focus is compute + sensors + UX + latency + power budget.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_AI, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_im_biomedical_interface(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BIOMEDICAL, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BIOMEDICAL)
    bundle = _research_bundle_for_domain(DOMAIN_BIOMEDICAL, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Blend G-suit, thermal management, wearable monitoring, and cognitive load under stress.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BIOMEDICAL, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


# --- Batman Beyond research functions --------------------------------------------


def research_bb_stealth_systems(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_STEALTH, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_STEALTH)
    bundle = _research_bundle_for_domain(DOMAIN_BB_STEALTH, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Emphasize passive-first and emission control. Distinguish lab demos vs fieldable TRL.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_STEALTH, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_bb_adaptive_structure(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_STRUCTURE, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_STRUCTURE)
    bundle = _research_bundle_for_domain(DOMAIN_BB_STRUCTURE, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Focus: STF textiles, smart materials, self-healing polymers, thin hard-plates without noise.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_STRUCTURE, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_bb_physical_augmentation(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_AUGMENTATION, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_AUGMENTATION)
    bundle = _research_bundle_for_domain(DOMAIN_BB_AUGMENTATION, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Prioritize soft exosuits and silent actuators; include human-factor constraints and injury risk.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_AUGMENTATION, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_bb_ai_interface(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_AI, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_AI)
    bundle = _research_bundle_for_domain(DOMAIN_BB_AI, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Stealth UX: haptics, eye-tracking, silent gesture, on-device processing.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_AI, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_bb_sensor_suite(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_SENSORS, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_SENSORS)
    bundle = _research_bundle_for_domain(DOMAIN_BB_SENSORS, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Prefer passive sensors (low-light/thermal) and RF awareness with minimal emissions.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_SENSORS, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


def research_bb_mobility_systems(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    tgt = get_domain_target(DOMAIN_BB_MOBILITY, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or _default_target_for_domain(DOMAIN_BB_MOBILITY)
    bundle = _research_bundle_for_domain(DOMAIN_BB_MOBILITY, context, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    syn = _synthesize_target_update(tgt, bundle, anthropic_client=anthropic_client, extra_notes="Focus: silent mobility improvements, climbing assist, short bursts without obvious propulsion signature.")
    upd = syn.get("update") if isinstance(syn.get("update"), dict) else {}
    return update_domain_target(DOMAIN_BB_MOBILITY, upd, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)


DOMAIN_RESEARCH_FUNCS: dict[str, Any] = {
    DOMAIN_POWER: research_im_power_core,
    DOMAIN_PROPULSION: research_im_propulsion,
    DOMAIN_STRUCTURE: research_im_structural_materials,
    DOMAIN_FLIGHT: research_im_flight_systems,
    DOMAIN_AI: research_im_ai_integration,
    DOMAIN_BIOMEDICAL: research_im_biomedical_interface,
    DOMAIN_BB_STEALTH: research_bb_stealth_systems,
    DOMAIN_BB_STRUCTURE: research_bb_adaptive_structure,
    DOMAIN_BB_AUGMENTATION: research_bb_physical_augmentation,
    DOMAIN_BB_AI: research_bb_ai_interface,
    DOMAIN_BB_SENSORS: research_bb_sensor_suite,
    DOMAIN_BB_MOBILITY: research_bb_mobility_systems,
}


def research_domain(
    domain: str,
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    dom = (domain or "").strip()
    fn = DOMAIN_RESEARCH_FUNCS.get(dom)
    if not fn:
        return {"ok": False, "error": f"Unknown domain: {dom}", "domains": sorted(ALL_DOMAINS)}
    return fn(
        context,
        anthropic_client=anthropic_client,
        memory_client=memory_client,
        user_id=user_id,
        use_mem0_cloud=use_mem0_cloud,
    )


def get_suit_roadmap(
    context: str,
    philosophy: DesignPhilosophy | None,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    targets = list_domain_targets(philosophy, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("targets") or []
    blob = json.dumps({"context": context, "targets": targets}, ensure_ascii=False, indent=2)[:140_000]
    system = """You are Angel's suit roadmap planner. Using ONLY the JSON, output ONE JSON:
- philosophy (iron_man|batman_beyond|both)
- top_gaps (array of strings)
- best_next_research_actions (array of strings)
- recommended_domain_priorities (array of {domain, why, mission_relevance})
- notes (string)
JSON only."""
    parsed = _claude_json(anthropic_client, system, blob)
    return {
        "ok": bool(parsed.get("ok")),
        "mode": "suit_roadmap",
        "philosophy": philosophy or "both",
        "targets": targets,
        "roadmap": parsed.get("data"),
        "error": parsed.get("error"),
    }


def analyze_suit_convergence(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    targets = list_domain_targets(None, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("targets") or []
    blob = json.dumps({"context": context, "targets": targets}, ensure_ascii=False, indent=2)[:140_000]
    system = """You analyze convergence between two suit philosophies. Using ONLY the JSON, output ONE JSON:
- shared_technologies (array of {name, why_shared, priority (1-5), domains (array), near_term (boolean)})
- fastest_dual_use_wins (array of strings)
- conflicts_or_tradeoffs (array of strings)
- notes (string)
JSON only."""
    parsed = _claude_json(anthropic_client, system, blob)
    return {
        "ok": bool(parsed.get("ok")),
        "mode": "suit_convergence",
        "analysis": parsed.get("data"),
        "error": parsed.get("error"),
    }


def run_full_ironman_assessment(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    ctx = (context or "").strip()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            DOMAIN_POWER: ex.submit(research_im_power_core, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_PROPULSION: ex.submit(research_im_propulsion, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_STRUCTURE: ex.submit(research_im_structural_materials, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_FLIGHT: ex.submit(research_im_flight_systems, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_AI: ex.submit(research_im_ai_integration, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BIOMEDICAL: ex.submit(research_im_biomedical_interface, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
        }
        results: dict[str, Any] = {}
        for dom, fut in futs.items():
            try:
                results[dom] = fut.result(timeout=220)
            except Exception as e:
                results[dom] = {"ok": False, "error": str(e), "domain": dom}
    assessment = {"ok": True, "mode": "ironman_full_assessment", "design": "iron_man", "context": ctx, "domains": results, "generated_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")}
    h = hashlib.sha256(json.dumps(assessment, ensure_ascii=False, default=str).encode()).hexdigest()[:10]
    fn = f"IM-{_now_utc().strftime('%Y%m%d')}-{h}"
    tags = ["design:iron_man", "assessment", "suit_targets"]
    filed = _file_assessment(IRONMAN_ENGINEERING_FOLDER, fn, assessment, files_cabinet=files_cabinet, tags=tags)
    assessment["filed_as"] = filed
    return assessment


def run_full_batman_beyond_assessment(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    ctx = (context or "").strip()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            DOMAIN_BB_STEALTH: ex.submit(research_bb_stealth_systems, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BB_STRUCTURE: ex.submit(research_bb_adaptive_structure, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BB_AUGMENTATION: ex.submit(research_bb_physical_augmentation, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BB_AI: ex.submit(research_bb_ai_interface, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BB_SENSORS: ex.submit(research_bb_sensor_suite, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
            DOMAIN_BB_MOBILITY: ex.submit(research_bb_mobility_systems, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud),
        }
        results: dict[str, Any] = {}
        for dom, fut in futs.items():
            try:
                results[dom] = fut.result(timeout=220)
            except Exception as e:
                results[dom] = {"ok": False, "error": str(e), "domain": dom}
    assessment = {"ok": True, "mode": "batman_beyond_full_assessment", "design": "batman_beyond", "context": ctx, "domains": results, "generated_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")}
    h = hashlib.sha256(json.dumps(assessment, ensure_ascii=False, default=str).encode()).hexdigest()[:10]
    fn = f"BB-{_now_utc().strftime('%Y%m%d')}-{h}"
    tags = ["design:batman_beyond", "assessment", "suit_targets"]
    filed = _file_assessment(BATMAN_BEYOND_ENGINEERING_FOLDER, fn, assessment, files_cabinet=files_cabinet, tags=tags)
    assessment["filed_as"] = filed
    return assessment


def run_full_suit_assessment(
    context: str,
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
    files_cabinet: Any,
) -> dict[str, Any]:
    ctx = (context or "").strip()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_im = ex.submit(run_full_ironman_assessment, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        f_bb = ex.submit(run_full_batman_beyond_assessment, ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud, files_cabinet=files_cabinet)
        im = f_im.result(timeout=360)
        bb = f_bb.result(timeout=360)
    conv = analyze_suit_convergence(ctx, anthropic_client=anthropic_client, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud)
    out = {"ok": True, "mode": "full_suit_assessment", "context": ctx, "iron_man": im, "batman_beyond": bb, "convergence": conv, "generated_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")}
    return out


def generate_suit_cad_component(
    design: DesignPhilosophy,
    domain: str,
    specs: dict[str, Any],
    *,
    user_id: str,
) -> dict[str, Any]:
    """
    Returns CAD output for a conceptual component. Uses angel_cad.generate_shape.
    """
    from angel_cad import generate_shape

    dsgn = design
    dom = (domain or "").strip()
    sp = specs if isinstance(specs, dict) else {}
    session_id = user_id or "default-user"

    # Simple conceptual geometry (not a production design).
    if dsgn == "iron_man" and dom == DOMAIN_POWER:
        diameter_mm = float(sp.get("diameter_mm") or 280)
        thickness_mm = float(sp.get("thickness_mm") or 90)
        return generate_shape(
            "lenticular",
            {"diameter": diameter_mm, "thickness": thickness_mm, "units": "mm"},
            session_id=session_id,
            design_name="ironman_power_core_concept",
            context="Iron Man power core conceptual geometry",
        )
    if dsgn == "iron_man" and dom == DOMAIN_PROPULSION:
        throat = float(sp.get("throat_radius_mm") or 12)
        exit_r = float(sp.get("exit_radius_mm") or 30)
        length = float(sp.get("length_mm") or 120)
        return generate_shape(
            "nozzle",
            {"throat_radius": throat, "exit_radius": exit_r, "length": length, "units": "mm"},
            session_id=session_id,
            design_name="ironman_repulsor_nozzle_concept",
            context="Palm/boot thruster nozzle concept",
        )
    if dsgn == "batman_beyond" and dom == DOMAIN_BB_MOBILITY:
        chord = float(sp.get("chord_mm") or 260)
        span = float(sp.get("span_mm") or 900)
        naca = str(sp.get("naca_code") or "0012")
        return generate_shape(
            "airfoil",
            {"naca_code": naca, "chord": chord, "span": span, "units": "mm"},
            session_id=session_id,
            design_name="bb_glide_wing_concept",
            context="Batman Beyond glide wing concept",
        )
    if dsgn == "batman_beyond" and dom == DOMAIN_BB_STRUCTURE:
        outer_r = float(sp.get("outer_radius_mm") or 55)
        thick = float(sp.get("thickness_mm") or 4)
        return generate_shape(
            "disc",
            {"outer_radius": outer_r, "inner_radius": 0, "thickness": thick, "units": "mm"},
            session_id=session_id,
            design_name="bb_armor_pad_concept",
            context="Adaptive armor pad concept (STF textile stack placeholder)",
        )
    # Fallback: small box component
    return generate_shape(
        "box",
        {"length": float(sp.get("length_mm") or 80), "width": float(sp.get("width_mm") or 40), "height": float(sp.get("height_mm") or 20), "units": "mm"},
        session_id=session_id,
        design_name=f"{dsgn}_{dom}_component_concept",
        context="Generic suit component concept",
    )


# --- Chat intent integration ------------------------------------------------------

_SUIT_TRIGGERS = re.compile(
    r"(?i)\b(?:"
    r"iron\s*man|batman\s*beyond|powered\s+armor|exosuit|exoskeleton\s+suit|"
    r"repulsor|arc\s+reactor|flight\s+suit|stealth\s+suit|second\s+skin\s+suit|"
    r"suit\s+roadmap|engineering\s+roadmap|design\s+targets|domain\s+targets"
    r")\b"
)


def detect_suit_chat_intent(user_message: str) -> tuple[str | None, dict[str, Any]]:
    msg = (user_message or "").strip()
    if len(msg) < 10 or not _SUIT_TRIGGERS.search(msg):
        return None, {}
    low = msg.lower()
    payload: dict[str, Any] = {"original": msg[:600]}
    if "batman" in low or "beyond" in low:
        payload["philosophy"] = "batman_beyond"
    elif "iron" in low or "arc reactor" in low or "repulsor" in low:
        payload["philosophy"] = "iron_man"
    else:
        payload["philosophy"] = "both"

    # Domain selection by keyword
    dom = None
    dom_map = {
        "power": DOMAIN_POWER,
        "reactor": DOMAIN_POWER,
        "arc": DOMAIN_POWER,
        "fusion": DOMAIN_POWER,
        "battery": DOMAIN_POWER,
        "propulsion": DOMAIN_PROPULSION,
        "repulsor": DOMAIN_PROPULSION,
        "thrust": DOMAIN_PROPULSION,
        "structural": DOMAIN_STRUCTURE,
        "armor": DOMAIN_STRUCTURE,
        "materials": DOMAIN_STRUCTURE,
        "flight": DOMAIN_FLIGHT,
        "aerodynamic": DOMAIN_FLIGHT,
        "ai": DOMAIN_AI,
        "sensor fusion": DOMAIN_AI,
        "biomedical": DOMAIN_BIOMEDICAL,
        "g-suit": DOMAIN_BIOMEDICAL,
        "stealth": DOMAIN_BB_STEALTH,
        "camouflage": DOMAIN_BB_STEALTH,
        "adaptive": DOMAIN_BB_STRUCTURE,
        "self-heal": DOMAIN_BB_STRUCTURE,
        "augmentation": DOMAIN_BB_AUGMENTATION,
        "exosuit": DOMAIN_BB_AUGMENTATION,
        "haptic": DOMAIN_BB_AI,
        "sensors": DOMAIN_BB_SENSORS,
        "mobility": DOMAIN_BB_MOBILITY,
    }
    for k, v in dom_map.items():
        if k in low:
            dom = v
            break
    if dom:
        payload["domain"] = dom
        return "suit_domain", payload
    return "suit_roadmap", payload


def run_suit_intent_for_chat(
    intent: str,
    payload: dict[str, Any],
    *,
    anthropic_client: Any,
    memory_client: Any,
    user_id: str,
    use_mem0_cloud: bool,
) -> dict[str, Any]:
    ctx = str(payload.get("original") or "")
    if intent == "suit_domain":
        dom = str(payload.get("domain") or "").strip()
        existing = get_domain_target(dom, memory_client=memory_client, user_id=user_id, use_mem0_cloud=use_mem0_cloud).get("target") or {}
        if _is_stale(str(existing.get("last_researched") or ""), days=7):
            return research_domain(
                dom,
                ctx,
                anthropic_client=anthropic_client,
                memory_client=memory_client,
                user_id=user_id,
                use_mem0_cloud=use_mem0_cloud,
            )
        return {"ok": True, "mode": "suit_domain_cached", "target": existing}
    if intent == "suit_roadmap":
        ph = payload.get("philosophy")
        phv: DesignPhilosophy | None = ph if ph in ("iron_man", "batman_beyond") else None
        return get_suit_roadmap(
            ctx,
            phv,
            anthropic_client=anthropic_client,
            memory_client=memory_client,
            user_id=user_id,
            use_mem0_cloud=use_mem0_cloud,
        )
    return {"ok": False, "error": f"Unknown intent: {intent}"}


def format_suit_block_for_prompt(result: dict[str, Any]) -> str:
    """
    Formats a structured appendix for Claude. This is appended to the user message.
    """
    try:
        mode = str(result.get("mode") or "suit").strip() or "suit"
    except Exception:
        mode = "suit"
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)[:90_000]
    return f"[Angel suit — {mode}]\n{payload}"

