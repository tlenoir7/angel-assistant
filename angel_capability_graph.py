"""
Capability Graph system for Angel.

Goal: a structured map of how Angel's systems feed into each other, plus
fast per-turn recognition of multi-capability opportunities.

Design notes:
- `build_capability_context_for_prompt()` MUST be fast/light (called every turn).
- Deeper analysis (Claude-assisted) is exposed via API helpers, but per-turn
  prompt injection uses heuristics only.
"""

from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Capability map --------------------------------------------------------------

CAPABILITIES: dict[str, dict[str, Any]] = {
    "osint": {
        "name": "OSINT Deep Background",
        "description": "Systematic open source intelligence on people and organizations",
        "feeds_into": ["network_graph", "threat_detection", "predictions", "proactive_intel"],
        "fed_by": ["research_agent", "web_search"],
        "api": "/api/osint/background",
        "trigger_keywords": ["who is", "background on", "research", "dossier"],
        "output_types": ["dossier", "intelligence_file"],
        "folder": "OSINT Dossiers",
    },
    "network_graph": {
        "name": "Relationship Network Mapping",
        "description": "Maps connections between people and organizations",
        "feeds_into": ["threat_detection", "predictions", "proactive_intel"],
        "fed_by": ["osint", "proactive_intel"],
        "api": "/api/network/add-node",
        "trigger_keywords": ["connected to", "relationship", "network", "who knows"],
        "output_types": ["network_node", "network_edge"],
        "folder": "Network Intelligence",
    },
    "threat_detection": {
        "name": "Threat Detection",
        "description": "Monitors for threats to Tyler and his mission",
        "feeds_into": ["predictions", "proactive_intel"],
        "fed_by": ["osint", "network_graph", "proactive_intel", "web_search"],
        "api": "/api/threats/scan",
        "trigger_keywords": ["threat", "risk", "danger", "monitor", "watch"],
        "output_types": ["threat_filing"],
        "folder": "Threat Intelligence",
    },
    "predictions": {
        "name": "Predictive Modeling",
        "description": "Forecasts future developments based on current intelligence",
        "feeds_into": ["proactive_intel", "briefings"],
        "fed_by": ["osint", "network_graph", "threat_detection", "research_agent"],
        "api": "/api/predictions/generate",
        "trigger_keywords": ["predict", "forecast", "likely", "what will happen"],
        "output_types": ["prediction"],
        "folder": "Predictions",
    },
    "proactive_intel": {
        "name": "Proactive Background Intelligence",
        "description": "Angel monitors topics autonomously without being asked",
        "feeds_into": ["threat_detection", "network_graph", "briefings"],
        "fed_by": ["osint", "web_search", "research_agent"],
        "api": "/api/proactive/run",
        "trigger_keywords": ["watch", "monitor", "track", "keep an eye on"],
        "output_types": ["proactive_finding"],
        "folder": "Proactive Intelligence",
    },
    "research_agent": {
        "name": "Theoretical Research Agent",
        "description": "Pulls from ArXiv, NASA, DARPA, patents simultaneously",
        "feeds_into": ["physics_sim", "chemistry", "suit_design", "osint"],
        "fed_by": ["web_search"],
        "api": "/api/research/query",
        "trigger_keywords": ["research", "papers", "studies", "what does science say"],
        "output_types": ["research_brief"],
        "folder": "Research Intelligence",
    },
    "physics_sim": {
        "name": "Physics Simulation Engine",
        "description": "Runs actual physics calculations and simulations",
        "feeds_into": ["suit_design", "cad_generation"],
        "fed_by": ["research_agent"],
        "api": "/api/physics/simulate",
        "trigger_keywords": ["simulate", "calculate", "would it work", "physics of"],
        "output_types": ["simulation_result"],
        "folder": "Physics Simulations",
    },
    "chemistry": {
        "name": "Chemical and Materials Synthesis",
        "description": "PubChem, NIST, Materials Project database access",
        "feeds_into": ["suit_design", "cad_generation", "medical"],
        "fed_by": ["research_agent"],
        "api": "/api/chemistry/compound",
        "trigger_keywords": ["material", "compound", "synthesis", "chemical"],
        "output_types": ["chemistry_brief"],
        "folder": "Chemistry Intelligence",
    },
    "cad_generation": {
        "name": "CAD Generation",
        "description": "Generates actual STEP and STL CAD files",
        "feeds_into": ["visualization_3d"],
        "fed_by": ["physics_sim", "chemistry", "suit_design"],
        "api": "/api/cad/from-brief",
        "trigger_keywords": ["design", "generate", "cad", "model", "build specs"],
        "output_types": ["cad_file"],
        "folder": "Engineering Designs",
    },
    "visualization_3d": {
        "name": "3D Visualization",
        "description": "Renders CAD models in iPhone app",
        "feeds_into": [],
        "fed_by": ["cad_generation"],
        "api": "/api/cad/mesh-json",
        "trigger_keywords": ["view in 3d", "show me", "visualize"],
        "output_types": ["3d_render"],
        "folder": None,
    },
    "medical": {
        "name": "Medical Intelligence Core",
        "description": "PubMed, FDA, ClinicalTrials, MedlinePlus access",
        "feeds_into": ["biomedical_research", "treatment_design", "health_profile"],
        "fed_by": ["research_agent"],
        "api": "/api/medical/condition",
        "trigger_keywords": ["medical", "condition", "treatment", "drug", "clinical"],
        "output_types": ["medical_brief"],
        "folder": "Medical Intelligence",
    },
    "biomedical_research": {
        "name": "Biomedical Research Agent",
        "description": "UniProt, NCBI Gene, KEGG, PDB database access",
        "feeds_into": ["treatment_design"],
        "fed_by": ["medical", "research_agent"],
        "api": "/api/medical/biomedical-research",
        "trigger_keywords": ["gene", "protein", "pathway", "molecular", "genomics"],
        "output_types": ["biomedical_brief"],
        "folder": "Medical Intelligence",
    },
    "treatment_design": {
        "name": "Theoretical Treatment Design",
        "description": "Designs novel theoretical treatment approaches",
        "feeds_into": [],
        "fed_by": ["medical", "biomedical_research", "chemistry"],
        "api": "/api/medical/design-treatment",
        "trigger_keywords": ["treat", "therapy", "design a treatment", "what would work for"],
        "output_types": ["treatment_brief"],
        "folder": "Theoretical Medicine",
    },
    "health_profile": {
        "name": "Personal Health Intelligence",
        "description": "Tyler's personal health profile and monitoring",
        "feeds_into": ["medical"],
        "fed_by": ["medical"],
        "api": "/api/health/profile",
        "trigger_keywords": ["my health", "how am i doing", "my fitness"],
        "output_types": ["health_assessment"],
        "folder": None,
    },
    "suit_design": {
        "name": "Theoretical Suit Design",
        "description": "Iron Man and Batman Beyond engineering roadmaps",
        "feeds_into": ["cad_generation", "physics_sim"],
        "fed_by": ["research_agent", "chemistry", "physics_sim", "medical"],
        "api": "/api/ironman/assess",
        "trigger_keywords": ["iron man", "batman beyond", "suit", "powered armor"],
        "output_types": ["suit_assessment"],
        "folder": "Iron Man Engineering",
    },
    "vision_forensic": {
        "name": "Computer Vision on Demand",
        "description": "Forensic analysis of images and camera input",
        "feeds_into": ["osint", "network_graph", "threat_detection"],
        "fed_by": [],
        "api": "/api/vision/forensic",
        "trigger_keywords": ["look at this", "analyze this image", "forensic"],
        "output_types": ["forensic_analysis"],
        "folder": "Visual Intelligence",
    },
    "translation": {
        "name": "Real Time Translation",
        "description": "Multilingual intelligence and foreign source monitoring",
        "feeds_into": ["osint", "proactive_intel"],
        "fed_by": ["web_search"],
        "api": "/api/translate",
        "trigger_keywords": ["translate", "foreign", "language"],
        "output_types": ["translation"],
        "folder": "Foreign Intelligence",
    },
    "web_search": {
        "name": "Web Search",
        "description": "Real-time web search via Tavily",
        "feeds_into": ["osint", "research_agent", "proactive_intel", "threat_detection"],
        "fed_by": [],
        "api": None,
        "trigger_keywords": ["search", "latest", "recent", "news"],
        "output_types": ["search_results"],
        "folder": None,
    },
    "file_cabinet": {
        "name": "Intelligence File Cabinet",
        "description": "Angel's full filing system for all intelligence",
        "feeds_into": [],
        "fed_by": [
            "osint",
            "threat_detection",
            "predictions",
            "research_agent",
            "medical",
            "chemistry",
            "suit_design",
            "vision_forensic",
        ],
        "api": "/api/files/summary",
        "trigger_keywords": ["file", "save", "intelligence", "what have you filed"],
        "output_types": ["intelligence_file"],
        "folder": None,
    },
    "self_modification": {
        "name": "Self Modification — Stage 6",
        "description": "Angel permanently adapts behavior based on observations",
        "feeds_into": [],
        "fed_by": [],
        "api": "/api/selfmod/proposals",
        "trigger_keywords": ["improve yourself", "adapt", "modify"],
        "output_types": ["modification_proposal"],
        "folder": None,
    },
    "multi_agent": {
        "name": "Multi Agent Coordination",
        "description": "Spawns parallel specialist agents for complex tasks",
        "feeds_into": [],
        "fed_by": [],
        "api": "/api/agents/run",
        "trigger_keywords": ["deep analysis", "comprehensive", "full intelligence"],
        "output_types": ["agent_synthesis"],
        "folder": None,
    },
}


def _normalize_cap_id(cap: str) -> str:
    return (cap or "").strip().lower()


def _edges() -> dict[str, list[str]]:
    e: dict[str, list[str]] = {}
    for k, v in CAPABILITIES.items():
        outs = v.get("feeds_into") if isinstance(v.get("feeds_into"), list) else []
        e[_normalize_cap_id(k)] = [str(x).strip() for x in outs if str(x).strip()]
    return e


def get_graph_as_json() -> dict[str, Any]:
    nodes = []
    edges = []
    for cap_id, meta in CAPABILITIES.items():
        nodes.append({"id": cap_id, **meta})
        for dst in meta.get("feeds_into") or []:
            edges.append({"source": cap_id, "target": dst})
    return {"ok": True, "generated_at": _now_utc_iso(), "nodes": nodes, "edges": edges}


# --- Chain analysis --------------------------------------------------------------


def get_capability_chain(starting_capability: str) -> dict[str, Any]:
    start = _normalize_cap_id(starting_capability)
    if start not in CAPABILITIES:
        return {"ok": False, "error": "unknown capability", "capability": start, "known": sorted(CAPABILITIES.keys())}
    e = _edges()
    visited: set[str] = set()
    order: list[str] = []
    q: deque[str] = deque([start])
    visited.add(start)
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in e.get(cur, []):
            n = _normalize_cap_id(nxt)
            if not n or n in visited:
                continue
            visited.add(n)
            q.append(n)
    return {"ok": True, "start": start, "chain": order}


def get_feeding_capabilities(target_capability: str) -> dict[str, Any]:
    target = _normalize_cap_id(target_capability)
    if target not in CAPABILITIES:
        return {"ok": False, "error": "unknown capability", "capability": target, "known": sorted(CAPABILITIES.keys())}
    # reverse edges
    rev: dict[str, list[str]] = {k: [] for k in CAPABILITIES.keys()}
    for src, meta in CAPABILITIES.items():
        for dst in meta.get("feeds_into") or []:
            d = _normalize_cap_id(dst)
            if d in rev:
                rev[d].append(src)
    visited: set[str] = set()
    q: deque[str] = deque([target])
    visited.add(target)
    ups: list[str] = []
    while q:
        cur = q.popleft()
        for prev in rev.get(cur, []):
            p = _normalize_cap_id(prev)
            if not p or p in visited:
                continue
            visited.add(p)
            ups.append(p)
            q.append(p)
    return {"ok": True, "target": target, "fed_by": sorted(ups)}


# --- Combination recognition (fast heuristics) -----------------------------------

_WORD = re.compile(r"[a-z0-9_]+", re.I)


def _msg_has_any(msg: str, phrases: list[str]) -> bool:
    m = (msg or "").lower()
    return any((p or "").lower() in m for p in phrases if p)


def _score_capability_from_keywords(msg: str, cap_id: str) -> int:
    meta = CAPABILITIES.get(cap_id) or {}
    kws = meta.get("trigger_keywords") if isinstance(meta.get("trigger_keywords"), list) else []
    score = 0
    low = (msg or "").lower()
    for kw in kws:
        k = str(kw).lower().strip()
        if not k:
            continue
        if k in low:
            score += 3
    # Extra weighting for explicit api-like words
    if cap_id == "vision_forensic" and re.search(r"(?i)\b(image|photo|screenshot|picture|camera)\b", msg or ""):
        score += 2
    if cap_id == "medical" and re.search(r"(?i)\b(symptom|diagnos|treat|medicine|drug)\b", msg or ""):
        score += 2
    if cap_id == "suit_design" and re.search(r"(?i)\b(iron\s*man|batman|powered\s+armor|exosuit)\b", msg or ""):
        score += 3
    if cap_id == "osint" and re.search(r"(?i)\bwho\s+is\b", msg or ""):
        score += 2
    return score


KNOWN_HIGH_VALUE_COMBINATIONS: list[dict[str, Any]] = [
    {
        "name": "Person + threat triage",
        "example": "Who is [person] and are they a threat?",
        "capabilities": ["osint", "network_graph", "threat_detection"],
    },
    {
        "name": "Engineering stack",
        "example": "What material should I use for [application]?",
        "capabilities": ["chemistry", "physics_sim", "cad_generation"],
    },
    {
        "name": "UAP disclosure intelligence",
        "example": "What's the latest on UAP disclosure?",
        "capabilities": ["web_search", "osint", "network_graph", "predictions"],
    },
    {
        "name": "Medical synthesis + design",
        "example": "Design a treatment for [condition].",
        "capabilities": ["medical", "biomedical_research", "chemistry", "treatment_design"],
    },
    {
        "name": "Field photo escalation",
        "example": "Analyze this photo from the field.",
        "capabilities": ["vision_forensic", "osint", "network_graph", "threat_detection"],
    },
]


def recognize_capability_combinations(user_message: str) -> dict[str, Any]:
    msg = (user_message or "").strip()
    if len(msg) < 3:
        return {
            "ok": True,
            "primary_capability": None,
            "supporting_capabilities": [],
            "combination_rationale": "",
            "execution_order": [],
            "compounding_effect": "",
        }
    scores = {cap: _score_capability_from_keywords(msg, cap) for cap in CAPABILITIES.keys()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = ranked[0][0] if ranked and ranked[0][1] > 0 else None
    supporting = [cap for cap, sc in ranked[1:6] if sc > 0]

    # Add graph-adjacent supporters when certain primaries fire
    if primary == "osint":
        supporting = list(dict.fromkeys(supporting + ["network_graph", "threat_detection"]))
    if primary == "vision_forensic":
        supporting = list(dict.fromkeys(supporting + ["osint", "network_graph"]))
    if primary == "medical":
        supporting = list(dict.fromkeys(supporting + ["biomedical_research", "health_profile"]))
    if primary == "suit_design":
        supporting = list(dict.fromkeys(supporting + ["research_agent", "physics_sim", "chemistry", "cad_generation"]))

    # Build an execution order from a set using dependency cues (fed_by relationships).
    chosen = [primary] if primary else []
    chosen += supporting
    chosen = [c for c in chosen if c in CAPABILITIES]
    execution_order = _topologicalish_order(chosen)

    rationale = ""
    compounding = ""
    if primary == "osint" and "threat_detection" in supporting:
        rationale = "OSINT establishes facts; network mapping adds context; threat detection evaluates risk."
        compounding = "A person dossier + relationship context + threat posture, better than any one alone."
    elif primary == "medical" and "treatment_design" in supporting:
        rationale = "Medical evidence grounds the question; biomedical data refines mechanisms; chemistry checks feasibility; treatment design synthesizes."
        compounding = "Mechanism-aware treatment hypothesis with safety constraints and evidence framing."
    elif primary == "suit_design":
        rationale = "Research finds real programs/papers; physics constrains feasibility; chemistry constrains materials; suit design integrates; CAD makes it tangible."
        compounding = "An engineering roadmap that stays grounded in physics/material limits, plus concrete CAD artifacts."
    elif primary == "vision_forensic":
        rationale = "Forensics extracts signal from the image; OSINT + network graph can attribute entities and connect them."
        compounding = "Image-derived entities become actionable intelligence links."

    return {
        "ok": True,
        "primary_capability": primary,
        "supporting_capabilities": supporting,
        "combination_rationale": rationale,
        "execution_order": execution_order,
        "compounding_effect": compounding,
    }


def _topologicalish_order(caps: list[str]) -> list[str]:
    """
    Lightweight ordering: prefer upstream (fed_by empty or external) before downstream.
    Not a strict topo sort (graph can contain cycles), but stable enough for prompting.
    """
    unique = [c for i, c in enumerate(caps) if c and c in CAPABILITIES and c not in caps[:i]]
    # Score: lower means earlier (more upstream)
    def up_score(c: str) -> int:
        meta = CAPABILITIES.get(c) or {}
        fed_by = meta.get("fed_by") if isinstance(meta.get("fed_by"), list) else []
        # count only dependencies within chosen set
        inside = sum(1 for x in fed_by if _normalize_cap_id(x) in unique)
        return inside

    # iteratively pick nodes with minimal inside-deps
    remaining = unique[:]
    out: list[str] = []
    while remaining:
        remaining.sort(key=lambda c: (up_score(c), c))
        pick = remaining.pop(0)
        out.append(pick)
        # remove pick from others' dependency counts by recomputing (cheap at this size)
    return out


def find_optimal_combination(
    problem_description: str,
    *,
    anthropic_client: Any | None = None,
) -> dict[str, Any]:
    """
    Returns an ordered capability combination with rationale.

    - Fast heuristic path always runs.
    - If anthropic_client is provided, we ask Claude to refine ordering/rationale using ONLY the capability ids.
    """
    base = recognize_capability_combinations(problem_description)
    if not anthropic_client:
        return {"ok": True, "mode": "heuristic", **base}

    # Claude refinement (small, bounded)
    try:
        bundle = {
            "problem": (problem_description or "")[:1200],
            "primary": base.get("primary_capability"),
            "supporting": base.get("supporting_capabilities") or [],
            "order": base.get("execution_order") or [],
            "capabilities": {k: {"name": v.get("name"), "feeds_into": v.get("feeds_into"), "fed_by": v.get("fed_by")} for k, v in CAPABILITIES.items()},
        }
        sys = (
            "You are choosing an optimal sequence of Angel capabilities to solve a problem.\n"
            "Output ONE JSON with keys: primary_capability, supporting_capabilities, execution_order, rationale, compounding_effect.\n"
            "Keep rationale concise. Use only capability ids from the provided map."
        )
        user = json.dumps(bundle, ensure_ascii=False)[:8000]
        resp = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=350,
            temperature=0.2,
            system=sys,
            messages=[{"role": "user", "content": user}],
        )
        txt = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                txt += str(getattr(block, "text", "") or "")
        raw = (txt or "").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0)) if m else {}
        if isinstance(data, dict) and data.get("execution_order"):
            return {"ok": True, "mode": "claude_refined", "result": data, "base": base}
    except Exception:
        pass
    return {"ok": True, "mode": "heuristic_fallback", **base}


def suggest_capability_combinations(context: str) -> dict[str, Any]:
    """
    Proactive suggestion helper. Returns a suggested multi-capability action plan.
    This does NOT execute anything; it's for conversational guidance.
    """
    rec = recognize_capability_combinations(context)
    primary = rec.get("primary_capability")
    supporting = rec.get("supporting_capabilities") or []
    if not primary or not supporting:
        return {"ok": True, "suggest": None}
    # Keep suggestion short and optionally phrased as a "want me to do X+Y+Z?"
    chain = " → ".join(rec.get("execution_order") or [])
    sug = f"I can coordinate {primary} + {', '.join(supporting[:3])} (chain: {chain}). Want me to run the full chain?"
    return {"ok": True, "suggest": sug, "recognition": rec}


def explain_capability_compounding() -> dict[str, Any]:
    return {
        "ok": True,
        "primary_chain": ["web_search", "osint", "network_graph", "threat_detection", "predictions", "proactive_intel", "briefings"],
        "engineering_chain": ["research_agent", "physics_sim", "chemistry", "cad_generation", "visualization_3d"],
        "medical_chain": ["medical", "biomedical_research", "chemistry", "treatment_design"],
        "suit_chain": ["research_agent", "chemistry", "physics_sim", "suit_design", "cad_generation"],
        "cross_chain_connections": [
            "vision_forensic → osint → network_graph",
            "translation → osint → proactive_intel",
            "health_profile → medical → treatment_design",
        ],
    }


def build_capability_context_for_prompt(user_message: str) -> str:
    """
    Returns a compact capability context block for the system prompt.
    Keep this under ~150 tokens.
    """
    rec = recognize_capability_combinations(user_message)
    primary = rec.get("primary_capability")
    supporting = rec.get("supporting_capabilities") or []
    order = rec.get("execution_order") or []
    if not primary:
        return ""
    sup = ", ".join(supporting[:4]) if supporting else ""
    chain = " → ".join(order[:7]) if order else primary
    # Proactive suggestion only when clearly multi-capability
    ps = ""
    if supporting:
        ps = f"Suggestion: coordinate {primary} + {', '.join(supporting[:2])}."
    block = (
        "[Capability Graph — active systems for this turn]\n"
        f"Primary: {primary}\n"
        + (f"Supporting: {sup}\n" if sup else "")
        + f"Chain: {chain}\n"
        + (f"{ps}\n" if ps else "")
    )
    # Hard cap for safety
    return block[:900].rstrip()

