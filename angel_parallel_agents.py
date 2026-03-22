"""
Multi-agent parallel coordination for Angel — specialized agents + coordinator synthesis.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

# Populated for GET /api/agents/status/<task_id>
_REGISTRY_LOCK = threading.Lock()
TASK_REGISTRY: dict[str, "AgentTask"] = {}

AGENT_ROLES = frozenset(
    {
        "osint_researcher",
        "threat_analyst",
        "network_mapper",
        "historian",
        "translator",
        "pattern_analyst",
        "surveillance_monitor",
        "biological_analyst",
        "general",
    }
)

MAX_PARALLEL_AGENTS = 5
MAX_TAVILY_PER_AGENT = 3
AGENT_MODEL = "claude-haiku-4-5"

# Coordinator: Haiku for standard (speed); Sonnet snapshot for deep runs.
COORDINATOR_MODEL_STANDARD = "claude-haiku-4-5-20251001"
COORDINATOR_MODEL_DEEP = "claude-sonnet-4-20250514"

# Per-agent output caps (by depth); keeps synthesis input small.
AGENT_WORD_LIMIT_STANDARD = 400
AGENT_WORD_LIMIT_DEEP = 600
# ~chars per word for max_tokens hint to Haiku (call_claude)
_AGENT_MAX_TOKENS_STANDARD = 750
_AGENT_MAX_TOKENS_DEEP = 1100

# Shared memory injected into all agents + coordinator (~2000 tokens max; no per-agent Mem0).
MAX_MEMORY_CONTEXT_TOKENS = 2000
_CHARS_PER_TOKEN_EST = 4
MAX_MEMORY_CONTEXT_CHARS = MAX_MEMORY_CONTEXT_TOKENS * _CHARS_PER_TOKEN_EST


def _cap_memory_context(text: str) -> str:
    """Truncate to a concise shared context (~MAX_MEMORY_CONTEXT_TOKENS tokens, char-estimate)."""
    t = (text or "").strip()
    if len(t) <= MAX_MEMORY_CONTEXT_CHARS:
        return t
    return t[: MAX_MEMORY_CONTEXT_CHARS - 1].rstrip() + "…"


def prepare_shared_memory_context(memory_summary: str | None) -> str:
    """Cap a pre-fetched memory blob for parallel runs (call once per turn in generate_reply)."""
    return _cap_memory_context(memory_summary or "")


def _normalize_depth(depth: str | None) -> str:
    d = (depth or "standard").strip().lower()
    return "deep" if d == "deep" else "standard"


def _agent_word_limit_for_depth(depth: str) -> int:
    return AGENT_WORD_LIMIT_DEEP if depth == "deep" else AGENT_WORD_LIMIT_STANDARD


def _truncate_to_word_count(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).rstrip() + "…"


def _coordinator_model_for_depth(depth: str) -> str:
    return COORDINATOR_MODEL_DEEP if depth == "deep" else COORDINATOR_MODEL_STANDARD


def _ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class AgentTask:
    task_id: str
    agent_role: str
    instruction: str
    context: str = ""
    status: str = "pending"  # pending | running | complete | failed
    result: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _register_task(t: AgentTask) -> None:
    with _REGISTRY_LOCK:
        TASK_REGISTRY[t.task_id] = t


def _update_task(t: AgentTask, **kwargs: Any) -> None:
    with _REGISTRY_LOCK:
        for k, v in kwargs.items():
            setattr(t, k, v)
        TASK_REGISTRY[t.task_id] = t


def get_task_status(task_id: str) -> dict[str, Any] | None:
    with _REGISTRY_LOCK:
        t = TASK_REGISTRY.get(task_id)
        if t is None:
            return None
        return t.to_dict()


def _role_system_prompt(role: str) -> str:
    specs = {
        "osint_researcher": (
            "You are an OSINT researcher. Focus on public records, news, and credible open sources. "
            "Extract key facts, red flags, and cite themes (no fabricated URLs)."
        ),
        "threat_analyst": (
            "You are a threat intelligence analyst. Assess risk level (LOW/MEDIUM/HIGH), implications, "
            "and practical recommendations from open-source signals only."
        ),
        "network_mapper": (
            "You are a network analyst. Describe likely entities, relationships, clusters, and paths of interest "
            "from open information — not classified data."
        ),
        "historian": (
            "You are an intelligence historian. Provide historical context, precedent, and timeline-relevant notes "
            "from public knowledge."
        ),
        "translator": (
            "You are a translator / foreign-open-source analyst. Note language angles, international angles, "
            "and mission-relevant nuance from public material."
        ),
        "pattern_analyst": (
            "You are a pattern analyst. Identify recurring themes, anomalies, correlations, and confidence levels."
        ),
        "surveillance_monitor": (
            "You are a surveillance-intel synthesizer (legal OSINT only). Summarize active signals and correlated events."
        ),
        "biological_analyst": (
            "You are a biological/medical intelligence analyst (open sources). Assess patterns and uncertainties; "
            "no clinical diagnosis."
        ),
        "general": "You are a general intelligence analyst assisting Angel for Tyler's mission.",
    }
    return specs.get(role, specs["general"])


def _tavily_for_agent(
    api_key: str,
    topic: str,
    role: str,
    instruction: str,
) -> str:
    from angel import _tavily_search_one

    # Up to 3 fixed-angle queries (no extra Claude call for query gen — keeps cost down)
    qbase = (topic or "").strip() or instruction[:200]
    queries = [
        f"{qbase} latest news context",
        f"{qbase} background facts analysis",
        f"{qbase} implications risk",
    ][:MAX_TAVILY_PER_AGENT]
    chunks: list[str] = []
    for q in queries:
        rows = _tavily_search_one(q, api_key, max_results=3, search_depth="basic")
        for r in rows[:3]:
            title = (r.get("title") or "")[:200]
            content = (r.get("content") or "")[:600]
            url = (r.get("url") or "")[:200]
            chunks.append(f"- {title}\n  {content}\n  {url}")
    return "\n".join(chunks) if chunks else "(No Tavily results for this angle.)"


def _run_one_agent(
    task: AgentTask,
    *,
    shared_context: str,
    memory_context: str,
    anthropic_client: Any,
    api_key: str | None,
    depth: str = "standard",
) -> AgentTask:
    """Run one specialist agent: Tavily + Haiku only (no Mem0; memory_context is pre-fetched)."""
    from angel import call_claude

    d = _normalize_depth(depth)
    word_limit = _agent_word_limit_for_depth(d)
    agent_max_tokens = (
        _AGENT_MAX_TOKENS_DEEP if d == "deep" else _AGENT_MAX_TOKENS_STANDARD
    )
    density = (
        f"Keep your response under {word_limit} words. Be dense and specific. "
        "No preamble. Lead with the most important finding."
    )

    _update_task(task, status="running", started_at=_ts(), error=None)
    try:
        web_blob = ""
        if api_key:
            web_blob = _tavily_for_agent(
                api_key,
                shared_context[:500],
                task.agent_role,
                task.instruction,
            )
        system = (
            _role_system_prompt(task.agent_role)
            + "\n\n"
            + density
        )
        user = (
            f"Shared mission context (Tyler):\n{shared_context[:4000]}\n\n"
            f"Task-specific context:\n{task.context[:3000]}\n\n"
            f"Your instruction:\n{task.instruction}\n\n"
            f"Relevant memory context (shared, pre-loaded for this run):\n{memory_context}\n\n"
            f"Open-web snippets (Tavily):\n{web_blob[:8000]}\n\n"
            "Respond in plain text with clear sections: Summary, Key findings, "
            "Sources/themes (no fake URLs), Notes/uncertainty."
        )
        out = call_claude(
            anthropic_client,
            system,
            user,
            model=AGENT_MODEL,
            prior_turns=None,
            max_tokens=agent_max_tokens,
        )
        out = _truncate_to_word_count((out or "").strip(), word_limit)
        _update_task(
            task,
            status="complete",
            result=out,
            completed_at=_ts(),
        )
    except Exception as e:
        _update_task(
            task,
            status="failed",
            error=str(e),
            result="",
            completed_at=_ts(),
        )
    return task


def _synthesize(
    anthropic_client: Any,
    *,
    topic: str,
    shared_context: str,
    tasks: list[AgentTask],
    memory_context: str,
    agent_phase_sec: float,
    estimated_sequential_sec: float,
    depth: str = "standard",
) -> str:
    from angel import call_claude

    d = _normalize_depth(depth)
    coord_model = _coordinator_model_for_depth(d)

    blocks = []
    for t in tasks:
        blocks.append(
            f"### Agent: {t.agent_role} (id={t.task_id})\n"
            f"Status: {t.status}\n"
            f"{t.result or t.error or ''}\n"
        )
    n = len(tasks)
    roles_csv = ", ".join(t.agent_role for t in tasks) if tasks else "(none)"
    system = (
        "You are Angel coordinating parallel specialist agents for Tyler. "
        "Merge their outputs into one coherent intelligence report. "
        "Resolve contradictions cautiously; note corroboration; flag highest-priority findings. "
        "Use clear headings; be concise but substantive. Open sources only.\n\n"
        "Begin your response by briefly noting that you ran parallel analysis, how many agents, "
        "the specialist roles involved, the wall-clock time for the parallel agent phase "
        f"(~{agent_phase_sec:.1f}s), and how that compares to estimated sequential processing "
        f"(~{estimated_sequential_sec:.0f}s for {n} agents). "
        "Then provide the full synthesis."
    )
    user = (
        f"Topic / focus: {topic}\n\n"
        f"Coordination metadata: {n} agents ({roles_csv}); parallel agent phase ~{agent_phase_sec:.1f}s; "
        f"estimated sequential wall time ~{estimated_sequential_sec:.0f}s.\n\n"
        f"Shared context:\n{shared_context[:4000]}\n\n"
        f"Memory context (shared, pre-loaded; no additional memory retrieval):\n{memory_context}\n\n"
        "Agent outputs:\n"
        + "\n".join(blocks)
    )
    return call_claude(
        anthropic_client,
        system,
        user,
        model=coord_model,
        prior_turns=None,
    ).strip()


def run_parallel_agents(
    tasks: list[AgentTask | dict[str, Any]],
    shared_context: str,
    *,
    anthropic_client: Any,
    user_id: str,
    memory_summary: str | None = None,
    memory_client: Any | None = None,
    use_mem0_cloud: bool = False,
    depth: str = "standard",
) -> dict[str, Any]:
    """
    Run up to ``MAX_PARALLEL_AGENTS`` agents in parallel; coordinator synthesizes (Haiku if
    ``depth`` is standard, Sonnet snapshot if ``depth`` is ``deep``).

    Memory is resolved **once** before any agent thread runs:
    - If ``memory_summary`` is non-empty, it is capped and used (no Mem0 call).
    - Else if ``memory_client`` is set, ``fetch_combined_memories`` runs once and a summary is built.
    - Otherwise agents run with empty memory context.

    Agents receive only this shared string — they never call Mem0.

    Returns ``results``, ``synthesis``, ``agents_used``, ``total_time``, ``task_ids``.
    """
    depth_norm = _normalize_depth(depth)
    t_total_start = time.perf_counter()
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip() or None

    norm: list[AgentTask] = []
    for raw in tasks[:MAX_PARALLEL_AGENTS]:
        if isinstance(raw, AgentTask):
            norm.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        tid = (raw.get("task_id") or "").strip() or secrets.token_hex(8)
        role = (raw.get("agent_role") or "general").strip()
        if role not in AGENT_ROLES:
            role = "general"
        norm.append(
            AgentTask(
                task_id=tid,
                agent_role=role,
                instruction=(raw.get("instruction") or "").strip() or "Analyze the topic.",
                context=(raw.get("context") or "").strip(),
            )
        )

    for t in norm:
        t.status = "pending"
        _register_task(t)

    sc = (shared_context or "").strip() or f"User id: {user_id}"

    if not norm:
        return {
            "ok": False,
            "error": "no tasks",
            "results": [],
            "synthesis": "",
            "agents_used": 0,
            "total_time": 0.0,
            "task_ids": [],
        }

    t_mem0 = time.perf_counter()
    mem_in = (memory_summary or "").strip()
    if mem_in:
        memory_context = _cap_memory_context(mem_in)
        dt_mem = time.perf_counter() - t_mem0
        print(
            f"[parallel] shared memory context ready in {dt_mem:.2f}s "
            f"(pre-built, no Mem0 fetch)",
            flush=True,
        )
    elif memory_client is not None:
        from angel import build_memory_summary_with_sections, fetch_combined_memories

        merged = fetch_combined_memories(memory_client, user_id, use_mem0_cloud)
        memory_context = _cap_memory_context(
            build_memory_summary_with_sections(merged, None)
        )
        dt_mem = time.perf_counter() - t_mem0
        print(f"[parallel] memory fetch complete in {dt_mem:.2f}s", flush=True)
    else:
        memory_context = ""
        dt_mem = time.perf_counter() - t_mem0
        print(
            f"[parallel] no memory context ({dt_mem:.2f}s); agents use Tavily + task context only",
            flush=True,
        )

    print("[parallel] agents starting with shared context", flush=True)
    t_pool_start = time.perf_counter()
    ordered: list[AgentTask] = []
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_AGENTS, len(norm))) as ex:
        future_to_index: dict[Any, int] = {}
        for i, t in enumerate(norm):
            fut = ex.submit(
                _run_one_agent,
                t,
                shared_context=sc,
                memory_context=memory_context,
                anthropic_client=anthropic_client,
                api_key=api_key,
                depth=depth_norm,
            )
            future_to_index[fut] = i
        slot: list[AgentTask | None] = [None] * len(norm)
        for fut in as_completed(future_to_index):
            idx = future_to_index[fut]
            try:
                slot[idx] = fut.result()
            except Exception as e:
                tt = norm[idx]
                _update_task(tt, status="failed", error=str(e), completed_at=_ts())
                slot[idx] = tt
        ordered = [s for s in slot if s is not None]

    agent_phase_sec = time.perf_counter() - t_pool_start
    print(f"[parallel] all agents complete in {agent_phase_sec:.2f}s", flush=True)

    seq_est = float(len(ordered) * 48)
    topic = sc[:500]
    t_synth_start = time.perf_counter()
    synthesis = _synthesize(
        anthropic_client,
        topic=topic,
        shared_context=sc,
        tasks=ordered,
        memory_context=memory_context,
        agent_phase_sec=agent_phase_sec,
        estimated_sequential_sec=seq_est,
        depth=depth_norm,
    )
    synth_elapsed = time.perf_counter() - t_synth_start
    print(f"[parallel] synthesis complete in {synth_elapsed:.2f}s", flush=True)

    elapsed = time.perf_counter() - t_total_start
    print(f"[parallel] total time {elapsed:.2f}s", flush=True)
    return {
        "ok": True,
        "results": [t.to_dict() for t in ordered],
        "synthesis": synthesis,
        "agents_used": len(ordered),
        "total_time": elapsed,
        "task_ids": [t.task_id for t in ordered],
        "depth": depth_norm,
        "coordinator_model": _coordinator_model_for_depth(depth_norm),
        "estimated_sequential_time_sec": seq_est,
        "time_saved_note": (
            f"Parallel run ~{elapsed:.1f}s vs ~{seq_est:.0f}s estimated sequential wall time "
            f"({len(ordered)} agents)."
        ),
    }


def detect_parallel_opportunity(user_message: str) -> tuple[bool, str | None]:
    """
    True if multi-agent parallel research is appropriate.
    Returns (True, topic_hint) or (False, None).
    """
    raw = (user_message or "").strip()
    if len(raw) < 12:
        return False, None
    low = raw.lower()

    phrase_topic = None
    patterns = [
        r"full\s+briefing\s+on\s+(.+)",
        r"deep\s+dive\s+(?:on|into)\s+(.+)",
        r"everything\s+you\s+know\s+about\s+(.+)",
        r"comprehensive\s+analysis\s+of\s+(.+)",
        r"research\s+(.+?)\s+thoroughly",
        r"thoroughly\s+research\s+(.+)",
        r"analyze\s+(.+?)\s+comprehensively",
    ]
    for pat in patterns:
        m = re.search(pat, low, re.I | re.DOTALL)
        if m:
            phrase_topic = m.group(1).strip()
            break

    multi_topic = False
    if raw.count(",") >= 2 or re.search(
        r"\b(and|vs\.?|versus)\b.+\b(and|vs\.?|versus)\b", low
    ):
        multi_topic = True

    keyword_hit = any(
        k in low
        for k in (
            "full briefing",
            "deep dive",
            "everything you know",
            "comprehensive analysis",
            "research thoroughly",
            "thoroughly research",
            "parallel",
            "all angles",
        )
    )

    if phrase_topic:
        if len(phrase_topic) > 400:
            phrase_topic = phrase_topic[:400] + "…"
        return True, phrase_topic

    if multi_topic and len(raw) > 40:
        return True, raw[:500]

    if keyword_hit and len(raw) > 25:
        return True, raw[:500]

    return False, None


def decompose_into_parallel_tasks(
    topic: str,
    user_message: str,
    *,
    depth: str = "standard",
) -> list[AgentTask]:
    """Build 3–5 complementary AgentTask rows."""
    base = (topic or user_message or "").strip()[:2000]
    um = (user_message or "").strip()[:1500]
    n = 5 if (depth or "").lower() == "deep" else 4

    specs: list[tuple[str, str, str]] = [
        (
            "osint_researcher",
            f"Open-source background on: {base}",
            "Emphasize key facts, red flags, and source themes.",
        ),
        (
            "threat_analyst",
            f"Threat and risk framing for: {base}",
            "Assess risk level and implications for Tyler's mission (open sources).",
        ),
        (
            "network_mapper",
            f"Entity and relationship map around: {base}",
            "Describe plausible connections, clusters, and gaps.",
        ),
        (
            "historian",
            f"Historical precedent and timeline context for: {base}",
            "Cite public historical patterns relevant to the topic.",
        ),
        (
            "pattern_analyst",
            f"Cross-cutting patterns and anomalies for: {base}",
            "Note correlations and confidence.",
        ),
    ]
    if n <= 4:
        specs = specs[:4]

    out: list[AgentTask] = []
    for role, instr, ctx in specs[:n]:
        out.append(
            AgentTask(
                task_id=secrets.token_hex(8),
                agent_role=role,
                instruction=instr,
                context=f"{ctx}\n\nUser message excerpt:\n{um}",
            )
        )
    return out


def run_research_decomposed(
    topic: str,
    *,
    depth: str,
    anthropic_client: Any,
    user_id: str,
    user_message: str = "",
    memory_summary: str | None = None,
    memory_client: Any | None = None,
    use_mem0_cloud: bool = False,
) -> dict[str, Any]:
    tasks = decompose_into_parallel_tasks(topic, user_message or topic, depth=depth)
    return run_parallel_agents(
        tasks,
        topic,
        anthropic_client=anthropic_client,
        memory_summary=memory_summary,
        memory_client=memory_client,
        use_mem0_cloud=use_mem0_cloud,
        user_id=user_id,
        depth=depth,
    )
