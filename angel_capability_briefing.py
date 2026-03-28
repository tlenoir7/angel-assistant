"""
Living capability briefing: filed to the Intelligence File Cabinet and referenced from the system prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

ANGEL_CAPABILITY_BRIEFING = """# ANGEL CAPABILITY BRIEFING — OPERATIONAL REFERENCE
Last updated: {date}
This document is Angel's authoritative self-reference
for all capabilities, limitations, and optimal usage.

## INTELLIGENCE SYSTEMS

### OSINT Deep Background (/api/osint/background)
- Trigger: "run background on [person/org]", "who is [name]", "research [target]"
- What it does: Systematic open source intelligence — searches news, social media,
  public records, professional background across multiple sources simultaneously
- Best use: Always run OSINT before threat assessment — it feeds the network graph
- Limitation: Only public information, no classified sources
- Auto-files to: OSINT Dossiers folder
- Chain: OSINT → Network Graph → Threat Detection → Predictions

### Relationship Network Mapping (/api/network/)
- Trigger: "how is [A] connected to [B]", "show me the network", "who knows [name]"
- What it does: Maps connections between people and organizations, builds a visual
  intelligence graph, tracks relationship strength and evidence
- Best use: After OSINT runs, nodes are added automatically. Ask "show me the network"
  to see the full picture
- Key people currently in network: Grusch, Elizondo, Mellon, Coulthart, Rubio
- Auto-updates: Every OSINT run adds new nodes and edges

### Threat Detection (/api/threats/scan)
- Trigger: "threat assessment", "is [person] a threat", "scan for threats"
- What it does: Monitors 22 threat categories, searches for risks to Tyler and mission
- Best use: Run after OSINT for full context. Scheduled automatically every 6 hours
- Auto-files to: Threat Intelligence folder
- Note: Does NOT treat Tyler's own requests as threats (trusted operator)

### Predictive Modeling (/api/predictions/generate)
- Trigger: "what's likely to happen with [topic]", "predict", "forecast"
- What it does: Generates evidence-based forecasts from current intelligence
- Best use: After threat detection and OSINT have run — predictions are more accurate
  with more intelligence context
- Auto-files to: Predictions folder

### Proactive Background Intelligence (/api/proactive/run)
- Trigger: "watch [topic/person]", "monitor [subject]", "track [name]"
- What it does: Angel monitors topics autonomously without being asked, runs every 4 hours
- Currently watching: 14+ high-priority targets including Grusch, UAP legislation,
  room-temperature superconductors, FBI developments
- Auto-files to: Proactive Intelligence folder

## RESEARCH & ENGINEERING SYSTEMS

### Theoretical Research Agent (/api/research/query)
- Trigger: "research [topic]", "what does science say about", "papers on [subject]"
- What it does: Parallel searches across ArXiv, NASA NTRS, DARPA programs, USPTO patents
- Best use: Ask for research before physics simulation — research informs parameters
- Returns: TRL estimate, key papers, gaps in current knowledge
- Auto-files to: Research Intelligence folder

### Physics Simulation Engine (/api/physics/simulate)
- Trigger: "simulate [scenario]", "would it work if", "calculate [physics problem]"
- Domains: Propulsion, orbital mechanics, EM fields, structural analysis, energy systems,
  theoretical/exotic physics (Alcubierre, Casimir, inertial reduction)
- Best use: Provide specific numbers — "10 ton craft, 500kN thrust, 30km altitude"
- Returns: Feasibility rating (FEASIBLE/MARGINAL/INFEASIBLE/THEORETICAL), limiting factors

### Chemical and Materials Synthesis (/api/chemistry/)
- Trigger: "properties of [material]", "synthesis route for [compound]", "what material for [use]"
- Databases: PubChem (100M+ compounds), NIST WebBook, Materials Project (needs API key)
- Best use: Combine with physics simulation for complete engineering analysis
- Returns: Precise scientific data, synthesis routes, safety profiles

### CAD Generation (/api/cad/from-brief)
- Trigger: "generate [shape]", "design [component]", "create CAD for [description]"
- Shapes available: lenticular, disc, fuselage, airfoil, nozzle, box, cylinder, sphere,
  cone, torus, pressure vessel
- IMPORTANT LIMITATION: CAD files are stored in /tmp on Railway — they are DELETED
  on every redeploy. Always download STEP/STL files immediately after generation.
- Best use: Download files right away, don't wait
- View in 3D: Tap the 🔺 View in 3D button that appears after generation

### 3D Visualization (iPhone App)
- Trigger: Tap 🔺 View in 3D button after CAD generation
- Controls: One finger to rotate, two fingers to zoom and pan
- Limitation: Model must still exist on server — if server redeployed, regenerate first
- Note: The button appears automatically after CAD generation is detected

## MEDICAL INTELLIGENCE SYSTEMS

### Medical Intelligence Core (/api/medical/condition)
- Trigger: Medical conditions, drug names, symptoms, treatments, clinical trials
- Databases: PubMed, FDA, MedlinePlus, ClinicalTrials.gov (live data)
- Best use: Ask about specific conditions for live research, not general health advice
- Returns: Evidence quality ratings, active clinical trials with real NCT IDs

### Biomedical Research Agent (/api/medical/biomedical-research)
- Trigger: Gene names (BRCA1, TP53), protein names, molecular pathways, genomics
- Databases: UniProt, NCBI Gene, KEGG, RCSB PDB, ClinVar
- Best use: For molecular-level research on specific genes or proteins

### Theoretical Treatment Design (/api/medical/design-treatment)
- Trigger: "design a treatment for", "what would work for [condition]", "theoretical approach to"
- What it does: Combines known mechanisms to propose novel treatment strategies
- ALWAYS labeled THEORETICAL — not medical advice, requires clinical validation
- Best use: Research tool for understanding treatment landscape

### UAP Medical Effects (/api/medical/uap-medical)
- Trigger: "UAP medical effects", "witness symptoms", "radiation exposure from UAP"
- What it does: Research on documented medical effects from UAP encounters
- Key cases in database: Cash-Landrum 1980, Colares 1977, Rendlesham 1980
- Auto-files to: UAP Medical Intelligence folder

### Personal Health Intelligence (/api/health/profile)
- Tyler's current profile: Age 27, weight 75kg, runs 5x/week, sleep 6-8 hours
- Passive monitoring: Angel automatically extracts health data from conversations
- When to update: Just mention health information naturally in conversation
- Privacy: Health data NEVER appears in briefings or shared externally

## SUIT DESIGN SYSTEMS

### Theoretical Suit Design (/api/ironman/assess)
- Trigger: "Iron Man", "Batman Beyond", "suit design", "powered armor", "exosuit"
- Two design philosophies:
  * Iron Man: Maximum power, flight, weaponization — arc reactor, repulsors, Mach 3+
  * Batman Beyond: Stealth, agility, augmentation — skin-tight, silent, AI-directed
- Best use: Ask about specific domains (power, propulsion, materials, stealth, sensors)
- Returns: TRL ratings, gap analysis, research vectors with lead organizations
- Batman Beyond is significantly more achievable near-term than Iron Man

## COMPUTER VISION SYSTEMS

### Describe Mode (/api/vision)
- Trigger: Camera icon → Describe mode → take photo → Analyze
- What it does: General description of what's in the image with mission context
- Best use: Quick visual check, environmental awareness

### Forensic Vision (/api/vision/forensic)
- Trigger: Camera icon → toggle to Forensic → take photo → Analyze
- What it does: Structured forensic analysis — classifies image type, runs appropriate
  pipeline (document, scene, person, object, media/authenticity)
- Processing time: 2-5 minutes for full analysis — be patient
- IMPORTANT: Images are compressed to under 4MB before sending
- Auto-files: HIGH/CRITICAL mission relevance → Visual Intelligence folder
- Best use: Document authenticity, scene analysis, UAP photo verification

## VOICE SYSTEMS

### Standard Voice Mode
- How it works: Hold button to speak, release to send, Angel responds with voice
- Best for: Precise, deliberate communication

### Realtime Voice Mode (GPT-4o)
- How it works: Always-on listening — just speak naturally, no button holding
- Server VAD handles turn detection automatically
- Known issue: Audio occasionally cuts out — switch to text and back to reset
- Best for: Natural conversation, hands-free operation
- Note: Responses also appear in text chat for reference

## INTELLIGENCE FILE CABINET

### Filing System
- All intelligence auto-files to organized folders
- Key folders: OSINT Dossiers, Threat Intelligence, Predictions, Proactive Intelligence,
  Research Intelligence, Chemistry Intelligence, Medical Intelligence, UAP Medical
  Intelligence, Visual Intelligence, Foreign Intelligence, Engineering Designs,
  Iron Man Engineering, Batman Beyond Engineering, Physics Simulations,
  Theoretical Medicine, Biological Intelligence, Historical Archives,
  Network Intelligence, Surveillance Intelligence

### Accessing Files
- "What have you filed?" → summary of all intelligence folders
- "Show me [folder name]" → contents of specific folder
- "Search for [topic]" → find relevant intelligence files

## MEMORY SYSTEM

### How Memory Works
- Local storage on Railway persistent volume (/app/data/tyler_memories.json)
- Currently storing 6,700+ memories and growing
- Mem0 cloud DISABLED — fully local, no API limits
- Memory persists across Railway redeploys via persistent volume
- Auto-saves: Every conversation turn adds new memories

### Memory Categories
- personal_health: Tyler's health profile (private, never in briefings)
- suit_targets: Iron Man and Batman Beyond engineering targets
- chemistry_cache: Chemical database lookup cache (7-day TTL)
- research_cache: Research database cache (7-day TTL)
- medical_cache: Medical database cache (7-day TTL)
- osint_dossier: People and organization profiles
- proactive_watch: Active monitoring targets

## OPTIMAL USAGE PATTERNS

### Full Intelligence Profile on a Person
1. "Angel, run OSINT on [name]" → builds dossier
2. "Add [name] to the network graph" → maps connections
3. "Run threat assessment on [name]" → evaluates risk
4. "What predictions do you have about [name]?" → forecasts

### Engineering Research Workflow
1. "Research [technology]" → Research Agent pulls papers
2. "Simulate [specific scenario with numbers]" → Physics Engine
3. "What material would work for [application]?" → Chemistry
4. "Generate CAD for [design]" → CAD Generation
5. Tap 🔺 View in 3D → visualize the model

### UAP Investigation Workflow
1. "Research [UAP topic]" → Research Agent
2. "Run OSINT on [witness/official]" → Background
3. "Add to network graph" → Connections
4. "What are the medical implications?" → UAP Medical
5. "What predictions do you have?" → Forecasts

## KNOWN LIMITATIONS

1. CAD files delete on Railway redeploy — download immediately
2. Realtime voice audio occasionally cuts out — switch modes to reset
3. Forensic vision takes 2-5 minutes — be patient
4. File reading works with: PDF, TXT, DOCX, CSV, XLSX, images
5. Physics simulation theoretical domain (Alcubierre etc.) is clearly labeled THEORETICAL
6. Medical intelligence is research only — not medical advice
7. OSINT is public information only — no classified sources
8. Memory writes happen after each turn — very recent memories may not
   be available in the same conversation
9. Proactive intelligence runs every 4 hours — not real-time
10. Morning briefings currently DISABLED (DISABLE_BRIEFING=true in Railway)

## SELF-MODIFICATION

### Stage 6 Self Modification
- Angel can propose permanent behavioral changes based on observations
- All modifications require Tyler's explicit approval before applying
- To trigger: "Angel, propose a self-modification for [behavior]"
- To review: "Angel, show me proposed modifications"
- To approve: "Approve modification [ID]"
- Current approved modifications: Communication style (short, conversational,
  plain language, offer to go deeper)
"""

BRIEFING_FILE_NAME = "Angel-Capability-Briefing"
BRIEFING_FOLDER = "System Intelligence"


def render_capability_briefing_text() -> str:
    """Full briefing body with today's UTC date."""
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ANGEL_CAPABILITY_BRIEFING.replace("{date}", d)


def get_capability_briefing_summary() -> str:
    """Short block for system prompt injection."""
    return """
[SELF-KNOWLEDGE: Full capability briefing available in
Intelligence File Cabinet → System Intelligence →
Angel-Capability-Briefing. Reference it when Tyler asks
about your capabilities, limitations, or how to best
use your systems.]
"""


def file_capability_briefing(angel_instance: Any) -> str:
    """
    File the capability briefing to the Intelligence File Cabinet.
    Returns the briefing text (whether or not a new file was created).
    """
    content = render_capability_briefing_text()
    fc = getattr(angel_instance, "files_cabinet", None) if angel_instance else None
    if fc is None:
        return content
    try:
        fc.create_file(
            BRIEFING_FOLDER,
            BRIEFING_FILE_NAME,
            content,
            tags=[
                "capability_briefing",
                "self_knowledge",
                "operational_reference",
                "system",
            ],
        )
    except ValueError:
        # Already exists (unique name constraint)
        pass
    except Exception:
        pass
    return content


def ensure_capability_briefing_filed(angel_instance: Any) -> None:
    """On startup: create the cabinet file once if missing."""
    if not angel_instance or not getattr(angel_instance, "files_cabinet", None):
        return
    fc = angel_instance.files_cabinet
    try:
        if fc.get_file(BRIEFING_FILE_NAME):
            return
    except Exception:
        return
    file_capability_briefing(angel_instance)
