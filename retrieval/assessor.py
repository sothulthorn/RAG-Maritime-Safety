"""Safety Risk Assessor — analyzes a vessel profile against accident investigation reports.

This is NOT Q&A. It takes structured input about a vessel's situation,
retrieves relevant accident patterns from the database, and produces
a risk assessment with evidence-backed recommendations.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE
from retrieval.retriever import retrieve
from retrieval.rag_chain import _format_context, _extract_sources


RISK_QUERIES_BY_VESSEL = {
    "fishing": [
        "fishing vessel capsize stability",
        "fishing vessel man overboard fatality",
        "single handed fishing vessel accident",
        "fishing vessel flooding sinking",
        "fishing vessel safety recommendations",
    ],
    "cargo": [
        "cargo vessel collision",
        "cargo vessel grounding",
        "cargo vessel fire engine room",
        "cargo vessel structural failure",
        "cargo vessel safety recommendations",
    ],
    "tanker": [
        "tanker explosion fire",
        "tanker collision grounding pollution",
        "tanker cargo operations accident",
        "tanker mooring accident",
        "tanker safety recommendations",
    ],
    "passenger": [
        "passenger vessel fire safety",
        "passenger vessel collision",
        "passenger vessel evacuation",
        "passenger vessel propulsion failure",
        "passenger vessel safety recommendations",
    ],
    "tug": [
        "tug capsize girting",
        "tug towing accident",
        "tug stability failure",
        "tug crew safety",
        "tug safety recommendations",
    ],
    "recreational": [
        "recreational vessel capsize",
        "recreational vessel man overboard",
        "recreational vessel collision",
        "recreational vessel carbon monoxide",
        "recreational vessel safety recommendations",
    ],
}

ASSESSMENT_PROMPT = """You are a maritime safety risk assessor. Based on the vessel profile provided and the evidence from real accident investigation reports in the context, produce a structured risk assessment.

IMPORTANT: You must ground every finding in the actual accident reports provided in the context. Cite specific vessel names, incident details, and source documents. Do NOT make generic safety recommendations — only recommend actions that are directly supported by patterns found in the accident evidence.

VESSEL PROFILE:
{profile}

EVIDENCE FROM ACCIDENT INVESTIGATION REPORTS:
{context}

Produce a risk assessment in this EXACT structure:

## RISK ASSESSMENT

**Overall Risk Level:** [CRITICAL / HIGH / MEDIUM / LOW]
(Based on how many fatal incidents in the evidence match this vessel's profile)

## CRITICAL FINDINGS

For each risk factor identified from the vessel profile, provide:
- The risk factor from the profile
- Severity: [CRITICAL / HIGH / MEDIUM / LOW]
- Evidence: Which specific accidents from the context match this risk factor (vessel name, what happened, outcome)
- Source: The specific report filename and page
- Recommended Action: What to do, based on the investigation recommendations

## MATCHING PAST INCIDENTS

List the specific accidents from the context that most closely match this vessel's profile. For each:
- Vessel name and type
- What happened
- Outcome (fatalities/injuries)
- What went wrong
- Source document

## PRIORITY ACTIONS

Numbered list of actions ranked by urgency, each citing the evidence that supports it.

Remember: ONLY cite information that actually appears in the context. Do not invent incidents or statistics."""


def _build_profile_text(profile: dict) -> str:
    """Convert a vessel profile dict to readable text."""
    lines = []
    field_labels = {
        "vessel_type": "Vessel Type",
        "vessel_length": "Length",
        "operation": "Operation Type",
        "crew_size": "Crew Size",
        "operating_area": "Operating Area",
        "season": "Season/Weather",
        "stability_booklet": "Stability Booklet",
        "pfd_policy": "PFD/Lifejacket Policy",
        "epirb": "EPIRB/PLB",
        "mob_plan": "Man Overboard Recovery Plan",
        "fire_detection": "Fire Detection System",
        "weather_routing": "Weather Routing",
        "maintenance": "Maintenance Status",
        "safety_training": "Safety Training",
        "communication": "Communication Equipment",
        "additional_info": "Additional Information",
    }
    for key, label in field_labels.items():
        val = profile.get(key, "")
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "No profile data provided."


def assess_risk(profile: dict, source_filter: str | None = None) -> dict:
    """Run a safety risk assessment for a vessel profile.

    Args:
        profile: Dict with vessel characteristics (vessel_type, operation, etc.)
        source_filter: Optional organization filter.

    Returns:
        dict with "assessment", "sources", "profile_text"
    """
    vessel_type = profile.get("vessel_type", "").lower()
    profile_text = _build_profile_text(profile)

    # Build ONE combined query from vessel type + key risk factors
    risk_parts = [f"{vessel_type} vessel accident"]

    operation = profile.get("operation", "").lower()
    if "single" in operation:
        risk_parts.append("single handed fatality")
    if profile.get("stability_booklet", "").lower() in ("no", "none", "not available"):
        risk_parts.append("capsize stability")
    if profile.get("pfd_policy", "").lower() in ("no", "not worn", "none", "no policy"):
        risk_parts.append("overboard not wearing lifejacket")
    if profile.get("fire_detection", "").lower() in ("no", "none", "not fitted", "unknown"):
        risk_parts.append("fire")

    # Two focused queries: one for incidents, one for recommendations
    query_incidents = " ".join(risk_parts)
    query_recs = f"{vessel_type} vessel safety recommendations findings"

    all_docs = []
    seen = set()
    for q in [query_incidents, query_recs]:
        docs = retrieve(q, k=8, source_filter=source_filter, fast=True)
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    docs = all_docs[:10]

    if not docs:
        return {
            "assessment": "Unable to perform risk assessment — no accident investigation reports found in the database. Please ingest documents first.",
            "sources": [],
            "profile_text": profile_text,
        }

    context = _format_context(docs)

    # Generate assessment
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content="You are a maritime safety risk assessor who analyzes vessel profiles against real accident investigation evidence."),
        HumanMessage(content=ASSESSMENT_PROMPT.format(profile=profile_text, context=context)),
    ]
    response = llm.invoke(messages)

    return {
        "assessment": response.content,
        "sources": _extract_sources(docs),
        "profile_text": profile_text,
    }
