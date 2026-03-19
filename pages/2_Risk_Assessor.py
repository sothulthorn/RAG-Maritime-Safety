"""Safety Risk Assessor — analyze a vessel profile against real accident data."""

import streamlit as st

st.set_page_config(page_title="Safety Risk Assessor", page_icon="🚨", layout="wide")

from ui.sidebar import render_sidebar
from retrieval.assessor import assess_risk


def _render_source(src: dict):
    """Render a single source entry."""
    source_label = src.get("source", "Unknown")
    page = src.get("page")
    org = src.get("organization", "")
    header = f"**{source_label}**"
    if page:
        header += f" — Page {page}"
    if org:
        header += f" ({org})"
    st.markdown(header)
    st.text(src.get("snippet", ""))
    st.divider()


# --- Sidebar ---
render_sidebar()

# --- Main ---
st.header("Maritime Safety Risk Assessor")
st.caption(
    "Enter a vessel profile to get a risk assessment backed by real accident investigation evidence. "
    "This is not generic advice — every finding is grounded in actual MAIB, NTSB, and TSB reports."
)

st.divider()

# --- Vessel Profile Form ---
st.subheader("Vessel Profile")

col1, col2 = st.columns(2)

with col1:
    vessel_type = st.selectbox(
        "Vessel Type",
        ["fishing", "cargo", "tanker", "passenger", "tug", "recreational"],
        key="vessel_type",
    )
    vessel_length = st.text_input("Vessel Length", placeholder="e.g., 8.5m", key="vessel_length")
    operation = st.text_input("Operation Type", placeholder="e.g., Single-handed potting, Towing, Container transport", key="operation")
    crew_size = st.text_input("Crew Size", placeholder="e.g., 1, 5, 25", key="crew_size")
    operating_area = st.text_input("Operating Area", placeholder="e.g., Western Approaches, English Channel", key="operating_area")
    season = st.text_input("Season / Weather Conditions", placeholder="e.g., Winter, heavy weather", key="season")

with col2:
    stability_booklet = st.selectbox("Stability Booklet Available?", ["Yes", "No", "Unknown"], key="stability")
    pfd_policy = st.selectbox("PFD / Lifejacket Policy", ["Worn at all times", "Worn on deck", "Not worn", "No policy"], key="pfd")
    epirb = st.selectbox("EPIRB / PLB Carried?", ["Yes", "No", "Unknown"], key="epirb")
    mob_plan = st.selectbox("Man Overboard Recovery Plan?", ["Yes", "No", "Unknown"], key="mob")
    fire_detection = st.selectbox("Fire Detection System?", ["Yes", "No", "Unknown"], key="fire")
    weather_routing = st.selectbox("Weather Routing Used?", ["Yes", "No", "Unknown"], key="weather")

with st.expander("Additional Details"):
    maintenance = st.text_input("Maintenance Status", placeholder="e.g., Up to date, Overdue", key="maintenance")
    safety_training = st.text_input("Safety Training", placeholder="e.g., Annual drills, No formal training", key="training")
    communication = st.text_input("Communication Equipment", placeholder="e.g., VHF, Satellite phone", key="comms")
    additional_info = st.text_area("Any Other Relevant Information", placeholder="e.g., Operating at night, new crew member, unfamiliar waters", key="additional", height=80)

st.divider()

# --- Quick Presets ---
st.subheader("Quick Presets")
st.caption("Or try a pre-built scenario:")

preset_col1, preset_col2, preset_col3 = st.columns(3)

with preset_col1:
    preset_fishing = st.button("High-Risk Fishing Vessel", key="preset_fish")
with preset_col2:
    preset_tug = st.button("Tug Towing Operation", key="preset_tug")
with preset_col3:
    preset_cargo = st.button("Cargo Vessel Passage", key="preset_cargo")

# Apply presets
profile = {}
run_assessment = False

if preset_fishing:
    profile = {
        "vessel_type": "fishing",
        "vessel_length": "9.8m",
        "operation": "Single-handed, crab potting",
        "crew_size": "1",
        "operating_area": "Western Approaches",
        "season": "Winter, frequent gales",
        "stability_booklet": "No",
        "pfd_policy": "Not worn",
        "epirb": "Yes",
        "mob_plan": "No",
        "fire_detection": "No",
        "weather_routing": "No",
        "maintenance": "Unknown",
        "safety_training": "No formal training",
        "communication": "VHF only",
        "additional_info": "Operating alone in adverse weather conditions, no shore contact procedure",
    }
    run_assessment = True

elif preset_tug:
    profile = {
        "vessel_type": "tug",
        "vessel_length": "25m",
        "operation": "Ship assist, harbour towing",
        "crew_size": "4",
        "operating_area": "Port approaches",
        "season": "Year-round",
        "stability_booklet": "Yes",
        "pfd_policy": "Worn on deck",
        "epirb": "Yes",
        "mob_plan": "Yes",
        "fire_detection": "Yes",
        "weather_routing": "No",
        "maintenance": "Up to date",
        "safety_training": "Annual drills",
        "communication": "VHF, mobile phone",
        "additional_info": "Towing large vessels in confined waters, crew experienced",
    }
    run_assessment = True

elif preset_cargo:
    profile = {
        "vessel_type": "cargo",
        "vessel_length": "90m",
        "operation": "Short sea trading, general cargo",
        "crew_size": "12",
        "operating_area": "North Sea, English Channel",
        "season": "Winter",
        "stability_booklet": "Yes",
        "pfd_policy": "Worn on deck",
        "epirb": "Yes",
        "mob_plan": "Yes",
        "fire_detection": "Yes",
        "weather_routing": "Yes",
        "maintenance": "Up to date",
        "safety_training": "ISM compliant",
        "communication": "VHF, GMDSS, satellite",
        "additional_info": "Operating in TSS with dense traffic, winter passage",
    }
    run_assessment = True

# --- Manual Run ---
if st.button("Run Risk Assessment", type="primary", key="run_assess"):
    profile = {
        "vessel_type": vessel_type,
        "vessel_length": vessel_length,
        "operation": operation,
        "crew_size": crew_size,
        "operating_area": operating_area,
        "season": season,
        "stability_booklet": stability_booklet,
        "pfd_policy": pfd_policy,
        "epirb": epirb,
        "mob_plan": mob_plan,
        "fire_detection": fire_detection,
        "weather_routing": weather_routing,
        "maintenance": maintenance,
        "safety_training": safety_training,
        "communication": communication,
        "additional_info": additional_info,
    }
    run_assessment = True

# --- Run Assessment ---
if run_assessment and profile:
    st.divider()

    # Show the profile being assessed
    with st.expander("Vessel Profile Being Assessed", expanded=False):
        for key, val in profile.items():
            if val:
                st.markdown(f"- **{key.replace('_', ' ').title()}**: {val}")

    with st.spinner("Analyzing vessel profile against accident investigation database..."):
        source_filter = st.session_state.get("source_filter")
        result = assess_risk(profile, source_filter=source_filter)

    # Display assessment
    st.subheader("Risk Assessment Results")
    st.markdown(result["assessment"])

    # Sources
    if result["sources"]:
        with st.expander(f"Evidence Sources ({len(result['sources'])} documents referenced)"):
            for src in result["sources"]:
                _render_source(src)
