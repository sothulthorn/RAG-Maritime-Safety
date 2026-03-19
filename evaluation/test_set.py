"""
Ground-truth test set for evaluating the Maritime Safety RAG system.

Each question includes:
- question: the natural language query
- ground_truth: the expected factual answer (from actual documents)
- source_keywords: keywords that MUST appear in cited sources for citation to be valid
- key_facts: specific facts the answer MUST contain to be considered accurate
- category: question type for grouped analysis
"""

TEST_SET = [
    # --- Category: Specific Incident Facts ---
    {
        "id": "incident_01",
        "question": "What happened to the tug Biter?",
        "ground_truth": "The tug Biter capsized and sank while assisting the passenger vessel Hebridean Princess, resulting in the loss of two lives.",
        "source_keywords": ["Biter", "Hebridean"],
        "key_facts": ["capsize", "Hebridean Princess", "two lives", "tug"],
        "category": "specific_incident",
    },
    {
        "id": "incident_02",
        "question": "What caused the loss of propulsion on Spirit of Discovery?",
        "ground_truth": "The passenger vessel Spirit of Discovery experienced loss of propulsion in heavy weather, leading to over 100 injuries and one fatality.",
        "source_keywords": ["Spirit", "Discovery"],
        "key_facts": ["propulsion", "heavy weather", "injuries"],
        "category": "specific_incident",
    },
    {
        "id": "incident_03",
        "question": "What happened in the collision between Polesie and Verity?",
        "ground_truth": "The bulk carrier Polesie collided with the general cargo ship Verity, resulting in the sinking of Verity and the loss of lives.",
        "source_keywords": ["Polesie", "Verity"],
        "key_facts": ["collision", "bulk carrier", "sinking", "Verity"],
        "category": "specific_incident",
    },
    {
        "id": "incident_04",
        "question": "Describe the Honeybourne III incident.",
        "ground_truth": "A deckhand suffered a fatal injury following a chain failure on the scallop dredger Honeybourne III.",
        "source_keywords": ["Honeybourne"],
        "key_facts": ["chain failure", "scallop dredger", "fatal", "deckhand"],
        "category": "specific_incident",
    },
    {
        "id": "incident_05",
        "question": "What happened to the fishing vessel Opportune?",
        "ground_truth": "The stern trawler Opportune (LK 209) experienced flooding and foundered.",
        "source_keywords": ["Opportune"],
        "key_facts": ["flooding", "foundering", "stern trawler"],
        "category": "specific_incident",
    },
    # --- Category: Cause Analysis ---
    {
        "id": "cause_01",
        "question": "What are the common causes of fishing vessel capsizing based on investigation reports?",
        "ground_truth": "Common causes include stability issues, overloading, adverse weather conditions, and inadequate safety management.",
        "source_keywords": [],
        "key_facts": ["stability", "capsize"],
        "category": "cause_analysis",
    },
    {
        "id": "cause_02",
        "question": "What factors contribute to collision incidents between vessels?",
        "ground_truth": "Factors include inadequate lookout, failure to comply with COLREG, bridge watchkeeping failures, and communication breakdowns.",
        "source_keywords": [],
        "key_facts": ["collision"],
        "category": "cause_analysis",
    },
    {
        "id": "cause_03",
        "question": "What causes engine room fires on ships?",
        "ground_truth": "Engine room fires are commonly caused by fuel oil leaks, hot surface ignition, inadequate maintenance, and failure of fire detection systems.",
        "source_keywords": [],
        "key_facts": ["fire", "engine"],
        "category": "cause_analysis",
    },
    # --- Category: Safety Recommendations ---
    {
        "id": "safety_01",
        "question": "What safety recommendations were made after tug incidents?",
        "ground_truth": "Recommendations typically address towing procedures, crew training, vessel stability assessment, and emergency preparedness.",
        "source_keywords": ["tug", "Biter"],
        "key_facts": ["recommendation", "safety"],
        "category": "safety_recommendations",
    },
    {
        "id": "safety_02",
        "question": "What safety measures are recommended for preventing person overboard incidents on fishing vessels?",
        "ground_truth": "Recommendations include wearing personal flotation devices, man overboard recovery procedures, working alone policies, and guardrail requirements.",
        "source_keywords": [],
        "key_facts": ["overboard", "fishing"],
        "category": "safety_recommendations",
    },
    # --- Category: Regulatory Knowledge ---
    {
        "id": "reg_01",
        "question": "What does the Port Marine Safety Code require?",
        "ground_truth": "The Port Marine Safety Code sets out a national standard for port marine safety, covering risk assessment, safety management systems, and competence standards.",
        "source_keywords": ["port", "marine", "safety", "code"],
        "key_facts": ["port", "safety", "risk"],
        "category": "regulatory",
    },
    # --- Category: Cross-Document Analysis ---
    {
        "id": "cross_01",
        "question": "Compare the safety findings between the Biter capsize and the Opportune foundering.",
        "ground_truth": "Both incidents involved vessel loss. The Biter capsized during towing operations with loss of two lives, while the Opportune foundered due to flooding.",
        "source_keywords": ["Biter", "Opportune"],
        "key_facts": ["Biter", "Opportune", "capsize", "flooding"],
        "category": "cross_document",
    },
    {
        "id": "cross_02",
        "question": "What common safety themes emerge from MAIB investigation reports on fishing vessel losses?",
        "ground_truth": "Common themes include inadequate stability awareness, failure to wear PFDs, single-handed operations, and poor safety culture.",
        "source_keywords": [],
        "key_facts": ["fishing", "safety"],
        "category": "cross_document",
    },
    # --- Category: Detail Extraction ---
    {
        "id": "detail_01",
        "question": "What type of vessel was the Finnhawk?",
        "ground_truth": "The Finnhawk was a cargo vessel involved in a pilot ladder incident.",
        "source_keywords": ["Finnhawk"],
        "key_facts": ["Finnhawk", "cargo"],
        "category": "detail_extraction",
    },
    {
        "id": "detail_02",
        "question": "How many lives were lost in the Nicola Faith sinking?",
        "ground_truth": "Three lives were lost in the capsize and sinking of the whelk potter Nicola Faith.",
        "source_keywords": ["Nicola Faith"],
        "key_facts": ["three", "3", "Nicola Faith"],
        "category": "detail_extraction",
    },
]
