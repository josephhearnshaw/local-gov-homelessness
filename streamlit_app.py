"""
AnalyseMe - Housing Support Assessment Service
Conversational assessment with AWS Bedrock Claude integration
"""

import os
import json
import html
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import boto3
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AnalyseMe",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# AWS BEDROCK CONFIGURATION
# -----------------------------------------------------------------------------

BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY")

if BEDROCK_API_KEY:
    # New Bedrock bearer-token auth uses AWS_BEARER_TOKEN_BEDROCK
    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY
else:
    st.error("❌ BEDROCK_API_KEY environment variable is not set; LLM will not be available.")

BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # adjust if needed

# -----------------------------------------------------------------------------
# SYSTEM PROMPT FOR CLAUDE
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a compassionate housing support assistant for AnalyseMe, a Birmingham City Council housing vulnerability assessment service. Your role is to analyse assessments and provide helpful, empathetic responses with relevant local support links.

## Your Tasks

When you receive an assessment payload, return a JSON response with:

1. **user_response**: Friendly content for the citizen with relevant support links
2. **officer_summary**: Professional summary for housing officers

## Birmingham Support Resources Database

[TRUNCATED HERE IN THIS MESSAGE FOR BREVITY – KEEP YOUR FULL PROMPT TEXT]
"""

# Keep your full prompt – I’ve shortened it here just to save space in the chat.
# In your actual file, paste the entire prompt block you had.

# -----------------------------------------------------------------------------
# CSS STYLES
# -----------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .header-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .header-banner h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .header-banner p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    .question-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .question-number {
        background: #1e3a5f;
        color: white;
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .question-text {
        font-size: 1.25rem;
        font-weight: 500;
        color: #1e293b;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .risk-low { background: #dcfce7; color: #166534; }
    .risk-medium { background: #fef3c7; color: #92400e; }
    .risk-high { background: #fee2e2; color: #991b1b; }
    
    .score-display {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# QUESTIONS CONFIGURATION
# -----------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "housing_stability",
        "prompt": "Can you tell me a bit about how secure or stable your current living situation feels?",
        "weight": 20,
        "options": {
            "Very secure - I have stable, long-term housing": 0,
            "Fairly secure - but some uncertainty": 4,
            "Uncertain - my situation could change soon": 10,
            "Unstable - I may lose my home in the coming weeks": 16,
            "Crisis - I am homeless or about to be": 20
        },
        "risk_keywords": ["eviction", "homeless", "rough sleeping", "temporary", "sofa surfing", "notice"]
    },
    {
        "id": "financial_pressure",
        "prompt": "How are finances affecting your day-to-day life — do money pressures feel overwhelming right now?",
        "weight": 25,
        "options": {
            "Finances are comfortable - no concerns": 0,
            "Managing okay - occasional tight moments": 5,
            "Struggling - frequently worried about money": 12,
            "Severe pressure - behind on essential bills": 19,
            "Crisis - cannot afford basic needs, facing debt action": 25
        },
        "risk_keywords": ["debt", "arrears", "benefits", "universal credit", "income", "unemployed", "bills"]
    },
    {
        "id": "health_work_impact",
        "prompt": "Do you have any health conditions or disabilities that affect your ability to work or could put pressure on your finances?",
        "weight": 15,
        "options": {
            "No - my health doesn't affect my ability to work": 0,
            "Minor impact - I can work with some adjustments": 3,
            "Moderate impact - limits the work I can do": 7,
            "Significant impact - unable to work full-time": 11,
            "Severe - unable to work due to health/disability": 15
        },
        "risk_keywords": ["disability", "unable to work", "PIP", "ESA", "long-term sick", "chronic illness", "limited capability"]
    },
    {
        "id": "mental_health",
        "prompt": "Would you like to talk about your mental health or emotional wellbeing?",
        "weight": 10,
        "options": {
            "I'm doing well emotionally": 0,
            "Some stress but coping okay": 2,
            "Struggling - anxiety, low mood affecting me": 5,
            "Significant difficulties - impacting daily life": 7,
            "In crisis - need urgent mental health support": 10
        },
        "risk_keywords": ["mental health", "depression", "anxiety", "stress", "suicidal", "crisis"]
    },
    {
        "id": "abuse_safety",
        "prompt": "Have you experienced situations, like abuse or violence, that impact your safety or stability?",
        "weight": 10,
        "options": {
            "No - I feel safe": 0,
            "Past experiences - but currently safe": 3,
            "Some concerns about safety": 6,
            "Currently experiencing abuse or control": 8,
            "In immediate danger - need to leave": 10
        },
        "risk_keywords": ["domestic abuse", "violence", "fleeing", "refuge", "controlling", "fear"]
    },
    {
        "id": "care_leaver",
        "prompt": "Are you preparing to leave care services, or have you recently transitioned out of them?",
        "weight": 5,
        "options": {
            "This doesn't apply to me": 0,
            "Left care several years ago": 1,
            "Left care in the past 2-3 years": 2,
            "Recently left care (within 12 months)": 4,
            "Currently preparing to leave care": 5
        },
        "risk_keywords": ["care leaver", "foster care", "looked after", "leaving care", "personal adviser"]
    },
    {
        "id": "institutional_discharge",
        "prompt": "Have you recently left prison or another institution, and are you adjusting to life outside?",
        "weight": 5,
        "options": {
            "This doesn't apply to me": 0,
            "Left an institution over a year ago": 1,
            "Left within the past year": 2,
            "Recently released (past 3 months)": 4,
            "Just released / about to be released": 5
        },
        "risk_keywords": ["prison", "hospital discharge", "rehabilitation", "probation", "resettlement"]
    },
    {
        "id": "benefits_access",
        "prompt": "Do you feel confident about the benefits or entitlements you can access, or would support help?",
        "weight": 5,
        "options": {
            "Yes - I receive everything I'm entitled to": 0,
            "Mostly - but might be missing something": 1,
            "Unsure - I don't know what I can claim": 3,
            "Struggling to access benefits I need": 4,
            "Benefits stopped or refused - need urgent help": 5
        },
        "risk_keywords": ["benefits", "universal credit", "PIP", "housing benefit", "sanctions", "appeal"]
    },
    {
        "id": "support_network",
        "prompt": "Who do you feel you can rely on right now — family, friends, or community networks?",
        "weight": 3,
        "options": {
            "Strong support network - family and friends": 0,
            "Some support available": 1,
            "Limited support - one or two people": 2,
            "Very little support - mostly alone": 2,
            "No support - completely isolated": 3
        },
        "risk_keywords": ["isolated", "alone", "no family", "estranged", "breakdown"]
    },
    {
        "id": "service_interest",
        "prompt": "Would it be useful if I showed you local and national services that match your needs?",
        "weight": 2,
        "options": {
            "Not needed right now - just exploring": 0,
            "Maybe - would like to know what's available": 1,
            "Yes - I'd like some guidance": 1,
            "Definitely - I need help finding services": 2,
            "Urgent - I need immediate support connections": 2
        },
        "risk_keywords": []
    }
]

assert sum(q["weight"] for q in QUESTIONS) == 100, "Weights must sum to 100"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_bedrock_client():
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        return client
    except Exception as e:
        st.error("❌ Could not create Bedrock client")
        st.code(repr(e))
        return None


def clean_text_for_html(value: Optional[str]) -> str:
    """Strip tags, collapse whitespace, escape again."""
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.escape(text)


def get_risk_band(score: int) -> Tuple[str, str, str, str]:
    if score >= 60:
        return ("HIGH", "risk-high", "Urgent intervention recommended", "Within 24 hours")
    elif score >= 35:
        return ("MEDIUM", "risk-medium", "Priority support pathway", "Within 3 working days")
    else:
        return ("LOW", "risk-low", "Standard support pathway", "Within 10 working days")


def build_llm_payload(responses: Dict, additional_context: str, reference: str) -> Dict:
    total_score = 0
    category_scores = {}
    risk_flags = []
    crisis_override = False

    for q in QUESTIONS:
        q_id = q["id"]
        if q_id in responses:
            answer = responses[q_id]
            if answer in q["options"]:
                score = q["options"][answer]
                category_scores[q_id] = {
                    "score": score,
                    "max": q["weight"],
                    "answer": answer
                }
                total_score += score

                if score > q["weight"] * 0.6:
                    risk_flags.append({
                        "category": q_id,
                        "concern": q["prompt"],
                        "response": answer,
                        "keywords": q.get("risk_keywords", [])
                    })

                # Crisis overrides
                if q_id == "housing_stability":
                    if answer.startswith("Unstable") or answer.startswith("Crisis"):
                        crisis_override = True
                if q_id == "abuse_safety":
                    if answer.startswith("In immediate danger"):
                        crisis_override = True
                if q_id == "mental_health":
                    if answer.startswith("In crisis"):
                        crisis_override = True

    risk_level, _, risk_desc, response_time = get_risk_band(total_score)

    if crisis_override and risk_level != "HIGH":
        risk_level = "HIGH"
        risk_desc = "Crisis indicators present – urgent intervention recommended"
        response_time = "Within 24 hours"

    return {
        "reference": reference,
        "timestamp": datetime.now().isoformat(),
        "assessment": {
            "total_score": total_score,
            "max_score": 100,
            "risk_level": risk_level,
            "risk_description": risk_desc,
            "recommended_response_time": response_time
        },
        "category_scores": category_scores,
        "risk_flags": risk_flags,
        "additional_context": additional_context,
        "request": {
            "generate_support_links": True,
            "generate_risk_summary": True,
            "locale": "en-GB"
        }
    }


def generate_reference() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"AM-{timestamp}-{abs(hash(timestamp)) % 10000:04d}"


def call_bedrock_claude(payload: Dict) -> Optional[Dict]:
    client = get_bedrock_client()
    if client is None:
        st.warning("⚠️ Bedrock client unavailable; using fallback.")
        return None

    st.info("🔍 Calling Bedrock Claude via boto3.converse()")

    user_prompt = (
        "Analyse this Birmingham housing support assessment and provide personalised support "
        "recommendations.\n\nAssessment Data:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Return ONLY valid JSON following the specified format. No extra text."
    )

    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 2500},
        )
    except Exception as e:
        st.error("❌ Bedrock request error")
        st.code(repr(e))
        return None

    try:
        content = response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        st.error("❌ Unexpected response format from Bedrock")
        st.code(json.dumps(response, indent=2))
        return None

    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        st.error("❌ LLM returned invalid JSON")
        st.code(content[:2000])
        return None


def get_fallback_response(payload: Dict) -> Dict:
    risk_level = payload.get("assessment", {}).get("risk_level", "MEDIUM")

    return {
        "user_response": {
            "greeting": "Thank you for completing this assessment. Based on your responses, we've identified some services that may be able to help you.",
            "support_links": [
                {
                    "name": "Birmingham Council Homelessness Line",
                    "url": "https://www.birmingham.gov.uk/homeless",
                    "phone": "0121 303 7410",
                    "description": "24/7 housing advice and emergency support for Birmingham residents",
                    "priority": "high"
                },
                {
                    "name": "Shelter Birmingham",
                    "url": "https://england.shelter.org.uk/get_help/local_services/birmingham",
                    "phone": "0808 800 4444",
                    "description": "Free expert housing advice and support",
                    "priority": "high"
                },
                {
                    "name": "Citizens Advice Birmingham",
                    "url": "https://www.citizensadvice.org.uk/local/birmingham/",
                    "phone": "0808 278 7973",
                    "description": "Free advice on benefits, debt, and housing issues",
                    "priority": "medium"
                }
            ],
            "next_steps": f"Your reference number is {payload.get('reference')}. A housing support officer will review your case and be in touch within the recommended timeframe.",
            "emergency_note": "If you need immediate help tonight, call Birmingham Council on 0121 303 7410" if risk_level == "HIGH" else None
        },
        "officer_summary": {
            "risk_level": risk_level,
            "key_concerns": [f"Score: {payload.get('assessment', {}).get('total_score', 'N/A')}/100"] +
                           [f['category'].replace('_', ' ').title() for f in payload.get("risk_flags", [])],
            "recommended_actions": ["Review full assessment", "Contact citizen within recommended timeframe"],
            "referral_suggestions": ["Based on risk flags - see assessment details"],
            "notes": "Automated fallback response - LLM analysis was unavailable. Manual review recommended."
        }
    }

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "additional_context" not in st.session_state:
    st.session_state.additional_context = ""
if "reference" not in st.session_state:
    st.session_state.reference = None
if "llm_payload" not in st.session_state:
    st.session_state.llm_payload = None
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "used_fallback" not in st.session_state:
    st.session_state.used_fallback = False

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------

def render_welcome():
    st.markdown("""
        <div class="header-banner">
            <h1>🏠 AnalyseMe</h1>
            <p>Housing Support Assessment</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Welcome
    
    This short assessment helps us understand your situation so we can connect you with the right support in Birmingham.
    
    **What to expect:**
    - 10 simple questions about your circumstances
    - Takes about **5 minutes**
    - Your answers are **confidential**
    - You'll get **personalised support links**
    
    All information is protected under GDPR and used only to help identify services for you.
    """)

    st.divider()

    if st.button("Begin Assessment", type="primary", use_container_width=True):
        st.session_state.page = "question"
        st.session_state.current_question = 0
        st.rerun()

    st.markdown("---")
    st.caption("🆘 **Need help now?** Call Birmingham Council: 0121 303 7410 | Shelter: 0808 800 4444 | Emergency: 999")


def render_question():
    q_idx = st.session_state.current_question
    total = len(QUESTIONS)

    if q_idx >= total:
        st.session_state.page = "additional"
        st.rerun()
        return

    q = QUESTIONS[q_idx]

    st.markdown("""
        <div class="header-banner">
            <h1>🏠 AnalyseMe</h1>
            <p>Tell us about your situation</p>
        </div>
    """, unsafe_allow_html=True)

    progress = (q_idx + 1) / (total + 1)
    st.progress(progress, text=f"Question {q_idx + 1} of {total}")

    st.markdown(f"""
        <div class="question-container">
            <span class="question-number">Question {q_idx + 1}</span>
            <div class="question-text">{q['prompt']}</div>
        </div>
    """, unsafe_allow_html=True)

    options = list(q["options"].keys())
    current = st.session_state.responses.get(q["id"])

    selected = st.radio(
        "Select your answer",
        options=options,
        index=options.index(current) if current in options else None,
        key=f"q_{q['id']}_{q_idx}",
        label_visibility="collapsed"
    )

    if selected:
        st.session_state.responses[q["id"]] = selected

    st.markdown("---")
    col1, _, col3 = st.columns([1, 1, 1])

    with col1:
        if q_idx > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()

    with col3:
        if selected:
            btn_text = "Next →" if q_idx < total - 1 else "Continue →"
            if st.button(btn_text, type="primary", use_container_width=True):
                st.session_state.current_question += 1
                st.rerun()
        else:
            st.button("Next →", type="primary", use_container_width=True, disabled=True)
            st.caption("Please select an answer")


def render_additional():
    st.markdown("""
        <div class="header-banner">
            <h1>🏠 AnalyseMe</h1>
            <p>Almost done</p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(1.0, text="Final step")

    st.markdown("""
        <div class="question-container">
            <span class="question-number">Final Step</span>
            <div class="question-text">Is there anything else you'd like to share about your situation?</div>
        </div>
    """, unsafe_allow_html=True)

    st.caption("This helps us better understand your circumstances and find the most relevant support. You can skip this if you prefer.")

    additional = st.text_area(
        "Additional context",
        value=st.session_state.additional_context,
        height=180,
        placeholder="For example:\n• Specific challenges you're facing\n• Support you've already tried\n• Any other circumstances we should know about...",
        label_visibility="collapsed"
    )
    st.session_state.additional_context = additional

    st.markdown("---")
    col1, _, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_question = len(QUESTIONS) - 1
            st.session_state.page = "question"
            st.rerun()

    with col3:
        if st.button("Complete Assessment →", type="primary", use_container_width=True):
            st.session_state.reference = generate_reference()
            st.session_state.llm_payload = build_llm_payload(
                st.session_state.responses,
                st.session_state.additional_context,
                st.session_state.reference
            )
            with st.spinner("Analysing your responses..."):
                llm_result = call_bedrock_claude(st.session_state.llm_payload)
                if llm_result is not None:
                    st.session_state.llm_response = llm_result
                    st.session_state.used_fallback = False
                else:
                    st.session_state.llm_response = get_fallback_response(st.session_state.llm_payload)
                    st.session_state.used_fallback = True

            st.session_state.page = "results"
            st.rerun()


def render_results():
    if st.session_state.get("used_fallback", False):
        st.warning("⚠️ Using fallback — LLM analysis was unavailable.")

    st.markdown("""
        <div class="header-banner">
            <h1>🏠 AnalyseMe</h1>
            <p>Your Assessment Results</p>
        </div>
    """, unsafe_allow_html=True)

    st.success(f"📋 **Reference:** `{st.session_state.reference}` — Save this for your records")

    payload = st.session_state.llm_payload
    llm_response = st.session_state.llm_response

    assessment = payload["assessment"]
    score = assessment["total_score"]
    risk_level = assessment["risk_level"]
    risk_desc = assessment["risk_description"]
    response_time = assessment["recommended_response_time"]

    if risk_level == "HIGH":
        risk_class = "risk-high"
    elif risk_level == "MEDIUM":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    tab_user, tab_officer = st.tabs(["👤 Your Support & Advice", "👔 Officer View"])

    # USER VIEW
    with tab_user:
        user_resp = llm_response.get("user_response", {})

        raw_greeting = user_resp.get("greeting", "Thank you for completing this assessment.")
        greeting = clean_text_for_html(raw_greeting)
        st.markdown("### ")
        st.markdown(greeting)

        raw_emergency = user_resp.get("emergency_note")
        if raw_emergency:
            emergency = clean_text_for_html(raw_emergency)
            st.error(f"🆘 **Urgent:** {emergency}")

        st.markdown("---")
        st.markdown("### Support Services For You")

        support_links = user_resp.get("support_links", [])
        for link in support_links:
            priority = (link.get("priority") or "medium").lower()

            if priority == "high":
                bg = "#fef2f2"
                border = "#dc2626"
                badge = "<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:999px;font-size:0.75rem;font-weight:600;'>HIGH PRIORITY</span>"
            elif priority == "medium":
                bg = "#fffbeb"
                border = "#f59e0b"
                badge = ""
            else:
                bg = "#f9fafb"
                border = "#e5e7eb"
                badge = ""

            name = clean_text_for_html(link.get("name", "Support Service"))
            desc = clean_text_for_html(link.get("description", ""))
            phone_raw = link.get("phone")
            phone = clean_text_for_html(phone_raw) if phone_raw else ""
            url = link.get("url") or "#"

            st.markdown(f"""
                <div style="border-radius:12px;border:1px solid {border};background:{bg};padding:1.1rem 1.25rem;margin-bottom:1rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem;">
                        <strong style="font-size:1.05rem;">{name}</strong>
                        {badge}
                    </div>
                    <div style="margin:0.4rem 0 0.6rem 0;color:#4b5563;font-size:0.95rem;">
                        {desc}
                    </div>
                    <div style="margin:0;font-size:0.95rem;">
                        {("📞 <strong>" + phone + "</strong>&nbsp;&nbsp;" ) if phone else ""}
                        🔗 <a href="{url}" target="_blank">Visit website</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### What Happens Next")
        raw_next_steps = user_resp.get(
            "next_steps",
            f"A housing support officer will review your case within {response_time}."
        )
        next_steps = clean_text_for_html(raw_next_steps)
        st.markdown(next_steps)

        if risk_level == "HIGH":
            st.error(f"⚡ **Priority Case** — Aim to contact within {response_time}")
        elif risk_level == "MEDIUM":
            st.warning(f"📞 **Case Review** — Expected response within {response_time}")
        else:
            st.info(f"📧 **Standard Pathway** — Response within {response_time}")

    # OFFICER VIEW
    with tab_officer:
        officer_resp = llm_response.get("officer_summary", {})

        st.markdown("### Risk Assessment")

        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div class="score-display">{score}/100</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="text-align: center; margin-top: 0.5rem;">
                    <span class="risk-badge {risk_class}">{risk_level} RISK</span>
                </div>
            """, unsafe_allow_html=True)
            st.caption(risk_desc)

        st.markdown("""
        | Threshold | Level | Response Time |
        |-----------|-------|---------------|
        | 0-34 | LOW | Within 10 working days |
        | 35-59 | MEDIUM | Within 3 working days |
        | 60-100 | HIGH | Within 24 hours |
        """)

        st.markdown("---")
        st.markdown("### Key Concerns")
        for concern in officer_resp.get("key_concerns", []):
            st.markdown(f"- {concern}")

        st.markdown("---")
        st.markdown("### Recommended Actions")
        for action in officer_resp.get("recommended_actions", []):
            st.markdown(f"- {action}")

        st.markdown("---")
        st.markdown("### Referral Suggestions")
        for ref in officer_resp.get("referral_suggestions", []):
            st.markdown(f"- {ref}")

        if officer_resp.get("notes"):
            st.markdown("---")
            st.markdown("### Officer Notes")
            st.info(officer_resp.get("notes"))

        st.markdown("---")
        st.markdown("### Score Breakdown")

        breakdown_data: List[Dict] = []
        for q in QUESTIONS:
            q_id = q["id"]
            if q_id in payload["category_scores"]:
                cat = payload["category_scores"][q_id]
                breakdown_data.append({
                    "Category": q_id.replace("_", " ").title(),
                    "Score": f"{cat['score']}/{cat['max']}",
                    "Response": cat["answer"][:50] + ("..." if len(cat["answer"]) > 50 else "")
                })

        st.dataframe(breakdown_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Full Payload")
        with st.expander("View JSON payload"):
            st.code(json.dumps(payload, indent=2), language="json")

        with st.expander("View LLM Response"):
            st.code(json.dumps(llm_response, indent=2), language="json")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📧 Create Case", use_container_width=True)
        with col2:
            st.button("📞 Schedule Call", use_container_width=True)
        with col3:
            st.button("📄 Export PDF", use_container_width=True)

    st.markdown("---")
    if st.button("Start New Assessment", use_container_width=True):
        st.session_state.page = "welcome"
        st.session_state.current_question = 0
        st.session_state.responses = {}
        st.session_state.additional_context = ""
        st.session_state.reference = None
        st.session_state.llm_payload = None
        st.session_state.llm_response = None
        st.session_state.used_fallback = False
        st.rerun()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    if st.session_state.page == "welcome":
        render_welcome()
    elif st.session_state.page == "question":
        render_question()
    elif st.session_state.page == "additional":
        render_additional()
    elif st.session_state.page == "results":
        render_results()

if __name__ == "__main__":
    main()
