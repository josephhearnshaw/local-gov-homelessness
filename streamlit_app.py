"""
AnalyseMe - Housing Support Assessment Service
Conversational assessment with AWS Bedrock Claude integration
"""

import os
import json
import html
import re
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

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
BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
if BEDROCK_API_KEY:
    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY
else:
    msg = "BEDROCK_API_KEY environment variable is not set; LLM will not be available."
    logging.warning(msg)
    st.error("❌ " + msg)

    import requests



BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"


# -----------------------------------------------------------------------------
# SYSTEM PROMPT FOR CLAUDE
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a compassionate housing support assistant for AnalyseMe, a Birmingham City Council housing vulnerability assessment service. Your role is to analyse assessments and provide helpful, empathetic responses with relevant local support links.

## Your Tasks

When you receive an assessment payload, return a JSON response with:

1. **user_response**: Friendly content for the citizen with relevant support links
2. **officer_summary**: Professional summary for housing officers

## Birmingham Support Resources Database

Use these verified Birmingham-specific services. Match services to the person's identified needs based on their assessment scores and risk flags.

### HOMELESSNESS SERVICES (Birmingham Council)

**Birmingham Council Homelessness Line**
- URL: https://www.birmingham.gov.uk/homeless
- Phone: 0121 303 7410 (Over 18) | 0121 303 1888 (Under 18)
- Description: 24/7 homelessness advice, emergency contacts, duty to refer
- Use when: Housing stability score high, immediate homelessness risk

**Housing Needs Assessment**
- URL: https://www.birmingham.gov.uk/housingneeds
- Description: Online form to apply if at risk within 56 days
- Use when: Housing uncertain but not immediate crisis

**Emergency Accommodation (25+ or 18-25 with children)**
- URL: https://www.birmingham.gov.uk/info/20207/homelessness/1196/i_need_accommodation_now
- Description: Temporary accommodation process and DA hub contacts
- Use when: Immediate housing need, domestic abuse situation

### YOUTH HOMELESSNESS (16-24)

**St Basils Youth Hub**
- URL: https://stbasils.org.uk/hub/birmingham/
- Phone: 0121 439 7766
- Description: Single access hub for 16-24s: prevention, mediation, housing options
- Use when: Under 25, any housing concern

### DOMESTIC VIOLENCE & ABUSE

**Birmingham & Solihull Women's Aid**
- URL: https://bswaid.org/i-need-help/
- Phone: 0808 800 0028
- Description: Helpline, refuges, drop-in, webchat for women and children
- Use when: Female, abuse/safety concerns

**Anawim Centre for Women**
- URL: https://anawim.co.uk/
- Description: Trauma-informed support including housing, mental health, legal advocacy
- Use when: Female, abuse + mental health concerns

**Gilgal Birmingham Refuge**
- URL: https://gilgalbham.org.uk/
- Description: Safe housing for women and children fleeing abuse
- Use when: Female with/without children, fleeing abuse

**WAITS Refuge**
- URL: https://www.waitsaction.org/who-we-are/refuge-accommodation
- Description: Refuge for single women 21+
- Use when: Female 21+, fleeing abuse, no children

**Council DA Services Directory**
- URL: https://www.birmingham.gov.uk/info/50350/domestic_abuse/2980/help_for_victims_and_survivors_of_domestic_abuse
- Description: Comprehensive list including services for women, men, LGBT+
- Use when: Any abuse situation, need multiple options

**No Excuse for Abuse Directory**
- URL: https://noexcuseforabuse.info/locations/birmingham/
- Description: Aggregated contacts including LGBT and ethnic minority services
- Use when: Abuse situation, diverse needs

### MENTAL HEALTH

**Birmingham Mind - Housing Support**
- URL: https://birminghammind.org/services/housing/
- Phone: 0121 262 3555
- Description: Mental health support and housing advice
- Use when: Mental health flagged, any housing concern

**Birmingham Mind - Supported Accommodation**
- URL: https://birminghammind.org/what-we-do/supported-accommodation/
- Description: Supported housing for adults with MH difficulties
- Use when: Mental health significant, needs supported housing

**Birmingham Mind - Tenancy Support**
- URL: https://birminghammind.org/what-we-do/support-in-your-community-and-your-own-home/
- Description: Support to sustain tenancies for adults with MH conditions
- Use when: Mental health concerns + at risk of losing tenancy

**Birmingham Mind - Homelessness Contacts**
- URL: https://birminghammind.org/what-we-do/helpline/homelessness/
- Description: Signposting for homelessness including council and St Basils contacts
- Use when: Mental health + homelessness combination

**Birmingham Crisis Centre**
- URL: https://www.birminghamcrisis.org.uk/
- Description: Emergency accommodation for those in mental health crisis
- Use when: Mental health crisis, needs emergency accommodation

### SUBSTANCE MISUSE

**Change Grow Live Birmingham**
- URL: https://www.changegrowlive.org/service/birmingham-drug-alcohol
- Description: Adult drug & alcohol hubs across Birmingham
- Use when: Substance misuse mentioned in additional context

**Aquarius (Under 25s)**
- URL: https://aquarius.org.uk/
- Description: Drug & alcohol service for up to 25s
- Use when: Under 25, substance issues

**Cranstoun (Women - DA + Addiction)**
- URL: https://www.cranstoun.org/services/domestic-abuse/
- Description: Specialist services for women facing domestic abuse and addiction
- Use when: Female, abuse + substance issues

### DEBT & FINANCIAL

**Citizens Advice Birmingham**
- URL: https://www.citizensadvice.org.uk/local/birmingham/
- Phone: 0808 278 7973
- Description: Rent arrears, eviction prevention, benefits advice
- Use when: Financial pressure high, debt, benefits uncertainty

**Citizens Advice - Rent Arrears Guide**
- URL: https://www.citizensadvice.org.uk/debt-and-money/rent-arrears/
- Description: National guidance on dealing with arrears and eviction
- Use when: Rent arrears specifically mentioned

**Birmingham Council Rent Arrears**
- URL: https://www.birmingham.gov.uk/rentarrears
- Description: Council tenant support
- Use when: Council tenant with rent issues

**StepChange**
- URL: https://www.stepchange.org/
- Phone: 0800 138 1111
- Description: Free debt advice and solutions
- Use when: Debt issues, financial crisis

### CARE LEAVERS

**Birmingham Children's Trust (18-21)**
- URL: https://www.birminghamchildrenstrust.co.uk/info/2/information_for_children_and_young_people/28/support_for_18_to_21_year_olds_who_have_been_in_care
- Description: Support for care leavers transitioning to independent living
- Use when: Care leaver identified

**Shelter - Care Leaver Housing Rights**
- URL: https://england.shelter.org.uk/housing_advice/homelessness/housing_help_and_homelessness_if_you_are_a_care_leaver
- Description: Explains entitlements and priority need criteria
- Use when: Care leaver, needs to understand rights

### EX-OFFENDERS / PRISON LEAVERS

**Birmingham Council - Prisoners & Ex-Offenders**
- URL: https://www.birmingham.gov.uk/info/50113/housing_advice_and_support/1221/prisoners_and_ex-offenders/2
- Description: Guidance on support on release, homelessness assistance
- Use when: Prison/institutional discharge identified

**St Giles Trust**
- URL: https://www.stgilestrust.org.uk/
- Description: Helps ex-offenders reintegrate and avoid homelessness
- Use when: Prison release, resettlement needs

**Nacro Birmingham**
- URL: https://homeless.org.uk/homeless-england/service/nacro-birmingham-offender-housing/
- Description: Supported accommodation for single adults including MH and substance issues
- Use when: Ex-offender + mental health or substance issues

### REFUGEES & ASYLUM SEEKERS

**Refugee Council**
- URL: https://www.refugeecouncil.org.uk/
- Description: Housing advice for asylum seekers
- Use when: Refugee/asylum status identified

**Refugee & Migrant Centre**
- URL: https://rmcentre.org.uk/get-help/
- Description: Holistic support including immigration, housing & homelessness
- Use when: Migrant, asylum seeker, refugee in Birmingham

**St Chad's Sanctuary**
- URL: http://www.stchadssanctuary.com/
- Description: Welcome centre providing practical support
- Use when: Asylum seeker, immediate practical needs

**Birmingham City of Sanctuary**
- URL: https://www.birmingham.gov.uk/info/50227/city_of_sanctuary/2512/asylum_support_in_birmingham
- Description: Local support directories and contacts
- Use when: Asylum seeker, needs directory of services

### ARMED FORCES / VETERANS

**SSAFA Greater Birmingham**
- URL: https://www.ssafa.org.uk/greater-birmingham
- Description: Local welfare and housing support for veterans
- Use when: Armed forces/veteran identified

**Birmingham Council - Ex-Armed Forces**
- URL: https://www.birmingham.gov.uk/info/50113/housing_advice_and_support/1222/ex-armed_forces_personnel/2
- Description: Housing options including supported accommodation
- Use when: Veteran with housing needs

**Veterans Aid**
- URL: https://www.veteransaid.net/
- Description: Emergency accommodation for homeless veterans
- Use when: Veteran, immediate homelessness

**Haig Housing**
- URL: https://www.haighousing.org.uk/
- Description: Supported housing provider for veterans
- Use when: Veteran needs longer-term housing

### ROUGH SLEEPING

**Trident Reach Homeless Services**
- URL: https://tridentgroup.org.uk/care-support/homeless-services/
- Description: Outreach, emergency beds, supported housing
- Use when: Rough sleeping, immediate street homelessness

**Washington Court Hub**
- URL: https://homeless.org.uk/homeless-england/service/trident-reach-washington-court-hub/
- Phone: 0121 675 4249 (Gateway referral)
- Description: Large supported facility including rooms for women (21+)
- Use when: 21+, needs supported accommodation

**SIFA Fireside**
- URL: https://www.birmingham.gov.uk/info/20207/homelessness/1216/homeless_advice
- Description: Day centre with housing advice, welfare and health services (25+)
- Use when: 25+, rough sleeping, needs daytime support

### GENERAL ADVICE & DIRECTORIES

**Shelter Birmingham**
- URL: https://england.shelter.org.uk/get_help/local_services/birmingham
- Phone: 0808 800 4444
- Description: Legal Aid housing advice, emergency helpline, webchat
- Use when: Any housing issue, needs expert advice

**Crisis Skylight Birmingham**
- URL: https://www.crisis.org.uk/get-help/birmingham/
- Description: Support to leave homelessness, find housing/work/health support
- Use when: Homeless or at risk, needs comprehensive support

**Route 2 Wellbeing Directory**
- URL: https://r2wbirmingham.info
- Description: Directory of housing advice/support services in Birmingham
- Use when: Needs to browse multiple options

## Response Guidelines

1. **Tone**: Warm, non-judgmental, empowering. Avoid bureaucratic language.
2. **Prioritise by risk flags**: List most urgent services first based on their specific concerns.
3. **Be specific**: Match 3-5 services to their actual situation, don't overwhelm with options.
4. **Include contact methods**: Phone numbers are essential for urgent cases.
5. **Age-appropriate**: Under 25 → St Basils; 25+ → Council/adult services.
6. **Acknowledge urgency**: HIGH risk = emphasise immediate options and phone numbers.

## Response Format

Return ONLY valid JSON with no additional text:

{
  "user_response": {
    "greeting": "Empathetic opening acknowledging their situation",
    "support_links": [
      {
        "name": "Service Name",
        "url": "https://...",
        "phone": "phone number or null",
        "description": "How this specifically helps them",
        "priority": "high|medium|low"
      }
    ],
    "next_steps": "Clear guidance on what happens next",
    "emergency_note": "Only if HIGH risk or crisis - include key phone numbers, otherwise null"
  },
  "officer_summary": {
    "risk_level": "HIGH|MEDIUM|LOW",
    "key_concerns": ["List of main issues"],
    "recommended_actions": ["Suggested officer actions"],
    "referral_suggestions": ["Services to refer to"],
    "notes": "Additional observations"
  }
}"""

# -----------------------------------------------------------------------------
# CSS STYLES – DARK BACKGROUND + CLEAR TEXT
# -----------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020617;
        color: #e5e7eb;
    }
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    a { color: #38bdf8; }

    .header-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .header-banner h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .header-banner p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    .question-container {
        background: #020617;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        color: #e5e7eb;
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
        color: #e5e7eb;
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
    .risk-low { background: #14532d; color: #bbf7d0; }
    .risk-medium { background: #92400e; color: #fef3c7; }
    .risk-high { background: #7f1d1d; color: #fee2e2; }
    
    .score-display {
        font-size: 3rem;
        font-weight: 700;
        color: #e5e7eb;
        text-align: center;
    }

    .stTextArea textarea {
        font-size: 1rem !important;
        line-height: 1.6 !important;
        min-height: 150px !important;
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
    """
    Create a Bedrock client.

    IMPORTANT: with current boto3/botocore, you STILL need normal AWS creds
    (env vars, shared config, or IAM role). The API key alone is not enough
    unless you're on the very latest SDK with bearer-token support.
    """
    logging.info("Creating Bedrock client in us-east-1")
    print("Creating Bedrock client in us-east-1")

    # Quick sanity check for credentials so you don't get a mysterious traceback
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        msg = (
            "No AWS credentials found. Configure either:\n"
            "- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optional AWS_SESSION_TOKEN), or\n"
            "- an IAM role / profile with bedrock:InvokeModel permissions."
        )
        logging.error(msg)
        print(msg)
        st.error("❌ " + msg)
        return None

    try:
        client = session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        return client
    except Exception as e:
        logging.exception("Could not create Bedrock client")
        print("Could not create Bedrock client:", repr(e))
        traceback.print_exc()
        st.error("❌ Could not create Bedrock client")
        st.code(repr(e))
        return None


def clean_text_for_html(value: Optional[str]) -> str:
    """
    Make absolutely sure no tags leak through.
    """
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)  # strip all tags
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

                # Crisis overrides HIGH even if total score is borderline
                if q_id == "housing_stability" and (answer.startswith("Unstable") or answer.startswith("Crisis")):
                    crisis_override = True
                if q_id == "abuse_safety" and answer.startswith("In immediate danger"):
                    crisis_override = True
                if q_id == "mental_health" and answer.startswith("In crisis"):
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
    """
    Call Bedrock Converse endpoint over HTTPS using the API key.
    No boto3, no IAM creds. Logs errors to both Streamlit and terminal.
    """
    if not BEDROCK_API_KEY:
        msg = "No Bedrock API key set (AWS_BEARER_TOKEN_BEDROCK or BEDROCK_API_KEY)."
        st.error("❌ " + msg)
        print(msg)
        return None

    url = (
        f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/"
        f"model/{BEDROCK_MODEL_ID}/converse"
    )

    user_prompt = (
        "Analyse this Birmingham housing support assessment and provide personalised "
        "support recommendations.\n\nAssessment Data:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Return ONLY valid JSON following the specified format. No extra text."
    )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            }
        ],
        "system": [{"text": SYSTEM_PROMPT}],
        "inferenceConfig": {"maxTokens": 2500},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
    except requests.RequestException as e:
        st.error("❌ Bedrock HTTP request error")
        print("Bedrock HTTP request error:", repr(e))
        return None

    if not resp.ok:
        # Show enough to debug, but cap length so Streamlit isn’t flooded
        msg = f"Bedrock HTTP {resp.status_code}: {resp.text[:500]}"
        st.error("❌ " + msg)
        print(msg)
        return None

    try:
        data = resp.json()
    except Exception as e:
        st.error("❌ Could not decode Bedrock JSON response")
        print("JSON decode error:", repr(e), "raw:", resp.text[:500])
        return None

    # Same structure as SDK: output.message.content[0].text
    try:
        content = data["output"]["message"]["content"][0]["text"]
    except Exception as e:
        st.error("❌ Unexpected Bedrock response shape")
        print("Unexpected shape:", repr(e))
        print("Raw data:", json.dumps(data, indent=2)[:2000])
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error("❌ LLM returned invalid JSON")
        print("JSON parse error:", repr(e))
        print("Raw LLM text:", content[:2000])
        return None



def get_fallback_response(payload: Dict) -> Dict:
    logging.warning("Using fallback response – LLM unavailable")
    print("Using fallback response – LLM unavailable")
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
# UI HELPERS
# -----------------------------------------------------------------------------

def render_support_card(link: Dict):
    """
    Render one support service card without ever letting raw HTML from the LLM
    leak into the DOM.
    """
    priority = (link.get("priority") or "medium").lower()

    if priority == "high":
        bg = "#fef2f2"
        border = "#dc2626"
        badge_html = "<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:999px;font-size:0.75rem;font-weight:600;'>HIGH PRIORITY</span>"
    elif priority == "medium":
        bg = "#fffbeb"
        border = "#f59e0b"
        badge_html = ""
    else:
        bg = "#f9fafb"
        border = "#e5e7eb"
        badge_html = ""

    name = clean_text_for_html(link.get("name", "Support Service"))
    desc = clean_text_for_html(link.get("description", ""))
    phone_raw = link.get("phone")
    phone = clean_text_for_html(phone_raw) if phone_raw else ""
    url = link.get("url") or "#"

    phone_bit = f"📞 <strong>{phone}</strong>&nbsp;&nbsp;" if phone else ""

    # IMPORTANT: no leading newline or indentation – HTML starts at column 0
    card_html = (
        f"<div style='border-radius:12px;border:1px solid {border};"
        f"background:{bg};padding:1.1rem 1.25rem;margin-bottom:1rem;color:#111827;'>"
        f"  <div style='display:flex;align-items:center;justify-content:space-between;gap:0.5rem;'>"
        f"    <strong style='font-size:1.05rem;color:#111827;'>{name}</strong>"
        f"    {badge_html}"
        f"  </div>"
        f"  <div style='margin:0.4rem 0 0.6rem 0;color:#4b5563;font-size:0.95rem;'>"
        f"    {desc}"
        f"  </div>"
        f"  <div style='margin:0;font-size:0.95rem;color:#111827;'>"
        f"    {phone_bit}"
        f"    🔗 <a href='{html.escape(url)}' target='_blank'>Visit website</a>"
        f"  </div>"
        f"</div>"
    )

    st.markdown(card_html, unsafe_allow_html=True)

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

    payload = st.session_state.llm_payload or {}
    llm_response = st.session_state.llm_response or {}

    # If LLM gave something weird (string / list), drop back to fallback
    if not isinstance(llm_response, dict):
        print("Unexpected LLM response type:", type(llm_response), llm_response)
        st.error("❌ Unexpected response format from support agent. Using fallback summary.")
        llm_response = get_fallback_response(payload)

    st.success(f"📋 **Reference:** `{st.session_state.reference}` — Save this for your records")

    assessment = payload.get("assessment", {})
    score = assessment.get("total_score", 0)
    risk_level = assessment.get("risk_level", "MEDIUM")
    risk_desc = assessment.get("risk_description", "")
    response_time = assessment.get("recommended_response_time", "Within 10 working days")

    if risk_level == "HIGH":
        risk_class = "risk-high"
    elif risk_level == "MEDIUM":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    tab_user, tab_officer = st.tabs(["👤 Your Support & Advice", "👔 Officer View"])

    # ---------------- USER VIEW ----------------
    with tab_user:
        user_resp_raw = llm_response.get("user_response", {})

        if isinstance(user_resp_raw, dict):
            user_resp = user_resp_raw
            greeting_src = user_resp.get(
                "greeting",
                "Thank you for completing this assessment."
            )
        else:
            # If model just gave a string, treat it as the greeting text
            print("user_response is not a dict:", type(user_resp_raw), user_resp_raw)
            user_resp = {}
            greeting_src = str(user_resp_raw) or "Thank you for completing this assessment."

        greeting = clean_text_for_html(greeting_src)
        st.markdown("### ")
        st.markdown(greeting)

        emergency_raw = user_resp.get("emergency_note") if isinstance(user_resp, dict) else None
        if emergency_raw:
            emergency = clean_text_for_html(emergency_raw)
            st.error(f"🆘 **Urgent:** {emergency}")

        st.markdown("---")
        st.markdown("### Support Services For You")

        support_links = user_resp.get("support_links", []) if isinstance(user_resp, dict) else []
        if not isinstance(support_links, list):
            print("support_links not a list:", type(support_links), support_links)
            support_links = []

        for link in support_links:
            if isinstance(link, dict):
                render_support_card(link)
            else:
                # Log but don't crash if the model put something odd in the list
                print("Non-dict support link:", type(link), link)

        st.markdown("---")
        st.markdown("### What Happens Next")

        next_steps_src = (
            user_resp.get("next_steps")
            if isinstance(user_resp, dict)
            else None
        )
        if not next_steps_src:
            next_steps_src = f"A housing support officer will review your case within {response_time}."

        next_steps = clean_text_for_html(next_steps_src)
        st.markdown(next_steps)

        if risk_level == "HIGH":
            st.error(f"⚡ **Priority Case** — Aim to contact within {response_time}")
        elif risk_level == "MEDIUM":
            st.warning(f"📞 **Case Review** — Expected response within {response_time}")
        else:
            st.info(f"📧 **Standard Pathway** — Response within {response_time}")

    # ---------------- OFFICER VIEW ----------------
    with tab_officer:
        officer_raw = llm_response.get("officer_summary", {})
        officer_resp = officer_raw if isinstance(officer_raw, dict) else {}
        if not isinstance(officer_raw, dict):
            print("officer_summary is not a dict:", type(officer_raw), officer_raw)

        st.markdown("### Risk Assessment")

        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div class="score-display">{score}/100</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="text-align: center; margin-top: 0.5rem;">
                    <span class="risk-badge {risk_class}">{risk_level} RISK</span>
                </div>
            """, unsafe_allow_html=True)
            if risk_desc:
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
            if q_id in payload.get("category_scores", {}):
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
