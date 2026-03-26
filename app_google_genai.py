"""
app.py — Aahar AI · Personalised Indian Nutrition & Fitness Coach
Supports: Anthropic (Claude) and Google Gemini APIs
"""

import streamlit as st
from config import SYSTEM_PROMPT, build_user_prompt

# ─────────────────────────────────────────────────────────────────────────────
# Package-level imports (safe — never crash at startup)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from anthropic import Anthropic as _Anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STEPS = ["Profile", "Activity", "Diet", "Lifestyle", "Constraints", "Review & Generate"]

DEFAULT_GEMINI_KEY = ""

PROVIDER_META = {
    "Google Gemini": {
        "icon": "🔷",
        "placeholder": "AIza...",
        "chat_model": "gemini-2.0-flash",
        "fast_model": "gemini-2.0-flash",
        "help": "aistudio.google.com",
        "badge_color": "#60A5FA",
        "badge_bg": "rgba(66,133,244,0.12)",
        "badge_border": "rgba(66,133,244,0.35)",
    },
    "Anthropic (Claude)": {
        "icon": "🔶",
        "placeholder": "sk-ant-...",
        "chat_model": "claude-opus-4-5",
        "fast_model": "claude-haiku-4-5-20251001",
        "help": "console.anthropic.com",
        "badge_color": "#F59E0B",
        "badge_bg": "rgba(217,119,6,0.12)",
        "badge_border": "rgba(217,119,6,0.35)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Key verification — actually hits the API
# ─────────────────────────────────────────────────────────────────────────────
def verify_key(provider: str, key: str) -> tuple:
    """Returns (success: bool, message: str)"""
    if provider == "Anthropic (Claude)":
        if not _HAS_ANTHROPIC:
            return False, "Package missing — run: pip install anthropic"
        try:
            client = _Anthropic(api_key=key)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "✓ Anthropic key verified"
        except Exception as exc:
            err = str(exc)
            if any(x in err for x in ["401", "403", "authentication", "invalid", "x-api-key"]):
                return False, "Invalid key — check console.anthropic.com"
            return False, f"Error: {err[:150]}"

    elif provider == "Google Gemini":
        if not _HAS_GEMINI:
            return False, "Package missing — run: pip install google-genai"
        try:
            client = _genai.Client(api_key=key)
            client.models.generate_content(
                model="gemini-1.5-flash",
                contents="hi",
                config=_genai_types.GenerateContentConfig(max_output_tokens=5),
            )
            return True, "✓ Gemini key verified"
        except Exception as exc:
            err = str(exc)
            if any(x in err for x in ["400", "403", "API_KEY", "invalid", "key"]):
                return False, "Invalid key — check aistudio.google.com"
            return False, f"Error: {err[:150]}"

    return False, "Unknown provider"


# ─────────────────────────────────────────────────────────────────────────────
# Unified API caller
# ─────────────────────────────────────────────────────────────────────────────
def call_api(messages: list, system: str, max_tokens: int = 4096) -> str:
    """
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns the model's reply as a string.
    """
    provider = st.session_state["provider"]
    key      = st.session_state["keys"][provider]

    if provider == "Anthropic (Claude)":
        if not _HAS_ANTHROPIC:
            raise RuntimeError("anthropic package not installed — run: pip install anthropic")
        client = _Anthropic(api_key=key)
        resp = client.messages.create(
            model=PROVIDER_META[provider]["chat_model"],
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    elif provider == "Google Gemini":
        if not _HAS_GEMINI:
            raise RuntimeError("google-genai not installed — run: pip install google-genai")

        # Convert message history → Gemini chat history (all except last message)
        history = []
        for m in messages[:-1]:
            history.append({
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
            })

        client = _genai.Client(api_key=key)
        chat = client.chats.create(
            model=PROVIDER_META[provider]["chat_model"],
            config=_genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
            history=history,
        )
        resp = chat.send_message(messages[-1]["content"])
        return resp.text

    raise ValueError(f"Unknown provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aahar AI · Indian Nutrition Coach",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --saffron:  #E8540A;
  --turmeric: #F0A500;
  --leaf:     #2D6A4F;
  --leaf-lt:  #40916C;
  --cream:    #FEF9F2;
  --cream-dk: #F5EDE0;
  --ink:      #1C1C1E;
  --muted:    #717171;
  --card:     #FFFFFF;
  --border:   #E9DDD0;
  --shadow:   rgba(0,0,0,0.07);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--cream); }
.block-container { padding-top: 1rem !important; max-width: 1100px; }

/* Hero */
.hero {
  background: linear-gradient(130deg,#B83208 0%,#E8540A 45%,#F0A500 80%,#FFD166 100%);
  border-radius: 20px; padding: 2.6rem 2.4rem 2rem;
  margin-bottom: 1.6rem; color: white; position: relative; overflow: hidden;
  box-shadow: 0 8px 32px rgba(232,84,10,0.26);
}
.hero::before { content:"🌿"; position:absolute; font-size:10rem; right:2rem; top:-1rem; opacity:0.13; }
.hero h1 { color:white !important; font-size:2.5rem; margin:0 0 0.3rem;
           font-family:'Playfair Display',serif; letter-spacing:-0.5px; }
.hero p  { color:rgba(255,255,255,0.87); font-size:1.04rem; margin:0; }
.hero-pill {
  display:inline-block; background:rgba(255,255,255,0.18);
  border:1px solid rgba(255,255,255,0.32); border-radius:20px;
  padding:0.18rem 0.8rem; font-size:0.78rem; font-weight:600;
  margin-bottom:0.65rem; color:white; letter-spacing:0.4px;
}

/* Step bar */
.step-bar { display:flex; align-items:center; margin-bottom:1.3rem; gap:0; }
.step-pill {
  padding:5px 13px; border-radius:20px; font-size:0.76rem;
  font-weight:600; white-space:nowrap; transition:all 0.25s;
}
.step-pill.done   { background:#D1FAE5; color:var(--leaf); }
.step-pill.active { background:var(--saffron); color:white;
                    box-shadow:0 3px 12px rgba(232,84,10,0.32); }
.step-pill.todo   { background:var(--cream-dk); color:var(--muted); }
.step-line        { flex:1; height:2px; background:var(--border); margin:0 3px; }
.step-line.done   { background:var(--leaf-lt); }

/* Buttons */
.stButton > button {
  background:linear-gradient(135deg,var(--saffron),var(--turmeric)) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  font-weight:600 !important; padding:0.5rem 1.5rem !important;
  transition:all 0.2s !important;
}
.stButton > button:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 6px 18px rgba(232,84,10,0.36) !important;
}

/* Sidebar */
div[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0f0f0f 0%,#1a1a1e 100%) !important;
  border-right:1px solid #2a2a2a !important;
}
div[data-testid="stSidebar"] * { color:#ddd !important; }
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color:#fff !important; }
div[data-testid="stSidebar"] .stTextInput input {
  background:#252528 !important; border:1px solid #3a3a3a !important;
  color:white !important; border-radius:8px !important;
}
div[data-testid="stSidebar"] .stSelectbox > div > div {
  background:#252528 !important; border-color:#3a3a3a !important;
}
div[data-testid="stSidebar"] .stButton > button {
  width:100% !important; background:#252528 !important;
  border:1px solid #3a3a3a !important; color:#ddd !important;
  text-align:left !important; font-weight:400 !important;
  border-radius:8px !important; box-shadow:none !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
  background:#2e2e32 !important; border-color:var(--saffron) !important;
  transform:none !important; box-shadow:none !important;
}

/* Cards */
.review-card {
  background:var(--card); border:1px solid var(--border);
  border-radius:13px; padding:1rem 1.2rem; margin-bottom:0.75rem;
  box-shadow:0 1px 8px var(--shadow);
}
.review-card h4 { margin:0 0 0.45rem; color:var(--saffron); font-size:0.92rem; }

/* Chat */
.chat-user {
  background:linear-gradient(135deg,#FFF3E0,#FFE8CC);
  border-radius:14px 14px 4px 14px; padding:0.85rem 1.05rem;
  margin:0.45rem 0; border:1px solid #FFD4A3;
}
.chat-ai {
  background:white; border-radius:14px 14px 14px 4px;
  padding:0.85rem 1.05rem; margin:0.45rem 0;
  border:1px solid var(--border); box-shadow:0 1px 6px var(--shadow);
}

/* Sidebar disclaimer */
.disclaimer {
  font-size:0.74rem; color:#666;
  background:#161618; border:1px solid #2a2a2a;
  border-radius:9px; padding:0.75rem 0.9rem;
  margin-top:0.6rem; line-height:1.55;
}

/* Form labels */
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stNumberInput label, .stRadio label, .stTextInput label,
.stTimeInput label { color:var(--ink) !important; font-weight:500; font-size:0.9rem; }

hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    if "step"          not in st.session_state: st.session_state["step"]          = 0
    if "profile"       not in st.session_state: st.session_state["profile"]       = {}
    if "chat_history"  not in st.session_state: st.session_state["chat_history"]  = []
    if "plan_generated"not in st.session_state: st.session_state["plan_generated"]= False
    if "plan_result"   not in st.session_state: st.session_state["plan_result"]   = ""
    if "provider"      not in st.session_state: st.session_state["provider"]      = "Google Gemini"
    if "quick_prompt"  not in st.session_state: st.session_state["quick_prompt"]  = ""
    # Per-provider key storage — Gemini pre-filled with provided key
    if "keys" not in st.session_state:
        st.session_state["keys"] = {
            "Google Gemini":       DEFAULT_GEMINI_KEY,
            "Anthropic (Claude)":  "",
        }
    # Per-provider verification status
    if "key_status" not in st.session_state:
        st.session_state["key_status"] = {
            "Google Gemini":       None,  # None = not verified, True = ok, False = bad
            "Anthropic (Claude)":  None,
        }

_init()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Configuration")

    # Provider selector
    provider = st.selectbox(
        "AI Provider",
        list(PROVIDER_META.keys()),
        index=list(PROVIDER_META.keys()).index(st.session_state["provider"]),
    )
    if provider != st.session_state["provider"]:
        st.session_state["provider"] = provider
        st.rerun()

    meta = PROVIDER_META[provider]

    # Key input — bound to per-provider slot
    raw_key = st.text_input(
        f"{meta['icon']} {provider} API Key",
        type="password",
        placeholder=meta["placeholder"],
        help=f"Get your key at {meta['help']}",
        value=st.session_state["keys"][provider],
        key=f"key_input_{provider}",
    )

    # Update stored key whenever it changes
    if raw_key != st.session_state["keys"][provider]:
        st.session_state["keys"][provider]      = raw_key
        st.session_state["key_status"][provider] = None  # reset verification

    # Verify button
    if st.button("🔍 Verify Key", key="verify_btn"):
        if not raw_key.strip():
            st.error("Paste a key first.")
        else:
            with st.spinner("Verifying…"):
                ok, msg = verify_key(provider, raw_key.strip())
            st.session_state["key_status"][provider] = ok
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # Status badge
    status = st.session_state["key_status"][provider]
    has_key = bool(st.session_state["keys"][provider].strip())

    if status is True:
        st.markdown(
            f'<div style="background:{meta["badge_bg"]};border:1px solid {meta["badge_border"]};'
            f'border-radius:8px;padding:6px 12px;font-size:0.81rem;font-weight:600;'
            f'color:{meta["badge_color"]};margin-top:4px">'
            f'{meta["icon"]} {provider} · {meta["chat_model"]} ✓</div>',
            unsafe_allow_html=True,
        )
    elif status is False:
        st.warning("⚠ Key invalid — re-enter and verify.")
    elif has_key:
        st.info("ℹ Key entered but not verified yet.")
    else:
        st.warning(f"⚠ Enter your {provider} API key.")

    st.divider()

    # Navigation
    st.markdown("### 🧭 Navigation")
    for i, name in enumerate(STEPS):
        icon = "✅" if i < st.session_state["step"] else ("▶" if i == st.session_state["step"] else "○")
        if st.button(f"{icon} {name}", key=f"nav_{i}"):
            st.session_state["step"] = i
            st.rerun()

    if st.session_state["plan_generated"]:
        st.divider()
        if st.button("💬 Open Chat Coach"):
            st.session_state["step"] = 6
            st.rerun()

    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚕ Aahar AI is for general wellness only — not medical advice.
    Always consult a qualified healthcare professional before
    making significant dietary or exercise changes.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check if current provider is ready to use
# ─────────────────────────────────────────────────────────────────────────────
def _api_ready() -> bool:
    provider = st.session_state["provider"]
    return bool(st.session_state["keys"][provider].strip())


# ─────────────────────────────────────────────────────────────────────────────
# Hero + Step bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-pill">AI-Powered · {st.session_state["provider"]}</div>
  <h1>🥗 Aahar AI</h1>
  <p>Your personalised Indian nutrition &amp; fitness coach — evidence-based, culturally grounded</p>
</div>
""", unsafe_allow_html=True)


def _render_steps(current: int):
    parts = []
    for i, name in enumerate(STEPS):
        cls = "done" if i < current else ("active" if i == current else "todo")
        lbl = "✓" if i < current else str(i + 1)
        parts.append(f'<div class="step-pill {cls}">{lbl} {name}</div>')
        if i < len(STEPS) - 1:
            lc = "done" if i < current else ""
            parts.append(f'<div class="step-line {lc}"></div>')
    st.markdown(f'<div class="step-bar">{"".join(parts)}</div>', unsafe_allow_html=True)

_render_steps(min(st.session_state["step"], len(STEPS) - 1))

p = st.session_state["profile"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Basic Profile
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["step"] == 0:
    st.markdown("### 👤 Basic Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        p["age"]    = st.number_input("Age (years)", 10, 90, int(p.get("age", 25)))
        p["gender"] = st.selectbox("Gender",
            ["Male", "Female", "Non-binary / Prefer not to say"],
            index=["Male", "Female", "Non-binary / Prefer not to say"].index(p.get("gender", "Male")))
    with c2:
        p["height_cm"] = st.number_input("Height (cm)", 100, 230, int(p.get("height_cm", 165)))
        p["weight_kg"] = st.number_input("Current Weight (kg)", 30, 200, int(p.get("weight_kg", 65)))
    with c3:
        p["target_weight"] = st.number_input("Target Weight (kg)", 30, 200, int(p.get("target_weight", 60)))
        p["goal"] = st.selectbox("Primary Goal",
            ["Weight Loss", "Muscle Gain", "Maintenance", "Body Recomposition"],
            index=["Weight Loss", "Muscle Gain", "Maintenance", "Body Recomposition"].index(
                p.get("goal", "Weight Loss")))

    # Live BMI badge
    try:
        h = p["height_cm"] / 100
        bmi = round(p["weight_kg"] / (h ** 2), 1)
        if bmi < 18.5:   cat, col = "Underweight", "#3B82F6"
        elif bmi < 25:   cat, col = "Normal weight","#10B981"
        elif bmi < 30:   cat, col = "Overweight",   "#F59E0B"
        else:            cat, col = "Obese",         "#EF4444"
        st.markdown(
            f'<div style="display:inline-block;background:{col}18;border:1px solid {col}55;'
            f'border-radius:9px;padding:5px 14px;margin:6px 0">'
            f'<b style="color:{col}">BMI {bmi}</b> · {cat}</div>',
            unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("#### 🏥 Health Conditions")
    p["conditions"] = st.multiselect("Select any that apply",
        ["Diabetes (Type 1 or 2)", "PCOS", "Thyroid Disorder",
         "High Blood Pressure", "High Cholesterol", "None"],
        default=p.get("conditions", ["None"]))
    p["medications"] = st.text_input("Medications (optional, comma-separated)", p.get("medications", ""))

    if st.button("Next → Activity ▶"):
        st.session_state["step"] = 1; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Activity & Workout
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 1:
    st.markdown("### 🏃 Activity & Workout")
    c1, c2 = st.columns(2)
    ACTIVITY_OPTS = [
        "Sedentary (desk job, little movement)",
        "Lightly Active (light walks, household chores)",
        "Moderately Active (regular exercise 3-4x/week)",
        "Very Active (intense exercise daily / physical job)",
    ]
    with c1:
        p["activity_level"] = st.radio("Daily Activity Level", ACTIVITY_OPTS,
            index=ACTIVITY_OPTS.index(p.get("activity_level", ACTIVITY_OPTS[0])))
    with c2:
        p["workout_type"] = st.multiselect("Workout Type(s)",
            ["Gym (weights/machines)", "Home Workout", "Cardio (running/cycling)",
             "Yoga", "Sports / Martial arts", "None"],
            default=p.get("workout_type", ["None"]))
        p["workout_days"] = st.slider("Workout days per week", 0, 7, int(p.get("workout_days", 3)))
        p["workout_mins"] = st.slider("Duration per session (min)", 15, 120, int(p.get("workout_mins", 45)), step=5)
        EXP_OPTS = ["Beginner (< 6 months)", "Intermediate (6 months – 2 years)", "Advanced (2+ years)"]
        p["experience"]   = st.selectbox("Fitness Experience", EXP_OPTS,
            index=EXP_OPTS.index(p.get("experience", EXP_OPTS[0])))
        p["equipment"]    = st.multiselect("Available Equipment",
            ["Full Gym", "Dumbbells", "Resistance Bands", "Pull-up Bar", "Kettlebell", "None / Bodyweight only"],
            default=p.get("equipment", ["None / Bodyweight only"]))

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button("◀ Back"):  st.session_state["step"] = 0; st.rerun()
    with col_n:
        if st.button("Next → Diet ▶"): st.session_state["step"] = 2; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Dietary Patterns
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 2:
    st.markdown("### 🍛 Dietary Patterns")
    c1, c2 = st.columns(2)
    DIET_OPTS   = ["Vegetarian", "Eggetarian (Veg + Eggs)", "Non-Vegetarian", "Vegan"]
    REGION_OPTS = ["North Indian", "South Indian",
                   "West Indian (Gujarati/Maharashtrian)",
                   "East Indian (Bengali/Odia)", "Mixed / Pan-Indian"]
    with c1:
        p["diet_type"]    = st.radio("Diet Type", DIET_OPTS,
            index=DIET_OPTS.index(p.get("diet_type", "Vegetarian")))
        p["restrictions"] = st.multiselect("Religious / Cultural Restrictions",
            ["Jain (no root vegetables)", "Halal", "Sattvic (no onion/garlic)", "Kosher", "None"],
            default=p.get("restrictions", ["None"]))
    with c2:
        p["region"]    = st.selectbox("Regional Cuisine Preference", REGION_OPTS,
            index=REGION_OPTS.index(p.get("region", "Mixed / Pan-Indian")))
        p["allergies"] = st.multiselect("Allergies / Intolerances",
            ["Nuts", "Dairy", "Gluten", "Soy", "Seafood", "Eggs", "None"],
            default=p.get("allergies", ["None"]))
        p["dislikes"]  = st.text_input("Food dislikes (comma-separated)", p.get("dislikes", ""))
        p["likes"]     = st.text_input("Favourite foods / must-haves", p.get("likes", ""))

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button("◀ Back"):  st.session_state["step"] = 1; st.rerun()
    with col_n:
        if st.button("Next → Lifestyle ▶"): st.session_state["step"] = 3; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Lifestyle
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 3:
    st.markdown("### 🌙 Lifestyle & Habits")
    c1, c2, c3 = st.columns(3)
    with c1:
        p["meals_per_day"] = st.slider("Meals per day", 2, 6, int(p.get("meals_per_day", 3)))
        SCHED_OPTS = ["Fixed timing", "Flexible / irregular"]
        p["meal_schedule"] = st.radio("Meal Schedule", SCHED_OPTS,
            index=SCHED_OPTS.index(p.get("meal_schedule", "Fixed timing")))
        p["wake_time"]  = st.time_input("Usual wake-up time", value=None)
        p["sleep_time"] = st.time_input("Usual sleep time",   value=None)
    with c2:
        p["sleep_hours"]   = st.slider("Average sleep (hours/night)", 3, 12, int(p.get("sleep_hours", 7)))
        p["sleep_quality"] = st.slider("Sleep quality (1=poor, 5=great)", 1, 5, int(p.get("sleep_quality", 3)))
        p["stress_level"]  = st.slider("Stress level (1=low, 5=high)", 1, 5, int(p.get("stress_level", 2)))
    with c3:
        p["water_liters"] = st.slider("Daily water (litres)", 0.5, 5.0,
            float(p.get("water_liters", 2.0)), step=0.25)
        ALC_OPTS = ["None", "Occasional", "Weekly", "Daily"]
        p["alcohol"]  = st.selectbox("Alcohol", ALC_OPTS,
            index=ALC_OPTS.index(p.get("alcohol", "None")))
        p["smoking"]  = st.radio("Smoking", ["No", "Yes"],
            index=["No", "Yes"].index(p.get("smoking", "No")))

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button("◀ Back"):  st.session_state["step"] = 2; st.rerun()
    with col_n:
        if st.button("Next → Constraints ▶"): st.session_state["step"] = 4; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Practical Constraints
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 4:
    st.markdown("### 🛒 Practical Constraints")
    c1, c2 = st.columns(2)
    BUDGET_OPTS = ["Low (₹3,000–5,000/month)", "Medium (₹5,000–10,000/month)", "High (₹10,000+/month)"]
    SKILL_OPTS  = ["Beginner (basic recipes only)", "Intermediate (comfortable in kitchen)", "Advanced (complex dishes OK)"]
    TIME_OPTS   = ["< 15 minutes (very quick)", "15–30 minutes", "30–45 minutes", "45+ minutes (OK with long preps)"]
    GROCERY_OPTS= ["Daily fresh market (sabzi mandi)", "Weekly supermarket", "Online delivery", "Limited / small town"]
    with c1:
        p["budget_level"]  = st.selectbox("Monthly Food Budget", BUDGET_OPTS,
            index=BUDGET_OPTS.index(p.get("budget_level", BUDGET_OPTS[1])))
        p["daily_budget"]  = st.number_input("Daily food budget ₹ (0 = use range above)",
            0, 2000, int(p.get("daily_budget", 0)), step=50)
        p["cooking_skill"] = st.selectbox("Cooking Skill", SKILL_OPTS,
            index=SKILL_OPTS.index(p.get("cooking_skill", SKILL_OPTS[0])))
    with c2:
        p["cook_time"]      = st.radio("Cooking time per meal", TIME_OPTS,
            index=TIME_OPTS.index(p.get("cook_time", TIME_OPTS[1])))
        p["kitchen_tools"]  = st.multiselect("Kitchen equipment",
            ["Pressure Cooker", "Mixer/Grinder", "Microwave", "Oven",
             "Air Fryer", "Induction", "Gas Stove", "Basic (just pan/kadai)"],
            default=p.get("kitchen_tools", ["Gas Stove", "Pressure Cooker"]))
        p["grocery_access"] = st.radio("Grocery access", GROCERY_OPTS,
            index=GROCERY_OPTS.index(p.get("grocery_access", GROCERY_OPTS[1])))

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button("◀ Back"):  st.session_state["step"] = 3; st.rerun()
    with col_n:
        if st.button("Next → Review ▶"): st.session_state["step"] = 5; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Review & Generate
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 5:
    st.markdown("### 📋 Review Your Profile")

    col1, col2 = st.columns(2)
    with col1:
        try:
            h   = p.get("height_cm", 165) / 100
            bmi = round(p.get("weight_kg", 65) / (h ** 2), 1)
        except Exception:
            bmi = "—"
        st.markdown(f"""
        <div class="review-card">
          <h4>👤 Basic Info</h4>
          <p><b>{p.get('age')} yrs · {p.get('gender')}</b> · BMI {bmi}<br>
          {p.get('height_cm')} cm · {p.get('weight_kg')} kg → Target {p.get('target_weight')} kg<br>
          Goal: <b>{p.get('goal')}</b><br>
          {'⚕ ' + ', '.join(p.get('conditions', [])) if p.get('conditions') and 'None' not in p.get('conditions', []) else ''}
          </p>
        </div>
        <div class="review-card">
          <h4>🍛 Diet</h4>
          <p>{p.get('diet_type')} · {p.get('region')}<br>
          {'Restrictions: ' + ', '.join(p.get('restrictions', [])) if p.get('restrictions') and 'None' not in p.get('restrictions', []) else ''}
          {'<br>Allergies: ' + ', '.join(p.get('allergies', [])) if p.get('allergies') and 'None' not in p.get('allergies', []) else ''}
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="review-card">
          <h4>🏃 Activity</h4>
          <p>{p.get('activity_level', '')[:50]}…<br>
          {p.get('workout_days')} days/week × {p.get('workout_mins')} min<br>
          {p.get('experience', '')}
          </p>
        </div>
        <div class="review-card">
          <h4>🛒 Constraints</h4>
          <p>{p.get('budget_level', '')}<br>Cook time: {p.get('cook_time', '')[:22]}…<br>Skill: {p.get('cooking_skill', '')[:30]}</p>
        </div>
        <div class="review-card">
          <h4>🌙 Lifestyle</h4>
          <p>Sleep {p.get('sleep_hours')}h · Water {p.get('water_liters')}L · Stress {p.get('stress_level')}/5</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    what = st.multiselect(
        "📦 What to generate?",
        ["📅 7-Day Meal Plan", "💪 Weekly Workout Plan",
         "📊 Macro & Calorie Targets", "🛒 Grocery List", "📝 Daily Tips & Habits"],
        default=p.get("generate_items",
                      ["📅 7-Day Meal Plan", "💪 Weekly Workout Plan", "📊 Macro & Calorie Targets"]),
    )
    p["generate_items"] = what

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button("◀ Back"): st.session_state["step"] = 4; st.rerun()
    with col_n:
        go = st.button("🚀 Generate My Personalised Plan")

    if go:
        if not _api_ready():
            st.error(f"⚠ Enter your {st.session_state['provider']} API key in the sidebar first.")
        elif not what:
            st.warning("Select at least one item to generate.")
        else:
            with st.spinner(f"🧠 Crafting your plan with {st.session_state['provider']}…"):
                try:
                    user_prompt = build_user_prompt(p)
                    result = call_api(
                        [{"role": "user", "content": user_prompt}],
                        SYSTEM_PROMPT,
                        max_tokens=4096,
                    )
                    st.session_state["plan_result"]   = result
                    st.session_state["plan_generated"] = True
                    st.session_state["chat_history"]   = [
                        {"role": "user",      "content": user_prompt},
                        {"role": "assistant", "content": result},
                    ]
                    st.session_state["step"] = 6
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ {exc}")
                    st.info("💡 Make sure your API key is correct and the relevant package is installed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results & Chat
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["step"] == 6:
    if not st.session_state["plan_generated"]:
        st.warning("No plan yet — complete your profile first.")
        if st.button("← Go to Profile"):
            st.session_state["step"] = 0; st.rerun()
    else:
        tab1, tab2 = st.tabs(["📄 Your Plan", "💬 Chat with Coach"])

        with tab1:
            st.markdown("### 🎉 Your Personalised Plan")
            prov  = st.session_state["provider"]
            model = PROVIDER_META[prov]["chat_model"]
            meta  = PROVIDER_META[prov]
            st.markdown(
                f'<div style="margin-bottom:1rem">'
                f'<span style="background:{meta["badge_bg"]};border:1px solid {meta["badge_border"]};'
                f'border-radius:8px;padding:4px 12px;font-size:0.8rem;font-weight:600;color:{meta["badge_color"]}">'
                f'{meta["icon"]} Generated by {prov} · {model}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state["plan_result"])

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇ Download Plan (TXT)",
                    data=st.session_state["plan_result"],
                    file_name="aahar_ai_plan.txt", mime="text/plain")
            with c2:
                if st.button("🔄 Regenerate"):
                    st.session_state["plan_generated"] = False
                    st.session_state["step"] = 5; st.rerun()

        with tab2:
            st.markdown("### 💬 Ask Your AI Coach")
            st.caption("Substitutions, adjustments, follow-ups — ask anything!")

            history = st.session_state["chat_history"]
            if len(history) >= 2:
                with st.expander("📄 Initial Plan", expanded=False):
                    st.markdown(history[1]["content"])

            for msg in history[2:]:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai">🥗 <b>Aahar AI:</b><br>{msg["content"]}</div>',
                                unsafe_allow_html=True)

            st.markdown("**⚡ Quick questions:**")
            qcols = st.columns(3)
            QUICK = [
                "Suggest a paneer substitute",
                "Missed workout — adjust the week?",
                "High-protein breakfast swap",
                "Wedding this weekend — modify Sunday?",
                "How much water post-workout?",
                "Simple grocery list for this week",
            ]
            for i, q in enumerate(QUICK):
                with qcols[i % 3]:
                    if st.button(q, key=f"qp_{i}"):
                        st.session_state["quick_prompt"] = q

            user_input = st.chat_input("Ask your coach anything…")
            if st.session_state["quick_prompt"]:
                user_input = st.session_state["quick_prompt"]
                st.session_state["quick_prompt"] = ""

            if user_input:
                if not _api_ready():
                    st.error("⚠ API key required — add it in the sidebar.")
                else:
                    with st.spinner("Coach is thinking…"):
                        try:
                            history.append({"role": "user", "content": user_input})
                            reply = call_api(history, SYSTEM_PROMPT, max_tokens=1024)
                            history.append({"role": "assistant", "content": reply})
                            st.session_state["chat_history"] = history
                            st.rerun()
                        except Exception as exc:
                            history.pop()  # remove failed user message
                            st.error(f"❌ {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#aaa;font-size:0.76rem;">'
    '🥗 Aahar AI · Personalised Indian Nutrition &amp; Fitness · '
    'Supports Claude (Anthropic) &amp; Gemini (Google) · '
    'Not a substitute for medical advice</p>',
    unsafe_allow_html=True,
)
