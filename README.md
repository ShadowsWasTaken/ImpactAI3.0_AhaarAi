# ImpactAI3.0_AhaarAi
Ai nutritionist
# 🥗 Aahar AI — Personalised Indian Nutrition & Fitness Coach

An AI-powered nutrition and fitness assistant tailored for Indian users, built with Streamlit.
Now supports **both Anthropic (Claude)** and **Google Gemini** as AI backends.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Visit `http://localhost:8501`

### 4. Configure your API key
In the left sidebar:
- **Select your AI Provider** (Anthropic or Google Gemini)
- Paste your API key

| Provider | Key format | Get it from |
|---|---|---|
| Anthropic (Claude) | `sk-ant-...` | https://console.anthropic.com |
| Google Gemini | `AIza...` | https://aistudio.google.com |

---

## 📁 File Structure

```
nutrition_coach/
├── app.py            # Main Streamlit application
├── config.py         # System prompt + user prompt builder
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## 🎛️ Features

- **Multi-step onboarding** across 5 screens (Profile → Activity → Diet → Lifestyle → Constraints)
- **Dual AI backend**: Anthropic Claude or Google Gemini — switch any time in the sidebar
- **Live BMI indicator** on the profile screen
- **Generates**: 7-day meal plan, workout plan, macro targets, grocery list, daily tips
- **Indian-first**: regional cuisines, ₹ budget, Indian food names, katori/roti portions
- **Chat coach**: follow-up Q&A with quick-prompt buttons after plan generation
- **Download plan** as plain text

---

## 🔑 API Key Security

Your API key is:
- **Never stored to disk** — session memory only
- **Never sent anywhere except the chosen AI provider's API**
- Cleared when you close the browser tab

---

## ⚕️ Disclaimer

Aahar AI provides general wellness guidance only. It is not a substitute for professional medical or dietetic advice. Always consult a qualified healthcare provider before making significant changes to your diet or exercise routine, especially if you have an existing health condition.
