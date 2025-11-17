# app.py
import streamlit as st
import yaml
import time
import threading
import pandas as pd
import os

from google import genai
from google.genai import types
from openai import OpenAI

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(page_title="Assertion–Reason Generator (Costed)", page_icon="🧠", layout="wide")
st.title("🧠 Assertion–Reason Generator — Gemini 2.5 Pro + GPT-5 (Costed, Modular)")

# ====================================================
# API KEY INPUT
# ====================================================
st.markdown("### 🔐 Enter API Keys")
OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
st.markdown("---")

# ====================================================
# PRICING (FORCED ≤200k TIER)
# ====================================================
GPT5_PRICES = {
    "input": 1.25,
    "cached_input": 0.125,
    "output": 10.00
}

GEMINI_PRICES = {
    "input": 1.25,
    "output": 10.00     # includes thinking tokens
}

# ====================================================
# PROMPT TEMPLATE LOADING
# ====================================================
if not os.path.exists("prompts.yaml"):
    st.error("⚠️ Missing prompts.yaml in current folder")
    st.stop()

with open("prompts.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if "prompt" not in data:
    st.error("prompts.yaml must contain a top-level 'prompt' key")
    st.stop()

PROMPT_TEMPLATE = data["prompt"]

# ====================================================
# INPUT FORM
# ====================================================
with st.form("input_form"):
    subject = st.text_input("📘 Subject", "Mathematics")
    chapter = st.text_input("📖 Chapter", "Real Numbers")
    num_questions = st.number_input("🔢 Number of Questions", min_value=1, max_value=20, value=5)

    key_concepts = st.text_area(
        "🧩 Key Concepts (one per line)",
        "Euclid’s Division Lemma\nFundamental Theorem of Arithmetic\nIrrational Numbers"
    )

    generate_btn = st.form_submit_button("🚀 Generate Questions")

# ====================================================
# PROMPT BUILDER
# ====================================================
def build_prompt(subject, chapter, num_questions, key_concepts):
    inputs = {
        "subject": subject,
        "chapter": chapter,
        "num_questions": num_questions,
        "key_concepts": key_concepts
    }

    try:
        prompt = PROMPT_TEMPLATE.format_map(inputs)
    except:
        prompt = PROMPT_TEMPLATE
        for k, v in inputs.items():
            prompt = prompt.replace(f"{{{{{k}}}}}", str(v))

    return f"{prompt}\n\nGenerate {num_questions} Assertion–Reason questions."

# ====================================================
# TOKEN EXTRACTION
# ====================================================
def extract_gpt_tokens(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {"input":0, "cached_input":0, "output":0}

    input_tokens = int(getattr(usage, "input_tokens", 0))
    output_tokens = int(getattr(usage, "output_tokens", 0))

    cached = 0
    try:
        cached = usage.input_tokens_details.cached_tokens
    except:
        pass

    return {
        "input": input_tokens,
        "cached_input": cached,
        "output": output_tokens
    }


def extract_gemini_tokens(response):
    um = getattr(response, "usage_metadata", None)
    if not um:
        return {"input":0, "candidates":0, "thinking":0, "output_total":0}

    inp = int(getattr(um, "prompt_token_count", 0))
    cand = int(getattr(um, "candidates_token_count", 0))
    think = int(getattr(um, "thoughts_token_count", 0))

    return {
        "input": inp,
        "candidates": cand,
        "thinking": think,
        "output_total": cand + think
    }

# ====================================================
# COST CALCULATORS
# ====================================================
def calculate_gpt_cost(tokens):
    normal_input = max(0, tokens["input"] - tokens["cached_input"])

    return {
        "normal_input": (normal_input / 1_000_000) * GPT5_PRICES["input"],
        "cached_input": (tokens["cached_input"] / 1_000_000) * GPT5_PRICES["cached_input"],
        "output":       (tokens["output"] / 1_000_000) * GPT5_PRICES["output"]
    }


def calculate_gemini_cost(tokens):
    return {
        "input": (tokens["input"] / 1_000_000) * GEMINI_PRICES["input"],
        "candidates": (tokens["candidates"] / 1_000_000) * GEMINI_PRICES["output"],
        "thinking":   (tokens["thinking"] / 1_000_000) * GEMINI_PRICES["output"],
        "output_total": (tokens["output_total"] / 1_000_000) * GEMINI_PRICES["output"]
    }

# ====================================================
# COST TABLE BUILDERS
# ====================================================
def build_gpt_cost_table(tokens, costs):
    rows = [
        ["Input (non-cached)", tokens["input"] - tokens["cached_input"], f"${GPT5_PRICES['input']}/1M", costs["normal_input"]],
        ["Cached Input", tokens["cached_input"], f"${GPT5_PRICES['cached_input']}/1M", costs["cached_input"]],
        ["Output (incl. reasoning)", tokens["output"], f"${GPT5_PRICES['output']}/1M", costs["output"]],
        ["TOTAL", tokens["input"] + tokens["output"], "-", sum(costs.values())]
    ]

    df = pd.DataFrame(rows, columns=["Component", "Tokens", "Price/1M", "Cost"])
    df["Cost"] = df["Cost"].apply(lambda x: f"${x:.6f}")
    return df


def build_gemini_cost_table(tokens, costs):
    correct_total_cost = costs["input"] + costs["output_total"]

    rows = [
        ["Input", tokens["input"], f"${GEMINI_PRICES['input']}/1M", costs["input"]],
        ["Candidates", tokens["candidates"], f"${GEMINI_PRICES['output']}/1M", costs["candidates"]],
        ["Thinking", tokens["thinking"], f"${GEMINI_PRICES['output']}/1M", costs["thinking"]],
        ["Total Output (billed)", tokens["output_total"], f"${GEMINI_PRICES['output']}/1M", costs["output_total"]],
        ["TOTAL", tokens["input"] + tokens["output_total"], "-", correct_total_cost]
    ]

    df = pd.DataFrame(rows, columns=["Component", "Tokens", "Price/1M", "Cost"])
    df["Cost"] = df["Cost"].apply(lambda x: f"${x:.6f}")
    return df


# ====================================================
# MODEL RUNNERS
# ====================================================
gpt_result = {}
gemini_result = {}

gpt_done = threading.Event()
gemini_done = threading.Event()
# ====================================================
# MODEL RUNNERS (SAFE, NO THREADING)
# ====================================================

def run_gpt(prompt, api_key):
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "medium"},
            input=prompt
        )
        text = response.output_text or str(response)
        return {"raw": response, "text": text}

    except Exception as e:
        return {"raw": None, "text": f"[GPT-5 error] {e}"}


def run_gemini(prompt, api_key):
    client = genai.Client(api_key=api_key)
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=4000
            )
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt],
            config=config
        )

        # Extract text
        text = ""
        for c in response.candidates or []:
            for p in c.content.parts or []:
                if hasattr(p, "text"):
                    text += p.text + "\n\n"

        return {"raw": response, "text": text.strip()}

    except Exception as e:
        return {"raw": None, "text": f"[Gemini error] {e}"}


# ====================================================
# MAIN ORCHESTRATOR (SEQUENTIAL + STABLE)
# ====================================================

def orchestrate(final_prompt):
    st.info("Running GPT-5...")
    gpt_result = run_gpt(final_prompt, OPENAI_API_KEY)

    st.info("Running Gemini 2.5 Pro...")
    gem_result = run_gemini(final_prompt, GEMINI_API_KEY)

    # ===========================
    # Extract tokens & costs
    # ===========================
    gpt_tokens = extract_gpt_tokens(gpt_result["raw"])
    gem_tokens = extract_gemini_tokens(gem_result["raw"])

    gpt_costs = calculate_gpt_cost(gpt_tokens)
    gem_costs = calculate_gemini_cost(gem_tokens)

    total_gpt = sum(gpt_costs.values())
    total_gem = sum(gem_costs.values())
    total_all = total_gpt + total_gem

    st.success("Both models finished ✔")

    # ===========================
    # OUTPUTS
    # ===========================
    st.markdown("## 🧠 Model Outputs")

    st.subheader("GPT-5 Output")
    st.write(gpt_result["text"])

    st.subheader("Gemini 2.5 Pro Output")
    st.write(gem_result["text"])

    # ===========================
    # COST TABLES
    # ===========================
    st.markdown("---")
    st.markdown("## 💰 Cost Breakdown")

    st.markdown("### GPT-5 Cost")
    st.table(build_gpt_cost_table(gpt_tokens, gpt_costs))

    st.markdown("### Gemini 2.5 Pro Cost")
    st.table(build_gemini_cost_table(gem_tokens, gem_costs))

    st.markdown("---")
    st.markdown(f"### Final Total Cost: **${total_all:.6f}**")


    # st.markdown("### 🔍 Final Prompt")
    # st.code(final_prompt)

# ====================================================
# RUN
# ====================================================
if generate_btn:
    if not OPENAI_API_KEY or not GEMINI_API_KEY:
        st.error("Enter both API keys.")
        st.stop()

    prompt = build_prompt(subject, chapter, num_questions, key_concepts)
    orchestrate(prompt)
