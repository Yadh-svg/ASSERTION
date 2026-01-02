# app.py
import streamlit as st
import yaml
import pandas as pd
import os
from google import genai
from google.genai import types

# ====================================================
# PAGE CONFIG
# ====================================================
# import socket, requests, ssl
# import streamlit as st

# st.title("🔍 Network Diagnostic")

# try:
#     r = requests.get("https://generativelanguage.googleapis.com", timeout=5)
#     st.write("Direct connection:", r.status_code)
# except Exception as e:
#     st.write("Direct connection error:", str(e))

# try:
#     ctx = ssl.create_default_context()
#     with ctx.wrap_socket(socket.socket(), server_hostname="generativelanguage.googleapis.com") as s:
#         s.settimeout(5)
#         s.connect(("generativelanguage.googleapis.com", 443))
#         st.write("TLS handshake OK")
# except Exception as e:
#     st.write("TLS handshake error:", str(e))

st.set_page_config(page_title="Assertion–Reason Generator", page_icon="🧠", layout="wide")
st.markdown(
    """
    <script type="text/javascript"
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    """,
    unsafe_allow_html=True
)
st.title("🧠 Assertion–Reason Generator")

# ====================================================
# API KEY INPUT
# ====================================================
st.markdown("### 🔐 Enter Gemini API Key")
GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")

st.markdown("---")

# ====================================================
# PRICING (FORCED ≤200k TIER)
# ====================================================
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
# New Concept Source Selection (outside form for dynamic updates)
new_concept_source = st.radio(
    "🆕 New Concept Source",
    ["Text Input", "PDF Upload"],
    horizontal=True
)

with st.form("input_form"):
    subject = st.text_input("📘 Subject", "")
    grade = st.text_input("🎓 Grade", "")
    chapter = st.text_input("📖 Chapter", "")
    num_questions = st.number_input("🔢 Number of Questions", min_value=1, max_value=20, value=5)

    topics = st.text_area(
        "📚 Topics",
        ""
    )

    old_concept = st.text_area(
        "📖 Old Concept (Prerequisite Knowledge)",
        ""
    )

    # Conditionally show input based on selection
    if new_concept_source == "Text Input":
        new_concept_text = st.text_area(
            "🧩 New Concept (Current Chapter Content)",
            ""
        )
        new_concept_pdf = None
    else:
        new_concept_pdf = st.file_uploader(
            "📄 Upload New Concept PDF",
            type=["pdf"],
            help="Upload a PDF containing the new concepts covered in this chapter"
        )
        new_concept_text = ""

    additional_notes = st.text_area(
        "📝 Additional Notes (Optional)",
        ""
    )

    generate_btn = st.form_submit_button("🚀 Generate Questions")

# ====================================================
# PROMPT BUILDER
# ====================================================
def build_prompt(subject, grade, chapter, num_questions, old_concept, new_concept, additional_notes, topics, has_pdf=False):
    """
    Build the prompt with new field structure.
    If has_pdf is True, new_concept will indicate PDF is attached.
    """
    # If PDF is uploaded, indicate it in the new_concept field
    if has_pdf:
        new_concept_value = """The new concept content is provided in the attached PDF document. 
Please carefully read and analyze the PDF to understand all topics, subtopics, definitions, theorems, 
formulas, and examples covered in this chapter. Use this PDF content as the primary source for 
generating questions about the new concepts the student is currently learning.But you should only make questions based on the Topic given the pdf is just for reference """
    else:
        new_concept_value = new_concept
    
    inputs = {
        "subject": subject,
        "grade": grade,
        "chapter": chapter,
        "num_questions": num_questions,
        "old_concept": old_concept,
        "new_concept": new_concept_value,
        "additional_notes": additional_notes,
        "topics": topics
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
# COST CALCULATOR
# ====================================================
def calculate_gemini_cost(tokens):
    return {
        "input": (tokens["input"] / 1_000_000) * GEMINI_PRICES["input"],
        "candidates": (tokens["candidates"] / 1_000_000) * GEMINI_PRICES["output"],
        "thinking":   (tokens["thinking"] / 1_000_000) * GEMINI_PRICES["output"],
        "output_total": (tokens["output_total"] / 1_000_000) * GEMINI_PRICES["output"]
    }

# ====================================================
# COST TABLE BUILDER
# ====================================================
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
# GEMINI RUNNER
# ====================================================
gemini_result = {}

def run_gemini(prompt, api_key, pdf_file=None):
    client = genai.Client(api_key=api_key)
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=5000)
        )

        # Prepare content parts
        content_parts = []
        
        # If PDF is provided, add it using Part.from_bytes
        if pdf_file is not None:
            content_parts.append(
                types.Part.from_bytes(
                    data=pdf_file.read(),
                    mime_type='application/pdf'
                )
            )
        
        # Add the prompt
        content_parts.append(prompt)

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=content_parts,
            config=config
        )

        # extract text
        text = ""
        for c in response.candidates or []:
            for p in c.content.parts or []:
                if hasattr(p, "text"):
                    text += p.text + "\n\n"

        gemini_result["raw"] = response
        gemini_result["text"] = text.strip()
    except Exception as e:
        gemini_result["raw"] = None
        gemini_result["text"] = f"[Gemini error] {e}"

# ====================================================
# MAIN ORCHESTRATOR
# ====================================================
def run_generation(final_prompt, pdf_file=None):
    # Run Gemini generation
    overall_status = st.empty()
    overall_status.info("🤖 Generating questions with Gemini 2.5 Pro...")
    
    run_gemini(final_prompt, GEMINI_API_KEY, pdf_file)
    
    overall_status.success("✅ Generation finished!")

    # ===========================
    # Extract tokens & costs
    # ===========================
    gem_tokens = extract_gemini_tokens(gemini_result.get("raw"))
    gem_costs = calculate_gemini_cost(gem_tokens)
    total_cost = sum(gem_costs.values())

    # ===========================
    # OUTPUT
    # ===========================
    st.markdown("## 🧠 Generated Questions")
    st.markdown(gemini_result.get("text", ""), unsafe_allow_html=True)

    # ===========================
    # COST TABLE
    # ===========================
    st.markdown("---")
    st.markdown("## 💰 Cost Breakdown")
    st.table(build_gemini_cost_table(gem_tokens, gem_costs))
    st.markdown(f"### Total Cost: **${total_cost:.6f}**")


# ====================================================
# RUN
# ====================================================
if generate_btn:
    if not GEMINI_API_KEY:
        st.error("Please enter Gemini API Key.")
        st.stop()

    # Validate new concept input
    if new_concept_source == "PDF Upload":
        if new_concept_pdf is None:
            st.error("Please upload a PDF file for new concepts or switch to text input.")
            st.stop()
        # Use PDF
        has_pdf = True
        new_concept_final = ""
        pdf_to_upload = new_concept_pdf
    else:
        # Use text
        has_pdf = False
        new_concept_final = new_concept_text
        pdf_to_upload = None

    prompt = build_prompt(
        subject, 
        grade, 
        chapter, 
        num_questions, 
        old_concept, 
        new_concept_final, 
        additional_notes, 
        topics, 
        has_pdf=has_pdf
    )
    run_generation(prompt, pdf_file=pdf_to_upload)

