import streamlit as st
from io import StringIO
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
import requests, os, faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import time
import cProfile
import pstats
import textstat

load_dotenv()

# ---------------- CSS -----------------
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 450px; max-width: 450px; }
</style>
""", unsafe_allow_html=True)

# ---------------- INIT -----------------
st.title("🤖 RefineCode.ai – AI Agent Demo Mode")
st.write("Now featuring tools + vector memory! 🧠⚙️")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------- EMBEDDING MODEL -----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)
stored_texts = []

# ---------------- TOOL SIMULATION -----------------
def code_complexity_analyzer(code_text):
    """Estimates code complexity using text statistics."""
    score = textstat.flesch_reading_ease(code_text)
    if score < 30:
        level = "🧩 Hard to read – refactoring is must."
    elif score < 60:
        level = "🪶 Medium complexity."
    else:
        level = "✨ Easy to read and well structured even without refactoring."
    return f"Readability score before refactoring was: {score:.1f}\n{level}"

# ---------------- SIDEBAR -----------------
with st.sidebar:
    st.header("💬 Enter/Upload Your Code")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Hi, How can I help you today?", height=50)
        uploaded_file = st.file_uploader("Attach a Python file (optional)", type=["py"])
        submitted = st.form_submit_button("Submit")

if not submitted:
    st.stop()

if uploaded_file:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text_data = stringio.read()
else:
    text_data = ""

query = user_input + "\n" + text_data

# ---------------- VECTOR STORAGE -----------------
def store_in_vector_db(text: str):
    embedding = embedder.encode([text])
    index.add(np.array(embedding, dtype=np.float32))
    stored_texts.append(text)

def search_in_vector_db(query: str, k=1):
    if len(stored_texts) == 0:
        return []
    embedding = embedder.encode([query])
    D, I = index.search(np.array(embedding, dtype=np.float32), k)
    return [stored_texts[i] for i in I[0]]

# store user query for future memory
store_in_vector_db(query)
retrieved_context = search_in_vector_db(query, k=1)
context_text = retrieved_context[0] if retrieved_context else "No memory yet."

# ---------------- TOOL DECISION -----------------
tool_result = None
#if "fetch doc" in user_input.lower() or "python doc" in user_input.lower():
#    tool_result = fetch_python_doc(user_input.split()[-1])

if "code readability" in user_input.lower() or "code complexity" in user_input.lower():
    tool_result = code_complexity_analyzer(user_input + "\n" + text_data)
# ---------------- CHAT COMPLETION -----------------
messages = [
    {"role": "system", "content": "You are a professional AI Code Refactoring and Debugging Assistant. "
                "You are specialized in Python best practices, clean code principles (PEP8, SOLID), performance optimization, bug detection, and automated refactoring. "
                "You can help users improve code readability, maintainability, and efficiency while preserving functionality. "
                "You are also skilled in modularization, design patterns, type hinting, test-driven development (TDD), and documentation improvements. "

                "Your role is to:"
                "- Analyze Python code provided by the user."
                "- Suggest improvements in structure, style, and performance."
                "- Detect possible bugs, anti-patterns, or vulnerabilities."
                "- Refactor the code into cleaner, more efficient versions."
                "- Provide detailed explanations for your changes so the user can learn."
                "- If requested, output results in different formats (markdown, text, code blocks, or even documentation-ready format)."
                "- When applicable, show comparisons between the original and refactored code in a tabular format."
                "- Always show code complexity and readability score as per textstat.flesch_reading_ease"

                "If a user asks about anything unrelated to Python, debugging, or refactoring, reply:"
                "I am here to help with Python code refactoring, debugging, and clean coding practices. "
                "Please share your code or ask about refactoring, optimization, or debugging."
},
    {"role": "system", "content": f"Relevant past context:\n{context_text}"},
    {"role": "user", "content": query},
]
if tool_result:
    messages.append({"role": "system", "content": f"Tool output: {tool_result}"})

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"
)

response = chat_completion.choices[0].message.content

# ---------------- DISPLAY -----------------
st.subheader("🧠 Agent Thought Process")
st.code(f"""
🧩 Context Retrieved: {context_text[:200]}...
🔧 Tool Used: {'Yes' if tool_result else 'No'}
""")

st.subheader("💬 LLM Response")
st.markdown(response)

if tool_result:
    st.info(f"🔍 Tool Output: {tool_result}")  
