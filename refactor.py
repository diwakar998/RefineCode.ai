import streamlit as st
from io import StringIO
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import os

#choice1 = int(input("Enter your choice 1. Groq 2.OpenAI: "))
#if choice1 == 1:
client = Groq(
api_key=os.environ.get("GROQ_API_KEY"),
)
# Title of the app
st.title("RefineCode.ai – clean and professional 💎")

# Write text
st.write("Welcome to your code refinement app!")

# Initialize chat history with a health-focused system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a professional AI Code Refactoring and Debugging Assistant. "
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

                "If a user asks about anything unrelated to Python, debugging, or refactoring, reply:"
                "I am here to help with Python code refactoring, debugging, and clean coding practices. "
                "Please share your code or ask about refactoring, optimization, or debugging."

            )
        }
    ]

with st.sidebar:
    st.header("💬 Enter/Upload Your Code/File")
    with st.form("chat_form", clear_on_submit=True):
        #user_input = st.text_input("Hi, How can I help you today ?", key="input_box")
        user_input=st.text_area("Hi, How can I help you today ?", height=50)
        uploaded_file = st.file_uploader(
        "Attach a supporting file (optional)", 
        type=["py"]
        )
        submitted = st.form_submit_button("Submit")
    
    #user_input=st.text_input("Hi, How can I help you today ?")
    #Submit button to control execution
    if submitted:
        if not user_input:  # Text is mandatory
            st.error("⚠️ Please enter text before submitting.")
        else:
            st.success("✅ Processing your request...")
            #st.write("Text entered:", text_val)    
            if uploaded_file:
                st.write("File uploaded:", uploaded_file.name)
            else:
                st.info("No file uploaded (that’s okay!)")
                
#text_data=""    
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    content = ""
    contentTbl=""
    #st.write(file_type)
    if file_type == "py":
        # Read as text
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        content = stringio.read()
        text_data=content
        #st.write(text_data)
    else:
        st.error("⚠️ Unsupported file type. Please upload a .py file.")
        text_data=""
else:
    text_data=""

messages = st.session_state.messages
messages.append({"role":"user", "content": user_input+text_data})
chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile"
    )

response=chat_completion.choices[0].message.content
st.markdown(response)


