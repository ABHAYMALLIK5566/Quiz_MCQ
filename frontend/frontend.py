import streamlit as st
import requests
import os
import re

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def clean_mcq_output(raw_text: str) -> str:
    # Convert all literal '\n' to real newlines
    text = raw_text.replace('\\n', '\n')
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Ensure 50 dashes are followed by two newlines
    text = re.sub(r'-{5,}\n+', '-' * 50 + '\n\n', text)
    return text.strip()

def main():
    st.set_page_config(page_title="Spiritual MCQ Generator", page_icon="🧘", layout="centered")
    if "token" not in st.session_state:
        st.session_state.token = None
    if not st.session_state.token:
        show_auth()
    else:
        show_main_interface()

def show_auth():
    st.title("Spiritual MCQ Generator")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                response = requests.post(
                    f"{BASE_URL}/token",
                    data={"username": email, "password": password, "grant_type": "password"}
                )
                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Login failed")
    with tab2:
        with st.form("register"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            api_key = st.text_input("Gemini API Key", type="password", help="Get your Gemini API key from Google AI Studio.")
            if st.form_submit_button("Register"):
                response = requests.post(
                    f"{BASE_URL}/register",
                    json={"email": email, "password": password, "api_key": api_key}
                )
                if response.status_code == 200:
                    st.success("Registration successful! Please login")
                else:
                    st.error("Registration failed")

def show_main_interface():
    st.title("Generate Spiritual MCQs")
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, DOC, DOCX, PPT, PPTX, TXT)",
        type=["pdf", "txt", "doc", "docx", "ppt", "pptx"]
    )
    
    with st.expander("⚙️ Settings"):
        num_questions = st.slider("Number of Questions", 1, 50, 10)
        difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"])
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1)
        with col2:
            end_page = st.number_input("End Page", min_value=1, value=1)
    
    if uploaded_file and st.button("Generate MCQs"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {
            "difficulty": difficulty,
            "num_questions": num_questions,
            "start_page": start_page,
            "end_page": end_page
        }
        with st.spinner("Generating spiritual MCQs..."):
            try:
                response = requests.post(
                    f"{BASE_URL}/generate",
                    files=files,
                    data=data,
                    headers=headers
                )
                if response.status_code == 200:
                    st.success("MCQs Generated!")
                    # Clean and format the output before download and display
                    formatted_mcqs_string = clean_mcq_output(response.text)
                    st.download_button(
                        label="Download MCQs",
                        data=formatted_mcqs_string,
                        file_name="spiritual_mcqs.txt",
                        mime="text/plain"
                    )
                    st.code(formatted_mcqs_string)
                else:
                    st.error(f"Generation failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
