import streamlit as st
import time
from chromadb import PersistentClient
from google import genai
import os

# Page Config
st.set_page_config(
    page_title="College Calendar RAG Assistant",
    page_icon="📅",
    layout="centered"
)

st.title("📅 College Calendar Assistant")
st.caption("Ask anything about the 2022-2023 Academic Calendar, Holidays, and Exams!")

# 1. Initialize Gemini Client & ChromaDB
@st.cache_resource
def init_rag():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    chroma_client = PersistentClient(path="./calendar_vector_db")
    collection = chroma_client.get_or_create_collection(name="college_calendar")
    return client, collection

try:
    client, collection = init_rag()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

# 2. Core RAG Pipeline Function (Notebook Logic Exact Port)
def ask_calendar(user_query):
    months = ["january", "february", "march", "april", "may", "june", 
              "july", "august", "september", "october", "november", "december"]
    
    found_month = None
    for m in months:
        # substring search to catch typos like septamber
        if m[:4] in user_query.lower(): 
            found_month = m.capitalize()
            break

    # Fetch ALL documents from ChromaDB collection
    all_records = collection.get()
    all_chunks = all_records['documents']

    # Filter documents matching ONLY the queried month
    if found_month:
        month_chunks = [
            chunk for chunk in all_chunks 
            if found_month.lower() in chunk.lower()
        ]
    else:
        month_chunks = all_chunks

    # Filter ONLY 'is_holiday: true' or 'true' chunks flexibly
    holiday_chunks = [
        chunk for chunk in month_chunks 
        if "is_holiday" in chunk.lower() and "true" in chunk.lower()
    ]

    # Combine context
    final_context = "\n\n".join(holiday_chunks) if holiday_chunks else "\n\n".join(month_chunks)

    # Strict Prompt
    prompt = f"""
    You are an intelligent College Calendar Assistant.
    You ONLY have data for the 2022-2023 Academic Year.

    CRITICAL RULES:
    1. Check if the user is asking about a specific year (e.g., 2024, 2025, 2026).
    2. If the user asks about ANY year other than 2022 or 2023, IMMEDIATELY state: "Information Not Available in Calendar Context for the requested year."
    3. Do NOT substitute 2022 or 2023 data if the user asks for another year (like 2025).
    4. If the question is valid for 2022-2023, list ALL matching details from the context below accurately without skipping any date or holiday.

    Calendar Context:
    {final_context}

    User Question: {user_query}

    Answer:
    """

    # Retry Mechanism using gemini-3.6-flash
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return f"Error: {e}"

# 3. Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I help you with the college calendar today?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input Box
if user_input := st.chat_input("Ex: How many holidays are in September?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching calendar..."):
            response_text = ask_calendar(user_input)
            st.write(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})