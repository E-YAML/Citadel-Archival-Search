import streamlit as st
import json
import uuid
import requests
from typing import Dict, Any

# Set premium page configs
st.set_page_config(
    page_title="ASOIAF Citadel Archival Search",
    page_icon="📜",
    layout="centered",
)

# Premium Custom CSS Injection for high-end maester theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;0,800;1,400&display=swap');
    
    .stApp {
        background-color: #0d0e12;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
    }
    
    .main-title {
        font-size: 2.6rem;
        background: linear-gradient(135deg, #a5f3fc 0%, #38bdf8 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .sidebar-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .divider-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #f59e0b, transparent);
        margin: 15px 0 25px 0;
    }
</style>
""", unsafe_allow_html=True)

# Render main header layout
st.markdown('<div class="main-title">Citadel Archival Search</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Self-Corrective RAG Intelligence Engine of Westeros</div>', unsafe_allow_html=True)
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# Initialize session state structures
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar panel configurations
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Citadel Registry")
    st.markdown("Accessing archival records from the Citadel at Oldtown. Our agent verifies accuracy and filters misinformation automatically.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**Session Lineage ID**")
    st.code(st.session_state.thread_id, language="text")
    st.caption("Enables stateful, thread-isolated conversational memory inside the graph checkpointer.")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Clear Archival Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query submission logic
if user_prompt := st.chat_input("Ask a question about Westeros lore..."):
    # Append user prompt to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Begin streaming assistant response
    with st.chat_message("assistant"):
        # Setup status log to show reasoning cycles
        status_box = st.status("Initializing Maester's search...", expanded=True)
        response_placeholder = st.empty()
        
        full_response = ""
        api_url = "http://localhost:8000/api/chat/stream"
        payload = {
            "message": user_prompt,
            "thread_id": st.session_state.thread_id
        }

        try:
            # Connect to FastAPI streaming backend endpoint
            with requests.post(api_url, json=payload, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    status_box.update(label="Search Failed", state="error")
                    st.error(f"Error: API returned status code {response.status_code}")
                else:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode("utf-8")

                        # Parse Server-Sent Events (SSE) data frame
                        if decoded_line.startswith("data:"):
                            raw_json = decoded_line[5:].strip()
                            try:
                                frame = json.loads(raw_json)
                                event_type = frame.get("event")
                                content = frame.get("data")

                                if event_type == "node_start":
                                    if content == "retrieve":
                                        status_box.write("📚 Retrieving contextual references from databases...")
                                    elif content == "grade_documents":
                                        status_box.write("⚖️ Checking source documents for relevance...")
                                    elif content == "generate":
                                        status_box.write("✍️ Formulating answer using verified facts...")
                                    elif content == "rewrite":
                                        status_box.write("🔄 Documents irrelevant. Optimizing query coordinates for retry...")

                                elif event_type == "node_end":
                                    if content == "grade_documents":
                                        status_box.write("✅ Document review completed.")

                                elif event_type == "token":
                                    # Auto-collapse processing block on first generated token
                                    status_box.update(label="Processing Complete", state="complete", expanded=False)
                                    full_response += content
                                    response_placeholder.markdown(full_response)

                                elif event_type == "error":
                                    status_box.update(label="Processing Error", state="error")
                                    st.error(f"Backend Graph Error: {content}")

                            except json.JSONDecodeError:
                                # Skip invalid lines
                                pass

            # Save assistant output
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            status_box.update(label="Connection Failed", state="error")
            st.error(f"Could not connect to the backend server: {str(e)}")
            st.caption("Please check if the FastAPI server is running on http://localhost:8000")
