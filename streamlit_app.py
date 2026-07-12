"""
Citadel Archival Search — Streamlit App (Collapsed Architecture)
================================================================
Single-process deployment: LangGraph graph runs inline, no FastAPI backend needed.
Streams tokens directly from astream_events() to the Streamlit UI.

Deployment: Streamlit Community Cloud (free tier)
External services: Qdrant Cloud + Neo4j Aura Free + Groq (all free tier)
"""

import os
import sys
import asyncio
import queue
import threading
import uuid
import tempfile

import streamlit as st

# ── Must be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="Citadel Archival Search",
    page_icon="🐉",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit secrets into os.environ ───────────────────────────────────
# This must happen BEFORE importing any backend modules (they read config at
# import time via pydantic-settings which reads os.environ).
_SECRET_KEYS = [
    "GROQ_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "CHECKPOINT_DB_PATH",
]
for _k in _SECRET_KEYS:
    if _k not in os.environ:
        try:
            if _k in st.secrets:
                os.environ[_k] = str(st.secrets[_k])
        except Exception:
            pass

# Default checkpoint to system temp dir if not set (works locally + Streamlit Cloud)
if "CHECKPOINT_DB_PATH" not in os.environ:
    os.environ["CHECKPOINT_DB_PATH"] = os.path.join(
        tempfile.gettempdir(), "citadel_checkpoints.db"
    )

# Add backend package root to Python path
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ── Backend imports (after path + env setup) ───────────────────────────────────
_IMPORT_ERROR: str | None = None
try:
    from app.graph.workflow import app as graph_app
except Exception as _e:
    _IMPORT_ERROR = str(_e)
    graph_app = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Targaryen / Citadel Dark Theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;800&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&display=swap');

    /* ── Base ── */
    .stApp {
        background: radial-gradient(ellipse at top, #1a0000 0%, #0d0a0a 50%, #050303 100%);
        color: #e8d5b7;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.1rem;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'Cinzel', serif;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    /* ── Main title ── */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #daa520 0%, #8b0000 45%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 4px;
        font-family: 'Cinzel', serif;
        letter-spacing: 2px;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    @keyframes titleGlow {
        from { filter: brightness(1); }
        to   { filter: brightness(1.2); }
    }

    /* ── Subtitle ── */
    .subtitle {
        text-align: center;
        color: #a08060;
        font-size: 1rem;
        margin-bottom: 4px;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Cinzel', serif;
    }

    /* ── Divider ── */
    .divider-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b0000 20%, #daa520 50%, #8b0000 80%, transparent);
        margin: 12px 0 28px 0;
        opacity: 0.6;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(10, 5, 5, 0.95) !important;
        border-right: 2px solid rgba(139, 0, 0, 0.3) !important;
    }

    /* ── Sidebar cards ── */
    .sidebar-card {
        background: rgba(30, 10, 10, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(218, 165, 32, 0.15);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        transition: border-color 0.3s ease;
    }
    .sidebar-card:hover {
        border-color: rgba(218, 165, 32, 0.3);
    }

    /* ── Chat message styling ── */
    [data-testid="stChatMessage"] {
        background: rgba(20, 10, 8, 0.7) !important;
        border: 1px solid rgba(218, 165, 32, 0.12) !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin-bottom: 10px !important;
        backdrop-filter: blur(8px);
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {
        border-top: 1px solid rgba(139, 0, 0, 0.3) !important;
        padding-top: 12px;
    }

    /* ── Status box ── */
    [data-testid="stStatusWidget"] {
        background: rgba(20, 10, 8, 0.9) !important;
        border: 1px solid rgba(218, 165, 32, 0.2) !important;
        border-radius: 8px !important;
    }

    /* ── Code blocks ── */
    code {
        background: rgba(218, 165, 32, 0.1) !important;
        color: #daa520 !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.85em;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'Cinzel', serif !important;
        color: #daa520 !important;
        font-size: 0.9rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d0a0a; }
    ::-webkit-scrollbar-thumb { background: #5c1010; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #daa520; }

    /* ── How it works cards ── */
    .how-step {
        background: rgba(20, 10, 8, 0.5);
        border: 1px solid rgba(218, 165, 32, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: border-color 0.3s ease;
    }
    .how-step:hover {
        border-color: rgba(218, 165, 32, 0.25);
    }
    .how-step-number {
        font-family: 'Cinzel', serif;
        color: #8b0000;
        font-weight: 700;
        font-size: 1.2rem;
        margin-right: 8px;
    }
    .how-step-title {
        font-family: 'Cinzel', serif;
        color: #daa520;
        font-size: 1rem;
        font-weight: 600;
    }
    .how-step-desc {
        color: #c4a882;
        font-style: italic;
        margin-top: 2px;
    }

    /* ── Welcome heading ── */
    .welcome-heading {
        font-family: 'Cinzel', serif;
        color: #daa520;
        text-align: center;
        font-size: 1.4rem;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🐉 Citadel Archival Search</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask a Maester — Search the Archives of Westeros</div>', unsafe_allow_html=True)
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ══════════════════════════════════════════════════════════════════════════════
# How It Works — Welcome Section (shown when no messages yet)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown('<div class="welcome-heading">📜 How the Citadel Serves You</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Your Question Arrives", "A raven brings your query to the Citadel. The Maesters begin their work."),
        ("2", "The Archives Are Searched", "Seven tomes across 371 chapters are scanned via arcane vector arts (Qdrant). Lineage records in the iron graph (Neo4j) are consulted for character ties."),
        ("3", "Scrolls Are Examined", "Each retrieved passage is graded for relevance. Only the truest scrolls reach the Grand Maester."),
        ("4", "An Answer Is Composed", "The Grand Maester (Groq) assembles a verified reply, drawing from the selected archives."),
        ("5", "The Raven Returns", "Your answer is delivered, rooted in the source texts and cross-checked against the lineage records."),
    ]

    st.markdown(
        '<div style="max-width: 680px; margin: 0 auto;">',
        unsafe_allow_html=True,
    )
    for num, title, desc in steps:
        st.markdown(
            f'<div class="how-step">'
            f'<span class="how-step-number">{num}.</span> '
            f'<span class="how-step-title">{title}</span>'
            f'<div class="how-step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center; margin-top: 24px; color: #8b0000; '
        'font-family: Cinzel, serif; font-size: 0.85rem; letter-spacing: 1px;">'
        "❝ A maester's word is only as good as the texts he has read. ❞</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider-line" style="margin-top: 28px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🐉 Citadel Registry")
    st.markdown(
        "Accessing archival records spanning the known histories of Westeros. "
        "The Maesters verify every answer before it is delivered."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**📖 The Archive**")
    st.markdown(
        "- A Game of Thrones\n"
        "- A Clash of Kings\n"
        "- A Storm of Swords\n"
        "- A Feast for Crows\n"
        "- A Dance with Dragons\n"
        "- Fire & Blood\n"
        "- Tales of Dunk & Egg"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    with st.expander("📜 How It Works", expanded=False):
        st.markdown(
            "**1. Ask** — A question arrives by raven.\n\n"
            "**2. Search** — The archives (Qdrant) and lineage records (Neo4j) "
            "are consulted.\n\n"
            "**3. Examine** — Retrieved scrolls are graded for relevance.\n\n"
            "**4. Compose** — The Grand Maester (Groq) forms a verified reply.\n\n"
            "**5. Deliver** — Your answer is returned, rooted in the source texts."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("**🔗 Conversational Memory**")
    st.markdown(
        "Ask follow-up questions naturally. The Maesters remember "
        "what was discussed and resolve references like *he*, *that battle*, "
        "or *the same house* from prior conversation."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Session", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # Show import error if backend failed to load
    if _IMPORT_ERROR:
        st.error(f"⚠️ Backend failed to load:\n\n`{_IMPORT_ERROR}`")
        st.caption("Check that all secrets are configured.")


# ══════════════════════════════════════════════════════════════════════════════
# Async Graph Streaming Bridge
# ══════════════════════════════════════════════════════════════════════════════

def stream_graph_events(question: str, thread_id: str, chat_history: list):
    """
    Generator that streams (event_type, data) tuples from the LangGraph graph.

    Runs the async graph in a dedicated background thread with its own event loop,
    using a thread-safe queue to pass events back to this sync generator.
    This pattern is safe with Streamlit's own event loop and avoids nest_asyncio.

    Yields tuples:
        ("node_start", node_name: str)
        ("node_end",   node_name: str)
        ("token",      text: str)
        ("error",      error_msg: str)
    """
    if graph_app is None:
        yield ("error", f"Graph not loaded: {_IMPORT_ERROR}")
        return

    event_queue: queue.Queue = queue.Queue()
    _SENTINEL = object()

    async def _run_graph():
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "question": question,
            "original_question": question,
            "search_retry_count": 0,
            "generation": "",
            "documents": [],
            "is_hallucination": False,
            "chat_history": chat_history,
        }
        try:
            async for event in graph_app.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                event_queue.put(("raw", event))
        except Exception as exc:
            event_queue.put(("error", str(exc)))
        finally:
            event_queue.put(("done", _SENTINEL))

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_graph())
        finally:
            loop.close()

    worker = threading.Thread(target=_thread_target, daemon=True)
    worker.start()

    # Process queue items and yield structured events to the Streamlit UI
    while True:
        kind, payload = event_queue.get()

        if kind == "done":
            break

        if kind == "error":
            yield ("error", payload)
            break

        # Parse LangGraph astream_events v2 event structure
        event = payload
        event_name = event.get("event", "")
        node_name = event.get("name", "")

        if event_name == "on_chain_start" and node_name in (
            "retrieve", "grade_documents", "generate", "rewrite"
        ):
            yield ("node_start", node_name)

        elif event_name == "on_chain_end" and node_name in (
            "retrieve", "grade_documents", "generate", "rewrite"
        ):
            yield ("node_end", node_name)

        elif event_name == "on_chat_model_stream":
            # Only stream tokens from the Maester generator chain (tagged)
            tags = event.get("tags", [])
            if "citadel_generation" in tags:
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield ("token", chunk.content)

        elif event_name == "on_chain_error":
            error_msg = str(event.get("data", {}).get("error", "Unknown graph error"))
            yield ("error", error_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Chat History Rendering
# ══════════════════════════════════════════════════════════════════════════════

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ══════════════════════════════════════════════════════════════════════════════
# Query Submission & Streaming Response
# ══════════════════════════════════════════════════════════════════════════════

if user_prompt := st.chat_input(
    "Ask the Maester a question about Westeros lore...",
    disabled=(_IMPORT_ERROR is not None),
    key="chat_input",
):
    # Guard: backend must be loaded
    if _IMPORT_ERROR:
        st.error(f"Cannot process query — backend failed to load: {_IMPORT_ERROR}")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        status_box = st.status("🐉 A Maester takes up your query...", expanded=True)
        response_placeholder = st.empty()

        full_response = ""
        generation_started = False
        error_occurred = False

        for event_type, data in stream_graph_events(
            user_prompt,
            st.session_state.thread_id,
            st.session_state.messages[-12:],  # last 6 turns
        ):

            if event_type == "node_start":
                if data == "retrieve":
                    status_box.write("📜 Searching the archives and lineage records...")
                elif data == "grade_documents":
                    status_box.write("⚖️ The Maesters examine each scroll for truth...")
                elif data == "generate":
                    status_box.write("✍️ The Grand Maester composes a verified reply...")
                elif data == "rewrite":
                    status_box.write("🔄 The raven returns — query must be refined...")

            elif event_type == "node_end":
                if data == "grade_documents":
                    status_box.write("✅ Scrolls have been examined and approved.")

            elif event_type == "token":
                if not generation_started:
                    status_box.update(
                        label="🐉 The Maester's reply arrives",
                        state="complete",
                        expanded=False,
                    )
                    generation_started = True
                full_response += data
                response_placeholder.markdown(full_response + "▌")  # blinking cursor

            elif event_type == "error":
                status_box.update(label="❌ The Maesters could not complete their work", state="error")
                st.error(f"Graph Error: {data}")
                error_occurred = True
                break

        # Remove blinking cursor from final response
        if full_response:
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        elif not error_occurred:
            status_box.update(
                label="📜 The archives hold no answer",
                state="complete",
                expanded=True,
            )
            response_placeholder.warning(
                "⚠️ The Maesters searched every scroll and lineage record, "
                "but found nothing that speaks to your question. "
                "Perhaps ask the question another way."
            )
