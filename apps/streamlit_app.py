"""
streamlit_app.py
-----------------
College Books RAG — a chatbot-style Streamlit UI.

Layout:
  - Left  : chat interface (ask questions, see retrieved chunks as
            assistant messages)
  - Right : file upload panel + index controls
  - Bottom: a live logger panel that streams backend log messages
            (extraction / chunking / indexing progress) in real time

Run:
    streamlit run apps/streamlit_app.py
"""
import logging
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from college_rag.config import Config  # noqa: E402
from college_rag.exceptions import CollegeRAGError  # noqa: E402
from college_rag.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="College Books RAG", layout="wide", page_icon="📚")


# --------------------------------------------------------------------------- #
# Live logger: a logging.Handler that writes straight into a Streamlit
# placeholder, so log lines appear on screen as they are emitted — not just
# after the whole operation finishes.
# --------------------------------------------------------------------------- #
class StreamlitLogHandler(logging.Handler):
    MAX_LINES = 300

    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.lines = []
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            return
        self.lines.append(msg)
        self.lines = self.lines[-self.MAX_LINES:]
        try:
            self.placeholder.code("\n".join(self.lines) or "(no log output yet)", language="log")
        except Exception:
            # Placeholder may not be mounted yet on the very first log line
            pass


def _attach_live_logger(placeholder):
    """Attaches a fresh StreamlitLogHandler to the 'college_rag' logger,
    replacing any handler from a previous run so logs don't duplicate."""
    logger = logging.getLogger("college_rag")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        if isinstance(h, StreamlitLogHandler):
            logger.removeHandler(h)
    handler = StreamlitLogHandler(placeholder)
    logger.addHandler(handler)
    return handler


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

st.title("📚 College Books RAG")
st.caption("Semantic Chunking + FAISS Vector Search — pure retrieval, no LLM involved")

left, right = st.columns([2, 1], gap="large")

# --------------------------------------------------------------------------- #
# RIGHT SIDE — file upload + indexing controls
# --------------------------------------------------------------------------- #
with right:
    st.subheader("📤 Upload books")
    uploaded_files = st.file_uploader(
        "PDF or DOCX files (multiple allowed)",
        type=["pdf", "docx"], accept_multiple_files=True,
    )

    with st.expander("⚙️ Chunking settings", expanded=False):
        min_chars = st.slider("Min chunk size (chars)", 100, 500, 200, 50)
        max_chars = st.slider("Max chunk size (chars)", 500, 2000, 1200, 100)
        threshold = st.slider("Topic-switch sensitivity (%)", 50, 95, 75, 5,
                               help="Higher = more, smaller chunks")
        top_k = st.slider("Chunks to retrieve per question (top-K)", 1, 10, 5)

    build_btn = st.button("🔨 Build index", type="primary", use_container_width=True)

    log_placeholder = st.empty()
    st.session_state.show_logs = st.toggle("🪵 Live logger", value=st.session_state.show_logs)
    if not st.session_state.show_logs:
        log_placeholder.empty()

    if build_btn:
        if not uploaded_files:
            st.warning("Upload at least one PDF or DOCX file first.")
        else:
            handler = _attach_live_logger(log_placeholder if st.session_state.show_logs else st.container())

            config = Config(
                min_chunk_chars=min_chars,
                max_chunk_chars=max_chars,
                breakpoint_percentile=float(threshold),
                default_top_k=top_k,
            )
            with st.spinner("Loading embedding model (first run can take a while)..."):
                pipeline = RAGPipeline(config=config)

            tmp_paths = []
            try:
                for uf in uploaded_files:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_paths.append(tmp.name)

                with st.spinner("Extracting text, chunking, and indexing..."):
                    try:
                        stats = pipeline.build_index_from_files(tmp_paths)
                    except CollegeRAGError as e:
                        st.error(f"❌ {e}")
                        stats = None
            finally:
                for p in tmp_paths:
                    if os.path.exists(p):
                        os.unlink(p)
                logging.getLogger("college_rag").removeHandler(handler)

            if stats:
                st.session_state.pipeline = pipeline
                st.session_state.stats = stats
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"✅ Indexed **{stats.total_chunks} chunks** from "
                        f"**{stats.total_source_files} file(s)**: "
                        f"{', '.join(stats.source_files)}\n\nAsk me anything about these books!"
                    ),
                })
                st.rerun()

    if st.session_state.stats:
        st.divider()
        st.markdown("**Current index**")
        st.metric("Chunks", st.session_state.stats.total_chunks)
        st.metric("Source files", st.session_state.stats.total_source_files)
        for f in st.session_state.stats.source_files:
            st.caption(f"📄 {f}")

# --------------------------------------------------------------------------- #
# LEFT SIDE — chatbot interface
# --------------------------------------------------------------------------- #
with left:
    st.subheader("💬 Chat")

    chat_container = st.container(height=520)
    with chat_container:
        if not st.session_state.messages:
            st.info("Upload your books on the right, build the index, then ask a question here.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your books...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        if not st.session_state.pipeline:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ No index yet — upload your books on the right and click **Build index** first.",
            })
        else:
            results = st.session_state.pipeline.query(question, top_k=top_k)
            if not results:
                answer = "I couldn't find anything relevant to that question in the indexed books."
            else:
                parts = [f"Found **{len(results)}** relevant passage(s):\n"]
                for i, r in enumerate(results, 1):
                    parts.append(
                        f"**#{i} — {r.chunk.location_label()}**  ·  similarity `{r.score:.3f}`\n\n"
                        f"> {r.chunk.text}\n"
                    )
                answer = "\n".join(parts)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()
