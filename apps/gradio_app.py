"""Gradio app — College Books RAG for Hugging Face Spaces."""

import logging
import os
import sys
import tempfile

import gradio as gr
import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from college_rag.config import Config
from college_rag.exceptions import CollegeRAGError
from college_rag.pipeline import RAGPipeline

logger = logging.getLogger("college_rag")


# ---------------------------------------------------------------------------
# Monkey-patch Gradio 4.40.0 bug: _json_schema_to_python_type crashes when
# schema is a bool instead of a dict (https://github.com/gradio-app/gradio/issues/8756)
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_jstt = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    return _orig_jstt(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        msg = self.format(record)
        self.lines.append(msg)

    def get_logs(self):
        return "\n".join(self.lines[-300:]) or "(no log output yet)"


@spaces.GPU
def build_index(files, min_chars, max_chars, threshold, top_k, pipeline_state):
    if not files:
        return pipeline_state, "⚠️ Upload at least one PDF or DOCX file first.", gr.update(value="No index")

    handler = LogCaptureHandler()
    logger = logging.getLogger("college_rag")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    config = Config(
        min_chunk_chars=min_chars,
        max_chunk_chars=max_chars,
        breakpoint_percentile=float(threshold),
        default_top_k=top_k,
    )

    pipeline = RAGPipeline(config=config)

    tmp_paths = []
    stats = None
    try:
        for f in files:
            suffix = os.path.splitext(f)[1] or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_paths.append(tmp.name)
            tmp.close()
            with open(f, "rb") as src, open(tmp.name, "wb") as dst:
                dst.write(src.read())

        stats = pipeline.build_index_from_files(tmp_paths)
    except CollegeRAGError as e:
        return pipeline, handler.get_logs(), f"❌ {e}"
    except Exception as e:
        return pipeline, handler.get_logs(), f"❌ Unexpected error: {e}"
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)
        logger.removeHandler(handler)

    idx_text = f"**Chunks:** {stats.total_chunks} | **Files:** {stats.total_source_files}"
    return pipeline, handler.get_logs(), gr.update(value=idx_text)


def respond(question, history, pipeline):
    if not question:
        return "", history

    if pipeline is None:
        history.append((question, "⚠️ No index yet — upload your books and click **Build index** first."))
        return "", history

    top_k = 5
    results = pipeline.query(question, top_k=top_k)

    if not results:
        answer = "I couldn't find anything relevant to that question in the indexed books."
    else:
        parts = ["Found **{}** relevant passage(s):\n".format(len(results))]
        for i, r in enumerate(results, 1):
            parts.append(
                "**#{} — {}**  ·  similarity `{:.3f}`\n\n> {}\n".format(
                    i, r.chunk.location_label(), r.score, r.chunk.text
                )
            )
        answer = "\n".join(parts)

    history.append((question, answer))
    return "", history


with gr.Blocks(title="College Books RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 College Books RAG")
    gr.Markdown("Semantic Chunking + FAISS Vector Search — pure retrieval, no LLM involved")

    pipeline = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Chat",
                height=500,
                placeholder="Upload your books on the right, build the index, then ask a question here.",
                bubble_full_width=False,
            )
            msg = gr.Textbox(label="Ask a question", placeholder="Ask a question about your books...", lines=1)
            with gr.Row():
                clear = gr.Button("🗑️ Clear")
                submit = gr.Button("Send", variant="primary", scale=2)

        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload books")
            file_input = gr.File(
                label="PDF or DOCX files",
                file_count="multiple",
            )

            with gr.Accordion("⚙️ Chunking settings", open=False):
                min_chars = gr.Slider(100, 500, value=200, step=50, label="Min chunk size (chars)")
                max_chars = gr.Slider(500, 2000, value=1200, step=100, label="Max chunk size (chars)")
                threshold = gr.Slider(50, 95, value=75, step=5, label="Topic-switch sensitivity (%)")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Chunks to retrieve (top-K)")

            build_btn = gr.Button("🔨 Build index", variant="primary")

            log_output = gr.Textbox(label="🪵 Log output", value="(log output will appear here)", lines=8, max_lines=20)

            index_info = gr.Markdown("No index")

    build_btn.click(
        fn=build_index,
        inputs=[file_input, min_chars, max_chars, threshold, top_k, pipeline],
        outputs=[pipeline, log_output, index_info],
    )

    msg.submit(fn=respond, inputs=[msg, chatbot, pipeline], outputs=[msg, chatbot])
    submit.click(fn=respond, inputs=[msg, chatbot, pipeline], outputs=[msg, chatbot])
    clear.click(fn=lambda: ([], None), inputs=[], outputs=[chatbot, pipeline])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
