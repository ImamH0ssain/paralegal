"""Streamlit UI for the grounded legal memo RAG workflow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from legal_rag.evaluation import run_evaluation
from legal_rag.feedback import save_operator_feedback
from legal_rag.generation import DEFAULT_MEMO_QUERY, generate_memo
from legal_rag.ingestion import ingest_paths
from legal_rag.providers import OpenAIClient, make_embedding_provider, make_reranker
from legal_rag.real_data import download_real_inputs
from legal_rag.retrieval import EvidenceIndex
from legal_rag.sample_data import create_sample_inputs
from legal_rag.io_utils import read_json


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_inputs"
UPLOAD_DIR = ROOT / "data" / "uploads"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "evidence_index"
FEEDBACK_DIR = ROOT / "data" / "feedback"
OUTPUT_DIR = ROOT / "sample_outputs"


def available_inputs() -> list[Path]:
    """Return uploaded and sample inputs that the UI can ingest."""
    manifest = read_json(SAMPLE_DIR / "real_source_manifest.json", default=[]) or []
    real_paths = [
        Path(item["path"])
        for item in manifest
        if item.get("downloaded") and Path(item.get("path", "")).exists()
    ]
    if real_paths:
        return real_paths + [
            p for p in UPLOAD_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}
        ]
    inputs = [p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}]
    inputs += [p for p in UPLOAD_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}]
    return inputs


st.set_page_config(page_title="Grounded Legal Memo RAG", layout="wide")
st.title("Grounded Legal Memo RAG")

with st.sidebar:
    st.header("Workflow")
    use_openai = st.toggle("Use OpenAI when configured", value=False)
    use_ollama = st.toggle("Use local Ollama", value=False)
    use_docling = st.toggle("Use local Docling OCR/layout", value=False)
    use_local_embeddings = st.toggle("Use local Qwen embeddings", value=False)
    use_local_reranker = st.toggle("Use local BGE reranker", value=False)
    force_ocr = st.toggle("Force OCR path", value=False)
    query = st.text_area("Drafting task", value=DEFAULT_MEMO_QUERY, height=150)
    uploaded = st.file_uploader("Add documents", accept_multiple_files=True, type=["pdf", "txt", "md", "png", "jpg", "jpeg"])
    if uploaded:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded:
            (UPLOAD_DIR / file.name).write_bytes(file.getbuffer())
        st.success(f"Saved {len(uploaded)} uploaded file(s).")

left, right = st.columns([1, 1])

with left:
    st.subheader("Prepare Evidence")
    if st.button("Download public real packets"):
        paths = download_real_inputs(SAMPLE_DIR)
        st.success(f"Downloaded/found {len(paths)} public document(s).")
    if st.button("Create synthetic fallback packets"):
        paths = create_sample_inputs(SAMPLE_DIR)
        st.success(f"Created {len(paths)} sample PDFs.")
    inputs = available_inputs()
    st.caption(f"{len(inputs)} input file(s) available.")
    if inputs:
        st.write([p.name for p in inputs])
    if st.button("Ingest documents", disabled=not inputs):
        records = ingest_paths(
            inputs,
            PROCESSED_DIR,
            force_ocr=force_ocr,
            ocr_backend="docling" if use_docling else "auto",
            use_openai=use_openai,
        )
        st.success(f"Wrote {len(records)} processed records.")
    if st.button("Build retrieval index"):
        count = EvidenceIndex(
            INDEX_DIR,
            embedding_provider=make_embedding_provider("local" if use_local_embeddings else "auto"),
        ).build(PROCESSED_DIR / "processed_documents.jsonl")
        st.success(f"Indexed {count} evidence blocks.")

with right:
    st.subheader("Evaluate")
    if st.button("Run evaluation"):
        report = run_evaluation(
            index_dir=INDEX_DIR,
            processed_dir=PROCESSED_DIR,
            ground_truth_path=SAMPLE_DIR / "sample_ground_truth.json",
            feedback_dir=FEEDBACK_DIR,
            output_dir=OUTPUT_DIR,
            use_openai=use_openai,
            use_ollama=use_ollama,
            reranker=make_reranker("local" if use_local_reranker else "none"),
        )
        st.json(report)

st.divider()
st.subheader("Generate Draft")
if st.button("Generate grounded memo"):
    try:
        st.session_state["draft_result"] = generate_memo(
            INDEX_DIR,
            PROCESSED_DIR,
            query=query,
            feedback_dir=FEEDBACK_DIR,
            output_dir=OUTPUT_DIR,
            use_openai=use_openai,
            use_ollama=use_ollama,
            reranker=make_reranker("local" if use_local_reranker else "none"),
        )
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("draft_result")
if result:
    memo_col, evidence_col = st.columns([1.2, 0.8])
    with memo_col:
        edited = st.text_area("Editable memo", value=result["memo"], height=520)
        st.write("Citation validation", result["validation"])
        if st.button("Save operator edit"):
            saved = save_operator_feedback(
                FEEDBACK_DIR,
                task_type="internal_case_memo",
                original_draft=result["memo"],
                edited_draft=edited,
                evidence_ids=[item["evidence_id"] for item in result["evidence"]],
                openai=OpenAIClient() if use_openai else None,
            )
            st.success("Feedback saved and learned patterns updated.")
            st.json(saved["learned_patterns"])
    with evidence_col:
        st.markdown("### Retrieved Evidence")
        for item in result["evidence"]:
            with st.expander(f"{item['evidence_id']} - {item['source_file']} p.{item['page']}"):
                st.write(item["text"])
                st.caption(f"score={item['score']:.4f} confidence={item['confidence']:.2f}")
