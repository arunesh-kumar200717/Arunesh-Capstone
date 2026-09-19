import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📚", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(
        data_dir="data/papers",
        persist_dir="./chroma_db",
        top_k=5,
        embedding_choice="open_source",
        retrieval_strategy="dense",
        use_local_llm=False,
    )

pipeline: RAGPipeline = st.session_state.pipeline

st.title("Research Paper Answer Bot")
st.caption("Ask a question and retrieve grounded answers from your PDF research library.")

with st.sidebar:
    st.header("Active settings")
    st.write(f"Embedding model: {pipeline.embedding_choice}")
    st.write(f"Retrieval strategy: {pipeline.retrieval_strategy}")
    st.write(f"LLM: {'Local Ollama' if pipeline.use_local_llm else 'OpenAI GPT-4o-mini'}")
    st.write(f"Documents indexed: {len(pipeline.final_chunks)} chunks")

question = st.text_input("Ask a question about the papers", placeholder="What is the main contribution of this paper?")

if st.button("Search") and question.strip():
    with st.spinner("Searching and generating an answer..."):
        result = pipeline.answer_question(question, strategy="dense", k=5)

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Top supporting passages")
    for idx, passage in enumerate(result["supporting_passages"], start=1):
        with st.expander(f"Passage {idx}: {passage['paper_title']} (p. {passage['page_number']})"):
            st.write(passage["excerpt"])
else:
    st.info("Add PDFs to data/papers and enter a question to begin.")
