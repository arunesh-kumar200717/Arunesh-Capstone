import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

USE_LOCAL_LLM = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful research assistant. Answer the user's question using only the provided context. "
                "If the context does not provide enough evidence, respond exactly: 'I don't know based on the provided papers'. "
                "Cite the supporting evidence in the final answer by referencing the paper title and page number from the metadata.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )


def load_environment() -> None:
    load_dotenv()


def normalize_document_metadata(doc: Document) -> Document:
    metadata = dict(doc.metadata or {})
    metadata["source"] = metadata.get("source") or metadata.get("paper_title") or "unknown"
    metadata["paper_title"] = metadata.get("paper_title") or metadata["source"]
    metadata["page"] = int(metadata.get("page", 1) or 1)
    doc.metadata = metadata
    return doc


def safe_pdf_load(data_dir: str = "data/papers") -> List[Document]:
    documents: List[Document] = []
    pdf_dir = Path(data_dir)

    if not pdf_dir.exists():
        print(f"Data directory not found: {pdf_dir}. Add PDFs under data/papers/.")
        return documents

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files were found in {pdf_dir}. Add research papers to continue.")
        return documents

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
        except Exception as exc:
            print(f"Failed to load {pdf_path.name}: {exc}")
            continue

        for doc in loaded_docs:
            metadata = dict(doc.metadata or {})
            page_number = int(metadata.get("page", 1) or 1)
            metadata.update(
                {
                    "source": pdf_path.stem,
                    "paper_title": pdf_path.stem,
                    "page": page_number,
                    "filename": pdf_path.name,
                }
            )
            doc.metadata = metadata
            documents.append(doc)

    return documents


def recursive_chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [normalize_document_metadata(chunk) for chunk in chunks]


def semantic_chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    if not documents:
        return []

    paragraph_documents: List[Document] = []
    for doc in documents:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", doc.page_content) if part.strip()]
        if not paragraphs:
            paragraphs = [doc.page_content.strip()]

        for paragraph in paragraphs:
            chunk_doc = Document(
                page_content=paragraph,
                metadata={
                    "source": doc.metadata.get("source") or doc.metadata.get("paper_title") or "unknown",
                    "paper_title": doc.metadata.get("paper_title") or doc.metadata.get("source") or "unknown",
                    "page": int(doc.metadata.get("page", 1) or 1),
                },
            )
            paragraph_documents.append(chunk_doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(paragraph_documents)
    return [normalize_document_metadata(chunk) for chunk in chunks]


def make_embedding_model(embedding_choice: str = "open_source") -> Any:
    load_environment()
    embedding_choice = (embedding_choice or "open_source").lower()

    if embedding_choice in {"open_source", "huggingface", "local"}:
        return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    if embedding_choice in {"commercial", "openai"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file before using OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    raise ValueError("embedding_choice must be 'open_source' or 'commercial'.")


def build_chroma_store(chunks: List[Document], embedding_model: Any, persist_dir: str = "./chroma_db") -> Optional[Chroma]:
    if not chunks:
        print("No chunks available to index in Chroma. Skipping vector store creation.")
        return None

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )


def build_faiss_store(chunks: List[Document], embedding_model: Any) -> Optional[FAISS]:
    if not chunks:
        print("No chunks available to index in FAISS. Skipping FAISS store creation.")
        return None

    return FAISS.from_documents(chunks, embedding_model)


def dense_retriever(vectorstore: Optional[Chroma], query: str, k: int = 5) -> List[Document]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def mmr_retriever(vectorstore: Optional[Chroma], query: str, k: int = 5, fetch_k: int = 20) -> List[Document]:
    if vectorstore is None:
        return []
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def hybrid_retriever(
    vectorstore: Optional[Chroma],
    documents: List[Document],
    query: str,
    k: int = 5,
) -> List[Document]:
    if vectorstore is None:
        return []

    dense_hits = vectorstore.similarity_search(query, k=max(10, k))
    if not documents:
        return dense_hits[:k]

    bm25 = BM25Retriever.from_documents(documents, k=max(10, k))
    keyword_hits = bm25.invoke(query)

    scored: Dict[str, float] = {}
    docs_by_key: Dict[str, Document] = {}

    def add_doc(doc: Document, weight: float) -> None:
        key = f"{doc.metadata.get('source', 'unknown')}::{doc.metadata.get('page', 1)}::{doc.page_content[:150]}"
        scored[key] = scored.get(key, 0.0) + weight
        docs_by_key[key] = doc

    for rank, doc in enumerate(dense_hits):
        add_doc(doc, 1.0 / (60 + rank))

    for rank, doc in enumerate(keyword_hits):
        add_doc(doc, 0.6 / (60 + rank))

    ranked_keys = sorted(scored, key=scored.get, reverse=True)[:k]
    return [docs_by_key[key] for key in ranked_keys]


class RAGPipeline:
    def __init__(
        self,
        data_dir: str = "data/papers",
        persist_dir: str = "./chroma_db",
        top_k: int = 5,
        embedding_choice: str = "open_source",
        retrieval_strategy: str = "dense",
        use_local_llm: bool = False,
    ):
        set_seed(42)
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.embedding_choice = embedding_choice
        self.retrieval_strategy = retrieval_strategy
        self.use_local_llm = use_local_llm

        self.documents = self.load_documents()
        self.recursive_chunks = recursive_chunk_documents(self.documents, chunk_size=1000, chunk_overlap=150)
        self.semantic_chunks = semantic_chunk_documents(self.documents, chunk_size=1000, chunk_overlap=150)
        self.final_chunks = self.recursive_chunks if self.recursive_chunks else self.semantic_chunks

        self.embedding_model: Optional[Any] = None
        self.vectorstore: Optional[Chroma] = None
        self.faiss_store: Optional[FAISS] = None
        self.llm: Optional[Any] = None
        self.retriever: Optional[Any] = None

        if self.documents:
            try:
                self.embedding_model = make_embedding_model(self.embedding_choice)
                self.vectorstore = build_chroma_store(self.final_chunks, self.embedding_model, persist_dir=self.persist_dir)
                self.faiss_store = build_faiss_store(self.final_chunks, self.embedding_model)
                self.llm = self._build_llm()
                self.retriever = (
                    self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})
                    if self.vectorstore is not None
                    else None
                )
            except Exception as exc:
                print(f"Warning: vector store initialization failed: {exc}")
                self.vectorstore = None
                self.faiss_store = None
                self.llm = None
                self.retriever = None

    def load_documents(self) -> List[Document]:
        return safe_pdf_load(self.data_dir)

    def _build_llm(self) -> Any:
        load_environment()

        if self.use_local_llm:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError("Install langchain-ollama to use the local LLM fallback.") from exc
            return ChatOllama(model="llama3.1:8b")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before running the final generation step.")

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant context was retrieved."

        formatted_chunks: List[str] = []
        for doc in docs:
            title = doc.metadata.get("paper_title") or doc.metadata.get("source") or "Unknown paper"
            page = doc.metadata.get("page", 1)
            formatted_chunks.append(f"[Source: {title}, Page {page}]\n{doc.page_content.strip()}")

        return "\n\n".join(formatted_chunks)

    def _retrieve(self, query: str, strategy: Optional[str] = None, k: Optional[int] = None) -> List[Document]:
        if not self.documents:
            return []

        retrieval_strategy = strategy or self.retrieval_strategy
        top_k = k if k is not None else self.top_k

        if retrieval_strategy == "dense":
            return dense_retriever(self.vectorstore, query, k=top_k)
        if retrieval_strategy == "mmr":
            return mmr_retriever(self.vectorstore, query, k=top_k)
        if retrieval_strategy == "hybrid":
            return hybrid_retriever(self.vectorstore, self.final_chunks, query, k=top_k)

        raise ValueError(f"Unsupported retrieval strategy: {retrieval_strategy}")

    def answer_question(self, question: str, strategy: Optional[str] = None, k: Optional[int] = None) -> Dict[str, Any]:
        if not self.documents:
            return {"answer": "I don't know based on the provided papers", "supporting_passages": []}

        if self.llm is None:
            return {
                "answer": "The RAG pipeline is not ready. Please add PDFs and configure OPENAI_API_KEY.",
                "supporting_passages": [],
            }

        docs = self._retrieve(question, strategy=strategy, k=k)
        context = self.format_context(docs)
        prompt = build_prompt_template()
        answer = (prompt | self.llm | StrOutputParser()).invoke({"question": question, "context": context})
        answer_text = answer.strip() or "I don't know based on the provided papers"

        passages: List[Dict[str, Any]] = []
        for doc in docs[:3]:
            passages.append(
                {
                    "paper_title": doc.metadata.get("paper_title") or doc.metadata.get("source") or "Unknown paper",
                    "page_number": int(doc.metadata.get("page", 1) or 1),
                    "excerpt": doc.page_content.strip()[:500],
                }
            )

        return {"answer": answer_text, "supporting_passages": passages}
