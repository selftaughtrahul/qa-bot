import streamlit as st
import os
import tempfile

from src.ingestion.pipeline import ingest_document
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.conversational_rag_chain import ConversationalRAGChain

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📚 QA Bot — Ask Your Documents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-card {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .badge {
        background: #4f8ef7;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
def init_session():
    if "embedder" not in st.session_state:
        st.session_state.embedder = DocumentEmbedder()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISSVectorStore()
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = []

init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Document Manager")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files to query."
    )

    if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Ingest
                    chunks = ingest_document(tmp_path)
                    all_chunks.extend(chunks)
                    st.session_state.documents_loaded.append(uploaded_file.name)
                    os.unlink(tmp_path)

                # Build vector store
                st.session_state.vector_store.add_chunks(
                    all_chunks, st.session_state.embedder
                )

                # Create RAG chain
                st.session_state.rag_chain = ConversationalRAGChain(
                    vector_store=st.session_state.vector_store,
                    embedder=st.session_state.embedder,
                    llm_provider="groq"
                )

            st.success(f"✅ {len(uploaded_files)} document(s) processed! ({len(all_chunks)} chunks indexed)")

    # Loaded documents list
    if st.session_state.documents_loaded:
        st.markdown("**📄 Loaded Documents:**")
        for doc in st.session_state.documents_loaded:
            st.markdown(f"- `{doc}`")

    st.markdown("---")

    # Clear button
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.vector_store = FAISSVectorStore()
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.documents_loaded = []
        st.session_state.embedder = DocumentEmbedder()
        st.rerun()

    # Settings
    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=4)
    llm_provider = st.selectbox("LLM Provider", ["groq", "huggingface", "gemini"])


# ── Main Chat Area ─────────────────────────────────────────────────────────────
st.title("🤖 QA Bot — Ask Your Documents")
st.markdown("Upload documents in the sidebar, then ask any question below.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="badge">{src['source']} | Page {src['page']}</span>
                        <br><br>{src['content'][:300]}...
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.rag_chain is None:
        st.error("⚠️ Please upload and process documents first!")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.rag_chain.answer(prompt)

            st.markdown(answer)

            # Show sources
            if sources:
                with st.expander("📄 View Sources", expanded=False):
                    for src in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="badge">{src['source']} | Page {src['page']} | Score: {src.get('similarity_score', 0):.3f}</span>
                            <br><br>{src['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })