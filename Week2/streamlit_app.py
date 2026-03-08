import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Page settings
st.set_page_config(
    page_title="AI Travel Concierge",
    page_icon="🌍",
    layout="centered"
)

# Background styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#e0f7fa,#fff3e0);
}

h1 {
    text-align:center;
    color:#1f4e79;
}

.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.15);
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1>🌍 AI Travel Concierge</h1>
<p style='text-align:center;font-size:18px'>
Your AI assistant for exploring travel guides ✈️
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.title("Travel AI")
    st.write("Upload travel guides and ask questions from them.")
    st.markdown("---")
    st.write("✨ Features")
    st.write("✔ PDF Travel Guides")
    st.write("✔ AI Answers")
    st.write("✔ RAG Retrieval")

# Card container
st.markdown("<div class='card'>", unsafe_allow_html=True)

api_key = st.text_input("🔑 Enter Groq API Key", type="password")

uploaded_file = st.file_uploader(
    "📄 Upload Travel Guide PDF",
    type="pdf"
)

query = st.text_input("💬 Ask a travel question")

st.markdown("</div>", unsafe_allow_html=True)

# Main logic
if uploaded_file and api_key and query:

    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.7
    )

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f"""
        Use the context below to answer.

        Context:
        {context}

        Question:
        {query}
        """
    )

    st.markdown("### 🤖 AI Travel Assistant")

    st.markdown(
        f"""
        <div style="
        background:white;
        padding:20px;
        border-radius:10px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1)">
        {response.content}
        </div>
        """,
        unsafe_allow_html=True
    )
