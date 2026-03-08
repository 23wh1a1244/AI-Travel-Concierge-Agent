import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="TravelMate AI",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------
# ANIMATED GRADIENT BACKGROUND
# ---------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#ff9a9e,#fad0c4,#fbc2eb,#a6c1ee,#a1c4fd,#c2e9fb);
background-size: 400% 400%;
animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.main-title{
text-align:center;
font-size:60px;
font-weight:800;
color:#1a1a1a;
}

.sub-title{
text-align:center;
font-size:20px;
margin-bottom:40px;
}

.feature-card{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------

st.markdown(
"""
<div class="main-title">✈️ TravelMate AI</div>
<div class="sub-title">
Discover destinations • Explore food • Plan your journey 🌍
</div>
""",
unsafe_allow_html=True
)

# ---------------------------
# FEATURE SECTION
# ---------------------------

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(
    """
    <div class="feature-card">
    🏝️ <h3>Destinations</h3>
    Discover tourist attractions
    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown(
    """
    <div class="feature-card">
    🍜 <h3>Food</h3>
    Explore local cuisines
    </div>
    """,unsafe_allow_html=True)

with col3:
    st.markdown(
    """
    <div class="feature-card">
    🏨 <h3>Hotels</h3>
    Find places to stay
    </div>
    """,unsafe_allow_html=True)

with col4:
    st.markdown(
    """
    <div class="feature-card">
    📚 <h3>Travel Guides</h3>
    Upload travel PDFs
    </div>
    """,unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------
# LOAD GROQ API FROM SECRETS
# ---------------------------

api_key = st.secrets["GROQ_API_KEY"]

# ---------------------------
# PDF UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
"📄 Upload Travel Guide PDF",
type="pdf"
)

# ---------------------------
# RAG PIPELINE
# ---------------------------

if uploaded_file:

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

    st.success("✅ Travel guide loaded successfully!")

    query = st.text_input("💬 Ask a travel question")

    if query:

        docs = retriever.invoke(query)

        context = "\n".join([d.page_content for d in docs])

        response = llm.invoke(
        f"""
        You are a travel assistant.

        Use the context below to answer clearly and attractively.

        Context:
        {context}

        Question:
        {query}
        """
        )

        st.markdown("###  AI Travel Answer")

        st.success(response.content)
