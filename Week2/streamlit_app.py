import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Page settings
st.set_page_config(
    page_title="TravelMate AI",
    page_icon="✈️",
    layout="wide"
)

# Animated gradient background
st.markdown("""
<style>

body {
    background: linear-gradient(135deg,#ff9a9e,#fad0c4,#fbc2eb,#a6c1ee,#a1c4fd,#c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
0% {background-position:0% 50%;}
50% {background-position:100% 50%;}
100% {background-position:0% 50%;}
}

.stApp{
background: transparent;
}

.block-container{
background: rgba(255,255,255,0.9);
padding: 2rem;
border-radius: 15px;
box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

h1{
text-align:center;
font-size:55px;
color:#0b3d91;
font-family:Trebuchet MS;
}

.subtitle{
text-align:center;
font-size:20px;
color:#444;
margin-bottom:30px;
}

.answer-box{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 6px 20px rgba(0,0,0,0.2);
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1>✈️ TravelMate AI</h1>
<p class="subtitle">
Discover destinations • Explore food • Plan your journey 🌍
</p>
""", unsafe_allow_html=True)

# Feature icons
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown("### 🏝 Destinations")

with col2:
    st.markdown("### 🍜 Food")

with col3:
    st.markdown("### 🏨 Hotels")

with col4:
    st.markdown("### 🗺 Travel Guides")

# Sidebar
with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=90)

    st.title("Travel AI")

    st.write("Your smart travel companion ✈️")

    st.markdown("---")

    st.write("✨ Features")

    st.write("✔ Upload travel guide PDFs")

    st.write("✔ Ask travel questions")

    st.write("✔ AI answers using RAG")

    st.write("✔ Smart document search")

# Inputs
api_key = st.text_input("🔑 Enter Groq API Key", type="password")

uploaded_file = st.file_uploader("📄 Upload Travel Guide PDF", type="pdf")

query = st.text_input("💬 Ask a travel question")

# RAG logic
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
        Use the context below to answer the travel question.

        Context:
        {context}

        Question:
        {query}
        """
    )

    st.markdown("### Travel AI Answer")

    st.markdown(
        f"""
        <div class="answer-box">
        {response.content}
        </div>
        """,
        unsafe_allow_html=True
    )
