import os

import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

st.title("📚 AI-Powered Wikipedia QA System")
st.write("Enter your question below to get an AI-powered response.")

def save_bm25_docs(wiki_title, chunks):
    """Saves BM25 documents locally to avoid re-fetching Wikipedia data"""
    bm25_data = [{"title": wiki_title, "content": chunk} for chunk in chunks]

    with open("bm25_docs.json", "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=4)

def getWikipediaContent(title):
    """Fetches the full Wikipedia page content using the MediaWiki API"""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explainText": True,
        "format": "json"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            raw_html = page_data.get("extract", "No content available")
            soup = BeautifulSoup(raw_html, "html.parser")
            return soup.get_text(separator="\n")  # Convert to plain text

    return f"Error: {response.status_code}"

def ingestData(wiki_title):
    try :

        def getWikipediaSummary(title):
            """Fetches the summary of a Wikipedia page using the REST API"""
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return data.get("extract","No summary available")
            else:
                return f"Error:{response.status_code}"



        wiki_data = getWikipediaContent(wiki_title)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(wiki_data)
        docs = [Document(page_content=chunk, metadata={"title": wiki_title}) for chunk in chunks]

        #Save for BM25 retreiver
        save_bm25_docs(wiki_title, chunks)

        # Create FAISS Vector Store from Documents
        faiss_index = FAISS.from_documents(docs, embeddings)

        # Save for future use
        faiss_index.save_local("wikipedia_faiss_index")
    except Exception as e :
        print(f"❌ Error during ingestion: {e}")




def load_model(model_name):
    """Loads the appropriate Gemini model."""
    try:
        if model_name=="gemini-1.5-pro":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        else:
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        return llm
    except Exception as e:
        print(f"❌ Error loading LLM model: {e}")
        return None


def load_bm25_docs(wiki_title):
    """Loads BM25 documents from local storage"""
    try:
        if os.path.exists("bm25_docs.json"):
            with open("bm25_docs.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Document(page_content=item["content"], metadata={"title": item["title"]}) for item in data]
        print("BM25 documents not found. Re-ingesting data...")
        ingestData(wiki_title)  # Re-fetch Wikipedia content & save BM25 data
        return load_bm25_docs(wiki_title)  # Load again after ingestion
    except Exception as e:
        print(f"❌ Error loading BM25 documents: {e}")
        return None

def faiss_index_exists():
    return os.path.exists("wikipedia_faiss_index/index.faiss") and os.path.exists("wikipedia_faiss_index/index.pkl")


wiki_title = "Indian_independence_movement"


if not faiss_index_exists():
    ingestData(wiki_title)


faiss_index = FAISS.load_local("wikipedia_faiss_index", embeddings,allow_dangerous_deserialization=True)

# Create retriever
vectorstore_retreiver = faiss_index.as_retriever(search_type="similarity")

bm25_docs = load_bm25_docs(wiki_title)
keyword_retriever = BM25Retriever.from_documents(bm25_docs)
keyword_retriever.k = 5


llm = load_model("gemini-1.5-pro")
normal_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = vectorstore_retreiver
)


ensemble_retriever =EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.5,0.5])

question = st.text_input("Ask a question:")

if question:
    if llm  and ensemble_retriever:
        try:
            ensemblel_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever = ensemble_retriever
            )

            print("getting ensemble answer")
            response = ensemblel_chain.invoke(question)

            answer= response.get("result","No response received")
            st.success("✅ AI Answer:")
            st.write(answer)

        except Exception as e:
            print(f"❌ Error during question-answering: {e}")
    else:
        st.error("⚠️ Model or retriever is not properly initialized.")