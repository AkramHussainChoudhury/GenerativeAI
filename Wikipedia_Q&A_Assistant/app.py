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
FAISS_DIR="faiss_indices"
bm25_dir = "bm25_indices"
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(bm25_dir, exist_ok=True)

st.title("📚 AI-Powered Wikipedia QA System")
#st.write("Enter your question below to get an AI-powered response.")



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

def save_bm25_docs(wiki_title, chunks):
    """Saves BM25 documents locally to avoid re-fetching Wikipedia data"""
    bm25_data = [{"title": wiki_title, "content": chunk} for chunk in chunks]
    file_path = os.path.join(bm25_dir, f"{wiki_title}.json")
    with open(file_path, "w", encoding="utf-8") as f:
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
        if "Error" in wiki_data:
            st.error(f"Failed to fetch Wikipedia content for '{wiki_title}'")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(wiki_data)
        docs = [Document(page_content=chunk, metadata={"title": wiki_title}) for chunk in chunks]

        #Save for BM25 retreiver
        save_bm25_docs(wiki_title, chunks)

        # Create FAISS Vector Store from Documents
        faiss_index = FAISS.from_documents(docs, embeddings)
        # Save for future use
        index_path = os.path.join(FAISS_DIR, wiki_title)
        faiss_index.save_local(index_path)
    except Exception as e :
        os.remove(os.path.join(FAISS_DIR, wiki_title))  # Remove partial FAISS index
        os.remove(os.path.join(bm25_dir, f"{wiki_title}.json"))  # Remove partial BM25 file
        print(f"❌ Error during ingestion: {e}")


def load_bm25_docs():
    """Loads BM25 documents from local storage"""
    bm25_docs = []
    if not os.path.exists(bm25_dir):
        return []
    for file_name in os.listdir(bm25_dir):
        file_path = os.path.join(bm25_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                bm25_docs.extend([Document(page_content=item["content"], metadata={"title": item["title"]}) for item in data])
            #print("BM25 documents not found. Re-ingesting data...")
            #ingestData(wiki_title)  # Re-fetch Wikipedia content & save BM25 data
            #return load_bm25_docs(wiki_title)  # Load again after ingestion
        except Exception as e:
            st.warning(f"⚠️ Failed to load BM25 index '{file_name}': {e}")
    return bm25_docs

def load_faiss_indexes():
    """Loads all available FAISS indexes"""
    retrievers=[]
    for index_name in os.listdir(FAISS_DIR):
        index_path = os.path.join(FAISS_DIR,index_name)
        try:
            faiss_index = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
            retrievers.append(faiss_index.as_retriever(search_type="similarity"))
        except Exception as e:
            st.warning(f"failed to load FAISS index '{index_name}':{e}")

    return retrievers


# Load all retrievers at startup
bm25_docs = load_bm25_docs()
faiss_retrievers = load_faiss_indexes()
bm25_retriever = BM25Retriever.from_documents(bm25_docs) if bm25_docs else None

# Create retriever
retrievers = load_faiss_indexes()
llm = load_model("gemini-1.5-pro")


if faiss_retrievers or bm25_retriever:
    retriever_list = [*faiss_retrievers]  # Add FAISS retrievers first
    if bm25_retriever:
        retriever_list.append(bm25_retriever)  # Add BM25 if available

    # Ensure retriever_list is not empty before creating ensemble
    if retriever_list:
        ensemble_retriever = EnsembleRetriever(
            retrievers=retriever_list,
            weights=[1 / len(retriever_list)] * len(retriever_list)
        )
    else:
        ensemble_retriever = None  # No retrievers available
else:
    ensemble_retriever = None  # No retrievers available


# UI for adding new topics
topic_input = st.text_input("Enter a Wikipedia topic to ingest (optional):")
if st.button("Ingest Topic") and topic_input:
    ingestData(topic_input)
    st.success(f"Ingested data for {topic_input}")



# UI for asking questions
query = st.text_input("Ask a question:")
if st.button("Search") and query:
    ensemble_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever
    )
    response = ensemble_chain.invoke(query)
    answer = response.get("result", "No response received")
    st.success("✅ AI Answer:")
    st.write(answer)