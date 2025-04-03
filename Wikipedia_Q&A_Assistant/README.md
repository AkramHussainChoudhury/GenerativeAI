📚 AI-Powered Wikipedia QA System

🚀 Overview

This project is an AI-powered question-answering (QA) system that retrieves and answers queries based on Wikipedia articles. It leverages a Retrieval-Augmented Generation (RAG) approach, combining BM25 and FAISS vector search for efficient and accurate retrieval of relevant text chunks.

🛠️ Features

Fetches Wikipedia content dynamically.

Splits text into chunks for better retrieval.

Uses FAISS for vector-based similarity search.

Uses BM25 for keyword-based retrieval.

Employs Ensemble Retrieval to combine FAISS and BM25.

Uses Google's Gemini 1.5 Pro or Gemini 2.0 Flash LLM for answering questions.

Streamlit-based UI for easy interaction.

📂 Project Structure

RAG_Project/
│── myenv/               # Virtual environment (Python dependencies)
│── faiss_indices/       # Stores FAISS vector indexes
│── bm25_indices/        # Stores BM25 JSON indexes
│── main.py              # Main application script
│── requirements.txt     # Required dependencies
│── README.md            # Project documentation

🏗️ Setup & Installation

1️⃣ Clone the Repository

git clone https://github.com/AkramHussainChoudhury/GenerativeAI.git
cd RAG_Project

2️⃣ Set Up Virtual Environment

python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate     # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up Environment Variables

Create a .env file in the root directory and add your Google API Key:

GOOGLE_API_KEY=your_google_gemini_api_key

5️⃣ Run the Application

streamlit run main.py

🛠️ How It Works

User Inputs a Wikipedia Topic: The system fetches the article content.

Preprocessing:

Splits text into chunks.

Creates FAISS embeddings.

Saves BM25 indexes.

Question-Answering:

Retrieves relevant text using FAISS & BM25.

Uses an ensemble retriever for better accuracy.

Passes retrieved text to Gemini LLM for generating answers.