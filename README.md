# 📅 College Calendar RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) web application designed to answer natural language queries regarding college academic calendars, scheduled events, exam dates, and holidays with high accuracy.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://college-calendar-rag-kpq5qxthzzeuolz5kfearb.streamlit.app/)

---

## ✨ Features

* **Natural Language Querying:** Ask questions about academic schedules in simple English or Tamil.
* **Context-Aware Responses:** Powered by **Google Gemini API (`gemini-3.6-flash`)** for accurate and context-aware answers.
* **Vector Search:** Uses **ChromaDB** for efficient similarity searching across calendar embeddings.
* **Custom Filtering Logic:** Implements precise Python-based filtering logic to eliminate hallucination and ensure **100% data retrieval accuracy**.
* **User-Friendly Interface:** Built with **Streamlit** for a simple and intuitive chat interface.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Framework:** Streamlit
* **LLM & Embeddings:** Google Gemini API (`gemini-3.6-flash`), SentenceTransformers / Gemini Embeddings
* **Vector Database:** ChromaDB / LangChain VectorStores
* **Document Processing:** PyMuPDF / PDFPlumber / Pandas
* **Version Control & Deployment:** Git, GitHub, Streamlit Cloud

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit web application
├── data/                       # Raw calendar PDF / CSV data
├── src/
│   ├── data_loader.py          # Document parsing & text chunking
│   ├── vector_store.py         # ChromaDB indexing & retriever
│   └── rag_chain.py            # Gemini LLM integration & prompt template
├── requirements.txt            # Project dependencies
├── .env.example                # Sample environment variables file
└── README.md                   # Project documentation 
