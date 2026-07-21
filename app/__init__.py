"""DocQ — a document-intelligence RAG application.

Layered so each concern is isolated:
    api / streamlit   presentation (thin, no business logic)
    service           orchestration of the RAG flow
    retrieval         vector-search backends (Chroma / Bedrock KB)
    providers         text-generation backends (Ollama / Bedrock)
    ingestion         builds the vector store from a document folder
    config            environment-driven settings
"""
