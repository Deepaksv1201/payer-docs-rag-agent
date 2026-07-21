"""Chat UI for the RAG service.

A thin presentation layer: it sends questions to the API and renders the
answer, its sources, and per-answer latency. Run the API first, then:
    streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/ask")

st.set_page_config(page_title="DocQ — Document Q&A", page_icon="📄")

st.title("DocQ — Document Q&A")
st.caption(
    "Grounded answers over your document corpus. Every reply cites the chunks "
    "it used and refuses when the answer isn't in the documents."
)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the documents..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            data = requests.post(API_URL, json={"question": question}, timeout=120).json()
        st.markdown(data["answer"])
        st.caption(f"{data['provider']} · {data['latency_seconds']}s")
        with st.expander(f"Sources ({len(data['sources'])})"):
            for s in data["sources"]:
                score = f" · score {s['score']:.2f}" if s.get("score") is not None else ""
                st.markdown(f"**{s['source']}**{score}")
                st.caption(s["preview"])
    st.session_state.history.append({"role": "assistant", "content": data["answer"]})
