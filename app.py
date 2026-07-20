"""Streamlit chat UI — talks to the FastAPI backend.

Run the API first:  uvicorn api:app --port 8000
Then:               streamlit run app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Payer Docs Q&A Agent", page_icon="📄")

st.title("Payer Docs Q&A Agent")
st.caption(
    "Retrieval-augmented answers over healthcare payer documentation. "
    "FastAPI backend · pluggable LLM providers · grounded with visible sources."
)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about the payer documentation..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            r = requests.post(API_URL, json={"question": question}, timeout=120)
            data = r.json()
        st.markdown(data["answer"])
        st.caption(f"{data['provider']} · {data['latency_seconds']}s")
        with st.expander(f"Sources ({len(data['sources'])})"):
            for s in data["sources"]:
                st.markdown(f"**{s['source']}**" + (f" · score {s['score']:.2f}" if s.get("score") else ""))
                st.caption(s["preview"])
    st.session_state.history.append({"role": "assistant", "content": data["answer"]})
