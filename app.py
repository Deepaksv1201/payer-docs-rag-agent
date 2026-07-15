"""Streamlit UI for the payer-docs retrieval chain.

Usage: streamlit run app.py
"""
import streamlit as st

from ask import answer

st.set_page_config(page_title="Payer Docs Q&A Agent", page_icon="📄")

st.title("Payer Docs Q&A Agent")
st.caption(
    "Retrieval-augmented answers over healthcare payer documentation — "
    "LangChain + Amazon Bedrock Knowledge Base. Every answer shows its sources."
)

question = st.text_input("Ask about the payer documentation:")

if question:
    with st.spinner("Retrieving and generating…"):
        result, docs = answer(question)

    st.markdown(result)

    with st.expander(f"Sources — {len(docs)} chunks retrieved"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}**")
            st.caption(d.page_content[:400])
