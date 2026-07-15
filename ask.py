"""Retrieval chain over a Bedrock Knowledge Base.

Usage: python ask.py "your question"
"""
import logging
import sys
import time

from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import AWS_REGION, CHAT_MODEL_ID, KB_ID, NUM_RESULTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("payer-rag")

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=KB_ID,
    region_name=AWS_REGION,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}},
)

llm = ChatBedrockConverse(model=CHAT_MODEL_ID, region_name=AWS_REGION)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the context does not contain the answer, say
"I don't have that information in the provided documents."

Context:
{context}

Question: {question}"""
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def answer(question: str):
    """Run the chain and return (answer, retrieved_docs), logging retrieval
    count and end-to-end latency."""
    start = time.perf_counter()
    docs = retriever.invoke(question)
    result = chain.invoke(question)
    elapsed = time.perf_counter() - start
    log.info("query answered | chunks=%d | latency=%.2fs", len(docs), elapsed)
    return result, docs


if __name__ == "__main__":
    result, docs = answer(sys.argv[1])
    print(result)
    print("\n--- sources ---")
    for d in docs:
        print("•", d.page_content[:150].replace("\n", " "))
