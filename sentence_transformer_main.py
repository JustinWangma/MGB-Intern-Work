import argparse

from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings
import create_database

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

If the context does not have anything related to the question, state "The provided documents do not have related information. I will answer based on my own knowledge." and then try your best to answer the question from your own knowledge if you know about it.

Answer the question based on the above context and state whether you are using your own knowledge or provided contexts: {question}

---

"""


def search_vectorstore_chroma_merger_retriever(query, db, embedding_function):
    retriever_sim = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    retriever_mmr = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 3}
    )

    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])

    filter = EmbeddingsRedundantFilter(embeddings=embedding_function)

    reordering = LongContextReorder()

    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger
    )

    docs = compression_retriever.invoke(query)

    return docs


def main():
    #create_database.main()

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    model_kwargs = {'device': 'cpu'}
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # results = db.similarity_search_with_relevance_scores(query_text, k=3)
    results = search_vectorstore_chroma_merger_retriever(query_text, db, embedding_function)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        #return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
