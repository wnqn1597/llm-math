# -*- coding: utf-8 -*-

# from langchain_community.llms import VLLMOpenAI
from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores.faiss import FAISS
from langgraph.graph import START, StateGraph
from typing import List, TypedDict

embeddings_path = "emb/"
base_url = "http://114.212.85.164:8000/v1"
llm = ChatOpenAI(
    model="Llama",
    temperature=1.0,
    max_tokens=1024,
    max_retries=3,
    base_url=base_url,
    api_key="EMPTY",
)

fp = "files/ABL-reflection.pdf"
loader = PyPDFLoader(file_path=fp)
pages = []

for page in loader.lazy_load():
    pages.append(page)
print(f"PDF pages: {len(pages)}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(pages)
print(f"Splits: {len(all_splits)}")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)
# vector_store = InMemoryVectorStore(embeddings)
# vector_store.add_documents(all_splits)
vector_store = FAISS.from_documents(all_splits, embeddings)
# vector_store.save_local()
# FAISS.load_local()
prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=6)
    # print(len(retrieved_docs))
    # for doc in retrieved_docs:
    #     print(doc.page_content)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # print("-" * 50)
    # print(messages)
    # print("-" * 50)
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
response = graph.invoke({"question": "What is ABL-refl?"})
print(response["answer"])
# print(response["context"])
