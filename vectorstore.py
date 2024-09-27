import os
import warnings
from typing import List

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_docs(
    knowledge_base: str = "knowledge_base.txt", n: int = 100
) -> List[Document]:
    pdf_urls = []
    with open(knowledge_base, "r") as f:
        for pdf_url in f:
            pdf_urls.append(pdf_url.strip())

    docs = []
    print("Loading documents...")
    for pdf_url in tqdm(pdf_urls[:n]):
        loader = PDFMinerLoader(
            pdf_url,
            concatenate_pages=True,
        )
        docs += loader.load()
    print("Documents loaded!")

    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    print("Splitting documents...")
    splitted_docs = text_splitter.split_documents(docs)
    print(f"Splitted into {len(splitted_docs)} chunks")
    return splitted_docs


def create_vectorstore(splitted_docs: List[Document]) -> None:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(
        documents=splitted_docs, embedding=embedding, docstore=InMemoryDocstore()
    )

    vectorstore.save_local("store")
    print("Vector store saved!")


def load_vectorstore(path: str = "store") -> VectorStore:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vectorstore = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


def get_vector_store_for_pdf(pdf):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    loader = PDFMinerLoader(pdf)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def main():
    docs = load_docs()
    splitted_docs = split_docs(docs)
    create_vectorstore(splitted_docs)


if __name__ == "__main__":
    os.environ["CURL_CA_BUNDLE"] = ""
    warnings.filterwarnings("ignore")
    main()
