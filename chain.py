import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.base import Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.chat_models import GigaChat
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores.base import VectorStore

load_dotenv()


def get_retriever_chain(vectorstore: VectorStore) -> Chain:
    llm = GigaChat(
        credentials=os.getenv("CREDENTIALS"),
        auth_url=os.getenv("AUTH_URL"),
        verify_ssl_certs=False,
    )

    splitted_docs = []
    for value in vectorstore.index_to_docstore_id.values():
        splitted_docs.append(vectorstore.docstore.search(value))
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(splitted_docs, k=3)

    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, keyword_retriever], weights=[0.7, 0.3]
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=ensemble, llm=llm)
    contextualize_q_system_prompt = (
        "Учитывая историю чатов и последний вопрос пользователя, "
        "который может ссылаться на контекст в истории чата, "
        "сформулируй отдельный вопрос, который может быть понят "
        "без истории чата. НЕ отвечай на вопрос, "
        "только переформулируй его, если это необходимо, "
        "а в противном случае верни его как есть."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever_from_llm, contextualize_q_prompt
    )

    return history_aware_retriever


def get_conversational_rag(history_aware_retriever: Chain) -> Chain:
    llm = GigaChat(
        credentials=os.getenv("CREDENTIALS"),
        auth_url=os.getenv("AUTH_URL"),
        verify_ssl_certs=False,
    )

    system_prompt = (
        "Ты - помощник для решения задач, связанных с ответами на вопросы. "
        "Используй следующие фрагменты найденного контекста, чтобы ответить на вопрос: "
        "Если ты не знаешь ответа, скажи, что ты не знаешь. Ответ должен быть кратким"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
