import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from chain import get_conversational_rag, get_retriever_chain
from vectorstore import get_vector_store_for_pdf, load_vectorstore

load_dotenv()
warnings.filterwarnings("ignore")


def get_response(user_input):
    hsc = get_retriever_chain(st.session_state.vectorstore)
    conv_chain = get_conversational_rag(hsc)
    response = conv_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )
    answer = response["answer"]

    sources = []
    for doc in response["context"]:
        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])
    if sources:
        answer += "\n\nИсточники: " + ", ".join(sources)
    return answer


st.header("RAG ChatBot")

chat_history = []
vectorstore = []

with st.sidebar:
    st.header("Опции")
    option = st.radio(
        "Выберите опцию: ", ("Использовать базу знаний", "Загрузить свой PDF")
    )

if option == "Загрузить свой PDF":

    with st.sidebar:
        st.header("Загрузить PDF")
        pdf = st.file_uploader("Загрузить PDF", "pdf")

    if pdf is None:
        st.info("Загрузите PDF")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Как я могу помочь?")]

        if vectorstore not in st.session_state:
            with open("tmp/" + pdf.name, mode="wb") as w:
                w.write(pdf.getvalue())

            st.session_state.vectorstore = get_vector_store_for_pdf("tmp/" + pdf.name)

        user_input = st.chat_input("Напиши здесь")
        if user_input is not None and user_input.strip() != "":
            response = get_response(user_input)

            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            else:
                with st.chat_message("Human"):
                    st.write(message.content)

else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Как я могу помочь?")]
    if vectorstore not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()

    user_input = st.chat_input("Напиши здесь")
    if user_input is not None and user_input.strip() != "":
        response = get_response(user_input)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)
