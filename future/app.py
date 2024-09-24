from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# db = FAISS.load_local(r"faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization = True)  

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    user_query = text

    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_chroma import Chroma
    from langchain_community.chat_message_histories import ChatMessageHistory
    # from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY']='sk-proj-l6lhdh6zadWETeCgSQyqT3BlbkFJEfofIqmuPzNqxoW3E2Ys'
    # print(os.environ['OPENAI_API_KEY'])
    # import getpass
    # import os

    # os.environ["OPENAI_API_KEY"] = getpass.getpass()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    import PyPDF2
    from PyPDF2 import PdfReader

    # def get_pdf_text(pdf_docs):
    #     text=""
    #     for pdf in pdf_docs:
    #         pdf_reader= PdfReader(pdf)
    #         for page in pdf_reader.pages:
    #             text+= page.extract_text()
    #     return  text

    # pdf = r'C:\Users\chira\OneDrive\Desktop\frshr\LLM_roadmap\future\future.pdf'
    # pdfFileObject = open(pdf, 'rb')
    # pdfReader = PyPDF2.PdfReader (pdfFileObject)
    # # pdfReader
    # count = len(pdfReader.pages)
    # output = ''
    # for i in range(count):
    #     page = pdfReader.pages[i]
    #     output += page.extract_text()

    # output

    # from langchain.docstore.document import Document

    # doc =  Document(page_content=output, metadata={"source": "local"})
    # # type(doc)

    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    # documents=text_splitter.split_documents([doc])
    # # print(documents)

    # # ## Vector Embedding And Vector Store
    # from langchain_openai import OpenAIEmbeddings
    # from langchain_community.vectorstores import Chroma
    # db = Chroma.from_documents(documents,OpenAIEmbeddings())

    from langchain_community.vectorstores import FAISS

    # db = FAISS.from_documents(documents, OpenAIEmbeddings())
    # db.save_local(r'C:\Users\chira\OneDrive\Desktop\frshr\LLM_roadmap\future\faiss_index')
    db = FAISS.load_local(r"C:\Users\chira\OneDrive\Desktop\frshr\LLM_roadmap\future\faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization = True)

    query = "What does Frshr technology does"
    retireved_results=db.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [    ("ai", contextualize_q_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retireved_results, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt =  """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say "Apologies, I do not have the answer to this question. Kindly send us an email at **EMAIL** and we will revert back to you urgently."\
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("ai", qa_system_prompt),
            (MessagesPlaceholder("chat_history")),
           ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}


    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     if session_id not in store:
    #         store[session_id] = ChatMessageHistory()
    #     return store[session_id]


    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )

    # import streamlit as st
    from langchain_core.messages import HumanMessage
    chat_history = []

    response = rag_chain.invoke({"input":user_query, "chat_history": chat_history})
    # response = user_input(user_query)
    chat_history.extend([HumanMessage(content=user_query), response["answer"]])

    # if request.method == 'POST':
    #     # user_query = str(request.form['user_input'])
    #     return render_template('index.html', data=response['answer'])
    # return render_template("index.html")


    return response['answer']


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=True, port=5005)