import langchain_community.vectorstores.utils
import os
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import GigaChat

load_dotenv()

llm = GigaChat(credentials=os.getenv("API"), scope=os.getenv("SCOPE"), verify_ssl_certs=False)

loader = UnstructuredExcelLoader("./kokos.xlsx", mode="elements")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs)
docs = langchain_community.vectorstores.utils.filter_complex_metadata(docs)

embedding_function = SentenceTransformerEmbeddings(model_name="sergeyzh/rubert-tiny-turbo")

db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever()
prompt = hub.pull("gigachat/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("KOKOS Q&A BOT")
user_question = st.text_input("Задайте свой вопрос о клубе:")

if user_question:
    answer = rag_chain.invoke(user_question)
    st.write(answer)

