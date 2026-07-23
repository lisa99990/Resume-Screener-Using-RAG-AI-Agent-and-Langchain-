import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# load API key
load_dotenv()
st.title("GC Questionaire - Feasible for anyone applying Marriage based Green card with greencard holder spouse")
@st.cache_resource
def build_chain():
    #loads all docs from the docs folder and splits them into chunks of 1000 characters with an overlap of 150 characters
    loader=DirectoryLoader("docs",glob="*pdf",loader_cls=PyPDFLoader)
    docs=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
    task_type="retrieval_document")
    #Creates a FAISS vector store from the chunks and embeddings
    vectordb=FAISS.from_documents(chunks,embeddings)

    #Creates a ChatGoogleGenerativeAI model with the specified parameters as LLM
    llm=ChatGoogleGenerativeAI( model="gemini-3.5-flash",         
        temperature=0,
        )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return qa


qa = build_chain()
question=st.text_input("Ask anything about Greencard application")
if st.button("Ask") and question:
    result=qa.invoke({"query",question})
    st.subheader("Answer")
    st.write(result['result'])
    st.subheader("Source Documents")
    shown=set()
    for doc in result["source_documents"]:
        src=os.path.basename(doc.metadata['source'])
        if src not in shown:
            st.write("*", src)
            shown.add(src)




