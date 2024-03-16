import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title('News Reader Tool 📰')
st.sidebar.title("New Article url")
urls = []
for i in range(2):
    url=st.sidebar.text_input(f"Enter url {i+1}", key= f"url {i+1}")
    urls.append(url)
main_placeholder = st.empty()


if st.sidebar.button("Summarize"):
    for url in urls:
        loader = UnstructuredURLLoader(urls=[url])
        data = loader.load()
        main_placeholder.text("Data Loading...Started...✅✅✅")
        time.sleep(2)
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.'],
                                                        chunk_size=1000, 
                                                        chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        main_placeholder.text("Data Splitting...Started...✅✅✅")
        time.sleep(2)
        
        embeddings = OpenAIEmbeddings()
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
        
        vectorestore = FAISS.from_documents(texts, embeddings)
        main_placeholder.text("Vector Store Started Building...✅✅✅")
        time.sleep(2)


        retriever = vectorestore.as_retriever(search_type="mmr", search_k=3, search_threshold=0.5)

        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(temperature=0, model_name="textdavinci-003"), chain_type="stuff", retriever=retriever)

        query = st.text_input("Enter your query here",key="query")
        if query:
            result= chain({"query": query}, return_only_outputs=True) 
            st.header("Answer")
            st.write(result['answer'])

            sources = result.get('sources', '')
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
            
