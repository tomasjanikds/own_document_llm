import os
import openai
import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def initialize_openai():
    openai.api_key = os.environ['OPENAI_API_KEY']

def get_llm_name():
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        return "gpt-3.5-turbo-0301"
    else:
        return "gpt-3.5-turbo"

def load_pdf_documents():
    loaders = [
        PyPDFLoader("pdf_docs/Harnessing the Falcon 40B Model, the Most Powerful Open-Source LLM.pdf"),
        PyPDFLoader("pdf_docs/The Power of OpenAI’s Function Calling in Language Learning Models_ A Comprehensive Guide.pdf"),
        PyPDFLoader("pdf_docs/Testing the Massively Multilingual Speech (MMS) Model that Supports 1162 Languages.pdf"),
        PyPDFLoader("pdf_docs/Whisper JAX vs PyTorch_ Uncovering the Truth about ASR Performance on GPUs.pdf")
    ]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    return docs

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    splits = text_splitter.split_documents(docs)
    return splits

def create_vector_db(splits):
    embedding = OpenAIEmbeddings()
    persist_directory = 'db/chroma/'
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def initialize_chat_model(llm_name):
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    return llm

def similarity_search(vectordb, question):
    return vectordb.similarity_search(question, k=3)

def retrieve_results(llm, vectordb, question, chain_type="stuff"):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type=chain_type
    )
    result = qa_chain({"query": question})
    return result["result"]

def main():
    initialize_openai()
    llm_name = get_llm_name()
    print("LLM Name:", llm_name)

    docs = load_pdf_documents()
    splits = split_documents(docs)
    vectordb = create_vector_db(splits)

    question = "What is Falcon-40b and can I use it for commercial use"
    llm = initialize_chat_model(llm_name)

    # Perform similarity search
    similarity_results = similarity_search(vectordb, question)
    for doc in similarity_results:
        print(doc.metadata)
    
    results_standard = retrieve_results(llm, vectordb, question)
    print("Standard Results:", results_standard)

    results_map_reduce = retrieve_results(llm, vectordb, question, chain_type="map_reduce")
    print("Map-Reduce Results:", results_map_reduce)

    results_refine = retrieve_results(llm, vectordb, question, chain_type="refine")
    print("Refine Results:", results_refine)
    
if __name__ == "__main__":
    main()
