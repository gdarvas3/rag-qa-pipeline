from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_retriever():
    # Embedding modell
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Load DB
    db = Chroma(persist_directory="db/", embedding_function=embeddings)
    
    # Retriver
    # Return top 3 results
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

def retrieve_context(query):
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)
    
    # Merge context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context