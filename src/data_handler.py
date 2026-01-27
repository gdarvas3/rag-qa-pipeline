import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def clean_text(text):
    """
    Normalizes text by removing redundant whitespaces and newlines.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_vector_db():
    """
    Full ingestion workflow:
    1. Load PDFs from disk.
    2. Clean and preprocess text.
    3. Split text into chunks.
    4. Generate embeddings and persist the vector database.
    """
    # Document Loading
    loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Apply normalization to all loaded documents.
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Data chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embed and save
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory="db/"
        )