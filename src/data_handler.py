import re
import os
import shutil
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class DocumentIndexer:
    """
    Handles the end-to-end document ingestion pipeline:
    Loading, preprocessing, chunking, and vector storage.
    """

    def __init__(
        self, 
        data_dir: str = "data/", 
        db_dir: str = "db/", 
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    def _clean_text(self, text: str) -> str:
        """Internal helper to normalize text content."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_documents(self, glob_pattern: str = "*.pdf") -> List:
        """Loads documents from the specified directory."""
        loader = DirectoryLoader(self.data_dir, glob=glob_pattern, loader_cls=PyPDFLoader)
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
        
        return documents

    def create_chunks(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 50):
        """Splits documents into smaller semantic segments."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def build_and_save_index(self):
            """Clears the db content and builds a new one."""
            # Delete content of db
            if os.path.exists(self.db_dir):
                for filename in os.listdir(self.db_dir):
                    if filename == ".gitkeep":
                        continue
                    file_path = os.path.join(self.db_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                os.makedirs(self.db_dir, exist_ok=True)
            
            # Load documents
            docs = self.load_documents()
            if not docs:
                return None
                
            chunks = self.create_chunks(docs)
            print(f"Indexing {len(chunks)} chunks")
            
            # Save db
            db = Chroma.from_documents(
                chunks, 
                self.embeddings, 
                persist_directory=self.db_dir
            )
            print("Database successfully rebuilt.")
            return db
    
if __name__ == "__main__":
    indexer = DocumentIndexer()
    indexer.build_and_save_index()