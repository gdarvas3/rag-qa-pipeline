from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class ContextRetriever:
    """
    Handles the semantic search and context extraction from the vector database.
    """

    def __init__(self, db_dir: str = "db/", embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the retriever with a persistent vector store.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load the existing database from disk
        self.db = Chroma(
            persist_directory=db_dir, 
            embedding_function=self.embeddings
        )

    def search(self, query: str, k: int = 3) -> str:
        """
        Performs a similarity search and returns a concatenated string of the top-k results.
        """
        # We use the 'as_retriever' interface for standard LangChain compatibility
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return ""

        # Join the retrieved segments with double newlines for clear separation in the prompt
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context