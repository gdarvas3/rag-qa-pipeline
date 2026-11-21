import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.errors import NotFoundError

from src.graph_extractor import GeminiGraphExtractor
from src.graph_store import GraphStore

class ContextRetriever:
    """
    Handles the semantic search and context extraction from the vector database.
    """

    def __init__(self, db_dir: str = "db/", embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the retriever with a persistent vector store.
        """
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.graph_enabled = os.getenv("GRAPH_ENABLED", "false").lower() == "true"
        self.graph_hops = int(os.getenv("GRAPH_HOPS", "1"))
        self.graph_topk_nodes = int(os.getenv("GRAPH_TOPK_NODES", "8"))
        self.graph_store = GraphStore(enabled=self.graph_enabled)
        self.graph_extractor = GeminiGraphExtractor(enabled=self.graph_enabled)

        self._init_db()

    def _init_db(self):
        # Load the existing database from disk
        self.db = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )

    @staticmethod
    def _format_graph_facts(graph_facts):
        if not graph_facts:
            return ""

        lines = []
        for fact in graph_facts:
            lines.append(f"- {fact['head']} --[{fact['relation']}]--> {fact['tail']}")
        return "\n".join(lines)

    @staticmethod
    def _collect_graph_nodes(graph_facts):
        """Returns unique head/tail node names from graph facts."""
        nodes = set()
        for fact in graph_facts or []:
            head = str(fact.get("head", "")).strip()
            tail = str(fact.get("tail", "")).strip()
            if head:
                nodes.add(head)
            if tail:
                nodes.add(tail)
        return sorted(nodes)

    def search_with_metadata(self, query: str, k: int = 10) -> dict:
        metadata = {
            "vector_hits": 0,
            "graph_edges": 0,
            "graph_hops_used": 0,
            "seed_entities": 0,
            "graph_used": False,
            "graph_nodes": [],
        }

        try:
            retriever = self.db.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(query)
        except NotFoundError:
            # Collection may have been rebuilt during ingest; reopen the handle once.
            self._init_db()
            retriever = self.db.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return {"context": "", "metadata": metadata}

        metadata["vector_hits"] = len(relevant_docs)
        vector_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        graph_context = ""
        graph_available = self.graph_store.enabled and self.graph_extractor.enabled
        if graph_available:
            try:
                seed_entities = self.graph_extractor.extract_entities(
                    query,
                    max_entities=self.graph_topk_nodes,
                )
                metadata["seed_entities"] = len(seed_entities)

                if seed_entities:
                    graph_facts = self.graph_store.get_neighbors(
                        seed_entities=seed_entities,
                        hops=self.graph_hops,
                        limit=max(5, self.graph_topk_nodes * 3),
                    )
                    metadata["graph_edges"] = len(graph_facts)
                    metadata["graph_hops_used"] = self.graph_hops
                    formatted_facts = self._format_graph_facts(graph_facts)
                    if formatted_facts:
                        metadata["graph_used"] = True
                        metadata["graph_nodes"] = self._collect_graph_nodes(graph_facts)
                        graph_context = f"\n\n[Graph Facts]\n{formatted_facts}"
            except Exception:
                # Graph errors should never break the vector retrieval path.
                pass

        return {
            "context": f"{vector_context}{graph_context}",
            "metadata": metadata,
        }

    def search(self, query: str, k: int = 10) -> str:
        """
        Performs a similarity search and returns a concatenated string of the top-k results.
        """
        result = self.search_with_metadata(query=query, k=k)
        return result["context"]
