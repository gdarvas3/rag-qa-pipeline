import re
import os
import time
from pathlib import Path
from typing import List
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from llama_parse import LlamaParse

from src.graph_extractor import GeminiGraphExtractor
from src.graph_store import GraphStore

load_dotenv("app.env")
load_dotenv(".env", override=False)


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
        self.use_llamaparse = os.getenv("USE_LLAMAPARSE", "true").lower() == "true"
        self.graph_enabled = os.getenv("GRAPH_ENABLED", "false").lower() == "true"
        self.graph_store = GraphStore(enabled=self.graph_enabled)
        self.graph_extractor = GeminiGraphExtractor(enabled=self.graph_enabled)
        self.last_ingest_stats = {
            "chunks": 0,
            "entities": 0,
            "relations": 0,
            "graph_enabled": int(self.graph_store.enabled and self.graph_extractor.enabled),
        }

    def _clean_text(self, text: str) -> str:
        """Internal helper to normalize text content."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _load_with_llamaparse(self, pdf_file: Path) -> List[Document]:
        """Parses a PDF in the cloud and returns markdown-backed LangChain documents."""
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is not set.")

        print(f"[Ingest] LlamaParse started: {pdf_file.name}", flush=True)

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown"
        )
        parsed_docs = parser.load_data(str(pdf_file))
        print(f"[Ingest] LlamaParse completed: {pdf_file.name}, pages={len(parsed_docs)}", flush=True)

        return [
            Document(
                page_content=getattr(doc, "text", ""),
                metadata={"source": str(pdf_file), "parser": "llamaparse"}
            )
            for doc in parsed_docs
        ]

    def _load_with_pypdf(self, pdf_file: Path) -> List[Document]:
        """Fallback parser used when LlamaParse is unavailable or fails."""
        print(f"[Ingest] PyPDF started: {pdf_file.name}", flush=True)
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        print(f"[Ingest] PyPDF completed: {pdf_file.name}, pages={len(docs)}", flush=True)
        for doc in docs:
            doc.metadata = {**doc.metadata, "parser": "pypdf"}
        return docs

    def load_documents(self, glob_pattern: str = "*.pdf") -> List:
        """Loads documents from the specified directory.

        LlamaParse is used first for robust extraction of complex layouts.
        PyPDF is used as a local fallback when needed.
        """
        data_path = Path(self.data_dir)
        pdf_files = sorted(data_path.glob(glob_pattern))

        if not pdf_files:
            return []

        documents = []
        failed_files = []
        for pdf_file in pdf_files:
            print(f"[Ingest] Processing file: {pdf_file.name}", flush=True)
            if self.use_llamaparse:
                try:
                    documents.extend(self._load_with_llamaparse(pdf_file))
                except Exception as e:
                    print(f"LlamaParse failed for '{pdf_file.name}', falling back to PyPDF: {e}", flush=True)
                    try:
                        documents.extend(self._load_with_pypdf(pdf_file))
                    except Exception as fallback_error:
                        failed_files.append((pdf_file.name, str(fallback_error)))
            else:
                try:
                    documents.extend(self._load_with_pypdf(pdf_file))
                except Exception as fallback_error:
                    failed_files.append((pdf_file.name, str(fallback_error)))

        for failed_name, failed_reason in failed_files:
            print(f"Skipping unreadable PDF '{failed_name}': {failed_reason}")
        
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)

        # Drop empty pages created by malformed PDFs.
        documents = [doc for doc in documents if doc.page_content]
        
        return documents

    def create_chunks(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 50):
        """Splits documents into smaller semantic segments."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def build_and_save_index(self):
            """Rebuilds the db content from current PDFs."""
            os.makedirs(self.db_dir, exist_ok=True)

            # Clear existing collections safely through Chroma API.
            client = chromadb.PersistentClient(path=self.db_dir)
            for collection in client.list_collections():
                try:
                    client.delete_collection(name=collection.name)
                except Exception as e:
                    print(f"Failed to delete collection '{collection.name}'. Reason: {e}")
            
            # Load documents
            docs = self.load_documents()
            if not docs:
                return None
                
            chunks = self.create_chunks(docs)
            print(f"Indexing {len(chunks)} chunks")

            graph_entities = 0
            graph_relations = 0
            graph_available = self.graph_store.enabled and self.graph_extractor.enabled

            if not graph_available:
                print(
                    "Graph indexing disabled: "
                    f"store_enabled={self.graph_store.enabled}, "
                    f"extractor_enabled={self.graph_extractor.enabled}, "
                    f"store_error={getattr(self.graph_store, 'last_error', '')}"
                )

            if graph_available:
                try:
                    self.graph_store.reset_graph()
                    triplets = []
                    extraction_failures = 0
                    max_graph_chunks = int(os.getenv("GRAPH_MAX_CHUNKS", "0"))
                    batch_size = max(1, int(os.getenv("GRAPH_EXTRACT_BATCH_SIZE", "12")))
                    iterable_chunks = chunks[:max_graph_chunks] if max_graph_chunks > 0 else chunks
                    total_batches = (len(iterable_chunks) + batch_size - 1) // batch_size if iterable_chunks else 0
                    started_at = time.time()

                    print(
                        "Graph extraction started: "
                        f"chunks={len(iterable_chunks)}, batch_size={batch_size}, total_batches={total_batches}"
                    )

                    for start in range(0, len(iterable_chunks), batch_size):
                        batch = iterable_chunks[start:start + batch_size]
                        batch_number = (start // batch_size) + 1
                        batch_payload = []
                        for idx, chunk in enumerate(batch):
                            absolute_idx = start + idx
                            batch_payload.append(
                                {
                                    "chunk_id": f"chunk-{absolute_idx}",
                                    "source": str(chunk.metadata.get("source", "")) if chunk.metadata else "",
                                    "text": chunk.page_content,
                                }
                            )

                        try:
                            extracted = self.graph_extractor.extract_triplets_batch(batch_payload)
                            triplets.extend(extracted)
                            elapsed = time.time() - started_at
                            print(
                                "Graph extraction progress: "
                                f"batch={batch_number}/{total_batches}, "
                                f"batch_chunks={len(batch_payload)}, "
                                f"batch_triplets={len(extracted)}, "
                                f"total_triplets={len(triplets)}, "
                                f"elapsed_s={elapsed:.1f}"
                            )
                        except Exception as extraction_error:
                            extraction_failures += len(batch_payload)
                            print(
                                "Graph extraction failed for batch "
                                f"start={start}, size={len(batch_payload)}: {extraction_error}"
                            )

                    if extraction_failures:
                        print(f"Graph extraction failures: {extraction_failures}")

                    graph_stats = self.graph_store.upsert_triplets(triplets)
                    graph_entities = graph_stats["entities"]
                    graph_relations = graph_stats["relations"]
                    print(f"Graph indexed: entities={graph_entities}, relations={graph_relations}")
                except Exception as e:
                    print(f"Graph indexing skipped due to error: {e}")

            # Save db
            db = Chroma.from_documents(
                chunks, 
                self.embeddings, 
                persist_directory=self.db_dir
            )

            self.last_ingest_stats = {
                "chunks": len(chunks),
                "entities": graph_entities,
                "relations": graph_relations,
                "graph_enabled": int(graph_available),
            }
            print("Database successfully rebuilt.")
            return db
    
if __name__ == "__main__":
    indexer = DocumentIndexer()
    indexer.build_and_save_index()