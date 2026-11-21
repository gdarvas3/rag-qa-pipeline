import os
import traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai.errors import ClientError

from src.retriever import ContextRetriever
from src.generator import RAGEngine
from src.data_handler import DocumentIndexer

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / "app.env")
load_dotenv(project_root / ".env", override=False)


def is_quota_error(error: Exception) -> bool:
    if not isinstance(error, ClientError):
        return False

    code = getattr(error, "code", None)
    status = str(getattr(error, "status", "")).upper()
    message = str(error).upper()

    return code == 429 or "RESOURCE_EXHAUSTED" in status or "RESOURCE_EXHAUSTED" in message

# --- FastAPI App Configuration ---
app = FastAPI(
    title='RAG Q&A API',
    description='A production-ready FastAPI backend for document RAG with Gemini-based generation and self-evaluation.'
)

retriever = ContextRetriever(
    db_dir="db/", 
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)

engine = RAGEngine(
    embedding_model_name='all-MiniLM-L6-v2',
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
)

# --- Pydantic Data Models ---
class EvaluationMetrics(BaseModel):
    """Schema for the validation metrics provided by the engine."""
    similarity_score: float
    judge_feedback: str


class RetrievalMetrics(BaseModel):
    """Schema for retrieval details from vector and graph stages."""
    vector_hits: int
    graph_edges: int
    graph_hops_used: int
    seed_entities: int
    graph_used: bool = False
    graph_nodes: list[str] = []

class QuestionRequest(BaseModel):
    """Schema for incoming user questions."""
    question: str

class AnswerResponse(BaseModel):
    """Schema for the final API response, structured for frontend consumption."""
    question: str
    answer: str
    context: str
    evaluation: EvaluationMetrics
    retrieval: RetrievalMetrics


class IngestResponse(BaseModel):
    """Schema for ingestion endpoint responses."""
    status: str
    detail: str
    stats: dict[str, int] | None = None

# --- API Endpoints ---

@app.post("/ask", response_model=AnswerResponse)
async def ask_rag(request: QuestionRequest):
    """
    Primary RAG endpoint:
    1. Semantic Search: Retrieves relevant chunks from the persistent vector store.
    2. Generation: Processes the query and context through Gemini.
    3. Self-Evaluation: Runs a judge-audit and similarity check.
    """
    try:
        # Context Retrieval
        retrieval_result = retriever.search_with_metadata(request.question, k=10)
        context = retrieval_result["context"]
        
        if not context:
            raise HTTPException(status_code=404, detail="No relevant information found in the local documents.")

        # Generation and Evaluation
        pipeline_result = engine.process_query(request.question, context)


        # Return structured response
        return AnswerResponse(
            question=request.question,
            answer=pipeline_result["answer"],
            context=context,
            evaluation=EvaluationMetrics(
                similarity_score=pipeline_result["metrics"]["similarity_score"],
                judge_feedback=pipeline_result["metrics"]["judge_feedback"]
            ),
            retrieval=RetrievalMetrics(**retrieval_result["metadata"]),
        )

    except Exception as e:
        if is_quota_error(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini quota exceeded for the configured key/model. "
                    "Check Gemini API usage limits or switch to a key/project with available quota."
                ),
            )

        print(f"CRITICAL ERROR in /ask endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error during RAG processing.")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Triggers document indexing from the data directory into ChromaDB.
    """
    try:
        global retriever

        indexer = DocumentIndexer(data_dir="data/", db_dir="db/")
        result = indexer.build_and_save_index()

        if result is None:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF content could be loaded from the data directory."
            )

        retriever = ContextRetriever(
            db_dir="db/",
            embedding_model='sentence-transformers/all-MiniLM-L6-v2'
        )

        return IngestResponse(
            status="ok",
            detail="Document ingestion completed successfully.",
            stats=indexer.last_ingest_stats,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in /ingest endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error during ingestion.")