import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retriever import ContextRetriever
from src.generator import RAGEngine
from src.data_handler import DocumentIndexer

# --- FastAPI App Configuration ---
app = FastAPI(
    title='RAG Q&A API',
    description='A production-ready FastAPI backend for local document RAG with self-evaluation.'
)

retriever = ContextRetriever(
    db_dir="db/", 
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)

engine = RAGEngine(
    model_path="models/Mistral-7B-Instruct-v0.3-Q6_K.gguf",
    embedding_model_name='all-MiniLM-L6-v2'
)

# --- Pydantic Data Models ---
class EvaluationMetrics(BaseModel):
    """Schema for the validation metrics provided by the engine."""
    similarity_score: float
    judge_feedback: str

class QuestionRequest(BaseModel):
    """Schema for incoming user questions."""
    question: str

class AnswerResponse(BaseModel):
    """Schema for the final API response, structured for frontend consumption."""
    question: str
    answer: str
    context: str
    evaluation: EvaluationMetrics

# --- API Endpoints ---

@app.post("/ask", response_model=AnswerResponse)
async def ask_rag(request: QuestionRequest):
    """
    Primary RAG endpoint:
    1. Semantic Search: Retrieves relevant chunks from the persistent vector store.
    2. Generation: Processes the query and context through the Mistral LLM.
    3. Self-Evaluation: Runs a judge-audit and similarity check.
    """
    try:
        # Context Retrieval
        context = retriever.search(request.question, k=3)
        
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
            )
        )

    except Exception as e:
        print(f"CRITICAL ERROR in /ask endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error during RAG processing.")