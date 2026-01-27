import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retriever import retrieve_context
from src.generator import final_rag_pipeline

# Initialize FastAPI application
app = FastAPI(
    title='RAG Q&A API'
)

# --- Pydantic Schemas for Request/Response Validation ---

class EvaluationMetrics(BaseModel):
    similarity_score: float
    judge_feedback: str

class QuestionRequest(BaseModel):
    question : str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    context: str
    evaluation: EvaluationMetrics
    is_reliable: bool

# --- API Endpoints ---

@app.post("/ask", response_model=AnswerResponse)
async def ask_rag(request: QuestionRequest):
    """
    Core RAG endpoint:
    1. Retrieves relevant context from ChromaDB.
    2. Passes query and context to the LLM pipeline.
    3. Evaluates the response for reliability.
    """
    try:
        # Context retrival
        context = retrieve_context(request.question)
        if not context:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        # Generation and Evaluation
        result = final_rag_pipeline(request.question, context)

        # We define 'reliable' as having a semantic similarity score above 0.4
        reliable = result["metrics"]["similarity_score"] > 0.4

        # Construct and return the validated response
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            context=context,
            evaluation=EvaluationMetrics(
                similarity_score=result["metrics"]["similarity_score"],
                judge_feedback=result["metrics"]["judge_feedback"]
            ),
            is_reliable=reliable
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))