from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

# --- Model Initialization ---

llm = Llama(
    model_path="models/Mistral-7B-Instruct-v0.3-Q6_K.gguf",
    n_ctx=2048,
    n_threads=4
)

# Lightweight embedding model for similarity checks
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_answer(query, context):
    """
    Generates a response using the LLM based on provided context.
    Uses a standard Instruction template for Mistral models.
    """
    prompt = (
        f"[INST]Answer the question based on the context"
        f"Context:\n{context}\n\n"
        f"Question: {query} [/INST]\n"
        "Answer:"
    )
    
    response = llm(
        prompt, 
        stop=["[/INST]", "</s>", "Question:", "Context:"], 
        echo=False,
        temperature=0.2,
        max_tokens=200
    )
    
    answer = response["choices"][0]["text"].strip()
    return answer

def evaluate_answer(query, context, answer):
    """
    LLM-as-a-Judge: Performs a secondary pass where the model acts as an auditor.
    Checks for alignment with context and assigns a quality score.
    """
    eval_prompt = (
        f"[INST] You are a strict quality auditor. Is the following answer supported by the context? \n"
        f"1. Rate it 1-5 (5 is best).\n"
        f"2. Explain why in one sentence.\n\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"AI Answer: {answer} [/INST]\n"
        f"Evaluation: "
    )
    res = llm(eval_prompt, stop=["[/INST]", "</s>"], temperature=0.0, max_tokens=100)
    return res["choices"][0]["text"].strip()


def get_similarity_score(context, answer):
    """
    Calculates the mathematical Cosine Similarity between the context and the answer.
    Provides a non-LLM based metric to detect semantic drift or grounding.
    """
    # Convert text to vector embeddings
    emb1 = sim_model.encode(context)
    emb2 = sim_model.encode(answer)

    # Compute the cosine of the angle between vectors
    cos_sim = util.cos_sim(emb1, emb2)
    
    return float(cos_sim[0][0])

def final_rag_pipeline(query, context):
    """
    Generation and Evaluation.
    Returns a structured dictionary containing the answer and validation metrics.
    """
    # Generate the initial response
    ans = generate_answer(query, context)

    # Calculate similarity
    sim = get_similarity_score(context, ans)

    # Perform LLM-based evaluation
    eval_text = evaluate_answer(query, context, ans)
    
    return {
        "answer": ans,
        "metrics": {
            "similarity_score": round(sim, 4),
            "judge_feedback": eval_text
        }
    }