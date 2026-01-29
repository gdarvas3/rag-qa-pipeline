from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

class RAGEngine:
    """
    Core engine for handling RAG-based generation and automated evaluation.
    """
    def __init__(self, model_path: str, embedding_model_name: str):
        # Initialize the LLM with specific parameters
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )
        # Initialize the embedding model for similarity checks
        self.sim_model = SentenceTransformer(embedding_model_name)

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a response based strictly on the provided context.
        Uses Mistral-style [INST] tags for optimal instruction following.
        """
        prompt = (
            f"[INST]Answer the question based on the context\n"
            f"Context:\n{context}\n\n"
            f"Question: {query} [/INST]\n"
            "Answer:"
        )
        response = self.llm(
            prompt,
            stop=["[/INST]", "</s>", "Question:", "Context:"],
            echo=False,
            temperature=0.2,
            max_tokens=200
        )
        return response["choices"][0]["text"].strip()

    def evaluate_answer(self, query: str, context: str, answer: str) -> str:
        """
        LLM-as-a-Judge: The AI audits its own output.
        """
        eval_prompt = (
            f"[INST] You are a strict quality auditor. Is the following answer supported by the context? \n"
            f"1. Rate it 1-5 (5 is best).\n"
            f"If the answer correctly states that the information is NOT present in the context, award a HIGH SCORE (5). This is a successful avoidance of hallucination."
            f"2. Explain why in one sentence.\n\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"AI Answer: {answer} [/INST]\n"
            f"Evaluation: "
        )
        res = self.llm(eval_prompt, stop=["[/INST]", "</s>"], temperature=0.0, max_tokens=100)
        return res["choices"][0]["text"].strip()

    def get_similarity_score(self, context: str, answer: str) -> float:
        """
        Calculates the cosine similarity between context and answer.
        """
        if "no information found" in answer.lower():
            return 1.0

        emb1 = self.sim_model.encode(context)
        emb2 = self.sim_model.encode(answer)
        cos_sim = util.cos_sim(emb1, emb2)
        return float(cos_sim[0][0])

    def process_query(self, query: str, context: str) -> dict:
        """
        Orchestrates the full RAG pipeline.
        Returns a dictionary containing the answer and all calculated metrics.
        """
        ans = self.generate_answer(query, context)
        sim = self.get_similarity_score(context, ans)
        eval_text = self.evaluate_answer(query, context, ans)
        
        return {
            "answer": ans,
            "metrics": {
                "similarity_score": round(sim, 4),
                "judge_feedback": eval_text
            }
        }