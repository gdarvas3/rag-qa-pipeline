import os

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util

class RAGEngine:
    """
    Core engine for handling RAG-based generation and automated evaluation.
    """
    def __init__(self, embedding_model_name: str, gemini_model: str = "gemini-2.5-flash-lite", api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to use the Gemini model.")

        self.client = genai.Client(api_key=api_key)
        self.model = gemini_model

        # Initialize the embedding model for similarity checks
        self.sim_model = SentenceTransformer(embedding_model_name)

    def _generate_text(self, prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        message = str(error).upper()
        return "RESOURCE_EXHAUSTED" in message or "QUOTA" in message

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a response based strictly on the provided context.
        """
        prompt = (
            "You are a precise RAG assistant. Answer only from the provided context. "
            "If the context does not contain the answer, say that the information is not available in the retrieved documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return self._generate_text(prompt, temperature=0.2, max_output_tokens=200)

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
            f"AI Answer: {answer}\n"
            f"Evaluation: "
        )
        return self._generate_text(eval_prompt, temperature=0.0, max_output_tokens=100)

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
        try:
            eval_text = self.evaluate_answer(query, context, ans)
        except Exception as e:
            if self._is_quota_error(e):
                eval_text = "Evaluation skipped due to Gemini quota limits."
            else:
                raise
        
        return {
            "answer": ans,
            "metrics": {
                "similarity_score": round(sim, 4),
                "judge_feedback": eval_text
            }
        }