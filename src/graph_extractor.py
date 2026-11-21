import json
import os
import re
import time
from typing import Dict, List

from google import genai
from google.genai import types


class GeminiGraphExtractor:
    """Extracts entities and relations from free text using Gemini."""

    def __init__(self, model: str | None = None, api_key: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.max_retries = int(os.getenv("GRAPH_EXTRACT_RETRIES", "3"))
        self.initial_backoff_seconds = float(os.getenv("GRAPH_EXTRACT_BACKOFF_SECONDS", "1.0"))
        self.last_error = ""

        self.client = None
        if not enabled or not api_key:
            self.enabled = False
            return

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception:
            self.enabled = False
            self.client = None

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        message = str(error).upper()
        return (
            "503" in message
            or "UNAVAILABLE" in message
            or "RESOURCE_EXHAUSTED" in message
            or "RATE" in message
            or "TIMEOUT" in message
        )

    def _generate_json(self, prompt: str, max_output_tokens: int = 400) -> str:
        if not self.enabled or self.client is None:
            return ""

        backoff = max(0.1, self.initial_backoff_seconds)
        attempts = max(1, self.max_retries)
        for attempt in range(attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
                self.last_error = ""
                return (response.text or "").strip()
            except Exception as e:
                self.last_error = str(e)
                if attempt == attempts - 1 or not self._is_transient_error(e):
                    return ""
                time.sleep(backoff)
                backoff *= 2

        return ""

    @staticmethod
    def _safe_load_json(text: str) -> Dict:
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Tolerate fenced-code responses from some model variants.
            cleaned = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {}

    def extract_triplets(self, text: str, source: str = "", chunk_id: str = "") -> List[Dict]:
        if not self.enabled:
            return []

        prompt = (
            "Extract factual relationship triplets from the text. "
            "Return STRICT JSON with this schema: "
            "{\"triplets\":[{\"head\":\"...\",\"relation\":\"...\",\"tail\":\"...\",\"confidence\":0.0}]}\n"
            "Rules:\n"
            "- Use concise entity names.\n"
            "- Keep only explicit factual relations from the text.\n"
            "- Confidence must be between 0 and 1.\n"
            f"Text:\n{text[:4000]}"
        )

        raw = self._generate_json(prompt, max_output_tokens=700)
        payload = self._safe_load_json(raw)
        triplets = payload.get("triplets", []) if isinstance(payload, dict) else []

        output: List[Dict] = []
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue
            output.append(
                {
                    "head": triplet.get("head", ""),
                    "relation": triplet.get("relation", ""),
                    "tail": triplet.get("tail", ""),
                    "confidence": triplet.get("confidence", 0.0),
                    "source": source,
                    "chunk_id": chunk_id,
                }
            )

        return output

    def extract_triplets_batch(self, chunk_items: List[Dict], max_chars_per_chunk: int = 1200) -> List[Dict]:
        """Extracts triplets for multiple chunks with a single model call."""
        if not self.enabled or not chunk_items:
            return []

        compact_items = []
        source_by_chunk_id: Dict[str, str] = {}
        for item in chunk_items:
            chunk_id = str(item.get("chunk_id", "")).strip()
            text = str(item.get("text", "")).strip()
            source = str(item.get("source", "")).strip()
            if not chunk_id or not text:
                continue
            source_by_chunk_id[chunk_id] = source
            compact_items.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[:max_chars_per_chunk],
                }
            )

        if not compact_items:
            return []

        payload_text = json.dumps({"chunks": compact_items}, ensure_ascii=False)
        prompt = (
            "You are an information extraction engine. "
            "Extract factual relationship triplets for each chunk. "
            "Return STRICT JSON with this schema: "
            "{\"items\":[{\"chunk_id\":\"...\",\"triplets\":[{\"head\":\"...\",\"relation\":\"...\",\"tail\":\"...\",\"confidence\":0.0}]}]}.\n"
            "Rules:\n"
            "- Preserve the input chunk_id exactly.\n"
            "- Keep only explicit relations from each chunk text.\n"
            "- Confidence must be between 0 and 1.\n"
            "Input chunks JSON:\n"
            f"{payload_text}"
        )

        raw = self._generate_json(prompt, max_output_tokens=1500)
        parsed = self._safe_load_json(raw)
        items = parsed.get("items", []) if isinstance(parsed, dict) else []

        output: List[Dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", "")).strip()
            triplets = item.get("triplets", [])
            if not chunk_id or not isinstance(triplets, list):
                continue

            for triplet in triplets:
                if not isinstance(triplet, dict):
                    continue
                output.append(
                    {
                        "head": triplet.get("head", ""),
                        "relation": triplet.get("relation", ""),
                        "tail": triplet.get("tail", ""),
                        "confidence": triplet.get("confidence", 0.0),
                        "source": source_by_chunk_id.get(chunk_id, ""),
                        "chunk_id": chunk_id,
                    }
                )

        return output

    def extract_entities(self, text: str, max_entities: int = 8) -> List[str]:
        if not self.enabled:
            return []

        prompt = (
            "Extract the most important named entities from the question. "
            "Return STRICT JSON with schema: {\"entities\":[\"...\"]}. "
            f"Return at most {max_entities} entities.\n"
            f"Question:\n{text[:2000]}"
        )

        raw = self._generate_json(prompt, max_output_tokens=200)
        payload = self._safe_load_json(raw)
        entities = payload.get("entities", []) if isinstance(payload, dict) else []

        normalized: List[str] = []
        for entity in entities:
            if not isinstance(entity, str):
                continue
            clean = entity.strip()
            if clean:
                normalized.append(clean)

        return normalized[:max_entities]
