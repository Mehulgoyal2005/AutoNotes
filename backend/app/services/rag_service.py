import json
import re
import numpy as np
from typing import List, Tuple
from langchain_groq import ChatGroq

NOT_FOUND_MESSAGE = (
    "I couldn't find a reliable answer to this question in the document. "
    "It may not be covered by the uploaded content — try rephrasing, or ask "
    "about a topic the document discusses."
)


class RAGService:
    """RAG service for answering questions based on document context"""

    def __init__(
        self,
        vector_store,
        groq_api_key: str = None,
        groundedness_threshold: float = 0.4,
        groundedness_min_ratio: float = 0.6
    ):
        self.vector_store = vector_store
        self.groundedness_threshold = groundedness_threshold
        self.groundedness_min_ratio = groundedness_min_ratio

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to the .env file at the project root."
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
        print("✅ RAG Service: Using Groq API")

    def _rewrite_query(self, question: str) -> List[str]:
        """
        Generate 2-3 paraphrases of the question via one standard Groq call.
        Returns [] on any failure so retrieval degrades gracefully to the
        original question alone.
        """
        prompt = f"""Rewrite the following question in 2-3 alternative phrasings that mean the same thing but use different vocabulary. This helps a document search engine find relevant passages.

Question: {question}

Respond with ONLY a JSON array of strings, nothing else. Example:
["variant one", "variant two", "variant three"]"""

        try:
            response = self.llm.invoke(prompt).content.strip()
            # Strip markdown code fences if the model added them
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            # Tolerate leading/trailing prose around the array
            start, end = response.find("["), response.rfind("]") + 1
            if start == -1 or end <= start:
                return []
            variants = json.loads(response[start:end])
            return [v for v in variants if isinstance(v, str) and v.strip()][:3]
        except Exception as e:
            print(f"⚠️  Query rewriting failed ({e}), using original question only")
            return []

    def _check_groundedness(
        self,
        answer: str,
        source_chunks: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Verify the answer against the retrieved chunks using embeddings.
        Splits the answer into sentences and checks each one's best cosine
        similarity against the chunks; sentences below the threshold are
        flagged as potentially unsupported.
        Returns (grounded_ratio, flagged_sentences).
        """
        # Strip citation markers and markdown before comparing meaning
        clean = re.sub(r"\[Source \d+\]", "", answer)
        clean = re.sub(r"[*_#`]", "", clean)
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean)
            if len(s.split()) >= 5  # skip headings/glue like "Here's how it works:"
        ]
        if not sentences:
            return 1.0, []

        sent_emb = self.vector_store.create_embeddings(sentences)
        chunk_emb = self.vector_store.create_embeddings(source_chunks)

        # Cosine similarity = dot product of L2-normalized vectors
        sent_emb = sent_emb / np.linalg.norm(sent_emb, axis=1, keepdims=True)
        chunk_emb = chunk_emb / np.linalg.norm(chunk_emb, axis=1, keepdims=True)
        best_sim = (sent_emb @ chunk_emb.T).max(axis=1)

        flagged = [s for s, sim in zip(sentences, best_sim)
                   if sim < self.groundedness_threshold]
        grounded_ratio = 1.0 - len(flagged) / len(sentences)
        return grounded_ratio, flagged

    def generate_answer(
        self,
        document_id: str,
        question: str,
        top_k: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Generate a citation-aware answer using RAG, verified for groundedness.
        Returns (answer, source_chunks).
        """

        # Rewrite the question into paraphrases, then retrieve with all
        # phrasings voting (hybrid: vector + BM25 -> RRF -> re-rank)
        variants = self._rewrite_query(question)
        if variants:
            print(f"🔁 Query variants: {variants}")
        results = self.vector_store.hybrid_search(
            document_id, question, top_k, variants=variants
        )

        if not results:
            return "No relevant information found in the document.", []

        source_chunks = [chunk for chunk, _, _ in results]

        # Citation-aware prompt: numbered sources, cite-per-claim instruction
        numbered_sources = "\n\n".join(
            f"[Source {i}]\n{chunk}" for i, chunk in enumerate(source_chunks, 1)
        )
        prompt = f"""You are a helpful assistant that answers questions using ONLY the numbered sources below.

RULES:
1. Base every claim on the sources. After each claim, cite the source it came from, like [Source 1] or [Source 2].
2. If the sources do not contain the information needed, say exactly: "This information is not available in the document." Do not guess or use outside knowledge.
3. Keep the answer clear and well-organized.

{numbered_sources}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        # Groundedness check: catch hallucination the prompt didn't prevent
        grounded_ratio, flagged = self._check_groundedness(answer, source_chunks)
        if flagged:
            print(f"🚩 Groundedness: {grounded_ratio:.0%} grounded; flagged sentences:")
            for s in flagged:
                print(f"   - {s[:120]}")

        if grounded_ratio < self.groundedness_min_ratio:
            print(f"❌ Answer failed groundedness check "
                  f"({grounded_ratio:.0%} < {self.groundedness_min_ratio:.0%}) — returning fallback")
            return NOT_FOUND_MESSAGE, source_chunks

        return answer, source_chunks
