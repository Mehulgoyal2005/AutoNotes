from typing import List, Tuple
from langchain_groq import ChatGroq
import httpx


class RAGService:
    """RAG service for answering questions based on document context"""

    def __init__(
        self,
        vector_store,
        groq_api_key: str = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3"
    ):
        self.vector_store = vector_store
        self.groq_api_key = groq_api_key
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        # Initialize LLM (prefer Groq API if available)
        self.use_ollama = True
        self.llm = None

        if groq_api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama-3.3-70b-versatile"
                )
                self.use_ollama = False
                print("✅ RAG Service: Using Groq API")
            except Exception as e:
                print(f"⚠️ Groq init failed ({e}), using Ollama")
        else:
            print("ℹ️ Using local Ollama")

    def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama model"""
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No response from model")
                else:
                    print("Ollama error:", response.text)
                    return "Error: Unable to get response from Ollama"

        except Exception as e:
            print("Connection error:", str(e))
            return f"Error connecting to Ollama: {str(e)}"

    def generate_answer(
        self,
        document_id: str,
        question: str,
        top_k: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Generate an answer using RAG.
        Returns (answer, source_chunks).
        """

        # Retrieve relevant chunks
        results = self.vector_store.search(document_id, question, top_k)

        if not results:
            return "No relevant information found in the document.", []

        # Prepare context
        context = "\n\n".join([chunk for chunk, _, _ in results])
        source_chunks = [chunk for chunk, _, _ in results]

        # Prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        # Generate answer
        if self.use_ollama:
            print("🤖 Using Ollama...")
            answer = self._query_ollama(prompt)
        else:
            print("🤖 Using Groq...")
            response = self.llm.invoke(prompt)
            answer = response.content

        return answer, source_chunks