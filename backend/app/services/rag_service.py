from typing import List, Tuple
from langchain_groq import ChatGroq


class RAGService:
    """RAG service for answering questions based on document context"""

    def __init__(self, vector_store, groq_api_key: str = None):
        self.vector_store = vector_store

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to the .env file at the project root."
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
        print("✅ RAG Service: Using Groq API")

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

        response = self.llm.invoke(prompt)
        return response.content, source_chunks
