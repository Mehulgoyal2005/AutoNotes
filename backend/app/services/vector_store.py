import faiss
import numpy as np
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Local vector store using FAISS + BM25 hybrid retrieval"""

    # Shared across instances; loaded lazily on first hybrid search so
    # instances that never retrieve (e.g. the upload route) skip the cost
    _reranker = None

    @classmethod
    def _get_reranker(cls):
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder
            print("🔄 Loading cross-encoder re-ranker (first hybrid search)...")
            cls._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return cls._reranker

    def __init__(self, embedding_model: str, vector_db_dir: Path):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db_dir = vector_db_dir
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.documents = {}  # document_id -> {index, chunks, metadata}

    def _get_index_path(self, document_id: str) -> Path:
        """Get path to FAISS index file for a document"""
        return self.vector_db_dir / f"{document_id}.faiss"

    def _get_metadata_path(self, document_id: str) -> Path:
        """Get path to metadata file for a document"""
        return self.vector_db_dir / f"{document_id}_metadata.pkl"

    def _get_bm25_path(self, document_id: str) -> Path:
        """Get path to BM25 index file for a document"""
        return self.vector_db_dir / f"{document_id}_bm25.pkl"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase word tokenization for BM25"""
        return re.findall(r"\w+", text.lower())

    def _build_bm25(self, document_id: str, texts: List[str]) -> BM25Okapi:
        """Build a BM25 index from chunk texts and persist it to disk"""
        bm25 = BM25Okapi([self._tokenize(t) for t in texts])
        with open(self._get_bm25_path(document_id), 'wb') as f:
            pickle.dump(bm25, f)
        return bm25

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype('float32')

    def add_document(self, document_id: str, chunks: List[Tuple[str, dict]]) -> int:
        """
        Add a document to the vector store.
        Returns the number of chunks added.
        """
        # Extract texts and metadata
        texts = [chunk[0] for chunk in chunks]
        metadata = [chunk[1] for chunk in chunks]

        # Create embeddings
        embeddings = self.create_embeddings(texts)

        # Create FAISS index
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)

        # Save index and metadata
        faiss.write_index(index, str(self._get_index_path(document_id)))

        with open(self._get_metadata_path(document_id), 'wb') as f:
            pickle.dump({'chunks': texts, 'metadata': metadata}, f)

        # Build BM25 keyword index from the same chunks
        bm25 = self._build_bm25(document_id, texts)

        # Store in memory
        self.documents[document_id] = {
            'index': index,
            'chunks': texts,
            'metadata': metadata,
            'bm25': bm25
        }

        return len(chunks)

    def load_document(self, document_id: str) -> bool:
        """Load a document from disk into memory"""
        index_path = self._get_index_path(document_id)
        metadata_path = self._get_metadata_path(document_id)

        if not index_path.exists() or not metadata_path.exists():
            return False

        # Load index
        index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        # Load BM25 index; rebuild it for documents ingested before BM25 existed
        bm25_path = self._get_bm25_path(document_id)
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                bm25 = pickle.load(f)
        else:
            bm25 = self._build_bm25(document_id, data['chunks'])

        self.documents[document_id] = {
            'index': index,
            'chunks': data['chunks'],
            'metadata': data['metadata'],
            'bm25': bm25
        }

        return True

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, dict, float]]:
        """
        Search for relevant chunks in a document.
        Returns list of (chunk_text, metadata, distance) tuples.
        """
        # Load document if not in memory
        if document_id not in self.documents:
            if not self.load_document(document_id):
                return []

        doc_data = self.documents[document_id]

        # Create query embedding
        query_embedding = self.create_embeddings([query])

        # Search in FAISS index
        distances, indices = doc_data['index'].search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # FAISS returns -1 for empty slots when top_k exceeds the number of vectors
            if 0 <= idx < len(doc_data['chunks']):
                results.append((
                    doc_data['chunks'][idx],
                    doc_data['metadata'][idx],
                    float(distance)
                ))

        return results

    def bm25_search(
        self,
        document_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, dict, float]]:
        """
        Keyword search using BM25.
        Returns list of (chunk_text, metadata, score) tuples — same shape as
        search(), except higher score = better (FAISS distance is the opposite).
        """
        if document_id not in self.documents:
            if not self.load_document(document_id):
                return []

        doc_data = self.documents[document_id]
        scores = doc_data['bm25'].get_scores(self._tokenize(query))

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (doc_data['chunks'][i], doc_data['metadata'][i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # ignore chunks with zero keyword overlap
        ]

    def hybrid_search(
        self,
        document_id: str,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
        rrf_k: int = 60,
        variants: Optional[List[str]] = None
    ) -> List[Tuple[str, dict, float]]:
        """
        Hybrid retrieval: vector search + BM25 in parallel, merged with
        Reciprocal Rank Fusion, then re-ranked by a cross-encoder.
        If `variants` (paraphrases of the query) are given, every variant
        votes into the same RRF pool; re-ranking always uses the original query.
        Returns list of (chunk_text, metadata, relevance_score) tuples,
        same shape as search() — higher score = more relevant.
        """
        if document_id not in self.documents:
            if not self.load_document(document_id):
                return []

        doc_data = self.documents[document_id]
        n_chunks = len(doc_data['chunks'])
        candidate_k = min(candidate_k, n_chunks)

        queries = [query] + [
            v.strip() for v in (variants or [])
            if isinstance(v, str) and v.strip() and v.strip() != query
        ]

        # Reciprocal Rank Fusion: merge by rank position, not raw score.
        # Both engines run for every query phrasing; all rankings vote.
        rrf_scores: dict = {}
        for q in queries:
            # Engine 1: vector search (ranked chunk indices)
            query_embedding = self.create_embeddings([q])
            _, indices = doc_data['index'].search(query_embedding, candidate_k)
            vector_ranking = [int(i) for i in indices[0] if 0 <= i < n_chunks]

            # Engine 2: BM25 keyword search (ranked chunk indices)
            scores = doc_data['bm25'].get_scores(self._tokenize(q))
            bm25_ranking = [
                int(i) for i in np.argsort(scores)[::-1][:candidate_k] if scores[i] > 0
            ]

            for ranking in (vector_ranking, bm25_ranking):
                for rank, chunk_idx in enumerate(ranking):
                    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        if not rrf_scores:
            return []
        candidates = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        # Cross-encoder re-ranking: score each (query, chunk) pair directly
        reranker = self._get_reranker()
        ce_scores = reranker.predict(
            [(query, doc_data['chunks'][ci]) for ci in candidates]
        )

        order = np.argsort(ce_scores)[::-1][:top_k]
        return [
            (
                doc_data['chunks'][candidates[i]],
                doc_data['metadata'][candidates[i]],
                float(ce_scores[i])
            )
            for i in order
        ]

    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the vector store"""
        return (
            self._get_index_path(document_id).exists() and
            self._get_metadata_path(document_id).exists()
        )

    def get_all_chunks(self, document_id: str) -> Optional[List[str]]:
        """Get all chunks for a document"""
        if document_id not in self.documents:
            if not self.load_document(document_id):
                return None

        return self.documents[document_id]['chunks']
