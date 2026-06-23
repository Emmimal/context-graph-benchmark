"""
Architecture 2: Vector-Only RAG.

Each turn (fact and distractor alike -- a real vector store does not know
in advance which turns matter) is embedded with TF-IDF and stored as a
chunk. On query, we retrieve the top-K most similar chunks by cosine
similarity and use only those chunks (not the full transcript) to answer.

TF-IDF is used instead of a neural embedding API because:
  1. The user explicitly ruled out API calls.
  2. TF-IDF is deterministic given fixed input -- no model weights to drift,
     no sampling, fully reproducible at seed level (scikit-learn's
     TfidfVectorizer has no randomness in its default configuration).
  3. It is a legitimate vector retrieval method, not a toy stand-in: it's
     widely used in production RAG systems as a sparse-retrieval baseline
     or hybrid component, so this is a fair representative of "vector-only
     RAG," not a strawman weakened to make the graph win.

This architecture's structural weakness is exactly what the article claims:
it retrieves chunks, not relationships. A "join" query that requires
combining two separate facts has no single chunk that contains the answer,
so similarity search alone cannot resolve it -- regardless of how good the
embedding is. We measure this honestly rather than asserting it.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tokenizer import count_tokens
from scenarios import Turn

TOP_K = 3


class VectorOnlyRAG:
    name = "vector_only_rag"

    def __init__(self, top_k: int = TOP_K):
        self.chunks: list[str] = []
        self.top_k = top_k

    def ingest(self, turn: Turn) -> None:
        self.chunks.append(f"{turn.speaker}: {turn.text}")

    def _retrieve(self, query_text: str) -> list[str]:
        if not self.chunks:
            return []
        # Fit fresh on each query (as a real vector DB's index reflects all
        # ingested docs at query time). Deterministic: TfidfVectorizer has
        # no random_state because it has no stochastic component.
        corpus = self.chunks + [query_text]
        vectorizer = TfidfVectorizer()
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Degenerate corpus (e.g. all-stopword query) -- no signal.
            return []
        query_vec = matrix[-1]
        doc_vecs = matrix[:-1]
        sims = cosine_similarity(query_vec, doc_vecs).flatten()
        ranked_idx = sims.argsort()[::-1]
        top_idx = [i for i in ranked_idx[: self.top_k] if sims[i] > 0]
        return [self.chunks[i] for i in top_idx]

    def answer_query(self, query_turn: Turn) -> tuple[str, int]:
        retrieved = self._retrieve(query_turn.text)
        prompt = "\n".join(retrieved) + f"\n{query_turn.speaker} (query): {query_turn.text}"
        tokens = count_tokens(prompt)
        answer = retrieved[0] if retrieved else "UNKNOWN"
        return answer, tokens
