# embedding-based relevance scoring

from typing import List, Tuple
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config


class RelevanceScorer:

    def __init__(self, embedding_model: str = None):
        model_name = embedding_model or config.EMBEDDING_MODEL
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def compute_embedding(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text))

    def compute_similarity(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_emb)
        if norm_q == 0 or norm_d == 0:
            return 0.0
        return float(np.dot(query_emb, doc_emb) / (norm_q * norm_d))

    def score_documents(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        query_emb = self.compute_embedding(query)
        scored = []
        for doc in documents:
            doc_emb = self.compute_embedding(doc.page_content[:500])
            score = self.compute_similarity(query_emb, doc_emb)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def filter_documents(
        self, query: str, documents: List[Document], threshold: float = 0.5
    ) -> List[Document]:
        scored = self.score_documents(query, documents)
        return [doc for doc, score in scored if score >= threshold]

    def rank_by_credibility(self, documents: List[Document]) -> List[Document]:
        return sorted(
            documents,
            key=lambda d: d.metadata.get("citations", 0),
            reverse=True,
        )
