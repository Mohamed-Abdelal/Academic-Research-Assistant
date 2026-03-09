# FAISS vector store for research docs

from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


class LocalKnowledgeBase:

    def __init__(self, embedding_model: Optional[str] = None):
        model_name = embedding_model or config.EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.vector_store = None

    def add_documents(self, documents: List[Document]) -> None:
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

    def search(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
