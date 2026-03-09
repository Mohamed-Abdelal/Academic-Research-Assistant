# web search via DuckDuckGo

from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.documents import Document


class WebResearchTool:

    def __init__(self):
        self._ddgs = None

    def _get_ddgs(self):
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                self._ddgs = None
        return self._ddgs

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        ddgs = self._get_ddgs()
        if ddgs is None:
            return []
        try:
            return list(ddgs.text(query, max_results=num_results))
        except Exception:
            return []

    def search_with_filtering(
        self, query: str, num_results: int = 5, domain_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        raw = self.search(query, num_results=num_results * 2)
        if not domain_filter:
            return raw[:num_results]
        filtered = [
            r for r in raw
            if any(d in r.get("href", "") for d in domain_filter)
        ]
        return filtered[:num_results]

    def search_to_documents(self, query: str, num_results: int = 5) -> List[Document]:
        results = self.search(query, num_results=num_results)
        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.get("body", ""),
                metadata={
                    "title": r.get("title", "Web Result"),
                    "source": r.get("href", ""),
                    "authors": "Web Source",
                    "year": datetime.now().year,
                    "source_type": "web",
                    "journal": r.get("href", "").split("/")[2] if r.get("href") else "Web",
                },
            ))
        return docs
