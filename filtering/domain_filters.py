# filters by domain, year, source type

from typing import List, Optional
from langchain_core.documents import Document


DOMAIN_KEYWORDS = {
    "computer_science": ["machine learning", "deep learning", "neural", "algorithm", "software", "computing", "AI"],
    "medicine": ["clinical", "patient", "diagnosis", "treatment", "medical", "health", "disease"],
    "physics": ["quantum", "particle", "energy", "relativity", "physics", "matter"],
    "environment": ["climate", "carbon", "renewable", "emission", "sustainability", "ecology"],
    "social_science": ["education", "society", "policy", "economic", "behavior", "culture"],
}


class DomainFilter:

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.keywords = DOMAIN_KEYWORDS.get(domain, [])

    def filter_by_domain(self, documents: List[Document], strict: bool = False) -> List[Document]:
        if self.domain == "general" or not self.keywords:
            return documents
        filtered = []
        for doc in documents:
            text = doc.page_content.lower()
            title = doc.metadata.get("title", "").lower()
            combined = text + " " + title
            matches = sum(1 for kw in self.keywords if kw in combined)
            if strict and matches >= 2:
                filtered.append(doc)
            elif not strict and matches >= 1:
                filtered.append(doc)
        return filtered if filtered else documents

    def filter_by_year(self, documents: List[Document], min_year: int = 2000) -> List[Document]:
        return [d for d in documents if d.metadata.get("year", 9999) >= min_year]

    def filter_by_source_type(
        self, documents: List[Document], source_types: Optional[List[str]] = None
    ) -> List[Document]:
        if not source_types:
            return documents
        return [d for d in documents if d.metadata.get("source_type", "") in source_types]
