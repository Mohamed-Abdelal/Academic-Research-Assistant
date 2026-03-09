# tracks sources and generates citations

from typing import List, Dict, Optional
from langchain_core.documents import Document
from citation.citation_styles import STYLES


class CitationManager:

    def __init__(self):
        self.sources: List[Dict] = []

    def clear(self) -> None:
        self.sources = []

    def add_source(self, source: Dict) -> None:
        if source.get("title") not in {s.get("title") for s in self.sources}:
            self.sources.append(source)

    def add_sources_from_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            meta = doc.metadata
            self.add_source({
                "title": meta.get("title", "Untitled"),
                "authors": meta.get("authors", "Unknown"),
                "year": meta.get("year", "n.d."),
                "journal": meta.get("journal", ""),
                "doi": meta.get("doi", ""),
                "citations": meta.get("citations", 0),
                "source_type": meta.get("source_type", "unknown"),
            })

    def format_citation(self, source: Dict, style: str = "apa") -> str:
        formatter = STYLES.get(style.lower())
        if formatter is None:
            raise ValueError(f"Unknown style: {style}. Available: {list(STYLES.keys())}")
        return formatter(source)

    def generate_bibliography(self, style: str = "apa") -> str:
        if not self.sources:
            return "No sources recorded."
        formatter = STYLES.get(style.lower())
        if formatter is None:
            raise ValueError(f"Unknown style: {style}")
        lines = [f"## References ({style.upper()})", ""]
        sorted_sources = sorted(self.sources, key=lambda s: s.get("authors", ""))
        for i, src in enumerate(sorted_sources, 1):
            lines.append(f"{i}. {formatter(src)}")
        return "\n".join(lines)

    def assess_credibility(self, source: Dict) -> str:
        citations = source.get("citations", 0)
        source_type = source.get("source_type", "unknown")
        if source_type == "academic" and citations > 5000:
            return "High"
        elif source_type == "academic" and citations > 1000:
            return "Medium-High"
        elif source_type == "academic":
            return "Medium"
        elif source_type == "web":
            return "Low-Medium"
        return "Unknown"
