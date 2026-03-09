# CLI entry point for research assistant

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_groq import ChatGroq
from langchain_core.documents import Document

import config
from sources.web_search import WebResearchTool
from sources.paper_database import PaperDatabaseTool
from sources.knowledge_base import LocalKnowledgeBase
from synthesis.summarizer import InformationSynthesizer
from synthesis.comparator import InformationComparator
from reporting.report_generator import ReportGenerator
from citation.citation_manager import CitationManager
from citation.citation_styles import list_styles
from filtering.relevance_scorer import RelevanceScorer
from filtering.domain_filters import DomainFilter


class ResearchAssistant:

    def __init__(self):
        print("\n[INFO] Initializing Academic Research Assistant...")

        if not config.GROQ_API_KEY:
            print("[ERROR] Please set GROQ_API_KEY in .env or environment.")
            sys.exit(1)

        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

        self.web_search = WebResearchTool()
        self.paper_db = PaperDatabaseTool()
        self.local_kb = LocalKnowledgeBase()

        self.synthesizer = InformationSynthesizer(llm=self.llm)
        self.comparator = InformationComparator(llm=self.llm)
        self.report_generator = ReportGenerator(llm=self.llm)
        self.citation_manager = CitationManager()
        self.relevance_scorer = RelevanceScorer()
        self.domain_filter = DomainFilter()

        print("[INFO] Research Assistant ready.\n")

    def _collect_sources(self, topic: str, depth: str = "medium") -> List[Document]:
        depth_map = {"shallow": 3, "medium": 5, "deep": 8}
        n = depth_map.get(depth, 5)

        print(f"  [Collect] Searching web for: '{topic}'")
        web_docs = self.web_search.search_to_documents(topic, num_results=n)

        print(f"  [Collect] Searching academic papers for: '{topic}'")
        paper_docs = self.paper_db.search_to_documents(topic, num_results=n)

        all_docs = web_docs + paper_docs
        print(f"  [Collect] Retrieved {len(all_docs)} total documents")
        return all_docs

    def _filter_and_rank(
        self, documents: List[Document], query: str,
        domain: str = "general", min_year: Optional[int] = None,
    ) -> List[Document]:
        df = DomainFilter(domain)
        docs = df.filter_by_domain(documents, strict=False)
        if min_year:
            docs = df.filter_by_year(docs, min_year=min_year)
        docs = self.relevance_scorer.filter_documents(
            query, docs, threshold=config.RELEVANCE_THRESHOLD
        )
        docs = self.relevance_scorer.rank_by_credibility(docs)
        print(f"  [Filter] {len(docs)} documents passed filtering")
        return docs

    def conduct_research(
        self, topic: str, depth: str = "medium", report_type: str = "summary",
        citation_style: str = "apa", domain: str = "general",
        min_year: Optional[int] = None, custom_sections: Optional[List[str]] = None,
    ) -> str:
        print(f"\n[Research] Topic: '{topic}'")
        print(f"[Research] Depth: {depth} | Report: {report_type} | Citations: {citation_style}")

        documents = self._collect_sources(topic, depth)
        if not documents:
            return "No research sources found. Please try a different topic."

        filtered_docs = self._filter_and_rank(documents, topic, domain, min_year)
        if not filtered_docs:
            print("  [Filter] No docs passed threshold; using all documents.")
            filtered_docs = documents[:5]

        self.citation_manager.clear()
        self.citation_manager.add_sources_from_documents(filtered_docs)

        print("  [Synthesize] Generating synthesis...")
        synthesis = self.synthesizer.synthesize_with_query_focus(filtered_docs[:6], topic)

        print("  [Report] Generating report...")
        if custom_sections:
            report = self.report_generator.generate_structured_report(
                content=synthesis, topic=topic, sections=custom_sections,
            )
        else:
            report = self.report_generator.generate_report(
                content=synthesis, topic=topic, report_type=report_type,
            )

        bibliography = self.citation_manager.generate_bibliography(style=citation_style)
        return report + "\n\n" + bibliography

    def quick_summary(self, topic: str) -> str:
        print(f"\n[QuickSearch] Topic: '{topic}'")
        web_docs = self.web_search.search_to_documents(topic, num_results=3)
        if not web_docs:
            return "No results found."
        return self.synthesizer.synthesize_documents(web_docs[:3], chain_type="stuff")

    def compare_perspectives(self, topic: str) -> str:
        documents = self._collect_sources(topic, depth="medium")
        if not documents:
            return "No sources found."
        return self.comparator.compare_sources(documents[:6], topic)

    def run_interactive(self) -> None:
        print("=" * 70)
        print("  Academic Research Assistant")
        print("  Powered by Groq LLM + DuckDuckGo + LangChain")
        print("=" * 70)
        print("\nCommands:")
        print("  research <topic>       - Full research report")
        print("  detailed <topic>       - Detailed multi-section report")
        print("  compare <topic>        - Comparative analysis")
        print("  quick <topic>          - Fast web-only summary")
        print("  citations              - Show citation styles")
        print("  quit                   - Exit\n")

        while True:
            try:
                user_input = input("Research> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Session ended]")
                break

            if not user_input:
                continue

            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            topic = parts[1].strip() if len(parts) > 1 else ""

            if command == "quit":
                print("Goodbye!")
                break
            elif command == "citations":
                print("\n" + list_styles() + "\n")
            elif command == "quick":
                if not topic:
                    print("Usage: quick <topic>")
                    continue
                print(self.quick_summary(topic))
            elif command == "compare":
                if not topic:
                    print("Usage: compare <topic>")
                    continue
                print(self.compare_perspectives(topic))
            elif command == "detailed":
                if not topic:
                    print("Usage: detailed <topic>")
                    continue
                result = self.conduct_research(topic, depth="deep", report_type="detailed")
                print(f"\n{result}\n")
            elif command == "research":
                if not topic:
                    print("Usage: research <topic>")
                    continue
                result = self.conduct_research(topic)
                print(f"\n{result}\n")
            else:
                result = self.conduct_research(user_input)
                print(f"\n{result}\n")


if __name__ == "__main__":
    assistant = ResearchAssistant()
    assistant.run_interactive()
