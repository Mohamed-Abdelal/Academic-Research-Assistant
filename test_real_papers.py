# test_real_papers.py - verify real academic paper search works

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sources.paper_database import PaperDatabaseTool

def test_paper_search():
    print("=" * 70)
    print("Testing Real Academic Paper Search")
    print("=" * 70)

    tool = PaperDatabaseTool()

    test_queries = [
        "machine learning transformers",
        "climate change renewable energy",
        "quantum computing algorithms",
        "healthcare artificial intelligence"
    ]

    for query in test_queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 70)

        papers = tool.search(query, num_results=3)

        if papers:
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper['title']}")
                print(f"   Authors: {paper['authors']}")
                print(f"   Year: {paper['year']} | Journal: {paper['journal']}")
                print(f"   Citations: {paper['citations']} | DOI: {paper.get('doi', 'N/A')}")
                print(f"   Abstract: {paper['abstract'][:150]}...")
        else:
            print("   No papers found")

    print("\n\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_paper_search()
