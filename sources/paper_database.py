# paper_database.py - searches Semantic Scholar and arXiv for real papers

import re
import time
import requests
from typing import List, Dict, Optional
from langchain_core.documents import Document


def _keyword_relevance(query: str, title: str, abstract: str) -> float:
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are',
                  'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but',
                  'can', 'will', 'its', 'into', 'use', 'using', 'based'}
    query_words -= stop_words
    if not query_words:
        return 0.5

    text = (title + " " + abstract).lower()
    matches = sum(1 for w in query_words if w in text)
    return round(matches / len(query_words), 2)


class SemanticScholarAPI:

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Research-Assistant/1.0'
        })

    def search_papers(self, query: str, limit: int = 10, fields: Optional[List[str]] = None) -> List[Dict]:
        if fields is None:
            fields = ['title', 'abstract', 'authors', 'year', 'citationCount',
                     'venue', 'externalIds', 'publicationTypes', 'fieldsOfStudy']

        url = f"{self.BASE_URL}/paper/search"
        params = {
            'query': query,
            'limit': min(limit, 100),
            'fields': ','.join(fields)
        }

        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                if papers:
                    print(f"    -> Got {len(papers)} results from Semantic Scholar")
                return papers
            elif response.status_code == 429:
                print(f"    -> Semantic Scholar rate limited, retrying...")
                time.sleep(2)
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('data', [])
            else:
                print(f"    -> Semantic Scholar returned status {response.status_code}")
            return []
        except Exception as e:
            print(f"    -> Semantic Scholar error: {str(e)[:80]}")
            return []


class ArXivAPI:

    BASE_URL = "https://export.arxiv.org/api/query"

    def _build_query(self, query: str) -> str:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this',
                      'are', 'was', 'not', 'but', 'can', 'will', 'its', 'use'}
        terms = [w for w in words if w.lower() not in stop_words]
        if not terms:
            return f'all:{query}'
        # spaces become + in URL encoding, which arXiv interprets correctly
        return ' AND '.join(f'all:{t}' for t in terms)

    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        params = {
            'search_query': self._build_query(query),
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            if response.status_code == 200:
                papers = self._parse_arxiv_response(response.text)
                if papers:
                    print(f"    -> Got {len(papers)} results from arXiv")
                return papers
            return []
        except Exception as e:
            print(f"    -> arXiv error: {str(e)[:80]}")
            return []

    def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        import xml.etree.ElementTree as ET

        papers = []
        try:
            root = ET.fromstring(xml_text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)
                authors = entry.findall('atom:author', ns)
                id_elem = entry.find('atom:id', ns)

                paper = {
                    'title': title.text.strip() if title is not None else '',
                    'abstract': summary.text.strip() if summary is not None else '',
                    'year': int(published.text[:4]) if published is not None else 0,
                    'authors': ', '.join([a.find('atom:name', ns).text for a in authors if a.find('atom:name', ns) is not None]),
                    'venue': 'arXiv',
                    'arxiv_id': id_elem.text.split('/')[-1] if id_elem is not None else '',
                }
                papers.append(paper)
        except Exception as e:
            print(f"[Warning] arXiv parsing error: {e}")

        return papers


class PaperDatabaseTool:

    RELEVANCE_THRESHOLD = 0.4

    def __init__(self):
        self.semantic_scholar = SemanticScholarAPI()
        self.arxiv = ArXivAPI()

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        all_papers = []

        print(f"  [PaperDB] Searching Semantic Scholar for: {query}")
        s2_papers = self.semantic_scholar.search_papers(query, limit=num_results * 2)

        for paper in s2_papers:
            doi = ""
            if paper.get('externalIds'):
                doi = (paper['externalIds'].get('DOI', '') or
                      paper['externalIds'].get('ArXiv', ''))

            authors = "Unknown"
            if paper.get('authors'):
                author_list = [a.get('name', '') for a in paper['authors'] if a.get('name')]
                if author_list:
                    if len(author_list) <= 3:
                        authors = ', '.join(author_list)
                    else:
                        authors = f"{author_list[0]} et al."

            title = paper.get('title', 'Untitled')
            abstract = paper.get('abstract', '') or ''
            kw_rel = _keyword_relevance(query, title, abstract)

            citation_bonus = min(0.15, (paper.get('citationCount', 0) / 50000))
            relevance = round(min(0.98, kw_rel * 0.85 + citation_bonus), 2)

            standardized = {
                'title': title,
                'abstract': abstract if abstract else 'No abstract available.',
                'authors': authors,
                'year': paper.get('year', 0),
                'journal': paper.get('venue', 'Unknown Venue'),
                'citations': paper.get('citationCount', 0),
                'doi': doi,
                'relevance': relevance,
            }
            all_papers.append(standardized)

        print(f"  [PaperDB] Searching arXiv for: {query}")
        arxiv_papers = self.arxiv.search_papers(query, max_results=max(3, num_results))

        for paper in arxiv_papers:
            title_lower = paper.get('title', '').lower()
            is_duplicate = any(
                title_lower[:50] == existing['title'].lower()[:50]
                for existing in all_papers
            )

            if not is_duplicate:
                title = paper.get('title', 'Untitled')
                abstract = paper.get('abstract', '') or ''
                kw_rel = _keyword_relevance(query, title, abstract)

                standardized = {
                    'title': title,
                    'abstract': abstract if abstract else 'No abstract available.',
                    'authors': paper.get('authors', 'Unknown'),
                    'year': paper.get('year', 0),
                    'journal': 'arXiv',
                    'citations': 0,
                    'doi': f"arXiv:{paper.get('arxiv_id', '')}",
                    'relevance': round(kw_rel * 0.85, 2),
                }
                all_papers.append(standardized)

        # Filter out papers below relevance threshold
        before = len(all_papers)
        all_papers = [p for p in all_papers if p['relevance'] >= self.RELEVANCE_THRESHOLD]
        if before > len(all_papers):
            print(f"  [PaperDB] Filtered out {before - len(all_papers)} irrelevant papers")

        all_papers.sort(
            key=lambda p: (p.get('relevance', 0), p.get('year', 0)),
            reverse=True
        )

        print(f"  [PaperDB] Returning {min(num_results, len(all_papers))} relevant papers")
        return all_papers[:num_results]

    def search_to_documents(self, query: str, num_results: int = 5) -> List[Document]:
        papers = self.search(query, num_results)

        documents = []
        for p in papers:
            abstract = p.get("abstract", "")
            if not abstract or abstract == "No abstract available.":
                abstract = f"Paper titled '{p.get('title', 'Unknown')}' by {p.get('authors', 'Unknown')}."

            doc = Document(
                page_content=abstract,
                metadata={
                    "title": p.get("title", "Untitled"),
                    "authors": p.get("authors", "Unknown"),
                    "year": p.get("year", 0),
                    "journal": p.get("journal", "Unknown"),
                    "citations": p.get("citations", 0),
                    "doi": p.get("doi", ""),
                    "source_type": "academic",
                    "relevance": p.get("relevance", 0.5),
                },
            )
            documents.append(doc)

        return documents

    def get_developments(self, query: str) -> str:
        papers = self.search(f"{query} recent developments 2024 2025", num_results=5)

        if papers:
            recent_titles = [p['title'] for p in papers[:3]]
            return f"Recent developments: " + "; ".join(recent_titles)

        return "Active research area with substantial recent publications."
