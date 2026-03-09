# Academic Research Assistant

Research assistant that searches real academic databases (Semantic Scholar + arXiv) and the web (DuckDuckGo), synthesizes findings, and generates reports with proper citations. Built with LangChain + FAISS + Groq (LLaMA 3.3 70B).

## What it does

- **Searches real academic papers** via Semantic Scholar API (200M+ papers) and arXiv API (preprints)
- **Web search** via DuckDuckGo for current information
- Synthesizes findings using LangChain summarize chains (map-reduce and stuff)
- Generates 5 report types: summary, detailed, comparative, literature review, executive summary
- Tracks all sources and generates bibliographies in APA, MLA, or Chicago format
- Scores document relevance using embedding-based cosine similarity (all-MiniLM-L6-v2)
- Filters results by domain, year, and source type
- Assesses source credibility based on citation count and source type
- Works as both a CLI app and a Streamlit web app

## Data Sources

| Source | What it provides | Notes |
|--------|-----------------|-------|
| Semantic Scholar API | Academic papers across all fields | Free, may be rate-limited (~100 req/5 min) |
| arXiv API | Preprints (CS, Physics, Math, Biology) | Free, no rate limits |
| DuckDuckGo | Live web search results | Free, no API key needed |

## Project Structure

```
project2_research_assistant_v2/
├── sources/
│   ├── web_search.py           # DuckDuckGo search
│   ├── paper_database.py       # Semantic Scholar + arXiv API clients
│   └── knowledge_base.py       # FAISS local vector store
├── synthesis/
│   ├── summarizer.py           # LangChain summarize chains + query-focused synthesis
│   └── comparator.py           # Compare findings across sources
├── reporting/
│   ├── report_generator.py     # 5 report formats via LLMChain
│   └── templates/
│       ├── summary_template.py
│       ├── detailed_template.py
│       └── comparative_template.py
├── citation/
│   ├── citation_manager.py     # Source tracking + bibliography generation
│   └── citation_styles.py      # APA, MLA, Chicago formatters
├── filtering/
│   ├── relevance_scorer.py     # Cosine similarity scoring with HuggingFace embeddings
│   └── domain_filters.py       # Domain/year/type filters
├── config.py                   # Centralized settings
├── main.py                     # CLI entry point
├── streamlit_app.py            # Streamlit web UI
├── test_real_papers.py         # Test script for paper search APIs
└── requirements.txt
```

## LangChain Components Used

| Component | Where | What for |
|---|---|---|
| `load_summarize_chain` | summarizer.py | Map-reduce / stuff summarization |
| `LLMChain` + `PromptTemplate` | summarizer.py, comparator.py, report_generator.py, streamlit_app.py | Synthesis, comparison, reports |
| `HuggingFaceEmbeddings` | relevance_scorer.py, knowledge_base.py | Embedding vectors for cosine similarity |
| `FAISS` | knowledge_base.py | Local vector store for document storage |
| `RecursiveCharacterTextSplitter` | knowledge_base.py | Chunking documents |
| `Document` | everywhere | LangChain document format with metadata |
| `ChatGroq` | main.py, streamlit_app.py | LLM (Groq LLaMA 3.3 70B) |

## Setup

You need Python 3.10+ and a free Groq API key from https://console.groq.com/keys.

```bash
# clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd project2_research_assistant_v2

# (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# install dependencies
pip install -r requirements.txt
```

First run downloads the embedding model (~80 MB), after that it's cached locally.

## Running

**Streamlit UI (recommended):**
```bash
python -m streamlit run streamlit_app.py
```
Opens at http://localhost:8501. Paste your Groq API key in the sidebar and start researching. No `.env` file needed. The app has 3 tabs: Research Chat, Generate Report, and Sources Library.

**CLI:**
```bash
# for CLI you need a .env file with your key
echo GROQ_API_KEY=gsk_your_key_here > .env
python main.py
```

CLI commands:
- `research machine learning in healthcare` - full research report
- `detailed quantum computing` - detailed multi-section report
- `compare AI ethics approaches` - comparative analysis
- `quick neural networks` - fast web-only summary
- `citations` - show collected citation styles
- `quit` - exit

## How it works

1. **Data collection** - Searches DuckDuckGo, Semantic Scholar, and arXiv for the given topic. arXiv uses AND queries so all search terms must appear in the paper.
2. **Relevance filtering** - Each paper is scored by keyword overlap between the query and the paper's title + abstract. Papers below 40% relevance are discarded to prevent the LLM from citing unrelated papers. The CLI path also uses embedding-based cosine similarity for deeper filtering.
3. **Synthesis** - Processes each relevant document individually with LangChain LLMChain, then combines partial results into a unified synthesis. The LLM is instructed to only cite papers genuinely relevant to the query.
4. **Report generation** - Applies the selected report template (summary/detailed/comparative/literature review/executive summary)
5. **Citations** - Generates a bibliography in the chosen style (APA/MLA/Chicago) with credibility assessment based on citation count and source type

Searches take ~10-25 seconds depending on API response times.

## Testing the paper search

```bash
python test_real_papers.py
```
Runs test queries against Semantic Scholar and arXiv to verify the APIs are working.

## Notes

- Semantic Scholar free tier has rate limits (~100 requests per 5 minutes). If rate-limited, the code retries once then falls back to arXiv results.
- arXiv has no strict rate limits but please use it politely.
- The `LangChainDeprecationWarning` about memory migration is harmless and doesn't affect functionality.
- On Python 3.13 you may see a `torch._classes RuntimeError` warning from Streamlit's file watcher. This is cosmetic and doesn't affect the app.

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io and click New App
3. Pick your repo, set main file to `streamlit_app.py`
4. In Advanced settings > Secrets, add: `GROQ_API_KEY = "gsk_your_key_here"`
5. Deploy

---
Built for Lab 4 - AI-Based Applications course.

