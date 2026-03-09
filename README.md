# Academic Research Assistant

Research tool that gathers info from web search + a paper database, synthesizes it, and generates reports with proper citations. Uses LangChain + FAISS + Groq (LLaMA 3.3 70B).

## What it does

- Searches DuckDuckGo and a built-in academic paper database (10 domains, ~30 papers)
- Synthesizes findings using LangChain summarize chains
- Generates 5 types of reports (summary, detailed, comparative, literature review, executive summary)
- Tracks all sources and generates bibliographies in APA, MLA, or Chicago
- Scores document relevance using embedding-based cosine similarity
- Filters by domain, year, and source type
- Works as both a CLI app and a Streamlit web app

## Project Structure

```
academic_research_assistant/
├── sources/
│   ├── web_search.py           # DuckDuckGo search
│   ├── paper_database.py       # Mock academic papers (10 domains)
│   └── knowledge_base.py       # FAISS local KB
├── synthesis/
│   ├── summarizer.py           # Summarize chains + query-focused synthesis
│   └── comparator.py           # Compare findings across sources
├── reporting/
│   ├── report_generator.py     # 5 report formats
│   └── templates/
│       ├── summary_template.py
│       ├── detailed_template.py
│       └── comparative_template.py
├── citation/
│   ├── citation_manager.py     # Source tracking + bibliography
│   └── citation_styles.py      # APA, MLA, Chicago
├── filtering/
│   ├── relevance_scorer.py     # Cosine similarity scoring
│   └── domain_filters.py       # Domain/year/type filters
├── config.py
├── main.py                     # CLI
├── streamlit_app.py            # Web UI
└── requirements.txt
```

## LangChain Components

| Component | Where | What for |
|---|---|---|
| load_summarize_chain | summarizer.py | Map-reduce / stuff summarization |
| LLMChain + PromptTemplate | summarizer.py, comparator.py, report_generator.py | Synthesis, comparison, reports |
| HuggingFaceEmbeddings | relevance_scorer.py, knowledge_base.py | Embedding vectors for similarity |
| FAISS | knowledge_base.py | Local vector store |
| RecursiveCharacterTextSplitter | knowledge_base.py | Chunking docs |
| Document | everywhere | LangChain doc format with metadata |
| ChatGroq | everywhere | LLM (Groq) |

## Setup

You need Python 3.10+ and a Groq API key (free at https://console.groq.com/keys).

```bash
# clone and cd into the project
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd project2_research_assistant_v2

# (optional) virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# install packages
pip install -r requirements.txt
```

First run downloads the embedding model (~80 MB), after that it's cached.

## Running

**Streamlit UI (recommended):**
```bash
python -m streamlit run streamlit_app.py
```
Opens at http://localhost:8501. Paste your Groq API key in the sidebar and start researching — no `.env` file needed. Has 3 tabs: Research Chat, Generate Report, Sources Library.

**CLI:**
```bash
# for CLI you need a .env file
echo GROQ_API_KEY=gsk_your_key_here > .env
python main.py
```
Commands:
- `research machine learning in healthcare` - search + synthesize
- `detailed quantum computing` - generate detailed report
- `compare AI ethics approaches` - comparative analysis
- `citations` - show collected sources
- `quit` - exit

## Research Domains

The paper database covers: Machine Learning, AI, Climate Change, Quantum Computing, Blockchain, Healthcare, Cybersecurity, Education, Robotics, Neuroscience.

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io, click New App
3. Pick your repo, set main file to `streamlit_app.py`
4. In Advanced settings > Secrets, add: `GROQ_API_KEY = "gsk_your_key_here"`
5. Deploy

---
Built for Lab 4 - AI-Based Applications course.
