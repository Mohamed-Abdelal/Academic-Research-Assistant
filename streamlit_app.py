# Streamlit UI for research assistant

import os
import sys
import streamlit as st
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_groq import ChatGroq

import config
from sources.web_search import WebResearchTool
from sources.paper_database import PaperDatabaseTool, DEVELOPMENTS
from synthesis.summarizer import InformationSynthesizer
from synthesis.comparator import InformationComparator
from reporting.report_generator import ReportGenerator
from citation.citation_manager import CitationManager
from citation.citation_styles import format_apa, format_mla, format_chicago, STYLES
from filtering.domain_filters import DomainFilter

# ── Streamlit Cloud secrets support ──────────────────────────────────────────
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1a237e 0%, #4a148c 60%, #1a237e 100%);
    padding: 22px 28px; border-radius: 16px; color: white; margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
.hero p  { margin: 4px 0 0; opacity: 0.75; font-size: 0.9rem; }

.source-card {
    border-left: 4px solid #7c3aed; background: rgba(124,58,237,0.08);
    padding: 12px 16px; border-radius: 8px; margin: 6px 0;
}
.source-high   { border-left-color: #10b981; background: rgba(16,185,129,0.08); }
.source-medium { border-left-color: #f59e0b; background: rgba(245,158,11,0.08); }
.source-low    { border-left-color: #ef4444; background: rgba(239,68,68,0.08); }

.stat-box {
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.3);
    border-radius: 10px; padding: 10px; text-align: center;
}
.stat-num { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
.stat-lbl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #2d1b69 100%);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3);
    color: #c4b5fd !important; border-radius: 8px; width: 100%;
    font-size: 0.82rem; transition: all 0.2s; margin-bottom: 2px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(124,58,237,0.25); border-color: #7c3aed;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(api_key=api_key, model=config.GROQ_MODEL, temperature=config.LLM_TEMPERATURE)


def search_papers(query: str, use_live: bool = True) -> List[Dict]:
    web_tool = WebResearchTool()
    paper_db = PaperDatabaseTool()

    live_results = []
    if use_live:
        for r in web_tool.search(query, num_results=3):
            live_results.append({
                "title": r.get("title", "Web Result"),
                "authors": "Web Source",
                "year": datetime.now().year,
                "journal": r.get("href", "").split("/")[2] if r.get("href") else "Web",
                "abstract": r.get("body", "")[:400],
                "citations": 0,
                "doi": r.get("href", ""),
                "relevance": 0.72,
                "source_type": "web",
            })

    mock_results = paper_db.search(query, num_results=5)

    all_results = mock_results + live_results
    seen, unique = set(), []
    for p in all_results:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)
    unique.sort(key=lambda x: x.get("relevance", 0.5), reverse=True)
    return unique[:7]


def get_research_response(api_key, query, history, report_format, depth):
    llm = get_llm(api_key)
    synthesizer = InformationSynthesizer(llm=llm)
    paper_db = PaperDatabaseTool()

    papers = search_papers(query)
    devs = paper_db.get_developments(query)

    context = "=== Academic Papers ===\n"
    for p in papers:
        context += f"- {p['title']} — {p['authors']} ({p['year']}) in {p['journal']}\n"
        context += f"  Abstract: {p['abstract'][:250]}...\n"
        context += f"  Citations: {p.get('citations', 0):,} | DOI: {p.get('doi', '')}\n\n"
    context += f"\n=== Recent Developments ===\n{devs}\n"

    from langchain_classic.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        template=(
            "You are an Academic Research Assistant.\n"
            "Report Format: {report_format} | Depth: {depth}\n"
            "Date: {date}\n\n"
            "Instructions:\n"
            "1. Synthesize the provided research data\n"
            "2. Cite sources using [Author, Year] format\n"
            "3. Present conflicting findings objectively\n"
            "4. Highlight practical implications\n"
            "5. Suggest research gaps\n"
            "6. Use markdown headings and bullet points\n\n"
            "RESEARCH DATA:\n{context}\n\n"
            "USER QUERY: {query}"
        ),
        input_variables=["report_format", "depth", "date", "context", "query"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(
        report_format=report_format,
        depth=depth,
        date=datetime.now().strftime("%B %d, %Y"),
        context=context,
        query=query,
    )
    return response, papers


def generate_full_report(api_key, topic, papers, report_format):
    llm = get_llm(api_key)
    generator = ReportGenerator(llm=llm)

    sources_text = ""
    for i, p in enumerate(papers[:6], 1):
        sources_text += f"\n[{i}] {p.get('title')} — {p.get('authors')} ({p.get('year')})\n"
        sources_text += f"Abstract: {p.get('abstract', '')[:350]}...\n"

    format_map = {
        "Structured Report": "detailed",
        "Literature Review": "literature_review",
        "Executive Summary": "executive_summary",
        "Annotated Bibliography": "summary",
        "Comparative Analysis": "comparative",
    }
    report_type = format_map.get(report_format, "summary")
    return generator.generate_report(content=sources_text, topic=topic, report_type=report_type)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages": [], "chat_history": [], "sources": [],
        "sessions": 0, "total_papers": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()

    st.markdown("""
    <div class="hero">
        <h1>Academic Research Assistant</h1>
        <p>AI-powered research synthesis · LangChain chains · Citation management (APA/MLA/Chicago) · 10 domains</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Configuration")
        default_key = os.environ.get("GROQ_API_KEY", "")
        api_key = st.text_input(
            "GROQ API Key",
            value=default_key,
            type="password",
            placeholder="gsk_...",
            help="Get a free key at https://console.groq.com/keys",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            config.GROQ_API_KEY = api_key

        st.divider()
        st.markdown("### Report Settings")
        report_format = st.selectbox(
            "Report Format",
            ["Structured Report", "Literature Review", "Executive Summary",
             "Annotated Bibliography", "Comparative Analysis"],
        )
        depth = st.selectbox("Research Depth", ["Comprehensive", "Overview", "Deep Dive"])
        use_live_search = st.toggle("Live Web Search", value=True)

        st.divider()
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.sessions}</div><div class="stat-lbl">Searches</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.total_papers}</div><div class="stat-lbl">Papers</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Quick Topics")
        topics = [
            "Machine learning in healthcare",
            "Climate change mitigation strategies",
            "Quantum computing error correction",
            "AI ethics and bias in LLMs",
            "Blockchain in supply chain",
            "Cybersecurity zero-trust architecture",
            "Robotics and autonomous systems",
            "Neuroscience and brain-computer interfaces",
            "AI in education and personalised learning",
        ]
        for t in topics:
            if st.button(t, key=f"topic_{t[:15]}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": f"Research topic: {t}"})
                st.rerun()

        st.divider()
        if st.button("Clear Session", use_container_width=True):
            for k in ["messages", "chat_history", "sources", "sessions"]:
                st.session_state[k] = [] if k != "sessions" else 0
            st.rerun()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Research Chat", "Generate Report", "Sources Library"])

    with tab1:
        st.subheader("Ask Your Research Question")

        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center;padding:40px;color:#64748b;'>
                <h3>Start Your Research</h3>
                <p>Ask any academic question. I'll synthesise findings from papers and web sources.</p>
                <p style='font-size:0.8rem'>Powered by Groq · LLaMA 3.3 70B · LangChain Chains · DuckDuckGo · 10 Domains</p>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask a research question…")

        if query:
            if not api_key:
                st.error("Please enter your GROQ API Key in the sidebar.")
            else:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching databases and synthesizing..."):
                        try:
                            response, papers = get_research_response(
                                api_key, query, st.session_state.chat_history, report_format, depth
                            )
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.session_state.chat_history.append({"role": "user", "content": query})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            if len(st.session_state.chat_history) > 12:
                                st.session_state.chat_history = st.session_state.chat_history[-12:]
                            st.session_state.sessions += 1

                            if papers:
                                existing = {p["title"] for p in st.session_state.sources}
                                new_papers = [p for p in papers if p["title"] not in existing]
                                st.session_state.sources.extend(new_papers)
                                st.session_state.total_papers += len(new_papers)

                                with st.expander(f"{len(papers)} papers found"):
                                    for p in papers[:4]:
                                        rel = p.get("relevance", 0.5)
                                        cls = "source-high" if rel >= 0.85 else "source-medium" if rel >= 0.7 else "source-low"
                                        cred = CitationManager()
                                        cred.add_source(p)
                                        credibility = cred.assess_credibility(p)
                                        st.markdown(
                                            f'<div class="source-card {cls}"><b>{p["title"]}</b> — {p["authors"]} ({p["year"]})<br>'
                                            f'<small>{p["journal"]} | {p.get("citations", 0):,} citations | '
                                            f'Relevance: {rel*100:.0f}% | Credibility: {credibility}</small></div>',
                                            unsafe_allow_html=True,
                                        )
                        except Exception as e:
                            st.error(f"Error: {e}")

    with tab2:
        st.subheader(f"Generate {report_format}")
        col1, col2 = st.columns([3, 1])
        with col1:
            report_topic = st.text_input("Research Topic", placeholder="e.g. Applications of AI in medical diagnosis")
        with col2:
            min_papers = st.slider("Min. papers", 2, 6, 3)

        if st.button("Generate Report", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your GROQ API Key.")
            elif not report_topic.strip():
                st.warning("Please enter a research topic.")
            else:
                with st.spinner(f"Writing {report_format}..."):
                    try:
                        papers = search_papers(report_topic, use_live=use_live_search)
                        if len(papers) < min_papers:
                            papers += search_papers("artificial intelligence", use_live=False)
                        papers = list({p["title"]: p for p in papers}.values())[:6]
                        report = generate_full_report(api_key, report_topic, papers, report_format)

                        cm = CitationManager()
                        for p in papers:
                            cm.add_source(p)
                        bib = cm.generate_bibliography(style="apa")
                        full = report + "\n\n" + bib

                        st.markdown(full)
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("Download (Markdown)", data=full,
                                file_name=f"report_{report_topic[:30].replace(' ', '_')}.md", mime="text/markdown")
                        with col_b:
                            st.download_button("Download (Text)",
                                data=full.replace("**", "").replace("*", "").replace("#", ""),
                                file_name=f"report_{report_topic[:30].replace(' ', '_')}.txt", mime="text/plain")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")

    with tab3:
        st.subheader("Research Sources Library")

        if not st.session_state.sources:
            st.info("Sources collected during research will appear here.")
        else:
            citation_style = st.radio("Citation Style", ["APA", "MLA", "Chicago"], horizontal=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                min_year = st.number_input("Min Year", 1990, 2025, 2000)
            with col2:
                min_cit = st.number_input("Min Citations", 0, 100_000, 0, step=1000)
            with col3:
                sort_by = st.selectbox("Sort By", ["Relevance", "Citations", "Year"])

            filtered = [
                p for p in st.session_state.sources
                if p.get("year", 0) >= min_year and p.get("citations", 0) >= min_cit
            ]
            if sort_by == "Citations":
                filtered.sort(key=lambda x: x.get("citations", 0), reverse=True)
            elif sort_by == "Year":
                filtered.sort(key=lambda x: x.get("year", 0), reverse=True)
            else:
                filtered.sort(key=lambda x: x.get("relevance", 0), reverse=True)

            st.markdown(f"**Showing {len(filtered)} / {len(st.session_state.sources)} papers**")

            if filtered:
                fmt_fn = {"APA": format_apa, "MLA": format_mla, "Chicago": format_chicago}[citation_style]
                bib_text = f"Bibliography ({citation_style})\n{'=' * 50}\n\n"
                bib_text += "\n\n".join(fmt_fn(p) for p in filtered)
                st.download_button(
                    f"Export {citation_style} Bibliography",
                    data=bib_text,
                    file_name=f"bibliography_{citation_style.lower()}.txt",
                    mime="text/plain",
                )

            for paper in filtered:
                rel = paper.get("relevance", 0.5)
                cls = "source-high" if rel >= 0.85 else "source-medium" if rel >= 0.7 else "source-low"
                cm = CitationManager()
                cm.add_source(paper)
                credibility = cm.assess_credibility(paper)
                with st.expander(f"{paper['title']} ({paper['year']}) — Credibility: {credibility}"):
                    st.markdown(
                        f'<div class="source-card {cls}">'
                        f'<b>Authors:</b> {paper["authors"]}<br>'
                        f'<b>Journal:</b> {paper["journal"]} · <b>Year:</b> {paper["year"]} · '
                        f'<b>Citations:</b> {paper.get("citations", 0):,} · '
                        f'<b>Relevance:</b> {rel*100:.0f}% · <b>Credibility:</b> {credibility}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    fmt_fn = {"APA": format_apa, "MLA": format_mla, "Chicago": format_chicago}[citation_style]
                    st.code(fmt_fn(paper), language=None)

            st.divider()
            if st.button("Clear Sources Library"):
                st.session_state.sources = []
                st.session_state.total_papers = 0
                st.rerun()


if __name__ == "__main__":
    main()
