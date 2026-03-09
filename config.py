import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LLM_TEMPERATURE = 0.2
MAX_TOKENS = 2048

MAX_SEARCH_RESULTS = 5
RELEVANCE_THRESHOLD = 0.5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

DEFAULT_REPORT_TYPE = "summary"
DEFAULT_CITATION_STYLE = "apa"
DEFAULT_RESEARCH_DEPTH = "medium"
