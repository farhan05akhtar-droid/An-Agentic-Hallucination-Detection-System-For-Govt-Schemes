import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data & Persistence paths
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# RAG & Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 6
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Deterministic Temperature Lock (Rule from PRD: T=0.0 to 0.1)
LLM_TEMPERATURE = 0.0

# Hallucination Verification Thresholds (Rule from PRD)
# If Entailment score < 0.70 or Contradiction/Neutral > 0.30 -> Trigger loop/strip
ENTAILMENT_THRESHOLD = 0.70
CONTRADICTION_THRESHOLD = 0.30
MAX_REVISION_COUNT = 2

# LLM Provider Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Preferred model selections
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# System Execution Limits
MAX_LATENCY_BUDGET_SECONDS = 8.0
TARGET_FAITHFULNESS = 0.98
TARGET_PRECISION = 0.95
