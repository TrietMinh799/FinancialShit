# -*- coding: utf-8 -*-
"""config.py - Global configuration, paths, constants, and domain vocabularies."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & storage
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.cwd() / "data"))
ROOT = DATA_DIR
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = ROOT / "rag.sqlite3"
CHROMA_DIR = ROOT / "chroma"

# In-memory run cache (populated by analysis.analyze_report)
RUNS: OrderedDict = OrderedDict()
MAX_RUNS: int = 50

# ---------------------------------------------------------------------------
# Chunking & retrieval tunables
# ---------------------------------------------------------------------------
# Default chunk sizing for reference books.
CHUNK_CHARS: int = 1600
CHUNK_OVERLAP: int = 250
CHUNK_MIN_STEP: int = 400          # minimum slide between chunk windows
CHUNK_MIN_LEN: int = 120           # discard chunks shorter than this
# Denser annual-report chunks improve retrieval precision.
REPORT_CHUNK_CHARS: int = 800
REPORT_CHUNK_OVERLAP: int = 180
# Snippet extraction.
SNIPPET_MAX_CHARS: int = 500
# Query tokenisation cap.
QUERY_TERM_LIMIT: int = 20
# Number of recent documents surfaced by Store.stats().
RECENT_DOCS_LIMIT: int = 20
# Cross-encoder rerank cutoff (how many final evidence chunks passed to LLM).
RERANK_TOP_K: int = 15
# LLM HTTP timeouts (seconds).
LLM_TIMEOUT: int = 120
LLM_STREAM_TIMEOUT: int = 300  # longer for streaming: free models can take minutes to first token
LLM_REPORT_TIMEOUT: int = 300  # longer timeout for report analysis (larger prompts)
LLM_DECOMPOSE_TIMEOUT: int = 25  # reduced: free/slow models fall back to raw question faster

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# Base URL for any OpenAI-compatible Chat Completions provider.
# Examples:
#   OpenAI      -> https://api.openai.com/v1
#   OpenRouter  -> https://openrouter.ai/api/v1
#   Groq        -> https://api.groq.com/openai/v1
#   Together    -> https://api.together.xyz/v1
#   Ollama      -> http://localhost:11434/v1
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    # English stopwords
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "business",
    "can",
    "company",
    "from",
    "growth",
    "have",
    "into",
    "its",
    "more",
    "not",
    "that",
    "the",
    "their",
    "this",
    "with",
    "what",
    "when",
    "where",
    "whether",
    "will",
    # Vietnamese stopwords (with and without diacritics)
    "của",
    "và",
    "có",
    "trong",
    "cho",
    "với",
    "các",
    "được",
    "một",
    "này",
    "đã",
    "sẽ",
    "đang",
    "là",
    "không",
    "tại",
    "theo",
    "khi",
    "về",
    "từ",
    "hoặc",
    "như",
    "trên",
    "giữa",
    "sau",
    "trước",
    "nên",
    "rằng",
    "mà",
    "vì",
    "nếu",
    "thì",
    "để",
    "qua",
    "lại",
    "ra",
    "vào",
    "những",
    "nhiều",
    "mọi",
    "mỗi",
    "đó",
    "đây",
    "ấy",
    "nay",
    "vậy",
    "bị",
    "phải",
    "làm",
    "hơn",
    "rất",
    "cùng",
    "đến",
    "nơi",
    "việc",
    "người",
    "cả",
    "đều",
    "tôi",
    "bạn",
    "chúng",
    # Diacritic-free variants (for users typing without diacritics)
    "cua",
    "va",
    "co",
    "duoc",
    "mot",
    "da",
    "se",
    "dang",
    "la",
    "khong",
    "ve",
    "tu",
    "hoac",
    "nhu",
    "tren",
    "giua",
    "truoc",
    "nen",
    "rang",
    "ma",
    "vi",
    "neu",
    "thi",
    "de",
    "lai",
    "vao",
    "nhung",
    "nhieu",
    "moi",
    "do",
    "day",
    "ay",
    "vay",
    "bi",
    "phai",
    "lam",
    "hon",
    "rat",
    "cung",
    "den",
    "noi",
    "viec",
    "nguoi",
    "ca",
    "deu",
    "toi",
    "ban",
    "chung",
    # Additional missing diacritic-free variants
    "cac",
    "tai",
    "voi",
}

# ---------------------------------------------------------------------------
# Domain vocabularies  (term → human-readable label)
# ---------------------------------------------------------------------------
MOAT_TERMS: dict[str, str] = {
    "scale": "Scale advantage",
    "market share": "Market share",
    "brand": "Brand strength",
    "switching cost": "Switching costs",
    "network effect": "Network effects",
    "cost advantage": "Cost advantage",
    "vertical integration": "Vertical integration",
    "distribution": "Distribution reach",
    "patent": "Protected know-how",
    "technology": "Technology capability",
    "capacity": "Capacity advantage",
    "customer relationship": "Customer relationships",
    "barrier to entry": "Entry barriers",
}

GROWTH_TERMS: dict[str, str] = {
    "capacity expansion": "Capacity expansion",
    "new product": "New products",
    "export": "Export growth",
    "market expansion": "Market expansion",
    "demand": "Demand growth",
    "investment": "Investment program",
    "innovation": "Innovation",
    "digital": "Digital capability",
    "research and development": "R&D",
    "infrastructure": "Infrastructure demand",
    "penetration": "Penetration upside",
}

EXECUTION_TERMS: dict[str, str] = {
    "completed": "Project completion",
    "delivered": "Delivery record",
    "strategy": "Clear strategy",
    "governance": "Governance",
    "risk management": "Risk management",
    "operational efficiency": "Operational efficiency",
    "productivity": "Productivity",
}

RISK_TERMS: dict[str, str] = {
    "competition": "Competitive pressure",
    "cyclical": "Cyclicality",
    "commodity": "Commodity exposure",
    "foreign exchange": "FX exposure",
    "regulatory": "Regulatory pressure",
    "debt": "Debt load",
    "overcapacity": "Overcapacity",
    "inflation": "Inflation",
    "interest rate": "Interest-rate risk",
    "raw material": "Input-cost exposure",
    "tariff": "Trade barrier",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create ROOT and UPLOAD_DIR on first use."""
    ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
