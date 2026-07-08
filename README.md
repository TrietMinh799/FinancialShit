# Valuation RAG — Chatbot

Vietnamese-language financial analysis chatbot powered by Retrieval-Augmented Generation (RAG). Upload textbooks and annual reports, then ask natural-language questions about competitive moat, growth, risk, and valuation.

## Features

- **Chatbot UI** — Chat-first interface with conversation history, typing indicators, starter prompts, and inline analysis cards
- **RAG Q&A** — Ask questions in natural language; the system retrieves relevant passages from your uploaded library via BM25/FTS5 and synthesises answers via LLM
- **Knowledge Base Builder** — Upload PDF, EPUB, DOCX, TXT, MD files to build a searchable library
- **Annual Report Analyzer** — Score companies on 5 dimensions (moat, growth, execution, resilience, risk pressure), generate SWOT analysis, prioritised growth actions, and a structured KB report
- **Multi-LLM Support** — Works with any OpenAI-compatible provider: OpenAI, OpenRouter, Groq, Together, Ollama, and more
- **Fallback Mode** — No API key? Still get keyword-evidence answers from your documents
- **Flask Backend** — Lightweight, dependency-minimal, easy to extend

## Architecture

```
├── server.py              # Flask app with all API routes
├── core/
│   ├── config.py          # Paths, model names, LLM_BASE_URL, domain vocabularies
│   ├── store.py           # SQLite document store + FTS5 search
│   ├── llm.py             # LLM integration (OpenAI-compatible /chat/completions)
│   ├── analysis.py        # Report scoring engine + SWOT
│   ├── extractors.py      # PDF/EPUB/DOCX/TXT parser
│   ├── text_utils.py      # Chunking, cleaning, snippet extraction
│   └── rag_platform.py    # Backward-compat re-export shim
├── web/
│   ├── index.html         # Chatbot UI (single-file, inline JS)
│   └── static/
│       ├── styles.css     # Design system & layout
│       └── chat.js        # Additional chat interactions
└── .env                   # API keys & configuration
```

## Installation

Requires Python 3.8+.

```bash
python -m venv env

# Windows:
.\env\Scripts\activate
# macOS/Linux:
source env/bin/activate

pip install flask python-dotenv pypdf
```

## Usage

```bash
python server.py --host 127.0.0.1 --port 8767
# or with auto-reload for development:
python server.py --debug
```

Open `http://127.0.0.1:8767` in your browser.

### LLM Provider Setup

Set environment variables in `.env` or export them in your shell:

```ini
# Provider base URL (defaults to OpenAI)
LLM_BASE_URL=https://api.openai.com/v1

# Model name
OPENAI_MODEL=gpt-4o

# API key (can also be entered in the UI)
OPENAI_API_KEY=sk-...
```

| Provider | `LLM_BASE_URL` | Model example |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| OpenRouter | `https://openrouter.ai/api/v1` | `google/gemini-2.0-flash-exp:free` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together | `https://api.together.xyz/v1` | `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.2` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the chat UI |
| `GET` | `/api/library` | Library stats & recent documents |
| `POST` | `/api/ask` | RAG question-answering |
| `POST` | `/api/upload-book` | Upload & index a document |
| `POST` | `/api/analyze-report` | Score a company annual report |
| `POST` | `/api/test-key` | Validate API key connectivity |

## Best Practices

- Never commit your `.env` file or expose API keys publicly
- The project uses strict Python type hints — run `mypy` before submitting changes
- On first use, the app creates temp directories (`%TEMP%/valuation_rag_platform_epub_clean/`)
