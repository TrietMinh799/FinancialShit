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

The app uses the standard OpenAI-compatible **Chat Completions** API (`/v1/chat/completions`), so it works with any provider that supports this format.

#### Configuration options

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | Base URL of any OpenAI-compatible API endpoint |
| `OPENAI_API_KEY` | No* | — | Default API key (can also be entered per-session in the UI) |
| `OPENAI_MODEL` | No | `gpt-4.1-mini` | Default model name |
| `VALUATION_RAG_USERNAME` | No | — | Username for HTTP Basic Authentication (Basic auth is enabled only if both username and password are set). |
| `VALUATION_RAG_PASSWORD` | No | — | Password for HTTP Basic Authentication (Basic auth is enabled only if both username and password are set). |

*\*Without a key, the app still works in fallback mode — answers are built from keyword-matched evidence only.*

#### How to configure

**Option A — via `.env` file** (permanent, in project root):

```ini
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
```

**Option B — via the Chat UI** (per-session, no restart needed):

1. Open the app in your browser
2. In the **sidebar**, paste your API key into the key field
3. Set the **model name** (e.g. `gpt-4o`, `google/gemini-2.0-flash-exp:free`, `llama-3.3-70b-versatile`)
4. Check **"Ghi nhớ trong trình duyệt"** to persist the key in localStorage

The UI key + model override `.env` values for that session.

---

#### Provider-specific guides

##### OpenAI

```ini
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-<your-key>
OPENAI_MODEL=gpt-4o
```

> Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (requires a paid account).

##### OpenRouter (free models available, no credit card required)

```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-<your-key>
OPENAI_MODEL=google/gemini-2.0-flash-exp:free
```

> 1. Sign up at [openrouter.ai](https://openrouter.ai)
> 2. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys)
> 3. Browse free models at [openrouter.ai/models](https://openrouter.ai/models) — recommended: `google/gemini-2.0-flash-exp:free`, `meta-llama/llama-3.2-3b-instruct:free`
> 4. Set the model name to **exactly** what appears in OpenRouter's model selector (e.g. `google/gemini-2.0-flash-exp:free`)

##### Groq (free tier available)

```ini
LLM_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_<your-key>
OPENAI_MODEL=llama-3.3-70b-versatile
```

> 1. Sign up at [groq.com](https://groq.com)
> 2. Create a key at [console.groq.com/keys](https://console.groq.com/keys)
> 3. Available models: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`

##### Together AI

```ini
LLM_BASE_URL=https://api.together.xyz/v1
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=mistralai/Mixtral-8x22B-Instruct-v0.1
```

> Get a key at [api.together.xyz](https://api.together.xyz).

##### Ollama (local, fully offline)

```ini
LLM_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama           # any non-empty value; Ollama doesn't verify keys
OPENAI_MODEL=llama3.2
```

> 1. Install Ollama from [ollama.com](https://ollama.com)
> 2. Pull a model: `ollama pull llama3.2`
> 3. Ensure Ollama is running with the API enabled (it runs by default on port 11434)
> 4. The model name must match `ollama list` output
> 5. **Important**: Ollama must allow cross-origin requests. If you see CORS errors, set `OLLAMA_ORIGINS=*` before starting:

```bash
# Windows PowerShell:
$env:OLLAMA_ORIGINS="*"; ollama serve

# macOS / Linux:
OLLAMA_ORIGINS=* ollama serve
```

##### Anthropic (via proxy)

Anthropic's API is **not** OpenAI-compatible natively. Use OpenRouter or a proxy like [LiteLLM](https://github.com/BerriAI/litellm):

```bash
pip install litellm
litellm --model claude-sonnet-4-20250514 --port 4000
```

Then configure:

```ini
LLM_BASE_URL=http://localhost:4000/v1
OPENAI_API_KEY=sk-ant-<your-key>
OPENAI_MODEL=claude-sonnet-4-20250514
```

---

#### Custom / self-hosted provider

Any service that exposes an OpenAI-compatible `/v1/chat/completions` endpoint works:

1. Set `LLM_BASE_URL` to the service root URL (up to but **not including** `/chat/completions`)
2. Set `OPENAI_MODEL` to whatever model name the provider expects
3. Set `OPENAI_API_KEY` to the required authentication token (or leave empty if not needed)

Examples:
- **vLLM**: `http://<host>:8000/v1`
- **LocalAI**: `http://localhost:8080/v1`
- **Text Generation Inference (TGI)**: `http://<host>:3000/v1`
- **llama.cpp server**: `http://localhost:8080/v1`

#### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Empty response" or `None` returned | Model name doesn't exist on the provider | Double-check the exact model identifier |
| `401 Unauthorized` | Wrong API key | Check the key and the provider's dashboard |
| `404 Not Found` | Wrong `LLM_BASE_URL` | Verify the URL (include `/v1` if required) |
| CORS error in browser console | Ollama missing `OLLAMA_ORIGINS=*` | Restart Ollama with the env var |
| Fallback mode (no LLM) | No API key configured | Enter a key in the UI sidebar

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
