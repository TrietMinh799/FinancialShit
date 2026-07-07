# Valuation RAG Project

Valuation RAG is a lightweight, zero-dependency-framework Retrieval-Augmented Generation (RAG) web application designed for processing, analyzing, and querying books and corporate annual reports. 

## Features

- **Built-in HTTP Server:** Zero heavy framework dependencies. The server routes are built entirely using Python's standard `http.server`.
- **Intelligent RAG System:** Upload corporate reports or books and instantly ask contextual questions powered by Large Language Models (LLMs).
- **SQLite Document Store:** Documents and chunks are stored securely using SQLite with Fast Text Search (FTS5) capabilities for rapid retrieval.
- **Report Analysis:** Specialized features for generating comprehensive knowledge-base reports based on corporate annual disclosures.

## Architecture & Structure

- `core/`: Contains the core application business logic.
  - `store.py`: Document storage handling, chunking, multipart parsing, and full-text search.
  - `llm.py` / `rag_platform.py`: AI model integrations, prompting, and RAG execution.
  - `analysis.py`: Report evaluation logic.
  - `extractors.py` / `text_utils.py`: Text cleaning and manipulation.
- `web/`: The user interface frontend. Contains `index.html` and static assets.
- `data/`: Storage directory for local SQLite databases and uploaded documents.
- `server.py`: The single-threaded-safe HTTP handler and entry point for the server.

## Installation

Ensure you have Python 3.8+ installed on your machine.

1. **Clone the project repository** (or navigate to the folder).
2. **Set up a Virtual Environment**:
   ```bash
   python -m venv env
   # Activate on Windows:
   .\env\Scripts\activate
   # Activate on macOS/Linux:
   source env/bin/activate
   ```
3. **Install Dependencies**:
   Install required Python packages via pip:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` file should contain the necessary packages like langchain, sqlite, etc.)*

## Usage

1. Start the HTTP server from the project root:
   ```bash
   python server.py --host 127.0.0.1 --port 8767
   ```
2. Open your browser and navigate to `http://127.0.0.1:8767` to access the RAG platform.
3. Configure your LLM API Key inside the platform.
4. Upload documents (PDF, DOCX, TXT, EPUB) and start asking questions!

## Best Practices
- **Security**: Be careful not to expose your `.env` or data files to the public. 
- **Type Checking**: The project adheres strictly to Python type hints and `pep8-naming` conventions.
