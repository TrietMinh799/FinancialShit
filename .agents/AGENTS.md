# Valuation RAG Project Rules

These rules apply to all tasks performed in this workspace. Follow them carefully to ensure consistency and quality.

## Architecture and Structure
1. **Backend**: Built using Python's built-in `http.server` module. Do not introduce large web frameworks (like FastAPI, Flask, or Django) without explicit user permission.
2. **Frontend**: The frontend is primarily a single-page HTML application located at `web/index.html`. Static assets (CSS, JS, images) should be placed in `web/static/`.
3. **Core Logic**: Keep all Retrieval-Augmented Generation (RAG), Large Language Model (LLM) interactions, and business logic inside the `core/` directory. Leave `server.py` strictly for HTTP request/response routing.

## Python Coding Standards
1. **Type Hints**: Always use Python type hints (`from __future__ import annotations` and standard type hinting) for function arguments and return types.
2. **Linting and Naming**: The project adheres to `pep8-naming` standards (e.g., `# noqa: N802` seen in code). Stick strictly to PEP 8 naming conventions.
3. **Imports**: Keep standard library imports first, followed by third-party libraries, and finally internal project imports (e.g., `from core...`).

## Dependencies
1. Minimize external dependencies. When parsing documents or interacting with LLMs, use the existing modules in `core/` before adding new libraries.
2. Any new package dependencies must be justified and explicitly approved by the user.

## Data and File Handling
1. **Document Storage**: Books, reports, and extracted texts are handled by the `Store` class (`core/store.py`). Always interact with the store via this class rather than direct file I/O where possible.
2. **Paths**: Use `pathlib.Path` for all file system path operations. Avoid using `os.path`.
