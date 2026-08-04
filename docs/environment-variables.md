# Environment variables

All configuration lives in `.env` (copy from [`.env.example`](../.env.example)). Nothing here is required beyond `OPENAI_API_KEY` (or `OPENAI_BASE_URL` for a compatible provider) — everything else has a sane default.

| Variable | Default | Required? | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | — | Yes, unless using a custom `OPENAI_BASE_URL` that doesn't need one | API key for the LLM (chat completions + query rewriting) |
| `OPENAI_MODEL` | `gpt-4o-mini` | No | Chat model used for answers and query rewriting |
| `OPENAI_BASE_URL` | OpenAI's default | No | Point at an OpenAI-compatible provider (Ollama, Groq, a corporate gateway, ...) |
| `OPENAI_EXTRA_HEADERS` | — | No | JSON object of extra HTTP headers to send with every LLM request (corporate gateways) |
| `RAG_STRATEGY` | `hybrid_rerank_rewrite` | No | Retrieval strategy used by the CLI (`assistant.py`) — see [Retrieval strategies](../README.md#retrieval-strategies) |
| `MOVEBANK_USERNAME` / `MOVEBANK_PASSWORD` | — | No | Movebank credentials for `make download`; omit to use the bundled sample dataset instead |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `5432` | No | Where to reach Postgres (set automatically inside Docker Compose) |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | `wildlife_rag` / `wildlife` / `wildlife` | No | Postgres credentials for conversation/feedback logging |
| `INDEX_PATH` / `EMBEDDINGS_PATH` / `DOCUMENTS_PATH` | `data/index/wildlife_index.pkl` / `wildlife_embeddings.npy` / `wildlife_documents.json` | No | Where `ingest.py` writes (and the app reads) the search index artifacts |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | No | Sentence-transformers model used for both indexing and query embeddings |

See [`../README.md`](../README.md) for the full quick-start guide.
