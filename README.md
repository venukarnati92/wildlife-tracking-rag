# Wildlife Tracking RAG

**LLM Zoomcamp final project** — a retrieval-augmented generation (RAG) assistant that answers questions about wildlife tracking studies (species, locations, sensors, investigators, time periods, citations) grounded in **[Movebank](https://www.movebank.org)** study metadata.

Ask things like:

- *Which studies track turkey vultures?*
- *Are there any studies on African elephant movement corridors?*
- *What sensors are used to track Galapagos albatrosses?*
- *Who is the principal investigator for the Serengeti lion project?*
- *Which studies span more than 5 years of data?*

The assistant retrieves relevant Movebank studies from an indexed knowledge base and answers with study names + study IDs. It refuses to answer (`"I don't know..."`) when the corpus does not cover the question.

---

## Contents

- [Problem statement](#problem-statement)
- [Architecture](#architecture)
- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [Screenshots](#screenshots)
- [Local setup (without Docker)](#local-setup-without-docker)
- [Ingestion pipeline (dlt)](#ingestion-pipeline-dlt)
- [Retrieval strategies](#retrieval-strategies)
- [Evaluation](#evaluation)
- [Monitoring dashboard](#monitoring-dashboard)
- [Evaluation criteria checklist](#evaluation-criteria-checklist)
- [Repo layout](#repo-layout)

---

## Problem statement

Movebank is the largest open repository of animal tracking data (GPS, accelerometer, ARGOS, geolocator, ...), hosted by the Max Planck Institute of Animal Behavior. Researchers upload studies documenting who tracked what species, where, when, and with which sensors — plus citations and licensing.

Discovering the *right* study for a research question is hard: the metadata is spread across thousands of studies. This project builds a RAG assistant over the study metadata so a biologist, journalist, or student can ask natural-language questions and get grounded answers with study IDs and citations.

The knowledge base is study metadata, **not** raw GPS fixes (which are billions of rows and rarely helpful as unstructured context). Each Movebank study is represented as one searchable document containing name, taxa, PI, contact, location, time period, animal/tag counts, sensors, objective, citation, license, and acknowledgements.

---

## Architecture

```
Movebank CSV ── scripts/download_movebank.py ─┐
                                              │
                                     data/raw/*.csv
                                              │
                              pipelines/movebank_pipeline.py  (dlt filesystem + read_csv)
                                              │
                                    DuckDB: movebank.studies
                                              │
                                          ingest.py
                                              │
              ┌───────────────────────────────┼─────────────────────────────┐
              │                               │                             │
     minsearch text index          sentence-transformers          documents.json
     (BM25-ish)                    embeddings (all-MiniLM-L6-v2)
              │                               │
              └───────────┐         ┌─────────┘
                          │         │
                       search.py: TextSearcher, VectorSearcher, HybridSearcher (RRF),
                                  RerankedSearcher (cross-encoder), RewritingSearcher (LLM)
                                              │
                                        assistant.py  ──►  RAGWithMetrics (metrics.py)
                                              │                   │
                                              │                   ▼
                                              │            Postgres (conversations,
                                              │             feedback + judge verdict)
                                              │                   │
                                              ▼                   ▼
                                          Streamlit             Streamlit
                                          app.py (chat)         dashboard.py (7 charts)
```

Every module mirrors the LLM Zoomcamp course flow (RAGBase → RAGWithMetrics → Streamlit + Postgres feedback → dashboard).

---

## Quick start (Docker Compose)

Everything (Postgres, Streamlit app, monitoring dashboard) comes up with one command.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY (or point OPENAI_BASE_URL at Ollama / another gateway)

# Bootstrap the knowledge base BEFORE bringing compose up:
uv sync
uv run python -c "import shutil; shutil.copy('data/movebank_studies.sample.json','data/movebank_studies.json')"
uv run python ingest.py

docker compose up --build
```

Then open:

- App:       http://localhost:8501
- Dashboard: http://localhost:8502

To seed the dashboard with sample conversations (~15 real RAG calls across all strategies):

```bash
docker compose run --rm app uv run python generate_data.py
```

Tear down:

```bash
docker compose down -v
```

---

## Screenshots

**Chat UI** (`app.py`, http://localhost:8501) — ask a question, pick a retrieval strategy, and get a grounded answer with citations, latency/token/cost metrics, and the retrieved study context on a successful RAG run:

![Wildlife Tracking RAG chat UI](assets/screenshots/app-chat.png)

**Monitoring dashboard** (`dashboard.py`, http://localhost:8502) — conversations, cost, and response time over time, plus judge/feedback breakdowns:

![Wildlife Tracking RAG monitoring dashboard](assets/screenshots/dashboard.png)

---

## Local setup (without Docker)

```bash
# 1. Python + deps (uv is required)
uv sync

# 2. Environment
cp .env.example .env
# set OPENAI_API_KEY (or OPENAI_BASE_URL for a compatible provider)

# 3. Data + index (bundled sample)
make bootstrap        # copies sample -> data/movebank_studies.json + builds indexes

# 4. Postgres (via docker) + tables
docker compose up -d postgres
uv run python db_init.py

# 5. Run
uv run streamlit run app.py --server.port 8501       # chat UI
uv run streamlit run dashboard.py --server.port 8502 # monitoring
```

CLI quick test (no UI):

```bash
uv run python assistant.py "Which studies track Yellowstone wolves?"
```

---

## Ingestion pipeline (dlt)

Two-step, automated:

1. `scripts/download_movebank.py` — fetches public Movebank study metadata via the REST endpoint `https://www.movebank.org/movebank/service/direct-read?entity_type=study`. Writes CSV to `data/raw/` and JSON to `data/movebank_studies.json`. Falls back to the bundled `data/movebank_studies.sample.json` (12 well-known studies) if the endpoint requires credentials that aren't set.

   Optionally set `MOVEBANK_USERNAME` / `MOVEBANK_PASSWORD` in `.env` for authenticated access.

2. `pipelines/movebank_pipeline.py` — a **dlt** pipeline that reads any CSV under `data/raw/` (`dlt.sources.filesystem` + `read_csv`) and loads it into `data/movebank_pipeline.duckdb` under the `movebank.studies` table.

3. `ingest.py` reads the DuckDB output (or the JSON fallback), turns each study row into a canonical document, then builds:
   - `data/index/wildlife_index.pkl` — minsearch text index
   - `data/index/wildlife_embeddings.npy` — dense vectors from `sentence-transformers/all-MiniLM-L6-v2`
   - `data/index/wildlife_documents.json` — the source docs

To re-ingest end to end:

```bash
make download    # (or copy your own CSVs into data/raw/)
make pipeline    # dlt -> DuckDB
make index       # build search + embeddings artifacts
```

---

## Retrieval strategies

Implemented in [`search.py`](search.py) and selectable at runtime via the app sidebar or `RAG_STRATEGY` env var:

| Strategy                | What it does                                                                 |
| ---                     | ---                                                                          |
| `text`                  | minsearch text search with field boosts (name, taxa, objective)              |
| `vector`                | dense retrieval over sentence-transformers embeddings (cosine similarity)    |
| `hybrid`                | Reciprocal Rank Fusion (RRF) of `text` and `vector`                          |
| `hybrid_rerank`         | RRF, then cross-encoder rerank with `ms-marco-MiniLM-L-6-v2`                 |
| `hybrid_rerank_rewrite` | Add an LLM query-rewrite step (e.g. common name → scientific name)           |

The RAG production path defaults to `hybrid_rerank_rewrite`.

---

## Evaluation

### Retrieval evaluation (all strategies)

Ground truth is [`data/ground_truth.json`](data/ground_truth.json) (30 human-written questions, each with the gold Movebank `study_id`). Run:

```bash
uv run python scripts/eval_retrieval.py
```

Results committed at [`data/retrieval_eval.csv`](data/retrieval_eval.csv):

| Strategy                | Hit@1 | Hit@3 | Hit@5 | MRR   |
| ---                     | ---   | ---   | ---   | ---   |
| text                    | 0.967 | 0.967 | 1.000 | 0.975 |
| vector                  | 0.900 | 0.967 | 1.000 | 0.942 |
| hybrid                  | 0.967 | 1.000 | 1.000 | 0.978 |
| **hybrid_rerank**       | **1.000** | **1.000** | **1.000** | **1.000** |

`hybrid_rerank_rewrite` also runs when `OPENAI_API_KEY` is set; on this corpus it matches `hybrid_rerank` at MRR 1.0 while adding query normalization for common → scientific names, which is where it shines on paraphrased user queries.

### LLM evaluation (multiple prompts)

Notebook: [`notebooks/03-llm-eval.ipynb`](notebooks/03-llm-eval.ipynb).

Compares two prompt variants on the same retrieval backend:

- **Prompt A** — concise, cite study id, refuse if not in context
- **Prompt B** — "helpful research librarian" style, cite study id in parentheses, refuse if not in context

Each answer is graded by [`judge.py`](judge.py) (LLM-as-judge) as `RELEVANT` / `PARTLY_RELEVANT` / `NON_RELEVANT`. The winning prompt is wired into `rag_helper.INSTRUCTIONS` for the app.

### Ground-truth generation

[`notebooks/01-ground-truth.ipynb`](notebooks/01-ground-truth.ipynb) uses the LLM to expand every study into 5 synthetic questions. The committed [`data/ground_truth.json`](data/ground_truth.json) is a hand-curated version of the same idea, so the retrieval eval is reproducible **without an OpenAI key**.

---

## Monitoring dashboard

Every RAG call is written to Postgres by [`db_save.py`](db_save.py). The LLM judge verdict and user thumbs (from the app) go to a linked `feedback` table via [`db_feedback.py`](db_feedback.py).

`dashboard.py` (http://localhost:8502) renders **7 charts** on top of that:

1. Conversations over time (per hour)
2. Cost per call over time
3. Response time over time
4. Judge relevance distribution (bar)
5. User feedback thumbs up/down (metrics)
6. Retrieval strategy usage & average latency (table + bars)
7. Top asked questions (table)

Plus 4 headline KPIs at the top (total conversations, avg latency, total cost, avg tokens/call).

To seed it quickly with real data:

```bash
uv run python generate_data.py
```

---

## Evaluation criteria checklist

| Criterion              | Points | Where it lives                                                                                       |
| ---                    | ---    | ---                                                                                                  |
| Problem description    | 2      | This README, [Problem statement](#problem-statement)                                                 |
| Retrieval flow         | 2      | KB in [`ingest.py`](ingest.py) + LLM in [`rag_helper.py`](rag_helper.py) / [`metrics.py`](metrics.py) |
| Retrieval evaluation   | 2      | [`scripts/eval_retrieval.py`](scripts/eval_retrieval.py), 4 strategies compared, best selected       |
| LLM evaluation         | 2      | [`notebooks/03-llm-eval.ipynb`](notebooks/03-llm-eval.ipynb) — 2 prompt variants                     |
| Interface              | 2      | Streamlit UI [`app.py`](app.py)                                                                      |
| Ingestion pipeline     | 2      | Automated with **dlt** in [`pipelines/movebank_pipeline.py`](pipelines/movebank_pipeline.py)         |
| Monitoring             | 2      | Postgres feedback + [`dashboard.py`](dashboard.py) with 7 charts                                     |
| Containerization       | 2      | Everything in [`docker-compose.yaml`](docker-compose.yaml) (postgres + db-init + app + dashboard)    |
| Reproducibility        | 2      | Pinned deps in [`pyproject.toml`](pyproject.toml), `make bootstrap`, bundled sample data             |
| Hybrid search          | +1     | [`search.HybridSearcher`](search.py) (RRF)                                                           |
| Re-ranking             | +1     | [`search.RerankedSearcher`](search.py) (cross-encoder)                                               |
| Query rewriting        | +1     | [`search.QueryRewriter`](search.py) + `RewritingSearcher`                                            |

---

## Repo layout

```
wildlife-tracking-rag/
├── assets/screenshots/           # README screenshots (app chat UI, dashboard)
├── app.py                       # Streamlit chat UI
├── assistant.py                 # Factory: index + searcher + RAGWithMetrics
├── dashboard.py                 # Streamlit monitoring (7 charts)
├── db_init.py                   # Postgres schema
├── db_save.py / db_feedback.py  # Persist conversations + feedback
├── db_query.py                  # Read queries for the dashboard
├── generate_data.py             # Seed real RAG conversations for the dashboard
├── ingest.py                    # Build text + vector indexes from studies
├── judge.py                     # LLM-as-judge (RelevanceVerdict)
├── llmclient.py                 # OpenAI-compatible client (OpenAI / Ollama / custom)
├── metrics.py                   # RAGWithMetrics + LLMCallRecord + cost table
├── rag_helper.py                # RAGBase, PROMPT_TEMPLATE, format_study_doc
├── search.py                    # Text / Vector / Hybrid(RRF) / Rerank / Rewrite
├── data/
│   ├── movebank_studies.sample.json   # 12 curated public studies (bootstrap)
│   ├── ground_truth.json              # 30 hand-curated Q -> study_id pairs
│   └── retrieval_eval.csv             # Committed retrieval eval results
├── pipelines/movebank_pipeline.py     # dlt: filesystem CSV -> DuckDB
├── scripts/
│   ├── download_movebank.py           # Fetch studies from Movebank REST
│   └── eval_retrieval.py              # Runnable retrieval eval
├── notebooks/
│   ├── 01-ground-truth.ipynb          # LLM-generated ground truth
│   ├── 02-retrieval-eval.ipynb        # Interactive retrieval eval
│   └── 03-llm-eval.ipynb              # Prompt A/B eval with judge
├── Dockerfile
├── docker-compose.yaml                # postgres + db-init + app + dashboard
├── Makefile                           # bootstrap / download / pipeline / index / app / dashboard
├── pyproject.toml                     # pinned deps, python >=3.12
└── .env.example
```

---

## Data attribution

Sample studies bundled in [`data/movebank_studies.sample.json`](data/movebank_studies.sample.json) are metadata about publicly published Movebank studies. Please cite the individual studies (via each study's `citation` field) and Movebank itself when using this data. See [Movebank's citation guidelines](https://www.movebank.org/cms/movebank-content/citation-guidelines).

---

## License

MIT — see [LICENSE](LICENSE).
