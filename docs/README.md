# Design documents

The design documents for Ask Zero to One are engagement material and are not published in this repository. They are git-ignored and shared with contributors by the project owner. This page describes what they contain, so the code can be read without them.

| Document | Local path | Contents |
|---|---|---|
| Feature specification | `docs/agentic_rag_spec.md` | Use cases, requirements, ingestion, retrieval, tools, answer contract, models, deployment, security and licensing, evaluation, delivery phases, risks, decision log, open questions |
| Reference architecture (v3) | `docs/architecture_v3.png` | The approved software architecture diagram |
| Earlier drafts | `docs/archive/` | Superseded diagrams (v1, v2, first draft), kept for reference |

Code comments cite the specification by section, for example "spec section 6.1". The sections are:

| § | Topic | Where it lands in code |
|---|---|---|
| 3 | Sources and chapter coverage | `sources/` |
| 6.1 | Ingestion pipeline | `rag/ingest` |
| 6.2 | Retrieval | `rag/ingest` (indexes), `rag/tools` (search) |
| 6.3 | Agent tools | `rag/tools` |
| 6.4 | Answer contract | `rag/agent` (verifier) |
| 6.5 | Models | `rag/config.py` |
| 8 | Deployment | `deploy/` (planned) |
| 9 | Security, compliance and licensing | `sources/README.md`, `RAG_BOOK_LICENSED` |
| 10 | Evaluation and the release gate | `rag/eval` |

## Architecture in brief

The system has three parts:

- **Offline build.** CI parses the sources (the book, the analyst's reading notes, and this repo's notebooks and `output/`) into candidate stores, checks the notes against the book, and runs the release evaluation.
- **Released stores.** Only a corpus that passes the evaluation is published, as an immutable version with a manifest of source hashes, index versions and models.
- **Online serving.** An agent answers authenticated users' questions through read-only tools, each scoped to one released store. Every answer is checked before it is returned and logged to an append-only audit store.

Claude handles reasoning (alignment, evaluation, answering). OpenAI handles embeddings. These are the only calls that leave the deployment environment.

Every answer labels which source each statement comes from: Thiel's text, the analyst's interpretation, or the empirical analysis. Verdicts come only from the claims registry built from this repo's findings.
