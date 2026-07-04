# Architecture Decision Records

One file per decision, format `NNNN-short-slug.md`. Write new ones with `/new-adr`.

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-duplicate-course-ids.md) | Collapse duplicate course_ids to one representative row | Accepted | 2026-06-02 |
| [0002](0002-eval-harness-design.md) | Cross-listing ground truth and a technique-agnostic diversity space | Accepted | 2026-06-02 |
| [0003](0003-judged-query-lens.md) | A hand-labeled judged-query set for the free-text lens | Accepted | 2026-06-03 |
| [0004](0004-semantic-vectors.md) | Semantic-vector recommenders (SBERT local + API), caching and ANN | Accepted | 2026-06-03 |
| [0005](0005-rerank-mmr.md) | Retrieve → cross-encoder rerank → MMR diversity | Accepted | 2026-06-06 |
| [0006](0006-graph-heldout.md) | Course graph (PPR) on a held-out cross-listing edge split | Accepted | 2026-06-19 |
| [0007](0007-clustering-diagnostic.md) | Clustering + 2-D map as a diagnostic, not a recommender | Accepted | 2026-06-19 |
| [0008](0008-metadata-fusion.md) | Metadata fusion — weighted one-hot facets ⊕ TF-IDF | Accepted | 2026-06-19 |
| [0009](0009-llm-enrichment-ollama.md) | LLM enrichment via local Ollama (tag-extraction rung) | Accepted | 2026-06-19 |
| [0010](0010-llm-reranker.md) | Zero-shot LLM reranker (retrieve → LLM reorder over full text) | Accepted | 2026-06-19 |
| [0011](0011-llm-explainer.md) | "Why this fits" explainer — a UI helper, not a scored recommender | Accepted | 2026-06-20 |
| [0012](0012-streamlit-ui.md) | Phase 8 minimal Streamlit UI — testable registry + thin view layer | Accepted | 2026-06-20 |
| [0013](0013-deploy-warm-docker-image.md) | Deploy the UI as a warm, offline CPU Docker image | Accepted | 2026-06-20 |
| [0014](0014-teaching-notebooks.md) | Teaching notebooks — reimplement-from-scratch over a tested library | Accepted | 2026-06-20 |
| [0015](0015-docs-site.md) | Documentation site — MkDocs Material on GitHub Pages, with pre-executed notebooks | Accepted | 2026-07-04 |
| [0016](0016-ci-test-gating.md) | CI test gating with a lean install and a notebook render-freshness guard | Accepted | 2026-07-04 |
