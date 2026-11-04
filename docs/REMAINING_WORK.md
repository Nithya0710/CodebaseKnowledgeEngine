# REMAINING_WORK.md — Handoff for Claude Code

You already know the project (local-first RAG over 8 Python repos: flask, click, jinja, itsdangerous, markupsafe, werkzeug, requests, httpx). This file covers only: where we are, what is left, and the task list.

Days 1–11 are built. We are on **Day 12** (gold test set + RAGAS). Days 13–16 follow. The original curriculum's summary and quiz prompts are dropped; only build tasks remain.

---

## 0. Master checklist

Work top to bottom. Tick a box only when its acceptance criterion (section 5) is met.

- [ ] **T0** Environment rebuilt and full test suite green
- [ ] **T1** Implement numeric `[S#]` citations; fix stale tests and `guardrails.py __main__`
- [ ] **T2** Finish `eval/test_set.json` (24 entries, real chunk_ids, user review)
- [ ] **T3** RAGAS harness with injectable local judge + judge smoke test
- [ ] **T4** Run full RAGAS suite -> `eval/ragas_baseline.csv`
- [ ] **T5** Day 12 metric interpretation in `docs/TECHNICAL.md`
- [ ] **T6** Gemini generation backend + rate limiter (swappable, Ollama kept), compare on gold set
- [ ] **T7** Make rerank / augmentation / dense retrieval toggleable (feature flags)
- [ ] **T8** 4-way ablation -> `eval/ablation_results.csv`
- [ ] **T9** `src/eval/code_correctness.py` (py_compile + ruff) wired into every run
- [ ] **T10** Ablation interpretation in `docs/TECHNICAL.md`
- [ ] **T11** Known-issue fixes (section 4)
- [ ] **T12** Final cleanup: ruff on `src/`, full pytest, pinned `requirements.txt`, `.env.example`, README, `docs/architecture.png`, interview walkthrough script
- [ ] **T13 (optional, only after T1–T12)** FastAPI `/search` + `/health`
- [ ] **T14 (optional, after T13)** Multi-stage Dockerfile + docker-compose + `.dockerignore`, end-to-end smoke test

---

## 1. Current situation

### Built and working (as of last successful runs)
- AST chunking (tree-sitter) -> `CodeChunk`; chunk ID = `VectorStoreManager._chunk_id(chunk)` = `repo_name:file_path:name:start_line` (file_path is the full relative path, e.g. `data/repos/httpx/httpx/_client.py`).
- Dual Chroma collections (`code_chunks`, `doc_chunks`; doc IDs carry a `_doc` suffix, normalized with `.removesuffix("_doc")` before fusion). Embeddings: `all-MiniLM-L6-v2`. The embedding backend is a permanent choice — do not swap it.
- BM25 (`src/retrieval/keyword_index.py`) with identifier-aware tokenizer; RRF fusion (k=60) in `src/retrieval/hybrid.py`.
- Dependency-graph augmentation (`src/retrieval/augment.py`, `src/chunking/dependency_graph.py`): depth-2, `MAX_AUGMENTED_CHUNKS=10`, round-robin, deterministic query-hash rotation, dynamic hub threshold (p95 degree; 14.0 on the 268-node / 641-edge graph).
- Cross-encoder rerank (`src/retrieval/rerank.py`, `ms-marco-MiniLM-L-6-v2`), blend `alpha*minmax(ce) + (1-alpha)*1/(1+prior_rank)` with `alpha=0.65`. `rerank` returns **3-tuples** `(chunk, blended_score, raw_ce_score)`.
- Pre-generation guardrail on the **raw** cross-encoder score (`RAW_SCORE_GUARDRAIL_THRESHOLD = -10.0`, defined in `rerank.py`). Post-generation citation validator.
- Idempotent ingestion `load_or_ingest_corpus(repo_names, repo_roots, force=False)` -> `(all_chunks, bm25_index, vector_store, graph)`; caches in `data/chroma`, `data/bm25_chunks.json`, `data/combined_dependency_graph.json`, `data/ingestion_manifest.json`.
- Generation (`src/generation/generator.py`): Ollama `llama3.2:3b`, temperature 0.1, streaming. The prompt currently asks the model to cite chunks as `[cite: chunk_id]` (no numeric labels yet).
- Eval scaffolding: `src/eval/run_pipeline.py` (`run_pipeline_on_test_set`, writes `eval/pipeline_results.json`; records: question, answer, contexts, ground_truth, ground_truth_context_chunk_ids, retrieved_chunk_ids, blocked). `eval/test_set.json` currently has **8** entries with ground truth.
- Docs: `docs/TECHNICAL.md` has Day 10 and Day 11 sections. `docs/day11_transcripts.json` has 6 transcripts (5 baseline + blocked weather question).

### Citation scheme: proposed change NOT yet applied
Today's repo still uses `[cite: chunk_id]` citations (`build_prompt` labels each context block with its chunk_id; `guardrails.validate_citations` matches `\[cite:\s*([^\]]+)\]`). The 3B model drifts on this format and produces off-by-1/2 line-number citations like `BaseClient.auth:274`. The agreed fix, **to be implemented in T1**, is numeric labels:
- `build_prompt` numbers each context block `[S{i}] file :: name (lines a-b)` and tells the model the valid label range (`S1 through Sn`, or "none" when no context).
- `validate_citations` matches `\[(S\d+(?:\s*,\s*S\d+)*)\]` (comma groups handled), strips out-of-range labels with a logged warning (`Hallucinated citation detected and stripped`), then collapses double spaces and spaces before punctuation.
- New `resolve_citations(answer, ranked_chunks) -> list[str]` maps labels back to chunk_ids deterministically (ordered, de-duplicated, invalid ignored).
- Label is `S#` (not a bare number) to avoid colliding with things like `args[0]`.
- Decision gate: if hallucinated-citation warnings are still frequent with llama3.2:3b after this change, the answer is the Gemini backend (T6), not more prompt tinkering.

Also verify the rest of the guardrail/rerank state: some copies of `tests/test_guardrails.py`, `tests/test_rerank.py` and `guardrails.py` that were reviewed earlier were still on 2-tuples and the old 0.65 blended-score threshold. Confirm everything uses 3-tuples and the raw `-10.0` threshold (T1).

### Environment: NOT working yet
The old venv was broken (torch `libtorch_cpu.dylib` missing, then scipy dlopen "zero-fill section type" error — a venv that had been copied between machines). Plan: rebuild a fresh venv on a different laptop. `requirements.txt` is **stale**; `freeze_old.txt` holds the real versions from the old venv (do not trust torch/scipy builds in it blindly — they came from the broken env, but the version numbers are the reference; torch was 2.13.0). Treat T0 as the first task.

---

## 2. Decisions already made (do not relitigate)

1. **Everything local except generation.** Embeddings, BM25, cross-encoder, RAGAS judge: local. **Generation will use the Gemini API behind a rate limiter** (see T6). Ollama `llama3.2:3b` stays as the swappable/fallback backend and the Day 5–11 baseline for comparison.
2. **RAGAS judge:** try `llama3.2:3b` via ChatOllama + HuggingFace embeddings first. Switch to Groq **only if** the judge is "unusable" (criterion in T3). No bigger local model (8 GB RAM machine). Document the small-judge noise caveat either way.
3. **Chunk IDs in ground truth come only from real `chunk_repo` output**, never from memory. User reviews the finished set.
4. **Secrets:** API keys (Gemini, Groq) via `.env` + `python-dotenv`, never hardcoded, never committed. Provide `.env.example`.
5. Out-of-scope questions are expected to be **blocked by the guardrail**; scoring them is handled separately from in-scope questions (see T4).
6. Claude Code writes the code directly. Append design notes to `docs/TECHNICAL.md` as you go (rationale, parameters, tradeoffs, failures) — the user wants to defend every choice in an interview.

---

## 3. Engineering standards (apply to every task)

- Full type hints; docstrings on every function/class.
- Structured logging via `get_logger(__name__)`; no `print` in library code. Call `setup_logging()` only at entrypoints (`__main__` / CLI / app startup).
- No bare `except`, no silent failures; catch specific exceptions and log with context.
- Pydantic models for structured data crossing module boundaries.
- Deterministic, idempotent behavior; stable IDs.
- Tests: synthetic fixtures (no network, no Ollama, no real corpus in unit tests); cover edge cases (empty input, boundaries, error paths). Integration tests that need Ollama/Gemini/corpus go behind a pytest marker and are skipped by default.
- No secrets in code or git. Respect `.gitignore`.
- Keep stages swappable behind config/flags rather than deleting code.

---

## 4. Known issues (all must be addressed or consciously documented)

| # | Issue | Where | Fix / action |
|---|---|---|---|
| K1 | Ingestion staleness check compares chunk **count** only; changed code with the same count looks fresh | `src/retrieval/ingest.py` | Hash the sorted list of chunk IDs (plus repo list) into the manifest; compare hashes |
| K2 | `original_rank` lookup keyed by `chunk.name + chunk.file_path` can collide (same name, different start_line) | `src/retrieval/rerank.py` | Key by `VectorStoreManager._chunk_id(chunk)` |
| K3 | Routing question retrieval is inconsistent: top-5 for "How does Flask handle routing for a URL?" is URL-building heavy (`Flask.url_for`, `url_for`, `FormDataRoutingRedirect.__init__`, `send_static_file` x2) and `Flask.dispatch_request` appears inconsistently in candidates | retrieval | Investigate with rank-delta logs (BM25 vs dense vs fused vs rerank); report root cause; fix only if cheap and non-regressive, else document in TECHNICAL.md |
| K4 | Leftover `openai_api_key` in settings | `src/config.py` | Remove; add `gemini_api_key` / `groq_api_key` as optional settings |
| K5 | Cosmetic: blank-line noise in logs/output; Chroma/posthog telemetry log spam | logging config, Chroma client | Disable anonymized telemetry (`Settings(anonymized_telemetry=False)`), tidy output |
| K6 | `requirements.txt` stale vs `freeze_old.txt` | repo root | Regenerate from the new working venv; add ragas, langchain-ollama, langchain-huggingface, groq, google-genai (or current Gemini SDK), python-dotenv, ruff |
| K7 | Guardrail threshold `-10.0` was set from essentially one negative example (weather raw score -10.70) | `rerank.py` | After the 24-question set exists, report raw score distributions for in-scope vs out-of-scope; adjust threshold only with evidence; document |
| K8 | Round-robin augmentation: when the cap (10) is below the number of fused chunks/sources, later sources can still be starved | `augment.py` | Document; test with the gold set; do not expand scope unless ablation shows it matters |
| K9 | `guardrails.py __main__` transcript loop unpacks 2-tuples (`for chunk, score in top5`) | `src/generation/guardrails.py` | Unpack `(chunk, blended, raw)` and save both scores |
| K10 | Citation **validity != claim support**: a valid `[S2]` can still cite a chunk that doesn't back the sentence | design | RAGAS faithfulness is the check; call this out in T5 write-up |

---

## 5. Tasks

### T0 — Environment rebuild and green tests
1. `python3.10` (matches old env) or newer; create a **fresh venv on the machine that will run this** (never copy a venv between machines).
2. Install from `freeze_old.txt` versions: `pip install --no-cache-dir -r freeze_old.txt`; if a pin fails to resolve on the new machine, relax only that pin and note it.
3. Import-check: `python -c "import torch, scipy, sklearn, sentence_transformers; print(torch.__version__)"`.
4. Install Ollama, `ollama pull llama3.2:3b`, `ollama serve`; confirm `curl localhost:11434/api/tags`.
5. Ensure `data/repos/{flask,click,jinja,itsdangerous,markupsafe,werkzeug,requests,httpx}` are cloned; run `load_or_ingest_corpus` once (caches regenerate; they are gitignored).
6. `.gitignore` must include: `venv/`, `venv_old/`, `freeze_old.txt`, `data/chroma/`, `data/repos/`, `.env`.

**Acceptance:** all imports succeed; `pytest` passes entirely (approximate counts before T1: hybrid 8, augment 12, dependency_graph 4, ingest 5, keyword_index 14, naive_baseline 3, plus generator, rerank and guardrails; some of those may still be on the old tuple/citation format until T1 — if so, note which fail and why, and fix them in T1 rather than here); corpus loads (~268-node graph for the 8 repos).

### T1 — Implement numeric citations; fix stale tests and `__main__`
1. First, with Ollama running, run `python -m src.eval.run_pipeline` on the existing test set **before changing anything** and record the "before" numbers: count of "Hallucinated citation detected" warnings and answers with zero valid citations.
2. Implement the numeric `[S#]` scheme described in section 1 in `generator.py` and `guardrails.py`.
3. Update tests: `tests/test_guardrails.py` should have 4 pre-generation tests (raw scores, `-10.0`, 3-tuples), 6 validator tests (S-labels, comma groups, out-of-range stripped + warning logged, plain text unchanged, whitespace/punctuation cleanup), 3 resolver tests (ordered, de-duplicated, invalid ignored). `tests/test_rerank.py` on 3-tuples. `tests/test_generator.py` covering the S-label prompt, label-range line, and empty-context case. Update anything still on 2-tuples, `[cite: ...]`, or threshold 0.65.
4. Fix K9.
5. Re-run the pipeline and record the "after" numbers next to the "before" ones in TECHNICAL.md.

**Acceptance:** full pytest green; written count of hallucinated-citation warnings per run; decision recorded: stay on Ollama for generation or proceed to T6 as primary.

### T2 — Finish the gold-standard test set (`eval/test_set.json`, 24 entries)
Existing 8 entries stay (auth middleware, Flask routing, rate limiter [no real rate limiter exists; closest is `LimitedStream`/Limits], payment failure [empty chunk-id list on purpose], `add_url_rule`, `LimitedStream.on_exhausted`, Blueprint/Scaffold, `LimitedStream` byte-limit).

Add 16 more:
- **11 in-scope** across the 8 external repos: werkzeug routing `Map`/`Rule` matching; flask `app.py` exception-handling dependencies; Jinja2 template compilation; requests redirects in a session; httpx `Auth` class / `BaseClient`; click option parsing; itsdangerous sign/verify; MarkupSafe `escape`; jinja2 `Environment` load/cache templates; click `Context`; httpx `Client` vs `AsyncClient`.
- **5 out-of-scope** (expected block, empty chunk-id list): weather in Bangalore, chocolate chip cookie recipe, who won the World Cup, capital of France, a poem about the ocean.

Entry schema: `{question, ground_truth_answer, ground_truth_context_chunk_ids}`. Mix of types: identifier lookup, cross-file tracing, conceptual "how does X work", out-of-scope.

**Rule:** every chunk_id must be copied from real `chunk_repo` output (or the loaded `all_chunks` via `VectorStoreManager._chunk_id`). Write a small helper that, given a repo/name search, prints candidate IDs, and a validator script that fails if any ground-truth ID is not present in the corpus. Ground-truth answers are short (2–4 sentences) and grounded in the actual code you read.

**Acceptance:** 24 entries; validator passes (all IDs exist); >=4 out-of-scope; type mix present; list handed to the user for review before T4 runs.

### T3 — RAGAS harness (`src/eval/run_ragas.py`) with injectable judge
1. Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Context precision/recall use ground-truth chunk IDs/contexts — map chunk IDs to chunk text from the corpus so RAGAS gets reference contexts.
2. Judge LLM and embeddings must be **injected** (default RAGAS is OpenAI — do not use it). Default: `ChatOllama(model="llama3.2:3b")` and `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`. Provide an alternative factory for Groq via env var, selected by a config flag.
3. Smoke test the judge on 3 hand-picked records (one clearly good, one clearly wrong, one out-of-scope) before the full run.
4. **"Unusable" criterion for switching to Groq** (any one triggers it): (a) judge output fails to parse/format on more than ~25% of metric calls (NaN scores); (b) it assigns identical scores (e.g., all 1.0 or all 0.0) to the good and the clearly wrong sample; (c) scores for the wrong-answer sample are >= the good one. Log which criterion fired.
5. Out-of-scope/blocked rows: do not feed refusal messages into faithfulness as if they were answers; exclude from the four metrics (or score separately as "correctly blocked" yes/no) and report guardrail precision/recall instead.

**Acceptance:** `python -m src.eval.run_ragas` runs offline-capable (Ollama only), judge is injectable via config, smoke-test result recorded, unit tests for the sample-to-RAGAS-dataset conversion (synthetic).

### T4 — Run the full suite -> `eval/ragas_baseline.csv`
Run the pipeline over all 24 questions (hybrid -> augment -> rerank -> guardrail -> generate -> validate citations), then RAGAS. Save `eval/pipeline_results.json` and `eval/ragas_baseline.csv` (24 rows x question + 4 metric columns, plus `blocked`, `n_valid_citations`, and the generation backend/judge used). Also report guardrail results on the 5 out-of-scope vs in-scope raw-score distributions (feeds K7).

**Acceptance:** CSV exists with 24 rows; NaN count reported; means per metric printed in the log; judge used is recorded in the file.

### T5 — Day 12 metric interpretation (append to `docs/TECHNICAL.md`)
Plain-language section covering:
- context_precision low -> retrieval returns noise or ranks relevant chunks too low (look at rerank/augmentation).
- context_recall low -> relevant chunks never retrieved (look at chunking, BM25/dense coverage, candidate depth).
- faithfulness low -> generator makes claims not in the context (hallucination; generation model/prompt problem), even with good retrieval.
- answer_relevancy low -> answer drifts from the question.
- Low precision + high faithfulness = retrieval weak but generator honest; high precision + low faithfulness = retrieval fine, generator weak; etc.
- Caveats: small local judge is noisy; sample size 24; citation validity != claim support (K10).
- Report what the actual numbers say about where this pipeline is weak.

**Acceptance:** section present, tied to the real `ragas_baseline.csv` numbers, not generic.

### T6 — Gemini generation backend with rate limiter
- Add a generation backend interface (e.g., `GenerationBackend` protocol with streaming `generate`) with two implementations: `OllamaBackend` (existing) and `GeminiBackend`. Selected by config (`GENERATION_BACKEND=ollama|gemini`); default stays `ollama` until T6 results justify switching.
- Rate limiter: a small, tested, thread-safe limiter (token bucket or sliding window) configurable via settings (requests/minute and a daily cap sized to the free tier of the chosen model — look up current limits rather than assuming). On 429/transient errors: exponential backoff with jitter, bounded retries, clear logged failure; automatic fallback to Ollama only if configured.
- Same `build_prompt` (S-labels), same guardrails and validators on both backends. Temperature low/fixed. Streaming preserved.
- Privacy note for TECHNICAL.md: with Gemini, retrieved code snippets and questions leave the machine (the corpus is public OSS here; retrieval and ranking remain local). Key from `.env`.
- Unit tests with a fake client: rate limiter spacing/cap, 429 retry path, missing key error, streaming passthrough. No live API calls in unit tests.
- Run the 24-question set with both backends; compare hallucinated-citation counts, faithfulness, answer_relevancy.

**Acceptance:** both backends pass the same interface tests; limiter tests deterministic (inject a fake clock); comparison table in TECHNICAL.md; a recommended default with justification.

### T7 — Toggleable pipeline stages (prereq for ablation)
Refactor into one function (e.g., `run_retrieval(question, components, flags)`) with a `PipelineConfig` (Pydantic): `use_dense: bool`, `use_augmentation: bool`, `use_rerank: bool`. **Same code path; one flag disables one stage — no deleted code.**
- No rerank: still return 3-tuples; use a defined fallback (e.g., fused score as blended, and note the raw-score guardrail cannot apply — define and document what the guardrail does in this mode, e.g., skip it or run the cross-encoder only for gating; pick one and state it).
- No augmentation: skip graph step.
- BM25-only: skip both dense collections; fusion over the single BM25 list.
- Default config must reproduce today's behavior exactly (add a regression test comparing outputs on a fixed query with a fake/small corpus).

**Acceptance:** flags covered by unit tests; defaults identical to the current pipeline; behavior of the guardrail in each mode documented.

### T8 — 4-way ablation -> `eval/ablation_results.csv`
Runs on the same 24-question set, same judge, same generation backend: (1) full, (2) no rerank, (3) no augmentation, (4) BM25-only. Columns: `run_name`, the four RAGAS metrics (mean), `code_correctness_pass_rate`, n rows, n blocked, n NaN. Also keep per-question results per run. Seed everything possible; note residual non-determinism. Because the generation model is non-deterministic, consider running each configuration at least twice and reporting the spread, if time allows.

**Acceptance:** CSV with 4 run-name rows; per-question details saved; reproducible command documented.

### T9 — `src/eval/code_correctness.py`
Extract fenced code blocks from generated answers (handle ```python, bare ```, multiple blocks, none); `py_compile` each block for syntax; `ruff check` each block (subprocess, timeouts, specific exception handling); report pass/fail percentage per run and per answer. Define behavior for snippets that are intentionally fragments (document the policy; e.g., select only syntax-error-class ruff rules plus a configured set, and say so). Answers with no code block are excluded from the denominator, reported as `n_with_code`.

**Acceptance:** unit tests with synthetic answers (valid, syntax error, lint-dirty, no block, multiple blocks); integrated into ablation CSV.

### T10 — Ablation interpretation (append to `docs/TECHNICAL.md`)
Results table + interpretation: which removed component hurt the most and whether that matches architectural expectations (e.g., does graph augmentation help or hurt precision? does BM25-only collapse on conceptual questions?). Be honest where differences are within noise.

**Acceptance:** table is generated from the real CSV; interpretation cites specific numbers.

### T11 — Known-issue fixes
Work through section 4 (K1–K10). Each fix gets a test where testable (K1, K2, K9, K4). K3, K7, K8, K10 get written findings in TECHNICAL.md.

**Acceptance:** each K-item either fixed (with test) or documented with evidence.

### T12 — Final cleanup and packaging
- `ruff check src/` clean; full `pytest` green and not flaky (run twice).
- `requirements.txt` regenerated and pinned from the working venv; `.env.example` listing all keys with blank values.
- `README.md`: setup (venv, Ollama pull, repo clone, ingest, run eval), how to run each eval script, headline ablation result.
- `docs/TECHNICAL.md` consolidated into a coherent design-rationale document: the 3 core problems and fixes, the full pipeline architecture (ingestion -> chunking -> dependency graph -> dual vector store + BM25 -> RRF -> graph augmentation -> rerank -> guarded generation), ablation table + interpretation, known limitations.
- `docs/architecture.png`: system diagram of the above (generate with graphviz/mermaid/matplotlib and commit the image).
- `docs/INTERVIEW_WALKTHROUGH.md`: ~5-minute bullet script covering the system end to end.
- No new architecture in this task.

**Acceptance:** all of the above exist; fresh-clone smoke run of the offline path works (ingest -> query -> eval) following the README only.

### T13 (optional — only after T1–T12) FastAPI service
- `POST /search` `{question: str, repo_filter: Optional[str]}` -> `StreamingResponse`: stream answer tokens, then a final structured JSON payload with `sources: [{chunk_id, file_path, start_line, end_line, snippet}]` (use `resolve_citations`/retrieved chunks).
- `repo_filter` becomes a Chroma metadata filter (metadata is set at upsert, see `_chunk_to_metadata`); also filter BM25/graph candidates consistently.
- Errors: 422 malformed body (Pydantic), 503 when Chroma unreachable (health-check dependency), 500 with sanitized message (no stack trace to client; full structured log server-side).
- `GET /health`: Chroma connectivity + BM25 loaded + graph loaded.
- Load heavy resources once at startup (lifespan), not per request. Guardrail-blocked questions return a clean structured "insufficient context" response.
- Tests with FastAPI `TestClient`: valid question (backend faked), malformed body, simulated Chroma failure -> 503, `/health`.

**Acceptance:** tests pass; curl examples in README; no secrets or stack traces in responses.

### T14 (optional — after T13) Docker
- Multi-stage `Dockerfile`: stage 1 `python:3.11` + `build-essential` (tree-sitter C extensions) installing deps into a venv/wheelhouse; stage 2 `python:3.11-slim` copying only the installed packages and app code. Comment explaining why single-stage bloats the image. Verify the app works on 3.11 (the dev env was 3.10).
- `docker-compose.yml`: service `api`, port mapping, `env_file: .env`, named volume `chroma_data` mounted at the Chroma persist dir (not baked into the image). Note: Ollama runs on the host or as a separate service; document how the container reaches it (`OLLAMA_BASE_URL`), or use the Gemini backend.
- `.dockerignore`: `data/repos`, `__pycache__`, `.git`, venvs, `.env`.
- Verify: image size (single vs multi-stage if feasible), `/health` OK, data persists in `chroma_data` across restart.
- End-to-end smoke: fresh clone -> compose up -> ingest -> issue the 4 example queries (auth middleware, rate limiter, payment failure, `add_url_rule`) through the live API; confirm grounded, cited answers and that the payment-failure/rate-limiter cases degrade honestly.

**Acceptance:** image builds; `/health` green; volume persistence demonstrated; results logged in TECHNICAL.md.

---

## 6. Notes on working with the user
- She is learning from this project and must be able to defend every design choice; keep TECHNICAL.md rationale concrete (parameter values, alternatives rejected, failures encountered and fixes).
- Surface the T2 question list and the T4/T6 numbers for her review rather than silently deciding.
- Do not add website/multi-user features or git-SHA caching; that is tracked separately.
