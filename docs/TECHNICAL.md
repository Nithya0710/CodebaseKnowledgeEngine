## Day-1
This project is about a system that ingests 5-10 open-source Python GitHub repositories, indexes them with code-aware chunking and multi-strategy retrieval, and allows engineers to query across the entire codebase in natural language and receive grounded, code-citing answers with exact file paths and line numbers.

Instead of using a simple RAG pipeline, using an advanced one allows the LLM to have better context. In case of character boundaries, it can be modified to allow it to break after functions, instead of cutting off in the middle which doesn't give the LLM the correct context for a given function in a file. With respect to semantic embeddings which will miss identifiers, we can modify BM25 search with a custom tokenizer so that it searches for the exact identifier and splits camel case and snake case identifiers into sub-tokens before indexing. And we already know that relevant code is spread across files, so this system helps us to understand how code connects and where. This is solved by using a dependency graph that maps which files import which and enables context expansion at query time.

This project is built to run fully locally/free via Ollama, with Gemini available as an optional swap-in later.

## Day-5

### Why does Naive RAG Fail here?

Q1 (auth middleware) — Problem 3, cross-file context. 
The model found httpbasicauth.py (a real, relevant hit) but then explicitly said it couldn't determine where middleware is defined and wired in without more files. That's not a bad LLM — that's naive top-5 retrieval genuinely not surfacing the file that registers/calls the auth check. Single-hop cosine retrieval doesn't know that answering "where is X wired in" often requires the caller, not just the definition.

Q2 (rate limiter) — Problem 2, identifier/semantic mismatch, plus corpus coverage. This one's subtler: none of the 8 repos actually implement classic API rate-limiting (that's more of a Flask-extension or infra concern), so the model correctly hedged rather than hallucinating — it found LimitedStream and Limits by loose semantic association with "limit," not by understanding intent. This is a good one to note as not strictly a retrieval failure — it's the naive baseline being asked about something arguably out-of-corpus.

Q3 (payment failure) — textbook Problem 3. No payment function exists in these repos at all, so the model reached for handle_user_exception/InternalServerError by generic "error handling" similarity and confabulated a plausible-sounding trace, even including a fabricated code snippet. This is the best example of embedding similarity producing fluent nonsense rather than an honest "not found".

Q4 (getUserById) — It actually found two real, correct call/definition sites (BlueprintSetupState.add_url_rule, Scaffold.add_url_rule) — solid retrieval, because add_url_rule as a literal string is distinctive enough that cosine similarity had less room to wander. But then it tacked on URL.is_absolute_url/is_relative_url from httpx and asserted a connection between them and Flask's routing that doesn't actually exist — those are unrelated URL-parsing utilities from a different library entirely, pulled in by superficial "URL" keyword/semantic overlap. That's a clean, concrete example of naive top-k retrieval mixing a correct hit with an irrelevant one and the LLM confidently stitching a false narrative across both. Also worth noting: what it found were definitions of add_url_rule, not call sites — the question asked "where is it called," but retrieval surfaced where it's defined instead. That's a subtly different failure mode from Q1-Q3: not "wrong topic," but "right term, wrong relationship to that term".

Q5 (Flask routing) — Problem 1, confirmed twice now. Same URL-building-vs-URL-matching conflation as the earlier smoke test, consistent across runs.

## Day-9
### The cap (`augment.py`)
The cap exists because uncapped depth-2 traversal through hub nodes pulls in architecturally-adjacent-but-conceptually-irrelevant files, and there's no cheap way to distinguish a meaningful edge from a hub-node pass-through without something like edge weighting or hub-exclusion, which is out of scope for today.

Cap via early break in fused-order: it privileges the top-ranked chunk's neighborhood entirely, rather than spreading the cap budget across multiple fused chunks' neighborhoods. Not something to fix today — RRF's fused order is a reasonable priority signal, and a more balanced allocation strategy (e.g. round-robin across fused chunks, one related file each, until the cap is hit) is a legitimate future improvement.

Augmentation didn't surface a JWT file (none exists), but it did prove the mechanism works — the uncapped run found a real, valuable cross-file connection (werkzeug/security.py) that hybrid retrieval alone completely missed, and the capped run demonstrates the guard rail functions correctly, with a documented, real tradeoff about allocation order.

**Problem:** 
`dependency-graph` augmentation over-weights highly-connected "hub" files. get_related_files (Day 4) uses unweighted BFS (nx.single_source_shortest_path_length) to find files within depth hops — every import edge counts equally, whether it's a meaningful architectural dependency or a generic utility import shared by dozens of unrelated files. At depth=2, this let augmentation traverse through hub files (e.g. werkzeug/utils.py-style common modules) and pull in conceptually unrelated chunks from across the corpus — confirmed empirically: an uncapped run on a single query expanded 15 fused chunks to 94 augmented chunks, many with no real relevance to the query.

The original cap implementation exhausted one fused chunk's entire related-file neighborhood before giving any other fused chunk a turn, meaning a single highly-connected chunk (e.g. httpx's BaseClient.auth) could consume the entire max_augmented budget on its own dependencies alone, crowding out more relevant discoveries from other fused chunks' neighborhoods (e.g. werkzeug/security.py's _hash_internal, found only in the uncapped run).

Today's fixes solved "does the same chunk's neighborhood always win" (yes, fixed) and "does the same rank position always win across different queries" (yes, fixed) — but they don't solve "does every fused chunk always get a turn when cap < fused count," which remains an open, honestly-documented tradeoff, not a bug.

1. Round-robin allocation — replaced exhaust-one-chunk-then-move-on with one-addition-per-fused-chunk-per-pass, so a single highly-connected chunk can't consume the entire cap alone.
2. Hub-node exclusion (HUB_DEGREE_THRESHOLD = 12, chosen empirically from the real graph's degree distribution: median 3, mean 4.78, clear elbow around 12-14) — prevents depth-2 traversal from bridging through generic utility files to reach unrelated importers, while still allowing hub files to appear directly as depth-1 neighbors.
3. Deterministic query-hash rotation — round-robin now starts from an offset derived from a SHA-256 hash of the query, so no single RRF rank position is systematically favored across different queries, while staying fully reproducible for testing.
4. Known residual limitation, left undone by design: when max_augmented < len(fused_chunks), a single pass can exhaust the cap before every fused chunk gets a turn — a smaller-scale version of the original fairness problem, not eliminated, just significantly reduced in scope and no longer always privileging the same position.

#### Hub-threshold generalization
The original HUB_DEGREE_THRESHOLD = 12 was a constant derived by manually inspecting this project's specific 268-node, 8-repo graph — correct for this corpus, but not something that would generalize to an arbitrary user-supplied repo of a different size, which matters given the project's eventual goal of accepting any cloned GitHub repo, not a fixed pre-selected set. Replaced with compute_hub_threshold(), which derives the cutoff dynamically as the 95th percentile of the graph's own degree distribution at call time, with a minimum-node-count guard (20 nodes) below which hub detection disables entirely rather than produce a statistically meaningless cutoff on a tiny repo. Validated against the real corpus: the dynamically computed threshold (14.0) landed close to the originally hand-picked value (12), confirming the automated approach finds a comparable, sensible cutoff without requiring per-corpus manual tuning.

## Day-10
### Cross-Encoder
This cross-encoder judges "does this text seem relevant to this query" in a general sense, not "is this idiomatically correct code for this task" — it wasn't trained on code-query pairs.

Reranking isn't strictly "better," it's "differently opinionated," and that's worth being explicit about rather than assuming rerank=improvement by default.

The weighted blend measurably fixed the exact failure mode it was built for — strong multi-signal consensus (Q1) no longer gets casually overridden by a single-model surface-level quirk. It did not fix Q5, but that's because Q5's problem lives upstream in retrieval, not in reranking.

### Architecture
Days 6-9 built increasingly sophisticated retrieval — dense embeddings (two collections), BM25 with identifier-aware tokenization, RRF fusion across three channels, and dependency-graph augmentation — but none of it actually judged fine-grained relevance between a specific query and a specific candidate. Every prior stage is bi-encoder-style: query and document are encoded/scored independently, then compared, which is fast enough for first-stage retrieval over thousands of chunks but caps achievable accuracy.

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) removes that independence: it takes the (query, document) pair jointly and lets a transformer's attention mechanism compare every query token against every document token simultaneously. This is far more accurate, but nothing can be precomputed — every candidate must be scored fresh, per query — which is exactly why it's used only at this small-candidate-set stage (Day 9's ~25 augmented chunks -> 5), never as first-stage retrieval over the full multi-thousand-chunk corpus.

### Weighted blend, not pure cross-encoder override
An initial pure-cross-encoder-sort implementation was found, on real pipeline output, to override strong multi-signal consensus based on surface-level quirks — e.g. displacing a chunk that three independent retrieval channels agreed was the top match, in favor of a chunk that merely contained the literal query word "middleware" in its docstring. Since `ms-marco-MiniLM-L-6-v2` is a general-purpose model trained on web search relevance (not code), it has no special understanding that a
literal keyword match shouldn't automatically outrank hard-won multi-source agreement.

Fixed by blending the cross-encoder's score with the chunk's incoming rank (its position after hybrid retrieval + graph augmentation), rather than letting the cross-encoder fully override it: `final_score = alpha * normalized(cross_encoder_score) + (1 - alpha) * normalized(prior_rank)`, with `alpha=0.65` as a tunable default (cross-encoder remains the majority signal, but prior-stage consensus is a real counterweight, not discarded entirely). Verified on real data: the blend correctly restored a three-channel-agreed chunk to rank #1 that pure cross-encoder sorting had displaced, while still allowing the cross-encoder to dominate when its
signal is strong (confirmed via alpha=0/alpha=1 boundary tests).

### Rank delta logging
Every rerank call logs each surviving chunk's prior rank vs. its final rank, as empirical proof reranking is changing the ordering rather than reproducing input order — deltas of ±7 to ±12 positions were observed on real queries, confirming the cross-encoder's joint-attention scoring performs real, substantial work.

### Bug found: embedding non-determinism causing flaky rankings
A query that intermittently included or excluded a known-relevant chunk (`Flask.dispatch_request`) across separate full-pipeline runs was traced, via isolated diagnostics, to floating-point non-determinism in batched `sentence-transformers` inference — re-embedding identical text in separate forward passes does not guarantee bit-identical output, and this was small but large enough to flip ranking among near-tied candidates competing for the last top-15 slots. Retrieval itself (BM25, Chroma queries, RRF fusion) was confirmed fully deterministic when re-run against already-stored data within a single process — the instability existed only
because the `__main__` pipeline was re-chunking and re-embedding the entire 4,273-chunk corpus on every single run, unconditionally, defeating the persistence guarantee `VectorStoreManager` was built to provide back on Day 6.

Notably, this bug was invisible to every prior day's unit tests (all of which passed throughout), because those tests use small, synthetic 2-3 chunk fixtures — at that scale, batching-related floating-point drift either doesn't occur or is far too small to ever flip a ranking among clearly-distinct candidates. The bug only became observable at the real corpus's scale (4,273 chunks, dozens of near-tied candidates), where a correct, fully-passing test suite still failed to surface it. A concrete reminder that small-fixture unit tests validate logic correctness but cannot substitute for testing at realistic data scale. 

### Fix: idempotent ingestion (`src/retrieval/ingest.py`)
`load_or_ingest_corpus()` now ensures chunking, embedding, BM25 indexing, and graph building happen exactly once per corpus, tracked via a manifest file (`data/ingestion_manifest.json`) recording ingested repo names and per-repo chunk counts. Subsequent runs check the manifest against the requested repos AND cross-verify the vector store's actual stored count against the manifest's recorded total before trusting the cache — any mismatch (new repos, missing files, or a count disagreement, e.g. from a manually cleared Chroma directory) triggers full re-ingestion rather than silently serving stale data.

This eliminated the observed non-determinism entirely (confirmed via repeated identical runs producing byte-identical chunk rankings) and cut full-pipeline run time from ~90 seconds to ~3 seconds on a cache hit — both a correctness fix and a necessary step toward the project's eventual deployment goal, where re-embedding an entire corpus on every user query would never be viable.

Known limitation, not yet addressed: the cache-validity check compares total chunk COUNT, not content — a repo modified in a way that keeps the same total count (e.g. one file removed, a different file added with the same chunk count) would produce a false cache hit, silently serving stale data. A content hash (e.g. of sorted chunk IDs) rather than a raw count would close this gap; deferred as a documented future improvement rather than solved today, since it wasn't the failure mode actually observed.

### Generalization note (multi-repo deployment)
Day 9's hub-exclusion threshold was originally a hardcoded constant tuned by manually inspecting this project's specific 8-repo graph — flagged as a problem given the project's eventual goal of accepting arbitrary user-supplied repos of any size. Replaced with a threshold computed dynamically per-graph (95th percentile of that graph's own degree distribution, with a minimum-node-count guard disabling hub detection on graphs too small for a percentile cutoff to be meaningful). This same principle — deriving thresholds from the data actually presented, rather than hardcoding values tuned to one fixed corpus — should be applied to any future constant introduced in this pipeline.


## Day 11

### Architecture
Days 6-10 built increasingly sophisticated retrieval, but nothing before today actually generated a user-facing answer or checked whether an answer was trustworthy before showing it. Generation uses a local Ollama model (llama3.2:3b).

Two independent guardrails were built, catching two genuinely different failure modes:
- **Pre-generation**: checks whether there's enough relevant context to answer at all, BEFORE calling the LLM — blocks generation entirely on out-of-corpus questions rather than letting the model improvise.
- **Post-generation**: parses every citation out of the generated answer and verifies each cited chunk_id was actually retrieved — catches fabricated citations even when overall context was sufficient.

### Citation format: simplified for a small local model
The spec's multi-field format (`[source: chunk_id, file: ..., lines: ...]`) was not reliably reproduced by llama3.2:3b — real output mixed formats, invented extra citations, and dropped required fields. Simplified to a single mechanical tag, `[cite: chunk_id]`, which the model reproduced correctly. chunk_id is computed via `VectorStoreManager._chunk_id`, the same deterministic ID scheme used since Day 6, so the post-generation validator can directly string-compare cited IDs against retrieved ones with no fuzzy matching. Real generations were still observed occasionally echoing the context block's own label (`[chunk_id: ...]`) instead of the instructed `[cite: ...]` wrapper — the validator's regex was widened to accept both forms rather than fighting the model toward perfect format compliance.

### Bug found: guardrail threshold checked the wrong score entirely
The pre-generation guardrail's first implementation checked Day 10's BLENDED rerank score against a fixed threshold (0.65) — and failed to block a deliberately out-of-scope question ("what's the weather in Bangalore"), which scored 0.72 despite every retrieved candidate being irrelevant. Root cause: the blended score is min-max normalized WITHIN each query's own batch of candidates, so something always scores near 1.0 regardless of whether anything in the batch is actually relevant — this is a structural property of relative normalization, not a calibration issue, and no threshold value on the blended scale could have fixed it (the weather question would have cleared any threshold below 0.72).

Fixed by having `rerank()` return the RAW, un-normalized cross-encoder score alongside the blended score (`(chunk, blended_score, raw_ce_score)`), and switching the guardrail to check the raw score instead. Raw scores are comparable ACROSS queries (unlike blended scores, which only mean something within their own batch), which is what an absolute "is there real signal here" check requires. Threshold set to -10.0, empirically derived from real data: every genuine in-corpus question's best raw score observed was -9.35 or better; the out-of-scope question's best raw score was -10.70. Re-run after the fix: the same weather question was correctly blocked (best_raw_score=-10.70 < -10.0), while all 5 real baseline questions passed through correctly. Noted as a thin, data-limited threshold (one negative example) worth revisiting with more out-of-scope test cases, especially once the corpus becomes arbitrary user-supplied repos rather than a fixed, known set.

### Known limitation: citation validity ≠ claim support
The post-generation guardrail verifies a cited chunk_id was actually retrieved — it does NOT verify the citation actually supports the specific claim it's attached to. Observed in a real transcript: the model correctly hedged "the rate limiter is not explicitly used in the Limits class" while still citing the Limits chunk — a technically valid citation (the chunk was genuinely retrieved) attached to a claim the citation doesn't specifically substantiate either way. This is why stripped citations only remove the tag, never the surrounding sentence — citation correctness and claim correctness are independent properties, and a wrong citation doesn't imply a wrong claim. This gap — valid citation vs. genuinely supported claim — is exactly what Day 12's RAGAS `faithfulness` metric is designed to measure, since faithfulness checks semantic support against retrieved context, not citation-set membership.

### Model choice reconsidered twice, kept local both times
Switching to a hosted model (Gemini) was considered twice today — once over citation format reliability, once over general hallucination concern — and rejected both times. Rationale: a stronger model reduces hallucination frequency but does not eliminate the need for guardrails, since even large hosted models fabricate citations and unsupported claims. Today's evidence supports keeping guardrails as the primary safety mechanism regardless of model choice: the pre-generation gate correctly blocked a real out-of-scope question, and the post-generation validator correctly caught and stripped a real fabricated citation (`check_auth`) mid-session. The bar for actually switching models isn't "answers could be marginally better" — it's measured evidence that the local model structurally cannot produce output the guardrail architecture can act on (e.g., citation format the validator can never reliably parse regardless of prompt tuning), which was not observed after the format simplification.

## Day 12

### Environment rebuild
The venv was rebuilt on a fresh machine (macOS 27, 16 GB RAM). `scipy==1.15.3`, the newest scipy published for Python 3.10, fails to load on macOS 27: dyld rejects its compiled `_propack` extension ("zero-fill section type, but offset field is not zero"), and sklearn and sentence-transformers fail with it because they import `scipy.sparse.linalg`. A freshly created 3.10 venv failed the same way, which rules out the earlier theory that the problem came from copying a venv between machines. Fix: Python 3.11 with `scipy==1.17.1` (verified to load in a throwaway venv first); every other pin from the old freeze is unchanged. `scripts/check_env.py` confirms the rebuilt corpus reproduces all 4,273 committed chunk IDs and the 268-node / 641-edge dependency graph, so ground-truth chunk_ids written against `data/bm25_chunks.json` stay valid.

With 16 GB rather than 8 GB, the default Ollama model moves from `llama3.2:3b` to `qwen2.5:7b` (Q4_K_M, about 4.7 GB) for both generation and the RAGAS judge. llama3.2:3b remains selectable through `OLLAMA_MODEL`.

### Citation labels: [S#] instead of copied chunk_ids
Day 11's `[cite: chunk_id]` format asked the model to copy a long `repo:path:name:line` string. llama3.2:3b found the right function but copied the string imprecisely (e.g. `BaseClient.auth:274` for line 273) and sometimes echoed the context label `[chunk_id: ...]` instead. Each context block is now labeled `[S1]`..`[Sn]`, the prompt states the valid range ("S1 through Sn", or "none" with no context), and `resolve_citations` maps labels back to chunk_ids in code. The `S` prefix keeps citations from colliding with code such as `args[0]` in an answer.

"Before" numbers for this change were not captured: the S-label scheme was merged before a counted baseline run, and the saved Day 11 results (`eval/pipeline_results.json`) contain only post-validation answers, so the stripped hallucinated citations cannot be recovered from them. To make this measurable from now on, `run_pipeline` saves the raw (pre-validation) answer plus `n_valid_citations` and `n_stripped_citations` per question.

### Bug found: citation cleanup flattened the whole answer
After stripping invalid labels, `validate_citations` ran two regexes over the entire answer: `\s{2,}` -> one space and "no whitespace before punctuation". These were meant to tidy the gap a removed tag leaves, but they also collapsed paragraph breaks and code-block indentation everywhere, even when nothing was stripped. A snippet like `def f(x):\n    if x:\n        return 1` came out as `def f(x): if x: return 1`, which would make every multi-line code answer fail the planned `py_compile` correctness check, a failure caused by the guardrail rather than the model. The existing tests used only single-line answers, so they never saw it. Fix: the citation regex now captures the spaces and tabs on either side of a tag, and whitespace is repaired only where a whole tag was removed (one space kept if the tag sat between two words, none otherwise). Newlines are never touched. A regression test with a multi-line code block now covers it.
