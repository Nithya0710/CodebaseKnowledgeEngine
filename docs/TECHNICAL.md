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