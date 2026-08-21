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