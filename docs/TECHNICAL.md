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