import requests
import json
from src.config import settings
from src.retrieval.vector_store import VectorStoreManager
from src.chunking.ast_parser import CodeChunk
from src.logging_config import get_logger
from src.logging_config import setup_logging

logger = get_logger(__name__)

GENERATION_TEMPERATURE = 0.1


def build_prompt(question: str, ranked_chunks: list[tuple[CodeChunk, float, float]]) -> str:
    """
    Build a generation prompt that requires the model to cite every
    factual claim inline as [cite: chunk_id].

    chunk_id is computed via VectorStoreManager._chunk_id — the SAME
    deterministic ID scheme used everywhere else in this project since
    Day 6 — so the post-generation guardrail can independently recompute
    each retrieved chunk's ID and directly string-compare it against
    whatever the model cites, with no fuzzy matching required.

    Uses a deliberately SIMPLE citation format ([cite: chunk_id], not a
    multi-field bracket structure) after finding that llama3.2:3b did
    not reliably reproduce a more complex [source: ..., file: ..., 
    lines: ...] format — small local models are less reliable at
    multi-field structured output than large hosted models, so the
    format was simplified to something more mechanically reproducible
    rather than switching away from local generation.
    """
    context_blocks = []
    for chunk, score, _ in ranked_chunks:
        chunk_id = VectorStoreManager._chunk_id(chunk)
        context_blocks.append(
            f"[chunk_id: {chunk_id}]\n"
            f"# {chunk.file_path} :: {chunk.name} (lines {chunk.start_line}-{chunk.end_line})\n"
            f"{chunk.code_text}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant answering questions about a Python codebase.\n"
        "Use ONLY the code context below to answer the question. Every factual "
        "claim you make MUST be followed by an inline citation in EXACTLY this "
        "format: [cite: chunk_id]\n"
        "Use the exact chunk_id shown in brackets above each context block — copy "
        "it exactly, character for character. Do not invent, modify, or guess at "
        "a chunk_id.\n"
        "If the context does not contain enough information to answer, say so "
        "plainly rather than guessing.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (with inline citations):"
    )
    return prompt



def generate_streaming(question: str, ranked_chunks: list[tuple[CodeChunk, float, float]]):
    """
    Stream a generated, citation-required answer token by token from
    the local Ollama model, given the question and final ranked chunks
    from the retrieval pipeline (hybrid -> augment -> rerank).

    Yields individual response text fragments as they arrive, for later
    use by a streaming API endpoint (Day 14). Low temperature (0.1) is
    used to favor factual, grounded output over creative variation,
    since citation accuracy matters more than response diversity here.
    """
    prompt = build_prompt(question, ranked_chunks)

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": GENERATION_TEMPERATURE},
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Generation request failed: {e}")
        yield f"[ERROR: generation failed: {e}]"
        return

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse streamed line: {e}")
            continue

        if "response" in chunk:
            yield chunk["response"]
        if chunk.get("done"):
            break



def generate(question: str, ranked_chunks: list[tuple[CodeChunk, float, float]]) -> str:
    """
    Non-streaming convenience wrapper: collect the full streamed
    response into a single string. Used for testing and for today's
    Step 2/5 work, where a complete answer is easier to validate than a
    token stream.
    """
    return "".join(generate_streaming(question, ranked_chunks))


# if __name__=='__main__':
    # testing generation with a sample question and dummy ranked chunks
    # setup_logging()
    # from src.chunking.ast_parser import CodeChunk

    # fake_chunks = [
    #     (CodeChunk(
    #         repo_name="flask", file_path="flask/app.py", chunk_type="function",
    #         name="add_url_rule", start_line=10, end_line=20,
    #         code_text="def add_url_rule(self, rule, endpoint=None, view_func=None):\n    ...",
    #         docstring=None,
    #     ), 0.91),
    # ]

    # answer = generate("What does add_url_rule do?", fake_chunks)
    # print(answer)