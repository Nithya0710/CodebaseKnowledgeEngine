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
    Build a generation prompt in which each retrieved chunk gets a short
    positional label ([S1], [S2], ...) and the model must cite by label.

    Earlier versions asked the model to copy a long chunk_id string
    (repo:path:name:line) verbatim. llama3.2:3b reliably identified the
    right function but kept reproducing the string imprecisely (e.g. line
    273 -> 274). A short label is trivial to copy exactly, and the label
    -> chunk_id mapping is done deterministically in code
    (guardrails.resolve_citations), so the model never handles the long ID.
    The 'S' prefix avoids collisions with ordinary code in answers such as
    args[0], which a bare [1] label would collide with.
    """
    n = len(ranked_chunks)
    context_blocks = []
    for i, (chunk, _, _) in enumerate(ranked_chunks, start=1):
        context_blocks.append(
            f"[S{i}] {chunk.file_path} :: {chunk.name} "
            f"(lines {chunk.start_line}-{chunk.end_line})\n"
            f"{chunk.code_text}"
        )
    context = "\n\n".join(context_blocks)

    label_range = f"S1 through S{n}" if n else "none (no context was provided)"

    prompt = (
        "You are a helpful assistant answering questions about a Python codebase.\n"
        "Use ONLY the code context below to answer the question. Every factual "
        "claim you make MUST be followed by a citation giving the label of the "
        "source it came from, written exactly like [S1] or [S2].\n"
        f"Valid labels are: {label_range}. Do not invent labels, and never "
        "write file paths or line numbers inside a citation.\n"
        "If the context does not contain enough information to answer, say so "
        "plainly rather than guessing.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (with citations like [S1]):"
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