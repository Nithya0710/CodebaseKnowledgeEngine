import pytest
import json
from src.generation.generator import build_prompt, generate, generate_streaming
from src.chunking.ast_parser import CodeChunk


def make_chunk(name="foo", file_path="a.py", repo_name="test", start_line=1, code_text=None):
    return CodeChunk(
        repo_name=repo_name, file_path=file_path, chunk_type="function",
        name=name, start_line=start_line, end_line=start_line + 1,
        code_text=code_text or f"def {name}(): pass", docstring=None,
    )


class FakeStreamingResponse:
    """
    Stand-in for requests. Response when stream=True — mimics Ollama's
    newline-delimited JSON streaming format without a real network call.
    """
    def __init__(self, lines, status_ok=True):
        self._lines = lines
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise Exception("simulated HTTP error")

    def iter_lines(self):
        return iter(self._lines)


class TestBuildPrompt:
    def test_empty_ranked_chunks_produces_empty_context(self):
        """
        build_prompt should not crash on zero chunks — should produce
        a valid prompt string with an empty CONTEXT section, letting the
        model's own 'say so if not enough information' instruction
        handle it, rather than the prompt-builder crashing.
        """
        prompt = build_prompt("some question", [])
        assert "CONTEXT:" in prompt
        assert "some question" in prompt

    def test_multiple_chunks_all_included_in_context(self):
        """
        Every chunk passed in must appear in the built prompt — a
        chunk silently dropped from context would be a serious,
        hard-to-notice bug (the model would just never know it existed).
        """
        chunks = [
            (make_chunk(name="foo", code_text="def foo(): pass"), 0.9, -2.0),
            (make_chunk(name="bar", code_text="def bar(): pass"), 0.8, -3.0),
            (make_chunk(name="baz", code_text="def baz(): pass"), 0.7, -4.0),
        ]
        prompt = build_prompt("question", chunks)
        assert "def foo(): pass" in prompt
        assert "def bar(): pass" in prompt
        assert "def baz(): pass" in prompt

    def test_chunks_are_labeled_in_order(self):
        chunks = [
            (make_chunk(name="foo", code_text="def foo(): pass"), 0.9, -2.0),
            (make_chunk(name="bar", code_text="def bar(): pass"), 0.8, -3.0),
        ]
        prompt = build_prompt("question", chunks)
        assert prompt.index("[S1]") < prompt.index("def foo(): pass")
        assert prompt.index("[S2]") < prompt.index("def bar(): pass")
        assert prompt.index("def foo(): pass") < prompt.index("[S2]")

    def test_prompt_states_valid_label_range(self):
        chunks = [(make_chunk(name=f"f{i}"), 0.9, -2.0) for i in range(3)]
        prompt = build_prompt("question", chunks)
        assert "S1 through S3" in prompt

    def test_prompt_does_not_expose_raw_chunk_ids(self):
        """The model should never see the long ID string, since imprecise
        copying of it was the failure this design removes."""
        from src.retrieval.vector_store import VectorStoreManager
        chunk = make_chunk(name="add_url_rule", file_path="flask/app.py", repo_name="flask", start_line=10)
        prompt = build_prompt("question", [(chunk, 0.9, -2.0)])
        assert VectorStoreManager._chunk_id(chunk) not in prompt


class TestGenerateStreaming:
    def test_ollama_connection_failure_yields_error_string(self, monkeypatch):
        """
        If Ollama is unreachable, generate_streaming should yield a
        clean error string rather than raising — matches the established
        pattern from Day 5's naive_query and every retrieval stage since.
        """
        def fake_post(*args, **kwargs):
            raise ConnectionError("simulated: Ollama not running")

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = list(generate_streaming("question", chunks))
        assert len(result) == 1
        assert "[ERROR: generation failed:" in result[0]

    def test_non_200_status_yields_error_string(self, monkeypatch):
        """
        A non-2xx HTTP response (e.g. Ollama returns a 500) should be
        caught via raise_for_status() and surfaced as a clean error
        string, not propagate as an uncaught exception.
        """
        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=[], status_ok=False)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = list(generate_streaming("question", chunks))
        assert len(result) == 1
        assert "[ERROR: generation failed:" in result[0]

    def test_malformed_json_line_is_skipped_not_fatal(self, monkeypatch):
        """
        A single corrupted line in the stream (e.g. a truncated
        network chunk) should be logged and skipped, not crash the whole
        generation — subsequent valid lines should still be yielded.
        """
        lines = [
            b'{"response": "valid token 1", "done": false}',
            b'{not valid json at all',
            b'{"response": "valid token 2", "done": false}',
            b'{"response": "", "done": true}',
        ]

        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=lines)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = list(generate_streaming("question", chunks))
        assert "valid token 1" in result
        assert "valid token 2" in result
        # The malformed line contributed nothing; the final line's empty
        # "response" (paired with done=true) is still yielded — that's
        # correct behavior, matching real Ollama output.
        assert result == ["valid token 1", "valid token 2", ""]

    def test_empty_lines_in_stream_are_skipped(self, monkeypatch):
        """
        Ollama's streaming output can include blank keep-alive lines —
        these should be silently skipped, not cause an error.
        """
        lines = [
            b"",
            b'{"response": "hello", "done": false}',
            b"",
            b'{"response": "", "done": true}',
        ]

        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=lines)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = list(generate_streaming("question", chunks))
        # Blank keep-alive lines (b"") are skipped via `if not line: continue`.
        # The final line's empty "response" is still yielded (see note above).
        assert result == ["hello", ""]

    def test_stops_at_done_true_ignoring_later_lines(self, monkeypatch):
        """
        Once a line with done=true is received, generation should
        stop yielding — any lines after that (shouldn't normally happen,
        but worth guarding) must be ignored.
        """
        lines = [
            b'{"response": "first", "done": false}',
            b'{"response": "final", "done": true}',
            b'{"response": "should never appear", "done": false}',
        ]

        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=lines)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = list(generate_streaming("question", chunks))
        assert result == ["first", "final"]
        assert "should never appear" not in result


class TestGenerate:
    def test_generate_joins_streamed_tokens_into_single_string(self, monkeypatch):
        """
        The non-streaming generate() wrapper should concatenate every
        yielded fragment into one complete string.
        """
        lines = [
            b'{"response": "Hello ", "done": false}',
            b'{"response": "world", "done": false}',
            b'{"response": "!", "done": true}',
        ]

        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=lines)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        chunks = [(make_chunk(), 0.9, -2.0)]
        result = generate("question", chunks)
        assert result == "Hello world!"

    def test_generate_with_empty_ranked_chunks_does_not_crash(self, monkeypatch):
        """
        generate() with zero retrieved chunks (context-less generation)
        should still complete without crashing — the model will likely
        just say it lacks information, per the prompt's own instruction.
        """
        lines = [b'{"response": "I don\'t have enough context.", "done": true}']

        def fake_post(*args, **kwargs):
            return FakeStreamingResponse(lines=lines)

        monkeypatch.setattr("src.generation.generator.requests.post", fake_post)

        result = generate("question", [])
        assert "context" in result.lower()