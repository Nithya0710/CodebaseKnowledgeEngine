import pytest
import networkx as nx
from src.retrieval.augment import augment_with_related_files
from src.chunking.ast_parser import CodeChunk


def make_chunk(file_path, name, chunk_type="function", code_text="def foo(): pass"):
    return CodeChunk(
        repo_name="testrepo",
        file_path=file_path,
        chunk_type=chunk_type,
        name=name,
        start_line=1,
        end_line=5,
        code_text=code_text,
        docstring=None,
    )


class TestAugmentWithRelatedFiles:
    def test_adds_related_file_chunks_not_already_present(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")
        graph.add_edge("main.py", "utils.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_chunk = make_chunk("helpers.py", "help_fn")
        utils_chunk = make_chunk("utils.py", "util_fn")

        all_chunks = [main_chunk, helpers_chunk, utils_chunk]
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        result_files = {c.file_path for c in result}
        assert result_files == {"main.py", "helpers.py", "utils.py"}
        assert len(result) == 3

    def test_prefers_module_docstring_over_first_chunk(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_fn_chunk = make_chunk("helpers.py", "help_fn")
        helpers_docstring_chunk = make_chunk(
            "helpers.py", "helpers_docstring", chunk_type="module_docstring"
        )

        all_chunks = [main_chunk, helpers_fn_chunk, helpers_docstring_chunk]
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        added = [c for c in result if c.file_path == "helpers.py"]
        assert len(added) == 1
        assert added[0].chunk_type == "module_docstring"

    def test_falls_back_to_first_chunk_when_no_docstring(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_chunk_a = make_chunk("helpers.py", "help_fn_a")
        helpers_chunk_b = make_chunk("helpers.py", "help_fn_b")

        all_chunks = [main_chunk, helpers_chunk_a, helpers_chunk_b]
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        added = [c for c in result if c.file_path == "helpers.py"]
        assert len(added) == 1
        assert added[0].name == "help_fn_a"

    def test_does_not_duplicate_files_already_in_fused_set(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_chunk = make_chunk("helpers.py", "help_fn")

        all_chunks = [main_chunk, helpers_chunk]
        fused = [main_chunk, helpers_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        assert len(result) == 2
        helpers_count = sum(1 for c in result if c.file_path == "helpers.py")
        assert helpers_count == 1

    def test_cap_limits_total_additions(self):
        graph = nx.DiGraph()
        related_files = [f"related_{i}.py" for i in range(5)]
        for rf in related_files:
            graph.add_edge("main.py", rf)

        main_chunk = make_chunk("main.py", "main_fn")
        related_chunks = [make_chunk(rf, f"fn_{i}") for i, rf in enumerate(related_files)]
        all_chunks = [main_chunk] + related_chunks
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=2)

        added = [c for c in result if c.file_path != "main.py"]
        assert len(added) == 2
        assert len(result) == 3


class TestAugmentEdgeCases:
    def test_chunk_file_not_in_graph_handled_gracefully(self):
        graph = nx.DiGraph()
        graph.add_edge("known_a.py", "known_b.py")

        orphan_chunk = make_chunk("not_in_graph.py", "orphan_fn")
        known_a_chunk = make_chunk("known_a.py", "a_fn")
        known_b_chunk = make_chunk("known_b.py", "b_fn")

        all_chunks = [orphan_chunk, known_a_chunk, known_b_chunk]
        fused = [orphan_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        assert result == [orphan_chunk]

    def test_max_augmented_zero_adds_nothing(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_chunk = make_chunk("helpers.py", "help_fn")
        all_chunks = [main_chunk, helpers_chunk]
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=0)

        assert result == [main_chunk]

    def test_empty_fused_chunks_returns_empty_list(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "helpers.py")

        helpers_chunk = make_chunk("helpers.py", "help_fn")
        all_chunks = [helpers_chunk]
        fused = []

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        assert result == []

    def test_related_file_with_zero_chunks_in_corpus_skipped(self):
        graph = nx.DiGraph()
        graph.add_edge("main.py", "empty_file.py")
        graph.add_edge("main.py", "helpers.py")

        main_chunk = make_chunk("main.py", "main_fn")
        helpers_chunk = make_chunk("helpers.py", "help_fn")
        all_chunks = [main_chunk, helpers_chunk]
        fused = [main_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=1, max_augmented=10)

        result_files = {c.file_path for c in result}
        assert result_files == {"main.py", "helpers.py"}
        assert "empty_file.py" not in result_files

    def test_cyclic_imports_do_not_infinite_loop_or_duplicate(self):
        graph = nx.DiGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")

        a_chunk = make_chunk("a.py", "a_fn")
        b_chunk = make_chunk("b.py", "b_fn")
        all_chunks = [a_chunk, b_chunk]
        fused = [a_chunk]

        result = augment_with_related_files("test query", fused, graph, all_chunks, depth=2, max_augmented=10)

        result_files = [c.file_path for c in result]
        assert result_files.count("b.py") == 1
        assert len(result) == 2


class TestRotationFairness:
    def test_different_questions_can_produce_different_offsets(self):
        """Sanity check that _rotation_offset actually varies across
        different query strings, rather than always returning 0 —
        proves the rotation mechanism has a real effect, not just a
        no-op that happens to pass other tests."""
        from src.retrieval.augment import _rotation_offset

        n = 10
        offsets = {_rotation_offset(f"query number {i}", n) for i in range(20)}
        assert len(offsets) > 1, "Expected varied offsets across different queries, got all the same"

    def test_same_question_produces_same_offset(self):
        """Determinism check: the same question string must always
        produce the same rotation offset, since tests and debugging
        rely on reproducibility."""
        from src.retrieval.augment import _rotation_offset

        offset_a = _rotation_offset("How does Flask handle routing?", 10)
        offset_b = _rotation_offset("How does Flask handle routing?", 10)
        assert offset_a == offset_b