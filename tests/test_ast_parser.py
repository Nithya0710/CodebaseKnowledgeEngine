from pathlib import Path

from src.chunking.ast_parser import parse_file


def test_parse_clean_file(clean_python_file: Path):
    chunks = parse_file(clean_python_file, "test_repo")

    names = [c.name for c in chunks]

    # TODO 1: assert "top_level_func" is in names
    assert "top_level_func" in names
    # TODO 2: assert "Sample" is in names (the class skeleton chunk)
    assert "Sample" in names
    # TODO 3: assert "Sample.undecorated_method" is in names
    assert "Sample.undecorated_method" in names
    # TODO 4: assert "Sample.decorated_method" is in names — this
    #         specifically proves the decorator-handling fix from
    #         earlier today still works
    assert "Sample.decorated_method" in names
    # TODO 5: find the chunk named "top_level_func" specifically
    #         (hint: use a loop or a generator expression to find the
    #         one chunk whose .name matches) and assert its .docstring
    #         equals "Doubles the input."
    top_level_func_chunk = next((c for c in chunks if c.name == "top_level_func"), None)
    assert top_level_func_chunk is not None
    assert top_level_func_chunk.docstring == "Doubles the input."


def test_parse_file_with_syntax_error(syntax_error_file: Path):
    chunks = parse_file(syntax_error_file, "test_repo")

    names = [c.name for c in chunks]

    # TODO 6: assert "good_function" IS in names — proving error
    #         recovery preserved the valid part of the file
    # (We're not asserting anything about "broken_function" here —
    #  tree-sitter's error recovery behavior on malformed code can
    #  vary, so the meaningful claim is just "the good part survived")
    assert "good_function" in names

def test_parse_file_nested_functions_not_chunked(nested_functions_file: Path):
    chunks = parse_file(nested_functions_file, "test_repo")

    names = [c.name for c in chunks]

    # TODO 7: assert "outer_function" IS in names
    assert "outer_function" in names
    # TODO 8: assert "inner_function" is NOT in names — this is the
    #         actual point of the test
    assert "inner_function" not in names