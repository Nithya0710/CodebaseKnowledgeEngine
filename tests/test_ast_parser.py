from pathlib import Path

from src.chunking.ast_parser import parse_file


def test_parse_clean_file(clean_python_file: Path):
    chunks = parse_file(clean_python_file, "test_repo")

    names = [c.name for c in chunks]

    assert "top_level_func" in names

    assert "Sample" in names

    assert "Sample.undecorated_method" in names

    assert "Sample.decorated_method" in names

    top_level_func_chunk = next((c for c in chunks if c.name == "top_level_func"), None)
    assert top_level_func_chunk is not None
    assert top_level_func_chunk.docstring == "Doubles the input."


def test_parse_file_with_syntax_error(syntax_error_file: Path):
    chunks = parse_file(syntax_error_file, "test_repo")

    names = [c.name for c in chunks]

    assert "good_function" in names

def test_parse_file_nested_functions_not_chunked(nested_functions_file: Path):
    chunks = parse_file(nested_functions_file, "test_repo")

    names = [c.name for c in chunks]

    assert "outer_function" in names

    assert "inner_function" not in names