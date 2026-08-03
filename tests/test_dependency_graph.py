from pathlib import Path

from src.chunking.dependency_graph import build_dependency_graph, get_related_files


def test_get_related_files_depth_1(synthetic_import_graph: Path):
    graph = build_dependency_graph(synthetic_import_graph)
    a_path = str(synthetic_import_graph / "fakepkg" / "a.py")
    b_path = str(synthetic_import_graph / "fakepkg" / "b.py")

    related = get_related_files(graph, a_path, depth=1)

    assert related == [b_path]


def test_get_related_files_depth_2(synthetic_import_graph: Path):
    graph = build_dependency_graph(synthetic_import_graph)
    a_path = str(synthetic_import_graph / "fakepkg" / "a.py")
    b_path = str(synthetic_import_graph / "fakepkg" / "b.py")
    c_path = str(synthetic_import_graph / "fakepkg" / "c.py")

    related = get_related_files(graph, a_path, depth=2)

    assert set(related) == {b_path, c_path}


def test_isolated_file_has_no_related_files(synthetic_import_graph: Path):
    graph = build_dependency_graph(synthetic_import_graph)
    e_path = str(synthetic_import_graph / "fakepkg" / "e.py")

    related = get_related_files(graph, e_path, depth=2)

    assert related == []


def test_unknown_file_returns_empty_list(synthetic_import_graph: Path):
    graph = build_dependency_graph(synthetic_import_graph)

    related = get_related_files(graph, "nonexistent/path.py", depth=2)

    assert related == []