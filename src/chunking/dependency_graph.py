import tree_sitter_python as tspython
import networkx as nx
import json

from pathlib import Path
from tree_sitter import Language, Parser
from src.logging_config import get_logger
from src.logging_config import setup_logging
from src.ingestion.repo_loader import walk_python_files

logger = get_logger(__name__)
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def parse_imports(file_path: Path) -> list[str]:
    """
    Extract the raw import targets from a Python file.

    Returns strings like 'os', 'flask', 'pathlib', '.', '.utils' —
    unresolved to actual file paths. Resolution to real files within
    the repo happens later, in build_dependency_graph.
    """
    try:
        source_code = file_path.read_bytes()
    except OSError as e:
        logger.error(f"Could not read {file_path}: {e}")
        return []
    
    tree = parser.parse(source_code)
    root_node = tree.root_node

    import_targets: list[str] = []

    for node in root_node.children:
        if node.type == "import_statement":
            # import os, sys
            for child in node.named_children:
                if child.type == "dotted_name":
                    import_targets.append(child.text.decode("utf-8"))
        elif node.type == "import_from_statement":
            # from flask import Flask
            # from . import app
            # from .utils import helper
            module_node = node.child_by_field_name("module_name")

            if module_node is None:
                continue
        
            module_text = module_node.text.decode("utf-8")

            is_bare_relative = module_node.type == "relative_import" and not any(child.type == "dotted_name" for child in module_node.named_children)

            if is_bare_relative:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    module_text += name_node.text.decode("utf-8")

            import_targets.append(module_text)

    return import_targets


def _resolve_import(raw_import: str, current_file: Path, repo_root: Path) -> Path | None:
    """
    Resolve a raw import string to an actual file within repo_root.

    Returns None if the import doesn't correspond to any file in this
    repo (i.e., it's an external package/stdlib import).
    """
    if raw_import.startswith("."):
        leading_dots = len(raw_import) - len(raw_import.lstrip("."))
        remaining_name = raw_import[leading_dots:]

        candidate_dir = current_file.parent
        for _ in range(max(leading_dots - 1, 0)):
            candidate_dir = candidate_dir.parent

        if remaining_name:
            base = candidate_dir / Path(*remaining_name.split("."))
        else:
            return _find_file_for_module(candidate_dir)
    else:
        segments = raw_import.split(".")
        for candidate_root in (repo_root, repo_root / "src"):
            base = candidate_root / Path(*segments)
            resolved = _find_file_for_module(base)
            if resolved is not None:
                return resolved
        return None

    return _find_file_for_module(base)


def _find_file_for_module(base: Path) -> Path | None:
    """
    Given a path with no extension yet, check both possible real
    forms of a Python module: a flat file (base.py), or a package
    directory (base/__init__.py). Returns whichever actually exists,
    or None if neither does.
    """
    flat_file = base.with_suffix(".py")
    if flat_file.is_file():
        return flat_file

    package_init = base / "__init__.py"
    if package_init.is_file():
        return package_init

    return None


def build_dependency_graph(repo_root: Path) -> nx.DiGraph:
    """
    Build a directed graph of file-level import dependencies across a repo.

    Nodes are file paths (as strings). A directed edge A -> B means
    "file A imports something from file B" — directionality matters:
    A depends on B, but B has no knowledge of or dependency on A.
    """
    graph = nx.DiGraph()

    py_files = walk_python_files(repo_root, [])

    for file_path in py_files:
        graph.add_node(str(file_path))

        raw_imports = parse_imports(file_path)

        for raw_import in raw_imports:
            resolved = _resolve_import(raw_import, file_path, repo_root)
            if resolved is None:
                continue

            graph.add_edge(str(file_path), str(resolved))

    logger.info(f"Built dependency graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def get_related_files(graph: nx.DiGraph, file_path: str, depth: int = 2) -> list[str]:
    """
    Return files within `depth` hops of file_path, in either direction
    (files it imports, and files that import it), excluding file_path
    itself.
    """
    if file_path not in graph:
        logger.warning(f"{file_path} not found in dependency graph.")
        return []

    successors = nx.single_source_shortest_path_length(graph, file_path, cutoff=depth)

    predecessors = nx.single_source_shortest_path_length(graph.reverse(), file_path, cutoff=depth)

    related = set(successors.keys()) | set(predecessors.keys())
    related.discard(file_path)

    return list(related)


def save_graph(graph: nx.DiGraph, dest: Path) -> None:
    """
    Persist the dependency graph to dest, so it can be reloaded
    later without rebuilding from scratch.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        graph_data = nx.node_link_data(graph)
        with open(dest, "w") as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Saved dependency graph to {dest}")
    except OSError as e:
        logger.error(f"Failed to save dependency graph to {dest}: {e}")


def load_graph(src: Path) -> nx.DiGraph:
    """Load a previously-saved dependency graph from src."""
    try:
        with open(src) as f:
            graph_data = json.load(f)
        return nx.node_link_graph(graph_data, directed=True)
    except OSError as e:
        logger.error(f"Failed to load dependency graph from {src}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid dependency graph JSON in {src}: {e}")

    return nx.DiGraph()


if __name__ == "__main__":
#     Example usage
#     file_path = Path("scratch.py")
#     imports = parse_imports(file_path)
#     print(imports)

    # testing for _resolve_import()
    # repo_root = Path("data/repos/flask")
    # current = repo_root / "src/flask/app.py"
    # print(_resolve_import(".helpers", current, repo_root))

    # testing for build_dependency_graph()
    setup_logging()
    repo_root = Path("data/repos/flask")
    graph = build_dependency_graph(repo_root)
    # print("Graph has edge from src/flask/app.py to src/flask/helpers.py:", graph.has_edge(str(repo_root / "src/flask/app.py"), str(repo_root / "src/flask/helpers.py")))
    # # Negative check — app.py imports from sansio/app.py, but never
    # # references sansio/blueprints.py or sansio/scaffold.py directly
    # print("app.py -> sansio/blueprints.py:", graph.has_edge(
    #     "data/repos/flask/src/flask/app.py",
    #     "data/repos/flask/src/flask/sansio/blueprints.py",
    # ))  
    # print("app.py -> sansio/scaffold.py:", graph.has_edge(
    #     "data/repos/flask/src/flask/app.py",
    #     "data/repos/flask/src/flask/sansio/scaffold.py",
    # ))

    # testing for get_related_files()
    file_path = str(repo_root / "src/flask/app.py")
    # related = get_related_files(graph, file_path, depth=1)
    # print(f"Files related to {file_path} (depth=1):")
    # for f in related:
    #     print(f"  {f}")
    # related_1 = get_related_files(graph, file_path, depth=1)
    # related_2 = get_related_files(graph, file_path, depth=2)
    # print(f"\ndepth=1 count: {len(related_1)}")
    # print(f"depth=2 count: {len(related_2)}")

    # testing for save_graph() and load_graph()
    save_graph(graph, Path("data/flask_dependency_graph.json"))
    loaded = load_graph(Path("data/flask_dependency_graph.json"))

    print(f"Original: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Loaded:   {loaded.number_of_nodes()} nodes, {loaded.number_of_edges()} edges")

    file_path = str(repo_root / "src/flask/app.py")
    print(f"has_edge app->helpers (original): {graph.has_edge(file_path, str(repo_root / 'src/flask/helpers.py'))}")
    print(f"has_edge app->helpers (loaded):   {loaded.has_edge(file_path, str(repo_root / 'src/flask/helpers.py'))}")