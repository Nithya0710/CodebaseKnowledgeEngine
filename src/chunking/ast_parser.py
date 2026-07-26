import tree_sitter_python as tspython

from tree_sitter import Language, Parser
from src.logging_config import get_logger
from pydantic import BaseModel
from pathlib import Path
from typing import Literal, Optional
from src.ingestion.repo_loader import walk_python_files
from src.logging_config import setup_logging

logger = get_logger(__name__)

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class CodeChunk(BaseModel):
    """Represents a chunk of code extracted from a Python file"""
    repo_name: str
    file_path: str
    chunk_type: Literal['function', 'class', 'module_docstring']
    name: str
    code_text: str
    start_line: int
    end_line: int
    docstring: Optional[str]


def _build_function_chunk(node, source_code: bytes, repo_name: str, file_path: Path, name_prefix: str = "") -> CodeChunk:
    """Build a CodeChunk for a function or method definition node."""
    name_node = node.child_by_field_name("name")
    func_name = name_node.text.decode("utf-8")
    full_name = f"{name_prefix}{func_name}"

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    code_text = source_code[node.start_byte:node.end_byte].decode("utf-8")

    docstring_node = _extract_docstring_from_statements(node.child_by_field_name("body"), source_code)
    docstring = None
    if docstring_node is not None:
        raw_text = source_code[docstring_node.start_byte:docstring_node.end_byte].decode("utf-8")
        docstring = raw_text.strip("'\"")

    return CodeChunk(
        repo_name=repo_name,
        file_path=str(file_path),
        chunk_type="function",
        name=full_name,
        code_text=code_text,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
    )


def _extract_docstring_from_statements(container_node, source_code: bytes):
    """... same docstring, but now skips leading comment nodes before
    checking whether the first real statement is a bare string literal."""
    if container_node is None or not container_node.named_children:
        return None

    for child in container_node.named_children:
        if child.type == "comment":
            continue
        first_stmt = child
        break
    else:
        return None  # every child was a comment, no real statement exists

    if first_stmt.type != "expression_statement":
        return None
    if not first_stmt.named_children:
        return None

    string_node = first_stmt.named_children[0]
    if string_node.type != "string":
        return None

    return string_node

def parse_file(file_path: Path, repo_name: str) -> list[CodeChunk]:
    """Parse a single Python file into a list of CodeChunk objects.

    Handles syntax errors gracefully — tree-sitter's error recovery
    means malformed files still yield whatever chunks can be extracted
    from their valid portions, with a warning logged for the rest.
    """    
    try:
        source_code = file_path.read_bytes()
    except OSError as e:
        logger.error(f"Could not read {file_path}: {e}")
        return []
    
    tree = parser.parse(source_code)

    if tree.root_node.has_error:
        logger.warning(f"Tree-sitter detected syntax errors in {file_path}; continuing with partial parse.")

    chunks: list[CodeChunk] = []

    docstring_node = _extract_docstring_from_statements(tree.root_node, source_code)

    if docstring_node is not None:
        raw_text = source_code[docstring_node.start_byte:docstring_node.end_byte].decode("utf-8")
        module_docstring_text = raw_text.strip("'\"")
        chunks.append(
            CodeChunk(
                repo_name=repo_name,
                file_path=str(file_path),
                chunk_type="module_docstring",
                name=f"{file_path.stem}_docstring",
                code_text=module_docstring_text,
                start_line=docstring_node.start_point[0] + 1,
                end_line=docstring_node.end_point[0] + 1,
                docstring=module_docstring_text,
        )
    )
    
    for node in tree.root_node.children:
        if node.type == "function_definition":
            chunks.append(_build_function_chunk(node, source_code, repo_name, file_path))

        elif node.type == "class_definition":
            class_name_node = node.child_by_field_name("name")
            class_name = class_name_node.text.decode("utf-8")
            body_node = node.child_by_field_name("body")
            class_end_byte = node.end_byte
            class_end_line = node.end_point[0] + 1
            if body_node is not None and body_node.named_children:
                class_end_byte = body_node.named_children[0].start_byte
                class_end_line = body_node.named_children[0].start_point[0] + 1

            class_code_text = source_code[node.start_byte:class_end_byte].decode("utf-8")
            chunks.append(
                CodeChunk(
                    repo_name=repo_name,
                    file_path=str(file_path),
                    chunk_type="class",
                    name=class_name,
                    code_text=class_code_text,
                    start_line=node.start_point[0] + 1,
                    end_line=class_end_line,
                    docstring=None,
                )
            )

            if body_node is None:
                continue

            for child in body_node.children:
                method_node = None
                if child.type == "function_definition":
                    method_node = child
                elif child.type == "decorated_definition":
                    method_node = child.child_by_field_name("definition")

                if method_node is not None:
                    chunks.append(
                        _build_function_chunk(
                            method_node, source_code, repo_name, file_path,
                            name_prefix=f"{class_name}.",
                        )
                    )
    return chunks


def chunk_repo(repo_root: Path, repo_name: str) -> list[CodeChunk]:
    """Parse every Python file in a repo into a flat list of CodeChunks."""
    chunks: list[CodeChunk] = []
    py_files = walk_python_files(repo_root, [])

    for py_file in py_files:
        chunks.extend(parse_file(py_file, repo_name))

    logger.info(f"Chunked {repo_name}: {len(py_files)} files -> {len(chunks)} chunks")
    return chunks


# if __name__ == "__main__":
    # Quick test: parse this file itself and print the root node
    # test_file = Path("data/repos/flask/src/flask/app.py")
    # chunks = parse_file(test_file, "flask")
    # print(f"Parsed {len(chunks)} chunks from {test_file}")

    # print out each chunk's name, chunk_type, and maybe the first line of code_text
    # for chunk in chunks:
    #     print(f"Name: {chunk.name}, Type: {chunk.chunk_type}, First Line: {chunk.code_text.splitlines()[0] if chunk.code_text.splitlines() else 'N/A'}")

    # testing the docstring extraction
    # docstring_chunks = [c for c in chunks if c.docstring is not None]
    # print(f"\n{len(docstring_chunks)} chunks have a docstring:")
    # for c in docstring_chunks[:3]:
    #     print(f"  {c.name}: {c.docstring[:80]!r}")

    # testing chunk_repo()
    # setup_logging()
    # chunks=chunk_repo(Path("data/repos/click"), "click")
    # print(f"Total chunks: {len(chunks)}")

    # testing parse_file()
    # chunks=parse_file(Path("data/repos/requests/src/requests/__init__.py"), "requests")
    # docstring_chunks = [c for c in chunks if c.chunk_type == "module_docstring"]
    # for c in docstring_chunks:
    #     print(c.name, c.start_line, c.end_line)
    #     print(c.code_text[:100])