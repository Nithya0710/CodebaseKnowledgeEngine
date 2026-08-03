import tree_sitter_python as tspython

from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

parser = Parser(PY_LANGUAGE)

source = b"import os\nimport flask\nfrom pathlib import Path\nfrom . import app\nfrom .utils import helper\n"
tree = parser.parse(source)
print(tree.root_node)