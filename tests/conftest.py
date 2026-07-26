import subprocess
from pathlib import Path

import pytest

@pytest.fixture()
def bare_repo(tmp_path: Path) -> Path:
    """Create a local bare git repo with one commit, usable as a clone source."""
    bare = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True)

    # A bare repo has no working directory to add files to directly —
    # so we clone it into a temporary "seed" folder, add a file there,
    # commit, and push back into the bare repo. That gives the bare
    # repo real commit history to clone from later in your tests.
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True)

    (seed / "app.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(seed), "branch", "-M", "main"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "Test User"], check=True)

    subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed commit"], check=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"], check=True)

    return bare


@pytest.fixture()
def clean_python_file(tmp_path: Path) -> Path:
    """A well-formed file with a module docstring, a top-level function,
    and a class with both a decorated and an undecorated method."""
    content = '''"""A clean sample module."""

def top_level_func(x: int) -> int:
    """Doubles the input."""
    return x * 2


class Sample:
    """A sample class."""

    def undecorated_method(self):
        return 1

    @property
    def decorated_method(self):
        return 2
'''
    file_path = tmp_path / "clean.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture()
def syntax_error_file(tmp_path: Path) -> Path:
    """A file with one valid top-level function, followed by a
    structurally broken function definition (unclosed parenthesis)."""
    content = '''def good_function():
    return 1


def broken_function(
    x, y
'''
    file_path = tmp_path / "broken.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture()
def nested_functions_file(tmp_path: Path) -> Path:
    """A top-level function containing a nested inner function —
    only the outer one should be chunked."""
    content = '''def outer_function():
    def inner_function():
        return "should not be chunked"
    return inner_function()
'''
    file_path = tmp_path / "nested.py"
    file_path.write_text(content)
    return file_path