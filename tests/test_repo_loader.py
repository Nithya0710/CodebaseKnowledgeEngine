import json

from pathlib import Path
from src.ingestion.repo_loader import _repo_name_from_url
from src.ingestion.repo_loader import walk_python_files
from src.ingestion.repo_loader import load_repo_urls
from src.ingestion.repo_loader import clone_repo

# test cases for _repo_name_from_url()
def test_repo_name_from_plain_url():
    assert _repo_name_from_url("https://github.com/pallets/flask") == "flask"

def test_repo_name_from_url_with_dotgit_suffix():
    assert _repo_name_from_url("https://github.com/pallets/flask.git") == "flask"

def test_repo_name_from_url_with_trailing_slash():
    assert _repo_name_from_url("https://github.com/pallets/flask/") == "flask"

def test_repo_name_from_url_with_both():
    assert _repo_name_from_url("https://github.com/pallets/flask.git/") == "flask"

# test cases for walk_python_files()
def test_walk_python_files_finds_py_files(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "utils.py").write_text("y = 2")
    (tmp_path / "README.md").write_text("# docs")

    result = walk_python_files(tmp_path, [])
    
    assert len(result) == 2
    assert (tmp_path / "app.py") in result
    assert (tmp_path / "utils.py") in result
    assert (tmp_path / "README.md") not in result


def test_walk_python_files_skips_tests_dir(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text("assert True")
    top_level_file = tmp_path / "app.py"
    top_level_file.write_text("x = 1")

    result = walk_python_files(tmp_path, [])

    assert (tmp_path / "app.py") in result
    assert (tmp_path / "tests" / "test_app.py") not in result

# test cases for load_repos_urls()
def test_load_repo_urls_valid_file(tmp_path: Path):
    config_path = tmp_path / "config.json"
    expected = [
        "https://github.com/pallets/flask",
        "https://github.com/tiangolo/fastapi",
    ]
    config_path.write_text(json.dumps({"repos": expected}))
    
    result = load_repo_urls(config_path)

    assert result == expected


def test_load_repo_urls_missing_file(tmp_path: Path):
    # No file created here on purpose — you're testing the
    # FileNotFoundError branch you wrote inside load_repo_urls.

    result = load_repo_urls(tmp_path / "does_not_exist.json")

    assert result == []


def test_load_repo_urls_malformed_json(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json {{{")

    result = load_repo_urls(config_path)

    assert result == []


def test_load_repo_urls_missing_repos_key(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"repository_urls": ["https://github.com/pallets/flask"]}))

    result = load_repo_urls(config_path)

    assert result == []

# test cases for clone_repo()
def test_clone_repo_fresh_clone(bare_repo: Path, tmp_path: Path):
    dest = tmp_path / "cloned"

    result = clone_repo(str(bare_repo), dest)

    assert result == dest
    assert (dest / ".git").is_dir()
    assert (dest / "app.py").exists()


def test_clone_repo_idempotent(bare_repo: Path, tmp_path: Path):
    dest = tmp_path / "cloned"

    first_result = clone_repo(str(bare_repo), dest)
    second_result = clone_repo(str(bare_repo), dest)

    assert first_result == second_result == dest