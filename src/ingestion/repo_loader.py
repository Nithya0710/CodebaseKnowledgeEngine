import subprocess
import json
import os
import fnmatch

from pathlib import Path
from git import GitCommandError, Repo
from src.logging_config import get_logger
from src.logging_config import setup_logging
from datetime import datetime, timezone
from pydantic import BaseModel

logger = get_logger(__name__)

class RepoManifest(BaseModel):
    """Summary of a single ingested repository."""
    repo_name: str
    repo_url: str
    local_path: Path
    file_count: int
    cloned_at: datetime


def _repo_name_from_url(url: str) -> str:
    """Derive a short directory/repo name from a git URL."""
    # TODO: you already wrote this exact logic once, inline, inside
    #       clone_all. Move it here as its own function instead of
    #       duplicating it — then call THIS function from both
    #       clone_all and build_manifest, so there's one source of truth.
    repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    return repo_name


def build_manifest(repo_url: str, local_path: Path) -> RepoManifest:
    """Build a RepoManifest for an already-cloned repo."""
    file_count = len(walk_python_files(local_path, []))
    repo_name = _repo_name_from_url(repo_url)
    cloned_at = datetime.now(timezone.utc)
    return RepoManifest(
        repo_name=repo_name,
        repo_url=repo_url,
        local_path=local_path,
        file_count=file_count,
        cloned_at=cloned_at,
    )


def save_manifest(manifests: list[RepoManifest], dest: Path) -> None:
    """Persist a list of manifests to dest as JSON."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = [manifest.model_dump(mode="json") for manifest in manifests]
        with open(dest, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Wrote manifest for {len(manifests)} repos to {dest}")
    except OSError as e:
        logger.error(f"Failed to write manifest to {dest}: {e}")


def clone_repo(repo_url: str, dest_dir: Path) -> Path:
    """
    Clone a Git repository to the specified directory.

    Idempotent: if dest_dir already contains a valid git repo, the clone
    is skipped and the existing path is returned.

    Args:
        repo_url: The URL of the Git repository to clone.
        dest_dir: The directory where the repository should be cloned.

    Returns:
        The path to the cloned (or already-existing) repository.

    Raises:
        subprocess.CalledProcessError: If both GitPython and the
            subprocess fallback fail to clone the repository.
    """
    if (dest_dir / ".git").is_dir():
        logger.info(f"Directory {dest_dir} already has a repo. Skipping clone.")
        return dest_dir
    try:
        Repo.clone_from(repo_url, dest_dir)
        logger.info(f"Successfully cloned {repo_url} into {dest_dir} (GitPython)")
        return dest_dir
    except GitCommandError as e:
        logger.warning(f"GitPython failed for {repo_url}, falling back to subprocess: {e}")
    try:
        subprocess.run(["git", "clone", repo_url, str(dest_dir)], check=True)
        logger.info(f"Successfully cloned {repo_url} into {dest_dir} (subprocess fallback)")
        return dest_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess fallback also failed for {repo_url}: {e}")
        raise


def clone_all(repo_urls: list[str], base_dir: Path) -> list[Path]:
    """
    Clone every URL in repo_urls into its own subdirectory under base_dir.

    Repos that fail to clone are logged and skipped — one bad URL should
    not prevent the rest of the batch from cloning.

    Args:
        repo_urls: List of git URLs to clone.
        base_dir: Parent directory under which each repo gets its own subfolder.

    Returns:
        Paths to every repo that cloned successfully (failed ones are omitted).
    """
    cloned_paths: list[Path] = []
    manifests: list[RepoManifest] = []
    for url in repo_urls:
        repo_name=_repo_name_from_url(url)
        dest_dir = base_dir / repo_name
        try:
            cloned_path = clone_repo(url, dest_dir)
            manifest = build_manifest(url, dest_dir)
            manifests.append(manifest)
            cloned_paths.append(cloned_path)
        except Exception as e:
            logger.error(f"Failed to clone {url}: {e}")
    save_manifest(manifests, Path("data/manifest.json"))
    return cloned_paths


_DEFAULT_SKIP_DIRS = {"test", "tests", "vendor", "node_modules", "__pycache__"}


def walk_python_files(root: Path, ignore_patterns: list[str]) -> list[Path]:
    """
    Walk root and return every .py file, skipping unwanted directories.

    Directories named test, tests, vendor, node_modules, or __pycache__
    are always pruned. Additional glob patterns in ignore_patterns are
    matched against each file's path relative to root.
    """
    if not root.is_dir():
        logger.info("No root directory.")
        return []
    python_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, root)
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            rel_path = os.path.join(rel_dir, filename) if rel_dir != "." else filename
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in ignore_patterns):
                continue
            python_files.append(Path(dirpath) / filename)
    return python_files


def load_repo_urls(config_path: Path) -> list[str]:
    """Load the list of repo URLs from a JSON config file."""
    try:
        with open(config_path) as f:
            data = json.load(f)
        return data["repos"]
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        return []
    except KeyError:
        logger.error(f"Missing 'repos' key in {config_path}")
        return []


# if __name__ == "__main__":
#     # Example usage for clone_repo() 
#     clone_repo("https://github.com/Nithya0710/AskMyNotes", Path("data/repos"))
    
    # setup_logging()

    # # # Example usage for clone_all() with repos.json
    # with open("repos.json") as f:
    #     repos = json.load(f)["repos"]

    # results = clone_all(repos, Path("data/repos"))
    # print(f"Cloned {len(results)} of {len(repos)} repos.")