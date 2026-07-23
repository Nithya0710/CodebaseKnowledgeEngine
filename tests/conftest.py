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