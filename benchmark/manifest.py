from __future__ import annotations

import hashlib
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.rstrip("\r\n")


def repository_metadata(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    status = _git(root, "status", "--short")
    branch = _git(root, "branch", "--show-current") or "DETACHED"
    return {
        "repository_path": str(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status_short": status.splitlines(),
        "submodule_status": _git(root, "submodule", "status", "--recursive").splitlines(),
    }


def runtime_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
    }
