"""Fakeable I/O layer for GPU detection.

ALL system I/O that detect.py needs lives here. Tests monkeypatch these
functions to inject fake sysfs content and clinfo output, so detect.py
never reads files or runs subprocesses directly.
"""

import os
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# KFD topology
# ---------------------------------------------------------------------------

KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")


def list_kfd_nodes() -> list[Path]:
    """List /sys/class/kfd/kfd/topology/nodes/* directories.

    Returns directories sorted by name (node number). Returns an empty
    list if the KFD topology path does not exist.
    """
    if not KFD_TOPOLOGY.is_dir():
        return []
    return sorted(p for p in KFD_TOPOLOGY.iterdir() if p.is_dir() and p.name.isdigit())


def read_kfd_properties(node_path: Path) -> str:
    """Read raw text of a KFD node's ``properties`` file.

    Args:
        node_path: Path to a KFD topology node directory
            (e.g., ``/sys/class/kfd/kfd/topology/nodes/1``).

    Returns:
        Raw file content as a string.

    Raises:
        FileNotFoundError: If the properties file does not exist.
    """
    return (node_path / "properties").read_text()


# ---------------------------------------------------------------------------
# Windows: clinfo subprocess
# ---------------------------------------------------------------------------


def run_clinfo() -> str:
    """Run ``clinfo`` and return its stdout.

    Used on Windows where sysfs is not available. Returns empty string
    if clinfo is not found or fails.
    """
    try:
        result = subprocess.run(
            ["clinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


def get_env(key: str) -> str | None:
    """Read an environment variable. Patchable for tests."""
    return os.environ.get(key)
