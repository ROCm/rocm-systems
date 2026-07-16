"""Payload contract: the workload-specific plugin interface (Python spike).

A Payload isolates everything that differs per workload from the generic
Orchestrator. Only `run` is required; the rest have no-op defaults. Each method
receives the Orchestrator (`orch`), giving access to orch.cfg and helpers like
orch.ssh_head(...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Payload(ABC):
    #: short, stable container-name tag (e.g. "cov" -> rccl-cov-<arch>)
    tag: str = "workload"
    #: GitHub env var the collected artifacts dir is exported under
    artifact_env_var: str = "RESULT_ARTIFACT_DIR"

    def validate(self, orch) -> None:
        """Early input checks (before allocation)."""

    def mounts(self, orch) -> List[str]:
        """Extra `--volume host:ctr[:ro]` args (already shlex-quoted) for mnctl."""
        return []

    def prelaunch(self, orch) -> None:
        """After the hostfile is written, before container launch."""

    def prepare(self, orch) -> None:
        """In-container prep after launch (build/install, cleanup)."""

    @abstractmethod
    def run(self, orch) -> None:
        """The in-container command; must write the dir it produced to
        orch.cfg.artifacts_pointer_ctr for deterministic collection."""

    def artifact_glob(self, orch) -> str:
        """Fallback glob (relative to RCCL_DIR) if the pointer file is missing."""
        return f"rccl_test_artifacts_{orch.cfg.run_id}_*"
