"""Container runtime abstraction layer.

Defines the ``ContainerRuntime`` interface and a factory function.
Docker is the default runtime.  The interface is designed so that
Pyxis+Enroot or other SLURM-native container runtimes can be added
by implementing a new subclass in a separate module (e.g. ``pyxis_ops.py``).

Extension point for a new runtime:
    1. Create ``mnctl/<name>_ops.py``
    2. Subclass ``ContainerRuntime`` and implement all abstract methods
    3. Register the name in ``get_runtime()`` below
"""

from abc import ABC, abstractmethod
from typing import Optional


class ContainerRuntime(ABC):
    """Abstract base class for container runtimes.

    Each subclass encapsulates one way to build images, launch containers,
    run ephemeral build tasks, and stop containers.  The orchestration
    layer (``orchestrate.py``) fans operations out to nodes and uses
    this interface for the per-node work.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self):
        # type: () -> str
        """Short runtime identifier (e.g. ``'docker'``, ``'pyxis'``)."""
        pass

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------
    @abstractmethod
    def image_exists(self):
        # type: () -> bool
        """Return True if the configured image is available locally."""
        pass

    @abstractmethod
    def build_image(self):
        # type: () -> None
        """Build or import the container image (idempotent unless --rebuild)."""
        pass

    def image_details(self):
        # type: () -> Optional[str]
        """Return a human-readable string with image metadata, or None."""
        return None

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def launch(self):
        # type: () -> None
        """Launch a container on the current node (idempotent)."""
        pass

    @abstractmethod
    def get_stop_cmd(self):
        # type: () -> str
        """Return a shell command that stops/removes the container.

        This command is executed on each node (locally or via SSH) by the
        orchestration layer.  It must be safe to run even if no container
        is running (i.e. idempotent).
        """
        pass

    # ------------------------------------------------------------------
    # Ephemeral build tasks
    # ------------------------------------------------------------------
    @abstractmethod
    def run_build_task(self, script, log_path):
        # type: (str, str) -> int
        """Run a bash *script* inside a throwaway container.

        The container mounts ``cfg.shared_dir`` at ``/opt/shared`` so
        the build output persists.  All stdout/stderr is written to
        *log_path*.

        Returns the process exit code (0 = success).
        """
        pass


def get_runtime(cfg):
    """Instantiate the container runtime selected by ``cfg.runtime_name``.

    Raises ``SystemExit`` if the name is not recognised.
    """
    name = cfg.runtime_name

    if name == "docker":
        from .docker_ops import DockerRuntime
        return DockerRuntime(cfg)

    # Future runtimes:
    # if name == "pyxis":
    #     from .pyxis_ops import PyxisRuntime
    #     return PyxisRuntime(cfg)

    from .utils import error
    error(
        "Unknown runtime: '{}'. Available: docker".format(name)
    )
    raise SystemExit(1)
