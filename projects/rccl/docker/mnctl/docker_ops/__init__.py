"""Docker container runtime implementation.

The package layout is:

  * :mod:`._helpers` — small utilities with no Config dependency
    (render-gid lookup, RDMA bind-mount, docker-output capture).
  * :mod:`.env_map`  — host->container env-var mapping and post-setup dir
    resolution; pure data, referenced by both ``launch`` and the
    multi-node forwarding code in :mod:`mnctl.orchestrate.forward`.
  * :mod:`.build`    — Dockerfile parsing, base-image preparation,
    ``docker build`` driver.
  * :mod:`.launch`   — argv assembly + ``docker run`` + post-launch
    summary; calls into :mod:`.wait` once the container is up.
  * :mod:`.wait`     — entrypoint readiness watcher with wall-clock
    deadline and state probing.

This ``__init__`` only contains :class:`DockerRuntime`, the thin
adapter implementing the :class:`mnctl.runtime.ContainerRuntime` ABC by
delegating to the submodules.  Keep it small; non-trivial logic belongs
in a submodule so it can be tested without instantiating a runtime.
"""

from typing import Optional

from ..runtime import ContainerRuntime
from ..utils import run_capture
from . import build, launch


__all__ = ["DockerRuntime"]


class DockerRuntime(ContainerRuntime):
    """Docker CLI-based container runtime."""

    @property
    def name(self):
        # type: () -> str
        return "docker"

    # --- Image management ---

    def image_exists(self, tag=None):
        # type: (Optional[str]) -> bool
        """Return True if *tag* (or ``cfg.image_tag``) is present locally."""
        target = tag or self.cfg.image_tag
        return run_capture(["docker", "image", "inspect", target]).ok

    def build_image(self):
        # type: () -> None
        build.build_image(self.cfg, self.image_exists)

    def image_details(self):
        # type: () -> Optional[str]
        return build.image_details(self.cfg)

    # --- Container lifecycle ---

    def launch(self):
        # type: () -> None
        launch.launch(self.cfg, self.image_exists)

    def get_stop_cmd(self):
        # type: () -> str
        return (
            "docker rm -f {name} 2>/dev/null "
            "&& echo 'removed' || echo 'not running'"
        ).format(name=self.cfg.container_name)

    # --- Ephemeral build tasks ---

    def run_build_task(self, script, log_path):
        # type: (str, str) -> int
        import subprocess
        with open(log_path, "w") as lf:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", "{}:/opt/shared".format(self.cfg.shared_dir),
                    self.cfg.image_tag,
                    "bash", "-c", script,
                ],
                stdout=lf, stderr=subprocess.STDOUT,
            )
        return result.returncode
