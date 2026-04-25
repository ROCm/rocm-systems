"""Image-build helpers: Dockerfile parsing, base-image preparation, ``docker build``.

All entry points take ``cfg`` plus an ``image_exists`` callable so this
module never reaches into a runtime instance directly (keeps it
unit-testable with a fake ``image_exists``).
"""

import os
import re
import subprocess
from typing import Callable, Dict, Optional

from ..utils import error, log, log_verbose, run_capture, Timer
from ._helpers import docker_output


def resolve_dockerfile_base(dockerfile_path):
    # type: (str) -> Optional[str]
    """Parse a Dockerfile to resolve the actual base image from ARG defaults.

    Handles patterns like::

        ARG ROCM_IMAGE=some/image:tag
        FROM ${ROCM_IMAGE}

    and::

        ARG ROCM_IMAGE_NAME=some/image
        ARG ROCM_IMAGE_TAG=1.0
        FROM "${ROCM_IMAGE_NAME}:${ROCM_IMAGE_TAG}"
    """
    try:
        with open(dockerfile_path, "r") as f:
            lines = f.readlines()
    except (IOError, OSError):
        return None

    args = {}  # type: Dict[str, str]
    from_line = None

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^ARG\s+(\w+)=(.+)$", stripped)
        if m:
            args[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        if stripped.upper().startswith("FROM "):
            from_line = stripped[5:].strip().strip('"').strip("'")
            break

    if from_line is None:
        return None

    def _sub(match):
        # type: (re.Match) -> str
        name = match.group(1)
        return args.get(name, match.group(0))

    resolved = re.sub(r"\$\{(\w+)\}", _sub, from_line)
    resolved = re.sub(r"\$(\w+)", _sub, resolved)
    return resolved if "$" not in resolved else None


def sync_base_from_dockerfile(cfg, dockerfile_path):
    # type: (object, str) -> None
    """Update ``cfg.rocm_image`` from the Dockerfile when not user-specified.

    Parses ARG/FROM defaults so that ``image_tag`` and
    :func:`ensure_base_image` use the correct base for any Dockerfile.
    """
    if cfg.rocm_image_explicit:
        return
    parsed = resolve_dockerfile_base(dockerfile_path)
    if parsed:
        log_verbose(
            "Resolved base image from {}: {}".format(cfg.dockerfile, parsed)
        )
        cfg.rocm_image = parsed


def ensure_base_image(cfg, image_exists, base):
    # type: (object, Callable[[str], bool], str) -> None
    """Verify the base image is available locally; pull if needed."""
    if image_exists(base):
        log_verbose("Base image '{}' found locally".format(base))
        return

    log("  Base image '{}' not found locally, pulling...".format(base))
    pull = run_capture(["docker", "pull", base])
    if pull.ok:
        log("  Pulled '{}'".format(base))
        return

    _print_missing_base_image_hint(cfg, base)
    raise SystemExit(1)


def _print_missing_base_image_hint(cfg, base):
    # type: (object, str) -> None
    """Emit the user-facing options block when the base image is missing.

    Mirrors the structured-hint pattern used in
    :func:`mnctl.ssh._print_ssh_fix_hints` so all "here's how to recover"
    blocks live in their own helpers and are easy to grep for.
    """
    log("")
    error("Base image '{}' is not available".format(base))
    log("  It does not exist locally and could not be pulled.")
    log("")
    log("  The Dockerfile ({}) requires this image in its".format(
        cfg.dockerfile
    ))
    log("  FROM directive. You must make it available first:")
    log("")
    log("  Options:")
    log("    1. Pull from a registry:")
    log("       docker pull {}".format(base))
    log("    2. Build it locally (if you have a Dockerfile for it):")
    log("       docker build -t {} -f <Dockerfile> .".format(base))
    log("    3. Use a different base image:")
    log("       python3 -m mnctl <other-image>")


def build_image(cfg, image_exists):
    # type: (object, Callable[[Optional[str]], bool]) -> None
    """End-to-end image build for the configured Dockerfile/tag.

    Idempotent unless ``cfg.force_rebuild`` is set.
    """
    dockerfile_path = os.path.join(cfg.script_dir, cfg.dockerfile)
    sync_base_from_dockerfile(cfg, dockerfile_path)

    if not cfg.force_rebuild and image_exists():
        log(
            "=== Image {} already exists "
            "(use --rebuild to force) ===".format(cfg.image_tag)
        )
        if cfg.verbose:
            details = image_details(cfg)
            if details:
                log_verbose("Image details: {}".format(details))
        log("")
        return

    with Timer("Image build"):
        log("=== Building image ===")
        log("  Base       : {}".format(cfg.rocm_image))
        log("  Dockerfile : {}".format(cfg.dockerfile))
        log("  Tag        : {}".format(cfg.image_tag))
        log("")

        ensure_base_image(cfg, image_exists, cfg.rocm_image)

        cmd = [
            "docker", "build",
            "--build-arg", "SSH_PORT={}".format(cfg.ssh.port),
            "-t", cfg.image_tag,
            "-f", dockerfile_path,
        ]

        if cfg.force_rebuild:
            cmd.append("--no-cache")

        if cfg.rocm_image_explicit:
            cmd += [
                "--build-arg",
                "ROCM_IMAGE={}".format(cfg.rocm_image),
            ]

        if cfg.gpu_targets:
            cmd += [
                "--build-arg",
                "GPU_TARGETS={}".format(cfg.gpu_targets),
            ]

        if cfg.verbose:
            cmd.append("--progress=plain")
            log_verbose(
                "docker build {}".format(
                    " ".join(cmd + [cfg.script_dir])
                )
            )

        cmd.append(cfg.script_dir)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log("")
            error("Image build failed (exit {})".format(e.returncode))
            log("  Check the Docker build output above for details.")
            log("  Common fixes:")
            log("    - Ensure Docker daemon is running")
            log("    - Check Dockerfile syntax and base image availability")
            log("    - Retry: python3 -m mnctl --rebuild")
            raise SystemExit(1)

    log("")
    log("  Built: {}".format(cfg.image_tag))
    if cfg.verbose:
        log_verbose(
            "Image ID: {}".format(
                docker_output([
                    "docker", "image", "inspect", cfg.image_tag,
                    "--format", "{{.Id}}",
                ])[:12]
            )
        )
    log("")


def image_details(cfg):
    # type: (object) -> Optional[str]
    """Return a one-line summary (id/created/size) of ``cfg.image_tag``."""
    text = docker_output([
        "docker", "image", "inspect", cfg.image_tag,
        "--format", "{{.Id}} created={{.Created}} size={{.Size}}",
    ])
    return text or None
