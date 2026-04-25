"""Container launch path: argv assembly, ``docker run``, summary, wait.

The only mutable state touched here is ``cfg`` (read-only) and the
container itself; helpers are pure where possible.  See
:func:`assemble_run_args` for the canonical list of host->container
mounts and ``-e`` flags.
"""

import os
import subprocess
from typing import List

from ..utils import error, log, log_verbose, Timer
from ._helpers import bind_mount_rdma_libs, docker_output, get_render_gid
from .env_map import container_env_pairs, resolve_post_setup_dirs
from .wait import wait_for_entrypoint


def remove_container(cfg):
    # type: (object) -> None
    """Force-remove the container by name (no-op if absent)."""
    from ..utils import run_capture
    run_capture(["docker", "rm", "-f", cfg.container_name])


def assemble_run_args(cfg):
    # type: (object) -> List[str]
    """Build the full argument list for ``docker run``.

    See :func:`mnctl.docker_ops.env_map.container_env_pairs` for the
    authoritative host-to-container environment-variable mapping.
    """
    render_gid = get_render_gid()

    args = [
        "-d",
        "--name", cfg.container_name,
        "--privileged",
        "--security-opt", "apparmor=unconfined",
        "--security-opt", "seccomp=unconfined",
        "--restart", "unless-stopped",
        "--group-add", "video",
        "--group-add", render_gid,
        "--cap-add", "SYS_PTRACE",
        "--network", "host",
        "--ipc", "host",
        "--shm-size", cfg.shm_size,
        "--ulimit", "memlock=-1",
    ]

    for name, value in container_env_pairs(cfg, render_gid):
        args += ["-e", "{}={}".format(name, value)]

    if os.path.exists("/dev/kfd"):
        args += ["--device", "/dev/kfd"]
    if os.path.isdir("/dev/dri"):
        args += ["--device", "/dev/dri"]

    if os.path.isdir("/dev/infiniband"):
        args += ["--device", "/dev/infiniband:/dev/infiniband"]
        for entry in sorted(os.listdir("/dev/infiniband")):
            dev = "/dev/infiniband/{}".format(entry)
            if os.path.exists(dev):
                args += ["--device", "{}:{}".format(dev, dev)]
        log_verbose(
            "InfiniBand devices: {}".format(
                " ".join(os.listdir("/dev/infiniband"))
            )
        )
        if cfg.nic_type == "mellanox":
            bind_mount_rdma_libs(args)
        else:
            log_verbose(
                "Skipping host RDMA lib bind-mount (nic_type={})"
                .format(cfg.nic_type)
            )
    else:
        log_verbose("No InfiniBand devices found at /dev/infiniband")

    if os.path.isfile(cfg.hostfile):
        args += ["-v", "{}:{}:ro".format(cfg.hostfile, cfg.hostfile)]

    args += [
        "-v", "{}:/opt/shared".format(cfg.shared_dir),
        "-v", "{}:/opt/builds".format(cfg.builds_dir),
        "-v", "{}:/opt/ssh-keys:ro".format(cfg.ssh.key_dir),
    ]

    # Mount each resolved post-setup dir at /opt/post-setup.N (read-only).
    # The numeric suffix preserves CLI order so the entrypoint can iterate
    # POST_SETUP_DIRS (set in container_env_pairs) without reparsing.
    for i, host_dir in enumerate(resolve_post_setup_dirs(cfg)):
        args += [
            "-v", "{}:/opt/post-setup.{}:ro".format(host_dir, i)
        ]

    for vol in cfg.extra_volumes:
        args += ["-v", vol]

    return args


def print_ready_summary(cfg):
    # type: (object) -> None
    """Print the post-Ready banner once the entrypoint is confirmed ready.

    Must be called AFTER :func:`wait_for_entrypoint` returns ``True`` --
    otherwise the user is shown shell hints for a container that may
    not yet be (or never become) usable.
    """
    log("")
    log("=== Container '{}' is running ===".format(cfg.container_name))
    log("")
    log(
        "  Shell (root)  : docker exec -it {} bash".format(
            cfg.container_name
        )
    )
    log(
        "  Shell (ubuntu): docker exec -it -u ubuntu {} bash".format(
            cfg.container_name
        )
    )
    log("")
    log("  Verify SSH across nodes:")
    log("    python3 -m mnctl --verify")
    log_verbose("SSH port      : {}".format(cfg.ssh.port))
    log_verbose("Shared directories (same on all nodes):")
    log_verbose("  /opt/shared    <- {}".format(cfg.shared_dir))
    log_verbose("  /opt/builds    <- {}".format(cfg.builds_dir))
    log_verbose("  /opt/ssh-keys  <- {} (read-only)".format(
        cfg.ssh.key_dir
    ))


def launch(cfg, image_exists):
    # type: (object, callable) -> None
    """Idempotent container launch: replace, skip, or start fresh as needed."""
    existing = docker_output(
        ["docker", "ps", "-a", "--format", "{{.Names}}"]
    ).splitlines()

    if cfg.container_name in existing:
        state = docker_output([
            "docker", "inspect", cfg.container_name,
            "--format", "{{.State.Status}}",
        ])

        if cfg.force_rebuild or cfg.force_replace:
            log(
                "=== Replacing container '{}' ({}) ===".format(
                    cfg.container_name,
                    "--rebuild" if cfg.force_rebuild else "--replace",
                )
            )
            remove_container(cfg)
        elif state == "running":
            log(
                "=== Container '{}' is already running "
                "(use --rebuild to replace) ===".format(cfg.container_name)
            )
            log_verbose(
                "Container ID: {}".format(
                    docker_output([
                        "docker", "inspect", cfg.container_name,
                        "--format", "{{.Id}}",
                    ])[:12]
                )
            )
            return
        else:
            log(
                "=== Container '{}' exists but is {}, "
                "removing and re-launching ===".format(
                    cfg.container_name, state
                )
            )
            remove_container(cfg)

    with Timer("Container launch"):
        log("=== Launching container ===")
        log("  Image     : {}".format(cfg.image_tag))
        log("  Container : {}".format(cfg.container_name))
        log("  GPUs      : {}".format(cfg.gpus))
        log_verbose("SSH port  : {}".format(cfg.ssh.port))
        log_verbose("SHM size  : {}".format(cfg.shm_size))
        log("")

        run_args = assemble_run_args(cfg)

        if cfg.verbose:
            log_verbose("docker run arguments:")
            for arg in run_args:
                log_verbose("  {}".format(arg))

        cmd = ["docker", "run"] + run_args + [cfg.image_tag]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log("")
            error(
                "Container launch failed (exit {})".format(e.returncode)
            )
            log("  Troubleshooting:")
            log("    docker image inspect {}".format(cfg.image_tag))
            log("    docker rm -f {}".format(cfg.container_name))
            log("    python3 -m mnctl --run --rebuild")
            raise SystemExit(1)

    # Wait BEFORE printing the summary so the "is running" banner only
    # appears once the entrypoint has actually reached '=== Ready ==='.
    if wait_for_entrypoint(cfg):
        print_ready_summary(cfg)
