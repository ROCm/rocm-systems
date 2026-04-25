"""Launch the configured container on every node in parallel.

Per-node flow (issued in parallel via :func:`run_parallel_streaming`):

    python3 -u <script_dir>/run_mnctl.py --setup-deps ... <image>
        && python3 -u <script_dir>/run_mnctl.py --run        ... <image>

The output of every node is streamed line-by-line to the console with a
``[host]`` prefix; serialized through ``print_lock`` to avoid interleaving.
"""

import os
import shlex
import sys
import threading
from typing import Dict, List

from ..config import Config
from ..utils import (
    Timer, get_local_hostnames, host_ssh_cmd, log, log_verbose,
    parse_hostfile, run_parallel_streaming,
)
from .distribute import distribute_files
from .failures import report_launch_failures
from .forward import build_forward_args


def launch_all(cfg, runtime):
    # type: (Config, object) -> None
    """Build + launch a container on every node in the hostfile.

    *runtime* is currently unused inside ``launch_all`` (each remote node
    re-instantiates its own runtime via ``run_mnctl.py``) but is part of
    the public signature so the dispatch layer treats every multi-node
    entry point uniformly and so any future inline runtime probing can
    use it without re-plumbing.

    1. Distribute SSH keys, tool code, hostfile, and post-setup to remotes
    2. Spawn deps-build + container-launch on every node concurrently
    3. Stream output line-by-line from each node as it arrives
    """
    del runtime  # currently unused; see docstring
    with Timer("Launch all nodes"):
        log("=== Launching containers on all nodes ===")
        log("")

        hosts = parse_hostfile(cfg.hostfile)
        local_names = get_local_hostnames()
        script = os.path.join(cfg.script_dir, "run_mnctl.py")

        log("  Hostfile  : {} ({} nodes)".format(cfg.hostfile, len(hosts)))
        log("  Image     : {}".format(cfg.image_tag))
        log("  Container : {}".format(cfg.container_name))
        log_verbose("Script    : {}".format(script))
        log("")

        # Distribute files to remote hosts (handles non-shared FS)
        remote_hosts = [h for h in hosts if h not in local_names]
        distribute_files(cfg, remote_hosts)

        # Build compound command: setup-deps (idempotent) then run.
        # python3 -u disables output buffering so lines stream in real time.
        # --setup-deps gets --rebuild (image build happens here).
        # --run reuses the image that --setup-deps just built, so we
        # downgrade any --rebuild request to a container --replace to
        # avoid a redundant full image rebuild on every node.
        deps_args = build_forward_args(cfg, action="--setup-deps")
        run_args = build_forward_args(
            cfg, action="--run",
            force_rebuild=False,
            force_replace=cfg.force_rebuild or cfg.force_replace,
        )

        def _quote_cmd(args):
            # type: (List[str]) -> str
            return " ".join(shlex.quote(a) for a in args)

        deps_cmd = "python3 -u {} {}".format(
            shlex.quote(script), _quote_cmd(deps_args),
        )
        run_cmd = "python3 -u {} {}".format(
            shlex.quote(script), _quote_cmd(run_args),
        )
        compound = "{} && {}".format(deps_cmd, run_cmd)
        log_verbose("Per-node command: {}".format(compound))

        # Build per-host argv mapping (local hosts run via bash -c; remote
        # hosts via SSH).  run_parallel_streaming spawns + reaps + streams.
        jobs = {}  # type: Dict[str, List[str]]
        for host in hosts:
            if host in local_names:
                jobs[host] = ["bash", "-c", compound]
            else:
                jobs[host] = host_ssh_cmd(cfg, host) + [compound]

        label_len = max(len(h) for h in hosts)
        print_lock = threading.Lock()
        completed = {"n": 0}

        def _on_line(host, line):
            # type: (str, str) -> None
            with print_lock:
                log("  [{}] {}".format(host.ljust(label_len), line))

        results = run_parallel_streaming(jobs, on_line=_on_line)

        # Render terminal status per host (preserves prior behavior of
        # showing [OK]/[FAIL] as each one finishes -- here we summarize
        # after the fact since streaming already showed live progress).
        for host in hosts:
            r = results.get(host)
            rc = r.returncode if r is not None else 1
            completed["n"] += 1
            status = "[OK]  " if rc == 0 else "[FAIL]"
            log("  {} {:<40} ({}/{})".format(
                status, host, completed["n"], len(hosts),
            ))

        failed = [h for h in hosts if results[h].returncode != 0]
        succeeded = [h for h in hosts if results[h].returncode == 0]

        if failed:
            # Adapt streaming results into the (rc, bytes) shape that
            # report_launch_failures consumes.
            shaped = {h: (results[h].returncode, results[h].stdout)
                      for h in hosts}
            report_launch_failures(cfg, hosts, shaped, failed, succeeded)
            sys.exit(1)

    log("")
    log("=== All {} containers launched ===".format(len(hosts)))
    log("")
    log("  Verify container SSH:")
    log("    python3 -m mnctl --verify")
