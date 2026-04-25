"""Stop and remove the container on every node in the hostfile."""

from typing import Dict, List

from ..config import Config
from ..utils import (
    get_local_hostnames, host_ssh_cmd, log, parse_hostfile, run_parallel,
)


def stop_all(cfg, runtime):
    # type: (Config, object) -> None
    """Stop and remove containers on every node in the hostfile."""
    log("=== Stopping containers on all nodes ===")
    log("")

    hosts = parse_hostfile(cfg.hostfile)
    local_names = get_local_hostnames()
    stop_cmd = runtime.get_stop_cmd()

    # Spawn all stop commands at once
    jobs = {}  # type: Dict[str, List[str]]
    for host in hosts:
        if host in local_names:
            jobs[host] = ["sh", "-c", stop_cmd]
        else:
            jobs[host] = host_ssh_cmd(cfg, host) + [stop_cmd]
    results = run_parallel(jobs)

    unreachable = []
    for host in hosts:
        r = results[host]
        output = r.stdout_text.strip()
        if not r.ok:
            err = r.stderr_text.strip()
            output = "[UNREACHABLE] {}".format(
                err[:200] if err else "exit {}".format(r.returncode)
            )
            unreachable.append(host)
        log("  {:<20} {}".format(host, output))

    if unreachable:
        log("")
        log("WARNING: {} node(s) could not be reached for cleanup:".format(
            len(unreachable)
        ))
        for h in unreachable:
            log("  - {}".format(h))
        log("")
        log("  Manual cleanup on unreachable nodes:")
        log("    ssh <node> docker rm -f {}".format(cfg.container_name))

    log("")
    log("=== Done ===")
