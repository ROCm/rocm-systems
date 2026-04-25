"""Partial-failure reporting for :func:`mnctl.orchestrate.launch_all`."""

from typing import Dict, List, Tuple

from ..config import Config
from ..utils import log


def report_launch_failures(cfg, hosts, results, failed, succeeded):
    # type: (Config, List[str], Dict[str, Tuple[int, bytes]], List[str], List[str]) -> None
    """Print detailed failure info, partial-deployment summary, and next steps."""
    del cfg  # accepted for caller symmetry; not currently consumed
    log("")
    log("=== Failed nodes ({}/{}) ===".format(len(failed), len(hosts)))
    for host in failed:
        rc, output = results[host]
        log("")
        log("--- {} (exit {}) ---".format(host, rc))
        text = output.decode("utf-8", errors="replace")
        for line in text.splitlines()[-30:]:
            log("  {}".format(line))

    log("")
    log("=== Partial deployment summary ===")
    log("  Succeeded : {}/{} nodes".format(len(succeeded), len(hosts)))
    log("  Failed    : {}/{} nodes".format(len(failed), len(hosts)))
    if len(failed) <= 20:
        log("  Failed on : {}".format(", ".join(failed)))
    log("")
    log("  Next steps:")
    log("    1. Retry (idempotent — skips already-running containers):")
    log("       python3 -m mnctl --launch-all")
    log("    2. Clean up ALL nodes and start fresh:")
    log("       python3 -m mnctl --stop-all")
