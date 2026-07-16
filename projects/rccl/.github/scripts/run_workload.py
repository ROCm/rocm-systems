#!/usr/bin/env python3
"""Generic multi-node workload entrypoint (Python spike).

Selects a payload by --workload / $WORKLOAD, builds a RunConfig from the
environment, and runs the Orchestrator flow. Parallel to run_workload.sh; kept
under py/ so it can be compared without disturbing the working bash pipeline.

    WORKLOAD=coverage python3 run_workload.py
    python3 run_workload.py --workload coverage

Add a workload: implement payloads/<name>.py (subclass Payload) and register it
in payloads/__init__.py:REGISTRY.
"""

from __future__ import annotations

import argparse
import os
import sys

from orchestrator import Orchestrator, RunConfig
from payloads import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-node GPU workload runner (spike).")
    ap.add_argument("--workload", default=os.environ.get("WORKLOAD", ""),
                    help="workload plugin name (or set $WORKLOAD)")
    args = ap.parse_args()

    if not args.workload:
        sys.exit(f"ERROR: set --workload / $WORKLOAD (available: {', '.join(sorted(REGISTRY))})")
    if args.workload not in REGISTRY:
        sys.exit(f"ERROR: unknown workload '{args.workload}' "
                 f"(available: {', '.join(sorted(REGISTRY))})")

    payload = REGISTRY[args.workload]()
    cfg = RunConfig(workload=args.workload, artifact_env_var=payload.artifact_env_var)
    cfg.resolve(entrypoint=os.path.abspath(__file__), workload_tag=payload.tag)
    Orchestrator(cfg, payload, entrypoint=os.path.abspath(__file__)).run()


if __name__ == "__main__":
    main()
