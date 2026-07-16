"""Coverage payload: test_runner.py --coverage-report (Python spike).

Mirrors payloads/coverage.sh. Notice how much of the bash gymnastics collapses:
  * suite-name mapping reads the JSON config directly (no python-in-bash heredoc);
  * the test_runner command is assembled here with filters already resolved, so the
    in-container step is a single shlex-quoted string (no `filters=()` arg-array
    rebuild across the ssh->docker->bash layers).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from typing import List

from .base import Payload


class CoveragePayload(Payload):
    tag = "cov"                              # keeps the historical rccl-cov-<arch> name
    # artifact_env_var inherited from Payload (RESULT_ARTIFACT_DIR) -- standardized so
    # the reusable workflow uploads $RESULT_ARTIFACT_DIR for every workload.

    def __init__(self):
        self.test_config = os.environ.get("TEST_CONFIG", "mi300x_mellanox_ib.json")
        self.test_suite = os.environ.get("TEST_SUITE", "")
        self.test_name = os.environ.get("TEST_NAME", "")
        self.rccl_tests_dir = os.environ.get("RCCL_TESTS_DIR", "")
        self.mpi_hostfile = ""              # resolved in prelaunch/validate

    # rccl-tests source (perf binaries): override wins, else the sibling checkout.
    def validate(self, orch) -> None:
        if not self.rccl_tests_dir:
            cand = os.path.abspath(os.path.join(orch.cfg.rccl_dir, "..", "rccl-tests"))
            self.rccl_tests_dir = cand if os.path.isdir(cand) else ""
        if not (self.rccl_tests_dir and os.path.isdir(self.rccl_tests_dir)):
            sys.exit("ERROR: rccl-tests source not found; set RCCL_TESTS_DIR to a checkout "
                     "(expected sibling projects/rccl-tests).")

    def _resolve_mpi_hostfile(self, orch) -> str:
        return os.environ.get("RCCL_TEST_MPI_HOSTFILE") or orch.cfg.hostfile

    def mounts(self, orch) -> List[str]:
        mpi = self._resolve_mpi_hostfile(orch)
        m = [f"--volume {shlex.quote(self.rccl_tests_dir + ':/work/rccl-tests')}"]
        if mpi != orch.cfg.hostfile:
            m.append(f"--volume {shlex.quote(f'{mpi}:{mpi}:ro')}")
        return m

    def prelaunch(self, orch) -> None:
        self.mpi_hostfile = self._resolve_mpi_hostfile(orch)
        if self.mpi_hostfile != orch.cfg.hostfile and not os.path.isfile(self.mpi_hostfile):
            sys.exit(f"ERROR: RCCL_TEST_MPI_HOSTFILE not found: {self.mpi_hostfile}")
        self.test_suite = self._map_suite(orch)

    # --suite-name globs the DISPLAY name, not the config key. If TEST_SUITE is a
    # config key, translate to the matching display name(s) (':'-joined = OR).
    def _map_suite(self, orch) -> str:
        if not self.test_suite:
            return self.test_suite
        cfg = os.path.join(orch.cfg.rccl_dir, "tools", "scripts", "test_runner",
                          "configs", self.test_config)
        if not os.path.isfile(cfg):
            return self.test_suite
        with open(cfg) as f:
            data = json.load(f)
        names = [s["name"] for s in data.get("test_suites", [])
                 if s.get("config") == self.test_suite and "name" in s]
        if names:
            mapped = ":".join(names)
            orch and print(f">>> Suite filter '{self.test_suite}' (config key) -> "
                          f"display name(s) '{mapped}'", flush=True)
            return mapped
        return self.test_suite

    # Wipe build/ only when its baked CMAKE_HOME_DIRECTORY != the container mount.
    def prepare(self, orch) -> None:
        print(">>> Checking for stale CMake cache inside the container ...", flush=True)
        script = r'''set -euo pipefail
for src in /work/rccl /work/rccl-tests; do
  [ -d "$src/build" ] || continue
  stale=""
  while IFS= read -r cache; do
    home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -n1)"
    if [ -n "$home" ] && [ "$home" != "$src" ]; then stale="$home"; break; fi
  done < <(find "$src/build" -name CMakeCache.txt 2>/dev/null)
  if [ -n "$stale" ]; then
    echo "[cleanup] WARNING: stale CMake cache in $src/build (configured for '$stale', expected '$src'); removing."
    rm -rf "$src/build"
  else
    echo "[cleanup] $src/build is compatible; keeping it (incremental build)."
  fi
done'''
        orch.ssh_head(f"docker exec -i {shlex.quote(orch.cfg.container)} bash -s <<'REMOTE'\n{script}\nREMOTE")

    def run(self, orch) -> None:
        c = orch.cfg
        print(">>> Running test_runner.py --coverage-report ...", flush=True)
        filters = ""
        if self.test_suite:
            filters += f" --suite-name {shlex.quote(self.test_suite)}"
        if self.test_name:
            filters += f" --test-name {shlex.quote(self.test_name)}"
        inner = (
            "cd /work/rccl && "
            "ROCM_PATH=/opt/rocm MPI_PATH=/opt/shared/ompi RCCL_TESTS_DIR=/work/rccl-tests "
            f"RCCL_TEST_MPI_HOSTFILE={shlex.quote(self.mpi_hostfile)} "
            "python3 tools/scripts/test_runner/test_runner.py "
            f"--config tools/scripts/test_runner/configs/{shlex.quote(self.test_config)}"
            f"{filters} --report-suffix {shlex.quote(c.run_id)} "
            "--coverage-report --verbose --emit-results"
        )
        docker = (
            f"docker exec -e RCCL_ARTIFACTS_DIR_FILE={shlex.quote(c.artifacts_pointer_ctr)} "
            f"{shlex.quote(c.container)} bash -lc {shlex.quote(inner)}"
        )
        orch.ssh_head(docker)
