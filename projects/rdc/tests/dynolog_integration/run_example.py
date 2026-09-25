#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run dynolog's ``RdcWrapperExample`` and verify it drives RDC end to end.

The example embeds RDC exactly as dynolog does: it constructs ``RdcWrapper``,
discovers GPU entities, watches the RDC field set, then loops forever polling
metrics. This helper launches it with ``ROCM_PATH`` / ``LD_LIBRARY_PATH`` set,
lets it run for a bounded window, terminates it (the poll loop never returns by
design), and inspects the captured output.

The RDC build/link step is the primary integration signal; this runtime check is
a bonus smoke test. Gating:

* Not started (no ``Initializing RDC example`` banner) -> hard fail: the binary
  could not load/run against the freshly-built RDC (a missing shared library or
  an undefined symbol at load time) -- a real runtime-linkage break.
* Fatal setup error (``RDC example failed: ...``): soft-pass only when no GPU is
  present or the message is a recognized capability/field limitation (e.g. an
  unsupported field on this arch). Any other fatal -- ``rdc_init``,
  ``rdc_start_embedded``, GPU group create/add, field-watch, ... -- is a genuine
  RDC regression and hard-fails.
* Self-exit before we stop it: the poll loop is meant to run until terminated,
  so a process that exits on its own (e.g. a crash after a healthy-looking
  sample) hard-fails even if some metrics were printed first.
* Healthy (started, >=1 entity, >=1 metric, still running) -> pass.
* Never reported an entity count on a GPU box -> hard fail: RdcWrapper setup
  neither finished nor threw within the window, which is a hang, not arch
  variance.
* Started, discovered entities, but collected no metric on a GPU box ->
  soft-pass with a warning (usually all enabled fields are N/A for the arch);
  ``--strict`` turns this into a hard failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from ._common import configure_logging, gh_error, gh_warning

logger = logging.getLogger("dynolog.run_example")

_KFD_DEVICE = Path("/dev/kfd")

_STARTED_RE = re.compile(r"Initializing RDC example", re.MULTILINE)
_FATAL_RE = re.compile(r"RDC example failed:\s*(?P<msg>.*)", re.MULTILINE)
_ENTITIES_RE = re.compile(r"Monitoring (?P<n>\d+) entities", re.MULTILINE)
_COLLECTED_RE = re.compile(r"collected (?P<m>\d+) metrics", re.MULTILINE)

# A fatal message matching this is an acceptable capability/arch limitation
# (the build already validated the integration surface), not an RDC regression.
_CAPABILITY_RE = re.compile(r"not[ _]?supported|unsupported", re.IGNORECASE)


def gpu_present() -> bool:
    """True if an AMD GPU compute device (``/dev/kfd``) is available."""
    return _KFD_DEVICE.exists()


def _monitored_entities(log: str) -> int | None:
    matches = _ENTITIES_RE.findall(log)
    return int(matches[-1]) if matches else None


def _max_metrics_collected(log: str) -> int:
    return max((int(m) for m in _COLLECTED_RE.findall(log)), default=0)


def evaluate(
    log: str, *, gpu_available: bool, self_exited: bool, strict: bool = False
) -> tuple[bool, bool, str]:
    """Assess an example run from its captured output.

    Returns ``(ok, soft_pass, reason)``. A pure function of the inputs so it is
    unit-testable without hardware. See the module docstring for the gating
    rationale.
    """
    if not _STARTED_RE.search(log):
        return (
            False,
            False,
            "example binary did not start (no 'Initializing RDC example' banner) "
            "-- possible runtime link/load failure against the built RDC",
        )

    fatal = _FATAL_RE.search(log)
    if fatal is not None:
        msg = fatal.group("msg").strip() or "no detail"
        # Acceptable: no hardware, or a recognized capability/field limitation.
        if not gpu_available or _CAPABILITY_RE.search(msg):
            return (True, True, f"setup did not complete ({msg})")
        # Anything else (init / embedded-start / group / field-watch) is a real
        # RDC runtime regression.
        return (False, False, f"RdcWrapper reported a fatal error: {msg}")

    # The free-running poll loop should only stop when we terminate it; a
    # self-exit (e.g. a crash after a healthy-looking prefix) is unexpected.
    if self_exited:
        return (
            False,
            False,
            "example exited on its own before it was stopped -- unexpected for the "
            "free-running poll loop (possible crash after start)",
        )

    entities = _monitored_entities(log)
    metrics = _max_metrics_collected(log)

    # Healthy run: discovered entities and read at least one metric.
    if entities and metrics > 0:
        return (True, False, f"collected up to {metrics} metrics from {entities} entities")

    if entities is None:
        # Never printed an entity count and never threw: RdcWrapper construction
        # did not return. No arch explains that, so on a GPU box it is a hang.
        reason = (
            "example never reported an entity count -- RdcWrapper setup neither "
            "completed nor failed within the run window (possible hang in "
            "rdc_init / rdc_start_embedded / field-watch)"
        )
        if gpu_available:
            return (False, False, reason)
    elif entities == 0:
        reason = "no GPU entities were discovered (Monitoring 0 entities)"
    else:
        # Usually every enabled field is N/A for this arch.
        reason = "no metrics were collected from any entity"

    if not gpu_available:
        return (True, True, f"no GPU present; runtime validation skipped ({reason})")
    if strict:
        return (False, False, reason)
    return (True, True, reason)


def run(
    *,
    binary: Path,
    duration: float,
    rocm_dir: Path,
    ld_library_path: str | None,
    log_file: Path,
    strict: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    if not binary.is_file():
        gh_error(f"example binary not found: {binary}")
        return 1

    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_dir)
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = ld_library_path

    cmd = [str(binary), *(extra_args or [])]
    logger.info("running example for %.0fs: %s", duration, " ".join(cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    self_exited = False
    with log_file.open("wb") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            # The example loops forever on success, so a return within the window
            # means it stopped on its own (threw, or crashed). TimeoutExpired is
            # the healthy path.
            proc.wait(timeout=duration)
            self_exited = True
            logger.info("example exited on its own with code %s", proc.returncode)
        except subprocess.TimeoutExpired:
            logger.info("example still running after %.0fs -- stopping it", duration)
            _terminate(proc)

    log = log_file.read_text(errors="replace")
    _echo_log(log)

    present = gpu_present()
    ok, soft, reason = evaluate(log, gpu_available=present, self_exited=self_exited, strict=strict)
    ctx = "GPU present" if present else "no GPU detected"
    if not ok:
        gh_error(f"dynolog RdcWrapperExample validation failed ({ctx}): {reason}")
        return 1
    if soft:
        gh_warning(f"dynolog RdcWrapperExample soft-passed ({ctx}): {reason}")
    else:
        logger.info("dynolog RdcWrapperExample OK (%s): %s", ctx, reason)
    return 0


def _terminate(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue


def _echo_log(log: str, lines: int = 40) -> None:
    tail = log.splitlines()[-lines:]
    if tail:
        logger.info("---- RdcWrapperExample output (last %d lines) ----", len(tail))
        for line in tail:
            print(line, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rocm-dir", default=Path("/opt/rocm"), type=Path)
    parser.add_argument("--ld-library-path", default=None)
    parser.add_argument("--log-file", default=Path("/tmp/rdc-dynolog-example.log"), type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Hard-fail when a started example collects no metrics on a GPU box "
        "(default: soft-pass, since that is usually arch field-availability).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run(
        binary=args.binary,
        duration=args.duration,
        rocm_dir=args.rocm_dir,
        ld_library_path=args.ld_library_path,
        log_file=args.log_file,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
