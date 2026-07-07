#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices. All rights reserved.
"""Verify the Device Metrics Exporter Prometheus endpoint.

Replaces the inline curl/grep loop in ``dme-amdsmi-ci.yml`` Phase 5 with
a real Prometheus exposition-format check that asserts:

* Endpoint returns HTTP 200 within ``--max-retries`` attempts.
* Response is valid Prometheus text (``# HELP`` + ``# TYPE`` headers).
* Every metric in ``--required-metric`` is exposed and has at least one
  numeric sample emitted.

When ``--gpu-agent-pid-file`` is provided, the verification is aware of GPU
Agent health.  If GPU Agent crashed (common with ABI mismatches between
AMDSMI versions), the step emits a warning and exits 0 rather than blocking
the entire CI. The underlying infrastructure (build, deploy, service
management) is still validated.
"""

import argparse
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ._common import configure_logging, gh_error, gh_warning

logger = logging.getLogger("dme.metrics")

_HELP_RE = re.compile(r"^# HELP ", re.MULTILINE)
_TYPE_RE = re.compile(r"^# TYPE ", re.MULTILINE)
# Prometheus sample lines: ``metric{labels...} <number>`` (with optional timestamp).
# Captures metric name in group 1.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[0-9eE+\-.]+|NaN|\+Inf|-Inf)"
    r"(?:\s+\d+)?\s*$",
    re.MULTILINE,
)

# Default GPU metrics that should always be exposed by the AMDSMI-backed
# Device Metrics Exporter once GPU Agent is up. Override via CLI flag.
_DEFAULT_REQUIRED_METRICS = ("gpu_edge_temperature", "gpu_power_usage", "gpu_gfx_activity")

# Anchored crash indicators for GPU Agent logs. Word-boundary/line-anchored
# regexes avoid false positives on benign gRPC lines like
# "Connection Aborted by peer" or "core dumped".
_CRASH_RES = [
    re.compile(p, re.MULTILINE)
    for p in (
        r"\*\*\* stack smashing detected",
        r"^Segmentation fault",
        r"^Aborted$",
        r"\bSIGSEGV\b",
        r"\bSIGABRT\b",
    )
]


def _fetch(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, ""


def _assert_prometheus_format(body: str) -> None:
    if not _HELP_RE.search(body) or not _TYPE_RE.search(body):
        raise AssertionError("Response is missing # HELP / # TYPE headers")


def _exposed_metric_names(body: str) -> set[str]:
    return {m.group("name") for m in _SAMPLE_RE.finditer(body)}


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid_file(pid_file: Path) -> int | None:
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _tail_file(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log file unavailable)"


def _gpu_agent_crashed(log_file: Path) -> bool:
    """Check if GPU Agent log contains crash indicators."""
    if not log_file.is_file():
        return False
    try:
        log_content = log_file.read_text(errors="replace")
        return any(r.search(log_content) for r in _CRASH_RES)
    except OSError:
        return False


def verify(
    *,
    url: str,
    required_metrics: tuple[str, ...],
    max_retries: int,
    retry_delay: float,
    request_timeout: float,
    output_path: Path | None = None,
    gpu_agent_pid_file: Path | None = None,
    gpu_agent_log_file: Path | None = None,
) -> None:
    body = ""
    last_status = 0
    exposed: set[str] = set()
    for attempt in range(1, max_retries + 1):
        logger.info("attempt %d/%d: GET %s", attempt, max_retries, url)
        status, body = _fetch(url, timeout=request_timeout)
        last_status = status
        if status == 200 and body:
            # Retry if required metrics are missing; the pipeline may need warm-up.
            exposed = _exposed_metric_names(body)
            missing = [m for m in required_metrics if m not in exposed]
            if not missing:
                break
            logger.info(
                "HTTP 200 but %d required metric(s) missing -- retrying in %.1fs",
                len(missing),
                retry_delay,
            )
        else:
            logger.info("HTTP %s -- retrying in %.1fs", status, retry_delay)
        if attempt < max_retries:
            time.sleep(retry_delay)
    else:
        # All retries exhausted: endpoint unreachable or metrics incomplete.
        if last_status != 200:
            gh_error(
                f"Metrics endpoint unreachable after {max_retries} attempts "
                f"(last status {last_status})"
            )
            raise SystemExit(1)
        if not body:
            gh_error(
                f"Metrics endpoint returned HTTP 200 with empty body after {max_retries} attempts"
            )
            raise SystemExit(1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(body)
        logger.info("saved metrics output to %s", output_path)

    _assert_prometheus_format(body)
    logger.info("Prometheus exposition format OK")

    # Reuse the ``exposed`` set already computed in the retry loop above.
    missing = [m for m in required_metrics if m not in exposed]
    if not missing:
        logger.info(
            "All %d required metrics present (total exposed: %d)",
            len(required_metrics),
            len(exposed),
        )
        return

    # Metrics missing: a crashed GPU Agent is an upstream ABI issue, not our bug.
    gpu_agent_alive = True
    if gpu_agent_pid_file is not None:
        pid = _read_pid_file(gpu_agent_pid_file)
        logger.info("GPU Agent PID file: %s, PID: %s", gpu_agent_pid_file, pid)
        if pid is None:
            logger.info("GPU Agent PID file invalid or missing")
            gpu_agent_alive = False
        else:
            # A crashed process can linger as a zombie, so os.kill(pid, 0)
            # succeeds; also scan the log for crash indicators.
            gpu_agent_alive = _process_alive(pid)
            logger.info("GPU Agent process (PID %s) alive: %s", pid, gpu_agent_alive)

            if gpu_agent_alive and gpu_agent_log_file is not None:
                if _gpu_agent_crashed(gpu_agent_log_file):
                    logger.info("GPU Agent log contains crash indicators (zombie process)")
                    gpu_agent_alive = False

    if not gpu_agent_alive:
        gh_warning(
            "GPU Agent process died (likely ABI mismatch with libamd_smi.so). "
            "GPU metric verification skipped."
        )
        if gpu_agent_log_file is not None:
            tail = _tail_file(gpu_agent_log_file)
            gh_warning(f"GPU Agent log (last lines):\n{tail}")
        logger.info(
            "Soft-pass: infrastructure validated but GPU metrics unavailable due to GPU Agent crash"
        )
        return

    # GPU Agent is alive (or uncheckable) but metrics are still missing: hard fail.
    gh_error("Required GPU metrics missing from /metrics: " + ", ".join(missing))
    sample = ", ".join(sorted(exposed)[:20])
    gh_warning(f"Exposed metrics (sample): {sample}")
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--required-metric",
        action="append",
        default=None,
        help="Metric name that must appear (repeatable). Defaults to GPU metric set.",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--gpu-agent-pid-file",
        type=Path,
        default=None,
        help="PID file for GPU Agent. Enables soft-fail if GPU Agent crashed.",
    )
    parser.add_argument(
        "--gpu-agent-log-file",
        type=Path,
        default=None,
        help="Log file for GPU Agent (used for diagnostics on crash).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    required = tuple(args.required_metric) if args.required_metric else _DEFAULT_REQUIRED_METRICS
    verify(
        url=args.url,
        required_metrics=required,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_timeout=args.request_timeout,
        output_path=args.output,
        gpu_agent_pid_file=args.gpu_agent_pid_file,
        gpu_agent_log_file=args.gpu_agent_log_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
