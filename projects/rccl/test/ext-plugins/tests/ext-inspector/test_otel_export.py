# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Tests for the Inspector's OTEL metric export."""

import glob
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# The aggregated path (inspectorOtelAddCollBucket) emits these two; the verbose
# per-collective path (inspectorOtelAddCompletedColl) adds algobw.
OTEL_AGGREGATED_METRICS = (
    "nccl_bus_bandwidth_gbs",
    "nccl_collective_exec_time_microseconds",
)
OTEL_VERBOSE_ONLY_METRIC = "nccl_collective_algobw_gbs"


class _OtlpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.received.append((self.path, self.headers.get("Content-Type"), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


class _OtlpCollector:
    """Minimal OTLP/HTTP endpoint that records what the exporter sends."""

    def __enter__(self):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _OtlpHandler)
        self._server.received = []
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def received(self):
        return self._server.received


def _run_all_reduce(paths, dump_dir, extra_env, log_name):
    """Run a short 8-rank all_reduce with the inspector enabled."""
    perf_bin = os.path.join(paths.RCCL_TESTS_DIR, "build", "all_reduce_perf")
    if not os.path.exists(perf_bin):
        pytest.skip(f"rccl-tests all_reduce_perf not found at: {perf_bin}")

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        # DDA claims AllReduce and is not profiler-traced.
        "RCCL_DDA_ENABLE": "0",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
    })
    for name in (
        "NCCL_INSPECTOR_OTEL_EXPORT",
        "NCCL_INSPECTOR_OTEL_VERBOSE",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    ):
        env.pop(name, None)
    env.update(extra_env)

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        perf_bin,
        "-b", "1M",
        "-e", "4M",
        "-f", "2",
        "-g", "1",
        "-n", "2",
        "-w", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "inspector_otel_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)
    with open(log_file, "w") as logfile:
        try:
            result = subprocess.run(
                args, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                universal_newlines=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Inspector OTEL run timed out after 300s, see {log_file}")

    with open(log_file) as logfile:
        return result.returncode, logfile.read(), log_file


def _decode_bodies(received):
    """Validate each payload is OTLP/JSON and return them joined for name checks."""
    bodies = []
    for _, content_type, body in received:
        assert content_type == "application/json", \
            f"OTLP payload should be JSON, got {content_type!r}"
        text = body.decode("utf-8", "replace")
        json.loads(text)
        bodies.append(text)
    return "\n".join(bodies)


@pytest.mark.ext_inspector
@pytest.mark.allreduce
def test_otel_export_posts_metrics(paths):
    """With OTEL export on, the inspector must POST OTLP/JSON metrics to the endpoint."""

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "otel_export", "enabled")
    os.makedirs(dump_dir, exist_ok=True)

    with _OtlpCollector() as collector:
        endpoint = f"http://127.0.0.1:{collector.port}"
        rc, log, log_file = _run_all_reduce(
            paths,
            dump_dir,
            {
                "NCCL_INSPECTOR_OTEL_EXPORT": "1",
                "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
                "OTEL_SERVICE_NAME": "rccl-ext-plugins-test",
            },
            "otel_export_enabled.log",
        )
        received = list(collector.received)

    assert rc == 0, f"AllReduce with OTEL export failed, see {log_file}"
    assert received, (
        f"Inspector sent no OTLP request to {endpoint}; see {log_file}"
    )

    paths_seen = {path for path, _, _ in received}
    assert any(p.endswith("/v1/metrics") for p in paths_seen), \
        f"Expected a POST to /v1/metrics, got {paths_seen}"

    joined = _decode_bodies(received)
    missing = [name for name in OTEL_AGGREGATED_METRICS if name not in joined]
    assert not missing, (
        f"OTLP payloads are missing metrics {missing}; saw {len(received)} request(s). See {log_file}"
    )


@pytest.mark.ext_inspector
@pytest.mark.allreduce
def test_otel_verbose_export_adds_per_collective_metric(paths):
    """Verbose export switches to the per-collective path, which adds algobw."""

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "otel_export", "verbose")
    os.makedirs(dump_dir, exist_ok=True)

    with _OtlpCollector() as collector:
        rc, log, log_file = _run_all_reduce(
            paths,
            dump_dir,
            {
                "NCCL_INSPECTOR_OTEL_EXPORT": "1",
                "NCCL_INSPECTOR_OTEL_VERBOSE": "1",
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{collector.port}",
            },
            "otel_export_verbose.log",
        )
        received = list(collector.received)

    assert rc == 0, f"AllReduce with verbose OTEL export failed, see {log_file}"
    assert received, f"Inspector sent no OTLP request, see {log_file}"

    joined = _decode_bodies(received)
    assert OTEL_VERBOSE_ONLY_METRIC in joined, (
        f"Verbose export should include {OTEL_VERBOSE_ONLY_METRIC}; "
        f"saw {len(received)} request(s). See {log_file}"
    )


@pytest.mark.ext_inspector
@pytest.mark.allreduce
def test_otel_export_off_by_default(paths):
    """Without NCCL_INSPECTOR_OTEL_EXPORT the inspector must not contact the endpoint."""

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "otel_export", "disabled")
    os.makedirs(dump_dir, exist_ok=True)
    for path in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(path)

    with _OtlpCollector() as collector:
        rc, log, log_file = _run_all_reduce(
            paths,
            dump_dir,
            {"OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{collector.port}"},
            "otel_export_disabled.log",
        )
        received = list(collector.received)

    assert rc == 0, f"AllReduce without OTEL export failed, see {log_file}"
    assert not received, (
        f"Inspector exported metrics with OTEL export disabled: {len(received)} request(s). See {log_file}"
    )
    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert dump_files, (
        f"Inspector produced neither OTLP requests nor JSON dumps; see {log_file}"
    )
