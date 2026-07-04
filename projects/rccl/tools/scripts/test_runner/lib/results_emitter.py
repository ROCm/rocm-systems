#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Results Emitter

Structured, machine-readable result emission for the RCCL test runner. Feeds the
results dashboard.

Two independent sinks, both driven off the same in-memory records so they never
diverge:

  1. Local files (always, when --emit-results is set) -- the durable source of
     truth. A per-run directory of JSON/JSONL plus a ``<run_id>.tar.gz`` snapshot
     and a stable ``latest.tar.gz`` that an hourly ``cron`` + ``scp`` job on the
     dashboard host scrapes.

  2. PostgreSQL (optional, when --db-push is set) -- best effort. If the connect
     or write fails or times out we log a warning and carry on; the local tarball
     is still written, so the dashboard-side ingest job can pick the run up on the
     next scrape. The DB push is never allowed to fail the test run.

The DSN is read from the ``RCCL_RESULTS_DSN`` environment variable (libpq URL or
key/value string). It is never hardcoded and never written to the emitted files.
"""

import datetime
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tarfile

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EMITTER_VERSION = "1.0.0"

# rccl-tests binary name -> logical collective name.
_COLLECTIVE_FROM_BINARY = {
    "all_reduce_perf": "all_reduce",
    "all_gather_perf": "all_gather",
    "alltoall_perf": "alltoall",
    "alltoallv_perf": "alltoallv",
    "broadcast_perf": "broadcast",
    "gather_perf": "gather",
    "scatter_perf": "scatter",
    "reduce_perf": "reduce",
    "reduce_scatter_perf": "reduce_scatter",
    "sendrecv_perf": "sendrecv",
    "hypercube_perf": "hypercube",
}


def collective_from_binary(binary):
    """Map an rccl-tests perf binary name to a logical collective name."""
    if not binary:
        return None
    base = os.path.basename(str(binary))
    if base in _COLLECTIVE_FROM_BINARY:
        return _COLLECTIVE_FROM_BINARY[base]
    # Fall back to trimming a trailing _perf.
    return base[:-5] if base.endswith("_perf") else base


def _to_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def parse_perf_output(text):
    """
    Parse rccl-tests perf table output into structured rows.

    rccl-tests prints two performance triplets per size -- out-of-place then
    in-place -- each ``time algbw busbw #wrong``. Leading columns are
    ``size count type`` and, for reduction collectives, ``redop root``. We anchor
    on the trailing 8 numeric columns so the parser works for both reduction and
    non-reduction collectives regardless of how many label columns precede them.

    Returns (rows, avg_busbw) where rows is a list of per-(size, place) dicts and
    avg_busbw is the run's "Avg bus bandwidth" summary line if present.
    """
    rows = []
    avg_busbw = None
    if not text:
        return rows, avg_busbw

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"Avg bus bandwidth\s*:\s*([0-9.]+)", line)
            if m:
                avg_busbw = _to_float(m.group(1))
            continue

        toks = line.split()
        # Need at least: size count type + 8 trailing perf columns.
        if len(toks) < 11:
            continue
        if not toks[0].isdigit() or not toks[1].isdigit():
            continue

        tail = toks[-8:]
        oop_time, oop_alg, oop_bus = _to_float(tail[0]), _to_float(tail[1]), _to_float(tail[2])
        ip_time, ip_alg, ip_bus = _to_float(tail[4]), _to_float(tail[5]), _to_float(tail[6])
        # A real data row has numeric perf columns; header/units lines do not.
        if None in (oop_time, oop_alg, oop_bus, ip_time, ip_alg, ip_bus):
            continue

        size = int(toks[0])
        count = int(toks[1])
        dtype = toks[2]
        mid = toks[3:len(toks) - 8]  # 0..2 of {redop, root}
        redop = mid[0] if len(mid) >= 1 else None
        root = mid[1] if len(mid) >= 2 else None

        rows.append({
            "size_bytes": size, "count_elements": count, "dtype": dtype,
            "redop": redop, "root": root, "place": "out_of_place",
            "time_us": oop_time, "algbw_gbs": oop_alg, "busbw_gbs": oop_bus,
            "wrong": tail[3],
        })
        rows.append({
            "size_bytes": size, "count_elements": count, "dtype": dtype,
            "redop": redop, "root": root, "place": "in_place",
            "time_us": ip_time, "algbw_gbs": ip_alg, "busbw_gbs": ip_bus,
            "wrong": tail[7],
        })

    return rows, avg_busbw


def parse_coverage_report(report_txt_path):
    """
    Parse the ``TOTAL`` line of an llvm-cov ``report`` text file into coverage
    percentages. llvm-cov emits, in order, region / function / line / branch
    coverage, each with a trailing ``NN.NN%`` column. Returns None if the file is
    absent or has no TOTAL line.
    """
    if not report_txt_path or not os.path.isfile(report_txt_path):
        return None
    try:
        with open(report_txt_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    total_line = next((ln for ln in lines if ln.strip().startswith("TOTAL")), None)
    if not total_line:
        return None

    pcts = [float(p) for p in re.findall(r"([0-9]+\.[0-9]+)%", total_line)]
    # Region, Function, Line, Branch (branch may be absent in older llvm-cov).
    keys = ["regions_pct", "functions_pct", "lines_pct", "branches_pct"]
    cov = {k: (pcts[i] if i < len(pcts) else None) for i, k in enumerate(keys)}
    cov["raw"] = {"total_line": total_line.strip(), "percents": pcts}
    return cov


def git_info(start_dir):
    """Return (sha, branch) for the git repo containing start_dir, or (None, None).

    Normal pipeline runs build a named branch, so branch is e.g. 'develop'. When
    the checkout is on a raw commit/ref, git reports the branch as 'HEAD'; record
    it explicitly as 'detached (<short-sha>)' so it reads as detached-HEAD state
    rather than a branch literally named 'HEAD'."""
    def _git(*args):
        try:
            out = subprocess.run(
                ["git", "-C", start_dir, *args],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        branch = f"detached ({sha[:12]})" if sha else "detached HEAD"
    return sha, branch


def gpu_arch():
    """Best-effort GPU arch (e.g. gfx942). Tries rocm_agent_enumerator then rocminfo."""
    try:
        out = subprocess.run(["rocm_agent_enumerator"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            for tok in out.stdout.split():
                if tok.startswith("gfx") and tok != "gfx000":
                    return tok
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            m = re.search(r"\bgfx[0-9a-f]+\b", out.stdout)
            if m:
                return m.group(0)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def node_names_from_mpi_hosts(mpi_hosts):
    """Distinct compute node hostnames a job used, from SLURM host_list or a hostfile."""
    if not mpi_hosts:
        return []
    seen = []
    if "host_list" in mpi_hosts:
        for part in str(mpi_hosts["host_list"]).split(","):
            host = part.strip().split(":")[0].strip()
            if host and host not in seen:
                seen.append(host)
    elif "hostfile" in mpi_hosts:
        try:
            with open(mpi_hosts["hostfile"], encoding="utf-8", errors="replace") as hf:
                for line in hf:
                    line = line.split("#")[0].strip()
                    if not line:
                        continue
                    host = line.split()[0].strip()
                    if host and host not in seen:
                        seen.append(host)
        except OSError:
            pass
    return seen


def _rocm_version(rocm_path):
    for rel in (".info/version", ".info/version-dev"):
        p = os.path.join(rocm_path, rel)
        try:
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    return f.read().strip()
        except OSError:
            pass
    return None


def _atomic_write(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class ResultsEmitter:
    """Collects run + test + perf + coverage records and emits them locally and
    (optionally) to PostgreSQL."""

    def __init__(self, run_manifest, results_dir):
        self.manifest = dict(run_manifest)
        self.manifest.setdefault("schema_version", SCHEMA_VERSION)
        self.manifest.setdefault("emitter_version", EMITTER_VERSION)
        self.results_dir = results_dir
        self.run_id = self.manifest["run_id"]
        self.tests = []   # list of test_result dicts (+ nested "perf_rows")

    def add_test(self, record):
        """Attach one test result. If record has a readable ``log_file`` and looks
        like a perf (non-gtest) test, parse perf rows out of its captured log."""
        rec = dict(record)
        perf_rows = []
        avg_busbw = None
        log_file = rec.get("log_file")
        if log_file and os.path.isfile(log_file) and not rec.get("is_gtest", True):
            try:
                with open(log_file, encoding="utf-8", errors="replace") as f:
                    perf_rows, avg_busbw = parse_perf_output(f.read())
            except OSError as e:
                log.warning("results_emitter: could not read log %s: %s", log_file, e)
        collective = collective_from_binary(rec.get("binary"))
        for row in perf_rows:
            row["collective"] = collective
            row["avg_busbw_gbs"] = avg_busbw
        rec["perf_rows"] = perf_rows
        # Store log path relative to the run dir for portability in the tarball.
        if log_file:
            rec["log_relpath"] = os.path.join("logs", os.path.basename(log_file))
        self.tests.append(rec)

    def set_coverage(self, coverage):
        if coverage:
            self.manifest["coverage"] = coverage

    def finalize_summary(self, summary):
        self.manifest["summary"] = summary
        self.manifest.setdefault(
            "finished_at", datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    # -- local emission (durable) ------------------------------------------
    def write_local(self, log_dir=None, report_dir=None):
        """Write the per-run directory, roll a ``<run_id>.tar.gz`` snapshot,
        refresh ``latest.tar.gz``/``latest.json`` and append to ``index.jsonl``.
        Returns the tarball path (or None on failure)."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            run_dir = os.path.join(self.results_dir, self.run_id)
            os.makedirs(run_dir, exist_ok=True)

            _atomic_write(os.path.join(run_dir, "run.json"),
                          json.dumps(self.manifest, indent=2, default=str))

            with open(os.path.join(run_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
                for t in self.tests:
                    slim = {k: v for k, v in t.items() if k != "perf_rows"}
                    f.write(json.dumps(slim, default=str) + "\n")

            with open(os.path.join(run_dir, "perf.jsonl"), "w", encoding="utf-8") as f:
                for t in self.tests:
                    for row in t.get("perf_rows", []):
                        out = dict(row)
                        out["suite"] = t.get("suite")
                        out["test_name"] = t.get("test_name")
                        f.write(json.dumps(out, default=str) + "\n")

            if self.manifest.get("coverage"):
                _atomic_write(os.path.join(run_dir, "coverage.json"),
                              json.dumps(self.manifest["coverage"], indent=2, default=str))

            # Bundle captured logs + coverage report into the run dir so the
            # tarball is self-contained for the dashboard scrape.
            if log_dir and os.path.isdir(log_dir):
                dst = os.path.join(run_dir, "logs")
                shutil.copytree(log_dir, dst, dirs_exist_ok=True)
            if report_dir and os.path.isdir(report_dir):
                dst = os.path.join(run_dir, "report")
                shutil.copytree(report_dir, dst, dirs_exist_ok=True)

            tarball = os.path.join(self.results_dir, f"{self.run_id}.tar.gz")
            tmp_tar = f"{tarball}.tmp"
            with tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(run_dir, arcname=self.run_id)
            os.replace(tmp_tar, tarball)

            # Stable scrape targets. Write latest.tar.gz atomically (tmp + replace)
            # so a concurrent scraper never observes a partially written gzip.
            latest_tar = os.path.join(self.results_dir, "latest.tar.gz")
            tmp_latest = f"{latest_tar}.tmp"
            shutil.copyfile(tarball, tmp_latest)
            os.replace(tmp_latest, latest_tar)
            _atomic_write(os.path.join(self.results_dir, "latest.json"),
                          json.dumps(self.manifest, indent=2, default=str))

            index_line = json.dumps({
                "run_id": self.run_id,
                "created_at": self.manifest.get("created_at"),
                "rccl_sha": self.manifest.get("rccl_sha"),
                "tarball": os.path.basename(tarball),
                "summary": self.manifest.get("summary"),
            }, default=str)
            with open(os.path.join(self.results_dir, "index.jsonl"), "a", encoding="utf-8") as f:
                f.write(index_line + "\n")

            log.info("results_emitter: wrote local results + tarball: %s", tarball)
            print(f"Results written: {tarball}")
            return tarball
        except Exception as e:  # never let emission break the run
            log.warning("results_emitter: local write failed: %s", e)
            print(f"WARNING: results emission (local) failed: {e}")
            return None

    # -- optional PostgreSQL push (best effort) ----------------------------
    def push_postgres(self, dsn=None, timeout=10):
        """Push records to PostgreSQL. Best effort: any failure/timeout is logged
        and swallowed so the test run's exit status is unaffected. Idempotent on
        run_id."""
        dsn = dsn or os.environ.get("RCCL_RESULTS_DSN")
        if not dsn:
            print("WARNING: --db-push set but RCCL_RESULTS_DSN is empty; skipping DB push")
            return False

        connect, driver = _load_pg_driver()
        if connect is None:
            print("WARNING: no PostgreSQL driver (psycopg / psycopg2) available; "
                  "skipping DB push. Local tarball still written.")
            return False

        conn = None
        try:
            conn = connect(dsn, timeout)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {int(timeout) * 1000}")
                    self._upsert(cur)
            print("Results pushed to PostgreSQL")
            return True
        except Exception as e:  # graceful: local tarball is the source of truth
            log.warning("results_emitter: DB push failed (%s): %s", driver, e)
            print(f"WARNING: DB push failed ({driver}): {e}. "
                  "Local tarball retained for the dashboard scrape.")
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _upsert(self, cur):
        m = self.manifest
        cur.execute(
            """
            INSERT INTO runs (run_id, schema_version, created_at, started_at,
                finished_at, rccl_sha, rccl_branch, rocm_version, host, hosts,
                gpu_arch, node_names, num_nodes, gpus_per_node, config_name,
                config_description, runner_version, label, tags, env, summary, metadata)
            VALUES (%(run_id)s, %(schema_version)s, %(created_at)s, %(started_at)s,
                %(finished_at)s, %(rccl_sha)s, %(rccl_branch)s, %(rocm_version)s,
                %(host)s, %(hosts)s, %(gpu_arch)s, %(node_names)s, %(num_nodes)s,
                %(gpus_per_node)s, %(config_name)s, %(config_description)s,
                %(runner_version)s, %(label)s, %(tags)s, %(env)s, %(summary)s, %(metadata)s)
            ON CONFLICT (run_id) DO UPDATE SET
                finished_at = EXCLUDED.finished_at,
                summary     = EXCLUDED.summary,
                ingested_at = now()
                -- NOTE: tags intentionally NOT updated on conflict; once a run is
                -- in the DB, tags are mutable only via the admin dashboard.
            """,
            {
                "run_id": m["run_id"],
                "schema_version": m.get("schema_version", SCHEMA_VERSION),
                "created_at": m.get("created_at"),
                "started_at": m.get("started_at"),
                "finished_at": m.get("finished_at"),
                "rccl_sha": m.get("rccl_sha"),
                "rccl_branch": m.get("rccl_branch"),
                "rocm_version": m.get("rocm_version"),
                "host": m.get("host"),
                "hosts": json.dumps(m.get("hosts")) if m.get("hosts") is not None else None,
                "gpu_arch": m.get("gpu_arch"),
                "node_names": json.dumps(m.get("node_names")) if m.get("node_names") is not None else None,
                "num_nodes": m.get("num_nodes"),
                "gpus_per_node": m.get("gpus_per_node"),
                "config_name": m.get("config_name"),
                "config_description": m.get("config_description"),
                "runner_version": m.get("emitter_version"),
                "label": m.get("label"),
                "tags": m.get("tags"),  # list -> TEXT[]; None -> NULL
                "env": json.dumps(m.get("env")) if m.get("env") is not None else None,
                "summary": json.dumps(m.get("summary")) if m.get("summary") is not None else None,
                "metadata": json.dumps(m.get("metadata")) if m.get("metadata") is not None else None,
            },
        )

        for t in self.tests:
            cur.execute(
                """
                INSERT INTO test_results (run_id, suite, test_name, test_binary,
                    is_gtest, num_nodes, gpus_per_node, num_ranks, dtype, exec_mode,
                    nthreads, status, duration_s, exit_code, error, log_relpath)
                VALUES (%(run_id)s, %(suite)s, %(test_name)s, %(test_binary)s,
                    %(is_gtest)s, %(num_nodes)s, %(gpus_per_node)s, %(num_ranks)s,
                    %(dtype)s, %(exec_mode)s, %(nthreads)s, %(status)s, %(duration_s)s,
                    %(exit_code)s, %(error)s, %(log_relpath)s)
                ON CONFLICT (run_id, suite, test_name) DO UPDATE SET
                    status = EXCLUDED.status, duration_s = EXCLUDED.duration_s,
                    exit_code = EXCLUDED.exit_code, error = EXCLUDED.error
                RETURNING id
                """,
                {
                    "run_id": m["run_id"],
                    "suite": t.get("suite"),
                    "test_name": t.get("test_name"),
                    "test_binary": t.get("binary"),
                    "is_gtest": t.get("is_gtest"),
                    "num_nodes": t.get("num_nodes"),
                    "gpus_per_node": t.get("num_gpus"),
                    "num_ranks": t.get("num_ranks"),
                    "dtype": t.get("dtype"),
                    "exec_mode": t.get("exec_mode"),
                    "nthreads": t.get("nthreads"),
                    "status": t.get("status") or t.get("result"),
                    "duration_s": t.get("duration_s") or t.get("duration"),
                    "exit_code": t.get("exit_code"),
                    "error": t.get("error"),
                    "log_relpath": t.get("log_relpath"),
                },
            )
            test_result_id = cur.fetchone()[0]

            # Re-ingest safety: clear any prior perf rows for this test result.
            cur.execute("DELETE FROM perf_rows WHERE test_result_id = %s", (test_result_id,))
            for row in t.get("perf_rows", []):
                cur.execute(
                    """
                    INSERT INTO perf_rows (run_id, test_result_id, collective,
                        size_bytes, count_elements, dtype, redop, root, place,
                        time_us, algbw_gbs, busbw_gbs, wrong, avg_busbw_gbs)
                    VALUES (%(run_id)s, %(trid)s, %(collective)s, %(size_bytes)s,
                        %(count_elements)s, %(dtype)s, %(redop)s, %(root)s,
                        %(place)s, %(time_us)s, %(algbw_gbs)s, %(busbw_gbs)s,
                        %(wrong)s, %(avg_busbw_gbs)s)
                    """,
                    {"run_id": m["run_id"], "trid": test_result_id, **row},
                )

        cov = m.get("coverage")
        if cov:
            cur.execute(
                """
                INSERT INTO coverage (run_id, regions_pct, functions_pct,
                    lines_pct, branches_pct, raw)
                VALUES (%(run_id)s, %(regions_pct)s, %(functions_pct)s,
                    %(lines_pct)s, %(branches_pct)s, %(raw)s)
                ON CONFLICT (run_id) DO UPDATE SET
                    regions_pct = EXCLUDED.regions_pct,
                    functions_pct = EXCLUDED.functions_pct,
                    lines_pct = EXCLUDED.lines_pct,
                    branches_pct = EXCLUDED.branches_pct,
                    raw = EXCLUDED.raw
                """,
                {
                    "run_id": m["run_id"],
                    "regions_pct": cov.get("regions_pct"),
                    "functions_pct": cov.get("functions_pct"),
                    "lines_pct": cov.get("lines_pct"),
                    "branches_pct": cov.get("branches_pct"),
                    "raw": json.dumps(cov.get("raw")) if cov.get("raw") is not None else None,
                },
            )


def _load_pg_driver():
    """Return (connect_fn, driver_name). connect_fn(dsn, timeout) -> connection.
    Tries psycopg (v3) then psycopg2. Returns (None, None) if neither is present."""
    try:
        import psycopg  # type: ignore

        def _connect(dsn, timeout):
            return psycopg.connect(dsn, connect_timeout=int(timeout))

        return _connect, "psycopg"
    except ImportError:
        pass
    try:
        import psycopg2  # type: ignore

        def _connect(dsn, timeout):
            return psycopg2.connect(dsn, connect_timeout=int(timeout))

        return _connect, "psycopg2"
    except ImportError:
        pass
    return None, None
