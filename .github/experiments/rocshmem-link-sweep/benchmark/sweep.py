#!/usr/bin/env python3
"""Fixed-input Linux rocSHMEM link-width experiment. Never builds prerequisites.

RSS is sampled and summed across processes: shared pages are double-counted,
short-lived processes can be missed, and this is not unique or exact peak memory.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import time

WIDTHS = (1, 2, 4, 8, 16, 31)
ARCHITECTURES = (
    "gfx900", "gfx90c", "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1033",
    "gfx1034", "gfx1035", "gfx1036", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1170", "gfx1171", "gfx1200",
    "gfx1201", "gfx1250-strict", "gfx1250",
)
TARGETS = {"info": ("tools/rocshmem_info",),
           "functional": ("tests/functional_tests/rocshmem_functional_tests",),
           "both": ("tools/rocshmem_info", "tests/functional_tests/rocshmem_functional_tests")}
BLOCKED_ENV = {"CCC_OVERRIDE_OPTIONS", "CCC_PRINT_OPTIONS", "CL", "_CL_", "HIPCC_VERBOSE",
               "HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "LD_PRELOAD",
               "LD_AUDIT", "MAKEFLAGS", "MFLAGS", "LLVM_PROFILE_FILE", "LLVM_PROFILE_VERBOSE"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guard_environment(env):
    # Presence-based Clang controls can activate even when exported empty.
    bad = sorted(key for key, value in env.items()
                 if key.startswith("CC_PRINT_") or (value and key in BLOCKED_ENV))
    if env.get("HIP_CLANG_NUM_PARALLEL_JOBS", "1") not in ("", "1"):
        bad.append("HIP_CLANG_NUM_PARALLEL_JOBS")
    if bad:
        # Only variable names are reported, never values or the full environment.
        raise ValueError("Confounding environment controls: " + ", ".join(bad))


def guard_argv(argv):
    for i, arg in enumerate(argv):
        if (arg in ("-v", "--verbose", "-###", "--driver-mode=cl", "-gen-reproducer") or
                arg.startswith(("-fproc-stat-report", "-ccc-print-", "-gen-reproducer=",
                                "-save-temps", "--save-temps")) or
                (arg == "--driver-mode" and i + 1 < len(argv) and argv[i + 1] == "cl")):
            raise ValueError("Serializing/diagnostic driver flag: " + arg)
        if arg.startswith("@"):
            raise ValueError("Response files are not audited; configure direct Ninja link commands")


def canonical_link(argv, width, target):
    guard_argv(argv)
    flags = [arg for arg in argv if arg.startswith("--offload-jobs")]
    if flags != ([] if width == 1 else [f"--offload-jobs={width}"]):
        raise ValueError(f"Expected exactly the scoped width-{width} option")
    arches = [arch for arg in argv if arg.startswith("--offload-arch=")
              for arch in arg.split("=", 1)[1].split(",")]
    if sorted(arches) != sorted(ARCHITECTURES):
        raise ValueError("Expected each of the original 31 architecture IDs exactly once")
    if argv.count("-o") != 1 or argv[argv.index("-o") + 1] != target:
        raise ValueError("Unexpected output target")
    return [arg for arg in argv if arg not in flags]


def parse_link_command(text):
    argv = shlex.split(text.strip().splitlines()[-1])
    if argv[:2] == [":", "&&"]:
        argv = argv[2:]
    if argv[-2:] == ["&&", ":"]:
        argv = argv[:-2]
    if not argv or any(arg in ("&&", "||", ";", "|", ">", "<") for arg in argv):
        raise ValueError("Expected a single compiler command with optional CMake colon wrappers")
    return argv


def guard_dry_run(text):
    edges = [line for line in text.splitlines() if re.match(r"^\[\d+/\d+\]", line)]
    if len(edges) != 1 or "Linking CXX executable" not in edges[0]:
        raise ValueError("Dry run must contain exactly one selected final link")


def repeat_order(winner):
    return [2, 2] if winner == 2 else [winner, 2, winner, 2]


def eligible_totals(rows, targets, phase="sweep"):
    groups = {}
    for row in rows:
        if row["phase"] == phase and row["status"] == "passed":
            groups.setdefault(row["width"], {})[row["target"]] = row["wall_seconds"]
    return {width: sum(values.values()) for width, values in groups.items() if set(values) == set(targets)}


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    pgrp: int
    started: int
    rss_kib: int
    name: str


def parse_proc_stat(pid, text, page_size):
    end = text.rfind(")")
    fields = text[end + 2:].split()
    if end < 0 or len(fields) < 22:
        raise ValueError("Malformed /proc stat")
    return Proc(pid, int(fields[1]), int(fields[2]), int(fields[19]),
                max(0, int(fields[21])) * page_size // 1024, text[text.find("(") + 1:end])


def group_members(snapshot, leader, known):
    selected = {pid for pid, proc in snapshot.items()
                if proc.pgrp == leader or known.get(pid) == proc.started}
    while True:
        before = len(selected)
        selected.update(pid for pid, proc in snapshot.items() if proc.ppid in selected)
        if len(selected) == before:
            break
    return [snapshot[pid] for pid in selected]


def read_proc_snapshot():
    snapshot, denied = {}, 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            snapshot[int(path.name)] = parse_proc_stat(int(path.name), (path / "stat").read_text(errors="replace"), page_size)
        except PermissionError:
            denied += 1
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    return snapshot, denied


def meminfo():
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            result[key + "_kib"] = int(value.split()[0])
    return result


def cgroup_context():
    candidates = [Path("/sys/fs/cgroup")]
    try:
        relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
        path = Path("/sys/fs/cgroup") / relative.lstrip("/")
        if ".." not in path.parts:
            candidates.insert(0, path)
    except (OSError, StopIteration):
        pass
    for directory in candidates:
        if (directory / "memory.current").exists():
            values = {"path": str(directory)}
            for name in ("memory.current", "memory.max", "memory.events", "cpu.max"):
                try:
                    values[name] = (directory / name).read_text().strip()
                except OSError:
                    values[name] = None
            return values
    return {"path": None}


def memory_budget_kib(host, cgroup, explicit_limit, reserve):
    """Conservative available headroom, not a dedicated allocation guarantee."""
    budgets = [max(0, host["MemAvailable_kib"] - reserve)]
    if explicit_limit:
        budgets.append(explicit_limit)
    try:
        if cgroup.get("memory.max") not in (None, "max"):
            budgets.append(max(0, (int(cgroup["memory.max"]) - int(cgroup["memory.current"])) // 1024 - reserve))
    except (ValueError, TypeError, KeyError):
        pass
    return min(budgets)


def cgroup_remaining_kib(context):
    try:
        if context.get("memory.max") not in (None, "max"):
            return max(0, (int(context["memory.max"]) - int(context["memory.current"])) // 1024)
    except (ValueError, TypeError, KeyError):
        pass
    return None


def predicted_rss_kib(rows, target, width):
    observed = [row["memory"]["peak_summed_rss_kib"] / row["width"]
                for row in rows if row.get("target") == target and row["status"] == "passed"]
    # Shared memory makes linear scaling conservative in many cases, but this
    # remains an estimate, not an upper bound or proof against OOM.
    return max(observed) * width * 1.25 if observed else None


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def timed_link(command, env, case, interval, timeout, rss_limit_kib, min_available_kib):
    import resource
    known, peak_rss, peak_count, peak_lld, denied, samples = {}, 0, 0, 0, 0, 0
    available_min, cgroup_min, max_gap, previous, stop_reason = None, None, 0.0, None, None
    cgroup_before = cgroup_context()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    observer_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.monotonic()
    with (case / "build.log").open("w") as log, (case / "memory.jsonl").open("w") as telemetry:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while process.poll() is None:
                snapshot, errors = read_proc_snapshot()
                members = group_members(snapshot, process.pid, known)
                known.update((proc.pid, proc.started) for proc in members)
                elapsed = time.monotonic() - start
                rss = sum(proc.rss_kib for proc in members)
                available = meminfo().get("MemAvailable_kib")
                cgroup_remaining = cgroup_remaining_kib(cgroup_context())
                lld_count = sum("lld" in proc.name for proc in members)
                telemetry.write(json.dumps(dict(elapsed_seconds=elapsed, summed_rss_kib=rss,
                    process_count=len(members), resident_lld_count=lld_count,
                    host_mem_available_kib=available, cgroup_memory_remaining_kib=cgroup_remaining,
                    proc_permission_errors=errors)) + "\n")
                telemetry.flush()
                samples += 1
                denied += errors
                peak_rss = max(peak_rss, rss)
                peak_count, peak_lld = max(peak_count, len(members)), max(peak_lld, lld_count)
                if available is not None:
                    available_min = available if available_min is None else min(available_min, available)
                if cgroup_remaining is not None:
                    cgroup_min = cgroup_remaining if cgroup_min is None else min(cgroup_min, cgroup_remaining)
                if previous is not None:
                    max_gap = max(max_gap, elapsed - previous)
                previous = elapsed
                if rss_limit_kib and rss > rss_limit_kib:
                    stop_reason = "summed-rss-limit"
                elif min_available_kib and available is not None and available < min_available_kib:
                    stop_reason = "host-memory-floor"
                elif min_available_kib and cgroup_remaining is not None and cgroup_remaining < min_available_kib:
                    stop_reason = "cgroup-memory-floor"
                elif timeout and elapsed > timeout:
                    stop_reason = "timeout"
                if stop_reason:
                    stop_group(process)
                    break
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_group(process)
            raise
    wall = time.monotonic() - start
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    observer_after = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_seconds=wall, returncode=process.returncode, stop_reason=stop_reason,
        child_user_seconds=child_after.ru_utime - child_before.ru_utime,
        child_system_seconds=child_after.ru_stime - child_before.ru_stime,
        observer_cpu_seconds=(observer_after.ru_utime + observer_after.ru_stime -
                              observer_before.ru_utime - observer_before.ru_stime),
        memory=dict(sample_interval_requested_seconds=interval, samples=samples,
            max_sample_gap_seconds=max_gap, peak_summed_rss_kib=peak_rss,
            peak_resident_processes=peak_count, peak_resident_lld_processes=peak_lld,
            host_mem_available_min_kib=available_min, cgroup_memory_remaining_min_kib=cgroup_min,
            proc_permission_errors=denied,
            scope="owned process group and observed descendants; RSS double-counts shared pages and may miss short-lived processes"),
        cgroup_before=cgroup_before, cgroup_after=cgroup_context())


def explicit_inputs(argv, build, target):
    hashes = {}
    for arg in argv:
        if arg.startswith("-") or arg == target:
            continue
        path = Path(arg)
        if not path.is_absolute():
            path = build / path
        if path.is_file():
            hashes[str(path.resolve())] = sha(path)
    if not any(path.endswith("librocshmem.a") for path in hashes):
        raise ValueError("Prepared rocSHMEM static archive missing from explicit inputs")
    return hashes


@contextmanager
def measurement_lock(path):
    import fcntl
    with path.open("a") as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Sweep:
    def __init__(self, args):
        self.args = args
        self.source, self.build, self.output = args.source.resolve(), args.build.resolve(), args.output.resolve()
        if self.output.is_relative_to(self.build):
            raise ValueError("Evidence output must be outside the build directory")
        self.targets = TARGETS[args.targets]
        self.env = os.environ.copy()  # Held stable in memory; never dumped.
        guard_environment(self.env)
        self.env["NINJA_STATUS"] = "[%f/%t] "
        self.compile_bytes = (self.build / "compile_commands.json").read_bytes()
        cache = (self.build / "CMakeCache.txt").read_text()
        if not re.search(r"^HIP_CLANG_NUM_PARALLEL_JOBS:[^=]+=1$", cache, re.MULTILINE):
            raise ValueError("Prepare with HIP_CLANG_NUM_PARALLEL_JOBS=1")
        self.commands, self.inputs, self.reference_outputs, self.rows = {}, {}, {}, []
        self.owns_output = False

    def record(self, case, row):
        self.rows.append(row)
        write_json(case / "result.json", row)
        write_json(self.output / "results.json", self.rows)
        return row

    def case(self, width, target, phase):
        case = self.output / f"{len(self.rows):02d}-{phase}-width{width}-{Path(target).name}"
        case.mkdir()
        budget = memory_budget_kib(meminfo(), cgroup_context(),
            self.args.max_summed_rss_gib * 1024**2, self.args.min_available_gib * 1024**2)
        prediction = predicted_rss_kib(self.rows, target, width)
        preflight = dict(available_budget_kib=budget, predicted_summed_rss_kib=prediction,
                         prediction_margin=1.25, prediction_is_not_guaranteed_upper_bound=True)
        write_json(case / "memory-preflight.json", preflight)
        if budget <= 0 or (prediction is not None and prediction > budget):
            return self.record(case, dict(width=width, target=target, phase=phase,
                case=case.name, status="skipped-predicted-memory", memory_preflight=preflight))
        header = self.build / "include/rocshmem/rocshmem_config_embedded.hpp"
        content, stat = header.read_bytes(), header.stat()
        with (case / "configure.log").open("w") as log:
            subprocess.run(["cmake", "-S", str(self.source), "-B", str(self.build),
                            f"-DROCSHMEM_NATIVE_LINK_JOBS={width}"], env=self.env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if header.read_bytes() != content:
            raise ValueError("Generated-header content changed")
        # Upstream file(WRITE) touches identical embedded-config contents. Only
        # restore mtime after equality, so the experiment holds compiled inputs fixed.
        os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        write_json(case / "generated-header.json", dict(sha256=hashlib.sha256(content).hexdigest(),
                   byte_identical_after_configure=True, restored_mtime_ns=stat.st_mtime_ns))
        if (self.build / "compile_commands.json").read_bytes() != self.compile_bytes:
            raise ValueError("Compile commands changed")
        argv = parse_link_command(subprocess.check_output(
            ["ninja", "-C", str(self.build), "-t", "commands", target], env=self.env, text=True))
        canonical = canonical_link(argv, width, target)
        if target not in self.commands:
            if width != 1:
                raise ValueError("Width-one baseline must be first")
            self.commands[target] = canonical
        elif canonical != self.commands[target]:
            raise ValueError("Unexpected final-link command changes")
        write_json(case / "link-command.json", argv)
        inputs = explicit_inputs(argv, self.build, target)
        if target not in self.inputs:
            self.inputs[target] = inputs
        elif inputs != self.inputs[target]:
            raise ValueError("Explicit link input bytes changed")
        write_json(case / "input-hashes.json", inputs)
        path = self.build / target
        if path.is_symlink() or not path.resolve().is_relative_to(self.build):
            raise ValueError("Output must be a regular path inside the isolated build")
        if path.exists():
            if target not in self.reference_outputs:
                shutil.copy2(path, self.output / (path.name + ".initial"))
            path.unlink()
        preview = subprocess.check_output(["ninja", "-C", str(self.build), "-n", target],
                                          env=self.env, text=True)
        (case / "dry-run.log").write_text(preview)
        guard_dry_run(preview)
        row = timed_link(["ninja", "-C", str(self.build), "-j1", target], self.env, case,
            self.args.sample_interval, self.args.timeout_seconds, budget,
            self.args.min_available_gib * 1024**2)
        row.update(width=width, phase=phase, target=target, case=case.name, memory_preflight=preflight)
        row["status"] = row["stop_reason"] or ("passed" if row["returncode"] == 0 else "link-failed")
        if row["returncode"] == 0 and not row["stop_reason"]:
            row["output_sha256"] = sha(path)
            shutil.copy2(path, case / path.name)
            if target not in self.reference_outputs:
                self.reference_outputs[target] = row["output_sha256"]
            row["raw_output_parity"] = row["output_sha256"] == self.reference_outputs[target]
            if not row["raw_output_parity"]:
                row["status"] = "parity-failed"
        elif path.is_file():
            shutil.copy2(path, case / (path.name + ".partial"))
        if explicit_inputs(argv, self.build, target) != self.inputs[target]:
            row["status"] = "inputs-changed"
        self.record(case, row)
        if row["status"] == "inputs-changed":
            raise ValueError("Input mutation during timed link")
        return row

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.owns_output = True
        affinity = sorted(os.sched_getaffinity(0))
        if self.args.expected_affinity_cpus and len(affinity) != self.args.expected_affinity_cpus:
            raise ValueError("CPU affinity differs from requested CI allocation")
        write_json(self.output / "context.json", dict(
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), platform=platform.platform(),
            visible_logical_cpus=os.cpu_count(), affinity_cpus=affinity, memory=meminfo(),
            cgroup=cgroup_context(), architectures=ARCHITECTURES, widths=self.args.order,
            compile_commands_sha256=hashlib.sha256(self.compile_bytes).hexdigest(),
            environment_policy="stable in-memory snapshot; profiling/override controls rejected; no secret values dumped",
            scope="separate selected native links; not project or CI time; visible CPUs do not prove dedicated entitlement"))
        unsafe = {}
        with measurement_lock(self.args.slot_lock):
            for width in self.args.order:
                for target in self.targets:
                    if target in unsafe:
                        row = dict(width=width, phase="sweep", target=target,
                                   status="skipped-after-failure", reason=unsafe[target])
                        self.rows.append(row)
                        write_json(self.output / "results.json", self.rows)
                        continue
                    row = self.case(width, target, "sweep")
                    if width in (1, 2) and row["status"] != "passed":
                        raise ValueError("Both baseline and width-two reference must pass")
                    if row["status"] in ("summed-rss-limit", "host-memory-floor", "cgroup-memory-floor", "timeout", "link-failed"):
                        unsafe[target] = f"width {width}: {row['status']}"
            totals = eligible_totals(self.rows, self.targets)
            winner = min(totals, key=lambda width: (totals[width], width))
            summary = dict(status="sweep-complete", screening_winner=winner,
                sweep_link_totals_seconds=totals, targets=self.targets, product_default_changed=False,
                selection_basis="sum of separately timed selected links; not full project/CI elapsed",
                memory_limitations="sampled RSS double-counts shared pages and may miss peaks; predictions are conservative estimates, not guaranteed upper bounds")
            write_json(self.output / "summary.json", summary)
            if self.args.repeat_best:
                orders = repeat_order(winner)
                for repeat_index, width in enumerate(orders):
                    for target in self.targets:
                        row = self.case(width, target, "repeat")
                        row["repeat_index"] = repeat_index
                        write_json(self.output / row["case"] / "result.json", row)
                        write_json(self.output / "results.json", self.rows)
                        if row["status"] != "passed":
                            raise ValueError("Selected-width repeat failed; no accepted repeat comparison")
                repeats = [row for row in self.rows if row["phase"] == "repeat"]
                grouped = {}
                for row in repeats:
                    grouped.setdefault((row["repeat_index"], row["width"]), []).append(row["wall_seconds"])
                reference = statistics.median(sum(values) for (_, width), values in grouped.items() if width == 2)
                selected = statistics.median(sum(values) for (_, width), values in grouped.items() if width == winner)
                summary.update(status="complete", repeat_order=orders,
                    repeat_width2_median_link_total_seconds=reference,
                    repeat_winner_median_link_total_seconds=selected,
                    repeat_savings_vs2_percent=100 * (reference - selected) / reference,
                    repeat_benefit_observed=winner != 2 and selected < reference)
            summary["parity_failures"] = [row["case"] for row in self.rows if row["status"] == "parity-failed"]
            summary["skipped_or_failed_cases"] = [dict(width=row["width"], target=row["target"], status=row["status"])
                                                   for row in self.rows if row["status"] != "passed"]
            write_json(self.output / "summary.json", summary)
            return 2 if summary["parity_failures"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "build", "output", "slot-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--order", default="1,2,4,8,16,31")
    parser.add_argument("--targets", choices=TARGETS, default="both")
    parser.add_argument("--repeat-best", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=0)
    parser.add_argument("--max-summed-rss-gib", type=float, default=0)
    parser.add_argument("--min-available-gib", type=float, default=4.0)
    parser.add_argument("--expected-affinity-cpus", type=int)
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        parser.error("Actual sweep requires Linux /proc, process groups and flock")
    try:
        args.order = tuple(map(int, args.order.split(",")))
    except ValueError:
        parser.error("Order must be comma-separated integer widths")
    if args.order[:2] != (1, 2) or len(set(args.order)) != len(args.order) or any(w not in WIDTHS for w in args.order):
        parser.error("Order must begin 1,2 and contain distinct supported widths up to 31")
    if args.sample_interval <= 0 or min(args.timeout_seconds, args.max_summed_rss_gib, args.min_available_gib) < 0:
        parser.error("Sample interval must be positive and limits nonnegative")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; stopping owned link process group")
    signal.signal(signal.SIGTERM, interrupted)
    sweep = Sweep(args)
    try:
        return sweep.run()
    except BaseException as error:
        if sweep.owns_output:
            write_json(sweep.output / "failure.json", dict(type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
