#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the AMD SMI KPI coverage tables from ``api_summary.py`` output.

``api_summary.py`` answers "which APIs did the suites touch?"; this turns that
into the tables tracked on the AMD SMI KPI page, plus machine-readable CSVs and
an appendable history file so coverage can be trended across runs.

Consumes:
    <log_dir>/_api_summary.csv   written by api_summary.py
    <log_dir>/_py_cli_test.log   optional, for the CLI command-count column
    include/amd_smi/amdsmi.h     for the @ingroup -> subsystem mapping

Writes into --log_dir:
    _api_coverage.md             the report (also echoed to stdout)
    _api_coverage.csv            the report's tables, for opening as a sheet
    _api_coverage.tsv            the same tables, for pasting into a sheet
    api_coverage_trend.csv       this run's row, after the inherited ones

Every run directory is self-contained: the trend carries forward the rows of
the newest earlier ``api-coverage-results-*`` sibling, so the latest directory
holds the whole trend and can be copied on its own. It records the hardware and
driver configuration behind each number, but never the hostname.

Coverage is hardware- and driver-dependent -- CPU/ESMI needs an EPYC host with
HSMP, NIC needs a supported NIC, IFoE needs ifoe.ko. The run metadata block
records all of it so a low number can be explained rather than guessed at.

Run manually (after api_summary.py has produced the CSV):

    python3 projects/amdsmi/tests/api_coverage_report.py --log_dir build

Normally invoked for you by ``tests/run_api_coverage.sh``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_HEADER = PROJECT_ROOT / "include" / "amd_smi" / "amdsmi.h"
DEFAULT_BASELINE = PROJECT_ROOT / "tests" / "api_coverage_baseline.csv"

# Each run writes a self-contained directory, so the trend lives inside it
# rather than in one shared file. It is seeded by carrying forward the newest
# sibling run directory's rows, which keeps every run dir a complete bundle: the
# newest one holds the full trend and can be copied on its own.
#
# Deliberately not '_'-prefixed. Every '_' file is disposable output that can be
# deleted or regenerated; this is the one generated file worth keeping, because
# deleting it is the only way to lose the trend.
HISTORY_NAME = "api_coverage_trend.csv"
RUN_DIR_PREFIX = "api-coverage-results-"

# Subsystem buckets. Everything is keyed off the doxygen @ingroup tag that
# precedes each declaration in amdsmi.h, because the function name alone is
# ambiguous (amdsmi_get_clk_freq and amdsmi_get_fw_info are GPU APIs despite
# carrying no _gpu_ infix). Only NIC is name-based: its APIs are split across
# tagNicInfo and tagProcDiscovery.
SYSTEM_GROUPS = frozenset(
    {
        "tagInitShutdown",
        "tagVersionQuery",
        "tagSoftwareVersion",
        "tagSystemInfo",
        "tagHWTopology",
        "tagErrorQuery",
        # Node/chassis scope (amdsmi_node_handle): NPM status and tray info.
        # Tested under functional/ifoe/tray/ but node-scoped, not fabric.
        "tagNodeInfo",
    }
)

# Infinity Fabric over Ethernet -- XGMI's counterpart, the GPU scale-up fabric.
# Reported separately from GPU because it is gated on its own kernel driver
# (ifoe.ko exposing /dev/cbl-cfg-* plus a netlink family, see src/ualoe_lib/),
# so on a host without it these read untested for a reason unrelated to the GPU.
IFOE_GROUPS = frozenset({"tagFabric"})

SUBSYSTEM_ORDER = ("GPU", "System/Topo", "CPU/ESMI", "NIC", "IFoE")

# api_coverage_baseline.csv is laid out as the two report tables; these map its
# (row, metric) pairs onto the flat column names used by the history CSV.
BASELINE_SUBSYSTEM_KEYS = {
    "GPU": "gpu",
    "System/Topo": "systopo",
    "CPU/ESMI": "cpu_esmi",
    "NIC": "nic",
    "IFoE": "ifoe",
    "Total": "api",
}
BASELINE_FRAMEWORK_KEYS = {
    ("Gtest C/C++", "API"): "c_api",
    ("Gtest C/C++", "Unit"): "c_unit",
    ("Gtest C/C++", "Functional"): "c_func",
    ("Python", "API"): "py_api",
    ("Python", "Unit"): "py_unit",
    ("Python", "Functional"): "py_func",
    ("Python", "CLI"): "cli_cmds",
    ("Total", "API"): "total_api",
    ("Total", "Unit"): "total_unit",
    ("Total", "Functional"): "total_func",
    ("Total", "CLI"): "cli_cmds",
}

# Targets from the AMD SMI KPI page. API, Functional and CLI each target 100% of
# their own scale. Unit is held at 80% for now because it is scaled against the
# full API count, which is a placeholder -- not every API is reachable without a
# device. When a qualifying no-hardware subset is defined, this percentage and
# Unit's entry in scale_of are the only things that need to change.
SUBSYSTEM_GOAL_PCT = 100
FRAMEWORK_GOALS = {"API": 100, "Unit": 80, "Functional": 100, "CLI": 100}

# RunCmds() logs one "<cmd> : <result>" line per amd-smi invocation. The KPI
# "CLI (#)" column counts those invocations, not the far smaller number of
# unittest methods driving them.
CLI_INVOCATION_RE = re.compile(r"^\s+(?:sudo\s+)?amd-smi\s.*\s:", re.MULTILINE)
CLI_METHOD_RE = re.compile(r"\[-+\]\s+(\d+)\s+tests?\s+ran")

C_COLUMNS = ("c_unit_test", "c_func_test", "c_intg_test")
PY_COLUMNS = ("py_unit_test", "py_func_test", "py_intg_test")

HISTORY_COLUMNS = [
    "run_date",
    "os",
    "kernel",
    "amdgpu",
    "amdsmi_tool",
    "amdsmi_lib",
    "rocm",
    "hsmp_driver",
    "nic_driver",
    "ifoe_driver",
    "gpu_model",
    "gfx_target",
    "api_total",
    "api_tested",
    "api_pct",
    "gpu_total",
    "gpu_tested",
    "systopo_total",
    "systopo_tested",
    "cpu_esmi_total",
    "cpu_esmi_tested",
    "nic_total",
    "nic_tested",
    "ifoe_total",
    "ifoe_tested",
    "c_api",
    "c_unit",
    "c_func",
    "py_api",
    "py_unit",
    "py_func",
    "total_api",
    "total_unit",
    "total_func",
    "cli_cmds",
    "cli_tests",
    "notes",
]


# --------------------------------------------------------------------------- #
# Run metadata -- so a low number can be explained rather than guessed at.
# --------------------------------------------------------------------------- #
def _run(cmd, timeout=120):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ""


def collect_metadata():
    """Best-effort environment capture; missing metadata is reported as N/A.

    The hostname is deliberately not collected. What explains a coverage number
    is the hardware and driver configuration -- GPU, gfx target, HSMP/NIC/IFoE
    drivers -- so recording the machine's name would identify the host without
    adding anything. Use --notes to attribute a run on purpose.
    """
    meta = dict.fromkeys(HISTORY_COLUMNS, "N/A")
    meta["run_date"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    meta["kernel"] = platform.release() or "N/A"
    meta["amdgpu"] = _read("/sys/module/amdgpu/version") or "N/A"
    meta["notes"] = ""

    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            meta["os"] = line.split("=", 1)[1].strip().strip('"')

    # ifoe.ko exposes /dev/cbl-cfg-* misc devices; without it the IFoE row is
    # untestable regardless of how good the GPU is.
    meta["ifoe_driver"] = (
        "present" if list(pathlib.Path("/sys/class/misc").glob("cbl-cfg-*")) else "N/A"
    )

    if not shutil.which("amd-smi"):
        return meta

    try:
        ver = json.loads(_run(["amd-smi", "version", "--json"]) or "[]")
        if ver:
            meta["amdsmi_tool"] = ver[0].get("version", "N/A")
            meta["amdsmi_lib"] = ver[0].get("amdsmi_library_version", "N/A")
            meta["rocm"] = ver[0].get("rocm_version", "N/A")
            meta["hsmp_driver"] = ver[0].get("amd_hsmp_driver_version", "N/A")
            meta["nic_driver"] = ver[0].get("nic_driver_version", "N/A")
    except (json.JSONDecodeError, IndexError, AttributeError):
        pass

    try:
        static = json.loads(_run(["amd-smi", "static", "-g", "0", "--asic", "--board", "--json"]))
        gpu0 = static["gpu_data"][0]
        meta["gpu_model"] = gpu0["asic"].get("market_name", "N/A")
        meta["gfx_target"] = gpu0["asic"].get("target_graphics_version", "N/A")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    return meta


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def classify(group, func):
    if "_nic_" in func:
        return "NIC"
    if group.startswith("tagEsmi"):
        return "CPU/ESMI"
    if group in IFOE_GROUPS:
        return "IFoE"
    if group in SYSTEM_GROUPS:
        return "System/Topo"
    return "GPU"


def parse_header(header_path):
    """Return {func_name: subsystem}, scanning blocks like api_summary.py does."""
    content = header_path.read_text()
    mapping = {}
    end = 0
    while True:
        start = content.find("@ingroup", end)
        if start == -1:
            break
        group = content[start + len("@ingroup") : content.find("\n", start)].strip()
        end = content.find(");", start)
        if end == -1:
            break
        open_paren = content.rfind("(", start, end)
        name_start = content.rfind("amdsmi_", start, open_paren)
        mapping[content[name_start:open_paren].strip()] = classify(
            group, content[name_start:open_paren].strip()
        )
    return mapping


def read_summary(csv_path):
    """Return {func_name: {column: count}} from _api_summary.csv."""
    rows = {}
    with csv_path.open() as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        header = [column.strip() for column in next(reader)]
        for row in reader:
            if not row or not row[0].strip():
                continue
            values = [value.strip() for value in row]
            rows[values[0]] = {k: int(v) for k, v in zip(header[1:], values[1:])}
    return rows


def count_cli(log_path):
    """Return (amd-smi invocations, unittest methods) from a cli_unit_test.py log."""
    if not log_path.exists():
        return (0, 0)
    text = log_path.read_text()
    methods = CLI_METHOD_RE.search(text)
    return (len(CLI_INVOCATION_RE.findall(text)), int(methods.group(1)) if methods else 0)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def pct(num, den):
    return round(num / den * 100) if den else 0


def pct_s(num, den):
    return f"{pct(num, den)}%" if den else "N/A"


def delta(current, previous):
    """Render the change vs the previous run, for the Trend column."""
    if previous is None or str(previous).strip().lower() in ("", "n/a"):
        return "-"
    try:
        diff = int(current) - int(float(previous))
    except (TypeError, ValueError):
        return "-"
    return "=" if diff == 0 else f"{diff:+d}"


def md_table(rows):
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for index, row in enumerate(rows):
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
        if index == 0:
            out.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    return "\n".join(out)


def read_history_rows(path):
    """Every row of a history CSV, oldest first. Missing/unreadable -> []."""
    if path is None or not path.exists():
        return []
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def find_previous_history(log_dir):
    """History file of the newest earlier run directory, or None.

    Run directories are named ``api-coverage-results-<timestamp>`` with a
    sortable timestamp, so the newest sibling by name is the previous run. Only
    siblings of --log_dir are considered: a directory placed next to existing
    run dirs picks the trend up, one somewhere else starts a fresh trend.
    """
    log_dir = log_dir.resolve()
    if not log_dir.parent.is_dir():
        return None
    candidates = [
        entry
        for entry in log_dir.parent.glob(f"{RUN_DIR_PREFIX}*")
        if entry.is_dir() and entry.resolve() != log_dir and (entry / HISTORY_NAME).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name) / HISTORY_NAME


def read_baseline(path):
    """Return {'start': {...}, 'previous': {...}} keyed like api_coverage_history.csv.

    The file is laid out as the two report tables (section/row/metric matching
    _api_coverage.csv) with 'start' and 'previous' as value columns, so rows can
    be copy-pasted straight from a generated run. This maps that back onto the
    flat history column names the report and history CSV use.
    """
    points = {"start": {}, "previous": {}}
    if not path.exists():
        return points
    try:
        lines = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
        for row in csv.DictReader(lines):
            section = (row.get("section") or "").strip().lower()
            name = (row.get("row") or "").strip()
            metric = (row.get("metric") or "").strip()
            if section == "subsystem":
                prefix = BASELINE_SUBSYSTEM_KEYS.get(name)
                key = f"{prefix}_{metric.lower()}" if prefix else None
            elif section == "framework":
                key = BASELINE_FRAMEWORK_KEYS.get((name, metric))
            else:
                key = None
            if not key:
                continue
            for point in points:
                value = (row.get(point) or "").strip()
                if value:
                    points[point][key] = value
    except (OSError, csv.Error):
        pass
    return points


def read_suite_status(log_dir):
    """Return [(suite, state, ran, passed, failed, skipped)] from _suite_status.csv.

    run_api_coverage.sh records each suite's exit code; the counts are parsed
    back out of the suite's own log. A report whose suites failed is reporting
    partial coverage, so this is rendered alongside the numbers rather than
    leaving a silent gap.
    """
    status_path = log_dir / "_suite_status.csv"
    if not status_path.exists():
        return []
    rows = []
    try:
        with status_path.open() as handle:
            for row in csv.DictReader(handle):
                suite = (row.get("suite") or "").strip()
                code = (row.get("exit_code") or "").strip()
                log = log_dir / (row.get("log") or "").strip()
                text = log.read_text(errors="ignore") if log.exists() else ""

                def count(pattern):
                    found = re.findall(pattern, text)
                    return int(found[-1]) if found else 0

                ran = count(r"\[[=-]+\]\s+(\d+)\s+tests?\s+(?:from .*?)?ran")
                passed = count(r"\[\s*PASSED\s*\]\s+(\d+)\s+tests?")
                failed = count(r"\[\s*FAILED\s*\]\s+(\d+)\s+tests?")
                skipped = count(r"\[\s*SKIPPED\s*\]\s+(\d+)\s+tests?")
                if code == "skipped":
                    state = "not run"
                elif code == "0":
                    state = "pass"
                else:
                    state = f"FAIL (rc={code})"
                rows.append((suite, state, ran, passed, failed, skipped))
    except (OSError, csv.Error):
        return []
    return rows


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def cli_surface_total():
    """Total amd-smi CLI command surface, measured by cli_surface.py.

    Returns None when the CLI cannot be imported, so the caller can fall back to
    the baseline file rather than silently reporting a zero denominator.
    """
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import cli_surface

        parser = cli_surface.load_parser()
        if parser is None:
            return None
        _, _, features, _ = cli_surface.measure(parser)
        return cli_surface.compute(features)["total"] if features else None
    except Exception:  # noqa: BLE001 - any import/introspection failure falls back
        return None


def arrow(start, today, end, denom=None):
    """Render 'S -> Today -> End'.

    ``end`` is already formatted: framework rows end at the 100% scale (a count),
    the Total row ends at the target percentage. Passing ``denom`` renders the
    first two values as percentages of it. Missing values render as '-'.
    """

    def one(value):
        num = as_int(value)
        if num is None:
            return "-"
        return f"{pct(num, denom)}%" if denom else str(num)

    return f"{one(start)} → {one(today)} → {end}"


def write_history(path, inherited, row):
    """Write the carried-forward rows plus this run's, newest last.

    Rewritten whole rather than appended so an inherited history and an
    explicit --history file are handled by the same path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for old in inherited:
            writer.writerow({column: old.get(column, "") for column in HISTORY_COLUMNS})
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log_dir",
        default="build",
        type=pathlib.Path,
        help="Directory holding _api_summary.csv and the suite logs, default=%(default)s",
    )
    parser.add_argument(
        "--amdsmi",
        default=DEFAULT_HEADER,
        type=pathlib.Path,
        help="Path to amdsmi.h, default=%(default)s",
    )
    parser.add_argument(
        "--cli_log",
        default="_py_cli_test.log",
        help="Filename of the CLI suite log inside --log_dir, default=%(default)s",
    )
    parser.add_argument(
        "--history",
        default=None,
        type=pathlib.Path,
        help="Trend CSV to inherit from and write; default=<log_dir>/" + HISTORY_NAME,
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        type=pathlib.Path,
        help="Hand-maintained start/previous/goal values, default=%(default)s",
    )
    parser.add_argument(
        "--no_history", action="store_true", help="Do not write a history file for this run"
    )
    parser.add_argument("--notes", default="", help="Free-text note stored with this run")
    args = parser.parse_args()

    csv_path = args.log_dir / "_api_summary.csv"
    if not csv_path.exists():
        print(f"error: {csv_path} not found -- run tests/api_summary.py first", file=sys.stderr)
        return 1
    if not args.amdsmi.exists():
        print(f"error: header not found, {args.amdsmi}", file=sys.stderr)
        return 1

    subsystems = parse_header(args.amdsmi)
    summary = read_summary(csv_path)
    cli_cmds, cli_tests = count_cli(args.log_dir / args.cli_log)
    num_api = len(summary)
    meta = collect_metadata()
    meta["notes"] = args.notes
    baseline = read_baseline(args.baseline)
    start = baseline.get("start", {})
    # Self-contained run dirs: inherit the previous run's rows so this dir ends
    # up holding the whole trend.
    #
    # Sources, in order: an explicit --history, the newest sibling run dir, or
    # the trend already sitting in this directory. That last fallback matters
    # because the file is rewritten whole -- without it, reusing one fixed
    # --log_dir would silently truncate its own trend on every run. Siblings are
    # preferred over self so re-running the report inside a dated dir stays
    # idempotent rather than appending a row each time.
    history_path = args.history or (args.log_dir / HISTORY_NAME)
    inherited = read_history_rows(
        args.history or find_previous_history(args.log_dir) or history_path
    )
    # An explicit 'previous' row in the baseline pins what Trend compares
    # against; otherwise fall back to the last inherited history row.
    previous = baseline.get("previous") or (inherited[-1] if inherited else None)
    out = []

    # ---- Run metadata ------------------------------------------------------ #
    out.append("# AMD SMI API Coverage\n")
    out.append(f"**Run:** {meta['run_date']}\n")
    env_rows = [["Field", "Value"]]
    env_rows += [
        ["GPU", f"{meta['gpu_model']} ({meta['gfx_target']})"],
        ["amd-smi tool", meta["amdsmi_tool"]],
        ["amdsmi library", meta["amdsmi_lib"]],
        ["ROCm", meta["rocm"]],
        ["amdgpu driver", meta["amdgpu"]],
        ["Kernel", meta["kernel"]],
        ["OS", meta["os"]],
        ["HSMP driver (CPU/ESMI)", meta["hsmp_driver"]],
        ["NIC driver", meta["nic_driver"]],
        ["IFoE driver (ifoe.ko)", meta["ifoe_driver"]],
    ]
    out.append(md_table(env_rows))
    out.append(
        "\n> Coverage is hardware- and driver-gated: CPU/ESMI needs HSMP, NIC needs a\n"
        "> supported NIC, IFoE needs ifoe.ko. An `N/A` above explains a 0% row.\n"
    )

    # ---- Test suite status -------------------------------------------------- #
    # Rendered before the coverage tables: numbers produced by a run whose suites
    # failed are partial, and that has to be visible next to them.
    suites = read_suite_status(args.log_dir)
    suite_rows = []
    if suites:
        broken = [s for s in suites if s[1] != "pass"]
        if broken:
            out.append(
                f"\n> **WARNING: {len(broken)} of {len(suites)} suites did not pass.**"
                " The coverage below is partial --\n> an API only counts when the test"
                " that calls it actually runs.\n"
            )
        suite_rows = [["Suite", "Result", "Ran", "Passed", "Failed", "Skipped"]]
        suite_rows += [[s[0], s[1], s[2], s[3], s[4], s[5]] for s in suites]
        out.append("\n## Test suites\n")
        out.append(md_table(suite_rows))

    # ---- Table 1: coverage by subsystem ------------------------------------ #
    totals = {name: [0, 0] for name in SUBSYSTEM_ORDER}
    untested = {name: [] for name in SUBSYSTEM_ORDER}
    for func, flags in summary.items():
        bucket = subsystems.get(func, "GPU")
        totals[bucket][0] += 1
        if flags["Tested"]:
            totals[bucket][1] += 1
        else:
            untested[bucket].append(func)

    hist_key = {
        "GPU": "gpu",
        "System/Topo": "systopo",
        "CPU/ESMI": "cpu_esmi",
        "NIC": "nic",
        "IFoE": "ifoe",
    }
    rows = [["Subsystem", "Total", "Tested", "Untested", "Coverage", "Trend"]]
    grand_total = grand_tested = 0
    for name in SUBSYSTEM_ORDER:
        total, tested = totals[name]
        grand_total += total
        grand_tested += tested
        prev = previous.get(f"{hist_key[name]}_tested") if previous else None
        rows.append(
            [name, total, tested, total - tested, pct_s(tested, total), delta(tested, prev)]
        )
        meta[f"{hist_key[name]}_total"] = total
        meta[f"{hist_key[name]}_tested"] = tested
    rows.append(
        [
            "**Total**",
            grand_total,
            grand_tested,
            grand_total - grand_tested,
            pct_s(grand_tested, grand_total),
            delta(grand_tested, previous.get("api_tested") if previous else None),
        ]
    )
    out.append(
        f"\n## API Tests -- coverage by subsystem\n\n**Goal: {SUBSYSTEM_GOAL_PCT}% coverage**\n"
    )
    out.append(md_table(rows))
    subsys_rows = rows

    meta["api_total"] = grand_total
    meta["api_tested"] = grand_tested
    meta["api_pct"] = pct(grand_tested, grand_total)

    # ---- Table 2: coverage by framework ------------------------------------ #
    def tally(*columns):
        return sum(1 for flags in summary.values() if any(flags[c] for c in columns))

    counts = {
        "c_api": tally(*C_COLUMNS),
        "c_unit": tally("c_unit_test"),
        "c_func": tally("c_func_test", "c_intg_test"),
        "py_api": tally(*PY_COLUMNS),
        "py_unit": tally("py_unit_test"),
        "py_func": tally("py_func_test", "py_intg_test"),
        "total_api": tally(*C_COLUMNS, *PY_COLUMNS),
        "total_unit": tally("c_unit_test", "py_unit_test"),
        "total_func": tally("c_func_test", "c_intg_test", "py_func_test", "py_intg_test"),
    }
    meta.update(counts)
    meta["cli_cmds"] = cli_cmds
    meta["cli_tests"] = cli_tests

    # Goal counts: API/Unit/Functional are a percentage of the measured API total
    # (the APIs in amdsmi.h). The CLI 100% mark is the measured command surface
    # from cli_surface.py -- base commands, valid option/format combinations and
    # negative cases -- falling back to the baseline 'goal' row if that import
    # fails (e.g. the CLI is not installed).
    # Every goal is derived, nothing is hand-entered: API/Unit/Functional are a
    # percentage of the APIs measured in amdsmi.h, and the CLI 100% mark is the
    # command surface measured by cli_surface.py. If the CLI cannot be
    # introspected the column degrades to '-' rather than inventing a number.
    # The 100% mark each column is measured against: the APIs counted in
    # amdsmi.h, or for CLI the command surface measured by cli_surface.py.
    # Framework rows end the arrow at this scale; the Total row ends at the
    # target percentage from FRAMEWORK_GOALS.
    cli_total = cli_surface_total() or 0
    scale_of = {"API": num_api, "Unit": num_api, "Functional": num_api, "CLI": cli_total}
    # The goal is a percentage of the scale, and is the arrow's endpoint
    # everywhere: as a count in the framework rows, as a percentage in Total.
    goal_of = {
        metric: round(scale_of[metric] * FRAMEWORK_GOALS[metric] / 100)
        for metric in FRAMEWORK_GOALS
    }

    def fw_cell(metric, key, as_pct=False):
        """S -> Today -> Goal, as counts or (for the Total row) percentages."""
        scale = scale_of[metric]
        today = cli_cmds if metric == "CLI" else counts[key]
        s_val = start.get(key, "")
        if as_pct:
            return arrow(s_val, today, f"{FRAMEWORK_GOALS[metric]}%", denom=scale)
        return arrow(s_val, today, str(goal_of[metric]) if scale else "-")

    prev_total_api = previous.get("total_api") if previous else None
    rows = [
        [
            "Framework",
            "API (#)<br>S→Today→Goal",
            "Unit (#)\\*<br>S→Today→Goal",
            "Functional (#)\\*\\*<br>S→Today→Goal",
            "CLI (#)\\*\\*\\*<br>S→Today→Goal",
            "Trend",
        ]
    ]
    rows.append(
        [
            "Gtest, C/C++",
            fw_cell("API", "c_api"),
            fw_cell("Unit", "c_unit"),
            fw_cell("Functional", "c_func"),
            "N/A",
            "",
        ]
    )
    rows.append(
        [
            "Python",
            fw_cell("API", "py_api"),
            fw_cell("Unit", "py_unit"),
            fw_cell("Functional", "py_func"),
            fw_cell("CLI", "cli_cmds"),
            "",
        ]
    )
    rows.append(
        [
            "**Total**",
            fw_cell("API", "total_api", as_pct=True),
            fw_cell("Unit", "total_unit", as_pct=True),
            fw_cell("Functional", "total_func", as_pct=True),
            fw_cell("CLI", "cli_cmds", as_pct=True),
            delta(counts["total_api"], prev_total_api),
        ]
    )
    out.append(f"\n## Coverage by framework (of {num_api} APIs in {args.amdsmi.name})\n")
    out.append(md_table(rows))
    fw_rows = rows

    # ---- Footnotes: how each column is scaled and how its goal is derived --- #
    out.append(
        "\nEach column is measured against a **100% scale**, and its goal is a"
        " percentage of that scale:\n\n"
        "```\ngoal = scale x goal%\n```\n\n"
        "The arrow ends at the goal in every row -- as a count in the framework"
        " rows, as a percentage in **Total**. Nothing here is hand-entered -- both the"
        " scale and the goal are measured on every run."
    )
    legend = [["Column", "100% scale", "Goal", "Scale measured from"]]
    legend.append(
        [
            "API",
            num_api,
            f"{FRAMEWORK_GOALS['API']}% = {goal_of['API']}",
            f"`@ingroup` blocks in `{args.amdsmi.name}`",
        ]
    )
    legend.append(
        [
            "Unit \\*",
            scale_of["Unit"],
            f"{FRAMEWORK_GOALS['Unit']}% = {goal_of['Unit']}",
            f"same as API (`{args.amdsmi.name}`)",
        ]
    )
    legend.append(
        [
            "Functional \\*\\*",
            scale_of["Functional"],
            f"{FRAMEWORK_GOALS['Functional']}% = {goal_of['Functional']}",
            f"same as API (`{args.amdsmi.name}`)",
        ]
    )
    legend.append(
        [
            "CLI \\*\\*\\*",
            scale_of["CLI"] or "-",
            f"{FRAMEWORK_GOALS['CLI']}% = {goal_of['CLI'] or '-'}",
            "`tests/cli_surface.py`",
        ]
    )
    out.append("\n" + md_table(legend))
    out.append(
        "\n\\* **Unit** -- APIs reached by tests that need no hardware: the"
        " `*Unit*` GTest suites and `unit_tests.py`. An API counts only when the"
        " test actually invokes it, so wrapper-level argument validation that"
        " raises before the C call does not count.\n"
        "\n  Its scale is currently the full API count -- a placeholder, since not"
        " every API can be exercised without a device. That is why Unit is the one"
        " column held below 100%: the goal is 80% of a scale known to be too"
        " large. Both should be revisited together once a qualifying no-hardware"
        " subset is agreed.\n"
        "\n\\*\\* **Functional** -- APIs reached against a live device: the"
        " `*Functional*` GTest suites and `integration_test.py` (which drives"
        " `tests/python/functional/`). Every API is expected to have a functional"
        " test, so the scale is every API in the header.\n"
        "\n\\*\\*\\* **CLI** -- counts `amd-smi` command invocations, not APIs, so"
        " it has its own scale. `cli_surface.py` walks the real argparse tree in"
        " `amdsmi_cli/amdsmi_parser.py` and sums base commands, valid"
        " option/format combinations, and negative cases. It forces the"
        " device-presence probes on, so the scale reflects the whole product"
        " rather than the devices in this host.\n"
    )

    # Tidy long format: one row per framework/metric so counts, percentages and
    # goals never share a column and the file pivots cleanly.
    # ---- Untested detail ---------------------------------------------------- #
    out.append("\n## Untested APIs\n")
    for name in SUBSYSTEM_ORDER:
        missing = sorted(untested[name])
        out.append(f"### {name} ({len(missing)})")
        out.extend(f"  - {func}()" for func in missing)
        out.append("")

    report = "\n".join(out)
    print(report)

    # ---- Artifacts ---------------------------------------------------------- #
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "_api_coverage.md").write_text(report + "\n")

    # _api_coverage.csv/.tsv mirror _api_coverage.md table for table, so opening
    # or pasting one reproduces the report's layout. Each block carries its own
    # title and header row, separated by a blank line; markdown decoration is
    # stripped so the cells hold plain values.
    def plain(cell):
        # Order matters: strip markdown bold first (adjacent '**'), then unescape
        # the literal footnote markers ('\*'), which are not adjacent stars.
        text = str(cell)
        for old, new in (("<br>", " "), ("**", ""), ("\\*", "*"), ("`", ""), ("\t", " ")):
            text = text.replace(old, new)
        return text

    blocks = [
        ([f"AMD SMI API Coverage - {meta['run_date']}"], []),
        (["Run metadata"], env_rows),
        (["Test suites"], suite_rows),
        ([f"API Tests - coverage by subsystem (goal {SUBSYSTEM_GOAL_PCT}%)"], subsys_rows),
        ([f"Coverage by framework (of {num_api} APIs in {args.amdsmi.name})"], fw_rows),
        (["Scale and goal per column"], legend),
    ]
    coverage_csv = []
    for title, table in blocks:
        coverage_csv.append(title)
        coverage_csv.extend([plain(c) for c in row] for row in table)
        coverage_csv.append([])
    width = max(len(row) for row in coverage_csv)
    coverage_csv = [row + [""] * (width - len(row)) for row in coverage_csv]
    # utf-8-sig: the BOM makes Excel read the arrows as UTF-8 instead of latin-1.
    with (args.log_dir / "_api_coverage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(coverage_csv)

    # Same tables, tab-separated. Excel splits pasted clipboard text on tabs and
    # never on commas, so this is the copy/paste form; the .csv is the open-it
    # form. Tabs also remove the need to quote cells containing commas.
    with (args.log_dir / "_api_coverage.tsv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle, delimiter="\t").writerows(coverage_csv)

    if not args.no_history:
        write_history(history_path, inherited, meta)
        print(f"\nhistory -> {history_path} ({len(inherited) + 1} runs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
