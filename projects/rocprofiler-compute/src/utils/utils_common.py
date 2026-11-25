# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import copy
import csv
import errno
import io
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import uuid
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Match,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import config
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
)
from vendored import yaml

# Global constants
NS_TO_MS = 1.0 / 1_000_000.0
METRIC_ID_RE = re.compile(pattern=r"^\d{1,2}(?:\.\d{1,2}){0,2}$")
# Metric library id format: {category}.{subcategory}_{name}, in lowercase with
# underscores.
METRIC_LIBRARY_ID_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z0-9]+(_[a-z0-9]+)*$")
PC_SAMPLING_BLOCK_IDS = ("21", "pc_sampling")
# Block 4 and its alias; ids such as 4.1 are roofline metrics
ROOFLINE_BLOCK_IDS = ("4", "roof")
PROFILE_OUTPUT_FORMAT = "rocpd"
# Profile names the native tool's per-pid CSVs <prefix>_<fbase>_<pid>.csv.gz
# in the workload directory, and analyze reads them back by these prefixes.
NATIVE_COUNTERS_PREFIX = "counters"
NATIVE_DISPATCH_PREFIX = "dispatch"
NATIVE_KERNEL_SYMBOLS_PREFIX = "kernel_symbols"
NATIVE_AGENTS_PREFIX = "agents"
# Panel id of block 30, Memory Bandwidth Analysis
MEMBW_ANALYSIS_PANEL_ID: int = 3000
# Panel id of block 3, Memory Chart, drawn as the memory chart in the CLI
MEMORY_CHART_PANEL_ID: int = 300
# Panel id of block 4, Roofline, drawn as the per-kernel roofline layout in the CLI
ROOFLINE_PANEL_ID: int = 400
# --list-metrics value that lists every metric
LIST_ALL_METRICS = "all"
# The metric_table key that expands one row into a row per channel, e.g. per L2 channel
PLACEHOLDER_RANGE_KEY = "placeholder_range"
# Kinds of view tables, which are also the panel configs' data source keys
METRIC_TABLE = "metric_table"
RAW_CSV_TABLE = "raw_csv_table"
# File name of a view: its panel id and a name, e.g. 0300_memory_chart.yaml
VIEW_FILE_RE = re.compile(r"^(\d\d00)_[a-z0-9_]+\.yaml$")
# The panel --list-stats shows.
TOP_STATS_PANEL_ID = 0
# The views every arch has, by panel id. They show the workload's CSV files,
# not metrics.
BUILT_IN_VIEWS: Dict[int, Dict[str, Any]] = {
    TOP_STATS_PANEL_ID: {
        "view": "Top Stats",
        "alias": "topstats",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "Top Kernels",
                "source": "pmc_kernel_top.csv",
            },
            {
                "render": RAW_CSV_TABLE,
                "title": "Dispatch List",
                "source": "pmc_dispatch_info.csv",
            },
        ],
    },
    100: {
        "view": "System Info",
        "alias": "sysinfo",
        "tables": [
            {
                "render": RAW_CSV_TABLE,
                "title": "System Info",
                "source": "sysinfo.csv",
                "columnwise": True,
            }
        ],
    },
}
# Narrowest the --list-metrics Description column gets on a narrow terminal
DESCRIPTION_WIDTH_MIN = 40

# Shared suffix for invalid --block errors.
INVALID_BLOCK_HINT = (
    "\n\tRun rocprof-compute --list-metrics to see all block ids/aliases."
)


def is_gfx9(gpu_arch: Optional[str]) -> bool:
    """Return True if gpu_arch is a gfx9xx (CDNA) architecture."""
    return bool(gpu_arch and gpu_arch.startswith("gfx9"))


def is_gfx115x(gpu_arch: Optional[str]) -> bool:
    """Return True if gpu_arch is a gfx115x (RDNA 3.5 APU) architecture."""
    return bool(gpu_arch and gpu_arch.startswith("gfx115"))


def is_gfx1250(gpu_arch: Optional[str]) -> bool:
    """Return True if gpu_arch is a gfx1250 architecture."""
    return gpu_arch == "gfx1250"


def canonical_config_arch(gpu_arch: Optional[str]) -> Optional[str]:
    """Map a GPU arch to the directory name its metric library and views use, gfx115x
    for gfx1150 to gfx1153."""
    if gpu_arch is None:
        return None
    if is_gfx115x(gpu_arch):
        return "gfx115x"
    return gpu_arch


# Supported expression field names for metric tables
SUPPORTED_FIELD: List[str] = [
    "Value",
    "Minimum",
    "Maximum",
    "Average",
    "Median",
    "Min",
    "Max",
    "Avg",
    "Peak",
    "Peak (Empirical)",
    "Count",
    "Mean",
    "Percent",
    "Std Dev",
    "Q1",
    "Q3",
    "Expression",
    "Percent of Peak",
    # Special keywords for L2 channel
    "L2 Cache Hit Rate",
    "Requests",
    "L2 Read",
    "L2 Write",
    "L2 Atomic",
    "L2-Fabric Requests",
    "L2-Fabric Read",
    "L2-Fabric Write and Atomic",
    "L2-Fabric Atomic",
    "L2 Read Req",
    "L2 Write Req",
    "L2 Atomic Req",
    "L2-Fabric Read Req",
    "L2-Fabric Write and Atomic Req",
    "L2-Fabric Atomic Req",
    "L2-Fabric Read Latency",
    "L2-Fabric Write Latency",
    "L2-Fabric Atomic Latency",
    "L2-Fabric Read Stall (PCIe)",
    "L2-Fabric Read Stall (Infinity Fabric™)",
    "L2-Fabric Read Stall (HBM)",
    "L2-Fabric Write Stall (PCIe)",
    "L2-Fabric Write Stall (Infinity Fabric™)",
    "L2-Fabric Write Stall (HBM)",
    "L2-Fabric Write Starve",
]

# Global state: rocprof being used by profile mode
_rocprof_cmd = ""


def get_rocprof_cmd() -> str:
    """Get the current rocprof command."""
    return _rocprof_cmd


def set_rocprof_cmd(cmd: str) -> None:
    """Set the rocprof command."""
    global _rocprof_cmd
    _rocprof_cmd = cmd


def version_to_numeric(version_parts: List[int], max_len: int) -> int:
    """Convert version tuple to numeric value using base-1000 positional system."""
    version_numeric = 0
    for i, part in enumerate(version_parts):
        version_numeric += part * (1000 ** (max_len - i - 1))
    return version_numeric


def resolve_rocm_library_path(library_path: Optional[str]) -> Optional[str]:
    """
    Resolve ROCm library path with automatic version fallback.
    Tries exact path first, then falls back to versioned variants
    (e.g., .so.1, .so.1.2.3).
    """
    if not library_path:
        return library_path

    path = Path(library_path)

    # Try exact path first (handles both unversioned and explicit versioned paths)
    if path.exists():
        console_debug(f"Resolved library (exact match): {path}")
        return str(path)

    # Use iterdir to avoid glob metacharacter issues in library_path.
    if not path.parent.is_dir():
        return None
    prefix = f"{path.name}."
    matches = [str(p) for p in path.parent.iterdir() if p.name.startswith(prefix)]

    # First pass: filter to numeric versions and collect version tuples
    version_tuples: list[tuple[list[int], str]] = []
    for candidate in matches:
        # Compute the suffix relative to the requested library path.
        if not candidate.startswith(library_path):
            continue
        suffix = candidate[len(library_path) :]
        # Expect a suffix like ".1" or ".1.2.3"
        if not suffix.startswith("."):
            continue
        parts = suffix.split(".")[1:]  # drop leading empty element
        if not parts:
            continue
        if not all(part.isdigit() for part in parts):
            continue
        version_tuples.append(([int(p) for p in parts], candidate))

    # Find max version length to normalize all versions
    if not version_tuples:
        console_debug(f"ROCm library .so file not found: {library_path}")
        return library_path

    # Second pass: convert to numeric values with normalized length
    max_version_len = max(len(vt[0]) for vt in version_tuples)
    versioned_candidates: list[tuple[int, str]] = []
    for version_parts, candidate in version_tuples:
        version_numeric = version_to_numeric(version_parts, max_version_len)
        versioned_candidates.append((version_numeric, candidate))

    # Select the candidate with the highest numeric version.
    versioned_candidates.sort(key=lambda item: item[0], reverse=True)
    resolved = versioned_candidates[0][1]
    console_debug(f"Resolved library (versioned): {library_path} -> {resolved}")
    return resolved


def is_tcc_channel_counter(counter: str) -> bool:
    return counter.startswith("TCC") and counter.endswith("]")


def add_counter_extra_config_input_yaml(
    data: Dict[str, Any],
    counter_name: str,
    description: str,
    expression: str,
    architectures: List[str],
    properties: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Add a new counter to the rocprofiler-sdk dictionary.
    Initialize missing parts if data is empty or incomplete.
    Enforces that 'architectures' and 'properties' are lists
    for correct YAML list serialization.
    Overwrites the counter if it already exists.

    Args:
        data (dict): The loaded YAML dictionary (can be empty).
        counter_name (str): The name of the new counter.
        description (str): Description of the new counter.
        architectures (list): List of architectures for the definitions.
        expression (str): Expression string for the counter.
        properties (list, optional): Optional list of properties, default to empty list.

    Returns:
        dict: Updated YAML dictionary.
    """
    if properties is None:
        properties = []

    # Enforce type checks for YAML list serialization
    if not isinstance(architectures, list):
        raise TypeError(
            f"'architectures' must be a list, got {type(architectures).__name__}"
        )
    if not isinstance(properties, list):
        raise TypeError(f"'properties' must be a list, got {type(properties).__name__}")

    # Initialize the top-level 'rocprofiler-sdk' dict if missing
    if "rocprofiler-sdk" not in data or not isinstance(data["rocprofiler-sdk"], dict):
        data["rocprofiler-sdk"] = {}

    sdk = data["rocprofiler-sdk"]

    # Initialize schema version if missing
    if "counters-schema-version" not in sdk:
        sdk["counters-schema-version"] = 1

    # Initialize counters list if missing or not a list
    if "counters" not in sdk or not isinstance(sdk["counters"], list):
        sdk["counters"] = []

    # Build the new counter dictionary
    new_counter = {
        "name": counter_name,
        "description": description,
        "properties": properties,
        "definitions": [
            {
                "architectures": architectures,
                "expression": expression,
            }
        ],
    }

    # Check if the counter already exists and overwrite if found
    for idx, counter in enumerate(sdk["counters"]):
        if counter.get("name") == counter_name:
            sdk["counters"][idx] = new_counter
            break
    else:
        # Not found, append new counter
        sdk["counters"].append(new_counter)

    return data


def get_version(rocprof_compute_home: Path) -> Dict[str, str]:
    """Return ROCm Compute Profiler versioning info"""

    # semantic version info - note that version file(s) can reside in
    # two locations depending on development vs formal install
    search_dirs = [rocprof_compute_home, rocprof_compute_home.parent]
    found = False
    version_dir: Optional[Path] = None
    VER = "unknown"
    SHA = "unknown"
    MODE = "unknown"

    for directory in search_dirs:
        version_file = directory / "VERSION"
        try:
            with open(version_file, encoding="utf-8") as file:
                VER = file.read().replace("\n", "")
                found = True
                version_dir = directory
                break
        except Exception:
            pass
    if not found:
        console_error(f"Cannot find VERSION file at {search_dirs}")

    # git version info
    if version_dir is not None:
        try:
            success, output = capture_subprocess_output(
                ["git", "-C", version_dir, "log", "--pretty=format:%h", "-n", "1"],
            )
            if success:
                SHA = output
                MODE = "dev"
            else:
                raise Exception(output)
        except Exception:
            try:
                sha_file = version_dir / "VERSION.sha"
                with open(sha_file, encoding="utf-8") as file:
                    SHA = file.read().replace("\n", "")
                    MODE = "release"
            except Exception:
                pass

    return {"version": VER, "sha": SHA, "mode": MODE}


def get_version_display(version: str, sha: str, mode: str) -> str:
    """Pretty print versioning info"""
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print(f"rocprofiler-compute version: {version} ({mode})", file=buf)
    print(f"Git revision:     {sha}", file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()


def detect_rocprof(args: argparse.Namespace) -> str:
    """Detect loaded rocprof version. Resolve path and set cmd globally."""
    # Default is rocprofiler-sdk
    if os.environ.get("ROCPROF", "rocprofiler-sdk") == "rocprofiler-sdk":
        if not Path(args.rocprofiler_sdk_tool_path).exists():
            console_error(
                "Could not find rocprofiler-sdk tool at "
                f"{args.rocprofiler_sdk_tool_path}"
            )
        set_rocprof_cmd("rocprofiler-sdk")
        console_debug(f"rocprof_cmd is {get_rocprof_cmd()}")
        console_debug(f"rocprofiler_sdk_tool_path is {args.rocprofiler_sdk_tool_path}")
    else:
        # If ROCPROF is not set to rocprofiler-sdk
        set_rocprof_cmd(os.environ["ROCPROF"])
        rocprof_path = shutil.which(get_rocprof_cmd())
        if not rocprof_path:
            console_error(
                f"Unable to resolve path to {get_rocprof_cmd()} binary. "
                "Please verify installation or set ROCPROF "
                "environment variable with full path."
            )
        rocprof_path = str(Path(rocprof_path.rstrip("\n")).resolve())
        console_debug(f"rocprof_cmd is {get_rocprof_cmd()}")
        console_debug(f"ROC Profiler: {rocprof_path}")
    return get_rocprof_cmd()


def capture_subprocess_output(
    subprocess_args: List[str],
    new_env: Optional[Dict[str, str]] = None,
    profileMode: bool = False,
) -> Tuple[bool, str]:
    sanitized_env = (
        None
        if new_env is None
        else {
            k: ":".join(str(i) for i in v) if isinstance(v, list) else str(v)
            for k, v in new_env.items()
        }
    )

    # Use a PTY in profile mode to prevent instrumentation output from
    # being interleaved with workload output.
    if profileMode:
        pty_parent_fd, pty_child_fd = os.openpty()
        stdout_arg = pty_child_fd
        stderr_arg = pty_child_fd
    else:
        stdout_arg = subprocess.PIPE
        stderr_arg = subprocess.STDOUT

    # env=None is Popen's default (inherit parent env).
    try:
        process = subprocess.Popen(
            subprocess_args,
            stdin=subprocess.PIPE,
            stdout=stdout_arg,
            stderr=stderr_arg,
            universal_newlines=True,
            errors="replace",
            env=sanitized_env,
        )
    except BaseException:
        # Close PTY fds if Popen fails.
        if profileMode:
            os.close(pty_parent_fd)
            os.close(pty_child_fd)
        raise

    if profileMode:
        # Close the child end; parent only reads. errors="replace" skips
        # bad bytes instead of crashing the read loop.
        os.close(pty_child_fd)
        process_stdout = os.fdopen(pty_parent_fd, "r", errors="replace")
    else:
        process_stdout = process.stdout

    # Create buffer for captured process output
    buf = io.StringIO()

    def forward_input() -> None:
        """
        Forward the keyboard input from the terminal to the inside subprocess
        """

        try:
            sys.stdin.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # Stdin can't be used in select; skip input forwarding
            return

        if sys.stdin.isatty():
            for line in sys.stdin:
                if process.poll() is not None:
                    break
                process.stdin.write(line)
                process.stdin.flush()
        else:
            while process.poll() is None:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                except (io.UnsupportedOperation, AttributeError):
                    break
                if rlist:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    process.stdin.write(line)
                    process.stdin.flush()
        try:
            process.stdin.close()
        except Exception:
            console_warning("forward_input: the stdin did not close properly!")

    input_thread = threading.Thread(target=forward_input, daemon=True)
    input_thread.start()

    # Read until the child closes its end. Pipes signal EOF with an empty
    # string; PTYs signal it with OSError(EIO). The two never overlap.
    while True:
        try:
            line = process_stdout.readline()
        except OSError as e:
            if e.errno == errno.EIO:
                break
            raise
        if not line:
            break
        buf.write(line)
        if profileMode:
            console_log(get_rocprof_cmd(), line.strip(), indent_level=1)
        else:
            console_log(line.strip())

    process_stdout.close()
    input_thread.join(timeout=1)

    # Get process return code
    return_code = process.wait()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return success, output


def parse_pmc_perf(pmc_perf_file: str) -> List[str]:
    """
    Parse the YAML file to get the pmc counters.
    Assumes only one job per file.
    """
    with open(pmc_perf_file, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    jobs = data.get("jobs", [])
    if not jobs:
        return []
    return jobs[0].get("pmc") or []


def selects_roofline(filter_blocks: Optional[List[str]]) -> bool:
    """Return True if nothing is selected (everything) or a block is 4, 4.x or roof."""
    return not filter_blocks or any(
        str(block) in ROOFLINE_BLOCK_IDS or str(block).startswith("4.")
        for block in filter_blocks
    )


def is_only_pc_sampling(filter_blocks: List[str]) -> bool:
    """Return True if all requested blocks are PC sampling (block 21 or the
    ``pc_sampling`` alias)."""
    return bool(filter_blocks) and all(
        block in PC_SAMPLING_BLOCK_IDS for block in filter_blocks
    )


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts: list[str] = []

    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    if len(parts) <= 1:
        return parts[0] if parts else "0 seconds"

    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def parse_sets_yaml(arch: str) -> Dict[str, Any]:
    config_arch = canonical_config_arch(arch) or arch
    filename = (
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sets"
        / f"{config_arch}_sets.yaml"
    )
    with open(filename, encoding="utf-8") as file:
        content = file.read()
    data = yaml.safe_load(content)

    sets_data = data.get("sets", [])

    sets_info: dict[str, Any] = {}
    for set_item in sets_data:
        set_option = set_item.get("set_option", "")
        if set_option:
            sets_info[set_option] = set_item
    return sets_info


def metric_tables_by_id(
    panel_configs: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Return each metric table of an arch's panel configs, by table id."""
    return {
        source[METRIC_TABLE]["id"]: source[METRIC_TABLE]
        for panel_config in panel_configs.values()
        for source in panel_config["data source"]
        if METRIC_TABLE in source
    }


def expand_placeholder_ranges(
    panel_configs: OrderedDict[int, Dict[str, Any]],
    sys_info: Optional[Dict[str, Any]],
) -> OrderedDict[int, Dict[str, Any]]:
    """
    Expand placeholder_range entries in metric_table data configs in-place.

    Mutates panel_configs directly and returns the same object for convenience.

    Some metric tables define a range of metrics via a placeholder key that is
    expanded into individual entries at load time. sys_info is required to
    resolve built-in range variables; if it is None the table is cleared.
    """
    for _panel_id, panel in panel_configs.items():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if type_key != "metric_table":
                    continue
                if "metric" not in data_config:
                    continue
                if PLACEHOLDER_RANGE_KEY not in data_config["metric"]:
                    continue
                if sys_info is None:
                    data_config["metric"] = {}
                    continue

                # Resolved here (not in eval_metric) because the range may
                # itself be a built-in var. Single placeholder only.
                p_range = data_config["metric"].pop(PLACEHOLDER_RANGE_KEY)
                metric, metric_expr = data_config["metric"].popitem()
                new_metrics: dict[str, Any] = {}
                for p, r in p_range.items():
                    if isinstance(r, int):
                        var = r
                    elif isinstance(r, str) and r.startswith("$total_l2_chan"):
                        var = int(sys_info["total_l2_chan"])
                    else:
                        console_error(f'Built-in var "{r}" is not supported')
                    for i in range(var):
                        new_key = metric.replace(p, str(i))
                        new_val = {
                            k: v.replace(p, str(i)) for k, v in metric_expr.items()
                        }
                        new_metrics[new_key] = new_val
                data_config["metric"] = new_metrics

    return panel_configs


def _metric_has_valid_expr(entries: dict, data_config: dict) -> bool:
    """
    Return True if a metric entry has at least one evaluatable expression field
    that is not None and not the string "None".

    Expression fields are identified by matching the header display name against
    SUPPORTED_FIELD, excluding Peak-prefixed fields which are empirical values.
    """
    for header_key, header_display in data_config["header"].items():
        if header_display in SUPPORTED_FIELD and not header_display.startswith("Peak"):
            expr_value = entries.get(header_key)
            if expr_value is not None and expr_value != "None":
                return True
    return False


def build_metric_details(
    panel_configs: Mapping[int, Dict[str, Any]],
    sys_info: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[str, str, str]]:
    """Map panel, table and metric ids to (name, unit, description). Without
    sys_info, per-channel tables have no metrics."""
    expanded_configs = expand_placeholder_ranges(panel_configs, sys_info)
    return dict(_iter_metric_list_rows(expanded_configs))


def panel_id_of(table_id: int) -> int:
    """Return the panel id of a table config id, e.g. 1802 -> 1800."""
    return table_id // 100 * 100


def format_table_id(config_id: int) -> str:
    """Return the "block.table" id of a table config id, e.g. 1802 -> "18.2"."""
    return f"{config_id // 100}.{config_id % 100}"


def format_row_id(config_id: int, index: int) -> str:
    """Return the "block.table.index" id of a table config's row, e.g. 1802, 3 ->
    "18.2.3"."""
    return f"{format_table_id(config_id)}.{index}"


def metric_ids_by_name(data_config: Dict[str, Any]) -> Dict[str, str]:
    """Map each metric name of a metric_table config to its "block.table.index" id."""
    metrics = data_config.get("metric") or {}
    return {
        name: format_row_id(data_config["id"], index)
        for index, name in enumerate(metrics)
    }


def resolve_block_token(token: str, alias_map: Dict[str, str]) -> Optional[str]:
    """Return the metric id for a block or metric id or alias, or None if unknown."""
    if METRIC_ID_RE.match(token):
        return token
    return alias_map.get(token)


def select_listed_ids(
    metric_ids: List[str],
    selection: List[str],
    alias_to_block_id: Dict[str, str],
    listable_block_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Return the ids to list: the blocks when nothing is selected, every id for
    "all", else each selected id with its parents and children. listable_block_ids
    limits the result to those blocks."""
    if listable_block_ids is not None:
        metric_ids = [
            metric_id
            for metric_id in metric_ids
            if metric_id.split(".")[0] in listable_block_ids
        ]
    if not selection:
        return [metric_id for metric_id in metric_ids if "." not in metric_id]
    if selection == [LIST_ALL_METRICS]:
        return metric_ids
    if LIST_ALL_METRICS in selection:
        console_error(
            f'"{LIST_ALL_METRICS}" lists every metric; pass it without other ids.'
        )

    selected_ids = []
    for token in selection:
        selected_id = resolve_block_token(token, alias_to_block_id)
        if selected_id not in metric_ids:
            console_error(f"Unknown block or metric id {token!r}.{INVALID_BLOCK_HINT}")
        selected_ids.append(selected_id)
    return [
        metric_id
        for metric_id in metric_ids
        if any(_is_same_id_branch(metric_id, selected) for selected in selected_ids)
    ]


def format_block_table(
    metric_details: Dict[str, Tuple[str, str, str]],
    block_ids: List[str],
    block_aliases: Dict[str, str],
) -> str:
    """Format the Block_ID, Alias and Block table of --list-metrics."""
    rows = [
        {
            "Block_ID": block_id,
            "Alias": block_aliases.get(block_id, ""),
            "Block": metric_details[block_id][0],
        }
        for block_id in block_ids
    ]
    return format_table_ascii(rows, ["Block_ID", "Alias", "Block"], show_index=False)


def format_metric_tables(
    metric_details: Dict[str, Tuple[str, str, str]],
    listed_ids: List[str],
    block_aliases: Dict[str, str],
    terminal_width: int,
) -> str:
    """Format a heading per listed block and a table per listed table."""
    lines: List[str] = []
    for block_id in [metric_id for metric_id in listed_ids if "." not in metric_id]:
        heading = f"{block_id}. {metric_details[block_id][0]}"
        if block_aliases.get(block_id):
            heading += f" (alias: {block_aliases[block_id]})"
        block_tables = [
            _format_listed_table(table_id, metric_details, listed_ids, terminal_width)
            for table_id in listed_ids
            if table_id.count(".") == 1 and table_id.split(".")[0] == block_id
        ]
        block_tables = [table for table in block_tables if table]
        lines.extend([heading, *(block_tables or ["No metrics to list."]), ""])
    return "\n".join(lines)


def get_uuid() -> str:
    return uuid.uuid4().hex[:8]


def convert_filter_blocks_to_panel_ids(
    filter_blocks: List[str], arch: Optional[str] = None
) -> Set[int]:
    """Return the panel ids (e.g. {200, 1100}) that -b tokens select.

    Numeric ids ("2", "11.1") select their panel, also when the views leave their row
    out, so workloads saved by older versions keep their panels. Metric library ids
    select the panels that show the metric; other tokens are arch's panel aliases.
    """
    if arch:
        filter_blocks = [
            placement
            for bid in map(str, filter_blocks)
            for placement in (
                metric_placements(bid, arch)
                if METRIC_LIBRARY_ID_RE.match(bid)
                else [bid]
            )
        ]
    alias_map: dict[str, str] = (
        get_arch_alias_to_panel_id(arch)
        if arch and any(not METRIC_ID_RE.match(str(bid)) for bid in filter_blocks)
        else {}
    )
    resolved: set[int] = set()
    for bid in filter_blocks:
        metric_id = resolve_block_token(str(bid), alias_map)
        if metric_id is None:
            console_error(f"Invalid --block value {str(bid)!r}.{INVALID_BLOCK_HINT}")
        resolved.add(int(convert_metric_id_to_panel_info(metric_id)[0]))
    return resolved


def convert_metric_id_to_panel_info(
    metric_id: str,
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Convert metric id into panel information.
    Output is a tuples of the form (file_id, panel_id, metric_id).

    For example:

    Input: "2"
    Output: ("0200", None, None)

    Input: "11"
    Output: ("1100", None, None)

    Input: "11.1"
    Output: ("1100", 1101, None)

    Input: "11.1.1"
    Output: ("1100", 1101, 1)

    Raises exception for invalid metric id.
    """
    tokens = metric_id.split(".")
    if not (0 < len(tokens) < 4):
        raise ValueError(f"Invalid metric id: {metric_id}")

    # File id
    file_id = str(int(tokens[0]))
    # 4 -> 04
    if len(file_id) == 1:
        file_id = f"0{file_id}"
    # Multiply integer by 100
    file_id = f"{file_id}00"

    # Panel id
    panel_id = None
    if len(tokens) > 1:
        panel_id = int(tokens[0]) * 100 + int(tokens[1])

    # Metric id
    metric_id_int = None
    if len(tokens) > 2:
        metric_id_int = int(tokens[2])

    return (file_id, panel_id, metric_id_int)


def load_yaml(filepath: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load YAML file and return as dictionary."""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mapping_field_errors(
    entry: Dict[str, Any], required: Tuple[str, ...], optional: Tuple[str, ...]
) -> List[str]:
    """Report each missing required field and each unknown field of a YAML mapping."""
    errors = [f"'{field}' is required" for field in required if field not in entry]
    errors.extend(
        f"'{field}' is not a known field"
        for field in entry
        if field not in required and field not in optional
    )
    return errors


def read_view_files(arch: str) -> Dict[Path, Any]:
    """Return arch's view file documents by path, in file name order."""
    arch_dir = config.analysis_views_dir / canonical_config_arch(arch)
    return {path: load_yaml(path) for path in sorted(arch_dir.glob("*.yaml"))}


def load_views(arch: str) -> "OrderedDict[int, Dict[str, Any]]":
    """Return arch's views, built-in ones included, sorted by panel id."""
    return views_from_files(read_view_files(arch))


def views_from_files(documents: Dict[Path, Any]) -> "OrderedDict[int, Dict[str, Any]]":
    """Return the built-in views and those of read_view_files(), sorted by panel id."""
    views = copy.deepcopy(BUILT_IN_VIEWS)
    views.update((view_panel_id(path.name), view) for path, view in documents.items())
    return OrderedDict(sorted(views.items()))


def view_panel_id(file_name: str) -> int:
    """Return the panel id of a view file name, e.g. 300 for 0300_memory_chart.yaml."""
    match = VIEW_FILE_RE.match(file_name)
    if match is None:
        raise ValueError(f"{file_name}: not a view file name")
    return int(match.group(1))


def panel_aliases(views: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
    """Return panel number -> -b alias, "" for none, of an arch's views."""
    return {
        str(panel_id // 100): view.get("alias") or ""
        for panel_id, view in views.items()
    }


def iter_view_rows(
    views: Dict[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield each metric table row of an arch's views with its numeric id."""
    for panel_id, view in views.items():
        for table_id, table in numbered_view_tables(panel_id, view):
            if view_table_render(table) != METRIC_TABLE:
                continue
            for index, row in enumerate(table["metrics"]):
                yield format_row_id(table_id, index), row


def row_metric_ids(row: Dict[str, Any]) -> List[str]:
    """Return the metric ids a row shows: its metric, or each column's."""
    if "metric" in row:
        return [row["metric"]]
    return [metric_id for metric_id in row.get("columns", {}).values() if metric_id]


def view_table_render(table: Dict[str, Any]) -> str:
    """Return the kind of a view table: its render, a metric table by default."""
    return table.get("render", METRIC_TABLE)


def numbered_view_tables(
    panel_id: int, view: Dict[str, Any]
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield each table of a view with its table id: the panel id plus its position."""
    for index, table in enumerate(view["tables"]):
        yield panel_id + index + 1, table


def has_views(arch: str) -> bool:
    """Return whether arch has analysis views, as every arch with a metric library
    has."""
    return (config.analysis_views_dir / canonical_config_arch(arch)).is_dir()


def get_arch_panel_id_to_alias(arch: str) -> Dict[str, str]:
    """Return panel_aliases() of arch's views; {} for an arch without views."""
    if not has_views(arch):
        return {}
    return panel_aliases(load_views(arch))


def get_arch_alias_to_panel_id(arch: str) -> Dict[str, str]:
    """Return -b alias -> panel number, for the panels that have an alias."""
    return {
        alias: pid for pid, alias in get_arch_panel_id_to_alias(arch).items() if alias
    }


def metric_placements(metric_id: str, arch: str) -> List[str]:
    """Return the "block.table.row" ids of arch's rows that show metric_id, in view
    order; exit with an error for an arch without views or an unknown metric_id."""
    if not has_views(arch):
        console_error(f"Metric ids such as {metric_id!r} are not supported on {arch}.")
    placements = [
        position
        for position, row in iter_view_rows(load_views(arch))
        if metric_id in row_metric_ids(row)
    ]
    if not placements:
        console_error(
            f"Unknown metric id {metric_id!r} for {canonical_config_arch(arch)}."
            f"{INVALID_BLOCK_HINT}"
        )
    return placements


def expand_metric_ids(tokens: List[str], arch: str) -> List[str]:
    """Return -b tokens with each metric library id replaced by the numeric ids of
    arch's rows that show it."""
    expanded: List[str] = []
    for token in tokens:
        if METRIC_LIBRARY_ID_RE.match(token):
            expanded.extend(metric_placements(token, arch))
        else:
            expanded.append(token)
    return expanded


def row_ids_as_metric_ids(tokens: List[str], arch: str) -> List[str]:
    """Return -b tokens, without duplicates, with each numeric row id replaced by its
    row's metric ids, so a saved workload keeps its rows when later views renumber them.
    """
    rows = _rows_by_position(arch)
    replaced: List[str] = []
    for token in tokens:
        for new in (
            row_metric_ids(rows[token])
            if _is_row_id(token) and token in rows
            else [token]
        ):
            if new not in replaced:
                replaced.append(new)
    return replaced


def warn_numeric_metric_ids(tokens: List[str], arch: str) -> None:
    """Warn that each numeric row id in tokens is deprecated, naming its metric ids."""
    rows = _rows_by_position(arch)
    for token in tokens:
        if not _is_row_id(token) or token not in rows:
            continue
        hint = " or ".join(repr(metric_id) for metric_id in row_metric_ids(rows[token]))
        console_warning(
            f"Numeric metric id {token!r} is deprecated. Use {hint} instead."
        )


def get_job_rank_and_size() -> Tuple[Optional[str], Optional[int]]:
    """Detect job rank and total ranks from runtime environment variables.

    Returns a (rank, total_ranks) tuple, ensuring both values come from the same
    runtime.
    """
    # Note: PMIX_RANK is intentionally excluded. PMIx has no standard size env var,
    # and PMIx is never a standalone launcher — it always runs behind SLURM, OpenMPI,
    # PALS, etc., which set their own paired rank/size vars checked here. If only
    # PMIX_RANK is set, the launcher also sets generic MPI_RANK/MPI_SIZE caught below.
    rank_size_env_vars = [
        ("PBS_NODENUM", "PBS_O_TASKNUM"),  # PBS/Torque
        ("SLURM_PROCID", "SLURM_NTASKS"),  # SLURM
        ("FLUX_TASK_RANK", "FLUX_JOB_SIZE"),  # Flux
        ("PMI_RANK", "PMI_SIZE"),  # PMI
        ("PALS_RANKID", "PALS_WORLD_SIZE"),  # PALS (HPE Cray)
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),  # OpenMPI
        ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),  # MVAPICH2
        ("MPI_RANKID", "MPI_NRANKS"),  # Generic
        ("MPI_LOCALRANKID", "MPI_LOCALNRANKS"),  # Generic (local)
        ("MPI_RANK", "MPI_SIZE"),  # Generic
    ]
    matched_rank = None
    matched_rank_var = None
    matched_size = None
    matched_size_var = None
    for rank_var, size_var in rank_size_env_vars:
        rank_value = os.environ.get(rank_var)
        try:
            _ = int(rank_value)
        except (TypeError, ValueError):
            continue

        # Rank is valid; try to get a matching size for a complete pair
        size_value = os.environ.get(size_var)
        try:
            matched_size = int(size_value)
        except (TypeError, ValueError):
            # Size missing or invalid — remember rank as fallback but keep
            # searching for a runtime that provides both rank and size
            if matched_rank is None:
                matched_rank = rank_value
                matched_rank_var = rank_var
                matched_size_var = size_var
            continue

        # Complete pair found
        matched_rank = rank_value
        matched_rank_var = rank_var
        matched_size_var = size_var
        break

    console_debug(
        f"Parallel runtime detected: {matched_rank_var}='{matched_rank}',"
        f" {matched_size_var}={matched_size}"
    )

    return (matched_rank, matched_size)


def replace_rank(name: str) -> str:
    def rank(match: Match[str]) -> str:
        value, _ = get_job_rank_and_size()
        if value is not None:
            return value + match.group(1)  # preserve trailing slash
        else:
            return ""  # Ignore %rank% and trailing slash

    # Replace %rank% (and optional trailing slash) with MPI process rank
    pattern = re.compile(r"%rank%(/?)")

    return pattern.sub(rank, name)


def replace_env(name: str) -> str:
    def env(match: Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")  # Default to empty string if not found

    # Replace %env{VAR}% with environment variable values
    pattern = re.compile(r"%env{([^}]+)}%")

    return pattern.sub(env, name)


def normalize_filter_to_str_list(value: Any) -> List[str]:  # noqa: ANN401
    """Normalize a filter value (scalar or list) to a list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def print_status(msg: str) -> None:
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def create_temp_rocprofiler_metrics_path(
    sdk_config: Dict[str, Any], parent_dir: Path = Path("/tmp")
) -> str:
    """Create a temporary directory with the rocprofiler-sdk counter definitions.

    The directory is created in parent_dir. It holds config.yaml with the full
    config, and counter_defs.yaml without firmware restrictions for older
    rocprofiler-sdk versions. Return the path of the directory.
    """
    tmpfile_parent = Path(
        tempfile.mkdtemp(prefix="rocprof_compute_sdk_config_", dir=parent_dir)
    )

    # counter_defs.yaml is for backward compatibility with previous versions of sdk
    tmpfile_path_old = tmpfile_parent / "counter_defs.yaml"
    # Current version of sdk uses config.yaml instead of counter_defs.yaml
    tmpfile_path_new = tmpfile_parent / "config.yaml"

    with open(tmpfile_path_old, "w", encoding="utf-8") as tmpfile:
        # Old sdk does not support firmware restrictions
        sdk_config_old = {
            **sdk_config,
            "rocprofiler-sdk": {
                k: v
                for k, v in sdk_config["rocprofiler-sdk"].items()
                if k not in {"fw-restriction-schema-version", "firmware_restrictions"}
            },
        }
        yaml.dump(sdk_config_old, tmpfile, default_flow_style=False, sort_keys=False)

    with open(tmpfile_path_new, "w", encoding="utf-8") as tmpfile:
        yaml.dump(sdk_config, tmpfile, default_flow_style=False, sort_keys=False)

    return str(tmpfile_parent)


def reconfigure_stdio_utf8() -> None:
    """Force sys.stdout and sys.stderr to UTF-8 so tabulated output never
    crashes with UnicodeEncodeError on systems started under a non-UTF-8
    locale.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass


def _workload_base_dir(workload_dir: Union[str, list, None]) -> Optional[str]:
    """Extract the base workload directory from the (possibly nested) value."""
    if isinstance(workload_dir, list):
        return (
            workload_dir[0][0]
            if isinstance(workload_dir[0], (list, tuple))
            else workload_dir[0]
        )
    return workload_dir


def validate_roofline_csv(workload_dir: Union[str, Path, list]) -> Tuple[bool, str]:
    """
    Validate roofline.csv exists and has consistent structure.

    Returns:
        tuple: (is_valid, error_message)
               is_valid=True if CSV is valid, False otherwise
               error_message contains description if invalid
    """
    base_dir = _workload_base_dir(workload_dir)
    if base_dir is None:
        return False, "Workload directory is not set"

    benchmark_results = Path(base_dir) / "roofline.csv"

    # Check if file exists
    if not benchmark_results.exists():
        return False, f"Benchmark results file not found: {benchmark_results}"

    # Validate CSV structure
    try:
        with open(benchmark_results, newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            row_count = 0
            num_headers = 0

            for row in csv_reader:
                if row_count == 0:
                    num_headers = len(row) - 1
                    if num_headers <= 0:
                        return (
                            False,
                            "Empty or invalid header row in benchmark_results",
                        )
                else:
                    if len(row) - 1 != num_headers:
                        return (
                            False,
                            f"Inconsistent row length in benchmark_results at "
                            f"row {row_count + 1}. "
                            f"Expected {num_headers + 1} columns, "
                            f"found {len(row)}. "
                            "Roofline data appears corrupted or incomplete.",
                        )
                row_count += 1

            if row_count < 2:
                return (
                    False,
                    f"Insufficient data in benchmark_results. "
                    f"Found {row_count} rows (need at least 2)."
                    f" Roofline data appears corrupted or incomplete.",
                )
    except Exception as e:
        return False, f"Failed to read benchmark_results: {e}"

    return True, ""


def format_table_ascii(
    data: List[Dict[str, Any]],
    columns: List[str],
    description_wrap_width: int = 40,
    decimal: int = 2,
    show_index: bool = True,
) -> str:
    """
    Format a list of dicts as an ASCII table string using only stdlib.

    Args:
        data: List of dictionaries containing row data
        columns: List of column names to display (in order)
        description_wrap_width: Width to wrap Description column (default 40)
        decimal: Number of decimal places for floats (default 2)
        show_index: Whether to add an index column numbering the rows (default True)

    Returns:
        Formatted ASCII table string
    """
    if not data:
        return ""

    # Calculate column widths
    col_widths: dict[str, int] = {}
    wrapped_data: list[dict[str, list[str]]] = []

    for col in columns:
        col_widths[col] = len(col)

    # Process each row and wrap text as needed
    for row in data:
        wrapped_row: dict[str, list[str]] = {}
        for col in columns:
            value = row.get(col, "")
            if value is None:
                value = ""

            # Format floats with specified decimal places (matches tabulate floatfmt)
            if isinstance(value, float):
                value = f"{value:.{decimal}f}"
            elif isinstance(value, int):
                value = str(value)
            else:
                value = str(value)

            # Wrap Description column using textwrap.fill behavior
            # (collapses whitespace, wraps at word boundaries)
            if col == "Description":
                # Use textwrap.fill to match original behavior exactly
                wrapped_text = textwrap.fill(value, width=description_wrap_width)
                lines = wrapped_text.split("\n") if wrapped_text else [""]
            else:
                lines = [value]

            wrapped_row[col] = lines

            # Update column width based on longest line
            for line in lines:
                col_widths[col] = max(col_widths[col], len(line))

        wrapped_data.append(wrapped_row)

    # Build the table
    result: list[str] = []

    # Index column width: max of "index" (5) and the widest row number
    idx_width = max(5, len(str(len(data) - 1)))

    # Create separator line
    def make_separator() -> str:
        parts = ["+"]
        if show_index:
            parts.append("-" * (idx_width + 2) + "+")
        for col in columns:
            parts.append("-" * (col_widths[col] + 2) + "+")
        return "".join(parts)

    # Create header
    def make_header() -> str:
        parts = ["|"]
        if show_index:
            parts.append(f" {'index':^{idx_width}} |")
        for col in columns:
            parts.append(f" {col:<{col_widths[col]}} |")
        return "".join(parts)

    # Create data row (handles multi-line cells)
    def make_row(row_idx: int, wrapped_row: Dict[str, List[str]]) -> List[str]:
        # Find max lines in this row
        max_lines = max(len(lines) for lines in wrapped_row.values())

        row_lines: list[str] = []
        for line_idx in range(max_lines):
            parts = ["|"]
            # Index column - only show on first line
            if show_index and line_idx == 0:
                parts.append(f" {row_idx:>{idx_width}} |")
            elif show_index:
                parts.append(" " * (idx_width + 1) + " |")

            for col in columns:
                lines = wrapped_row[col]
                if line_idx < len(lines):
                    cell_value = lines[line_idx]
                else:
                    cell_value = ""
                parts.append(f" {cell_value:<{col_widths[col]}} |")
            row_lines.append("".join(parts))
        return row_lines

    # Build table
    separator = make_separator()
    result.append(separator)
    result.append(make_header())
    result.append(separator)

    for idx, wrapped_row in enumerate(wrapped_data):
        row_lines = make_row(idx, wrapped_row)
        for line in row_lines:
            result.append(line)
        result.append(separator)

    return "\n".join(result)


def _iter_metric_list_rows(
    panel_configs: Mapping[int, Dict[str, Any]],
) -> Iterator[Tuple[str, Tuple[str, str, str]]]:
    """Yield (id, (name, unit, description)) for each block, table and metric."""
    for panel in panel_configs.values():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if type_key == "metric_table":
                    yield from _iter_metric_table_rows(panel, data_config)
                elif type_key in ("raw_csv_table", "pc_sampling_table"):
                    yield str(data_config["id"] // 100), (panel["title"], "", "")


def _iter_metric_table_rows(
    panel: Dict[str, Any], data_config: Dict[str, Any]
) -> Iterator[Tuple[str, Tuple[str, str, str]]]:
    """Yield the block, table and metric rows of one metric_table."""
    block_id = str(data_config["id"] // 100)
    if block_id != "0":
        yield block_id, (panel["title"], "", "")

    yield format_table_id(data_config["id"]), (data_config["title"], "", "")

    descriptions = panel.get("metrics_description") or {}
    metric_ids = metric_ids_by_name(data_config)
    for name, entries in data_config["metric"].items():
        if not _metric_has_valid_expr(entries, data_config):
            continue
        unit = entries.get("unit")
        description = " ".join(str(descriptions.get(name) or "").split())
        yield metric_ids[name], (name, str(unit or ""), description)


def _is_same_id_branch(metric_id: str, selected_id: str) -> bool:
    """Return True if metric_id is selected_id, or above or below it."""
    return (
        metric_id == selected_id
        or metric_id.startswith(f"{selected_id}.")
        or selected_id.startswith(f"{metric_id}.")
    )


def _format_listed_table(
    table_id: str,
    metric_details: Dict[str, Tuple[str, str, str]],
    listed_ids: List[str],
    terminal_width: int,
) -> str:
    """Format one table with its heading, or "" if it lists no metrics."""
    columns = ["Metric_ID", "Metric", "Unit", "Description"]
    rows = [
        dict(zip(columns, [metric_id, *metric_details[metric_id]]))
        for metric_id in listed_ids
        if metric_id.count(".") == 2 and metric_id.rsplit(".", 1)[0] == table_id
    ]
    if not rows:
        return ""
    # Description fills the terminal width left by the other columns and the
    # borders (3 characters per column, plus 1), down to DESCRIPTION_WIDTH_MIN.
    other_widths = sum(
        max(len(column), *(len(row[column]) for row in rows)) for column in columns[:-1]
    )
    description_width = max(
        DESCRIPTION_WIDTH_MIN, terminal_width - other_widths - (3 * len(columns) + 1)
    )
    table = format_table_ascii(
        rows,
        columns,
        description_wrap_width=description_width,
        show_index=False,
    )
    return f"{table_id} {metric_details[table_id][0]}\n{table}"


def _is_row_id(token: str) -> bool:
    return bool(METRIC_ID_RE.match(token)) and token.count(".") == 2


def _rows_by_position(arch: str) -> Dict[str, Dict[str, Any]]:
    """Return numeric row id -> row of arch's views; {} for an arch without views."""
    if not has_views(arch):
        return {}
    return dict(iter_view_rows(load_views(arch)))
