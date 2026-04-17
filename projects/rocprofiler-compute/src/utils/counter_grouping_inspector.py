#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
# Co-authored-by: @vedithal-amd
"""
Counter grouping inspector for rocprofiler-compute.

Parses GFX architecture YAML configs and outputs counter grouping analysis
without requiring GPU, rocprofiler, or full rocprof-compute initialization.

Perfmon bin-packing uses ``OmniSoC_Base._allocate_perfmon_counter_files`` from
``soc_base.py`` (including ``profiling_counter_grouping_policy.yaml`` via the
same metric-aware coalesce pass as profiling). This script does not implement a
parallel grouping algorithm.

Usage:
    ./src/utils/counter_grouping_inspector.py --arch gfx942
    ./src/utils/counter_grouping_inspector.py --arch gfx942 --block 2 3 4
    ./src/utils/counter_grouping_inspector.py --arch gfx942 --output plan.txt
    ./src/utils/counter_grouping_inspector.py --arch gfx942 --output plan.svg
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure src directory is in Python path for imports
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import from existing modules to maintain single source of truth
from rocprof_compute_soc.soc_base import (  # noqa: E402
    CounterFile,
    OmniSoC_Base,
    flat_counters_in_perfmon_file,
    is_tcc_channel_counter,
)
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils_common import (  # noqa: E402
    METRIC_ID_RE,
    convert_metric_id_to_panel_info,
    get_panel_alias,
)
from vendored import yaml  # noqa: E402


def _counter_display_ip_prefix(counter: str) -> str:
    """First IP-style token of a PMC name (column key for bucket tables)."""
    if is_tcc_channel_counter(counter):
        base = counter.split("[")[0]
        return base.split("_", 1)[0] if "_" in base else base
    if "_" not in counter:
        return counter
    return counter.split("_", 1)[0]


def _counters_grouped_by_ip_sorted(counters: list[str]) -> dict[str, list[str]]:
    by_ip: dict[str, list[str]] = {}
    for ctr in counters:
        ip = _counter_display_ip_prefix(ctr)
        by_ip.setdefault(ip, []).append(ctr)
    for names in by_ip.values():
        names.sort()
    return by_ip


def parse_counters_from_text(text: str) -> tuple[set[str], set[str]]:
    """Parse hardware counters and variables from YAML text."""
    _blk = (
        r"(?:CHA|CHC|CPC|CPF|GC_CANE|GC_EA_SE|GCR|GL1A|GL1C|GL2A|GL2C|"
        r"GLARBA|GLARBC|GRBM|SDMA|SPI|SQ|SP|SQC|SQG|TX|UTCL1|"
        r"TA|TD|TCP|TCC|GCEA)_[0-9A-Z_]*[0-9A-Z](?:\[|e|_sum|_avr|_max|_min)*"
    )
    hw_counter_regex = _blk
    variable_regex = r"\$([0-9A-Za-z_]*[0-9A-Za-z])"
    hw_counter_matches = set(re.findall(hw_counter_regex, text))
    variable_matches = set(re.findall(variable_regex, text))
    hw_counter_matches = hw_counter_matches - variable_matches
    return hw_counter_matches, variable_matches


def parse_counters(config_text: str) -> set[str]:
    """Extract all hardware counters from config text."""
    hw_counters, variables = parse_counters_from_text(config_text)

    # Import BUILD_IN_VARS from utils_common for consistency
    from utils.utils_common import BUILD_IN_VARS

    while variables:
        subvariables: set[str] = set()
        for var in variables:
            if var in BUILD_IN_VARS:
                hw_new, var_new = parse_counters_from_text(BUILD_IN_VARS[var])
                hw_counters.update(hw_new)
                subvariables.update(var_new)
        variables = subvariables - variables

    return hw_counters


def iter_yaml_metrics(
    config_dir: Path,
    arch: str,
) -> Iterator[tuple[str, Any, int, str, str]]:
    """Iterate over all metrics in analysis YAML files."""
    config_root = config_dir / arch
    if not config_root.is_dir():
        return

    for ypath in sorted(config_root.glob("*.yaml")):
        stem_id = ypath.name.split("_")[0]
        try:
            with open(ypath, encoding="utf-8") as stream:
                doc = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        panel_cfg = doc.get("Panel Config")
        if not isinstance(panel_cfg, dict):
            continue
        sources = panel_cfg.get("data source")
        if not isinstance(sources, list):
            continue
        for section in sources:
            if not isinstance(section, dict) or "metric_table" not in section:
                continue
            mt = section["metric_table"]
            if not isinstance(mt, dict):
                continue
            metrics = mt.get("metric")
            if not isinstance(metrics, dict):
                continue
            panel_id = mt.get("id")
            for idx, (metric_name, metric_body) in enumerate(metrics.items()):
                try:
                    metric_text = yaml.dump(
                        metric_body, sort_keys=False, allow_unicode=True
                    )
                except (TypeError, yaml.YAMLError):
                    continue
                yield stem_id, panel_id, idx, metric_name, metric_text


def append_analysis_yaml_for_filter_token(
    raw_token: str,
    config_filename_dict: dict[str, Path],
    config_root_dir: Path,
    texts: list[str],
    panel_alias_dict: dict[str, str],
) -> None:
    """Append YAML content for a filter token (metric ID or alias)."""
    block_id = raw_token

    # Check if it's a metric ID (x.x.x format)
    if METRIC_ID_RE.match(block_id):
        pass
    elif block_id in panel_alias_dict:
        # It's an alias like "lds", "l1i", etc.
        block_id = panel_alias_dict[block_id]
        print(f"alias: {raw_token} -> block id: {block_id}", file=sys.stderr)
    else:
        # Treat as file ID (like "0200")
        if block_id in config_filename_dict:
            with open(config_filename_dict[block_id]) as stream:
                texts.append(stream.read())
            return
        print(
            f"Warning: Unknown block/alias: {raw_token} not found in {config_root_dir}",
            file=sys.stderr,
        )
        return

    file_id, panel_id, metric_id = convert_metric_id_to_panel_info(block_id)

    if file_id not in config_filename_dict:
        print(
            f"Warning: Skipping {block_id}: file id {file_id} not found in "
            f"{config_root_dir}",
            file=sys.stderr,
        )
        return

    with open(config_filename_dict[file_id]) as stream:
        try:
            file_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(
                f"Warning: Skipping {block_id}: failed to parse YAML from "
                f"{config_filename_dict[file_id]}: {exc}",
                file=sys.stderr,
            )
            return

    if not isinstance(file_config, dict):
        print(
            f"Warning: Skipping {block_id}: expected mapping at root of "
            f"{config_filename_dict[file_id]}",
            file=sys.stderr,
        )
        return

    if panel_id is None:
        texts.append(yaml.dump(file_config, sort_keys=False))
        return

    panel_config = file_config.get("Panel Config", {})
    if not isinstance(panel_config, dict):
        print(
            f"Warning: Skipping {block_id}: invalid 'Panel Config' in "
            f"{config_filename_dict[file_id]}",
            file=sys.stderr,
        )
        return

    data_source = panel_config.get("data source", [])
    if not isinstance(data_source, list):
        print(
            f"Warning: Skipping {block_id}: invalid 'data source' in "
            f"{config_filename_dict[file_id]}",
            file=sys.stderr,
        )
        return

    panel_dict = {
        metric_table["id"]: metric_table
        for section in data_source
        if isinstance(section, dict)
        for metric_table in [section.get("metric_table")]
        if isinstance(metric_table, dict) and "id" in metric_table
    }

    if panel_id not in panel_dict:
        print(
            f"Warning: Skipping {block_id}: metric table {panel_id} not found in "
            f"{config_filename_dict[file_id]}",
            file=sys.stderr,
        )
        return

    if metric_id is None:
        texts.append(yaml.dump(panel_dict[panel_id], sort_keys=False))
        return

    metrics = panel_dict[panel_id].get("metric", {})
    if not isinstance(metrics, dict):
        print(
            f"Warning: Skipping {block_id}: invalid metric table format for "
            f"panel id {panel_id}",
            file=sys.stderr,
        )
        return

    metric_list = list(metrics.items())
    if metric_id >= len(metric_list):
        print(
            f"Warning: Skipping {block_id}: metric id {metric_id} not found in "
            f"panel id {panel_id}",
            file=sys.stderr,
        )
        return

    metric_name, metric_body = metric_list[metric_id]
    texts.append(yaml.dump({metric_name: metric_body}, sort_keys=False))


def detect_counters(
    config_dir: Path,
    arch: str,
    filter_blocks: list[str] | None = None,
) -> tuple[set[str], list[str]]:
    """Detect all counters from YAML configs for the given architecture."""
    config_root = config_dir / arch
    if not config_root.is_dir():
        print(f"Error: Config directory not found: {config_root}", file=sys.stderr)
        sys.exit(1)

    config_files = {f.name.split("_")[0]: f for f in config_root.glob("*.yaml")}
    panel_alias_dict = get_panel_alias()

    texts: list[str] = []
    effective_blocks: list[str] = filter_blocks or []

    if not effective_blocks:
        default_config_files = {
            block_id: filename
            for block_id, filename in config_files.items()
            if block_id != "3000"
        }
        for filename in default_config_files.values():
            with open(filename) as stream:
                texts.append(stream.read())
    else:
        for block_token in effective_blocks:
            append_analysis_yaml_for_filter_token(
                block_token,
                config_files,
                config_root,
                texts,
                panel_alias_dict,
            )

    counters = parse_counters("\n".join(texts))
    counters.discard("SQ_ACCUM_PREV_HIRES")

    return counters, effective_blocks


def allocate_buckets(
    counters: set[str],
    perfmon_config: dict[str, int],
    arch: str,
    config_dir: Path,
) -> list[CounterFile]:
    """Allocate counters using ``OmniSoC_Base._allocate_perfmon_counter_files`` (profiling path)."""
    mspec = MagicMock()
    mspec.rocminfo_lines = None
    # TCC template expansion only; values mirror unit-test defaults when no GPU.
    mspec.num_xcd = 1
    mspec.l2_banks = 4

    args = MagicMock()
    args.config_dir = str(config_dir.resolve())
    args.membw_analysis = False

    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, mspec)
    soc.set_arch(arch)
    soc.set_perfmon_config(perfmon_config)

    work = set(counters)
    work.discard("SQ_ACCUM_PREV_HIRES")
    work = soc._expand_tcc_template_counters(work)

    output_files, _file_count, _accu = soc._allocate_perfmon_counter_files(work)
    return output_files


def _global_ip_column_widths(
    output_files: list[CounterFile],
) -> tuple[list[str], dict[str, int]]:
    """Column order and cell widths from the longest header or counter name."""
    max_cell: dict[str, int] = {}
    for counter_file in output_files:
        flat = flat_counters_in_perfmon_file(counter_file)
        by_ip = _counters_grouped_by_ip_sorted(flat)
        for ip, names in by_ip.items():
            longest = max((len(n) for n in names), default=0)
            max_cell[ip] = max(max_cell.get(ip, 0), longest, len(ip))
    columns = sorted(max_cell.keys())
    widths = {ip: max_cell[ip] for ip in columns}
    return columns, widths


def _format_bucket_markdown(
    counters: list[str],
    global_columns: list[str],
    widths: dict[str, int],
) -> str:
    """GitHub-style pipe table with fixed column widths."""
    by_ip = _counters_grouped_by_ip_sorted(counters)
    height = max((len(by_ip.get(c, [])) for c in global_columns), default=0)

    def pipe_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        pipe_row([ip.ljust(widths[ip]) for ip in global_columns]),
        pipe_row(["-" * widths[ip] for ip in global_columns]),
    ]
    for row_idx in range(height):
        lines.append(
            pipe_row([
                (by_ip[c][row_idx] if row_idx < len(by_ip.get(c, [])) else "").ljust(
                    widths[c]
                )
                for c in global_columns
            ])
        )
    return "\n".join(lines)


def generate_bucket_plan(
    output_files: list[CounterFile],
    arch: str,
) -> str:
    """Generate the bucket allocation plan as markdown tables."""
    from io import StringIO

    buf = StringIO()
    buf.write(f"Perfmon bucket allocation plan (architecture: {arch})\n\n")

    global_columns, col_widths = _global_ip_column_widths(output_files)
    total_assignments = 0

    for counter_file in output_files:
        bucket_label = counter_file.file_name_txt.replace(".txt", "")
        flat = flat_counters_in_perfmon_file(counter_file)
        total_assignments += len(flat)
        buf.write(f"Bucket: {bucket_label}\n")
        if not flat:
            buf.write("(no PMC counters)\n\n")
            continue
        if not global_columns:
            continue
        buf.write(_format_bucket_markdown(flat, global_columns, col_widths))
        buf.write("\n\n")

    buf.write(
        f"Summary: {len(output_files)} bucket(s), "
        f"{total_assignments} counter assignment(s).\n\n"
    )

    return buf.getvalue()


def print_bucket_plan(output_files: list[CounterFile], arch: str) -> None:
    """Print the bucket allocation plan as markdown tables."""
    output = generate_bucket_plan(output_files, arch)
    print(output, end="")


def _counter_to_bucket_map(
    output_files: list[CounterFile],
) -> dict[str, str]:
    """Map each PMC counter string to its perfmon bucket label."""
    result: dict[str, str] = {}
    for counter_file in output_files:
        label = counter_file.file_name_txt.replace(".txt", "")
        for ctr in flat_counters_in_perfmon_file(counter_file):
            result[ctr] = label
    return result


def generate_multi_bucket_metrics(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
) -> str:
    """Generate metrics that span multiple buckets as a string."""
    from io import StringIO

    buf = StringIO()
    counter_to_bucket = _counter_to_bucket_map(output_files)

    multi_rows: list[tuple[str, str, int, str, int, str]] = []
    single_rows: list[tuple[str, str, int, str, str]] = []
    total_metrics = 0

    for file_id, panel_id, metric_idx, metric_name, metric_yaml in iter_yaml_metrics(
        config_dir, arch
    ):
        total_metrics += 1
        hw = parse_counters(metric_yaml)
        buckets: set[str] = set()
        for c in hw:
            b = counter_to_bucket.get(c)
            if b is not None:
                buckets.add(b)

        panel_s = str(panel_id) if panel_id is not None else "-"
        n_b = len(buckets)
        if n_b > 1:
            multi_rows.append((
                file_id,
                panel_s,
                metric_idx,
                metric_name,
                n_b,
                ", ".join(sorted(buckets)),
            ))
        elif n_b == 1:
            single_rows.append((
                file_id,
                panel_s,
                metric_idx,
                metric_name,
                next(iter(buckets)),
            ))

    multi_pct = (100.0 * len(multi_rows) / total_metrics) if total_metrics else 0.0
    single_pct = (100.0 * len(single_rows) / total_metrics) if total_metrics else 0.0

    buf.write(
        f"Metrics with PMC counters assigned to more than one perfmon bucket "
        f"({len(multi_rows)} of {total_metrics} metrics, {multi_pct:.1f}%)\n"
    )
    buf.write(
        "All *.yaml under the arch are scanned. Listed rows are metrics where at "
        "least one formula counter is in this plan's collection and those "
        "counters map to 2+ buckets in the layout above.\n"
    )
    buf.write(f"Config tree: {config_dir / arch}\n")

    if multi_rows:
        # Calculate column widths
        headers = ["File", "Panel", "Idx", "Metric name", "#Bkts", "Buckets"]
        widths = [len(h) for h in headers]
        str_rows = [[r[0], r[1], str(r[2]), r[3], str(r[4]), r[5]] for r in multi_rows]
        for row in str_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def pipe_line(parts: list[str]) -> str:
            return (
                "| "
                + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))
                + " |"
            )

        buf.write(pipe_line(headers) + "\n")
        buf.write(pipe_line(["-" * widths[i] for i in range(len(widths))]) + "\n")
        for row in str_rows:
            buf.write(pipe_line(row) + "\n")
    else:
        buf.write(
            "(none listed — in-collection counters stay in one bucket per metric)\n"
        )

    buf.write("\n")
    buf.write(
        f"Metrics with PMC counters assigned to one perfmon bucket "
        f"({len(single_rows)} of {total_metrics} metrics, {single_pct:.1f}%)\n"
    )
    buf.write(
        "Listed rows are metrics where at least one formula counter is in this "
        "plan's collection and every such counter maps to the same single "
        "bucket above (metrics with no in-collection PMCs are omitted).\n"
    )

    if single_rows:
        headers = ["File", "Panel", "Idx", "Metric name", "Bucket"]
        widths = [len(h) for h in headers]
        str_rows = [[r[0], r[1], str(r[2]), r[3], r[4]] for r in single_rows]
        for row in str_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def pipe_line(parts: list[str]) -> str:
            return (
                "| "
                + " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))
                + " |"
            )

        buf.write(pipe_line(headers) + "\n")
        buf.write(pipe_line(["-" * widths[i] for i in range(len(widths))]) + "\n")
        for row in str_rows:
            buf.write(pipe_line(row) + "\n")
    else:
        buf.write(
            "(none listed — no metric has in-collection PMCs confined to one bucket)\n"
        )
    buf.write("\n")

    return buf.getvalue()


def print_multi_bucket_metrics(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
) -> None:
    """Print metrics that span multiple buckets."""
    output = generate_multi_bucket_metrics(output_files, config_dir, arch)
    print(output, end="")


def render_perfmon_plan_svg(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
    title: str = "Perfmon Bucket Plan",
) -> str:
    """Render the perfmon plan as SVG using Rich's export_svg.

    Uses markdown pipe-table style (same as CLI) with all columns aligned
    across all buckets for consistent visual appearance.
    """
    from io import StringIO

    try:
        from rich.console import Console
    except ImportError:
        raise ImportError("Rich library required for SVG output: pip install rich")

    # Calculate required width based on table content
    global_columns, col_widths = _global_ip_column_widths(output_files)
    total_width = sum(col_widths.values()) + len(col_widths) * 3 + 10
    console_width = max(200, total_width)

    # Create console with recording enabled
    console = Console(
        file=StringIO(),
        force_terminal=True,
        width=console_width,
        height=500,
        record=True,
    )

    # Print header
    header = f"Perfmon bucket allocation plan (architecture: {arch})"
    console.print(f"[bold cyan]{header}[/bold cyan]\n")

    # Print each bucket using markdown pipe-table style with global column alignment
    total_assignments = 0
    for counter_file in output_files:
        bucket_label = counter_file.file_name_txt.replace(".txt", "")
        flat = flat_counters_in_perfmon_file(counter_file)
        total_assignments += len(flat)

        console.print(f"[bold blue]Bucket: {bucket_label}[/bold blue]")

        if not flat:
            console.print("[dim](no PMC counters)[/dim]\n")
            continue

        if not global_columns:
            continue

        # Use markdown pipe-table format - plain text for consistency
        by_ip = _counters_grouped_by_ip_sorted(flat)
        height = max((len(by_ip.get(c, [])) for c in global_columns), default=0)

        # Header row - use plain padding then wrap with color
        header_parts = [col.ljust(col_widths[col]) for col in global_columns]
        console.print("[bold cyan]| " + " | ".join(header_parts) + " |[/bold cyan]")

        # Separator row
        sep_parts = ["-" * col_widths[col] for col in global_columns]
        console.print("[dim]| " + " | ".join(sep_parts) + " |[/dim]")

        # Data rows - no special coloring, just consistent padding
        for row_idx in range(height):
            row_parts = []
            for col in global_columns:
                counters = by_ip.get(col, [])
                cell = counters[row_idx] if row_idx < len(counters) else ""
                row_parts.append(cell.ljust(col_widths[col]))
            console.print("| " + " | ".join(row_parts) + " |")

        console.print()

    console.print(
        f"[bold]Summary:[/bold] {len(output_files)} bucket(s), "
        f"{total_assignments} counter assignment(s).\n"
    )

    # Add multi-bucket metrics section
    metrics_output = generate_multi_bucket_metrics(output_files, config_dir, arch)
    console.print(metrics_output)

    return console.export_svg(title=title)


def get_default_config_dir() -> Path:
    """Get the default analysis configs directory."""
    return Path(__file__).parent.parent / "rocprof_compute_soc" / "analysis_configs"


def get_supported_archs() -> list[str]:
    """Get list of supported architectures from mi_gpu_specs."""
    return list(mi_gpu_specs.get_gpu_series_dict().keys())


def main() -> None:
    # Get supported architectures dynamically from mi_gpu_specs
    supported_archs = get_supported_archs()

    parser = argparse.ArgumentParser(
        description="Counter grouping inspector for rocprofiler-compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m utils.counter_grouping_inspector --arch gfx942
  python -m utils.counter_grouping_inspector --arch gfx942 --block 0200 0400
  python -m utils.counter_grouping_inspector --arch gfx942 --output plan.txt
  python -m utils.counter_grouping_inspector --arch gfx942 --output plan.svg
""",
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=sorted(supported_archs),
        help="GPU architecture (e.g., gfx942, gfx950)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to analysis_configs directory (default: auto-detect)",
    )
    parser.add_argument(
        "--block",
        "-b",
        nargs="+",
        default=None,
        help=(
            "Filter to specific metric IDs or aliases.\n"
            "Metric ID format: x, x.x, x.x.x (e.g., 2, 2.1, 2.1.1)\n"
            "Aliases: lds, l1i, sl1d, tcp, vl1d, l2, xgmi\n"
            "File IDs: 0200, 0400, etc."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output to file. Supported formats determined by file suffix:\n"
            "  .txt - Plain text output\n"
            "  .svg - SVG image output (requires rich library)"
        ),
    )

    args = parser.parse_args()

    config_dir = args.config_dir or get_default_config_dir()
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    arch = args.arch

    # Get perfmon config from mi_gpu_specs (single source of truth)
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    if not perfmon_config:
        print(
            f"Error: No perfmon config found for architecture: {arch}",
            file=sys.stderr,
        )
        sys.exit(1)

    counters, filter_blocks = detect_counters(config_dir, arch, args.block)

    if not counters:
        print("No counters found!", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Collected {len(counters)} unique counters:")
        for c in sorted(counters):
            print(f"  - {c}")
        print()

    output_files = allocate_buckets(counters, perfmon_config, arch, config_dir)

    # Handle output formats
    if args.output:
        output_path = args.output
        suffix = output_path.suffix.lower()

        if suffix == ".svg":
            try:
                svg_content = render_perfmon_plan_svg(output_files, config_dir, arch)
                output_path.write_text(svg_content, encoding="utf-8")
            except ImportError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            except OSError as e:
                print(
                    f"Error: could not write SVG output to {output_path}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"SVG saved to {output_path}")
        elif suffix == ".txt":
            bucket_output = generate_bucket_plan(output_files, arch)
            metrics_output = generate_multi_bucket_metrics(
                output_files, config_dir, arch
            )
            try:
                output_path.write_text(bucket_output + metrics_output, encoding="utf-8")
            except OSError as e:
                print(
                    f"Error: could not write text output to {output_path}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Output written to {output_path}")
        else:
            print(
                f"Warning: Unsupported output format '{suffix}'. "
                f"Supported formats: .txt, .svg",
                file=sys.stderr,
            )
            print("Falling back to stdout output.\n", file=sys.stderr)
            print_bucket_plan(output_files, arch)
            print_multi_bucket_metrics(output_files, config_dir, arch)
    else:
        print_bucket_plan(output_files, arch)
        print_multi_bucket_metrics(output_files, config_dir, arch)


if __name__ == "__main__":
    main()
