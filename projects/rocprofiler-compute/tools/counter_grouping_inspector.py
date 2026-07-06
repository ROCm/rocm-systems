#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""
Counter grouping inspector for rocprofiler-compute.

Parses GFX architecture YAML configs and outputs counter grouping analysis
without requiring GPU, rocprofiler, or full rocprof-compute initialization.

Counter discovery uses ``OmniSoC_Base.detect_counters``; perfmon layout and
YAML emission use ``OmniSoC_Base.perfmon_coalesce`` (which calls
``_allocate_perfmon_counter_files`` and writes under ``<workload>/perfmon/``),
matching the profiling path in ``soc_base.py``. ``get_rocprof_supported_counters``
is stubbed so the tool runs without rocprofiler. Bucket views reuse a second
``_allocate_perfmon_counter_files`` call on the same counter set (same result as
the coalesce pass).

Usage (from the ``rocprofiler-compute`` project root):
    ./tools/counter_grouping_inspector.py --arch gfx942
    ./tools/counter_grouping_inspector.py --arch gfx942 --block 2 3 4
    ./tools/counter_grouping_inspector.py --arch gfx942 --output plan.txt
    ./tools/counter_grouping_inspector.py --arch gfx942 --output plan.svg
"""

import argparse
import logging
import sys
import tempfile
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from rich.console import Console

# This file lives under tools/; add src/ to path for rocprof_compute_* imports.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Import from existing modules to maintain single source of truth
from rocprof_compute_soc.soc_base import (  # noqa: E402
    CounterFile,
    OmniSoC_Base,
    flat_counters_in_perfmon_file,
    is_tcc_channel_counter,
)
from utils.logger import console_error  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils_counter_defs import (  # noqa: E402
    get_build_in_vars,
    parse_counters_text,
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


def parse_counters(config_text: str, gpu_series: str) -> set[str]:
    """Extract all hardware counters from config text.

    Args:
        config_text: Metric formula text
        gpu_series: GPU series for resolving built-in vars
    """
    hw_counters, variables = parse_counters_text(config_text)
    build_in_vars = get_build_in_vars(gpu_series)

    while variables:
        subvariables: set[str] = set()
        for var in variables:
            if var in build_in_vars:
                hw_new, var_new = parse_counters_text(build_in_vars[var])
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


def _rocprof_supported_superset(counters: set[str]) -> set[str]:
    """Return a fake rocprofiler avail set.

    Used so perfmon_coalesce skips unsupported-counter warnings.
    """
    out = set(counters)
    for ctr in counters:
        if is_tcc_channel_counter(ctr):
            out.add(ctr.split("[", 1)[0])
    return out


def run_soc_detect_and_coalesce(
    arch: str,
    config_dir: Path,
    filter_blocks: Optional[list[str]],
    perfmon_config: dict[str, int],
    workload_root: Path,
) -> tuple[set[str], list[CounterFile]]:
    """Run SoC counter detection and perfmon coalesce.

    Writes YAML under ``workload_root/perfmon/``.
    """
    machine_spec = SimpleNamespace(
        rocminfo_lines=None,
        num_xcd=1,
        l2_banks=4,
    )

    filter_block_list = list(filter_blocks) if filter_blocks else []
    cli_args = argparse.Namespace(
        output_directory=str(workload_root.resolve()),
        config_dir=str(config_dir.resolve()),
        filter_blocks=filter_block_list,
        membw_analysis=False,
        set_selected=None,
        roof_only=False,
        no_roof=True,
        device=0,
    )

    soc = OmniSoC_Base(cli_args, machine_spec)
    soc.set_arch(arch)
    soc.set_perfmon_config(perfmon_config)

    counters, _unused_filter_blocks = soc.detect_counters()
    # Same as OmniSoC_Base.perfmon_filter before perfmon_coalesce: drop
    # SQ_ACCUM_PREV_HIRES here; it is injected again when LEVEL counters are
    # allocated so pass grouping matches profiling.
    counters = counters - {"SQ_ACCUM_PREV_HIRES"}
    if not counters:
        return set(), []

    soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
        lambda c=counters: _rocprof_supported_superset(c)
    )

    soc.perfmon_coalesce(counters)

    perfmon_counter_files, _unused_perfmon_files, _unused_accumulator_files = (
        soc._allocate_perfmon_counter_files(counters)
    )
    return counters, perfmon_counter_files


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


def _bucket_plan_sections(
    output_files: list[CounterFile],
) -> tuple[list[str], dict[str, int], list[tuple[str, list[str]]], int]:
    """Shared bucket layout: columns, widths, per-bucket lists, assignment total."""
    global_columns, column_widths = _global_ip_column_widths(output_files)
    sections: list[tuple[str, list[str]]] = []
    total_assignments = 0
    for counter_file in output_files:
        bucket_label = counter_file.file_name_txt.replace(".txt", "")
        flat_counters = flat_counters_in_perfmon_file(counter_file)
        total_assignments += len(flat_counters)
        sections.append((bucket_label, flat_counters))
    return global_columns, column_widths, sections, total_assignments


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
    buf = StringIO()
    buf.write(f"Perfmon bucket allocation plan (architecture: {arch})\n\n")

    global_columns, column_widths, sections, total_assignments = _bucket_plan_sections(
        output_files
    )

    for bucket_label, flat_counters in sections:
        buf.write(f"Bucket: {bucket_label}\n")
        if not flat_counters:
            buf.write("(no PMC counters)\n\n")
            continue
        if not global_columns:
            continue
        buf.write(_format_bucket_markdown(flat_counters, global_columns, column_widths))
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


def generate_bucket_metrics(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
) -> str:
    """Generate metrics that span multiple buckets as a string.

    Note: Despite handling both multi-bucket and single-bucket metrics,
    this function's name reflects the primary bucket analysis use case.
    """
    buf = StringIO()
    counter_to_bucket = _counter_to_bucket_map(output_files)

    multi_rows: list[tuple[str, str, int, str, int, str]] = []
    single_rows: list[tuple[str, str, int, str, str]] = []
    total_metrics = 0

    gpu_series = mi_gpu_specs.get_gpu_series(arch)
    for file_id, panel_id, metric_idx, metric_name, metric_yaml in iter_yaml_metrics(
        config_dir, arch
    ):
        total_metrics += 1
        hardware_counters = parse_counters(metric_yaml, gpu_series)
        buckets: set[str] = set()
        for formula_counter in hardware_counters:
            bucket_label = counter_to_bucket.get(formula_counter)
            if bucket_label is not None:
                buckets.add(bucket_label)

        panel_label = str(panel_id) if panel_id is not None else "-"
        bucket_count = len(buckets)
        if bucket_count > 1:
            multi_rows.append((
                file_id,
                panel_label,
                metric_idx,
                metric_name,
                bucket_count,
                ", ".join(sorted(buckets)),
            ))
        elif bucket_count == 1:
            single_rows.append((
                file_id,
                panel_label,
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


def print_bucket_metrics(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
) -> None:
    """Print metrics that span multiple buckets."""
    output = generate_bucket_metrics(output_files, config_dir, arch)
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
    global_columns, column_widths, sections, total_assignments = _bucket_plan_sections(
        output_files
    )

    total_width = sum(column_widths.values()) + len(column_widths) * 3 + 10
    console_width = max(200, total_width)

    console = Console(
        file=StringIO(),
        force_terminal=True,
        width=console_width,
        height=500,
        record=True,
    )

    header = f"Perfmon bucket allocation plan (architecture: {arch})"
    console.print(f"[bold cyan]{header}[/bold cyan]\n")

    for bucket_label, flat_counters in sections:
        console.print(f"[bold blue]Bucket: {bucket_label}[/bold blue]")

        if not flat_counters:
            console.print("[dim](no PMC counters)[/dim]\n")
            continue

        if not global_columns:
            continue

        table_lines = _format_bucket_markdown(
            flat_counters, global_columns, column_widths
        ).split("\n")
        if table_lines:
            console.print(f"[bold cyan]{table_lines[0]}[/bold cyan]")
        if len(table_lines) > 1:
            console.print(f"[dim]{table_lines[1]}[/dim]")
        for line in table_lines[2:]:
            console.print(line)
        console.print()

    console.print(
        f"[bold]Summary:[/bold] {len(output_files)} bucket(s), "
        f"{total_assignments} counter assignment(s).\n"
    )

    metrics_output = generate_bucket_metrics(output_files, config_dir, arch)
    console.print(metrics_output)

    return console.export_svg(title=title)


def get_default_config_dir() -> Path:
    """Get the default analysis configs directory."""
    return _SRC_DIR / "rocprof_compute_soc" / "analysis_configs"


def get_supported_archs() -> list[str]:
    """Get list of supported architectures from mi_gpu_specs."""
    return list(mi_gpu_specs.get_gpu_series_dict().keys())


def _configure_inspector_logging() -> None:
    """Suppress debug noise from SoC code paths (console_debug)."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in root.handlers:
        handler.setLevel(logging.WARNING)


def main() -> None:
    _configure_inspector_logging()

    # Get supported architectures dynamically from mi_gpu_specs
    supported_archs = get_supported_archs()

    parser = argparse.ArgumentParser(
        description="Counter grouping inspector for rocprofiler-compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/counter_grouping_inspector.py --arch gfx942
  python tools/counter_grouping_inspector.py --arch gfx942 --block 0200 0400
  python tools/counter_grouping_inspector.py --arch gfx942 --output plan.txt
  python tools/counter_grouping_inspector.py --arch gfx942 --output plan.svg
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
        console_error(f"Error: Config directory not found: {config_dir}")

    arch = args.arch

    # Get perfmon config from mi_gpu_specs (single source of truth)
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    if not perfmon_config:
        console_error(f"Error: No perfmon config found for architecture: {arch}")

    config_arch = config_dir / arch
    if not config_arch.is_dir():
        console_error(f"Error: Architecture config directory not found: {config_arch}")

    with tempfile.TemporaryDirectory(prefix="rocprof_counter_inspector_") as tmpdir:
        workload_root = Path(tmpdir)
        counters, output_files = run_soc_detect_and_coalesce(
            arch, config_dir, args.block, perfmon_config, workload_root
        )

        if not counters:
            console_error("No counters found!")

        if args.verbose:
            print(f"Collected {len(counters)} unique counters:")
            for ctr in sorted(counters):
                print(f"  - {ctr}")
            print()
            perfmon_dir = workload_root / "perfmon"
            if perfmon_dir.is_dir():
                written = sorted(p.name for p in perfmon_dir.iterdir())
                print(f"Perfmon YAML directory ({perfmon_dir}): {len(written)} file(s)")
                for name in written:
                    print(f"  - {name}")
                print()

        _emit_inspector_output(args, output_files, config_dir, arch)


def _emit_inspector_output(
    args: argparse.Namespace,
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
) -> None:
    """Write or print bucket plan and multi-bucket metrics (stdout or --output)."""
    # Handle output formats
    if args.output:
        output_path = args.output
        suffix = output_path.suffix.lower()

        if suffix == ".svg":
            try:
                svg_content = render_perfmon_plan_svg(output_files, config_dir, arch)
                output_path.write_text(svg_content, encoding="utf-8")
            except OSError as exc:
                console_error(
                    f"Error: could not write SVG output to {output_path}: {exc}"
                )
            print(f"SVG saved to {output_path}")
        elif suffix == ".txt":
            bucket_output = generate_bucket_plan(output_files, arch)
            metrics_output = generate_bucket_metrics(output_files, config_dir, arch)
            try:
                output_path.write_text(bucket_output + metrics_output, encoding="utf-8")
            except OSError as exc:
                console_error(
                    f"Error: could not write text output to {output_path}: {exc}"
                )
            print(f"Output written to {output_path}")
        else:
            print(
                f"Warning: Unsupported output format '{suffix}'. "
                f"Supported formats: .txt, .svg",
                file=sys.stderr,
            )
            print("Falling back to stdout output.\n", file=sys.stderr)
            print_bucket_plan(output_files, arch)
            print_bucket_metrics(output_files, config_dir, arch)
    else:
        print_bucket_plan(output_files, arch)
        print_bucket_metrics(output_files, config_dir, arch)


if __name__ == "__main__":
    main()
