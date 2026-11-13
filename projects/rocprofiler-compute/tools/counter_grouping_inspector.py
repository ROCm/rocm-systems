#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""
Counter grouping inspector for rocprofiler-compute.

Parses GFX architecture YAML configs and outputs counter grouping analysis
without requiring GPU, rocprofiler, or full rocprof-compute initialization.

Counter discovery uses `OmniSoC_Base.detect_counters`; perfmon layout and
YAML emission use `OmniSoC_Base.perfmon_coalesce` (which calls
`_allocate_perfmon_counter_files` and writes under `<workload>/perfmon/`),
matching the profiling path in `soc_base.py`. `get_rocprof_supported_counters`
is stubbed so the tool runs without rocprofiler. Bucket views reuse a second
`_allocate_perfmon_counter_files` call on the same counter set (same result as
the coalesce pass).

Each plan summary ends with the arch's counter limits per block and a note on
which table columns are real packing blocks.

Usage (from the `rocprofiler-compute` project root):
    ./tools/counter_grouping_inspector.py --arch gfx942
    ./tools/counter_grouping_inspector.py --arch gfx942 --block 2 3 4
    ./tools/counter_grouping_inspector.py --arch gfx942 --output plan.txt
    ./tools/counter_grouping_inspector.py --arch gfx942 --output plan.svg
"""

import argparse
import logging
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from rich.console import Console

# This file lives under tools/; add src/ to path for rocprof_compute_* imports.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Import from existing modules to maintain single source of truth
import config
from rocprof_compute_soc.soc_base import (
    OmniSoC_Base,
    is_tcc_channel_counter,
    iter_analysis_config_metrics,
)
from utils.counters.allocator import metric_counter_sets, reads_one_pass
from utils.counters.counter_file import (
    CounterFile,
    flat_counters_in_perfmon_file,
)
from utils.logger import console_error
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import canonical_config_arch
from utils.utils_counter_defs import BLOCK_REMAP, Collectables


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
    filter_blocks: Optional[list[str]],
    perfmon_config: dict[str, int],
    workload_root: Path,
) -> tuple[set[str], list[CounterFile]]:
    """Run SoC counter detection and perfmon coalesce.

    Writes YAML under `workload_root/perfmon/`.
    """
    machine_spec = SimpleNamespace(
        rocminfo_lines=None,
        num_xcd=1,
        l2_banks=4,
        gpu_arch=arch,
        gpu_series=mi_gpu_specs.get_gpu_series(arch),
    )

    filter_block_list = list(filter_blocks) if filter_blocks else []
    cli_args = argparse.Namespace(
        output_directory=str(workload_root.resolve()),
        filter_blocks=filter_block_list,
        membw_analysis=False,
        set_selected=None,
    )

    soc = OmniSoC_Base(cli_args, machine_spec)
    soc.set_arch(arch)
    soc.set_perfmon_config(perfmon_config)

    counters, _unused_filter_blocks = soc.detect_counters()
    if not counters:
        return set(), []

    soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
        lambda c=counters: _rocprof_supported_superset(c)
    )

    soc.perfmon_coalesce(counters)

    perfmon_counter_files = soc._allocate_perfmon_counter_files(counters)
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
        bucket_label = counter_file.name.replace(".txt", "")
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


def _format_hw_limits_line(perfmon_config: dict[str, int]) -> str:
    """Format the counter limit of each perfmon block."""
    limits = ", ".join(f"{block}: {cap}" for block, cap in perfmon_config.items())
    return f"Counter limits per block: {limits}"


def _format_display_column_comments() -> str:
    """Explain which table columns are real packing blocks."""
    remaps = ", ".join(f"{src}->{dst}" for src, dst in sorted(BLOCK_REMAP.items()))
    return (
        "Note: the table columns group counters by name prefix. Some blocks "
        f"count against another block's limit ({remaps}); CPC and CPF have "
        "their own limits."
    )


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
        f"{total_assignments} counter assignment(s).\n"
    )
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch) or {}
    buf.write(f"\n{_format_hw_limits_line(perfmon_config)}\n")
    buf.write(f"{_format_display_column_comments()}\n\n")

    return buf.getvalue()


def print_bucket_plan(output_files: list[CounterFile], arch: str) -> None:
    """Print the bucket allocation plan as markdown tables."""
    output = generate_bucket_plan(output_files, arch)
    print(output, end="")


def _bucket_counter_sets(
    output_files: list[CounterFile],
) -> dict[str, set[str]]:
    """Map each perfmon bucket label to the PMC counters it collects."""
    return {
        counter_file.name.replace(".txt", ""): set(
            flat_counters_in_perfmon_file(counter_file)
        )
        for counter_file in output_files
    }


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
    bucket_counters = _bucket_counter_sets(output_files)
    plan_counters = set().union(*bucket_counters.values())

    multi_rows: list[tuple[str, str, int, str, int, str]] = []
    single_rows: list[tuple[str, str, int, str, str]] = []
    total_metrics = 0

    collectables = Collectables.of_arch(arch)
    config_arch = canonical_config_arch(arch)
    for metric in iter_analysis_config_metrics(config_dir / config_arch, set()):
        total_metrics += 1
        metric_sets = metric_counter_sets(
            [metric.expressions], collectables, plan_counters
        )
        if not metric_sets:
            continue
        metric_set = metric_sets[0]
        panel_label = str(metric.table_id) if metric.table_id is not None else "-"

        # A counter can sit in several buckets, so a metric is single-pass
        # when one bucket holds all of its counters.
        holders = [
            label
            for label, counters in bucket_counters.items()
            if metric_set <= counters
        ]
        # A split metric reads each of its collectables from its own pass
        if holders or reads_one_pass(
            metric.expressions, collectables, list(bucket_counters.values())
        ):
            single_rows.append((
                metric.stem_id,
                panel_label,
                metric.metric_idx,
                metric.metric_name,
                holders[0] if holders else "collectables",
            ))
            continue
        buckets = [
            label
            for label, counters in bucket_counters.items()
            if metric_set & counters
        ]
        multi_rows.append((
            metric.stem_id,
            panel_label,
            metric.metric_idx,
            metric.metric_name,
            len(buckets),
            ", ".join(buckets),
        ))

    multi_pct = (100.0 * len(multi_rows) / total_metrics) if total_metrics else 0.0
    single_pct = (100.0 * len(single_rows) / total_metrics) if total_metrics else 0.0

    buf.write(
        f"Multi-pass metrics: no perfmon bucket holds all of their PMC counters "
        f"({len(multi_rows)} of {total_metrics} metrics, {multi_pct:.1f}%)\n"
    )
    buf.write(
        "Lists every arch config metric that reads a collected counter. Buckets "
        "are the ones holding any of its counters.\n"
    )
    buf.write(f"Config tree: {config_dir / config_arch}\n")

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
            "(none listed: in-collection counters stay in one bucket per metric)\n"
        )

    buf.write("\n")
    buf.write(
        f"Single-pass metrics: one perfmon bucket holds all of their PMC counters "
        f"({len(single_rows)} of {total_metrics} metrics, {single_pct:.1f}%)\n"
    )
    buf.write(
        "Lists every arch config metric that reads a collected counter. Bucket "
        "is the first one holding all of its counters.\n"
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
            "(none listed: no metric has in-collection PMCs confined to one bucket)\n"
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
        f"{total_assignments} counter assignment(s)."
    )
    console.print()
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch) or {}
    console.print(_format_hw_limits_line(perfmon_config))
    console.print(_format_display_column_comments())
    console.print()

    metrics_output = generate_bucket_metrics(output_files, config_dir, arch)
    console.print(metrics_output)

    return console.export_svg(title=title)


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

    config_dir = config.analysis_configs_dir

    arch = args.arch

    # Get perfmon config from mi_gpu_specs (single source of truth)
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    if not perfmon_config:
        console_error(f"Error: No perfmon config found for architecture: {arch}")

    config_arch_name = canonical_config_arch(arch)
    config_arch_path = config_dir / config_arch_name
    if not config_arch_path.is_dir():
        console_error(
            f"Error: Architecture config directory not found: {config_arch_path}"
        )

    with tempfile.TemporaryDirectory(prefix="rocprof_counter_inspector_") as tmpdir:
        workload_root = Path(tmpdir)
        counters, output_files = run_soc_detect_and_coalesce(
            arch, args.block, perfmon_config, workload_root
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
                output_path.write_text(
                    bucket_output + metrics_output,
                    encoding="utf-8",
                )
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
