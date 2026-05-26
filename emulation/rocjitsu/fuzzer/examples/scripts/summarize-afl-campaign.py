#!/usr/bin/env python3

import argparse
import datetime as _datetime
import importlib.util
import subprocess
import sys
from pathlib import Path

from patch_report_samples import coverage_mix_from_summaries, selected_edge_sample_text


TARGETS = {
    "rocblas-sgemm": {
        "classification": "Device branch coverage",
        "reports": ["high-edge-reports/rocblas_sgemm_default_hybrid_high_edge.jsonl"],
        "limitation": "Coverage depends on the current default hybrid branch path; deeper coverage still needs safer relocation and temporary-register planning.",
    },
    "rocfft-c2c": {
        "classification": "Device branch coverage with unresolved twiddle gap",
        "reports": ["high-edge-reports/rocfft_c2c_high_edge.jsonl"],
        "limitation": "Generated FFT kernels produce device deltas; twiddle_gen_* remains explicitly unpatched until low-register or spill-backed probes are safe.",
    },
    "rocrand-uniform": {
        "classification": "Device branch coverage via loader-scoped fixed branches",
        "reports": ["rocrand_uniform_report.jsonl"],
        "limitation": "Coverage is shallow but exercises KPACK/HSA-reader launch scoping and runtime shadow modules.",
    },
    "rocsparse-spmv": {
        "classification": "Device branch coverage via loader-scoped fixed branches",
        "reports": ["rocsparse_spmv_report.jsonl"],
        "limitation": "The target is non-persistent today because the rocSPARSE handle path is not stable after AFL's deferred forkserver.",
    },
    "rocsolver-getrf": {
        "classification": "Device branch coverage via loader/runtime-shadow fixed branches",
        "reports": ["rocsolver_getrf_report.jsonl"],
        "limitation": "Coverage is wired through several short helper kernels; deeper edge identity needs more previous-BB safe sites.",
    },
    "miopen-activation": {
        "classification": "Device branch coverage through hybrid planning",
        "reports": ["miopen_activation_report.jsonl"],
        "limitation": "The launched activation kernel is small; the value is proving hybrid entry plus fixed-branch fallback on a real library path.",
    },
}


def load_summarizer(path):
    spec = importlib.util.spec_from_file_location("rocfuzz_patch_report_summary", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load summarizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_stats(path):
    stats = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            stats[key.strip()] = value.strip()
    return stats


def int_stat(stats, key):
    try:
        return int(float(stats.get(key, "0")))
    except ValueError:
        return 0


def find_runs(root, explicit_target):
    direct = root / "default" / "fuzzer_stats"
    if direct.exists():
        target = explicit_target or root.name
        return [(target, direct)]

    runs = []
    for stats in sorted(root.glob("*/default/fuzzer_stats")):
        target = stats.parent.parent.name
        if explicit_target is not None and target != explicit_target:
            continue
        runs.append((target, stats))
    order = {name: index for index, name in enumerate(TARGETS)}
    runs.sort(key=lambda run: (order.get(run[0], len(order)), run[0]))
    return runs


def format_int(value):
    return f"{value:,}"


def format_runtime(seconds):
    if seconds < 90:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def edge_text(stats):
    found = stats.get("edges_found", "?")
    total = stats.get("total_edges", "?")
    return f"{found} / {total}"


def finding_count(run_default, name, stat_value):
    directory = run_default / name
    if not directory.exists():
        return stat_value
    files = [path for path in directory.iterdir() if path.is_file() and not path.name.startswith("README")]
    return max(stat_value, len(files))


def campaign_revision(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "log", "-1", "--oneline"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize_patch_report(path, summarizer):
    if not path.exists():
        return None
    rows = summarizer.read_jsonl(str(path))
    return summarizer.summarize(rows)


def dict_text(values, limit=3):
    if not values:
        return "none"
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    text = ", ".join(f"{key}={value}" for key, value in items[:limit])
    if len(items) > limit:
        text += ", ..."
    return text


def result_text(stats, crashes, hangs):
    if crashes:
        return "Crashes found"
    if hangs:
        return "Hangs found"
    if int_stat(stats, "corpus_found"):
        return "Working; corpus grew, no crash/hang findings"
    return "Working; no crash/hang findings"


def write_report(args, runs, summarizer):
    now = _datetime.date.today().isoformat()
    repo_root = args.repo_root.resolve()
    revision = args.source_revision or campaign_revision(repo_root)
    lines = [
        f"# rocfuzz AFL campaign, {now}",
        "",
        "Campaign root:",
        f"`{args.campaign_root.resolve()}`",
        "",
        "Source revision:",
        f"`{revision}`",
        "",
        "Run policy:",
        "This report is generated from AFL++ `fuzzer_stats` plus the current rocfuzz patch-report JSONL summaries. It is a crash-focused campaign report; numeric mismatches are not treated as findings here. AFL runs normally leave `ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1` off so valid mutations that skip patched device edges do not become false crashes.",
        "",
        "## Summary",
        "",
        "| Target | Coverage classification | Coverage mix | Runtime | Execs | Corpus | New corpus | Edges | Bitmap coverage | Crashes | Hangs | Result |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    details = ["", "## Per-Example Details", ""]
    for target, stats_path in runs:
        stats = read_stats(stats_path)
        run_default = stats_path.parent
        crashes = finding_count(run_default, "crashes", int_stat(stats, "saved_crashes"))
        hangs = finding_count(run_default, "hangs", int_stat(stats, "saved_hangs"))
        target_info = TARGETS.get(target, {})
        classification = target_info.get("classification", "Unclassified coverage")
        report_summaries = []
        for rel in target_info.get("reports", []):
            summary = summarize_patch_report(args.examples_build / rel, summarizer)
            if summary is not None:
                report_summaries.append(summary)
        runtime = format_runtime(int_stat(stats, "run_time"))
        result = result_text(stats, crashes, hangs)
        lines.append(
            f"| `{target}` | {classification} | "
            f"{coverage_mix_from_summaries(report_summaries)} | {runtime} | "
            f"{format_int(int_stat(stats, 'execs_done'))} | "
            f"{format_int(int_stat(stats, 'corpus_count'))} | "
            f"{format_int(int_stat(stats, 'corpus_found'))} | "
            f"{edge_text(stats)} | {stats.get('bitmap_cvg', '?')} | "
            f"{crashes} | {hangs} | {result} |"
        )

        details.append(f"### `{target}`")
        details.append("")
        details.append(
            f"- AFL: {format_int(int_stat(stats, 'execs_done'))} execs in {runtime}, "
            f"{stats.get('execs_per_sec', '?')} exec/s, stability {stats.get('stability', '?')}, "
            f"target mode `{stats.get('target_mode', 'unknown').strip()}`."
        )
        details.append(
            f"- Corpus: {format_int(int_stat(stats, 'corpus_count'))} total, "
            f"{format_int(int_stat(stats, 'corpus_found'))} generated by this run, "
            f"max depth {stats.get('max_depth', '?')}."
        )
        details.append(f"- Findings: {crashes} crash artifacts, {hangs} hang artifacts.")

        report_paths = target_info.get("reports", [])
        if report_paths:
            details.append("- Patch reports:")
            for rel in report_paths:
                report_path = args.examples_build / rel
                summary = summarize_patch_report(report_path, summarizer)
                if summary is None:
                    details.append(f"  - `{report_path}`: missing.")
                    continue
                patches = summary.get("patches", {})
                deltas = summary.get("device_edge_delta", {})
                details.append(
                    f"  - `{report_path}`: patch events {patches.get('events', 0)}, "
                    f"successes {patches.get('successful_events', 0)}, "
                    f"edge sites {patches.get('edge_sites_patched', 0)} "
                    f"({patches.get('hashed_edge_sites', 0)} hashed, "
                    f"{patches.get('fixed_edge_sites', 0)} fixed, "
                    f"{patches.get('branch_edges_degraded_to_fixed', 0)} degraded), "
                    f"device edge slots {deltas.get('edge_slot_delta_count', 0)}, "
                    f"contexts {dict_text(patches.get('contexts', {}))}, "
                    f"low-edge causes {dict_text(patches.get('low_edge_causes', {}))}."
                )
                selected_edges = selected_edge_sample_text([summary])
                if selected_edges:
                    details.append(f"  - Selected edges: {selected_edges}.")
        else:
            details.append("- Patch reports: no report mapping configured for this target.")

        limitation = target_info.get("limitation")
        if limitation:
            details.append(f"- Known limitation: {limitation}")
        details.append("")

    return "\n".join(lines + details).rstrip() + "\n"


def main():
    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir.parent
    repo_root = examples_dir.parents[3]
    parser = argparse.ArgumentParser(
        description="Generate a rocfuzz AFL campaign markdown report."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--target", help="Restrict a single-target AFL output root.")
    parser.add_argument("--examples-build", type=Path, default=examples_dir / "build")
    parser.add_argument(
        "--summarizer",
        type=Path,
        default=examples_dir / "../afl-dbi/tools/summarize_patch_report.py",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = find_runs(args.campaign_root, args.target)
    if not runs:
        raise SystemExit(f"error: no AFL fuzzer_stats found under {args.campaign_root}")
    summarizer = load_summarizer(args.summarizer.resolve())
    report = write_report(args, runs, summarizer)
    if args.output is None:
        sys.stdout.write(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
