#!/usr/bin/env python3

import argparse
import datetime as _datetime
import hashlib
import importlib.util
import subprocess
from pathlib import Path

from patch_report_samples import coverage_mix_from_reports, selected_edge_sample_text


TARGETS = [
    {
        "name": "rocblas-sgemm",
        "reports": [
            ("high-edge fixed diagnostic", "high-edge-reports/rocblas_sgemm_high_edge.jsonl"),
            (
                "default hybrid",
                "high-edge-reports/rocblas_sgemm_default_hybrid_high_edge.jsonl",
            ),
        ],
        "limitation": (
            "The default path has useful previous-BB hashed Tensile coverage, but "
            "more sites still need safer relocation, opaque-instruction modeling, "
            "and temporary-register planning."
        ),
    },
    {
        "name": "rocfft-c2c",
        "reports": [
            ("high-edge FFT/twiddle", "high-edge-reports/rocfft_c2c_high_edge.jsonl"),
        ],
        "limitation": (
            "Generated FFT kernels produce device deltas; twiddle_gen_* is still "
            "reported as no-patchable-sites until its descriptor/prologue "
            "interaction is understood."
        ),
    },
    {
        "name": "rocrand-uniform",
        "reports": [("default", "rocrand_uniform_report.jsonl")],
        "showmap": "rocfuzz_example_rocrand_uniform_showmap",
        "limitation": (
            "Coverage is shallow but validates launch-scoped KPACK/HSA-reader "
            "coverage and runtime shadow launch redirection."
        ),
    },
    {
        "name": "rocsparse-spmv",
        "reports": [("default", "rocsparse_spmv_report.jsonl")],
        "showmap": "rocfuzz_example_rocsparse_spmv_showmap",
        "limitation": (
            "The retained wrapper uses the non-persistent target because the "
            "rocSPARSE handle path is not yet stable across AFL deferred "
            "forkserver boundaries."
        ),
    },
    {
        "name": "rocsolver-getrf",
        "reports": [("default", "rocsolver_getrf_report.jsonl")],
        "limitation": (
            "Coverage is wired through short helper kernels; deeper edge identity "
            "needs more previous-BB-safe sites."
        ),
    },
    {
        "name": "miopen-activation",
        "reports": [
            ("default", "miopen_activation_report.jsonl"),
            ("high-edge fixed diagnostic", "high-edge-reports/miopen_activation_high_edge.jsonl"),
        ],
        "limitation": (
            "Activation kernels are small. The current value is proving that the "
            "entry-unsafe MIOpen path uses self-contained fixed branch counters "
            "without entry redirection."
        ),
    },
]


def load_summarizer(path):
    spec = importlib.util.spec_from_file_location("rocfuzz_patch_report_summary", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load summarizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_revision(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "log", "-1", "--oneline"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def display_path(path, repo_root):
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def counter_text(values, limit=3):
    if not values:
        return "none"
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    text = ", ".join(f"`{key}`={value}" for key, value in items[:limit])
    if len(items) > limit:
        text += ", ..."
    return text


def word_list_text(words, limit=2):
    if not isinstance(words, list) or not words:
        return ""
    formatted = []
    for word in words[:limit]:
        if isinstance(word, int):
            formatted.append(f"0x{word:08x}")
        else:
            formatted.append(str(word))
    text = ", ".join(formatted)
    if len(words) > limit:
        text += ", ..."
    return text


def sampled_skip_instruction_text(reports, limit=5):
    counts = {}
    examples = {}
    for report in reports:
        for kernel in report["summary"].get("kernels", []):
            for sample in kernel.get("sampled_skip_instructions", []):
                if not isinstance(sample, dict):
                    continue
                reason = sample.get("reason")
                mnemonic = sample.get("mnemonic")
                if not reason or not mnemonic:
                    continue
                key = (str(reason), str(mnemonic))
                counts[key] = counts.get(key, 0) + 1
                examples.setdefault(key, sample)
    if not counts:
        return ""

    parts = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        reason, mnemonic = key
        sample = examples[key]
        words = word_list_text(sample.get("words"))
        detail = f"`{mnemonic}` for `{reason}`"
        if count > 1:
            detail += f" x{count}"
        if words:
            detail += f" ({words})"
        parts.append(detail)
    if len(counts) > limit:
        parts.append("...")
    return ", ".join(parts)


def sum_field(reports, section, field):
    return sum(report["summary"].get(section, {}).get(field, 0) for report in reports)


FALLBACK_CAUSE_FIELDS = (
    ("aggregate-cap", "fixed_counter_branch_edge_aggregate_fallback_used"),
    ("exec-safety", "fixed_counter_branch_edge_safety_fallback_used"),
    ("liveness", "fixed_counter_branch_edge_liveness_fallback_used"),
    ("placement", "fixed_counter_branch_edge_placement_fallback_used"),
)


def fixed_fallback_cause_counts(reports):
    return {
        label: sum_field(reports, "patches", field)
        for label, field in FALLBACK_CAUSE_FIELDS
    }


def fixed_fallback_cause_text(counts):
    nonzero = [(label, count) for label, count in counts.items() if count]
    if not nonzero:
        return "none"
    return ", ".join(f"{label}={count}" for label, count in nonzero)


def patch_accounting(reports):
    return {
        "events": sum_field(reports, "patches", "events"),
        "successful_events": sum_field(reports, "patches", "successful_events"),
        "patched_sites": sum_field(reports, "patches", "edge_sites_patched"),
        "policy_hashed_sites": sum_field(reports, "patches", "hashed_edge_sites"),
        "policy_fixed_sites": sum_field(reports, "patches", "fixed_edge_sites"),
        "degraded_branch_edges": sum_field(
            reports, "patches", "branch_edges_degraded_to_fixed"
        ),
        "previous_bb_branch_sites": sum_field(
            reports, "patches", "previous_bb_branch_sites_selected"
        ),
        "previous_bb_branch_site_fallbacks": sum_field(
            reports, "patches", "previous_bb_branch_sites_degraded_to_fixed"
        ),
        "fixed_fallback_causes": fixed_fallback_cause_counts(reports),
    }


def patch_accounting_from_summary(summary):
    return {
        "events": summary["patches"]["events"],
        "successful_events": summary["patches"]["successful_events"],
        "patched_sites": summary["patches"]["edge_sites_patched"],
        "policy_hashed_sites": summary["patches"]["hashed_edge_sites"],
        "policy_fixed_sites": summary["patches"]["fixed_edge_sites"],
        "degraded_branch_edges": summary["patches"]["branch_edges_degraded_to_fixed"],
        "previous_bb_branch_sites": summary["patches"][
            "previous_bb_branch_sites_selected"
        ],
        "previous_bb_branch_site_fallbacks": summary["patches"][
            "previous_bb_branch_sites_degraded_to_fixed"
        ],
        "fixed_fallback_causes": {
            label: summary["patches"].get(field, 0)
            for label, field in FALLBACK_CAUSE_FIELDS
        },
    }


def merge_counter(reports, section, field):
    merged = {}
    for report in reports:
        values = report["summary"].get(section, {}).get(field, {})
        for key, value in values.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def collect_patch_values(rows, field):
    values = {}
    for row in rows:
        if row.get("event") != "patch_device_elf":
            continue
        value = row.get(field)
        if value in (None, ""):
            continue
        values[str(value)] = values.get(str(value), 0) + 1
    return values


def read_report(path, summarizer):
    if not path.exists():
        return None
    rows = summarizer.read_jsonl(str(path))
    return {
        "path": path,
        "rows": rows,
        "summary": summarizer.summarize(rows),
        "strategies": collect_patch_values(rows, "coverage_strategy"),
        "strategy_reasons": collect_patch_values(rows, "coverage_strategy_reason"),
        "branch_policies": collect_patch_values(rows, "branch_edge_slot_policy"),
    }


def classify_target(reports):
    if not reports:
        return "missing reports"
    edge_sites = sum_field(reports, "patches", "edge_sites_patched")
    delta_slots = sum_field(reports, "device_edge_delta", "edge_slot_delta_count")
    failures = merge_counter(reports, "patches", "reasons")
    if edge_sites > 0 and delta_slots > 0:
        if failures.get("no_patchable_sites", 0) > 0:
            return "device branch coverage with known skipped payload"
        return "device branch coverage"
    if edge_sites > 0:
        return "patched device code, no observed edge delta"
    if delta_slots > 0:
        return "device delta without selected edge-site evidence"
    return "launch/report-only or blocked"


def read_showmap(build_dir, base_name):
    if not base_name:
        return None
    out = []
    for suffix in ("a", "b"):
        path = build_dir / f"{base_name}.{suffix}.device"
        if not path.exists():
            return None
        data = path.read_bytes()
        lines = [line for line in data.splitlines() if line.strip()]
        digest = hashlib.sha256(data).hexdigest()[:12] if data else "empty"
        out.append((path, len(lines), digest))
    return out


def status_detail(reports):
    if not reports:
        return "No JSONL report was found."
    accounting = patch_accounting(reports)
    delta_slots = sum_field(reports, "device_edge_delta", "edge_slot_delta_count")
    delta_total = sum_field(reports, "device_edge_delta", "edge_counter_delta_total")
    return (
        f"{accounting['successful_events']}/{accounting['events']} patch events "
        f"succeeded, {accounting['patched_sites']} edge sites were patched; "
        f"policy accounting selected hashed sites={accounting['policy_hashed_sites']}, "
        f"fixed sites={accounting['policy_fixed_sites']}, degraded logical "
        f"branch edges={accounting['degraded_branch_edges']}, previous-BB branch "
        f"sites={accounting['previous_bb_branch_sites']}, and previous-BB sites "
        f"degraded to fixed={accounting['previous_bb_branch_site_fallbacks']} "
        f"({fixed_fallback_cause_text(accounting['fixed_fallback_causes'])}); "
        f"{delta_slots} device slots changed with counter delta {delta_total}."
    )


def format_report_detail(label, report, repo_root):
    summary = report["summary"]
    accounting = patch_accounting_from_summary(summary)
    deltas = summary["device_edge_delta"]
    path = display_path(report["path"], repo_root)
    return (
        f"- `{label}` (`{path}`): {accounting['successful_events']}/"
        f"{accounting['events']} successful patch events, "
        f"{accounting['patched_sites']} patched sites; policy accounting "
        f"hashed={accounting['policy_hashed_sites']}, "
        f"fixed={accounting['policy_fixed_sites']}, "
        f"degraded_branch_edges={accounting['degraded_branch_edges']}, "
        f"previous_bb_branch_sites={accounting['previous_bb_branch_sites']}, "
        f"previous_bb_site_fallbacks={accounting['previous_bb_branch_site_fallbacks']}, "
        f"fixed_fallback_causes={fixed_fallback_cause_text(accounting['fixed_fallback_causes'])}, "
        f"{deltas['edge_slot_delta_count']} device slots, strategies "
        f"{counter_text(report['strategies'])}, reasons "
        f"{counter_text(report['strategy_reasons'])}."
    )


def write_report(args, targets, out):
    today = _datetime.date.today().isoformat()
    revision = args.source_revision or repo_revision(args.repo_root)
    print(f"# rocfuzz coverage baseline, {today}", file=out)
    print("", file=out)
    print(
        "This report is generated from the maintained real-library smoke and "
        "patch-report gates. It is a coverage-quality baseline, not an AFL crash "
        "campaign report; crash and hang findings are reported by "
        "`summarize-afl-campaign.py` from `afl-fuzz` output roots.",
        file=out,
    )
    print("", file=out)
    print(f"Source revision: `{revision}`", file=out)
    print(f"Examples build: `{display_path(args.examples_build, args.repo_root)}`", file=out)
    print("", file=out)
    print("## Summary", file=out)
    print("", file=out)
    print(
        "`Patched sites` is the successfully applied instrumentation count; "
        "`Device slots` is the observed runtime delta count. `Policy hashed`, "
        "`Policy fixed`, `Degraded branch edges`, `PrevBB sites`, and "
        "`PrevBB fallbacks` come from planner accounting in patch reports and "
        "can be higher when failed or partially patched events kept diagnostic "
        "plan data.",
        file=out,
    )
    print("", file=out)
    print(
        "| Example | Status | Coverage mix | Patch events | Patched sites | "
        "Policy hashed | Policy fixed | Degraded branch edges | PrevBB sites | "
        "PrevBB fallbacks | Device slots | Main low-edge cause | Findings |",
        file=out,
    )
    print(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        file=out,
    )
    prepared = []
    for target in targets:
        reports = []
        missing = []
        for label, relpath in target["reports"]:
            path = args.examples_build / relpath
            report = read_report(path, args.summarizer_module)
            if report is None:
                missing.append(path)
            else:
                reports.append((label, report))
        report_values = [report for _, report in reports]
        low_edge_causes = merge_counter(report_values, "patches", "low_edge_causes")
        degradation_causes = merge_counter(report_values, "patches", "degradation_reasons")
        main_cause = counter_text(low_edge_causes, limit=1)
        status = "missing reports" if missing else classify_target(report_values)
        findings = "not assessed by smoke baseline"
        accounting = patch_accounting(report_values)
        print(
            f"| `{target['name']}` | {status} | "
            f"{coverage_mix_from_reports(report_values)} | "
            f"{accounting['events']} | "
            f"{accounting['patched_sites']} | "
            f"{accounting['policy_hashed_sites']} | "
            f"{accounting['policy_fixed_sites']} | "
            f"{accounting['degraded_branch_edges']} | "
            f"{accounting['previous_bb_branch_sites']} | "
            f"{accounting['previous_bb_branch_site_fallbacks']} | "
            f"{sum_field(report_values, 'device_edge_delta', 'edge_slot_delta_count')} | "
            f"{main_cause} | {findings} |",
            file=out,
        )
        prepared.append((target, reports, missing, low_edge_causes, degradation_causes))

    print("", file=out)
    print("## Per-Example Details", file=out)
    for target, reports, missing, low_edge_causes, degradation_causes in prepared:
        print("", file=out)
        print(f"### `{target['name']}`", file=out)
        print("", file=out)
        report_values = [report for _, report in reports]
        print(f"- Status: {classify_target(report_values) if not missing else 'missing reports'}.", file=out)
        print(f"- Coverage mix: {coverage_mix_from_reports(report_values)}.", file=out)
        print(f"- Progress: {status_detail(report_values)}", file=out)
        if missing:
            for path in missing:
                print(f"- Missing report: `{path}`.", file=out)
        for label, report in reports:
            print(format_report_detail(label, report, args.repo_root), file=out)
            selected_edges = selected_edge_sample_text([report["summary"]])
            if selected_edges:
                print(f"- Selected edges in `{label}`: {selected_edges}.", file=out)
        if low_edge_causes:
            print(f"- Low-edge attribution: {counter_text(low_edge_causes, limit=4)}.", file=out)
        if degradation_causes:
            print(
                f"- Coverage degradation attribution: "
                f"{counter_text(degradation_causes, limit=4)}.",
                file=out,
            )
        skip_instruction_samples = sampled_skip_instruction_text(report_values)
        if skip_instruction_samples:
            print(
                f"- Sampled skipped instructions: {skip_instruction_samples}.",
                file=out,
            )
        showmap = read_showmap(args.examples_build, target.get("showmap"))
        if showmap is not None:
            a_path, a_count, a_digest = showmap[0]
            b_path, b_count, b_digest = showmap[1]
            relation = "different" if a_digest != b_digest else "identical"
            print(
                f"- AFL-visible device showmap: `{a_path.name}` has {a_count} "
                f"device tuples, `{b_path.name}` has {b_count}, hashes are {relation}.",
                file=out,
            )
        print(f"- Findings: smoke baseline only; no crash/hang campaign findings assessed.", file=out)
        print(f"- Limitation: {target['limitation']}", file=out)


def main():
    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir.parent
    repo_root = examples_dir.parents[3]
    parser = argparse.ArgumentParser(
        description="Generate a rocfuzz smoke-level coverage baseline report."
    )
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
    args.summarizer_module = load_summarizer(args.summarizer)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as out:
            write_report(args, TARGETS, out)
    else:
        import sys

        write_report(args, TARGETS, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
