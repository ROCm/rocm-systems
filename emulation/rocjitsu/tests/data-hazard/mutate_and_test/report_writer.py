# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report writers for hazard mutation testing results."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .models import DetectionLabel, MutantResult, ShaderReport

if TYPE_CHECKING:
    from .benchmark import Benchmarker


def status_label(m: MutantResult) -> str:
    if not m.compiled:
        return "BUILD_FAIL"
    if m.exit_code == -1:
        return "TIMEOUT"
    if not m.ran:
        return "RUN_FAIL"
    if m.correct:
        return "MATCH"
    if not m.stdout_match:
        return "WRONG_OUTPUT"
    return f"EXIT_{m.exit_code}"


def is_false_negative(m: MutantResult, baseline_hazard_count: int) -> bool:
    """A mutant is a false negative when plugin did not report any *new* hazards beyond the baseline."""
    if not m.ran:
        return False
    return m.hazard_count <= baseline_hazard_count


def detection_label(m: MutantResult, baseline_hazard_count: int) -> DetectionLabel:
    """Return a detection-quality verdict for a mutant."""
    if not m.ran:
        return DetectionLabel.NA
    if is_false_negative(m, baseline_hazard_count):
        return DetectionLabel.FN
    if m.hazard_count > baseline_hazard_count:
        return DetectionLabel.DETECTED
    return DetectionLabel.NA


def compute_summary(reports: List[ShaderReport]) -> dict:
    """Compute aggregate summary statistics.

    A mutant that never executed says nothing about the detector, so it is
    counted as inconclusive rather than killed and is excluded from the
    mutation score. Counting it as killed turns a run where every process
    crashed into a 100% score.
    """
    total = sum(len(r.mutants) for r in reports)
    killed = sum(1 for r in reports for m in r.mutants if m.ran and not m.correct)
    inconclusive = sum(1 for r in reports for m in r.mutants if not m.ran)
    survived = total - killed - inconclusive
    scored = total - inconclusive
    by_status: Dict[str, int] = {}
    for r in reports:
        for m in r.mutants:
            sl = status_label(m)
            by_status[sl] = by_status.get(sl, 0) + 1

    # False positives: baseline (unmodified) shaders where hazards were reported
    false_positives = sum(
        1 for r in reports if r.build_ok and r.baseline_hazard_count > 0
    )
    # False negatives: killed mutants (wait was needed) where no new hazards detected
    false_negatives = sum(
        1
        for r in reports
        for m in r.mutants
        if is_false_negative(m, r.baseline_hazard_count)
    )

    # Confusion matrix: plugin detection vs ground truth
    tp = fn = fp = tn = na = 0
    for r in reports:
        if r.build_ok:
            if r.baseline_hazard_count > 0:
                fp += 1
            else:
                tn += 1
        for m in r.mutants:
            if not m.ran:
                na += 1
                continue
            if m.hazard_count > r.baseline_hazard_count:
                tp += 1
            else:
                fn += 1

    return {
        "total_shaders": len(reports),
        "total_mutants": total,
        "killed": killed,
        "survived": survived,
        "inconclusive": inconclusive,
        "scored_mutants": scored,
        "mutation_score": round(killed / scored * 100, 1) if scored else 0.0,
        "by_status": by_status,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "na": na,
    }


def extract_summary_section_markdown(md_text: str) -> str:
    """
    Return only the ``## Summary`` block (metrics table) from a full mutation report.

    Stops before ``### Confusion Matrix`` or ``## Results by Shader`` so CI logs stay compact.
    """
    start = md_text.find("## Summary")
    if start < 0:
        return "(no ## Summary section in report)"
    tail = md_text[start:]
    end = len(tail)
    for marker in ("\n## Results by Shader", "\n### Confusion Matrix"):
        i = tail.find(marker)
        if i != -1:
            end = min(end, i)
    return tail[:end].rstrip()


STATUS_EMOJI = {
    "MATCH": "\u2705",
    "WRONG_OUTPUT": "\u274c",
    "BUILD_FAIL": "\U0001f6a7",
    "TIMEOUT": "\u23f0",
    "RUN_FAIL": "\U0001f4a5",
}


class AbstractReportWriter(ABC):
    """Base class for writers that persist results to a file."""

    @abstractmethod
    def write(
        self, reports: List[ShaderReport], path: Path, *, arch: str = ""
    ) -> None: ...


class JSONReportWriter(AbstractReportWriter):
    """Write full results as machine-readable JSON."""

    def write(self, reports: List[ShaderReport], path: Path, *, arch: str = "") -> None:
        summary = compute_summary(reports)

        shaders = []
        for rpt in reports:
            shader_total = len(rpt.mutants)
            shader_killed = sum(1 for m in rpt.mutants if m.ran and not m.correct)
            shader_inconclusive = sum(1 for m in rpt.mutants if not m.ran)
            shader_fp = 1 if rpt.build_ok and rpt.baseline_hazard_count > 0 else 0
            shader_fn = sum(
                1
                for m in rpt.mutants
                if is_false_negative(m, rpt.baseline_hazard_count)
            )
            d = {
                "shader": rpt.shader,
                "asm_file": rpt.asm_file,
                "asm_content": rpt.asm_content,
                "compile_ok": rpt.compile_ok,
                "build_ok": rpt.build_ok,
                "error": rpt.error,
                "baseline_ok": rpt.baseline_ok,
                "baseline_exit_code": rpt.baseline_exit_code,
                "baseline_stdout": rpt.baseline_stdout,
                "baseline_hazard_count": rpt.baseline_hazard_count,
                "baseline_false_positive": shader_fp > 0,
                "baseline_hazard_report_path": rpt.baseline_hazard_report_path,
                "total_waits": len(rpt.waits_found),
                "killed": shader_killed,
                "survived": shader_total - shader_killed - shader_inconclusive,
                "inconclusive": shader_inconclusive,
                "false_negatives": shader_fn,
                "waits_found": [
                    {
                        "line": w.line_number,
                        "instruction": w.instruction,
                        "operand": w.operand,
                    }
                    for w in rpt.waits_found
                ],
                "mutants": [
                    {
                        "index": m.wait_index,
                        "instruction": m.wait_instruction,
                        "line": m.wait_line,
                        "status": status_label(m),
                        "compiled": m.compiled,
                        "linked": m.linked,
                        "ran": m.ran,
                        "correct": m.correct,
                        "exit_code": m.exit_code,
                        "stdout_match": m.stdout_match,
                        "stdout": m.stdout,
                        "error": m.error,
                        "hazard_count": m.hazard_count,
                        "hazard_report_path": m.hazard_report_path,
                    }
                    for m in rpt.mutants
                ],
            }
            shaders.append(d)

        report = {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": arch,
            "summary": summary,
            "shaders": shaders,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to {path}")


class CSVReportWriter(AbstractReportWriter):
    """Write a flat CSV of all mutant results."""

    def write(self, reports: List[ShaderReport], path: Path, *, arch: str = "") -> None:
        del arch  # unused; signature matches base for polymorphic use
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "shader",
                    "mutant_index",
                    "wait_instruction",
                    "asm_line",
                    "status",
                    "exit_code",
                    "stdout_match",
                    "hazard_count",
                    "detection",
                    "error",
                ]
            )
            for rpt in reports:
                for m in rpt.mutants:
                    det = detection_label(m, rpt.baseline_hazard_count)
                    w.writerow(
                        [
                            m.shader,
                            m.wait_index,
                            m.wait_instruction,
                            m.wait_line,
                            status_label(m),
                            m.exit_code,
                            m.stdout_match,
                            m.hazard_count,
                            det,
                            m.error,
                        ]
                    )
        print(f"CSV report written to {path}")


class MarkdownReportWriter(AbstractReportWriter):
    """Write a GitHub-flavored Markdown report."""

    _DETECTION_EMOJI = {
        DetectionLabel.FN: "\u274c FN",
        DetectionLabel.DETECTED: "\u2705 Detected",
        DetectionLabel.FP: "\u26a0\ufe0f FP",
        DetectionLabel.OK: "\u2705 OK",
    }

    def _truncate_preview(
        self, text: str, *, total_length: int, limit: int = 2000
    ) -> str:
        stripped = text.rstrip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit] + f"\n... ({total_length} chars total, truncated)"

    def _md_table_cell_escape(self, s: str) -> str:
        return s.rstrip().replace("|", "\\|").replace("\n", "&NewLine;")

    def _append_details_fenced(
        self,
        summary_text: str,
        body: str,
        fence_lang: str = "",
    ) -> None:
        lines = self.lines
        lines.append("<details>")
        lines.append(f"<summary>{summary_text}</summary>\n")
        if fence_lang:
            lines.append(f"```{fence_lang}")
        else:
            lines.append("```")
        lines.append(body)
        lines.append("```\n")
        lines.append("</details>\n")

    def _render_confusion_matrix_html(self, summary: dict) -> str:
        """Return an inline HTML table confusion matrix for the markdown report."""
        tp = summary["tp"]
        fn_val = summary["fn"]
        fp_val = summary["fp"]
        tn = summary["tn"]
        na = summary["na"]

        def bg(val: int, good: bool) -> str:
            if val == 0:
                return "#f5f5f5"
            return "#c6efce" if good else "#ffc7ce"

        return (
            "<table>\n"
            "  <tr>\n"
            '    <td colspan="2" rowspan="2"></td>\n'
            '    <th colspan="2" align="center" style="padding:8px 16px">'
            "<strong>Did find hazard?</strong></th>\n"
            "  </tr>\n"
            "  <tr>\n"
            '    <th align="center" style="padding:6px 12px">\u2705 Yes</th>\n'
            '    <th align="center" style="padding:6px 12px">\u274c No</th>\n'
            "  </tr>\n"
            "  <tr>\n"
            '    <th rowspan="2" style="padding:8px 12px">'
            "<strong>Should have<br/>found hazard?</strong></th>\n"
            f'    <th align="center" style="padding:6px 12px">\u2705 Yes</th>\n'
            f'    <td align="center" style="padding:12px 24px;background:{bg(tp, True)}">'
            f"<strong>{tp}</strong><br/><small>True Positive</small></td>\n"
            f'    <td align="center" style="padding:12px 24px;background:{bg(fn_val, False)}">'
            f"<strong>{fn_val}</strong><br/><small>False Negative</small></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            f'    <th align="center" style="padding:6px 12px">\u274c No</th>\n'
            f'    <td align="center" style="padding:12px 24px;background:{bg(fp_val, False)}">'
            f"<strong>{fp_val}</strong><br/><small>False Positive</small></td>\n"
            f'    <td align="center" style="padding:12px 24px;background:{bg(tn, True)}">'
            f"<strong>{tn}</strong><br/><small>True Negative</small></td>\n"
            "  </tr>\n"
            "</table>\n"
            f"\n<sub>Not applicable (build fail / timeout): {na}</sub>\n"
        )

    def write(self, reports: List[ShaderReport], path: Path, *, arch: str = "") -> None:
        self.lines = []
        summary = compute_summary(reports)
        self._append_document_header(arch)
        self._append_summary_section(summary, reports)
        self._append_results_by_shader(reports)
        self._append_legend()
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"Markdown report written to {path}")

    def _append_document_header(self, arch: str) -> None:
        lines = self.lines
        lines.append("# \U0001f9ea Mutation Testing Report\n")
        lines.append(
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        if arch:
            lines.append(f">")
            lines.append(f"> Architecture: `{arch}`")
        lines.append("")

    def _append_summary_section(
        self, summary: dict, reports: List[ShaderReport]
    ) -> None:
        lines = self.lines
        lines.append("## Summary\n")
        score = summary["mutation_score"]
        lines.append(f"| Metric | Value |")
        lines.append(f"| --- | --- |")
        lines.append(f"| Shaders tested | {summary['total_shaders']} |")
        lines.append(f"| Total mutants | {summary['total_mutants']} |")
        lines.append(f"| Killed | {summary['killed']} |")
        lines.append(f"| Survived | {summary['survived']} |")
        if summary["inconclusive"]:
            lines.append(f"| Inconclusive (never ran) | {summary['inconclusive']} |")
        lines.append(
            f"| **Mutation score** | **{score}%** of {summary['scored_mutants']} that ran |"
        )
        fp = summary["false_positives"]
        fn = summary["false_negatives"]
        if fp or fn:
            lines.append(f"| **False positives** (baseline hazards) | **{fp}** |")
            lines.append(f"| **False negatives** (missed hazards) | **{fn}** |")
        lines.append("")

        has_hazard_report = any(r.baseline_hazard_report_raw for r in reports)
        if has_hazard_report:
            lines.append("### Confusion Matrix\n")
            lines.append(self._render_confusion_matrix_html(summary))
            lines.append("")

        if summary["by_status"]:
            lines.append("<details>\n<summary>Breakdown by status</summary>\n")
            lines.append("| Status | Count |")
            lines.append("| --- | ---: |")
            for st, cnt in sorted(summary["by_status"].items()):
                emoji = STATUS_EMOJI.get(st, "")
                lines.append(f"| {emoji} {st} | {cnt} |")
            lines.append("\n</details>\n")

    def _append_results_by_shader(self, reports: List[ShaderReport]) -> None:
        lines = self.lines
        lines.append("## Results by Shader\n")
        for rpt in reports:
            self._append_one_shader(rpt)

    def _baseline_badge(self, rpt: ShaderReport) -> str:
        if rpt.baseline_ok:
            return "\u2705 PASS"
        if not rpt.compile_ok:
            return "\U0001f6a7 COMPILE_FAIL"
        if not rpt.build_ok:
            return "\U0001f6a7 BUILD_FAIL"
        return "\u274c FAIL"

    def _append_one_shader(self, rpt: ShaderReport) -> None:
        lines = self.lines
        shader_total = len(rpt.mutants)
        shader_killed = sum(1 for m in rpt.mutants if m.ran and not m.correct)
        baseline_badge = self._baseline_badge(rpt)

        lines.append(f"### `{rpt.shader}`\n")
        lines.append(f"| | |")
        lines.append(f"| --- | --- |")
        if rpt.compile_ok and rpt.build_ok:
            lines.append(
                f"| Baseline | {baseline_badge} (exit {rpt.baseline_exit_code}) |"
            )
        else:
            lines.append(f"| Baseline | {baseline_badge} |")
        lines.append(f"| Assembly | `{rpt.asm_file}` |")
        lines.append(f"| Wait instructions | {len(rpt.waits_found)} |")
        if shader_total:
            lines.append(f"| Killed / Total | {shader_killed} / {shader_total} |")
        if rpt.baseline_hazard_count > 0:
            lines.append(
                f"| Baseline hazards | \u274c **{rpt.baseline_hazard_count}** (false positive) |"
            )
        lines.append("")

        self._append_shader_collapsibles(rpt)

        if not rpt.mutants:
            lines.append("_No mutants generated._\n")
            return

        shader_has_hazard_report = rpt.baseline_hazard_report_raw or any(
            m.hazard_report_raw for m in rpt.mutants
        )

        self._append_mutant_table(rpt, shader_has_hazard_report)
        lines.append("")

        self._append_divergent_stdout_section(rpt)

    def _append_shader_collapsibles(self, rpt: ShaderReport) -> None:
        if rpt.error:
            self._append_details_fenced("Build error", rpt.error.rstrip(), "")

        if rpt.baseline_stdout:
            stdout_preview = self._truncate_preview(
                rpt.baseline_stdout, total_length=len(rpt.baseline_stdout)
            )
            self._append_details_fenced("Baseline stdout", stdout_preview, "")

        if rpt.asm_content:
            self._append_details_fenced(
                f"Assembly ({rpt.shader}.s)",
                rpt.asm_content.rstrip(),
                "asm",
            )

        if rpt.baseline_hazard_report_raw:
            self._append_details_fenced(
                f"Baseline hazard report ({rpt.baseline_hazard_count} hazards)",
                rpt.baseline_hazard_report_raw.rstrip(),
                "json",
            )

    def _detection_cell(self, m: MutantResult, baseline_hazard_count: int) -> str:
        det = detection_label(m, baseline_hazard_count)
        det_emoji = self._DETECTION_EMOJI.get(det, "\u2014")
        if m.hazard_report_raw:
            h_escaped = self._md_table_cell_escape(m.hazard_report_raw)
            return (
                f"{det_emoji} <details><summary>{m.hazard_count} hazard(s)</summary>"
                f"<pre>{h_escaped}</pre></details>"
            )
        return f"{det_emoji} ({m.hazard_count})"

    def _mutant_notes_cell(self, m: MutantResult) -> str:
        if m.stderr:
            escaped = self._md_table_cell_escape(m.stderr)
            return f"<details><summary>stderr</summary><pre>{escaped}</pre></details>"
        if m.error:
            escaped = self._md_table_cell_escape(m.error)
            return f"<details><summary>stderr</summary><pre>{escaped}</pre></details>"
        return ""

    def _append_mutant_table(
        self, rpt: ShaderReport, shader_has_hazard_report: bool
    ) -> None:
        lines = self.lines
        if shader_has_hazard_report:
            lines.append("| # | Line | Instruction | Status | Detection | Notes |")
            lines.append("| ---: | ---: | --- | --- | --- | --- |")
        else:
            lines.append("| # | Line | Instruction | Status | Notes |")
            lines.append("| ---: | ---: | --- | --- | --- |")
        for m in rpt.mutants:
            sl = status_label(m)
            emoji = STATUS_EMOJI.get(sl, "")
            notes = self._mutant_notes_cell(m)
            if shader_has_hazard_report:
                det_cell = self._detection_cell(m, rpt.baseline_hazard_count)
                lines.append(
                    f"| {m.wait_index} | {m.wait_line} | `{m.wait_instruction}` "
                    f"| {emoji} {sl} | {det_cell} | {notes} |"
                )
            else:
                lines.append(
                    f"| {m.wait_index} | {m.wait_line} | `{m.wait_instruction}` "
                    f"| {emoji} {sl} | {notes} |"
                )

    def _append_divergent_stdout_section(self, rpt: ShaderReport) -> None:
        lines = self.lines
        divergent = [
            m for m in rpt.mutants if m.ran and not m.stdout_match and m.stdout
        ]
        if not divergent:
            return
        lines.append("<details>")
        lines.append("<summary>Mutant stdout diffs</summary>\n")
        for m in divergent:
            lines.append(
                f"**Mutant {m.wait_index}** — `{m.wait_instruction}` "
                f"(line {m.wait_line}):\n"
            )
            mut_preview = self._truncate_preview(m.stdout, total_length=len(m.stdout))
            lines.append("```")
            lines.append(mut_preview)
            lines.append("```\n")
        lines.append("</details>\n")

    def _append_legend(self) -> None:
        lines = self.lines
        lines.append("## Legend\n")
        lines.append("### Status\n")
        lines.append("| Status | Meaning |")
        lines.append("| --- | --- |")
        lines.append(
            "| \u2705 MATCH | Mutant produced correct output — wait may be redundant |"
        )
        lines.append(
            "| \u274c WRONG_OUTPUT | Output differs from baseline — **data hazard detected** |"
        )
        lines.append("| EXIT\\_N | Non-zero exit code (crash or assertion) |")
        lines.append("| \u23f0 TIMEOUT | Execution exceeded timeout |")
        lines.append(
            "| \U0001f6a7 BUILD_FAIL | Modified assembly failed to assemble or link |"
        )
        lines.append(
            "| \U0001f4a5 RUN_FAIL | Executable failed to start; inconclusive, "
            "not counted in the mutation score |"
        )
        lines.append("")
        lines.append("### Detection\n")
        lines.append(
            "The Detection column merges hazard count and detection verdict.\n"
        )
        lines.append("| Detection | Meaning |")
        lines.append("| --- | --- |")
        lines.append(
            "| \u2705 Detected | Killed mutant with hazards reported — plugin correctly caught the issue |"
        )
        lines.append(
            "| \u274c FN | **False negative** — mutant was killed but plugin reported no hazards |"
        )
        lines.append(
            "| \u26a0\ufe0f FP | **False positive** — mutant survived but plugin reported hazards |"
        )
        lines.append(
            "| \u2705 OK | Mutant survived and no hazards reported — consistent |"
        )
        lines.append("| \u2014 | Not applicable (build failure, timeout, etc.) |")
        lines.append("")


class TerminalReportWriter:
    """Prints the human-readable mutation matrix to stdout (not a file)."""

    def write(
        self, reports: List[ShaderReport], benchmarker: Optional["Benchmarker"] = None
    ) -> None:
        """Print a human-readable matrix to the terminal."""
        print("\n" + "=" * 80)
        print("MUTATION TESTING RESULTS")
        print("=" * 80)

        has_hazard_report = any(r.baseline_hazard_report_raw for r in reports)
        benchmark_active = benchmarker is not None and benchmarker.enabled

        for rpt in reports:
            print(f"\n--- {rpt.shader} ---")
            print(f"    Assembly: {rpt.asm_file}")
            if not rpt.compile_ok:
                print(f"    Baseline: COMPILE_FAIL — {rpt.error}")
            elif not rpt.build_ok:
                print(f"    Baseline: BUILD_FAIL — {rpt.error}")
            else:
                print(
                    f"    Baseline: {'PASS' if rpt.baseline_ok else 'FAIL'} "
                    f"(exit {rpt.baseline_exit_code})"
                )
            if benchmark_active and benchmarker.has(rpt.shader):
                print(f"    Baseline time: {benchmarker.elapsed(rpt.shader):.3f}s")
            if has_hazard_report:
                fp_tag = " (FALSE POSITIVE)" if rpt.baseline_hazard_count > 0 else ""
                print(f"    Baseline hazards: {rpt.baseline_hazard_count}{fp_tag}")
            print(f"    Wait instructions found: {len(rpt.waits_found)}")

            if not rpt.mutants:
                print("    (no mutants)")
                continue

            time_hdr = f"  {'Time':>10}" if benchmark_active else ""
            if has_hazard_report:
                hdr = (
                    f"    {'#':>3}  {'Line':>5}  {'Instruction':<28}  "
                    f"{'Status':<14}  {'Detection':>15}{time_hdr}  Notes"
                )
            else:
                hdr = (
                    f"    {'#':>3}  {'Line':>5}  {'Instruction':<28}  "
                    f"{'Status':<14}{time_hdr}  Notes"
                )
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            for m in rpt.mutants:
                sl = status_label(m)
                notes = "(see below)" if m.error else ""
                time_col = ""
                if benchmark_active:
                    mut_label = f"{rpt.shader}_mut{m.wait_index}"
                    t_str = (
                        f"{benchmarker.elapsed(mut_label):.3f}s"
                        if benchmarker.has(mut_label)
                        else "—"
                    )
                    time_col = f"  {t_str:>10}"
                if has_hazard_report:
                    det = detection_label(m, rpt.baseline_hazard_count)
                    det_str = f"{det} ({m.hazard_count})"
                    print(
                        f"    {m.wait_index:>3}  {m.wait_line:>5}  "
                        f"{m.wait_instruction:<28}  {sl:<14}  {det_str:>15}{time_col}  {notes}"
                    )
                else:
                    print(
                        f"    {m.wait_index:>3}  {m.wait_line:>5}  "
                        f"{m.wait_instruction:<28}  {sl:<14}{time_col}  {notes}"
                    )

            errored = [m for m in rpt.mutants if m.error]
            if errored:
                print()
                for m in errored:
                    print(
                        f"    Mutant {m.wait_index} ({m.wait_instruction} "
                        f"L{m.wait_line}) stderr:"
                    )
                    for line in m.error.splitlines():
                        print(f"      {line}")

        summary = compute_summary(reports)
        total = summary["total_mutants"]
        killed = summary["killed"]
        survived = summary["survived"]
        inconclusive = summary["inconclusive"]
        scored = summary["scored_mutants"]
        print(f"\n{'=' * 80}")
        line = f"SUMMARY: {total} mutants | {killed} killed | {survived} survived"
        if inconclusive:
            line += f" | {inconclusive} inconclusive"
        print(line)
        if scored > 0:
            print(f"Mutation score: {killed / scored * 100:.1f}% of {scored} that ran")
        if has_hazard_report:
            fp = summary["false_positives"]
            fn = summary["false_negatives"]
            print(f"False positives (baseline hazards): {fp}")
            print(f"False negatives (missed hazards):   {fn}")
        print("=" * 80)
        if total and not scored:
            print(
                "\nNo mutant executed, so this run measured nothing. Every process "
                "failed to build, start or finish; check the stderr above. A ROCm "
                "runtime the emulator cannot serve is the usual cause — pass "
                "--rocm-path to select a working one."
            )

        if benchmark_active:
            self._print_benchmark_summary(reports, benchmarker)

    def _print_benchmark_summary(
        self,
        reports: List[ShaderReport],
        benchmarker: "Benchmarker",
    ) -> None:
        """Print a compact timing summary table to stdout."""
        col_w = 44
        print(f"\n{'=' * 60}")
        print("BENCHMARK TIMING SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Kernel':<{col_w}}  {'Time':>10}")
        print("  " + "-" * (col_w + 13))
        total = 0.0
        for rpt in reports:
            if benchmarker.has(rpt.shader):
                t = benchmarker.elapsed(rpt.shader)
                total += t
                print(f"  {rpt.shader:<{col_w}}  {t:>9.3f}s")
            for m in rpt.mutants:
                mut_label = f"{rpt.shader}_mut{m.wait_index}"
                if benchmarker.has(mut_label):
                    t = benchmarker.elapsed(mut_label)
                    total += t
                    print(f"  {mut_label:<{col_w}}  {t:>9.3f}s")
        print("  " + "-" * (col_w + 13))
        print(f"  {'TOTAL':<{col_w}}  {total:>9.3f}s")
        print("=" * 60)


def write_terminal_report(
    reports: List[ShaderReport],
    benchmarker: Optional["Benchmarker"] = None,
) -> None:
    TerminalReportWriter().write(reports, benchmarker=benchmarker)


def write_benchmark_report(benchmarker: "Benchmarker", path: Path) -> None:
    """Write a JSON benchmark timing report to *path*."""
    timings = benchmarker.timings
    entries = [
        {"kernel": label, "elapsed_seconds": seconds}
        for label, seconds in timings.items()
    ]
    report = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": sum(e["elapsed_seconds"] for e in entries),
        "kernels": entries,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark report written to {path}")


def write_json_report(reports: List[ShaderReport], path: Path, arch: str = "") -> None:
    JSONReportWriter().write(reports, path, arch=arch)


def write_csv_report(reports: List[ShaderReport], path: Path) -> None:
    CSVReportWriter().write(reports, path, arch="")


def write_markdown_report(
    reports: List[ShaderReport], path: Path, arch: str = ""
) -> None:
    MarkdownReportWriter().write(reports, path, arch=arch)
