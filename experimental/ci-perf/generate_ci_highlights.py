#!/usr/bin/env python3
"""
Generate CI Highlights PDF Report for rocm-systems.

Usage:
    python generate_ci_highlights.py [--repo OWNER/REPO] [--months N] [--output FILE]

Requires:
    - fpdf2: pip install fpdf2
    - requests: pip install requests
    - GH_TOKEN env var or `gh auth login`
"""

import argparse
import logging
from datetime import datetime

from fpdf import FPDF

from ci_analyzer import (
    CIAnalyzer,
    ReportData,
    _fmt_duration,
    _fmt_rate,
)
from ci_data_fetcher import CIDataFetcher


class CIReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "ROCm Systems CI - Executive Summary", align="L")
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(20, 60, 120)
            self.ln(5)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(20, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(40, 80, 140)
            self.ln(3)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(10)
        self.multi_cell(190, 5.5, f"{'':>{indent}}- {text}")

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 70, 130)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(30, 70, 130)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8.5)
                self.set_text_color(30, 30, 30)
                fill = False
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align=align)
            self.ln()
            fill = not fill
        self.ln(3)

    def key_finding(self, text):
        self.set_fill_color(255, 248, 230)
        self.set_draw_color(200, 170, 80)
        self.set_font("Helvetica", "I", 9.5)
        self.set_text_color(100, 70, 10)
        y = self.get_y()
        self.rect(12, y, 186, 8, style="DF")
        self.set_xy(15, y + 1.5)
        self.multi_cell(180, 5, text)
        self.ln(4)

    def stat_box(self, label, value, color_rgb, x, y, w=42, h=22):
        self.set_fill_color(*color_rgb)
        self.set_draw_color(max(0, color_rgb[0] - 30), max(0, color_rgb[1] - 30), max(0, color_rgb[2] - 30))
        self.rect(x, y, w, h, style="DF")
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(60, 60, 60)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 5, label, align="C")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 20, 20)
        self.set_xy(x + 2, y + 9)
        self.cell(w - 4, 10, value, align="C")

    def alert_box(self, title, text):
        self.set_fill_color(255, 235, 235)
        self.set_draw_color(200, 80, 80)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(150, 30, 30)
        y = self.get_y()
        lines = len(text) // 85 + text.count("\n") + 3
        box_h = max(16, lines * 5 + 8)
        self.rect(12, y, 186, box_h, style="DF")
        self.set_xy(15, y + 2)
        self.cell(180, 5, title)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 20, 20)
        self.set_xy(15, y + 8)
        self.multi_cell(178, 4.5, text)
        self.set_y(y + box_h + 3)

    def success_box(self, title, text):
        self.set_fill_color(230, 250, 235)
        self.set_draw_color(60, 160, 80)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 100, 30)
        y = self.get_y()
        lines = len(text) // 85 + text.count("\n") + 3
        box_h = max(16, lines * 5 + 8)
        self.rect(12, y, 186, box_h, style="DF")
        self.set_xy(15, y + 2)
        self.cell(180, 5, title)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(20, 70, 20)
        self.set_xy(15, y + 8)
        self.multi_cell(178, 4.5, text)
        self.set_y(y + box_h + 3)

    def code_block(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(245, 245, 248)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        for line in text.split("\n"):
            self.cell(190, 4.5, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(data: ReportData) -> CIReport:
    pdf = CIReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.ln(25)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "ROCm Systems", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "CI Analysis - Executive Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Repository: ROCm/rocm-systems  |  Branch: develop", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Report Date: {data.generated_date}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Data Period: {data.data_period}", align="C", new_x="LMARGIN", new_y="NEXT")

    # At-a-glance stats
    pdf.ln(12)
    y = pdf.get_y()
    pdf.stat_box("Workflow Files", f"{data.workflow_count}+", (220, 235, 255), 14, y)
    pdf.stat_box("Projects", f"{data.project_count}+", (220, 235, 255), 60, y)

    pr_build_str = _fmt_duration(data.avg_pr_build_m) if data.avg_pr_build_m else "N/A"
    pr_color = (255, 248, 230) if data.avg_pr_build_m else (230, 230, 230)
    pdf.stat_box("PR Build (avg)", pr_build_str, pr_color, 106, y)

    nightly_str = _fmt_rate(data.nightly_pass_rate) if data.nightly_pass_rate is not None else "N/A"
    nightly_color = (255, 235, 235) if (data.nightly_pass_rate or 1.0) < 0.8 else (230, 250, 235)
    pdf.stat_box("Nightly Pass Rate", nightly_str, nightly_color, 152, y)
    pdf.set_y(y + 30)

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.body_text(
        "Executive summary of the CI architecture, performance, and trends for the rocm-systems monorepo. "
        "Covers build/test timing, bottleneck analysis, historical trends, and actionable recommendations."
    )

    # =========================================================================
    # 1. CI ARCHITECTURE + WAIT TIMES
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("1. How CI Works")
    pdf.body_text(
        "rocm-systems is a git subtree monorepo with 28+ ROCm projects. CI uses 76+ GitHub Actions "
        "workflows organized into four systems: TheRock CI (primary build/test), Azure Pipelines "
        "(external CI with downstream dependency triggers), per-project standalone pipelines, and "
        "internal ROCm CI. Smart path-based routing ensures only affected projects are built on PRs."
    )

    pdf.chapter_title("CI Routing Logic", level=2)
    pdf.code_block(
        "PR opened/updated\n"
        "  --> pr_detect_changed_subtrees.py (which files changed?)\n"
        "  --> therock_configure_ci.py (map to project groups + cmake flags)\n"
        "  --> Build only affected groups | Skip if docs-only\n"
        "\n"
        "Project Groups:  core | runtimes | profiler | debug_tools | dc_tools | rocshmem\n"
        "Events:          PR (targeted) | push (all) | nightly (all + math) | dispatch (custom)"
    )

    # Dynamic wait times table
    if data.wait_times:
        pdf.chapter_title("Developer Wait Times by Change Area", level=2)
        wait_rows = [
            [w.change_area, _fmt_duration(w.avg_wait_m), w.bottleneck]
            for w in data.wait_times
        ]
        pdf.table(
            ["What Changed", "CI Wait", "Bottleneck"],
            wait_rows,
            col_widths=[55, 30, 95],
        )
        # Find the longest wait time for the finding
        longest = data.wait_times[0] if data.wait_times else None
        if longest and longest.avg_wait_m:
            pdf.key_finding(
                f"Longest average CI wait: {_fmt_duration(longest.avg_wait_m)} "
                f"for {longest.change_area} changes ({longest.bottleneck})."
            )

    # =========================================================================
    # 2. BOTTLENECKS + NIGHTLY HEALTH
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2. Top Bottlenecks")

    # Build bottleneck table from wait times (top 5 by duration)
    if data.wait_times:
        bottleneck_rows = []
        for i, w in enumerate(data.wait_times[:5], 1):
            severity = "HIGH" if (w.avg_wait_m or 0) > 60 else "MEDIUM" if (w.avg_wait_m or 0) > 20 else "LOW"
            bottleneck_rows.append([
                str(i), w.change_area, _fmt_duration(w.avg_wait_m), severity
            ])
        pdf.table(
            ["#", "Pipeline", "Avg Duration", "Severity"],
            bottleneck_rows,
            col_widths=[10, 68, 52, 25],
        )

    pdf.chapter_title("Where Time Goes (Typical PR)", level=2)
    pdf.code_block(
        "|||||||||||||||||||||||||||||||||||||||||||||||..............  TheRock Build   (~65-75%)\n"
        "..............................................|||||||||||||.  GPU Tests       (~15-20%)\n"
        "..............................................||||||||||||||  Setup / Upload  (~5-10%)"
    )

    # Dynamic nightly health table
    nh = data.nightly
    if nh.total_runs > 0:
        pdf.chapter_title("Nightly Run Health", level=2)
        nightly_rows = [
            ["Success rate", f"{_fmt_rate(nh.success_rate)} ({nh.successful_runs}/{nh.total_runs})"],
            ["Average duration", _fmt_duration(nh.avg_duration_m)],
            ["Duration range", f"{_fmt_duration(nh.min_duration_m)} - {_fmt_duration(nh.max_duration_m)}"],
            ["Fastest run", f"{_fmt_duration(nh.min_duration_m)} ({nh.fastest_run_date})"],
            ["Slowest run", f"{_fmt_duration(nh.worst_duration_m)} ({nh.worst_run_date})"],
        ]
        pdf.table(
            ["Metric", "Value"],
            nightly_rows,
            col_widths=[75, 105],
        )

    # =========================================================================
    # 3. TRENDS
    # =========================================================================
    pdf.add_page()
    month_headers = data.month_labels if data.month_labels else ["Month 1", "Month 2", "Month 3"]
    pdf.chapter_title(f"3. Pipeline Trends ({data.data_period})")

    if data.trends:
        trend_rows = []
        for t in data.trends:
            row = [t.workflow_label]
            for m in t.months:
                if m.total_runs == 0:
                    row.append("No runs")
                elif m.avg_duration_m is not None:
                    rate_str = f" ({_fmt_rate(m.success_rate)})" if m.success_rate is not None else ""
                    row.append(f"{_fmt_duration(m.avg_duration_m)}{rate_str}")
                else:
                    row.append(f"{m.total_runs} runs (all cancelled)")

            row.append(t.trend_description)
            trend_rows.append(row)

        headers = ["Pipeline"] + month_headers + ["Trend"]
        n_cols = len(headers)
        # Compute widths: pipeline=42, each month=28, trend=remaining
        month_count = len(month_headers)
        month_w = 28
        trend_w = 190 - 42 - (month_count * month_w)
        col_widths = [42] + [month_w] * month_count + [max(trend_w, 30)]

        pdf.table(headers, trend_rows, col_widths=col_widths)

    # =========================================================================
    # 4. IMPROVEMENTS
    # =========================================================================
    if data.improvements:
        pdf.add_page()
        pdf.chapter_title("4. What Improved")
        for title, desc in data.improvements:
            pdf.success_box(title, desc)
            pdf.ln(1)

    # =========================================================================
    # 5. CONCERNS
    # =========================================================================
    if data.concerns:
        pdf.add_page()
        pdf.chapter_title("5. What Needs Attention")
        for title, desc in data.concerns:
            pdf.alert_box(title, desc)
            pdf.ln(1)

    # =========================================================================
    # 6. RECOMMENDATIONS (static - these are strategic, not data-driven)
    # =========================================================================
    pdf.add_page()
    section_num = 4 + (1 if data.improvements else 0) + (1 if data.concerns else 0)
    pdf.chapter_title(f"{section_num}. Recommendations")

    recommendations = [
        ("Stabilize the nightly signal",
         "Identify the top flaky tests, quarantine or fix them, "
         "and target a 90%+ nightly pass rate to restore trust in the nightly signal."),
        ("Reduce GPU runner queue contention",
         "GPU queue waits often exceed actual test time. Options: stagger nightly "
         "schedules, add GPU runner capacity, or implement job priority for nightly runs."),
        ("Apply scoped builds to more subsystems",
         "The dedicated media pipeline demonstrates faster feedback by building only "
         "required components. Other isolated subsystems could benefit from similar carve-outs."),
        ("Investigate TheRock build time optimization",
         "The build dominates PR CI time. Potential avenues: more aggressive ccache warming, "
         "incremental builds, or splitting the monolithic build into parallelizable stages."),
        ("Expand nightly test coverage for uncovered components",
         "Some components (media libs, dc_tools, rocshmem) have no nightly test coverage. "
         "Adding them provides a safety net beyond their dedicated pipelines."),
        ("Monitor distro matrix growth",
         "Establish a policy for when to add vs. retire distro targets to prevent "
         "unbounded CI duration growth across all pipelines."),
        ("Eliminate redundant push-to-develop builds",
         "When PRs merge, both per-project and full TheRock CI trigger. The per-project builds "
         "are redundant on push. Consider skipping them, keeping only their unique tests."),
    ]

    for i, (title, desc) in enumerate(recommendations, 1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, f"{i}. {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.body_text(desc)

    return pdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CI Executive Summary PDF for rocm-systems"
    )
    parser.add_argument(
        "--repo",
        default="ROCm/rocm-systems",
        help="GitHub repository (default: ROCm/rocm-systems)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Number of months of history to analyze (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="rocm-systems-ci-highlights.pdf",
        help="Output PDF path (default: rocm-systems-ci-highlights.pdf)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Fetching CI data from {args.repo} ({args.months} months)...")
    fetcher = CIDataFetcher(repo=args.repo)
    raw_data = fetcher.fetch_all(months=args.months)

    print("Analyzing data...")
    analyzer = CIAnalyzer(raw_data, fetcher)
    report_data = analyzer.build_report_data()

    print("Generating PDF...")
    pdf = build_report(report_data)
    pdf.output(args.output)
    print(f"PDF generated: {args.output}")
