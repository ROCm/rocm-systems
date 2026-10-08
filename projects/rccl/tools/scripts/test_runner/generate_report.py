#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
RCCL Functional Test HTML Report Generator
Usage: python3 generate_report.py <run_dir> <output.html> [--mode SPX|DPX] [--bkc <version>]
                                   [--rocm-label <str>] [--real-total N]

<run_dir> must contain run.json and tests.jsonl (a single run directory from
the test_runner's results/<run_id>/ output).

--real-total N     Actual gtest case count from --gtest_list_tests across all
                   binaries.  When provided, shows a "Not Executed" stat and
                   uses N as the suite total instead of cases_total from run.json.
--rocm-label STR   Override ROCm version label (e.g. the /opt/rocm symlink name).
"""

import argparse
import collections
import html
import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    cases = {"failed": [], "timeout": [], "skipped": [], "disabled": [],
              "skipped_by_reason": collections.defaultdict(list)}
    with open(os.path.join(run_dir, "tests.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            result = t.get("result", "").upper()
            suite_label = t.get("suite", t.get("test_name", ""))
            skip_reason = t.get("error", "no reason given")
            details = t.get("case_details", [])
            test_name = t.get("test_name", "")
            if details:
                for c in details:
                    st = c.get("status", "").upper()
                    full_name = c.get("full_name", f"{c.get('suite','')}.{c.get('case','')}")
                    msg = c.get("message", c.get("failure_message", ""))
                    reason = c.get("reason", "")
                    if st == "FAILED":
                        cases["failed"].append({"test": full_name, "suite": suite_label, "message": msg})
                    elif st in ("TIMEOUT", "TIMED_OUT"):
                        cases["timeout"].append({"test": full_name, "suite": suite_label})
                    elif st == "SKIPPED":
                        cases["skipped"].append({"test": full_name, "suite": suite_label,
                                                 "reason": reason or skip_reason})
                    elif st == "DISABLED":
                        cases["disabled"].append({"test": full_name, "suite": suite_label})
            else:
                # Suite-level records (no case_details): use top-level result
                exit_code = t.get("exit_code", "")
                msg = skip_reason if result == "SKIPPED" else (
                    f"exit_code={exit_code}" if exit_code not in (0, None, "") else "")
                if result == "FAILED":
                    cases["failed"].append({"test": test_name, "suite": suite_label, "message": msg})
                elif result in ("TIMEOUT", "TIMED_OUT"):
                    cases["timeout"].append({"test": test_name, "suite": suite_label})
                elif result == "SKIPPED":
                    cases["skipped_by_reason"][skip_reason].append(test_name)
                elif result == "DISABLED":
                    cases["disabled"].append({"test": test_name, "suite": suite_label})
    return run, cases


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _h(s):
    return html.escape(str(s) if s else "")


def _badge(text, color):
    return f'<span class="badge" style="background:{color}">{_h(text)}</span>'


def _table(headers, rows, empty_msg="None"):
    if not rows:
        return f'<p class="empty">{empty_msg}</p>'
    cols = "".join(f"<th>{_h(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{_h(c)}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{cols}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #222; }
header { background: #1a1a2e; color: #eee; padding: 28px 40px; }
header h1 { font-size: 1.6rem; font-weight: 600; letter-spacing: .5px; }
header p  { font-size: .85rem; color: #aaa; margin-top: 4px; }
.content  { max-width: 1100px; margin: 30px auto; padding: 0 20px 60px; }
.meta-table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 28px; font-size: .88rem; overflow: hidden; }
.meta-table td { padding: 8px 18px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
.meta-table tr:last-child td { border-bottom: none; }
.meta-table .meta-key { width: 160px; color: #666; font-weight: 600; text-transform: uppercase;
                        font-size: .72rem; letter-spacing: .4px; white-space: nowrap; }
.meta-table .meta-val { word-break: break-all; }
.summary-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 36px; }
.stat { flex: 1; min-width: 120px; border-radius: 8px; padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
.stat .num  { font-size: 2.2rem; font-weight: 700; }
.stat .lbl  { font-size: .78rem; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }
.stat.total   { background: #fff; }
.stat.passed  { background: #d4edda; color: #155724; }
.stat.failed  { background: #f8d7da; color: #721c24; }
.stat.timeout { background: #fff3cd; color: #856404; }
.stat.skipped { background: #e2e3e5; color: #383d41; }
.stat.disabled{ background: #cfe2ff; color: #084298; }
section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
          margin-bottom: 24px; overflow: hidden; }
section h2 { padding: 16px 24px; font-size: 1rem; border-bottom: 1px solid #eee;
             display: flex; align-items: center; gap: 10px; }
.sec-body { padding: 16px 24px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th { background: #f8f9fa; text-align: left; padding: 8px 12px; font-weight: 600;
     border-bottom: 2px solid #dee2e6; }
td { padding: 7px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
tr:hover td { background: #fafafa; }
.badge { display: inline-block; border-radius: 4px; padding: 2px 8px;
         font-size: .75rem; font-weight: 600; color: #fff; }
.empty { color: #888; font-style: italic; padding: 8px 0; }
.reason-group { margin-bottom: 20px; }
.reason-group h3 { font-size: .85rem; color: #555; margin-bottom: 8px;
                   padding: 6px 10px; background: #f8f9fa; border-radius: 4px; }
.reason-group ul { list-style: none; padding-left: 12px; }
.reason-group ul li { padding: 3px 0; font-size: .83rem; color: #333; border-bottom: 1px solid #f5f5f5; }
footer { text-align: center; font-size: .78rem; color: #aaa; padding: 20px; }
.legend-table { width: 100%; border-collapse: collapse; font-size: .78rem; color: #555;
                margin-top: -20px; margin-bottom: 36px; }
.legend-table td { padding: 3px 8px; }
.leg-key { white-space: nowrap; font-weight: 600; width: 120px; color: #444; }
.leg-val { color: #666; }
"""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate(run, cases, mode, bkc_override, out_path, rocm_label=None, real_total=None):
    meta = run.get("metadata", {})
    versions = meta.get("versions", {})
    summary = run.get("summary", {})

    node_names = run.get("node_names", [])
    hostname = ", ".join(node_names) if node_names else run.get("host", "unknown")
    rocm_ver = rocm_label or versions.get("rocm", run.get("rocm_version", "unknown"))
    kernel = versions.get("kernel", meta.get("kernel", "unknown"))
    bkc = bkc_override or "unknown"
    gpu_arch = run.get("gpu_arch", "unknown")
    num_gpus = run.get("gpus_per_node", "?")
    config_name = run.get("config_name", "") or "mi455_ainic_roce_func"
    os_name = meta.get("os", "unknown")

    ts = run.get("started_at", run.get("created_at", ""))
    try:
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        date_short = dt.strftime("%Y-%m-%d")
    except Exception:
        date_str = ts
        date_short = ts[:10]

    passed   = summary.get("cases_passed", summary.get("passed", 0))
    failed   = summary.get("cases_failed", summary.get("failed", 0))
    timeout  = summary.get("cases_timeout", summary.get("timeout", 0))
    skipped  = summary.get("cases_skipped", summary.get("skipped", 0))
    disabled = summary.get("cases_disabled", summary.get("disabled", 0))
    duration = summary.get("duration_s", 0)
    executed = passed + failed + timeout + skipped

    suite_total  = real_total or summary.get("cases_total", summary.get("total", executed))
    not_executed = suite_total - executed if suite_total > executed else None
    pass_rate         = (passed / suite_total * 100) if suite_total else 0
    pass_rate_exec    = (passed / executed * 100) if executed else 0

    # ---- META TABLE ----
    meta_rows = [
        ("Date", date_str),
        ("Host", hostname),
        ("Mode", mode),
        ("BKC", bkc),
        ("ROCm", rocm_ver),
        ("GPU Arch", gpu_arch),
        ("GPUs / Node", num_gpus),
        ("OS", os_name),
        ("Kernel", kernel),
        ("Config", config_name),
        ("Duration", f"{duration:.0f}s ({duration/60:.1f}m)"),
        ("Pass Rate", f"{pass_rate:.1f}% of total  |  {pass_rate_exec:.1f}% of executed"),
    ]

    meta_rows_html = "".join(
        f"<tr><td class='meta-key'>{_h(label)}</td><td class='meta-val'>{_h(str(value))}</td></tr>"
        for label, value in meta_rows
    )
    cards_html = f"""
    <table class="meta-table">
      <tbody>{meta_rows_html}</tbody>
    </table>"""

    # ---- SUMMARY STATS ----
    not_exec_stat = ""
    if not_executed is not None:
        not_exec_stat = f'<div class="stat" style="background:#fff3cd;color:#856404"><div class="num">{not_executed}</div><div class="lbl">Not Executed</div></div>'
    stats_html = f"""
    <div class="stat total">  <div class="num">{suite_total}</div>  <div class="lbl">Total in Suite</div></div>
    <div class="stat passed"> <div class="num">{passed}</div> <div class="lbl">Passed</div></div>
    <div class="stat failed"> <div class="num">{failed}</div> <div class="lbl">Failed</div></div>
    <div class="stat timeout"><div class="num">{timeout}</div><div class="lbl">Timeout</div></div>
    <div class="stat skipped"><div class="num">{skipped}</div><div class="lbl">Skipped</div></div>
    <div class="stat disabled"><div class="num">{disabled}</div><div class="lbl">Disabled</div></div>
    {not_exec_stat}
    """
    legend_items = [
        ("Total in Suite", "All gtest cases registered across all test binaries (via --gtest_list_tests)."),
        ("Passed", "Binary ran, gtest case executed, and reported success."),
        ("Failed", "Binary ran, gtest case executed, and reported failure or exited non-zero."),
        ("Timeout", "Test config was killed after exceeding the time limit; no case results were captured."),
        ("Skipped", "Binary ran, gtest case was selected, but called GTEST_SKIP() at runtime (e.g. missing feature or hardware)."),
        ("Disabled", "Gtest case name prefixed with DISABLED_; excluded from execution by the framework."),
        ("Not Executed", "Cases whose test config was never launched (e.g. requires a second node, or a prior config timed out)."),
    ]
    legend_rows = "".join(
        f"<tr><td class='leg-key'>{_h(k)}</td><td class='leg-val'>{_h(v)}</td></tr>"
        for k, v in legend_items
    )
    legend_html = f"""
    <table class="legend-table">
      <tbody>{legend_rows}</tbody>
    </table>"""

    # ---- FAILED TABLE ----
    failed_rows = [(c["test"], c["suite"], c.get("message", "") or "process exited non-zero")
                   for c in cases["failed"]]
    failed_html = _table(["Test Case", "Suite", "Failure Reason"], failed_rows,
                         "No failures — all executed tests passed.")

    # ---- TIMEOUT TABLE ----
    timeout_rows = [(c["test"], c["suite"]) for c in cases["timeout"]]
    timeout_html = _table(["Test Case", "Suite"], timeout_rows, "No timeouts.")

    # ---- SKIPPED TABLE ----
    skipped_html = ""
    if cases["skipped"]:
        skipped_html += "<h3 style='margin:12px 0 8px;font-size:.85rem;color:#555'>Individual gtest cases skipped within executed suites</h3>"
        skipped_rows = [(c["test"], c["suite"], c.get("reason", "")) for c in cases["skipped"]]
        skipped_html += _table(["Test Case", "Suite", "Reason"], skipped_rows)
    if cases["skipped_by_reason"]:
        skipped_html += "<h3 style='margin:20px 0 8px;font-size:.85rem;color:#555'>Test configs skipped before execution (grouped by reason)</h3>"
        for reason, tests in sorted(cases["skipped_by_reason"].items(), key=lambda x: -len(x[1])):
            items = "".join(f"<li>{_h(t)}</li>" for t in sorted(tests))
            skipped_html += f"""
            <div class="reason-group">
              <h3>{_h(reason)} <span style="color:#aaa;font-weight:normal">({len(tests)} tests)</span></h3>
              <ul>{items}</ul>
            </div>"""
    if not skipped_html:
        skipped_html = '<p class="empty">No skipped tests.</p>'

    # ---- DISABLED TABLE ----
    disabled_rows = [(c["test"], c["suite"]) for c in cases["disabled"]]
    disabled_html = _table(["Test Case", "Suite"], disabled_rows, "No disabled tests.")

    def section(title, badge_text, badge_color, body):
        b = _badge(badge_text, badge_color) if badge_text else ""
        return f"""
        <section>
          <h2>{_h(title)} {b}</h2>
          <div class="sec-body">{body}</div>
        </section>"""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RCCL Functional Test Report — {_h(date_short)} {_h(mode)}</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>RCCL Functional Test Report</h1>
  <p>{_h(date_str)} &nbsp;|&nbsp; {_h(hostname)} &nbsp;|&nbsp; Mode: {_h(mode)} &nbsp;|&nbsp; BKC {_h(bkc)} &nbsp;|&nbsp; {_h(rocm_ver)}</p>
</header>
<div class="content">
  {cards_html}
  <div class="summary-row">{stats_html}</div>
  {legend_html}
  {section("Failed Tests", f"{failed} cases", "#dc3545", failed_html)}
  {section("Timed-Out Tests", f"{timeout} cases", "#fd7e14", timeout_html)}
  {section("Skipped Tests", f"{skipped} gtest cases", "#6c757d", skipped_html)}
  {section("Disabled Tests", f"{disabled} cases", "#0d6efd", disabled_html)}
</div>
{f'<div class="content"><div style="background:#fff3cd;border-left:4px solid #ffc107;padding:12px 18px;border-radius:6px;font-size:.88rem;color:#856404;margin-bottom:24px"><strong>Note:</strong> {_h(summary["note"])}</div></div>' if summary.get("note") else ""}
<footer>Generated by test_runner generate_report.py &nbsp;|&nbsp; {_h(date_str)}</footer>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(body)
    print(f"Report written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Generate RCCL functional test HTML report")
    p.add_argument("run_dir", help="Path to results/<run_id>/ directory")
    p.add_argument("output", help="Output HTML file path")
    p.add_argument("--mode", default="SPX", choices=["SPX", "DPX"], help="Test mode (default: SPX)")
    p.add_argument("--bkc", default="", help="BKC version string")
    p.add_argument("--rocm-label", default="", help="ROCm label override (e.g. /opt/rocm symlink target name)")
    p.add_argument("--real-total", type=int, default=0,
                   help="True gtest case count from --gtest_list_tests across all binaries")
    args = p.parse_args(argv)
    run, cases = load_run(args.run_dir)
    generate(run, cases, args.mode, args.bkc, args.output,
             rocm_label=args.rocm_label or None,
             real_total=args.real_total or None)


if __name__ == "__main__":
    main()
