#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
RCCL Test Coverage Checker

Identifies gtest cases that exist in test binaries but are never matched by
any test_filter in the config, so they are invisible to the test runner.

Usage:
  python3 check_coverage.py -c <config.json> [options]

Options:
  -c, --config CONFIG       Test runner config JSON (required)
  --binary BINARY           Restrict analysis to this binary name (repeatable)
  --build-dir DIR           Build directory containing test binaries
                            (default: <workdir>/build/debug or build/release)
  --report OUTPUT.HTML      Write an HTML coverage gap report to this path
  --rocm-path PATH          ROCm installation prefix (default: /opt/rocm)
  -v, --verbose             Show matched cases as well as unmatched ones

The script exits non-zero when any unmatched cases are found.
"""

import argparse
import fnmatch
import html
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _expand_env_vars(value):
    """Simple ${VAR:-default} and $VAR expansion."""
    if not isinstance(value, str):
        return value

    def _repl(m):
        var, default = m.group(1), m.group(2) or ""
        return os.environ.get(var, default)

    value = re.sub(r'\$\{(\w+)(?::-(.*?))?\}', _repl, value)
    value = re.sub(r'\$(\w+)', lambda m: os.environ.get(m.group(1), m.group(0)), value)
    return value


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def resolve_workdir(config):
    paths = config.get("paths", {})
    workdir = _expand_env_vars(paths.get("workdir", "$PWD"))
    return os.path.abspath(workdir)


def find_build_dir(workdir, build_dir_override=None):
    if build_dir_override:
        return build_dir_override
    for subdir in ("build/debug", "build/release", "build"):
        candidate = os.path.join(workdir, subdir)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(workdir, "build")


def collect_binaries_from_config(config, build_dir):
    """Return {binary_name: full_path} for every unique binary in the config."""
    binaries = {}
    for _cfg_name, cfg in config.get("test_configurations", {}).items():
        binary = cfg.get("binary", "")
        if binary and binary not in binaries:
            full = os.path.join(build_dir, binary)
            binaries[binary] = full
        for test in cfg.get("tests", []):
            binary = test.get("binary", binary)
            if binary and binary not in binaries:
                full = os.path.join(build_dir, binary)
                binaries[binary] = full
    return binaries


# ---------------------------------------------------------------------------
# Gtest case enumeration
# ---------------------------------------------------------------------------

def list_gtest_cases(binary_path, rocm_path="/opt/rocm"):
    """Run binary --gtest_list_tests and return list of 'Suite.Case' strings."""
    env = os.environ.copy()
    lib_path = os.path.join(rocm_path, "lib")
    env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
    env["HSA_HOTSWAP_DISABLE"] = "0"

    try:
        result = subprocess.run(
            [binary_path, "--gtest_list_tests"],
            capture_output=True, text=True, timeout=30, env=env
        )
    except FileNotFoundError:
        return None, f"Binary not found: {binary_path}"
    except subprocess.TimeoutExpired:
        return None, f"--gtest_list_tests timed out for {binary_path}"
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        return None, f"exit {result.returncode}: {result.stderr.strip()[:200]}"

    cases = []
    current_suite = ""
    for line in result.stdout.splitlines():
        if line.endswith(".") or (line.rstrip().endswith(".") and not line.startswith(" ")):
            current_suite = line.strip().rstrip(".")
        elif line.startswith("  ") and current_suite:
            case_name = line.strip().split("  ")[0]  # strip trailing comments
            if case_name:
                cases.append(f"{current_suite}.{case_name}")
    return cases, None


# ---------------------------------------------------------------------------
# Filter matching
# ---------------------------------------------------------------------------

def _gtest_glob_match(name, pattern):
    """Match a single gtest glob pattern (supports * and ? but not : / -)."""
    return fnmatch.fnmatchcase(name, pattern)


def filter_matches(full_name, filter_str):
    """Return True if full_name is matched by a gtest-style filter string.

    Syntax: positive[:positive...][-negative[:negative...]]
      * = any substring, ? = any char, : = OR within a group.
    """
    if not filter_str or filter_str == "*":
        return True

    # Split on '-' for negative group, but only on the first '-' at token boundary.
    parts = filter_str.split("-", 1)
    positive_str = parts[0]
    negative_str = parts[1] if len(parts) > 1 else ""

    # Check positive patterns (any must match)
    if positive_str:
        positives = [p for p in positive_str.split(":") if p]
        if not any(_gtest_glob_match(full_name, p) for p in positives):
            return False

    # Check negative patterns (none must match)
    if negative_str:
        negatives = [n for n in negative_str.split(":") if n]
        if any(_gtest_glob_match(full_name, n) for n in negatives):
            return False

    return True


# ---------------------------------------------------------------------------
# Config filter collection
# ---------------------------------------------------------------------------

def collect_filters(config, restrict_binaries=None):
    """Return {binary_name: [filter_str, ...]} from all enabled test configs."""
    filters = defaultdict(list)
    for _cfg_name, cfg in config.get("test_configurations", {}).items():
        if cfg.get("enabled") is False:
            continue
        binary = cfg.get("binary", "")
        if not binary:
            continue
        if restrict_binaries and binary not in restrict_binaries:
            continue
        # Suite-level filter (if any)
        suite_filter = cfg.get("test_filter", "")
        for test in cfg.get("tests", []):
            test_binary = test.get("binary", binary)
            if restrict_binaries and test_binary not in restrict_binaries:
                continue
            f = test.get("test_filter", suite_filter)
            if f:
                filters[test_binary].append(f)
            # A test with no filter and no suite filter covers everything in that binary
            elif not suite_filter:
                filters[test_binary].append("*")
    return filters


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(config, build_dir, restrict_binaries=None, rocm_path="/opt/rocm", verbose=False):
    """Return per-binary coverage report dicts.

    Each dict has: binary, path, error, all_cases, matched, unmatched.
    """
    all_binaries = collect_binaries_from_config(config, build_dir)
    all_filters = collect_filters(config, restrict_binaries)

    results = []
    for binary_name, binary_path in sorted(all_binaries.items()):
        if restrict_binaries and binary_name not in restrict_binaries:
            continue

        cases, err = list_gtest_cases(binary_path, rocm_path)
        if err:
            results.append({
                "binary": binary_name, "path": binary_path,
                "error": err, "all_cases": [], "matched": [], "unmatched": [],
            })
            continue

        filter_list = all_filters.get(binary_name, [])
        matched, unmatched = [], []
        for case in (cases or []):
            if any(filter_matches(case, f) for f in filter_list):
                matched.append(case)
            else:
                unmatched.append(case)

        results.append({
            "binary": binary_name, "path": binary_path,
            "error": None,
            "all_cases": cases or [],
            "matched": matched,
            "unmatched": unmatched,
            "filters": filter_list,
        })
    return results


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_report(results, verbose=False):
    total_all = total_matched = total_unmatched = 0
    any_unmatched = False

    for r in results:
        binary = r["binary"]
        if r["error"]:
            print(f"\n[{binary}]  ERROR: {r['error']}")
            continue

        n_all = len(r["all_cases"])
        n_matched = len(r["matched"])
        n_unmatched = len(r["unmatched"])
        total_all += n_all
        total_matched += n_matched
        total_unmatched += n_unmatched

        pct = f"{n_matched/n_all*100:.1f}%" if n_all else "n/a"
        print(f"\n[{binary}]  {n_matched}/{n_all} covered ({pct})"
              f"  —  {n_unmatched} unmatched")

        if n_unmatched:
            any_unmatched = True
            for case in r["unmatched"]:
                print(f"  UNMATCHED  {case}")

        if verbose and r["matched"]:
            for case in r["matched"]:
                print(f"  matched    {case}")

    print(f"\n{'='*60}")
    pct_total = f"{total_matched/total_all*100:.1f}%" if total_all else "n/a"
    print(f"TOTAL: {total_matched}/{total_all} covered ({pct_total})"
          f"  —  {total_unmatched} unmatched cases")
    return any_unmatched


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_H = lambda s: html.escape(str(s) if s else "")

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #222; }
header { background: #1a1a2e; color: #eee; padding: 28px 40px; }
header h1 { font-size: 1.6rem; font-weight: 600; }
header p  { font-size: .85rem; color: #aaa; margin-top: 4px; }
.content  { max-width: 1100px; margin: 30px auto; padding: 0 20px 60px; }
.summary-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 32px; }
.stat { flex: 1; min-width: 120px; border-radius: 8px; padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; background: #fff; }
.stat .num { font-size: 2.2rem; font-weight: 700; }
.stat .lbl { font-size: .78rem; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }
.stat.warn { background: #f8d7da; color: #721c24; }
.stat.ok   { background: #d4edda; color: #155724; }
section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
          margin-bottom: 24px; overflow: hidden; }
section h2 { padding: 14px 24px; font-size: 1rem; border-bottom: 1px solid #eee;
             display: flex; align-items: center; gap: 10px; }
.sec-body { padding: 16px 24px; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th { background: #f8f9fa; text-align: left; padding: 8px 12px; font-weight: 600;
     border-bottom: 2px solid #dee2e6; }
td { padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }
tr:hover td { background: #fafafa; }
.badge { display: inline-block; border-radius: 4px; padding: 2px 8px;
         font-size: .75rem; font-weight: 600; color: #fff; }
.empty { color: #888; font-style: italic; }
footer { text-align: center; font-size: .78rem; color: #aaa; padding: 20px; }
"""


def write_html_report(results, config_path, out_path):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    total_all = sum(len(r["all_cases"]) for r in results if not r["error"])
    total_matched = sum(len(r["matched"]) for r in results if not r["error"])
    total_unmatched = sum(len(r["unmatched"]) for r in results if not r["error"])
    pct = f"{total_matched/total_all*100:.1f}%" if total_all else "n/a"

    stats_html = f"""
    <div class="stat {'ok' if total_unmatched == 0 else ''}">
      <div class="num">{total_all}</div><div class="lbl">Total Cases</div></div>
    <div class="stat ok">
      <div class="num">{total_matched}</div><div class="lbl">Covered</div></div>
    <div class="stat {'warn' if total_unmatched else 'ok'}">
      <div class="num">{total_unmatched}</div><div class="lbl">Unmatched</div></div>
    <div class="stat">
      <div class="num">{_H(pct)}</div><div class="lbl">Coverage</div></div>
    """

    sections_html = ""
    for r in results:
        binary = r["binary"]
        if r["error"]:
            sections_html += f"""
            <section>
              <h2>{_H(binary)} <span class="badge" style="background:#dc3545">ERROR</span></h2>
              <div class="sec-body"><p>{_H(r['error'])}</p></div>
            </section>"""
            continue

        n_all = len(r["all_cases"])
        n_un = len(r["unmatched"])
        color = "#dc3545" if n_un else "#198754"
        badge = f'<span class="badge" style="background:{color}">{n_un} unmatched / {n_all} total</span>'

        if n_un:
            rows = "".join(
                f"<tr><td>{_H(c)}</td></tr>" for c in r["unmatched"]
            )
            table = f"""<table><thead><tr><th>Unmatched gtest case</th></tr></thead>
                        <tbody>{rows}</tbody></table>"""
        else:
            table = '<p class="empty">All cases are covered by at least one test_filter.</p>'

        sections_html += f"""
        <section>
          <h2>{_H(binary)} {badge}</h2>
          <div class="sec-body">{table}</div>
        </section>"""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RCCL Test Coverage Gap Report</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>RCCL Test Coverage Gap Report</h1>
  <p>{_H(now)} &nbsp;|&nbsp; Config: {_H(os.path.basename(config_path))}</p>
</header>
<div class="content">
  <div class="summary-row">{stats_html}</div>
  {sections_html}
</div>
<footer>Generated by test_runner check_coverage.py &nbsp;|&nbsp; {_H(now)}</footer>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(body)
    print(f"Coverage gap report written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Find gtest cases in test binaries that are not covered by any config filter"
    )
    p.add_argument("-c", "--config", required=True, help="Test runner config JSON")
    p.add_argument("--binary", action="append", default=[], metavar="BINARY",
                   help="Restrict to this binary name (repeatable)")
    p.add_argument("--build-dir", default="", help="Build directory containing test binaries")
    p.add_argument("--report", default="", metavar="OUTPUT.HTML",
                   help="Write HTML report to this path")
    p.add_argument("--rocm-path", default="/opt/rocm", help="ROCm installation prefix")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show matched cases as well as unmatched ones")
    args = p.parse_args()

    config = load_config(args.config)
    workdir = resolve_workdir(config)
    build_dir = find_build_dir(workdir, args.build_dir or None)
    restrict = set(args.binary) if args.binary else None

    print(f"Config:    {args.config}")
    print(f"Build dir: {build_dir}")
    if restrict:
        print(f"Binaries:  {', '.join(sorted(restrict))}")
    print()

    results = analyze(config, build_dir, restrict_binaries=restrict,
                      rocm_path=args.rocm_path, verbose=args.verbose)

    if not results:
        print("No binaries found in config.")
        sys.exit(0)

    any_unmatched = print_report(results, verbose=args.verbose)

    if args.report:
        write_html_report(results, args.config, args.report)

    sys.exit(1 if any_unmatched else 0)


if __name__ == "__main__":
    main()
