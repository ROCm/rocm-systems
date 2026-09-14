#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# perf_regression.py -- run the rocDecode performance test and compare measured
# decode FPS against the baseline to catch regressions.
#
# Baseline: an HTML report (rocDecode_perf_baseline.html) with one column per GPU
# (MI250X, MI300X, MI300A, MI350, MI355, Navi31, Navi48) and one row per perf
# stream. The local GPU is detected and mapped to the matching column. Measured
# FPS comes from running the videoDecodePerf sample directly on each stream.
# Streams that appear to regress on a single run are re-measured (3-run average)
# to rule out noise before being reported as regressions.
#
# Inputs are located via ROCDECODE_PERF_DIR (default: $HOME/rocDecodePerformance),
# which must contain the per-codec stream subdirectories AvcPerformance,
# HevcPerformance, Av1Performance, Vp9Performance and the baseline HTML file.

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# --- paths ---
SCRIPT_DIR = Path(__file__).resolve().parent          # .../test
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../rocdecode
PERF_EXE = PROJECT_ROOT / "samples" / "videoDecodePerf" / "build" / "videodecodeperf"

# (display name, stream subdirectory) in report order
CODECS = [
    ("AVC", "AvcPerformance"),
    ("HEVC", "HevcPerformance"),
    ("AV1", "Av1Performance"),
    ("VP9", "Vp9Performance"),
]

# ANSI colors (match validate.sh)
GREEN, RED, YELLOW, NC = "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[0m"

# GPU market-name patterns -> baseline column label. Order matters (MI355 before
# MI350 so "MI355X" is not swallowed by the "MI350" rule).
MARKET_MAP = [
    (re.compile(r"MI250", re.I), "MI250X"),
    (re.compile(r"MI300X", re.I), "MI300X"),
    (re.compile(r"MI300A", re.I), "MI300A"),
    (re.compile(r"MI355", re.I), "MI355"),
    (re.compile(r"MI350", re.I), "MI350"),
    (re.compile(r"7900\s*XTX|Navi\s*31", re.I), "Navi31"),
    (re.compile(r"9070|Navi\s*48", re.I), "Navi48"),
]
# gfx targets that map unambiguously to a single column (MI300X/MI300A share
# gfx942 and MI350/MI355 share gfx950, so those are intentionally omitted).
GFX_MAP = {"gfx90a": "MI250X", "gfx1100": "Navi31", "gfx1201": "Navi48"}


def find_tool(name):
    """Locate a ROCm tool on PATH, then under $ROCM_PATH/bin."""
    p = shutil.which(name)
    if p:
        return p
    candidate = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / name
    return str(candidate) if candidate.is_file() else None


def check_rocm():
    """Verify ROCM_PATH points at a usable ROCm toolchain (the sample build derives
    its compiler from it). Return (ok, message)."""
    rp = os.environ.get("ROCM_PATH", "/opt/rocm")
    compiler = Path(rp) / "lib" / "llvm" / "bin" / "amdclang++"
    if compiler.is_file() and os.access(compiler, os.X_OK):
        return True, f"Using ROCM_PATH={rp} (amdclang++ found)"
    note = f"ROCM_PATH={rp}" if os.environ.get("ROCM_PATH") else \
        "ROCM_PATH is unset; defaulting to /opt/rocm"
    msg = (f"ERROR: ROCm toolchain not found at: {compiler}\n"
           f"       ({note})\n"
           "  Fix: point ROCM_PATH at your ROCm install (the dir containing lib/llvm/bin).\n"
           "       Skills run cmake in a NON-interactive shell that does not source ~/.bashrc,\n"
           "       so set ROCM_PATH in ~/.profile (or export it before launching Claude Code).")
    return False, msg


def _run(cmd, **kw):
    """Run a command, capturing output. Surface non-zero exits (with stderr) so
    silent tool/decoder failures are visible; callers that treat failure as fatal
    can still inspect the returned CompletedProcess.returncode."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        print(f"    WARNING: command failed (exit {result.returncode}): {cmd[0]}",
              file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    return result


def detect_gpu():
    """Return (column_label, detail) or (None, reason)."""
    override = os.environ.get("ROCDECODE_PERF_GPU")
    if override:
        return override, f"ROCDECODE_PERF_GPU override = {override}"

    # 1. amd-smi market name (disambiguates same-gfx parts)
    amd_smi = find_tool("amd-smi")
    if amd_smi:
        out = _run([amd_smi, "static"]).stdout
        m = re.search(r"MARKET_NAME:\s*(.+)", out)
        if m:
            market = m.group(1).strip()
            for pat, label in MARKET_MAP:
                if pat.search(market):
                    return label, f"amd-smi MARKET_NAME='{market}'"

    # 2. rocm-smi product name
    rocm_smi = find_tool("rocm-smi")
    if rocm_smi:
        out = _run([rocm_smi, "--showproductname"]).stdout
        m = re.search(r"Card Series:\s*(.+)", out)
        if m:
            series = m.group(1).strip()
            for pat, label in MARKET_MAP:
                if pat.search(series):
                    return label, f"rocm-smi Card Series='{series}'"
        m = re.search(r"GFX Version:\s*(gfx[0-9a-f]+)", out)
        if m and m.group(1) in GFX_MAP:
            return GFX_MAP[m.group(1)], f"rocm-smi GFX Version={m.group(1)}"

    # 3. KFD topology gfx_target_version (unambiguous targets only)
    for props in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        try:
            txt = props.read_text()
        except OSError:
            continue
        m = re.search(r"^gfx_target_version\s+(\d+)", txt, re.M)
        if not m:
            continue
        v = int(m.group(1))
        if v == 0:
            continue
        major, minor, step = v // 10000, (v // 100) % 100, v % 100
        gfx = f"gfx{major}{minor:x}{step:x}" if (minor > 9 or step > 9) else f"gfx{major}{minor}{step}"
        if gfx in GFX_MAP:
            return GFX_MAP[gfx], f"KFD gfx_target_version={v} ({gfx})"

    return None, "no GPU market name / gfx target could be resolved"


def load_baseline(baseline_path, label):
    """Return {basename: baseline_fps} for the GPU column matching `label`."""
    tables = pd.read_html(str(baseline_path))
    frames = [t for t in tables if "File" in t.columns]
    if not frames:
        raise ValueError("no tables with a 'File' column found in baseline")
    df = pd.concat(frames, ignore_index=True)

    col = next((c for c in df.columns if str(c).split()[0] == label), None)
    if col is None:
        gpu_cols = [str(c) for c in df.columns if c not in
                    ("File", "Resolution", "Bit Depth", "FPS", "Bit Rate (Mb/s)")]
        raise KeyError(f"GPU '{label}' not found. Available columns: {gpu_cols}")

    baseline = {}
    for _, row in df.iterrows():
        fname = str(row["File"]).strip()
        fps = pd.to_numeric(row[col], errors="coerce")
        if fname and pd.notna(fps):
            baseline[fname] = float(fps)
    return baseline, str(col)


def measure_stream(stream_path, runs, device):
    """Run videodecodeperf directly N times on one stream; return averaged FPS or None."""
    vals = []
    for _ in range(runs):
        out = _run([str(PERF_EXE), "-i", str(stream_path), "-t", "1", "-d", str(device)])
        m = re.search(r"avg decode FPS:\s*([\d.]+)", out.stdout + out.stderr)
        if m:
            vals.append(float(m.group(1)))
    return sum(vals) / len(vals) if vals else None


def build_stream_index(codec_dir):
    """Map basename -> full path for every stream file under a codec dir."""
    exts = {".mp4", ".mov", ".mkv", ".webm", ".img", ".ivf", ".av1", ".265", ".h265", ".hevc"}
    idx = {}
    for p in codec_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            idx[p.name] = p
    return idx


def summary_box(rows, overall_ok):
    labels = [r[0] for r in rows] + ["Overall"]
    width = max(len(x) for x in labels + ["rocDecode Performance Regression"])
    inner = width + 12
    line = lambda l, f, r: l + f * inner + r
    status = {"PASSED": (GREEN, "✓ PASSED"), "REGRESSED": (RED, "✗ REGRESSED"),
              "WARNING": (YELLOW, "⚠ WARNING"), "SKIPPED": (YELLOW, "- SKIPPED")}
    print()
    print(line("╔", "═", "╗"))
    print("║  " + "rocDecode Performance Regression".ljust(inner - 2) + "║")
    print(line("╠", "═", "╣"))
    for name, st in rows:
        color, lbl = status.get(st, (NC, st))
        pad = inner - len(lbl) - 3
        print(f"║  {name.ljust(pad)} {color}{lbl}{NC}  ║")
    print(line("╠", "═", "╣"))
    color, lbl = (GREEN, "✓ PASSED") if overall_ok else (RED, "✗ REGRESSED")
    pad = inner - len(lbl) - 3
    print(f"║  {'Overall'.ljust(pad)} {color}{lbl}{NC}  ║")
    print(line("╚", "═", "╝"))


def main():
    ap = argparse.ArgumentParser(description="rocDecode performance regression check")
    ap.add_argument("--perf-dir", default=os.environ.get(
        "ROCDECODE_PERF_DIR", str(Path.home() / "rocDecodePerformance")),
        help="dir with per-codec stream subdirs and the baseline HTML")
    ap.add_argument("--baseline", default=None,
        help="baseline HTML path (default: <perf-dir>/rocDecode_perf_baseline.html)")
    ap.add_argument("--tolerance", type=float,
        default=float(os.environ.get("ROCDECODE_PERF_TOLERANCE", "5")),
        help="regression threshold: %% Avg FPS drop below baseline (default: 5)")
    ap.add_argument("--runs", type=int, default=3,
        help="runs to average when confirming a flagged stream (default: 3)")
    ap.add_argument("--device", type=int, default=0, help="GPU device id (default: 0)")
    ap.add_argument("--check-rocm", action="store_true",
        help="only verify the ROCm toolchain (ROCM_PATH) and exit")
    ap.add_argument("--quick", action="store_true",
        help="fast check: one baseline stream per leaf subfolder, capped at <=4K")
    args = ap.parse_args()

    # Verify the ROCm toolchain up front (and exit early if only checking).
    ok, msg = check_rocm()
    if not ok:
        print(msg, file=sys.stderr)
        return 1
    print(msg)
    if args.check_rocm:
        return 0

    perf_dir = Path(os.path.expanduser(args.perf_dir))
    baseline_path = Path(args.baseline) if args.baseline else perf_dir / "rocDecode_perf_baseline.html"
    tol = args.tolerance

    if not PERF_EXE.is_file():
        print(f"ERROR: videodecodeperf not built: {PERF_EXE}", file=sys.stderr)
        print("       Build it first: cmake --build build (or test/build_samples.sh)", file=sys.stderr)
        return 1
    if not baseline_path.is_file():
        print(f"ERROR: baseline not found: {baseline_path}", file=sys.stderr)
        print("       Download it from SharePoint and set ROCDECODE_PERF_DIR to its parent.", file=sys.stderr)
        return 1

    label, detail = detect_gpu()
    if label is None:
        print(f"ERROR: could not identify the GPU ({detail}).", file=sys.stderr)
        print("       Set ROCDECODE_PERF_GPU to a baseline column (e.g. Navi31).", file=sys.stderr)
        return 1
    try:
        baseline, col = load_baseline(baseline_path, label)
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"GPU: {label}  ({detail})")
    print(f"Baseline column: {col}  ({len(baseline)} streams)  tolerance: {tol}%")
    if args.quick:
        print("Quick mode: one stream per leaf subfolder, capped at <=4K")
    print()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Write results outside the repo tree (default: $HOME) so they do not clutter
    # `git status`. Override the base dir with ROCDECODE_PERF_RESULTS_DIR.
    results_base = Path(os.path.expanduser(os.environ.get(
        "ROCDECODE_PERF_RESULTS_DIR", str(Path.home() / "rocDecode_perf_results"))))
    out_dir = results_base / ts  # created only if there are results to write
    detail_rows = []
    summary_rows = []
    overall_ok = True
    t0 = time.time()

    for disp, subdir in CODECS:
        codec_dir = perf_dir / subdir
        if not codec_dir.is_dir():
            print(f"=== {disp}: stream dir not found ({codec_dir}) -- skipping ===")
            summary_rows.append((disp, "WARNING"))
            continue
        print(f"=== Measuring {disp} ({codec_dir}) ===")
        index = build_stream_index(codec_dir)
        # Baseline streams present in this codec dir, sorted for deterministic order.
        candidates = sorted((n, p) for n, p in index.items() if n in baseline)
        if args.quick:
            # one stream per leaf subfolder, capped at <=4K -- the 8K streams
            # dominate runtime, so skip them for a fast sanity check
            seen_dirs = set()
            picked = []
            for name, path in candidates:
                res = re.search(r"(\d{3,5})x(\d{3,5})", name)
                if res and int(res.group(1)) > 4096:
                    continue  # skip 8K
                if path.parent not in seen_dirs:
                    seen_dirs.add(path.parent)
                    picked.append((name, path))
            candidates = picked
        # Measure each selected stream once (flagged streams are re-measured below
        # to rule out run-to-run noise). Measuring videoDecodePerf directly avoids
        # writing any results into the repo tree.
        measured = {}
        for name, path in candidates:
            fps = measure_stream(path, 1, args.device)
            if fps is not None:
                measured[name] = fps
        if not measured:
            print(f"    no results produced for {disp}")
            summary_rows.append((disp, "WARNING"))
            continue

        regressions = 0
        compared = 0
        for name, m_fps in measured.items():
            compared += 1
            b_fps = baseline[name]
            delta = (m_fps - b_fps) / b_fps * 100.0
            verdict = "OK"
            note = ""
            if delta < -tol:
                # candidate regression -- confirm with an averaged re-run
                confirmed = m_fps
                if name in index:
                    avg = measure_stream(index[name], args.runs, args.device)
                    if avg is not None:
                        confirmed = avg
                        delta = (avg - b_fps) / b_fps * 100.0
                        note = f"re-ran {args.runs}x"
                m_fps = confirmed
                if delta < -tol:
                    verdict = "REGRESSED"
                    regressions += 1
                else:
                    verdict = "OK*"  # recovered on re-run
            detail_rows.append({
                "Codec": disp, "File": name, "Baseline FPS": round(b_fps, 2),
                "Measured FPS": round(m_fps, 2), "Delta %": round(delta, 2),
                "Verdict": verdict, "Note": note,
            })
            tag = {"OK": GREEN, "OK*": YELLOW, "REGRESSED": RED}[verdict]
            print(f"    {tag}{verdict:9}{NC} {name}  "
                  f"{m_fps:8.1f} vs {b_fps:8.1f} fps  ({delta:+.1f}%){(' '+note) if note else ''}")

        if compared == 0:
            summary_rows.append((f"{disp} (0 matched)", "WARNING"))
        elif regressions:
            summary_rows.append((f"{disp} ({regressions}/{compared} regressed)", "REGRESSED"))
            overall_ok = False
        else:
            summary_rows.append((f"{disp} ({compared} streams)", "PASSED"))

    if detail_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "perf_comparison.csv"
        pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
        summary_box(summary_rows, overall_ok)
        print(f"Results saved to: {out_dir}/")
    else:
        summary_box(summary_rows, overall_ok)
        print("No streams were compared (check ROCDECODE_PERF_DIR and the baseline).")
    elapsed = int(time.time() - t0)
    print(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")
    print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
