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
#
# The HTML baseline holds decode-only numbers (videoDecodePerf -m 3). Other output
# memory types (--mem-type 0/1/2) are compared against a local baseline CSV,
# <perf-dir>/local_baselines/<GPU>_m<N>.csv, recorded on this machine with
# --save-baseline.

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

# On Windows the default console encoding (cp1252) cannot handle the Unicode
# box-drawing characters used by summary_box(). Force UTF-8 so the summary
# renders correctly in Git Bash, Windows Terminal, and cmd.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- paths ---
SCRIPT_DIR = Path(__file__).resolve().parent          # .../test
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../rocdecode
IS_WINDOWS = sys.platform == "win32"
# On Windows the MSVC multi-config generator puts the exe under build/Release/;
# on Linux the single-config Makefile generator puts it directly under build/.
if IS_WINDOWS:
    PERF_EXE = PROJECT_ROOT / "samples" / "videoDecodePerf" / "build" / "Release" / "videodecodeperf.exe"
else:
    PERF_EXE = PROJECT_ROOT / "samples" / "videoDecodePerf" / "build" / "videodecodeperf"
_DEFAULT_ROCM = "C:/opt/rocm" if IS_WINDOWS else "/opt/rocm"

# (display name, stream subdirectory) in report order
CODECS = [
    ("AVC", "AvcPerformance"),
    ("HEVC", "HevcPerformance"),
    ("AV1", "Av1Performance"),
    ("VP9", "Vp9Performance"),
]

# videoDecodePerf -m values. The sample's default is 3 (decode only), which is
# what the HTML baseline measures.
MEM_TYPE_NAMES = {0: "device internal", 1: "device copied", 2: "host copied",
                  3: "not mapped / decode only"}
DEFAULT_MEM_TYPE = 3
LOCAL_BASELINE_DIR = "local_baselines"

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
    # Integrated GPUs — matched by hipInfo Name: field (e.g. "AMD Radeon(TM) 780M").
    # These share gfx1103, so they must be identified by market name, not gfx target.
    (re.compile(r"780M", re.I), "780M"),
    (re.compile(r"760M", re.I), "760M"),
    (re.compile(r"740M", re.I), "740M"),
]
# gfx targets that map unambiguously to a single baseline column (MI300X/MI300A
# share gfx942, MI350/MI355 share gfx950, and 780M/760M/740M share gfx1103,
# so those are intentionally omitted — use MARKET_MAP for disambiguation).
GFX_MAP = {"gfx90a": "MI250X", "gfx1100": "Navi31", "gfx1201": "Navi48"}


def find_tool(name):
    """Locate a ROCm tool on PATH, then under $ROCM_PATH/bin."""
    p = shutil.which(name)
    if p:
        return p
    candidate = Path(os.environ.get("ROCM_PATH", _DEFAULT_ROCM)) / "bin" / name
    return str(candidate) if candidate.is_file() else None


def check_rocm():
    """Verify ROCM_PATH points at a usable ROCm toolchain (the sample build derives
    its compiler from it). Return (ok, message)."""
    rp = os.environ.get("ROCM_PATH", _DEFAULT_ROCM)

    # On Windows the compiler is MSVC (no amdclang++); ROCM_PATH only selects the
    # install tree that rocDecode is resolved from, so a dir-exists check suffices.
    if IS_WINDOWS:
        if Path(rp).is_dir():
            return True, f"Using ROCM_PATH={rp} (Windows/MSVC)"
        note = f"ROCM_PATH={rp}" if os.environ.get("ROCM_PATH") else \
            "ROCM_PATH is unset; defaulting to C:/opt/rocm"
        msg = (f"ERROR: ROCM_PATH does not exist: {rp}\n"
               f"       ({note})\n"
               "  Fix: point ROCM_PATH at your rocDecode install prefix (default C:/opt/rocm)\n"
               "       and make sure its bin dirs are on PATH before launching Claude Code.")
        return False, msg

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


def _extract_hipinfo_device_block(output, device):
    """Extract the hipInfo output block for a specific device index.

    hipInfo prints one block per GPU, each starting with a '---' separator
    followed by 'device#  N'. Return only the text for the requested device.
    If no 'device#' line is present at all the format is unexpected, so return
    the full output (single-GPU fallback); if device blocks were parsed but
    none matches, return '' rather than another device's block."""
    blocks = re.split(r"^-{10,}$", output, flags=re.M)
    saw_device_block = False
    for block in blocks:
        m = re.search(r"^device#\s+(\d+)", block, re.M)
        if m:
            saw_device_block = True
            if int(m.group(1)) == device:
                return block
    # Parsed blocks but none for this device: better to fail detection than to
    # report another GPU's identity. Otherwise assume single-GPU/legacy output.
    return "" if saw_device_block else output


def detect_gpu(device=0, gpu_name=None):
    """Return (column_label, detail) or (None, reason)."""
    # --gpu flag takes priority, then ROCDECODE_PERF_GPU env var.
    if gpu_name:
        return gpu_name, f"--gpu {gpu_name}"
    override = os.environ.get("ROCDECODE_PERF_GPU")
    if override:
        return override, f"ROCDECODE_PERF_GPU override = {override}"

    # 1-2. amd-smi / rocm-smi (Linux only). On Windows the system amd-smi
    #       (C:\WINDOWS\system32\amd-smi) does not work ("Invalid platform")
    #       and may trigger a UAC elevation prompt (diskpart.exe), so skip it.
    if not IS_WINDOWS:
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

    # 3. hipInfo (available on both Linux and Windows). Restrict to the
    #    output block for the selected device so that --device N detects the
    #    right GPU on multi-GPU systems.
    hip_info = find_tool("hipInfo.exe" if IS_WINDOWS else "hipInfo")
    if hip_info:
        out = _extract_hipinfo_device_block(_run([hip_info]).stdout, device)
        # Try market-name matching on the Name: field first.
        m = re.search(r"^Name:\s*(.+)", out, re.M)
        if m:
            name = m.group(1).strip()
            for pat, label in MARKET_MAP:
                if pat.search(name):
                    return label, f"hipInfo Name='{name}' (device {device})"
        # Fall back to gfx target from gcnArchName.
        m = re.search(r"^gcnArchName:\s*(gfx[0-9a-f]+)", out, re.M)
        if m and m.group(1) in GFX_MAP:
            return GFX_MAP[m.group(1)], f"hipInfo gcnArchName={m.group(1)} (device {device})"

    # 4. KFD topology gfx_target_version (Linux only; unambiguous targets only)
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


def local_baseline_path(perf_dir, label, mem_type):
    """Default local baseline CSV for one GPU and output memory type."""
    return perf_dir / LOCAL_BASELINE_DIR / f"{label}_m{mem_type}.csv"


def load_local_baseline(csv_path, mem_type=None, label=None):
    """Return {basename: baseline_fps} from a baseline CSV written by --save-baseline.

    When the CSV has the Mem Type or GPU columns that --save-baseline writes, they
    must match the run's memory type and GPU: comparing, say, a decode-only run
    against a copied-mode baseline would hide regressions. A CSV with only File and
    FPS is accepted as is."""
    df = pd.read_csv(csv_path)
    if "File" not in df.columns or "FPS" not in df.columns:
        raise ValueError(f"baseline CSV {csv_path} needs 'File' and 'FPS' columns")
    if mem_type is not None and "Mem Type" in df.columns:
        recorded = set(pd.to_numeric(df["Mem Type"], errors="coerce").dropna().astype(int))
        if recorded and recorded != {mem_type}:
            raise ValueError(f"baseline CSV {csv_path} was recorded with memory type "
                             f"{sorted(recorded)}, but this run uses {mem_type}")
    if label is not None and "GPU" in df.columns:
        recorded = set(df["GPU"].dropna().astype(str).str.strip())
        if recorded and recorded != {label}:
            raise ValueError(f"baseline CSV {csv_path} was recorded on GPU "
                             f"{sorted(recorded)}, but this run is on {label}")
    baseline = {}
    for _, row in df.iterrows():
        fname = str(row["File"]).strip()
        # str(NaN) is "nan", so test the raw cell as well as the stripped name.
        if pd.isna(row["File"]) or not fname:
            raise ValueError(f"baseline CSV {csv_path} contains an empty File value")
        fps = pd.to_numeric(row["FPS"], errors="coerce")
        # FPS is a divisor in the comparison, so it must be finite and positive.
        if pd.isna(fps) or not (0 < float(fps) < float("inf")):
            raise ValueError(f"baseline CSV {csv_path} has invalid FPS for {fname!r}")
        if fname in baseline:
            # Streams are identified by file name alone, so a repeat (e.g. the same
            # name under two codecs) would make one row silently replace the other.
            raise ValueError(f"baseline CSV {csv_path} lists {fname!r} more than once")
        baseline[fname] = float(fps)
    if not baseline:
        # An empty baseline would compare no streams and report a pass.
        raise ValueError(f"baseline CSV {csv_path} contains no usable entries")
    return baseline


def measure_stream(stream_path, runs, device, mem_type=None):
    """Run videodecodeperf directly N times on one stream; return the averaged FPS,
    or None if any run fails. A failed run is not averaged away: a stream that
    decodes only some of the time is a failure, and an average of fewer runs than
    requested would misrepresent a saved baseline."""
    cmd = [str(PERF_EXE), "-i", str(stream_path), "-t", "1", "-d", str(device)]
    if mem_type is not None:
        cmd += ["-m", str(mem_type)]
    vals = []
    for _ in range(runs):
        out = _run(cmd)
        m = re.search(r"avg decode FPS:\s*([\d.]+)", out.stdout + out.stderr)
        if out.returncode != 0 or not m:
            return None
        vals.append(float(m.group(1)))
    return sum(vals) / len(vals)


STREAM_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".img", ".ivf", ".av1", ".265", ".h265", ".hevc"}


def list_streams(codec_dir):
    """Every stream file under a codec dir, in sorted order."""
    return sorted(p for p in codec_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in STREAM_EXTS)


def build_stream_index(codec_dir):
    """Map basename -> full path for every stream file under a codec dir."""
    return {p.name: p for p in list_streams(codec_dir)}


def find_duplicate_streams(perf_dir):
    """Return {basename: [paths]} for stream file names that occur more than once
    across all codec dirs. Baselines (HTML and CSV) identify a stream by its file
    name alone, so a repeated name would make two streams share one baseline."""
    paths = {}
    for _, subdir in CODECS:
        codec_dir = perf_dir / subdir
        if codec_dir.is_dir():
            for p in list_streams(codec_dir):
                paths.setdefault(p.name, []).append(p)
    return {name: ps for name, ps in paths.items() if len(ps) > 1}


SUMMARY_TITLES = {"compare": "rocDecode Performance Regression",
                  "measure": "rocDecode Performance (measure only)",
                  "save": "rocDecode Performance (saving baseline)"}


def summary_box(rows, overall_ok, mode="compare"):
    """mode: "compare" against a baseline, "measure" with no baseline, or "save" a baseline."""
    labels = [r[0] for r in rows] + ["Overall"]
    title = SUMMARY_TITLES[mode]
    width = max(len(x) for x in labels + [title])
    inner = width + 12
    line = lambda l, f, r: l + f * inner + r
    status = {"PASSED": (GREEN, "✓ PASSED"), "REGRESSED": (RED, "✗ REGRESSED"),
              "WARNING": (YELLOW, "⚠ WARNING"), "SKIPPED": (YELLOW, "- SKIPPED"),
              "NO BASELINE": (YELLOW, "— NO BASELINE"), "MEASURED": (GREEN, "✓ MEASURED"),
              "FAILED": (RED, "✗ FAILED")}
    print()
    print(line("╔", "═", "╗"))
    print("║  " + title.ljust(inner - 2) + "║")
    print(line("╠", "═", "╣"))
    for name, st in rows:
        color, lbl = status.get(st, (NC, st))
        pad = inner - len(lbl) - 3
        print(f"║  {name.ljust(pad)} {color}{lbl}{NC}  ║")
    print(line("╠", "═", "╣"))
    failed = any(st == "FAILED" for _, st in rows)
    if failed:
        color, lbl = RED, "✗ FAILED"
    elif mode != "compare":
        color, lbl = YELLOW, "— MEASURED"
    elif overall_ok:
        color, lbl = GREEN, "✓ PASSED"
    else:
        color, lbl = RED, "✗ REGRESSED"
    pad = inner - len(lbl) - 3
    print(f"║  {'Overall'.ljust(pad)} {color}{lbl}{NC}  ║")
    print(line("╚", "═", "╝"))


def positive_int(text):
    """argparse type for counts that must be at least 1 (e.g. --runs, which is a divisor)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not an integer")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main():
    ap = argparse.ArgumentParser(description="rocDecode performance regression check")
    ap.add_argument("--perf-dir", default=os.environ.get(
        "ROCDECODE_PERF_DIR", str(Path.home() / "rocDecodePerformance")),
        help="dir with per-codec stream subdirs and the baseline HTML")
    ap.add_argument("--baseline", default=None,
        help="baseline file, HTML or CSV (default: <perf-dir>/rocDecode_perf_baseline.html "
             "for --mem-type 3, <perf-dir>/local_baselines/<GPU>_m<N>.csv otherwise)")
    ap.add_argument("--tolerance", type=float,
        default=float(os.environ.get("ROCDECODE_PERF_TOLERANCE", "5")),
        help="regression threshold: %% Avg FPS drop below baseline (default: 5)")
    ap.add_argument("--runs", type=positive_int, default=3,
        help="runs to average when confirming a flagged stream, or per stream with "
             "--save-baseline (default: 3)")
    ap.add_argument("--device", type=int, default=0, help="GPU device id (default: 0)")
    ap.add_argument("--gpu", default=None,
        help="GPU name for baseline lookup (e.g. Navi31, MI300X). "
             "Overrides auto-detection; also overridable via ROCDECODE_PERF_GPU env var.")
    ap.add_argument("--check-rocm", action="store_true",
        help="only verify the ROCm toolchain (ROCM_PATH) and exit")
    ap.add_argument("--quick", action="store_true",
        help="fast check: one baseline stream per leaf subfolder, capped at <=4K")
    ap.add_argument("--mem-type", type=int, choices=range(4), default=DEFAULT_MEM_TYPE,
        help="output memory type passed to videodecodeperf as -m: 0 device internal, "
             "1 device copied, 2 host copied, 3 not mapped / decode only (default: 3). "
             "3 is compared against the HTML baseline; 0-2 against a local baseline "
             "recorded with --save-baseline")
    ap.add_argument("--save-baseline", action="store_true",
        help="measure every stream (averaging --runs runs each) and save the result as the "
             "local baseline for this GPU and --mem-type, instead of comparing")
    ap.add_argument("--force", action="store_true",
        help="with --save-baseline, overwrite an existing baseline file")
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
    html_baseline = perf_dir / "rocDecode_perf_baseline.html"
    mem = args.mem_type
    tol = args.tolerance

    if args.save_baseline and args.quick:
        # Later runs compare only the streams that are in the baseline, so a
        # quick (partial) baseline would silently shrink their coverage.
        print("ERROR: --save-baseline cannot be combined with --quick; a baseline must "
              "cover every stream.", file=sys.stderr)
        return 1
    if args.force and not args.save_baseline:
        print("ERROR: --force only applies together with --save-baseline.", file=sys.stderr)
        return 1

    if not PERF_EXE.is_file():
        print(f"ERROR: videodecodeperf not built: {PERF_EXE}", file=sys.stderr)
        print("       Build it first: cmake --build build (or test/build_samples.sh)", file=sys.stderr)
        return 1

    duplicates = find_duplicate_streams(perf_dir)
    if duplicates:
        print("ERROR: stream file names must be unique across the perf dir, because "
              "baselines are keyed by file name. Duplicates:", file=sys.stderr)
        for name, ps in sorted(duplicates.items()):
            print(f"       {name}: {', '.join(str(p) for p in ps)}", file=sys.stderr)
        return 1

    label, detail = detect_gpu(args.device, args.gpu)
    # label is replaced by a generic "device-N" below when detection fails, so
    # remember the failure for messages that need a real GPU name.
    gpu_unidentified = label is None
    baseline = {}
    col = None
    baseline_kind = None
    # measure_only: every stream is measured and none is compared. That is the case
    # both with no usable baseline and with --save-baseline; the reporting tells them
    # apart through summary_mode. measure_only_reason is set only for the former.
    measure_only = False
    measure_only_reason = ""
    save_path = None

    if args.save_baseline:
        if label is None:
            # The default file name is keyed by GPU, and a baseline from an
            # unidentified GPU could not be matched to later runs.
            print(f"ERROR: could not identify the GPU ({detail}); a baseline needs a GPU name.",
                  file=sys.stderr)
            print("       Use --gpu <name> (e.g. --gpu Navi31) or set ROCDECODE_PERF_GPU.",
                  file=sys.stderr)
            return 1
        save_path = Path(args.baseline) if args.baseline else local_baseline_path(perf_dir, label, mem)
        if save_path.suffix.lower() != ".csv":
            print(f"ERROR: a saved baseline must be a .csv file: {save_path}", file=sys.stderr)
            return 1
        if save_path.exists() and not args.force:
            print(f"ERROR: baseline already exists: {save_path}", file=sys.stderr)
            print("       Pass --force to overwrite it.", file=sys.stderr)
            return 1
        measure_only = True
    else:
        # Baseline files to try, in order, as (path, kind). The HTML baseline holds
        # decode-only (-m 3) numbers, so other memory types use a local CSV; for
        # -m 3 the local CSV is a fallback for GPUs the HTML has no column for.
        if args.baseline:
            p = Path(args.baseline)
            suffix = p.suffix.lower()
            if suffix == ".csv":
                kind = "csv"
            elif suffix in (".html", ".htm"):
                kind = "html"
            else:
                # Without this, any other suffix (e.g. a ".cvs" typo) was parsed as
                # HTML and failed with an unrelated "No tables found" error.
                print(f"ERROR: unsupported baseline file type: {p}", file=sys.stderr)
                print("       A baseline must be a .csv (local baseline) or .html "
                      "(SharePoint baseline).", file=sys.stderr)
                return 1
            tries = [(p, kind)]
            if kind == "html" and mem != DEFAULT_MEM_TYPE:
                print(f"WARNING: the HTML baseline holds decode-only (-m {DEFAULT_MEM_TYPE}) "
                      f"numbers; comparing -m {mem} against it will over-report regressions.")
        elif mem == DEFAULT_MEM_TYPE:
            tries = [(html_baseline, "html")]
            if label is not None:
                tries.append((local_baseline_path(perf_dir, label, mem), "csv"))
        else:
            tries = [(local_baseline_path(perf_dir, label, mem), "csv")] if label is not None else []

        reasons = []
        for path, kind in tries:
            if not path.is_file():
                reasons.append(f"baseline file not found: {path}")
                continue
            if kind == "csv":
                try:
                    baseline = load_local_baseline(path, mem, label)
                except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                    return 1
                col, baseline_kind = str(path), "csv"
                break
            if label is None:
                # GPU unknown but the HTML baseline exists — we need a column name
                # to look up, so detection failure is fatal.
                print(f"ERROR: could not identify the GPU ({detail}).", file=sys.stderr)
                print("       Use --gpu <name> (e.g. --gpu Navi31) or set ROCDECODE_PERF_GPU.",
                      file=sys.stderr)
                return 1
            try:
                baseline, col = load_baseline(path, label)
                baseline_kind = "html"
                break
            except ValueError as e:
                # Invalid baseline (e.g. no table with a 'File' column) — fatal.
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            except KeyError as e:
                # No baseline column for this GPU — try the next file, else
                # measure only.
                reasons.append(str(e))

        if baseline_kind is None:
            # No usable baseline — measure every stream and save the results to
            # CSV, which can serve as a future baseline.
            measure_only = True
            if label is None:
                label = f"device-{args.device}"
                detail = f"unidentified ({detail}); labelled as {label}"
                reasons.insert(0, f"GPU could not be identified ({detail})")
            measure_only_reason = "; ".join(reasons)

    print(f"GPU: {label}  ({detail})")
    print(f"Memory type: {mem} ({MEM_TYPE_NAMES[mem]})")
    if args.save_baseline:
        print(f"Saving baseline: {save_path}  (averaging {args.runs} runs per stream)")
    elif measure_only:
        print(f"WARNING: {measure_only_reason}")
        print("Running in measure-only mode — all streams will be measured and")
        print("saved to CSV; no regression comparison.")
        # Repeat the options that decide what gets measured and where the baseline
        # is written, so the suggested command records the baseline this run needs.
        hint = ["test/perf_regression.py", "--mem-type", str(mem)]
        if args.perf_dir != ap.get_default("perf_dir"):
            hint += ["--perf-dir", args.perf_dir]
        if args.device != 0:
            hint += ["--device", str(args.device)]
        if gpu_unidentified:
            hint += ["--gpu", "<name>"]
        elif args.gpu:
            hint += ["--gpu", args.gpu]
        hint.append("--save-baseline")
        print(f"To record a baseline: {' '.join(hint)}")
        if gpu_unidentified:
            print("  (the GPU could not be identified: replace <name> with its baseline name,")
            print("   e.g. Navi31, or set ROCDECODE_PERF_GPU)")
    elif baseline_kind == "csv":
        print(f"Baseline file: {col}  ({len(baseline)} streams)  tolerance: {tol}%")
    else:
        print(f"Baseline column: {col}  ({len(baseline)} streams)  tolerance: {tol}%")
    if args.quick:
        print("Quick mode: one stream per leaf subfolder, capped at <=4K")
    print()

    summary_mode = "save" if args.save_baseline else ("measure" if measure_only else "compare")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Write results outside the repo tree (default: $HOME) so they do not clutter
    # `git status`. Override the base dir with ROCDECODE_PERF_RESULTS_DIR.
    results_base = Path(os.path.expanduser(os.environ.get(
        "ROCDECODE_PERF_RESULTS_DIR", str(Path.home() / "rocDecode_perf_results"))))
    out_dir = results_base / f"{ts}_m{mem}"  # created only if there are results to write
    detail_rows = []
    summary_rows = []
    overall_ok = True
    # With --save-baseline, anything that could not be measured. Later runs
    # compare only the streams in the baseline, so saving without them would
    # silently drop them from regression coverage.
    save_failures = []
    t0 = time.time()

    for disp, subdir in CODECS:
        codec_dir = perf_dir / subdir
        if not codec_dir.is_dir():
            print(f"=== {disp}: stream dir not found ({codec_dir}) -- skipping ===")
            summary_rows.append((disp, "WARNING"))
            if args.save_baseline:
                save_failures.append(f"{disp} stream dir not found ({codec_dir})")
            continue
        print(f"=== Measuring {disp} ({codec_dir}) ===")
        index = build_stream_index(codec_dir)
        # In measure-only mode, run every stream; otherwise only those with a
        # baseline entry. Sorted for deterministic order.
        if measure_only:
            candidates = sorted(index.items())
        else:
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
        # to rule out run-to-run noise); a saved baseline averages --runs runs so
        # it is not skewed by one noisy run. Measuring videoDecodePerf directly
        # avoids writing any results into the repo tree.
        runs = args.runs if args.save_baseline else 1
        measured = {}
        failed = []  # streams of this codec that could not be measured
        for name, path in candidates:
            fps = measure_stream(path, runs, args.device, mem)
            if fps is not None:
                measured[name] = fps
            else:
                failed.append(name)
        if not candidates:
            print(f"    no results produced for {disp}")
            summary_rows.append((disp, "WARNING"))
            if args.save_baseline:
                save_failures.append(f"{disp} has no streams in {codec_dir}")
            continue
        # A stream that fails to decode is a failure in every mode: dropping it
        # would let a build that crashes on it pass, and saving without it would
        # remove it from later comparisons.
        for name in failed:
            print(f"    {RED}FAILED   {NC} {name}  (videodecodeperf failed or reported no FPS)")
            detail_rows.append({
                "Codec": disp, "File": name, "Mem Type": mem,
                "Baseline FPS": round(baseline[name], 2) if name in baseline else "",
                "Measured FPS": "", "Delta %": "", "Verdict": "FAILED", "Note": "",
            })
        save_failures.extend(failed)
        if failed:
            overall_ok = False
        if not measured:
            summary_rows.append((f"{disp} ({len(failed)}/{len(candidates)} failed)", "FAILED"))
            continue

        if measure_only:
            # No baseline — just record the measured FPS for each stream.
            for name, m_fps in measured.items():
                detail_rows.append({
                    "Codec": disp, "File": name, "Mem Type": mem, "Baseline FPS": "",
                    "Measured FPS": round(m_fps, 2), "Delta %": "",
                    "Verdict": "MEASURED", "Note": f"avg of {runs} runs" if runs > 1 else "",
                })
                print(f"    {GREEN}MEASURED {NC} {name}  {m_fps:8.1f} fps")
            if failed:
                summary_rows.append((f"{disp} ({len(failed)}/{len(candidates)} failed)", "FAILED"))
            else:
                summary_rows.append((f"{disp} ({len(measured)} streams)",
                                     "MEASURED" if args.save_baseline else "NO BASELINE"))
        else:
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
                    rerun_failed = False
                    if name in index:
                        avg = measure_stream(index[name], args.runs, args.device, mem)
                        if avg is not None:
                            confirmed = avg
                            delta = (avg - b_fps) / b_fps * 100.0
                            note = f"re-ran {args.runs}x"
                        else:
                            rerun_failed = True
                            note = "re-run failed"
                    m_fps = confirmed
                    if rerun_failed:
                        verdict = "FAILED"
                        failed.append(name)
                    elif delta < -tol:
                        verdict = "REGRESSED"
                        regressions += 1
                    else:
                        verdict = "OK*"  # recovered on re-run
                detail_rows.append({
                    "Codec": disp, "File": name, "Mem Type": mem, "Baseline FPS": round(b_fps, 2),
                    "Measured FPS": round(m_fps, 2), "Delta %": round(delta, 2),
                    "Verdict": verdict, "Note": note,
                })
                tag = {"OK": GREEN, "OK*": YELLOW, "REGRESSED": RED, "FAILED": RED}[verdict]
                print(f"    {tag}{verdict:9}{NC} {name}  "
                      f"{m_fps:8.1f} vs {b_fps:8.1f} fps  ({delta:+.1f}%){(' '+note) if note else ''}")

            if failed:
                overall_ok = False
                summary_rows.append((f"{disp} ({len(failed)}/{len(candidates)} failed)", "FAILED"))
            elif compared == 0:
                summary_rows.append((f"{disp} (0 matched)", "WARNING"))
            elif regressions:
                summary_rows.append((f"{disp} ({regressions}/{compared} regressed)", "REGRESSED"))
                overall_ok = False
            else:
                summary_rows.append((f"{disp} ({compared} streams)", "PASSED"))

    if measure_only and not args.save_baseline:
        print(f"\nWARNING: {measure_only_reason}")
    if detail_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "perf_comparison.csv"
        pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
        summary_box(summary_rows, overall_ok, mode=summary_mode)
        print(f"Results saved to: {out_dir}/")
        if args.save_baseline and save_failures:
            # Any existing baseline is left untouched: it is only written below.
            print(f"ERROR: baseline not saved; could not measure: {'; '.join(save_failures)}",
                  file=sys.stderr)
            return 1
        if args.save_baseline:
            bad = [r["File"] for r in detail_rows if not r["Measured FPS"] > 0]
            if bad:
                # load_local_baseline rejects non-positive FPS, so do not write a
                # baseline that could never be loaded.
                print(f"ERROR: no valid FPS measured for {', '.join(bad)}; baseline not saved.",
                      file=sys.stderr)
                return 1
            # The loader reads File and FPS; the other columns record where the
            # baseline came from.
            recorded = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rocm_path = os.environ.get("ROCM_PATH", _DEFAULT_ROCM)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "Codec": r["Codec"], "File": r["File"], "FPS": r["Measured FPS"],
                "Mem Type": mem, "GPU": label, "Recorded": recorded, "ROCM_PATH": rocm_path,
            } for r in detail_rows]).to_csv(save_path, index=False)
            print(f"Baseline saved to: {save_path}  ({len(detail_rows)} streams)")
        elif measure_only:
            print("These measurements can serve as a future baseline for this GPU.")
    else:
        if measure_only:
            summary_box(summary_rows, overall_ok, mode=summary_mode)
            print("ERROR: No streams were measured; no results CSV was saved.",
                  file=sys.stderr)
            return 1
        summary_box(summary_rows, overall_ok)
        print("No streams were compared (check ROCDECODE_PERF_DIR and the baseline).")
    elapsed = int(time.time() - t0)
    print(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")
    print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
