#!/usr/bin/env bash
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
#>>>HELP
# Generate code-coverage reports for amd-smi:
#   C/C++  -> gcovr (reads gcov data via llvm-cov / gcov)
#   Python -> coverage.py (branch coverage, incl. amd-smi CLI subprocesses)
#
# Usage:
#   run_coverage.sh [--verbose|--quiet] [--build[=TC]] [--stage[=DIR]|--installed] [all|cpp|py]
#   run_coverage.sh --clean | --clean-all
#
# Modes (positional, default: all):
#   all   C/C++ + Python
#   cpp   C/C++ only
#   py    Python only
#
# Flags:
#   -v, --verbose   per-test Python logging (-v shows every test name + skip reason)
#   -q, --quiet     terse Python logging (-q); the default
#   --build[=TC]    configure + build an instrumented build-cov with toolchain TC
#                   (gcc | clang | clang-NN; default clang-16), auto-matching GCOV_TOOL
#   --stage[=DIR]   cmake --install the build into DIR (default <repo>/build-cov-stage)
#                   and run the Python tier against it (no dpkg / no /opt/rocm changes).
#                   Auto-enabled when --build is combined with the Python tier (all/py).
#   --installed     Python tier measures the system-installed amd-smi package instead
#                   of staging (opts out of auto-stage; may be stale vs a fresh --build)
#   --clean         remove build artifacts (build-cov, build-cov-stage, .coverage*)
#                   but KEEP the coverage-report/ outputs; use alone.
#                   Only touches coverage-owned dirs (never a user BUILD_DIR).
#   --clean-all     remove everything, coverage-report/ included; use alone.
#   -h, --help      show this help
#
# Environment knobs:
#   BUILD_DIR     instrumented build tree            (default: <repo>/build)
#   OUT_DIR       where reports are written          (default: <repo>/coverage-report)
#   GCOV_TOOL     gcov reader matching the compiler  (default: "llvm-cov-14 gcov";
#                 gcc -> "gcov", clang-N -> "llvm-cov-N gcov"; mismatch => silent 0/0)
#   CLI_SRC       amd-smi CLI source dir             (default: /opt/rocm/libexec/amdsmi_cli)
#   TEST_FILTER   gtest --gtest_filter for amdsmitst (default: run all)
#   UNIT_ONLY     1 = unit tests only (C/C++ '*Unit*'; Python skips cli_unit_test.py)
#   VERBOSE       1 = same as --verbose
#   STAGE_DIR     staged-install prefix for --stage  (default: <repo>/build-cov-stage)
#
# Examples:
#   # Build (clang-16) + C/C++ only:
#   run_coverage.sh --build=clang-16 cpp
#
#   # Build + stage-install + BOTH tiers, verbose (no system install):
#   run_coverage.sh --build=clang-16 --stage --verbose all
#
#   # GCC build (reader auto-set to gcov):
#   run_coverage.sh --build=gcc cpp
#
#   # Reuse an existing build with a custom reader:
#   GCOV_TOOL="llvm-cov-16 gcov" BUILD_DIR=build-cov run_coverage.sh cpp
#
#   # Unit-only into a separate report dir:
#   UNIT_ONLY=1 OUT_DIR=coverage-report-unit run_coverage.sh all
#
#   # Clean up (each used alone)- keep the coverage-report/ outputs or wipe all:
#   run_coverage.sh --clean
#   run_coverage.sh --clean-all
#
# Prereqs:
#   - pip install --user gcovr coverage. C/C++ needs an instrumented build
#     (or pass --build)
#   - Python needs the amd-smi-lib package installed (or pass --stage).
#
# Runnable from any directory -- paths resolve relative to the script's location.
#<<<HELP
set +x
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"            # projects/amdsmi
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
# Normalize to an absolute path: run_cpp `cd`s into BUILD_DIR before running the
# binary, so a relative BUILD_DIR (e.g. "build-cov") would otherwise resolve the
# test path against the wrong dir and the binary would never run (0 .gcda).
BUILD_DIR="$(cd "$BUILD_DIR" 2>/dev/null && pwd || echo "$BUILD_DIR")"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/coverage-report}"
GCOV_TOOL="${GCOV_TOOL:-llvm-cov-14 gcov}"
CLI_SRC="${CLI_SRC:-/opt/rocm/libexec/amdsmi_cli}"
TEST_FILTER="${TEST_FILTER:-}"
UNIT_ONLY="${UNIT_ONLY:-0}"
VERBOSE="${VERBOSE:-0}"
DO_BUILD=0; BUILD_TC=""
DO_STAGE=0; STAGE_DIR="${STAGE_DIR:-$REPO_ROOT/build-cov-stage}"
DO_INSTALLED=0
DO_CLEAN=0; CLEAN_ALL=0
STAGED=0

# Parse optional flags around the positional mode. usage() prints the HELP block
# from the file header, so --help and the in-file docs stay in sync.
MODE=""
usage() { sed -n '/^#>>>HELP$/,/^#<<<HELP$/p' "$0" | sed '1d;$d; s/^#//; s/^ //'; }
argc=$#
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=1 ;;
        -q|--quiet)   VERBOSE=0 ;;
        --build)      DO_BUILD=1 ;;
        --build=*)    DO_BUILD=1; BUILD_TC="${1#*=}" ;;
        --stage)      DO_STAGE=1 ;;
        --stage=*)    DO_STAGE=1; STAGE_DIR="${1#*=}" ;;
        --installed)  DO_INSTALLED=1 ;;
        --clean)      DO_CLEAN=1 ;;
        --clean-all)  DO_CLEAN=1; CLEAN_ALL=1 ;;
        all|cpp|py)   MODE="$1" ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2
           echo "Run '$(basename "$0") --help' for usage." >&2; exit 2 ;;
    esac
    shift
done
# --clean/--clean-all are destructive standalone actions: refuse to combine them
# with a mode or any build/run flag so they can never trigger a run.
if [[ "$DO_CLEAN" == "1" && "$argc" -ne 1 ]]; then
    echo "--clean/--clean-all must be used alone (no other arguments)." >&2
    exit 2
fi
MODE="${MODE:-all}"

# The Python tier needs an importable binding + CLI. If you just built a fresh
# instrumented tree (--build) and want Python coverage, the only correct thing to
# measure is that build (a system install would be stale/uninstrumented), so
# auto-enable staging. --installed opts out to measure the system package instead.
if [[ "$MODE" != "cpp" && "$DO_INSTALLED" != "1" && "$DO_BUILD" == "1" && "$DO_STAGE" != "1" ]]; then
    DO_STAGE=1
    echo "(--build + Python tier: auto-staging to measure the freshly built binding;"
    echo " pass --installed to measure the system amd-smi package instead)"
fi
if [[ "$DO_INSTALLED" == "1" && "$DO_BUILD" == "1" ]]; then
    echo "(--installed: measuring the system amd-smi package, which may be stale vs your --build)" >&2
fi

# UNIT_ONLY scopes both tiers to unit tests. For C/C++ that's a gtest filter
# (unit suites are named '*Unit*', e.g. GpuUnit/SystemUnit); an explicit
# TEST_FILTER always wins.
if [[ "$UNIT_ONLY" == "1" && -z "$TEST_FILTER" ]]; then
    TEST_FILTER='*Unit*'
fi

# Locate gcovr. gcovr is a user-site pip install, so under sudo $HOME is /root
# and it won't be found there -- fall back to the invoking user's ~/.local.
resolve_gcovr() {
    command -v gcovr 2>/dev/null && return 0
    local homes=("$HOME")
    [[ -n "${SUDO_USER:-}" ]] && homes+=("$(getent passwd "$SUDO_USER" | cut -d: -f6)")
    for h in "${homes[@]}"; do
        [[ -x "$h/.local/bin/gcovr" ]] && { echo "$h/.local/bin/gcovr"; return 0; }
    done
    echo gcovr  # last resort; errors clearly if truly absent
}
GCOVR="$(resolve_gcovr)"

# --build: configure + build an instrumented build-cov with the chosen toolchain,
# then point BUILD_DIR/GCOV_TOOL at it so the reader always matches the compiler.
build_cov() {
    local tc="${BUILD_TC:-clang-16}" cc cxx
    case "$tc" in
        gcc)     cc=gcc;   cxx=g++ ;;
        clang)   cc=clang; cxx=clang++ ;;
        clang-*) cc="clang-${tc#clang-}"; cxx="clang++-${tc#clang-}" ;;
        *) echo "ERROR: --build toolchain '$tc' not recognized (use gcc|clang|clang-NN)." >&2; return 2 ;;
    esac
    local bdir="$REPO_ROOT/build-cov"
    echo "=================  Building instrumented tree ($tc)  ================="
    # CMake pins the compiler in the cache on first configure and refuses to change
    # it in place, so switching toolchains (e.g. clang-16 -> gcc) on an existing
    # build-cov errors out. If the cached compiler differs from the requested one,
    # wipe for a clean configure; same compiler reuses the cache (fast incremental).
    if [[ -f "$bdir/CMakeCache.txt" ]]; then
        local cached want
        cached="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$bdir/CMakeCache.txt")"
        want="$(command -v "$cxx" 2>/dev/null || echo "$cxx")"
        if [[ -n "$cached" \
              && "$(readlink -f "$cached" 2>/dev/null || echo "$cached")" \
              != "$(readlink -f "$want" 2>/dev/null || echo "$want")" ]]; then
            echo "(toolchain changed: cached '$cached' != requested '$want'; wiping $bdir)"
            rm -rf "$bdir"
        fi
    fi
    CC="$cc" CXX="$cxx" cmake -S "$REPO_ROOT" -B "$bdir" \
        -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON -DBUILD_EXAMPLES=OFF
    # Staging installs the whole tree (lib + python pkg + CLI); a plain cpp run
    # only needs the test binary.
    if [[ "$DO_STAGE" == "1" ]]; then
        cmake --build "$bdir" -j"$(nproc)"
    else
        cmake --build "$bdir" --target amdsmitst -j"$(nproc)"
    fi
    BUILD_DIR="$bdir"
    case "$tc" in
        gcc)     GCOV_TOOL="gcov" ;;
        clang)   GCOV_TOOL="llvm-cov gcov" ;;
        clang-*) GCOV_TOOL="llvm-cov-${tc#clang-} gcov" ;;
    esac
    echo "(auto-set BUILD_DIR=$BUILD_DIR, GCOV_TOOL='$GCOV_TOOL')"
}

# --stage: cmake --install the build into a user-writable prefix (install-tree
# layout the Python tests expect) so the Python tier runs against exactly what we
# built -- no 'dpkg -i', no /opt/rocm mutation. Wires the tests to the staged tree.
stage_install() {
    echo "=================  Staging install -> $STAGE_DIR  ================="
    # A prior staged run under sudo can leave root-owned __pycache__ here, which a
    # plain (non-root) rm can't remove -> fall back to sudo.
    rm -rf "$STAGE_DIR" 2>/dev/null || sudo rm -rf "$STAGE_DIR"
    if ! cmake --install "$BUILD_DIR" --prefix "$STAGE_DIR" >/dev/null 2>&1; then
        echo "ERROR: 'cmake --install $BUILD_DIR' failed -- the tree may be only partly" >&2
        echo "  built. Build the FULL tree first (e.g. add '--build', which does so)." >&2
        return 1
    fi
    # sudo strips LD_LIBRARY_PATH/PATH; run_py re-passes these via `env` (below).
    export AMDSMI_PATH="$STAGE_DIR/share/amd_smi"
    export LD_LIBRARY_PATH="$STAGE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="$STAGE_DIR/bin:$PATH"
    CLI_SRC="$STAGE_DIR/libexec/amdsmi_cli"
    STAGED=1
    echo "(Python tier -> staged tree: AMDSMI_PATH=$AMDSMI_PATH)"
}

run_cpp() {
    echo "=================  C/C++ coverage (gcovr)  ================="
    local tst="$BUILD_DIR/tests/amd_smi_test/amdsmitst"
    if [[ ! -x "$tst" ]]; then
        echo "ERROR: $tst not found." >&2
        echo "Configure + build with:  cmake -S . -B build -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON && cmake --build build" >&2
        return 1
    fi
    mkdir -p "$OUT_DIR/cpp"
    # Drop any stale machine-readable summary so a failed run this time reads as
    # ERROR (absent) rather than showing last run's numbers.
    rm -f "$OUT_DIR/cpp/summary.json"
    # The functional gtests need root for hardware access, so escalate just the
    # binary; gcovr must stay as the invoking user (root can't import the
    # user-site gcovr). Run this script WITHOUT sudo so gcovr resolves correctly.
    local sudo_prefix=()
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo_prefix=(sudo -E)
        echo "(escalating just the test binary via 'sudo -E'; gcovr runs as $USER)"
    fi
    # Run the suite to emit .gcda next to the objects. Coverage is the goal, so a
    # failing/skipped test (no hardware) must not abort the report.
    # Reset accumulated counters first so the report reflects only this run's
    # tests (gcov .gcda counts accumulate across runs otherwise).
    find "$BUILD_DIR" -name '*.gcda' -delete 2>/dev/null || true
    [[ -n "$TEST_FILTER" ]] && echo "(gtest filter: $TEST_FILTER)"
    local cpp_cmd=( "${sudo_prefix[@]}" "$tst" ${TEST_FILTER:+--gtest_filter="$TEST_FILTER"} )
    echo ">>> [$(date '+%H:%M:%S')] START amdsmitst"
    echo "    cmd: ( cd $BUILD_DIR && ${cpp_cmd[*]} )"
    local cpp_rc=0
    ( cd "$BUILD_DIR" && "${cpp_cmd[@]}" ) || cpp_rc=$?
    echo "<<< [$(date '+%H:%M:%S')] END   amdsmitst (exit $cpp_rc)"

    # No .gcda => the binary never reached a normal exit (sudo not completed, or a
    # crash before gcov flushes at exit). gcovr would then silently report 0% for
    # every file, which looks like "coverage is just low" -- fail loudly instead.
    local gcda_count
    gcda_count=$(find "$BUILD_DIR" -name '*.gcda' 2>/dev/null | wc -l)
    if [[ "$gcda_count" -eq 0 ]]; then
        echo "ERROR: no .gcda were produced under $BUILD_DIR." >&2
        echo "  The tests did not run to a normal exit, so there is NO coverage data" >&2
        echo "  (gcovr would report 0% for everything). Likely causes:" >&2
        echo "    * the 'sudo' escalation didn't run the binary (password not entered)," >&2
        echo "    * amdsmitst crashed/was killed before exit (no gcov flush), or" >&2
        echo "    * BUILD_DIR points at a build without -DENABLE_COVERAGE=ON." >&2
        echo "  Run it directly and watch it finish, then re-run coverage:" >&2
        echo "    ${sudo_prefix[*]:+${sudo_prefix[*]} }$tst" >&2
        # Leave a machine-readable note in place of the report so the final
        # summary shows the C/C++ report couldn't be generated.
        mkdir -p "$OUT_DIR"
        {
            echo "C/C++ coverage report NOT generated."
            echo "Reason: no .gcda produced under $BUILD_DIR (amdsmitst exit=$cpp_rc; tests did not run)."
            echo "See the ERROR details above in the run log."
        } > "$OUT_DIR/cpp-summary.txt"
        rm -f "$OUT_DIR/cpp/index.html" 2>/dev/null || true
        return 1
    fi
    echo "(collected $gcda_count .gcda files)"
    # .gcda written by the root-run binary are 0644, so gcovr reads them fine.
    "$GCOVR" \
        --root "$REPO_ROOT" \
        --gcov-executable "$GCOV_TOOL" \
        --filter "$REPO_ROOT/src/" \
        --filter "$REPO_ROOT/rocm_smi/src/" \
        --exclude '.*_test\.cc' \
        --print-summary \
        --txt "$OUT_DIR/cpp-summary.txt" \
        --json-summary-pretty \
        --json-summary "$OUT_DIR/cpp/summary.json" \
        --html-details "$OUT_DIR/cpp/index.html" \
        "$BUILD_DIR"
}

run_py() {
    echo "=================  Python coverage (coverage.py)  ================="
    # Resolve the binding the SAME way the tests do. common.common may stage a
    # copy (e.g. /opt/rocm/share/amd_smi/amdsmi) ahead of the dist-packages one
    # on sys.path; measuring a plain `import amdsmi` would target the wrong copy
    # and silently report 0% (the tests exercise the staged copy).
    local binding
    binding="$(cd "$REPO_ROOT/tests/python" \
        && python3 -c 'import common.common as c, os; print(os.path.dirname(c.amdsmi.__file__))' 2>/dev/null)"
    [[ -z "$binding" ]] && \
        binding="$(python3 -c 'import amdsmi, os; print(os.path.dirname(amdsmi.__file__))' 2>/dev/null)"
    if [[ -z "$binding" ]]; then
        echo "ERROR: Python tier could not import 'amdsmi' -- no binding to measure." >&2
        echo "  Pick one:" >&2
        echo "    --stage      measure a throwaway install of your tree (with --build to build it)" >&2
        echo "    --installed  measure the system amd-smi package (install amd-smi-lib first)" >&2
        echo "  e.g.  run_coverage.sh --build=gcc --stage all" >&2
        return 1
    fi
    echo "(binding source: $binding)"

    local pydir="$OUT_DIR/py"
    rm -rf "$pydir"
    mkdir -p "$pydir/_bootstrap"

    # Build an effective rc that pins the resolved source dirs. Both the parent
    # process and any spawned `amd-smi` subprocess read this same file.
    local eff_rc="$pydir/coveragerc.effective"
    python3 - "$HERE/.coveragerc" "$eff_rc" "$binding" "$CLI_SRC" <<'PY'
import sys
base, out, binding, cli = sys.argv[1:5]
src = f"source =\n    {binding}\n    {cli}"
text = open(base).read().replace(
    "# __SOURCE__ (run_coverage.sh replaces this marker with a resolved source = list)",
    src,
)
open(out, "w").write(text)
PY

    # Capture coverage from spawned amd-smi CLI subprocesses: a sitecustomize on
    # PYTHONPATH calls coverage.process_startup() when COVERAGE_PROCESS_START is set.
    printf 'import coverage\ncoverage.process_startup()\n' > "$pydir/_bootstrap/sitecustomize.py"

    # The runners (common.run_test_dir) sys.exit(1) unless euid==0, so the suites
    # only execute product code under root. Run coverage under sudo, but keep the
    # user site-packages on PYTHONPATH so both `coverage` and the bootstrap's
    # process_startup() resolve for root and for the amd-smi subprocesses.
    local user_site
    user_site="$(python3 -c 'import site; print(site.getusersitepackages())')"
    local pypath="$pydir/_bootstrap:$user_site${PYTHONPATH:+:$PYTHONPATH}"
    local cov_file="$pydir/.coverage"
    local sudo_prefix=()
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo_prefix=(sudo -E)
        echo "(not root -> running the Python suites under 'sudo -E'; the runners require it)"
    fi
    # Pass the coverage env explicitly through `env` so sudo can't strip it.
    # PYTHONDONTWRITEBYTECODE: the suites run under sudo, so any __pycache__ they
    # write would be root-owned and break a later (non-root) cleanup of the stage.
    local cov_env=(
        "PYTHONPATH=$pypath"
        "COVERAGE_PROCESS_START=$eff_rc"
        "COVERAGE_FILE=$cov_file"
        "PYTHONDONTWRITEBYTECODE=1"
    )
    # Staged run: the tests import the binding/CLI from the staged prefix. sudo
    # drops LD_LIBRARY_PATH/PATH, so pass them (and AMDSMI_PATH) explicitly via env.
    if [[ "$STAGED" == "1" ]]; then
        cov_env+=(
            "AMDSMI_PATH=$AMDSMI_PATH"
            "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
            "PATH=$PATH"
        )
    fi
    covpy() { "${sudo_prefix[@]}" env "${cov_env[@]}" python3 -m coverage "$@"; }
    # Log start/stop + the exact command for each suite so library stderr (e.g.
    # "Exception caught: map::at") can be attributed to the suite that triggered it.
    local vflag="-q"; [[ "$VERBOSE" == "1" ]] && vflag="-v"
    [[ "$VERBOSE" == "1" ]] && echo "(verbose: passing '$vflag' -> per-test names + skip reasons)"
    covrun() {
        local f="$1"
        echo ""
        echo ">>> [$(date '+%H:%M:%S')] START $f  ($vflag)"
        echo "    cmd: ${sudo_prefix[*]:+${sudo_prefix[*]} }env <coverage-env> python3 -m coverage run --rcfile=$eff_rc $f $vflag"
        local rc=0
        covpy run --rcfile="$eff_rc" "$f" "$vflag" || rc=$?
        echo "<<< [$(date '+%H:%M:%S')] END   $f (exit $rc)"
    }

    local tests_py="$REPO_ROOT/tests/python"
    pushd "$tests_py" >/dev/null
    # -q keeps the console readable; failures on absent hardware are fine, we only
    # need the executed-line data, so don't let a non-zero exit abort the report.
    covrun unit_tests.py
    # integration_test.py (functional, live-HW) and cli_unit_test.py (spawns the
    # real amd-smi binary) exercise far more of the binding/CLI than the mock
    # unit tests -- include them in the full run, skip under UNIT_ONLY.
    if [[ "$UNIT_ONLY" == "1" ]]; then
        echo "(UNIT_ONLY=1 -> skipping integration_test.py + cli_unit_test.py; unit tests only)"
    else
        covrun integration_test.py
        covrun cli_unit_test.py
    fi
    popd >/dev/null

    # combine/report resolve the [paths] source-tree aliases relative to CWD.
    pushd "$REPO_ROOT" >/dev/null
    covpy combine --rcfile="$eff_rc" || true
    covpy report --rcfile="$eff_rc" | tee "$OUT_DIR/py-summary.txt"
    covpy html --rcfile="$eff_rc" -d "$pydir/html"
    # gcovr-style per-metric covered/total. `coverage report` only prints the
    # blended % + raw branch counts; `coverage json` exposes the line & branch
    # numerators/denominators, so we format them like the C/C++ gcovr table.
    covpy json --rcfile="$eff_rc" -o "$pydir/coverage.json" || true
    python3 - "$pydir/coverage.json" <<'PY' | tee -a "$OUT_DIR/py-summary.txt"
import json, sys
try:
    t = json.load(open(sys.argv[1]))["totals"]
except Exception as e:
    print(f"\n(gcovr-style summary unavailable: {e})"); raise SystemExit(0)
pct = lambda c, n: f"{100.0*c/n:.1f}%" if n else "n/a"
sl = t.get("num_statements", 0)
cl = t.get("covered_lines", sl - t.get("missing_lines", 0))
nb = t.get("num_branches", 0)
cb = t.get("covered_branches", 0)
pb = t.get("num_partial_branches", 0)
print("\n=== gcovr-style summary (from coverage json) ===")
print(f"lines:     {pct(cl, sl):>6} ({cl:,} / {sl:,})")
print(f"branches:  {pct(cb, nb):>6} ({cb:,} / {nb:,}; {pb:,} partial)")
print("functions: n/a (coverage.py has no per-function metric)")
PY
    popd >/dev/null

    # Data + reports were written as root; hand them back to the invoking user.
    if [[ ${#sudo_prefix[@]} -gt 0 ]]; then
        sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR" || true
    fi
}

# List every artifact that was actually produced, so the paths are easy to find.
print_outputs() {
    echo
    echo "================  Output files  ================"
    echo "All reports under: $OUT_DIR"
    # C/C++: show the report, or a NOT-generated note if the run left the marker.
    if [[ -f "$OUT_DIR/cpp-summary.txt" ]] && grep -q "NOT generated" "$OUT_DIR/cpp-summary.txt"; then
        echo "  C/C++  report:  !! NOT GENERATED -- cannot generate report (see error"
        echo "                  details above); note written to $OUT_DIR/cpp-summary.txt"
    else
        [[ -f "$OUT_DIR/cpp/index.html"  ]] && echo "  C/C++  HTML:    $OUT_DIR/cpp/index.html"
        [[ -f "$OUT_DIR/cpp-summary.txt" ]] && echo "  C/C++  summary: $OUT_DIR/cpp-summary.txt"
    fi
    [[ -f "$OUT_DIR/py/html/index.html" ]] && echo "  Python HTML:    $OUT_DIR/py/html/index.html"
    [[ -f "$OUT_DIR/py-summary.txt"    ]] && echo "  Python summary: $OUT_DIR/py-summary.txt"
    [[ -f "$OUT_DIR/py/coverage.json"  ]] && echo "  Python JSON:    $OUT_DIR/py/coverage.json"
    [[ -f "$OUT_DIR/amdsmi_cc_summary.md" ]] && echo "  AMD SMI Summary (md):   $OUT_DIR/amdsmi_cc_summary.md"
    return 0   # never let a false [[ -f ... ]] test above become the exit code
}

# Build the combined C++/Python coverage table -> amdsmi_cc_summary.md + .txt.
# Per tier: real numbers if it ran, N/A if not selected this run, ERROR if it was
# selected but produced no data (e.g. no .gcda).
# Render "<driver> <version>" (e.g. "clang++-16 16.0.6", "g++ 9.4.0") from a
# compiler path, dropping distro noise like "Ubuntu clang version ... (++hash)".
pretty_compiler() {
    local p="$1" ver
    [[ -x "$p" ]] || { echo "unknown"; return; }
    ver="$("$p" -dumpfullversion 2>/dev/null || true)"          # gcc/g++ -> 9.4.0
    [[ -z "$ver" ]] && ver="$("$p" -dumpversion 2>/dev/null || true)"  # clang -> 16.0.6
    [[ -z "$ver" ]] && ver="$("$p" --version 2>/dev/null | head -1 || true)"
    echo "$(basename "$p") $ver"
}
write_summary_files() {
    local date_str cxx_comp c_comp cxx_path c_path comp_txt comp_md
    date_str="$(date +%m-%d-%Y)"
    cxx_comp="N/A (C/C++ tier not run)"; c_comp="$cxx_comp"
    if [[ "$MODE" != "py" ]]; then
        # gcc/clang ship separate C and C++ front-ends (gcc/g++, clang/clang++);
        # the library has both .c and .cc, so report each driver CMake used.
        cxx_path="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)"
        c_path="$(sed -n 's/^CMAKE_C_COMPILER:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)"
        cxx_comp="$(pretty_compiler "$cxx_path")"
        c_comp="$(pretty_compiler "$c_path")"
    fi
    if [[ "$cxx_comp" == N/A* ]]; then
        comp_txt="Compiler:     $cxx_comp"
        comp_md="**Compiler:** $cxx_comp"
    else
        comp_txt="C++ compiler: $cxx_comp"$'\n'"C compiler:   $c_comp"
        comp_md="**C++ compiler:** $cxx_comp  "$'\n'"**C compiler:** $c_comp"
    fi
    python3 - "$MODE" "$OUT_DIR/cpp/summary.json" "$OUT_DIR/py/coverage.json" \
        "$OUT_DIR/amdsmi_cc_summary.md" "$OUT_DIR/amdsmi_cc_summary.txt" \
        "$date_str" "$comp_txt" "$comp_md" <<'PY'
import json, sys
mode, cpp_json, py_json, out_md, out_txt, date_str, comp_txt, comp_md = sys.argv[1:9]
cpp_sel = mode in ("cpp", "all")
py_sel = mode in ("py", "all")

def pc(cov, tot):
    return "0.0% (0 / 0)" if not tot else f"{100.0*cov/tot:.1f}% ({cov:,} / {tot:,})"

def cpp_vals():
    if not cpp_sel:
        return "N/A"
    try:
        d = json.load(open(cpp_json))
        return {
            "Lines": pc(d["line_covered"], d["line_total"]),
            "Branches": pc(d["branch_covered"], d["branch_total"]),
            "Functions": pc(d["function_covered"], d["function_total"]),
        }
    except Exception:
        return "ERROR"

def py_vals():
    if not py_sel:
        return "N/A"
    try:
        t = json.load(open(py_json))["totals"]
        return {
            "Lines": pc(t.get("covered_lines", 0), t.get("num_statements", 0)),
            "Branches": pc(t.get("covered_branches", 0), t.get("num_branches", 0)),
            "Functions": "DNE",  # coverage.py has no per-function metric
        }
    except Exception:
        return "ERROR"

c, p = cpp_vals(), py_vals()
rows = ["Lines", "Branches", "Functions"]
def cell(v, k):
    return v if isinstance(v, str) else v[k]

legend = [
    "N/A   - tier not selected for this run.",
    "ERROR - tier selected but no data (e.g. no .gcda; tests did not run).",
    "DNE   - metric does not exist (coverage.py has no per-function metric).",
]

mw = max(len(x) for x in rows + ["Metric"])
cells = [cell(c, r) for r in rows] + [cell(p, r) for r in rows] + ["C++", "Python"]
cw = max(len(x) for x in cells)
def row(a, b, d):
    return f"{a:<{mw}} | {b:<{cw}} | {d:<{cw}}"
txt = [
    "================  AMD SMI Code Coverage  ================",
    f"Date:         {date_str}",
] + comp_txt.split("\n") + [
    "",
    row("Metric", "C++", "Python"),
    row("-" * mw, "-" * cw, "-" * cw),
]
txt += [row(r, cell(c, r), cell(p, r)) for r in rows]
txt += [""] + legend
open(out_txt, "w").write("\n".join(txt) + "\n")

md = [
    "# AMD SMI Code Coverage",
    "",
    f"**Date:** {date_str}  ",
] + comp_md.split("\n") + [
    "",
    "| Metric | C++ | Python |",
    "|--------|-----|--------|",
]
md += [f"| {r} | {cell(c, r)} | {cell(p, r)} |" for r in rows]
md += [
    "",
    "- **N/A** - tier not selected for this run.",
    "- **ERROR** - tier selected but no data (e.g. no `.gcda`; tests did not run).",
    "- **DNE** - metric does not exist (coverage.py has no per-function metric).",
]
open(out_md, "w").write("\n".join(md) + "\n")
PY
}

# sudo-aware removal: a prior root/staged run can leave root-owned artifacts.
rm_rf() { rm -rf "$@" 2>/dev/null || sudo rm -rf "$@"; }

# --clean drops the build/intermediate artifacts (build-cov, build-cov-stage,
# loose .coverage*) but keeps the whole coverage-report/ output; --clean-all also
# removes coverage-report/. Only ever touches coverage-owned paths -- never a
# user BUILD_DIR.
clean() {
    if [[ "$CLEAN_ALL" == "1" ]]; then
        rm_rf "$OUT_DIR"; echo "Removed $OUT_DIR"
    else
        echo "Kept report files under $OUT_DIR"
    fi
    rm_rf "$REPO_ROOT/build-cov" "$STAGE_DIR"
    rm -f "$REPO_ROOT"/.coverage "$REPO_ROOT"/.coverage.* 2>/dev/null || true
    echo "Removed build-cov, $STAGE_DIR, and loose .coverage* files"
}

if [[ "$DO_CLEAN" == "1" ]]; then clean; exit 0; fi
mkdir -p "$OUT_DIR"
if [[ "$DO_BUILD" == "1" ]]; then build_cov; fi
if [[ "$DO_STAGE" == "1" && "$MODE" != "cpp" ]]; then stage_install; fi
case "$MODE" in
    cpp) run_cpp ;;
    py)  run_py ;;
    all) run_cpp || true; run_py ;;
    *)   usage >&2; exit 2 ;;
esac
write_summary_files
print_outputs
echo
cat "$OUT_DIR/amdsmi_cc_summary.txt" 2>/dev/null || true
