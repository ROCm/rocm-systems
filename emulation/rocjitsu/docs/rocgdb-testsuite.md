# Running the official ROCgdb test suite

[`tools/run_rocgdb_official.py`](../tools/run_rocgdb_official.py) drives the
**real ROCgdb** against the **real `gdb.rocm` test suite**, with the inferior
running on the emulated GPU. It is the end-to-end gate for the debug path:
`ctest` covers the emulator's own units, but only this suite exercises
ROCgdb → rocm-dbgapi → interposer → daemon as a whole.

Each `.exp` file runs in its own freshly created, *verified* Mirage daemon
session, and the runner fails a file whose session it cannot account for. See
[Debugging with ROCgdb](rocgdb-debugging.md) for how the pieces fit together.

## 1. What it needs

The suite itself is **not** in rocm-systems — it lives in the ROCgdb repository.
Every path resolves from a flag, an environment variable, or a default, in
increasing order of precedence, so nothing is pinned to one developer's checkout:

| Piece | Default | Environment | Flag |
|---|---|---|---|
| rocm-systems checkout | inferred from the script's location | `ROCM_SYSTEMS_ROOT` | `--root` |
| ROCgdb sources (the suite) | `/tmp/ROCgdb-tests` | `ROCGDB_SUITE` | `--rocgdb-suite` |
| ROCgdb binary | `/tmp/ROCgdb-build/gdb/gdb` | `ROCGDB` | `--gdb` |
| ROCm SDK venv | `<root>/emulation/mirage/.venv-mi350` | `ROCM_SDK_VENV` | `--venv` |
| mirage binary | `<root>/emulation/mirage/target/debug/mirage` | — | `--mirage` |
| `librocjitsu.so` | `<root>/emulation/rocjitsu/build/librocjitsu.so` | — | `--rocjitsu` |

The venv supplies the ROCm SDK wheels: `amdclang++` compiles each test's HIP
program, and `_rocm_sdk_core/lib` and `_rocm_sdk_devel/lib` go on the inferior's
`LD_LIBRARY_PATH`. It is the SDK only — none of the code under test comes from
it — so an existing venv from another checkout is fine to reuse.

## 2. One-time setup

Clone and build ROCgdb out of tree:

```bash
git clone --filter=blob:none -b amd-staging https://github.com/ROCm/ROCgdb.git /tmp/ROCgdb-tests
mkdir -p /tmp/ROCgdb-build && cd /tmp/ROCgdb-build
../ROCgdb-tests/configure --enable-targets=all \
  --disable-binutils --disable-gas --disable-gold --disable-gprof --disable-ld --disable-sim \
  --disable-werror --with-amd-dbgapi=yes --with-python=/usr/bin/python3 \
  --with-system-readline --with-expat \
  --without-guile --without-debuginfod --without-libunwind --without-lzma --without-zstd
make -j"$(nproc)"
```

`--with-amd-dbgapi=yes` is the point of the exercise; the rest trims the build
to gdb. The suite is the same checkout — `gdb/testsuite/gdb.rocm/*.exp`.

## 3. Build the code under test

```bash
cd <root>/emulation/mirage && cargo build --workspace          # -> target/debug/mirage

cd <root>/emulation/rocjitsu
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Build everything rather than naming targets. A partial target list can leave
plugins linked against a stale dispatch ABI, which fails in ways that look like
emulator bugs rather than build staleness.

## 4. Clear stale sessions first

Mandatory before every run, not just after a crash:

```bash
pgrep -fa "mirage host --sessio[n]"          # expect no output
pkill -f "mirage host --sessio[n]"
rm -rf /run/user/$(id -u)/mirage/session/*/
```

Two traps:

- Write the pattern as `--sessio[n]`. Spelled literally, the pattern appears in
  `pkill`'s own command line, so it matches itself, kills the shell, and the run
  never starts (exit 144).
- Remove the session **directories**, not just the processes. Killing the hosts
  leaves their directories behind, and once roughly twenty accumulate the
  `session=None` failures start arriving in consecutive clusters. Those look
  like product bugs and are not.

## 5. Run

```bash
cd <root>
python3 emulation/rocjitsu/tools/run_rocgdb_official.py \
  --venv /path/to/.venv-mi350 \
  --expect-tests 89 \
  --output /tmp/rocgdb-run-$(date +%Y%m%dT%H%M%S)
```

`--expect-tests` fails fast if the suite checkout has drifted from the file
count you expect. Other useful flags:

| Flag | Effect |
|---|---|
| `--tests NAME.exp [...]` | Run only these files |
| `--timeout SECONDS` | Per-file limit (default 600) |
| `--stop-after-failure` | Stop at the first failing file |
| `--rj-log DIR` | Keep one rocjitsu log per test (default discards) |

**Run one suite at a time.** The runner locates the session each test created
under `/run/user/<uid>/mirage/session/`, so two concurrent runs — even a full
suite alongside a single-file run — steal each other's sessions. The failure is
silent and misleading: `rc=0`, no `FAIL` statuses, but `passed=false` because
the session bookkeeping did not close.

A full run takes roughly ten minutes and ends with:

```
output=... completed=89/89 passed=89 failed=0 aggregate={'PASS': 2376, ...}
```

## 6. Reading the results

`<output>/result.json` holds `aggregate_statuses`, `all_passed`, and a per-file
`records` array with each file's statuses, session id, and `passed` flag.
`manifest.json` records the workspace and ROCgdb commits plus SHA-256 hashes of
the `mirage` and `librocjitsu.so` actually exercised, so a run can be tied back
to the binaries it tested.

Compare against a known-good run rather than reading counts in isolation. The
current expected state:

| Status | Count |
|---|---|
| PASS | 2376 |
| UNSUPPORTED | 9 |
| KFAIL | 7 |
| UNTESTED | 6 |
| WARNING | 6 |
| XFAIL | 3 |

with 89/89 files passing and no `FAIL`, `ERROR`, or `UNRESOLVED`.

Two results that are **not** regressions:

- `device-attach.exp` and `lane-pc-vega20.exp` produce no test status at all.
  The runner warns about both on every clean run.
- `lane-info.exp` fails roughly 5–7% of runs, always the same four
  `lane apply` subtests. The test records lane state at a breakpoint and
  compares it against a later async stop, but the wave sits in an `-O0`
  divergent loop whose AGPR-spill windows legitimately leave `EXEC` all ones for
  about a sixth of its instruction slots. gdb reports correctly; the test's
  assumption is what breaks. Hardware hides it because the wave parks on
  `s_sleep` and an uncached load instead.

Before believing any single-file failure, re-run just that file with
`--tests <name>.exp` and compare its `aggregate_statuses` against the baseline.
An unchanged PASS count means no assertion was lost and the failure was
bookkeeping.
