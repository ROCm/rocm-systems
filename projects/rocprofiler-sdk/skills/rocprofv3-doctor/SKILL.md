---
name: rocprofv3-doctor
description: >-
  Diagnoses why rocprofv3 or ROCm GPU profiling does not work on a machine, by running
  `rocprofv3 --doctor` and turning its report into ordered, root-cause fixes. Use when rocprofv3
  fails, crashes, hangs, finds no GPU, or will not start; when an error such as "cannot open
  shared object file", "undefined symbol", "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
  "hipErrorNoDevice", "/dev/kfd: Permission denied", or "could not be locked for profiling"
  appears; when the user asks whether a ROCm installation, container, or TheRock install is
  ready for profiling; or when the user asks to run rocprofv3 --doctor.
  Not for profiling a working application, choosing counters, or reading profiling results:
  use the rocprofv3 skill for those.
---

# rocprofv3-doctor

`rocprofv3 --doctor` inspects the machine and the ROCprofiler-SDK installation and reports
what would stop `rocprofv3` from working. Run it, find the root causes in its report, and
give the user an ordered list of fixes.

The doctor makes no persistent changes, and neither does this skill: every fix is shown to
the user, never run on their behalf. A default run takes about a second. It loads the ROCm
libraries in child processes and initializes the GPU runtime once (for counter
enumeration).

## Prerequisites

- Linux, with a ROCm installation whose `rocprofv3` supports `--doctor`. ROCm releases
  that predate it can be inspected from a rocm-systems source checkout (see Step 1).
- Run it as the user who runs `rocprofv3`, never with `sudo`: as root it reports on root's
  access and environment, not the user's (it warns when that happens).
- Python 3 for the summary script (standard library only).
- No GPU is required to run it: a missing or inaccessible GPU is one of the things it
  reports.
- Run the script by absolute path: `python3 <this-skill-dir>/scripts/summarize_report.py`.

## Step 1: Find the tool

Use the first of these that runs (it prints version information; a source checkout
shows unfilled `@...@` placeholders, which is fine). A `rocprofv3` that predates `--doctor`
rejects the option ("unrecognized arguments"); move on to the next.

1. `rocprofv3 --doctor --version`
2. `<prefix>/bin/rocprofv3 --doctor --version`, for each ROCm prefix: `/opt/rocm`,
   `/opt/rocm/core-*`, `$ROCM_PATH`, `$ROCM_HOME`, or, in a TheRock Python virtualenv,
   `$(rocm-sdk path --root)`
3. From a rocm-systems checkout:
   `python3 <checkout>/projects/rocprofiler-sdk/source/bin/rocprofv3.py --doctor`, always
   with `--rocm-root <prefix>` so it inspects the installation, not the checkout

Use that same command in place of `rocprofv3 --doctor` in the steps below. If none works,
tell the user `rocprofv3 --doctor` is not available for their ROCm, and stop.

## Step 2: Run it and summarize

```bash
rocprofv3 --doctor --format json --output rocprofv3-doctor.json > /dev/null
echo "exit=$?"
python3 <this-skill-dir>/scripts/summarize_report.py rocprofv3-doctor.json
```

Add `--rocm-root <prefix>` when the user names an installation or has several.

| Exit code | Meaning |
| --- | --- |
| `0` | No check failed; warnings may remain. |
| `1` | At least one check failed. This is the normal "found problems" result, so summarize it. |
| `2` | The tool did not run, an `--only` pattern matched nothing, or a check broke and nothing failed. Read stderr: it is not a finding about the system. |

First, confirm the summary's ROCm root is the installation the user actually runs
`rocprofv3` from. Every other finding depends on it. If it is not, re-run with
`--rocm-root`.

## Step 3: Read the summary

The script has already done the mechanical part:
- It orders root causes as failures first, then warnings.
- It folds every skipped check into the root cause that blocked it ("Blocks:"), so a
  dependent check is never reported as a separate problem.
- It separates doctor errors (bugs in a check) from findings about the system.
- It lists what the run did not verify.

What it leaves to you:

- **Respect diagnosis confidence.**
  - `likely`: facts on the machine corroborate the cause. Present its fix.
  - `possible`: several causes fit. Say so, and propose the discriminating step from
    the fix text, not a definite fix.
- **Group causes that share a mechanism.** For example, a stale `LD_LIBRARY_PATH` can
  appear in both `install.no-mixed-rocm` and an `undefined symbol` in
  `runtime.libraries-load`.
- **Weigh warnings against the user's goal.** A missing kokkosp or rocattach library
  matters only for Kokkos tracing or process attach.

## Step 4: Present the fixes

Give an ordered list, root cause first. For each item give:
- the problem in one sentence, quoting the doctor's evidence
- the fix commands from the report
- whether it needs `sudo`, a new login session (group changes), or a container restart
  (device passthrough)

Rules:

- **Never run fixes yourself.** Never run `sudo`, `usermod`, `modprobe`, `sysctl`,
  `setcap`, a package manager, or `pip install` on the user's behalf. Show the command
  and let the user run it.
- **Add no fixes beyond the report's.** In particular:
  - Never suggest changing `kernel.perf_event_paranoid`; rocprofiler-sdk does not use
    perf events.
  - Never suggest `setcap` on `rocprofv3`. It is a Python script, so Linux ignores file
    capabilities on it. Locking the GPU for profiling needs CAP_PERFMON, for example
    `sudo -E rocprofv3 ...`.
- **Match the install kind** shown in the summary:
  - `python-wheel`: fix with `pip`, in that virtualenv
  - `therock-package`: `amdrocm-*` packages
  - `system-package`: `rocprofiler-sdk` / `rocm-*` packages
  - `standalone`: re-extract or rebuild that tree

## Step 5: Verify

After the user applies a fix, re-run only what it should change:

```bash
rocprofv3 --doctor --only <check-id-or-group>
```

- `--only` also runs the selected checks' prerequisites.
- List check ids with `rocprofv3 --doctor --list-checks`.
- Group-membership fixes take effect only in a new login session.

## Going further (ask first)

- **Smoke checks:** `rocprofv3 --doctor --run-smoke-test` (or `--only smoke`) runs
  `rocprofv3-avail info` and `rocprofv3` itself, and can take 30 seconds or more. A
  passing launcher check shows that rocprofv3 starts and injects its tool. It does not
  show that kernel tracing works, because the test program launches no GPU work.
- **Bug report:** `rocprofv3 --doctor --verbose --output rocprofv3-doctor-report.txt`.
- **An error from the user's own application:** the doctor cannot reproduce it. Report
  its findings as context, and say that the error itself was not reproduced. Once the
  setup is healthy, profile with the rocprofv3 skill.

## Known benign findings

These come from the environment, not from a broken install:

- `tools.rocprofv3-avail` and `install.avail-library` fail on ROCm releases that predate
  `rocprofv3-avail` (for example 6.2).
- `python.rocprofv3-package` warns when the doctor runs from a source checkout.
- `counters.no-profiler-lock` matches process command lines that mention profiler names,
  so a shell whose working directory is a rocprofiler checkout can trigger it. Check the
  listed processes before telling the user to stop anything.
