---
name: validate
description: Run full rocDecode build and test pipeline after code changes
allowed-tools:
  - Bash(cmake *)
  - Bash(make *)
  - Bash(test/validate.sh *)
  - Bash(test/build_samples.sh *)
  - Bash(nproc)
---

Run the full rocDecode validation pipeline. Execute these steps in order, stopping only
if a step fails. Run from the rocDecode project root (the directory containing this
repo's `build/`, `samples/`, and `test/` directories).

IMPORTANT: Run each command as a SEPARATE Bash tool call. Do not chain commands with && or |.

Conformance test data (Phase 3) is located via the `ROCDECODE_CONFORMANCE_DIR`
environment variable, which defaults to `$HOME/rocDecodeConformance`. It must contain the
per-codec subdirectories `AvcConformance`, `Av1Conformance`, `HevcConformance`, and
`Vp9Conformance`. Codecs whose directory is missing are reported as WARNING and skipped,
so the pipeline still runs without the full data set.

## Step 1 — Configure, clean, rebuild, and install the core library

The `build/` directory is not checked into the repo, so on a fresh checkout it must be
created and configured first. `cmake -B build -DENABLE_EXTENDED_TESTS=ON` is idempotent —
it creates and configures `build/` if missing, and is a cheap no-op if it is already
configured. `-DENABLE_EXTENDED_TESTS=ON` enables the additional FFmpeg-based CTest cases
(the CTest phase covers 6 tests without it, 15 with it); those extra tests are only added
when FFmpeg is found, so the flag is harmless on machines without FFmpeg.

Run these four commands separately:

1. `cmake -B build -DENABLE_EXTENDED_TESTS=ON`
2. `make clean -C build`
3. `make -j32 -C build`
4. `make install -C build`

## Step 2 — Build all sample apps

Run: `test/build_samples.sh`

## Step 3 — Run validation

Run: `test/validate.sh`

Report the final summary from validate.sh to the user.
