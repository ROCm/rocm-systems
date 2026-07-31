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

The build derives its compiler, install prefix, and CTest data path from `ROCM_PATH`
(default `/opt/rocm`). If ROCm is installed elsewhere (e.g. a TheRock or custom build),
`ROCM_PATH` must be set. NOTE: skills run cmake in a NON-interactive shell that does not
source `~/.bashrc`, so set `ROCM_PATH` in `~/.profile` (or export it before launching Claude
Code) — otherwise the build silently falls back to `/opt/rocm`. Step 1 verifies this and
fails fast with guidance if the toolchain is missing.

Conformance test data (Phase 4) is located via the `ROCDECODE_CONFORMANCE_DIR`
environment variable, which defaults to `$HOME/rocDecodeConformance`. It must contain the
per-codec subdirectories `AvcConformance`, `Av1Conformance`, `HevcConformance`, and
`Vp9Conformance`. Codecs whose directory is missing are reported as WARNING and skipped,
so the pipeline still runs without the full data set.

## Step 1 — Verify the ROCm toolchain

Run: `test/validate.sh --check-rocm`

This confirms `ROCM_PATH` points at a usable ROCm install before any build work. If it
fails, fix `ROCM_PATH` (see the note above) and re-run before continuing.

## Step 2 — Configure, clean, rebuild, and install the core library

The `build/` directory is not checked into the repo, so on a fresh checkout it must be
created and configured first. `cmake -B build -DENABLE_EXTENDED_TESTS=ON` is idempotent —
it creates and configures `build/` if missing, and is a cheap no-op if it is already
configured. `-DENABLE_EXTENDED_TESTS=ON` enables the additional FFmpeg-based CTest cases
(the CTest phase covers 6 tests without it, 15 with it); those extra tests are only added
when FFmpeg is found, so the flag is harmless on machines without FFmpeg.

Run these four commands separately:

1. `cmake -B build -DENABLE_EXTENDED_TESTS=ON`
2. `make clean -C build`
3. `make -j"$(nproc)" -C build`
4. `make install -C build`

## Step 3 — Build all sample apps

Run: `test/build_samples.sh`

## Step 4 — Run validation

Run: `test/validate.sh`

Report the final summary from validate.sh to the user.
