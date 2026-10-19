---
name: validate
description: Run full rocDecode build and test pipeline after code changes
allowed-tools:
  - Bash(cmake *)
  - Bash(make *)
  - Bash(ctest *)
  - Bash(test/validate.sh *)
  - Bash(test/build_samples.sh *)
  - Bash(nproc)
  - Bash(cmd *)
---

Run the full rocDecode validation pipeline. Execute these steps in order, stopping only
if a step fails. Run from the rocDecode project root (the directory containing this
repo's `build/`, `samples/`, and `test/` directories).

IMPORTANT: Run each command as a SEPARATE Bash tool call. Do not chain commands with && or |.

First determine the OS and follow the matching section below. Under Git Bash on Windows,
`uname -s` returns `MINGW64_NT-...`; on Linux it returns `Linux`. The `test/*.sh` scripts
are cross-platform and branch internally, so the commands are nearly identical — the
differences are the environment setup and the build generator.

---

## Linux

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

### Step 1 — Verify the ROCm toolchain

Run: `test/validate.sh --check-rocm`

This confirms `ROCM_PATH` points at a usable ROCm install (the compiler exists) and that
the install prefix is writable before any build work — `make install` installs into
`ROCM_PATH`, so a read-only prefix like a system `/opt/rocm` would need sudo, which this
skill does not use. If it fails, fix `ROCM_PATH` (see the note above; point it at a
user-writable install) and re-run before continuing. Do not retry without changing
`ROCM_PATH` — if it cannot be resolved, stop and ask the developer.

### Step 2 — Configure, clean, rebuild, and install the core library

The `build/` directory is not checked into the repo, so on a fresh checkout it must be
created and configured first. `cmake -B build -DENABLE_EXTENDED_TESTS=ON` is idempotent —
it creates and configures `build/` if missing, and is a cheap no-op if it is already
configured. `-DENABLE_EXTENDED_TESTS=ON` enables the additional FFmpeg-based CTest cases
(the CTest phase covers 6 tests without it, 15 with it); those extra tests are only added
when FFmpeg is found, so the flag is harmless on machines without FFmpeg.

Run these four commands separately:

1. `cmake -B build -DENABLE_EXTENDED_TESTS=ON`
2. `make clean -C build`
3. `make -j -C build`
4. `make install -C build`

(`make -j` lets make choose parallelism without a `$(nproc)` command substitution,
which would otherwise trigger a permission prompt on every run.)

### Step 3 — Build all sample apps

Run: `test/build_samples.sh`

### Step 4 — Run validation

Run: `test/validate.sh`

Report the final summary from validate.sh to the user.

---

## Windows

On Windows the shell is Git Bash (where the `test/*.sh` scripts run), but the build
toolchain is MSVC: cmake auto-selects the Visual Studio multi-config generator, so build,
CTest, and install all take an explicit `--config Release` / `-C Release`. There is no
`make`, `nproc` is optional, and the core library is built with `cmake --build` /
`cmake --install` rather than `make` / `make install`.

### Environment

There is NO top-level build system; rocDecode is built standalone. Before running the
pipeline, the developer's shell must have:

- `ROCM_PATH` pointing at the rocDecode install prefix (default `C:\opt\rocm`). This is
  the tree the samples resolve rocDecode from and where the core library installs.
- The install's runtime DLL directories on `PATH`: `%ROCM_PATH%\bin` (holds
  `rocdecode.dll`) and `%ROCM_PATH%\lib\rocm_sysdeps\bin` (holds `rocm_sysdeps_va*.dll`
  and `vaon12_drv_video.dll`). The Windows loader ignores `ROCM_PATH` for DLL search, so
  these PATH entries are what make the decode stack loadable at runtime.
- `LIBVA_DRIVERS_PATH` set to `%ROCM_PATH%\lib\rocm_sysdeps\bin` so the VA-API layer finds
  the VAOn12 driver.
- FFmpeg's `bin` on `PATH` at BOTH configure and runtime. The samples' `FindFFmpeg.cmake`
  AUTO-DISCOVERS FFmpeg from PATH (it probes PATH dirs for `avcodec*`/`avformat*`/`avutil*`
  DLLs and derives the root), so do NOT set `FFMPEG_ROOT` when ffmpeg's bin is already on
  PATH — doing so can fight the auto-discovery.

A per-developer batch file is the usual way to set all of this up (it typically calls
`vcvars64`, fixes Windows SDK paths, sets `ROCM_PATH`, prepends the two bin dirs to PATH,
and sets `LIBVA_DRIVERS_PATH`). Such a batch is machine-specific and NOT part of the repo —
reference your own, or set the variables above by hand. As a convenience, `test/validate.sh`
will itself prepend `%ROCM_PATH%\bin` and `%ROCM_PATH%\lib\rocm_sysdeps\bin` to PATH and set
`LIBVA_DRIVERS_PATH` when only `ROCM_PATH` is provided.

Because a sourced cmd environment only persists within a single `cmd //c "..."` call,
chain Windows build/install commands that need the MSVC env as:
`cmd //c "call <your_env.bat> && <command>"`. The `test/*.sh` entry points (build_samples.sh,
validate.sh) can be run directly from Git Bash once `ROCM_PATH` (and the PATH/FFmpeg setup
above) are in place; cmake finds MSVC via the Visual Studio generator without vcvars.

Conformance test data is located via `ROCDECODE_CONFORMANCE_DIR` (same variable as Linux),
whose per-codec subdirectories are `AvcConformance`, `Av1Conformance`, `HevcConformance`,
and `Vp9Conformance`. Each codec subdir must itself be the parent of `Streams\` and `MD5\`.
Point `ROCDECODE_CONFORMANCE_DIR` at the parent holding those four folders; any codec whose
directory is absent is reported as WARNING and skipped, so a partial data set still runs.

### Step 1 — Verify the install prefix

Run: `test/validate.sh --check-rocm`

On Windows this confirms `ROCM_PATH` exists (the compiler is MSVC, so there is no
`amdclang++` check, and the writable check is skipped — `cmake --install` does not need
sudo). If it fails, fix `ROCM_PATH` and re-run before continuing.

### Step 2 — Configure, rebuild, and install the core library

Multi-config cmake; run each separately (wrap in `cmd //c "call <env.bat> && ..."` if your
MSVC/ROCm env is set up by a batch rather than already active in this shell):

1. `cmake -B build -DENABLE_EXTENDED_TESTS=ON`
2. `cmake --build build --config Release --parallel`
3. `cmake --install build --config Release`

There is no `make`, no `-j`, and no `nproc` requirement — parallelism comes from
`cmake --build ... --parallel`.

CRITICAL stale-build-dir pitfall: cmake configure is a no-op when a `build/` cache already
exists, so a STALE `build/` dir silently keeps building against whatever `ROCM_PATH` it was
first configured with. A stale repo-level `build/` combined with CMake's Windows user
package registry (`HKCU\Software\Kitware\CMake\Packages`) can also make
`find_package(rocdecode-host)` in the samples resolve to a nonexistent path and fail
configure (rocdecode-host is intentionally not shipped in the Windows tarball). When
validating a tarball or switching install trees, do a CLEAN build: delete the stale
directories first. On Windows the Bash `rm -rf` may be blocked by permissions; use
`cmd //c "rmdir /s /q <path>"`. After a clean configure, verify a sample's cache has the
intended tree, e.g. `grep ROCM_PATH samples\videoDecode\build\CMakeCache.txt` should show
`ROCM_PATH:PATH=` matching `ROCM_PATH`.

### Step 3 — Build all sample apps

Run: `test/build_samples.sh`

This builds the full sample set (same as Linux) plus `videoDecodeCaps`, each with
`cmake --build ... --config Release`. Sample exes land in
`samples\<name>\build\Release\<name>.exe` (lowercased, e.g. `videodecode.exe`,
`videodecodecaps.exe`).

### Step 4 — Sanity-check hardware decode capabilities

Run: `samples\videoDecodeCaps\build\Release\videodecodecaps.exe`

This no-arg exe prints the GPU's hardware decode capability table and exits 0. It is a
cheap confirmation that the decode stack loads (rocdecode.dll + sysdeps + VAOn12 driver are
all resolvable) before the heavy conformance sweep. If it fails to load DLLs, revisit the
PATH / `LIBVA_DRIVERS_PATH` setup in the Environment section above.

### Step 5 — Run validation

Run: `test/validate.sh`

On Windows this drives CTest with `ctest -C Release` and launches the conformance script
with `python` (not `python3`, which is usually the inert Store shim). The conformance
script needs `pandas` installed. Report the final summary from validate.sh to the user.
