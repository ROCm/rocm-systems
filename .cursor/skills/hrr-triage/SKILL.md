---
name: hrr-triage
description: >-
  Decode and triage HRR capture archives (.hrr directories) with full GPU replay by default.
  Builds hrr-playback when the binary is missing. Never edits source files.
  Prints a structured finding summary in the chat reply.
  Use when the user mentions .hrr archives, HIP Record & Replay, hrr-playback, replay failures,
  D2H validation, capture truncation, or asks to inspect / replay / triage an HRR capture.
  Works on Linux (bash) and Windows (PowerShell + HIP SDK for Windows).
disable-model-invocation: true
---

# HRR Triage

## Platform detection

Detect the OS first and follow the matching path below.

```python
import sys
is_windows = sys.platform == "win32"
```

**Windows**: use `triage_archive.ps1` + `ensure_playback.ps1` from
`hipamd/src/hrr/skills/decode-and-triage/scripts/`.
**Linux**: use the bash scripts (`triage_archive.sh`, `ensure_playback.sh`) from the same dir,
or follow Steps 1–5 below using the shell directly.

Decode and triage an HRR capture archive and print a structured finding summary.
Never edit any source file. Never modify the archive.

---

## Windows quick path (use this when `sys.platform == "win32"`)

### W1 — Locate or build `hrr-playback.exe`

```powershell
# From the skill scripts directory:
.\ensure_playback.ps1          # find existing binary (checks HRR_PLAYBACK, CLR_BUILD, HIP_PATH\bin, PATH)
.\ensure_playback.ps1 --build  # also build from source if not found
```

Set `$env:HRR_PLAYBACK` to the path printed by the above, or let `triage_archive.ps1` call it automatically.

### W2 — Run triage (native GPU replay)

```powershell
# Metadata only (no GPU required):
.\triage_archive.ps1 --archive C:\path\to\capture.hrr\pid-138 --no-replay

# Full native replay:
.\triage_archive.ps1 --archive C:\path\to\capture.hrr\pid-138

# Override GPU ordinal:
$env:GPU = "1"
.\triage_archive.ps1 --archive C:\path\to\capture.hrr\pid-138
```

Environment variables on Windows:

| Variable | Purpose |
|---|---|
| `HRR_PLAYBACK` | Explicit `hrr-playback.exe` path |
| `HIP_PATH` or `ROCM_PATH` | HIP SDK root (default: `C:\Program Files\AMD\ROCm\6.2`) |
| `CLR_BUILD` | Existing build tree with `hrr-playback.exe` |
| `GPU` | GPU ordinal for replay (default: `0`) |
| `HRR_TRIAGE_WORKDIR` | Output dir for findings + logs |
| `HRR_CONTINUE=1` | Proceed past HIP/comgr version mismatch |
| `HRR_SKIP_COMPAT=1` | Skip manifest preflight entirely |

### W3 — Docker replay on Windows

Docker GPU replay (`--device=/dev/kfd`) is **Linux-only**. On Windows:

- Use WSL2 (Ubuntu 22.04+) and run `replay_docker.sh` inside WSL2:

  ```bash
  # In WSL2 shell:
  export HRR_DOCKER_IMAGE="rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1"
  bash scripts/replay_docker.sh \
    --archive /mnt/c/path/to/capture.hrr/pid-138 \
    --log /tmp/hrr.log
  ```

- Or use Cursor Remote SSH → Linux ROCm server and run the Linux skill there.

### W4 — Print finding summary

The `.finding.md` written by `triage_archive.ps1` is identical in schema to the Linux version.
Print or paste it directly in the chat reply using the same template in Step 5 below.

---

## Linux path (Steps 1–5)

## Step 1 — Locate hrr-playback

Search for the binary in the build tree:

```bash
find build -name hrr-playback -type f 2>/dev/null | head -5
```

If not found, build it (Step 2). Otherwise record the path and skip to Step 3.

## Step 2 — Build hrr-playback (only when missing)

Configure and build only the `hrr-playback` target. Use the existing build directory
if one is present, otherwise configure from scratch.

```bash
# Check for an existing build dir with a cache
BUILD_DIR=$(ls -d build* 2>/dev/null | head -1)

if [ -n "$BUILD_DIR" ] && [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  ninja -C "$BUILD_DIR" hrr-playback -j"$(nproc)"
else
  # No existing tree — configure once (adjust ROCM_PATH / HIP_COMMON_DIR as needed)
  cmake -S projects/clr -B build -GNinja \
    -DHIP_COMMON_DIR="${ROCM_PATH:-/opt/rocm}/include" \
    -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
    -DCMAKE_BUILD_TYPE=Release
  ninja -C build hrr-playback -j"$(nproc)"
fi
```

After building, locate the binary:

```bash
HRR_PB=$(find build -name hrr-playback -type f | head -1)
```

## Step 3 — Decode archive info (no GPU required)

```bash
"$HRR_PB" <archive.hrr> --info --events 2>&1 | tee /tmp/hrr_info.txt
```

Parse the output for:
- **Archive complete** (`complete: yes` / `NO`) — indicates clean vs. crash-truncated capture
- **Recovered events** — non-zero means the tail was torn; note the count
- **Event count**, **blob count** per process sub-archive
- **Processes** listed in the root manifest (multi-process captures)
- Any `writer_state.json` present (mid-capture remnant)

## Step 4 — Full GPU replay (default)

Run the full replay with D2H validation and timing:

```bash
"$HRR_PB" <archive.hrr> --timing 2>&1 | tee /tmp/hrr_replay.txt
```

Capture the exit code: `0` = all checks pass or no D2H blobs; `1` = D2H failure(s);
`2` = replay diverged (accumulated failures crossed abort threshold).

Parse the replay output for:
- **D2H pass / pass-tol / fail** counts
- **Kernel time**, **graph time**, combined total (from `--timing`)
- Any `MISSING` handle translations (null pointer warnings)
- Any `ERROR_STUB` or `not supported` messages (explicit graph, texture, symbol APIs)
- Any GPU fault or `hipError` returned from a launch
- Divergence abort messages

## Step 5 — Print finding summary

Print a structured summary directly in the chat reply. Use this template:

```
## HRR Triage — <archive name>

### Archive health
| Field           | Value |
|-----------------|-------|
| Complete        | yes / NO |
| Recovered tail  | <n> events (0 = clean) |
| Processes       | <pids> |
| Events          | <n> |
| Blobs           | <n> |

### Replay result
| Field           | Value |
|-----------------|-------|
| Exit code       | 0 / 1 / 2 |
| D2H pass        | <n> exact + <n> within-tol |
| D2H fail        | <n> |
| Kernel GPU time | <ms> ms |
| Graph GPU time  | <ms> ms |
| Total GPU time  | <ms> ms |

### Findings
- **F1** <finding> — <one-line explanation>
- **F2** ...

### Recommended next steps
1. ...
```

Classify each finding using these labels:
- **CRASH-TRUNCATED** — archive missing clean trailer; replay may be partial
- **D2H-FAIL** — output mismatch; note which blobs and max error if logged
- **MISSING-HANDLE** — null-translated pointer; likely `hipHostAlloc`/texture/IPC gap
- **NOOP-API** — explicit graph, texture object, or symbol API replayed as no-op
- **BUILD-NEEDED** — `hrr-playback` was absent and had to be built
- **MULTI-PROCESS** — root archive spans multiple pid sub-archives; triage each separately
- **CLEAN** — no issues found

## Key constraints

- Do **not** edit any source file (`.cpp`, `.h`, `.py`, `CMakeLists.txt`, `DESIGN.md`).
- Do **not** modify the archive directory or its contents.
- Do **not** run `hrr-playback --repair` unless the user explicitly asks.
- If the build step fails, report the CMake/Ninja error verbatim and stop; do not attempt source edits.
- If no GPU is available, run `--info --events` only (Steps 3) and note GPU replay was skipped.

## Quick reference — hrr-playback flags

| Flag | Purpose |
|------|---------|
| `--info` | Archive summary, no GPU |
| `--events` | Full event log with `--info` |
| `--timing` | Kernel + graph GPU time |
| `--verbose` | Per-event trace |
| `--sync-after-launch` | Surface GPU errors per kernel |
| `--kernel-filter STR` | Replay one kernel only |
| `--multi-thread` | One thread per captured thread |
| `--repair` | Rewrite crash-truncated archive (user must request) |

## Known limitations to flag automatically

If these patterns appear in replay output, add the corresponding finding:

| Pattern in output | Finding label |
|------------------|---------------|
| `complete: NO` or missing EOF | CRASH-TRUNCATED |
| `D2H FAIL` | D2H-FAIL |
| `(MISSING!)` or `nullptr` | MISSING-HANDLE |
| `explicit (node-API) graph construction is NOT supported` | NOOP-API |
| `hipCreateTextureObject` no-op | NOOP-API |
| `hipMemcpyToSymbol` / `hipGetSymbolAddress` | NOOP-API |
| multiple `pid-` directories in root manifest | MULTI-PROCESS |
