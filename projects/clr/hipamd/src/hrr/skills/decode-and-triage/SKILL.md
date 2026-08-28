---
name: decode-and-triage
description: >-
  Decode and triage HRR capture archives with full GPU replay by default (Linux).
  Builds hrr-playback when missing. On Windows: full native GPU replay via
  triage_archive.ps1 + ensure_playback.ps1 (PowerShell); Docker replay requires
  Linux or WSL2. Never edits source. Print finding summary in the chat reply.
inputs:
  - HRR archive path (`capture.hrr/pid-*` directory, or capture root to resolve)
  - Optional Docker image from capture (`HRR_DOCKER_IMAGE`)
  - Optional GPU ordinal (`GPU`)
outputs:
  - Finding summary markdown in chat (outcome, fault class, kernel, fault address, event seq, archive stats)
  - Optional finding file under `HRR_TRIAGE_WORKDIR` (script default)
---

# HRR Decode & Triage

**Platform (Linux):** AMD GPU host. Full skill workflow via `triage_archive.sh`:
native GPU replay (auto-builds `hrr-playback` when missing), optional Docker replay,
manifest preflight, and finding output.

**Platform (Windows):** Full native GPU replay via `triage_archive.ps1` +
`ensure_playback.ps1` (PowerShell). Docker replay is **not** supported on Windows
(ROCm containers are Linux-only) — see [Windows support](#windows-support) below.

**Run from:** any working directory. On Linux, invoke by absolute path:

```bash
<rocm-systems>/projects/clr/hipamd/src/hrr/skills/decode-and-triage/scripts/triage_archive.sh \
  --archive <path-to>/capture.hrr/pid-<pid>
```

On Windows, invoke by absolute path:

```powershell
<rocm-systems>\projects\clr\hipamd\src\hrr\skills\decode-and-triage\scripts\triage_archive.ps1 `
  --archive <path-to>\capture.hrr\pid-<pid>
```

Scripts locate the colocated CLR tree from their install path; you do not need to
`cd` into `rocm-systems` first.

Print the **finding markdown** that `triage_archive.sh` / `triage_archive.ps1` writes to stdout.

### Windows support

| Capability | Windows today |
|------------|---------------|
| Full skill via `triage_archive.ps1` | **Yes** — PowerShell port with native GPU replay, manifest preflight, and `.finding.md` output |
| Locate / build `hrr-playback.exe` | **Yes** — `ensure_playback.ps1` searches `HRR_PLAYBACK`, `CLR_BUILD`, `HIP_PATH\bin`, `PATH`; `--build` flag triggers CMake/Ninja build |
| Metadata-only triage (`--no-replay`) | **Yes** — `triage_archive.ps1 --no-replay` or `python analyze_replay_finding.py --archive <pid-dir>` |
| Native GPU replay | **Yes** — `triage_archive.ps1` launches `hrr-playback.exe` via `Start-Process` with stdout/stderr capture |
| Auto-build `hrr-playback.exe` | **Yes** — `ensure_playback.ps1 --build`; set `CLR_BUILD` to existing tree |
| Docker replay (`--replay docker`) | **No** — requires a Linux host with Docker and GPU passthrough; use WSL2 or a Linux SSH host |
| Container captures (e.g. vLLM image) | Replay on **Linux** (native or Docker); on Windows use `triage_archive.ps1` native replay if the Windows HIP stack matches |

On Windows, prefer `triage_archive.ps1` for full GPU replay. Fall back to
`--no-replay` (metadata-only) when no GPU is available or `hrr-playback.exe` is
absent and cannot be built.

#### W1 — Locate or build `hrr-playback.exe`

```powershell
# From the skill scripts directory:
.\ensure_playback.ps1          # find existing binary (checks HRR_PLAYBACK, CLR_BUILD, HIP_PATH\bin, PATH)
.\ensure_playback.ps1 --build  # also build from source if not found
```

Set `$env:HRR_PLAYBACK` to the path printed, or let `triage_archive.ps1` call it automatically.

#### W2 — Run triage (native GPU replay)

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
| `CLR_BUILD` | Existing build tree containing `hrr-playback.exe` |
| `GPU` | GPU ordinal for replay (default: `0`) |
| `HRR_TRIAGE_WORKDIR` | Output dir for findings + logs |
| `HRR_CONTINUE=1` | Proceed past HIP/comgr version mismatch |
| `HRR_SKIP_COMPAT=1` | Skip manifest preflight entirely |

#### W3 — Docker replay on Windows

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

#### W4 — Print finding summary

The `.finding.md` written by `triage_archive.ps1` is identical in schema to the Linux version.
Print or paste it directly in the chat reply using the same template in Step 5 below.

#### Copying scripts to a remote machine (no Cursor installed)

When the HIP app runs on a Remote Desktop machine without Cursor, copy the four
standalone script files and run them directly:

```
projects\clr\hipamd\src\hrr\skills\decode-and-triage\scripts\
  triage_archive.ps1
  ensure_playback.ps1
  analyze_replay_finding.py
  check_replay_compat.py
```

Requirements on the remote: PowerShell 5+, Python 3.8+, and `hrr-playback.exe`
(from HIP SDK `bin\` or built via `ensure_playback.ps1 --build`).

Bring the resulting `.finding.md` back and paste it into the chat for AI analysis.

---

## Example prompt

```
Use the decode-and-triage skill. Triage with docker replay:
<path-to>/capture.hrr/pid-<pid>

Docker image used for capture:
<registry>/<image>:<tag>
```

## Workflow

**Replay mode (`auto` default):** picks docker when `HRR_DOCKER_IMAGE` is set, else
native. Use `--no-replay` for metadata-only (no GPU).

1. Resolve `capture.hrr/pid-*` (largest `events.bin` if user gives root only).
2. **Preflight** when `manifest.json` has `metadata`: block if capture used more
   GPUs than replay exposes or requested `GPU` is missing. HIP/comgr mismatch
   prompts for confirmation (exit 2 in agents — ask user, then `HRR_CONTINUE=1`).
   Docker preflight uses the **image HRR stack** by default; `HRR_DOCKER_MOUNT_CLR=1`
   overlays host `CLR_BUILD`/`HRR_PLAYBACK` for WIP dev work. Legacy archives skip
   preflight. Overrides: `HRR_SKIP_COMPAT=1`, `HRR_STRICT_VERSION=1`, `HRR_STRICT_ARCH=1`.
3. **Native (Linux):** `triage_archive.sh --archive <pid-dir>` (builds/finds host `hrr-playback`).
4. **Native (Windows):** `triage_archive.ps1 --archive <pid-dir>` (finds/builds `hrr-playback.exe`).
5. **Docker (Linux only):** `export HRR_DOCKER_IMAGE='<capture image>'` then
   `triage_archive.sh --archive <pid-dir> --replay docker`. Image `hrr-playback` by
   default; dev overlay via `HRR_DOCKER_MOUNT_CLR=1`. Optional `GPU=1` when GPU 0 busy.
6. Print the finding summary from triage output. Add a one-line capture explainer if helpful.

### Finding outcomes

| Outcome | Meaning |
|---------|---------|
| `PASS` | Replay completed; D2H checks passed |
| `MAF` | GPU memory access fault during replay |
| `FAIL` | Replay finished but validation failed (e.g. D2H mismatch) |
| `ABORT` | Replay stopped early (fatal API, version mismatch, user abort) |
| `UNKNOWN` | Insufficient signal to classify (e.g. metadata-only triage, missing log) |

When outcome is `UNKNOWN` or fault class is `unknown`, say so explicitly in the
summary — do not invent a fault type.

### Finding summary template (Step 5 / W4)

Print a structured summary directly in the chat reply:

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

## Hard constraints

- **Windows hosts:** use `triage_archive.ps1` for full GPU triage. Do not run
  `triage_archive.sh` for full GPU replay unless the user explicitly has bash + a
  working Linux-style GPU path. Prefer `triage_archive.ps1 --no-replay` (or
  `analyze_replay_finding.py --archive <pid-dir>`) when no GPU is available.
- **No source edits.** One `ensure_playback.sh --build` / `ensure_playback.ps1 --build`
  attempt max; on failure stop (or use `--no-replay`).
- Do not ask about Docker until native replay fails **unless** user named the capture image.
- Do not ask user to run cmake/ninja — scripts handle it.
- Do not read stale `*.finding.md` / `hrr-replay-*.log` unless user points at them.
- Do **not** modify the archive directory or its contents.
- Do **not** run `hrr-playback --repair` unless the user explicitly asks.

## If build or replay fails

| Situation | Action |
|-----------|--------|
| `--build` fails (`rocdevice.cpp` etc.) | Set `CLR_BUILD` to existing `projects/clr/build-hrr*`; re-run. |
| Native replay fails (Linux) | Docker replay with capture image (Linux host). |
| Native replay fails (Windows) | Check `HRR_PLAYBACK`, `HIP_PATH\bin` on `PATH`; use `--no-replay` as fallback. |
| Windows host, no GPU | Metadata-only via `triage_archive.ps1 --no-replay` or `analyze_replay_finding.py --archive`. |
| Docker `hrr-playback` not in image | `HRR_DOCKER_MOUNT_CLR=1` with `CLR_BUILD` / `HRR_PLAYBACK`. |
| Docker HIP symbol/version mismatch | Dev overlay — `HRR_DOCKER_MOUNT_CLR=1`; `replay_docker.sh` puts mounted CLR first on `LD_LIBRARY_PATH`. |
| Preflight exit 2 | Ask user to confirm; re-run with `HRR_CONTINUE=1`. |
| HIP/comgr version mismatch warning | Set `HRR_CONTINUE=1` to proceed (non-fatal for same-arch captures). |

## Quick reference — hrr-playback flags

| Flag | Purpose |
|------|---------|
| `--info` | Archive summary (events, kernels, blobs, complete flag), no GPU |
| `--events` | Full event log with `--info` (includes Kernel Call Counts) |
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
