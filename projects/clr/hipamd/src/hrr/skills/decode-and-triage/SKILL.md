---
name: decode-and-triage
description: >-
  Decode and triage HRR capture archives with full GPU replay by default (Linux).
  Builds hrr-playback when missing. On Windows: metadata-only triage via Python, or
  manual hrr-playback.exe replay — bash orchestration is Linux-first; no Docker on
  Windows. Never edits source. Print finding summary in the chat reply.
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

**Platform (Windows):** `hrr-playback.exe` and native GPU replay exist, but the skill
orchestration scripts are **bash/Linux-oriented today** — see [Windows support](#windows-support)
below. Docker replay is **not** supported on Windows (ROCm containers are Linux-only).

**Run from:** any working directory. On Linux, invoke by absolute path:

```bash
<rocm-systems>/projects/clr/hipamd/src/hrr/skills/decode-and-triage/scripts/triage_archive.sh \
  --archive <path-to>/capture.hrr/pid-<pid>
```

Scripts locate the colocated CLR tree from their install path; you do not need to
`cd` into `rocm-systems` first.

Print the **finding markdown** that `triage_archive.sh` writes to stdout.

### Windows support

| Capability | Windows today |
|------------|---------------|
| Full skill via `triage_archive.sh` | **Unreliable** — scripts require bash (Git Bash or WSL) and use Linux GPU/lib checks (`/dev/kfd`, `LD_LIBRARY_PATH`) |
| Metadata-only triage (`--no-replay`) | **Yes** — `python3 analyze_replay_finding.py --archive <pid-dir>` |
| Native GPU replay (manual) | **Yes** — run prebuilt `hrr-playback.exe` yourself; prepend ROCm `bin` to `PATH` for HIP DLLs; pass `--log` to the analyzer |
| Auto-build `hrr-playback` (`ensure_playback.sh --build`) | **No** — Linux/Ninja only; set `HRR_PLAYBACK` to an existing Windows build |
| Docker replay (`--replay docker`) | **No** — requires a Linux host with Docker and GPU passthrough |
| Container captures (e.g. vLLM image) | Replay on **Linux** (native or Docker); on Windows use manual native replay only if the Windows HIP stack matches the capture |

On Windows, prefer **`--no-replay`** through the Python analyzer until a
cross-platform entry point lands. Do not claim full GPU triage succeeded if replay
did not run. For Linux container captures without a matching Windows HIP stack,
say replay must happen on Linux.

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
3. **Native:** `triage_archive.sh --archive <pid-dir>` (builds/finds host `hrr-playback`).
4. **Docker:** `export HRR_DOCKER_IMAGE='<capture image>'` then
   `triage_archive.sh --archive <pid-dir> --replay docker`. Image `hrr-playback` by
   default; dev overlay via `HRR_DOCKER_MOUNT_CLR=1`. Optional `GPU=1` when GPU 0 busy.
5. Print the finding summary from triage output. Add a one-line capture explainer if helpful.

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

## Hard constraints

- **Windows hosts:** follow [Windows support](#windows-support); do not run `triage_archive.sh`
  for full GPU replay unless the user explicitly has bash + a working Linux-style GPU path.
  Prefer `analyze_replay_finding.py --archive <pid-dir>` (`--no-replay`) or document manual
  `hrr-playback.exe` steps.
- **No source edits.** One `ensure_playback.sh --build` attempt max; on failure stop (or `--no-replay`).
- Do not ask about Docker until native replay fails **unless** user named the capture image.
- Do not ask user to run cmake/ninja — scripts handle it.
- Do not read stale `*.finding.md` / `hrr-replay-*.log` unless user points at them.

## If build or replay fails

| Situation | Action |
|-----------|--------|
| `--build` fails (`rocdevice.cpp` etc.) | Set `CLR_BUILD` to existing `projects/clr/build-hrr*`; re-run. |
| Native replay fails (Linux) | Docker replay with capture image (Linux host). |
| Windows host | Metadata-only via `analyze_replay_finding.py`, or manual `hrr-playback.exe` + analyzer; no Docker; no auto-build. |
| Docker `hrr-playback` not in image | `HRR_DOCKER_MOUNT_CLR=1` with `CLR_BUILD` / `HRR_PLAYBACK`. |
| Docker HIP symbol/version mismatch | Dev overlay — `HRR_DOCKER_MOUNT_CLR=1`; `replay_docker.sh` puts mounted CLR first on `LD_LIBRARY_PATH`. |
| Preflight exit 2 | Ask user to confirm; re-run with `HRR_CONTINUE=1`. |
