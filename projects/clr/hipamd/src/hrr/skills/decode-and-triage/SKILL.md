---
name: hrr-decode-and-triage
description: >-
  Decode and triage HRR capture archives with full GPU replay by default.
  Builds hrr-playback when missing. Never edits source. Print finding summary
  in the chat reply.
---

# HRR Decode & Triage

Run **`scripts/triage_archive.sh --archive <pid-dir>`** and **print the finding markdown** it outputs.

## Example prompt

```
Use the hrr-decode-and-triage skill. Triage with docker replay:
/var/lib/rancher/maf-repro/runs/fresh-v4-maf-20260716T102834Z/capture.hrr/pid-138

Docker image used for capture:
rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

## Workflow

1. Resolve `capture.hrr/pid-*` (largest `events.bin` if user gives root only).
2. **Preflight** when `manifest.json` has `metadata` (#8680): block if capture used
   more GPUs than replay exposes or requested `GPU` is missing. HIP/comgr mismatch
   prompts for confirmation (exit 2 in agents — ask user, then `HRR_CONTINUE=1`).
   Docker preflight uses the **image HRR stack** by default; `HRR_DOCKER_MOUNT_CLR=1`
   overlays host `CLR_BUILD`/`HRR_PLAYBACK` for WIP dev work. Legacy archives skip
   preflight. Overrides: `HRR_SKIP_COMPAT=1`, `HRR_STRICT_VERSION=1`, `HRR_STRICT_ARCH=1`.
3. **Native:** `triage_archive.sh --archive <pid-dir>` (builds/finds host `hrr-playback`).
4. **Docker:** `export HRR_DOCKER_IMAGE='<capture image>'` then
   `triage_archive.sh --archive <pid-dir> --replay docker`. Image `hrr-playback` by
   default; dev overlay via `HRR_DOCKER_MOUNT_CLR=1`. Optional `GPU=1` when GPU 0 busy.
5. Print the finding summary from triage output (outcome, fault class, kernel, fault addr,
   event seq, archive stats). Add a one-line capture explainer if helpful.

`auto` picks docker when `HRR_DOCKER_IMAGE` is set, else native. `--no-replay` for metadata-only.

## Install / smoke test

Copy `SKILL.md` and `scripts/` (exclude `__pycache__`). Validate:

```bash
python3 scripts/test_analyze_replay_finding.py
python3 scripts/test_check_replay_compat.py
scripts/triage_archive.sh --help
scripts/replay_docker.sh --help
```

Metadata-only smoke (no GPU):

```bash
HRR_TRIAGE_WORKDIR=/tmp/hrr-skill-smoke \
  scripts/triage_archive.sh --archive <pid-dir> --no-replay
```

A finding note like `archive wire version N does not match hrr-playback reader M` comes
from `hrr-playback --info` when the installed reader differs from the archive; it is
informational and does not block `--no-replay`.

## Hard constraints

- **No source edits.** One `ensure_playback.sh --build` attempt max; on failure stop (or `--no-replay`).
- Do not ask about Docker until native replay fails **unless** user named the capture image.
- Do not ask user to run cmake/ninja — scripts handle it.
- Do not read stale `*.finding.md` / `hrr-replay-*.log` unless user points at them.

## If build or replay fails

| Situation | Action |
|-----------|--------|
| `--build` fails (`rocdevice.cpp` etc.) | Set `CLR_BUILD` to existing `projects/clr/build-hrr*`; re-run. |
| Native replay fails | Docker replay with capture image. |
| Docker `hrr-playback` not in image | `HRR_DOCKER_MOUNT_CLR=1` with `CLR_BUILD` / `HRR_PLAYBACK`. |
| Docker `hip_7.14 not found` | Dev overlay only — `replay_docker.sh` puts mounted CLR first on `LD_LIBRARY_PATH`. |
| Preflight exit 2 | Ask user to confirm; re-run with `HRR_CONTINUE=1`. |
