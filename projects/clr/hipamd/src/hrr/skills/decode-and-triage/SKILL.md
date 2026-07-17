---
name: hrr-decode-and-triage
description: >-
  Decode and triage HRR capture archives with full GPU replay by default.
  Builds hrr-playback when missing. Never edits source. Print finding summary
  in the chat reply.
---

# HRR Decode & Triage

Run **`scripts/triage_archive.sh --archive <pid-dir>`** and **print the finding in your reply**.

## Typical user message

```
Use the hrr-decode-and-triage skill. Triage with docker replay:
/var/lib/rancher/maf-repro/runs/fresh-v4-maf-20260716T102834Z/capture.hrr/pid-138

Docker image used for capture:
rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

## Workflow

1. Resolve `capture.hrr/pid-*` (largest `events.bin` if user gives root only).
2. **Native first:** `"$SKILL/scripts/triage_archive.sh" --archive <pid-dir>`
3. **Docker capture** (user names the image, or native replay fails with library skew):
   ```bash
   export HRR_DOCKER_IMAGE='<image from user>'
   "$SKILL/scripts/triage_archive.sh" --archive <pid-dir> --replay docker
   ```
   `ensure_playback.sh` sets `ROCR_LIB` when in-tree ROCR exists. `replay_docker.sh`
   defaults `HRR_DOCKER_EXTRA_LD` for `rocm/vllm:*` images. Optional: `export GPU=1`
   when GPU 0 is busy.
4. Reply with finding summary + short capture explainer.

`auto` uses docker when `HRR_DOCKER_IMAGE` is already set, else native. `--no-replay`
only when user asks for metadata-only.

Execute in the same turn. Do not read stale `*.finding.md` / `hrr-replay-*.log` unless
the user points at them.

## Hard constraints

- **No source edits** (CLR, CMake, one-off parsers). One `ensure_playback.sh --build`
  attempt max; on failure report stderr and stop (optional `--no-replay` for manifest-only).
- Do not ask about Docker until native replay fails **unless** the user already named
  the capture image.
- Do not ask user to run cmake/ninja — scripts handle it.

## If build or replay fails

| Situation | Action |
|-----------|--------|
| `--build` fails (`rocdevice.cpp` etc.) | Set `CLR_BUILD` to existing `projects/clr/build-hrr*` if present; re-run. |
| Native replay fails | `LD_LIBRARY_PATH`: `<clr-build>/hipamd/lib` → in-tree ROCR → `/opt/rocm/lib`. Then docker replay with capture image. |
| Docker `hip_7.14 not found` | Mounted CLR libs must come **before** container SDK on `LD_LIBRARY_PATH` (handled by `replay_docker.sh`). |

## Reply template (required)

```markdown
## HRR finding summary

- **Outcome**: …
- **Fault class**: …
- **Kernel**: …
- **Fault address**: …
- **Failing call**: Event … (`…`) — or *n/a (GPU fault during kernel execution)*
- **Archive**: … events, Complete: …

### Capture explainer
Brief: `events.bin` trace, `blobs/` payloads, complete vs crash-truncated, wire version vs reader.
```

## Fault classes

`replay_pass` · `read_only_page_fault` · `illegal_memory_access` · `nan_inf_divergence` ·
`hang` · `replay_oom` · `replay_fatal_api` · `replay_aborted` · `version_mismatch`

## Scripts

| Script | Role |
|--------|------|
| `triage_archive.sh` | **Entry** — replay + finding |
| `ensure_playback.sh` | Find/build `hrr-playback` (scans `build-hrr*`, in-tree ROCR when needed) |
| `replay_docker.sh` | Docker replay (vLLM `EXTRA_LD` auto-default) |
| `analyze_replay_finding.py` | Parser (called by triage) |
