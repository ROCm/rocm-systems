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
2. **Preflight** (when `pid-*/manifest.json` has `metadata` from #8680): compare capture
   GPU count/arch and HIP/comgr versions vs replay host or Docker image. **Block replay**
   if capture used more GPUs than replay exposes, or requested `GPU` ordinal is missing.
   **HIP or comgr version mismatch** prints a confirmation prompt before replay; in
   non-interactive runs (agents) exit code 2 — ask the user *"Do you want to continue?"*
   and re-run with `HRR_CONTINUE=1` if yes. Docker preflight probes the image's own
   HRR/ROCm stack by default. Set `HRR_DOCKER_MOUNT_CLR=1` to overlay a host dev build
   (`CLR_BUILD` / `HRR_PLAYBACK` → `/opt/hrr/lib`) for WIP HRR work. Legacy archives
   without metadata skip preflight.
   Override: `HRR_SKIP_COMPAT=1`; hard block (no prompt): `HRR_STRICT_VERSION=1`,
   `HRR_STRICT_ARCH=1`.
3. **Native first:** `"$SKILL/scripts/triage_archive.sh" --archive <pid-dir>`
4. **Docker capture** (user names the image, or native replay fails with library skew):
   ```bash
   export HRR_DOCKER_IMAGE='<image from user>'
   "$SKILL/scripts/triage_archive.sh" --archive <pid-dir> --replay docker
   ```
   Docker replay uses **`hrr-playback` from the image** by default. For a host dev
   overlay (image lacks HRR or you need a WIP build): `export HRR_DOCKER_MOUNT_CLR=1`
   plus `CLR_BUILD` or `HRR_PLAYBACK`. `ensure_playback.sh --build` runs for native
   replay and for docker overlay only. `replay_docker.sh` defaults `HRR_DOCKER_EXTRA_LD`
   for `rocm/vllm:*` images. Optional: `export GPU=1` when GPU 0 is busy.
5. Reply with finding summary + short capture explainer.

`auto` uses docker when `HRR_DOCKER_IMAGE` is already set, else native. `--no-replay`
only when user asks for metadata-only.

## Manifest metadata (#8680)

Per-process `manifest.json` may embed capture metadata:

| Field | Use in triage |
|-------|----------------|
| `runtime.hip_runtime_version` | Prompt before replay on mismatch; block if `HRR_STRICT_VERSION=1` |
| `runtime.comgr_version` | Prompt before replay on mismatch; block if `HRR_STRICT_VERSION=1` |
| `device_count` / `devices[]` | Block replay when host/container exposes fewer GPUs |
| `devices[].properties.gcn_arch_name` | Warn/block arch mismatch (`HRR_STRICT_ARCH=1`) |
| `devices[].properties.name`, `total_global_mem`, `pci`, `uuid` | Surface in finding for context |

Legacy captures (no `metadata` key) still triage via `--info` + replay logs only.

Execute in the same turn. Do not read stale `*.finding.md` / `hrr-replay-*.log` unless
the user points at them.

If preflight exits **2** (HIP/comgr mismatch), **stop and ask the user**:
*"Capture and replay HIP/comgr versions differ. Do you want to continue?"*
Re-run triage with `HRR_CONTINUE=1` only after they confirm.

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
| Docker `hrr-playback` not in image | Set `HRR_DOCKER_MOUNT_CLR=1` with `CLR_BUILD` / `HRR_PLAYBACK` for dev overlay. |
| Docker `hip_7.14 not found` | With dev overlay, mounted CLR libs must come **before** container SDK on `LD_LIBRARY_PATH` (handled by `replay_docker.sh`). |

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
| `triage_archive.sh` | **Entry** — preflight + replay + finding |
| `check_replay_compat.py` | Manifest metadata preflight (#8680) |
| `ensure_playback.sh` | Find/build `hrr-playback` (scans `build-hrr*`, in-tree ROCR when needed) |
| `replay_docker.sh` | Docker replay (vLLM `EXTRA_LD` auto-default) |
| `analyze_replay_finding.py` | Parser (called by triage) |
