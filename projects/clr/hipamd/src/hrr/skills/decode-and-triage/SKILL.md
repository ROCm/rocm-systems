---
name: hrr-decode-and-triage
description: >-
  Decode and triage HIP Runtime Replay (HRR) capture archives. Runs archive
  decode, optional full GPU replay, and emits a structured finding (fault class,
  fault address, failing HIP call, implicated kernel) plus a capture explainer.
  Prints the summary in the chat reply. Use when the user asks to decode,
  triage, or analyze an HRR archive, replay log, or GPU crash capture
  (capture.hrr, pid-*, hrr-playback, memory access fault).
---

# HRR Decode & Triage

Decode an HRR archive, optionally replay it on GPU, and produce a structured **Finding** plus a **capture explainer**. **Always print the finding summary in your chat reply** — the user must not need to open a file.

## What the user should say

```
Use the hrr-decode-and-triage skill. Decode and triage this archive:
/path/to/capture.hrr/pid-138

Print the finding summary in your reply (outcome, fault class, kernel, fault
address, failing call, archive stats) plus a short capture explainer.
```

The user only needs the archive path. Full GPU replay is **allowed by default** when `/dev/kfd` is available.

## What to ask the user (only if missing)

| Missing | Ask once |
|---------|----------|
| Archive path | *"Which `capture.hrr/pid-*` directory should I use?"* |
| `hrr-playback` not found after discovery | *"Where is `hrr-playback` installed?"* |

Do **not** ask for GPU index, Docker, ROCm version, or HIP library paths unless replay fails.

## Agent workflow

```
1. Resolve archive — user path, or largest events.bin under capture.hrr/pid-*
2. Discover hrr-playback (see below)
3. Run triage_archive.sh --archive <pid-dir>   # decode + replay + finding
4. Print finding summary + capture explainer in the chat reply (required)
```

**Execute in the same turn** — do not narrate planning steps.

Do **not** read stale `hrr-replay-*.log` or `*.finding.md` files unless the user gives that path.

### Primary command

```bash
SKILL=<path-to>/.cursor/skills/hrr-decode-and-triage
# or: hipamd/src/hrr/skills/decode-and-triage

"$SKILL/scripts/triage_archive.sh" --archive <pid-dir>
```

- Runs `hrr-playback --info` + full GPU replay (docker if `scripts/maf-hrr-docker-playback.sh` exists, else native) + parser
- Writes fresh `hrr-replay-<pid>-<timestamp>.log` and `<pid>-<timestamp>.finding.md` under cwd
- Prints the finding markdown to stdout — **copy the summary into your reply**

Read-only (no replay): `"$SKILL/scripts/triage_archive.sh" --archive <dir> --no-replay`

### Discover `hrr-playback`

1. `command -v hrr-playback`
2. `${CLR_BUILD}/hipamd/src/hrr/playback/hrr-playback`
3. `/var/lib/rancher/hrr-develop-wt/projects/clr/build-hrr/hipamd/src/hrr/playback/hrr-playback`
4. `$ROCM_PATH/bin/hrr-playback` (default `ROCM_PATH=/opt/rocm`)
5. User-provided path → `HRR_PLAYBACK` for that run only

### Reply to the user (required)

After `triage_archive.sh`, **always** include this in your message:

```markdown
## HRR finding summary

- **Outcome**: …
- **Fault class**: …
- **Kernel**: …
- **Fault address**: …
- **Failing call**: Event … (`…`) — or *n/a (GPU fault during kernel execution)*
- **Archive**: … events, Complete: …

### Capture explainer
…
```

## Structured Finding (output contract)

| Field | Meaning |
|-------|---------|
| `fault_class` | Taxonomy bucket (see below) |
| `fault_address` | GPU faulting VA (if MAF) |
| `failing_call_index` | HIP event index at abort |
| `failing_api` | HIP API that failed |
| `kernel_name` | Kernel from ROCr fault line |
| `outcome` | `PASS`, `MAF`, `FAIL`, `ABORT`, or `UNKNOWN` |

## Capture explainer (always include)

- **`events.bin`** — HIP API trace for one process (`pid-<pid>/`)
- **`blobs/`** — host payloads referenced by the trace
- **`manifest.json`** — per-process completeness
- **Clean trailer vs crash** — missing trailer = original run crashed; reader recovers complete events
- **Version match** — capture wire version must match `hrr-playback` reader (v3 ≠ v4)

## Fault taxonomy

| `fault_class` | Meaning |
|---------------|---------|
| `replay_pass` | Clean replay |
| `read_only_page_fault` | Write to read-only page |
| `illegal_memory_access` | Other GPU memory fault |
| `nan_inf_divergence` | D2H numerical mismatch |
| `hang` | Device/queue hang |
| `replay_oom` | Out of VRAM |
| `replay_fatal_api` | HIP API error stopped replay |
| `version_mismatch` | Archive wire version ≠ playback reader |

## Building `hrr-playback`

See [reference.md](reference.md).

## Further reading

- [reference.md](reference.md) — log patterns, archive layout, build
- [examples.md](examples.md) — user phrasing and agent responses
