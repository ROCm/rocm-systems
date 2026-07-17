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

The user only needs the archive path. **Default is read-only** (`--info` + log parse). Run full GPU replay only when the user explicitly asks.

## What to ask the user (only if missing)

| Missing | Ask once |
|---------|----------|
| Archive path | *"Which `capture.hrr/pid-*` directory should I use?"* |
| Full replay requested but no GPU/docker | Confirm native replay is OK or use `--no-replay` |
| `ensure_playback.sh` fails (no CLR tree, no cmake/ninja) | Only then ask where `rocm-systems` / `hrr-playback` lives |

Do **not** ask for GPU index, Docker, ROCm version, or HIP library paths unless replay fails.
Do **not** ask the user to run cmake/ninja manually — the skill builds playback itself.

## Agent workflow

```
1. Resolve archive — user path, or largest events.bin under capture.hrr/pid-*
2. Bootstrap playback — ensure_playback.sh (find or build hrr-playback from CLR tree)
3. Run decode_finding.sh --archive <pid-dir>   # read-only by default
4. Print finding summary + capture explainer in the chat reply (required)
5. If user asked for full replay: triage_archive.sh --archive <dir> --replay
```

**Execute in the same turn** — do not narrate planning steps.

Do **not** read stale `hrr-replay-*.log` or `*.finding.md` files unless the user gives that path.

### Primary commands

```bash
SKILL=<path-to>/skills/decode-and-triage
# In-tree: hipamd/src/hrr/skills/decode-and-triage
# Optional when symlinked outside the repo: export HRR_ROOT=<workspace-with-rocm-systems>

# Bootstrap (find or build hrr-playback; sets CLR_BUILD / HRR_PLAYBACK / LD_LIBRARY_PATH):
HRR_PLAYBACK="$("$SKILL/scripts/ensure_playback.sh")"

# Read-only (default):
"$SKILL/scripts/decode_finding.sh" --archive <pid-dir>

# Full replay + finding (only when user asks):
"$SKILL/scripts/triage_archive.sh" --archive <pid-dir> --replay
# or: --replay native | --replay docker | --replay auto
```

**Replay environment:** Prefer `--replay docker` when native replay fails with a HIP
library mismatch (`libamdhip64` version/symbol errors against `/opt/rocm`). Native replay
requires `hrr-playback` and `libamdhip64` from the same build on `LD_LIBRARY_PATH`.

- Read-only path runs `hrr-playback --info` + parser (no GPU)
- Full replay writes `hrr-replay-<pid>-<timestamp>.log` and a finding file under cwd
- Prints the finding to stdout — **copy the summary into your reply**

### Bootstrap `hrr-playback` (agent runs this — not the user)

Run **`ensure_playback.sh`** before decode/replay. It:

1. Reuses an existing binary (`PATH`, `HRR_PLAYBACK`, `CLR_BUILD`, `/opt/rocm/bin`)
2. Else finds CLR source from the in-tree skill path or `$HRR_ROOT/projects/clr`
3. Else runs `cmake` + `ninja amdhip64 hrr-playback` into `build-hrr/`
4. Exports `CLR_BUILD`, `HRR_PLAYBACK`, and `LD_LIBRARY_PATH`

Manual build details: [reference.md](reference.md).

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
| `replay_aborted` | Replay stopped before classification |
| `version_mismatch` | Archive wire version ≠ playback reader |

## Building `hrr-playback`

See [reference.md](reference.md).

## Further reading

- [reference.md](reference.md) — log patterns, archive layout, build
- [examples.md](examples.md) — user phrasing and agent responses
