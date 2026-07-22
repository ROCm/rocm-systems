# HIP Record & Replay (HRR)

HRR captures HIP API traces into a binary archive and replays them on a live GPU for bug reproduction and validation.

| Doc | Purpose |
|-----|---------|
| [DESIGN.md](DESIGN.md) | Full architecture, archive format, limitations, build system |

## Quick start

### Capture

```bash
HIP_HRR_CAPTURE_OUTPUT=./my_capture.hrr ./my_hip_app
```

Use the in-tree `libamdhip64` from a developer build when testing capture changes:

```bash
export LD_LIBRARY_PATH=<clr-build>/hipamd/lib:$LD_LIBRARY_PATH
HIP_HRR_CAPTURE_OUTPUT=./out.hrr ./my_hip_app
```

## Build `hrr-playback`

From the ROCm CLR tree (`projects/clr`):

```bash
cmake -S . -B build -GNinja \
  -DHIP_COMMON_DIR=<path-to-HIP> \
  -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build amdhip64 hrr-playback -j"$(nproc)"
```

Locate the binary: `find build -name hrr-playback -type f`

Details: [DESIGN.md § Build System](DESIGN.md#build-system).

## Inspecting and replaying captures

Point at a specific `pid-<pid>/` subdirectory when the capture root has multiple processes.

**Summary only (no GPU):**

```bash
hrr-playback ./my_capture.hrr/pid-<pid>/ --info
```

**Full replay** (Windows or Linux; requires AMD GPU and matching `hrr-playback` + HIP libraries):

```bash
hrr-playback ./my_capture.hrr/pid-<pid>/
```

**HIP library matching:** `hrr-playback` must load the same `libamdhip64` it was built against. If native replay fails with a symbol/version error (e.g. `hip_7.14 not found` from `/opt/rocm/lib`), either run with only the build tree on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=<clr-build>/hipamd/lib
hrr-playback ./my_capture.hrr/pid-<pid>/
```

Point `LD_LIBRARY_PATH` at the same `<clr-build>/hipamd/lib` used for capture when
`/opt/rocm/lib` triggers a symbol/version mismatch.

Useful flags when debugging a fault or hang: `--sync-after-launch`, `--sync-watchdog-ms N`, `--progress-seconds S`, `--trace-kernels`. See [Configuration reference](#configuration-reference) below.

## Configuration reference

User-facing capture, replay, and validation knobs. Implementation details remain in [DESIGN.md](DESIGN.md).

### Capture environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `HIP_HRR_CAPTURE_OUTPUT` | *(unset)* | Enable capture; path to the `.hrr` archive directory |
| `HIP_HRR_DEBUG_ARGS` | off | Dump every captured kernel arg to the log (debug / provenance) |

### `hrr-playback` CLI options

| Option | Purpose |
|--------|---------|
| `--info` | Print archive summary and exit (no GPU) |
| `--repair` | Rewrite a crash-truncated archive with a clean trailer |
| `--events` | With `--info`: print the full event log |
| `--verbose` | Print each event as it is replayed |
| `--skip-device-sync` | Skip `hipDeviceSynchronize` / `hipStreamSynchronize` events |
| `--multi-thread` | One replay thread per captured thread (default: single-threaded) |
| `--timing` | Report wall time and GPU kernel/graph time |
| `--kernel-filter STR` | Only launch kernels whose name contains `STR` (warm-up pass first) |
| `--replace-kernel N=P` | Launch recorded kernel `N` from external code object `P` instead |
| `--sync-after-launch` | `hipDeviceSynchronize` after every kernel launch |
| `--sync-after-event` | Sync after every event (slow; pinpoints faults/hangs) |
| `--sync-watchdog-ms N` | Abort if any device sync exceeds `N` ms (`0` = disabled) |
| `--trace-kernels` | One compact line before every kernel launch |
| `--trace-sync` | Log sync begin/done around kernel syncs |
| `--progress-kernels N` | Heartbeat every `N` launched kernels |
| `--progress-seconds S` | Heartbeat at most every `S` seconds |

### Replay environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `HIP_HRR_REPLAY_ALLOC_PAD_FACTOR` | `1` | Multiply replayed `hipMalloc` size for pool-style headroom (`256` for legacy MIOpen-style workloads; uses more VRAM) |
| `HIP_HRR_REPLAY_ALLOC_PAD_MAX` | `1073741824` (1 GiB) | Cap per-allocation padded size |
| `HIP_HRR_REPLAY_ZERO_INIT` | on | Zero-fill replay allocations so OOB reads see zeros (`0` to skip) |
| `HIP_HRR_REPLAY_TRACE_KERNELS` | off | Same as `--trace-kernels` |
| `HIP_HRR_REPLAY_TRACE_SYNC` | off | Same as `--trace-sync` |
| `HIP_HRR_REPLAY_PROGRESS_KERNELS` | `0` | Same as `--progress-kernels N` |
| `HIP_HRR_REPLAY_PROGRESS_SECONDS` | `0` | Same as `--progress-seconds S` |
| `HIP_HRR_REPLAY_SYNC_WATCHDOG_MS` | `0` | Same as `--sync-watchdog-ms N` |
| `HIP_HRR_REPLAY_DIVERGENCE_ABORT` | `0.25` | Exit early when D2H failure ratio exceeds this fraction (`0` disables) |
| `HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES` | `64` | Minimum D2H attempts before divergence abort applies |
| `HIP_HRR_REPLAY_NO_RESCAN` | off | Disable suballoc pointer rescan at replay |
| `HIP_HRR_PTR_RELAX` | off | Disable replay-side stale-pointer guard (debug only) |
| `HIP_HRR_REPLAY_FORCE_EXT_CIJK` | off | Force external Cijk kernel binding workaround (debug) |
| `HIP_HRR_REPLAY_DUMP_PTRS_ORDINAL` | `0` | Dump pointer translation map at event ordinal `N` (debug) |

### D2H validation

| Variable | Default | Purpose |
|----------|---------|---------|
| `HIP_HRR_D2H_ATOL` | tuned for bf16/fp16 | Absolute tolerance for float D2H compare |
| `HIP_HRR_D2H_RTOL` | tuned for bf16/fp16 | Relative tolerance for float D2H compare |
| `HIP_HRR_D2H_EXACT` | off | Require byte-exact D2H match (disable tolerance) |

**Exit codes:** `0` pass; `1` D2H failure; `2` early divergence abort.

## Archive layout (short)

```
capture.hrr/
  manifest.json
  pid-<pid>/
    events.bin
    blobs/
    manifest.json
```

- **events.bin** — HIP API event stream
- **blobs/** — host payloads referenced by the trace
- **Complete: NO** — original run crashed before clean shutdown; reader still recovers complete events

Capture wire version must match the `hrr-playback` reader (see DESIGN.md wire-format notes).

## Agent tooling

Optional Cursor/agent skill: [skills/decode-and-triage/SKILL.md](skills/decode-and-triage/SKILL.md)
(Windows or Linux native replay; Docker replay on Linux hosts). Docker replay uses the
image HRR stack by default; set `HRR_DOCKER_MOUNT_CLR=1` to overlay a host dev build
(`CLR_BUILD` / `HRR_PLAYBACK`).

## Copyright

AMD SPDX MIT — see individual source files.
