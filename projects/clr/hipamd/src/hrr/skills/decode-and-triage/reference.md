# HRR decode & triage reference

## Directory layout

```
capture.hrr/
  manifest.json       # root index (multi-process captures)
  pid-<capture_pid>/
    events.bin          # primary event stream (may be GB-scale)
    manifest.json       # { complete, event_count, blob_count, ... }
    blobs/              # content-addressed host payloads
    code_objects/       # captured ELFs (when used)
```

Pick the `pid-*` directory with the **largest `events.bin`** for the faulting process.

If the user points at `capture.hrr/` (root), resolve to one `pid-*` child before replay or `--info`.

## events.bin record model (conceptual)

Each event has:

| Field | Role |
|-------|------|
| Thread id | Capturing host thread |
| Sequence / event index | Monotonic call index in replay |
| API id | HIP API (malloc, launch, memcpy, sync, …) |
| Payload | API-specific bytes (variable-length for kernel launches) |

**Kernel launch payload** includes: stream, kernel name, code-object hash, grid, block, shared memory, **kernarg blob** (pointer table + struct args), optional D2H snapshot descriptors.

Wire format version is in the file header. **`hrr-playback` reader version must match** the capture (e.g. v4 widened `payload_length` to `uint32_t`; v3 archives fail on v4 playback).

## Completeness markers

| Signal | Meaning |
|--------|---------|
| `Complete: YES` (`--info`) | Clean shutdown trailer present |
| `recovered N events` | Crash capture; trailer missing; reader kept all complete records |
| `Torn trailing record` | Last record partial; preceding events valid |

Crash captures are **expected** to lack a trailer and still be replayable.

## Replay log lines

### Progress

```
[HRR progress] elapsed_s=... seq=13118764 kernels=797227 d2h_pass=4303 d2h_fail=0 ...
```

- `seq` — last replayed event sequence number (proxy for **failing_event_seq** when fault follows)
- `kernels` — kernel launch count so far
- `d2h_*` — device-to-host validation counters

### GPU memory fault (ROCr)

```
Memory access fault by GPU node-N (Agent handle: 0x...) on address 0xADDR. Reason: ...
:0:rocdevice.cpp:NNNN: Memory Fault Error [..., faulting addr: 0xADDR, kernel: Cijk_...]
```

Extract: **fault_address**, **kernel_name**, **gpu_node**, **fault_reason**.

### Fatal API abort

```
[HRR] Fatal: T146 Event 9268 (hipMalloc) returned 2 (out of memory) — aborting replay
```

Extract: **failing_thread**, **failing_call_index**, **failing_api**.

### Suballoc fidelity (optional playback feature)

```
[HRR] SUBALLOC OOB: kernel arg[10] rec 0x... resolves inside a captured segment but in no active tensor block
```

High count on one arg index with `d2h_fail=0` and later MAF → likely **stale/OOB device pointer** in kernarg.

## Tensile / hipBLASLt kernel name cheat sheet

Example:

```
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x192x128_..._SK3_..._WS64_WG16_16_1
```

| Token | Meaning |
|-------|---------|
| `Cijk_*` | Contraction GEMM family |
| `MT128x192x128` | Macro-tile dimensions |
| `SK3` | StreamK variant |
| `WS64` | Workspace-related sizing hint |
| `Bias_HA` | Bias + HPA layout flags |

**read_only_page_fault** on StreamK GEMM → investigate edge tile / workspace (`AddressWS`) / output (`AddressD`) stores.

## Build `hrr-playback` from this tree

Playback is built as part of the CLR `amdhip` CMake tree (`hrr/CMakeLists.txt` → `add_subdirectory(playback)`). Linux/Ninja only.

### Configure (once)

```bash
cd <path-to-ROCm-CLR>/projects/clr

cmake -S . -B build -GNinja \
  -DHIP_COMMON_DIR=<path-to-HIP> \
  -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  -DCMAKE_BUILD_TYPE=Release
```

### Build capture runtime + playback tool

```bash
ninja -C build amdhip64 hrr-playback -j"$(nproc)"

# Runtime: build/hipamd/lib/libamdhip64.so*
# Tool:    find build -name hrr-playback -type f
```

### Deploy for capture or replay

```bash
export LD_LIBRARY_PATH="$PWD/build/hipamd/lib:${LD_LIBRARY_PATH:-}"

# Capture with in-tree runtime:
HIP_HRR_CAPTURE_OUTPUT=./out.hrr ./my_hip_app

# Replay (requires GPU + /dev/kfd):
build/.../hrr-playback ./out.hrr/pid-<pid>/

# Read-only archive summary (no GPU):
build/.../hrr-playback ./out.hrr/pid-<pid>/ --info
```

### Validation tool (test tree)

```bash
ninja -C build_test hrr-validate
```

Full build notes: [DESIGN.md](../../DESIGN.md#build-system).

## Playback binary discovery

| Path | Role |
|------|------|
| `hrr-playback` on `PATH` | Preferred if present |
| `$ROCM_PATH/bin/hrr-playback` | Installed ROCm (default `/opt/rocm`) |
| `<clr-build>/.../hrr-playback` | Developer build from this tree |

Set `HRR_PLAYBACK=<absolute-path>` when the binary is not in a standard location.

## Parser script

```bash
python3 skills/decode-and-triage/scripts/analyze_replay_finding.py --help
```

Outputs JSON or Markdown **Finding** with fields:

`outcome`, `fault_class`, `fault_address`, `failing_event_seq`, `failing_call_index`, `failing_api`, `kernel_name`, `kernarg_address`, `d2h_fail`, `archive_events`, …

## Read-only entry script

```bash
skills/decode-and-triage/scripts/decode_finding.sh --archive <pid-dir> [--log replay.log]
```

Runs `hrr-playback --info` (when available) and the parser. Does **not** run full GPU replay.

## Knobs (replay / capture)

Environment variables and CLI flags are documented in [DESIGN.md](../../DESIGN.md) (playback options, `HIP_HRR_REPLAY_*`, D2H tolerance, alloc padding). Formal knob reference + tests are tracked as **R9** in the productization roadmap.
