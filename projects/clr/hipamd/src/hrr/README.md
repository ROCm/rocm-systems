# HIP Record & Replay (HRR)

HRR captures HIP API traces into a binary archive and replays them on a live GPU for bug reproduction and validation.

| Doc | Purpose |
|-----|---------|
| [DESIGN.md](DESIGN.md) | Full architecture, archive format, limitations, build system |
| [skills/decode-and-triage/SKILL.md](skills/decode-and-triage/SKILL.md) | **A1** — Cursor/agent skill for read-only decode & triage |

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

### Archive info (no GPU)

```bash
hrr-playback ./my_capture.hrr/pid-<pid>/ --info
```

### Replay on GPU

Requires AMD GPU, `/dev/kfd`, and matching `hrr-playback` + HIP libraries:

```bash
hrr-playback ./my_capture.hrr/pid-<pid>/
```

Point at a specific `pid-<pid>/` subdirectory when the capture root has multiple processes.

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

Details: [DESIGN.md § Build System](DESIGN.md#build-system) and [skills/decode-and-triage/reference.md](skills/decode-and-triage/reference.md).

## Decode & triage (A1)

Read-only triage of replay logs and archive metadata — **no GPU required**.

```bash
SKILL=hipamd/src/hrr/skills/decode-and-triage
"$SKILL/scripts/decode_finding.sh" --archive /path/to/capture.hrr/pid-<pid>/
# With an existing replay log:
"$SKILL/scripts/decode_finding.sh" --archive /path/to/pid-<pid>/ --log replay.log
```

Produces a structured **Finding**: fault class, fault address, failing HIP call index/API, implicated kernel, plus a capture explainer.

For Cursor agents, install or reference `skills/decode-and-triage/SKILL.md`.

## Productization roadmap

| ID | Item | Status |
|----|------|--------|
| **A1** | **Decode & triage skill** — read replay fault/divergence; output structured finding (fault class, address, failing call, implicated kernel) + capture explainer; read-only; ships first | **In tree** (`skills/decode-and-triage/`) |
| **R9** | **Knob docs + tests + basic user guide** — formal reference for `HIP_HRR_*` / CLI knobs, regression tests for playback options, and expanded end-user documentation beyond this README | Planned |

Knob behavior today is described inline in [DESIGN.md](DESIGN.md) (playback options, D2H tolerance, alloc padding, debug flags). R9 will consolidate that into tested, user-facing docs.

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

## Copyright

AMD SPDX MIT — see individual source files.
