# reduceOpBench

GPU probe for FP8/BF8 hardware intrinsics. Reports, per GPU, which ones the toolchain
can emit for that target and — optionally — which ones actually execute correctly.

This is not an RCCL capability table, and it does not measure performance yet: there is
no timing anywhere in the tool.

## Build

```bash
cd projects/rccl/tools/reduceOpBench
make
./reduceOpBench
```

`make` generates `fp8_auto_probe.h` from the installed toolchain, then compiles with
`--offload-arch=native`. Rebuild when probing a different GPU architecture.

| Target / variable | Purpose |
|---|---|
| `make regen` | Re-derive the probe list after a ROCm change (not automatic) |
| `make clean` | Remove the binary and the generated header |
| `ROCM_PATH` | ROCm install to read builtins from (default `/opt/rocm`) |
| `AUTO_CATEGORIES` | `cvt` (default), or any of `mfma,wmma,dot,other`, or `all` |

## Usage

```
./reduceOpBench [-d DEVICE] [-f] [-h]

  -d DEVICE         Probe one GPU (default: all GPUs)
  -f, --functional  Also run the hand-written execute+sanity suite
  -h                Show help
```

## Modes

**Default — availability.** Intrinsics are discovered from the installed toolchain and
the device reports `__builtin_amdgcn_is_invocable()` for each. This is a compile-time
answer for the offload target, so nothing is executed and nothing can fail.

| Status | Meaning |
|--------|---------|
| `[yes]` | Invocable for this offload target |
| `[n/a]` | Known to the toolchain but not invocable on this GPU |

**`-f` — functional.** Runs the curated suite in `fp8_probe.h`, which executes each
intrinsic in a device kernel and checks the result. Covers the HIP `__hip_fp8_*` types,
upcast and downcast for e4m3 and e5m2 (`cvt_pk_f32_*`, `cvt_sr_*`, `cvt_scalef32_pk_*`,
inline `v_cvt_f32_fp8`), arithmetic (`v_pk_add_f32`, `v_pk_add_f16`, `fmed3f`), and
end-to-end add chains (upcast → add → downcast) for e4m3/e5m2 in pk1 and pk2 form.

| Status | Meaning |
|--------|---------|
| `[yes]` | Executed on device and passed its sanity check |
| `[fail]` | Executed but the sanity check failed |
| `[n/a]` | Not available for this GPU target |

## How discovery works

`gen_auto_probe.py` emits the probe kernel, so the list tracks whatever the local ROCm
knows rather than a hand-maintained checklist. It prefers Clang's
`BuiltinsAMDGPU.def`/`.inc` under `$ROCM_PATH`. Runtime-only installs ship no
`clang/Basic` headers, so it falls back to the builtin names embedded in the clang
executable's string table.

Output is grouped by Clang's target-feature string (`Feature: fp8-conversion-insts`)
when the headers were available. The binary fallback cannot recover those strings, so
it groups by a label derived from the instruction name instead (`Group: fp8 convert`).
The per-intrinsic verdicts are identical either way.

Pass `--builtins` to point the generator at a specific `.def`, `.inc`, or clang binary.
