# Example 7: Kernel Occupancy Analysis

## Objective

Optimize kernel occupancy by **tuning resource usage** — specifically register
(VGPR) usage — and see the effect. Three kernels do the same work with different
register pressure; the program computes register-limited occupancy and rocjitsu
confirms the register allocation under `RJ_LOG=1`.

## What rocjitsu does and does not provide

| rocjitsu provides | rocjitsu does **not** provide |
|---|---|
| Real, compiler-allocated `vgprs`/`sgprs` per dispatch (`RJ_LOG=1`) | An occupancy number — it is a **functional** emulator, not a perf model |
| A functional run so `hipFuncGetAttributes` returns register usage | LDS-limited occupancy (RJ_LOG does not surface LDS usage) |

The occupancy figures here are a **host-side calculation** using the standard AMD
register-limited formula and the CU limits declared in
`configs/amdgpu_cdna4_kmd.json`. rocjitsu supplies the *inputs* (register usage);
it does not compute occupancy.

## The three kernels

| Kernel | Register pressure |
|---|---|
| `low_reg` | trivial (few VGPRs) |
| `high_reg` | 32 live accumulators (high VGPRs — occupancy-limiting) |
| `capped_reg` | same body as `high_reg` + `__launch_bounds__(256, 8)` to cap VGPRs |

`__launch_bounds__(maxThreadsPerBlock, minWavesPerSIMD)` tells the compiler you
need `minWavesPerSIMD` waves resident, so it caps VGPRs to fit — this is the
"tuning" knob.

## Occupancy formula

Per SIMD, register-limited:

```
alloc_vgprs        = round_up(used_vgprs, granularity)          # granularity = 8
waves_per_simd     = min(max_waves_per_simd, VGPR_budget / alloc_vgprs)
occupancy          = waves_per_simd / max_waves_per_simd
```

CU limits from the config: `max_waves_per_simd=8`, `simd_per_cu=4`
(`num_wf_slots=32`), `wave_front_size=64`, and the per-SIMD VGPR budget
(`vgprs_per_wf=512`).

## Files

- `src/occupancy.cpp` — the three kernels + host occupancy calculation
- `Makefile`

## Build and run

```bash
cd usage-examples/07-occupancy-analysis
make
make run        # RJ_LOG=1: dispatch vgprs + host occupancy report
make run-quiet  # host occupancy report only
```

## Expected output

Verbatim from `sharkmi300x-4` (ROCm 7.2.1, `amdgpu_cdna4_kmd.json`).

`RJ_LOG=1` dispatch metadata — note `vgprs=` changes per kernel:

```text
[rocjitsu] Kernel #1 dispatch
  entry_pc=0x5400012400  grid=[4096,1,1]  wg=[256,1,1]
  wgs=16  wfs/wg=4  sgprs=16  vgprs=8
[rocjitsu] Kernel #2 dispatch
  entry_pc=0x5400012500  grid=[4096,1,1]  wg=[256,1,1]
  wgs=16  wfs/wg=4  sgprs=16  vgprs=72
[rocjitsu] Kernel #3 dispatch
  entry_pc=0x5400013900  grid=[4096,1,1]  wg=[256,1,1]
  wgs=16  wfs/wg=4  sgprs=16  vgprs=64
```

Host occupancy report:

```text
Occupancy vs register usage - emulated MI350X CU (from amdgpu_cdna4_kmd.json):
  4 SIMDs/CU, max 8 waves/SIMD (= 32 wave slots/CU), wave = 64 threads,
  VGPR budget/SIMD = 512, allocation granularity = 8
  block = 256 threads = 4 waves/block

Register-limited occupancy (computed on host; rocjitsu does NOT compute this):
  low_reg     used_vgprs=4   alloc_vgprs=8   -> 8/8 waves/SIMD  (100.0% occupancy)
  high_reg    used_vgprs=66  alloc_vgprs=72  -> 7/8 waves/SIMD  ( 87.5% occupancy)
  capped_reg  used_vgprs=64  alloc_vgprs=64  -> 8/8 waves/SIMD  (100.0% occupancy)
```

## What this shows

- **Tuning resource usage changes occupancy.** `high_reg` needs 72 VGPRs/wave, so
  only `floor(512/72)=7` of 8 wave slots per SIMD can be filled → **87.5%**.
- **`__launch_bounds__` is the fix.** It caps `capped_reg` to 64 VGPRs
  (`512/64=8`), restoring **100%** occupancy — same work, better occupancy.
- **Cross-check:** the program's `alloc_vgprs` (8 / 72 / 64) matches `RJ_LOG`'s
  `vgprs=` exactly. `RJ_LOG` reports *allocated* VGPRs (rounded up to
  granularity 8); `hipFuncGetAttributes` reports *used* (4 / 66 / 64).

## Caveats

- rocjitsu **does not compute occupancy**; the numbers above are standard AMD
  math on the config's CU limits. On real hardware, confirm with `rocprofv3` /
  `rocprof-compute`.
- **LDS-limited occupancy is not shown.** `RJ_LOG` does not print LDS usage;
  `hipFuncGetAttributes().sharedSizeBytes` exposes it on the host if you want to
  extend the calculation.
- SGPRs (16 here) are far below the per-wave budget, so they do not limit
  occupancy in this example.

## Key takeaways

- Occupancy is limited by whichever resource runs out first: VGPRs, SGPRs, LDS,
  or wave slots. Here it is **VGPRs**.
- Reducing per-wave register usage (leaner code or `__launch_bounds__`) raises
  the number of resident waves.
- rocjitsu gives you the **real register usage** to drive the calculation; the
  occupancy math and the hardware limits come from you / the config.

## Related

- [Example 6: Hook profiling](../06-memory-coalescing/) — another "rocjitsu is
  functional, not a perf model" case
- `configs/amdgpu_cdna4_kmd.json` — the CU limits used above
