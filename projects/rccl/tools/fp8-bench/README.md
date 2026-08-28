# fp8-bench

V1 GPU hardware probe for FP8 intrinsics. **Executes each intrinsic on the device** and reports whether it works — this is not an RCCL capability table.

## Build

```bash
cd projects/rccl/tools/fp8-bench
make
./fp8_bench
```

Built with `--offload-arch=native`. Rebuild if probing a different GPU architecture.

## Output legend

| Status | Meaning |
|--------|---------|
| `[yes]` | Intrinsic compiled for this GPU target, executed on device, sanity check passed |
| `[fail]` | Executed on device but sanity check failed |
| `[n/a]` | Not compiled for this GPU target (hardware/compiler does not expose it) |

## What is probed

- HIP FP8 types and conversion
- Legacy/gfx942 paths: `cvt_pk_f32_fp8`, `cvt_pk_fp8_f32`, `cvt_sr_*`, gfx950 scalef32 pk2
- **gfx1250 paths**: `cvt_pk_f16_fp8`, `cvt_pk_fp8_f16`, `cvt_scalef32_pk8_*`, `cvt_scale_pk8_*`
- Assembly: `v_cvt_f32_fp8` (gfx942/950), `v_pk_add_f32`, `v_pk_add_f16`
- End-to-end FP8 add chains (upcast → add → downcast) for e4m3/e5m2, pk1 and pk2

Availability is detected with `__builtin_amdgcn_is_invocable()` on the device, then each
intrinsic is executed with a sanity check. This correctly distinguishes gfx1250 from gfx950.

Each test runs in a device kernel on the selected GPU.
