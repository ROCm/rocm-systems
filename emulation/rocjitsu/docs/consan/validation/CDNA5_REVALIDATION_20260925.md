# CDNA5 emulator revalidation — September 25, 2026

Campaign artifacts: `/home/benoit/workspace/consan-validation/cdna5-20260925/`.
Initial RocJITsu revision: `6aff016c3ce`, normal GCC build. Runtime and external
workloads follow [the audited setup](CDNA5_WORKLOAD_AUDIT_20260925.md).
The live qualification ledger is [STATUS_CDNA5.md](STATUS_CDNA5.md).

## Discovery prerequisite

The initial rocprofv3 kernel-trace attempt under the gfx1250 launcher aborted
before dispatch: canonicalizing `librocprofiler-sdk.so` returned ENOSYS.
The same profiler loads outside emulation with `/bin/true`. Evidence is in
`discovery/d128-block/probe.log` and `discovery/host-loader-check.log`.
The early legacy-stat failure was fixed in `3cc0f3ba80b`, and early open/close
handling in `a318db66e08`. Each passed a normal GCC rebuild and the interposer /
fork suite (70 passed, seven configuration-dependent skips).

Further tracing showed the profiler reading host topology and opening a host
render node before the emulator constructor. Do **not** pass those GPU ioctls
through. Prepending **both** profiler libraries with rocprofv3's `--preload`
corrects the constructor order:

```sh
--preload "$CONSAN_VALIDATION_ROCM_ROOT/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" \
          "$CONSAN_VALIDATION_ROCM_ROOT/lib/librocprofiler-sdk.so"
```

With this added to the rocprofv3 command inside the gfx1250 launcher, D128's
two numerical tests pass and kernel/agent CSVs are produced. Retained evidence:
`discovery/preload-both.log`, `discovery/d128-preload-both/`, and the generated
`discovery/d128-block.allowlist`. This is emulated dispatch discovery, not a
physical trace or performance measurement.

## Initial unfiltered clean control

`unfiltered-d128-block/` retains independent baseline, Default and SuperCollider
clean runs with all kernels eligible for instrumentation (no allowlist). All
three were accepted. Both engines patched 267/267 applicable access sites;
Default additionally patched 276/276 barriers. Static and dynamic completeness
passed with no unexpected diagnostics. These runs establish yellow cells;
reviewed prospective fault trials remain pending. Future filtered results will
use separate artifact roots and matching clean controls.

## Remaining hip-moi clean controls

`unfiltered-d128-pressure/`, `unfiltered-wmma-attention/`,
`unfiltered-streamk-arrival/`, and `unfiltered-tree-atomic-or/` each passed
baseline, Default, and SuperCollider clean qualification. Their cells are yellow
until prospective fault trials complete. The current D128 exact fault inventory
is retained under `unfiltered-inventory/d128-block/inventory/`; it is accepted
and has fresh code-object hashes and instruction addresses.
