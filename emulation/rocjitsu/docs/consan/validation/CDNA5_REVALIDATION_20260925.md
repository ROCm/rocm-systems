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
Investigation is in progress. No native hardware trace is claimed.

## Initial unfiltered clean control

`unfiltered-d128-block/` retains independent baseline, Default and SuperCollider
clean runs with all kernels eligible for instrumentation (no allowlist). All
three were accepted. Both engines patched 267/267 applicable access sites;
Default additionally patched 276/276 barriers. Static and dynamic completeness
passed with no unexpected diagnostics. These runs establish yellow cells;
reviewed prospective fault trials remain pending. Future filtered results will
use separate artifact roots and matching clean controls.
