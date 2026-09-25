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

## Filtered campaign

All 21 non-Tensile validation IDs now have successful emulated kernel-dispatch
traces and generated allowlists under `allowlists/gfx1250/`. The profiler agent
CSV identifies gfx1250 (target version 120500), not the host gfx1201.
`discovery-batch/` retains standard SDK discovery; `discovery-pytorch-matched/`
retains PyTorch discovery. `discover.py` and `discover_pytorch.py` preserve the
exact commands and environment construction.

PyTorch must use its matching `_rocm_sdk_core/bin/rocprofv3`, `--rocm-root`
pointing at that package, and its versioned `librocprofiler-sdk-tool.so.1` and
`librocprofiler-sdk.so.1` in the preload pair. Put that package's `lib`,
`lib/llvm/lib`, and `lib/rocm_sysdeps/lib` first in the profiler's library search
path. Using the newer home-venv profiler with PyTorch's older LLVM bundle fails
with duplicate `PointerFlowAnalysisResult` registration. All eight PyTorch
operations successfully profile with the matched bundle.

Filtered clean results are under `filtered-clean/<id>/<profile>/`; each profile
has an independent baseline and matching coverage/oracle checks. Tensile uses
its host driver and an inner emulated client, with discovery performed by
`profile_tensile_client.sh` through `CONSAN_VALIDATION_TENSILE_WRAPPER`.
`discovery-tensile/` includes every selected shard, not just a canary shard.
Actual filtered Tensile clean results use `filtered-clean-tensile/`; the earlier
`filtered-clean/tensile-*` attempt stopped at argument parsing before execution.

D128's reviewed grouped K-publication fault is committed in the maintained
`consan_validation_faults_gfx1250.json`. Pristine disassembly and dry-run inventory
establish the current code hash and two physical publication barrier pairs.
`d128-fault-default/` runs eight precommitted trials per detector using that spec,
with gfx1250 emulator health discovery and dispatch smoke before/after each.

`torch.topk` Default hit the initial 30-second deadline during patching. The
otherwise identical `--timeout 180` retry passed baseline and clean qualification
in `topk-longer-clean/` (instrumented elapsed 50.98 s under campaign load).
The initial timeout remains preserved, and this is not a detector failure.

D128 delay-zero SuperCollider completed all eight admitted/reached trials with
healthy pre/post emulator probes and zero detections. The grouped publication
fault and expectations were fixed before the trials. Delay calibration remains
pending; no trial was replaced or omitted to improve the fraction.

## Issues uncovered by filtered clean runs

- `tree-atomic-or` Default completes its numerical test, but ConSan reports three
  cross-wave partial-store/final-consumer conflicts (return 89). Coverage is
  complete: accesses 32/32, barriers 6/6, atomics 2/2, fences 3/3. The bounded
  atomic-publication journal is currently enabled only for RDNA4 and CDNA4 in
  `consan_observation_policy.cpp` and `consan_sync.inc`; CDNA5 emits zero journal
  events. The missing CDNA5 port is the leading explanation, to be checked by
  regression and workload tests. Its native FLAT/GLOBAL format adds
  `scale_offset` to the RDNA4 layout, so blindly reusing the RDNA4 rewrite is
  insufficient. Earlier unfiltered clean success did not exercise a conflicting
  sampled access pair and is superseded by this failure.
- HipKittens naive BF16 aborts with `corrupted double-linked list` after both
  instrumented profiles. Static/dynamic coverage passes; baseline and profiler
  discovery succeed. The fault reproduces under GDB; retained logs are
  `hipkittens-teardown-gdb*.log`. This is a real clean execution failure, with
  root cause still under investigation.

D128's first Default matrix detected 1/8 admitted/reached faults; delay-zero
SuperCollider detected 0/8. Both matching filtered clean controls pass. The next
Default calibration is `high`, with a new matching clean control and the same
reviewed fault; artifacts are under `d128-high/`.

Tensor-descriptor add also reproduces the teardown heap corruption in both
profiles. HipKittens' full GDB stack reaches `KfdProcess::unmap_pages` while
freeing `LegacyPageTableEntry::host_extents`; this identifies where corruption is
detected, not yet where it originates. `hipkittens-teardown-gdb-full.log`
preserves the stack for follow-up.

`pytorch-scatter-reduce` returns zero and passes its numerical oracle in both
profiles, with no applicable LDS code objects (0/0 accesses). These are yellow
scope-limit cells, not detector failures. All three TP2 IDs now pass filtered
clean controls independently in each profile.

D128 Default qualifies at `high`: matching filtered clean passes and all eight
admitted/reached faults are detected, with complete coverage and healthy probes.
The qualification audit compares clean/fault provenance files and rehashes every
input. `default` had 1/8, so `high` is the lowest passing preset at or above
default. Evidence: `d128-high/`.

### Full-shard Tensile discovery completed

The discovery batch completed ten of twelve families across every shard.
`mxf8f4gemm-tdm` shard 2 and `sgemm-quick` shards 3–5 returned nonzero;
the mixed-format log records profiler finalization on SIGTERM at the client
deadline. Their allowlists are not admitted as complete. These two rows remain
orange while the interrupted shards are retried with longer deadlines.
Evidence: `cdna5-20260925/discovery-tensile/*/result.json` and command logs.

### CDNA5 publication journal activated and clean tree recovered

The CDNA5 target now enables publication modification capture, using its native
return-observation encoding (including the scaled-address bit). The out-of-tree
`tree-atomic-or` filtered Default clean run now passes with complete coverage:
`cdna5-20260925/publication-tree-clean/tree-atomic-or/clean/default/result.json`.
Fault qualification remains pending.

Normal GCC verification: all 779 `ConSan.*` host tests and all 67 gfx1250
atomic/ordering/publication device tests pass. Logs: `publication-host-tests.log`
and `publication-device-retest.log` under the campaign directory. The three
arrival/store/tree positive fixtures initially failed because their shared gfx1250
helper hid its LDS store in inline assembly, preventing compiler wait insertion.
Disassembly confirmed no producer `s_wait_dscnt`; the publication trace correctly
retained the producers without release evidence. Use the compiler-visible LDS
store already used by gfx950/gfx1201 for gfx1250 as well. The three positive and
three broken-address negative controls now pass (`publication-fixture-tests.log`).
A temporary expanded journal print was reverted after diagnosis.

The shared helper change also requires rebuilding and running the rest of the
gfx1250 device fixtures before final qualification. Earlier external results
retain their recorded hook provenance; final qualification must match the updated
hook. The HipKittens/TDM teardown corruption remains a separate open issue.

### Host OOM and campaign memory containment

At 13:11 on September 25 the host hit global OOM and the kernel killed a
VS Code process. The retained `cdna5-20260925/oom-kernel.log` records 581 Python
processes totaling approximately 51 GiB RSS and 13 Tensile clients. Four outer
workload workers, per-workload shard concurrency, and Tensile's unrestricted
internal generation pools compounded. The discovery retry was also concurrent.
The interrupted runs have no accepted results and must not qualify cells.

All subsequent campaign commands, including discovery and builds, must use:

```sh
emulation/rocjitsu/scripts/consan-bounded-run.sh COMMAND [ARGUMENTS...]
```

This wrapper places all invocations in the **same** user systemd slice, with
aggregate `MemoryMax=40G`, `MemoryHigh=36G`, and `MemorySwapMax=1G`.
The limit applies to the campaign, leaving the desktop outside its group.
Parent/child cgroup membership and the effective 42949672960-byte limit were
verified. A limit hit is an infrastructure failure, never detector evidence.
Tensile generation now defaults to two CPU workers per shard (`--cpu-threads`);
outer campaign workload batches are serialized. Do not overlap discovery with
large clean batches. The 38 Tensile validation tests pass with an assertion that
the bounded worker setting reaches the actual Tensile driver invocation.
