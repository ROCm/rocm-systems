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
