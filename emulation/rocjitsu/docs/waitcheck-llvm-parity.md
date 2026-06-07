# Waitcheck LLVM Parity Map

`rj_waitcheck` is an object-code checker for gfx12 wait hazards. It does
not run LLVM's `si-insert-waitcnts` or `post-RA-hazard-rec` passes, so parity is
defined as detecting the same missing waits that LLVM would insert for hazards
that still have enough information in assembled code.

## Current Scope

- Supported targets: `gfx1200`, `gfx1201`, and `gfx1250`.
- Analysis input: decoded executable code-object sections and kernel descriptor
  entry points.
- Output: diagnostics for missing or too-weak waits; no code rewriting.
- Non-goal: proving scheduler-only hazards that depend on pre-RA metadata,
  pseudo instructions, killed operands, or pass pipeline state not present in
  final object code.

## LD_PRELOAD Hook

The runtime prototype builds `librocjitsu_waitcheck.so`. Preload it into an HSA
process to analyze code objects as ROCR creates readers:

```sh
LD_PRELOAD=/path/to/librocjitsu_waitcheck.so ROCJITSU_WAITCHECK_FAIL=1 ./app
```

The shim intercepts `hsa_code_object_reader_create_from_memory`,
`hsa_code_object_reader_create_from_file`, and AMD's offset-size file reader
`hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`.
It also wraps `hsa_system_get_extension_table` and
`hsa_system_get_major_extension_table` so clients that fetch the AMD loader
extension table get the same offset-size reader hook.
`ROCJITSU_WAITCHECK=0` disables the checker. With `ROCJITSU_WAITCHECK_FAIL=1`,
missing waits and supported-gfx12 analysis failures return
`HSA_STATUS_ERROR_INVALID_CODE_OBJECT`; otherwise the shim reports diagnostics to
stderr and chains to the real runtime reader.

## Implemented Coverage

| LLVM source area | LLVM behavior | RocJITsu coverage |
| --- | --- | --- |
| `SIInsertWaitcnts.cpp` gfx12 split counters | Tracks independent `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and `expcnt`. | `WaitcheckTest` covers missing/correct load, store, DS, KM, sample, BVH, and EXP waits. |
| `waitcnt-overflow.mir` gfx12 queue-depth cases | Long outstanding queues require nonzero waits such as `S_WAIT_DSCNT 39` before using the oldest result. | Overflow-sized tests report `required_count=39` for DS, load, store, KM, sample, BVH, and EXP queues. Register-result queues accept the matching nonzero wait for the oldest result, and the DS fixture still flags the second-oldest result as pending. |
| `waitcnt-flat.ll` and flat counter logic | Flat operations may require both VMEM/load and LDS/DS waits. | Flat load/store tests require both split counters and accept combined waits. |
| `waitcnt-global-inv-wb.mir` and `insert-waitcnts-gfx12-wbinv.mir` | `global_inv` increments the load counter; `global_wb` and `global_wbinv` require store waits before later memory operations. | `ReportsMissingLoadcntAfterGlobalInvBeforeMemoryOp`, `ReportsMissingStorecntAfterGlobalWbBeforeMemoryOp`, and `ReportsMissingStorecntAfterGlobalWbinvBeforeMemoryOp`. |
| `waitcnt-sample-out-order.mir` and `waitcnt-sample-waw.mir` | gfx12 separates image load and sample counters and orders WAW cases through the right split counter. | Image load/sample overwrite tests distinguish `loadcnt` from `samplecnt`; ordered sample overwrite is accepted. MSAA loads are tracked by `samplecnt`; returning image atomics are tracked by `loadcnt` plus EXP/source waits before cross-family overwrites. |
| `waitcnt-bvh.mir` | BVH operations have an independent counter and interact with VMEM/sample ordering. | BVH use tests require `bvhcnt`; cross-family WAW tests require `bvhcnt` before VMEM/sample overwrites and `loadcnt` before BVH overwrites VMEM results. |
| `waitcnt-gfx1250.mir` high VGPR cases | `s_set_vgpr_msb` selects the high VGPR bank per operand role, so encoded `v0` can mean `v0`, `v256`, `v512`, or `v768`. | gfx1250 analysis tracks `s_set_vgpr_msb`, remaps role-qualified VGPR refs up to `v1023`, preserves the mode across waits, and conservatively expands refs when CFG predecessors disagree. Tests cover the six object-visible MIR shapes: low/high non-aliasing for `v256` and `v512`, same-bank hazards for `v511`, `v512`, and `v768`, and different-bank `v768`/`v512` non-aliasing. |
| `lds-direct-hazards-gfx12.mir` | LDS direct loads carry embedded wait fields and participate in EXP-style hazards. | `ds_param_load`, `ds_direct_load`, `s_wait_expcnt`, VINTERP embedded `wait_exp`, DSDIR embedded `wait_vm_vsrc`, and DSDIR `wait_va_vdst` VALU/TRANS distance tests cover missing and correct object-visible waits. |
| `valu-read-sgpr-hazard.mir` and `AMDGPUWaitSGPRHazards.cpp` | gfx12 SGPR read hazards require `s_wait_alu` depctr waits; enough `ds_nop`s or eligible memory ops can clear tracked hazards. | Tests cover `depctr_sa_sdst`, `depctr_va_sdst`, `depctr_va_vcc`, four-`ds_nop` culling, SMEM/non-FLAT VMEM culling, and scratch non-culling. |
| `waitcnt-kmcnt-scc-different-block.mir` | SCC writes from barrier operations are KM-counter hazards across blocks. | Barrier signal/wait tests cover same-block and cross-block SCC use/clear behavior. |
| `waitcnt-loop-*.mir` | Backedges and joins can require stricter waits when event order is uncertain. | Object-level CFG tests cover skipped paths, mixed order at joins, and loop-carried DS hazards. |

## Object-Code Parity Boundary

Some LLVM lit tests are still important for compiler coverage, but they do not
define a useful success criterion for a post-link code-object checker. The table
below records how to handle those cases when expanding the RocJITsu corpus.

| LLVM test category | Examples | Waitcheck treatment |
| --- | --- | --- |
| Pass debugging and forced insertion controls | `waitcnt-debug.mir`, `expand-waitcnt-profiling.ll`, `-amdgpu-waitcnt-forcezero`, and `si-insert-waitcnts-*` debug counters. | Not modeled. These tests validate LLVM pass controls and instrumentation, not whether final machine code has a missing wait. |
| MIR-only meta, pseudo, bundle, and debug placement | `waitcnt-meta-instructions.mir`, `waitcnt-skip-meta.mir`, `waitcnt-debug-non-first-terminators.mir`, `hazard-recognizer-meta-insts.mir`, `hazard-pseudo-machineinstrs.mir`, `hazard-kill.mir`, `hazard-in-bundle.mir`, and `hazard-hidden-bundle.mir`. | Not object-code-verifiable after assembly removes pseudo/meta instructions and bundle/debug placement. Add fixtures only if the final bytes still contain a real wait hazard. |
| Wait preservation and redundancy optimization | `waitcnt-preexisting*.mir`, `waitcnt-no-redundant.mir`, and `preserve-user-waitcnt.ll`. | Treat correct final waits as accepted and missing final waits as diagnostics. Do not require the checker to prove whether LLVM preserved, removed, or avoided redundant waits. |
| Scheduler latency and non-waitcnt hazard recognizer cases | `wmma-hazards*.mir`, `wmma-coexecution-valu-hazards.mir`, `trans-forwarding-hazards.mir`, `partial-forwarding-hazards.mir`, `gfx11-sgpr-hazard-latency.mir`, and `hazard-buffer-store-v-interp.mir`. | Mostly out of scope for the gfx12 wait-counter checker because the observable fix may be instruction scheduling, `s_nop`, or `s_delay_alu`, not a waitcnt counter. Promote only cases that leave an object-visible wait-like dependency. |
| Non-gfx12 or non-RDNA4 hazards | GFX9/GFX10/GFX11 hazard files, `mai-hazards-gfx90a.mir`, `mai-hazards-gfx942.mir`, `mai-hazards-gfx950.mir`, and GWS/LDS-DMA tests without a gfx12 wait-counter analogue. | Out of current scope. Revisit when waitcheck grows architecture-specific modes outside `gfx1200`, `gfx1201`, and `gfx1250`. |
| IR/codegen tests that incidentally print waits | `call-waitcnt.ll`, `call-waw-waitcnt.mir`, `insert-waitcnts-crash.ll`, `statepoint-insert-waitcnts.mir`, memory legalizer tests, and other lowering tests whose checks include `s_waitcnt`. | Use only when the final object can be reduced to an explicit correct-wait or missing-wait fixture. The broader lowering behavior belongs to LLVM tests, not the preload checker. |

## Corpus Evidence To Track

Earlier one-off RDNA4 HSACO sweeps exposed useful decoder coverage gaps. A
`D7190002` skip was traced to `gfx1250` target gating and is now covered by a
`v_max_u64` smoke fixture. A `CC350000` skip in `TensileLibrary_gfx1250.co` was
a gfx1250 scaled-WMMA decode gap; the generated decoder now recognizes the
paired `v_wmma_scale_f32_16x16x128_f8f6f4` encoding, and the exact Tensile code
object analyzes with `skipped=0`.

The recursive `rocjitsu-corpus` sweep uses a bounded first-hazard mode:

```sh
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' rj_waitcheck \
  "$HOME/rocjitsu/rocjitsu-corpus/corpus" \
  --recursive --all-code-objects --skip-unsupported --no-fail \
  --max-diagnostics 0 --stop-after-first-diagnostic --summary-only
```

On 2026-06-07 this scanned 520 inputs, skipped 240 inputs without an analyzed
supported gfx12 code object, analyzed 280 code objects, and reported
`diagnostics=>=296` in 36.55 seconds with 378120 KB max RSS.
