# Waitcheck LLVM Parity Map

`rj_waitcheck` is an object-code checker for AMDGPU wait hazards. It does
not run LLVM's `si-insert-waitcnts` or `post-RA-hazard-rec` passes, so parity is
defined as detecting the same missing waits that LLVM would insert for hazards
that still have enough information in assembled code.

## Wait-Strength Auditing

Counter-parity mode uses intact final kernels from a known LLVM-based compiler
pipeline as a one-sided oracle. It does not invoke LLVM and it does not rewrite,
weaken, or remove waits. At each encoded wait instruction or consecutive wait
group, waitcheck snapshots the pending state before applying the wait and
compares each emitted counter value `M` with the weakest value `N` that its
model says is sufficient for the guarded dependency.

Counter values are ordered by strength: lower values wait for more operations.
The comparison is therefore interpreted as follows:

| Relation | Meaning |
| --- | --- |
| `N == M` | Exact agreement with the emitted counter field. |
| `N > M` | LLVM emitted a stronger wait than waitcheck believes is required. This is an under-accounting and potential false-negative candidate. |
| `N < M` | Waitcheck requires a stronger wait than the kernel emits. This is ordinary hazard/false-positive triage, not the parity audit's target. |

A stronger emitted field with no attributable pre-wait dependency is reported
separately as an `unmodeled-emitted-wait`. This commonly identifies compiler
policy waits, explicit user waits, or information that did not survive into
final ISA; it is not silently counted as exact agreement. Consecutive explicit
waits, combined wait encodings, embedded instruction wait fields, and waits
implied by another counter are normalized before comparison. Every finding
retains kernel identity, wait/producer/consumer offsets and instructions, a
stable catalog key, and a single-kernel replay command.

Run a strict corpus audit with:

```sh
rj_waitcheck /path/to/code-objects \
  --target gfx950 --exhaustive --check-counter-parity \
  --counter-parity-jsonl gfx950-counter-parity.jsonl \
  --diagnostics-jsonl gfx950-hazards.jsonl \
  --summary-only --progress -j12 --slowest-kernels 20 --no-fail
```

The final summary distinguishes `parity-exact`, `counter-underaccounting`,
`unmodeled-waits`, and `parity-indeterminate`. A completed validation requires
matching discovered/completed code-object and kernel totals, zero analysis
errors, and individual root-cause classification of every modeled `N > M`
finding. `parity-indeterminate` is never treated as a clean result.

### Reproducible corpus inventory

Use `scripts/inventory_waitcheck_corpus.py` to record where each artifact came
from without checking large framework binaries into the repository. The JSONL
manifest includes project, source revision or package build, provenance,
target, container and extracted-artifact hashes, code-object index, kernel name
and entry offset, and the exact extraction command. Run provenance is emitted
once, each artifact's hashes are emitted once, and code-object/kernel rows
refer to that record by `artifact_id`; this keeps large framework inventories
lossless without repeating package metadata on every kernel:

```sh
scripts/inventory_waitcheck_corpus.py \
  --waitcheck build/tools/rj_waitcheck \
  --project iree --source "$IREE_REV" --provenance llvm \
  --target gfx950 --extract-dir /tmp/iree-gfx950-code-objects \
  --output /tmp/iree-gfx950-manifest.jsonl \
  /path/to/program.vmfb
```

The same command accepts TheRock split-wheel `.kpack` archives when run from an
environment containing `rocm_kpack`. For example, a gfx1250 PyTorch wheel can
be inventoried and extracted directly from
`torch/.kpack/torch_gfx1250.kpack`; each manifest artifact retains the kpack
member name and original code-object ordinal.

Use `--provenance llvm` only when the pipeline that selected the final encoded
wait values is known to be LLVM-based. Assembling generated `.s` source with
LLVM tools does not make its wait choices LLVM-produced; record such artifacts
as `generated-assembly`. Only the former is a valid oracle for modeled `N > M`
triage.

### Durable finding classes

The initial corpus work established these recurring classes:

| Class | Attribution |
| --- | --- |
| Pending physical-register replacement definitions | Waitcheck originally followed only the runtime-visible committed generation. LLVM post-RA accounting follows pending physical definitions; parity mode now does the same. |
| Shared scalar-memory counter ordering | SMEM may complete out of order relative to other events on a shared legacy counter. A nonzero partial wait cannot retire a particular dependency; waitcheck now requires zero. |
| CDNA flat-memory ordering | Flat operations make the relevant legacy VM/LGKM completion order uncertain. Partial waits cannot select a particular result; waitcheck now follows LLVM's zero-wait rule. |
| DS read-result overwrite | LLVM treats a second DS read defining the same VGPR as a pending-definition WAW. Waitcheck's former DS-to-DS ordered-WAW exemption was removed. |
| Loop-header and forward-scheduled waits | A wait may guard a use after independent instructions or on a later CFG path. Bounded lookahead attributes it to the pre-wait event without rebuilding the CFG per counter. |
| Waits at basic-block labels | A label can split a fallthrough block immediately after an emitted wait. The pending parity group now continues into the sole CFG successor instead of becoming indeterminate at the structural boundary. |
| Adjacent non-counter waits | `s_waitcnt_depctr` and `s_wait_event` do not terminate a pending memory-counter wait group. Lookahead continues through them, while `s_wait_idle` is normalized to zero for every supported counter. |
| Barrier safety waits | `SIInsertWaitcnts` may emit all-zero waits around barriers as a target policy for exception safety. These remain explicit `unmodeled-emitted-wait` rows when no final-ISA dependency can be attributed. |
| Acquire-fence soft waits | GFX12 `SIMemoryLegalizer` emits all-zero memory-counter waits before `global_inv` for cross-address-space acquire fences. Parity attributes the object-visible wait group to that memory-order marker without making `global_inv` an ordinary register dependency. |
| Consecutive waits split by a CFG label | A branch target may fall between adjacent waits even though no instruction executes between them. A non-overlapping first counter field now follows the sole fallthrough edge to its guarded dependency; the labeled successor wait is audited once in its own block. |
| Implicit EXEC definitions | GFX12 `v_cmpx_*` instructions encode no explicit destination but still replace EXEC. XCNT parity now treats that implicit definition exactly like an explicit EXEC write. |

### Root-cause catalog

The table below retains the modeled `N > M` classes and completeness gaps that
led to analyzer changes. `W/P/C` gives the representative wait, producer, and
consumer `.text` offsets; the lossless JSONL retains every kernel name and
duplicate site.

| Corpus / target | Root-cause class | Rows / keys and representative evidence | Status and disposition |
| --- | --- | --- | --- |
| PyTorch / `gfx950` | Pending physical-register replacement definitions | 138 rows / 12 keys, `M/N` in `0/1`, `2/3`, or `6/7`; `global_load_dword -> v_mad_u64_u32` at entry `0x145800`, W/P/C `0x146804/0x1467bc/0x146808`. | Resolved, waitcheck fixed. Parity now uses LLVM's pending-definition view; `Gfx950CounterParityMatchesPendingReplacementDefinitionsLikeLlvm` is the minimized regression. |
| PyTorch / `gfx950` | Legacy LGKMCNT with younger scalar memory | 5 rows / 1 key, `M/N=0/1` before the fix; `ds_read_u8 -> v_and_b32_e32` at entry `0x72400`, W/P/C `0x7247c/0x72460/0x72480`. | Resolved, waitcheck fixed. SMEM makes the shared counter out of order, so the corrected requirement is zero and the artifact is exact. |
| IREE and Triton / `gfx950` | Audit stopped at the first guarded dependency | 3 modeled rows / 3 keys, all initially `M/N=0/1`; representative IREE `ds_read2_b64 -> v_mfma_f32_16x16x32_f16` at entry `0x0`, W/P/C `0x7a4/0x794/0x7e8`. | Resolved, waitcheck fixed. The audit now follows the decoded CFG and collects every guarded dependency before the next same-counter wait; a later dependency requires zero. |
| PyTorch / `gfx1201` | Split-barrier DS drains were attributed past the barrier | 163 rows / 20 keys in the 10,773-kernel before-fix prefix, generally `M/N=0/1`. Object 148, entry `0x0`, has a DS producer at `0xe18`, `s_wait_dscnt 0` at `0xe94`, `s_barrier_signal -1` at `0xe98`, and the formerly attributed register overwrite at `0xedc`. | Resolved, waitcheck fixed. A split-barrier signal is the immediate memory-order consumer of outstanding DS events. This follows LLVM's `memory-legalizer-barriers*.ll` and `fence-barrier-latency.ll` sequences; `Gfx1201CounterParityAttributesDsDrainToSplitBarrierSignal` is the minimized final-ISA regression. |
| PyTorch / `gfx1201` | Acquire-fence DS drain was attributed past `global_inv` | 1 row / 1 key, `M/N=0/1`. Object 172, entry `0x34f00`, has an older `ds_load_b64` at `0x353c4`, a younger `ds_store_b64` at `0x35604`, `s_wait_loadcnt_dscnt 0` at `0x3560c`, `global_inv` at `0x35610`, and the formerly attributed overwrite at `0x3562c`. | Resolved, waitcheck fixed. LLVM's GFX12 `SIMemoryLegalizer` emits this all-zero wait plus `global_inv` sequence for a cross-address-space acquire fence. `Gfx1201CounterParityAttributesAcquireFenceWaitBeforeGlobalInv` retains the final-ISA pattern. |
| PyTorch / `gfx1201` | Consecutive split-counter waits crossed a structural CFG label | 315 indeterminate groups in 18 objects. Object 145, entry `0x5e00`, has `s_wait_dscnt 0` at `0x6a04` and adjacent `s_wait_loadcnt 0` at the labeled `0x6a08`. | Resolved, waitcheck fixed. The first non-overlapping field follows the sole fallthrough edge while the successor wait remains independently audited, eliminating both incompleteness and double counting. All 2,640 kernels in the affected objects reran with zero indeterminate groups; `Gfx1201CounterParityHandlesConsecutiveWaitsAcrossCfgLabel` is the regression. |
| PyTorch / `gfx1250` | `v_cmpx_*` implicit EXEC definitions were omitted | 3 rows in masked-scatter kernels, all initially `M/N=0/1`: a global load remained outstanding when `s_wait_xcnt 0` preceded a `v_cmpx_*` instruction. | Resolved, waitcheck fixed. `v_cmpx_*` has no encoded destination operand but implicitly defines EXEC. `Gfx1250CounterParityMatchesXcntBeforeImplicitCmpxExecDef` and the ordinary missing-wait regression cover the final-ISA shape; the focused artifact rerun is exact. |
| PyTorch / `gfx950` | LLVM loop-preheader flush policy | 2 rows / 1 key, `M/N=0/1`; `global_load_dword -> v_cmp_lt_i32_e32` in `__amd_rocclr_batchMemOp`, entry `0x1400`, waits at `0x15ac` and `0x17cc`. | Resolved, compiler-policy exception. LLVM's `waitcnt-vmcnt-loop.mir` documents the deliberately stronger zero flush when a partial preheader flush would suffice. No suppression or analyzer change. |

The full PyTorch `gfx950` sweep found 211 additional modeled rows. They reduce
to the 13 final-ISA catalog keys below. The package is
`torch 2.9.1+rocm7.13.0a20260422`; its code objects identify AMD Clang commit
`43215c73116c407735c85a180d174f718798c328` plus patch set
`2506c552d8428e2cc1778bef048b20f818e06bb3`. All 211 are resolved as safe
compiler conservatism: reduced MIR through current LLVM
`ee5d6f537da4bc81b956b4813afbc0877f82b59d` emits waitcheck's `N`. The three
representative oracles emit `lgkmcnt(5)` for six DS reads, `vmcnt(4)` for five
buffer-load replacement definitions, and `vmcnt(1)` for two scratch loads.

| Catalog-key suffix | Rows | Encoded/required `M/N` | Representative object, entry, W/P/C | Disposition |
| --- | ---: | --- | --- | --- |
| `dscnt/use/acc-vgpr/ds_read2st64_b64/v_mfma_f32_32x32x8_bf16` | 4 | `1/7`, `3/5`, `7/10`, `11/13` | 205, `0x35800`, `0x3c3e8/0x3b30c/0x3c3fc` | Current LLVM agrees with `N`; artifact wait is stronger. |
| `dscnt/use/vgpr/ds_bpermute_b32/v_add_u32_e32` | 3 | `0/1` | 115, `0x31d700`, `0x31dbb0/0x31db98/0x31dbb4` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b32/flat_store_dword` | 8 | `0/1`, `0/5` | 78, `0x6600`, `0x8d90/0x8d0c/0x8f18` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b32/v_accvgpr_write` | 65 | `0/1`, `0/10`, `4/5` | 202, `0x100`, `0x132c/0x12dc/0x1334` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b32/v_mov_b32_e32` | 2 | `2/4` | 202, `0x100`, `0x8f2c/0x8b60/0x90b4` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b64/flat_store_dwordx2` | 58 | `0/1` through `0/6` | 223, `0x14400`, `0x15350/0x15314/0x15374` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b64/global_store_dwordx2` | 8 | `0/1`, `0/2` | 223, `0x14400`, `0x15610/0x153b4/0x15614` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read2st64_b64/v_mfma_f32_32x32x8_bf16` | 2 | `3/4`, `4/5` | 205, `0x1b500`, `0x1ffc4/0x1fb50/0x1ffe4` | Same DS-age conservatism. |
| `dscnt/use/vgpr/ds_read_b64/flat_store_dwordx2` | 48 | `0/1` through `0/6` | 225, `0xa00`, `0x274c/0x2724/0x2770` | Same DS-age conservatism. |
| `loadcnt/def/vgpr/buffer_load_dwordx2/v_accvgpr_read` | 1 | `1/4` | 205, `0x5f200`, `0x62f60/0x629d4/0x63034` | Current LLVM emits `vmcnt(4)`; artifact wait is stronger. |
| `loadcnt/use/vgpr/scratch_load_dword/v_mov_b32_e32` | 4 | `0/1` | 115, `0x174a00`, `0x174be8/0x174aec/0x174c38` | Current LLVM emits `vmcnt(1)`; artifact wait is stronger. |
| `loadcnt/use/vgpr/scratch_load_dwordx2/v_mov_b32_e32` | 1 | `0/1` | 115, `0x149600`, `0x149800/0x1496ec/0x149850` | Same scratch-load conservatism. |
| `loadcnt/use/vgpr/scratch_load_ubyte/v_cmp_lt_i16_e32` | 7 | `0/1` | 115, `0x15c500`, `0x15e8e0/0x15e8c4/0x15e8ec` | Same scratch-load conservatism. |

The full `gfx942` package from the same PyTorch build contributes 237 modeled
rows across 11 keys, all repetitions of those two proven families: DS event
age and scratch-load scheduling. The per-key counts are 1
`ds_read2st64_b64 -> MFMA`, 3 `ds_bpermute_b32 -> add`, 4
`ds_read2st64_b32 -> flat store`, 66 `ds_read2st64_b32 -> ACC write`, 76
`ds_read2st64_b64 -> flat store`, 28 `ds_read2st64_b64 -> global store`, 5
`ds_read2st64_b64 -> MFMA`, 42 `ds_read_b64 -> flat store`, and 4/1/7
scratch dword/dwordx2/ubyte rows. Running the three reduced family oracles for
`gfx942` also emits waitcheck's `N`; no additional semantic class or analyzer
change is needed.

Replay any row directly from the packaged library by substituting its object
and entry above:

```sh
rj_waitcheck libtorch_hip.so --target gfx950 --code-object-index OBJECT \
  --kernel-entry ENTRY --check-counter-parity --no-fail
```

The full gfx1150 PyTorch sweep adds three source-backed dispositions. The
nightly package is `amd-torch-device-gfx1150
2.14.0a0+rocm7.15.0a20260719`; the code objects identify AMD Clang commit
`92fe31231dd83aedaedd822daa2079ca8a445547` in their `.comment` sections.

| Catalog class | Occurrences and representative final ISA | Disposition |
| --- | --- | --- |
| `gfx1150/modeled/dscnt/use/vgpr/ds_load_b32/v_add_nc_u32_e32` | 48 unique edges in 24 rocPRIM 4.5.0 `onesweep_iteration_helper` entries (`M=18`, `N=19`); 432 pre-dedup JSONL rows included 384 descriptor aliases. In object 113, kernel entry `0x7100`, `.text+0xb790` defines `v84`, `.text+0xb844` contains `lgkmcnt(18)`, and `.text+0xb848` uses `v84`. | Safe compiler conservatism. There are 19 younger DS events at the use; a reduced MIR through current `SIInsertWaitcnts` emits `lgkmcnt(19)`, agreeing with waitcheck's `N`. The artifact's value is one step stronger. |
| `gfx1150/modeled/dscnt/use/vgpr/ds_load_b32/v_lshlrev_b32_e32` | 64 unique edges in 32 rocPRIM `onesweep_iteration_helper` entries (`M/N=6/7` or `17/18`); 576 pre-dedup rows included 512 aliases. In object 113, entry `0x71400`, the representative producer/wait/consumer offsets are `0x75cb4`, `0x75d0c`, and `0x75d10`. | Same safe one-step compiler conservatism as the add class. The checked source is `rocprim/device/detail/device_radix_sort.hpp`; no waitcheck change or suppression is warranted. |
| `gfx1150/modeled/loadcnt/use/vgpr/scratch_load_b128/v_dual_mov_b32` | One CK batched-GEMM kernel in object 143, entry `0x8200`: four `scratch_load_b128` operations under `s_clause 0x3`, `vmcnt(2)` at `0xb400`, and a VOPD use of `v[24:25]` at `0xb404`; waitcheck computes `N=3`. | Safe compiler conservatism. A reduced four-load MIR through current `SIInsertWaitcnts` emits `vmcnt(3)`. The exact encoded sequence is retained by `Gfx1150CounterParityCapturesConservativeScratchClauseWait`. |
| CK WMMA GEMM overflow-distance VMEM uses | 20 ordinary hazards in six folded kernel entries across objects 135 and 138. Each producer has 61 younger VMEM events at an earlier `vmcnt(62)` and eight more loads before its use, so the producer may still be pending at saturated age 62. | Compiler-output hazard candidate in the recorded LLVM revision, not a parity false-negative. A reduced 70-load true16 MIR through current `SIInsertWaitcnts` inserts a second `vmcnt(62)` before the use; the artifacts do not. `Gfx1150ReportsOldLoadAfterMaximumPartialWaitAndNewLoads` locks the final-ISA accounting. |

The full PyTorch `gfx1100` sweep found 1,520 modeled rows across 37 final-ISA
keys. They reduce to three semantic classes. The package is
`torch 2.9.1+rocm7.13.0a20260422` and identifies the same AMD Clang
`43215c73116c407735c85a180d174f718798c328` plus patch set
`2506c552d8428e2cc1778bef048b20f818e06bb3` as the gfx950 package. Reduced
MIR through current LLVM `ee5d6f537da4bc81b956b4813afbc0877f82b59d`
agrees with waitcheck's `N` in all three classes:

| Semantic class | Rows / keys and representative final ISA | Disposition |
| --- | --- | --- |
| Paired D16 scheduling | 1,322 rows / 32 keys. Object 139, entry `0x0`, has `ds_load_u8_d16_hi v0` at `0x4e8`, `lgkmcnt(0)` at `0x4f8`, and `v_and_b16 ... v0` at `0x504`; representative `M/N=0/1`. The same zero wait is coupled to a following opposite-half D16 load, so nearby non-D16 producers also deduplicate into this class. | Safe compiler conservatism. A true-16 MIR reduction makes current `SIInsertWaitcnts` emit `lgkmcnt(1)`. `Gfx1100CounterParityCatalogsConservativePairedD16Wait` retains the exact final-ISA shape. |
| DS event-age scheduling | 112 rows / 2 keys, `M/N=18/19` or `17/18`. Object 2, entry `0x7900`, has `ds_load_b32 v84` at `0xbf08`, `lgkmcnt(18)` at `0xbfbc`, and `v_add_nc_u32_e32 ... v84` at `0xbfc0`. | Safe one-step compiler conservatism, identical to the gfx1150 rocPRIM class. Current LLVM emits waitcheck's `N`. |
| Scratch reload scheduling | 86 rows / 3 keys, with `M/N` values `1/4`, `2/5`, `4/5`, `4/6`, `6/7`, and `8/9`. Object 150, entry `0x131500`, has `scratch_load_b128 v[4:7]` at `0x13b400`, `vmcnt(4)` at `0x13b480`, and a VOPD overwrite at `0x13b494`. | Safe compiler conservatism. An eight-load MIR reduction makes current LLVM emit the dependency ages computed by waitcheck (`vmcnt(7)` for the oldest result and `vmcnt(6)` for the next group), rather than the artifact's coarser reload schedule. |

The full `gfx1201` sweep found 95 modeled rows after fixing the split-barrier,
acquire-fence, and cross-label completeness defects above. Ninety-four are
scratch-reload scheduling instances of the already reduced `gfx1100` class:
`scratch_load_b32/b128` results are consumed or replacement-defined after an
encoded wait that is one to three steps stronger than the dependency age
computed by waitcheck. The remaining row is a control-flow join in object 39,
entry `0x4bc200`: `global_load_u16 v15` at `.text+0x4c1d40`,
`s_wait_loadcnt 0` at `0x4c2370`, and `global_store_b16 ... v15` at
`0x4c26d8` produce `M/N=0/1`. One predecessor has already waited for `v15`,
while the other reaches the join with it pending; a partial wait is sufficient
on the latter path. The all-zero join wait is therefore safe compiler
conservatism, not waitcheck under-accounting. The completed rerun has zero
indeterminate groups, ordinary hazards, or analysis errors.

The full `gfx1250` split-wheel sweep used
`amd-torch-device-gfx1250 2.11.0+rocm7.15.0a20260717`; the code objects identify
AMD Clang commit `723bffa5dfbf92e452b0d4a0df674bdd849fcf12`. The original
`.kpack` contains 232 code objects and 41,124 descriptor-backed kernels. After
the implicit-EXEC fix above, its 209 modeled rows reduce to four conservative
scheduling classes:

| Semantic class | Rows / keys and representative final ISA | Disposition |
| --- | --- | --- |
| Scratch-store XCNT source lifetime | 194 rows / 2 keys in warp-top-k spill code. Representative object 167, entry `0x123e00`, has four `scratch_store_b128` operations, `s_wait_xcnt 0`, then a replacement definition of a source from the second store (`M/N=0/2`). | Safe compiler conservatism. A reduced four-store MIR through current `SIInsertWaitcnts` emits `s_wait_xcnt 3` before replacing the oldest source, exactly the age computed by waitcheck. The artifacts use stronger zero or smaller partial values; `Gfx1250CounterParityCatalogsConservativeScratchStoreXcnt` retains the final-ISA shape. |
| Global-load XCNT address/EXEC lifetime | 13 rows / 9 keys in elementwise and collective kernels, with `M/N` from `0/1` through `1/6`. Representative global-load clauses overwrite an address VGPR after an earlier all-zero XCNT wait. | Safe compiler conservatism. A reduced four-load clause through current `SIInsertWaitcnts` emits `s_wait_xcnt 3`, matching waitcheck's event age. `Gfx1250CounterParityCatalogsConservativeXcntClauseDrain` retains the final-ISA policy shape. |
| Scratch-load replacement definition | 1 row / 1 key in warp-top-k: `scratch_load_b128 v[80:83]`, `s_wait_loadcnt 8`, then a VOPD replacement of `v[82:83]`; `M/N=8/15`. | Safe compiler conservatism. A reduced 16-load MIR makes current `SIInsertWaitcnts` emit `s_wait_loadcnt 15` before replacing the oldest result, agreeing with waitcheck; `Gfx1250CounterParityCatalogsConservativeScratchLoadReplacement` retains the final-ISA shape. |
| Control-flow load flush | 1 row / 1 key in a double-precision floor-divide elementwise kernel: an older `global_load_b64 v[12:13]`, a control-flow-scheduled `s_wait_loadcnt 0`, and a later `global_store_b64` of that result produce `M/N=0/1`. | Safe compiler conservatism. LLVM's current `wait-xcnt.mir` oracle for two contiguous loads followed by a store of the older result emits `s_wait_loadcnt 1`, agreeing with waitcheck; the artifact's control-flow flush is stronger. |

The completed run reports 7,708,884 audited fields, 6,183,582 exact fields,
209 modeled `N > M` rows, 1,267,530 unmodeled policy fields, zero
indeterminate groups, and zero analysis errors. The 19,426 ordinary waitcheck
diagnostics are deliberately reported separately: they are not parity
under-accounting findings and are outside this one-sided audit's disposition.
The manifest was regenerated directly from `torch_gfx1250.kpack`; 231 extracted
hashes match the prior sweep inputs byte-for-byte, and the one packaging-metadata
variant has identical kernel, instruction, event, diagnostic, and parity totals.

For example, the rocPRIM class is replayed with:

```sh
rj_waitcheck 113__lib__libtorch_hip.so__2__gfx1150.co \
  --target gfx1150 --code-object-index 0 --kernel-entry 0x7100 \
  --check-counter-parity --no-fail
```

The JSONL catalog deliberately keeps every edge, while triage deduplicates by
target, entry, counter, wait offset, producer, and consumer. Kernel descriptors
that alias the same entry and wave mode are analyzed once; the packaged
`libtorch_hip.so` contains many such identical-code-folding aliases.

The minimized final-ISA fixtures in `tests/analysis/waitcheck_test.cpp` cover
these classes plus RAW/WAW/WAR dependencies, CFG joins and loops, ordered-queue
overflow, combined and consecutive wait groups, X_CNT and expert-scheduling
waits, embedded/implied waits, calls, and barriers. Their source patterns are
derived from `SIInsertWaitcnts`, `SIMemoryLegalizer`, AMDGPU hazard tests, and
the IREE, CK, Triton, and PyTorch corpus findings summarized here; the checked-in
bytes are stable test inputs and normal test execution does not require LLVM.

Triage each modeled finding in this order:

1. Deduplicate by `catalog_key`, target, and final producer/wait/consumer ISA.
2. Run the JSONL `repro_command` to confirm the single-kernel result.
3. Inspect decoded operands, event classification, counter aging, pending-state
   joins, CFG feasibility, and the original kernel source when available.
4. Assume under-accounting is a waitcheck defect until the final-ISA evidence
   establishes a compiler-policy or provenance exception.
5. Only after isolating a compact repro, selectively inspect LLVM wait logic or
   intermediate pass state to confirm attribution.
6. Fix analyzer or decoder defects with a minimized final-ISA regression and
   rerun the originating intact corpus. Never suppress a class merely to make a
   sweep clean.

### Validation snapshots

These provenance-backed intact-kernel sweeps establish the baseline:

| Corpus | Target | Kernels | Exact fields | Modeled `N > M` and disposition | Unmodeled policy fields | Ordinary hazards / errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IREE VMFB convolution, matmul, math, and helpers | `gfx950` | 91 | 578 | 0 | 0 | 0 / 0 |
| MIOpen CK grouped convolution | `gfx1150` | 13 | 449 | 0 | 0 | 0 / 0 |
| Triton matmul and flash-attention fixtures | `gfx942` | 4 | 139 | 0 | 6 barrier-policy waits | 0 / 0 |
| Triton matmul and flash-attention fixtures | `gfx950` | 4 | 174 | 0 | 6 barrier-policy waits | 0 / 0 |
| Full PyTorch library | `gfx942` | 24,508 | 4,627,029 | 237, all resolved compiler conservatism | 46,360 | 0 / 0 |
| Full PyTorch library | `gfx950` | 24,448 | 4,494,494 | 211, all resolved compiler conservatism | 41,673 | 0 / 0 |
| Full PyTorch library | `gfx1100` | 24,544 | 4,569,962 | 1,520, all resolved compiler conservatism | 57,905 | 0 / 0 |
| Full PyTorch library | `gfx1150` | 25,938 | 3,307,745 | 113, all resolved compiler conservatism | 36,086 | 20 candidates / 0 |
| Full PyTorch library | `gfx1201` | 24,544 | 4,559,409 | 95, all resolved compiler conservatism | 433,630 | 0 / 0 |
| Full PyTorch split wheel | `gfx1250` | 41,124 | 6,183,582 | 209, all resolved compiler conservatism | 1,267,530 | 19,426 untriaged / 0 |

The exact column intentionally excludes `N < M` fields, where waitcheck asks
for a stronger wait than the compiler emitted. Those are outside this
one-sided false-negative audit and commonly reflect source-level alias or
memory information unavailable in final ISA; ordinary hazard triage remains
the authority for that direction.

The Triton flash-attention fixtures cover reduction, shared-memory,
software-pipelined patterns, and elementwise epilogues in addition to the
explicit matmul fixtures. The PyTorch rows are LLVM-produced captures, not the
generated-assembly hipBLASLt
or Tensile libraries shipped alongside the wheels; those are useful ordinary
waitcheck inputs but are not LLVM wait-strength oracles.

### Performance snapshot

An alternating two-pair benchmark on the intact 336-kernel gfx950 PyTorch
object used `-j12`, `--exhaustive`, and no JSONL writer so that it measures the
analysis itself. Ordinary mode averaged 63.87 seconds and parity mode averaged
67.60 seconds, a 5.8% wall-time increase. Average peak RSS rose from 9,411,122
KiB to 9,641,592 KiB (2.4%). Both modes analyzed the same 1,804,537
instructions and 117,788 memory events and reported zero ordinary diagnostics
or analysis errors in both runs.

Parity is computed inside the existing per-kernel decode and CFG analysis; it
does not rebuild either per wait. Corpus mode uses the shared bounded thread
pool (maximum 16 workers), one-kernel work items for lossless writers, a 16 MiB
per-analysis reachability-cache bound, periodic allocator trimming, and the
existing slowest-kernel report. Ordinary analysis follows the same path with
parity collection disabled, as the identical benchmark totals demonstrate.

## Current Scope

- Supported targets: `gfx942`, `gfx950`, `gfx1100`, `gfx1150`, `gfx1151`,
  `gfx1200`, `gfx1201`, and `gfx1250`.
- Analysis input: decoded executable code-object sections and kernel descriptor
  entry points.
- Output: diagnostics for missing or too-weak waits; no code rewriting.
- Non-goal: proving scheduler-only hazards that depend on pre-RA metadata,
  pseudo instructions, killed operands, or pass pipeline state not present in
  final object code.

## HSA Tools Hook

The runtime prototype builds `librocjitsu_waitcheck_hooks.so`. Load it through
ROCR's environment-driven HSA tools path:

```sh
HSA_TOOLS_DISABLE_REGISTER=1 \
HSA_TOOLS_LIB=/path/to/librocjitsu_waitcheck_hooks.so \
ROCJITSU_WAITCHECK_FAIL=1 ./app
```

The tool patches `hsa_code_object_reader_create_from_memory`,
`hsa_code_object_reader_create_from_file`, and AMD's offset-size file reader
`hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`.
It also wraps `hsa_system_get_extension_table` and
`hsa_system_get_major_extension_table` so clients that fetch the AMD loader
extension table get the same offset-size reader hook. Analysis runs from the
`hsa_executable_load_agent_code_object` wrapper, so a waitcheck tool installed
before DBT or DBI sees the final replacement reader passed toward ROCR.
`ROCJITSU_WAITCHECK=0` disables the checker. With `ROCJITSU_WAITCHECK_FAIL=1`,
missing waits and supported-target analysis failures return
`HSA_STATUS_ERROR_INVALID_CODE_OBJECT`; otherwise the tool reports diagnostics to
stderr and chains to the real runtime reader.

## Implemented Coverage

| LLVM source area | LLVM behavior | RocJITsu coverage |
| --- | --- | --- |
| `SIInsertWaitcnts.cpp` gfx12 split counters | Tracks independent `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and `expcnt`. | `WaitcheckTest` covers missing/correct load, store, DS, KM, sample, BVH, and EXP waits. |
| `waitcnt-overflow.mir` gfx12 queue-depth cases | Long outstanding ordered queues require nonzero waits such as `S_WAIT_DSCNT 39` before using the oldest result. Scalar-memory requests are different: they may return out of order and require `S_WAIT_KMCNT 0` before consuming a particular result. | Overflow-sized tests report `required_count=39` for ordered DS, load, store, sample, BVH, and EXP queues. The SMEM fixture requires `kmcnt(0)` and proves that a nonzero KMCNT does not retire the oldest scalar result. |
| `waitcnt-flat.ll` and flat counter logic | Flat operations may require both VMEM/load and LDS/DS waits. | Flat load/store tests require both split counters and accept combined waits. |
| `waitcnt-global-inv-wb.mir` and `insert-waitcnts-gfx12-wbinv.mir` | `global_inv` increments the load counter and changes the threshold needed for an older load; `global_wb` and `global_wbinv` increment the store counter. The waits in the latter MIR test originate as explicit soft waits, rather than implicit dependencies on every later memory instruction. | Cache-control events participate in counter accounting without inventing object-level memory dependencies. `GlobalInvContributesToLoadcntThreshold` and `AcceptsAdjustedLoadcntThresholdWithGlobalInv` cover the observable `global_inv` dependency. |
| `waitcnt-sample-out-order.mir` and `waitcnt-sample-waw.mir` | gfx12 separates image load and sample counters and orders WAW cases through the right split counter. | Image load/sample overwrite tests distinguish `loadcnt` from `samplecnt`; ordered sample overwrite is accepted. MSAA loads are tracked by `samplecnt`; returning image atomics are tracked by `loadcnt` plus EXP/source waits before cross-family overwrites. |
| `waitcnt-bvh.mir` | BVH operations have an independent counter and interact with VMEM/sample ordering. | BVH use tests require `bvhcnt`; cross-family WAW tests require `bvhcnt` before VMEM/sample overwrites and `loadcnt` before BVH overwrites VMEM results. |
| `waitcnt-gfx1250.mir` high VGPR cases | `s_set_vgpr_msb` selects the high VGPR bank per operand role, so encoded `v0` can mean `v0`, `v256`, `v512`, or `v768`. | gfx1250 analysis tracks `s_set_vgpr_msb`, remaps role-qualified VGPR refs up to `v1023`, preserves the mode across waits, and conservatively expands refs when CFG predecessors disagree. Tests cover the six object-visible MIR shapes: low/high non-aliasing for `v256` and `v512`, same-bank hazards for `v511`, `v512`, and `v768`, and different-bank `v768`/`v512` non-aliasing. |
| `lds-direct-hazards-gfx12.mir` | LDS direct loads carry embedded wait fields and participate in EXP-style hazards. | `ds_param_load`, `ds_direct_load`, `s_wait_expcnt`, VINTERP embedded `wait_exp`, DSDIR embedded `wait_vm_vsrc`, and DSDIR `wait_va_vdst` VALU/TRANS distance tests cover missing and correct object-visible waits. |
| `valu-read-sgpr-hazard.mir`, `AMDGPUWaitSGPRHazards.cpp`, and `AMDGPUInsertDelayAlu.cpp` | gfx12 SGPR read hazards require `s_wait_alu` depctr waits; enough `ds_nop`s, eligible memory ops, or a hardware SALU issue stall can clear tracked hazards. Calls, returns, and indirect branches require every outstanding depctr field at the transfer boundary. | Tests cover `depctr_sa_sdst`, `depctr_va_sdst`, `depctr_va_vcc`, four-`ds_nop` culling, SMEM/non-FLAT VMEM culling, scratch non-culling, direct and indirect calls, matching returns, resolved and unresolved indirect branches, link-register definitions, non-entry functions, and symbol-less sections. Linked objects do not retain LLVM's boundary-cull option, so waitcheck models its default disabled behavior. SALU issue stalls follow the hardware model in `AMDGPUInsertDelayAlu` rather than treating every SALU data use as a missing depctr wait. |
| `waitcnt-kmcnt-scc-different-block.mir` | SCC writes from barrier operations are KM-counter hazards across blocks. | Barrier signal/wait tests cover same-block and cross-block SCC use/clear behavior. |
| `waitcnt-loop-*.mir` | Backedges and joins can require stricter waits when event order is uncertain. | Object-level CFG tests cover skipped paths, mixed order at joins, and loop-carried DS hazards. |
| CDNA3 and RDNA3 generated-code validation | LLVM-generated kernels must use the target's actual legacy wait layout, not only pass instruction-sequence unit tests. | `RjWaitcheck.LlvmKernel.*` compiles clean HIP vector-add and deliberately wait-perturbed HIP kernels for gfx942 and gfx1100. The clean objects report zero diagnostics; each perturbed object reports the missing `lgkmcnt(0)` wait. |

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
| Hazards without an object-visible wait-counter analogue | GFX9/GFX10 and scheduler-only GFX11 hazard files, `mai-hazards-gfx90a.mir`, and GWS/LDS-DMA tests without a modeled wait-counter analogue. | Out of current scope. Legacy wait-counter analysis is supported for gfx942, gfx950, gfx1100, gfx1150, and gfx1151, but spacing and scheduling hazards remain outside waitcheck. |
| IR/codegen tests that incidentally print waits | `call-waitcnt.ll`, `call-waw-waitcnt.mir`, `insert-waitcnts-crash.ll`, `statepoint-insert-waitcnts.mir`, memory legalizer tests, and other lowering tests whose checks include `s_waitcnt`. | Use only when the final object can be reduced to an explicit correct-wait or missing-wait fixture. The broader lowering behavior belongs to LLVM tests, not the runtime checker. |

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
