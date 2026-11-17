# rj_isa_diff: incremental ISA-to-ISA differential testing

`rj_isa_diff` compares an old and a new compile of the same AMDGPU kernel or callable
function. It finds
where the machine code changed, cuts a contiguous region around each change, and
writes an explicit entry/exit contract for that region. It then runs both versions of
the region on paired inputs through RocJITsu's own instruction execution
(`ComputeUnitCore::execute_instruction`; memory instructions take the CU's issue path,
`ComputeUnitCore::step()`, against a modelled memory). Each case executes **only the
region**; the whole kernel is decoded and analysed once.

Scope: every GPU RocJITsu executes (see "Targets": CDNA1-4, RDNA1-4, gfx1250/gfx1251), in
wave64 or wave32, regions with one entry: straight-line code, or code with direct
branches and loops whose exits correspond on both sides (see "Control flow"), made of:
- SALU instructions on SGPRs, VCC, EXEC, M0 and SCC (an explicit allowlist with every
  generation's names, SALU float of GFX11.5+ included);
- VALU instructions, including MFMA/SMFMAC and AGPR moves (`v_accvgpr_*`) on CDNA (see
  "AGPRs and MFMA"); not interpolation, indirect VGPR access or exception state;
- memory instructions (see "Memory"): loads, stores and atomics of global, flat,
  scratch and buffer memory and of LDS (`ds_read*`/`ds_load*`, `ds_write*`/`ds_store*`
  including the two-address forms, DS atomics), scalar loads, and every counter wait
  (`s_waitcnt*`, `s_wait_*cnt`), and the LDS transpose loads (`ds_read_b64_tr_b16`,
  `_tr_b8`, `_tr_b4`, `ds_read_b96_tr_b6`: each lane reads at its own address and the LDS
  unit redistributes the elements across the lanes; a region issuing one under the entry
  EXEC gets a full EXEC, and an input address of a B64 form is eight-byte aligned, as the
  hardware requires). Not the LDS-DMA, global transpose, `block`, `addtid`, typed-format,
  GWS, ordered-count or `ds_swizzle` forms;
- the lane permutes `ds_bpermute_b32` and `ds_permute_b32` (through the LDS crossbar, no
  LDS memory; issued like LDS instructions, so `lgkmcnt` counts them; a backward permute
  reads only active source lanes);
- scheduling hints without a functional effect (`s_nop`, `s_delay_alu`, `s_clause`,
  `s_wait_alu`, `s_waitcnt_depctr`, prefetch hints). Alignment matches the other
  instructions first and hints only in the gaps between them; a hunk of hints alone
  (change kind `hint`) is not tested, and a function whose only differences are hints is
  `NO_SEMANTIC_DIFF`. Hardware hazards these hints avoid are not modelled, so not checked;
- direct branches (`s_branch`, `s_cbranch_scc0/1`, `vccz/vccnz`, `execz/execnz`), an
  early `s_endpgm` (see "Control flow");
- `s_barrier`, as a barrier of a one-wave workgroup whose shared-memory accesses keep their
  barrier epoch (see "Control flow"). Not a hint: adding, removing or moving one is a
  change.

The kernel around a region may contain anything; the region may not. Regions may also contain the comment-only
inline-asm blocks of LLVM tests (`; def v[0:1]`, `; use s4`; see "Inline asm").

Inputs are AMDGPU code objects, raw instruction words, and, through the LLVM plugin
(`llvm/`), `.s` and `.ll` sources. The core never runs LLVM; the plugin is a separate
library that turns sources into code objects and records how. A kernel is found through
its descriptor; a function symbol without a descriptor is a callable function (see
"Callable functions").

To test an LLVM change, `rj_isa_diff_pr` does everything in one command: it reads the
old and new assembly of every function the change affects from the CHECK lines of the
AMDGPU lit tests the change regenerated, and runs `analyze` and `test` on each (see
"Testing an LLVM change").

## Commands

```sh
# 0. One command for an LLVM change (base and head revisions of an llvm-project checkout).
rj_isa_diff_pr --repo ~/llvm-project --base BASE --head HEAD --out report/
#   --pr N             fetch GitHub PR N and test it against its merge base
#   --head WORKTREE    test the checkout's uncommitted (regenerated) tests
#   [--cases N] [--jobs N] [--tests GLOB] [--proxy-targets LIST] [--llvm-bin DIR] [--list]
#   [--strict-nan] [--no-poison-triage]
#   [--mc-before LLVM_MC --mc-after LLVM_MC]  each side's own assembler (see below)

# 1. Decode, align, classify, plan regions and contracts.
rj_isa_diff analyze --old old.hsaco --new new.hsaco --kernel my_kernel --out case/
#   [--context N]       execute N common-prefix instructions before the change
#   [--max-region N]    region length limit (default 256)
#   [--max-expansions N]  common instructions added per direction by variants (64)
#   [--no-merge]        do not merge hunks that share a basic block
#   [--no-derive-constraints]  no known-bits constraints from the common prefix
#   [--max-analysis-ms N]  planning time limit (stop rule); hitting it => UNRESOLVED
#   [--manual-region old:B-E,new:B-E]  hand-written region (at most ASSUMED)
#     [--map s4=s8,...] | [--entry-map ...] [--exit-map ...]  register correspondence
#   [--diagnostic-entry-prefix]  also build entry-prefix diagnostic regions
#   [--cache-dir DIR]   reuse an identical earlier analysis
#   [--old-kernel/--new-kernel NAME] when the symbol name differs
#   callable functions: [--return-vgprs N] (0..32, default 32) [--function-mode HEX]
#     [--return-masks M0,M1,...] (bits of each return VGPR)  [--entry-vgprs N]
#     [--entry-sgprs N] (defined argument registers; others are undefined until written)
#   .s/.ll inputs: [--llvm-bin DIR] [--llc-args S] [--old-llc-args S] [--new-llc-args S]
#     [--old-llvm-bin DIR] [--new-llvm-bin DIR] (per-side llc/llvm-mc over --llvm-bin)

# 1b. LLVM plugin: compile one .ll twice, with different llc options per side.
rj_isa_diff llvm-pair --input k.ll --common-args "-O3" --old-args "" \
    --new-args "-combiner-disabled" --out case/   # + any analyze option

# 2. Paired-input differential test of every READY region.
rj_isa_diff test --case case/ --seed 1 [--cases 1000] [--region r0]
#   the boundary/NaN/corner sweep always runs first; --cases adds random cases
#   [--max-boundary-cases N]  safety cap (default 250000); hitting it => PARTIAL
#   [--reduction-budget N | --no-reduce]  counterexample reduction (default 4096 runs)
#   [--no-case-log]  skip the per-case JSONL learning log (case/logs/<region>.jsonl)
#   [--cache-dir DIR]  reuse results of identical prepared regions and budgets
#   [--strict-nan]     compare NaNs bit for bit (default: same-format NaNs are equal)
#   [--poison-filter shift,fptoint,ctz,nsw,nuw,disjoint,exact,nnan,ninf|all]
#   after a local mismatch, precomputed larger regions (common context) are tried
#   automatically

# 3. Re-execute a counterexample from its recorded words (no artifact, no LLVM).
rj_isa_diff replay --counterexample case/counterexamples/r0-case17.json [--case case/]

# Validator self-test: inject a wrong opcode/immediate/operand (labelled synthetic).
rj_isa_diff inject --case case/ --region r0 --side new --inst 1 --xor 0x01000000 --out syn/

# Mutation analysis: how strong is a pass? (synthetic; see "Mutation analysis")
rj_isa_diff mutants --case case/ --out mutants.json [--max 48] [--cases 1000]
rj_isa_diff_mutants RESULTS_DIR... --out DIR [--jobs N]   # over rj_isa_diff_pr results
```

An input file is an ELF code object (detected by its magic), a `.s` or `.ll` source
(LLVM plugin), or a raw little-endian instruction stream treated as one kernel starting
at offset 0.

Exit codes:

| code | meaning |
|---|---|
| 0 | `NO_MISMATCH_FOUND` (or `NO_SEMANTIC_DIFF`), coverage `COMPLETE_FOR_DECLARED_SCOPE`, and no case needed the per-bit don't-care mask (`dont_care_masked_cases` 0) |
| 1 | `LOCAL_MISMATCH` (`replay`: did not reproduce) |
| 2 | `INCONCLUSIVE`, `UNSUPPORTED`, partial coverage, or a pass that needed the per-bit don't-care mask |
| 3 | harness error or bad usage |

## What a result means

Each evidence level is reported in its own field (`results.json`, `overall` and per region):

| field | values |
|---|---|
| `test_status` | `NO_MISMATCH_FOUND`, `LOCAL_MISMATCH`, `INCONCLUSIVE`, `UNSUPPORTED`, `HARNESS_ERROR` (`NO_SEMANTIC_DIFF` overall only) |
| `contract_status` | `ESTABLISHED`, `ASSUMED`, `UNRESOLVED` |
| `witness_scope` | `NONE`, `LOCAL_ONLY` (P0 never claims more) |
| `coverage_scope` | `COMPLETE_FOR_DECLARED_SCOPE`, `PARTIAL`, `NOT_ESTABLISHED` |
| `proof_status` | always `NOT_RUN` |

- `NO_MISMATCH_FOUND` means only that no counterexample was found under the given seed,
  budget and contract. It is not proven equivalence.
- A `LOCAL_MISMATCH` is a region-level reproducer under the recorded contract. The
  kernel/context reproducer is pending, because reachability of the inputs is not
  established in P0. It is never labelled an LLVM bug.
- A mismatch found under an `UNRESOLVED` contract does not make the overall result
  `LOCAL_MISMATCH`, because the relation itself may be what is wrong.
- Zero valid cases never count as a pass.
- When a contract carries constraints derived from the common prefix and the region
  passes, it is also run without them (stopping at the first mismatch), and the result's
  `unrestricted` field reports that run. The constrained result is the verdict;
  `constraint_dependent: true` says the rewrite is only correct under the derived facts.

## The contract

For each region pair (`case.json` → `regions[].contract`):

- **R_in**: equality on `entry_equal`, the union of both sides' live-ins at the
  cutpoint. It compares the same physical location on both sides; no renaming is
  inferred. Every other location (`canary`) gets an independent value per side, so a
  dependency that liveness missed shows up as a mismatch.
- **C**: `constraints` (known bits, unsigned range, equality), each with its source. A
  constraint written by hand is `user_assumption`, and editing a case after analysis
  downgrades `ESTABLISHED` to `ASSUMED`. The planner derives known bits for entry
  locations from the common prefix (`common_prefix_analysis`): a forward must-analysis
  over constants, masks, shifts, `s_bfe` and min/max, in which every write by changed
  code is unknown. These constraints never come from the changed code itself.
- **R_out**: equality on `observable`, the union of both continuations' live-ins. If
  the two continuations read different locations, the region is
  `INCONCLUSIVE_MAPPING` instead of guessing a mapping.
- **Frame**: STATUS (except SCC), MODE, TRAPSTS, TTMP, scratch base, wait counters,
  and trap/wave state are fixed and must be preserved. The sticky arithmetic exception
  flags (TRAPSTS.EXCP bits 0-6) are excluded: VALU arithmetic sets them, and LLVM's
  default floating-point environment gives them no meaning.
- **Effect check**: each side may change only its `declared_writes_*`. Any other
  change is a `HARNESS_ERROR`, because it means the effect model is wrong.
- **Pending effects** (`pending_effects`): memory operations that may still be in flight,
  from an `s_waitcnt` dataflow over the gfx9 counters.
  - VMEM completes in order, so `vmcnt(N)` retires all but the newest N vector memory
    operations; so does LDS, so `lgkmcnt(N)` retires all but the newest N LDS operations
    (software-pipelined kernels wait with N > 0). SMEM may complete out of order, so while
    a scalar load is outstanding only `lgkmcnt(0)` clears that counter's destinations.
  - The check runs before every region instruction, so a wait inside the region retires
    what it waits for.
  - `NONE` and `DISJOINT` (in flight, but no destination is read or written by the region)
    leave the contract unaffected.
  - `CONFLICT` and `UNKNOWN` leave the context unresolved: the isolated run completes
    every access at once, and real hardware might not.
- **Composition** (`composition`): a forward must-equality analysis from the kernel entry,
  through common code and upstream regions' R_out, must establish every `entry_equal`
  location (`HOLDS`). Otherwise it reports `FAILS`, naming the missing locations.
- **Status**:
  - `ESTABLISHED` when every path to the region runs byte-identical code from identical
    metadata.
  - `ASSUMED` when R_in relies on upstream regions' R_out, on untested upstream changes
    writing only their decoded destinations, or on an unchecked single-entry assumption
    (unknown CFG successors).
  - `UNRESOLVED` when kernel metadata differs (except register allocation, see
    "Register allocation"), a control-flow change can reach the entry, composition fails,
    or pending effects conflict. An unresolved context is also listed
    in `unresolved`, so the overall result is never a pass.

## Context expansion

A local mismatch may be compensated by common code around the change (handoff 5.2).
`analyze` precomputes up to 12 larger variants of each READY region, stored in
`case.json` under `expansion_variants`. Each variant adds matched common P0 instructions
from the same block, before and/or after the change (`r0.ctx-b<before>f<after>`). Each
has its own contract. When there are more shapes than variants, the largest contexts
(all common code forward, and both ways) are kept besides the smallest ones, because the
code that compensates a difference is often further away than a few instructions.

After a base region shows `LOCAL_MISMATCH`, `test` runs these variants smallest first. A
variant resolves only by passing, so it stops at its first mismatch, without reduction
(a register renaming under the identity mapping fails every variant at once). The base
result keeps its local mismatch, and `expansion.resolution` records the outcome:
- `RESOLVED_BY_CONTEXT`: a variant has no mismatch, and the overall verdict uses that
  variant's result and contract;
- `PERSISTS_WITH_CONTEXT`;
- `NO_EXPANSION_AVAILABLE`.

`--context N` still adds manual backward context to every base region.

## Control flow

A region may contain direct branches and span basic blocks (including control-flow
hunks: a changed condition, a retargeted branch):

- **Shape.** One entry: every block that starts inside the region is reached from
  inside it only. Calls, returns and indirect jumps stay outside. An early `s_endpgm`
  (a kernel's exit before its last block) is inside a region that reaches the final
  `s_endpgm`, where it is an exit to the end: it ends the wave as the final one does.
- **Barriers.** `s_barrier` (GFX9-GFX11) runs in regions. One wave runs, so a barrier
  waits for no other wave; it divides the run into epochs. Besides the final state, the
  two sides must execute as many barriers, store to the same shared dwords (LDS, and
  global memory outside the private segment) in each epoch, and load an LDS dword both
  load in the same epochs (`barriers` in the states). Global loads are not tracked: they
  are most of a GEMM's traffic (tracking them cost a third of `f0`'s time), and waves hand
  data to each other through LDS or by storing it. Scalar loads, which read kernel
  arguments and constants that allocation may rematerialize anywhere, are not tracked
  either. A store or LDS load moved across a barrier, or
  a value reused across one, is therefore a mismatch even though this wave sees the same
  values; other waves of the workgroup would not. What other waves store is not
  modelled: their part of memory keeps its initial content.
- **Exits.** Each side's exits are the targets of branches that leave the region, and
  falling off its end. They are paired through the instruction matching of the diff
  (falling off the end corresponds on both sides). An exit without a partner is
  `INCONCLUSIVE_MAPPING`.
- **R_out per exit.** Each exit pair has its own observable set: the liveness at its
  target (`exits[]` in the contract: `old_pc`/`new_pc`, 0 for falling off the end,
  `observable`, `vgpr_observable`, `vgpr_active_lanes_only`). Liveness that differs
  between the two targets is `INCONCLUSIVE_MAPPING`.
- **Execution.** The runner follows the branches RocJITsu executes (taken or not) and
  records the exit. Both sides must leave through corresponding exits, otherwise the
  case is an `exit_divergence` mismatch; then that exit's R_out is compared.
- **Loops.** A backward branch inside the region runs up to 64 times the region length
  (at most max(length, 65,536) steps).
  - Both sides past the limit: the case is excluded (`loop step limit`, outside the
    declared scope like inputs that fault on both sides), never read as non-termination.
  - One side exits and the other runs past the limit: the looping side re-runs at 16 times
    the limit (a slower loop then completes and the case is compared like any other; the
    larger limit holds for the rest of the case and for replay). If it still loops, the
    case is a possible non-termination (`counts.step_limit_divergences`): the region is
    `INCONCLUSIVE` with partial coverage, whatever its other cases show, and the whole
    function cannot pass on it either. After four such cases further one-sided cases are
    counted without a re-run. `counts.step_limit_reruns` counts the re-runs.
- **Undefined values.** When old's exit changes under the undefined-value refills, old
  branches on an undefined value (undefined behaviour in LLVM): the case is excluded.
- **Composition.** A region with control flow makes everything unknown downstream (as
  an untested change does). Edges inside a READY region (a loop back to its entry) are
  run by the region: the common-prefix analyses (composition, known bits) take the
  entry state from the edges that come from outside.

## Whole functions

Regions cut a function at matched instructions, and a cut needs an entry relation
(composition) and an exit relation (liveness that agrees on both sides). When those fail
(a register renamed across blocks, an untested change upstream, a mismatch that changed
code downstream may compensate), the regions alone leave the verdict open. The whole
function has neither cut:

- **Region `f0`.** `analyze` also builds the region from the function's entry to its
  final return (or `s_endpgm`) on both sides (`whole_function` in `case.json`;
  `--no-whole-function` turns it off; at most 4096 instructions per side). It needs
  what any region needs: every instruction executable (the list at the top of this file),
  direct branches only (and early `s_endpgm`), one return at the end, and no change in
  the return itself.
- **Kernels.** A kernel is an artifact with a descriptor, or a function listed without
  one that ends with `s_endpgm` on both sides (as `rj_isa_diff_pr` writes them; a C or
  gfx-convention function never does, a shader part returns to its epilog). Its `f0`:
  - drops all private memory and LDS at `s_endpgm` (`memory_dead_at_exit: private`);
  - takes workitem IDs below 256 in one dimension (v0-v2 ANDed with 0xff): arbitrary
    values put LDS addresses outside the allocation.
- **Contract.** Both sides start from the same call or dispatch state, so the entry
  relation is established unless the kernel metadata differs or memory is outstanding;
  entry values are arbitrary, which only widens the inputs. The exit is the return,
  where liveness is the calling convention's.
- **When it runs.** `test` runs `f0` only when the regions leave the verdict open
  (`INCONCLUSIVE`, `UNSUPPORTED`, or a pass with partial coverage). A pass of `f0`
  with complete coverage and at least 100 valid cases makes the function `NO_MISMATCH_FOUND` with complete
  coverage, unless an unresolved item lies outside the function's code (metadata,
  layout, relocations); a mismatch of `f0` is a `LOCAL_MISMATCH` verdict. Either way
  `overall.resolved_by` is `whole_function`, `results.json` has `whole_function`
  (`decides`, `reason`, `result`), and `f0` is listed with the regions.
- **Not a replacement.** Regions stay the first step: they are small, so a mismatch
  points at the change, and they work when other parts of the function cannot execute.

## Targets

A target is a GPU and a wave size (`Target`, `target.h`). Code objects name theirs (the
ELF machine number in `e_flags`; a kernel descriptor's `ENABLE_WAVEFRONT_SIZE32` sets the
wave size); raw words and `.s`/`.ll` inputs take `analyze --target NAME[:w32|w64]`
(default gfx950) and `--wave-size 32|64`, and `--mattr` passes target features to the
LLVM plugin. Both sides must run on the same target.

| family | targets | wave | ISA |
|---|---|---|---|
| CDNA1-4 | gfx908, gfx90a, gfx940-gfx942, gfx950 | 64 | GFX9 |
| RDNA1/2 | gfx1010-gfx1013, gfx1030-gfx1036 | 32 (default) or 64 | GFX10 |
| RDNA3/3.5 | gfx1100-gfx1103, gfx1150-gfx1153 | 32 (default) or 64 | GFX11 |
| RDNA4 | gfx1200, gfx1201 | 32 (default) or 64 | GFX12 |
| CDNA5 | gfx1250, gfx1251 | 32 | GFX12.5 |

GCN before CDNA (SI/CI/VI, Vega gfx900-gfx906/909/90c) and GFX13 have no RocJITsu ISA.
What depends on the target:

- **Decoder, compute unit and wave.** The target's ISA family decodes and executes; the
  wave is dispatched with the target's wave size and SGPRs (s0-s103 on GFX9, s0-s105
  from GFX10).
- **Scalar state.** SGPR locations are s0-s105; GFX9 addresses s0-s101 (its selectors
  102-105 are FLAT_SCRATCH/XNACK_MASK), so there s102-s105 are never named and stay 0.
  M0 is selector 124 before GFX11 and 125 from it; NULL (125 on GFX10, 124 from GFX11)
  reads as 0 and discards writes.
- **Wave32.** A wave mask is one SGPR: VCC and EXEC operands are `vcc_lo`/`exec_lo`,
  VALU instructions read `exec_lo`, a wave-mask SGPR operand (`s_and_saveexec_b32`,
  VOP3 carry/compare destinations) is one register. Lanes 32-63 do not exist: inputs
  keep them 0 and the runner neither loads nor captures them. Boundary sweeps spread
  their values over 32 lanes.
- **Waits and pending memory.** Outstanding memory operations are tracked per
  RocJITsu wait-counter domain (`WaitCounterPolicy`): vmcnt/lgkmcnt/expcnt on GFX9,
  plus vscnt on GFX10-11, and loadcnt, storecnt, dscnt, kmcnt, samplecnt, bvhcnt,
  expcnt (asynccnt, tensorcnt on gfx1250) on GFX12. Every wait form is decoded by
  RocJITsu (`explicit_wait_fields`); producers count through their memory-issue
  metadata.
- **MODE.** GFX12 has no DX10_CLAMP, IEEE or DEBUG bit: the descriptor's bits are not
  copied and the default function mode lacks them.
- **Descriptors.** The VGPR granule comes from RocJITsu
  (`descriptor_vgpr_granularity_for_wavefront`); ACCUM_OFFSET splits the register file
  on gfx90a and later CDNA only; GFX10+ gives every wave s0-s105.
- **Names.** Instruction names differ by generation (`s_andn2_b32`/`s_and_not1_b32`,
  `s_add_u32`/`s_add_co_u32`, `s_ff1_i32_b32`/`s_ctz_i32_b32`, `global_load_dword`/
  `global_load_b32`, `s_setpc_b64`/`s_set_pc_i64`); the allowlists hold all of them.
- **GFX12.5 VGPR banks.** After `s_set_vgpr_msb` the VGPR an operand names depends on
  dynamic state, so no instruction that names a VGPR is modelled in such a function.

## VALU support

VGPRs are part of the state, with all 64 lanes of each one (32 on wave32).

- **Effects.** VGPR operands become `vreads`/`vwrites`. A VALU instruction also reads
  EXEC, the lanes it writes.
- **Implicit reads.** Instructions outside the executable subset read the specials
  their class implies, beyond their operands:
  - SMEM: none;
  - vector memory: EXEC, plus M0 for LDS forms;
  - LDS: EXEC and M0;
  - anything else: EXEC/VCC/M0.

  Implicit uses reported by the decoder are translated: `global_store` names its SGPR
  base that way.
- **Liveness.** VGPR liveness is a backward dataflow. A write usually does not kill,
  because lanes outside EXEC keep their value. It kills when EXEC never changes again:
  from right before the write to every exit, no instruction writes EXEC or reads a lane
  regardless of EXEC (`v_readlane`/`v_writelane`, DPP, permutes, LDS swizzles), no
  effect or successor is unknown (`KernelModel::exec_stable_before`), and the write sets
  every bit (not a 16-bit result, SDWA, op_sel, true16 half or d16 load). The lanes it
  leaves alone are then never read again. A callable function's callee-saved VGPRs are
  never killed: the caller sees all their lanes.
  - `vgpr_entry_equal` and `vgpr_observable` come from this liveness, restricted to
    the tracked range.
  - A renamed dead VGPR temporary is not observable.
- **Tracked range.** The contract tracks `v0..v{vgpr_count-1}`: every VGPR either side
  names, rounded up to the allocation granule of 8. The runner allocates exactly that
  range, so an access beyond it is an invalid register access, never a silent write.
- **Comparison.** Every lane of every observable VGPR is compared, active or not, so a
  rewrite that clobbers lanes outside EXEC is caught, unless EXEC never changes after the
  exit on both sides: then nothing reads an inactive lane again, and observable VGPRs
  other than a callable function's callee-saved ones are compared on the lanes active in
  EXEC (`vgpr_active_lanes_only`).
  - The effect model also covers VGPRs: each side may change only its
    `vgpr_writes_*`.
  - Counterexamples list the differing lanes.
- **MODE.** Each side starts in its own kernel's MODE: round and denorm modes,
  DX10_CLAMP and IEEE, from the descriptor's `compute_pgm_rsrc1`. These are
  `mode_old`/`mode_new` in the contract. Raw word streams use the assembler defaults.
- **Composition.** Must-equality is tracked per VGPR. A VALU write is equal when all
  its inputs and its destination's previous value are equal.
- **Pending memory.** VGPR destinations of outstanding loads count: a region reading
  one gets `pending_effects: CONFLICT`.
- **Inputs are lane-parallel.**
  - Every case fills each VGPR with 64 different values: catalog values (with
    NaN/Inf/denormal encodings), 64-bit catalog values for 64-bit sources, single
    bits, copies of other registers' lanes, and random values.
  - The mandatory sweep adds vector phases:
    - catalog chunks of 64 values per VGPR (`vector.lanes`);
    - 64-bit chunks (`vector.lanes64`);
    - the cross product of two VGPR sources (`vector.cross`);
    - every scalar catalog value against every lane chunk for mixed SGPR/VGPR
      sources (`vector.scalar_cross`);
    - values around literals (`vector.constrel`);
    - alternate-lane and lane-pattern EXEC (`vector.exec`).
  - Vector phases run with all lanes active. The random phase uses a full EXEC on half
    of its cases.
- **Reduction.** After seed independence, lanes outside the mismatching ones are
  zeroed, then each shared VGPR is shrunk at the first mismatching lane.

A local mismatch whose locations are read by changed code downstream of the region
(another region, or a control-flow change) may be compensated there. Context expansion
cannot reach it, because the code in between is not common. Such a mismatch:

- stays a `LOCAL_ONLY` witness;
- is marked `compensable_downstream`;
- is listed in `unresolved`;
- leaves the overall result `INCONCLUSIVE`, never `LOCAL_MISMATCH`.

The contract records the consumers in `downstream_changed`, `downstream_reads` and
`vgpr_downstream_reads`.

## AGPRs and MFMA

- **Locations.** The accumulation registers a0..a255 of CDNA are vector locations 256..511
  (named `a0`..`a255` in contracts, states and messages), RocJITsu's logical layout: acc
  n is at the wave's VGPR base + 256 + n. Liveness, entry relations, undefined inputs,
  input lanes, masks and comparison treat them like VGPRs.
- **Instructions.** `v_accvgpr_read/write/mov`, MFMA and SMFMAC run in regions, as do
  memory instructions with AGPR data (`buffer_store_dword a0, ...`, `ds_write_b128 v4,
  a[0:3]`). MFMA results are compared bit for bit (they carry no floating-point tag);
  the runner executes them functionally, so MFMA hazards (`s_nop` counts) are not checked.
- **Lanes.** A matrix instruction reads every lane's A, B and C, whatever EXEC holds. So
  MFMA/SMFMAC count as lane-crossing: a write before one never kills its register
  (inactive lanes stay entry-equal), and a region that issues one under the entry EXEC
  gets EXEC all ones (`mfma_full_exec` constraints; LLVM issues MFMA whole-wave).
  Without these, inactive lanes holding per-side or undefined values fed the product and
  made equivalent code differ.
- **Calling convention.** From gfx90a on, a32..a255 are callee-saved (`CSR_AMDGPU_AGPRs`):
  live at a callable function's return and defined at its entry. gfx908 saves no AGPR.
- **Runner.** A region that names an AGPR dispatches 256 VGPRs, which gives both banks
  (v0..v255, a0..a255) without ACCUM_OFFSET.

## Memory

Memory instructions run inside regions. The runner writes the region's code bytes into
its memory at their PCs and issues each memory instruction with
`ComputeUnitCore::step()`: fetch, execute, route, and a functional access that completes
within the step.

- **Input.** Memory is one more input, shared by both sides.
  - Each case has a memory seed. A word that is not yet written reads
    `initial_memory_word(seed, word)`: half catalog values, half random.
  - An execution plugin sees every routed access before it reaches the caches, and
    seeds the words it touches. Both sides see the same image.
  - A case's states record the seed and the initial words read (`memory.initial`), so a
    bundle replays without the generator.
- **Pointers.** Some loaded words are pointers.
  - An `s_load` whose destination SGPRs reach a memory base, through SALU moves and
    adds, loads pointer-shaped words: the low half 64 KiB-aligned, the high half in
    0x10..0x7f.
  - Entry SGPR pairs used as memory bases get `memory_model` constraints of the same
    shape (low `known_zero 0xff`; high `known_one 0x10`, `known_zero 0xffffff80`).
    Boundary values that are not such pointers are rejected and counted, never changed.
- **Output.** Stored bytes are observable.
  - Both sides' stores are compared over the union of their addresses. An address one
    side did not write keeps its initial byte.
  - A difference is a memory mismatch: `memory_mismatch` in the result, and
    `memory_differs` and `memory_differences` in the bundle. Bundles replay.
- **Caches.** Before every memory step the runner writes back and invalidates the L1
  (write-through) and L2 (write-back) caches, so each access fills from the seeded memory.
  The CU's own wait checker is off, because the planner models waits (pending effects).
- **Faults.** A wave that halts or traps on an access is a `MEMORY_FAULT`.
  - On both sides, the input is excluded and counted as a rejection.
  - On the new side only, it is an `exit_status_divergence` mismatch: the change
    introduced a fault.
  - On the old side only, the input is excluded too (`memory fault on the old side only`):
    a fault is an access LLVM leaves undefined, so new may do anything there, as with
    reported undefined behaviour.
  - LDS-DMA and image accesses are `UNSUPPORTED_OPERAND`.
  - A load or store at either side's code addresses (a pointer that landed on the
    code; the program's objects never overlap it, and the sides' code bytes differ) is a
    `MEMORY_FAULT`, so an input both sides take it on is excluded. A case that stored into
    the code gets its bytes restored before the next one.
- **LDS.** The wave has a 64 KiB LDS window at offset 0 (one workgroup's allocation).
  - LDS words are seeded from the same image as global memory, as word
    `(kLdsTag | offset) / 4`; stored LDS bytes appear in the states as
    `0x4000000000000000 | offset`, so LDS and global memory never alias.
  - An access outside the window is a fault of the case (LLVM: out of the allocation,
    undefined): excluded on both sides, a mismatch on one.
  - An LDS address that is a region input (the first VGPR source of a DS instruction,
    not written before it) has every lane masked with `0x7ffc` (`vgpr_input_masks`,
    source `memory_model`), so it lies in the window with room for offsets.
  - Two-address forms (`ds_read2*`, `ds_write2*`) move both address sets.
- **Atomics.** Global, flat, buffer and DS atomics read and write: the word is seeded
  before the read-modify-write and the result is stored. A returning atomic's old value
  goes to its destination like a load.
- **Private memory (scratch and buffer spills).**
  - `scratch_*` instructions address the wave's scratch base, fixed in the frame at
    `0x0000008000000000`; lanes interleave by dword as the hardware's do.
  - A buffer resource that is a region input (the four SGPRs of a `buffer_*` instruction,
    not written before it) is the calling convention's private segment buffer
    (`memory_model` constraints): a 256-byte aligned base far above the code, unlimited
    records, and word 3 with ADD_TID and a dword swizzle (`0x00e00000` on GFX9,
    `0x31c16000`/`0x31e16000` on wave32/wave64 GFX10+). Single-SGPR offsets of buffer and
    scratch accesses (`s32`, the stack pointer) are 16-byte aligned.
  - **Stack frames.** A private access without a VGPR address (MUBUF without
    `offen`/`idxen`, scratch with an SGPR address only) goes through the stack or frame
    pointer: the function's own frame. Bytes stored that way are dead where nothing can
    read them again, and not compared (`memory_dead_at_exit: frame`, `memory.dead` in the
    states): at the exit of a region when no instruction outside it may read private
    memory (no buffer, scratch or flat load or atomic, no unknown effect), and at a
    callable function's return (`f0`). A kernel's `f0` drops all its private memory and
    LDS at `s_endpgm` (`private`), and a kernel's region outside which nothing reads
    private memory drops every private byte it stores, in any addressing form
    (`private_memory`: a kernel has no caller whose frame a VGPR address could reach). A
    spill slot that moved is therefore not a difference, while a reload from the wrong
    slot is.
  - **Small kernel arguments.** In a region with a loop, three cases of four read small
    values (below 2^k, k up to 12) from data words that scalar loads read (kernel
    arguments, constants), so loop bounds and sizes loaded from kernel arguments let loops
    end; the other cases keep the catalog and random values. Pointer words keep their
    pointer shape.
- **Contract.**
  - `memory_reads` and `memory_writes` record what the region does.
  - A region that loads needs memory equality at its entry in the composition check.
    Upstream stores in common code with equal inputs keep it.
  - `downstream_memory_reads` marks changed code downstream that loads, which may
    compensate a stored-byte difference (`compensable_downstream`).
- **Merging.** Hunks separated only by common memory instructions merge into one region.
  - A reordering of loads and the code consuming them is tested as one unit.
  - A register rename between the reordered loads becomes internal to the region,
    instead of an entry relation that composition cannot establish.

## Register allocation

A descriptor difference only in register allocation is informational when each kernel's
registers fit its own allocation. Register allocation means the granulated VGPR and
SGPR counts in `compute_pgm_rsrc1` and `ACCUM_OFFSET` in `compute_pgm_rsrc3`. Fitting
means:
- every VGPR a kernel names is within both `ACCUM_OFFSET` and the granulated VGPR count;
- the SGPRs it names, plus the six extra ones (VCC, FLAT_SCRATCH, XNACK), are within the
  granulated SGPR count (granule 16).

The note then says "register allocation only". Register-count metadata (`.vgpr_count`,
`.sgpr_count`, `.agpr_count`) follows the same rule. Any other descriptor bit, or an
allocation that a kernel overruns, stays semantic.

## Callable functions

A function symbol (`STT_FUNC`) without a kernel descriptor is a callable function under the
AMDGPU C calling convention (`CC_AMDGPU_Func`, `AMDGPUCallingConv.td`). The case records
it under `artifacts.*.callable_function`.

- **Return.** `s_setpc_b64 s[30:31]` is the return: a block without successors, instead of
  an unknown indirect branch. It reads only s[30:31].
- **Live at a return.** The caller observes:
  - the return VGPRs `v0..v{N-1}`, with N from `--return-vgprs` (default 32: every VGPR
    the convention can return in, which is conservative for a function returning less);
  - the callee-saved SGPRs s30-s39, s48-s55, s64-s71, s80-s87, s96-s105 (s96-s101 on GFX9; with the return
    address, stack and frame pointers);
  - the callee-saved VGPRs v40-v47, v56-v63, ..., v248-v255, on every lane;
  - EXEC.

  Caller-saved registers (s0-s29, v0-v39 beyond the return value, VCC, M0, SCC) are
  dead after the return. A rewrite that clobbers a callee-saved register is a mismatch.
- **Entry.** Both sides enter with the same caller state, so the composition check starts
  from full equality, as for a kernel. MODE at entry is `--function-mode` (default
  `0x3f0`: LLVM's default function mode, denormals kept, DX10_CLAMP and IEEE on).
- **Calls** (`s_swappc_b64`) stay unknown successors: everything is live there.
- A kernel body without its descriptor (as in LLVM test CHECK lines) also loads as a
  function symbol; it ends at `s_endpgm`, where nothing is live.

## Inline asm

LLVM tests feed values in and out of the code under test with comment-only inline asm:

```
;;#ASMSTART
; def v[0:1]
;;#ASMEND
```

Such a block assembles to nothing, so it is invisible in the code object. Ignored, its
inputs would look like unchanged entry values, which is wrong when old and new allocate them
differently, and a value consumed by a `; use` block would look dead.

- **Marking.** The LLVM plugin puts a label (`__rj_isa_diff_asm.<n>`) in front of every
  comment-only block before assembling. It records the block's text, and whether a label
  follows the block, in a non-loaded section, `.rj_isa_diff_inline_asm`. The code object
  carries both, and its hash covers them. Blocks that contain real instructions are
  ordinary code.
- **Pseudo-instructions.** The kernel model turns each block into a zero-size
  pseudo-instruction at its position. Its effects come from the text:
  - `; def R, ...` writes R: VGPR lanes in EXEC, so it reads EXEC;
  - `; use R, ...` reads R (and EXEC for VGPRs).

  R is `sN`, `s[N:M]`, `vN`, `v[N:M]`, `vcc`, `exec`, `m0`, `scc`, or one of their halves.
  AGPR operands are untracked: no region may contain the block. Any other text that names
  a register is opaque: its effects are unknown, and a diff note keeps coverage partial.
  A block followed by a label ends the preceding block. After a branch, it would sit on
  the fall-through edge alone; that case is opaque too.
- **Pairing.** Blocks are paired by their index in each function's sequence of blocks.
  - The diff key includes the index and the text: a block whose operands were renamed is
    part of a changed hunk, so the code around it forms one region.
  - Sequences of different length or operand shape are a diff note: no pairing, coverage
    partial.
  - A region whose two sides contain different blocks is `INCONCLUSIVE_MAPPING`.
- **Execution.**
  - A `def` writes per-case values from the case's inline-asm seed. The old and new block
    with the same index get the same values, whatever registers they name.
  - SGPRs that the region uses as a memory base get pointer-shaped values.
  - A `use` records what it reads: SGPRs, VGPR lanes in EXEC, and EXEC. The two sides'
    records are compared like stored bytes, so a renamed register that ends in a `use` is
    checked, and a wrong value in it is a mismatch (`inline asm block K use differs`).
  - States record the seed and the uses (`inline_asm`), so bundles replay.
- **Assumption.** Pairing by position assumes both compiles emit the blocks in the same
  order. Non-`sideeffect` inline asm may be scheduled. A reorder pairs the wrong values,
  which shows up as a mismatch rather than a pass.

## Testing an LLVM change

`rj_isa_diff_pr` (`llvm/rj_isa_diff_pr.py`, installed next to `rj_isa_diff` in the build
tree) tests an LLVM change on the AMDGPU lit tests it updates. LLVM is neither built nor
run.

- **Source of the assembly.** Tests whose assertions were autogenerated by
  `utils/update_llc_test_checks.py` contain each function's complete assembly in their
  CHECK lines. The driver reads both revisions of each changed test under
  `llvm/test/CodeGen/AMDGPU` from git, and rebuilds the old and new listing of every
  function whose CHECK body changed. A listing is the body with `{{\[\[}}`-style
  escapes undone, plus placeholder data for symbols it references but does not define.
  An assembler (`llvm-mc` and `ld.lld`, by default ROCm's) turns it into a code object.
- **Assembler.** Each side must assemble with its own LLVM's `llvm-mc`: an installed one
  can reject a change's syntax or checks (ROCm 7.2.3's rejected 7 functions of the
  commit sweep: VOPD banks, bf16 inline constants, operands "not valid for this GPU").
  - `--mc-before LLVM_MC --mc-after LLVM_MC` name them; with two llc builds the
    `llvm-mc` next to each `llc` is used (`--build-llc` builds it), if both have one.
  - The linker (`ld.lld`) stays `--llvm-bin`'s: an AMDGPU-only build has no lld.
  - A new listing that the new assembler rejects while the old one assembles is
    `asm-rejected` (a defect of the change, `MISMATCH`): upstream llvm-mc rejects the
    `v_rsq_bf16_e64 v0.l, 4.0` that commit 66cb135fa288 emits for gfx1250 ("bf16 inline
    constant is read from the high half ... use op_sel:[1,0]"). With `--llvm-bin`'s
    assembler a rejection stays `UNSUPPORTED` (it may only be version skew).
  - The report and `summary.json` (`meta.assembler`) name the assembler of each side,
    with a note when the llc builds have none and both sides fall back to `--llvm-bin`.
- **What is compared.** Only what changed. A function whose CHECK body differs only in
  comments (renumbered block names, spill notes) has the same code and is counted, not
  tested. Inline-asm blocks count as code. Within a changed function, only the regions
  around its hunks execute; the rest is analysed once. A function under a RUN line that
  targets a RocJITsu target (see "Targets"), or a proxy target, is compared when it is
  the same on both sides:
  - the test is autogenerated without `--filter`;
  - the RUN line is unchanged;
  - the function's IR is unchanged: body, attribute groups, referenced globals;
  - its CHECK lines contain no FileCheck patterns;
  - its calling convention is the C one or `amdgpu_kernel`.

  Every other changed function is listed with the reason it was skipped.
- **Targets and wave size.** A listing runs on its RUN line's `-mcpu`, at the RUN line's
  wave size (the last `+wavefrontsize32/64` in `-mattr`, else wave64 on GFX9 and wave32
  from GFX10), and llvm-mc gets the RUN line's other `-mattr` features (`real-true16`,
  ...). Vega listings (gfx900-gfx906, gfx909, gfx90c), which have no RocJITsu ISA, are
  assembled and executed as gfx908, which shares their encodings and syntax, and
  labelled `proxy`; a listing gfx908 cannot assemble is `UNSUPPORTED`. Changed functions
  on other targets (SI/CI/VI, GFX13, RUN lines without `-mcpu`) are counted, not
  executed.
- **Calling conventions.** Kernels and C functions as before; `amdgpu_gfx` functions
  return through `s_setpc_b64 s[30:31]` with callee-saved s4-s31 and s64-s105
  (`--calling-convention gfx`). Graphics shaders (`amdgpu_ps`, `vs`, `gs`, `hs`, `es`,
  `ls`, `cs`) take inreg arguments in s0.. and the others in v0.., and return to the
  shader epilog: the driver ends the listing with `s_endpgm`, where integer return values
  in s0..s43 and floating-point ones in v0.. are observable, nothing is callee-saved, and
  narrow integers (`i1`/`i8`/`i16` without `zeroext`/`signext`) compare their bits only
  (`--calling-convention shader --return-sgprs N --return-sgpr-masks M,...`). Their MODE
  defaults to IEEE off.
- **RUN lines whose flags changed.** A changed RUN line is compared against its old form
  when the target, check prefixes and wave size are the same and no floating-point
  semantics flag changed (denormals, fast-math, contraction): the old body (old flags)
  against the new body (new flags), each assembled with its own features
  (`--old-mattr`/`--new-mattr`). Other changed RUN lines are skipped.
- **Two llc builds.** `--llc-before LLC --llc-after LLC` compiles every changed test's
  head IR at each RUN line for a tested target with both, and compares the functions
  whose code differs; any test qualifies, not only autogenerated ones (hand-written
  tests, added tests, CHECK bodies present on one side only).
  - `--build-llc DIR` builds the two llc (and llvm-mc) itself: a worktree of `--repo` per side
    (`DIR/before`, `DIR/after`; `WORKTREE` builds the checkout) and a Release+assertions
    AMDGPU-only build per side (`DIR/before-build`, `DIR/after-build`), configured once and
    rebuilt incrementally, so a sweep over neighbouring commits rebuilds little.
  - `--all-tests` also compiles every unchanged test matching `--tests`: code a change
    moved in tests whose CHECK lines were not updated (or do not exist).
  - `--corpus PATH...` also compiles LLVM IR from outside the repository (real kernels;
    files, or directories searched for `.ll` and `.llir`, e.g. a Triton cache), at one `llc -mtriple=amdgcn-amd-amdhsa -mcpu=CPU` RUN
    line per `--corpus-mcpu` CPU (default `gfx950`), with `--corpus-args` added (e.g.
    Triton's `-O3 -relocation-model=pic`).
  - A corpus file that declares its target is compiled for that target only, and only
    if it is in `--corpus-mcpu`. The declaration is, in order:
    - the Triton cache metadata next to it, or next to the file a symlink points to (a
      sample of a cache): `<kernel>.json`'s `target.arch`; `target.warp_size` 64 on
      GFX10+ adds `-mattr=+wavefrontsize64`;
    - its functions' `"target-cpu"`;
    - its triple's subarch (`amdgpu9.50-amd-amdhsa` is gfx950).

    A file for another target (or another Triton backend, or a non-AMDGPU triple) is not
    applicable: the summary counts it per target (`corpus_other_target`). Compiling a
    gfx1250 kernel for gfx950 fails ("Cannot select", "requires target feature"), which
    used to make every run over a mixed Triton cache `INCOMPLETE`.
  - Corpus IR that both builds reject as input (a parse error, or the IR verifier: IR
    another LLVM wrote, e.g. an intrinsic whose arguments changed) cannot be compared by
    any build. The report lists it, and the summary records it (`corpus_input_rejected`),
    but it is not a coverage gap. A rejection by one build only, or a failure during code
    generation, stays an llc failure (below).
  - The compiles run in parallel (`--jobs`). Unchanged tests and corpus files without a
    RUN line for a tested target are not skips; the summary counts them
    (`extra_without_tested_run`).
  - An llc failure (or a timeout, 300 s) on a requested file is a skip, so the run is
    `INCOMPLETE`: changed tests, unchanged tests (`scope: unchanged test`) and the corpus
    (`scope: corpus`) alike (`extra_llc_failed` counts the last two). Both builds run on
    every RUN line first; a failure of one build where the other compiles is a new (or
    fixed) llc failure, reported as such and counted in `llc_one_sided_failures`.
- **ABI from the IR.**
  - `--return-vgprs` is the return type's size in dwords (`void` 0, `<3 x half>` 2,
    `iN`/`bN` ceil(N/32), `<N x i8>` N (one element per register), structs and arrays
    summed). An unknown type gives the conservative 32, with a note.
  - `--return-masks` covers narrow return types without `zeroext`/`signext`: `half`,
    `bfloat` and `i16` give `0xffff`, `i1` gives `0x1`, `<3 x half>` gives
    `[0xffffffff, 0xffff]`, `<N x i8>` gives `0xff` per register.
  - `--entry-vgprs` is the dwords of the non-`inreg` arguments (3 for kernels). Other
    VGPRs are undefined at the entry.
  - `--entry-sgprs` is the dwords of the `inreg` arguments.
    - Add 16 when the function may use special inputs or the private segment buffer
      (s0-s15): special-input intrinsics, calls, `alloca`, or a target without
      architected flat scratch.
    - Kernels get 24 (at most 16 user + 5 system SGPRs).
  - `--function-mode` comes from the denormal attributes (`denormal_fpenv(...)`,
    `"denormal-fp-math"`, `"denormal-fp-math-f32"`) and `"amdgpu-ieee"`.
- **Poison triage.** A function that mismatches is re-tested with the poison filters its
  IR calls for (`shl` → `shift`, `add nsw` → `nsw`, `fptosi` → `fptoint`, fast-math
  flags → `nnan`/`ninf`, ...).
  - The verdict stays `mismatch` either way. The filters match machine conditions, not
    IR values: llc may rely on a mask or range the IR guarantees (`shl i32 %x, (and %y,
    31)` becomes a bare `v_lshlrev_b32`, whose amount the hardware masks), so excluding
    machine shift amounts >= 32 can hide a real bug. Without a mapping from IR to machine
    values the triage cannot show that the inputs are poison in the IR.
  - If the mismatch disappears, the table marks it `mismatch (poison-sensitive)` and the
    details say so, with the number of inputs excluded: check the IR before dismissing
    it. Otherwise it is noted as "persists outside those inputs".
  - `shift` is called for only when some IR shift amount is not provably below the bit
    width (a constant below it, or an `and` with such a constant, is not poison).
  - `--no-poison-triage` turns this off.
- **Output.** `report.md` (the overall status, a verdict table, then details of every
  function that did not pass) and `summary.json` (`status` and every function). Each job
  has its own directory under `functions/`: the test's path, then
  `run<N>-<hash>-<function>`, the hash covering the test path, RUN line(s) and function,
  so tests with the same file name in different directories never share one. A job is
  built under a private `.partial-*` name, with the tool running inside it on relative
  paths, and published by a rename when it is complete.
- **Verdicts.** `pass`; `pass-masked` (equal only through the per-bit don't-care mask,
  see "NaNs, undefined values and poison"); `mismatch`; `asm-rejected` (with the
  change's own assembler: the new listing is rejected, the old one assembled, so the
  change emits invalid assembly); `inconclusive` (including `UNSUPPORTED`); `error` (the
  tool failed, or reported `HARNESS_ERROR`).
- **Overall status and exit code.**

  | code | status | meaning |
  |---|---|---|
  | 0 | `PASS` | every changed function was compared and passed; nothing skipped |
  | 1 | `MISMATCH` | a local mismatch, or an `asm-rejected` function |
  | 2 | `INCOMPLETE` | something was not covered: `inconclusive` or `pass-masked` functions, skipped functions or files (added tests, tests not autogenerated, changed RUN lines, changed IR, llc failures), or changed functions on targets not executed |
  | 3 | `ERROR` | a harness error |
  | 4 | `NO_APPLICABLE_TESTS` | nothing was compared: no changed function this tool can run |
- **IR identity.** A function is compared only when everything that decides its semantics
  is the same on both sides: its text, the attribute groups it and its callees use
  (call-site attributes included), the globals and declarations it references, the
  metadata they reference (transitively, so a changed `!range` counts), and module-level
  information (datalayout, triple, named metadata such as module flags, module asm).

## NaNs, undefined values and poison

LLVM's semantics leave some results open. A bit-exact comparison flags those as
differences, so equivalent compiles look like false mismatches. Each case below has a
switch for the stricter reading.

- **NaN results** (`test --strict-nan` turns this off).
  - **Rule.** LLVM leaves the payload, sign and quietness of a NaN produced by a
    floating-point operation unspecified, and commutative operations may swap operands.
    `v_add_f32 v0, v0, v1` and `v_add_f32 v0, v1, v0` return different NaNs when both
    inputs are NaN.
  - **Tags.** The runner tags every value with its floating-point provenance (`FpTag`):
    which f16/bf16 halves, f32, or f64 pair a floating-point instruction produced.
    - Moves, `v_cndmask_b32` and `v_pk_mov_b32` carry the tags.
    - Scalar locations are tagged too: SALU float results (GFX11.5+), `s_mov`,
      `s_cselect`/`s_cmov`, `v_readfirstlane_b32`/`v_readlane_b32`; a VALU instruction
      reading an SGPR (`v_mov_b32 v0, s6`) picks up its tag. NaN choice and the
      nnan/ninf poison checks cover them.
    - Stores turn them into floating-point units in memory.
    - Inline-asm uses record them.
    - Integer, bitwise and lane-shuffling instructions clear them.
  - **Comparison.** Two values that both sides tag as results of the same format, and
    that are both NaN, compare equal. Everything else stays bit-exact.
    - fneg/fabs/copysign done as bit operations must keep a NaN's bits. They produce no
      tag, so a rewrite into a multiplication by -1 (which quiets a signalling NaN) is
      still a mismatch.
  - **Sweep.** The mandatory sweep adds pairs of different NaNs (quiet/signalling,
    signs, payloads, mixed f16/bf16 halves) so this is exercised every run.
  - **NaN bits used as integers.** A NaN's unspecified bits may flow into integer code
    (a bitcast and a mask) before they are observed.
    - When old produced a NaN in a case, it re-runs with every NaN a floating-point
      instruction produces replaced by an alternative: the canonical quiet NaN, then all
      ones, then a random quiet NaN.
    - Output bits that change depend on the NaN's choice and are don't care.
    - New re-runs with its own alternative NaNs and must still match. Both sides draw a
      random NaN from the same seed, keyed by the NaN the hardware produced (with the
      lane and destination component): the same operation on the same inputs draws the
      same NaN on both sides, whatever order a scheduler gave the floating-point
      operations, and a moved hint, an extra move or a renamed register changes nothing.
      (Keying by position among the floating-point instructions executed made a
      rescheduled GEMM epilogue draw different NaNs for the same operations, which
      looked like new depending on NaN bits old does not.)
    - Moves, `v_cndmask` and `v_pk_mov` copy a NaN unchanged: only producing
      instructions choose.
    - A mismatch that new shows only under an alternative NaN records the choice
      (bundle `nan_choice`: alternative and seed); reduction, the bundle's trace and
      `replay` run the new side with it.
    - This uses the same mask and re-runs as undefined values.
- **Undefined values** (LLVM undef; `analyze --entry-vgprs N --entry-sgprs N`).
  - **Defined at entry:** the argument VGPRs `v0..v{N-1}` and SGPRs `s0..s{N-1}`, plus
    EXEC.
    - A callable function also has v31 (workitem IDs) and its callee-saved VGPRs and
      SGPRs defined.
    - VCC, M0 and SCC are undefined at every entry.
    - Defaults: callable functions 32 VGPRs and 30 SGPRs; kernels v0-v2 and their
      descriptor's user + system SGPR count; raw words treat everything as defined.
  - **Analysis.** A forward may-written analysis finds the VGPRs and scalar locations a
    region reads that no path from the entry has written. Contract fields:
    `vgpr_undef_old`, `vgpr_undef_new`, `undef_old`, `undef_new`.
  - **Test.** In each case, old re-runs with its undefined registers refilled three
    times: all zeros, all ones, random.
    - Output bits that change are don't care for that case.
    - **Limitation: per-bit masking loses the relation between bits.** Old `(u & 1) * 3`
      can only produce 0 or 3; masking the two bits that change also accepts 1. More
      re-runs do not fix this. A result that needed the mask is therefore not a plain
      pass: the case counts in `dont_care_masked` (results: `dont_care_masked_cases`),
      `test` exits 2, and the driver reports `pass-masked`. The exact NaN rule and an
      undefined input old passes through unchanged (its output is the input itself, in
      the base run and every refill) do not count: any value there is exact. A
      refinement check that keeps relations between bits is future work.
    - New re-runs with its own undefined registers refilled, and must still match old
      under that mask. Otherwise it is a mismatch: "new reads an undefined value".
  - **Effect.** A rewrite that materialises an undefined element as 0, or allocates an
    `IMPLICIT_DEF` to another register, passes. Storing garbage where old stored an
    argument does not.
- **Narrow return values** (`analyze --return-masks M0,M1,...`). A VGPR that is observable
  only through a return value narrower than the register compares only the returned
  bits (`vgpr_exit_masks`). For example, `0x0000ffff` for a `half` in v0.
- **Poison** (`test --poison-filter KINDS`). The tool runs no IR, so it cannot know when
  an input is poison in the IR. The filter excludes cases in which a region instruction
  meets the machine analogue of an IR poison condition, on either side; such inputs may
  legally produce anything.

  | kind | IR construct | machine condition |
  |---|---|---|
  | `shift` | `shl`, `lshr`, `ashr` | shift amount >= bit width |
  | `exact` | `lshr`/`ashr exact` | non-zero bits shifted out |
  | `fptoint` | `fptosi`, `fptoui` | NaN, infinity, or out of the integer range |
  | `ctz` | `ctlz`/`cttz` with `i1 true` | zero input |
  | `nsw`, `nuw` | `add`/`sub`/`mul nsw`/`nuw` | signed/unsigned 32-bit overflow of an add, sub or mul (not carry chains, not `clamp`) |
  | `disjoint` | `or disjoint` | operands share bits |
  | `nnan`, `ninf` | fast-math flags, `nofpclass`, `no-*-fp-math` | a NaN/infinite input or floating-point result |

  Excluded cases are counted as rejections (`poison input (excluded): KIND`). The filter
  is a triage tool: the machine condition approximates the IR one. `rj_isa_diff_pr` uses
  it on mismatches.
- **Reporting.** Results record `comparison.nan` (`llvm` or `strict`) and
  `comparison.poison_filter`. `counts.nan_tolerated` and `counts.undef_masked` count the
  valid pairs that compare equal only through these rules. Bundles carry `nan_policy`
  and `poison_filter`, so replay applies the same comparison; it recomputes the undef
  mask from the recorded seed and case.

## Manual regions and register maps

`--manual-region old:B-E,new:B-E` replaces automatic planning with one hand-picked
region `m0` (handoff 12). B and E are instruction indices or `0x` PCs, and E is
exclusive. The region goes through the same admissibility checks as an automatic one:
single block, P0 opcodes, no PC-derived input, no relocated bytes. Its contract:

- is `ASSUMED` (`UNRESOLVED` if metadata differs or pending effects conflict), never
  `ESTABLISHED`;
- records the boundaries as a `user_assumption` source;
- has composition `NOT_CHECKED`.

Every diff hunk outside the range stays in `unresolved`, so the overall result is never
a pass for the whole change.

Hand-written register correspondences, old=new (handoff 3.1), are allowed only with
`--manual-region`:

| flag | sets |
|---|---|
| `--entry-map s4=s8,...` | where the new side receives each input: old `s4`'s value goes to new `s8` |
| `--exit-map ...` | where the new side's R_out is read |
| `--map ...` | both maps |

The two ends need separate maps after a register-allocation change. In the real
`-enable-misched=false` pair, an input sits in `s[4:5]` vs `s[8:9]`, but the result is in
`s4` on both sides. That pair needs `--entry-map s4=s8,s5=s9`; `--map` is wrong there.

- The contract keeps old naming. Live-ins are translated through the entry map and
  live-outs through the exit map.
- Each map is completed to a permutation (`s4=s8` also sends old `s8` to new `s4`), so
  every location is still initialised exactly once.
- Only SGPRs (`s0`..`s105`) can be mapped, because EXEC/VCC/M0/SCC aliases must stay
  consistent.
- `mapping` becomes `user_register_map`, with a `user_assumption` source. The status is
  at most `ASSUMED`.

A wrong map shows up as a mismatch, or as `INCONCLUSIVE_MAPPING` when the translated exit
liveness disagrees. It is never inferred from outputs.

## LLVM plugin

`llvm/` (`rj_isa_diff_llvm`) finds `llc`, `llvm-mc`, `ld.lld` and `llvm-objcopy` in
`--llvm-bin`, `$RJ_ISA_DIFF_LLVM_BIN`, or `/opt/rocm/llvm/bin`, and runs them with
`posix_spawn` (no shell). `--old-llvm-bin DIR`/`--new-llvm-bin DIR` replace, for one side,
whichever of those tools `DIR` holds (at least `llc` or `llvm-mc`): an LLVM build's own
assembler with the default linker. The side's provenance records the overlay
(`toolchain.overlay`: directory, tools, `llvm-mc --version`).

- `.ll` goes through `llc -mtriple=amdgcn-amd-amdhsa -mcpu=TARGET [-mattr=...] <common>
  <side>` (`--target`, default gfx950; `--wave-size` adds `+wavefrontsize32/64` on
  GFX10+, `--mattr` other features); `.s` through `llvm-mc` with the same `-mcpu` and
  `-mattr`.
  Side arguments may not change the target (`-mcpu`, `-mattr`, `-mtriple`, `-march`,
  `-o` and `-filetype` are rejected).
- `.s` goes through `llvm-mc`, then `ld.lld -shared --emit-relocs`. `--emit-relocs`
  keeps `.rela.text`, so instructions patched by the linker stay identifiable; the
  executed bytes are unchanged.
- A listing with neither a kernel descriptor nor a function symbol falls back to its raw
  `.text` (`executed_object: raw_text`).
- Comment-only inline-asm blocks are marked in a copy of the source before assembly
  (`inline_asm_blocks_marked`; see "Inline asm").

`case.json` gets `provenance.{old,new}` with:

- every command line, its exit code and time;
- the input SHA-256;
- the toolchain version and revision;
- whether both sides came from the same input and toolchain.

Counterexample bundles carry the same provenance. Executed bytes are always the linked
object, decoded by RocJITsu; re-assembled code is not presented as an original emission.

## Relocations and metadata

- **Relocations**: `REL`/`RELA` entries that patch `.text` are kept per instruction.
  These are static `.rela.text` entries (relocatable objects, or `--emit-relocs`) and
  dynamic entries whose target lies in `.text`.
  - A region containing a relocated instruction is `UNSUPPORTED`.
  - Matched instructions whose relocations differ (same bytes, other symbol or addend)
    are a semantic `layout` change.
- **Metadata**: the AMDHSA note is parsed with `amdhsa::MetadataDocument`. This
  kernel's map and the global fields are compared field by field.
  - Argument records are compared as units: one inserted argument is one change.
  - A note that differs only in other kernels' entries is informational.
  - An unparsable note falls back to the whole-note hash, conservatively semantic.

## Caches

`--cache-dir DIR` enables two layers (handoff 11):

| layer | key binds |
|---|---|
| analysis (`analyze`) | both artifacts' SHA-256, target (GPU, wave size), kernel names and ranges, every planning option |
| result (`test`) | the prepared region (instructions, PCs, complete contract), the prepared keys of its expansion variants, policy, seed and every budget |

Both keys also bind the tool and schema versions and a build fingerprint: a SHA-256 of
the running executable and every executable file it maps. Any rebuild therefore
invalidates everything.

- Identical region bytes under a different contract miss: a downstream liveness change
  alters `observable`, and with it the key. The contract names the target and wave size.
- The same bytes analysed for another GPU or wave size miss (raw words take the target
  from `--target`). An analysis hit is also checked against the request (input hash,
  kernel, target, wave size); an entry that does not describe it is recomputed and
  replaced.
- A hit carries `cache.hit` and points its counterexamples at copies kept in the cache
  entry, so it stays replayable.
- Harness errors and time-limited analyses are never stored.
- Computing the fingerprint costs roughly 0.2 s per process. It is computed only when
  `--cache-dir` is given.

## Explicit diagnostic path

The default never falls back to whole-kernel simulation (handoff 5.3).
`--diagnostic-entry-prefix` additionally builds, for each READY region, the region
extended back to the kernel entry (`rN.diag-entry`). There the region's inputs are
produced by executing the common prefix instead of being sampled at the cut, and the
entry values stand in for the dispatch ABI, which is not modelled.

- Diagnostic regions are run by `test` and reported under `diagnostics`, never in the
  verdict.
- A prefix with control flow or instructions outside the executable subset (LDS, MFMA,
  `s_getpc_b64`, ...) makes the diagnostic `UNSUPPORTED` with that reason. Kernel-argument
  `s_load`s run.

## Stop rules

- `--max-region N`: region length limit.
- `--max-expansions N`: expansion limit.
- `--max-analysis-ms N`: planning time limit. Once reached, expansion variants, derived
  constraints and diagnostics are skipped. If the composition check has not run, every
  contract is `UNRESOLVED` and the scope is partial (`analysis.skipped_for_time`).

## Counterexample reduction

Saved counterexamples are shrunk before they are traced and written:
1. The seed-derived canaries are replaced with fixed distinct values. If the mismatch
   persists, the bundle is marked `seed_independent`.
2. Every shared entry value is tried at 0, at 1, at successive halvings, and with single
   bits cleared, keeping a candidate only if it is simpler, satisfies the constraints, and
   reproduces the same mismatch kind.

The bundle records `reduction` (runs, steps, the original initial state, and the non-zero
reduced inputs). `replay` runs the reduced case. The program is never reduced.

## Mutation analysis

`NO_MISMATCH_FOUND` means no counterexample under the budget. Mutation analysis measures
how much that says: it plants would-be miscompiles in passing regions and counts how
many the same test finds.

- **Mutants.** Every single-bit flip of each new-side instruction (not hints, branches
  or inline asm) that still decodes to a region-executable instruction of the same size
  with different text; one mutant per distinct text. Each is classed by what changed in
  the disassembly: `opcode`, `register`, `constant` (inline constant, literal, offset),
  `modifier` (neg, abs, op_sel, clamp, omod, cache bits) or `other`. Up to `--max` per
  region (48) are sampled round-robin over the classes, in an order fixed by the seed.
  The mutant's VGPR and SGPR writes are re-derived, so a mutant that writes another
  register is compared, not rejected as an undeclared effect.
- **Verdicts.** A mutant is tested like the region (same budget; reduction off; stops
  at its first mismatch):
  - `killed`: a mismatch, or a clobbered frame field. Phase `static` when the
    pending-memory analysis already rejects it: a mutant that reads a load's
    destination before a wait retires it (a weakened `s_waitcnt`) leaves the region's
    context unresolved in the real pipeline, while the isolated run completes loads at
    once;
  - `killed_late`: found only by a re-run with `--confirm-cases` random cases (20000): the
    budget is too small for it;
  - `survived`: never found;
  - `invalid`: the mutant left the model (no case runs on both sides); not counted.
- **Survivors.** Each survivor is run against the new side it came from, on the same
  inputs (`--effect-cases`, 2000): `no_effect` (identical final states: an ignored
  integer modifier, an aliased shift amount, a cache-policy bit), `dead_effect` (only
  locations not live at the exit differ: an extra write to a dead register),
  `undefined_effect` (a live-out location that is undefined at the entry, and that the
  original passes through, now holds another value: any value refines an undefined one)
  or `observable_effect` (a live-out value differs where the test compares it: the
  comparison with old tolerated it, for example a NaN payload, or the test is blind to
  it). Narrow return values and active-lanes-only VGPRs are compared as the test
  compares them. The first three are equivalent mutants, not miscompiles.
- **Score.** `killed / (killed + killed_late + survived)`, and without equivalent mutants
  `killed / (killed + killed_late + observable_effect survivors)`.
- **Campaign.** `rj_isa_diff_mutants` runs `mutants` over every plain pass of
  `rj_isa_diff_pr` results (`NO_MISMATCH_FOUND`, complete coverage, no don't-care
  masking) and writes `report.md` and `summary.json`: totals, the phase of the first
  kill, breakdowns by class, unit (SALU, VALU, memory) and target, survivors by effect,
  and the survivors themselves, observable effects first.
- Every mutant is synthetic: nothing here is an LLVM finding.

## Learning hooks

- `InputPolicy::propose(features, contract, history, case)` receives a bounded `History`:
  counts, categories, and the last 32 feedbacks.
- `observe(Feedback)` receives:
  - the phase and input events;
  - the acceptance or rejection reason;
  - both exit statuses and the outcome category;
  - the differing locations;
  - a bounded novelty bucket (outcome plus coarse value classes, never raw values) and
    whether it is new;
  - per-stage cost and instruction counts.
- `ConstraintChecker::validate` gates every proposal.
- `test` writes `logs/<region>.jsonl`:
  - a header with the per-instruction opcode/effect summary, roles, operand pairs, literal
    relations and the contract;
  - one line per case;
  - a summary.

## Rejections

A region is `UNSUPPORTED`, with a reason, if it contains any of:

- opcodes outside the executable subset: memory other than `global_load_*`,
  `global_store_*` and `s_load_*` (LDS, flat, buffer, scratch, atomics,
  `global_load_lds_*`, `_tr` loads), MFMA, `v_interp*`, `v_movrel*`, `v_clrexcp`,
  barriers, calls, returns, indirect jumps, `s_getreg`/`s_setreg`, `s_movrel*`,
  messages/traps, and SALU instructions not on the allowlist;
- operands outside the modelled state: TTMP, FLAT_SCRATCH, XNACK_MASK, apertures, AGPRs;
- PC-derived inputs, tracked by forward taint from `s_getpc_b64`;
- instructions whose bytes are patched by a relocation.

A region is `INCONCLUSIVE` if control enters it other than at its start, if an exit has
no partner on the other side, or if R_out cannot be established.

Kernel-level items keep coverage `PARTIAL`:

- control-flow hunks that no region covers;
- descriptor fields or this kernel's metadata fields that differ (a change in
  `kernel_code_entry_byte_offset` alone is layout only, and a register-allocation-only
  change that both kernels fit is informational);
- relocations that differ at matched instructions;
- changed unreachable bytes;
- unknown CFG successors;
- a layout shift in a kernel that reads the PC;
- diff hunks outside a manual region.

Execution failures are classified and never treated as no-ops: decode failure,
missing execution callback (model-only provider), unimplemented, illegal, host
exception, ISA undefined behaviour (RocJITsu's isa-undefined counter), PC written, and
memory fault.

## Files

- `case.json`:
  - artifacts with hashes, relocation counts, and whether metadata was parsed;
  - LLVM provenance;
  - change hunks, unresolved items, and regions (original words, PCs, disassembly);
  - contracts and `prepared_key`;
  - expansion variants and diagnostic regions;
  - the analysis cost split: `compile_ns`, `parse_ns`, `decode_cfg_ns`, `dataflow_ns`,
    `align_classify_ns`, `plan_ns`.
- `results.json`:
  - overall and per-region statuses, counts, and rejection reasons;
  - executed-instruction counters, which show that only the region ran;
  - per-phase timing (setup, propose, state load, old, new, compare, triage);
  - unrestricted runs, diagnostics, cache statistics, and counterexample paths.
- `counterexamples/<region>-case<k>.json`: a self-contained bundle with:
  - the region and contract;
  - both complete initial and final states (reduced);
  - seed and case index;
  - per-instruction traces and the first differences;
  - the reduction record;
  - artifacts and provenance;
  - tool/RocJITsu identity.
- `logs/<region>.jsonl`: the per-case learning log (see "Learning hooks").

`docs/schema/rj_isa_diff.v1.schema.json` (JSON Schema 2020-12) defines all four.
`docs/schema/validate_outputs.py` validates existing files, or drives the CLI over
built-in fixtures and validates everything it writes.

`prepared_key` hashes the region words, the PCs, the full contract and the execution
mode. A change in downstream liveness therefore yields a new key even when the region
bytes are identical.

## Input generation

Every `test` run has two phases.

**1. Mandatory boundary sweep** (`boundary.{h,cpp}`). It always runs, in full, before
any random case, and there is no flag to skip it. `--cases` counts only the random
phase, so `--cases 0` still runs the whole sweep. The sweep is deterministic: the same
region, contract and seed always give the same cases. It draws from named catalogs:

| catalog | values |
|---|---|
| `int` | 0, 1, 2, 3, -1, -2, INT_MIN, INT_MAX and their neighbours |
| `edge` | byte/half edges: 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, sign-extended forms |
| `small` / `bits` / `bit` | 31, 32, 33, 63, 64; alternating, nibble and byte patterns; each single bit and its inverse |
| `f32` | +-0, +-Inf, quiet/signalling NaN, NaN payload extremes, min/max denormal, min/max normal, +-1, 2^31, 2^32 |
| `f16`, `f16x2`, `bf16`, `bf16x2` | the same specials in the low half and packed |
| `int64`, `f64`, `bits64`, `exec` | for 64-bit operands: int64 limits, half-crossing values (2^32-1, 2^32), f64 specials, EXEC lane patterns (lane 0/31/32/63, low/high 32 lanes) |

Operand roles, found by decoding the region, add error-prone values where they matter:

| role | added values |
|---|---|
| shift amount or bit index (`s_lshl/lshr/ashr`, `s_bfm`, `s_bitset/bitcmp`) | 32, 33, 63, 64, 65, 127, 128, ...; high garbage over a zero low field |
| `s_bfe` control | offset x width edges, including width 0/32/33/64/127, offset 31/32/63/64, and garbage high bits |
| sign-extension input | 0x7f/0x80/0x7fff/0x8000, with and without garbage above |
| register combined with a literal `c` (add, sub, compare, min/max, mul, select, bitwise) | c, c+-1, c+-2, -c, -c+-1, ~c, c^sign, INT_MAX-c, INT_MIN-c and neighbours |
| two register sources of one such instruction | carry (UINT_MAX+1), signed overflow (INT_MAX+1, INT_MIN-1), borrow (0-1), equal, complementary, negated, adjacent, mul wrap, NaN/Inf pairs, and a 12x12 core cross product; 64-bit pairs get equal-high/differ-low, half-crossing and an 8x8 cross |
| contract constraints | umin/umax, both neighbours of each, and known-bit extremes |

Phases: uniform (every entry register set to each catalog value), single (each register
the region reads, over zeros/ones/seeded-random bases, with both SCC values when SCC is
an input), pair64, pairs, and constraint edges. Bases are projected onto the
constraints. A sweep value that violates a constraint is rejected and counted, never
changed. Identical input states are run once. `--max-boundary-cases` (default 250000)
is only a safety cap: reaching it is reported as `truncated` with coverage `PARTIAL`.
`results.json` reports `boundary_sweep` (cases, valid, mismatches, truncated, valid
cases per catalog family) separately from `random_phase`. Every boundary counterexample
records `phase: boundary` and the exact catalog label (for example
`single s2=constrel.0x00012345.c base=zeros`).

**2. Random phase.** `InputPolicy` is the pluggable interface (`propose`/`observe`), and
`validate_sample` keeps constraints out of the policy's hands. P0 ships `BaselinePolicy`,
which is deterministic in (seed, case) and draws, per register:

- 20%: a role value, then 25%: a catalog value (see the table above, including the
  NaN/Inf/denormal encodings), for a register with a role; 45% catalog values for a
  register without one;
- 10%: a single bit or its inverse;
- 15%: an equal or complementary copy of another input;
- the rest: uniformly random values.

64-bit operands draw from the 64-bit catalog one time in three. SCC alternates with the
case parity. Learned policies are future work, and may only propose inputs.

## Tests and measurements

- `rj_isa_diff_tests` (ctest prefix `isa_diff.`) holds the validator suite: positive,
  negative and rejection cases from handoff section 13, plus the manual region,
  register map, metadata, time limit, cache and diagnostic tests.
- Fixtures come from `tests/isa_diff/fixtures/fixtures.asm`, assembled once by
  `regen_fixtures.sh`; these tests never run LLVM.
- `rj_isa_diff_llvm_tests` (prefix `isa_diff_llvm.`) is built only when `llc` is found
  at configure time. It covers the plugin, the real LLVM pairs, relocations,
  per-kernel metadata, inline-asm marking, and a callable function with inline asm end to
  end.
- `validate_outputs.py` also drives a callable function with inline asm through the LLVM
  plugin (`fn`, `fn_wrong`), commuted NaN results (`nan`, `nan_strict` with replay), an
  undefined value (`undef`), and the poison filter (`poison`, `poison_filtered`), so the
  new fields are validated on real output.
- `isa_diff_llvm.pr_driver` (`llvm/test_rj_isa_diff_pr.py`) runs `rj_isa_diff_pr` on a
  synthetic two-commit llvm-project history and checks every verdict: NaN equivalence, an
  undefined element, a `half` return, poison triage (a hint on a `mismatch`; none for a
  masked shift amount), a real difference, and in two-llc mode an llc failure on one side
  (`INCOMPLETE`).
- `isa_diff.schema_and_cli` runs `validate_outputs.py` against the CLI and the committed
  samples. It is skipped when python3-jsonschema is missing.
- `docs/samples/regen.sh` regenerates the samples (`codeobject`, `synthetic`,
  `llvm_pair`).
- `bench/bench.py` measures cost against region size, with a whole-kernel control.
- Results are in `docs/findings.md`. Planned P1/P2 work is in `docs/p1_p2_design.md`.
