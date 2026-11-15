# rj_isa_diff: incremental ISA-to-ISA differential testing

`rj_isa_diff` compares an old and a new compile of the same AMDGPU kernel. It finds
where the machine code changed, cuts a contiguous region around each change, and
writes an explicit entry/exit contract for that region. It then runs both versions of
the region on paired inputs through RocJITsu's own instruction execution
(`ComputeUnitCore::execute_instruction`). Each case executes **only the region**; the
whole kernel is decoded and analysed once.

Scope: gfx950 (CDNA4), wave64, straight-line regions inside one basic block, made of
integer SALU instructions (an explicit allowlist) and plain VALU instructions. VALU
excludes memory, MFMA, interpolation, indirect VGPR access, and exception state. The
kernel around a region may contain anything; the region may not.

Inputs are AMDGPU code objects, raw instruction words, and, through the LLVM plugin
(`llvm/`), `.s` and `.ll` sources. The core never runs LLVM; the plugin is a separate
library that turns sources into code objects and records how.

## Commands

```sh
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
#   .s/.ll inputs: [--llvm-bin DIR] [--llc-args S] [--old-llc-args S] [--new-llc-args S]

# 1b. LLVM plugin: compile one .ll twice, with different llc options per side.
rj_isa_diff llvm-pair --input k.ll --common-args "-O3" --old-args "" \
    --new-args "-combiner-disabled" --out case/   # + any analyze option

# 2. Paired-input differential test of every READY region.
rj_isa_diff test --case case/ --seed 1 --cases 10000 [--region r0]
#   the boundary/NaN/corner sweep always runs first; --cases adds random cases
#   [--max-boundary-cases N]  safety cap (default 250000); hitting it => PARTIAL
#   [--reduction-budget N | --no-reduce]  counterexample reduction (default 4096 runs)
#   [--no-case-log]  skip the per-case JSONL learning log (case/logs/<region>.jsonl)
#   [--cache-dir DIR]  reuse results of identical prepared regions and budgets
#   after a local mismatch, precomputed larger regions (common context) are tried
#   automatically

# 3. Re-execute a counterexample from its recorded words (no artifact, no LLVM).
rj_isa_diff replay --counterexample case/counterexamples/r0-case17.json [--case case/]

# Validator self-test: inject a wrong opcode/immediate/operand (labelled synthetic).
rj_isa_diff inject --case case/ --region r0 --side new --inst 1 --xor 0x01000000 --out syn/
```

An input file is an ELF code object (detected by its magic), a `.s` or `.ll` source
(LLVM plugin), or a raw little-endian instruction stream treated as one kernel starting
at offset 0.

Exit codes:

| code | meaning |
|---|---|
| 0 | `NO_MISMATCH_FOUND` (or `NO_SEMANTIC_DIFF`) and coverage `COMPLETE_FOR_DECLARED_SCOPE` |
| 1 | `LOCAL_MISMATCH` (`replay`: did not reproduce) |
| 2 | `INCONCLUSIVE`, `UNSUPPORTED`, or partial coverage |
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
- When a contract carries constraints derived from the common prefix, the region is also
  run without them, and the result's `unrestricted` field reports that run. The
  constrained result is the verdict; `constraint_dependent: true` says the rewrite is
  only correct under the derived facts.

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
  and trap/wave state are fixed and must be preserved.
- **Effect check**: each side may change only its `declared_writes_*`. Any other
  change is a `HARNESS_ERROR`, because it means the effect model is wrong.
- **Pending effects** (`pending_effects`): memory operations that may still be in flight at
  the entry, from an `s_waitcnt` dataflow over the gfx9 counters.
  - `NONE` and `DISJOINT` (in flight, but no destination is read or written by the region)
    leave the contract unaffected.
  - `CONFLICT` and `UNKNOWN` leave the context unresolved: the isolated run starts with
    every input ready, and real hardware might not.
- **Composition** (`composition`): a forward must-equality analysis from the kernel entry,
  through common code and upstream regions' R_out, must establish every `entry_equal`
  location (`HOLDS`). Otherwise it reports `FAILS`, naming the missing locations.
- **Status**:
  - `ESTABLISHED` when every path to the region runs byte-identical code from identical
    metadata.
  - `ASSUMED` when R_in relies on upstream regions' R_out, on untested upstream changes
    writing only their decoded destinations, or on an unchecked single-entry assumption
    (unknown CFG successors).
  - `UNRESOLVED` when kernel metadata differs, a control-flow change can reach the entry,
    composition fails, or pending effects conflict. An unresolved context is also listed
    in `unresolved`, so the overall result is never a pass.

## Context expansion

A local mismatch may be compensated by common code around the change (handoff 5.2).
`analyze` precomputes up to 12 larger variants of each READY region, stored in
`case.json` under `expansion_variants`. Each variant adds matched common P0 instructions
from the same block, before and/or after the change (`r0.ctx-b<before>f<after>`). Each
has its own contract.

After a base region shows `LOCAL_MISMATCH`, `test` runs these variants smallest first. The
base result keeps its local mismatch, and `expansion.resolution` records the outcome:
- `RESOLVED_BY_CONTEXT`: a variant has no mismatch, and the overall verdict uses that
  variant's result and contract;
- `PERSISTS_WITH_CONTEXT`;
- `NO_EXPANSION_AVAILABLE`.

`--context N` still adds manual backward context to every base region.

## VALU support

VGPRs are part of the state, with all 64 lanes of each one.

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
- **Liveness.** VGPR liveness is a backward dataflow in which a write never kills,
  because lanes outside EXEC keep their value.
  - `vgpr_entry_equal` and `vgpr_observable` come from this liveness, restricted to
    the tracked range.
  - A renamed dead VGPR temporary is not observable.
- **Tracked range.** The contract tracks `v0..v{vgpr_count-1}`: every VGPR either side
  names, rounded up to the allocation granule of 8. The runner allocates exactly that
  range, so an access beyond it is an invalid register access, never a silent write.
- **Comparison.** Every lane of every observable VGPR is compared, active or not. A
  rewrite that clobbers lanes outside EXEC is caught.
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
- Only `s0`..`s101` can be mapped, because EXEC/VCC/M0/SCC aliases must stay consistent.
- `mapping` becomes `user_register_map`, with a `user_assumption` source. The status is
  at most `ASSUMED`.

A wrong map shows up as a mismatch, or as `INCONCLUSIVE_MAPPING` when the translated exit
liveness disagrees. It is never inferred from outputs.

## LLVM plugin

`llvm/` (`rj_isa_diff_llvm`) finds `llc`, `llvm-mc`, `ld.lld` and `llvm-objcopy` in
`--llvm-bin`, `$RJ_ISA_DIFF_LLVM_BIN`, or `/opt/rocm/llvm/bin`, and runs them with
`posix_spawn` (no shell).

- `.ll` goes through `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 <common> <side>`.
  Side arguments may not change the target (`-mcpu`, `-mattr`, `-mtriple`, `-march`,
  `-o` and `-filetype` are rejected).
- `.s` goes through `llvm-mc`, then `ld.lld -shared --emit-relocs`. `--emit-relocs`
  keeps `.rela.text`, so instructions patched by the linker stay identifiable; the
  executed bytes are unchanged.
- A listing without a kernel descriptor falls back to its raw `.text`
  (`executed_object: raw_text`).

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
| analysis (`analyze`) | both artifacts' SHA-256, kernel names, every planning option |
| result (`test`) | the prepared region (instructions, PCs, complete contract), the prepared keys of its expansion variants, policy, seed and every budget |

Both keys also bind the tool and schema versions and a build fingerprint: a SHA-256 of
the running executable and every executable file it maps. Any rebuild therefore
invalidates everything.

- Identical region bytes under a different contract miss: a downstream liveness change
  alters `observable`, and with it the key.
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
- A prefix with control flow or non-P0 instructions (`s_load`, ...) makes the
  diagnostic `UNSUPPORTED` with that reason.

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

- opcodes outside the executable subset: memory (SMEM, VMEM, LDS), MFMA, `v_interp*`,
  `v_movrel*`, `v_clrexcp`, `s_waitcnt`, barriers, branches, `s_getreg`/`s_setreg`,
  `s_movrel*`, messages/traps, and SALU instructions not on the allowlist;
- operands outside the modelled state: TTMP, FLAT_SCRATCH, XNACK_MASK, apertures, AGPRs;
- PC-derived inputs, tracked by forward taint from `s_getpc_b64`;
- instructions whose bytes are patched by a relocation.

A region is `INCONCLUSIVE` if it spans basic blocks or if R_out cannot be established.

Kernel-level items keep coverage `PARTIAL`:

- control-flow hunks;
- descriptor fields or this kernel's metadata fields that differ (a change in
  `kernel_code_entry_byte_offset` alone is layout only);
- relocations that differ at matched instructions;
- changed unreachable bytes;
- unknown CFG successors;
- a layout shift in a kernel that reads the PC;
- diff hunks outside a manual region.

Execution failures are classified and never treated as no-ops: decode failure,
missing execution callback (model-only provider), unimplemented, illegal, host
exception, ISA undefined behaviour (RocJITsu's isa-undefined counter), and PC written.

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
  at configure time. It covers the plugin, the real LLVM pairs, relocations and
  per-kernel metadata.
- `isa_diff.schema_and_cli` runs `validate_outputs.py` against the CLI and the committed
  samples. It is skipped when python3-jsonschema is missing.
- `docs/samples/regen.sh` regenerates the samples (`codeobject`, `synthetic`,
  `llvm_pair`).
- `bench/bench.py` measures cost against region size, with a whole-kernel control.
- Results are in `docs/findings.md`. Planned P1/P2 work is in `docs/p1_p2_design.md`.
