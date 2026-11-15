# rj_isa_diff: incremental ISA-to-ISA differential testing

`rj_isa_diff` compares an old and a new compile of the same AMDGPU kernel. It finds
where the machine code changed, cuts a contiguous region around each change, and
writes an explicit entry/exit contract for that region. It then runs both versions of
the region on paired inputs through RocJITsu's own instruction execution
(`ComputeUnitCore::execute_instruction`). Each case executes **only the region**; the
whole kernel is decoded and analysed once.

P0 scope: gfx950 (CDNA4), wave64, straight-line integer SALU regions inside one basic
block. The kernel around a region may contain anything; the region may not.

The P0 inputs are AMDGPU code objects and raw instruction words. Reading `.s` through
LLVM MC, and producing old/new pairs from LLVM switches, are a separate plugin task
(see `docs/plan.md`).

## Commands

```sh
# 1. Decode, align, classify, plan regions and contracts.
rj_isa_diff analyze --old old.hsaco --new new.hsaco --kernel my_kernel --out case/
#   [--context N]       execute N common-prefix instructions before the change
#   [--max-region N]    region length limit (default 256)
#   [--no-merge]        do not merge hunks that share a basic block
#   [--old-kernel/--new-kernel NAME] when the symbol name differs

# 2. Paired-input differential test of every READY region.
rj_isa_diff test --case case/ --seed 1 --cases 10000 [--region r0]
#   the boundary/NaN/corner sweep always runs first; --cases adds random cases
#   [--max-boundary-cases N]  safety cap (default 250000); hitting it => PARTIAL
#   [--reduction-budget N | --no-reduce]  counterexample reduction (default 4096 runs)
#   [--no-case-log]  skip the per-case JSONL learning log (case/logs/<region>.jsonl)
#   after a local mismatch, precomputed larger regions (common context) are tried
#   automatically

# 3. Re-execute a counterexample from its recorded words (no artifact, no LLVM).
rj_isa_diff replay --counterexample case/counterexamples/r0-case17.json [--case case/]

# Validator self-test: inject a wrong opcode/immediate/operand (labelled synthetic).
rj_isa_diff inject --case case/ --region r0 --side new --inst 1 --xor 0x01000000 --out syn/
```

An input file is either an ELF code object (detected by its magic) or a raw
little-endian instruction stream that is treated as one kernel starting at offset 0.

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

## The contract

For each region pair (`case.json` → `regions[].contract`):

- **R_in**: equality on `entry_equal`, the union of both sides' live-ins at the
  cutpoint. It compares the same physical location on both sides; no renaming is
  inferred. Every other location (`canary`) gets an independent value per side, so a
  dependency that liveness missed shows up as a mismatch.
- **C**: `constraints` (known bits, unsigned range, equality), each with its source. A
  constraint written by hand is `user_assumption`, and editing a case after analysis
  downgrades `ESTABLISHED` to `ASSUMED`.
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

- non-allowlisted opcodes: VALU, memory, MFMA, `s_waitcnt`, barriers, branches,
  `s_getreg`/`s_setreg`, `s_movrel*`, messages/traps;
- operands outside the modelled state: TTMP, FLAT_SCRATCH, XNACK_MASK, apertures;
- PC-derived inputs, tracked by forward taint from `s_getpc_b64`.

A region is `INCONCLUSIVE` if it spans basic blocks or if R_out cannot be established.

Kernel-level items keep coverage `PARTIAL`:

- control-flow hunks;
- descriptor or metadata differences (a change in `kernel_code_entry_byte_offset` alone
  is layout only);
- changed unreachable bytes;
- unknown CFG successors;
- a layout shift in a kernel that reads the PC.

Execution failures are classified and never treated as no-ops: decode failure,
missing execution callback (model-only provider), unimplemented, illegal, host
exception, ISA undefined behaviour (RocJITsu's isa-undefined counter), and PC written.

## Files

- `case.json`: artifacts with hashes, change hunks, unresolved items, regions
  (original words, PCs, disassembly), contracts, `prepared_key`, analysis timing.
- `results.json`: overall and per-region statuses, counts, rejection reasons,
  executed-instruction counters (proving region-only execution), per-phase timing,
  and counterexample paths.
- `counterexamples/<region>-case<k>.json`: a self-contained bundle with the region,
  contract, both complete initial and final states (reduced), seed, case index,
  per-instruction traces, first differences, the reduction record, and tool/RocJITsu
  identity.
- `logs/<region>.jsonl`: the per-case learning log (see "Learning hooks").

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
  negative and rejection cases from handoff section 13.
- Fixtures come from `tests/isa_diff/fixtures/fixtures.asm`, assembled once by
  `regen_fixtures.sh`; the tests never run LLVM.
- `bench/bench.py` measures cost against region size, with a whole-kernel control.
- Results are in `docs/findings.md`.
