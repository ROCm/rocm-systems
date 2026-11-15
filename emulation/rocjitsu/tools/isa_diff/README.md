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
- **Status**: `ESTABLISHED` when every path to the region runs byte-identical code from
  identical metadata. `ASSUMED` when R_in relies on upstream regions' R_out or on an
  unchecked single-entry assumption (unknown CFG successors). `UNRESOLVED` when a
  descriptor/metadata change, an unsupported or inconclusive upstream region, or a
  control-flow change can reach the entry.

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
  contract, both complete initial and final states, seed, case index, per-instruction
  traces, first differences, and tool/RocJITsu identity.

`prepared_key` hashes the region words, the PCs, the full contract and the execution
mode. A change in downstream liveness therefore yields a new key even when the region
bytes are identical.

## Input generation

`InputPolicy` is the pluggable interface (`propose`/`observe`). `ConstraintChecker`
(`validate_sample`) keeps constraints out of the policy's hands. P0 ships
`BaselinePolicy`, which is deterministic in (seed, case) and draws:

- boundary values;
- 64-bit pair boundaries;
- equal or complementary operands;
- random values;
- SCC alternating with the case parity.

Learned policies are future work, and may only propose inputs.

## Tests and measurements

- `rj_isa_diff_tests` (ctest prefix `isa_diff.`) holds the validator suite: positive,
  negative and rejection cases from handoff section 13.
- Fixtures come from `tests/isa_diff/fixtures/fixtures.asm`, assembled once by
  `regen_fixtures.sh`; the tests never run LLVM.
- `bench/bench.py` measures cost against region size, with a whole-kernel control.
- Results are in `docs/findings.md`.
