# Refactor Plan

Running list of things in the transpiler that should be made more
principled — data duplicated from LLVM, ad-hoc dispatch, string-based
lookups, copy-pasted code, stale abstractions, anything that's grown
past its original shape. Add a bullet when you spot one; delete it
when it's gone. One-liners unless a plan is actually useful.

## Open

- **M0 register: audit all producers and consumers for correctness.**
  M0 is used as an implicit address operand by several instruction families
  and the current handling is ad-hoc across handler files:
  - `handle_mubuf.cpp`: `buffer_load_*_lds` stores to LDS at M0, but does
    not advance M0 afterwards. If multiple `buffer_load_*_lds` fire in
    sequence the raiser relies on the kernel having explicit `s_mov_b32 m0`
    instructions between them. Verify this assumption against real kernels.
  - `handle_sopc.cpp`: `s_set_gpr_idx_on` writes M0 but the GPR dynamic
    indexing effect is not modeled. This is documented as a known limitation
    but should be revisited if AITER kernels start using GPR indexing.
  - `reg_file.hpp`: `LDS_DIRECT` reads from LDS at M0. On GFX9 there is
    no auto-increment of M0 (unlike GFX11+ DSDIR `lds_direct_load`), so
    the current implementation is correct. But if GFX11+ kernels are ever
    raised, the DSDIR auto-increment will need explicit modeling.
  - `handle_ds.cpp`: `ds_bpermute` uses M0 for byte-lane control but the
    handler passes M0 through correctly. No known issue.
  - General concern: M0 is a single 32-bit alloca shared by all these
    uses. If a kernel interleaves M0 uses (e.g. buffer_load_lds followed
    by ds_bpermute) the raiser must preserve M0's value correctly across
    the entire instruction stream. Add a test that exercises interleaved
    M0 usage patterns.

## How to work on one

- Prefer the authoritative source: LLVM helpers / generated tables
  (`MCInstrDesc::TSFlags`, `AMDGPU::getNamedOperandIdx`,
  `MCRegisterInfo`, `Intrinsic::getType`, feature bits, …),
  existing shared types, existing helpers in this tree. Don't
  re-derive what already exists.
- Make drift and unhandled cases loud: the "unexpected input" path is
  `report_fatal_error`, never a silent fallback.
- One surface per commit. Validate with `transpiler_tests`.

## What not to do

- Don't move hand-authored rows into a `.def` file — same data,
  different location.
- Don't try to eliminate `SemOp`. It is our dispatch key; LLVM has
  no equivalent.
