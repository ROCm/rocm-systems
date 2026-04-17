# Refactor Plan

Running list of things in the transpiler that should be made more
principled — data duplicated from LLVM, ad-hoc dispatch, string-based
lookups, copy-pasted code, stale abstractions, anything that's grown
past its original shape. Add a bullet when you spot one; delete it
when it's gone. One-liners unless a plan is actually useful.

## Open

*(none — add entries here)*

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
