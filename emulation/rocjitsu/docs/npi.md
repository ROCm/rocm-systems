# Adding a new GPU

This guide is not complete.

## Marking NPI tasks

Mark tasks that need to be done when introducing a new product with `\NPI` comment.

Single-line markers use any comment leader (e.g. `//`):

```c
// \NPI a one-line todo
```

Multi-line markers use a `/* */` block comment, ending each continued line with a
trailing `\` (a `//` comment ending in `\` is a line-continuation that GCC rejects
under `-Werror=comment`):

```c
/*
 * \NPI I am a big complicated todo \
 * that takes multiple lines to describe.
 */
```

`scripts/find-npi-tasks.sh` strips the comment leaders and joins the continued
lines into a single task.

## Find npi tasks

Use `scripts/find-npi-tasks.sh` to find tasks.

## Extra Steps

What the grep can't point at are the "create a new thing" steps:

- Construct a `configs/<gpu>.json` (plus the `_kmd` and any multi-GPU variant)
  for the new device.
- For a brand-new ISA family: sync `shared/machine-readable-isa` with
  `download.py`, regenerate the ISA/DBT sources per [codegen.md](codegen.md),
  and author the hand-written per-arch files (`isa.h`, `insts.h`, `mma_exec.h`,
  `addr_calc.h/.cpp`, ...) under
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<isa>/`.
- Add HIP kernel coverage under `tests/kernels/`.

## Reminder

Don't leak new products.
