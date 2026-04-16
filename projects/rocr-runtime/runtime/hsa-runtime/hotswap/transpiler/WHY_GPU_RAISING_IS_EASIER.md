# Why GPU Kernel Raising Is Easier Than Classical Binary Lifting

> **Notes:** Binary lifting (raising machine code to compiler IR) is notoriously
> hard on general-purpose CPUs. Our GPU transpiler sidesteps nearly all of these
> problems because GPU compute kernels lack the properties that make classical
> raising hard. We do a 1:1 instruction-to-IR translation, then let the LLVM
> backend re-derive instruction selection, register allocation, and scheduling
> from scratch for the new target. The raised IR is a **waypoint**, not a
> destination.

---

## Why GPU kernels avoid classical raising problems

1. **CFG recovery is trivial** — All AMDGPU branches (`s_branch`,
   `s_cbranch_scc0/1`, etc.) use PC-relative immediate offsets. No indirect
   jumps, no computed gotos, no jump tables. A single linear scan gives the
   full CFG.

2. **SSA construction is delegated** — The entire physical register file is
   modeled as LLVM allocas. One call to `PromoteMemToReg` produces proper SSA
   with phi nodes. Works cleanly because GPU registers have no aliasing (unlike
   x86 rax/eax/ax/al overlaps).

3. **Type recovery is free** — Types are encoded in instruction mnemonics
   (`v_add_f32` = float, `v_add_u32` = unsigned int). Kernel argument types
   come from `.amdgpu_metadata`. No type inference needed.

4. **Flag forwarding is structural** — SCC/VCC are modeled as `i1` allocas
   that participate in normal SSA promotion. Auto-SCC writeback uses hardware
   `implicit_defs()` metadata, making it structurally impossible to miss a
   flag write.

5. **No stack frame recovery** — GPU kernels have no traditional stack. Memory
   accesses use well-defined address spaces encoded in the opcode (global, LDS,
   buffer). Kernel arguments arrive via a known ABI with ELF metadata declaring
   every argument's offset, size, and type.

6. **No function boundary recovery** — Kernels are self-contained:
   single-entry, terminated by `s_endpgm`, no calls. The boundary comes from
   ELF symbol metadata; the calling convention is `AMDGPU_KERNEL` with the
   argument layout in `.amdgpu_metadata`.

7. **Information loss is a non-issue** — The information destroyed during
   original compilation (instruction selection, register allocation, scheduling,
   wait counters) is exactly what the LLVM backend re-derives from first
   principles for the new target.





## The one open problem: EXEC mask divergence

EXEC is a bitmask controlling which SIMT lanes are active. The raiser models
it as a scalar alloca, which is correct when all lanes execute uniformly (the
common case for production kernels we handle) but cannot represent true per-lane
divergence. A complete solution would require a vector-lane register model or
an IR with explicit lane semantics.
