# Why GPU Kernel Raising Is Easier Than Classical Binary Lifting

Binary lifting — raising machine code to a compiler IR — is a notoriously hard
problem in the general case.  Tools like McSema, Remill, and RetDec represent
decades of research and still struggle with fundamental ambiguities.  Our LLVM
IR raiser sidesteps most of these problems entirely, not through cleverness, but
because **GPU compute kernels lack the properties that make classical raising
hard**.

This document explains the classical problems, why they don't apply here, and
the one area where a GPU-specific challenge remains.

---

## The Core Insight

During compilation (lowering), information is destroyed.  A high-level `fma`
becomes a `mul` + `add`.  A struct access becomes pointer arithmetic + a load.
A loop becomes a flat stream of branches.  Classical raising tries to
**reconstruct** this lost information: pattern-match the `mul` + `add` back
into `fma`, infer the struct layout, recover the loop nest.

Our raiser does not attempt any of this.  It translates each machine instruction
to its LLVM IR equivalent **1:1**, then feeds the result into LLVM's standard
AMDGPU backend, which re-derives instruction selection, register allocation,
and scheduling from scratch.  The raised IR is a **waypoint**, not a
destination — we only need enough semantic content for `llc` to produce correct
machine code for a different target.

For example, a scalar add with carry-out detection (`raiser.cpp`):

```c++
// handle_sop2.cpp — s_add_u32 handler
if (sop == SemOp::S_ADD_U32) {
    Value *src0 = op.src(0), *src1 = op.src(1);
    Value *res = B.CreateAdd(src0, src1, "add");
    regs.writeReg32(B, op.dst(), res);
    auto *ov = B.CreateIntrinsic(Intrinsic::uadd_with_overflow, {i32Ty}, {src0, src1});
    regs.storeSCC(B, B.CreateExtractValue(ov, 1));
    sccHandled = true; handled = true; break;
}
```

A float add with VOP3 modifier support:

```c++
// handle_valu.cpp — v_add_f32 handler
if (sop == SemOp::V_ADD_F32) {
    Value *s0 = op.srcF(0), *s1 = op.srcF(1);  // srcF() applies neg/abs mods
    if (s0->getType() != f32Ty) s0 = B.CreateBitCast(s0, f32Ty);
    if (s1->getType() != f32Ty) s1 = B.CreateBitCast(s1, f32Ty);
    regs.writeReg32(B, op.dst(), B.CreateBitCast(B.CreateFAdd(s0, s1, "fadd"), i32Ty));
    handled = true; break;
}
```

No pattern recovery, no multi-instruction analysis — just 1:1 translation.

---

## Classical Problems and Why They Don't Apply

### 1. Control Flow Graph Recovery

**The classical problem.**  On x86/ARM, recovering the CFG is the hardest
unsolved problem.  Indirect jumps (`jmp rax`), computed gotos, jump tables,
virtual dispatch, and tail calls make it impossible in general to determine
where basic blocks start and end without solving a potentially undecidable
analysis.

**Why it doesn't apply.**  AMDGPU branch instructions (`s_branch`,
`s_cbranch_scc0/1`, `s_cbranch_vccnz`, `s_cbranch_execz/nz`) all use
**PC-relative immediate offsets**.  There are no indirect branches in compute
kernels.  A single linear scan over the `.text` section computes every branch
target.  The full CFG falls out of one pass — no heuristics, no iterative
analysis, no "is this code or data?" ambiguity.

The entire CFG recovery is this loop in Phase 1 of the raiser:

```c++
// raiser.cpp — branch target collection
if (di.isBranch) {
    for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
        if (inst.getOperand(i).isImm()) {
            int64_t raw = inst.getOperand(i).getImm();
            int64_t brOff = (int64_t)(int16_t)(uint16_t)(raw & 0xFFFF);
            blockStarts.insert(off + 4 + brOff * 4);
        }
    }
    if (di.isConditionalBranch)
        blockStarts.insert(off + instSize);
}
```

Every branch target is computed as `offset + 4 + immediate * 4`.  The set of
`blockStarts` directly becomes the set of `BasicBlock`s:

```c++
// raiser.cpp — basic blocks from branch targets
std::map<uint64_t, BasicBlock *> offsetToBB;
for (uint64_t addr : blockStarts)
    offsetToBB[addr] = BasicBlock::Create(C, "bb_0x" + utohexstr(addr - kernelOffset), F);
```

That's it.  No iterative refinement, no jump table analysis, no function
pointer resolution — just a `std::set` populated in a single linear scan.

### 2. SSA Construction

**The classical problem.**  Lifting flat machine code into SSA form normally
requires alias analysis, stack slot recovery, and complex dataflow analysis.
On x86, dozens of overlapping register aliases (rax/eax/ax/ah/al) and implicit
flag definitions make this especially painful.

**Why it doesn't apply.**  We model the entire physical register file as LLVM
`AllocaInst`s — 106 SGPRs, 256 VGPRs, 256 AGPRs, plus special registers (VCC,
SCC, EXEC, M0, FLAT_SCR, TTMPs):

```c++
// reg_file.hpp — the full GPU register file as allocas
struct AllocaRegFile {
  static constexpr int MAX_SGPR = 106;
  static constexpr int MAX_VGPR = 256;
  AllocaInst *sgpr[106] = {};
  AllocaInst *vgpr[256] = {};
  AllocaInst *agpr[256] = {};
  AllocaInst *vcc = nullptr;
  AllocaInst *scc = nullptr;
  AllocaInst *exec = nullptr;
  AllocaInst *m0 = nullptr;
  AllocaInst *flatScr[2] = {};
  static constexpr int MAX_TTMP = 16;
  AllocaInst *ttmp[16] = {};
  // ...
  void init(IRBuilder<> &B, Type *i32Ty, Type *i1Ty, const ISAProfile &isa) {
    execTy = isa.isWave32() ? (Type *)i32Ty : (Type *)B.getInt64Ty();
    for (int i = 0; i < MAX_SGPR; i++)
      sgpr[i] = B.CreateAlloca(i32Ty, nullptr, "sgpr" + std::to_string(i));
    for (int i = 0; i < MAX_VGPR; i++)
      vgpr[i] = B.CreateAlloca(i32Ty, nullptr, "vgpr" + std::to_string(i));
    // ...
    exec = B.CreateAlloca(execTy, nullptr, "exec");
    B.CreateStore(ConstantInt::getSigned(execTy, -1), exec);  // EXEC = all-ones
    // ...
  }
};
```

After raising all instructions, a **single call** to `PromoteMemToReg` converts
these allocas into proper SSA with phi nodes:

```c++
// raiser.cpp — alloca → SSA in one call
DominatorTree DT(*F);
AssumptionCache AC(*F);
SmallVector<AllocaInst *, 512> allocas;
regs.collectAllocas(allocas);
PromoteMemToReg(allocas, DT, &AC);
```

LLVM's standard SSA construction handles dominance, loops, and merge points
automatically — we write zero lines of SSA construction code.

This works cleanly because GPU registers have **no aliasing**.  Unlike x86
where `rax` overlaps `eax`/`ax`/`al`, each `sgpr[N]` or `vgpr[N]` is an
independent 32-bit slot.  Multi-dword accesses (64-bit, 128-bit) are explicit
lo/hi decomposition in the alloca model.

### 3. Type Recovery

**The classical problem.**  Machine code is untyped.  A register holds 32 bits
that could be an integer, a float, a pointer, or packed half-precision.
Traditional decompilers run iterative type inference passes that propagate
constraints through the dataflow graph, often requiring heuristics and
producing incomplete results.

**Why it doesn't apply.**  AMDGPU instructions encode the type in the
mnemonic: `v_add_f32` is float, `v_add_u32` is unsigned integer,
`v_cvt_f32_f16` names input and output types explicitly.  The raiser reads
the mnemonic and emits correctly-typed LLVM IR directly — `CreateFAdd` for
float ops, `CreateAdd` for integer ops.  There is no type inference pass
because the ISA tells you the types.

For kernel arguments, the ELF `.amdgpu_metadata` section declares which
arguments are `global_buffer` (pointer) and which are scalar values, along
with their sizes and offsets.  The raiser constructs a properly-typed LLVM
function signature directly from this metadata:

```c++
// raiser.cpp — build typed function signature from ELF metadata
SmallVector<Type *, 8> paramTypes;
for (auto &arg : meta.args) {
    if (arg.valueKind.rfind("hidden_", 0) == 0)
        continue;  // skip hidden implicit args
    bool isPtr = (arg.valueKind == "global_buffer");
    Type *ty;
    if (isPtr)      ty = ptrGlobalTy;    // ptr addrspace(1)
    else if (arg.size == 8) ty = i64Ty;
    else            ty = i32Ty;
    paramTypes.push_back(ty);
    // ...
}
auto *funcTy = FunctionType::get(voidTy, paramTypes, false);
Function *F = Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
F->setCallingConv(CallingConv::AMDGPU_KERNEL);
```

### 4. Condition Code / Flag Recovery

**The classical problem.**  On x86, `EFLAGS` is a single register written by
almost every ALU instruction and consumed by branches, conditional moves, and
setcc.  Recovering which flag-setting instruction feeds which flag-consuming
instruction — across arbitrary intervening code — is a major analysis problem
known as "flag forwarding."

**Why it doesn't apply.**  SCC and VCC are modeled as `i1` allocas that
participate in normal alloca-to-SSA promotion — no special analysis needed.
Additionally, the raiser uses an **auto-SCC writeback** mechanism: hardware
metadata (`MCInstrDesc::implicit_defs()`) determines which instructions
define SCC.  Handlers that compute SCC explicitly (carry, compare) set
`sccHandled = true`; everything else gets automatic `SCC = (result != 0)`
writeback.  Forgetting a flag write is **structurally impossible**.

The auto-writeback runs after every instruction handler:

```c++
// raiser.cpp — auto SCC writeback from hardware metadata
if (di.defsSCC && !sccHandled && sccResult) {
    Value *zero = Constant::getNullValue(sccResult->getType());
    regs.storeSCC(B, B.CreateICmpNE(sccResult, zero));
}
```

And `defsSCC` itself comes from a single metadata query during disassembly:

```c++
// raiser.cpp — implicit defs from MCInstrDesc
for (MCPhysReg r : desc.implicit_defs()) {
    StringRef rn = mc.regInfo->getName(r);
    if (rn == "SCC") di.defsSCC = true;
    else if (rn.starts_with("VCC")) di.defsVCC = true;
    else if (rn.starts_with("EXEC")) di.defsEXEC = true;
}
```

Hardware metadata is exhaustive by construction — if a new instruction defines
SCC, it will appear in `implicit_defs()` automatically.

### 5. Stack Frame / Memory Model Recovery

**The classical problem.**  x86 decompilers must recover stack frames from
flat memory accesses through `rsp`/`rbp` — identifying local variables,
spill slots, outgoing arguments, and frame layouts.

**Why it doesn't apply.**  GPU compute kernels have no stack in the traditional
sense.  Memory accesses fall into well-defined address spaces — `global` (flat
pointer), `LDS/shared` (`ds_read`/`ds_write`), `buffer` (descriptor-based
MUBUF) — and the address space is encoded in the instruction opcode.  Kernel
arguments arrive through a known ABI (kernarg pointer in `s[0:1]`, workgroup
and workitem IDs in fixed registers).  The ELF metadata tells you every
argument's offset, size, and type.

The raiser initializes entry-point registers from the ABI:

```c++
// raiser.cpp — ABI-defined register initialization
// s[0:1] = kernarg segment pointer
regs.storeSGPR64(B, 0, Constant::getNullValue(PointerType::get(C, 4)));
// s2 = workgroup_id_x
regs.storeSGPR32(B, 2, B.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
// s3 = workgroup_id_y
regs.storeSGPR32(B, 3, B.CreateCall(fnWorkgroupIdY, {}, "wg_id_y"));
// v0 = workitem_id_x
regs.storeVGPR32(B, 0, B.CreateCall(fnWorkitemIdX, {}, "tid"));
```

### 6. Function Boundary / Calling Convention Recovery

**The classical problem.**  Identifying function boundaries on x86 requires
heuristic analysis of prologues/epilogues, call/ret instruction patterns,
and stack frame setup.  Recovering calling conventions (which registers are
arguments, which are callee-saved) requires cross-function analysis.

**Why it doesn't apply.**  GPU compute kernels are self-contained — entered at
a known symbol offset, terminated by `s_endpgm`, with no calls to other
functions.  There are no return addresses, no callee-saved registers, no stack
frames to unwind.  The kernel boundary is given by ELF symbol metadata, and the
calling convention is `AMDGPU_KERNEL` with the argument layout declared in
`.amdgpu_metadata`.  There is nothing to recover.

---

## What the Backend Re-Derives

The information that IS lost during the original compilation is exactly the
information that the LLVM backend re-derives from first principles:

| Information lost during original lowering | Who recovers it |
|-------------------------------------------|-----------------|
| Instruction selection (fma vs. mul+add)   | `llc` instruction selector |
| Register allocation                       | `llc` register allocator (after `PromoteMemToReg`) |
| Instruction scheduling                    | `llc` machine scheduler |
| Wait counter placement                    | `SIInsertWaitcnts` pass in the AMDGPU backend |
| Kernel descriptor and metadata            | `llc` generates `.amdhsa_kernel` from function attributes |

We are not trying to undo the original compiler's decisions.  We are giving the
**target** compiler a fresh chance to make its own decisions for a different ISA.

---

## The One Problem That Does NOT Disappear: EXEC Mask Divergence

Classical raising has one GPU-specific analog with no CPU counterpart: the EXEC
mask.  EXEC is a 64-bit bitmask controlling which SIMT lanes are active.  When
EXEC narrows (e.g., via `s_and_saveexec_b64`), inactive lanes preserve their
old register values while active lanes receive new ones.

Our raiser models EXEC as a real alloca with truthful tracking — save, modify,
branch, and restore operations all use the actual EXEC value:

```c++
// handle_sop1.cpp — s_and_saveexec: save old EXEC, EXEC = EXEC & src
if (sop == SemOp::S_AND_SAVEEXEC_B32) {
    Value *oldExec = regs.loadExec(B);
    Value *src = op.srcExecWidth(0);
    regs.writeRegExecWidth(B, op.dst(), oldExec);   // save old EXEC to dest
    Value *newExec = B.CreateAnd(oldExec, src, "new_exec");
    regs.storeExec(B, newExec);                      // EXEC = EXEC & src
    sccResult = newExec;
    handled = true; break;
}
```

But each register holds **one** scalar value, not 64 per-lane values.  This is
correct when all lanes execute uniformly (the common case for the production
kernels we handle), but cannot represent true per-lane divergence.

This is the one area where information loss from the original compilation is
real and unrecoverable within a scalar IR model: the compiler knew which values
were uniform and which were divergent, and that knowledge is not encoded in the
binary.  A complete solution would require a vector-lane register model
(`<64 x i32>` per VGPR) or an intermediate representation with explicit lane
semantics (e.g., an MLIR dialect).

---

## Summary

| Classical raising problem          | CPU difficulty | GPU kernel difficulty | Why |
|------------------------------------|---------------|----------------------|-----|
| CFG recovery (indirect jumps)      | Very hard     | Trivial              | All branches are PC-relative immediates |
| SSA construction                   | Hard          | Delegated            | Alloca + `PromoteMemToReg` |
| Type recovery                      | Hard          | Free                 | Types encoded in mnemonics |
| Flag / condition code forwarding   | Hard          | Structural           | Auto-SCC via `implicit_defs()` metadata |
| Stack frame recovery               | Hard          | N/A                  | No stack; address spaces in opcodes |
| Function boundary recovery         | Medium        | N/A                  | Single-entry `s_endpgm`; ELF metadata |
| Information loss (instruction selection) | The core problem | Non-issue   | Backend re-derives from IR |
| EXEC mask / lane divergence        | N/A           | Open problem         | Scalar model; correct for uniform flow |

Classical binary lifting is hard because general-purpose CPUs have indirect
jumps, overlapping registers, untyped instructions, stack-based memory, and
cross-function calls.  GPU compute kernels have **none of these**.  Our
transpiler exploits every one of these structural properties to reduce what is
normally a research-grade problem to a modular ~30-file codebase with 100%
raise rate on 27 production kernels and verified cross-ISA transpilation
(gfx1250 RDNA4 → gfx942 CDNA3).
