# LLVM MIR Prototype — Three-Way Comparison

Comparison of the three binary translation prototype approaches: **waveasm**, **Aster (amdgcn dialect)**, and **LLVM MIR**.

## Pipeline Overview

| Aspect | waveasm | Aster | LLVM MIR |
|--------|---------|-------|----------|
| **IR Level** | Custom MLIR dialect (ISA-level) | MLIR amdgcn dialect (ISA-level with typed regs) | LLVM MachineInstr (post-RA MIR) |
| **Instruction ID** | Mnemonic string matching | Mnemonic string matching | Opcode integer (from MCInstrDesc) |
| **Implicit Operands** | Not modeled | Partially modeled (VCC only, manual) | **Automatic from TableGen** |
| **Register Model** | Precolored virtual regs | Typed physical regs (SGPR/VGPR/VCC/EXEC) | Physical regs with register classes |
| **Control Flow** | Linear (no CFG) | Multi-block with VCC branching | Multi-block with exec-mask branching |
| **Assembly Generation** | Custom emitter + splicing | Aster translateModule + patching | MCInstPrinter (disasm → reassembly) |
| **Kernel Metadata** | Spliced from original | Manual patching | Manual patching |
| **HSACO Pipeline** | External llvm-mc + ld.lld | External llvm-mc + ld.lld | External llvm-mc + ld.lld |

## Pipeline Results (vecadd, gfx942 same-ISA)

| Metric | waveasm | Aster | LLVM MIR |
|--------|---------|-------|----------|
| Instructions lifted | 292/292 (100%) | 23/23 unique (100%) | 23/23 (100%) |
| Unsupported mnemonics | 0 | 0 | 0 |
| GPU execution | PASS | PASS | PASS |
| Correct elements | 1024/1024 | 1024/1024 | 1024/1024 |

## Implicit Operand Comparison

The key differentiator for LLVM MIR is automatic implicit operand modeling.

**Example: `v_cmp_gt_i32_e32`**

- **waveasm**: `waveasm.raw "v_cmp_gt_i32_e32" %s4, %v0` — no VCC or EXEC tracked
- **Aster**: `amdgcn.cmpi(V_CMP_GT_I32, %s4, %v0) → %vcc : !amdgcn.vcc<0>` — VCC added manually
- **LLVM MIR**: `V_CMP_GT_I32_e32_vi $sgpr4, $vgpr0, implicit-def $vcc, implicit $exec` — VCC def and EXEC use auto-populated from MCInstrDesc

**Example: `s_and_saveexec_b64`**

- **waveasm**: Not modeled (EXEC implicit def/use invisible)
- **Aster**: Elided and converted to VCC-based branching
- **LLVM MIR**: `$sgpr2_sgpr3 = S_AND_SAVEEXEC_B64_vi $vcc, implicit-def $exec, implicit-def $scc, implicit $exec` — all three implicit operands visible

**Example: `v_add_f32_e32`**

- **waveasm**: `waveasm.raw "v_add_f32_e32" %v2, %v6, %v7` — no $mode or $exec
- **Aster**: Aster ops carry exec implicitly by convention
- **LLVM MIR**: `$vgpr2 = V_ADD_F32_e32_vi $vgpr6, $vgpr7, implicit $mode, implicit $exec` — $mode tracked too

## MIR Quality

The generated MIR is human-readable YAML, directly comparable to upstream LLVM test cases:

```yaml
body: |
  bb.0:
    successors: %bb.1(0x40000000), %bb.2(0x40000000)
    $sgpr3 = S_LOAD_DWORD_IMM_vi $sgpr0_sgpr1, 44, 0
    ...
    V_CMP_GT_I32_e32_vi $sgpr4, $vgpr0, implicit-def $vcc, implicit $exec
    $sgpr2_sgpr3 = S_AND_SAVEEXEC_B64_vi $vcc, implicit-def $exec, implicit-def $scc, implicit $exec
    S_CBRANCH_EXECZ_vi %bb.2, implicit $exec
  bb.1:
    ...
    $vgpr2 = V_ADD_F32_e32_vi $vgpr6, $vgpr7, implicit $mode, implicit $exec
    GLOBAL_STORE_DWORD_vi $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
  bb.2:
    S_ENDPGM_vi 0
```

## Key Advantages of LLVM MIR

1. **Implicit operands are free** — every `MachineInstr` automatically carries its implicit defs/uses from TableGen, solving the #1 limitation of both waveasm and Aster
2. **Opcode-based dispatch** — no mnemonic string parsing; `MCInstrDesc` carries all instruction metadata
3. **Register classes** — operands tagged with correct AMDGPU register classes
4. **Path to backend passes** — MIR is the right form for `SIInsertWaitcnts`, scheduling, hazard recognition
5. **Standard format** — MIR YAML is the same format used by LLVM developers for testing

## Key Limitations

See [SHORTCUTS_AND_LIMITATIONS.md](SHORTCUTS_AND_LIMITATIONS.md) for details. The main gap vs the other prototypes:
- No in-process AsmPrinter pipeline (llc couldn't process our post-RA MIR due to `declare` function)
- Assembly generation uses MCInstPrinter round-trip rather than MIR → AsmPrinter path
- Kernel metadata is hardcoded (same challenge as Aster)

## When to Choose Each Approach

| Use Case | Best Choice | Reason |
|----------|-------------|--------|
| Analysis / optimization passes | LLVM MIR | Full implicit operand info, backend pass compatibility |
| Cross-ISA translation | Aster or LLVM MIR | Need semantic understanding beyond raw bytes |
| Quick same-ISA hotswap | waveasm | Simplest pipeline, least overhead |
| Wait counter recomputation | LLVM MIR | Can run `SIInsertWaitcnts` pass directly |
| Register pressure analysis | LLVM MIR | `MachineRegisterInfo` with proper reg classes |
| High-level optimization | Aster → MLIR | Can raise to higher MLIR dialects |
