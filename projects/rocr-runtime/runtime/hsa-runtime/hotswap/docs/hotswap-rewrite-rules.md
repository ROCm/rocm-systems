# HotSwap Custom Rewrite Rules for gfx950 → gfx942

## Problem

AITER CK kernels compiled for gfx950 contain `v_cvt_scalef32_pk_fp4_f32` instructions
(432 out of 129,569 total — 0.3%) that don't exist on gfx942. The retarget engine
successfully re-encodes the other 99.7% of instructions, but the batch assembly
fails because of these 432 instructions.

## Key Discovery

gfx942 and gfx950 share **identical instruction encodings** for all standard
VALU/SMEM/VMEM/SOPP instructions. The batch re-assembly step is unnecessary
for this ISA pair — only the 432 gfx950-only instructions need handling.

## Strategy

**NOP pre-pass + emulation trampolines:**

1. **NOP pre-pass** (implemented): Replace gfx950-only instructions with `s_nop`
   pairs. The remaining 135,614 instructions are already valid gfx942 encoding.
2. **Emulation trampolines** (next): Replace the NOPs with `s_branch` to
   trampoline regions containing software emulation sequences assembled for gfx942.

## gfx950-Only Instructions in AITER

| Instruction | Count | Encoding | Description |
|---|---|---|---|
| `v_cvt_scalef32_pk_fp4_f32` | 432 | `D23D` (8 bytes) | Convert 2x f32 → packed FP4 E2M1 with scale |

### v_cvt_scalef32_pk_fp4_f32 Semantics

```
v_cvt_scalef32_pk_fp4_f32 vDst, vSrc0, vSrc1, vScale
```

- **Src0, Src1:** Two f32 values to convert
- **Scale:** f32 scale factor (multiply before quantization)
- **Dst:** uint32 with two 4-bit FP4 E2M1 values packed in bits [7:0]

FP4 E2M1 representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

### gfx942 Emulation

The FP4 conversion can be emulated using standard VALU instructions:

```asm
; Emulate v_cvt_scalef32_pk_fp4_f32 vDst, vSrc0, vSrc1, vScale
; Step 1: Scale the inputs
v_mul_f32 vTmp0, vSrc0, vScale
v_mul_f32 vTmp1, vSrc1, vScale
; Step 2: Clamp to FP4 range [-6, 6]
v_max_f32 vTmp0, -6.0, vTmp0
v_min_f32 vTmp0, 6.0, vTmp0
v_max_f32 vTmp1, -6.0, vTmp1
v_min_f32 vTmp1, 6.0, vTmp1
; Step 3: Quantize to nearest E2M1 value (lookup table or comparison chain)
; Step 4: Pack two 4-bit values into vDst
```

This expands from 8 bytes to ~60+ bytes — requires a trampoline.

## Implementation Plan

### Step 1: Pre-filter in RetargetCodeObject

Before the batch assembly, scan for gfx950-only instructions and replace them
with NOPs + trampoline branches. The trampoline contains the emulated sequence
assembled for gfx942.

### Step 2: FP4 Emulation Trampoline

Write the FP4 conversion emulation as an assembly sequence that:
- Uses 2 temporary VGPRs (requires `extra_vgprs: 2` in kernel descriptor)
- Implements E2M1 quantization via comparison chain
- Packs results into the destination register
- Branches back to the instruction after the original

### Step 3: Apply-before-retarget Hook

Add a new phase to `RetargetCodeObject` that runs rewrite rules BEFORE the
batch assembly. This removes gfx950-only instructions so the batch assembly
succeeds.

### Step 4: Kernel Descriptor Update

Update COMPUTE_PGM_RSRC1 VGPR count to account for temporary registers
used by the emulation trampolines.

## Alternatives Considered

- **Skip FP4 instructions entirely (NOP them out):** Would work for rmsnorm
  (the FP4 conversion is used for output quantization, not the core computation).
  Simpler but lossy — the output won't be quantized to FP4.

- **Use replace_bytes to patch each instruction individually:** Works but
  requires knowing every unique encoding. The trampoline approach is more general.

- **Compile CK without FP4 for gfx942 compatibility:** The right long-term
  solution but requires CK changes, not just hotswap rules.
