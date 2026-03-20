# gfx1250 → gfx942 Cross-Family ISA Transpiler

## Implementation Status (2026-03-19)

**18/20 test kernels passing** on MI300X (gfx942) via load-time transpilation of gfx1250 binaries.

| Category | Tests | Status |
|----------|-------|--------|
| Scalar compute (15 kernels) | vector_add, saxpy, relu, reduce_sum, dot_product, matrix_add_2d, clamp, stencil_1d, max_reduce, sqrt_vals, int_elementwise, fma_vals, exp_kernel, recip_vals, mag2d | All PASS |
| Dense matmul (12 shapes) | 8x8 to 2048x16x128, hipBLAS-LT-inspired | All PASS |
| Softmax (7 shapes) | Row-wise with shared-mem reduction, exp, reciprocal | All PASS |
| Fused attention (4 shapes) | QK^T + softmax + PV, with sqrtf/division | PASS (when code fits .text) |
| Split-K matmul (8 shapes) | 2D grid (blockIdx.y = split dimension) | 3/8 PASS (>256 blocks fail) |
| Multi-head attention (4 shapes) | 2D grid (blockIdx.y = head), 576 instructions | 0/4 (VGPR allocation mismatch) |

**Implementation:** ~3,200 lines in `transpiler.cpp`. Full disassemble→translate→reassemble pipeline
using LLVM MC. MCInst-level taint analysis for TTMP workgroup ID chains. 15 wave32→wave64 bugs
found and fixed.

**Target:** Currently transpiles to gfx942 (MI300X, CDNA3). The plan below was originally for
gfx950 (MI355X, CDNA4) but the implementation targets gfx942 for testing on available hardware.
gfx950 support would require minimal changes (different MACH value, some instruction differences).

---

## Original Plan

## Executive Summary

A load-time ISA transpiler that converts gfx1250 (RDNA4) GPU binaries to run on gfx950
(CDNA4/MI355X) hardware. Unlike the gfx950→gfx942 HotSwap (same-family, 98.4% encoding-
identical), this is a cross-family translation requiring full disassemble→translate→reassemble
of every instruction, plus wave size adaptation and matrix instruction substitution.

**Revised feasibility assessment:** More tractable than initially expected. VOP1, VOP2, VOPC,
and SOPP share identical binary encoding formats between GFX9 and GFX12. The hard parts are
wave size adaptation, memory instruction re-encoding, wait counter translation, and WMMA→MFMA
matrix instruction substitution.

## Architecture Comparison

| Property | gfx1250 (RDNA4, source) | gfx950 (CDNA4, target) |
|---|---|---|
| **Generation** | GFX12 | GFX9 |
| **MACH value** | 0x049 | 0x4f |
| **Encoding family** | GFX1250 (12) | GFX9 (5) / GFX940 (9) |
| **Wave size** | Wave32 only | Wave64 only |
| **VGPRs** | 1024 VGPR (no AGPR) | 512 VGPR + 512 AGPR |
| **LDS** | 327,680 bytes | 163,840 bytes |
| **SGPRs** | 106 | 102 |
| **Matrix unit** | WMMA/SWMMAC (32-wide) | MFMA (64-wide) |
| **Scratch model** | Architected flat scratch | Buffer scratch |
| **Wait model** | s_wait_*cnt (split) | s_waitcnt (combined) |
| **FLAT encoding** | 96-bit (24-bit offset) | 64-bit (13-bit offset) |

## Encoding Compatibility Analysis

### Shared Encodings (No Translation Needed for Format)

These encoding formats are **bit-for-bit identical** between GFX9 and GFX12:

| Format | Size | Encoding ID | Notes |
|---|---|---|---|
| **VOP1** | 32-bit | 0x3f [31:25] | src0[8:0], opcode[16:9], vdst[24:17] — identical |
| **VOP2** | 32-bit | 0x0 [31] | src0[8:0], src1[16:9], vdst[24:17], opcode[30:25] — identical |
| **VOPC** | 32-bit | 0x3e [31:25] | src0[8:0], src1[16:9], opcode[24:17] — identical |
| **SOPP** | 32-bit | 0x17f [31:23] | simm16[15:0], opcode[22:16] — identical |

**Caveat:** The encoding *format* is identical, but the *opcode values* within each format
may differ between GFX9 and GFX12. Each instruction needs an opcode mapping table.

### Different Encodings (Require Re-encoding)

| Format | GFX9 | GFX12/GFX1250 | Delta |
|---|---|---|---|
| **VOP3** | 64-bit, ID=0x34 | 64-bit, ID=0x35 | 1-bit encoding ID change only |
| **SMEM** | 64-bit, ID=0x30, 8-bit opcode, 21-bit offset | 64-bit, ID=0x3d, 6-bit opcode, 24-bit offset | Full restructure |
| **FLAT/GLOBAL** | 64-bit, ID=0x37, 13-bit offset | 96-bit, ID=0x3b, 24-bit offset | Size change + restructure |
| **DS** | 64-bit, ID=0x36, gds@[17] | 64-bit, ID=0x36, gds@[16], acc@[25] | Minor field shift |
| **VOP3P** | 64-bit | 64-bit | Opcode mappings differ |
| **MIMG** | 64-bit (GFX9) | 96-bit (GFX12) | Different format entirely |

### Encoding-Absent (Require Emulation or Substitution)

| Source (gfx1250) | Target (gfx950) | Strategy |
|---|---|---|
| WMMA | MFMA | Matrix shape substitution + lane remap |
| SWMMAC | MFMA + sparsity mask | Software sparsity emulation |
| s_wait_loadcnt/storecnt/etc | s_waitcnt | Counter merging |
| SALU float (s_add_f32, etc) | VALU (v_add_f32) | Promote to VALU with readfirstlane |
| VOPD (dual-issue) | VOP1+VOP2 pair | Split into two instructions |
| Flat scratch (architected) | Buffer scratch | Descriptor-based rewrite |
| DPP8 | DPP (dpp_ctrl) | Swizzle pattern translation |

## Implementation Plan

### Phase 1: Opcode Mapping Tables

Build exhaustive GFX1250→GFX9 opcode translation tables for each encoding format.

**Approach:** Parse LLVM TableGen definitions to auto-generate mapping tables.
The `getMCOpcodeGen` table in LLVM already contains the mapping — each row maps a
pseudo-opcode to encoding-family-specific opcodes. Extract columns 5 (GFX9) and 12
(GFX1250).

```
LLVM Source: compiler/amd-llvm/llvm/lib/Target/AMDGPU/SIInstrInfo.td (line 3350)
  getMCOpcodeGen table:
    KeyCol = SIEncodingFamily.NONE
    ValueCols = [SI, VI, SDWA, SDWA9, GFX80, GFX9, GFX10, SDWA10, GFX90A, GFX940, GFX11, GFX12, GFX1250]
                                                     ^^^                                           ^^^^^^^
```

**Deliverable:** `opcode_tables.h` with:
- `vop1_gfx1250_to_gfx9[256]` — VOP1 opcode remapping
- `vop2_gfx1250_to_gfx9[64]` — VOP2 opcode remapping
- `vop3_gfx1250_to_gfx9[1024]` — VOP3 opcode remapping
- `sopp_gfx1250_to_gfx9[128]` — SOPP opcode remapping
- `smem_gfx1250_to_gfx9[64]` — SMEM opcode remapping
- `flat_gfx1250_to_gfx9[128]` — FLAT opcode remapping
- `ds_gfx1250_to_gfx9[256]` — DS opcode remapping

**Estimated coverage:** ~80-90% of instructions will have direct 1:1 opcode mappings.
Instructions with no GFX9 equivalent go to the emulation table (Phase 4).

### Phase 2: Wave Size Adaptation

The core semantic challenge. gfx1250 kernels assume 32 lanes per wave; gfx950 has 64.

**Strategy: Half-wave execution.** Run the wave32 kernel in the lower 32 lanes of a
wave64, with the upper 32 lanes permanently masked off via EXEC.

```
Initialization (prepended to kernel):
  s_mov_b32 exec_hi, 0           ; mask off lanes 32-63
  s_mov_b32 exec_lo, 0xFFFFFFFF  ; enable lanes 0-31
```

**What this handles:**
- All VALU instructions: wave64 VALU processes 64 lanes but only 32 are active.
  Results in lanes 32-63 are don't-care. ~50% ALU efficiency, but correct.
- LDS operations: only 32 lanes participate, so LDS layout matches wave32 expectations
- Scalar operations: unaffected (scalar = 1 value per wavefront)
- Branch/control flow: EXEC masking for divergence uses 64-bit EXEC, but only lower
  32 bits are meaningful

**What needs special handling:**

| Pattern | gfx1250 (wave32) | gfx950 (wave64) | Translation |
|---|---|---|---|
| EXEC save/restore | `s_mov_b32 s[N], exec_lo` | `s_mov_b64 s[N:N+1], exec` (save both halves) | Widen to 64-bit, keep exec_hi=0 |
| EXEC manipulation | `s_and_b32 exec_lo, ...` | `s_and_b64 exec, ...` (zero-extend mask) | Widen operation, force exec_hi=0 |
| Ballot/readlane | `v_readlane_b32 sN, vN, lane` | Same instruction, but lane < 32 always | No change needed |
| readfirstlane | `v_readfirstlane_b32` | Same | No change (finds first active in 0-31) |
| DPP within 16 lanes | `dpp_ctrl` patterns | Same `dpp_ctrl` | Compatible — DPP row ops within 16 lanes |
| DPP8 | `dpp8` format | `dpp_ctrl` format | Translate swizzle; DPP8 not on GFX9 |
| Permute > 32 lanes | N/A (wave32) | N/A | Not an issue — source is wave32 |

**SGPR pressure:** Wave32→wave64 EXEC widening uses one extra SGPR pair per save/restore.
The kernel descriptor SGPR count needs adjustment.

**Workgroup dispatch:** The runtime dispatch must account for the wave size difference.
If the gfx1250 kernel expects `workgroup_size / 32` waves, gfx950 will launch
`workgroup_size / 64` waves (half as many). Each wave64 does the work of one wave32
(with 32 lanes idle). Need to double the number of waves or accept 50% occupancy.

**Implementation:** Scan for all EXEC-manipulating instructions and widen:
```
; Source (gfx1250 wave32):
s_mov_b32 exec_lo, s4          ; set exec from saved mask

; Target (gfx950 wave64):
s_mov_b32 exec_lo, s4          ; lower 32 bits = original mask
s_mov_b32 exec_hi, 0           ; upper 32 bits = disabled
```

### Phase 3: Instruction Re-encoding

For each encoding format that differs, implement a translation function.

#### 3a. VOP3 (trivial)

Change encoding ID from 0x35 to 0x34. All other fields identical.
```c
// DW0[31:26] = 0x35 (gfx1250) → 0x34 (gfx9)
dw0 = (dw0 & 0x03FFFFFF) | (0x34 << 26);
// Then remap opcode DW0[25:16]
uint16_t op = (dw0 >> 16) & 0x3FF;
dw0 = (dw0 & ~(0x3FF << 16)) | (vop3_map[op] << 16);
```

#### 3b. SMEM (significant)

GFX12 SMEM has 6-bit opcode and 24-bit offset; GFX9 has 8-bit opcode and 21-bit offset.

```
GFX12:  [sbase 5:0] [sdst 12:6] [opcode 18:13] [--- 25:19] [0x3d 31:26] | [offset 55:32] [soffset 63:57]
GFX9:   [sbase 5:0] [sdst 12:6] [--- 13] [SOffsetEn 14] [--- 15] [GLC 16] [imm 17] [opcode 25:18] [0x30 31:26] | [offset 52:32] [soffset 63:57]
```

Must extract fields, remap opcode, truncate offset (24→21 bits; abort if offset > 2^20),
reconstruct GFX9 format.

#### 3c. FLAT/GLOBAL (size change: 96→64 bit)

gfx1250 FLAT is 96-bit; gfx950 FLAT is 64-bit. This is a **size-reducing** translation
(good — the output is smaller, so it fits in the original space with room for NOPs).

Extract vaddr, vdata, saddr, vdst, offset from 96-bit GFX12 format.
Truncate offset (24→13 bits; abort if offset > 4095).
Reconstruct 64-bit GFX9 format.
Pad remaining 4 bytes with `s_nop 0`.

#### 3d. DS (minor)

Shift gds bit [16]→[17], remove acc bit [25], keep encoding ID 0x36.
Remap opcode.

#### 3e. MIMG (size change: 96→64 bit on GFX9)

Similar to FLAT — extract fields, remap to GFX9 MIMG format.
NSA (non-sequential address) encoding in GFX12 needs conversion to sequential
vaddr in GFX9 (may require register move preamble if addresses are non-contiguous).

### Phase 4: Instruction Emulation (Trampolines)

Instructions that exist on gfx1250 but not gfx950, requiring software emulation.

#### 4a. Wait Counter Translation

```
; Source (gfx1250):
s_wait_loadcnt 0        ; wait for all loads
s_wait_storecnt 0       ; wait for all stores
s_wait_dscnt 0          ; wait for all LDS/GDS

; Target (gfx950):
s_waitcnt vmcnt(0) lgkmcnt(0)  ; combined wait
```

**Mapping:**
- `s_wait_loadcnt N` → accumulate into vmcnt
- `s_wait_storecnt N` → accumulate into vscnt (GFX10+) or vmcnt
- `s_wait_dscnt N` → accumulate into lgkmcnt
- `s_wait_kmcnt N` → accumulate into lgkmcnt
- `s_wait_expcnt N` → accumulate into expcnt
- `s_wait_samplecnt N` → accumulate into vmcnt
- `s_wait_bvhcnt N` → accumulate into vmcnt

Multiple consecutive `s_wait_*` can merge into one `s_waitcnt`. Conservative: set all
counts to 0 (correctness over performance).

#### 4b. SALU Float → VALU Promotion

gfx1250 has scalar float instructions; gfx950 doesn't.

```
; Source (gfx1250):
s_add_f32 s4, s4, s5

; Target (gfx950):
v_mov_b32 v_tmp0, s4         ; move to VALU
v_add_f32 v_tmp0, v_tmp0, s5 ; VALU add
v_readfirstlane_b32 s4, v_tmp0 ; move back to SGPR
```

Each SALU float → 3-instruction trampoline. Needs 1 temp VGPR.

#### 4c. VOPD (Dual-Issue) → Two Separate Instructions

gfx1250 can pack two VOP operations into one VOPD encoding.

```
; Source (gfx1250, 32-bit):
v_dual_add_f32_fmac_f32 v0, v1, v2 :: v3, v4, v5

; Target (gfx950, 2x 32-bit):
v_add_f32 v0, v1, v2
v_fmac_f32 v3, v4, v5
```

Size: 32-bit → 64-bit (expands). Requires trampoline if in tight loop,
or use NOP sled space.

#### 4d. WMMA → MFMA Matrix Translation

The most complex emulation. WMMA and MFMA compute the same matrix math but with
different lane-to-element mappings and different wave widths.

**Compatible shape pairs (same M×N×K math):**

| gfx1250 WMMA (wave32) | gfx950 MFMA (wave64) | Src Regs | Dst Regs | Lane Remap? |
|---|---|---|---|---|
| v_wmma_f32_16x16x16_f16 | v_mfma_f32_16x16x16f16 | 8xf16→4xf16 | 8xf32→4xf32 | Yes |
| v_wmma_f32_16x16x16_bf16 | v_mfma_f32_16x16x16bf16 (via 1k) | Similar | Similar | Yes |
| v_wmma_i32_16x16x16_iu8 | v_mfma_i32_16x16x16i8 | V4I8→V4I8 | 4xI32→4xI32 | Yes |
| v_wmma_f32_16x16x32_f16 | v_mfma_f32_16x16x32_f16 | 8xf16→8xf16 | 4xf32→4xf32 | Yes |
| v_wmma_f32_16x16x32_bf16 | v_mfma_f32_16x16x32_bf16 | 8xbf16→8xbf16 | 4xf32→4xf32 | Yes |

**Lane mapping problem:** In wave32 WMMA, each lane holds 8 elements of the output
matrix (16×16 = 256 elements / 32 lanes = 8 per lane). In wave64 MFMA, each lane holds
4 elements (256 / 64 = 4 per lane). The data needs reshuffling.

**Two-phase approach:**
1. **Pre-MFMA shuffle:** Rearrange input register data from WMMA lane layout to MFMA lane layout
2. **Execute MFMA:** Run the gfx950-native matrix multiply
3. **Post-MFMA shuffle:** Rearrange output from MFMA lane layout back to WMMA lane layout

The shuffle sequences use `ds_permute_b32` / `ds_bpermute_b32` (LDS-based cross-lane
permutation) and `v_readlane_b32` / `v_writelane_b32` for lane-specific data movement.

**Estimated overhead per WMMA→MFMA:** ~20-40 extra instructions for shuffle (vs 1 WMMA).
For compute-bound kernels this is a ~20-40x slowdown on the matrix portion.

**Alternative:** If the kernel is from a framework (PyTorch, etc.), the matrix layout is
abstracted by the compiler. The data in memory is row-major or column-major regardless of
lane layout. We could skip the lane shuffle and instead pre/post-transpose the matrix
tiles in shared memory. This is what a JIT compiler would do.

#### 4e. Flat Scratch → Buffer Scratch

gfx1250 uses architected flat scratch (`scratch_load_*` instructions with FLAT_SCRATCH
register). gfx950 uses buffer scratch with an SGPR descriptor.

**Translation:** Replace `scratch_load_dword vN, off, soff` with:
```asm
buffer_load_dword vN, off, s[scratch_desc:scratch_desc+3], soff offen
```

Requires the kernel descriptor to set up a scratch buffer descriptor in the user SGPRs.
The descriptor base address comes from the runtime's scratch allocation.

### Phase 5: Kernel Descriptor Translation

The 64-byte AMD HSA kernel descriptor has the same base layout but different field
interpretations.

| Field | gfx1250 | gfx950 | Translation |
|---|---|---|---|
| COMPUTE_PGM_RSRC1[21] | ENABLE_WG_RR_EN | ENABLE_DX10_CLAMP | Set to 0 (disable both) |
| COMPUTE_PGM_RSRC1[23] | DISABLE_PERF | ENABLE_IEEE_MODE | Map appropriately |
| COMPUTE_PGM_RSRC1[4:0] | VGPR count (÷8, 1024 max) | VGPR count (÷4, 512 max) | Rescale granularity |
| COMPUTE_PGM_RSRC2 | USER_SGPR 6-bit | USER_SGPR 5-bit | Truncate (max 31) |
| COMPUTE_PGM_RSRC3 | INST_PREF_SIZE 8-bit | Shared VGPR count | Reinterpret |
| kernel_code_properties[10] | ENABLE_WAVEFRONT_SIZE32 | Must be 0 (wave64) | Clear bit |
| group_segment_size | Up to 327,680 | Max 163,840 | Abort if > 163,840 |

Additional adjustments:
- Increase VGPR count for temp registers used by emulation trampolines
- Increase SGPR count for wave64 EXEC save/restore pairs
- Set up scratch buffer descriptor if kernel uses scratch

### Phase 6: ELF and Loader Integration

Reuse the existing HotSwap ROCR loader hook. The transpiler runs in the same position
as `RetargetCodeObject`:

```
LoadCodeObject()
  → Parse ELF
  → Detect gfx1250 MACH in e_flags
  → TranspileCodeObject(gfx1250 → gfx950):
      1. Disassemble all .text using LLVM MC (GFX1250 decoder)
      2. Walk instruction stream:
         - Map opcodes via Phase 1 tables
         - Re-encode format via Phase 3 functions
         - Queue emulation trampolines via Phase 4
         - Track EXEC-width operations for Phase 2
      3. Assemble trampolines using LLVM MC (GFX9 assembler)
      4. Patch .text with translated instructions + trampoline branches
      5. Rewrite kernel descriptors (Phase 5)
      6. Patch ELF e_flags, .note ISA, MSGPACK metadata
  → LoadSegments() to GPU memory
```

### Phase 7: Runtime Dispatch Adaptation

The HIP runtime needs to be aware that gfx1250 code objects are being transpiled
for gfx950. Reuse the existing `hip_fatbin.cpp` cross-gen intercept:

- When target GPU is gfx950 and code object has gfx1250 ISA: accept it
- Patch COMGR lookup to include gfx1250 as an extra ISA
- The existing `rocr_hotswap_retarget` bridge handles the ROCR call

## Complexity Estimate

| Component | Estimated LOC | Difficulty | Risk |
|---|---|---|---|
| Opcode mapping tables (auto-gen) | 500 | Low | Low — mechanical extraction from LLVM |
| Wave size adaptation | 300 | High | Medium — EXEC widening is formulaic but edge cases exist |
| VOP3 re-encoding | 30 | Low | Low — 1-bit change + opcode remap |
| SMEM re-encoding | 150 | Medium | Medium — offset truncation may fail |
| FLAT/GLOBAL re-encoding | 200 | Medium | Medium — size change requires careful padding |
| DS re-encoding | 50 | Low | Low — minor field shift |
| Wait counter translation | 100 | Low | Low — conservative merge to waitcnt(0) |
| SALU float emulation | 100 | Low | Low — straightforward VALU promotion |
| VOPD splitting | 80 | Low | Low — decode + emit two instructions |
| WMMA→MFMA translation | 800 | Very High | High — lane remapping is complex |
| Kernel descriptor rewrite | 150 | Medium | Low — well-documented format |
| Flat scratch adaptation | 200 | Medium | Medium — requires descriptor setup |
| **Total** | **~2,660** | | |

## Phased Delivery

### MVP (Phase A): Basic VALU Kernels
- Opcode tables + VOP1/VOP2/VOP3 re-encoding + wave adaptation
- Wait counter translation
- Kernel descriptor rewrite
- **Target:** Simple element-wise kernels (add, mul, relu) — no matrix, no scratch

### Phase B: Memory Operations
- SMEM, FLAT/GLOBAL, DS re-encoding
- Flat scratch → buffer scratch
- **Target:** Kernels with global memory access (softmax, layernorm)

### Phase C: Matrix Operations
- WMMA→MFMA substitution with lane remapping
- SWMMAC emulation
- **Target:** GEMM kernels, attention kernels

### Phase D: Full Coverage
- SALU float emulation
- VOPD splitting
- MIMG re-encoding
- DPP8→DPP translation
- Edge cases (indirect branches, exception handling)
- **Target:** Arbitrary gfx1250 code objects

## Performance Expectations

| Kernel Type | Expected Overhead | Bottleneck |
|---|---|---|
| Element-wise VALU | ~50% throughput (32/64 lanes active) | Wave utilization |
| Memory-bound | ~50% throughput + bandwidth match | Wave utilization |
| Matrix-bound (WMMA→MFMA) | 5-20x slowdown | Lane shuffle overhead |
| Mixed compute | 2-5x slowdown | Depends on matrix fraction |

The 50% ALU penalty from half-wave execution is the baseline cost. Matrix-heavy kernels
pay an additional penalty for WMMA→MFMA lane remapping. Memory-bound kernels may
partially compensate via gfx950's higher memory bandwidth.

## Key Files

| File | Purpose |
|---|---|
| `hotswap/transpiler.cpp` | Core gfx1250→gfx950 instruction translation engine |
| `hotswap/opcode_tables.h` | Auto-generated opcode mapping tables |
| `hotswap/wave_adapt.cpp` | EXEC widening and wave32→wave64 adaptation |
| `hotswap/wmma_to_mfma.cpp` | WMMA→MFMA substitution + lane remap sequences |
| `hotswap/waitcnt_merge.cpp` | s_wait_*cnt → s_waitcnt merging |
| `hotswap/hotswap.cpp` | Extended with transpile path (vs existing retarget path) |
| `tools/gen_opcode_tables.py` | Script to extract opcode mappings from LLVM TableGen |

## Alternatives Considered

- **LLVM IR JIT (PTX-like):** Store LLVM IR in the binary and JIT to target ISA at load
  time. Correct but requires IR availability (not present in compiled .co files) and a
  full LLVM backend at runtime (~100MB+ memory overhead). The transpiler approach works
  with existing compiled binaries.

- **Source-level cross-compilation:** Compile for both targets via fat binary. The right
  long-term solution but doesn't help with pre-compiled libraries and third-party .co files.

- **Skip wave adaptation (rely on runtime dispatch):** Launch 2x waves, each processing
  32 elements. This would require rewriting the kernel's workgroup indexing (threadIdx,
  blockIdx calculations) which is more invasive than EXEC masking.

- **Full re-encode via LLVM MC text round-trip:** Disassemble gfx1250 → text → change
  target → reassemble for gfx950. Simpler but lossy — LLVM MC text form doesn't preserve
  all encoding details, and the assembler may choose different encodings. Direct binary
  translation is more faithful.

- **Run wave32 natively on gfx950:** GFX9 doesn't support wave32 at all — there's no
  hardware mode for it. Half-wave with EXEC masking is the only option.
