# Original HotSwap: Deep-Dive Design Document

This document provides a comprehensive technical analysis of the **original** ROCm
HotSwap system — the load-time ISA rewriter that lives inside the ROCR runtime.
This covers the core rewrite engine, the cross-generation retargeter, and the
cross-family transpiler. It does **not** cover any of the experimental prototype
approaches (WaveASM/MLIR pipeline, LLVM MIR, LLVM IR, or Aster).

> **Provenance:** The original hotswap was authored by **powderluv** starting
> with commit `e1575520fd` ("Add ROCm HotSwap: load-time ISA rewriter for AMD
> GPUs"). The MLIR pipeline files (`pipeline.cpp`, `lifter.cpp`,
> `cross_target.cpp`, `wave_width.cpp`, `emit_assembly.cpp`,
> `ssa_construction.cpp`, `code_object_builder.cpp`) were added later in commit
> `1a254c584a` by mluecke as an experiment and are explicitly excluded from
> this document.

> **How to navigate:** All file paths below are clickable in VS Code. Hold Ctrl
> (or Cmd on macOS) and click to open the referenced file.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Entry Points and Activation](#2-entry-points-and-activation)
3. [Component Architecture](#3-component-architecture)
   - [3.1 Rewrite Engine (hotswap.cpp)](#31-rewrite-engine)
   - [3.2 Rule Parser (hotswap_rules.cpp)](#32-rule-parser)
   - [3.3 Trampoline Builder (trampoline.cpp)](#33-trampoline-builder)
   - [3.4 Cross-Family Transpiler (transpiler.cpp)](#34-cross-family-transpiler)
4. [Design Decisions and Trade-offs](#4-design-decisions-and-trade-offs)
5. [Scalability Analysis](#5-scalability-analysis)
6. [Known Limitations](#6-known-limitations)
7. [File Index](#7-file-index)

---

## 1. System Overview

HotSwap intercepts GPU code object loading in the ROCR runtime to rewrite
machine instructions at load time. The system is **completely transparent** —
applications require no source changes, recompilation, or relinking.

There are two independent rewriting strategies, each targeting a different
cross-generation scenario:

| Strategy | Scope | ISA Pair | Mechanism |
|---|---|---|---|
| **Same-family retarget** | gfx950 → gfx942 | CDNA → CDNA | Encoding-compatible + NOP pre-pass + trampoline emulation |
| **Cross-family transpiler** | gfx1250 → gfx950 | RDNA → CDNA | Text round-trip with semantic rules |

The rule-based rewrite engine sits underneath both and provides surgical
per-instruction patching via JSON configuration files.

**Activation flow:**

```
HIP Application
    │
    ▼
__hipRegisterFatBinary ──── if no native code object, extract closest ISA
    │
    ▼
hsa_executable_load_agent_code_object
    │
    ├── RetargetCodeObject()     ← same-family or encoding-compatible retarget
    ├── TranspileCodeObject()    ← cross-family text round-trip
    ├── PatchElfIsa()            ← fix ELF metadata to match target
    └── RewriteCodeObject()      ← apply JSON rewrite rules + trampolines
    │
    ▼
GPU executes retargeted code
```

---

## 2. Entry Points and Activation

### Environment Variables

| Variable | Purpose |
|---|---|
| `HSA_HOTSWAP_RULES` | Path to a JSON rewrite rules file |
| `HSA_HOTSWAP_ISA_OVERRIDE` | Set to `1` to skip ISA compatibility checks |
| `HSA_HOTSWAP_DUMP` | Set to `1` for before/after instruction dump |

### Public API

The public C++ API is declared in
[hotswap/hotswap.hpp](../hotswap.hpp):

- **`IsEnabled()`** — Returns `true` if either `HSA_HOTSWAP_RULES` or
  `HSA_HOTSWAP_ISA_OVERRIDE` is set.
- **`IsIsaOverrideEnabled()`** — Returns `true` if `HSA_HOTSWAP_ISA_OVERRIDE`
  is set and non-zero.
- **`PatchElfIsa()`** — In-place patching of ELF `e_flags` and `.note`
  sections to match a target ISA.
- **`RetargetCodeObject()`** — Decode-and-reencode for same-encoding-family
  ISA pairs (e.g., gfx950 → gfx942).
- **`RewriteCodeObject()` / `RewriteCodeObjectGrow()`** — Apply JSON-defined
  rewrite rules, potentially growing the ELF to accommodate trampolines.

There is also a C-linkage bridge exported from `libhsa-runtime64.so`:

```c
extern "C" int rocr_hotswap_retarget(void* elf_data, size_t elf_size,
                                      const char* source_isa, const char* target_isa);
```

This allows the HIP/CLR layer to call the retarget function via `dlsym(RTLD_DEFAULT, ...)`
without a compile-time dependency on the hotswap headers.

The transpiler API is declared in
[hotswap/transpiler.hpp](../transpiler.hpp):

- **`NeedsTranspile()`** — Checks if source and target are from different ISA
  families (RDNA vs CDNA).
- **`TranspileCodeObject()`** — Full disassemble → translate → reassemble.
- **`TranslateInstruction()`** — Single-instruction semantic translation
  (used internally and for unit testing).

---

## 3. Component Architecture

### 3.1 Rewrite Engine

**File:** [hotswap/hotswap.cpp](../hotswap.cpp)

This is the largest single file in the system (~1750 lines). It contains:

#### LLVM MC Infrastructure

A lazy-initialized, cached-per-CPU set of LLVM Machine Code objects:

```
LLVMState {
    MCDisassembler    — decodes raw bytes → MCInst
    MCInstPrinter     — MCInst → assembly text
    MCCodeEmitter     — assembly → encoded bytes
    MCSubtargetInfo   — ISA feature flags
    MCInstrInfo       — instruction metadata (num defs, operand types)
}
```

Initialization is protected by `std::call_once` for thread safety. The AMDGPU
targets (`LLVMInitializeAMDGPUTargetInfo()`, etc.) are registered once per
process. Per-CPU `LLVMState` instances are cached in a `std::map<string, LLVMState>`
protected by a mutex.

#### ELF Parsing

The engine includes a self-contained minimal ELF64 parser (`ParseElfInfo`) that
extracts:
- Section table (name, type, offset, size, virtual address)
- Symbol table (name, value, size, info, section index)
- `.text` section location

This avoids dependencies on `libelf` or LLVM's `ObjectFile` layer, keeping the
runtime footprint minimal.

#### Instruction Decode

`DecodeTextSection()` walks the `.text` bytes sequentially, calling
`MCDisassembler::getInstruction()` for each instruction. Failed decodes advance
by 4 bytes (minimum AMDGPU instruction size). Each decoded instruction stores:
- Byte offset within `.text`
- Encoded size (4 or 8 bytes)
- The `MCInst` (LLVM's machine instruction representation)
- Mnemonic string (extracted from `MCInstPrinter` output)

#### Rule Matching and Application

For each decoded instruction, rules are tested in order:

1. **`MatchRule()`** checks mnemonic, operand patterns, kernel name, and byte
   offset against the rule's match criteria.
2. Three replacement strategies:
   - **`ApplyMnemonicSwap()`** — Prints the instruction to text, replaces the
     mnemonic token, reassembles via a full MC pipeline. Verifies the output
     is the same size as the original.
   - **`ApplyByteReplace()`** — Raw `memcpy` of replacement bytes. Pads any
     remaining space with `s_nop` (0xBF800000).
   - **`AsmReplace`** — Builds a trampoline (see [3.3](#33-trampoline-builder))
     for multi-instruction or size-changing replacements.

#### Same-Family Retarget (`RetargetCodeObject`)

The retarget path is the workhorse for gfx950 → gfx942 translation. A crucial
discovery drives its design:

> **gfx942 and gfx950 share identical binary encodings** for all standard
> VALU, SMEM, VMEM, and SOPP instructions. Only ~1.6% of instructions
> (gfx950-specific opcodes) need special handling.

This means the retarget can skip the expensive decode → print → assemble cycle
for 98.4% of instructions. The implementation:

1. **NOP pre-pass:** Scan for gfx950-only opcodes by checking the first dword:
   - `0xD23D`–`0xD243`: FP4/FP6/FP8 scale conversions
   - `0xD3AD`, `0xD3AE`: Mixed-format MFMA
   - `0xD267`: `v_cvt_pk_f16_f32`
   - `0xD268`: `v_cvt_pk_bf16_f32`
   - `0xD233`: `v_bitop3_b16`

2. **Instruction-specific emulation:** Each gfx950-only opcode gets a tailored
   replacement strategy:

   | Instruction | Strategy | Quality |
   |---|---|---|
   | `v_cvt_pk_f16_f32` | Opcode swap → `v_cvt_pkrtz_f16_f32` | Exact (different rounding) |
   | `v_cvt_pk_bf16_f32` | Trampoline: `v_lshrrev` × 2 + `v_lshl_or` (20 bytes) | Exact (truncation) |
   | `v_bitop3_b16` (0xEC) | `v_or_b32` | Good approximation |
   | `v_cvt_scalef32_pk_fp4_f32` | Trampoline: scale + truncate + clamp + pack (44 bytes) | Approximate E2M1 |
   | MFMA variants | NOP out (2 × `s_nop`) | Lossy placeholder |

3. **NOP sled allocation:** Trampolines are placed in post-`s_endpgm` padding
   regions (the 256-byte alignment NOPs between kernel bodies). The engine
   scans for `s_endpgm` → NOP sequences and maintains a write cursor for each
   sled. Trampolines include an `s_branch` back to the instruction after the
   original.

4. **Fallback batch assembly:** For ISA pairs that DON'T share encodings, the
   engine falls through to a full batch path: decode all → print all to one
   assembly string → assemble in one MC pass → patch `.text` per-instruction.

#### ELF ISA Patching (`PatchElfIsa`)

After instruction retargeting, ELF metadata must match the target ISA:

- **`e_flags`:** Patches bits [7:0] (`EF_AMDGPU_MACH`) using a lookup table
  mapping gfx names → machine IDs.
- **`.note` sections:** Finds `NT_AMDGPU_ISA` notes (type 27, owner "AMDGPU")
  and performs in-place string replacement. Shorter target names are null-padded
  to preserve note alignment.

#### Kernel Descriptor Patching

When emulation trampolines use extra registers, `UpdateKernelDescriptor()`
patches `COMPUTE_PGM_RSRC1` in the kernel descriptor:
- Bits [5:0]: `GRANULATED_WORKITEM_VGPR_COUNT` (granularity of 4 VGPRs)
- Bits [9:6]: `GRANULATED_WAVEFRONT_SGPR_COUNT` (granularity of 8 SGPRs)

---

### 3.2 Rule Parser

**Files:** [hotswap/hotswap_rules.hpp](../hotswap_rules.hpp), [hotswap/hotswap_rules.cpp](../hotswap_rules.cpp)

A fully self-contained JSON parser with **zero external dependencies**. This
was a deliberate choice to avoid pulling `nlohmann/json` or similar into the
ROCR runtime.

The parser is a hand-written recursive descent parser supporting the JSON subset
needed for rules files: objects, arrays, strings, integers, booleans, and null.

**Rule structure:**

```json
{
  "version": 1,
  "target": "amdgcn-amd-amdhsa--gfx1201",
  "rules": [
    {
      "name": "swap_dot2",
      "match": {
        "mnemonic": "v_dot2acc_f32_f16",
        "operands": [{"reg_class": "VGPR_32"}, {}],
        "kernel": "myKernel",
        "offset": "0x100"
      },
      "replace": { "mnemonic": "v_dot2c_f32_f16" },
      "extra_vgprs": 2
    }
  ]
}
```

**Match criteria (all must match):**
- `mnemonic`: Exact mnemonic string match
- `operands`: Array of operand matchers (Wildcard, Immediate value, RegClass)
- `kernel`: ELF symbol name match (finds which kernel an offset belongs to)
- `offset`: Exact byte offset within `.text`

**Replace actions:**
- `MnemonicSwap`: Replace mnemonic, preserve operands (same-size encoding)
- `AsmReplace`: Multi-line assembly replacement (triggers trampoline if larger)
- `ByteReplace`: Raw hex byte replacement

Rules are loaded once per process via `std::call_once` from the path in
`HSA_HOTSWAP_RULES` and cached in a singleton.

---

### 3.3 Trampoline Builder

**Files:** [hotswap/trampoline.hpp](../trampoline.hpp), [hotswap/trampoline.cpp](../trampoline.cpp)

For replacements larger than the original instruction, the trampoline mechanism
redirects execution to out-of-line code:

```
Original .text:                     Trampoline region:
┌─────────────────────┐            ┌──────────────────────────┐
│ ...                 │            │ <replacement sequence>   │
│ s_branch trampoline ├───────────►│ ...                      │
│ <NOP padding>       │      ┌─────┤ s_branch back            │
│ <resume point>      │◄─────┘     └──────────────────────────┘
│ ...                 │
└─────────────────────┘
```

**Key implementation details:**

- **`EncodeSBranch()`**: Encodes `s_branch` (SOPP opcode `0xBF82`) with a
  signed 16-bit dword offset relative to PC+4. Range: ±128KB.
- **`EncodeSNop()`**: Encodes `s_nop 0` (`0xBF800000`) for padding.
- **`BuildTrampoline()`**: Takes assembly lines, assembles them through a
  full LLVM MC pipeline (parser → streamer → code emitter → object writer),
  extracts `.text` bytes from the resulting mini-ELF, and appends an `s_branch`
  back to the resume point.

The trampoline bytes are appended to `.text` by `RewriteCodeObjectGrow()`,
which reallocates the ELF buffer and patches:
- `.text` section header (`sh_size`)
- Program headers containing `.text` (`p_filesz`, `p_memsz`)
- Section/segment offsets for everything after `.text`

#### Trampoline Placement Strategy

The retarget engine uses two trampoline allocation strategies:

1. **NOP sled reuse (preferred):** After `s_endpgm`, kernels are padded with
   `s_nop` instructions for 256-byte alignment. The engine scans for these
   NOP sleds and uses them as scratch space for trampolines. Each sled
   maintains a `write_pos` cursor. This avoids growing the ELF at all.

2. **ELF growth (fallback):** When NOP sleds are full or don't exist (JSON
   rule-based rewrites), `RewriteCodeObjectGrow()` allocates a new buffer,
   copies the original ELF, appends trampoline bytes after `.text`, and
   patches all section headers, program headers, and segment offsets.

---

### 3.4 Cross-Family Transpiler

**Files:** [hotswap/transpiler.hpp](../transpiler.hpp), [hotswap/transpiler.cpp](../transpiler.cpp)

The transpiler handles the hardest case: **cross-family ISA translation**
(RDNA → CDNA), specifically gfx1250 → gfx950. This is a ~5,800-line file
that implements full semantic translation.

#### Architecture

```
gfx1250 .text bytes
    │
    ▼
LLVM MC Disassembler (gfx1250)
    │
    ▼
MCInst → MCInstPrinter → assembly text
    │
    ▼
Per-instruction semantic translation
    │  ├── Mnemonic renaming (500+ mapping table entries)
    │  ├── Wait counter merging (split → unified s_waitcnt)
    │  ├── EXEC widening (wave32 → wave64)
    │  ├── SALU float emulation (s_mul_f32 → VALU sequence)
    │  ├── scale_offset addressing lowering
    │  ├── Operand format adaptation
    │  └── Unsupported instruction NOP-out
    │
    ▼
Batch assemble translated text (gfx950)
    │
    ▼
Replace .text + patch kernel descriptors + ELF metadata
```

#### Translation Rule Categories

1. **Mnemonic Renaming** (~500 entries across 15 tables):
   GFX12 uses bit-width naming (`global_load_b32`) while GFX9 uses type naming
   (`global_load_dword`). Tables cover: global memory, flat memory, DS (LDS),
   SMEM, scalar ALU, vector ALU, global atomics, buffer operations, flat
   atomics, image ops, comparison ops, and SDWA.

2. **Wait Counter Merging**: GFX12 has split wait counters
   (`s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_kmcnt`, `s_wait_dscnt`,
   `s_wait_expcnt`). GFX9 has a unified `s_waitcnt vmcnt(N) lgkmcnt(M) expcnt(K)`.
   The transpiler tracks pending counts per category and emits merged waitcnts.

3. **EXEC Widening**: GFX1250 (RDNA4) uses wave32 (32-lane execution mask).
   GFX950 (CDNA) uses wave64. The transpiler:
   - Expands `s_*_saveexec_b32` → manual exec save + ALU + `exec_hi = 0`
   - Converts `v_cmpx_*` → `v_cmp_*` + manual `exec AND` + VCC save/restore
   - Inserts `s_mov_b32 exec_hi, 0` after any `exec_lo` write
   - Inserts `s_mov_b32 vcc_hi, 0` after `v_cmp` instructions

4. **SALU Float Emulation**: GFX1250 has scalar ALU float instructions
   (`s_mul_f32`, `s_fmac_f32`, etc.) that don't exist on GFX9. The transpiler
   expands these to VALU sequences using temporary VGPRs, with proper exec
   masking and VCC save/restore.

5. **Scale-Offset Addressing**: GFX1250 global memory instructions use a
   `v_index(32) + s[base:base+1](64)` addressing mode. GFX9 requires a
   64-bit VGPR address. The transpiler synthesizes:
   ```asm
   v_lshlrev_b32 vTmp, <shift>, vIndex     ; index × element_size
   v_mov_b32     vAddrHi, sBaseHi
   v_add_co_u32  vAddrLo, vcc, sBaseLo, vTmp
   v_addc_co_u32 vAddrHi, vcc, 0, vAddrHi, vcc
   global_load_dword vDst, v[AddrLo:AddrHi], off
   ```

6. **Instruction Elimination**: GFX12-only scheduling hints are removed:
   `s_clause`, `s_delay_alu`, `s_code_end`.

#### Kernel Descriptor Update

The transpiler patches multiple kernel descriptor fields:
- **VGPR count**: Increased to accommodate temporary VGPRs used by emulation
  sequences (SALU float, address computation).
- **SGPR count**: May increase for saved VCC/exec state.
- **Kernel code properties**: Clears `enable_wavefront_size32` bit, sets
  `enable_sgpr_dispatch_ptr` if needed.

#### Development History

The transpiler was built incrementally over ~90 commits, tackling increasingly
complex real-world kernels from the AITER workload. Key milestones:
- Phase A: Basic mnemonic renaming (global/flat/SMEM/SALU)
- Phase B: Flat+saddr → global conversion, SMEM mapping
- Phase C: SALU float → VALU emulation
- Phase D: VOPD splitting, barrier translation
- Per-kernel debugging: attn_forward, split-K, multihead attention each
  required multi-commit investigation and targeted fixes
- Final state: 42/42 tests passing, 18/20 complex transpiler kernels passing

---

## 4. Design Decisions and Trade-offs

### Why LLVM MC Direct (Not COMGR)?

COMGR provides a text-based compilation API but round-trips through full
compilation. LLVM MC gives instruction-level control with much lower overhead.
COMGR's API is designed for whole-program compilation, not single-instruction
patching.

### Why a Custom JSON Parser?

The ROCR runtime is a critical system library loaded by every HIP/ROCm
application. Adding `nlohmann/json` or similar would increase binary size
and introduce a dependency that could conflict with applications. The
hand-written parser handles exactly the subset needed (~400 lines).

### Why Trampolines (Not Full .text Rewrite)?

Trampolines are **surgical** — only matched instructions are modified. A full
`.text` rewrite would reassemble everything, risking encoding errors in
unrelated code. The retarget engine uses full rewrite only as a fallback when
ISA pairs don't share encodings.

### Why Text-Level Translation (Not IR-Based)?

The transpiler (`transpiler.cpp`) operates at the assembly text level — it's a
string-manipulation engine with regex-based pattern matching. This was chosen
because:
- **Speed of iteration**: Text manipulation is simpler to debug and extend.
  Each new instruction can be added as a mapping table entry.
- **Battle-tested**: The transpiler was developed against real AITER and CK
  workloads through 90+ commits of incremental debugging.
- **No dependency on MLIR**: The transpiler only needs LLVM MC for
  disassembly and reassembly, avoiding the much larger MLIR dependency.

The trade-off is that text-level translation can't easily perform analyses
that require structured IR (register liveness, instruction scheduling, etc.).

### Why Source Integration (Not LD_PRELOAD)?

An `LD_PRELOAD` shim would avoid modifying ROCR source but would require
duplicating internal structures to parse code objects. The fat binary
intercept in HIP/CLR also cannot be done via LD_PRELOAD without fragile
symbol interposition. Source integration enables direct access to ROCR
internals (agent ISA info, loader state).

### Why Encoding-Compatibility First?

The key insight driving the gfx950 → gfx942 retarget is that identical
encodings eliminate the need for reassembly. This was discovered empirically:
135,614 instructions in the AITER workload shared identical encoding across
the two ISAs, with only 2,172 (1.6%) needing emulation. This means:
- No LLVM MC assembler bugs can affect the 98.4% majority
- No encoding-size mismatches to handle
- Sub-millisecond retarget time for typical code objects

---

## 5. Scalability Analysis

### Instruction Throughput

The same-family retarget path (gfx950 → gfx942) processes 135,614 instructions
across 1,315 kernels in the AITER workload. The encoding-compatibility
optimization means most instructions require zero processing — only the ~1.6%
gfx950-specific instructions need the NOP pre-pass.

For the cross-family transpiler, the overhead is proportional to `.text` size
since every instruction must be decoded, translated, and reassembled.

### LLVM MC State Management

A significant engineering challenge: **LLVM's AMDGPU backend has global state
that doesn't survive multiple MCContext lifecycles.** The code carefully manages
this by:
- Caching `LLVMState` per CPU with mutex protection
- Limiting retarget assembly to one code object per process (`s_retarget_count`)
- Using persistent static `TargetAssembler` instances instead of creating/destroying

### Memory Allocation

- **In-place patching**: `RewriteCodeObject()` modifies the ELF buffer directly
  for same-size replacements.
- **NOP sled reuse**: Trampolines placed in existing alignment padding require
  no allocation at all.
- **Growing**: `RewriteCodeObjectGrow()` allocates a new buffer
  (`malloc`) with space for trampolines, copies everything, and patches
  section/program headers to account for the inserted bytes.
- **Transpiler**: May reallocate the ELF buffer since translated `.text` can
  be larger (wave width expansion adds instructions).

### Thread Safety

- LLVM target initialization: `std::call_once`
- Per-CPU MC state cache: `std::mutex`
- Rules cache: `std::call_once`
- Assembler cache: `std::mutex`

The hotswap hooks are called from the ROCR loader which serializes code object
loading, so concurrent access to the same ELF buffer is not a concern.

### Benchmark Results (Same-Family Retarget)

Performance geomean across 20 AITER kernels: **1.000x** — zero overhead.

Accuracy: 15/20 kernels produce **bit-identical** output to native execution.
5/20 kernels have degraded accuracy due to FP4/bf16 emulation approximations.

All 1,315 precompiled `.co` kernels from AITER load successfully (100% load
rate). 17/17 execution validation tests pass.

---

## 6. Known Limitations

1. **LLVM MC global state**: The AMDGPU backend's internal tables only survive
   one MCContext lifecycle. Retarget is limited to one code object per process.
   The NOP pre-pass + encoding-skip approach works around this for gfx950 → gfx942.

2. **Trampoline distance**: `s_branch` uses signed 16-bit dword offset (±128KB).
   For very large kernels, a `s_setpc_b64` with literal address load (12 bytes)
   would be needed. Not implemented.

3. **Fixed temporary registers**: The transpiler uses hardcoded temporary
   registers (v255 for SALU float emulation, dynamically allocated VGPRs for
   address computation). This works because the kernel descriptor is patched
   to increase register allocation, but is not safe in general without
   live-range analysis.

4. **MFMA expansion**: `v_mfma_f32_16x16x128_f8f6f4` (128 elements/instruction)
   would need a 4:1 expansion to gfx942's 32-element MFMA. Currently NOPed out.

5. **FP4 emulation accuracy**: The `v_cvt_scalef32_pk_fp4_f32` emulation uses
   an approximate scale + truncate + clamp approach that doesn't precisely
   match hardware E2M1 rounding.

6. **No register pressure analysis**: Temporary registers for emulation are
   allocated statically (or dynamically based on kernel descriptor metadata).
   A proper implementation would need liveness analysis to find free registers.

7. **Cross-family transpiler complexity**: At ~5,800 lines, the transpiler is
   the most complex and hardest-to-maintain component. Each new ISA feature
   or instruction requires manual translation rule additions.

---

## 7. File Index

### Core Rewrite Engine

| File | Purpose | Lines |
|---|---|---|
| [hotswap/hotswap.hpp](../hotswap.hpp) | Public API: IsEnabled, PatchElfIsa, RetargetCodeObject, RewriteCodeObject | ~90 |
| [hotswap/hotswap.cpp](../hotswap.cpp) | Rewrite engine, ELF parser, retarget, ISA patching, kernel descriptor update | ~1750 |
| [hotswap/hotswap_rules.hpp](../hotswap_rules.hpp) | Rule data structures: OperandMatch, RewriteRule, RulesFile | ~80 |
| [hotswap/hotswap_rules.cpp](../hotswap_rules.cpp) | Self-contained JSON parser + rule loading | ~400 |
| [hotswap/trampoline.hpp](../trampoline.hpp) | Trampoline API: BuildTrampoline, EncodeSBranch, EncodeSNop | ~50 |
| [hotswap/trampoline.cpp](../trampoline.cpp) | Trampoline builder: assembly → ELF → .text extraction + branch encoding | ~230 |

### Cross-Family Transpiler

| File | Purpose | Lines |
|---|---|---|
| [hotswap/transpiler.hpp](../transpiler.hpp) | TranspileCodeObject, TranslateInstruction API | ~60 |
| [hotswap/transpiler.cpp](../transpiler.cpp) | Full semantic translator: 500+ mnemonic mappings, wait counter merge, EXEC widen, SALU float emulation, scale_offset lowering | ~5800 |

### Tests

| File | Purpose |
|---|---|
| [hotswap/tests/hotswap_test.cpp](../tests/hotswap_test.cpp) | Unit tests for rewrite engine |
| [hotswap/tests/test_transpiler.py](../tests/test_transpiler.py) | Python test suite for per-instruction transpiler translation |
| [hotswap/tests/test_transpiler_e2e.py](../tests/test_transpiler_e2e.py) | End-to-end transpiler tests against 50+ HIP kernel files |
| [hotswap/tests/test_rules.json](../tests/test_rules.json) | Example JSON rewrite rules |
| [hotswap/tests/*.hip](../tests/) | 50+ HIP kernel sources used as transpiler test payloads |

### Documentation

| File | Purpose |
|---|---|
| [hotswap/docs/hotswap-architecture.md](hotswap-architecture.md) | Architecture overview, status, benchmark results |
| [hotswap/docs/hotswap-rewrite-rules.md](hotswap-rewrite-rules.md) | Rewrite rules design for gfx950 → gfx942 |
| [hotswap/docs/hotswap-wheel-integration.md](hotswap-wheel-integration.md) | Integration with Python wheel packaging |
| [hotswap/docs/gfx1250-on-gfx950-analysis.md](gfx1250-on-gfx950-analysis.md) | Cross-family analysis: gfx1250 on gfx950 feasibility |

### Build

| File | Purpose |
|---|---|
| [hotswap/CMakeLists.txt](../CMakeLists.txt) | Standalone build: rocr-hotswap library + test executables |
| [hotswap/Dockerfile](../Dockerfile) | ROCm dev image for building and testing |
| [hotswap/run_tests.sh](../run_tests.sh) | Test runner (--no-gpu for Tier 1 only) |
