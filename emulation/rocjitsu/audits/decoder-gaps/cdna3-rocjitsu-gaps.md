# CDNA3 Rocjitsu Gaps

Architecture: CDNA3 / MI300

## Gaps

### CDNA3-RJ-001: FP8/BF8 conversions do not enforce the `SH_MEM_CONFIG[8]` prerequisite

Manual evidence:

- The cited manual passage defines CDNA3 FNUZ-style FP8/BF8
  numeric encodings.
- The cited manual passage defines `FP16_OVFL` conversion
  behavior and the `SH_MEM_CONFIG[8]` prerequisite.

Rocjitsu evidence:

- Generated CDNA3 packed and stochastic conversion paths now call FNUZ
  mode-aware helpers with `wf.fp16_ovfl()` in the implementation; the former fixed-overflow-mode subclaim is patched.
- Targeted searches of rocjitsu, amdisa codegen, and tests still find no
  `SH_MEM_CONFIG` state or prerequisite check for these conversions.

Impact:

CDNA3 now models the runtime `FP16_OVFL` alternatives, but it can execute
FP8/BF8 operations without enforcing the manual's `SH_MEM_CONFIG[8]`
configuration requirement.
### CDNA3-RJ-002: Generated source operands allow forms the manual says are illegal

Manual evidence:

- The cited manual passage says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

Rocjitsu evidence:

- Generated CDNA3 constructors use `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE` for
  `V_CVT_PK_FP8_F32` in the implementation.
- The stochastic forms use the same broad operand classes in the implementation.
- These generated classes come from XML classes that include scalar-register
  subtypes.

Impact:

Rocjitsu inherits the XML/manual legality mismatch. Hardware and LLVM oracle
behavior still need a focused check before deciding whether to enforce the
manual-only VGPR rule or update the audit classification.

### CDNA3-RJ-003: FP8/BF8 tests do not cover the `SH_MEM_CONFIG[8]` prerequisite

Rocjitsu evidence:

- `Cdna3CvtFp8Test` cases in
  the relevant tests now
  exercise packed, BF8, stochastic, and `S_SETREG_IMM32_B32`-driven
  `FP16_OVFL` behavior.
- Targeted searches still find no modeled `SH_MEM_CONFIG` state and therefore
  no test that rejects or changes FP8/BF8 execution when bit 8 is not enabled.

Impact:

The rebased tests cover the alternate overflow mode, but they cannot catch the
remaining CDNA3 configuration-prerequisite gap.
### CDNA3-RJ-004: FP8/BF8 forwarding hazard is not modeled

Manual evidence:

- The cited manual passage requires spacing between `CVT_*_F32` conversions that
  write low/high halves or bytes of the same destination register.

Rocjitsu evidence:

- The generated execution bodies are functional per-instruction operations and
  do not track a 4-cycle forwarding hazard for these conversions.

Impact:

Rocjitsu can execute sequences that the manual requires software to space on
hardware. This is a timing/scheduling gap rather than a single-instruction
decode bug.

### CDNA3-RJ-005: Packed F32 arithmetic broadcasts scalar sources instead of reading 64-bit pairs

Manual/XML evidence:

- CDNA3 section 6.7 says packed 32-bit instructions operate on two dwords at a
  time and that `OPSEL`/`OPSEL_HI` select the first or second dword for each
  source in the cited manual passage.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` in the detailed pseudocode in the cited manual passage.
- The CDNA3 XML encodes these sources as 64-bit operands in the machine-readable ISA XML.

Rocjitsu evidence:

- The generated CDNA3 constructors preserve the 64-bit operand sizes for
  `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` in the implementation.
- The shared generator reads a 64-bit pair only when the encoding value is a
  VGPR, otherwise it initializes the high word from the low 32-bit read in the code generator.
- The generated helpers follow that pattern for `V_PK_ADD_F32`,
  `V_PK_FMA_F32`, and `V_PK_MUL_F32` in the implementation.
- The operand layer already has scalar-pair support through `read_lane64()` and
  `resolve_src_scalar64()` in the implementation, and `V_PK_MOV_B32` uses that path
  unconditionally.

Impact:

A scalar-pair source such as `s6:s7` with different low/high dwords feeds the
low dword to both packed F32 components in rocjitsu. The manual and XML both
describe these operands as 64-bit pairs, so the high component should come from
the second dword of the pair before `OPSEL_HI` selection.

### CDNA3-RJ-006: Packed 32-bit VOP3P VGPR pair alignment is not validated

Manual evidence:

- CDNA3 section 6.7 says packed 32-bit operands must be two-dword aligned, with
  an even VGPR address, in the cited manual passage.
- The `V_PK_MOV_B32` notes repeat that sources are 64-bit operands subject to
  alignment restrictions for both SGPR and VGPR in the cited manual passage.

Rocjitsu evidence:

- The packed F32 and `V_PK_MOV_B32` constructors use raw 64-bit VGPR/source
  operands in the implementation.
- `Isa::resolved_vgpr_offset()` accepts any source VGPR encoding from 256
  through 511 and returns the unadjusted VGPR index in the implementation.
- `Operand::read_lane64()` and `write_lane64()` then read or write the pair
  starting at that index in the implementation.

Impact:

Odd VGPR encodings are treated as valid pairs rooted at the odd register,
whereas the manual restricts packed 32-bit operands to even-aligned pairs.

### CDNA3-RJ-007: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA3 VOP3P format includes `CLMP` as "1 = clamp result" in the cited manual passage.
- The CDNA3 XML describes generic VOP3P `CLAMP` as clamping output to
  `[0.0, 1.0]` in the machine-readable ISA XML.

Rocjitsu evidence:

- `V_PK_ADD_F16`, `V_PK_FMA_F16`, `V_PK_MAX_F16`, `V_PK_MIN_F16`, and
  `V_PK_MUL_F16` execute without applying `inst.inst_.clamp` in the implementation.
- MIX helpers do apply `std::clamp(result, 0.0f, 1.0f)` when
  `inst.inst_.clamp` is set in the implementation, so
  the generated code already has a working pattern for floating VOP3P clamp.

Impact:

Packed F16 results outside `[0, 1]` remain unclamped for the ordinary packed
F16 VOP3P arithmetic helpers even when the VOP3P `CLMP` bit is set.

### CDNA3-RJ-008: XF32 MFMA executes as ordinary F32

Manual evidence:

- The CDNA3 manual says XF32 MFMA takes 32-bit floats but rounds the mantissa
  to 10 bits for reduced-precision multiplication in the cited manual passage.

Rocjitsu evidence:

- The matrix execute generator maps `XF32` into the ordinary F32 specialized
  family and `amdgpu::extract_f32` extractor in the code generator.
- `VMfmaF3216x16x8Xf32Vop3pMfma::execute_impl()` and
  `VMfmaF3232x32x4Xf32Vop3pMfma::execute_impl()` both call
  `exec_f32_mfma_f32_spec` with ordinary F32 extractors in the implementation.
- The shared F32 MFMA path multiplies the extracted float values directly in the implementation.
- Searching rocjitsu for `xf32`, `mantissa`, and reduced-precision wording
  finds only the generated XF32 class names and unrelated comments, not an XF32
  rounding helper.

Impact:

CDNA3 XF32 MFMA results match ordinary F32 MFMA in rocjitsu, so inputs whose
low mantissa bits affect the product will diverge from the manual's reduced
precision rule.

### CDNA3-RJ-009: Dense MFMA register-block alignment is not validated

Manual evidence:

- MFMA input and output register blocks must be contiguous, and the first
  register must be aligned to the number of registers required by the operand,
  in the cited manual passage.
- The dense MFMA rule table also says `SRC0`, `SRC1`, `SRC2`, and `VDST` need
  even alignment when encoded as VGPR operands in the cited manual passage.

Rocjitsu evidence:

- Generated dense MFMA constructors expose raw register encodings and operand
  sizes without checking block-width alignment; representative XF32
  constructors are in the implementation, and I8/F64 constructors follow the same generated pattern.
- `dst_base()` and `src_base()` map the encoded register number to a physical
  VGPR/AccVGPR base but do not validate alignment in the implementation.
- The shared layout helpers then assume a contiguous block by adding computed
  register offsets from the base in the implementation.

Impact:

Illegal dense MFMA encodings with odd or under-aligned source, accumulator, or
destination bases execute as if valid. This is broader than the packed-VOP3P
two-register alignment gap because dense MFMA operands can require much larger
contiguous register blocks.

### CDNA3-RJ-010: Dense MFMA clamp and overflow state is not modeled

Manual evidence:

- The dense MFMA rule table says clamp is supported, uses `FP16_OVFL` from
  MODE, clamps F32 overflow to `+/-MAX` when set and otherwise produces
  `+/-INF`, and clamps I32 overflow/underflow to `+/-MAX` when set in the cited manual passage.

Rocjitsu evidence:

- Generated dense F32/XF32/F16/BF16 MFMA execution passes only source bases,
  destination base, constant-accumulator state, `CBSZ`, `ABID`, and `BLGP` into
  shared helpers; representative calls are in the implementation.
- Dense I8 MFMA generated paths call `exec_i32_i8()` without a clamp argument
  in the implementation.
- The generic dense I8 helper has no clamp parameter and accumulates through a
  32-bit path in the implementation; separate helpers do have clamp-capable integer primitives in the implementation, but the generated CDNA3 dense I8 MFMA
  paths do not use them.

Impact:

If the CDNA3 dense MFMA clamp/`FP16_OVFL` row is architecturally meaningful,
rocjitsu currently executes those cases as unclamped, mode-independent MFMA.

### CDNA3-RJ-011: MFMA broadcast field legality is not validated

Manual evidence:

- Section 7.1.5 says `CBSZ` values where `(1 << CBSZ)` exceeds the number of
  blocks are undefined, the largest legal `CBSZ` is 4, and `ABID` is illegal
  when `ABID >= (1 << CBSZ)` in the cited manual passage.

Rocjitsu evidence:

- Generated dense MFMA execution passes raw `inst_.cbsz`, `inst_.abid`, and
  `inst_.blgp` into shared helpers; representative XF32 calls are in the implementation.
- `permute_a_lane()` implements `S = 64 >> cbsz` and
  `(lane % S) + S * abid` without validating `CBSZ` or `ABID` in the implementation.

Impact:

Illegal or undefined MFMA broadcast encodings execute with a computed lane
permutation instead of being rejected or specially classified.

### CDNA3-RJ-012: Dense MFMA tests miss CDNA3-specific semantics

Rocjitsu evidence:

- The gfx942 CTS skip list excludes CDNA3 MFMA fpsan coverage, including F16,
  BF16, FP8/BF8, XF32, and F32 MFMA cases in the relevant tests.
- The expensive bit-exact MFMA SIMD suite is documented and instantiated as
  CDNA4-only in the relevant tests.
- CDNA3 vector instruction execute coverage is broadly excepted from the
  generated harness and delegated to kernel simulation in the coverage exceptions.

Impact:

The existing test surface can miss CDNA3-only dense MFMA behavior such as XF32
mantissa reduction, CDNA3 MODE-denorm distinctions, DGEMM exception behavior,
MAI dependency-wait rules, and CDNA3 field legality cases even when generic
CDNA4 MFMA SIMD checks pass.

### CDNA3-RJ-013: MFMA denorm, rounding, and DGEMM exception behavior are not modeled

Manual evidence:

- The dense MFMA rule table gives denorm, clamp, forced-RNE round mode,
  exception, and execution-mask behavior rows in the cited manual passage.
- Section 7.3 says ordinary `V_MFMA_F32_*_F32` honors MODE denormal flags,
  XF32 ignores `MODE.denorm`, matrix C input and D output do not flush
  denormals, sub-32-bit floating MFMA ignores MODE denorms, F64 MFMA ignores
  MODE and rounds to nearest even while allowing denorms, and DGEMM matrix
  operations support arithmetic exceptions in the cited manual passage.

Rocjitsu evidence:

- The generated dense MFMA paths pass only register bases, constant-accumulator
  state, and broadcast/negation fields into shared helpers; representative
  F32/XF32/F16/BF16/F64 calls are in
  the implementation.
- Shared extractors convert raw F16/BF16 input halves directly to host `float`
  in the implementation.
- The shared F32 MFMA helper accumulates with host `float` multiply-add and no
  shader MODE, denorm policy, or exception-state plumbing in the implementation.
- The shared F64 MFMA helper accumulates with host `double` arithmetic and
  `BLGP` negation bits, but has no DGEMM arithmetic-exception model or shader
  rounding/denorm state in the implementation.

Impact:

CDNA3 MFMA execution is currently a functional host-arithmetic model rather
than a model of the manual's per-family floating-point state. This is separate
from `CDNA3-RJ-010`, which records the missing clamp/`FP16_OVFL` behavior.

### CDNA3-RJ-014: MAI dependency-wait rules are not modeled

Manual evidence:

- Section 7.5 says users must insert NOPs or independent VALU instructions for
  specific MAI dependency cases and defines `DLop`, `XDLOP`, `DGEMM`, and
  `PASS` in the cited manual passage.
- Table 37 gives required waits for VALU-to-MFMA reads, DLop forwarding,
  XDL/SMFMA overlap cases, SGEMM overlap cases, DGEMM/F64 cases, and
  `V_CMPX*` execution-mask forwarding in the cited manual passage.

Rocjitsu evidence:

- Codegen only marks matrix instructions with a broad `MFMA` flag when the
  instruction name starts with `V_MFMA_` or `V_SMFMAC_` in the code generator.
- `ComputeUnit::issue_instruction()` executes non-memory instructions directly
  and deletes them after execution; only memory operations are routed through
  the memory wait-counter path in the implementation.
- `WaitCounters` tracks memory/export counter families such as VMCNT, LGKMCNT,
  EXPCNT, VSCNT, LOADCNT, STORECNT, DSCNT, KMCNT, TENSORCNT, and ASYNCCNT, but
  has no MAI pass/forwarding wait state in the implementation.

Impact:

Rocjitsu can execute instruction sequences that the CDNA3 manual requires
software to separate with independent instructions. This is a timing/scheduling
model gap rather than a single-instruction arithmetic mismatch.

### CDNA3-RJ-015: SMFMAC ignores sparse index-set selection from `CBSZ`/`ABID`

Manual evidence:

- For 16-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[1:0]` select one
  of four 8-bit sparse-index sets within the `SRC2` VGPR; if `CBSZ[1:0] != 0`,
  the first set is selected in the cited manual passage.
- For 8-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[0]` select one of
  two 16-bit sparse-index sets; if `CBSZ[1:0] != 0`, the first set is selected
  in the cited manual passage.
- The sparse restrictions say `CBSZ` and `ABID` only pick the sparse index from
  the VGPR read and do not perform ordinary MFMA A-matrix broadcast in the cited manual passage.

Rocjitsu evidence:

- The code generator emits F32-result SMFMAC calls that pass only `dst`, source
  bases, and an `idx` base to the sparse helpers; it does not pass `inst_.cbsz`
  or `inst_.abid` in the code generator.
- Representative generated CDNA3 F16/BF16 paths call sparse helpers with only
  `idx` in the implementation.
- Representative generated CDNA3 BF8/FP8 paths use the same index-base-only
  call shape in the implementation.
- The shared CDNA3 sparse helpers accept only `idx_base` and extract fixed
  low sparse-index fields from the `SRC2` VGPR in the implementation.

Impact:

SMFMAC encodings that should select a nonzero sparse-index set through
`ABID` execute with the helper's fixed/default index extraction. The helper
call shape correctly avoids ordinary dense-MFMA A-lane broadcast for SMFMAC, but
it drops the same fields before their sparse-specific selector meaning can be
applied.

### CDNA3-RJ-016: I32 SMFMAC variants are generated as unimplemented stubs

Manual evidence:

- The CDNA3 sparse matrix table includes `V_SMFMAC_I32_*_I8` variants in the cited manual passage.
- The detailed `V_SMFMAC_I32_16X16X64_I8` and
  `V_SMFMAC_I32_32X32X32_I8` definitions describe legal sparse I8-to-I32
  matrix multiply-accumulate instructions in the cited manual passage.

Rocjitsu evidence:

- The code generator intentionally emits a stub for non-F32-result SMFMAC
  variants in the code generator.
- Generated `VSmfmacI3216x16x64I8Vop3pMfma::execute_impl()` throws
  `UnimplementedInst` in the implementation.
- Generated `VSmfmacI3232x32x32I8Vop3pMfma::execute_impl()` does the same in the implementation.

Impact:

Legal CDNA3 sparse I8 matrix instructions decode and construct, but they throw
instead of executing.

### CDNA3-RJ-017: SMFMAC index-pair legality and VGPR alignment are not validated

Manual evidence:

- Each sparse index pair must satisfy `index0 < index1` and `index0 != index1`
  in the cited manual passage.
- `SRC0`, `SRC1`, and `VDST` VGPR addresses must be even-aligned in the cited manual passage.

Rocjitsu evidence:

- Shared SMFMAC helpers split each sparse-index nibble into two selectors and
  use both without checking ordering or distinctness in the implementation.
- Generated SMFMAC execution maps `SRC0`, `SRC1`, `VDST`, and `SRC2` through
  `src_base()` / `dst_base()` without sparse-specific validation; representative
  paths are in the implementation.
- `Isa::resolved_vgpr_offset()` accepts any encoded VGPR index and returns the
  unadjusted VGPR offset in the implementation.

Impact:

Invalid sparse-index pairs execute as ordinary selector pairs, and under-aligned
SMFMAC source or destination bases are accepted. The index-pair rule is unique
to SMFMAC; the alignment issue is the sparse instance of the broader dense-MFMA
register-block validation gap.

### CDNA3-RJ-018: Sparse SMFMAC tests miss selector and I8 execution contracts

Rocjitsu evidence:

- The gfx942 CTS skip list excludes the CDNA3 sparse MFMA fpsan test in the relevant tests.
- The generated-code tests only assert that CDNA3 FP8/BF8 SMFMAC generation
  uses FNUZ readers in the codegen tests.
- Searching the relevant tests for `V_SMFMAC` finds no
  sparse CDNA3 SMFMAC execution cases; the sparse exact SIMD tests in that
  directory cover RDNA4 `SWMMAC`, not CDNA3 `SMFMAC`.

Impact:

Current local tests can miss missing `CBSZ`/`ABID` sparse-index selection,
invalid index-pair handling, and the I8 SMFMAC stubs even when helper-selection
or unrelated sparse-WMMA tests pass.

### CDNA3-RJ-019: Generated CDNA3 DBT legalization rows duplicate sparse SMFMAC keys

Rocjitsu evidence:

- The generated legalization lookup key compares only `src_encoding_id` and
  `src_opcode`; `lookup()` uses `std::lower_bound()` and returns the first entry
  whose key matches in the implementation.
- CDNA3 VOP3P-MFMA opcode table entries identify sparse SMFMAC opcodes 98,
  100, 102, 104, 106, 108, and 120 through 127 in the cited manual passage.
- The generated CDNA3-to-CDNA4 table contains duplicate rows for those sparse
  opcodes under encoding id 423, with `Action::Identity` preceding
  `Action::Lower`, for example opcodes 98, 100, 102, 104, 106, 108, and
  120-127 in the implementation.
- The generated CDNA3-to-RDNA3 and CDNA3-to-RDNA4 tables contain duplicate rows
  for the same sparse opcode keys, usually with `Action::Lower` preceding
  `Action::Expand`, in the implementation.
- `select_legalization()` currently wires only CDNA4 guest legalization tables,
  so these CDNA3 guest tables are not active in the current DBT dispatch path in the implementation.

Impact:

If CDNA3 guest DBT legalization is enabled or used by tooling, the second row
for a duplicate sparse SMFMAC key is shadowed by the first matching row. For
CDNA3-to-CDNA4 that can turn a sparse opcode that also has a `Lower` entry into
an apparent identity mapping; for CDNA3-to-RDNA targets it can similarly hide
the alternate expansion/lowering action. Today this is generated-table
integrity debt rather than an active binary-translator runtime path.

### CDNA3-RJ-020: CDNA3 SMEM address calculation misses selector and alignment rules

Manual evidence:

- Chapter 8.1.1 gives scalar/global `S_LOAD`, `S_STORE`, and
  `S_DCACHE_DISCARD` addressing as `SBASE` plus an instruction offset plus M0,
  an SGPR offset, or zero, depending on `IMM` and `SOE`, in the cited manual passage.
- The same section says scratch SMEM uses the selected scalar offset multiplied
  by 64, and that all address components are byte quantities whose two low bits
  are ignored or forced to zero, in the cited manual passage.

Rocjitsu evidence:

- CDNA3 delegates SMEM address calculation to the shared helper in the implementation.
- The shared helper only adds `SOFFSET` when `SOFFSET_EN` is set and only adds
  `OFFSET` when `IMM` is set in the implementation.
- That path does not implement the `IMM=0, SOE=0` `OFFSET[6:0]` selector, does
  not treat offset value 124 as M0, does not mask the two low address bits, and
  is also used by generated CDNA3 scratch loads/stores.

Impact:

Legal CDNA3 SMEM encodings using the non-SOE offset selector, M0, unaligned
byte addresses, or scratch 64-byte offset units will execute against the wrong
address in rocjitsu.

### CDNA3-RJ-021: `S_BUFFER_*` SMEM ignores buffer resource descriptor and bounds semantics

Manual evidence:

- Chapter 8.1.1 says scalar buffer memory uses a four-SGPR resource descriptor
  containing base address, stride, `num_records`, and `NV`; the stride is used
  only for bounds checking and not for address calculation in the cited manual passage.
- Chapter 8.4 says an even `SBASE` is required for buffer loads and that
  out-of-bounds dwords are clamped by not performing those memory operations in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 scalar buffer loads and stores construct a 128-bit `SBASE`
  operand, but still call `smem_calculate_address()` like raw-pointer scalar
  memory operations in
  the implementation.
- The shared scalar address helper reads only an SGPR pair as a 64-bit base and
  returns base plus offset in the implementation.
- `ScalarMemState` carries only an address, width, data buffer, memory type,
  wait-counter type, and load/store flag, with no buffer descriptor or
  per-dword bounds state in the implementation.

Impact:

CDNA3 `S_BUFFER_*` instructions use raw-pointer-style addressing instead of the
manual's scalar-buffer descriptor semantics, and rocjitsu cannot suppress only
the out-of-bounds dwords of a buffer scalar-memory access.

### CDNA3-RJ-022: SMEM dependency counter, clause, and legality rules are not modeled

Manual evidence:

- Section 8.2 says scalar memory reads and writes can return out of order, can
  return partial results, and increment `LGKMCNT` by one for one dword or by two
  for two or more dwords in the cited manual passage.
- Sections 8.1.1, 8.3, and 8.4 describe source-overwrite hazards, scalar-memory
  clauses, SDATA/SBASE alignment restrictions, and out-of-range source/dest
  behavior in the cited manual passage.

Rocjitsu evidence:

- The scalar memory pipeline increments exactly one wait-counter slot per issued
  memory instruction and completes by writing the full data payload in the implementation.
- Generated CDNA3 SMEM constructors and execution paths do not validate the
  source-overwrite, clause, SDATA alignment, SBASE alignment, or out-of-range
  source/destination rules.

Impact:

Multi-dword SMEM instructions undercount `LGKMCNT`, and rocjitsu accepts or
executes scalar-memory sequences and operands that the manual marks as
restricted or undefined.

### CDNA3-RJ-023: Scalar atomics decode but do not execute

Manual evidence:

- Section 8.1.2 says scalar atomics support the same operations as vector memory
  atomics, use the same address calculations, and return the pre-atomic value
  when `GLC` is set in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 scalar atomic classes construct operands, but their
  `execute_impl()` bodies throw `UnimplementedInst` in
  the implementation.
- Python semantic derivation classifies SMEM atomics as `nop` in the code generator.

Impact:

Legal CDNA3 scalar atomic instructions cannot execute in rocjitsu, and Python
tooling can silently classify those instructions as no-ops rather than
unsupported read-modify-write operations.

### CDNA3-RJ-024: SMEM `GLC`/`NV` cache policy and discard are incomplete

Manual evidence:

- Chapter 8.1 describes `GLC` load/store/atomic policy and `NV` in the cited manual passage.
- The cache-policy table says `GLC=0` and `GLC=1` change read, write, and
  atomic persistence/return behavior in the cited manual passage.
- Section 8.1.3 defines scalar cache invalidate/write-back operations, and
  section 8.1.4 defines `S_MEMTIME`, `S_MEMREALTIME`, and discard operations in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 SMEM load/store execution sets `ScalarMemState::mtype` from
  `GLC`, but `ScalarMemPipeline::initiate_access()` calls scalar L1 load/store
  helpers without passing that instruction memory type.
- The scalar L1 cache derives memory type from the page-table entry rather than
  from the SMEM instruction flags in
  the implementation.
- Generated CDNA3 `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` execute paths
  throw `UnimplementedInst`, and `NV` is not consumed by the scalar-memory
  pipeline.

Impact:

Scalar-memory `GLC`/`NV` policy bits are decoded but mostly inert for execution,
and legal discard operations remain unimplemented.

### CDNA3-RJ-025: `S_MEMTIME` and `S_MEMREALTIME` use indistinguishable placeholder counters

Manual evidence:

- Section 8.1.4 describes `S_MEMTIME` as a scalar-memory timer query and
  `S_MEMREALTIME` as reading the 100 MHz real-time clock in the cited manual passage.

Rocjitsu evidence:

- Shared execution helpers for `S_MEMTIME` and `S_MEMREALTIME` each use a
  separate `thread_local` counter incremented by 100 per call in the implementation.

Impact:

Both timer instructions currently provide deterministic placeholder values, but
rocjitsu does not model the architectural distinction between the memory timer
and the 100 MHz real-time clock.

### CDNA3-RJ-026: Buffer descriptor addressing and range checking are incomplete

Manual evidence:

- Section 9.1.5 defines buffer addresses from resource base, SGPR offset, VGPR
  offset/index, stride, element size, `ADD_TID`, swizzle state, and `NumRecords`
  in the cited manual passage.
- Section 9.1.5.1 defines private, raw, and structured range-checking modes and
  the `dst_sel = SEL_1` OOB read exception in the cited manual passage.
- Section 9.1.8 defines the full 128-bit descriptor layout and says an all-zero
  resource acts as an unbound buffer returning zero and dropping writes in the cited manual passage.

Rocjitsu evidence:

- `mubuf_calculate_addresses()` reads base, 14-bit stride, `NumRecords`, and
  a single `oob_raw` bit from the descriptor in the implementation.
- Its range checks use `oob_raw`, `stride`, `index`, and `offset_part`, but do
  not model `ADD_TID`, swizzle enable, 18-bit stride extension, private-scratch
  no-range-check mode, descriptor type/user-VM/NV/reserved bits, unbound
  all-zero behavior, `dst_sel = SEL_1`, or all-or-nothing versus per-component
  distinctions in the implementation.
- `mtbuf_calculate_addresses()` follows the same reduced descriptor model in the implementation.

Impact:

Basic linear buffer cases can execute, but descriptor-driven scratch/private,
swizzled, unbound, and several OOB/read-channel cases can diverge from CDNA3
hardware semantics.

### CDNA3-RJ-027: Buffer format conversion and `dst_sel` semantics are missing

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and buffer data-format
  conversion in the cited manual passage.
- Section 9.1.4 says MTBUF takes format from the instruction, formatted MUBUF
  takes format and `dst_sel` from the resource, raw MUBUF derives size/type
  from the opcode, INVALID resource format remains unbound, and D16 variants
  pack/load/store 16-bit values in the cited manual passage.
- Section 9.1.11 begins the buffer data-format enum table in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 MUBUF formatted load/store bodies throw `UnimplementedInst`
  for representative non-D16 formatted operations in the implementation and for representative D16 formatted operations.
- Generated CDNA3 MTBUF formatted loads and stores use fixed 4-byte element
  sizes and raw VGPR payload transfers rather than `DFMT`/`NFMT` conversion;
  representative load/store bodies are in the implementation.
- Implemented CDNA3 `TBUFFER_STORE_FORMAT_D16_XY`, `_XYZ`, and `_XYZW` bodies
  repeatedly read the low half of `VDATA` for every component in the implementation, while Chapter 12.14 stores X/Y/Z/W from consecutive 16-bit component
  halves in the cited manual passage.
- Python semantic derivation maps MTBUF format mnemonics to fixed element
  counts and classifies unmatched MUBUF format mnemonics as `nop` in the code generator.

Impact:

Rocjitsu does not yet model CDNA3 typed-buffer conversion, resource-derived
formatted MUBUF conversion, destination-channel selection, INVALID/unbound
format behavior, or D16 formatted packing.

### CDNA3-RJ-131: MTBUF D16 formatted loads drop packed components

Manual evidence:

- Section 9.1.4 says D16 buffer loads convert returned data to 16 bits and pack
  pairs of data into each 32-bit VGPR, LSBs first and then MSBs, in the cited manual passage.
- Chapter 12.14 spells out that `TBUFFER_LOAD_FORMAT_D16_XY` writes X into
  `VDATA[15:0]` and Y into `VDATA[31:16]`, `TBUFFER_LOAD_FORMAT_D16_XYZ`
  writes Z into `VDATA[47:32]` while preserving `VDATA[63:48]`, and
  `TBUFFER_LOAD_FORMAT_D16_XYZW` writes all four 16-bit components in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 `TBUFFER_LOAD_FORMAT_D16_XY`, `_XYZ`, and `_XYZW` bodies set
  `elem_size = 2`, `num_elems = 2`, `3`, or `4`, and a single `d16_lo = true`
  flag before issuing the vector-memory operation in the implementation.
- The shared vector-memory completion path computes `vgpr_count` as
  `std::max(1u, total_bytes / 4)` and, for non-atomic sub-dword loads, copies
  only `elem_size` bytes at offsets `i * 4` before merging them as either the
  low or high half in the implementation.
  For D16 XY only X is written; for XYZ only X is written; for XYZW, X and Z are
  written into the low halves of successive VGPRs while Y and W are skipped.
- The MTBUF semantic derivation only records `(elem_size, num_elems, d16_lo)`;
  it has no semantic shape for packing successive 16-bit components into low and
  high halves of consecutive VGPRs in the code generator.

Impact:

End-to-end CDNA3 typed D16 buffer loads with two, three, or four components can
return only a subset of the components and place them in the wrong halves of the
destination VGPRs, even when address calculation and memory bytes are otherwise
valid. `CDNA3-RJ-027` covers the broader missing format conversion and D16 store
packing; this entry covers the separate implemented-load packing bug.

### CDNA3-RJ-028: Buffer `SOFFSET` and dword-alignment edge cases are not modeled

Manual evidence:

- The MUBUF/MTBUF field table says `SOFFSET` must be an SGPR, M0, or inline
  constant in the cited manual passage.
- Section 9.1.5 says the SGPR offset can come from an SGPR or M0 in the cited manual passage.
- Section 9.1.7 says dword-or-larger reads and writes ignore the two address
  LSBs, forcing dword alignment, in the cited manual passage.

Rocjitsu evidence:

- The MUBUF helper treats `SOFFSET == 0x80` as inline constant zero and all
  other values as SGPR indices in the implementation; MTBUF does the same.
- The final MUBUF and MTBUF addresses are written directly from
  `base_addr + total_offset` in the implementation; there is
  no element-size-dependent clearing of low address bits.

Impact:

M0 `SOFFSET`, nonzero inline constants, and misaligned dword-or-larger buffer
addresses can execute differently from the manual.

### CDNA3-RJ-029: Buffer-to-LDS subset, M0 offset, and clamping are incomplete

Manual evidence:

- Section 9.1.9 says load-to-LDS is supported only for
  `BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,format_x}`, defines
  `LDS_offset = M0[15:0]`, and requires active-mask clamping so return data is
  not written outside the LDS allocation for the wave in the cited manual passage.

Rocjitsu evidence:

- Allowed raw byte/short/dword loads do implement an `inst_.lds` path, but use
  `wf.m0() + wf.lds_base()` as the base in the implementation.
- The same `inst_.lds` pattern is also generated for `buffer_load_dwordx2`,
  `dwordx3`, and `dwordx4` in the implementation, and for raw D16
  loads, even though those forms are outside the
  manual's listed LDS subset.
- The vector-memory completion path writes LDS-destination loads
  `lds_base + lane * per_lane_bytes` when no per-lane LDS address is present in the implementation.
  The LDS backing drops writes beyond total LDS size in the implementation, but the
  buffer path does not derive the manual's allocation-aware active mask before
  the memory read.

Impact:

Rocjitsu can accept LDS forms the manual does not list, use high bits of M0 in
the LDS offset, and issue reads for lanes whose return data should be masked by
LDS allocation clamping.

### CDNA3-RJ-030: Buffer cache-control and cache-maintenance policies are coarse

Manual evidence:

- Section 9.1.10 gives detailed vector-memory load, store, atomic, invalidate,
  and writeback cache-policy tables for `SC[1:0]` and `NT`, including
  `TG_SPLIT` behavior and SC-dependent `BUFFER_WBL2`/`BUFFER_INV` effects, in the cited manual passage.

Rocjitsu evidence:

- `mtype_from_flags_gfx940()` collapses `SC0`, `SC1`, and `NT` into a coarse
  `Mtype` value in the implementation.
- Shared `BUFFER_INV`/`BUFFER_WBL2` helpers invalidate or flush broad cache
  levels without consulting the SC table or `TG_SPLIT` refinements in the implementation, and generated CDNA3 `BUFFER_WBL2`/`BUFFER_INV` dispatches
  simply call those helpers in the implementation.

Impact:

The current model has useful broad cache-behavior hooks, but not the
instruction- and scope-specific CDNA3 policy required for precise cache-control
validation.

### CDNA3-RJ-031: Vector-buffer tests miss descriptor and format edge cases

Rocjitsu evidence:

- The relevant tests covers dynamic VADDR width,
  SRSRC scaling, and CDNA ACC bank folding for MUBUF/MTBUF operands.
- The relevant tests covers the MUBUF `lds`
  modifier in disassembly.
- The codegen tests
  covers generator helpers for legacy buffer VADDR width and SRSRC scaling, and
  the codegen tests covers a
  derived `BUFFER_LOAD_B32` semantic shape.

Impact:

The existing tests cover important operand/codegen regressions, but not the
manual-derived descriptor modes, format conversion, M0/inline `SOFFSET`,
alignment, LDS clamping/subset, unbound resource, or cache-policy cases
identified in the vector-buffer audit slice.

### CDNA3-RJ-032: FLAT/GLOBAL/SCRATCH address-mode and fault behavior is incomplete

Manual evidence:

- Chapter 10 says flat memory uses aperture registers to distinguish
  video/system, LDS, and scratch spaces and that unmapped regions generate a
  memory violation in the cited manual passage.
- Section 10.3 says FLAT supports 32-bit and 64-bit addressing according to the
  `PTR32` mode register, and sections 10.4 and 10.5 say global/scratch address
  component size depends on `ADDRESS_MODE` in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 FLAT constructors fix the ordinary flat address operand as a
  64-bit VGPR pair, narrow scratch to a 32-bit VGPR offset, and narrow global
  to a 32-bit VGPR offset only when `saddr != 0x7F`; representative code is in
  the implementation.
- The shared helper uses fixed segment formulas: FLAT always reads a 64-bit VGPR
  pair, GLOBAL chooses either 64-bit VGPR pair or 64-bit SGPR base plus
  sign-extended 32-bit VGPR offset, and SCRATCH uses scratch base plus lane
  stride plus optional VGPR/SADDR offsets in the implementation.
- The helper maps private-aperture FLAT addresses into scratch backing memory
  and the compute unit maps shared-aperture addresses into LDS, but there is no
  visible `PTR32`/`ADDRESS_MODE` state or memory-violation path for aperture
  holes in the implementation.

Impact:

Rocjitsu has useful segment and aperture routing, but it cannot model
mode-dependent address-size behavior or flat aperture fault policy from the
CDNA3 manual.

### CDNA3-RJ-033: Global/scratch direct-to-LDS flat-memory forms are not modeled

Manual evidence:

- Sections 10.4 and 10.5 say GLOBAL and SCRATCH instructions can transfer data
  directly between LDS and memory in the cited manual passage.
- Section 10.3 gives LDS destination formulas using the hardware LDS base,
  `M0[17:2] * 4`, instruction offset, and `ThreadID` scaling in the cited manual passage.

Rocjitsu evidence:

- The generated flat-load body only sets a VGPR destination and issues a
  `VectorMemState(GLOBAL_MEM)` through `flat_calculate_addresses()`; it never
  sets `lds_dst`, `lds_base`, or per-lane LDS addresses in the implementation
  and the generator template in the code generator.
- A targeted search of CDNA3 flat generated code found no `LOAD_LDS`,
  `lds_dst`, or `wf.m0()` handling in the flat instruction family.
- The CDNA3 subdecode table marks FLAT opcodes 38 through 42 invalid in the implementation, even though the manual and XML reserve those opcode numbers for
  `GLOBAL_LOAD_LDS_*` and `SCRATCH_LOAD_LDS_*`.
- The profile generator skips segment-specific flat encodings, including
  `ENC_FLAT_GLBL` and `ENC_FLAT_SCRATCH`, in the code generator. The generic FLAT
  class sharing therefore drops segment-only opcodes 38 through 42.
- Semantic derivation strips `GLOBAL_LOAD_`/`SCRATCH_LOAD_` and looks up the
  remaining suffix in `_FLAT_DATA_MAP` in the code generator; suffixes such as
  `LDS_DWORD` miss the data map and fall through as non-modeled rows.
- The generated `FlatMachineInst` exposes bit 13 as `lds`, while the
  specialized global/scratch structs expose it as `sve`, and the builder calls
  the bit `lds`; this preserves the bit but does not create the direct-LDS
  destination path in the implementation.

Impact:

Ordinary VGPR-return global/scratch memory can execute, but the manual's
GLOBAL/SCRATCH-to-LDS forms lack the destination and `M0` address semantics
needed for end-to-end execution.

### CDNA3-RJ-034: FLAT wait-counter and ordering contract is simplified

Manual evidence:

- Section 10.2 says FLAT instructions execute internally as both LDS and Buffer,
  increment both `VM_CNT` and `LGKM_CNT`, and complete only when both counters
  decrement in the cited manual passage.
- Sections 10.2.1 and 10.2.2 describe out-of-order completion, same-VGPR return
  hazards, and the `S_WAITCNT 0` restriction in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 flat loads, stores, and atomics set
  `wait_counter_type = WaitCounterType::VMCNT` before issuing; representative
  bodies are in the implementation.
- The compute-unit router changes shared-aperture FLAT operations to the local
  pipeline and `LGKMCNT`, but this is an either/or counter choice rather than a
  dual VM+LGKM issue/retire model in the implementation.

Impact:

The current model tracks flat memory enough for functional completion, but not
the dual-counter and ordering behavior needed for wait-counter exactness.

### CDNA3-RJ-035: Floating flat atomics miss packed and FP-mode special cases

Manual evidence:

- Section 10.3.1 says floating-point atomics must use `SC[0]=0`, FP32 atomics
  flush denormals to zero, FP64 and FP16 atomics do not flush denormals, and
  rounding is fixed RNE in the cited manual passage.
- The FLAT instruction definitions include `FLAT_ATOMIC_ADD_F32`,
  `FLAT_ATOMIC_PK_ADD_F16`, `FLAT_ATOMIC_ADD_F64`, `FLAT_ATOMIC_MIN_F64`,
  `FLAT_ATOMIC_MAX_F64`, and `FLAT_ATOMIC_PK_ADD_BF16` in the cited manual passage.

Rocjitsu evidence:

- Generated floating flat atomics use `d->is_load = (inst_.sc0 != 0)`, so
  return-data mode is still allowed for FP atomics; representative F32/F64
  bodies are in the implementation.
- `FLAT_ATOMIC_PK_ADD_F16` and `FLAT_ATOMIC_PK_ADD_BF16` both set
  `elem_size = 4` and `atomic_op = AtomicOp::FADD`, the same path used for
  scalar F32 atomics, in the implementation.
- The memory pipeline treats 4-byte floating atomics as a single `float` and
  8-byte floating atomics as a single `double`; `apply_fp_atomic()` uses
  ordinary `+`, `std::fmin`, and `std::fmax` without FP32 denormal flushing or
  packed F16/BF16 lane handling in the implementation.

Impact:

F32/F64 atomic add/min/max have a coarse functional model, but packed F16/BF16
atomics are not type-correct and the CDNA3 FP atomic `SC0`, denormal, and
rounding rules are not enforced.

### CDNA3-RJ-036: Flat/global/scratch tests miss CDNA3-specific edge cases

Rocjitsu evidence:

- Shared address tests cover several scratch/global formulas through hand-built
  machine instructions, including CDNA4 and RDNA3 scratch/global cases in the relevant tests.
- Existing decode smoke coverage exercises ACC on global/scratch forms and
  adjacent memory tests exercise generic flat/global behavior, but the static
  pass did not find CDNA3 end-to-end cases for `PTR32`/`ADDRESS_MODE`,
  direct-to-LDS global/scratch forms, scratch atomic decode rejection, FLAT
  dual-counter behavior, aperture-hole faults, no-return atomic def-use
  metadata, or packed FP atomic execution.

Impact:

The tests cover useful address-helper regressions, but the manual-derived CDNA3
contracts identified in this slice can regress without focused coverage.

### CDNA3-RJ-037: LDS M0 clamping and bank-conflict behavior are not modeled

Manual evidence:

- Section 11.1 describes LDS as 32 banks and says bank conflicts are serialized
  for indexed and atomic operations in the cited manual passage.
- Section 11.3.1 says all LDS operations require `M0` initialization, and that
  `M0[16:0]` contains the LDS segment byte-size and clamps the final address in the cited manual passage.

Rocjitsu evidence:

- CDNA3 DS address calculation delegates to the shared DS helper in the implementation.
- The shared helper computes `VGPR[ADDR] + concatenated offset + lds_base` and
  never reads `wf.m0()` for normal DS operations in the implementation.
- Representative DS2 bodies compute the two addresses directly from `ADDR`,
  scaled offsets, and `wf.lds_base()` without M0 clamping in the implementation.
- The local-memory pipeline performs functional vector loads/stores and atomics
  in the implementation, with no LDS bank or conflict-timing model.

Impact:

Out-of-range LDS accesses can execute without the manual's `M0[16:0]` clamp,
and bank-conflict behavior is not visible to timing or hazard-sensitive tests.

### CDNA3-RJ-038: DS READ2/WRITE2 duplicate-offset collapse is not modeled

Manual evidence:

- Section 11.3.1 says a two-address operation can specify only one address by
  setting both offsets to the same value; this causes only one read/write and
  uses only the first `DATA0` field in the cited manual passage.

Rocjitsu evidence:

- `DsRead2B32Ds::execute_impl()` always sets `ds2_active = true`, computes both
  addresses from `offset0` and `offset1`, and sets a second destination register
  in the implementation.
- The local-memory pipeline always issues the second DS2 load when
  `ds2_active` is true and writes the second response during completion in the implementation.
- Generated WRITE2 bodies follow the same unconditional second-address pattern;
  representative B32 code computes both addresses and fills `ds2_store_data` in the implementation.

Impact:

Equal-offset READ2/WRITE2 encodings perform two modeled accesses in rocjitsu
instead of the manual's one-access form. Stores can use `DATA1` where hardware
should ignore it, and reads can write a second destination value where only one
access should occur.

### CDNA3-RJ-039: No-return DS atomics can clobber `VDST`

Manual evidence:

- Section 11.3.1 distinguishes LDS atomic updates from the optional return of
  the pre-operation value to VGPRs in the cited manual passage.

Rocjitsu evidence:

- The no-return `ds_add_u32` constructor has `num_dst_ = 0`, but its execute
  path still sets `d->dst_reg_base =... + inst_.vdst`, `d->is_load = true`,
  and `d->atomic_op = AtomicOp::ADD` in the implementation.
- `MemoryPipeline::issue()` always calls `complete_access()` after initiating a
  memory operation in the implementation.
- Local memory atomics execute `execute_lds_atomic_rmw()` and then call
  `vector_complete()` in the implementation.
- `vector_complete()` writes response data to `dst_reg_base` whenever
  `d.is_load` is true, regardless of the instruction's `num_dst_`, in the implementation.

Impact:

No-return DS atomics can write the returned old LDS value into the encoded
`VDST` register even though the instruction has no architectural destination.
This can silently corrupt a VGPR chosen as the unused `VDST` field.

### CDNA3-RJ-040: Packed F16/BF16 LDS atomics execute through the scalar F32 atomic path

Manual evidence:

- The DS opcode table includes packed F16/BF16 LDS atomic add forms in the cited manual passage.
- Section 11.3.1 says LDS floating atomics follow `MODE.FP_DENORM` denormal
  behavior and fixed round-to-nearest-even rounding in the cited manual passage.

Rocjitsu evidence:

- `DS_PK_ADD_F16` and `DS_PK_ADD_BF16` set `elem_size = 4` and
  `atomic_op = AtomicOp::FADD`, the same memory-pipeline operation used for
  scalar F32 atomics, in the implementation.
- The return forms `DS_PK_ADD_RTN_F16` and `DS_PK_ADD_RTN_BF16` do the same in the implementation.
- `execute_lds_atomic_rmw()` treats `elem_size == 4` floating atomics as one
  `float`, and `apply_fp_atomic()` uses ordinary C++ add/min/max without packed
  half or BF16 lane handling, denormal-mode logic, or explicit RNE behavior in the implementation.

Impact:

Packed LDS F16/BF16 atomics are type-incorrect: rocjitsu interprets the 32-bit
word as a scalar F32 value instead of two packed 16-bit lanes, and it cannot
model the manual's FP-mode requirements.

### CDNA3-RJ-041: GDS/GWS forms are decoded but not executed

Manual evidence:

- The DS format includes the `GDS` bit, and Chapter 11.4 adds GWS restrictions:
  all GWS instructions must be followed immediately by `s_waitcnt 0`, and VGPRs
  used by GWS instructions must be even in the cited manual passage.
- Section 12.12 says GWS instructions operate only on the first active lane in
  the EXEC mask in the cited manual passage.
- The detailed GWS definitions describe resource-id calculation and semaphore
  or barrier state-machine behavior in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 GWS classes decode `DS_GWS_SEMA_RELEASE_ALL`,
  `DS_GWS_INIT`, `DS_GWS_SEMA_V`, `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and
  `DS_GWS_BARRIER`, but every execute body throws `util::UnimplementedInst` in the implementation.
- Generated CDNA3 DS execute bodies throw `util::UnimplementedInst` whenever
  `inst_.gds` is set; representative load, store, atomic, and ADDTID paths are
  in the implementation.
- Codegen classifies `DS_GWS_*` semantics as `nop` because they are hardware
  scheduling primitives in the code generator.
- The static pass did not find a GDS/GWS execution path or validation for
  first-active-lane execution, resource state, immediate-following
  `s_waitcnt 0`, or even-VGPR restrictions.

Impact:

LDS forms execute functionally, but GDS/GWS encodings cannot execute and their
manual lane, state, sequence, and register constraints are not represented.

### CDNA3-RJ-042: Data-share tests miss CDNA3 LDS edge cases

Rocjitsu evidence:

- Existing VM atomic stress tests include CDNA4 `ds_add_rtn_u32` and a
  no-return `ds_add_u32` memory-result case in the relevant tests, but the no-return test only checks final LDS contents and
  does not assert that the unused `VDST` register remains unchanged.
- Decode smoke coverage exercises a DS ACC destination case in the relevant tests, and the HIP ACC transpose
  test checks one DS transpose/ACC behavior in the relevant tests.
- The static pass did not find focused CDNA3 tests for `M0[16:0]` LDS clamping,
  READ2/WRITE2 equal-offset collapse, packed F16/BF16 LDS atomics, GWS
  `GDS=1` encoding/disassembly, GDS/GWS runtime behavior, or LDS FP atomic MODE
  behavior.

Impact:

The current tests cover useful ordinary DS plumbing, but the manual-derived LDS
contracts identified in this slice can regress without detection.

### CDNA3-RJ-043: Generated GWS encodings and decode metadata miss required `GDS=1`

Manual/oracle evidence:

- Section 12.12 says `GDS` is set for GWS and clear for LDS in the cited manual passage.
- LLVM's assembler accepts GWS forms only with the explicit `gds` modifier:
  `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -show-encoding` encodes
  `ds_gws_init v2 gds` as bytes `[0x00,0x00,0x33,0xd9,0x02,0x00,0x00,0x00]`,
  where DS bit 16 is set.
- The same LLVM oracle rejects missing-`gds` assembly as "too few operands" and
  rejects odd data operands such as `v3` as "vgpr must be even aligned".
- LLVM disassembly treats the corresponding `GDS=0` byte sequence
  `0x00 0x00 0x32 0xd9 0x02 0x00 0x00 0x00` as an invalid instruction
  encoding rather than as `ds_gws_init`.

Rocjitsu evidence:

- Generated CDNA3 test encodings list the GWS opcodes with dword0 values
  `0xD9300000`, `0xD9320000`, `0xD9340000`, `0xD9360000`, `0xD9380000`, and
  `0xD93A0000`, all with the `GDS` bit clear, in the decode fixtures.
- The generated DS decoder table selects GWS decoders by opcode slots 152
  through 157 without a `GDS=1` condition in the implementation.
- `DsMachineInst` carries the `gds` bit, but the CDNA3 `Ds` encoding base has
  no `build_modifiers()` override to print a required `gds` modifier in the implementation.

Impact:

Rocjitsu's generated GWS fixture encodings and decode metadata can treat
non-GDS byte patterns as valid GWS instructions, while LLVM and the manual
require the `GDS` bit for these opcodes. Disassembly also lacks the required
`gds` modifier, so textual output can round-trip to a form LLVM rejects.

### CDNA3-RJ-044: SALU allocation-bound fallback and out-of-range destination side effects are not modeled

Manual evidence:

- Section 5.2 says SALU instructions cannot use VGPRs or LDS, source selector
  255 consumes the following 32-bit literal dword for all SALU formats except
  SOPP and SOPK, out-of-range source SGPRs read SGPR0, and out-of-range
  destination SGPRs suppress the SGPR write while still writing SCC and EXEC
  saveexec side effects in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` tracks a per-dispatch SGPR allocation count separately from the
  physical allocation base in the implementation.
- The CDNA3 scalar source resolver reads SGPR selectors by adding the raw
  selector to `wf.sgpr_alloc().base` and has no `wf.num_sgprs()` or
  `sgpr_alloc().count` bound check before reading in the implementation.
- Unsupported scalar source selectors throw in the implementation, and unsupported scalar destination selectors throw in the implementation.
- Some scalar helpers perform the destination write before the SCC side effect;
  for example `execute_s_add_u32_sop2()` writes `SDST` before `wf.write_scc()`
  in the implementation.

Impact:

Rocjitsu can read or write outside the wavefront's declared SGPR allocation, or
throw for raw unsupported destination selectors before completing side effects
that the manual says survive an out-of-range destination. That makes raw-SALU
edge encodings and low-SGPR-count dispatches diverge from the manual's fallback
contract.

### CDNA3-RJ-045: SALU 64-bit SGPR pair alignment is not enforced

Manual evidence:

- Section 5.2 says any instruction using 64-bit data in SGPRs must use an
  even-aligned SGPR pair, giving `s[2:3]` and `s[8:9]` as legal examples and
  `s[11:12]` as illegal in the cited manual passage.
- The SOP1, SOP2, and SOPC format tables expose 64-bit scalar operations that
  use the same SSRC/SDST selector space, for example `S_MOV_B64` in the SOP1
  opcode table in the cited manual passage.

Rocjitsu evidence:

- Generated 64-bit scalar instructions keep raw 64-bit `OPR_SDST`/`OPR_SSRC`
  operands; representative `S_MOV_B64` construction is in the implementation.
- `resolve_src_scalar64()` reads SGPR selector `ev` and `ev + 1` for ordinary
  scalar and TTMP pairs without checking that `ev` is even in the implementation.
- `resolve_dst_write64()` writes SGPR selector `ev` and `ev + 1` with the same
  missing even-boundary check in the implementation.

Impact:

Odd 64-bit SGPR pairs such as `s[11:12]` execute as ordinary consecutive
register pairs in rocjitsu even though the manual declares them illegal.

### CDNA3-RJ-046: SALU implicit SCC operands are not surfaced in C++ def-use metadata

Manual/XML evidence:

- Chapter 5 says many SALU operations set SCC for comparisons, carry-out, or
  zero-result testing in the cited manual passage.
- The XML includes implicit SCC output operands on representative writers such
  as `S_ADD_U32` and `S_CMP_EQ_I32` in the machine-readable ISA XML.
- The XML includes an implicit SCC input operand on `S_CBRANCH_SCC0` in the machine-readable ISA XML.

Rocjitsu evidence:

- Execution does read and write SCC directly; representative helpers call
  `wf.write_scc()` and `wf.read_scc()` in the implementation.
- Generated CDNA3 constructors omit the XML implicit SCC operands from the
  C++ operand arrays: `S_ADD_U32` exposes only `SDST`, `SSRC0`, and `SSRC1` in the implementation;
  `S_CMP_EQ_I32` exposes only its two explicit sources in the implementation;
  and `S_CBRANCH_SCC0/SCC1` expose only the label operand in the implementation.
- `InstDefUse` derives metadata from explicit operands plus
  `implicit_defs()`/`implicit_uses()` in the implementation,
  but `Instruction`'s default implicit hooks are no-ops in the implementation.
- Even if an implicit SCC `RegisterRef` were produced, `RegisterSet` currently
  documents SCC as untracked and ignores non-SGPR/VGPR/ACC classes in the implementation.

Impact:

Scalar execution has SCC behavior, but C++ def-use and liveness consumers do
not see the XML's implicit SCC reads/writes. Any analysis or patch planning
that depends on decoded `RegisterSet` metadata must handle SCC separately.

### CDNA3-RJ-047: `S_MAX_{I32,U32}` clears SCC for equal operands

Manual evidence:

- CDNA3 detailed `S_MAX_I32` and `S_MAX_U32` definitions set SCC with inclusive
  predicates, `SCC = S0.i32 >= S1.i32` and `SCC = S0.u32 >= S1.u32`, then
  select `D0 = SCC ? S0 : S1` in the cited manual passage.
- CDNA4 and RDNA4 detailed manuals use the same inclusive max predicate in the cited manual passage.

Rocjitsu evidence:

- The shared `S_MAX_I32` and `S_MAX_U32` helpers compute the correct numeric
  max but write SCC with strict `s0 > s1` in the implementation.
- The semantic-derivation regression explicitly requires strict `>` and rejects
  `>=` in the codegen tests.
- The cross-architecture scalar SCC runtime test expects `false` for equal
  `s_max_i32` and `s_max_u32` inputs in the relevant tests.

Impact:

Equal operands produce the same destination value under either predicate, so
ordinary destination-only tests pass. Any subsequent SCC consumer, such as
`S_CSELECT`, `S_CBRANCH_SCC*`, or a carry-chain probe using SCC state, can
observe the divergence.

### CDNA3-RJ-049: SETREG side effects, unsupported state, and spacing remain incomplete

Manual evidence:

- Section 5.8 requires an `S_NOP` between consecutive `S_SETREG` writes and
  applies privilege-sensitive `HwRegWriteMask` behavior with
  register-specific side effects in the cited manual passage.

Rocjitsu evidence:

- CDNA3 SOPK paths now use the correct architecture table, partial-field
  extraction/insertion, and read-only/user/privileged policies in
  the implementation.
- MODE is writable and STATUS read-only, but TRAPSTS, LDS_ALLOC, IB_STS,
  PC/debug, TBA/TMA, XCC_ID, and performance snapshot rows remain unsupported.
- No audited path models register-specific write side effects or the required
  dependency between consecutive SETREG writes.

Impact:

Basic ID and permission handling is patched; unsupported state, side effects,
and SETREG instruction-spacing behavior still diverge from the manual.

### CDNA3-RJ-051: Raw STATUS `EXECZ`/`VCCZ` bits can drift from EXEC/VCC

Manual evidence:

- The STATUS table defines `EXECZ` as "Exec mask is zero" and `VCCZ` as
  "Vector condition code is zero" in the cited manual passage.
- EXEC itself is a 64-bit execution mask, and `EXECZ` is a helper bit usable as a
  branch condition in the cited manual passage.
- STATUS fields are readable by shader code, initialized at wavefront creation,
  and some are set by shader instructions in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` stores EXEC and VCC as separate `exec_` and `vcc_` members, and
  `set_exec()`/`set_vcc()` update only those members in the implementation.
- Raw STATUS is a separate ISA-specific `StatusType status{0}` exposed through
  `status_raw()` and `set_status_raw()` in the implementation.
- CDNA status bit wrappers define `EXECZ` and `VCCZ` as bits 9 and 10 in the implementation, but the EXEC/VCC setters do not update those bits.
- Branch and scalar-source paths compute `VCCZ`/`EXECZ` directly from
  `wf.vcc()` and `wf.exec()` in the implementation.
- The generated CDNA3 HWREG path reads raw STATUS with `wf.status_raw()` for
  its handled status-like register in the implementation.

Impact:

Conditional branches and source operands can see the correct derived EXEC/VCC
zero state while raw STATUS readers see stale bits. Any code path that observes
STATUS rather than recomputing helper values can disagree with the architectural
helper-bit contract.

### CDNA3-RJ-052: `S_SETVSKIP` is decoded but unimplemented

Manual evidence:

- MODE bit 28 is `VSKIP`; when set, vector, VMEM, LDS, and GDS instructions are
  skipped rather than issued in the cited manual passage.
- The instruction overview describes `S_SETVSKIP` as setting a bit that causes
  vector instructions to be ignored in the cited manual passage.
- The detailed `S_SETVSKIP` definition says it enables or disables VSKIP and
  that VSKIPped memory instructions do not manipulate wait counters in the cited manual passage.

Rocjitsu evidence:

- CDNA3 decodes `S_SETVSKIP` and constructs two scalar source operands in the implementation.
- `SSetvskipSopc::execute_impl()` immediately throws `UnimplementedInst` in the implementation.
- The CU execution path directly dispatches each decoded instruction through
  `execute_instruction()` in the implementation,
  and the concrete execute hook simply calls `inst->execute()` in the implementation;
  there is no VSKIP-mode gate in the audited dispatch path.

Impact:

Programs that use `S_SETVSKIP` trap in the emulator instead of toggling
MODE.VSKIP. Even if the bit were set through another path, vector/memory/LDS
instruction issue is not suppressed by the execution dispatcher.

### CDNA3-RJ-053: LDS workgroup allocation uses 256-byte granularity instead of 512-byte blocks

Manual evidence:

- Chapter 3 says LDS is allocated per work-group or wavefront in contiguous
  512-byte blocks on 512-byte alignment, with no wrap, and LDS clamping uses the
  smaller of the SPI allocation size and M0 in the cited manual passage.

Rocjitsu evidence:

- The command-processor accounting helper aligns per-workgroup LDS to 256 bytes
  in the implementation.
- CU admission checks use `util::align_up(lds_bytes, 256u)` in the implementation.
- CU-local allocation also advances `next_lds_alloc_` by a 256-byte-aligned
  size in the implementation.
- The WGP placement path uses the same 256-byte alignment in the implementation, and dispatch
  validation reports the 256-byte-aligned value in the implementation.

Impact:

Rocjitsu can pack workgroups whose LDS allocations would occupy distinct
512-byte blocks on CDNA3 hardware. That can expose different workgroup
residency, LDS base addresses, and allocation-limit behavior for kernels with
nonzero LDS sizes below or not aligned to 512 bytes.

### CDNA3-RJ-054: Trap and exception state is not modeled

Manual evidence:

- Chapter 3.10 says traps load a hardware-generated `S_TRAP` payload into
  TTMP0/1, obey `STATUS.TRAP_EN`, reserve extra trap-handler SGPRs, and use
  `MODE.EXCP_EN` exception enables in the cited manual passage.
- Chapter 3.10.1 and 3.11 define sticky TRAPSTS fields, memory-violation
  sources, `TRAPSTS.mem_viol`, EXEC masking for buffer-to-LDS LDS address
  violations, and imprecise memory-violation saved-PC behavior in the cited manual passage.

Rocjitsu evidence:

- CDNA3 `S_TRAP` now has no generated branch or program-terminator flag and
  dispatches to the empty shared `execute_s_trap_sopp()` helper in the implementation.
- CDNA3 `S_GETREG_B32`/`S_SETREG_B32` only handle a small subset of HWREG IDs
  and default-log the rest; `S_GETREG_B32` handles ids 1, 4, 5, 6, and 7, and
  `S_SETREG_B32`/`S_SETREG_IMM32_B32` only write id 1 in the implementation.
- CDNA3 `V_CLREXCP` VOP1 and VOP3 bodies call shared helpers, but those
  helpers are empty in the implementation, and semantic derivation maps `V_CLREXCP` to `true_nop` in the code generator.
- Searching rocjitsu source and tests for `trapsts`, `TRAPSTS`, `mem_viol`,
  `MEM_VIOL`, `SAVECTX`, `ILLEGAL_INST`, `ADDR_WATCH`, `EXCP_CYCLE`, and
  `DP_RATE` finds no runtime trap-status or memory-violation model outside a
  semantic-fingerprint word list.

Impact:

Kernels that depend on trap enable state, trap status accumulation, memory
violation reporting, or `S_TRAP` NOP/trap behavior cannot be represented by the
current CDNA3 emulator path.

### CDNA3-RJ-055: TTMP privilege and CDNA3 TTMP launch initialization are missing

Manual evidence:

- Chapter 3.10 says all TTMP writes are privileged; outside a trap handler,
  writes are ignored and reads return zero in the cited manual passage.
- Chapter 3.13 says TTMP4/5 are initialized to zero, TTMP6/7 hold the dispatch
  packet address, TTMP8/9/10 hold dispatch grid dimensions, TTMP11 holds
  `wave_id_in_workgroup`, and other TTMPs are not initialized in the cited manual passage.

Rocjitsu evidence:

- CDNA3 scalar operand reads for encodings 108 through 123 read the wavefront
  SGPR file directly with no privilege check in the implementation.
- CDNA3 scalar destination writes for encodings 108 through 123 similarly write
  the wavefront SGPR file directly, and 64-bit TTMP-pair writes do the same for
  encodings 108 through 122, in the implementation.
- `CommandProcessor::init_wavefront_regs()` initializes user SGPRs, enabled
  workgroup-id SGPRs, a gfx12-only TTMP6/TTMP7/TTMP9 launch payload, and
  packed VGPR0 workitem IDs, but has no CDNA3 branch for TTMP4 through TTMP11
  in the implementation.

Impact:

Unprivileged CDNA3 code can observe and modify TTMP storage in rocjitsu instead
of seeing zero/ignored accesses, and kernels that use the CDNA3 TTMP launch
payload observe uninitialized simulator state.

### CDNA3-RJ-056: CDNA3 HW_ID and XCC_ID contents are incomplete

Manual evidence:

- Chapter 3.12 defines `HW_ID` bitfields for wave, SIMD, pipe, CU, shader
  engine, thread-group, VM, queue, state, and ME IDs, and defines `XCC_ID` bits
  3:0 in the cited manual passage.

Rocjitsu evidence:

- CDNA3 `S_GETREG_B32` treats HWREG id 4 as the low half of `wf.cu().id()` and
  id 5 as the high half, treats ids 6 and 7 as SGPR/VGPR allocation fields, and
  logs all other ids as unhandled in the implementation.
- There is no CDNA3 `S_GETREG_B32` case for the XML/manual `XCC_ID` register
  id 20, and the existing id-4 path does not pack the manual's HW_ID fields.

Impact:

`S_GETREG_B32` consumers see a CU-id placeholder instead of the CDNA3 `HW_ID`
layout and cannot read `XCC_ID`, which affects kernels or tests that inspect
placement identity.

### CDNA3-RJ-057: The TG_SIZE system SGPR launch payload is not initialized

Manual evidence:

- Chapter 3.13 says that when `tg_size_en` is enabled, SPI initializes a system
  SGPR containing `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0, 14'h0,
  work-group_size_in_waves[5:0]}` after the enabled workgroup-id SGPRs in the cited manual passage.

Rocjitsu evidence:

- `CommandProcessor::init_wavefront_regs()` writes user SGPRs, optional
  kernarg preload SGPRs, and enabled workgroup-id SGPRs, then proceeds to an
  RDNA4/gfx1250 TTMP payload and VGPR workitem IDs; there is no TG_SIZE system
  SGPR write in the CDNA3 path in the implementation.
- Searching rocjitsu for `TG_SIZE`, `tg_size`, `first_wave`,
  `wave_id_in_group`, and `workgroup_size_in_waves` finds no implementation;
  the only relevant descriptor plumbing found is workgroup-id enable handling.

Impact:

CDNA3 kernels compiled to consume the TG_SIZE system SGPR receive an
uninitialized or stale value from rocjitsu even though the manual defines a
precise launch-time payload.

### CDNA3-RJ-058: `S_RFE_B64` does not clear PRIV or branch to the return address

Manual evidence:

- Chapter 4.1 lists `S_RFE` as the trap-handler return instruction in the cited manual passage.
- The detailed `S_RFE_B64` definition says it may only be used within a trap
  handler, clears `WAVE_STATUS.PRIV`, and sets PC to the scalar source address
  in the cited manual passage.

Rocjitsu evidence:

- The CDNA3 generated `SRfeB64Sop1` constructor decodes the 64-bit source, but
  its execution delegates to `execute_s_rfe_b64_sop1()` in the implementation.
- The shared helper body is empty in the implementation.
- The CDNA3 `S_RFE_RESTORE_B64` form is decoded but throws
  `util::UnimplementedInst` in the implementation.

Impact:

Trap-handler return code cannot clear privileged mode or resume at the saved
PC in rocjitsu; `S_RFE_B64` behaves as a no-op and `S_RFE_RESTORE_B64` traps in
the emulator.

### CDNA3-RJ-059: Debug conditional branches never branch

Manual evidence:

- Chapter 4.2 says `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`, and
  `S_CBRANCH_CDBGSYS_AND_USER` branch based on the corresponding
  `COND_DBG_SYS` and `COND_DBG_USER` STATUS bits in the cited manual passage.

Rocjitsu evidence:

- CDNA3 constructors for the four debug conditional branches decode the label
  operand but do not set branch metadata flags, and each `execute_impl()`
  delegates to a shared helper in the implementation.
- The shared helpers for `S_CBRANCH_CDBGSYS`,
  `S_CBRANCH_CDBGSYS_AND_USER`, `S_CBRANCH_CDBGSYS_OR_USER`, and
  `S_CBRANCH_CDBGUSER` are empty in the implementation.
- Rocjitsu defines CDNA3/4 `COND_DBG_USER` and `COND_DBG_SYS` status accessors
  in the implementation, but the branch helpers do not read them.

Impact:

Debug-status conditional branches fall through regardless of STATUS, and
static branch analysis also lacks normal conditional-branch metadata for these
generated CDNA3 classes.

### CDNA3-RJ-060: Fork/join divergent control flow is not implemented

Manual evidence:

- Chapter 4.2 lists `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN` as conditional
  branch instructions for complex branching in the cited manual passage.
- The arbitrary divergent-control-flow section defines a six-deep CSP stack,
  `{EXEC, PC}` stack entries in SGPRs, pass/fail mask selection by bitcount,
  EXEC updates, branch target selection, and JOIN restoration in the cited manual passage.

Rocjitsu evidence:

- CDNA3 `S_CBRANCH_I_FORK` decodes the mask SGPR-pair and label but throws
  `util::UnimplementedInst` in `execute_impl()` in the implementation.
- CDNA3 `S_CBRANCH_G_FORK` decodes its two 64-bit sources and also throws
  `util::UnimplementedInst` in the implementation.
- CDNA3 `S_CBRANCH_JOIN` decodes the source but dispatches to an empty shared
  helper in the implementation.
- Searching rocjitsu for `CSP`, `control stack`, and fork/join helpers finds
  only generated decode fixtures and the unimplemented/empty execution paths,
  not a branch-stack state model.

Impact:

CDNA3 kernels that use compiler-emitted fork/join divergent control flow either
throw on the FORK instruction or fail to restore EXEC/PC at JOIN.

### CDNA3-RJ-061: Program-control status, priority, perf, trace, message, and wakeup side effects are mostly stubs

Manual evidence:

- Chapter 4.1 says `S_SETPRIO` modifies wave priority, `S_SLEEP` sleeps the
  wave, `S_SENDMSG` sends a host/upstream message, and `S_WAKEUP` wakes sleeping
  waves in the workgroup in the cited manual passage.
- Detailed SOPP definitions say `S_SETKILL` kills the wave when bit 0 is set,
  `S_SETHALT` sets or clears STATUS.HALT, `S_SETPRIO` updates the user priority
  bits, and `S_SENDMSGHALT` sends a message and halts in the cited manual passage.
- Detailed SOPP definitions also say `S_INCPERFLEVEL` and `S_DECPERFLEVEL`
  update performance counters and `S_TTRACEDATA` sends M0 to the thread-trace
  stream in the cited manual passage.
- Chapter 4.4 says `LGKM_CNT` increments by one for each `S_SENDMSG` and
  decrements when the message is sent out in the cited manual passage.

Rocjitsu evidence:

- CDNA3 `S_SETHALT`, `S_SETKILL`, `S_SETPRIO`, `S_SENDMSG`,
  `S_SENDMSGHALT`, `S_WAKEUP`, `S_INCPERFLEVEL`, `S_DECPERFLEVEL`, and
  `S_TTRACEDATA` dispatch to shared helpers, while `S_SLEEP` dispatches to a
  helper that only requests a functional yield. Representative generated calls
  are in the implementation.
- The shared helpers for `S_SENDMSG`, `S_SENDMSGHALT`, `S_SETHALT`,
  `S_SETKILL`, `S_SETPRIO`, and `S_WAKEUP` are empty in the implementation.
- The shared helpers for `S_DECPERFLEVEL`, `S_INCPERFLEVEL`, and
  `S_TTRACEDATA` are also empty in the implementation.
- `S_SLEEP` only calls `wf.cu().request_functional_yield()` and does not model
  the immediate value's sleep duration or `S_WAKEUP` interaction in the implementation.

Impact:

Control/status instructions used for halt, kill, priority, performance
counters, thread trace, host messages, message wait-counter accounting, and
sleep/wakeup scheduling have decode coverage but do not update the wave state,
side-channel state, wait-counter state, or scheduler behavior described by the
manual.

### CDNA3-RJ-128: Zero-target `S_SETPC_B64` and `S_SWAPPC_B64` hard-halt instead of branching

Manual evidence:

- `S_SETPC_B64` jumps to the byte address specified by the scalar source pair,
  with pseudocode `PC = S0.i64`, in the cited manual passage.
- `S_SWAPPC_B64` saves `PC + 4` and then jumps to the scalar input address,
  with pseudocode `PC = jump_addr.i64`, in the cited manual passage.
- The manual text for these default-form SOP1 PC instructions does not define a
  zero-address termination special case.

Rocjitsu evidence:

- Before the generated instruction body executes, `ComputeUnitCore` scans the
  mnemonic for `s_setpc` or `s_swappc`, reads the raw `SSRC0` scalar pair, and
  calls `active->halt()` when that target value is zero in the implementation.
- The generated `S_SETPC_B64` and `S_SWAPPC_B64` execute bodies otherwise branch
  through the scalar source pair, and `S_SWAPPC_B64` writes the link value, in the implementation.

Impact:

CDNA3 code that intentionally branches or calls through address zero is treated
as wave termination before the instruction semantics run. This changes control
flow and suppresses the `S_SWAPPC_B64` link write that the manual defines for a
zero target.

### CDNA3-RJ-062: Program-control tests are mostly decode fixtures

Rocjitsu evidence:

- Generated CDNA3 encoding fixtures include `s_rfe_b64`,
  `s_cbranch_join`, debug conditional branches, fork forms, `s_sethalt`,
  `s_setkill`, `s_setprio`, `s_sendmsg`, `s_sendmsghalt`, and `s_wakeup` in the decode fixtures.
- The instruction execution harness explicitly excludes several control
  instructions, including `s_sethalt`, `s_sendmsg`, `s_sendmsghalt`, and
  `s_rfe`, in the relevant tests.
- Searching the relevant tests found no behavior tests for CDNA3 debug
  conditional branch predicates, fork/join stack effects, `S_RFE` return
  behavior, or HALT/kill/message/wakeup side effects.

Impact:

The current tests can confirm these opcodes decode, but they would not catch
the no-op/unimplemented execution paths recorded above.

### CDNA3-RJ-063: `S_BARRIER` does not expose `STATUS.IN_BARRIER`

Manual evidence:

- Chapter 3 defines `STATUS.IN_BARRIER` bit 12 as "Wavefront is waiting at a
  barrier" in the cited manual passage.
- The detailed `S_BARRIER` definition says the wave waits at a threadgroup
  barrier until the release conditions are satisfied in the cited manual passage.

Rocjitsu evidence:

- The CDNA status wrapper defines `IN_BARRIER` as bit 12 in the implementation.
- CDNA3 `S_GETREG_B32` reads raw STATUS through `wf.status_raw()` for HWREG id
  1 in the implementation, and the wavefront stores raw status independently of scheduler state
  in the implementation.
- `execute_s_barrier_sopp()` only sets `WfState::BARRIER` in the implementation; it does not set bit 12 while the wave is waiting or clear it
  when the CU releases the barrier.
- The CU release path changes only scheduler state from `BARRIER` back to
  `RUNNING` in the implementation.

Impact:

Barrier scheduling can block and release waves, but shader/debug code that
observes raw STATUS through HWREG cannot see the architectural
`IN_BARRIER` bit while a wave is waiting.

### CDNA3-RJ-064: Workgroup size limits are not validated

Manual evidence:

- Chapter 4.3 says up to 16 wavefronts or 1024 work-items can be combined into
  a workgroup in the cited manual passage.

Rocjitsu evidence:

- The AQL dispatch path computes `wg_size` directly from packet workgroup
  dimensions, derives `wfs_per_wg = (wg_size + wave_size - 1) / wave_size`, and
  stores it in the dispatch entry in the implementation.
- The workgroup admission check only tests available wavefront slots, SGPR
  blocks, VGPR blocks, and LDS capacity in the implementation.
- A focused search found no architectural validation rejecting workgroups above
  16 waves or 1024 work-items.

Impact:

Rocjitsu can accept a dispatch shape that exceeds CDNA3's documented per
workgroup size limit if enough simulator resources are configured, which can
change residency, barrier membership, and launch-state behavior for malformed
or stress dispatches.

### CDNA3-RJ-065: Barrier tests cover only the basic all-live release case

Rocjitsu evidence:

- `HookOrderingTest.BarrierTwoWaves` runs `{S_BARRIER, S_ENDPGM}` with two
  waves in one workgroup and asserts one `BARRIER_RESOLVED` event plus two
  dispatched and halted wavefronts in the relevant tests.
- The instruction execution harness excludes `s_barrier` from its generic
  execution coverage in the relevant tests.
- Searching the tests found no behavior cases for early-terminated peers,
  `STATUS.IN_BARRIER` observation, wait counters remaining outstanding across
  barrier issue, single-wave immediate release, or oversized workgroup
  rejection.

Impact:

The current behavior test would catch a disconnected baseline barrier-release
hook, but it would not catch the `IN_BARRIER` state gap, size-limit gap, or
several manual-only release/issuing conditions.

### CDNA3-RJ-066: Scalar-memory `LGKM_CNT` accounting ignores dword counts

Manual evidence:

- Chapter 4.4 says `LGKM_CNT` is incremented by the dword count for scalar
  memory reads, with one count for one-dword loads and two counts for
  two-dword-or-larger loads; `S_MEMTIME` counts like `S_LOAD_DWORDX2` in the cited manual passage.
- The same section says `LGKM_CNT` decrements once for each dword returned from
  the data cache for SMEM in the cited manual passage.
- Chapter 4.4 also states scalar-memory reads can return out of order, so only
  `S_WAITCNT 0` is legitimate for such dependencies in the cited manual passage.

Rocjitsu evidence:

- `ScalarMemState` carries `num_dwords`, but only one
  `wait_counter_type = WaitCounterType::LGKMCNT` value in the implementation.
- The generated CDNA3 SMEM load bodies do set `num_dwords`, for example
  `S_LOAD_DWORD` uses `1` and `S_LOAD_DWORDX2` uses `2` in the implementation.
- `MemoryPipeline::issue()` increments exactly one wait counter for the
  instruction in the implementation.
- `ScalarMemPipeline::complete_access()` writes every returned dword to SGPRs,
  but completion returns once and the deferred callback releases that same
  single counter in the implementation.

Impact:

Functional execution writes the right SGPR payload, but outstanding
`LGKM_CNT` depth does not match CDNA3 for multi-dword SMEM loads or
`S_MEMTIME`. Thresholds such as `lgkmcnt(1)` can therefore unblock differently
from hardware in timing/deferred paths, and the model cannot represent the
manual's per-dword SMEM return accounting.

### CDNA3-RJ-067: GWS `EXP_CNT` producer behavior is not modeled in production

Manual evidence:

- Chapter 4.4 says `LGKM_CNT` is incremented by one for each GWS instruction in the cited manual passage.
- The same section defines `EXP_CNT` as the VGPR-export count for GWS,
  incremented when a GWS instruction issues from the wavefront buffer and
  decremented when the last GWS cycle is granted/executed and VGPRs have been
  read out in the cited manual passage.

Rocjitsu evidence:

- The generated CDNA3 GWS instruction classes decode the GWS opcodes, but all
  six `execute_impl()` bodies immediately throw `util::UnimplementedInst` in the implementation.
- A targeted search for `WaitCounterType::EXPCNT` found only the wait-counter
  primitive and unit/infrastructure tests, not a production instruction path
  that issues or retires `EXP_CNT`.
- Existing broader GWS behavior is already tracked in `CDNA3-RJ-041`; this
  item records the specific Chapter 4.4 wait-counter consequence.

Impact:

Rocjitsu can decode GWS instructions, but it cannot model the CDNA3
`LGKM_CNT`/`EXP_CNT` effects that make GWS VGPR sources safe to overwrite only
after the export count retires.

### CDNA3-RJ-068: Wait-counter tests miss CDNA3 producer-accounting edge cases

Rocjitsu evidence:

- The relevant tests covers primitive increments/decrements and
  saturation for individual counter types, including `EXPCNT`, in the relevant tests.
- The relevant tests manually seeds an RDNA3 `EXPCNT` value and
  checks `S_WAITCNT_EXPCNT` threshold behavior in the relevant tests.
- Searching the adjacent tests found no focused CDNA3 producer cases for
  multi-dword SMEM `LGKM_CNT` accounting, `S_MEMTIME` counting like
  `S_LOAD_DWORDX2`, `S_SENDMSG` message-counter accounting, GWS `EXP_CNT`
  issue/retire behavior, or producer-driven FLAT dual-counter accounting beyond
  the already-recorded FLAT gap
  `CDNA3-RJ-034`.

Impact:

The tests prove that counter storage and manual threshold waits work in
isolation, but not that CDNA3 instructions feed those counters according to the
manual's producer-specific rules.

### CDNA3-RJ-069: Manual NOP wait-state hazards are not modeled or diagnosed

Manual evidence:

- Chapter 4.5 says hardware does not check several dependency classes, so the
  shader must resolve them by inserting NOPs or independent instructions in the cited manual passage.
- Table 11 lists required waits for `S_SETREG`/`S_GETREG`, `MODE.VSKIP`,
  VALU-produced VCC/EXEC/SGPR/VGPR values, lane-select consumers,
  `V_DIV_FMAS`, wide store/atomic write-data hazards, M0 consumers,
  TRAPSTS/RFE, DPP, OPSEL/SDWA bit-position changes, and trans-op consumers in the cited manual passage.
- Table 12 defines the trans-op set used by the Table 11 trans-op wait row in the cited manual passage.
- The detailed `S_ICACHE_INV` row says 16 separate `S_NOP` instructions or a
  jump/branch must follow instruction-cache invalidation in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 `S_NOP` constructs a SOPP instruction and delegates to the
  shared helper in the implementation, but the shared helper is an empty function in the implementation.
- The functional memory pipeline documents synchronous completion and has a
  no-op `tick()` in functional mode in the implementation;
  `ExecComputeUnit` also leaves CLOCKED mode as a TODO in the implementation.
- The existing DBT `HazardTracker` is explicitly a GFX12 `s_delay_alu`
  auto-insertion helper in the implementation; it
  emits RDNA4 `S_DELAY_ALU` words in the implementation,
  not CDNA3 Table 11 `S_NOP` timing or runtime diagnostics.
- `S_ICACHE_INV` dispatches to an empty shared helper in the implementation, with no branch-or-16-`S_NOP` spacing validation.
- Some rows overlap existing behavior gaps, such as consecutive `S_SETREG`
  hazard handling in `CDNA3-RJ-049` and unimplemented `S_SETVSKIP` behavior in
  `CDNA3-RJ-052`; this finding records the broader absence of a CDNA3 Table 11
  wait-state/scoreboard model.

Impact:

Rocjitsu executes most producer/consumer pairs with fully updated architectural
state at instruction boundaries. A kernel that omits required CDNA3 NOPs,
including instruction-cache invalidation spacing, can therefore appear correct
under emulation even when hardware requires padding or independent instructions
to avoid races.

### CDNA3-RJ-070: Tests do not cover CDNA3 Table 11 wait-state hazards

Rocjitsu evidence:

- Decode smoke tests only assert that the shared `S_NOP` encoding decodes to
  `s_nop` for CDNA3 and other ISAs in the relevant tests.
- The generic instruction execution harness skips `s_nop`, `s_getreg`,
  `s_setreg`, memory, wait, barrier, branch, and message/control instructions
  in the relevant tests.
- DBT tests cover GFX12/RDNA4 `s_delay_alu` insertion paths, but the audited
  search found no CDNA3 tests that compare programs with and without the Table
  11 NOP counts or assert warnings/diagnostics for missing `S_NOP` hazards.

Impact:

Current tests cover `S_NOP` decoding and several unrelated functional
behaviors, but they would not catch a missing CDNA3 Table 11 hazard model or an
emulation path that silently accepts hardware-racy instruction sequences.

### CDNA3-RJ-071: VOP3 floating output modifiers ignore MODE denorm/IEEE gating

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point output modifiers are ignored when output
  denormals are enabled or when `MODE.IEEE` is 1; when output denormals are
  disabled, applying an output modifier also flushes denormals to zero and
  flushes `-0` to `+0` in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state through `mode_raw()` and `set_mode_raw()` in the implementation.
- Representative generated VOP3 FP helpers apply `OMOD` and `CLAMP`
  unconditionally: `execute_v_add_f32_vop3()` in the implementation, `execute_v_add_f64_vop3()`,
  `execute_v_mul_f32_vop3()`, and
  `execute_v_mul_f64_vop3()`.
- The shared `S_DENORM_MODE` helper is currently empty in the implementation, and the audited VOP3 helper bodies do not read `wf.mode_raw()`
  before applying the modifier or clamp.
- Existing SIMD correctness tests sweep modifier bits against the generated
  direct modifier model, for example
  the relevant tests, but they do not vary `MODE.IEEE` or `MODE.FP_DENORM`.

Impact:

CDNA3 VOP3 FP instructions can produce modified, clamped, denormal, or `-0`
results in rocjitsu even in MODE states where the manual says the modifier must
be ignored or the result must be flushed.

### CDNA3-RJ-072: VALU source validation allows manual-disallowed extra scalar sources

Manual evidence:

- Chapter 6.2.1 says VALU instructions can read at most one SGPR per instruction
  and can use at most one literal, only when neither SGPR nor M0 is used. It also
  says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC and therefore cannot use
  an additional SGPR or literal in the cited manual passage.
- The detailed definitions clarify that VOP3 `V_CNDMASK_B32` may take the VCC
  source from scalar GPR `S2`, and VOP3 `V_ADDC_CO_U32`/`V_SUBB_CO_U32` may take
  the VCC source from the SGPR pair at `S2.u`, in the cited manual passage. That explicit S2 source still consumes the scalar-source
  budget, so `S0`/`S1` cannot also be additional scalar or literal sources.

Rocjitsu evidence:

- Representative generated VOP3 `V_ADD_F32` independently constructs `src0` as
  `OPR_SRC_NOLIT` and `src1` as `OPR_SRC_SIMPLE` in the implementation; both operand classes include scalar-source encodings.
- `Operand::read_lane()` resolves each non-VGPR VALU source independently
  through `resolve_src_scalar()` in the implementation, and `resolve_src_scalar()` can read SGPRs, VCC, M0, EXEC, inline
  constants, and helper predicates in the implementation.
- Carry-consuming VOP3 forms such as `V_ADDC_CO_U32` construct broad
  `SRC0`/`SRC1` operands plus an explicit scalar-pair `SRC2` carry-in in the implementation, with no adjacent validation that the explicit `SRC2` scalar source
  disallows extra scalar/literal sources in `SRC0` or `SRC1`.

Impact:

This is a legality/diagnostic gap rather than proof that a legal encoding
executes incorrectly: rocjitsu can decode and execute source combinations that
the CDNA3 manual says the hardware contract disallows.

### CDNA3-RJ-073: VALU FP round/denorm modes and V_DOT2 denorm flushing are not modeled

Manual evidence:

- Chapter 6.4 says the shader program controls floating-point rounding and
  denormal input/result handling through MODE fields set by `S_SETREG`, with
  separate single-precision and double/half-precision fields in the cited manual passage.
- The same section says floating-point `V_DOT2` instructions do not support
  denormal or rounding modes and flush input and output denormals in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` can store raw MODE state in the implementation, but
  CDNA3 `S_SETREG_B32` and `S_SETREG_IMM32_B32` only handle HWREG id 1
  (`STATUS`) in the audited helpers in the implementation, so the Chapter 6.4 `S_SETREG` path cannot
  write MODE.
- The only `mode_raw()` consumers found in the audited rocjitsu tree are raw
  storage/GPR-indexing paths, not VALU FP arithmetic. Representative FP VOP3
  helpers compute with host arithmetic and direct modifier handling, for
  example `execute_v_add_f32_vop3()` in the implementation and `execute_v_add_f64_vop3()`.
- Floating `V_DOT2` helpers widen FP16 inputs with generic `util::f16_to_f32`,
  use ordinary host multiply/add, and optionally clamp; no explicit input/output
  denormal flush or MODE-independent V_DOT2 rule is visible in
  `execute_v_dot2_f32_f16_vop3p()` in the implementation, the `V_DOT2C` paths, or the SIMD path
  in the implementation.
- The shared `S_DENORM_MODE` and `S_ROUND_MODE` execution helpers are empty in the implementation; those are
  not the CDNA3 Chapter 6.4 route, but they show there is no alternate shared
  MODE update path for these fields.

Impact:

Ordinary CDNA3 VALU FP arithmetic is evaluated under host default rounding and
denormal behavior instead of the MODE fields, and floating `V_DOT2` does not
statically show the manual's required flush-in/flush-out denormal behavior.

### CDNA3-RJ-074: ALU clamp non-FP semantics are incomplete

Manual evidence:

- Chapter 6.5 says `V_CMP` clamp requests signaling compare behavior on FP
  exceptions, integer operations clamp to the largest/smallest representable
  value, and floating-point operations clamp to `[0.0, 1.0]` in the cited manual passage.
- Chapter 12.9.1 repeats the VOPC/VOP3A compare-specific rule: `CLAMP=1`
  signals an exception when either compare input is NaN, while `CLAMP=0` does
  not, in the cited manual passage.
- CDNA3 VOP3/VOP3B format tables expose `CLMP` as a result/output clamp in the cited manual passage, and
  the opcode tables include integer forms such as `V_ADD_U32` and `V_ADD_I32`
  in the cited manual passage.

Rocjitsu evidence:

- Representative integer VOP3 helpers ignore `inst.inst_.clamp` and execute
  wrapping arithmetic directly: `execute_v_add_i32_vop3()` in the implementation, `execute_v_add_u32_vop3()`, and
  `execute_v_sub_i32_vop3()`.
- The SIMD VOP3 integer fast-path comment acknowledges that clamp on an integer
  op means saturation, but says the wrap-around/bitwise twins do not request it
  in the implementation; the VOP3 integer-compare path similarly says clamp is unused
  in the implementation.
- Representative VOP3 compare helpers compute and store the condition result
  without inspecting `inst.inst_.clamp` or setting exception/signaling state, for
  example `execute_v_cmp_eq_f32_vop3()` in the implementation and `execute_v_cmp_lg_f32_vop3()`.

Impact:

Programs that encode `CLMP=1` on ordinary integer VOP3 arithmetic will see
wrap-around results rather than CDNA3's documented saturation, and `V_CMP`
encodings cannot request the documented signaling-compare behavior through the
clamp bit.

### CDNA3-RJ-075: VGPR indexing uses the wrong M0 layout and cannot honor source-role masks

Manual evidence:

- Chapter 6.6 defines VGPR indexing as a MODE-enabled M0 index for selected VALU
  VGPR sources or destinations in the cited manual passage.
- Table 27 says `S_SET_GPR_IDX_ON` and `S_SET_GPR_IDX_MODE` store the mode in
  `M0[15:12]`, while `S_SET_GPR_IDX_IDX` stores the index in `M0[7:0]`, in the cited manual passage.
- The prose defines `M0[15]` as destination enable, `M0[14]` as source-2
  enable, `M0[13]` as source-1 enable, and `M0[12]` as source-0 enable, limits
  indexing to VGPR operands, and makes out-of-range indexed VGPRs illegal in the cited manual passage.
- Section 6.6.2 gives instruction-specific source/destination role remapping for
  readlane, writelane, MAC/MAD, shift, `v_cvt_pkaccum`, and SDWA preserve forms
  in the cited manual passage.

Rocjitsu evidence:

- The shared `S_SET_GPR_IDX_MODE` and `S_SET_GPR_IDX_ON` helpers store the mode
  with `<< 8` and masks around bits `[11:8]`, not `M0[15:12]`, in the implementation.
- `Wavefront::gpr_idx_mode()` likewise reads `(m0_ >> 8) & 0xF`, and
  `apply_gpr_idx()` treats any low three source bits as applying to every source
  operand rather than src0/src1/src2 separately in the implementation.
- CDNA3 scalar fallback operand reads/writes pass only an `is_dst` boolean into
  `apply_gpr_idx()`, so the operand layer has no information about whether a
  VGPR read is src0, src1, or src2; representative paths are in the implementation.
- The generic SIMD operand helpers use the same `apply_gpr_idx()` helper with
  only source-versus-destination information in the implementation,; true16 write glue does the
  same in the implementation.

Impact:

CDNA3 programs that use `S_SET_GPR_IDX_*` according to the manual will write the
selector bits where rocjitsu does not read them, while programs that happen to
populate `M0[11:8]` can index the wrong source operands because rocjitsu lacks
per-source role mapping.

### CDNA3-RJ-076: Packed FP8/BF8-to-F32 converts do not validate even destination bases

Manual evidence:

- Table 31 says `CVT_PK_F32_FP8` and `CVT_PK_F32_BF8` write `dst,dst+1` and
  require an even destination VGPR in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 VOP1 constructors model `VCvtPkF32Fp8Vop1` and
  `VCvtPkF32Bf8Vop1` with 64-bit `OPR_VGPR` destinations in the implementation.
- Their execution writes through `RegisterAccess::write_lane64()` in the implementation.
- The CDNA3 operand and register-access layers turn a 64-bit VGPR operand into a
  two-register region but do not validate even alignment:
  the implementation.

Impact:

An odd-`VDST` encoding for these packed widening converts will execute as a
write to the odd/even+1 pair in rocjitsu instead of being rejected or diagnosed
as an illegal destination base.

### CDNA3-RJ-077: F32-to-FP8/BF8 VOP3 converts ignore supported source modifiers

Manual evidence:

- Table 31 says `CVT_PK_FP8_F32`, `CVT_PK_BF8_F32`, `CVT_SR_FP8_F32`, and
  `CVT_SR_BF8_F32` support `NEG` and `ABS` source modifiers while ignoring
  `CLAMP` and `OMOD` in the cited manual passage.

Rocjitsu evidence:

- `VCvtPkFp8F32Vop3::execute_impl()` and `VCvtPkBf8F32Vop3::execute_impl()`
  read `src0` and `src1` as raw F32 values and convert them directly, with no
  `inst_.abs` or `inst_.neg` handling, in the implementation.
- `VCvtSrFp8F32Vop3::execute_impl()` and `VCvtSrBf8F32Vop3::execute_impl()`
  likewise read raw `src0`, use `src1` as the seed, and merge the result byte
  without applying source `ABS` or `NEG` in the implementation.
- Other generated VOP3 conversion bodies do apply source modifiers locally, for
  example `VCvtF32F16Vop3::execute_impl()` applies `inst_.abs` and `inst_.neg`
  after reading the selected half in the implementation.

Impact:

CDNA3 FP8/BF8 narrow conversions with negative inputs and `ABS`, or positive
inputs and `NEG`, will produce results for the unmodified source value instead
of the modifier-adjusted value required by the manual.

### CDNA3-RJ-129: `V_CVT_PK_{FP8,BF8}_F32` low-half writes zero the preserved half

Manual evidence:

- Chapter 12.11 definitions 674 and 675 say `V_CVT_PK_FP8_F32` and
  `V_CVT_PK_BF8_F32` write the packed two-byte result into the selected
  16-bit half using `OPSEL[3]`, while preserving the other half of `D0`, in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 execution for both packed F32-to-FP8/BF8 forms reads the old
  destination through `implicit_uses`, but then calls
  `write_vop3_true16_dst(..., true)` in the implementation.
- The shared true16 destination helper interprets that final `true` as the
  CDNA low-half zero-high rule for CDNA1 through CDNA4 and, for low-half
  writes, returns `src_half` instead of merging with the old high 16 bits in the implementation.
- Existing gfx1250 tests assert the analogous packed FP8 low/high destination
  preservation behavior in the relevant tests, but no CDNA3 regression pins the Chapter 12.11 preservation
  rule for definitions 674 and 675.

Impact:

On CDNA3, a low-half `V_CVT_PK_FP8_F32` or `V_CVT_PK_BF8_F32` write can clear
destination bits 31:16 instead of preserving them, corrupting adjacent packed
data. The stochastic byte-write forms do their own read/mask/write merge and
are not covered by this specific bug.

### CDNA3-RJ-130: Stochastic FP8/BF8 helpers are not covered against the Chapter 12.11 seed-bit formula

Manual evidence:

- Chapter 12.11 definitions 676 and 677 say `V_CVT_SR_FP8_F32` adds
  `S1[31:12]` into the F32 mantissa before conversion, while
  `V_CVT_SR_BF8_F32` uses `S1[31:11]`, in the cited manual passage.
- Chapter 7 describes the same stochastic conversion family as adding the
  stochastic value and then truncating in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 stochastic execution delegates the conversion to
  `util::f32_to_fp8_e4m3_fnuz_sr()` and
  `util::f32_to_bf8_e5m2_fnuz_sr()` in the implementation, then masks the selected destination byte manually.
- The helper implementations add seed-derived bits inside the conversion
  helper, including `seed >> 12` for normal FP8 values and `seed >> 11` for
  normal BF8 values, with separate subnormal paths in the implementation.
- Existing helper/HIP conversion tests cover selected fixed outputs and finite
  conversion plumbing, but no CDNA3 end-to-end case asserts the Chapter 12.11
  FP8 `S1[31:12]` and BF8 `S1[31:11]` seed formulas across normal and
  subnormal inputs.

Impact:

The current helper structure may be equivalent to the manual's
add-random-then-truncate description for ordinary cases, but that equivalence
is not captured by a regression or hardware/oracle case. A future helper change
could use the wrong seed-bit range while the existing CDNA3 tests still pass.

### CDNA3-RJ-078: FP8/BF8 widening VOP1 SDWA converts honor ignored destination controls

Manual evidence:

- Table 31 says `CVT_F32_FP8`, `CVT_F32_BF8`, `CVT_PK_F32_FP8`, and
  `CVT_PK_F32_BF8` use SDWA source selection and ignore `ABS`, `NEG`, and
  `SEXT` in the cited manual passage.
- The section 7.2 prose says VOP1 8-bit-format converts use the SDWA word only
  for the `SRC0` VGPR and `SRC0_SELECT`, and that the other SDWA fields are
  ignored in the cited manual passage.

Rocjitsu evidence:

- The generated CDNA3 `VCvtF32Fp8Vop1` and `VCvtF32Bf8Vop1` SDWA paths copy
  generic SDWA source and destination fields, then merge the computed F32 result
  according to `sdwa_dst_sel_`/`sdwa_dst_unused_` in the implementation.
- The packed widening forms use the same generic SDWA source handling and
  destination merge controls around their 64-bit `VDST` writes in the implementation.

Impact:

SDWA encodings that set `DST_SEL` or `DST_UNUSED` on these widening converts
can partially merge or preserve destination bits in rocjitsu, even though the
manual says those SDWA controls are ignored and the conversion should be
governed only by `SRC0_SELECT`.

### CDNA3-RJ-079: Device-memory consistency and acknowledgment behavior is not represented

Manual evidence:

- Section 2.3 describes the CDNA device-memory hierarchy, cache-less loads,
  load-clause overlap caching, write-combining, atomic pre-op return storage,
  write-confirmation acknowledgments, relaxed consistency, per-PE/per-channel
  scatter-write ordering, and acknowledgment/fence use in the cited manual passage.

Rocjitsu evidence:

- Ordinary scalar memory accesses issue direct L1 load/store operations and
  complete load writeback with no modeled acknowledgment, per-channel order
  state, or relaxed-consistency fence state in the implementation.
- Vector atomics perform lane-by-lane L2 read-modify-write and store old values
  in per-lane response data for returned atomics, but do not model the manual's
  return-address write-confirmation acknowledgment path in the implementation.
- `L2Cache` documents a functional cache model: ordinary read/write paths are
  not thread-safe, only `atomic_rmw()` is mutex-protected, and atomics serialize
  under cache-line stripes rather than PE/channel ordering state in the implementation.
- AQL acquire-fence handling only invalidates modeled caches for agent/system
  scope in the implementation.

Impact:

rocjitsu can still serve as a functional memory-value oracle for many kernels,
but it cannot validate the Chapter 2.3 relaxed-consistency, acknowledgment, or
per-PE/per-channel ordering contracts.

### CDNA3-RJ-080: Buffer floating atomics miss L2 FP numeric and packed-lane rules

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and can be
  issued as LDS, Buffer, Flat, Global, and Scratch instructions in the cited manual passage.
- Float atomic ADD opcodes use RNE, and Table 52 defines L2 denormal behavior:
  packed F16/BF16 and F64 add/min/max do not flush denorms while F32 add
  flushes denorms in the cited manual passage.
- Chapter 9.2.3 defines SNaN quieting, NaN propagation/selection, signed-zero
  ordering, compare-store equality, and add edge cases in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 MUBUF floating atomics lower to the generic memory-pipeline
  `AtomicOp::FADD`, `FMIN`, and `FMAX` operations: representative F32, packed
  F16, F64 add, and F64 min/max paths are in the implementation.
- `BUFFER_ATOMIC_PK_ADD_F16` sets `elem_size = 4` and `AtomicOp::FADD`, the
  same scalar 32-bit path as `BUFFER_ATOMIC_ADD_F32`, in the implementation.
- The semantic derivation table also labels packed FP atomics as "treated as
  32-bit fadd for now" in the code generator.
- The shared L2 atomic executor treats 4-byte floating atomics as one host
  `float` and 8-byte floating atomics as one host `double`, using ordinary
  addition plus `std::fmin`/`std::fmax` in the implementation.
- `VectorMemState` carries the `AtomicOp` enum, element size, and dataflow
  fields, but not the floating-point subtype, packed-lane mode, or
  denormal/rounding policy needed by Chapter 9.2, in the implementation.

Impact:

F32/F64 buffer atomics have a coarse functional model, but rocjitsu does not
enforce CDNA3's L2 floating-atomic denormal, RNE, or NaN/signed-zero rules.
Packed F16 buffer atomics are not type-correct because their two lanes execute
as one scalar F32 operation. Existing `CDNA3-RJ-035` and `CDNA3-RJ-040` cover
the related flat/global/scratch and LDS floating-atomic paths.

### CDNA3-RJ-081: `S_ABSDIFF_I32` uses mathematical absolute difference instead of the wrapped SOP2 definition

Manual/XML evidence:

- Chapter 12.1 defines `S_ABSDIFF_I32` as a 32-bit `D0.i32 = S0.i32 - S1.i32`,
  followed by `D0.i32 = -D0.i32` only when that 32-bit result is negative, and
  gives overflow examples such as
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff` in the cited manual passage.
- The XML only summarizes the operation as "absolute value of difference" in the machine-readable ISA XML; the wrapped-subtract edge
  examples are recorded as `CDNA3-XML-055`.

Rocjitsu evidence:

- Generated `SAbsdiffI32Sop2::execute_impl()` delegates to the shared helper in the implementation.
- The shared helper promotes both signed inputs to `int64_t` and computes
  `a > b ? a - b : b - a` before truncating to 32 bits in the implementation.
- The generator lowering emits the same wide absolute-difference algorithm in the code generator, so
  regeneration preserves this behavior.

Impact:

The manual's overflow examples do not all execute correctly. For example,
`S_ABSDIFF_I32(0x80000000, 0x00000001)` should produce `0x7fffffff`, but the
wide absolute-difference implementation produces `0x80000001`.

### CDNA3-RJ-082: Chapter 12.1 SOP2 tests miss detailed semantic edge contracts

Evidence:

- The generated CDNA3 encoding fixture includes a single `s_absdiff_i32`
  encoding smoke entry in the decode fixtures.
- The Python semantic-derivation test for `S_ABSDIFF_I32` only checks that the
  derived block contains an `ABSDIFF` call in the codegen tests.
- The Python `S_BFE_U32` test only checks for a BFE call and SCC write in the codegen tests.
- This slice found no targeted runtime golden tests for the `S_ABSDIFF_I32`
  overflow examples, BFM/BFE width and offset edge cases, shifted-add carry
  cases, or scalar-pack literal/high-half selection.

Impact:

Full-suite passing can miss the concrete `S_ABSDIFF_I32` divergence above and
similar SOP2 definition-level regressions in bitfield, shifted-add, and pack
rows.

### CDNA3-RJ-083: `S_CALL_B64` direct calls are flagged as `INDIRECT_CALL`

Manual/XML evidence:

- Chapter 12.2 defines `S_CALL_B64` as `D0.i64 = PC + 4` and
  `PC = PC + signext(SIMM16.i16 * 4) + 4`, and says the instruction implements
  a short subroutine call that must be 4 bytes in the cited manual passage.
- The XML describes the target as a constant offset relative to the current PC
  and marks `IsIndirectBranch` false in the machine-readable ISA XML.

Rocjitsu evidence:

- CDNA3 `SCallB64Sopk` exposes a PC-relative `branch_offset_bytes()` and
  executes by writing `PC + size_` to `SDST` and adding the signed SIMM16
  instruction-count offset to `wf.pc` in the implementation.
- The same constructor sets `flags_ |= INDIRECT_CALL` in the implementation, even though
  `Instruction` documents that flag as an indirect call whose target comes from
  a register in the implementation.
- Some internal clients compensate by treating `INDIRECT_CALL` plus a present
  `branch_offset_bytes()` as a direct call in the implementation, but flag-only clients such as the probe-callable helper still
  classify calls from the flag alone in the implementation.

Impact:

Execution and relocation have a direct PC-relative offset available, but the
public instruction flag conflates direct PC-relative calls with register-target
indirect calls. Flag-only analysis, instrumentation, or reporting clients can
misclassify `S_CALL_B64` relative to the manual/XML branch metadata.

### CDNA3-RJ-084: M0-relative scalar moves truncate M0 and B64 forms scale the offset

Manual/XML evidence:

- Chapter 12.3 defines `S_MOVRELS_B32/B64` as `addr = SRC0.u32; addr +=
  M0.u32[31:0]`, then reads from `SGPR[addr]`; the B64 form also says both the
  index in M0 and the source address must be even in the cited manual passage.
- `S_MOVRELD_B32/B64` uses the same full-32-bit `addr += M0.u32[31:0]` rule for
  the instruction destination field, with even M0 plus an even destination
  address required for B64 in the cited manual passage.
- The XML exposes M0 as an implicit operand on `S_MOVRELS_*` and
  `S_MOVRELD_*` in the machine-readable ISA XML, while the
  detailed formula omission is recorded in `CDNA3-XML-057`.

Rocjitsu evidence:

- All four relative-move helpers first truncate the index with
  `index = wf.m0() & 0xFFu`, so values outside the low byte are ignored before
  address formation in the implementation.
- The B64 helpers then compute `width_words = size_bits / 32` and use
  `base + index * width_words`; for the 64-bit forms this is `base + 2 * M0` in the implementation.
- Neither path checks the manual's even-M0 requirement before forming the
  64-bit SGPR pair.

Impact:

For any relative move, an M0 value such as `0x100` should participate in the
manual's full-width `addr += M0.u32[31:0]` calculation, but rocjitsu uses zero.
For B64, a legal even M0 value such as 2 means `s_movrels_b64 s[10:11],
s[2:3]` should read from `s[4:5]`, but rocjitsu reads from `s[6:7]`. Odd M0
values are also silently remapped by the doubled formula instead of being
rejected or diagnosed as invalid 64-bit relative moves.

### CDNA3-RJ-085: Default-only SOP1 PC forms are accepted as 8-byte literal encodings

Manual/XML evidence:

- `S_GETPC_B64` saves `PC + 4` and its note says the instruction must be 4 bytes
  in the cited manual passage.
- `S_SETPC_B64` jumps to an address specified in a scalar register in the cited manual passage.
- `S_SWAPPC_B64` saves `PC + 4`, jumps to the scalar input, and its note says
  the instruction must be 4 bytes in the cited manual passage.
- The CDNA3 XML provides only default `ENC_SOP1` encodings for
  `S_GETPC_B64`, `S_SETPC_B64`, and `S_SWAPPC_B64`, with no
  `SOP1_INST_LITERAL` alternatives, in the machine-readable ISA XML.

Rocjitsu evidence:

- The generated CDNA3 `Sop1` base class treats every SOP1 instruction with
  `SSRC0 == 255` as a 64-bit instruction by adding an extension dword to
  `size_` in the implementation.
- `SGetpcB64Sop1::execute_impl()` writes `wf.pc + size_`, so a reserved
  `SSRC0 == 255` encoding would report `PC + 8` instead of the manual's
  required `PC + 4` in the implementation.
- `SSetpcB64Sop1` and `SSwappcB64Sop1` rewrite `SSRC0 == 255` into an
  `OPR_SIMM32` operand and then branch through it in the implementation,
  even though the XML/manual form is a scalar-register target.

Impact:

Reserved/default-only SOP1 PC encodings can consume the following dword as a
literal extension, change instruction length, and alter PC/link values or branch
targets. This can desynchronize decode streams and hide illegal encodings behind
apparently valid `S_GETPC_B64`, `S_SETPC_B64`, or `S_SWAPPC_B64` instructions.

### CDNA3-RJ-086: `S_SET_GPR_IDX_ON` treats operand 1 as a literal-capable scalar source

Manual/oracle evidence:

- The detailed `S_SET_GPR_IDX_ON` definition says vector operations use M0 for
  relative GPR addressing, source 0 supplies the index, and the raw bits of the
  `SRC1` field set the enable bits; the pseudocode writes `M0[15:12]` from
  `SRC1[3:0]` and says this is direct raw-field content in the cited manual passage.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942` accepts a 4-bit mode operand
  and a source-0 literal, producing `s_set_gpr_idx_on s0, 15` as a one-dword
  encoding and `s_set_gpr_idx_on 0x12345678, 15` as an 8-byte source-0 literal
  encoding, but rejects `s_set_gpr_idx_on s0, 16` and
  `s_set_gpr_idx_on s0, 0x12345678`.

Rocjitsu evidence:

- The generic CDNA3 `Sopc` base treats any `ssrc1 == 255` as a literal form and
  increases instruction size in the implementation.
- `SSetGprIdxOnSopc` initially declares operand 1 as `OPR_SIMM4`, but still
  replaces it with an `OPR_SIMM32` operand when the raw field is 255 in the implementation.
- The shared executor then reads operand 1 through `RegisterAccess` and masks the
  resulting value to four bits before writing M0 in the implementation. The separate M0 bit-position bug is tracked in
  `CDNA3-RJ-075`.

Impact:

rocjitsu accepts and sizes an operand-1 literal form that the ISA text and LLVM
assembler treat as invalid. If a raw `SSRC1=255` word is encountered, rocjitsu
consumes the next dword as a literal and derives the mode from that extension
word instead of treating operand 1 as a raw 4-bit mode field.

### CDNA3-RJ-087: XML-only SOPP opcodes are generated on CDNA3 with incomplete execution

Manual/XML evidence:

- The CDNA3 manual's detailed Chapter 12.5 SOPP definitions and Chapter 13.1.5
  SOPP opcode table stop at `S_SET_GPR_IDX_MODE` opcode 29, and a direct manual
  search finds no `S_ENDPGM_ORDERED_PS_DONE` or `S_SET_VALU_COEXEC_MODE` entry;
  this source drift is recorded in `CDNA3-XML-059`.
- CDNA3 XML nevertheless records `S_ENDPGM_ORDERED_PS_DONE` as `ENC_SOPP`
  opcode 30 with program-terminator metadata and records
  `S_SET_VALU_COEXEC_MODE` as opcode 31 with a `SIMM16` operand and text saying
  the value in `SIMM16[1:0]` controls vector ALU co-execution mode for the next
  VALU instruction in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated CDNA3 decoding maps SOPP opcodes 30 and 31 to
  `decodeSEndpgmOrderedPsDoneSopp` and `decodeSSetValuCoexecModeSopp` in the implementation.
- Generated CDNA3 smoke encodings include `s_endpgm_ordered_ps_done` and
  `s_set_valu_coexec_mode` in the decode fixtures.
- `SEndpgmOrderedPsDoneSopp::execute_impl()` only calls `wf.end()`, while
  `SSetValuCoexecModeSopp::execute_impl()` is a no-op in the implementation.
- Static source search found no CDNA3 wavefront or VALU issue state for the
  one-instruction co-execution mode described by the XML.

Impact:

Manual-based CDNA3 tools may reject SOPP opcodes 30 and 31 as absent, while
rocjitsu accepts and disassembles them. If the XML rows are treated as
authoritative, the generated execution still reduces the ordered-PS row to a
plain wave termination and ignores the one-instruction VALU co-execution mode.

### CDNA3-RJ-088: FP min/max helpers do not model NaN, signed-zero, or IEEE/MODE tie rules

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define SNaN quieting, NaN operand selection,
  signed-zero selection, and IEEE/non-IEEE equality behavior in the cited manual passage.
- `V_MAX_F16` and `V_MIN_F16` define the corresponding F16 edge behavior in the cited manual passage.

Rocjitsu evidence:

- The shared F16 and F32 VOP2/VOP3 max helpers call `util::stdx::fmax` in the
  SIMD path and `std::fmax` in the scalar path in the implementation.
- The corresponding min helpers call `util::stdx::fmin` and `std::fmin` in the implementation.
- The adjacent SIMD correctness test explicitly excludes NaN-input lanes and
  signed-zero ties from comparison in the relevant tests.

Impact:

The helpers may be adequate for ordinary finite lanes, but they do not encode
the ISA's operand-selection contract for NaNs, SNaNs, signed-zero ties, or
IEEE-mode equality differences. Existing SIMD/scalar parity coverage
intentionally avoids those lanes, so this gap can survive local correctness
tests.

### CDNA3-RJ-089: `V_PK_FMAC_F16` decodes but throws in VOP2 and VOP3 forms

Manual/XML evidence:

- The CDNA3 manual defines `V_PK_FMAC_F16` as a packed F16 accumulate operation
  in the cited manual passage.
- The checked-in XML exposes `V_PK_FMAC_F16` VOP2 literal/DPP alternatives and
  the promoted VOP3 form in the machine-readable ISA XML.

Rocjitsu evidence:

- The generated VOP2 constructor records `VDST` as both the accumulator source
  and destination, but `VPkFmacF16Vop2::execute_impl()` throws
  `UnimplementedInst` in the implementation.
- The generated VOP3 constructor has the same accumulator/destination operand
  shape, but `VPkFmacF16Vop3::execute_impl()` also throws
  `UnimplementedInst` in the implementation.

Impact:

Programs using this manual-defined packed accumulate opcode can decode and
disassemble, but execution terminates with an unimplemented-instruction
exception instead of performing the two packed F16 FMAs.

### CDNA3-RJ-090: VOP2 accumulator DPP forms permute `VDST` instead of real `SRC0`

Manual/XML evidence:

- Chapter 12.7 says VOP2 instructions may use DPP immediately after the
  instruction in the cited manual passage, and Chapter 12.16
  defines the DPP extension's `SRC0` field as the source-0 VGPR in the cited manual passage.
- The DPP exclusion list in the cited manual passage onward does not exclude
  accumulator-style VOP2 rows such as `V_FMAC_F64`, `V_MAC_F16`, `V_DOT*C`, or
  `V_FMAC_F32`, and the XML records DPP alternatives for these rows, for
  example `V_FMAC_F64` in the machine-readable ISA XML and
  `V_MAC_F16`.

Rocjitsu evidence:

- Generated accumulator-style VOP2 constructors place `VDST` in
  `src_operands_[0]` and the real source 0 in a later slot; `VFmacF64Vop2` is a
  representative example in the implementation.
- The generic DPP preamble still applies DPP to `src_operands_[0]`, then
  delegates `VDST` to the DPP operand before executing the helper in the implementation.
- The same pattern is present in `VMacF16Vop2` in the implementation, dot accumulators, and `VFmacF32Vop2`.

Impact:

Legal DPP forms for accumulator-style VOP2 instructions use the DPP-selected
old destination as the accumulator while leaving the real `SRC0` unpermuted.
That reverses the intended source role for cross-lane DPP selection and can
produce wrong results for any non-identity DPP control.

### CDNA3-RJ-091: Literal-only `_MK`/`_AK` VOP2 forms accept modifier-shaped encodings

Manual/XML evidence:

- `V_FMAMK_F32`, `V_FMAAK_F32`, `V_MADMK_F16`, and `V_MADAK_F16` are
  literal-only VOP2 rows, and the manual says they do not support VOP3, input
  modifiers, or output modifiers in the cited manual passage.
- The XML rows expose only the literal VOP2 forms for these instructions in the machine-readable ISA XML.

Rocjitsu evidence:

- The generic VOP2 encoding helper still classifies `src0 == 250` as DPP and
  `src0 == 249` as SDWA for all VOP2 opcodes, while separately marking opcodes
  23, 24, 36, and 37 as implied-literal forms in the implementation.
- Generated `V_FMAMK_F32` and `V_FMAAK_F32` constructors/executors parse DPP
  and SDWA sentinels while also reading the same extension dword as `SIMM32` in the implementation.
- The F16 literal forms follow the same DPP/SDWA parsing pattern in the implementation.

Impact:

rocjitsu can treat extension words for manual-defined literal-only opcodes as
DPP or SDWA modifier payloads, even though the ISA defines those rows as
literal-only and explicitly forbids input/output modifiers. That can admit
encodings the hardware/assembler contract rejects, or decode a required literal
extension through the wrong extension format.

### CDNA3-RJ-092: `V_READFIRSTLANE_B32` returns zero instead of lane 0 when `EXEC` is disabled

Manual evidence:

- `V_READFIRSTLANE_B32` says an all-disabled `EXEC` mask selects lane 0, while
  nonzero `EXEC` selects the lowest active lane; it also says the VGPR read
  overrides the `EXEC` mask in the cited manual passage.

Rocjitsu evidence:

- Generated VOP1 execution initializes `val` to zero, iterates only active
  `EXEC` lanes, and writes the untouched zero when no active lane exists in the implementation.
- The promoted VOP3 alias follows the same pattern in the implementation.
- Existing shared-infra coverage exercises an active-lane case for
  `v_readfirstlane_b32_e32 s24, v1` in the relevant tests, but does not seed
  `EXEC == 0`.

Impact:

An all-disabled wave reads zero instead of lane 0. Kernels using
`V_READFIRSTLANE_B32` as an EXEC-independent scalarization primitive can observe
incorrect results when no lanes are active.

### CDNA3-RJ-093: `V_SAT_PK_U8_I16` decodes but throws in both VOP1 and VOP3 forms

Manual/XML evidence:

- `V_SAT_PK_U8_I16` saturates two signed 16-bit integer inputs over an unsigned
  8-bit range and packs the results in the cited manual passage.
- The checked-in XML exposes base VOP1 and promoted VOP3 encodings in the machine-readable ISA XML.

Rocjitsu evidence:

- The generated VOP1 decoder and constructor cover opcode 79, including
  `implicit_uses()` expansion for the packed destination, but
  `execute_impl()` throws `util::UnimplementedInst` in the implementation.
- The promoted VOP3 form has the same generated dataflow and throwing executor
  in the implementation.
- Semantic derivation maps `V_SAT_PK_U8_I16` to `nop` in the code generator.

Impact:

Legal CDNA3 VOP1/VOP3 encodings decode but cannot execute, and semantic
fingerprints can understate the missing saturation/packing behavior.

### CDNA3-RJ-094: XML-only `V_SCREEN_PARTITION_4SE_B32` decodes but cannot execute

Manual/XML evidence:

- The CDNA3 manual VOP1 tables skip opcode 55, while the checked-in XML defines
  `V_SCREEN_PARTITION_4SE_B32` in the machine-readable ISA XML; `CDNA3-XML-065` records the source drift.

Rocjitsu evidence:

- The generated decoder maps VOP1 opcode 55 and promoted VOP3 opcode 375 in the implementation.
- Generated VOP1 construction accepts default, literal, DPP, SDWA, and VOP3
  forms, but both VOP1 and VOP3 executors throw `util::UnimplementedInst` in the implementation.

Impact:

rocjitsu exposes an XML-only CDNA3 opcode but has no executable semantics for
it. Depending on the source-of-truth decision, this either needs execution
support or architecture-specific removal/invalid-op handling.

### CDNA3-RJ-095: `V_MOV_B32` and `V_MOV_B64` VOP3 aliases ignore allowed floating modifiers

Manual evidence:

- `V_MOV_B32` allows input modifiers when the source is treated as F32, and
  `V_MOV_B64` allows input modifiers when the source is treated as F64, in the cited manual passage.

Rocjitsu evidence:

- The VOP3 aliases construct modifier fields, but shared execution helpers
  perform raw lane copies and ignore `inst_.abs`/`inst_.neg` in the implementation for B32 for B64.
- The adjacent SIMD test currently documents ignored `v_mov_b32` modifiers as
  expected raw-copy behavior in the relevant tests.

Impact:

VOP3 MOV encodings with legal floating input modifiers can produce the
unmodified bit pattern instead of the manual-defined F32/F64 modified value.

### CDNA3-RJ-096: `V_FRACT_F32/F64` helpers omit the ISA max-below-one clamp

Manual evidence:

- `V_FRACT_F32` and `V_FRACT_F64` define DX-style negative behavior, obey the
  selected rounding mode, and clamp results to the largest representable value
  below 1.0 in the cited manual passage.

Rocjitsu evidence:

- VOP1 and VOP3 F32 helpers compute `v - std::floor(v)` without the manual
  max-below-one clamp in the implementation.
- VOP1 and VOP3 F64 helpers use the same unclamped formula in the implementation.
- Existing `fract` coverage in
  the relevant tests compares scalar and SIMD execution paths, so it does not catch
  a shared manual-oracle mismatch.

Impact:

Boundary inputs just below a negative integer can round to exactly 1.0 in the
emulator instead of the ISA's max-below-one value.

### CDNA3-RJ-097: Chapter 12.16 VOP1 DPP/SDWA exclusions are not enforced

Manual/XML evidence:

- Chapter 12.16.1 says DPP cannot be used with `V_READFIRSTLANE_B32`, the F64
  VOP1 conversion/unary block from `V_CVT_I32_F64` through `V_FRACT_F64`,
  `V_CLREXCP`, and `V_SWAP_B32` in the cited manual passage.
- Chapter 12.16.2 similarly excludes SDWA for `V_READFIRSTLANE_B32`,
  `V_CLREXCP`, and `V_SWAP_B32` in the cited manual passage.
- `CDNA3-XML-070` records the F64 VOP1 DPP rows that also leak through the
  checked-in XML.

Rocjitsu evidence:

- The generic VOP1 encoding helper treats source selectors 250 and 249 as DPP
  and SDWA extension formats without instruction-local legality checks in the implementation.
- Generated F64 VOP1 constructors accept and execute DPP forms for prohibited
  rows, including representative paths for `V_CVT_I32_F64`,
  `V_CVT_F32_F64`, `V_TRUNC_F64`, `V_FREXP_EXP_I32_F64`, and `V_FRACT_F64` in the implementation.
- `V_READFIRSTLANE_B32` parses DPP/SDWA extension words despite the manual
  exclusion in the implementation.
- `V_SWAP_B32` accepts literal, DPP, and SDWA extension words and applies the
  generic DPP/SDWA machinery before swapping in the implementation.

Impact:

rocjitsu can admit and execute modifier-extension encodings that the CDNA3 ISA
declares illegal. The literal-only `_MK`/`_AK` VOP2 overlap is tracked
separately in `CDNA3-RJ-091`; this entry is limited to the Chapter 12.16 VOP1
exclusions.

### CDNA3-RJ-098: VOP3 `V_CMPX_CLASS_*` writes VCC instead of explicit SDST

Manual/XML evidence:

- `V_CMPX_CLASS_F32` stores the class result into both `EXEC` and `D0`, with
  `D0 = VCC` only for VOPC encoding, in the cited manual passage.
- Chapter 13.3.3 says compare results target VCC in VOPC encoding and an
  arbitrary SGPR in VOP3 encoding in the cited manual passage.
- The checked-in XML matches that split: `V_CMPX_CLASS_F32` `ENC_VOP3`
  exposes an explicit `VDST` `OPR_SDST` plus implicit `OPR_SDST_EXEC`, while
  `ENC_VOPC` exposes VCC plus implicit EXEC in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated CDNA3 VOP3 class-X constructors define `vdst` for
  `VCmpxClassF32Vop3`, `VCmpxClassF64Vop3`, and `VCmpxClassF16Vop3`, but their
  execute bodies end with `wf.set_vcc(result); wf.set_exec(result);` and never
  write the explicit scalar destination in the implementation.
- The class-compare generator has the same pattern: `gen_vector_cmp_class()`
  writes VCC for `is_cmpx` whenever `cmpx_writes_vcc` is true and has no
  `dst`/VOP3 SDST branch in that path in the code generator.
- Ordinary non-class `gen_vector_cmpx()` already handles the distinction by
  writing `dst[0]` for VOP3 and VCC for VOPC in the code generator, so the
  bug is specific to the class-X generator path.

Impact:

`v_cmpx_class_{f16,f32,f64}_e64` clobbers fixed VCC and leaves the requested
SDST stale. Programs that use a non-VCC scalar destination for a VOP3 compare
mask see incorrect mask data and incorrect VCC liveness.

### CDNA3-RJ-099: VOPC SDWA sub-dword sources ignore scalar-source mode

Manual/XML evidence:

- Chapter 13.3.8 defines SDWAB as the VOPC SDWA extension word and includes
  `S0`/`S1` bits selecting whether source 0/source 1 are VGPR or SGPR sources
  in the cited manual passage.
- Chapter 12.16.2 excludes SDWA only for `V_MAC_F32`, `V_MAD*`, `V_FMAC_F32`,
  `V_READFIRSTLANE_B32`, `V_CLREXCP`, and `V_SWAP_B32`, leaving VOPC compare
  rows in scope in the cited manual passage.

Rocjitsu evidence:

- Generated VOPC constructors preserve the SDWA scalar-source mode initially:
  for representative `VCmpClassF32Vopc`, `sw->s0` creates `src0` as
  `OPR_SRC`, and `sw->s1` creates `vsrc1` as `OPR_SRC`, in the implementation.
- The shared generated SDWA preamble then ignores that operand type whenever
  `SRC*_SEL != DWORD`: it builds a VGPR address from `encoding_value_`, reads
  `read_vgpr()`, and creates a `DppOperand` delegate in the implementation.
- The same template also applies ABS/NEG by bit-casting selected dwords to
  `float` regardless of whether the VOPC row is F32, F16, integer, or a class
  mask source. The code generator emits this generic template for all
  SDWA-capable VOPC operations in the code generator.

Impact:

VOPC SDWA encodings with scalar sources and byte/word source selectors can
execute with VGPR lane data instead of the selected SGPR value. Integer and F16
compare SDWA modifier cases can also see F32-style ABS/NEG transformations
instead of type-appropriate source handling.

### CDNA3-RJ-100: Illegal 64-bit VOPC DPP forms are accepted

Manual/XML evidence:

- Chapter 12.16.1 explicitly says DPP cannot be used with
  `V_CMP_CLASS_F64`, `V_CMPX_CLASS_F64`, `V_CMP*_F64`, `V_CMP*_I64`,
  `V_CMP*_U64`, and their CMPX forms in the cited manual passage.
- Chapter 13.3.9's generic 64-bit DPP row-only note in the cited manual passage does not remove those
  instruction-specific exclusions.

Rocjitsu evidence:

- The CDNA3 generator treats `ENC_VOPC` as DPP-capable on non-RDNA DPP16
  targets by returning true from `_supports_dpp_for_encoding('ENC_VOPC')`, and
  maps `ENC_VOPC` DPP through the VOP1 DPP machine-inst layout in the code generator.
- Representative generated 64-bit VOPC constructors accept `SRC_DPP`, read the
  DPP extension, and execute `apply_dpp()` without instruction-local legality
  checks. `VCmpClassF64Vopc` does this in the implementation.
- `Vopc::default_encoding()` also treats `SRC_DPP` as an extension-word form
  for all VOPC rows, with no VOPC-specific legality filter, in the implementation.

Impact:

Rocjitsu can decode and execute 64-bit VOPC compare DPP forms that the CDNA3
manual marks illegal, instead of rejecting or diagnosing those encodings.

### CDNA3-RJ-101: CMPX semantic derivation drops the VCC/SDST result

Manual/XML evidence:

- Chapter 13.3.3 says every compare writes VCC for VOPC or an SGPR for VOP3,
  and CMPX variants additionally write EXEC, in the cited manual passage.
- `V_CMPX_CLASS_F32` spells this out as `EXEC.u64[laneId] = D0.u64[laneId] =
  result` in the cited manual passage.

Rocjitsu evidence:

- `_VectorCmpx.derive()` contains a TODO noting that it writes only EXEC even
  though GFX9/CDNA CMPX also writes VCC, then emits only an EXEC lane
  assignment in the code generator.
- `_VectorCmpxClass.derive()` has the same shape for class compares: it builds
  the class-test result and assigns only `EXEC[laneId]` in the code generator.
- Existing semantic-derivation coverage checks only that CMPX mentions EXEC,
  not that it also preserves the CDNA VCC/SDST result contract, in the codegen tests.

Impact:

Any semantic/DBT consumer built from `vector_cmpx` or `vector_cmpx_class`
metadata can lose the architectural compare mask result even when the generated
C++ execute generator models some ordinary CMPX writeback cases correctly.

### CDNA3-RJ-102: Packed 16-bit VOP3P operand metadata mixes packed-width and element-width

Manual/XML evidence:

- CDNA3 Chapter 12.10 packed 16-bit rows produce both low and high 16-bit
  components and write the packed result to `D0.b32`; representative rows are
  `V_PK_MAD_I16`, `V_PK_MUL_LO_U16`, `V_PK_ADD_F16`, and `V_PK_MIN_F16` in the cited manual passage.
- `CDNA3-XML-074` records that the checked-in XML uses 16-bit operands for
  some packed rows and 32-bit operands for adjacent rows with the same packed
  one-dword dataflow.

Rocjitsu evidence:

- Generated CDNA3 constructors inherit that mixed metadata: `V_PK_MAD_I16`,
  `V_PK_MUL_LO_U16`, `V_PK_MAX_I16`, `V_PK_MAD_U16`, `V_PK_MAX_U16`,
  `V_PK_MIN_U16`, `V_PK_MIN_F16`, and `V_PK_MAX_F16` construct 16-bit operands
  in the implementation.
- Adjacent packed F16 rows construct 32-bit source operands in the implementation.
- The operand resolver switches inline floating constants between f16 bit
  patterns and f32 bit patterns solely from `size_bits_`: 16-bit operands use
  `resolve_src_scalar16()` in the implementation, while other widths use `resolve_src_scalar()`.

Impact:

Register-sourced scalar execution still reads the full 32-bit lane and writes a
packed 32-bit result, but scalar/inline-constant sources and generated operand
metadata are inconsistent across the packed 16-bit family. For example, packed
F16 min/max use half-precision inline constants while packed F16 add/mul/fma
use the 32-bit floating constant pattern before splitting the value into two
halves.

### CDNA3-RJ-103: `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` miss the `ACCVGPR` instruction flag

Manual/XML evidence:

- CDNA3 Chapter 12.10 defines `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` as
  accumulator VGPR move operations in the cited manual passage.
- The checked-in XML canonical names are `V_ACCVGPR_READ` and
  `V_ACCVGPR_WRITE`, with `_B32` only as aliases, in the machine-readable ISA XML.

Rocjitsu evidence:

- Rocjitsu documents `InstFlags::ACCVGPR` as covering `v_accvgpr_write`,
  `v_accvgpr_read`, and `v_accvgpr_mov`, and `Instruction::is_accvgpr()` reads
  that flag in the implementation.
- The generator only sets the flag for `_B32` canonical names
  `V_ACCVGPR_WRITE_B32`, `V_ACCVGPR_READ_B32`, and `V_ACCVGPR_MOV_B32` in the code generator.
- Generated CDNA3 `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` constructors do not
  set `flags_ |= ACCVGPR` in the implementation, while the generated `V_ACCVGPR_MOV_B32` VOP1 constructor does set
  the flag in the implementation.

Impact:

The lane-wise copy execution bodies are present, so this is a classification
and analysis gap rather than a base decode/execute gap. Any path using
`is_accvgpr()` to recognize accumulator move instructions will miss the
read/write VOP3P forms.

### CDNA3-RJ-104: Native VOP3A rows decode but throw at execution

Manual/XML evidence:

- CDNA3 Chapter 12.11 defines `V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and
  `V_MQSAD_U32_U8` in the cited manual passage.
- The same chapter defines `V_TRIG_PREOP_F64` in the cited manual passage and `V_CVT_PKNORM_I16_F16` /
  `V_CVT_PKNORM_U16_F16`.
- The checked-in XML exposes these rows as `ENC_VOP3` instructions with
  ordinary operands.

Rocjitsu evidence:

- Generated CDNA3 constructors and decoder entries exist for the QSAD/MQSAD
  group, but each `execute_impl()` throws `UnimplementedInst` in the implementation.
- `VTrigPreopF64Vop3::execute_impl()` throws in the implementation.
- `VCvtPknormI16F16Vop3::execute_impl()` and
  `VCvtPknormU16F16Vop3::execute_impl()` throw in the implementation.
- Other `UnimplementedInst` throws in the implementation are already covered by
  older entries such as `CDNA3-RJ-089`, `CDNA3-RJ-093`, and `CDNA3-RJ-094`.

Impact:

Legal CDNA3 VOP3A instructions pass decode and constructor coverage but abort
when executed. This is a runtime support gap, not an opcode-inventory gap.

### CDNA3-RJ-105: `V_LSHL_ADD_U64` uses the ordinary masked shift rule

Manual evidence:

- `V_LSHL_ADD_U64` says the shift count must be 0 through 4, higher counts are
  unsupported, and unsupported counts behave as a shift of zero in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 execution dispatches `V_LSHL_ADD_U64` to the shared helper in the implementation.
- That helper calls `lshl_masked()` for the 64-bit shift, and the shared
  64-bit implementation masks the count with `count & 63` in the implementation.

Impact:

Shift counts 5 through 63 produce real shifted results in rocjitsu instead of
the manual's zero-shift behavior. Counts outside 0 through 4 therefore diverge
from the CDNA3 instruction contract.

### CDNA3-RJ-106: F32 packed conversion VOP3A helpers ignore source modifiers and `PKRTZ` rounding

Manual evidence:

- Chapter 6.2.1 says VOP3 instructions with floating-point inputs may apply
  `ABS` and `NEG` to input operands in the cited manual passage.
- Chapter 12.11 defines F32-input packed conversions including
  `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` in the cited manual passage.
- `V_CVT_PKRTZ_F16_F32` explicitly uses round-toward-zero and ignores the
  current rounding mode in the cited manual passage.

Rocjitsu evidence:

- `execute_v_cvt_pk_u8_f32_vop3()` and
  `execute_v_cvt_pkaccum_u8_f32_vop3()` read raw `SRC0` as `float` without
  applying `inst.inst_.abs` or `inst.inst_.neg` in the implementation.
- `execute_v_cvt_pknorm_i16_f32_vop3()` and
  `execute_v_cvt_pknorm_u16_f32_vop3()` use
  `ROCJITSU_TRY_SIMD_VOP3_BINARY_INT`, whose comment says it reads
  `src0/src1` with no modifiers in the implementation, and their scalar fallbacks read raw `SRC0`/`SRC1` in the implementation.
- `execute_v_cvt_pkrtz_f16_f32_vop3()` uses the same no-modifier SIMD macro,
  reads raw scalar fallback sources, and calls `util::f32_to_f16()` rather than
  the available `util::f32_to_f16_rtz()` helper in the implementation.

Impact:

Encoded `ABS`/`NEG` bits are ignored for these F32-input VOP3A packed
conversions, and `V_CVT_PKRTZ_F16_F32` rounds like the ordinary F32-to-F16
helper instead of using round-toward-zero semantics.

### CDNA3-RJ-107: VOP3B VCC selector operands are invisible to def-use metadata

Manual/XML evidence:

- VOP3B is the scalar-destination encoding for carry/div-scale/wide-MAD
  opcodes in the cited manual passage.
- Chapter 13.3.5 lists VCC selector values in the VOP3B source selector table
  in the cited manual passage.
- The checked-in XML exposes `V_ADD_CO_U32` and carry-consuming VOP3B rows with
  `SDST` or `SRC2` operands typed as `OPR_SREG`, for example
  the machine-readable ISA XML.

Rocjitsu evidence:

- Generated VOP3B constructors preserve those operands as `OPR_SREG`; for
  example `VAddCoU32Vop3SdstEnc` uses `sdst(64, OperandType::OPR_SREG...)`
  and `VAddcCoU32Vop3SdstEnc` uses both `sdst` and `src2` as `OPR_SREG` in the implementation.
- Runtime scalar reads and writes special-case selector value `106` as VCC in the implementation, so execution can use the VCC selector.
- `Operand::to_register_ref()` maps `OPR_SREG` only for SGPR values 0 through
  101 and drops VCC selector values 106/107 in the implementation.
- `InstDefUse` relies on `to_register_ref()` for explicit source and
  destination operands in the implementation.

Impact:

Execution can read or write VCC through legal VOP3B selector operands, but
def-use/liveness analysis misses those explicit VCC uses and defs. Carry chains
using VCC-form VOP3B operands can therefore be misrepresented to downstream
analysis even though scalar-pair SGPR operands are tracked.

### CDNA3-RJ-108: `DS_WRAP_RTN_B32` decodes but always throws

Manual/XML evidence:

- Chapter 12.12 defines `DS_WRAP_RTN_B32` opcode 52 as a ring-buffer-oriented
  wraparound subtract/add RMW in the cited manual passage.
- The checked-in XML has the same row in the machine-readable ISA XML.

Rocjitsu evidence:

- The CDNA3 decoder table maps DS opcode 52 to `decodeDsWrapRtnB32Ds` in the implementation.
- `DsWrapRtnB32Ds::execute_impl()` throws `util::UnimplementedInst` in the implementation.

Impact:

Legal CDNA3 code using `DS_WRAP_RTN_B32` aborts in rocjitsu even though the
opcode decodes.

### CDNA3-RJ-109: `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` use the wrong address contract

Manual/XML evidence:

- Chapter 12.12 defines both ADDTID forms as using
  `{OFFSET1, OFFSET0} + M0[15:0] + laneID * 4`, with no `ADDR` VGPR operand, in the cited manual passage.
- The XML rows expose `DATA0` plus implicit DSMEM/M0 for the write form and
  `VDST` plus implicit DSMEM/M0 for the read form in the machine-readable ISA XML.

Rocjitsu evidence:

- `DsWriteAddtidB32Ds::execute_impl()` calls ordinary
  `ds_calculate_addresses()` in the implementation.
- The shared DS address helper reads the encoded `ADDR` VGPR and computes
  `ADDR + {OFFSET1, OFFSET0} + lds_base` in the implementation, even though ADDTID has no architectural `ADDR` source.
- `DsReadAddtidB32Ds::execute_impl()` computes a per-lane stride from
  `M0[24:16] * 4` and uses `lane * stride + offset + lds_base` in the implementation.
- The semantic derivation comment records the same stride model in the code generator.

Impact:

ADDTID accesses are wrong whenever `M0[15:0]` is nonzero, and the write form can
depend on an encoded `ADDR` field that the manual and XML operand list do not
expose. The older no-gap note for ADDTID has been retired in favor of this
specific gap.

### CDNA3-RJ-110: `DS_CONDXCHG32_RTN_B64` is modeled as generic compare-swap

Manual/XML evidence:

- Chapter 12.12 defines `DS_CONDXCHG32_RTN_B64` as two independent conditional
  dword writes: align `(ADDR + offset)` with `& 0xfff8`, return both old dwords,
  and write each dword only when the corresponding source MSB is set, clearing
  that MSB on write, in the cited manual passage.
- The XML row has one 64-bit `DATA0` source and generic 64-bit DSMEM
  input/output operands in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated execution sets `d->atomic_op = amdgpu::AtomicOp::CMPSWAP`,
  `elem_size = 8`, and `num_elems = 1` in the implementation.
- The same body reads both `DATA0` and `DATA1` register-pair fields into a
  four-dword compare-swap source payload in the implementation.
- Semantic derivation classifies `_CONDXCHG32_RTN_B64` as `cmpswap` in the code generator.

Impact:

rocjitsu can use the wrong source operands, skip the manual address alignment,
and perform ordinary compare-swap behavior instead of the manual's per-half
conditional write exchange.

### CDNA3-RJ-111: `DS_SWIZZLE_B32` omits FFT and rotate modes

Manual/XML evidence:

- Chapter 12.12 defines `DS_SWIZZLE_B32` offset `>= 0xe000` as FFT mode,
  `0xc000 <= offset < 0xe000` as rotate mode, and lower offsets as two basic
  modes in the cited manual passage.
- The XML row identifies the opcode and operands in the machine-readable ISA XML.

Rocjitsu evidence:

- CDNA3 `DsSwizzleB32Ds::execute_impl()` dispatches to
  `execute_ds_swizzle_b32_ds()` in the implementation.
- The shared helper only branches on `offset & 0x8000` for quad mode versus
  bit mode and never checks the manual's `0xc000`/`0xe000` rotate/FFT ranges in the implementation.
- Adjacent tests cover broadcast and quad-permutation offsets, not rotate or
  FFT offsets, in the relevant tests.

Impact:

High-offset swizzle encodings execute with the wrong lane mapping.

### CDNA3-RJ-112: DS lane-routing helpers ignore the `ACC` register bank

Manual/XML evidence:

- Chapter 13.4.1 defines DS bit 25 `ACC` as selecting an AccVGPR destination in the cited manual passage.
- The XML rows for `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, and
  `DS_BPERMUTE_B32` expose `VDST` as `OPR_VGPR_OR_ACCVGPR`, and the permute
  rows also expose `DATA0` as `OPR_VGPR_OR_ACCVGPR`, in the machine-readable ISA XML.

Rocjitsu evidence:

- The CDNA3 constructors adjust visible operands by `ACC_MIN` when `inst_.acc`
  is set in the implementation.
- The shared swizzle, permute, and bpermute execution helpers read and write
  `vb + inst.inst_.data0` or `vb + inst.inst_.vdst` directly, ignoring
  `inst.inst_.acc`, in the implementation.

Impact:

Encodings with `ACC=1` can display as AccVGPR operands but execute against the
ordinary VGPR bank. At minimum, accumulator destinations for DS lane-routing
forms write the wrong register file.

### CDNA3-RJ-113: `DS_APPEND` and `DS_CONSUME` add `M0` in the LDS path

Manual/XML evidence:

- Chapter 12.12 says LDS APPEND/CONSUME use `instr_offset`, while GDS uses
  `M0.base + instr_offset`, in the cited manual passage.

Rocjitsu evidence:

- `DsConsumeDs::execute_impl()` and `DsAppendDs::execute_impl()` both reject
  `inst_.gds`, then compute `wf.lds_base() + wf.m0() + offset` for the LDS
  access in the implementation.
- The memory pipeline has special active-count handling for APPEND/CONSUME in the implementation, so the gap is the LDS address base rather than the count update.

Impact:

LDS APPEND/CONSUME can access the wrong counter whenever `M0` is nonzero.

### CDNA3-RJ-114: Scalar DS floating atomics use the generic host FP path

Manual evidence:

- Chapter 11 says LDS floating atomics use `MODE.FP_DENORM` denormal behavior
  and fixed round-to-nearest-even rounding in the cited manual passage.
- Chapter 12.12 marks DS floating add/min/max/compare-store forms as handling
  NaN, Inf, and denorms, for example in the cited manual passage.

Rocjitsu evidence:

- Generated scalar DS floating atomics lower to the same memory-pipeline
  `AtomicOp::FADD`, `FMIN`, `FMAX`, or `CMPSWAP` machinery as other atomics.
- `apply_fp_atomic()` uses host `+`, `std::fmin`, and `std::fmax` without LDS
  denormal-mode policy, fixed RNE handling, SNaN quieting, or signed-zero/NaN
  selection rules in the implementation.

Impact:

Scalar `DS_ADD/MIN/MAX/CMPST_F32/F64` can differ from CDNA3 hardware on denorm
mode, NaN propagation/quieting, signed zero, and rounding edge cases. Existing
`CDNA3-RJ-040` covers packed LDS F16/BF16 atomics; `CDNA3-RJ-080` covers the
related L2/buffer floating-atomic path.

### CDNA3-RJ-116: `BUFFER_WBL2` and `BUFFER_INV` do not model returned ACK/wait behavior

Manual evidence:

- Chapter 12.13 says both `BUFFER_WBL2` and `BUFFER_INV` return an ACK to the
  shader in the cited manual passage.
- Chapter 9.1.10 gives scope-specific writeback/invalidate behavior for these
  operations in the cited manual passage.

Rocjitsu evidence:

- Generated `BufferWbl2Mubuf` and `BufferInvMubuf` constructors have no memory
  operation flag, no destination, and no wait-counter state in the implementation.
- Shared helpers synchronously flush or invalidate broad cache levels and
  return immediately in the implementation.
- `CDNA3-RJ-030` tracks the coarse cache-policy behavior; this entry covers the
  separate returned-ACK/dependency-accounting contract.

Impact:

Cache maintenance side effects are represented only as immediate helper calls.
Workloads or tests that depend on the architectural ACK/wait behavior cannot be
validated against rocjitsu.

### CDNA3-RJ-117: The FLAT subdecoder over-accepts scratch atomics

Manual/XML evidence:

- Chapter 13.6.3 lists SCRATCH opcodes only from 16 through 42, ending with
  `SCRATCH_LOAD_LDS_DWORD`, in the cited manual passage.
- The CDNA3 XML likewise has `ENC_FLAT_SCRATCH` instruction rows for opcodes
  16 through 42 only, beginning in the machine-readable ISA XML; no
  `SCRATCH_ATOMIC_*` rows are present.
- FLAT and GLOBAL tables, by contrast, include atomic opcodes 64 through 82 and
  96 through 108 in the cited manual passage.

Rocjitsu evidence:

- `Decoder::subDecodeFlat()` indexes `sub_decode_flat` only by `op.op` and does
  not validate the `SEG` field in the implementation.
- The same table maps opcodes 64 through 82 and 96 through 108 to
  `FlatAtomic*` constructors unconditionally in the implementation and later.
- `flat_mnemonic()` rewrites any `flat_*` mnemonic to `scratch_*` when
  `SEG==1` in the implementation, and the generated atomic constructors contain scratch
  `inst_.seg == 1` operand shaping.

Impact:

Illegal `SEG=SCRATCH` encodings with atomic opcode numbers decode and execute
as if scratch atomics existed on CDNA3, instead of rejecting the encoding.

### CDNA3-RJ-119: `V_MAC_F16` and `V_FMAC_F32` prohibited SDWA forms are accepted

Manual/XML evidence:

- Chapter 12.16.2 says SDWA cannot be used with `V_MAC_F16` and `V_FMAC_F32`
  in the cited manual passage.
- The checked-in XML matches that restriction for the listed rows: `V_MAC_F16`
  and `V_FMAC_F32` expose default, literal, DPP, and promoted VOP3 encodings,
  but no SDWA alternative, in the machine-readable ISA XML.
- `CDNA3-RJ-091` separately tracks literal-only `_MK`/`_AK` modifier
  over-acceptance, and `CDNA3-RJ-097` is limited to Chapter 12.16 VOP1
  exclusions.

Rocjitsu evidence:

- The generic CDNA3 VOP2 encoding helper treats `src0 == 249` as SDWA for every
  VOP2 opcode through `Vop2::has_sdwa()` and removes that selector from the
  default encoding in the implementation.
- Generated `VMacF16Vop2` accepts the SDWA sentinel in its constructor and runs
  the shared SDWA preamble in execution in the implementation.
- Generated `VFmacF32Vop2` has the same SDWA constructor and execution path in the implementation.

Impact:

rocjitsu can decode and execute SDWA extension encodings for VOP2 rows that
the CDNA3 manual and XML both classify as non-SDWA instructions.

### CDNA3-RJ-120: 64-bit DPP controls are not restricted to `DPP_ROW*`

Manual/XML evidence:

- Chapter 13.3.9 says DPP can follow VOP1, VOP2, or VOPC instructions, but
  states that for 64-bit input data the only legal DPP type is `DPP_ROW*` in the cited manual passage.
- `CDNA3-XML-092` records that the checked-in XML exposes 64-bit DPP rows such
  as `V_MOV_B64` and `V_FMAC_F64`, but does not carry the `DPP_CTRL` subset
  restriction.
- LLVM agrees with the manual restriction on `gfx942`: `row_newbcast:1`
  assembles for `v_mov_b64` and `v_fmac_f64`, while `row_shr:1` and
  `quad_perm:[0,1,2,3]` are rejected with "DP ALU dpp only supports
  row_newbcast".

Rocjitsu evidence:

- Generated 64-bit DPP constructors copy `dp->dpp_ctrl` without classifying the
  control value, for example `VMovB64Vop1` and `VFmacF64Vop2` in the implementation.
- The execute paths call the generic `amdgpu::dpp::apply_dpp()` without an
  operand-width or `DPP_CTRL` legality check in the implementation.
- The shared helper implements non-`DPP_ROW*` controls such as quad permute,
  row shifts, wave shifts, mirrors, and broadcasts, and unknown controls fall
  through to identity in the implementation.

Impact:

Invalid 64-bit DPP encodings can be decoded and routed into normal execution
instead of being rejected at decode or construction time. This is separate from
the `CDNA3-RJ-100` fully-illegal 64-bit VOPC DPP forms and from the
`CDNA3-RJ-121` valid-control data handling issue.

### CDNA3-RJ-121: 64-bit DPP source substitution only carries one dword

Manual/XML evidence:

- Chapter 13.3.9 allows `DPP_ROW*` for 64-bit input data in the cited manual passage.
- `V_MOV_B64` is a representative 64-bit VOP1 row with a DPP encoding and
  64-bit source/destination operands in the checked-in XML in the machine-readable ISA XML; LLVM accepts the legal
  `row_newbcast` form on `gfx942`.

Rocjitsu evidence:

- `VMovB64Vop1::execute_impl()` applies DPP, delegates `src0` to the resulting
  `DppOperand`, and then calls the shared 64-bit move helper in the implementation.
- `apply_dpp()` reads only one VGPR register, stores one `uint32_t` per lane,
  and constructs `DppOperand` from that 32-bit lane array in the implementation.
- `DppOperand` implements 32-bit lane reads and a 32-bit SIMD storage view, but
  does not override `read_lane64()` or `simd_vgpr_storage64_impl()` in the implementation; the base
  64-bit read throws in the implementation, and
  CDNA3 operand delegation forwards `read_lane64()` to that delegate in the implementation.

Impact:

Even a legal 64-bit DPP control such as `row_newbcast` cannot provide the full
64-bit source lane value through the current DPP proxy. Forced-scalar execution
falls into the missing `read_lane64()` path, and the SIMD path falls back to
the same delegate read when no 64-bit storage pair is available.

### CDNA3-RJ-122: Reserved FLAT `SEG=3` encodings are accepted

Manual/XML evidence:

- Chapter 10.1 defines `SEG` as memory segment `0=FLAT`, `1=SCRATCH`,
  `2=GLOBAL`, and `3=reserved` in the cited manual passage.
- Chapter 13.6 repeats the same segment set for the FLAT format in the cited manual passage.
- `CDNA3-RJ-117` covers illegal scratch atomic opcodes; this entry covers the
  separate reserved segment value.

Rocjitsu evidence:

- `Decoder::subDecodeFlat()` indexes the flat subdecode table only by `op.op`
  and does not validate `op.seg` in the implementation.
- `flat_mnemonic()` rewrites only `SEG==1` and `SEG==2`; other segment values
  keep the flat mnemonic in the implementation.
- The shared flat address helper treats non-scratch and non-global segment
  values as the ordinary FLAT path in the implementation.

Impact:

Reserved `SEG=3` flat-family encodings can decode and execute as ordinary FLAT
instructions instead of being rejected.

### CDNA3-RJ-123: Global SGPR-base VGPR offsets are sign-extended

Manual/XML evidence:

- Section 10.4 says GLOBAL instructions may use `SGPR-address + VGPR-offset +
  instruction offset` and that the VGPR offset is 32 bits in the cited manual passage.
- Chapter 13.6 says the single `ADDR` VGPR is a 32-bit unsigned offset when
  `SADDR` is not `0x7f` for GLOBAL and SCRATCH forms in the cited manual passage.

Rocjitsu evidence:

- `flat_calculate_addresses()` reads one VGPR when `SADDR != 0x7f`, casts that
  value to `int32_t`, then sign-extends it to `uint64_t` before adding the
  SGPR base and signed instruction offset in the implementation.
- The no-gap note below previously described this sign extension as modeled
  segment behavior; the manual's field table makes the global VGPR offset
  unsigned.

Impact:

GLOBAL instructions that use an SGPR base and a VGPR offset with bit 31 set can
address below the base in rocjitsu, while the CDNA3 manual describes that VGPR
component as an unsigned 32-bit byte offset.

### CDNA3-RJ-124: Signed GLOBAL/SCRATCH immediate offsets disassemble as unsigned

Manual/XML evidence:

- Chapter 10.1 says scratch and global use a 13-bit signed byte offset, while
  flat uses a 12-bit unsigned offset with the MSB ignored, in the cited manual passage.
- Chapter 13.6 repeats the same offset signedness split in the cited manual passage.

Rocjitsu evidence:

- Runtime address calculation sign-extends the flat-family immediate through
  `sign_extend(inst.offset | (inst.pad_12 << 12), 13)` in the implementation.
- `Flat::build_modifiers()` instead reconstructs the same 13 bits as an
  integer and prints it directly in the `offset:` modifier, without
  sign-extending global/scratch forms, in the implementation.

Impact:

Negative GLOBAL/SCRATCH immediate offsets can execute with the intended signed
address calculation but disassemble as large positive offsets, so text output
does not round-trip the encoded instruction semantics.

### CDNA3-RJ-125: Flat aperture routing samples only the first active lane

Manual/XML evidence:

- Chapter 10 says FLAT addresses can map to video/system memory, LDS, or
  scratch and that unmapped regions generate memory violations in the cited manual passage.
- Section 10.2 says FLAT internally uses both LDS and buffer paths and
  increments both `VM_CNT` and `LGKM_CNT` in the cited manual passage.
- Section 10.4 says GLOBAL instructions must not access LDS and return
  `MEM_VIOL` if they do in the cited manual passage.

Rocjitsu evidence:

- `ComputeUnitCore::route_memory_inst()` chooses a single `probe` address from
  the first active lane of a `GLOBAL_MEM` vector-memory state, then routes the
  entire instruction to LDS and rewrites all active lane addresses if that
  first address is in the shared aperture in the implementation.
- The same routing path changes the instruction tag to `LOCAL_MEM` and changes
  the wait counter to `LGKMCNT`, which overlaps but does not fully model the
  dual-counter FLAT behavior recorded in `CDNA3-RJ-034`.

Impact:

Mixed-lane FLAT operations can be routed as all-LDS or all-global based on the
first active lane instead of per-lane aperture behavior. GLOBAL instructions
that hit the LDS aperture can be silently converted into LDS operations rather
than reporting the manual's memory-violation case.

### CDNA3-RJ-126: `XNACK_MASK_LO/HI` use ordinary SGPR storage

Manual evidence:

- The Chapter 3.1 state table defines `XNACK_MASK` as a 64-bit bit mask of
  threads that have failed address translation in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA3 operand selector enums name `XNACK_MASK_LO` and
  `XNACK_MASK_HI` for scalar source and destination selector values 104 and
  105 in the implementation.
- `resolve_src_scalar()` special-cases `FLAT_SCRATCH_LO/HI` at selector values
  102 and 103, but then sends all selector values `<= 105`, including 104/105,
  to ordinary SGPR storage in the implementation.
- `resolve_dst_write()` mirrors the same behavior: 102/103 update the
  dedicated scratch-base field, while selector values `<= 105` write ordinary
  SGPR storage in the implementation.
- A targeted search of `Wavefront`, register access, command-processor, CU, and
  memory-pipeline state found no CDNA3 `XNACK_MASK` state or address-translation
  failure mask updates.

Impact:

Kernels that read or write `XNACK_MASK_LO/HI` observe simulator SGPR contents
rather than the architectural per-lane address-translation-failure mask. Memory
paths also cannot set the mask on replayable address-translation failures, so
stateful XNACK behavior is absent even though the selector names decode.

### CDNA3-RJ-127: Raw STATUS launch and reuse initialization is incomplete

Manual evidence:

- Chapter 3.4 says STATUS fields are initialized when a wavefront is created,
  are readable by shader code, and include bits such as `EXECZ`, `VCCZ`,
  `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`, `SCRATCH_EN`, and `IDLE` in the cited manual passage.
- Chapter 3.3 defines `EXECZ` as the helper bit for zero EXEC, and Chapter
  3.9 says `VCCZ` updates whenever VCC updates in the cited manual passage.

Rocjitsu evidence:

- The shared CDNA STATUS wrapper exposes only part of the CDNA3 STATUS table,
  from `SCC` through `ALLOW_REPLAY`, and has no helpers for later manual bits
  such as `SCRATCH_EN` or `IDLE` in the implementation.
- `ComputeUnitCore::allocate_wf()` initializes the wavefront PC, allocation
  records, EXEC, VCC, M0, apertures, scheduler state, and instruction count
  launch, but does not initialize or rebuild raw STATUS in the implementation.
- `Wavefront::reset()` explicitly says it does not change the status register,
  then resets dispatch state, EXEC, VCC, M0, MODE, apertures, counters, and
  scheduler state while leaving raw STATUS intact in the implementation.
- The ISA-specific wavefront still exposes raw STATUS through `status_raw()`
  and initializes the stored value to zero only at object construction in the implementation.
- The current CDNA3 HWREG read path returns that raw STATUS value through the
  correct STATUS ID in the shared architecture-aware helper, so the remaining
  problem is stale/incomplete STATUS state rather than a selector-map mismatch.

Impact:

Fresh or reused wavefronts can expose raw STATUS bits that do not match the
manual's wave-creation state. The first launch sees zero for bits that should
reflect the created/running wave, and reused slots can preserve stale raw STATUS
bits across dispatches even as the execution state, EXEC, VCC, M0, and MODE are
reset. This is distinct from `CDNA3-RJ-051`, which tracks later
`EXECZ`/`VCCZ` drift after EXEC/VCC mutations.

## No-Gap Notes

- The generated CDNA3 VOP3A/VOP3B decoder table and constructors cover every
  named Chapter 12.11 instruction found in the checked-in XML. The new VOP3A
  runtime gaps above are about hard-stubbed execute bodies or detailed
  semantics, not missing decode entries.
- VOP3B opcode inventory and production execution cover the basic
  scalar-destination dataflow for the ten manual VOP3B opcodes: add/sub carry
  helpers write arbitrary `SDST`, carry-consuming helpers read source 2 as the
  carry-in mask, division-scale helpers write the scalar mask, and wide-MAD
  helpers write the explicit scalar carry destination. The remaining VOP3B
  issue in this slice is the VCC selector's analysis metadata in
  `CDNA3-RJ-107`.
- Literal extension handling matches the manual's "not VOP3" prose in the
  audited generated path: `Vop3` and `Vop3SdstEnc` keep size to one 64-bit
  instruction and do not consume a following literal extension word.
- The generated CDNA3 VOP1 decoder covers all manual Chapter 12.8 rows and
  leaves the manual opcode holes invalid, except for XML-only opcode 55 tracked
  in `CDNA3-RJ-094`. The VOP1 gaps above are about execution semantics,
  source drift, or illegal extension forms rather than missing manual-listed
  base opcodes.
- The generated CDNA3 VOPC decoder covers the full Chapter 12.9 opcode
  inventory: class opcodes 16-21, FP compare ranges 32-127, and integer
  compare ranges 160-255 are decoded, while the documented holes remain
  invalid in the implementation.
- Ordinary non-class VOP3 `V_CMPX_*` rows already use the explicit SDST path in
  the generator and generated code; `CDNA3-RJ-098` is limited to
  `V_CMPX_CLASS_*` rows.
- VOPC E32 `CMPX` rows write VCC and EXEC in the generated execution path, and
  the generic DPP cleanup merges VCC/EXEC with the DPP write mask for masked
  lanes. The VOPC DPP findings above are about missing XML availability and
  64-bit legality, not absence of all DPP execution machinery.
- VOPC literal handling follows the expected format split in this static pass:
  E32 VOPC extends size for literal/DPP/SDWA sentinel encodings, while VOP3A
  compare constructors use no literal fixup and source operand classes exclude
  literal extension words.
- Outside the writeback/SDWA issues above, the audited generated class and
  relational predicate helpers distinguish sNaN/qNaN, infinities, normals,
  denormals, signed zero, unordered predicates, and negated-NaN compare cases
  in line with the detailed Chapter 12.9 formulas.
- Plain `V_SWAP_B32` VOP1/VOP3 dataflow is represented: generated classes mark
  both `VDST` and `SRC0` as read-write operands and the plain VOP3 executor
  performs the swap in the implementation. `CDNA3-RJ-097` is limited to illegal literal/DPP/SDWA extension
  forms on the VOP1 path.
- `V_CVT_OFF_F32_I4` is not currently a semantic-fallback gap: semantic
  derivation has a specific `vector_unary/cvt_off_f32_i4` override in the code generator.
- FP8/BF8 VOP1 widening behavior for rows 84-87 remains covered by
  `CDNA3-RJ-076` and `CDNA3-RJ-078`; this VOP1 pass did not find an additional
  opcode-inventory miss for those rows.
- `V_CLREXCP` no-op execution is recorded under the broader trap/exception
  state gap `CDNA3-RJ-054`, not duplicated as a separate VOP1 finding.
- The generated CDNA3 VOP2 inventory covers the Chapter 12.7 opcode table
  static-audit level: the decoder maps manual opcodes 0-21 and 23-61 and leaves
  the opcode 22 hole invalid, while the literal-only `_MK`/`_AK` VOP2 rows are
  not promoted to VOP3. The VOP2 gaps above concern execution semantics or
  illegal extension forms, not a missing base opcode inventory.
- VOP3 `V_MAC_F16` preserves the manual `OPSEL[3]` read/modify/write contract:
  generated execution reads the selected old destination half and writes back
  through `write_vop3_true16_dst(..., true)` in the implementation. The remaining F16 destination-half concerns are about XML
  recoverability and VOP2 literal/DPP legality, not this promoted true16 path.
- CDNA3 VOP2 carry/borrow helpers model the basic VOP2/VOP3 carry-out dataflow:
  `V_ADD_CO_U32` and `V_ADDC_CO_U32` write VCC or arbitrary VOP3 `SDST` forms,
  and the VOP3 add-with-carry path reads source 2 as the carry-in in the implementation. The existing clamp/saturation
  limitations remain covered by `CDNA3-RJ-074`.
- `V_LDEXP_F16` sign-extends the 16-bit exponent source before calling
  `std::ldexp`, matching the manual's signed-integer second operand in the implementation.
- CDNA3 Chapter 1-2 dispatch, 64-lane wavefront, initial `EXEC`, and packed
  work-item-ID basics are represented by the production dispatch path and
  existing Chapter 3 no-gap notes below. The new Chapter 2 issue above is
  limited to the device-memory consistency and acknowledgment model; LDS
  clamping/bank behavior, GWS execution, launch TTMP/TG_SIZE state, barrier
  behavior, and workgroup-size validation remain covered by their existing
  narrower entries.
- CDNA3 currently does use FNUZ-specific helper names for FP8/BF8 conversion
  execution, so the main format-family split from OCP RDNA4/CDNA4 is present.
- CDNA3 generated SOP literal handling matches the broad audited format split
  for ordinary literal-capable SOP1/SOP2/SOPC rows, while SOPK's
  `S_SETREG_IMM32_B32` opcode 20 is the only SOPK path that consumes an
  extension dword. Runtime stores that extension word in `Sopk::literal_` in the implementation. `CDNA3-RJ-085` records the separate default-only SOP1 PC-form
  legality/size issue; the `S_SETREG_IMM32_B32` gap above is specifically about
  missing literal operand metadata/disassembly, not extension-word fetch.
- The CDNA3 generated SOP1 inventory matches the Chapter 12.3 manual/XML opcode
  inventory: constructors span the generated SOP1 file, the decode table maps
  opcode holes 47 and 49 to invalid, and generated smoke encodings cover the
  ordinary 32-bit rows in the implementation,
  the implementation, and
  the decode fixtures. `CDNA3-RJ-085` records the separate legality issue for selected
  default-only PC encodings.
- Shared SOP1 helpers implement representative Chapter 12.3 unary and EXEC
  formulas at static-audit level: bit-count and bit-scan helpers are present in the implementation, `S_ABS_I32` preserves the
  `0x80000000` wrap result and writes SCC,
  bitset/bitreplicate helpers cover the manual bit-index behavior, and saveexec/wrexec/quadmask helpers update EXEC, SDST, and
  SCC in the reviewed ranges.
- The CDNA3 generated SOPC inventory matches the Chapter 12.4 manual/XML opcode
  inventory: constructors span `SCmpEqI32Sopc` through `SCmpLgU64Sopc`, the
  decode table maps opcodes 0 through 19 to those constructors, and generated
  smoke encodings cover the ordinary 32-bit rows in the implementation,
  the implementation, and
  the decode fixtures.
- Shared SOPC compare and bit-compare helpers implement the audited Chapter
  12.4 integer formulas at static-audit level: bit compare masks source 1 to
  five bits for B32 and six bits for B64 in the implementation, while signed, unsigned, EQ, LG, GT, GE, LT, and LE compare
  helpers use the expected integer comparisons.
- `S_SETVSKIP` is a SOPC row, but its missing execution semantics are already
  tracked under the program-control finding `CDNA3-RJ-052`; `S_SET_GPR_IDX_ON`
  has its broad M0 layout and role-mapping gap tracked in `CDNA3-RJ-075`, with
  the raw source-1 literal exception recorded separately in `CDNA3-RJ-086`.
- The CDNA3 generated SOPK inventory matches the Chapter 12.2 manual/XML opcode
  inventory after accounting for the literal-only opcode 20 form: constructors
  span `SMovkI32Sopk` through `SCallB64Sopk`, the decode table keeps opcode 19
  invalid and opcode 20 mapped to `SSetregImm32B32Sopk`, and generated smoke
  encodings cover the ordinary 32-bit SOPK rows in the implementation,
  the implementation, and
  the decode fixtures.
- The shared SOPK helpers implement the Chapter 12.2 signed/unsigned SIMM16
  rules for `S_MOVK_I32`, `S_CMOVK_I32`, and `S_CMPK_*`: signed forms cast
  through `int16_t`, while unsigned compare forms zero-extend the 16-bit
  immediate in the implementation.
- `S_ADDK_I32` and `S_MULK_I32` are not missing old-destination dataflow:
  generated constructors expose the destination as both source and destination,
  the shared helpers read the old scalar value before writeback, and adjacent
  tests assert def-use plus SCC behavior in the implementation,
  the implementation, and
  the relevant tests.
- `S_CBRANCH_I_FORK` is a SOPK row, but its missing execution semantics are
  already tracked under the program-control finding `CDNA3-RJ-060`.
- `S_RFE_RESTORE_B64` is a CDNA3 SOP2 row in XML but not in the CDNA3 manual;
  the source drift is recorded as `CDNA3-XML-056`, and the generated runtime
  throw is already covered by `CDNA3-RJ-058`.
- Generated stochastic narrow converts read the old destination and merge the
  selected byte manually, matching the manual's unwritten-byte preservation rule
  for the audited forms. The packed 16-bit low-half preservation divergence is
  recorded separately as `CDNA3-RJ-129`.
- Generated packed shift helpers mask the selected shift-count half with
  `& 15`, matching the Chapter 12.10 use of `S0[3:0]` and `S0[19:16]` for the
  low and high components. Representative helpers are in the implementation and analogous LSHL/LSHR helpers.
- `V_PK_MOV_B32` uses `read_lane64()` for both sources and selects output dwords
  with `OPSEL[0]` and `OPSEL[1]` in the implementation, matching the CDNA3 manual's special `V_PK_MOV_B32`
  selector behavior for scalar pairs and VGPR gather.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` have generated 32-bit lane-copy
  execution bodies under `EXEC` in the implementation. `CDNA3-RJ-103` is limited to the missing public instruction flag.
- `V_ACCVGPR_WRITE` uses the broad `OPR_SRC_NOLIT` source class in generated
  code, matching the checked-in XML and LLVM-accepted scalar/inline-constant
  source forms used by real kernels; this static pass did not record that as a
  runtime legality gap.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` in the implementation. They use multiply-add rather than fused FMA; the detailed
  instruction pseudocode and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- Packed 32-bit helpers do not apply clamp or other output modifiers, matching
  the packed 32-bit statement in the cited manual passage that output modifiers
  are not supported for those instructions.
- Generated dense MFMA constructors implement the manual's register-bank
  controls by applying `ACC` to source A/B operands and `ACC_CD` to destination
  and source C operands; representative generated code is in the implementation, and `mfma_src2_encoding()` applies the `ACC_CD` source-C rewrite.
- Shared dense MFMA layout helpers implement the manual's input and output
  lane/register formulas for 8/16/32/64-bit data in the implementation.
- Dense MFMA helper paths apply ordinary `CBSZ`/`ABID` A-lane broadcast and
  `BLGP` B-lane permutation through `permute_a_lane()` and `permute_b_lane()` in the implementation; the gap above is only about missing legality
  validation.
- F64 dense MFMA repurposes `BLGP` as A/B/C negation in rocjitsu: generated F64
  calls pass `inst_.blgp` as the `neg` parameter in the implementation, and `exec_f64()` applies bits `0x1`, `0x2`, and `0x4` to A, B, and C
  in the implementation.
- Dense MFMA execution writes full-wave results through `RegisterAccess(cu)`
  rather than `Wavefront` EXEC-filtered writes, matching the manual's statement
  that MFMA ignores the execution mask. Representative F32/I32/F64 writes are
  in the implementation.
- Generated SMFMAC constructors model `SRC2` as an Arch VGPR operand and do
  not apply `ACC_CD` to it; `ACC_CD` only adjusts the generated destination
  operand. Representative constructors are in the implementation.
- F32-result SMFMAC helpers read the old destination block as the accumulator
  and write the destination block after buffering results, matching the
  accumulate-only C/D aliasing contract. Representative helper bodies are in the implementation.
- CDNA3 FP8/BF8 SMFMAC generated execution uses FNUZ readers, matching CDNA3's
  FP8/BF8 numeric-family split; representative generated calls are in the implementation.
- CDNA3 SMEM SBASE operand display and def-use scale the encoded SBASE field to
  an even SGPR pair, and the relevant tests covers that
  operand-level behavior.
- Generated CDNA3 SMEM disassembly preserves the immediate offset as an
  `offset:` modifier when both `IMM` and `SOFFSET_EN` are set, so the paired
  immediate is not lost by the disassembly path.
- Generated CDNA3 SMEM class and decoder inventory matches the XML `ENC_SMEM`
  records exactly: 84 generated constructors and 84 non-invalid
  `sub_decode_smem` entries cover the 82 manual Chapter 12.6 instruction
  definitions plus the two XML-only `S_ATC_PROBE*` opcodes recorded in
  `CDNA3-XML-060`. The decoder table places the manual
  load/store/cache/time/discard/atomic opcodes in their documented slots and
  leaves the holes invalid in the implementation.
- Generated CDNA3 `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` decode from the XML,
  appear in the generated smoke encodings, and no-op in execution in the implementation, the decode fixtures, and
  the implementation. Because the CDNA3 manual does not list those opcodes, this
  audit records the mismatch as `CDNA3-XML-060` rather than a manual-derived
  runtime gap.
- Generated CDNA3 SMEM operand widths match the audited Chapter 12.6 data-width
  rows at static-audit level: load/store widths scale from dword through x16,
  buffer-resource `SBASE` operands are 128-bit while scalar/scratch bases are
  64-bit, and compare-swap atomics expose double-width `SDATA` operands in both
  32-bit and 64-bit families. The remaining SMEM gaps above are about address,
  descriptor, dependency, cache, and atomic execution behavior.
- Generated CDNA3 MUBUF/MTBUF constructors derive VADDR operand width from
  `IDXEN`/`OFFEN` with zero-, one-, or two-VGPR widths in the implementation, matching the manual's address-VGPR table rather than the
  fixed-width XML operand entries.
- The shared MUBUF/MTBUF address helpers combine instruction offset and VGPR
  offset before adding the SGPR offset in the implementation, so rocjitsu
  does not inherit the XML `OFFEN` wording that implies only one offset source.
- Basic raw CDNA3 MUBUF byte/short/dword load paths set element size, sign
  extension, D16 half selection, and SC/NT-derived memory type before issuing
  vector-memory operations; representative bodies are in
  the implementation.
- For single-half raw D16 buffer loads, the vector-memory completion path models
  the CDNA3 D16 ECC note by preserving the untouched half when SRAM ECC is
  disabled and zeroing unused bits when it is enabled in the implementation. Multi-component MTBUF D16 load packing is tracked separately in
  `CDNA3-RJ-131`.
- MUBUF integer atomic update formulas are implemented in the shared memory
  pipeline: compare-swap source/compare ordering and `INC`/`DEC` wrap behavior
  are handled by `apply_int_atomic()` and `execute_atomic_rmw()` in the implementation. The MUBUF runtime gaps above are about metadata and ACK/wait
  behavior, not the basic integer RMW formulas.
- The shared flat address helper models part of the main decoded segment split:
  unsigned 12-bit FLAT offsets, signed 13-bit GLOBAL/SCRATCH immediate
  offsets, 64-bit SGPR global bases, `saddr == 0x7F` VGPR-pair global
  addressing, scratch base plus lane stride, and optional scratch VGPR/SADDR
  offsets in the implementation. `CDNA3-RJ-123` records the separate unsigned global
  VGPR-offset issue.
- Rocjitsu partially models aperture routing: private-aperture FLAT addresses
  are remapped to scratch backing memory in the shared helper, and shared
  aperture addresses are routed to the LDS pipeline in the compute-unit router.
- Generated CDNA3 flat memory classes preserve D16 low/high-half load behavior,
  sign extension, ACC bank selection, `SC0` return selection for integer
  atomics, and `SC`/`NT` cache-flag plumbing in the audited representative
  bodies.
- FLAT/GLOBAL integer atomic update formulas are implemented in the shared
  memory pipeline: compare-swap source/compare ordering, signed/unsigned
  min/max, and `INC`/`DEC` wrap behavior are handled by `apply_int_atomic()`
  and `execute_atomic_rmw()` in the implementation. The flat-family runtime gaps above are about decode validity,
  metadata, wait counters, direct-LDS movement, and FP atomic edge behavior.
- Shared CDNA3 DS address paths cover the ordinary concatenated-offset form for
  single-address operations and the expected width/ST64 scaling in representative
  READ2/WRITE2 bodies. The remaining gaps are about M0 clamp, duplicate-offset,
  ADDTID, APPEND/CONSUME, and timing details.
- Generated narrow DS reads apply signed or zero extension for I8/U8/I16/U16
  forms in the audited representative bodies.
- DS_SWIZZLE, DS_PERMUTE, and DS_BPERMUTE execute through direct cross-lane
  helpers rather than the memory pipeline, matching the manual/XML statement
  that these forms do not write actual LDS memory. `CDNA3-RJ-111` and
  `CDNA3-RJ-112` record the remaining swizzle-mode and ACC-bank issues.
- APPEND/CONSUME have special active-count handling in the LDS atomic helper.
  `CDNA3-RJ-113` records the separate LDS address-base issue.
- Adjacent scalar SCC helpers reviewed in this slice match the CDNA3 manual
  static-audit level: `S_MIN_{I32,U32}` uses strict `<`; `S_ADD*`/`S_SUB*`,
  `S_ADDC_U32`, `S_SUBB_U32`, and `S_LSHL{1,2,3,4}_ADD_U32` write carry or
  overflow through SCC; logical, shift, BFE, ABS, and ABSDIFF helpers write
  result-nonzero SCC; and `S_BFM_{B32,B64}` does not write SCC.
- CDNA3 access-instruction constructors and decoders do recognize
  `S_GETREG_B32`, `S_SETREG_B32`, and the 64-bit `S_SETREG_IMM32_B32` opcode,
  and the execute helpers implement the generic `{offset,size}` extraction or
  insertion around the handled raw register value. The access gaps above are
  about the register map, write legality/side effects, and literal operand
  visibility.
- Ordinary CDNA3 `EXECZ`/`VCCZ` branch and scalar-source consumers compute from
  the current `wf.exec()` and `wf.vcc()` values, so the raw STATUS gap above is
  not a claim that conditional branches themselves use stale helper bits.
- Logical CDNA3 scalar operands special-case VCC encodings 106/107 to
  `wf.vcc()`, and EXEC encodings 126/127 to `wf.exec()`. The remaining Chapter 3
  register-state gaps are about raw/physical state exposure, `XNACK_MASK`,
  allocation granularity, and out-of-range semantics rather than absence of all
  logical VCC/EXEC operand handling.
- CDNA3 M0 has a dedicated wavefront field and scalar operand handling:
  encoding 124 reads and writes `wf.m0()` in the CDNA3 operand helpers, and
  `S_SET_GPR_IDX_*` updates M0 and MODE state in the shared SOPP helper. The
  current M0 gaps are about DS/LDS/GWS semantics and selector side effects, not
  total absence of M0 storage.
- CDNA3 packed workitem ID initialization for VGPR0 is present in the
  command-processor launch path on packed-TID targets: `pack_workitem_id()`
  writes X/Y/Z into v0 bitfields for CDNA3/4 and GFX11+ in the implementation. The launch-initialization gaps above are limited to TTMP4-11 and the
  TG_SIZE system SGPR payload.
- Ordinary CDNA3 PC-relative branches are modeled: `S_BRANCH` and the SCC,
  VCC, and EXEC conditional branches sign-extend the 16-bit label, scale it by
  four bytes, and update `wf.pc`; `S_CBRANCH_VCC*` masks VCC to wave size, and
  `S_CBRANCH_EXEC*` reads live EXEC in the implementation.
- For default encodings, CDNA3 direct PC helper instructions are mostly modeled:
  `S_GETPC_B64` writes the next PC, `S_SETPC_B64` jumps to the scalar-pair
  address, and `S_SWAPPC_B64` writes the next PC to SDST while jumping to SSRC0
  in the implementation. `CDNA3-RJ-085` records the separate default-only literal-size
  acceptance issue, and `CDNA3-RJ-128` records the zero-target pre-execution
  halt special case.
- `S_ENDPGM` and `S_ENDPGM_SAVED` terminate the wave in rocjitsu through
  `wf.end()`, matching the base Chapter 4.1 termination behavior. The
  remaining saved-context signal side effects overlap with the broader
  trap/context-save gaps.
- Rocjitsu's classic `S_BARRIER` path does implement the core multi-wave
  scheduler release: the direct command-processor path creates every wave in a
  workgroup together and registers the expected count in the implementation, the SPI path mirrors the same all-waves placement in the implementation,
  `Wavefront::halt()` decrements the workgroup refcount in the implementation; and
  `ComputeUnitCore::update_wf_states()` releases barrier waves only when every
  non-halted wavefront in the same dispatch/workgroup is also at `BARRIER` in the implementation.
- `S_BARRIER` does not alter wait-counter targets or wait for counters to
  drain in the audited helper; this matches the CDNA3 manual's statement that
  barriers do not wait for counters to become zero before issuing. Explicit
  `S_WAITCNT` behavior remains separate in the generated SOPP wait-counter
  helper in the implementation.
- The CDNA3 `S_WAITCNT` immediate decode matches the XML wait-count bit layout:
  rocjitsu combines `VM` low bits with `VM_HI`, extracts `EXP` and `LGKM`, and
  calls `wf.set_wait_target(vm, lgkm, exp)` in the implementation. The Chapter 4.4 gaps above are about producer accounting, not this
  threshold decode.
- Generated CDNA3 SOPP constructors and the decoder table cover the manual
  opcode inventory for opcodes 0 through 29 in the implementation. `CDNA3-RJ-087` records the separate XML-only opcode 30 and
  31 exposure; `CDNA3-RJ-054`, `CDNA3-RJ-059`, `CDNA3-RJ-061`,
  `CDNA3-RJ-063`, `CDNA3-RJ-075`, and `CDNA3-RJ-087` cover the remaining
  trap/debug/status/perf/trace/barrier/GPR-index/XML-only SOPP behavior gaps.
- `HookOrderingTest.BarrierTwoWaves` gives a basic end-to-end check that a
  two-wave workgroup reaches and resolves one barrier event before both waves
  halt.
