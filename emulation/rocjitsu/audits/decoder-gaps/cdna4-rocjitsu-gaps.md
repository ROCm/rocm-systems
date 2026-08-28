# CDNA4 Rocjitsu Gaps

Architecture: CDNA4

Audit snapshot: CDNA4 manual `workspace_docs/amdgpu-isa-manuals/cdna4/README.md`,
CDNA4 XML `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml`, and rocjitsu
source at `0e5333219682f52daf40713a4e6096fe2c78fc32` (rebased on
`origin/develop` `f226accd066d23a22339d10d103e4e0647d99f03`). The 2026-08-28
refresh was performed in manual subsection order before reconciling stable IDs.

Refresh note: the following prior IDs are resolved in the audited snapshot and
are no longer active gaps: `CDNA4-RJ-014`, `CDNA4-RJ-015`, `CDNA4-RJ-016`,
`CDNA4-RJ-018`, `CDNA4-RJ-027`, `CDNA4-RJ-120`, `CDNA4-RJ-121`,
`CDNA4-RJ-123`, and `CDNA4-RJ-125`. Exact current evidence is recorded in
**Resolved since the previous audit** below. Their older finding bodies remain
only in Git history; the resolution ledger preserves the stable-ID mapping.

## Gaps

Active findings after refresh: 117 (`P0`: 4, `P1`: 14, `P2`: 39, `P3`: 38,
`P4`: 22).

Previously reported gaps now verified as non-gaps because of fixes in the
audited snapshot: 9. Historical findings reviewed: 126 (117 active + 9 now
non-gaps).

Ordered for triage: incorrect legal-code semantics and execution behavior rank first, weighted by path frequency; state/runtime gaps and missing execution support follow; decoder/legality, metadata/fidelity, permissiveness, and test-only findings are later. Finding IDs remain stable.

## P0 — Critical correctness

### CDNA4-RJ-069: Allocated SGPR/VGPR range semantics and destination nullification are not modeled

ISA contract: CDNA4 sections 3.6.1, 3.6.2, 3.6.4, 6.2.3, and 8.4 define register range behavior from the wave's allocated SGPR/VGPR sizes. An out-of-range scalar source uses SGPR0; an out-of-range vector source uses VGPR0. An out-of-range scalar destination suppresses only the SGPR write, so SCC and saveexec's EXEC result still update. An out-of-range vector destination suppresses the instruction; if an instruction has multiple destinations, none are written when any destination is out of range. Memory and returning-atomic operations must check the complete return VGPR span and nullify the operation by issuing with EXEC cleared when any return register is out of range.

Current rocjitsu: `Wavefront::{sgpr_alloc,vgpr_alloc}` retain allocation counts, but `resolve_src_scalar*` and `resolve_dst_write*` add ordinary scalar selectors directly to `sgpr_alloc().base`, and `Operand::{read_lane_exec,write_lane_exec,read_lane64_exec,write_lane64_exec}` similarly add resolved vector offsets directly to `vgpr_alloc().base`; none compare the logical register span with the allocation count. The SIMD/chunk accessors follow the same direct-base pattern. Generated vector-memory bodies populate `VectorMemState::dst_reg_base`, and `vector_complete` and DS dual-return completion write each destination VGPR without an allocation-span preflight. Ordinary ALU helpers likewise write destinations independently rather than applying an instruction-wide nullification plan.

Impact and triage: out-of-range operands can access a neighboring physical allocation instead of SGPR0/VGPR0 or no-write behavior. Scalar destination failure is not separated from SCC/EXEC side effects, and wide, multi-destination, memory-load, and returning-atomic results can be partially written rather than wholly nullified. Centralize allocation-aware source resolution and destination-span planning before execution or memory issue, while preserving architecturally independent SCC/EXEC updates.

### CDNA4-RJ-028: `S_BUFFER_*` SMEM ignores buffer resource descriptor and bounds semantics

Manual evidence:

- Section 8.2.1.1 says scalar buffer memory uses a four-SGPR resource
  descriptor containing base address, stride, `num_records`, and `NV`; stride is
  used only for bounds checking and not for address calculation in the cited manual passage.
- Section 8.4 says `SBASE` must be even for `S_BUFFER_LOAD` and out-of-bounds
  dwords are clamped by not performing those memory operations in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 scalar-buffer loads and stores construct a 128-bit `SBASE`
  operand, but still call `smem_calculate_address()` like raw-pointer scalar
  memory operations in the implementation.
- The shared scalar address helper reads only an SGPR pair as a 64-bit base and
  returns base plus offset in the implementation.
- `ScalarMemState` carries only an address, width, data buffer, memory type,
  wait-counter type, and load/store flag, with no buffer descriptor or
  per-dword bounds state in the implementation.

Impact:

CDNA4 `S_BUFFER_*` instructions use raw-pointer-style addressing instead of the
manual's scalar-buffer descriptor semantics, and rocjitsu cannot suppress only
the out-of-bounds dwords of a buffer scalar-memory access.

### CDNA4-RJ-032: Buffer descriptor addressing and range checking are incomplete

Manual evidence:

- Section 9.1.5 defines buffer addresses from resource base, SGPR offset, VGPR
  offset/index, stride, element size, `ADD_TID`, swizzle state, and `NumRecords`
  in the cited manual passage.
- Section 9.1.5.1 defines private, raw, and structured range-checking modes and
  the `dst_sel = SEL_1` OOB read exception in the cited manual passage.
- Section 9.1.8 defines the full 128-bit descriptor layout and says an all-zero
  resource acts as an unbound buffer returning zero and dropping writes in the cited manual passage.

Rocjitsu evidence:

- `mubuf_calculate_addresses()` reads base, 14-bit stride, `NumRecords`, and a
  single `oob_raw` bit from the descriptor in the implementation.
- Its range checks use `oob_raw`, `stride`, `index`, and `offset_part`, but do
  not model `ADD_TID`, swizzle enable, 18-bit stride extension, private-scratch
  no-range-check mode, descriptor type/user-VM/NV/reserved bits, unbound
  all-zero behavior, `dst_sel = SEL_1`, or all-or-nothing versus per-component
  distinctions in the implementation.
- `mtbuf_calculate_addresses()` follows the same reduced descriptor model in the implementation.

Impact:

Basic linear buffer cases can execute, but descriptor-driven scratch/private,
swizzled, unbound, and several OOB/read-channel cases can diverge from CDNA4
hardware semantics.

### CDNA4-RJ-045: LDS M0 clamping, allocation granularity, and bank behavior are not modeled

Manual evidence:

- Section 11.1 describes CDNA4 LDS as 64 banks and says bank conflicts serialize
  indexed and atomic operations in the cited manual passage.
- Section 11.3.1 says all LDS operations require `M0` initialization and that
  `M0[16:0]` contains the LDS segment byte-size used to clamp final addresses in the cited manual passage.
- Section 3.6.5 says CDNA4 LDS allocation uses contiguous 1280-byte blocks on
  1280-byte alignment, and clamping uses the smaller of the SPI allocation size
  and `M0` in the cited manual passage.

Rocjitsu evidence:

- The shared DS address helper computes `VGPR[ADDR] + offset + lds_base` and
  never reads `wf.m0()` for normal DS operations in the implementation.
- Representative CDNA4 DS generated bodies delegate to that helper for ordinary
  loads/stores/atomics, for example `DS_ADD_U32` in the implementation.
- The CU LDS allocator aligns allocations to 256 bytes, not CDNA4's documented
  1280-byte granularity, in the implementation.
- The LDS backing and local-memory pipeline implement functional OOB read-zero
  and write-drop behavior against total backing size in the implementation and vector
  loads/stores, but do not model per-wave
  `min(lds_size, M0)` clamping or bank-conflict serialization.

Impact:

CDNA4 LDS accesses can address outside the wave/workgroup allocation and still
hit another modeled LDS region if they remain inside the CU backing. Residency,
OOB behavior, and timing-sensitive bank-conflict behavior can differ from the
manual.

## P1 — High-priority correctness

### CDNA4-RJ-034: Buffer `SOFFSET` and dword-alignment edge cases are not modeled

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

### CDNA4-RJ-039: FLAT/GLOBAL/SCRATCH address-mode and aperture behavior is incomplete

Manual evidence:

- Chapter 10 says flat addresses are routed by aperture registers across
  video/system memory, LDS, and scratch, and unmapped regions generate memory
  violations in the cited manual passage.
- Section 10.3 says FLAT supports 32-bit and 64-bit addressing selected by
  `PTR32`, and defines private-aperture scratch address conversion in the cited manual passage.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH address component size depends
  on `ADDRESS_MODE`, GLOBAL must not access LDS, and SCRATCH uses swizzled
  `FLAT_SCRATCH` addressing with unsigned byte offsets in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 FLAT constructors fix ordinary flat addresses as 64-bit VGPR
  pairs, narrow scratch to a 32-bit VGPR offset, and narrow global to a 32-bit
  VGPR offset only when `SADDR != 0x7f`; representative constructor code is in the implementation.
- The shared flat address helper uses fixed address-size formulas for FLAT,
  GLOBAL, and SCRATCH and does not consult visible `PTR32` or `ADDRESS_MODE`
  state in the implementation.
- The helper maps private-aperture FLAT addresses using a high-32-bit match and
  scratch lane stride, but does not model aperture limits, aperture holes, or
  the manual's scratch swizzle formula in the implementation.
- The compute-unit router converts the first active lane's shared-aperture
  address to LDS and `LGKMCNT`, but it has no memory-violation path for GLOBAL
  accesses that resolve to LDS or for unmapped aperture ranges in the implementation.

Impact:

Rocjitsu can run common flat/global/scratch addressing cases, but
mode-dependent address widths, scratch swizzling, aperture bounds, and
memory-violation behavior can diverge from CDNA4.

### CDNA4-RJ-033: Buffer format conversion and `dst_sel` semantics are missing

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and buffer data-format
  conversion in the cited manual passage.
- Section 9.1.4 says MTBUF takes format from the instruction, formatted MUBUF
  takes format and `dst_sel` from the resource, raw MUBUF derives size/type
  from the opcode, INVALID resource format remains unbound, and D16 variants
  pack/load/store 16-bit values in the cited manual passage.
- Chapter 13.5 lists MTBUF `DFMT`/`NFMT` values and MUBUF format/D16 opcodes in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 MUBUF formatted load/store bodies throw `UnimplementedInst`
  for representative non-D16 formatted operations in the implementation and for representative D16 formatted operations.
- Generated CDNA4 MTBUF formatted loads and stores use fixed 4-byte element
  sizes and raw VGPR payload transfers rather than `DFMT`/`NFMT` conversion;
  representative load/store bodies are in the implementation.
- Generated CDNA4 MTBUF D16 paths set fixed two-byte elements and D16
  writeback flags, but still do not derive typed conversion behavior from
  `DFMT`/`NFMT` in the implementation.
- Python semantic derivation maps MTBUF format mnemonics to fixed element
  counts and classifies unmatched MUBUF format mnemonics as `nop` in the code generator.

Impact:

Rocjitsu does not yet model CDNA4 typed-buffer conversion, resource-derived
formatted MUBUF conversion, destination-channel selection, INVALID/unbound
format behavior, or D16 formatted packing.

### CDNA4-RJ-072: `S_MAX_{I32,U32}` clears SCC for equal operands

Manual evidence:

- The detailed CDNA4 `S_MAX_I32` and `S_MAX_U32` definitions set SCC with
  inclusive predicates, `S0 >= S1`, in the cited manual passage.

Rocjitsu evidence:

- Shared execution for `S_MAX_I32` and `S_MAX_U32` writes SCC with strict
  `s0 > s1` in the implementation.
- The scalar SCC test currently locks in SCC=false for equal signed and
  unsigned max operands in the relevant tests.
- The semantic-derivation test likewise requires strict `s0 > s1` and rejects
  `s0 >= s1` in the codegen tests.

Impact:

rocjitsu produces the right max value for equal inputs but the wrong SCC value,
and current tests protect that behavior.

### CDNA4-RJ-093: FP min/max helpers do not model NaN, signed-zero, or IEEE/MODE tie rules

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define signaling-NaN quieting under
  `WAVE_MODE.IEEE`, NaN operand selection, signed-zero tie selection, and
  `V_MAX_F32` IEEE versus non-IEEE equality behavior in the cited manual passage.
- `V_MAX_F16` and `V_MIN_F16` define corresponding F16 NaN and signed-zero rules
  in the cited manual passage.
- `V_MIN_F64` and `V_MAX_F64` repeat the same style of signaling-NaN, NaN
  operand-selection, signed-zero, and IEEE-mode equality rules for F64 in the cited manual passage.

Rocjitsu evidence:

- Shared F32 VOP2/VOP3 min/max helpers use `util::stdx::fmax`/`fmin` in the SIMD
  path and host `std::fmax`/`std::fmin` in the scalar path, with no
  `WAVE_MODE.IEEE` handling or manual operand-selection logic, in the implementation.
- Shared F16 VOP2/VOP3 min/max helpers convert through F32 and likewise use
  `std::fmax`/`std::fmin` in the implementation.
- Shared F64 VOP3 min/max helpers use `util::stdx::fmax`/`fmin` in the SIMD
  path and host `std::fmax`/`std::fmin` in the scalar path, with no
  `WAVE_MODE.IEEE` handling or manual operand-selection logic, in the implementation.
- The SIMD correctness test documents that NaN-input lanes and signed-zero ties
  are excluded from comparison because scalar and SIMD paths may diverge there in the relevant tests.

Impact:

Finite non-tie min/max cases have coverage, but rocjitsu does not pin or emulate
the ISA's edge selection rules for NaNs, signaling NaNs, signed-zero ties, or
IEEE-mode equality predicates.

### CDNA4-RJ-078: VOP3 floating output modifiers ignore MODE denorm/IEEE gating

MODE is now backed and writable, but VOP3 floating execution still applies OMOD and clamp without the ISA-required IEEE/denormal gating or associated flush-to-zero and negative-zero behavior.

### CDNA4-RJ-079: VALU round/denorm modes and DOT2 forced denormal flushing remain unimplemented

rocjitsu now stores writable MODE state, but VALU floating helpers generally ignore its rounding and denormal fields; the SOPP mode setters are stubs and DOT2 execution lacks its instruction-specific forced denormal flushing.

### CDNA4-RJ-080: ALU clamp non-FP semantics are incomplete

Manual evidence:

- Chapter 6.5 says the VOP3 clamp bit signals FP exceptions for `V_CMP`,
  saturates integer results to the representable extrema, and clamps
  floating-point results to `[0.0, 1.0]` in the cited manual passage.

Rocjitsu evidence:

- Shared `execute_v_add_u32_vop3()` ignores `inst.inst_.clamp` and performs
  ordinary wrapping integer addition in the implementation.
- The same wrapping behavior is visible in the Chapter 12.11 signed add/sub
  forms: `execute_v_add_i32_vop3()` and `execute_v_sub_i32_vop3()` ignore
  `inst.inst_.clamp` in the implementation, while generated
  `VAddI16Vop3` and `VSubI16Vop3` write wrapped true16 results in the implementation.
- Shared compare helpers such as `execute_v_cmp_eq_f32_vop3()` also ignore
  `inst.inst_.clamp`, so they cannot model clamp-as-FP-exception signaling in the implementation.
- The floating helper cited in `CDNA4-RJ-078` implements only the
  floating-point `[0,1]` clamp case.

Impact:

rocjitsu handles ordinary floating-point clamp but not the manual's integer
saturation or compare exception-signaling overloads.

### CDNA4-RJ-117: Promoted VOP3 conversion aliases drop conversion modifiers

Manual/XML evidence:

- Chapter 12.8 says VOP1 instructions may also be encoded as VOP3 to access
  extra control bits such as `ABS` and `OMOD` in the cited manual passage; the
  later VOP3A section repeats the same promoted VOP1 conversion rows in the
  VOP3 opcode table in the cited manual passage.
- Chapter 6.2.1 says VOP3 floating-point inputs can use `ABS`/`NEG`, and
  Chapter 6.2.2 says VOP3 instructions with floating-point results can use
  `OMOD` and `CLAMP`, subject to MODE restrictions, in the cited manual passage.
- XML records promoted `ENC_VOP3` conversion forms with floating inputs or
  floating results, for example `V_CVT_F32_I32`, `V_CVT_U32_F32`,
  `V_CVT_I32_F32`, `V_CVT_F16_F32`, `V_CVT_F32_F16`, and `V_CVT_F32_F64`, in the machine-readable ISA XML.

Rocjitsu evidence:

- Some promoted conversion helpers with floating inputs read the raw source and
  ignore `ABS`/`NEG`, such as `execute_v_cvt_i32_f32_vop3()` and
  `execute_v_cvt_u32_f32_vop3()` in the implementation.
- Promoted conversion helpers with floating results write the converted value
  directly and do not apply `OMOD` or `CLAMP`, including
  `execute_v_cvt_f32_f64_vop3()`, `execute_v_cvt_f32_i32_vop3()`,
  `execute_v_cvt_f32_u32_vop3()`, `VCvtF16F32Vop3::execute_impl()`, and
  `VCvtF32F16Vop3::execute_impl()` in the implementation.
- `execute_v_cvt_off_f32_i4_vop3()` does apply VOP3 `OMOD` and `CLAMP` in the implementation, showing the omission is not an
  intentional blanket rule for all conversion-style helpers. Existing
  `CDNA4-RJ-078` covers MODE gating when modifiers are applied; this finding
  covers promoted conversion paths where the modifiers are not applied at all.

Impact:

Promoted VOP3 conversion encodings can ignore legal source modifiers for
floating inputs, and can ignore legal output modifiers for floating results.
That makes the VOP3 aliases semantically weaker than the manual's advertised
extra-control-bit form.

### CDNA4-RJ-009: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA4 VOP3P format includes `CLMP` as "1 = clamp result" in the cited manual passage.
- The CDNA4 XML describes generic VOP3P `CLAMP` as clamping output to
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

### CDNA4-RJ-133: `V_FRACT_F32` and `V_FRACT_F64` omit the maximum-below-one clamp

CDNA4 requires fractional results to be clamped to `0x3f7fffff` for F32 and
`0x3fefffffffffffff` for F64. The generated scalar and SIMD paths return only
`x - floor(x)`. At rounding edges where that subtraction produces exactly one,
rocjitsu returns an out-of-contract value instead of the largest representable
value below one.

### CDNA4-RJ-138: `V_MAD_MIX_*` performs separate multiply and add instead of fused FMA

CDNA4 explicitly defines `V_MAD_MIX_F32`, `V_MAD_MIXLO_F16`, and
`V_MAD_MIXHI_F16` as fused multiply-add operations. Their scalar and SIMD
helpers evaluate `a * b + c`, allowing the intermediate product to round before
the addition. Inputs near a rounding boundary can therefore differ by an ULP
from the required fused result before any F16 destination conversion.

### CDNA4-RJ-130: A lone `work_group_id0` payload receives a linearized ID

The launch ABI defines enabled `work_group_id0` as the X coordinate. For a
multidimensional dispatch, rocjitsu writes the flattened global workgroup index
whenever the Y and Z payloads are disabled, even if X is enabled. A 2D or 3D
kernel requesting only the X payload therefore observes a linearized ID rather
than its X coordinate.

### CDNA4-RJ-026: Section 7.4 MFMA floating-point mode and DGEMM exceptions are not modeled

Manual evidence:

- Section 7.4 says MAI denormal handling varies by datatype and sometimes by
  MODE in the cited manual passage.
- `V_MFMA_F32_*_F32` honors MODE denormal flags for 32-bit inputs, while
  matrix-C input and result output ignore `MODE.denorm` and preserve denormals
  in the cited manual passage.
- Sub-32-bit float MFMA inputs ignore `MODE.denorm` and preserve denormals;
  `V_MFMA_F64_*_F64` ignores MODE, rounds RNE, and allows input/output denorms;
  and `V_MFMA_I32_*_I8` ignores MODE because it is integer arithmetic with
  sign-extended intermediate values in the cited manual passage.
- The matrix core does not support arithmetic exceptions except for DGEMM matrix
  operations, which do support exceptions, in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state, but the generated CDNA4 MFMA call sites do
  not read it or pass it into helpers. Representative F32-input, I8, and F64
  generated paths pass only register bases, constant-accumulator state, and
  MFMA encoding fields in the implementation.
- The shared `exec_f32()` wrapper and F32 fast path accept no MODE or
  per-operand denorm policy, and use host/SIMD floating arithmetic through the
  generic MFMA helpers in the implementation.
- The F64 helper uses host floating arithmetic, with a scalar multiply/add loop
  and a SIMD fused-FMA path, and accepts only register bases,
  constant-accumulator state, and F64 negation bits. It does not explicitly
  force GPU RNE or record arithmetic exceptions in the implementation.
- `Wavefront` exposes raw MODE storage in the implementation, but
  shared `s_denorm_mode`, `s_round_mode`, and `v_clrexcp` execution helpers are
  empty stubs in the implementation.

Impact:

F32-input MFMA currently cannot vary A/B denormal handling with GPU MODE while
preserving matrix-C/result denormals independently. DGEMM arithmetic exception
support is also absent: F64 MFMA can produce numeric results through host FMA,
but rocjitsu has no visible exception flagging or trap-facing state for the
manual's DGEMM exception contract. The I8 path naturally ignores MODE because
it is integer arithmetic; the missing pieces are the floating-point state and
exception semantics.

## P2 — Specialized correctness and important missing support

### CDNA4-RJ-124: Zero-target `S_SETPC_B64` and `S_SWAPPC_B64` halt instead of branching

CDNA4 defines both instructions as ordinary PC transfers from the source SGPR
pair and assigns no special termination meaning to a zero target. Before the
generated instruction executes, rocjitsu checks these mnemonics and calls the
wave halt path whenever the resolved target is zero. A legal branch or swap to
address zero therefore terminates the wave instead of transferring control.

### CDNA4-RJ-035: Buffer-to-LDS subset, M0 offset, and clamping are incomplete

Manual evidence:

- Section 9.1.9 says load-to-LDS is supported only for
  `BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,dwordX3,dwordX4,format_x}`,
  defines `LDS_offset = M0[17:0]`, uses `TIDinWave * 16` for 3- and 4-dword
  loads, and requires active-mask clamping so return data is not written
  outside the LDS allocation for the wave in the cited manual passage.

Rocjitsu evidence:

- Allowed raw byte/short/dword loads do implement an `inst_.lds` path, but use
  `wf.m0() + wf.lds_base()` as the base in the implementation.
- The same `inst_.lds` pattern is also generated for `buffer_load_dwordx2` even
  though that form is outside the manual's listed LDS subset in the implementation.
- The D16 raw load paths also accept `inst_.lds` and use the full `M0` value
  for the LDS base in the implementation.
- The vector-memory completion path writes LDS-destination loads
  `lds_base + lane * per_lane_bytes` when no per-lane LDS address is present in the implementation. The buffer path does not derive the manual's allocation-aware active
  mask before issuing the memory read.

Impact:

Rocjitsu can accept LDS forms the manual does not list, use high bits of M0 in
the LDS offset, use a simplified lane-to-LDS address formula, and issue reads
for lanes whose return data should be masked by LDS allocation clamping.

### CDNA4-RJ-073: `S_MOVRELS` / `S_MOVRELD` scale M0 by operand width

Manual evidence:

- The Chapter 5 summary defines `MOVERELS: D = SGPR[S0+M0]` and
  `MOVERELD: SGPR[D+M0] = S0`, says M0 is an unsigned index, and requires an
  even index for 64-bit forms in the cited manual passage.
- The detailed B64 definitions repeat the raw `addr += M0.u32[31:0]` formula
  and evenness requirement in the cited manual passage.

Rocjitsu evidence:

- CDNA4 `S_MOVRELS_B32` / `S_MOVRELS_B64` compute `src_reg` as
  `ssrc0.encoding_value() + index * width_words`, so B64 doubles the M0 index,
  in the implementation.
- `S_MOVRELD_B32` / `S_MOVRELD_B64` compute `dst_reg` with the same
  `index * width_words` pattern in the implementation.
- These paths mask M0 to 8 bits and do not validate the manual's evenness
  requirement for the 64-bit source/destination index.

Impact:

CDNA4 B64 relative scalar moves address different SGPR pairs than the manual
pseudocode for nonzero M0 values, and invalid odd-index cases are not caught.

### CDNA4-RJ-102: VOP3 `V_CMPX_CLASS_*` writes VCC instead of the explicit SDST

Manual evidence:

- `V_CMPX_CLASS_F32` says the result is stored into EXEC and to VCC or a scalar
  register in the cited manual passage; F64 and F16 class-CMPX
  definitions repeat the same contract.

Rocjitsu evidence:

- The generated VOP3 class-CMPX constructor exposes an explicit `OPR_SDST`
  destination, for example `VCmpxClassF32Vop3` in the implementation.
- Its execute body computes a result mask, then calls `wf.set_vcc(result)` and
  `wf.set_exec(result)` without writing `vdst` in the implementation.
- Ordinary relational VOP3 CMPX helpers write the explicit scalar destination
  before updating EXEC, such as `VCmpxEqF32Vop3::execute_impl()` in the implementation.

Impact:

VOP3 class-CMPX encodings with an explicit scalar destination can leave that
destination stale and clobber VCC instead, diverging from the VOP3 destination
contract and from neighboring CMPX compare helpers.

### CDNA4-RJ-127: `V_PERMLANE16_SWAP_B32` overwrites EXEC-disabled lanes

Both the compact and VOP3 forms read and write all 64 lanes while swapping the
two 16-lane halves of each 32-lane region. Neither write loop checks `EXEC`, so
inactive destination lanes are overwritten even though ordinary VALU writes
must preserve them. Sparse-EXEC programs can therefore corrupt values in lanes
that should remain untouched.

### CDNA4-RJ-128: SMEM loads targeting `VCC` write an SGPR backing slot instead

CDNA4 permits `SDATA=VCC` for scalar-memory loads. Generated loads convert the
raw selector into a physical SGPR index, and scalar-memory completion writes
that location directly rather than updating wave `VCC` state. A legal load to
`VCC` thus leaves later VCC consumers seeing stale data while an unrelated SGPR
backing location receives the result.

### CDNA4-RJ-134: Packed FP8/BF8 low-half conversions clear the preserved half

`V_CVT_PK_FP8_F32` and `V_CVT_PK_BF8_F32` write the 16-bit half selected by
`OPSEL` and require the other destination half to be preserved. The generated
low-half path explicitly requests CDNA low-half zeroing from the shared true16
writer. A low-half conversion therefore clears `VDST[31:16]`, destroying live
data that the instruction must retain.

### CDNA4-RJ-046: `DS_*_ADDTID_B32` uses the wrong address formula

Manual evidence:

- `DS_WRITE_ADDTID_B32` stores to
  `{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4` in the cited manual passage.
- `DS_READ_ADDTID_B32` uses the same formula for the load address in the cited manual passage.

Rocjitsu evidence:

- Current generated CDNA4 `DS_WRITE_ADDTID_B32` calls the generic
  `ds_calculate_addresses()` helper in the implementation, so it reads the encoded `ADDR` VGPR instead of using ADDTID's
  lane-based formula.
- Current generated CDNA4 `DS_READ_ADDTID_B32` does use a special path, but it
  computes `lane * (((M0 >> 16) & 0x1ff) * 4) + offset + lds_base` in the implementation, not `{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4`.
- The generator's ADDTID templates contain the same high-M0 stride formula for
  both read and write in the code generator, so a regeneration would not recover the CDNA4 manual formula.

Impact:

CDNA4 ADDTID reads and writes can access completely different LDS addresses from
hardware. The write form is currently generic-DS addressing, while the generated
special form would still use the wrong M0 bit slice and lane scaling.

### CDNA4-RJ-047: DS READ2/WRITE2 duplicate-offset collapse is not modeled

Manual evidence:

- Section 11.3.1 says setting both two-address offsets to the same value
  specifies only one address, causes only one read/write, and uses only `DATA0`
  in the cited manual passage.

Rocjitsu evidence:

- `DsRead2B32Ds::execute_impl()` always sets `ds2_active = true`, computes both
  addresses from `offset0` and `offset1`, and sets a second destination register
  in the implementation.
- The local-memory pipeline always issues the second DS2 load when
  `ds2_active` is true and writes the second response during completion in the implementation.
- Generated WRITE2 bodies follow the same unconditional second-access pattern.

Impact:

Equal-offset READ2/WRITE2 encodings perform two modeled accesses in rocjitsu
instead of the manual's one-access form. Stores can use `DATA1` where hardware
should ignore it, and reads can write a second destination value where only one
access should occur.

### CDNA4-RJ-048: `DS_SWIZZLE_B32` misses FFT and rotate modes

Manual evidence:

- `DS_SWIZZLE_B32` supports FFT mode for offsets `>= 0xe000`, rotate mode for
  offsets `>= 0xc000 && < 0xe000`, quad mode, and 32-lane bit-mask mode, with
  invalid-thread reads returning zero in the cited manual passage.

Rocjitsu evidence:

- The shared generated helper only branches on `offset & 0x8000`: bit set uses
  quad selectors, bit clear uses the 32-lane bit-mask formula in the implementation.
- CDNA4 `DsSwizzleB32Ds::execute_impl()` calls that helper directly in the implementation.
- The generator profile test only asserts source selection and the quad
  selector expression, not FFT or rotate behavior, in the codegen tests.

Impact:

CDNA4 swizzle offsets in the FFT and rotate ranges execute as quad-mode
swizzles in rocjitsu, so any shader using those documented forms gets incorrect
cross-lane values.

### CDNA4-RJ-049: Packed F16/BF16 LDS atomics execute through the scalar F32 atomic path

Manual evidence:

- Chapter 9.2 says LDS packed F16/BF16 atomics have MODE-dependent denormal
  handling and RNE rounding in the cited manual passage.
- Section 12.12 defines `DS_PK_ADD_F16`, `DS_PK_ADD_BF16`, and their return
  variants as packed two-component 16-bit additions in the cited manual passage.

Rocjitsu evidence:

- CDNA4 packed no-return and return forms set `elem_size = 4` and
  `atomic_op = AtomicOp::FADD`, the same operation used for scalar F32 atomics,
  in the implementation.
- `execute_lds_atomic_rmw()` treats `elem_size == 4` floating atomics as one
  `float` and applies ordinary C++ `+`, `fmin`, or `fmax` without packed
  half/BF16 lane handling or MODE denormal policy in the implementation.

Impact:

Packed LDS F16/BF16 atomics are type-incorrect: rocjitsu interprets the 32-bit
word as scalar F32 instead of two packed 16-bit lanes and cannot model the
manual's FP-mode requirements.

### CDNA4-RJ-132: `DS_CONDXCHG32_RTN_B64` executes as ordinary compare-swap

CDNA4 defines two independent 32-bit conditional exchanges: each half writes
only when the corresponding source sign bit is set, with that sign bit cleared
in memory. Rocjitsu routes the instruction through the generic 64-bit
`CMPSWAP` operation. The compare predicate, memory updates, and returned value
therefore do not implement the legal instruction's semantics.

### CDNA4-RJ-135: LDS append/consume use the wrong address and return per-lane tickets

For LDS, `DS_APPEND` and `DS_CONSUME` address `instr_offset`, update one counter
by the number of active lanes, and return the same pre-operation value to every
active lane. Rocjitsu adds `M0` to the LDS address and synthesizes a different
ranked return value for each lane. Legal uses can update the wrong LDS word and
observe non-architectural per-lane results.

### CDNA4-RJ-136: Scalar F32/F64 LDS atomics use host floating-point rules

CDNA4 defines atomic-specific rounding, MODE-dependent denormal handling, NaN
behavior, infinities, and signed-zero results. Rocjitsu implements scalar F32
and F64 LDS add/min/max with host `+`, `fmin`, and `fmax`, without the CDNA4
edge tables or wave MODE. Legal atomic operations can therefore write different
bits from hardware for subnormal, NaN, infinity, and signed-zero inputs.

### CDNA4-RJ-139: `DS_SWIZZLE_B32` reads EXEC-disabled source lanes instead of zero

CDNA4 gates every selected swizzle source with the source lane's valid-thread
state and supplies zero when that lane is inactive. Rocjitsu snapshots all lane
values, then writes the selected value to each active destination without
testing the selected source against `EXEC`. An active lane that selects an
inactive lane therefore receives stale VGPR data instead of zero.

### CDNA4-RJ-081: VGPR indexing uses the wrong M0 layout and cannot honor source-role masks

Manual evidence:

- Chapter 6.6 defines `M0[7:0]` as the index and `M0[15:12]` as
  dest/src2/src1/src0 enable bits, with indexing applying only to VGPR
  operands and indexed out-of-range VGPRs illegal in the cited manual passage.
- Chapter 6.6.2 gives instruction-specific role remapping for readlane,
  writelane, MAC/MAD, reverse shifts, `v_cvt_pkaccum`, and SDWA
  read-modify-write in the cited manual passage.

Rocjitsu evidence:

- `Wavefront::gpr_idx_mode()` reads `(m0 >> 8) & 0xF`, and
  `execute_s_set_gpr_idx_mode_sopp()` / `execute_s_set_gpr_idx_on_sopc()` write
  the mode nibble at `M0[11:8]`, not `M0[15:12]`, in the implementation.
- `apply_gpr_idx()` applies any low source-enable bit to all source operands and
  only distinguishes destination versus non-destination in the implementation.
- CDNA4 operand reads/writes pass only a boolean source/destination role into
  that helper in the implementation, so they cannot distinguish src0, src1, src2, or the manual's
  instruction-specific remaps.

Impact:

CDNA4 indexed VGPR accesses use different M0 bits from the manual and can index
the wrong operands, while illegal indexed out-of-range accesses are not
diagnosed.

### CDNA4-RJ-083: FP8/BF8 widening converts still apply ignored SDWA fields

CDNA4 ISA Table 31 and section 7.3 say `V_CVT_F32_{FP8,BF8}` and `V_CVT_PK_F32_{FP8,BF8}` use SDWA only for source byte/word selection and ignore SEXT and destination-side controls. All four current VOP1 execute paths pass `sdwa_src0_sext_` to `sdwa::stage_source`, so sub-dword selections can still be sign-extended despite the ignored-field rule. The single-result `VCvtF32Fp8Vop1` and `VCvtF32Bf8Vop1` paths then write through `sdwa::write_lane`, which honors destination select/unused/merge controls that should also be ignored. The packed `VCvtPkF32Fp8Vop1` and `VCvtPkF32Bf8Vop1` paths use `sdwa::write_lane64`, which bypasses destination merging, so the destination-control subclaim is limited to the two single-result forms; ABS/NEG are already suppressed with `SourceModifierFormat::NONE`.

### CDNA4-RJ-087: `S_ABSDIFF_I32` uses 64-bit absolute difference instead of 32-bit wraparound

Manual evidence:

- The detailed CDNA4 `S_ABSDIFF_I32` definition first stores the 32-bit signed
  subtraction result and then negates that 32-bit result if it is negative in the cited manual passage.
- The manual examples pin the overflow edge cases, including
  `S_ABSDIFF_I32(0x80000000, 0x00000000) => 0x80000000` and
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff`, in the cited manual passage.

Rocjitsu evidence:

- The shared `execute_s_absdiff_i32_sop2()` helper widens both signed inputs to
  `int64_t` and returns the mathematical absolute difference before truncating
  to `uint32_t` in the implementation.
- For the manual's `0x80000000, 0x00000001` example, that formula computes
  `2147483649` and truncates to `0x80000001`, while the documented 32-bit
  subtract-then-negate behavior produces `0x7fffffff`.

Impact:

rocjitsu can return the wrong scalar result when the signed 32-bit subtraction
overflows before the absolute-value step. The audited overflow examples remain
nonzero, so this is primarily a destination-value mismatch rather than an SCC
predicate mismatch for those cases.

### CDNA4-RJ-094: `V_READFIRSTLANE_B32` returns zero instead of lane 0 when EXEC is disabled

Both generated execution forms default the result to zero and scan only active lanes. With EXEC=0 they must read lane 0 per the ISA, but instead write zero. DPP/SDWA encodings are now rejected and should be omitted from this finding.

### CDNA4-RJ-096: `V_PRNG_B32` execute bodies are no-ops

Manual evidence:

- `V_PRNG_B32` defines `D0.u32 = ((in << 1U) ^ (in[31] ? 197U : 0U))` and gives
  the nonzero period in the cited manual passage.

Rocjitsu evidence:

- The generated VOP1 body applies generic extension scaffolding, then never
  writes `VDST` before returning in the implementation.
- The generated VOP3 alias is a one-line no-op in the implementation.
- Existing data-type tests cover the PRNG helper itself, not the VOP1/VOP3
  instruction bodies.

Impact:

The decoded PRNG instruction leaves the destination unchanged instead of
advancing the documented LFSR sequence.

### CDNA4-RJ-099: `V_MOV_B32` and `V_MOV_B64` VOP3 aliases ignore allowed floating modifiers

Manual evidence:

- `V_MOV_B32` says floating-point modifiers are valid when the source is a
  32-bit floating-point value, and shows negation/absolute-value examples in the cited manual passage.
- `V_MOV_B64` carries the same floating-modifier note for 64-bit floating-point
  values in the cited manual passage.

Rocjitsu evidence:

- Shared `execute_v_mov_b32_vop3()` and `execute_v_mov_b64_vop3()` raw-copy the
  source bits to the destination without consulting `inst_.abs` or `inst_.neg`
  in the implementation.
- Adjacent VOP3 FP helper bodies explicitly apply `inst_.abs`/`inst_.neg` before
  arithmetic, for example in the implementation, showing the MOV aliases bypass the usual modifier path.
- The current VOP3 unary SIMD test locks in ignored `abs`/`neg` behavior for
  `v_mov_b32`, rather than the manual's modifier behavior.

Impact:

Promoted MOV encodings used as floating negation or absolute-value operations
execute as plain bit copies in rocjitsu.

### CDNA4-RJ-115: `V_LSHL_ADD_U64` treats unsupported shift counts as masked counts

Manual/XML evidence:

- `V_LSHL_ADD_U64` says the shift count must be between 0 and 4, and the notes
  say unsupported shift counts are treated as a shift of zero in the cited manual passage.
- XML records the VOP3 operand shape for opcode 520 but has no operand or
  semantic field that captures the unsupported-count-as-zero rule in the machine-readable ISA XML; see `CDNA4-XML-085`.

Rocjitsu evidence:

- Generated `execute_v_lshl_add_u64_vop3()` calls `lshl_masked()` on `SRC0`
  and `SRC1` in the implementation.
- The shared 64-bit helper masks the count with `& 63u`, not with a CDNA4
  `0..4` legality check, in the implementation.
- The SIMD codegen comment pins the same `(src0 << (src1 & 63)) + src2`
  behavior for `v_lshl_add_u64_vop3` in the code generator.

Impact:

Shift counts 5 through 63 produce real left shifts in rocjitsu, while the CDNA4
manual says unsupported counts should behave as shift zero. Kernels that rely
on the hardware fallback value can get different 64-bit results in both scalar
fallback and SIMD dispatch.

### CDNA4-RJ-126: The F64 inline `1/(2*pi)` constant is one ULP high

The CDNA4 handbook and XML specify the 64-bit selector value as
`0x3fc45f306dc9c882`. The shared scalar resolver returns
`0x3fc45f306dc9c883`. Any F64 or raw 64-bit consumer of the inline selector
therefore receives the next representable value above the architectural bit
pattern; lower-precision tests do not expose the mismatch.

### CDNA4-RJ-137: DPP floating source ABS/NEG fields are ignored

CDNA4 DPP extension words carry `SRC0_ABS`, `SRC0_NEG`, `SRC1_ABS`, and
`SRC1_NEG` for legal floating-point forms. Generated DPP constructors retain
the cross-lane control and masks but do not capture those source-modifier bits;
execution consequently permutes the unmodified values. Legal DPP arithmetic
with ABS or NEG can produce the wrong lane results.

### CDNA4-RJ-010: Scaled conversions ignore MODE-based F32 denormal control

Manual evidence:

- The cited manual passage says conversion from F32 supports MODE-based denormal
  control, while F4/F6/F8 allows denorms regardless of MODE.

Rocjitsu evidence:

- Generated scaled F32-to-FP4 code reads raw F32 sources and directly divides
  them by the decoded scale before calling the FP4 helper in the implementation.
- Generated wide F32-to-FP6 code follows the same raw-F32 pattern in the implementation.
- Generated FP6-to-F32 code multiplies the converted small-format value by the
  decoded scale and writes the result without consulting `wf.mode_raw()` in the implementation.
- `Wavefront` stores raw MODE state in the implementation, but
  the shared `s_denorm_mode` executor is currently a no-op in the implementation, and the audited CDNA4 scaled conversion bodies do not read
  MODE.

Impact:

F32 subnormal inputs and results follow host floating-point behavior regardless
of the shader MODE denormal settings the manual says should affect F32
conversion.

### CDNA4-RJ-017: Unscaled F8F6F4 MFMA C modifiers are ignored

Manual evidence:

- The detailed definitions for `V_MFMA_F32_16X16X128_F8F6F4` and
  `V_MFMA_F32_32X32X64_F8F6F4` say `NEG[1:0]` and `ABS[1:0]` must be zero,
  while `NEG[2]` and `ABS[2]` may control matrix C, in the cited manual passage.

Rocjitsu evidence:

- The generated constructors expose only `vdst`, `src0`, `src1`, and `src2`
  operands for the non-scale F8F6F4 MFMA classes in the implementation.
- Their execution paths read `SRC2` or a constant accumulator and dispatch to
  `exec_f32_mixed` without inspecting `inst_.neg` or `inst_.neg_hi` and without
  passing a C modifier in `generated/cdna4/vop3p_exec.cpp`. The scaled
  VOP3PX2 path now extracts the C modifier and passes it to
  `exec_f32_scaled_mixed`; it is not part of this remaining gap.
- `exec_f32_mixed` seeds the accumulator directly from `SRC2` or `const_acc`
  and has no `c_modifier` parameter in the implementation.
- The shared WMMA F32 path has an explicit `apply_wmma_c_modifier` pattern in the implementation and applies it in `exec_wmma_f32_mixed`,
  so this is not a missing primitive.

Impact:

Encodings that request negation or absolute value of matrix C execute as if the
modifier bits were clear.

### CDNA4-RJ-022: SMFMAC ignores sparse index-set selection from `CBSZ`/`ABID`

Manual evidence:

- For 16-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[1:0]` select one
  of four 8-bit sparse-index sets within the `SRC2` VGPR; if `CBSZ[1:0] != 0`,
  the first set is selected in the cited manual passage.
- For the later 16-bit large forms, the sparse index table narrows this to two
  sets selected by `ABID[0]`, while the 8-bit/IU8/F8 large forms ignore
  `CBSZ[1:0]` and `ABID[1:0]`, in the cited manual passage.
- The manual also says `CBSZ` and `ABID` only select the index from the VGPR
  read and do not affect source-A matrix broadcast for sparse MFMA in the cited manual passage.

Rocjitsu evidence:

- Generated SMFMAC execution derives an `idx` base from `SRC2` and calls sparse
  helpers without passing `inst_.cbsz` or `inst_.abid`; representative BF16 and
  F16 generated paths are in the implementation.
- The code generator emits the same index-base-only call shape for F32-result
  SMFMAC variants in the code generator.
- The shared sparse helpers accept only `idx_base` and use fixed bit extraction
  from the `SRC2` VGPR, with no selector argument, in the implementation.

Impact:

16-bit SMFMAC encodings that should select an alternate sparse-index set execute
with the helper's fixed/default index extraction. The large 8-bit/IU8/F8
selector-ignore rule may match the fixed extraction shape for those forms, but
the 16-bit selector contract is not represented.

### CDNA4-RJ-042: Floating flat atomics miss packed and FP-mode special cases

Manual evidence:

- Section 10.3.1 says floating-point atomics must set `SC[0]=0`, FP32 atomics
  flush denormals to zero, FP64 and FP16 atomics do not flush denormals, and
  rounding is fixed RNE in the cited manual passage.
- The FLAT/GLOBAL opcode inventory includes F32, packed F16/BF16, and F64
  floating atomics in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 floating flat atomics use `d->is_load = (inst_.sc0 != 0)`,
  so return-data mode remains executable for FP atomics; representative F32,
  packed F16, F64, and packed BF16 bodies are in the implementation.
- `FLAT_ATOMIC_PK_ADD_F16` and `FLAT_ATOMIC_PK_ADD_BF16` set
  `elem_size = 4` and use `AtomicOp::FADD`, the same scalar 32-bit floating
  atomic operation used by `FLAT_ATOMIC_ADD_F32`, in the implementation.
- The memory pipeline applies 4-byte FP atomics as scalar `float` and 8-byte
  FP atomics as scalar `double`, using ordinary `+`, `std::fmin`, and
  `std::fmax` without FP32 denormal flushing, packed F16/BF16 lanes, or
  explicit RNE handling in the implementation.

Impact:

F32/F64 atomic add/min/max have a coarse functional model, but packed F16/BF16
atomics are not type-correct and CDNA4's FP atomic return-mode, denormal, and
rounding rules are not enforced.

### CDNA4-RJ-090: Buffer floating atomics miss L2 FP numeric and packed-lane rules

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and can be
  issued as LDS, Buffer, Flat, Global, and Scratch instructions in the cited manual passage.
- Chapter 9.2 defines fixed RNE for float atomic adds, L2-specific denormal
  behavior for F32/F64/packed F16/BF16 forms, and NaN/signed-zero min/max,
  compare-swap, and add edge cases in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 MUBUF floating atomics lower to the generic memory-pipeline
  `AtomicOp::FADD`, `FMIN`, and `FMAX` operations: representative F32, packed
  F16, F64 add, F64 min/max, and packed BF16 paths are in the implementation.
- `BUFFER_ATOMIC_PK_ADD_F16` and `BUFFER_ATOMIC_PK_ADD_BF16` set
  `elem_size = 4` and `AtomicOp::FADD`, the same scalar 32-bit path as
  `BUFFER_ATOMIC_ADD_F32`, in the implementation.
- The shared L2 atomic executor treats 4-byte floating atomics as one host
  `float` and 8-byte floating atomics as one host `double`, using ordinary
  addition plus `std::fmin`/`std::fmax` in the implementation.
- `VectorMemState` carries the `AtomicOp` enum, element size, and dataflow
  fields, but not the floating-point subtype, packed-lane mode, or
  denormal/rounding policy needed by Chapter 9.2, in the implementation.

Impact:

F32/F64 buffer atomics have a coarse functional model, but rocjitsu does not
enforce CDNA4's L2 floating-atomic denormal, RNE, or NaN/signed-zero rules.
Packed F16/BF16 buffer atomics are not type-correct because their two lanes
execute as one scalar F32 operation. The rebased buffer-atomic metadata now
exposes the `VDATA` input and conditional return correctly.

### CDNA4-RJ-092: `V_DOT2C_F32_BF16` VOP3 ignores ABS-as-NEG_HI

Manual evidence:

- `V_DOT2C_F32_BF16` says `ABS[1:0]` are used as `NEG_HI[1:0]` during
  translation and that `NEG`/`ABS` input modifiers do not affect the accumulator
  source in the cited manual passage.

Rocjitsu evidence:

- The generated CDNA4 VOP3 body reads packed BF16 halves from `src0` and `src1`,
  accumulates into `vdst`, and never consults `inst_.abs` or `inst_.neg` in the implementation.
- Existing `CDNA4-RJ-079` covers DOT denormal flushing, but not this
  instruction-specific modifier remapping.

Impact:

VOP3 encodings that rely on the manual's high-half negation controls execute as
plain positive dot products in rocjitsu, so generated or decoded kernels can be
wrong even when the packed BF16 dataflow and accumulator are otherwise correct.

### CDNA4-RJ-106: `V_DOT2_F32_BF16` still applies `NEG[2]` to `S2`

Current execution correctly widens BF16 inputs, but it conditionally negates the accumulator from NEG[2]. The ISA says modifiers do not affect S2, so accumulator-modifier bits must be ignored.

### CDNA4-RJ-107: Packed F16 min/max3 ignore `DX10_CLAMP` NaN-to-zero MODE behavior

Manual evidence:

- The MODE register defines `DX10_CLAMP` as vector-ALU NaN-to-zero behavior and
  `IEEE` as signaling-NaN propagation/quieting state in the cited manual passage.
- `V_PK_MINIMUM3_F16` and `V_PK_MAXIMUM3_F16` say signaling NaNs propagate, then
  add that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for these
  operations in the cited manual passage.

Rocjitsu evidence:

- `VPkMinimum3F16Vop3p::execute_impl()` hard-codes an IEEE-style pairwise
  minimum helper that returns quiet NaN for any NaN input, then writes the F16
  result without inspecting `wf.mode_raw()` or zeroing NaNs in the implementation.
- `VPkMaximum3F16Vop3p::execute_impl()` has the same hard-coded IEEE-style NaN
  propagation and no MODE/DX10 handling in the implementation.

Impact:

When MODE.DX10_CLAMP is set, rocjitsu can preserve or produce NaN results where
hardware is documented to force NaN to zero for these packed min/max3
instructions.

### CDNA4-RJ-112: VOP3A F32 min/max3 ignore `DX10_CLAMP` NaN-to-zero MODE behavior

Manual/XML evidence:

- `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` say signaling NaNs propagate, then add
  that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for these
  operations in the cited manual passage.
- XML records the opcodes and F32 operand shapes for these two VOP3A
  instructions in the machine-readable ISA XML, but the MODE
  behavior is prose-only and recorded as
  `CDNA4-XML-081`.

Rocjitsu evidence:

- Generated CDNA4 `VMinimum3F32Vop3::execute_impl()` and
  `VMaximum3F32Vop3::execute_impl()` dispatch directly into the shared helpers
  in the implementation.
- `execute_v_maximum3_f32_vop3()` returns quiet NaN if any input is NaN and
  later applies only the VOP3 `clamp` bit in the implementation.
- `execute_v_minimum3_f32_vop3()` has the same NaN propagation and ordinary
  VOP3 `clamp` handling in the implementation.

Impact:

When MODE.DX10_CLAMP is set, rocjitsu can preserve or produce NaN results for
the scalar F32 min/max3 VOP3A instructions even though the CDNA4 manual says
NaNs are forced to zero for these operations.

### CDNA4-RJ-116: Packed F32-input VOP3A conversions ignore source modifiers

Manual/XML evidence:

- Chapter 6.2.1 says instructions using the VOP3 form with floating-point
  inputs can apply `ABS` and `NEG` to input operands in the cited manual passage,
  and the VOP3A field map carries per-source `ABS` and `NEG` fields in the cited manual passage.
- XML records native VOP3A packed conversion forms with F32 inputs for
  `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`, `V_CVT_PK_F16_F32`,
  `V_CVT_PK_BF16_F32`, `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` in the machine-readable ISA XML.

Rocjitsu evidence:

- Shared execution for `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` bit-casts raw source reads directly to F32 and never
  consults `inst_.abs` or `inst_.neg`, in the implementation.
- Generated CDNA4 bodies for `V_CVT_PK_F16_F32` and `V_CVT_PK_BF16_F32` do the
  same raw source reads before conversion in the implementation.
- `V_CVT_PKRTZ_F16_F32` now uses round-toward-zero conversion helpers; this
  finding is limited to legal source modifier bits on packed F32 inputs.

Impact:

Legal VOP3A encodings that negate or take the absolute value of the F32
conversion inputs execute as if the modifier bits were clear. That affects
integer-packed, normalized-packed, and F16/BF16 packed conversion results.

### CDNA4-RJ-053: Raw `STATUS` helper bits can drift from `EXEC` and `VCC`

Manual evidence:

- Chapter 3 defines `EXECZ` as the summary bit for zero `EXEC` in the cited manual passage.
- The `STATUS` table includes `EXECZ` and `VCCZ` bits in the cited manual passage.
- Section 3.9 says `VCCZ` is updated every time VCC is updated, including
  scalar writes to VCC, and that VCC is fully written in the cited manual passage.

Rocjitsu evidence:

- `Wavefront` stores `EXEC`, `VCC`, raw `MODE`, and raw `STATUS` as separate
  state; `set_exec()` and `set_vcc()` do not update raw `STATUS` in the implementation, while
  `IsaWavefront::status` is a separate raw register.
- CDNA status accessors expose `EXECZ` and `VCCZ` bits in the raw status layout
  in the implementation.
- Normal CDNA4 branch operands compute `VCCZ` and `EXECZ` from live `VCC` and
  `EXEC` in the implementation, and scalar special-source reads do the same in the implementation.
- `S_GETREG_B32`, however, reads the raw status register for its current
  `reg_id=1` mapping in the implementation.

Impact:

Control-flow instructions and scalar special-source operands can observe correct
live helper values while HWREG-visible `STATUS` reports stale `EXECZ`/`VCCZ`
bits. After the HWREG ID-map bug above is fixed, this still needs an explicit
status synchronization or computed-read policy. Wavefront reuse adds a second
lifecycle failure: reset clears other per-wave state but deliberately retains
the raw `STATUS` register, so a later dispatch can inherit privilege, trap,
exception, priority, or helper bits from the prior occupant of the slot.

### CDNA4-RJ-056: XNACK_MASK/TTMP dedicated state, TTMP privilege, and CDNA4 TTMP launch payloads are missing

ISA contract: CDNA4 sections 3.1 and 5.2 define XNACK_MASK as dedicated 64-bit state at scalar selectors 104/105 and TTMP0-TTMP15 at selectors 108-123. Section 3.10 permits TTMP writes only with STATUS.PRIV set; otherwise writes are ignored and reads return zero. Section 3.13 initializes TTMP4/5 to zero, TTMP6/7 to the dispatch-packet address, TTMP8/9/10 to grid X/Y/Z, and TTMP11 to the wave ID within the workgroup.

Current rocjitsu: `resolve_src_scalar*` and `resolve_dst_write*` route selectors 104/105 and 108-123 to `sgpr_alloc().base + selector`. There is no dedicated XNACK_MASK or TTMP register state and no STATUS.PRIV check around TTMP access, so shader code can read or overwrite unrelated physical SGPR storage. `CommandProcessor::init_wavefront_regs` has a property-gated gfx12 TTMP workgroup-ID path, but CDNA4 does not enable that property and has no initialization path for its TTMP4-TTMP11 launch payload.

Impact and triage: XNACK state is aliased to ordinary storage, unprivileged code can observe or modify TTMPs, and CDNA4 kernels do not receive the documented trap-temporary launch values. Add dedicated per-wave XNACK_MASK/TTMP state, enforce TTMP privilege on every scalar access width, and initialize the CDNA4 TTMP payload independently of the gfx12 ABI path.

### CDNA4-RJ-024: SMFMAC index legality, alignment, and floating-state controls are not modeled

Manual evidence:

- Sparse index pairs require `index0 < index1` and `index0 != index1` in the cited manual passage.
- The sparse MFMA state table says denorm handling ignores MODE and keeps
  denorms, clamp uses `FP16_OVFL`, rounding is forced RNE, exceptions are not
  supported, `SRC0`/`SRC1`/`VDST` VGPRs must be even-aligned, and `SRC2` is
  VGPR-only with no even-alignment requirement in the cited manual passage.

Rocjitsu evidence:

- Shared SMFMAC helpers split the raw sparse-index field into two selectors and
  use both without validating order or distinctness in the implementation.
- Generated SMFMAC execution maps `SRC0`, `SRC1`, `VDST`, and `SRC2` bases
  directly through `src_base()` / `dst_base()` without a sparse-specific
  alignment or selector-legality check; representative generated paths are in the implementation.
- The executed F32-result SMFMAC paths do not read MODE or `FP16_OVFL`, and the
  helpers perform host floating accumulation without a clamp/overflow policy.
  The I32-result SMFMAC paths are currently stubs, so their clamp behavior is
  not reachable yet.

Impact:

Invalid sparse-index pairs execute as ordinary selector pairs, under-aligned
SMFMAC source/destination bases are accepted, and sparse F32 overflow behavior
is independent of the manual's MODE/`FP16_OVFL` contract. These are
sparse-specific instances of broader MFMA legality/state gaps, with the
additional index-pair legality rule unique to SMFMAC.

### CDNA4-RJ-054: `S_SETVSKIP` and VSKIP issue-time suppression are unimplemented

ISA contract: CDNA4 MODE.VSKIP in section 3.5 and `S_SETVSKIP` in Chapter 12 require VSKIP to suppress issue of vector instruction families, including VALU/VOP, VMEM buffer/image, LDS/DS, and FLAT operations. A skipped memory instruction must not increment or decrement wait counters.

Current rocjitsu: `SSetvskipSopc::execute_impl` throws `UnimplementedInst`, and the execution harness still lists `s_setvskip` as expected unimplemented. `ComputeUnitCore::issue_instruction` decodes and executes every instruction without consulting MODE.VSKIP; vector memory then reaches `route_memory_inst` and the memory pipelines normally, so there is no common pre-issue gate capable of suppressing VALU, VMEM, LDS, and FLAT before their execution or wait-counter effects.

Impact and triage: code using the documented VSKIP alternative to control-flow branching cannot run, and merely backing the MODE bit would still allow skipped vector instructions and memory credits to execute. Implement `S_SETVSKIP`'s MODE update and one shared issue-time vector-family gate before execution and memory-pipeline accounting.

### CDNA4-RJ-062: Wave-control, message, performance, and trace side effects are mostly stubs

ISA contract: CDNA4 sections 4.1 and 4.4 plus the detailed SOPP definitions in Chapter 12 require `S_SETKILL` to terminate the wave when SIMM16[0] is 1, `S_SETHALT` to update HALT with its privilege rule, `S_SLEEP` to honor the encoded approximate duration, `S_WAKEUP` to wake sleeping peers in the workgroup, and `S_SETPRIO` to update user wave priority. `S_SENDMSG` sends an upstream message and consumes an LGKM credit until sent; `S_SENDMSGHALT` additionally halts the wave. `S_INCPERFLEVEL` and `S_DECPERFLEVEL` update the selected performance counter, and `S_TTRACEDATA` emits M0 to thread trace.

Current rocjitsu: the shared `execute_s_wakeup_sopp`, `execute_s_setkill_sopp`, `execute_s_sethalt_sopp`, `execute_s_setprio_sopp`, `execute_s_sendmsg_sopp`, `execute_s_sendmsghalt_sopp`, `execute_s_incperflevel_sopp`, `execute_s_decperflevel_sopp`, and `execute_s_ttracedata_sopp` helpers are empty. `execute_s_sleep_sopp` only calls `request_functional_yield`, without encoded duration or wakeup interaction. Consequently messages create no LGKM credit, SENDMSGHALT does not halt, and no kill, HALT, priority, performance-counter, or trace state is updated.

Impact and triage: these instructions decode but cannot reproduce architecturally visible wave lifecycle, scheduling, message/wait, performance, or trace behavior. Model the visible state transitions first, including message completion credits and halt/kill semantics; keep exact scheduler timing as a separable fidelity layer around sleep duration and priority arbitration.

## P3 — Missing state, rare support, and fidelity

### CDNA4-RJ-055: Trap, exception, and TRAPSTS state is not modeled

Manual evidence:

- Section 3.10 defines TTMP privilege, trap payload packing into
  `TTMP0`/`TTMP1`, `STATUS.TRAP_EN`, and `MODE.EXCP_EN` in the cited manual passage.
- Section 3.10.1 defines sticky `TRAPSTS` fields including `EXCP`, `SAVECTX`,
  `ILLEGAL_INST`, address-watch bits, `EXCP_CYCLE`, and `DP_RATE` in the cited manual passage.
- Section 3.11 defines sticky `TRAPSTS.mem_viol`, memory-violation trap enable,
  and imprecise saved PC behavior in the cited manual passage.

Rocjitsu evidence:

- A static source search finds no modeled `TRAPSTS`, `mem_viol`, `SAVECTX`,
  `ILLEGAL_INST`, `EXCP_CYCLE`, or `DP_RATE` state in rocjitsu runtime code.
- CDNA4 `S_TRAP` now has no generated branch or program-terminator flag and
  dispatches to the empty shared `execute_s_trap_sopp()` helper in the implementation.
- The shared `S_RFE_B64` executor is empty in the implementation, and `S_SENDMSG` is also a stub.
- The shared `V_CLREXCP` VOP1/VOP3 helpers are empty in the implementation, even though the CDNA4 VOP1 definition clears this wave's
  vector-ALU exception state in the cited manual passage.

Impact:

CDNA4 trap enable state, trap entry payloads, trap returns, sticky exception
status, and memory-violation reporting cannot be emulated or observed through
HWREG state. The missing trap trigger also includes `MODE.DEBUG`: CDNA4 requires
a trap after each executed non-`S_ENDPGM` instruction when `STATUS.TRAP_EN` is
set, but the issue/retirement path never checks the DEBUG bit, so legal
single-step execution runs through without entering the handler.

### CDNA4-RJ-058: Optional `TG_SIZE` system SGPR launch payload is not initialized

Manual evidence:

- Section 3.13 says compute launch may append a `TG_SIZE` system SGPR containing
  `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0, 14'h0,
  work-group_size_in_waves[5:0]}` when `COMPUTE_PGM_RSRC2.tg_size_en` is set in the cited manual passage.

Rocjitsu evidence:

- Dispatch initialization writes enabled workgroup-id system SGPRs after user
  SGPRs, then proceeds to the RDNA4/gfx1250 TTMP payload and packed workitem-ID
  initialization; it has no `TG_SIZE` write path in the implementation.
- A static source search for `TG_SIZE`, `tg_size`, `wave_id_in_group`,
  `first_wave`, and `work_group_size_in_waves` found no rocjitsu launch payload
  implementation.

Impact:

CDNA4 kernels that request the `TG_SIZE` system SGPR observe zero or stale SGPR
state instead of the documented wave/workgroup-size payload.

### CDNA4-RJ-066: `S_BARRIER` does not expose `STATUS.IN_BARRIER`

Manual evidence:

- Chapter 3 lists `STATUS.IN_BARRIER` among the hardware status bits in the cited manual passage.
- Section 4.3 describes `S_BARRIER` as waiting until other workgroup
  wavefronts reach the same barrier, with early-terminated waves satisfying the
  barrier, in the cited manual passage.

Rocjitsu evidence:

- `S_BARRIER` dispatches to `execute_s_barrier_sopp`, which sets only the
  internal `WfState::BARRIER` state in the implementation.
- The CU releases barriers when all non-halted wavefronts in the same
  dispatch/workgroup are either halted or at the barrier in the implementation.
- No corresponding update to raw `STATUS.IN_BARRIER` was found.

Impact:

The core multi-wave barrier release can work, but code that reads `STATUS`
during or around a barrier cannot observe the documented barrier status bit.

### CDNA4-RJ-075: SETREG side effects, unsupported state, and spacing remain incomplete

Manual evidence:

- Chapter 5.8 requires an `S_NOP` between consecutive `S_SETREG` writes and
  applies privilege-sensitive `HwRegWriteMask` behavior with
  register-specific side effects in the cited manual passage.

Rocjitsu evidence:

- CDNA4 SOPK paths now use the correct architecture table, partial-field
  extraction/insertion, and read-only/user/privileged policies in
  the implementation.
- MODE, STATUS, and packed GPR allocation have backing behavior, but TRAPSTS,
  LDS_ALLOC, IB_STS, PC/debug, TBA/TMA, XCC_ID, and performance snapshot rows
  remain unsupported.
- No audited path models register-specific write side effects or the required
  dependency between consecutive SETREG writes.

Impact:

Basic ID and permission handling is patched; unsupported state, side effects,
and SETREG instruction-spacing behavior still diverge from the manual.

### CDNA4-RJ-084: `S_SET_VALU_COEXEC_MODE` decodes but does not update co-execution state

XML evidence:

- CDNA4 XML records `S_SET_VALU_COEXEC_MODE` as `ENC_SOPP` opcode 31 with a
  `SIMM16` operand in the machine-readable ISA XML.
- The XML description says the instruction sets vector ALU co-execution mode
  from `SIMM16[1:0]` for the next VALU instruction and clears that mode after
  the next VALU instruction issues in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated CDNA4 decoding includes opcode 31 in `sub_decode_sopp` in the implementation.
- The generated constructor exposes the `SIMM16` operand, but
  `SSetValuCoexecModeSopp::execute_impl()` is a no-op in the implementation.
- Static source search found no CDNA4 wavefront or VALU issue state for this
  one-instruction co-execution mode.

Impact:

Code using this XML-described SOPP encoding decodes and disassembles, but it
has no effect on the following VALU instruction in rocjitsu.

### CDNA4-RJ-140: `S_ENDPGM_SAVED` loses the saved-context completion signal

CDNA4 defines `S_ENDPGM_SAVED` as both terminating the wave and signaling that
a context-switch trap handler completed saving it. The generated execute body
only invokes ordinary wave termination, with no saved-context cause or
control-plane notification. Restore and context-management logic therefore
cannot distinguish a successfully saved wave from an ordinary program end.

### CDNA4-RJ-029: SMEM dependency counter, time, clause, and legality rules are not modeled

Manual evidence:

- Section 8.2.1.1 describes scalar-memory source-overwrite and clause hazards,
  including the single-dword single-instruction exception and atomic
  single-instruction-clause requirement, in the cited manual passage.
- Section 8.3 says scalar memory reads and writes can return out of order, can
  return partial results, and increment `LGKM_CNT` by one for one dword or by
  two for two or more dwords in the cited manual passage.
- Section 8.4 describes SDATA/SBASE alignment restrictions and out-of-range
  behavior in the cited manual passage, and sections 8.2.4/8.2.5
  define 64-bit `S_MEMTIME`/`S_MEMREALTIME` writes.

Rocjitsu evidence:

- `ScalarMemState` carries `num_dwords`, but only one
  `wait_counter_type = WaitCounterType::LGKMCNT` value in the implementation.
- Generated CDNA4 SMEM load bodies set `num_dwords` and
  `wait_counter_type`, for example `S_LOAD_DWORD` and `S_LOAD_DWORDX2` in the implementation.
- `MemoryPipeline::issue()` increments exactly one wait counter for the issued
  instruction in the implementation; `ScalarMemPipeline::complete_access()` writes every returned
  dword and then completes once in the implementation.
- `S_MEMTIME` and `S_MEMREALTIME` execute as immediate SGPR writes outside the
  scalar memory pipeline in the implementation.
- Generated CDNA4 SMEM constructors and execution paths do not validate the
  source-overwrite, clause, SDATA alignment, SBASE alignment, or out-of-range
  source/destination rules.

Impact:

Multi-dword SMEM instructions undercount `LGKM_CNT`, time reads do not
participate in the scalar-memory dependency model, and rocjitsu accepts or
executes scalar-memory sequences and operands that the manual marks as
restricted or undefined.

### CDNA4-RJ-041: FLAT wait-counter and ordering contract is simplified

Manual evidence:

- Section 10.2 says FLAT instructions execute internally as both LDS and Buffer
  operations, increment both `VM_CNT` and `LGKM_CNT`, and complete only when
  both have decremented in the cited manual passage.
- Sections 10.2.1 and 10.2.2 describe out-of-order completion, same-VGPR
  return hazards, and the `S_WAITCNT 0` restriction in the cited manual passage.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH use only `VM_CNT`, not
  `LGKM_CNT`, in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 flat loads and atomics initially set
  `wait_counter_type = WaitCounterType::VMCNT`; representative bodies are in the implementation.
- The compute-unit router changes shared-aperture FLAT operations to
  `LOCAL_MEM` and `LGKMCNT`, which is an either/or counter choice rather than a
  dual VM+LGKM issue/retire model in the implementation.

Impact:

Functional LDS/global routing can work for simple cases, but rocjitsu does not
model the CDNA4 FLAT dual-counter and ordering hazards needed for wait-counter
exactness.

### CDNA4-RJ-063: Scalar-memory and message `LGKM_CNT` accounting is simplified

Manual evidence:

- Section 4.4 says scalar-memory reads increment `LGKM_CNT` by dword count
  (one for one dword, two for two or more dwords), `S_MEMTIME` counts like
  `s_load_dwordx2`, `S_SENDMSG` increments by one, and decrements occur for
  each scalar-memory dword returned and each message sent in the cited manual passage.

Rocjitsu evidence:

- The shared wait-counter model increments a selected counter by one per issue
  and decrements by one per completion in the implementation.
- `MemoryPipeline::issue()` calls `wf.wait_counters().increment(issue_counter)`
  once and releases the same counter once on completion in the implementation.
- Scalar memory uses the `LGKMCNT` pipeline by default in the implementation.
- `S_SENDMSG` uses an empty shared helper and therefore does not participate in
  `LGKM_CNT` in the implementation.

Impact:

Functional waiting works for many simple memory dependencies, but CDNA4 scalar
memory dword-count behavior and message-count behavior are not represented.

### CDNA4-RJ-064: Wait-counter ordering restrictions are not modeled

Manual evidence:

- Section 4.4 states that instructions of the same type return in order, that
  mixed reads/writes return in order for a given memory type, and that scalar
  memory reads can return out of order so only `S_WAITCNT 0` is legitimate in the cited manual passage.

Rocjitsu evidence:

- The wait target is a pure threshold over the current counter values in the implementation.
- The memory pipeline tracks a single counter type and releases that counter
  when an access completes in the implementation,
  with no per-instruction-class ordering metadata.

Impact:

Rocjitsu does not validate illegal partial scalar-memory waits or model the
manual's same-type ordering guarantees beyond simple outstanding-count
thresholds.

### CDNA4-RJ-122: `S_MEMTIME` and `S_MEMREALTIME` use call counters instead of architected clocks

CDNA4 ISA section 8.2.4 requires `S_MEMTIME` to return a 64-bit clock counter, while section 8.2.5 requires `S_MEMREALTIME` to return a 64-bit counter driven by a constant 100 MHz clock independent of power modes and core-clock changes. Current `execute_s_memtime_smem()` and `execute_s_memrealtime_smem()` each own a separate `static thread_local uint64_t counter`, increment it by 100 only when that instruction executes, and write the synthetic value to the destination SGPR pair. Neither value advances with simulator/core time, the real-time value does not progress at 100 MHz, and the two clocks have no defined relationship.

### CDNA4-RJ-031: SMEM cache policy, discard, and instruction-cache invalidation remain incomplete

CDNA4 ISA section 8.2.1.1 and Table 39 define scalar-memory `GLC`/`NV` policy and the 64-byte-aligned `S_DCACHE_DISCARD` behavior; section 8.2.3 distinguishes scalar-cache invalidation/writeback behavior, and the detailed `S_ICACHE_INV` definition requires invalidating the entire L1 instruction cache. Generated SMEM execution records `GLC`-derived `ScalarMemState::mtype`, but `ScalarMemPipeline::initiate_access()` does not pass that instruction policy to the scalar L1, `NV` is not consumed, and volatile invalidate/writeback forms collapse to the same whole-cache operations as their non-volatile forms. `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` still throw `UnimplementedInst`, while `execute_s_icache_inv_sopp()` is empty. The XML-only `S_ATC_PROBE` rows are intentionally excluded because the handbook does not provide an authoritative CDNA4 execution contract for them.

### CDNA4-RJ-036: Buffer cache-control and cache-maintenance policies are coarse

Manual evidence:

- Section 9.1.10 gives detailed vector-memory load, store, atomic, invalidate,
  and writeback cache-policy tables for `SC[1:0]` and `NT`, including
  `TG_SPLIT` behavior and SC-dependent `BUFFER_WBL2`/`BUFFER_INV` effects,
  beginning in the cited manual passage.

Rocjitsu evidence:

- `mtype_from_flags_gfx940()` collapses `SC0`, `SC1`, and `NT` into a coarse
  `Mtype` value in the implementation.
- Shared `BUFFER_INV`/`BUFFER_WBL2` helpers invalidate or flush broad cache
  levels without consulting the SC table or `TG_SPLIT` refinements in the implementation.
- Generated CDNA4 `BUFFER_WBL2`/`BUFFER_INV` dispatches simply call those
  helpers in the implementation.

Impact:

The current model has useful broad cache-behavior hooks, but not the
instruction- and scope-specific CDNA4 policy required for precise cache-control
validation.

### CDNA4-RJ-003: `SH_MEM_CONFIG[8]` is not modeled for BF8/FP8 operations

Manual evidence:

- The cited manual passage says `SH_MEM_CONFIG` bit 8 must be set for correct
  BF8/FP8 operation results.

Rocjitsu evidence:

- The audited generated conversion bodies read operands and destination state,
  but no scalar memory configuration state, before invoking FP8/BF8 helpers.

Impact:

Rocjitsu currently cannot distinguish configured and misconfigured BF8/FP8
operation behavior for this slice.

### CDNA4-RJ-020: Dense MFMA clamp and overflow state is not modeled

Manual evidence:

- The dense MFMA rule table, excluding F8F6F4, says clamp is supported, uses
  `FP16_OVFL` from MODE, clamps F32 overflow to `+/-MAX` when set and otherwise
  produces `+/-INF`, and clamps I32 overflow/underflow to `+/-MAX` when set
  and otherwise drops carry-out bits in the cited manual passage.

Rocjitsu evidence:

- Generated dense MFMA execution paths pass source/destination bases,
  `const_acc`, `CBSZ`, `ABID`, and `BLGP` into shared helpers, but do not read
  MODE, `FP16_OVFL`, or any clamp-control bit; representative F32/F16 calls are
  in the implementation.
- Dense I8 MFMA generated paths call `exec_i32_i8()` without a clamp argument,
  for example in the implementation.
- The shared integer helper has a clamp-capable primitive
  `exec_i32_mixed(..., bool clamp = false...)` in the implementation, but the dense I8 MFMA specialization documents that there is no
  clamp on its path and wraps in 32 bits.

Impact:

If the manual's dense MFMA clamp/`FP16_OVFL` row is architecturally meaningful,
rocjitsu currently executes those cases as unclamped, mode-independent MFMA.
The VOP3P-MAI field table does not expose a `CLAMP` bit, so this likely needs
manual/assembler reconciliation before a precise emulator contract can be
implemented.

### CDNA4-RJ-119: Device-memory consistency and acknowledgment behavior is not represented

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

### CDNA4-RJ-059: `S_RFE_B64` and `S_RFE_RESTORE_B64` do not model trap return

Manual evidence:

- Section 4.1 describes `S_RFE` as returning from the trap handler in the cited manual passage.
- Section 4.5 includes a required delay from `S_SETREG_B32` writing `TRAPSTS`
  to `S_RFE` / `S_RFE_restore`, tying trap-return behavior to trap status in the cited manual passage.

Rocjitsu evidence:

- CDNA4 `S_RFE_B64` constructs an operand and dispatches to the shared helper
  in the implementation.
- The shared `execute_s_rfe_b64_sop1` helper is empty in the implementation.
- CDNA4 `S_RFE_RESTORE_B64` throws `UnimplementedInst` in the implementation.

Impact:

Trap-handler return does not restore PC/privilege/trap state, and the restore
variant cannot execute at all.

### CDNA4-RJ-060: Debug conditional branches never branch

Manual evidence:

- Section 4.2 lists `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`,
  `S_CBRANCH_CDBGSYS_OR_USER`, and `S_CBRANCH_CDBGSYS_AND_USER` as debug-flag
  conditional branches in the cited manual passage.

Rocjitsu evidence:

- The CDNA4 constructors for these four SOPP branch instructions decode the
  label operand but do not set `COND_BRANCH` flags or implement a branch body in the implementation.
- The shared debug-branch helpers are empty in the implementation.

Impact:

Programs using debug conditional branches always fall through in rocjitsu,
regardless of debug flag state.

### CDNA4-RJ-061: Fork/join divergent control flow is not implemented

Manual evidence:

- Section 4.6 describes arbitrary divergent control flow with
  `S_CBRANCH_{I,G}_FORK` and `S_CBRANCH_JOIN`, including a six-deep branch
  stack, CSP mode bits, EXEC/PC stack entries, path selection, and pseudocode in the cited manual passage.

Rocjitsu evidence:

- CDNA4 `S_CBRANCH_G_FORK` throws `UnimplementedInst` in the implementation.
- CDNA4 `S_CBRANCH_I_FORK` throws `UnimplementedInst` in the implementation.
- CDNA4 `S_CBRANCH_JOIN` calls the shared helper in the implementation, and that helper is empty in the implementation.
- Static source search found no rocjitsu model for the CDNA4 CSP branch stack
  described by the manual.

Impact:

Compiler-generated fork/join divergent-control sequences either fail at fork or
fall through at join without restoring `EXEC`/PC from the branch stack.

### CDNA4-RJ-023: I32 SMFMAC variants are generated as unimplemented stubs

Manual evidence:

- The CDNA4 sparse MFMA inventory includes I8-to-I32 sparse forms, including
  `16x16x128_I8`, `32x32x64_I8`, and older/smaller I8 forms, in the cited manual passage.

Rocjitsu evidence:

- Generated `V_SMFMAC_I32_16X16X128_I8` and `V_SMFMAC_I32_32X32X64_I8`
  execution bodies throw `UnimplementedInst` in the implementation.
- The smaller `V_SMFMAC_I32_16X16X64_I8` and
  `V_SMFMAC_I32_32X32X32_I8` classes are also generated as stubs in the implementation.
- The code generator emits stubs for non-F32 SMFMAC variants in the code generator.

Impact:

Legal CDNA4 I8 sparse matrix instructions decode, but throw instead of
executing.

### CDNA4-RJ-030: Scalar atomics remain unimplemented and overstate unconditional SDATA returns

Generated scalar-atomic metadata now exposes SDATA as an input, but still declares SDATA as an unconditional destination even though the ISA returns the old value only with GLC=1. All scalar and scalar-buffer atomic execute methods still throw UnimplementedInst.

### CDNA4-RJ-057: CDNA4 `HW_ID` and `XCC_ID` HWREG reads remain unsupported

The shared HWREG table now maps CDNA4 HW_ID (ID 4) and XCC_ID (ID 20) correctly, fixing the old CU-only and ID-map behavior. Both registers are still explicitly unsupported, so S_GETREG_B32 returns zero with a warning instead of the packed hardware and XCC identifiers defined by ISA section 3.12.

### CDNA4-RJ-095: `V_SAT_PK_U8_I16` decodes but throws in both VOP1 and VOP3 forms

Manual evidence:

- `V_SAT_PK_U8_I16` defines signed 16-bit saturation into two packed unsigned
  8-bit results in the cited manual passage.

Rocjitsu evidence:

- The generated VOP1 executor throws `UnimplementedInst` in the implementation.
- The generated VOP3 executor also throws in the implementation.

Impact:

Kernels using this documented packed saturation instruction cannot execute
through rocjitsu despite generated decode and operand metadata being present.

### CDNA4-RJ-109: `V_PK_FMAC_F16` remains unimplemented in both encodings

The generated VOP2 and VOP3A constructors both decode legal
`V_PK_FMAC_F16` encodings, and the VOP3A form now surfaces the old `VDST`
accumulator read as well as the destination write. Both execute bodies still
throw `UnimplementedInst`, so neither legal encoding can execute.

### CDNA4-RJ-110: Several valid native VOP3A opcodes decode to hard stubs

Manual/XML evidence:

- `V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and `V_MQSAD_U32_U8` have detailed
  SAD/MSAD pseudocode in the cited manual passage, and XML
  records the matching VOP3A operand shapes in the machine-readable ISA XML.
- `V_TRIG_PREOP_F64` has detailed argument-reduction pseudocode in the cited manual passage, with XML metadata in the machine-readable ISA XML.
- `V_CVT_PKNORM_I16_F16` and `V_CVT_PKNORM_U16_F16` have packed normalized
  conversion pseudocode in the cited manual passage, with XML
  metadata in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated `VQsadPkU16U8Vop3`, `VMqsadPkU16U8Vop3`, and `VMqsadU32U8Vop3`
  constructors exist, but each `execute_impl()` throws `UnimplementedInst` in the implementation.
- `VTrigPreopF64Vop3` likewise decodes to a generated class whose
  `execute_impl()` throws in the implementation.
- `VCvtPknormI16F16Vop3` and `VCvtPknormU16F16Vop3` also throw in the implementation.

Impact:

These native CDNA4 VOP3A instructions have manual and XML semantic records and
decode as concrete rocjitsu classes, but runtime execution fails immediately.

### CDNA4-RJ-131: `DS_WRAP_RTN_B32` decodes but always throws

CDNA4 defines `DS_WRAP_RTN_B32` as a legal wrap-and-return atomic operation.
The generated execute body unconditionally throws `UnimplementedInst`, so any
legal use aborts simulation rather than conditionally subtracting or adding and
returning the old LDS value.

### CDNA4-RJ-098: `V_SCREEN_PARTITION_4SE_B32` is generated from XML but unimplemented

The current CDNA4 XML defines `V_SCREEN_PARTITION_4SE_B32` as a B32 VALU operation with `ENC_VOP1` opcode 55 and `ENC_VOP3` opcode 375, including destination and source operand metadata, although the detailed Chapter 12.8 definitions and Chapter 13 opcode table do not describe it. Rocjitsu generates and decodes `VScreenPartition4seB32Vop1` and `VScreenPartition4seB32Vop3`, but both `execute_impl` methods throw `UnimplementedInst`. This is a concrete XML-inventory implementation gap; absent fuller prose, the finding should not speculate about the missing operation's exact LUT semantics.

### CDNA4-RJ-040: Global/scratch direct-to-LDS flat-memory forms are not decoded or generated

Manual evidence:

- Sections 10.4 and 10.5 say GLOBAL and SCRATCH instructions can move data
  directly between LDS and memory in the cited manual passage.
- Section 10.3 gives direct-LDS destination formulas using the hardware LDS
  base, `M0[17:2] * 4`, instruction offset, and thread ID in the cited manual passage.
- Chapter 13.6 lists `GLOBAL_LOAD_LDS_*` and `SCRATCH_LOAD_LDS_*` opcode
  entries in the cited manual passage.

Rocjitsu evidence:

- A targeted search of generated CDNA4 and CDNA3 flat code found no
  `GLOBAL_LOAD_LDS`, `SCRATCH_LOAD_LDS`, `GlobalLoadLds`, or `ScratchLoadLds`
  instruction classes.
- CDNA4 generated flat decoding shares `FLAT_*` classes across `SEG` values for
  overlapping opcodes, with `flat_mnemonic()` rewriting `seg=1` to
  `scratch_*` and `seg=2` to `global_*` in the implementation. That only covers opcodes that have a
  `FLAT_*` base entry.
- The generated CDNA4 flat sub-decode table marks opcodes 38-42 invalid in the implementation, even though those are `GLOBAL_LOAD_LDS_{UBYTE,SBYTE,USHORT,SSHORT,DWORD}`
  and `SCRATCH_LOAD_LDS_{UBYTE,SBYTE,USHORT,SSHORT,DWORD}` in the manual/XML.
  It also leaves opcode slots 125-126 invalid in the implementation, matching the missing `GLOBAL_LOAD_LDS_DWORDX4` and
  `GLOBAL_LOAD_LDS_DWORDX3` forms.
- The generic flat-load generator only sets a VGPR destination and never sets
  `lds_dst`, `lds_base`, or per-lane LDS addresses in the code generator.
- The vector memory pipeline has an LDS-destination completion path, but it is
  only reached when an instruction sets `d.lds_dst` in the implementation.

Impact:

The CDNA4 decoder/runtime cannot decode or execute the manual's
GLOBAL/SCRATCH-to-LDS flat-memory forms end to end, even though those opcodes
are present in the XML. Overlapping GLOBAL/SCRATCH load/store/atomic opcodes
can still reach the shared FLAT classes through `SEG`-based mnemonic rewriting;
the missing part is the direct-to-LDS-only opcode subset.

### CDNA4-RJ-104: Legal 32-bit VOPC DPP forms are rejected

CDNA4 ISA section 12.16.1 prohibits DPP for the listed F64, I64, and U64 VOPC families, but does not prohibit ordinary 32-bit VOPC compares; Chapter 13.3.9 defines DPP as a second dword for VOPC and provides its lane-selection and source-modifier fields. Current generated constructors reject `SRC_DPP` even for legal 32-bit representatives including `VCmpLtF32Vopc`, `VCmpEqF32Vopc`, and `VCmpClassF32Vopc`, and their execute bodies also treat DPP as unimplemented. Thus the old accepted-but-ignored-modifier path is gone, but valid 32-bit VOPC DPP instructions still cannot execute at all.

### CDNA4-RJ-071: SALU exposes SCC operands but `InstDefUse` still drops them

CDNA4 ISA Chapters 5.3 through 5.7 define SCC producers and consumers across scalar arithmetic, comparison, carry, predicate, and conditional-select instructions. Generated constructors now surface these dependencies: representative examples include `SAddU32Sop2` and `SCmpEqI32Sopc` with SCC destinations, `SAddcU32Sop2` with SCC input and output, and `SCselectB32Sop2` with an SCC input. These SCC operands are fieldless, and `Operand::to_register_ref()` deliberately returns null for every fieldless operand; `InstDefUse` therefore still receives no actionable SCC definition or use. The missing operand-array entries are fixed, but RegisterRef-based scheduling and analysis remain blind to the architectural SCC dependency.

### CDNA4-RJ-101: E32 VOPC exposes VCC/EXEC operands but RegisterRef def-use still drops fieldless special-state writes

Generated VOPC classes now surface VCC and EXEC destination operands. Those operands are fieldless, and Operand::to_register_ref deliberately returns null for fieldless values, so InstDefUse still cannot represent the architectural VCC/EXEC definitions.

### CDNA4-RJ-111: VOP3B `SDST=VCC` writes are invisible to register metadata

Manual/XML evidence:

- Chapter 13.3.5 defines VOP3B as the encoding with a unique scalar
  destination and lists the ten scalar-destination opcodes in the cited manual passage.
- XML records `SDST` as a scalar mask destination for the VOP3B records, using
  `OPR_SREG` for carry/MAD forms and `OPR_SDST` for div-scale forms, for
  example in the machine-readable ISA XML.

Rocjitsu evidence:

- Generated VOP3B constructors mirror those operand classes: carry and MAD
  `sdst` operands use `OPR_SREG`, while `V_DIV_SCALE_*` use `OPR_SDST`, in the implementation.
- Runtime writes go through `inst.sdst`, and the scalar-write resolver handles
  selector 106 as VCC in the implementation.
- `Operand::to_register_ref()` only returns `RegisterRef` values for the SGPR
  numeric ranges of `OPR_SDST` and `OPR_SREG`, and drops special scalar
  selectors such as VCC, M0, and EXEC in the implementation.

Impact:

An encoding with explicit `SDST=VCC` can update VCC during execution, but
liveness, def-use, and probe-clobber analysis built from `RegisterRef` metadata
can miss the explicit VCC clobber.

### CDNA4-RJ-108: `V_ACCVGPR_READ/WRITE` do not set the ACCVGPR instruction flag

Manual/XML evidence:

- The CDNA4 VOP3P opcode table lists `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE`
  opcodes 88 and 89 in the cited manual passage.
- XML gives those records `_B32` aliases and the expected accumulator operand
  classes in the machine-readable ISA XML.

Rocjitsu evidence:

- `InstFlags::ACCVGPR` is documented as the flag for `v_accvgpr_write`,
  `v_accvgpr_read`, and `v_accvgpr_mov`, and `Instruction::is_accvgpr()` exposes
  it in the implementation.
- The generator sets this flag only when `inst.name` is one of the `_B32`
  aliases in the code generator.
- Generated CDNA4 constructors use the primary XML names
  `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE`, construct the correct accumulator
  operands, and execute lane copies, but they never set `flags_ |= ACCVGPR` in the implementation.

Impact:

Execution dataflow is present, but instruction-metadata consumers using
`Instruction::is_accvgpr()` can miss the explicit accumulator move opcodes.

### CDNA4-RJ-089: GPR-indexing exposes M0 operands but RegisterRef def-use still drops them

Generated GPR-indexing constructors now surface the XML-described M0 reads and writes as fieldless OPR_SDST_M0 operands. Fieldless operands deliberately return no RegisterRef, OPR_SDST_M0 has no mapping, and RegisterSet has no tracked M0 class, so InstDefUse still cannot order or analyze the architectural M0 dependency.

### CDNA4-RJ-006: FP8/BF8 forwarding hazard is not modeled

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

### CDNA4-RJ-025: Section 7.6 MAI dependency-wait rules are not modeled

Manual evidence:

- Section 7.6 states that the VOP3P-matrix table gives timing conditions where
  users must insert NOPs or independent VALU instructions in the cited manual passage.
- The table covers DLop, XDL, SGEMM, DGEMM, and SMFMA producer/consumer
  combinations with exact-opcode Source C forwarding exceptions, Source C
  overlap wait tiers, SrcA/SrcB waits, VM/LDS/FLAT/Export overlap waits, VALU
  RAW+WAW waits, `V_CMPX*` EXEC-mask forwarding waits, and an XDL/SMFMA Source C
  read versus VALU write WAR anti-dependency in the cited manual passage.

Rocjitsu evidence:

- The generic `MFMA` instruction flag only marks matrix FMA instructions, and
  `is_mfma()` only exposes that broad classification in the implementation.
- Codegen sets `flags_ |= MFMA` for names starting with `V_MFMA_` or
  `V_SMFMAC_`, but does not attach Section 7.6 producer/consumer class, pass
  count, overlap, forwarding, or required-wait metadata in the code generator.
- `ComputeUnit::issue_instruction()` executes the decoded instruction directly
  and only routes memory operations through the wait-counter memory pipeline in the implementation.
- `WaitCounters` are documented as outstanding memory-operation counters for
  `vmcnt`, `lgkmcnt`, `expcnt`, vector-store, DS/K, tensor, and async memory
  forms, not MAI producer/consumer timing hazards, in the implementation.
- The shared MFMA helpers do buffer inputs before writes to avoid hazards inside
  a single helper call, but that is not an inter-instruction Section 7.6
  scheduling model.

Impact:

Rocjitsu functional execution can run MAI producer/consumer sequences that the
manual requires software to separate with independent instructions, and it
cannot diagnose or model the Section 7.6 forwarding and overlap contract. If
rocjitsu remains intentionally non-cycle-accurate, this is still a missing
metadata/diagnostic surface for scheduler-sensitive MAI rules rather than a
basic arithmetic helper mismatch.

### CDNA4-RJ-065: Manual wait-state hazards are not modeled or diagnosed

Manual evidence:

- Section 4.5 lists required manually inserted wait states for control/status,
  VALU helper-bit, readlane/lane-select, store data overwrite, SALU-to-VMEM,
  SALU-to-message, DPP, mixed VCC aliasing, RFE/TRAPSTS, M0-to-LDS/addressed
  operations, `S_MOVEREL`, `V_CMPX`, and transcendental hazards in the cited manual passage.

Rocjitsu evidence:

- `S_NOP` dispatches to an empty shared helper in the implementation.
- The instruction issue path executes functional instruction bodies and wait
  counters, but no static or dynamic check was found for the Chapter 4.5
  hazard table.

Impact:

Rocjitsu can execute instruction sequences that the CDNA4 manual requires
software to separate with NOPs or independent instructions. This is primarily a
timing/diagnostic gap, but it can also hide hazards when rocjitsu is used as an
oracle for hand-written ISA.

### CDNA4-RJ-077: SDWA/OPSEL next-VALU hazards are not modeled or diagnosed

Manual evidence:

- Chapter 6.2.1 says DOT instructions must not use SDWA or OPSEL, and that a
  VALU instruction using SDWA or OPSEL must not have its result consumed by the
  next VALU instruction; an independent instruction or `V_NOP` is required in the cited manual passage.

Rocjitsu evidence:

- rocjitsu has instruction flags for branches, wait counters, barriers, MFMA,
  AccVGPR, and predicated defs, but no generic SDWA/OPSEL hazard flag in the implementation.
- Generated CDNA4 SDWA/OPSEL-bearing instructions execute through normal
  instruction helpers and operand delegates; the instruction metadata has no
  producer/consumer hazard state analogous to the manual's next-VALU rule.

Impact:

Execution and scheduling analyses can accept instruction streams that require a
manual-inserted independent VALU separation on hardware.

## P4 — Permissiveness and coverage gaps

### CDNA4-RJ-076: VALU source validation allows manual-disallowed extra scalar sources

Manual evidence:

- Chapter 6.2.1 limits VALU to at most one SGPR source and at most one literal,
  with literals disallowed when an SGPR or M0 is used, in the cited manual passage.
- The same section says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC and
  therefore cannot use an additional SGPR or literal in the cited manual passage.

Rocjitsu evidence:

- Generated CDNA4 VOP3 constructors expose broad operand classes independently:
  `V_CNDMASK_B32` uses `OPR_SRC_NOLIT`, `OPR_SRC_SIMPLE`, and an explicit
  `OPR_SREG` third source in the implementation, while `V_ADD_F32` uses the two broad source classes.
- Carry-in forms likewise expose broad `SRC0`/`SRC1` classes plus an explicit
  scalar-pair `SRC2` for `V_ADDC_CO_U32` and `V_SUBB_CO_U32` in the implementation.
- `Operand::read_lane()` resolves non-VGPR operands through scalar/immediate
  fallback independently for each operand in the implementation; this path does not enforce a per-instruction scalar-source budget.

Impact:

rocjitsu can decode and execute VALU operand combinations that the manual
forbids, especially implicit-VCC instructions with an extra SGPR or literal.

### CDNA4-RJ-004: Generated source operands allow forms the manual says are illegal

Manual evidence:

- The cited manual passage says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

Rocjitsu evidence:

- Generated CDNA4 constructors use `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE` for
  non-scaled `V_CVT_PK_FP8_F32` in the implementation.
- The same broad operand classes appear in scaled FP8/BF8 constructors, for
  example the implementation.
- The underlying CDNA4 operand classes include scalar-register and inline-source
  subtypes in the generated operand metadata.

Impact:

Rocjitsu inherits the XML/manual legality mismatch for both non-scaled and
scaled conversion sources. Hardware and LLVM oracle behavior still need a
focused decision pass before turning this into an enforcement change.

### CDNA4-RJ-007: Packed F32 VOP3P accepts illegal non-VGPR sources

CDNA4 ISA section 6.7 requires packed F32 operands to be even-aligned VGPR pairs. Current constructors still admit scalar and inline source classes, and execution broadcasts the low dword for non-VGPR encodings instead of rejecting or diagnosing them. The live issue is source legality, not missing scalar-pair behavior.

### CDNA4-RJ-008: Packed 32-bit VOP3P VGPR pair alignment is not validated

Manual evidence:

- CDNA4 section 6.7 says packed 32-bit operands must be two-dword aligned, with
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

### CDNA4-RJ-011: Scaled conversion scale operands allow SGPR/literal forms the manual excludes

Manual evidence:

- The cited manual passage says the scale can come from a VGPR or an inline
  constant, using the float exponent portion.

Rocjitsu evidence:

- Generated scale operands use `OperandType::OPR_SRC_SIMPLE` in representative
  scaled conversion constructors, including FP4 F32 narrowing in the implementation, wide FP6 narrowing, stochastic FP6
  narrowing, and FP6 widening.
- The CDNA4 operand layer resolves `OPR_SRC_SIMPLE` scalar-register encodings
  as SGPRs in the implementation and VGPR encodings; the XML
  source class also includes scalar-register source classes in the machine-readable ISA XML.

Impact:

Rocjitsu inherits the XML source-class broadness for scale operands. A scale
encoded as an SGPR or literal can be decoded and executed even though the manual
limits the scale source to VGPR or inline constant forms.

### CDNA4-RJ-012: Wide scaled/packed conversion VGPR alignment is not validated

Manual/XML evidence:

- CDNA4 section 6.7.1 says convert opcodes operating on FP6/BF6/FP4 data must
  use VGPR sources for operand slots providing more than 32 bits of data in the cited manual passage.
- The CDNA4 XML's `OPR_SRC_VGPR` class says 64-bit and wider VGPR values must
  be even-aligned in the machine-readable ISA XML.
- The CDNA4 XML's plain `OPR_VGPR` class repeats the same alignment rule for
  64-bit and wider VGPR destinations in the machine-readable ISA XML.
- Table 31 says `CVT_PK_F32_FP8` and `CVT_PK_F32_BF8` write `dst,dst+1` and
  require an even destination in the cited manual passage; their
  instruction definitions write `D0[31:0]` and `D0[63:32]`.
- The scaled FP8/BF8 packed widening definitions also produce 64-bit F32 pairs
  in the cited manual passage.

Rocjitsu evidence:

- Generated scaled conversion constructors preserve 64-bit and wider VGPR data
  operands, for example `V_CVT_SCALEF32_SR_PK_FP4_F32` uses a 64-bit VGPR
  source in the implementation, `V_CVT_SCALEF32_2XPK16_FP6_F32` uses 192/512-bit VGPR
  operands, and `V_CVT_SCALEF32_PK32_F32_FP6` uses
  1024/192-bit VGPR operands.
- Generated `V_CVT_PK_F32_FP8/BF8` VOP1/VOP3 forms and
  `V_CVT_SCALEF32_PK_F32_FP8/BF8` construct 64-bit `VDST` operands and write via
  `write_lane64()` in the implementation.
- `Isa::resolved_vgpr_offset()` accepts any VGPR encoding from 256 through 511
  and returns the unadjusted register index in the implementation.
- `Operand::read_lane64()` and `write_lane64()` then read or write pairs rooted
  at that returned index in the implementation; the wide
  generated conversion paths use the same unvalidated root index in their
  multi-dword VGPR access lambdas.

Impact:

Odd VGPR encodings can root 64-bit packed F32 results or 64-bit, 192-bit,
512-bit, and 1024-bit scaled conversion operands in rocjitsu, despite the XML
operand-class alignment rule for 64-bit and wider VGPR values.

### CDNA4-RJ-019: Dense MFMA register-block alignment is not validated

Manual evidence:

- Section 7.1 says MFMA input/output register blocks must be contiguous and the
  first register must be aligned to the number of registers required by that
  operand in the cited manual passage.
- The dense MFMA rule table also says `SRC0`, `SRC1`, `SRC2`, and `VDST` need
  VGPR alignment in the cited manual passage.

Rocjitsu evidence:

- Generated dense MFMA constructors expose large register ranges from the raw
  encoded base without legality checks, for example the 32-register F32
  destination and accumulator for `V_MFMA_F32_32X32X1_2B_F32` in the implementation, and the 64-bit F16 source ranges for
  `V_MFMA_F32_32X32X4_2B_F16`.
- Execution maps encoded bases through `dst_base()` / `src_base()` and passes
  them directly into shared helpers, for example the F32 path in the implementation and the F16 path.
- `dst_base()` and `src_base()` only translate VGPR versus AccVGPR numbering in the implementation; the layout helpers compute offsets from that base; and the contiguous-region read/write helpers size regions but do not
  validate base alignment.

Impact:

Illegal dense MFMA encodings with odd or under-aligned source, accumulator, or
destination bases execute against shifted register blocks instead of being
rejected or diagnosed. This is broader than the packed 32-bit VOP3P alignment
gap because dense MFMA operands can require 4, 16, or 32 contiguous registers.

### CDNA4-RJ-021: MFMA broadcast field legality is not validated

Manual evidence:

- Section 7.1.6.1 says the largest legal `CBSZ` value is 4, `(1 << CBSZ)` must
  not exceed the number of blocks the MFMA instruction processes, and `ABID`
  must be less than `(1 << CBSZ)` in the cited manual passage.

Rocjitsu evidence:

- Generated dense MFMA execution passes raw `inst_.cbsz`, `inst_.abid`, and
  `inst_.blgp` into shared helpers for representative F32/F16/BF16/I8 paths,
  for example `V_MFMA_F32_32X32X1_2B_F32` in the implementation, `V_MFMA_F32_32X32X4_2B_F16`, and I8
  MFMA.
- `permute_a_lane()` computes `S = 64 >> cbsz` and returns
  `(lane % S) + S * abid` without checking `cbsz <= 4`, whether
  `(1 << cbsz)` fits the instruction's block count, or whether
  `abid < (1 << cbsz)` in the implementation.
- The scalar/SIMD MFMA helpers only gate whether the raw fields are non-zero
  before applying the permutation; representative F32 and I32 call sites are in the implementation.

Impact:

Illegal or undefined MFMA broadcast encodings execute with a computed lane
mapping instead of being rejected or diagnosed. Extreme illegal values can also
produce nonsensical helper behavior, such as `CBSZ=7` yielding `S=0` before the
modulus in `permute_a_lane()`.

### CDNA4-RJ-050: LDS transpose-load preconditions are not validated

Manual evidence:

- Section 11.4 says MFMA transpose loads require `EXEC` all ones, LDS address
  alignment to data size, and even-aligned VGPRs for 64-bit-or-larger data
  except `DS_READ_B96_TR_B6` in the cited manual passage.

Rocjitsu evidence:

- CDNA4 transpose load execute bodies set `d->transpose` and call generic DS
  address calculation, with no checks for all-ones `EXEC`, LDS address
  alignment, or even VGPR alignment in the implementation.
- The shared transpose helpers implement data shuffles for TR_B4/B6/B8/B16 in the implementation, but they do not validate instruction preconditions.
- Existing transpose tests check ACC destination routing for `DS_READ_B64_TR_B16`
  in the relevant tests, not precondition
  enforcement or all layout variants.

Impact:

Illegal or underspecified CDNA4 transpose loads can execute in rocjitsu instead
of being rejected/nullified according to the manual's constraints. Layout bugs
in TR_B4/B6/B8/B16 can also escape because adjacent tests mostly protect ACC
routing.

### CDNA4-RJ-067: Workgroup size limits are not validated against the ISA cap

Manual evidence:

- Section 4.3 says a workgroup can contain up to 16 wavefronts, or 1024
  work-items, in the cited manual passage.

Rocjitsu evidence:

- Dispatch computes `wfs_per_workgroup` from packet workgroup size and wave
  size in the implementation.
- Placement checks whether a CU has enough free wavefront slots and LDS in the implementation.
- `ComputeUnitCore::can_accept_workgroup()` compares requested wavefronts
  against free wavefront slots in the implementation,
  but no static check was found for the ISA's 16-wavefront / 1024-work-item
  workgroup limit.

Impact:

Oversized CDNA4 workgroups may be accepted or rejected based on emulator
resource configuration rather than the architectural limit.

### CDNA4-RJ-070: SALU 64-bit SGPR pair alignment is not enforced

Manual evidence:

- Chapter 5.2 says 64-bit SGPR operands must be aligned to an even SGPR index
  in the cited manual passage.

Rocjitsu evidence:

- `resolve_src_scalar64()` accepts raw 64-bit scalar sources such as `ev <= 105`
  and `108 <= ev <= 122` without checking that `ev` is even in the implementation.
- `resolve_dst_write64()` similarly writes 64-bit destinations without an even
  alignment check in the implementation.

Impact:

Odd 64-bit scalar pairs can execute silently in rocjitsu even though the CDNA4
ISA treats them as illegal or constrained.

### CDNA4-RJ-085: SOP1 no-literal forms still consume an illegal extra dword

The SOP1 base class treats SSRC0=255 as an eight-byte encoding for every opcode, including register-only and operandless forms with no literal alternative. Constructors now retain register-only operand types, but the decoder still consumes an extension dword and PC-relative forms use the inflated size, producing PC+8 rather than the required PC+4.

### CDNA4-RJ-086: `S_RFE_RESTORE_B64` accepts literal selectors despite having no literal encoding

Manual/XML evidence:

- CDNA4 XML records `S_RFE_RESTORE_B64` as `ENC_SOP2` opcode 43 with 64-bit
  `SSRC0`, 32-bit `SSRC1`, and implicit PC output in the machine-readable ISA XML.
- The same XML instruction has no `SOP2_INST_LITERAL` alternative; in the full
  audited SOP2 inventory, opcode 43 was the only XML `ENC_SOP2` record without
  any literal-encoding variant.
- The detailed CDNA4 SOP2 manual inventory omits this opcode entirely; the
  manual/XML source drift is recorded separately as `CDNA4-XML-060`.

Rocjitsu evidence:

- The shared CDNA4 `Sop2` base class treats every SOP2 instruction with
  `SSRC0 == 255` or `SSRC1 == 255` as an 8-byte literal form, with no
  opcode-specific filter, in the implementation.
- `SRfeRestoreB64Sop2` rewrites either source selector 255 into `OPR_SIMM32` in the implementation, even though the XML record has only the default encoding.

Impact:

Reserved or unsupported `S_RFE_RESTORE_B64` selector-255 encodings decode as
literal forms and consume a following dword, which can desynchronize
instruction-stream decoding and produce operand metadata that does not match the
XML encoding contract.

### CDNA4-RJ-088: `S_SET_GPR_IDX_ON` treats operand 1 as a literal-capable scalar source

Manual/oracle evidence:

- The detailed `S_SET_GPR_IDX_ON` definition says vector operations use M0 for
  relative GPR addressing, source 0 supplies the index, and the raw bits of the
  `SRC1` field set the enable bits; the pseudocode writes `M0[15:12]` from
  `SRC1[3:0]` and says this is direct raw-field content in the cited manual passage.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950` accepts a 4-bit mode operand
  and a source-0 literal, producing `s_set_gpr_idx_on s0, 15` as a one-dword
  encoding and `s_set_gpr_idx_on 0x12345678, 15` as an 8-byte source-0 literal
  encoding, but rejects `s_set_gpr_idx_on s0, 16` and
  `s_set_gpr_idx_on s0, 0x12345678`.

Rocjitsu evidence:

- The generic CDNA4 `Sopc` base treats any `ssrc1 == 255` as a literal form and
  increases instruction size in the implementation.
- `SSetGprIdxOnSopc` initially declares operand 1 as `OPR_SIMM4`, but still
  replaces it with an `OPR_SIMM32` operand when the raw field is 255 in the implementation.
- The shared executor then reads operand 1 through `RegisterAccess` and masks the
  resulting value to four bits before writing M0 in the implementation.

Impact:

rocjitsu accepts and sizes an operand-1 literal form that the ISA text and LLVM
assembler treat as invalid. If a raw `SSRC1=255` word is encountered, rocjitsu
consumes the next dword as a literal and derives the mode from that extension
word instead of treating operand 1 as a raw 4-bit mode field.

### CDNA4-RJ-100: VOP1 SDWA accepts unsupported modifier bits on untyped bit operations

Untyped bit operations now stage SDWA with SourceModifierFormat::NONE, preventing accidental floating ABS/NEG transforms. Their constructors still accept nonzero modifier fields that the ISA does not support and execute while silently ignoring them instead of rejecting the encoding.

### CDNA4-RJ-118: Selected F64 VOP1 DPP exclusions remain unenforced

CDNA4 ISA section 12.16.1 forbids DPP for the enumerated F64 VOP1 conversions and unary operations. Generated CDNA4 constructors still parse `SRC_DPP` for verified residual forms `VCvtI32F64Vop1`, `VCvtF64I32Vop1`, `VCvtF32F64Vop1`, `VTruncF64Vop1`, `VFrexpExpI32F64Vop1`, and `VFractF64Vop1`, allowing prohibited encodings into generic DPP execution. Other previously reported families are now rejected, and `VSqrtF64Vop1` is specifically excluded from this residual list because its constructor now throws `InvalidInst` for DPP.

### CDNA4-RJ-005: Scaled FP8/BF8 selector and legality edge coverage remains incomplete

Rocjitsu evidence:

- Generated CDNA4 conversion paths now pass `wf.fp16_ovfl()` to mode-aware
  FP8/BF8 helpers, including scaled RNE and stochastic paths in the implementation and non-scaled paths.
- The rebased execution-harness tests exercise alternate `FP16_OVFL` outcomes,
  so the former fixed-helper-mode test gap is patched.
- Targeted test inspection still does not establish complete coverage for BF8
  byte selection, high-half destination preservation, ignored SDWA fields, or
  64-bit destination-pair alignment on the scaled conversion family.

Impact:

Mode-dependent overflow behavior is now protected, but less common selector,
preservation, and legality contracts can still regress without a targeted
failure.

### CDNA4-RJ-013: FP4/FP6/BF6 scaled-conversion tests miss state and legality edges

Rocjitsu evidence:

- HIP FP4 tests round-trip exact FP4 values with scale 1.0 in the relevant tests.
- HIP FP6/BF6 tests round-trip exact representable values with scale 1.0 in the relevant tests.
- The explicit scale edge test covers scaled FP8 only in the relevant tests; the NaN/Inf edge test
  covers FP4 finite helper behavior.
- The gfx950 corpus skip list still skips the fpsan scaled FP4/FP6/SR tests in the relevant tests.

Impact:

The current tests cover useful finite plumbing and helper behavior, but would
not catch the MODE-denorm, scale-source legality, wide-VGPR alignment, E8M0
`0xff`, or multi-pass stochastic edge cases recorded above.

### CDNA4-RJ-038: Vector-buffer tests miss descriptor and format edge cases

Rocjitsu evidence:

- The relevant tests covers dynamic VADDR width,
  SRSRC scaling, and CDNA ACC bank folding for MUBUF/MTBUF operands.
- The relevant tests covers the MUBUF `lds`
  modifier in disassembly.
- The codegen tests
  covers generator helpers for legacy buffer VADDR width and SRSRC scaling, and
  the codegen tests covers a
  derived buffer-load semantic shape.

Impact:

The existing tests cover important operand/codegen regressions, but not the
manual-derived descriptor modes, format conversion, M0/inline `SOFFSET`,
alignment, LDS clamping/subset, unbound resource, atomic dataflow, or
cache-policy cases identified in the vector-buffer audit slice.

### CDNA4-RJ-044: Flat/global/scratch tests miss CDNA4-specific edge cases

Rocjitsu evidence:

- Shared tests cover CDNA4 scratch-base and global-address basics in the relevant tests.
- Generic flat VM tests include a CDNA3/CDNA4 encoding helper in the relevant tests and exercise some global
  atomic behavior later in that file.
- The static pass did not find CDNA4 end-to-end cases for `PTR32` or
  `ADDRESS_MODE`, direct-to-LDS global/scratch flat forms, FLAT dual-counter
  behavior, aperture-hole or GLOBAL-to-LDS memory violations, packed
  F16/BF16 atomics, FP atomic `SC0=0`, or the flat `NV` manual ambiguity.

Impact:

Existing tests protect useful core addressing behavior, but not the
manual-derived edge cases identified by the CDNA4 Chapter 10 slice.

### CDNA4-RJ-051: Data-share tests miss CDNA4 LDS edge cases

Rocjitsu evidence:

- Existing runtime coverage includes CDNA4 `ds_add_rtn_u32` and no-return
  `ds_add_u32` stress cases in the relevant tests.
- Existing transpose coverage checks ACC routing in the relevant tests.
- Python generator/semantics tests cover DS swizzle source selection,
  READ2/atomic semantic-class derivation, and transpose-kind derivation in the codegen tests.
- The static pass did not find focused CDNA4 tests for `M0[16:0]` LDS clamping,
  1280-byte allocation granularity, ADDTID addressing, duplicate-offset
  READ2/WRITE2 collapse, `DS_SWIZZLE_B32` FFT/rotate modes, packed F16/BF16 LDS
  atomics, transpose preconditions/layout variants, or the CDNA4 `GDS` reserved
  versus GDS wording ambiguity.

Impact:

The current tests protect useful DS plumbing and ACC routing, but the CDNA4
manual-derived LDS edge cases identified in this slice can regress without
focused coverage.

### CDNA4-RJ-068: Chapter 4 program-flow tests miss side-effect-heavy cases

Rocjitsu evidence:

- The generic instruction execution harness lists `s_waitcnt`, `s_barrier`,
  `s_sleep`, `s_sendmsg`, `s_sendmsghalt`, and `s_rfe` as decode/execute
  surface mnemonics in the relevant tests.
- The adjacent plugin barrier test covers the basic two-wave barrier-resolved
  hook path in the relevant tests.
- Static test search found ordinary waitcnt decode/execution fixtures and
  kernel fixtures containing `s_waitcnt` / `s_barrier`, but no focused CDNA4
  coverage for debug branch predicates, RFE/RFE_RESTORE, fork/join stack
  behavior, `S_SENDMSG` `LGKM_CNT`, scalar-memory dword-count waits,
  `STATUS.IN_BARRIER`, or the Chapter 4.5 wait-state hazard table.

Impact:

The current tests can catch basic decode, ordinary waiting, and a simple
barrier release, but they would not fail for most of the Chapter 4 semantic
gaps above.

## Resolved since the previous audit

- `CDNA4-RJ-027` is fixed for the selector/alignment behavior it originally
  reported. `generated/cdna4/smem.cpp:17` now preserves the non-SOE
  `OFFSET[6:0]` selector, and `shared/addr_calc_scalar.h:44` resolves M0,
  applies scratch x64 scaling, and clears the required dword low bits. The
  remaining `S_DCACHE_DISCARD` cache-line behavior is already tracked by
  `CDNA4-RJ-031`.
- `CDNA4-RJ-125` is fixed: no-return DS atomics set `is_load = false` beginning
  at `generated/cdna4/ds_exec.cpp:35`, preventing a return write to encoded
  `VDST`.
- `CDNA4-RJ-120` is fixed: the shared `V_BCNT_U32_B32` executors add `SRC1` to
  the popcount at `generated/shared/execute_shared.h:3603` and `:3611`.
- `CDNA4-RJ-121` is fixed: scalar selected-lane reads and writes mask by the
  dispatched wave size in `vm/amdgpu/register_access.h:870` and `:931`.
- `CDNA4-RJ-123` is fixed: `shared/ds_transpose.h:197` implements the CDNA4
  4-by-16 halfword permutation, with an exact layout regression beginning at
  `tests/amdgpu_vm_test.cpp:4068`.
- `CDNA4-RJ-014`, `CDNA4-RJ-015`, and `CDNA4-RJ-016` are fixed together by
  explicit scaled VOP3PX2 classes and operands in `generated/cdna4/vop3p.h:1037`
  and `generated/cdna4/vop3p.cpp:4240`; execution extracts independent scale
  byte selectors and inline scale sources in `generated/cdna4/vop3p_exec.cpp:1372`
  and `shared/mma_exec.h:3311`.
- `CDNA4-RJ-018` is fixed by direct decode, legality, selector, layout,
  reference, rounding, inline-scale, and C-modifier tests in
  `tests/instruction_execution_harness_test.cpp:1624` through `:2255`.

## No-Gap Notes

- CDNA4 Chapter 1-2 dispatch, 64-lane wavefront, initial `EXEC`, and packed
  work-item-ID basics are represented by the production dispatch path and
  existing Chapter 3 no-gap notes below. The new Chapter 2 issue above is
  limited to the device-memory consistency and acknowledgment model; LDS
  allocation/bank behavior, launch TTMP/TG_SIZE state, barrier behavior, and
  workgroup-size validation remain covered by their existing narrower entries.
- The CDNA4 VOP3P opcode inventory is generated for all 104 Chapter 12.10
  detailed opcode headings, with decode-table entries for the checked DOT,
  packed, ACCVGPR, and MFMA families. The only additional XML VOP3P-family
  entries are the two scaled `ENC_VOP3PX2` forms. Those forms now have explicit
  classes, operand metadata, execution, and direct tests; the remaining
  unscaled C-modifier issue is `CDNA4-RJ-017`.
- Generated CDNA4 VOP3A/VOP3B class inventory matches the XML decode split for
  this slice: 500 `ENC_VOP3` classes and 10 `VOP3_SDST_ENC` classes. The gaps
  above are semantic/runtime and metadata gaps, not missing decoder entries.
- CDNA4 Chapter 12.11 definitions 640-659 have generated constructors and
  encoding tests for the manual/XML opcodes 640-653 and 655-659; opcode 654 is
  absent from both sources. For this slice, `V_MBCNT_*` adds `SRC1`, 64-bit
  shift helpers mask counts with `&63`, and `V_BFM_B32` masks width/offset with
  `&31`. `V_BCNT_U32_B32` now adds `SRC1`, and readlane/writelane now mask the
  lane selector; the remaining `V_TRIG_PREOP_F64` hard stub is already covered
  by `CDNA4-RJ-110`.
- CDNA4 Chapter 12.11 definitions 660-673 have generated constructors and
  encoding tests for the manual/XML opcodes 660-666 and 668-673; opcode 667 is
  absent from both sources. The F32 normalized converts, packed integer
  narrowing converts, `V_PACK_B32_F16`, and `V_MUL_LEGACY_F32` have generated
  execution bodies matching the audited source-selection and pack/legacy-zero
  dataflow. Existing gaps cover the two F16 normalized-convert hard stubs
  (`CDNA4-RJ-110`) and integer clamp/saturation overloads (`CDNA4-RJ-080`);
  packed F32-to-F16 RTZ conversion now uses the correct rounding helper.
- CDNA4 Chapter 12.11 definitions 680-681 have generated constructors and
  encoding tests for `V_MINIMUM3_F32` and `V_MAXIMUM3_F32`, and the generated
  bodies call the expected IEEE min/max helper composition. The remaining
  runtime issue for this pair is the MODE.DX10_CLAMP NaN-to-zero behavior
  recorded as `CDNA4-RJ-112`.
- Generated CDNA4 VOP3B class coverage exists for all ten Chapter 12.11/13.3.5
  VOP3B opcodes, and the shared carry/div-scale/MAD helpers write their
  explicit `inst.sdst` scalar destinations. `CDNA4-RJ-111` is limited to the
  public register-metadata view for special scalar destinations such as VCC.
- Ordinary `V_FMAC_F64` and `V_FMAC_F32` accumulator metadata is present: the
  generated constructors put `VDST` in both `src_operands_` and
  `dst_operands_` in the implementation. `CDNA4-RJ-109` remains limited to the
  hard-stubbed `V_PK_FMAC_F16` path.
- XML carries the `V_CVT_PKRTZ_F16_F32` round-toward-zero description and VOP3
  operand shape, and generated execution now uses the matching RTZ helper.
- Native F16 VOP3A destination-half preservation is implemented for
  `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`, and
  `V_DIV_FIXUP_F16`: the generated classes add old `VDST` to implicit uses and
  call `write_vop3_true16_dst(..., true)` in the implementation. The corresponding XML gap is `CDNA4-XML-083`.
- `V_BITOP3_B16` and `V_BITOP3_B32` are implemented as truth-table operations
  over the overloaded `{OMOD, ABS, NEG}` bits rather than ordinary modifiers,
  in the implementation; the generator documents the same packing rule in the code generator.
  The corresponding XML gap is `CDNA4-XML-084`.
- A fresh CDNA4 VOP3 hard-stub scan found only the VOP3A stubs already covered
  by `CDNA4-RJ-095`, `CDNA4-RJ-098`, `CDNA4-RJ-109`, and `CDNA4-RJ-110`, in the implementation.
- The shared CDNA VOP3B machine-instruction struct has no `ABS` field, and the
  generator's `has_abs_modifier()` rule excludes `VOP3_SDST_ENC`, matching the
  manual's VOP3B field map rather than leaking generic VOP3A `ABS` handling
  into VOP3B.
- Generated packed shift helpers mask the selected shift-count half with
  `& 15`, matching the manual's use of `S0[3:0]` and `S0[19:16]` for the low
  and high components. The generated constructors and shared helpers for
  `V_PK_LSHLREV_B16`, `V_PK_LSHRREV_B16`, and `V_PK_ASHRREV_I16` dispatch
  through the implementation.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` are not missing execution bodies:
  generated CDNA4 code copies one 32-bit lane between the VGPR and ACCVGPR
  register classes under `EXEC` in the implementation. `CDNA4-RJ-108` is limited to the public instruction flag metadata.
- CDNA4 generated VALU instruction coverage is not broadly absent for the
  audited Chapter 6.3 inventory. Representative VOP2/VOP3/VOPC/VOP3P DOT,
  compare, carry-out, CNDMASK, `_MK`/`_AK`, and GPR-indexing instruction
  classes exist; the Chapter 6 VALU gaps above are about legality, stateful
  semantics, and hazard modeling.
- CDNA4 VALU execution honors the EXEC mask for ordinary per-lane VGPR writes in
  shared helpers such as `execute_v_add_f32_vop3()` in the implementation, matching the basic Chapter 6.2.2 write-mask contract.
- Generated CDNA4 VOPC class/decode inventory exists for all 198 XML VOPC
  opcode rows; the VOPC gaps above are about special-state metadata and
  extension semantics, not missing base opcode classes.
- CDNA4 relational VOP3 CMPX helpers write the explicit scalar destination and
  then update EXEC, matching the VOP3 destination contract. `CDNA4-RJ-102`
  records the separate class-CMPX exception, while `CDNA4-RJ-101` covers E32
  VOPC implicit VCC/EXEC metadata.
- Chapter 6.2.3's ordinary out-of-range VGPR behavior overlaps the broader GPR
  allocation/out-of-range execution gap already recorded in the Chapter 3
  rocjitsu pass; this Chapter 6 slice adds the distinct indexed-VGPR
  out-of-range legality issue under `CDNA4-RJ-081`.
- Generated CDNA4 SOP1 class and decoder inventory matches XML `ENC_SOP1`
  records exactly: all 54 manual/XML opcode rows have constructors and
  non-invalid `sub_decode_sop1` entries, while opcode holes 47 and 49 decode as
  invalid. `CDNA4-RJ-085` records the separate no-literal legality and size
  issue for selected SOP1 opcodes.
- Generated CDNA4 SOP2 class and decoder inventory matches XML `ENC_SOP2`
  records exactly: all 53 XML opcode rows, including XML-only opcode 43, have
  constructors and non-invalid primary decode-table entries. `CDNA4-RJ-086`
  records the separate no-literal legality and size issue for
  `S_RFE_RESTORE_B64`, and `CDNA4-XML-060` records the manual-source drift for
  opcode 43.
- Generated CDNA4 SOPK class and decoder inventory matches XML `ENC_SOPK`
  records exactly: all 21 XML opcode rows have constructors and non-invalid
  primary decode-table entries, opcode 19 decodes invalid, and only opcode 20
  uses an instruction extension word as an operand. Existing gaps cover the
  semantic issues found in this inventory: `CDNA4-RJ-061` for
  `S_CBRANCH_I_FORK`, `CDNA4-RJ-075` for remaining HWREG/SETREG behavior, and
  `CDNA4-RJ-071` for implicit SCC metadata. The HWREG ID map and
  `S_SETREG_IMM32_B32` literal visibility are patched.
- Generated CDNA4 SOPC class and decoder inventory matches XML `ENC_SOPC`
  records exactly: all 20 XML opcode rows have constructors and non-invalid
  `sub_decode_sopc` entries, while the remaining slots 20 through 127 decode
  invalid. `CDNA4-RJ-054` records the existing `S_SETVSKIP` execution gap,
  `CDNA4-RJ-071` records the SOPC compare/bitcmp SCC metadata gap, and
  `CDNA4-RJ-088` / `CDNA4-RJ-089` record the distinct GPR-indexing operand and
  M0 metadata issues found in this full SOPC pass.
- CDNA4 normal SALU literal replacement exists for SOP1/SOP2/SOPC instructions
  that use selector 255; representative constructors replace the source operand
  with `OPR_SIMM32` in the implementation.
  The SETREG_IMM and `S_SET_GPR_IDX_ON` gaps above are limited to their
  instruction-specific operand contracts.
- CDNA4 shared SALU helpers implement the audited signed/unsigned add/sub SCC
  behavior for ordinary arithmetic: signed add/sub use overflow, unsigned
  add/sub and addc/subb use carry/borrow, and ADDK uses signed overflow. The
  scalar SCC predicate gap from this slice is the max-equality predicate;
  `CDNA4-RJ-087` records the separate `S_ABSDIFF_I32` destination-value issue.
- CDNA4 special scalar selectors for VCCZ, EXECZ, and SCC read live wavefront
  state in `resolve_src_scalar()` in the implementation. The remaining STATUS/raw-state drift is covered by the earlier
  `CDNA4-RJ-053` finding.
- Chapter 5's former HWREG-map and `S_SETREG_IMM32_B32` literal-metadata
  findings are patched. `CDNA4-RJ-075` retains only unsupported register
  state/side effects and SETREG spacing behavior.
- CDNA4 ordinary PC-relative branch behavior is implemented for `S_BRANCH` and
  SCC/VCC/EXEC conditional branches. The constructors set branch flags and the
  execute bodies apply the `pc + 4 + simm16*4 - size_` formula in the implementation.
- Generated CDNA4 SOPP class and decoder inventory matches XML `ENC_SOPP`
  records exactly: 32 constructors and 32 non-invalid `sub_decode_sopp` entries
  cover opcodes 0 through 31. The SOPP gaps above are semantic/runtime gaps,
  plus the separate manual/XML source drift recorded as `CDNA4-XML-059`.
- CDNA4 direct PC operations have concrete behavior: `S_GETPC_B64` writes
  `PC+size`, `S_SETPC_B64` sets PC from the scalar source, and `S_SWAPPC_B64`
  writes the next PC while branching through the scalar source in the implementation.
- CDNA4 `S_CALL_B64` writes `PC+size` to the destination SGPR pair and uses the
  documented PC-relative branch formula in the implementation. Although the constructor uses the generic `INDIRECT_CALL` call marker,
  it also exposes `branch_offset_bytes()`, and CFG recovery distinguishes this
  branch-offset-bearing case as a direct call edge.
- CDNA4 `S_WAITCNT` immediate decoding matches the XML bit layout for the
  audited fields: low VM bits, EXP bits, LGKM bits, and high VM bits are
  extracted in the implementation.
- The core CDNA4 `S_BARRIER` release path is not absent: CU state release waits
  until all non-halted wavefronts in the same dispatch/workgroup are at the
  barrier and then releases the blocked waves together in the implementation.
  The remaining barrier gaps are architectural status exposure, ISA-limit
  validation, and edge-case tests.
- The CDNA4 generated SMEM SBASE operand scaling regression is covered by
  the relevant tests, including CDNA4
  64-bit loads, 128-bit scalar-buffer descriptors, and store-family source
  ordering. The SMEM gaps above are about later address semantics, dependency
  behavior, and cache/atomic side effects rather than that operand-width fix.
- Generated CDNA4 SMEM class and decoder inventory matches the XML
  `ENC_SMEM` records exactly: 84 generated constructors and 84 non-invalid
  `sub_decode_smem` entries cover the 82 manual opcode-table entries plus the
  two XML-only `S_ATC_PROBE*` opcodes. The decoder table places the manual
  load/store/cache/time/discard/atomic opcodes in their documented slots and
  leaves the holes invalid in the implementation.
- Generated CDNA4 `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` decode from the XML and
  no-op in execution in the implementation. Because the CDNA4 manual does not list those opcodes, this audit
  records the mismatch as `CDNA4-XML-058` rather than a manual-derived runtime
  gap.
- Production dispatch sets the initial `EXEC` mask from
  `initial_exec_mask_for_wave()` before register initialization in the implementation, and that helper accounts for partial workgroups and grid bounds in the implementation.
  The Chapter 3 EXEC gap above is therefore about HWREG-visible raw `STATUS`,
  not missing active-lane initialization.
- CDNA4 packed `VGPR0` workitem IDs are initialized by the dispatch path using
  the Chapter 3 `{Z,Y,X}` packing in the implementation. The remaining launch-initialization gaps are TTMP payloads
  and the optional `TG_SIZE` system SGPR.
- CDNA4 generated scaled code does extract the scale exponent with
  `(scale_bits >> 23) & 0xFFu`, matching the manual's E8M0/exponent-only
  direction at a high level. The remaining gap is the missing mode/configuration
  behavior around that conversion.
- Generated scaled FP8/BF8 conversion definitions 565-572 implement the
  high-level `OPSEL` and scale plumbing checked in this slice: packed and
  stochastic narrow forms preserve the old destination, packed widening forms
  use `OPSEL[0]`, single widening forms use `OPSEL[1:0]`, and the scale source
  is reduced to the F32 exponent field in the implementation. The remaining issues are the recorded state, source-legality, and
  alignment gaps.
- Generated scaled FP4 code preserves the old destination for byte writes via
  implicit destination uses and read-modify-write execution, for example
  `V_CVT_SCALEF32_PK_FP4_F32` in the implementation and `V_CVT_SCALEF32_SR_PK_FP4_F32`.
- Generated scaled FP6/BF6 wide paths exist for the CDNA4 `PK32` and `2XPK16`
  forms and use 32-element 6-bit pack/unpack helpers, for example
  `V_CVT_SCALEF32_2XPK16_FP6_F32` in the implementation and `V_CVT_SCALEF32_PK32_F32_FP6`.
- Generated stochastic FP4/FP6/BF6 scaled conversions advance the PRNG between
  elements with `util::prng_advance`, matching the manual's "internal
  V_PRNG_B32 but not written" direction at a high level; representative
  generated calls appear in the implementation.
- `util::fp4_e2m1_to_f32`, `util::fp6_e2m3_to_f32`, and
  `util::bf6_e3m2_to_f32` plus their RNE/SR narrow helpers encode the CDNA4
  FP4/FP6/BF6 range shape, no-Inf/no-NaN destinations, saturation, and tiny
  underflow-to-zero behavior in the implementation.
- CDNA4 generated MUBUF and MTBUF constructors do compensate for XML's fixed
  64-bit `VADDR` metadata by deriving the public address operand width from
  `IDXEN`/`OFFEN`, and scale `SRSRC` to the aligned SGPR descriptor base in the implementation.
- CDNA4 generated MUBUF/MTBUF class inventory covers the audited manual opcode
  tables: 16 MTBUF declarations in the implementation
  and 74 MUBUF declarations in the implementation,
  with generated constructor/execution bodies running through
  the implementation.
- CDNA4 raw MUBUF byte/short/dword loads and stores are generated with useful
  basic element sizes, sign extension, D16 low/high writeback flags, and global
  memory-pipeline routing. The vector-buffer gaps above are about descriptor,
  formatted conversion, address edge cases, LDS, atomic dataflow metadata,
  floating-atomic numeric behavior, and precise cache policy rather than
  absence of all raw-buffer execution.
- The vector memory pipeline does preserve/zero D16 results according to the
  emulator's SRAM ECC flag, matching the high-level direction of the CDNA4 D16
  ECC note for implemented raw D16 paths in the implementation.
- CDNA4 special decode does consume the four-word VOP3PX2 scale form and stores
  the two prefix words in `raw_words_`; the generated constructors set
  `size_ = 16` and `raw_encoding_` in the implementation.
- `ABID[0]=0` dispatches the unscaled mixed-format MFMA helper, matching the
  manual's "runs without scale source" behavior for that bit in the implementation.
- Dense MFMA generated classes do use `ACC[0]`/`ACC[1]` to select Arch VGPR
  versus AccVGPR sources and `ACC_CD` for C/D register-bank selection. The
  constructor rewrites representative dense operands in the implementation, and execution resolves physical bases with `dst_base()`,
  `src_base()`, and `resolve_acc()`.
- Generated MFMA and SMFMAC classes are tagged with the broad `MFMA`
  instruction flag, and rocjitsu has generated `WAITCNT` flags plus memory wait
  counters for ordinary memory dependencies. The Section 7.6 MAI gap above is
  about missing producer/consumer hazard predicates and wait-count rules, not a
  total absence of MFMA or wait-counter classification.
- CDNA4 dense I8 MFMA paths execute integer multiply-add helpers and therefore
  naturally ignore floating-point MODE state. The Section 7.4 MAI gap above is
  about missing floating-point denorm/RNE/exception modeling, not a claim that
  the integer I8 helper should consult MODE.
- Shared dense MFMA layout helpers implement the manual's lane/item formulas:
  `input_loc()` computes input item/register/lane placement, `output_loc_32()`
  implements the 4-row F32/I32 output grouping, and `output_loc_64()`
  implements the paired-register F64 output layout in the implementation.
- Dense MFMA helper paths apply `CBSZ`/`ABID` A-lane broadcast and `BLGP`
  B-lane permutation in the scalar path and route non-zero broadcast fields out
  of the SIMD fast paths, for example `permute_a_lane()` / `permute_b_lane()`
  in the implementation and `exec_f32_mixed()`.
- `permute_b_lane()` implements all documented ordinary `BLGP` values 0
  through 7, including the rotate-16 case, in the implementation.
- F64 MFMA generated code passes `inst_.blgp` as the `exec_f64()` negation mask
  and does not pass `CBSZ`/`ABID`, matching the manual's alternate F64 field
  meaning. Representative generated calls are in the implementation, and `exec_f64()` applies bits 0/1/2 to
  A/B/C in the implementation.
- Generated SMFMAC constructors model `SRC2` as an Arch VGPR operand and do
  not let `ACC_CD` redirect it to AccVGPR, while destination/C selection still
  uses `ACC_CD`. Representative constructors and execution paths are in the implementation.
- Sparse MFMA helper paths ignore `EXEC` by construction and write the full
  matrix footprint from `ComputeUnit` state, matching the manual's
  execution-mask ignore rule at a high level. Representative helpers are in the implementation.
- Existing tests cover the F64 `BLGP`-as-negation path, including generated
  CDNA4 instruction execution, in the relevant tests. SIMD exact tests also cover representative non-default
  `CBSZ` and `BLGP` helper cases in the relevant tests.
- Dense MFMA reads and writes use full-wave masks, matching the manual's
  execution-mask ignore rule at a register-access level:
  `mfma_full_lane_mask()` and the MFMA region helpers are in the implementation.
- The generated F8F6F4 MFMA execution dispatches by `CBSZ`/`BLGP` matrix format
  through `dispatch_matrix_fmt_pair`, matching the manual's use of those fields
  as A/B format selectors for this instruction family. The dispatch table covers
  all 25 documented A/B combinations for `FP8`, `BF8`, `FP6`, `BF6`, and `FP4`
  in the implementation.
- F8F6F4 MFMA execution ignores `EXEC` and writes the full wave, matching the
  manual's "forces it to 1 for all threads" rule: MFMA region helpers use
  `mfma_full_lane_mask` in the implementation, and the F8/F6/F4 paths use those full-lane
  region writes.
- The generated F8F6F4 MFMA operand classes match the manual's source legality
  at decode/disassembly level: `SRC0` and `SRC1` are
  `OPR_SRC_VGPR_OR_ACCVGPR`, while `SRC2` is
  `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST`, for example
  the implementation.
- Generated FP8/BF8 packed and stochastic narrow converts, scaled FP4 byte
  converts, and `V_CVT_PKACCUM_U8_F32` now surface and read the old destination
  before partial writes through operand metadata and read-modify-write code,
  even where XML marks `VDST` as output-only.
- Generated VOP3 FP8/BF8 widening aliases for definitions 84-87 select source
  bytes through `OPSEL[1:0]` or packed source words through `OPSEL[0]`, matching
  the definition-level VOP3 behavior in the implementation. `CDNA4-RJ-012` covers the separate 64-bit destination-alignment
  issue, and `CDNA4-RJ-083` covers VOP1 SDWA ignored fields.
- Generated `V_ASHR_PK_{I8,U8}_I32` masks the shift count to five bits, performs
  signed arithmetic shifts, clamps to the documented signed/unsigned 8-bit
  intervals, and packs the two bytes before the true16 destination write in the implementation.
- Generated `V_CVT_PK_F16_F32` and `V_CVT_PK_BF16_F32` write full 32-bit
  packed destinations and use the RNE F16/BF16 conversion helpers in the implementation; the Python semantic tests also pin BF16 RNE lowering in the codegen tests.
- Generated `V_CVT_SCALEF32_PK_BF16_{FP8,BF8}` extracts `OPSEL[0]` source-word
  selection and the exponent-only scale field in the implementation. The remaining scaled-conversion gaps are the previously recorded
  MODE/source-legality/alignment issues, not absence of these two generated
  BF16 widening forms.
- Generated CDNA4 VOP3P machine state keeps bit 14 as a separate
  `op_sel_hi_2` field in the implementation, and the generator has an explicit CDNA4 field-rename override
  in the code generator. That preserves
  source-2 high-half selection despite the CDNA4 XML folding bit 14 into
  `OP_SEL_HI`.
- `V_PK_MOV_B32` uses `read_lane64()` for both sources and selects output dwords
  with `OPSEL[0]` and `OPSEL[1]` in the implementation, matching the CDNA4 manual's special `V_PK_MOV_B32`
  selector behavior for scalar pairs and VGPR gather.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` in the implementation. They use multiply-add rather than fused FMA; the detailed
  instruction pseudocode and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- Packed 32-bit helpers do not apply clamp or other output modifiers, matching
  the packed 32-bit statement in the cited manual passage that output modifiers
  are not supported for those instructions.
- CDNA4 generated VOP2/VOP3 inventory covers all Chapter 12.7 opcodes 0 through
  61. The expected VOP3 holes for the four literal-only `_MK`/`_AK` opcodes are
  invalid in the VOP3 decode table, while the VOP2 decode table dispatches the
  literal forms through their generated classes.
- `V_CNDMASK_B32` VOP3 correctly uses source 2 as the scalar condition source
  and applies B32 source modifiers only to sources 0 and 1. `CDNA4-RJ-076`
  records the separate scalar-source-combination validation issue.
- Generated VOP3 carry-out forms use arbitrary SGPR-pair `SDST` destinations
  and read carry-in from `SRC2` for ADDC/SUBB/SUBBREV. `CDNA4-RJ-080` records
  the separate integer saturation/clamp gap.
- Generated `V_MAC_F16` VOP3 handles the manual's non-standard `OPSEL[3]`
  destination-half rule by selecting the old destination half and passing the
  OPSEL-derived write location into the true16 destination helper.
- Generated `V_LDEXP_F16` sign-extends the 16-bit exponent source, matching the
  instruction-local source-size rule.
- Generated CDNA4 VOP1 and VOP3 class inventory covers all 85 detailed Chapter
  12.8 VOP1 opcode definitions. `CDNA4-RJ-098` records the concrete XML-only
  `V_SCREEN_PARTITION_4SE_B32` hard stub without speculating about legacy
  EXP/LOG semantics absent from the current XML.
- Generated VOP3 FP8/BF8 widening aliases for definitions 84-87 select source
  bytes through `OPSEL[1:0]` or packed source words through `OPSEL[0]`, matching
  the definition-level VOP3 behavior. `CDNA4-RJ-083` records only the VOP1 SDWA
  ignored-field issue for these widening converts.
- `V_PERMLANE32_SWAP_B32` implements the manual's lane-0-through-31 row-pair
  swap shape in both generated VOP1 and VOP3 execute bodies in the implementation. Swap-style source metadata is now surfaced, and
  `V_PERMLANE16_SWAP_B32` performs its wave64 second pass.
- Generated CDNA4 FLAT classes intentionally cover the 54 FLAT opcode-table
  entries and use `SEG`-based mnemonic rewriting to present overlapping
  scratch/global forms; the 54 declarations run from
  the implementation,
  with constructor/execution bodies from
  the implementation. `CDNA4-RJ-040` records the separate missing direct-to-LDS-only
  `GLOBAL_LOAD_LDS_*`/`SCRATCH_LOAD_LDS_*` opcodes. These constructors adjust
  public address operand width based on `SEG` and `SADDR`, and preserve
  ACC-bank addressing for data/destination operands.
- The shared flat address helper implements the common 64-bit FLAT,
  SGPR-base-plus-VGPR-offset GLOBAL, VGPR-pair GLOBAL, and basic
  `FLAT_SCRATCH + lane_stride + offset` SCRATCH formulas in the implementation.
- Generated CDNA4 FLAT load/store paths implement raw byte/short/dword element
  sizes, sign extension, D16 low/high writeback flags, ACC addressing, and
  vector-memory routing for ordinary VGPR-return memory operations.
- CDNA4 `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` execute through shared helpers
  that use 64-lane grouping on CDNA4, apply `EXEC` to source/writeback as
  expected, and use ascending lane order so the highest-numbered active
  `DS_PERMUTE_B32` source wins collisions in the implementation. The swizzle gap above is limited to
  `DS_SWIZZLE_B32`'s missing FFT/rotate modes.
- Generated CDNA4 DS class and mnemonic inventory covers the audited opcode
  table: 126 declarations in the implementation,
  constructor/execution bodies from
  the implementation,
  and `sub_decode_ds` dispatch entries in the implementation. The data-share gaps above are semantic/runtime gaps rather than
  missing generated DS instruction records.
- CDNA4 transpose-load execution is not absent: generated DS bodies set
  `d->transpose`, and local-memory completion invokes `transpose_response()` in the implementation.
  The remaining transpose gap is validation and focused layout coverage.
