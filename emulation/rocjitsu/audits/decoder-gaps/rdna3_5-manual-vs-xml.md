# RDNA3.5 Manual vs XML Gaps

Architecture: RDNA3.5

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna3.5/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_rdna3_5.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1.1-1.2 Introduction, terminology, hardware overview, LDS/GDS/cache/device memory | Audited statically | Checked document scope, notation/suffix terminology, wave/workgroup/WGP/CU/SIMD/LDS/GDS/VMEM terms, host/command-processor/memory-controller overview, LDS/GDS/cache hierarchy, FP-exception interrupt capability, and relaxed memory/acknowledgment prose against XML architecture metadata, data-format records, functional groups, and instruction descriptions. Detailed operational gaps remain tracked in later semantic sections. |
| 2 intro / 2.1-2.4 Shader concepts, wave32/wave64, shader types, work-groups, CU/WGP modes, and shader padding | Audited statically | Checked shader execution roles, `EXEC` participation summary, wave32/wave64 issue model, compute/graphics shader launch concepts, workgroup sharing/barrier limits, CU/WGP LDS placement rules, WGP-mode LDSDIR exclusions, and shader-padding/prefetch-distance prose against XML functional groups, operand descriptions, instruction entries, and existing detailed semantic gaps. |
| 3.1-3.2 Wave state overview, PC, EXEC, and `EXEC==0` skipping | Audited statically | Checked the readable/writable wave-state inventory, per-wave/shared ownership notes, PC alignment and direct-control transfer rules, branch PC-relative formula, debug PC wording, `EXEC`/`VCC` wave32 low-half handling, and zero-`EXEC` skip/issue/counter exceptions against XML state operands, HWREG metadata, wait-counter operands, branch instruction entries, and functional groups. |
| 3.3.1 SGPRs | Audited statically | Checked SGPR/VCC/TTMP allocation, VCC-as-SGPR dependency wording, scalar alignment and endian ordering, region-crossing restrictions, out-of-range source/destination behavior, TTMP privilege, SAVEEXEC/WREXEC exceptions, SMEM return-data range notes, and S_MOVREL index/range behavior against XML scalar operand tables and instruction entries. |
| 3.3.2 VGPRs | Audited statically | Checked VGPR allocation granularity, nonzero allocation requirement, `S_SENDMSG` deallocation contract, VGPR out-of-range detection and consequences for destinations, `V_SWAP`/`V_SWAPREL`, multiple destinations, VMEM/export sources, VALU sources, and `VOPD`/`V_MOVREL` exceptions against XML VGPR operand tables and send-message metadata. |
| 3.3.3 Memory alignment and out-of-range behavior | Audited statically | Checked memory source/destination GPR range rules, destination-nullification behavior, PRT extra-VGPR range checks, image `DMASK` sizing, `SH_MEM_CONFIG.alignment_mode`, formatted-buffer alignment, and atomic MEMVIOL behavior against XML memory field descriptions and image/PRT metadata. |
| 3.3.4 LDS | Audited statically | Checked LDS allocation granularity and per-workgroup bounds, CU/WGP side placement, pixel-parameter LDS side restrictions, LDS/GDS alignment and out-of-range behavior, `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT MEMVIOL reporting, source-VGPR fallback, destination-VGPR nullification, and native alignment masks against XML DS/LDSDIR fields and DSMEM operands. |
| 3.4.1-3.4.2 STATUS and MODE registers | Audited statically | Checked HWREG register IDs and access permissions, STATUS field layout/write permissions/initialization/effects, and MODE floating-point, trap, exception, FP16 overflow, and performance controls against XML HWREG operand names and S_GETREG/S_SETREG instruction shells. |
| 3.4.3 M0 | Audited statically | Checked the complete per-consumer M0 use table, including LDSDIR, LDS ADDTID, GDS, ordered count, GWS, S/V_MOVREL, send-message, export, SMEM, and temporary uses against XML M0 operand classes, LDSDIR descriptions, send-message payload descriptions, and memory offset operand records. |
| 3.4.4-3.4.6 NULL, SCC, VCC, and VCCZ | Audited statically | Checked NULL read-zero/write-discard rules, NULL SALU destination SCC behavior, SCC producer/consumer roles, VCC compare and carry-out masks, VCCZ summary behavior, and wave32 low-half restrictions against XML scalar operands, SCC special-source operands, VCC operands, VOPC EXEC-mask descriptions, and branch instruction shells. |
| 3.4.7-3.4.8 FLAT_SCRATCH and hardware internal registers | Audited statically | Checked FLAT_SCRATCH launch initialization, no-scratch zero behavior, 256-byte alignment, trap-only write policy, HW_ID1/HW_ID2 field layouts, FLUSH_IB, SH_MEM_BASES, PC, and FLAT_SCRATCH HWREG targets against XML `OPR_FLAT_SCRATCH`, HWREG selector names, S_GETREG/S_SETREG shells, and scratch instruction operands. |
| 3.4.9 Trap and exception registers | Audited statically | Checked TTMP privilege overlap, trap-entry payload, `STATUS.TRAP_EN`, `MODE.EXCP_EN`, unmaskable MEMVIOL/illegal-instruction exceptions, `TRAPSTS` field semantics, and TMA/TBA access paths against XML trap/message instruction shells, HWREG selectors, and send-message return operands. |
| 3.4.10 Time | Audited statically | Checked `TIME`/`SHADER_CYCLES` and `REALTIME` clock domains, widths, synchronization, latency, and wait-counter usage against XML HWREG and return-message selectors plus `S_GETREG`/`S_SENDMSG_RTN` instruction shells. |
| 3.5.1-3.5.2 Initial `EXEC` and `FLAT_SCRATCH` state | Audited statically | Checked initial active-lane/null-wave `EXEC` semantics and scratch/no-scratch `FLAT_SCRATCH` launch initialization against XML `OPR_EXEC`, `OPR_FLAT_SCRATCH`, and scratch instruction operand records. |
| 3.5.3 SGPR Initialization | Audited statically | Checked PS/GS/HS/CS SGPR preload ordering, graphics-stage packed payloads, compute workgroup-id/TG_SIZE/TTMP system payloads, and no-alignment packing rules against XML scalar operand records and launch-control field absence. |
| 7.5 Packed Math | Audited for VOP3P packed-math slice only | Checked packed opcode inventory, VOP3P field rules, inline constants, MIX selectors, DPP legality, and clamp behavior. |
| 12.5.3 DS Stack Operations for Ray Tracing | Audited statically | Checked `DS_BVH_STACK_RTN_B32` opcode and operand coverage, LDS-only/offset semantics, packed `ADDR` state, stack-size encoding, push/pop behavior, and implicit DSMEM metadata. |
| 15.3.6 VOP3P | Audited for field inventory only | Checked generic VOP3P field positions, `OPSEL_HI2`, and opcode table coverage for the packed-math slice. |
| Instruction definitions 32-34 | Audited for MIX slice only | Checked FMA_MIX fused semantics, selector prose, and `NEG_HI` modifier wording. |
| Remaining RDNA3.5 manual sections | Not started | Full chapter-by-chapter audit remains outside the rows above. |

## Gaps

### RDNA3_5-XML-001: Packed VOP3P OPSEL legality restrictions are prose-only

Manual evidence:

- RDNA3.5 section 7.5 says packed math uses VOP3P and defines
  `OPSEL`/`OPSEL_HI` restrictions at `rdna3.5/README.md:2607` through `:2657`.
- `OPSEL` must be zero when the corresponding source or destination is 32-bit,
  except for MIX instructions; inline constants also require `OPSEL` zero
  because their value exists only in the low 16 bits.
- `OPSEL_HI` must be zero when the corresponding source or destination is
  32-bit or is a constant, except for MIX instructions.

XML evidence:

- The generic VOP3P field descriptions define `OP_SEL` and `OP_SEL_HI` as
  lower/upper 16-bit source selectors at
  `amdgpu_isa_rdna3_5.xml:2947` through `:2967`.
- Packed instruction entries such as `V_PK_FMA_F16` carry operand sizes and
  formats at `amdgpu_isa_rdna3_5.xml:120617` through `:120886`, but do not
  encode the per-source selector legality rules, the inline-constant
  restriction, or the MIX exception.

Impact:

An XML-only decoder or legality fuzzer can see the selector bit positions, but
cannot derive which selector values are legal for each packed source class.

### RDNA3_5-XML-002: Packed inline-constant behavior and DOT exceptions are prose-only

Manual evidence:

- Section 7.5.1 says inline constants used with packed math produce a value
  only in the low 16 bits, and float 16-bit sources receive an F16 constant
  value, at `rdna3.5/README.md:2678` through `:2686`.
- The DOT exception table at `rdna3.5/README.md:2690` through `:2700` gives
  per-opcode behavior for 8-bit, 4-bit, F16, and BF16 DOT forms, including
  which forms ignore `OPSEL` and which duplicate or derive upper halves.

XML evidence:

- The XML has inline-source operand classes and VOP3P literal extension
  encodings such as `VOP3P_INST_LITERAL` at `amdgpu_isa_rdna3_5.xml:13073`.
- Packed entries use formats such as `FMT_NUM_PK2_F16`, but the entries do not
  encode the low-16-only inline value rule, the F16-inline conversion rule, the
  BF16 upper-16 selection rule, or the DOT exception table.

Impact:

Generated semantics need manual-derived special cases for packed inline
constants. Operand size and data format alone do not distinguish all of these
behaviors.

### RDNA3_5-XML-003: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 7.5 says `V_FMA_MIX_*` uses VOP3P but is not packed math at
  `rdna3.5/README.md:2628` through `:2636`.
- The MIX selector note says `{OPSEL_HI[i], OPSEL[i]}` is a 2-bit source
  selector choosing between full FP32, low FP16, and high FP16 inputs at
  `rdna3.5/README.md:2661` through `:2674`.
- The MIX definitions say `NEG_HI` acts as an absolute-value modifier at
  `rdna3.5/README.md:16215` through `:16282`.

XML evidence:

- The generic VOP3P field description says `NEG_HI` negates the high operation
  at `amdgpu_isa_rdna3_5.xml:2927`, and says `OP_SEL`/`OP_SEL_HI` choose lower
  or upper 16-bit inputs at `:2947` through `:2967`.
- `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` are present at
  `amdgpu_isa_rdna3_5.xml:123059`, `:123385`, and `:123711`, but their entries
  do not encode the MIX-only selector mapping or `NEG_HI` absolute-value
  behavior.

Impact:

The generic XML field descriptions imply the wrong interpretation for MIX. XML
consumers need a hard-coded MIX override or manual-derived metadata.

### RDNA3_5-XML-004: `DS_BVH_STACK_RTN_B32` stack-state semantics are prose-only

Manual evidence:

- Section 12.5.3 defines `DS_BVH_STACK_RTN_B32` as an LDS-only per-thread BVH
  stack instruction and gives the field contract at `rdna3.5/README.md:4952`
  through `:4973`: `GDS` must be zero, `OFFSET0` is unused, `OFFSET1[5:4]`
  carries stack size, and `ADDR` is a packed in-out stack address.
- The instruction definition repeats that `OFFSET1[5:4]` selects stack sizes
  `{8, 16, 32, 64}`, describes `DATA0` as the last visited node pointer and
  `DATA1` as four candidate pointers, and gives the push/pop pseudocode at
  `rdna3.5/README.md:24705` through `:24750`.

XML evidence:

- The generic `ENC_DS` field descriptions only say `OFFSET0` and `OFFSET1`
  are DS offsets whose meanings are instruction-specific at
  `amdgpu_isa_rdna3_5.xml:3463` through `:3480`.
- The `DS_BVH_STACK_RTN_B32` XML entry has the opcode and explicit operand
  widths, plus implicit DSMEM input/output operands, at
  `amdgpu_isa_rdna3_5.xml:21850` through `:21898`; it does not structure
  `GDS == 0`, the `OFFSET1[5:4]` stack-size field, the packed `ADDR` layout,
  `DATA_VALID`, stack wrap/exhaustion, memory invalidation, or push/pop order.

Impact:

An XML-only consumer can decode the instruction and see that DSMEM is touched,
but cannot reconstruct the state machine or field invariants needed for
execution, legality checking, or focused test generation.

### RDNA3_5-XML-005: Chapter 1 architecture and host-control overview is not machine-readable

Manual evidence:

- Chapter 1 defines the manual's scope as RDNA3.5 instruction set and
  shader-program accessible state at `rdna3.5/README.md:337`, then introduces
  RDNA3.5 as a parallel micro-architecture at `rdna3.5/README.md:339`.
- The terminology table defines dispatch, work-group, wave, WGP, CU, SIMD32,
  LDS, GDS, VMEM, literal, sampler, texture resource, and buffer resource terms
  at `rdna3.5/README.md:366` through `:407`.
- Chapter 1.2 says the device includes a processor array, command processor,
  and memory controller; the command processor reads host-written
  memory-mapped registers and sends hardware-generated completion interrupts,
  while the memory controller has direct access to device and host-specified
  system memory, at `rdna3.5/README.md:461`.
- Chapter 1.2.1 through 1.2.3 describe independent WGP pipelines, automatic
  instruction fetch into on-chip caches, hardware FP-exception interrupts, LDS
  geometry and the 64KiB per-workgroup allocation limit, 4KiB GDS with
  append/consume support, cache hierarchy, cache-scope acknowledgments, and
  relaxed consistency at `rdna3.5/README.md:477`, `:479`, `:481`, `:497`,
  `:501`, and `:509` through `:516`.

XML evidence:

- The top-level XML architecture metadata records only `AMD RDNA 3.5` and
  architecture ID `9` at `amdgpu_isa_rdna3_5.xml:11` through `:12` before
  entering instruction encoding records.
- Related records are narrow: data-format layouts exist under `FMT_NUM_BF16`,
  `FMT_NUM_F16`, `FMT_NUM_F32`, and `FMT_NUM_F64` at
  `amdgpu_isa_rdna3_5.xml:173083` through `:173205`, and functional groups
  such as SALU, SMEM, VALU, VMEM, MESSAGE, WAVE_CONTROL, and TRAP are listed at
  `amdgpu_isa_rdna3_5.xml:189879` through `:189916`.
- Searching XML for command-processor, memory-controller, host-interrupt,
  relaxed-consistency, 128KiB/64KiB LDS geometry, 4KiB GDS, and append/consume
  architecture wording finds no architecture-topology or host-control schema;
  the GDS hits are only instruction-level descriptions such as GS streamout
  register operations at `amdgpu_isa_rdna3_5.xml:21375` and `:21411`.

Impact:

XML consumers cannot derive the Chapter 1 device topology, host launch and
completion relationship, FP-exception interrupt capability, memory hierarchy,
LDS/GDS topology, or relaxed-consistency overview from XML alone. Validators,
emulators, and documentation generators need manual prose or separate
architecture configuration for that layer, even when instruction-level field
decode comes from XML.

### RDNA3_5-XML-006: Chapter 2 wave32/wave64 issue model is prose-only

Manual evidence:

- Chapter 2.1 says both wave32 and wave64 are supported for all operations, and
  that shader programs are compiled for one fixed wave size at
  `rdna3.5/README.md:543` through `:545`.
- Wave32 issues each instruction once, while wave64 typically issues VALU and
  vector-memory instructions, including LDS, texture, buffer, and flat, twice;
  scalar ALU, scalar memory, branches, messages, and exports issue once at
  `rdna3.5/README.md:547`.
- Chapter 2.1 records wave64 skip-half rules for zero `EXEC` halves, the VMEM
  outstanding-operation exception, and the rule that VALU instructions writing
  SGPRs are not half-skipped at `rdna3.5/README.md:549` through `:553`.
- The same section says both wave64 passes use the pre-instruction wave state,
  the second pass increments selected carry/divergence inputs and outputs, and
  wave32 ignores upper `EXEC`/`VCC` bits at `rdna3.5/README.md:555` through
  `:565`.

XML evidence:

- XML encodes instruction families and operands, and one mask operand
  description notes that 64-bit masks may be truncated to 32 bits in wave32 at
  `amdgpu_isa_rdna3_5.xml:173395` through `:173396`.
- Searches found no architecture record for fixed shader wave size, the
  functional-family issue-once/issue-twice split, wave64 half-skip rules, the
  pre-instruction-state rule for both wave64 passes, or the second-pass scalar
  input/output increments.
- Local instruction descriptions can mention wave32 or wave64 restrictions, but
  those entries do not structure the Chapter 2 general issue model.

Impact:

XML-only consumers can identify instruction families and operands, but cannot
derive RDNA3.5's wave32/wave64 issue scheduling, skipped-half behavior, or
wave64 scalar-input/pass interaction from XML alone.

### RDNA3_5-XML-007: Shader-type launch modes are absent from XML

Manual evidence:

- Chapter 2.2.1 describes compute shaders as dispatch programs over a 1D, 2D,
  or 3D grid, with the processor walking the grid, creating waves, and
  initializing each work-item with a unique grid address/index at
  `rdna3.5/README.md:571`.
- Chapter 2.2.2 defines pixel, geometry, and hull shader waves, says normal NGG
  geometry-engine launch initializes VGPRs with primitive/index and
  vertex-buffer data, and describes mesh-shader plus amplification-shader
  launch modes at `rdna3.5/README.md:573` through `:586`.

XML evidence:

- Searches found no shader-stage launch schema, compute grid-walk model,
  work-item initialization model, geometry-engine launch payload, mesh-shader
  mode, or amplification-shader mode in XML.
- XML contains only narrow stage-adjacent instruction names or message
  descriptions, such as `msg_hs_tessfactor` and `msg_gs_alloc_req` in the
  message table at `amdgpu_isa_rdna3_5.xml:178062` through `:178088`, not a
  machine-readable launch taxonomy or per-stage VGPR initialization contract.

Impact:

XML cannot drive or validate compute-vs-graphics shader wave creation, normal
geometry-engine VGPR setup, mesh-shader launch conversion, or amplification
shader control without manual prose or another launch ABI source.

### RDNA3_5-XML-008: Workgroup and CU/WGP mode constraints are prose-only

Manual evidence:

- Chapter 2.3 says a workgroup's waves share LDS, synchronize at barriers, are
  issued to the same WGP, and can run on any of that WGP's four SIMD32 units at
  `rdna3.5/README.md:590`.
- The same section states the WGP supports up to 32 workgroups and at most
  1024 work-items per workgroup; single-wave workgroups do not count against
  the 32-workgroup limit, do not allocate a barrier resource, and treat barrier
  operations as `S_NOP` at `rdna3.5/README.md:590` through `:592`.
- CU mode and WGP mode change LDS sharing and placement: CU mode splits LDS
  into upper/lower halves and keeps all workgroup waves resident within one CU,
  while WGP mode exposes one contiguous WGP LDS and may distribute workgroup
  waves across both CUs. `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are not
  supported in WGP mode at `rdna3.5/README.md:594` through `:604`.

XML evidence:

- Searches found no XML record for WGP workgroup capacity, the 1024-work-item
  workgroup limit, single-wave barrier-resource elision, CU-mode LDS
  split/isolation, WGP-mode LDS sharing, mixed CU/WGP residency, or WGP-mode
  exclusion for LDSDIR instructions.
- XML does contain `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` instruction shells at
  `amdgpu_isa_rdna3_5.xml:28998` and `:29043`, but those entries do not encode
  the Chapter 2 WGP-mode legality condition.

Impact:

An XML-only validator can decode the LDSDIR instructions and identify
workgroup-related instruction families, but cannot infer the RDNA3.5 placement,
capacity, barrier-resource, or CU/WGP-mode legality contracts.

### RDNA3_5-XML-009: Shader padding and prefetch-distance requirements are prose-only

Manual evidence:

- Chapter 2.4 says shaders must be padded with 64 extra DWORDs, or 256 bytes,
  beyond the end of the shader and recommends `S_CODE_END` padding to avoid
  prefetching uninitialized or unmapped memory at `rdna3.5/README.md:612`.
- The same section says the shader can prefetch 1, 2, or 3 64-byte cachelines
  ahead of the current PC, controlled by wave-launch state or
  `S_SET_INST_PREFETCH_DISTANCE`, at `rdna3.5/README.md:614`.

XML evidence:

- XML models `S_SET_INST_PREFETCH_DISTANCE` and its `S_INST_PREFETCH` alias as
  an instruction that changes instruction-prefetch mode at
  `amdgpu_isa_rdna3_5.xml:60859` through `:60863`.
- XML models `S_CODE_END` as an instruction that causes an illegal-instruction
  interrupt and marks the end of a shader buffer for debug tools at
  `amdgpu_isa_rdna3_5.xml:61145` through `:61146`, and models `S_ENDPGM` as a
  program terminator at `amdgpu_isa_rdna3_5.xml:61527` through `:61528`.
- Searches found no XML representation of the 64-DWORD/256-byte padding length,
  the code-object validation rule, the 1/2/3-cacheline legal prefetch-distance
  set, or the wave-launch state relation.

Impact:

XML consumers can decode the prefetch and padding-related instructions, but
cannot validate the Chapter 2 shader-buffer padding requirement or prefetch
distance contract from XML alone.

### RDNA3_5-XML-010: Chapter 3.1 wave-state inventory is only partially structured

Manual evidence:

- Chapter 3 opens by saying each wave has a private copy of visible shader
  state unless otherwise specified at `rdna3.5/README.md:616` through `:618`.
- The state table lists PC, VGPRs, SGPRs, LDS, `EXEC`, `EXECZ`, `VCC`, `VCCZ`,
  `SCC`, `FLAT_SCRATCH`, `STATUS`, `MODE`, `M0`, `TRAPSTS`, `TBA`, `TMA`,
  `TTMP0`-`TTMP15`, and the VM/VScnt/EXP/LGKM wait counters with sizes and
  ownership notes at `rdna3.5/README.md:620` through `:653`.

XML evidence:

- XML has nominal operand records for several pieces of state: `OPR_EXEC` is a
  64-bit vector execute mask at `amdgpu_isa_rdna3_5.xml:177125` through
  `:177131`; `OPR_PC` is a program-counter implied operand at `:177325`
  through `:177331`; `FMT_NUM_M64` describes 64-bit lane masks and wave32
  truncation at `:173395` through `:173396`; `OPR_HWREG` lists
  hardware-register IDs such as `hw_reg_mode`, `hw_reg_status`,
  `hw_reg_trapsts`, TBA, and flat scratch at `:177147` through `:177280`;
  `OPR_WAITCNT` structures exp/lgkm/vm fields at `:189743` through `:189779`.
- These records do not form a complete Chapter 3.1 state table: they do not
  encode the full visible-state inventory, per-wave versus shared ownership,
  shader read/write access policy, `TBA` bit-63/address replication wording,
  all counter names and sizes as wave state, or the relationship between raw
  `STATUS` bits and live `EXEC`/`VCC`.

Impact:

An XML-only state model can discover individual operand names and some encoded
fields, but cannot reconstruct the Chapter 3.1 architectural wave-state
contract without manual prose.

### RDNA3_5-XML-011: PC control-state rules are prose-only

Manual evidence:

- Chapter 3.2.1 says PC is a DWORD-aligned byte address, initialized to the
  first shader instruction, with the two low bits forced to zero at
  `rdna3.5/README.md:659`.
- The same section says `S_GETPC_B64`, `S_SETPC_B64`, `S_CALL_B64`,
  `S_RFE_B64`, and `S_SWAPPC_B64` transfer PC to and from an even-aligned SGPR
  pair sign-extended, that ordinary branches target the next instruction plus
  `offset * 4`, and that `S_TRAP` saves the PC of the trap instruction itself
  at `rdna3.5/README.md:661` through `:663`.
- The debug wording says a read PC points to the next instruction to issue and
  earlier instructions may not have completed at `rdna3.5/README.md:665`.

XML evidence:

- XML carries individual instruction entries for `S_GETPC_B64`,
  `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_RFE_B64` at
  `amdgpu_isa_rdna3_5.xml:44814` through `:44969`, and for `S_CALL_B64` at
  `:60558` through `:60592`.
- XML's relative-label operand describes a signed dword target relative to
  `PC+4` at `amdgpu_isa_rdna3_5.xml:177307` through `:177313`.
- Those entries do not encode the forced low-zero PC bits, sign-extension and
  even-SGPR-pair transfer details, debugger PC visibility, or the `S_TRAP`
  current-instruction-PC save rule.

Impact:

XML can identify the PC-manipulating instructions, but an XML-derived emulator,
validator, or fuzzer needs manual text to enforce PC alignment, PC save
semantics, and direct-PC operand invariants.

### RDNA3_5-XML-012: `EXEC==0` instruction skip policy is absent

Manual evidence:

- Chapter 3.2.2 defines `EXEC` as affecting vector ALU, vector memory, LDS,
  GDS, and export instructions, but not scalar execution or branches, and says
  wave32 only uses bits 31:0 with `EXECZ` reflecting that low half at
  `rdna3.5/README.md:669` through `:673`.
- Chapter 3.2.3 defines timing-visible `EXEC==0` skip behavior and exceptions:
  ordinary VALU skip except SGPR/VCC writers and WMMA, listed lane/control
  instructions and buffer invalidations never skip, selected SGPR/VCC-writing
  VALU issue twice in wave64, export skip depends on `Done`, `POS0`, and
  `SKIP_EXPORT`, and VMEM/LDS skipping depends on outstanding VM/VScnt, EXPcnt,
  or LGKM counters at `rdna3.5/README.md:675` through `:696`.

XML evidence:

- XML's `FMT_NUM_M64` notes mask truncation in wave32 at
  `amdgpu_isa_rdna3_5.xml:173395` through `:173396`, and branch entries for
  `S_CBRANCH_VCCZ` and `S_CBRANCH_EXECZ` reference implicit mask operands at
  `:61267` through `:61384`.
- Some encoding descriptions say vector compares respect `EXEC`, for example
  `ENC_VOPC` at `amdgpu_isa_rdna3_5.xml:1540`; DS and export encoding prose
  also mention `EXEC` at `:3397` through `:3399` and `:4331` through `:4352`.
- Searches found instruction names such as `V_NOP`, `V_READLANE_B32`, and
  `BUFFER_GL1_INV`, but no machine-readable per-family zero-`EXEC` skip table,
  no no-skip exception list, and no counter-dependent VMEM/LDS/export skip
  metadata.

Impact:

XML-only consumers can model mask operands and some per-instruction effects,
but cannot derive the Chapter 3 scheduler/issue/skip behavior or its
wait-counter and export exceptions.

### RDNA3_5-XML-013: SGPR alignment enforcement and scalar region rules are prose-only

Manual evidence:

- Chapter 3.3.1.3 says 64-bit SGPR data, PC pairs, scalar-memory base pairs,
  and VALU 64-bit SGPR sources require even SGPR alignment; values wider than
  64 bits and wide scalar-memory data GPRs require quad alignment; LSBs are in
  `SGPR[n]` and MSBs in `SGPR[n+1]`; and hardware enforces alignment by
  ignoring low address bits, with `*MOVREL*_B64` also ignoring the low index bit
  at `rdna3.5/README.md:714` through `:733`.
- Chapter 3.3.1.4 says GPR indexing and multi-DWORD operands must not cross
  scalar regions: SGPR/VCC, TTMP, and all other scalar-source values, at
  `rdna3.5/README.md:741` through `:745` and `:765` through `:773`.

XML evidence:

- XML enumerates scalar destination/source operand maps and carries a generic
  SGPR alignment description in `OPR_SDST`, `OPR_SREG`, and `OPR_SSRC` at
  `amdgpu_isa_rdna3_5.xml:177902` through `:178557`,
  `:184997` through `:185634`, and `:186734` through `:187390`.
- XML lists the `S_MOVRELS_*`, `S_MOVRELD_*`, and `S_MOVRELSD_2_B32`
  instruction entries with `M0` implicit operands at
  `amdgpu_isa_rdna3_5.xml:43113` through `:43380`.
- These records do not encode the hardware behavior of zeroing low SGPR address
  bits, the special `*MOVREL*_B64` index-LSB rule, the prohibition on crossing
  SGPR/VCC into TTMP or other scalar-source regions, or the fallback rule for
  out-of-range MOVREL source/destination addresses.

Impact:

An XML consumer can format scalar operands and see broad alignment text, but
cannot derive the exact architectural address-masking, region-validation, or
relative-index behavior from XML alone.

### RDNA3_5-XML-014: Scalar out-of-range, TTMP privilege, and EXEC-write exceptions are prose-only

Manual evidence:

- Chapter 3.3.1.4 says scalar sources and destinations use the 7-bit map
  `0-105=SGPR`, `106/107=VCC`, `108-123=TTMP`, and
  `124-127={NULL,M0,EXEC_LO,EXEC_HI}` at `rdna3.5/README.md:735` through
  `:740`.
- It then defines consequences: out-of-range scalar sources return zero, writes
  are ignored, user-mode TTMP reads return zero, user-mode TTMP writes are
  ignored, failed TTMP SALU writes do not update SCC, `WREXEC` and `SAVEEXEC`
  still write `EXEC` when `SDST` is out of range, and SMEM/MOVREL have
  additional range fallback rules at `rdna3.5/README.md:747` through `:773`.

XML evidence:

- XML enumerates the selector values for `s0` through `s105`, TTMP0-TTMP15,
  `VCC_LO`, `VCC_HI`, `NULL`, `M0`, `EXEC_LO`, and `EXEC_HI` in scalar operand
  tables at `amdgpu_isa_rdna3_5.xml:177902` through `:178592`,
  `:184997` through `:185634`, and `:186734` through `:187390`; it also has a
  separate `OPR_VCC` record at `:188947` through `:188955`.
- XML marks `S_AND_SAVEEXEC_B32` and `S_AND_NOT0_WREXEC_B32` destinations as
  `OPR_SREG` and exposes implicit `EXEC` and `SCC` operands at
  `amdgpu_isa_rdna3_5.xml:40989` through `:41023` and `:42753` through
  `:42789`.
- These entries do not encode the dynamic out-of-range read/write
  consequences, TTMP privilege checks via `STATUS.PRIV`, SCC suppression on
  failed TTMP writes, SMEM return-data range behavior, or the special rule that
  SAVEEXEC/WREXEC update `EXEC` even when the scalar destination is invalid.

Impact:

XML-derived validators can identify legal scalar selector spellings, but cannot
model Chapter 3.3.1's user/trap privilege policy or out-of-range execution
side effects without manual prose.

### RDNA3_5-XML-015: VGPR allocation and out-of-range consequences are not structured

Manual evidence:

- Chapter 3.3.2.1 says VGPRs are allocated in blocks of 16 for wave32 or 8 for
  wave64, up to 256 VGPRs per shader, and that a wave may not be created with
  zero VGPRs; devices with 1536 VGPRs per SIMD use blocks of 24 and 12 instead,
  at `rdna3.5/README.md:777` through `:781`.
- The same subsection says `S_SENDMSG` can voluntarily deallocate all VGPRs,
  after which the wave cannot reallocate them and the only valid action is
  termination, at `rdna3.5/README.md:781`.
- Chapter 3.3.2.2 defines `Vs`/`Ve` out-of-range tests for VGPR operands,
  `V_MOVREL` indexed cases, and consequences: destination OOR makes the
  instruction a NOP, `V_SWAP`/`V_SWAPREL` discard when either argument is OOR,
  multiple-destination VALU writes no GPRs, VMEM/export and VALU source OOR use
  VGPR0, and VOPD source addresses use a different modulo rule, at
  `rdna3.5/README.md:783` through `:815`.

XML evidence:

- XML enumerates VGPR operand namespaces in `OPR_SRC`,
  `OPR_SRC_VGPR`, `OPR_SRC_VGPR_OR_INLINE`, and `OPR_VGPR`; the first VGPR
  descriptions mention `NUM_VGPR` and, for 9-bit source operands,
  `SQ_SRC_VGPR_BIT`, at `amdgpu_isa_rdna3_5.xml:180116`,
  `:181411`, `:183150`, and `:188462`.
- XML includes the `msg_dealloc_vgprs` predefined `S_SENDMSG` value and a prose
  description that later vector instructions are ignored after deallocation at
  `amdgpu_isa_rdna3_5.xml:178068`.
- These records do not structure wave-size allocation block sizing, the
  nonzero launch rule, the live `VGPR_SIZE` bound used by each instruction, the
  per-instruction OOR consequences, or the `V_MOVREL` and VOPD exception rules.

Impact:

XML consumers can format VGPR operands and find a textual deallocate-VGPR
message, but cannot mechanically derive the Chapter 3.3.2 allocation and
out-of-range execution contract.

### RDNA3_5-XML-016: Memory out-of-range and alignment policy is only partially encoded

Manual evidence:

- Chapter 3.3.3 says the rules apply to LDS, GDS, buffer, global, flat, and
  scratch memory accesses, and defines memory-read and return-atomic behavior
  for out-of-range source and destination GPRs at `rdna3.5/README.md:820`
  through `:827`.
- The same subsection says destination-range testing includes all potentially
  returned VGPRs, includes the extra PRT status VGPR, nullifies return-atomic
  operations with out-of-range destinations by issuing with `EXEC==0`, and uses
  image `DMASK` bits when making image out-of-bounds determinations at
  `rdna3.5/README.md:825` through `:829`.
- Chapter 3.3.3 defines `SH_MEM_CONFIG.alignment_mode`, `DWORD` automatic
  alignment, `UNALIGNED` mode, formatted-operation alignment by component size,
  and atomic MEMVIOL on data-size misalignment at `rdna3.5/README.md:834`
  through `:844`.

XML evidence:

- XML carries related field fragments: image `DMASK` describes consecutive VGPR
  packing and D16 behavior at `amdgpu_isa_rdna3_5.xml:4140` through `:4155`,
  `TFE` is described as partially resident texture enable at
  `amdgpu_isa_rdna3_5.xml:3712` through `:3713`, and selected memory fields
  mention alignment or MEMVIOL cases such as SMEM negative descriptor offsets
  at `amdgpu_isa_rdna3_5.xml:852`.
- These records do not structure the Chapter 3.3.3 execution policy: source
  GPR OOR data consequences, destination VGPR OOR nullification by issuing with
  cleared `EXEC`, all-returned-VGPR and PRT-extra-VGPR range sizing, image
  `DMASK` as a range predicate, `SH_MEM_CONFIG.alignment_mode`, formatted-op
  alignment by data size, or MEMVIOL generation for misaligned atomics.

Impact:

XML-derived memory decoders can recover operand fields and some local field
descriptions, but cannot mechanically derive the general memory OOR,
nullification, PRT sizing, alignment-mode, or MEMVIOL contract from XML alone.

### RDNA3_5-XML-017: LDS allocation, placement, and LDS/GDS alignment policy are incomplete

Manual evidence:

- Chapter 3.3.4 says waves may allocate 0-64 KiB of LDS per wave/workgroup,
  that the allocation is shared by all waves in the workgroup, that LDS is
  allocated in 1024-byte blocks, and that all LDS accesses are restricted to
  that wave/workgroup allocation at `rdna3.5/README.md:846` through `:847`.
- The same section describes the two 64 KiB LDS blocks, CU-mode same-side
  placement and no-cross/no-wrap rule, WGP-mode placement or straddling, and
  pixel-parameter placement on the same CU side as the pixel-shader wave at
  `rdna3.5/README.md:848` through `:855`.
- Chapter 3.3.4.1 defines DS/LDS/GDS alignment behavior,
  `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT MEMVIOL
  reporting, read-zero and write-discard OOR behavior, source VGPR0 fallback,
  destination VGPR nullification, and native alignment masks at
  `rdna3.5/README.md:856` through `:890`.

XML evidence:

- XML exposes the generic DS field map, including `ADDR`, `DATA0`, `DATA1`,
  `GDS`, offsets, and `VDST`, at `amdgpu_isa_rdna3_5.xml:3252` through
  `:3493`.
- XML has representative DS load/store shells such as `DS_STORE_B32` and
  `DS_LOAD_B128` at `amdgpu_isa_rdna3_5.xml:16875` through `:16910` and
  `:22210` through `:22235`, plus `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, and
  `OPR_DSMEM` records at `amdgpu_isa_rdna3_5.xml:28998` through `:29059` and
  `:177114`.
- Searches found no structured XML records for `LDS_CONFIG`,
  `ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT alignment modes, native
  LDS/GDS alignment masks, per-workgroup LDS allocation granularity and bounds,
  CU/WGP side placement, pixel-parameter side constraints, or the source/dest
  VGPR range consequences specific to LDS operations.

Impact:

An XML-only DS/LDS consumer can decode LDS memory operands and the LDS/GDS
selector bit, but cannot derive the manual's allocation, placement, alignment,
or violation policy that determines whether an LDS operation reaches storage,
returns zero, is nullified, or reports a memory violation.

### RDNA3_5-XML-018: STATUS and MODE field semantics are prose-only

Manual evidence:

- Chapter 3.4 lists the wave-state registers readable or writable through
  `S_GETREG` and `S_SETREG`, including `MODE`, `STATUS`, `TRAPSTS`,
  `FLUSH_IB`, `SH_MEM_BASES`, flat scratch, hardware ID registers, and
  `SHADER_CYCLES`, with read/write, read-only, write-only, or privileged access
  qualifiers at `rdna3.5/README.md:894` through `:912`.
- Chapter 3.4.1 defines STATUS fields, bit positions, privileged-write
  permissions, initialization, and effects for `SCC`, wave priorities, `PRIV`,
  `TRAP_EN`, `EXPORT_RDY`, `EXECZ`, `VCCZ`, `IN_WG`, `IN_BARRIER`, `HALT`,
  `TRAP`, `VALID`, `SKIP_EXPORT`, `PERF_EN`, conditional-debug bits,
  `FATAL_HALT`, `NO_VGPRS`, `LDS_PARAM_RDY`, `MUST_GS_ALLOC`, `MUST_EXPORT`,
  `IDLE`, and `SCRATCH_EN` at `rdna3.5/README.md:914` through `:950`.
- Chapter 3.4.2 defines MODE fields and their effects for `FP_ROUND`,
  `FP_DENORM`, `DX10_CLAMP`, `IEEE`, `LOD_CLAMPED`, `TRAP_AFTER_INST`,
  `EXCP_EN`, `FP16_OVFL`, and `DISABLE_PERF` at `rdna3.5/README.md:954`
  through `:977`.

XML evidence:

- XML exposes `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` as
  HWREG-access instructions at `amdgpu_isa_rdna3_5.xml:60454` through
  `:60524`.
- XML's `OPR_HWREG` predefined values name `hw_reg_mode`, `hw_reg_status`,
  `hw_reg_trapsts`, `hw_reg_flush_ib`, `hw_reg_sh_mem_bases`,
  `hw_reg_shader_flat_scratch_lo/hi`, `hw_reg_hw_id1/2`, and related
  selectors, but leave their descriptions empty at
  `amdgpu_isa_rdna3_5.xml:177160` through `:177248`.
- Searches found no XML records for the STATUS field names or MODE field names
  above, their bit positions, read/write/privileged access policy,
  initialization rules, export/trap/liveness effects, floating-point mode
  effects, or exception-enable behavior.

Impact:

An XML-only state-register model can find the HWREG selector names and the
get/set instruction forms, but cannot reconstruct the STATUS/MODE layout or the
field-level behavior needed to model export waits, trap state, priority, live
summary bits, floating-point modes, or exception controls.

### RDNA3_5-XML-019: The complete M0 use table is prose-only

Manual evidence:

- Chapter 3.4.3 says each wave has one 32-bit `M0` register and lists distinct
  layouts or meanings for `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, LDS ADDTID,
  Global Data Share, GDS ordered count, global wave sync, `S/V_MOVREL`,
  `S_SENDMSG`/`_RTN`, export, SMEM, and temporary uses at
  `rdna3.5/README.md:979` through `:999`.
- The same table gives field-level constraints such as `LDS_PARAM_LOAD`
  byte-offset alignment, wave32 new-primitive-mask layout, `LDS_DIRECT_LOAD`
  data-type bits, LDS ADDTID 4-byte alignment, and GDS base/size packing at
  `rdna3.5/README.md:987` through `:996`.

XML evidence:

- XML exposes `M0` as operand type `OPR_SDST_M0`, with a description naming
  LDS/GDS addresses, relative indices, and send-message values, at
  `amdgpu_isa_rdna3_5.xml:178010` through `:178016`.
- `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` include implicit `OPR_SDST_M0`
  operands, and the direct-load description locally documents
  `M0[15:0]`/`M0[18:16]`, at `amdgpu_isa_rdna3_5.xml:28998` through
  `:29064`.
- XML includes scattered instruction-local M0 references, such as send-message
  payload descriptions at `amdgpu_isa_rdna3_5.xml:178058`, `:178063`, and
  `:178088`, and lists `OPR_SDST_M0` as a subtype of `OPR_SMEM_OFFSET` at
  `amdgpu_isa_rdna3_5.xml:178320` through `:178326`.
- These records do not form a structured table for all Chapter 3.4.3 M0
  consumers, layouts, and alignment constraints, and do not provide a single
  machine-readable source for the per-instruction interpretation of the same
  32-bit M0 state.

Impact:

An XML-only consumer can decode uses of the `M0` operand and recover a few
instruction-local descriptions, but cannot derive the full per-consumer M0
interpretation needed for legality checks, operand-state modeling, or targeted
test generation.

### RDNA3_5-XML-020: FLAT_SCRATCH and hardware-internal register behavior is prose-only

Manual evidence:

- Chapter 3.4 lists the S_GETREG/S_SETREG hardware-register target IDs and
  access permissions, including `FLUSH_IB`, `SH_MEM_BASES`,
  `FLAT_SCRATCH_LO/HI`, `HW_ID1/2`, and `SHADER_CYCLES`, at
  `rdna3.5/README.md:904` through `:912`.
- Chapter 3.4.7 says `FLAT_SCRATCH` is initialized by wave-launch hardware for
  waves with scratch allocation, is read-only outside the trap handler, returns
  zero when no scratch space is allocated, and must be a 256-byte-aligned byte
  address computed from `scratch_base + spi_scratch_offset` at
  `rdna3.5/README.md:1046` through `:1056`.
- Chapter 3.4.8 defines read-only hardware internal registers, the
  unpredictable/context-switch caveat for `HW_ID` and `*_BASE` values, the
  `HW_ID1` and `HW_ID2` bit layouts, the `FLUSH_IB` side effect,
  `SH_MEM_BASES` private/shared aperture bits, the warning against reading PC
  via `S_GETREG`, and the TMA/TBA access path at `rdna3.5/README.md:1058`
  through `:1093`.

XML evidence:

- XML has an `OPR_FLAT_SCRATCH` operand whose predefined value is only
  `flat_scratch[63:0]` at `amdgpu_isa_rdna3_5.xml:177691` through `:177699`.
- XML's `OPR_HWREG` operand records the HWREG selector field and names
  `HW_REG_FLUSH_IB`, `HW_REG_SH_MEM_BASES`,
  `HW_REG_SHADER_FLAT_SCRATCH_LO/HI`, `HW_REG_HW_ID1/2`, and
  `HW_REG_SHADER_CYCLES`, but their descriptions are `N/A` at
  `amdgpu_isa_rdna3_5.xml:177773` through `:177845`.
- XML exposes the `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` instruction shells at
  `amdgpu_isa_rdna3_5.xml:59399` through `:59486`, and scratch instruction
  entries carry implicit `OPR_FLAT_SCRATCH` operands at
  `amdgpu_isa_rdna3_5.xml:18849` through `:18852` and `:25081` through
  `:25084`. Searches found no structured FLAT_SCRATCH launch formula,
  256-byte alignment rule, no-scratch zero behavior, trap-only write policy,
  HW_ID bit layouts, SH_MEM_BASES private/shared bit layout, FLUSH_IB side
  effect, PC read warning, or TMA/TBA access note.

Impact:

An XML-only consumer can identify the flat-scratch operand and HWREG selector
names, but cannot reconstruct the launch, access-control, alignment,
field-layout, side-effect, or unpredictability rules needed for emulation and
legality testing.

### RDNA3_5-XML-021: Trap-status and exception-control behavior is prose-only

Manual evidence:

- Chapter 3.4.9 defines trap and exception register behavior, including
  user-mode TTMP read-zero/write-ignore behavior, trap-handler-only TTMP
  writes, read-only TMA/TBA access through `S_SENDMSG_RTN`, and the trap-entry
  payload `{TTMP1, TTMP0} = {7'h0, HT[0], trapID[7:0], PC[47:0]}` at
  `rdna3.5/README.md:1097` through `:1113`.
- The same section says `STATUS.TRAP_EN` controls whether traps are taken and
  that `s_trap` is converted to NOP when no trap handler is present at
  `rdna3.5/README.md:1115` through `:1119`.
- `MODE.EXCP_EN[8:0]` controls exception enables, while MEMVIOL and illegal
  instruction traps cannot be masked off, at `rdna3.5/README.md:1121` through
  `:1137`.
- The `TRAPSTS` table defines sticky exception status bits, host/context-save
  trap bits, illegal-instruction behavior, buffer-OOB reporting, wave-start/end
  markers, and trap-after-instruction reporting at `rdna3.5/README.md:1139`
  through `:1165`.

XML evidence:

- XML names `hw_reg_trapsts` and `hw_reg_shader_tba_lo/hi` in `OPR_HWREG`, but
  gives empty descriptions for those selectors at
  `amdgpu_isa_rdna3_5.xml:177171` through `:177224`.
- XML defines the `OPR_SENDMSG_RTN` operand and names message values including
  `msg_rtn_get_tma`, `msg_rtn_get_tba`, and `msg_rtn_get_tba_to_pc` at
  `amdgpu_isa_rdna3_5.xml:178149` through `:178205`.
- XML exposes `S_RFE_B64`, `S_SENDMSG_RTN_B32`, `S_SENDMSG_RTN_B64`, and
  `S_TRAP` instruction shells at `amdgpu_isa_rdna3_5.xml:44927` through
  `:45039` and `:61058` through `:61077`.
- Searches found no structured TTMP privilege rules, trap-entry TTMP payload,
  `STATUS.TRAP_EN` NOP conversion, `MODE.EXCP_EN` mask/unmaskable-exception
  policy, `TRAPSTS` sticky-field layout, or host/context-save trap-status
  semantics.

Impact:

An XML-only consumer can decode trap-control instructions and identify some
trap-related selectors, but cannot reconstruct the trap state machine, exception
enable/reporting rules, or handler-visible TTMP/TBA/TMA behavior needed for
emulation, legality checking, or focused tests.

### RDNA3_5-XML-022: Time-counter clock-domain and wait semantics are prose-only

Manual evidence:

- Section 3.4.10 distinguishes `TIME` from `REALTIME`: `TIME` measures graphics
  core clock cycles as a 20-bit counter, while `REALTIME` is a 64-bit
  fixed-frequency clock that usually runs at 100 MHz and continues regardless
  of shader or memory clock speed at `rdna3.5/README.md:1167` through `:1174`
  and `:1176`.
- The manual says `SHADER_CYCLES` is read through `S_GETREG`, is not
  synchronized across different SIMDs, should be used only for within-wave
  deltas, and has typical SALU latency around 8 cycles at
  `rdna3.5/README.md:1174`.
- The manual's `REALTIME` sequence uses `S_SENDMSG_RTN_B64` followed by
  `S_WAITCNT LGKMcnt == 0` at `rdna3.5/README.md:1180` through `:1183`.

XML evidence:

- XML names `hw_reg_shader_cycles` but gives it an empty description at
  `amdgpu_isa_rdna3_5.xml:177277` through `:177279`.
- XML names `msg_rtn_get_realtime` and says it returns the constant-frequency
  REFCLK time counter for 32-bit or 64-bit requests at
  `amdgpu_isa_rdna3_5.xml:178189` through `:178191`.
- XML exposes the `S_GETREG_B32` and `S_SENDMSG_RTN_B64` instruction shells at
  `amdgpu_isa_rdna3_5.xml:60454` through `:60480` and
  `amdgpu_isa_rdna3_5.xml:45014` through `:45039`, but does not encode the
  20-bit width, per-SIMD synchronization caveat, SALU latency, fixed-frequency
  clock-rate expectation, idle-clock independence, or required LGKM wait
  dependency.

Impact:

XML-only consumers can find the selector and message names, but cannot recover
the time-counter domains, widths, synchronization limits, or dependency
behavior needed for timing-sensitive emulation and tests.

### RDNA3_5-XML-023: Initial `EXEC` and `FLAT_SCRATCH` launch state is prose-only

Manual evidence:

- Section 3.5 says wave state is initialized from state data and dynamic
  wave-launch state before execution, and that some state is common across a
  draw while some is unique per wave, at `rdna3.5/README.md:1185` through
  `:1193`.
- Section 3.5.1 says `EXEC` normally starts as the active-thread mask, but some
  launch cases initialize `EXEC == 0` for "Null waves" that should do no work
  and exit immediately, at `rdna3.5/README.md:1195` through `:1197`.
- Section 3.5.2 says waves with scratch memory are initialized with
  `FLAT_SCRATCH` pointing at global memory, while waves without scratch get
  zero, at `rdna3.5/README.md:1199` through `:1201`.

XML evidence:

- XML records `OPR_EXEC` as the 64-bit vector execute mask and names the
  `exec` selector at `amdgpu_isa_rdna3_5.xml:177125` through `:177131`.
- XML records `OPR_FLAT_SCRATCH` as an implied 64-bit flat-scratch operand and
  names the `flat_scratch` selector at `amdgpu_isa_rdna3_5.xml:177136`
  through `:177142`.
- Scratch instructions reference `OPR_FLAT_SCRATCH` as an implicit operand,
  for example at `amdgpu_isa_rdna3_5.xml:22340` through `:22344`, but XML does
  not encode wave-launch state, active-lane mask derivation, null-wave exit
  behavior, or scratch/no-scratch initialization policy.

Impact:

XML-only consumers can identify the architectural registers and implicit
scratch operand, but cannot derive the initial `EXEC`/`FLAT_SCRATCH` values a
wave should see at launch. That launch-state model has to come from manual
prose or another non-XML source.

### RDNA3_5-XML-024: SGPR launch payload tables are prose-only

Manual evidence:

- Section 3.5.3 says SGPRs are initialized from SPI/COMPUTE program-resource
  register settings, only enabled values are loaded, enabled values are packed
  consecutively, and no SGPRs are skipped for alignment at
  `rdna3.5/README.md:1203` through `:1207`.
- The PS table defines user-data, `{bc_optimize, prim_mask, lds_offset}`,
  `{ps_wave_id, ps_wave_index}`, and provoking-vertex SGPR payloads at
  `rdna3.5/README.md:1209` through `:1228`.
- The GS and HS sections define combined-stage launch behavior, first-eight
  automatically initialized SGPRs, program-address payloads, off-chip LDS and
  stage-specific packed fields, FAST_LAUNCH variants, and user-SGPR ranges at
  `rdna3.5/README.md:1230` through `:1280`.
- The CS table defines up-to-16 user-data SGPRs, optional
  `work_group_id0/1/2` SGPRs, a packed `tg_size_en` payload, zeroed
  `TTMP4/5`, and `TTMP6` through `TTMP11` dispatch-packet/grid/wave payloads
  at `rdna3.5/README.md:1284` through `:1308`.

XML evidence:

- XML has scalar operand classes such as `OPR_SDST`, `OPR_SREG`, and
  `OPR_SSRC` that enumerate SGPR selector values and alignment notes at
  `amdgpu_isa_rdna3_5.xml:177336` through `:178021`,
  `amdgpu_isa_rdna3_5.xml:184431` through `:185071`, and
  `amdgpu_isa_rdna3_5.xml:186168` through `:187312`.
- Searches for `COMPUTE_PGM_RSRC2`, `SPI_SHADER_PGM_RSRC`,
  `SPI_SHADER_USER_DATA`, `tg_size`, `wave_id_in_group`,
  `ordered_append`, `work_group_size_in_waves`, `PS_wave_id`,
  `FAST_LAUNCH`, `LOAD_PROVOKING_VTX`, `dispatch packet addr`, and
  `dispatch grid` found no RDNA3.5 XML records.
- The scalar operand records do not encode shader-stage SGPR initialization
  order, enable-field names, no-alignment packing, graphics combined-stage
  payloads, compute workgroup-id payloads, the packed `TG_SIZE` system SGPR,
  or the RDNA3.5 compute TTMP launch payloads.

Impact:

XML-only consumers can decode instructions that reference SGPRs, but cannot
derive which SGPRs are preloaded for PS/GS/HS/CS launch or how enabled launch
payloads are packed. Emulators and launch-state tests need the manual or a
separate launch ABI source.

## No-Gap Notes

- Chapter 3.1 state-operand narrow match: XML has concrete records for
  `OPR_EXEC`, `OPR_VCC`, `OPR_PC`, `OPR_HWREG`, and the exp/lgkm/vm wait-count
  fields. `RDNA3_5-XML-010` covers the missing full wave-state table and
  ownership/access semantics rather than a complete absence of state operands.
- Chapter 3.2 mask-behavior boundary: XML exposes mask operands and some
  per-family `EXEC` wording for VOPC, DS, and export encodings. The remaining
  gap is the manual's zero-`EXEC` scheduler skip table, no-skip exceptions, and
  counter-dependent issue rules.
- Chapter 3.3.1 scalar-selector narrow match: XML does enumerate the RDNA3.5
  scalar selector values for SGPRs, VCC, TTMPs, NULL, M0, EXEC, inline
  constants, and `OPR_VCC`. `RDNA3_5-XML-013` and `RDNA3_5-XML-014` cover the
  missing dynamic rules rather than a missing scalar operand namespace.
- Chapter 3.3.2 VGPR-selector/message narrow match: XML does enumerate
  RDNA3.5 VGPR selector values for 8-bit and 9-bit VGPR operands, including the
  high-bank source range, and carries a `msg_dealloc_vgprs` predefined send
  message. `RDNA3_5-XML-015` covers missing structured allocation and
  out-of-range semantics rather than absence of VGPR operand records.
- Chapter 3.3.3 memory-field narrow match: XML carries image `DMASK`,
  `TFE`/`LWE`, memory operand widths, and scattered memory-field descriptions
  for individual encodings. `RDNA3_5-XML-016` covers the missing general
  execution policy rather than absence of every memory field.
- Chapter 3.3.4 DS/LDS instruction-shell narrow match: XML has the `ENC_DS`
  field map, `GDS` field, `OPR_DSMEM`, representative DS load/store opcode
  shells, and LDSDIR parameter/direct load entries. `RDNA3_5-XML-017` is about
  missing LDS allocation, side-placement, alignment, and reporting semantics,
  not missing DS/LDSDIR decode metadata.
- Chapter 3.4.1-3.4.2 HWREG shell narrow match: XML has `OPR_HWREG` selector
  names and the `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32`
  instruction shells. `RDNA3_5-XML-018` covers missing STATUS/MODE field
  layout, access policy, initialization, and behavioral effects rather than
  absence of HWREG decode.
- Chapter 3.4.3 M0 operand narrow match: XML has an explicit `OPR_SDST_M0`
  operand type, M0-bearing scalar/memory operand subtypes, and selected
  instruction-local M0 descriptions. `RDNA3_5-XML-019` is about the missing
  complete per-consumer use table rather than absence of any M0 operand record.
- Chapter 3.4.4-3.4.6 NULL/SCC/VCC narrow match: XML enumerates `NULL`,
  `VCC_LO`, and `VCC_HI` in scalar operand tables, has a dedicated
  `OPR_SDST_NULL` discard operand at `amdgpu_isa_rdna3_5.xml:178587` through
  `:178595`, exposes `SRC_SCC` via `OPR_SSRC_SPECIAL_SCC` at
  `amdgpu_isa_rdna3_5.xml:188845` through `:188853`, carries `OPR_VCC` at
  `amdgpu_isa_rdna3_5.xml:188947` through `:188955`, states the VOPC
  `EXEC`-masked result rule at `amdgpu_isa_rdna3_5.xml:1149`, and has
  `S_CBRANCH_VCCZ`/`S_CBRANCH_EXECZ` instruction shells at
  `amdgpu_isa_rdna3_5.xml:60288` through `:60400`. The remaining VCCZ
  auto-summary behavior is part of the STATUS field semantics already covered
  by `RDNA3_5-XML-018`, rather than absence of NULL/SCC/VCC operand records.
- Chapter 3.4.7-3.4.8 FLAT_SCRATCH/HWREG selector narrow match: XML records
  the `OPR_FLAT_SCRATCH` implied operand, scratch instruction implicit
  operands, `OPR_HWREG` selector encoding, and the relevant HWREG selector
  names. `RDNA3_5-XML-020` is about the missing launch, field-layout, access,
  and side-effect behavior rather than absence of flat-scratch or HWREG
  selector shells.
- Chapter 3.4.9 trap instruction-shell narrow match: XML records
  `hw_reg_trapsts`, TMA/TBA return-message IDs, `S_TRAP`, `S_RFE_B64`, and
  `S_SENDMSG_RTN_B32/B64`. `RDNA3_5-XML-021` is about the missing trap state
  machine, TTMP payload/privilege rules, exception-enable policy, and
  `TRAPSTS` field semantics rather than absence of trap-related opcodes or
  selectors.
- Chapter 3.4.10 time selector narrow match: XML records
  `hw_reg_shader_cycles`, `msg_rtn_get_realtime`, `S_GETREG_B32`, and
  `S_SENDMSG_RTN_B64`. `RDNA3_5-XML-022` is about the missing clock-domain,
  width, synchronization, latency, and LGKM wait-counter semantics.
- Chapter 3.5.1-3.5.2 register-name narrow match: XML records `OPR_EXEC` and
  `OPR_FLAT_SCRATCH`, and scratch instruction entries reference the scratch
  operand where needed. `RDNA3_5-XML-023` is about the missing launch-time
  initialization contract, not absence of those architectural names.
- Chapter 3.5.3 scalar-operand narrow match: XML records ordinary scalar
  operand namespaces and SGPR selector values. `RDNA3_5-XML-024` is about the
  missing shader-stage launch payload tables and packing rules, not absence of
  scalar register operands.
- Chapter 2 instruction-role narrow match: XML functional groups such as SALU,
  SMEM, VALU, VMEM, EXPORT, BRANCH, MESSAGE, WAVE_CONTROL, and TRAP at
  `amdgpu_isa_rdna3_5.xml:189879` through `:189916` map to the manual's broad
  shader-processor roles. The Chapter 2 gaps above cover lifecycle,
  scheduling, and launch contracts rather than missing instruction families.
- Chapter 2 instruction-shell narrow match: XML has concrete
  `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, `S_SET_INST_PREFETCH_DISTANCE`,
  `S_INST_PREFETCH`, `S_CODE_END`, and `S_ENDPGM` entries; the gaps are about
  WGP-mode legality, padding length, launch/prefetch state, and global wave
  scheduling rules, not missing opcode shells.
- The RDNA3.5 XML has a distinct `OP_SEL_HI_2` field at
  `amdgpu_isa_rdna3_5.xml:2967`, matching the VOP3P split high-selector field
  in the manual at `rdna3.5/README.md:6566` and `:6611`.
- The XML carries explicit packed data formats and packed VOP3P entries for the
  audited ordinary packed-F16 forms, including `V_PK_FMA_F16` at
  `amdgpu_isa_rdna3_5.xml:120617`.
- `V_PK_FMAC_F16` is not missing from VOP3P: the XML lists it as an `ENC_VOP2`
  instruction at `amdgpu_isa_rdna3_5.xml:86076`, matching its
  accumulator-style packed-F16 contract.
- RDNA3.5 MIX descriptions in both manual and XML say fused multiply-add.
- The manual's DPP table says `V_PK_*` and WMMA do not support DPP, while
  `V_FMA_MIX_*` and `V_DOT2_F32_{BF16,F16}` do, at
  `rdna3.5/README.md:2771` through `:2788`. The inspected `V_PK_FMA_F16` XML
  entry exposes base and literal VOP3P encodings, not DPP encodings, while the
  MIX XML entries include DPP encodings.
- The XML does carry the RDNA3.5 `DS_BVH_STACK_RTN_B32` opcode 173, the
  expected explicit operands (`VDST`, in-out `ADDR`, `DATA0`, `DATA1`),
  implicit DSMEM operands, and `VMEM`/`DATA_SHARE` functional group metadata at
  `amdgpu_isa_rdna3_5.xml:21850` through `:21898`.
