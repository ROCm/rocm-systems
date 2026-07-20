# RDNA3 Manual vs XML Gaps

Architecture: RDNA3

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna3/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_rdna3.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1.1-1.2 Introduction, terminology, hardware overview, LDS/GDS/cache/device memory | Audited statically | Checked document scope, notation/suffix terminology, wave/workgroup/WGP/CU/SIMD/LDS/GDS/VMEM terms, host/command-processor/memory-controller overview, LDS/GDS/cache hierarchy, FP-exception interrupt capability, and relaxed memory/acknowledgment prose against XML architecture metadata, data-format records, functional groups, and instruction descriptions. Detailed operational gaps remain tracked in later semantic sections. |
| 2 intro / 2.1-2.4 Shader concepts, wave32/wave64, shader types, work-groups, CU/WGP modes, and shader padding | Audited statically | Checked shader execution roles, `EXEC` participation summary, wave32/wave64 issue model, compute/graphics shader launch concepts, workgroup sharing/barrier limits, CU/WGP LDS placement rules, WGP-mode LDSDIR exclusions, and shader-padding/prefetch-distance prose against XML functional groups, operand descriptions, instruction entries, and existing detailed semantic gaps. |
| 3.1-3.2 Wave state overview, PC, EXEC, and `EXEC==0` skipping | Audited statically | Checked the readable/writable wave-state inventory, per-wave/shared ownership notes, PC alignment and direct-control transfer rules, branch PC-relative formula, debug PC wording, `EXEC`/`VCC` wave32 low-half handling, and zero-`EXEC` skip/issue/counter exceptions against XML state operands, HWREG metadata, wait-counter operands, branch instruction entries, and functional groups. |
| 3.3.1 SGPR allocation, VCC, alignment, and out-of-range behavior | Audited statically | Checked scalar selector regions, VCC implicit producer/consumer entries, SGPR pair/quad alignment wording, scalar region-crossing restrictions, TTMP privilege behavior, SMEM destination range rules, and `S_MOVREL` range/index behavior against scalar operand classes and representative instruction entries. |
| 3.3.2 VGPR allocation, deallocation, and out-of-range behavior | Audited statically | Checked VGPR allocation granularity, zero-allocation prohibition, `S_SENDMSG` deallocation semantics, generic VGPR out-of-range predicates and consequences, `V_MOVREL*` indexed operands, `V_SWAPREL`, memory/export source fallback, multi-destination suppression, and the VOPD source modulo exception against VGPR operand classes, `MSG_DEALLOC_VGPRS`, and representative `V_MOVREL*`/`V_SWAPREL` entries. |
| 3.3.3 Memory alignment and memory-operation out-of-range behavior | Audited statically | Checked source/destination GPR and memory-address out-of-range behavior, return-VGPR nullification, PRT/TFE extra return VGPRs, image `DMASK` out-of-bounds determination, `SH_MEM_CONFIG.alignment_mode`, formatted-buffer alignment, and atomic alignment/MEMVIOL against DS/MUBUF/MTBUF/MIMG/FLAT field records, implicit memory operands, and `DMASK`/`TFE` field text. |
| 3.3.4 LDS allocation, placement, alignment, and out-of-range behavior | Audited statically | Checked LDS per-workgroup allocation size/granularity, CU/WGP side placement, pixel-parameter LDS placement, DS/LDS/GDS alignment mode, native alignment masks, `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, `MEMVIOL`, source/destination VGPR range effects, and read/write out-of-range results against DS/LDSDIR instruction entries, `ENC_DS`, DSMEM operands, and `HW_REG_LDS_ALLOC`. |
| 3.4.1-3.4.2 STATUS and MODE registers | Audited statically | Checked STATUS field layout, privilege/write permissions, initialization and dynamic status bits, plus MODE round/denorm/DX10/IEEE/trap/exception/FP16 overflow controls against HWREG selector names and S_GETREG/S_SETREG/S_ROUND_MODE/S_DENORM_MODE instruction shells. |
| 3.4.3-3.4.6 M0, NULL, SCC, VCC, and VCCZ | Audited statically | Checked the M0 use table, NULL read/write and SCC side-effect rules, SCC producer/consumer roles, VCC compare/carry destinations, live VCCZ summary behavior, wave32 mask truncation, and VCC SGPR-source accounting against scalar operand records, VOPC encoding prose, LDSDIR M0 descriptions, and SCC/VCC operand types. |
| 3.4.7-3.4.8 FLAT_SCRATCH and hardware internal registers | Audited statically | Checked FLAT_SCRATCH launch initialization, zero/no-scratch behavior, 256-byte alignment, trap-only write policy, HW_ID1/HW_ID2 field layouts, SH_MEM_BASES/PC/FLUSH_IB register behavior, and TBA/TMA access notes against XML `OPR_FLAT_SCRATCH`, HWREG selector names, S_GETREG/S_SETREG shells, and scratch instruction operands. |
| 3.4.9 Trap and Exception registers | Audited statically | Checked TTMP privilege, TBA/TMA read-only access, trap-entry TTMP payload, `STATUS.TRAP_EN`, exception-enable and unmaskable trap rules, and `TRAPSTS` sticky/status bits against XML S_TRAP/S_RFE/S_SENDMSG_RTN instruction entries, `OPR_HWREG`, and return-message metadata. |
| 3.4.10 Time | Audited statically | Checked `TIME`/`SHADER_CYCLES` and `REALTIME` clock domains, widths, synchronization, latency, and wait-counter usage against XML HWREG and return-message selectors plus S_GETREG/S_SENDMSG_RTN instruction shells. |
| 3.5.1-3.5.2 Initial `EXEC` and `FLAT_SCRATCH` state | Audited statically | Checked initial active-lane/null-wave `EXEC` semantics and scratch/no-scratch `FLAT_SCRATCH` launch initialization against XML `OPR_EXEC`, `OPR_FLAT_SCRATCH`, and scratch instruction operand records. |
| 3.5.3 SGPR Initialization | Audited statically | Checked PS/GS/HS/CS SGPR preload ordering, graphics-stage packed payloads, compute workgroup-id/TG_SIZE system SGPRs, and no-alignment packing rules against XML scalar operand records and launch-control field absence. |
| 3.5.4 VGPR Initialization | Audited statically | Checked HS/GS combined-stage initial VGPR tables, GS fast-launch variants, PS input CAM ordering, `SPI_PS_INPUT_ENA`/`SPI_PS_INPUT_ADDR` routing rules, and full/packed/skipped pixel VGPR examples against XML VGPR operand records and launch-control field absence. |
| 3.5.5 LDS Initialization | Audited statically | Checked PS-only LDS prelaunch vertex-parameter preload and barycentric interpolation source against XML LDSDIR instruction shells and LDS operand records. |
| 4 intro / 4.1-4.1.1 Shader Instruction Set, common fields, and cache controls | Audited statically | Checked instruction group inventory, unused-field defaulting, inline/literal source materialization, selector table, and SLC/GLC/DLC cache policy tables against XML encoding fields and operand classes. |
| 5.1-5.7 Program Flow Control | Audited statically | Checked program-control, scheduling, clause, message, branch, barrier, wait-counter, wait-event, and `S_DELAY_ALU` prose against XML SOP1/SOPK/SOPP instruction entries and operand classes. |
| 6.1-6.2 SALU formats and operands | Audited statically | Checked SALU format inventory, scalar operand selector table, source/destination and literal rules, out-of-range/alignment overlap, and `S_SETREG_IMM32_B32` literal-exception metadata. |
| 6.3-6.7 SALU SCC, arithmetic, conditional, compare, and bit-wise operations | Audited statically | Checked SCC write/read rules, signed-overflow versus carry-out wording, WREXEC destination restrictions, relative SGPR moves, bit-compare masks, and representative arithmetic/bitwise instruction entries. |
| 6.8-6.9 SALU access instructions and memory aperture query | Audited statically | Checked GETREG/SETREG metadata, `S_ROUND_MODE`/`S_DENORM_MODE` shells, and shared/private aperture source-selector formulas. |
| 7.1 VALU microcode encodings | Audited statically | Checked VOP1/VOP2/VOPC/VOP3/VOP3SD/VOP3P/VOPD inventory, VOP3 promotion exclusions, VOP3SD opcode inventory and `SDST` semantics, generic modifier fields, `OPSEL`, and literal-format notes. |
| 7.2.1-7.2.2 and 7.2.5-7.2.6 VALU operand/source rules | Audited statically | Checked non-standard operand-field uses, readlane lane-selector masking, DOT2 inline constants, input modifiers, literal expansion, scalar-source/literal restrictions, `OPSEL` allow-list, SGPR mask/carry operands, and wave64 SGPR caveats. |
| 7.2.3-7.2.4 and 7.2.7-7.2.8 VALU output/mode/edge rules | Audited statically | Checked output destinations, `V_CMPX` `EXEC` writes, carry-out destinations, VOP3 `OMOD`/`CLAMP` behavior and unsupported opcode lists, wave64 SGPR destination restrictions, round/denorm mode summary, VALU out-of-range GPR rules, and `PERMLANE` sequencing hazards. |
| 7.3 VALU instruction inventory | Audited statically | Checked the non-VOP3P VALU mnemonic inventory, VOP1/VOP2/VOP3/VOP3SD classification, XML alias spellings, and compact compare-family summary against concrete XML instruction entries and detailed opcode tables. |
| 7.4 16-bit Math and VGPRs | Audited statically | Checked true16 VGPR half naming/addressing, compact VOP1/VOP2/VOPC bit-7 half selection, VOP3/VOP3P/VINTERP `OPSEL` half selection, and 256-versus-512 true16 VGPR reach. |
| 7.5 Packed Math | Audited for VOP3P packed-math slice only | Checked packed opcode inventory, VOP3P field rules, inline constants, MIX selectors, DPP legality, and clamp behavior. |
| 7.6 Dual Issue VALU | Audited statically | Checked VOPD wave32-only and pair-legality prose, source-cache and literal restrictions, destination parity, ignored MOV source fields, implicit VCC use, paired-exception behavior, and the VOPD field/opcode tables against XML encoding and instruction records. |
| 7.7 DPP | Audited statically | Checked DPP16/DPP8 legality/exclusion table, row/bank/bounds/fetch-inactive semantics, compare full-mask behavior, VOP3/VOP3P modifier precedence, and DPP8 lane-selection/BC=1 rules against XML DPP encoding records and representative instruction entries. |
| 7.8 VGPR Indexing | Audited statically | Checked `M0`-indexed `V_MOVREL*`/`V_SWAPREL` formulas, unsigned and split-index uses of `M0`, and source/destination swap read-write behavior against XML VALU instruction entries and existing indexed-VGPR gaps. |
| 7.9 WMMA | Audited statically | Checked WMMA opcode inventory, VOP3P encoding, A/B/C operand classes, data-format records, C-only inline constants, matrix layout prose, wave32/wave64 replication, modifier overloads, rounding/exception behavior, and dependent-WMMA scheduling rules. |
| 8.1-8.4 Scalar Memory Operations | Audited statically | Checked SMEM field layout, raw scalar-load addressing, scalar buffer-resource addressing, buffer range behavior, scalar cache invalidation, dependency-counter rules, SMEM clause/group restrictions, and alignment/bounds prose against XML SMEM encodings, operands, and instruction entries. |
| 9.1-9.6 Vector Memory Buffer Instructions | Audited statically | Checked MUBUF/MTBUF field layout, VGPR address/data usage, format/resource/D16 data rules, buffer addressing/range/swizzle/resource descriptor behavior, alignment, TFE/cache/atomic-return summaries, and MUBUF/MTBUF opcode definitions against XML encoding records, operand classes, opaque descriptor formats, and representative instruction entries. |
| 10.1-10.9 Vector Memory Image Instructions | Audited statically | Checked MIMG field behavior, TFE/LWE, D16/A16/G16, NSA, no-sampler and sampler address tables, VGPR data/DMASK rules, image resource and sampler descriptors, data-format conversion, vector-memory dependency prose, and BVH ray-tracing behavior against XML MIMG field records, opaque descriptor formats, and representative image instruction entries. |
| 11 intro / 11.1-11.4 Global, Scratch and Flat Address Space | Audited statically | Checked FLAT/GLOBAL/SCRATCH field behavior, SEG-specific address forms, aperture checks, scratch swizzle, wait-counter and ordering rules, memory-error/MEMVIOL behavior, and data/D16 movement against XML FLAT field records and representative instruction entries. |
| 12.1-12.7 Data Share Operations | Audited statically | Checked LDS overview/access forms, LDSDIR parameter/direct-load behavior, indexed load/store/atomic address forms, ADDTID, append/consume, lane permute, DS stack/BVH, GDS and GS streamout behavior, and alignment/error prose against XML DS/LDSDIR field records and instruction entries. |
| 13.1-13.3 Float Memory Atomics | Audited statically | Checked float atomic surfaces, fixed add rounding, LDS and cache-atomic denormal controls, SNaN/QNaN propagation, min/max signed-zero and NaN ordering, compare-swap equality, and add special cases against XML DS, FLAT/GLOBAL, and MUBUF float-atomic opcode shells. |
| 13.4 Global Wave Sync and Atomic Ordered Count | Audited statically | Checked GWS/ordered-count clause and wait rules, `EXEC` handling, ordered-count fields/targets, append/consume behavior, GWS resource state and resource-ID formulas, and GWS instruction pseudocode against XML DS GWS and ordered-count opcode shells. |
| 14.1-14.3 Export: Position, Color/MRT | Audited statically | Checked export target and `EN` fields, `DONE`, `ROW_EN`/`M0`, 16-bit packing, pixel/depth/sample-mask constraints, dual-source blend masks, primitive position exports, `EXPCNT`, ordering, and `SKIP_EXPORT` against XML `ENC_EXP`, `EXP`, target enums, and wait-counter records. |
| 15 intro / microcode summary | Audited statically | Checked format-width summary, literal/DPP extension note, reserved-field defaulting, and suffix terminology against XML encoding records and existing literal/DPP policy gaps. |
| 15.1.1-15.1.5 SOP2/SOPK/SOP1/SOPC/SOPP | Audited statically | Checked scalar/control field positions, opcode-table coverage, scalar selector reuse, destination-field widths, SIMM16 placement, and literal-extension conditions against XML `ENC_SOP1`, `ENC_SOP2`, `ENC_SOPK`, `ENC_SOPC`, and `ENC_SOPP`. |
| 15.3.1-15.3.5 VOP2/VOP1/VOPC/VOP3/VOP3SD | Audited statically | Checked base VALU field positions, selector enumerations, compact destination/source fields, VOP3 modifiers, VOP3SD `SDST`, opcode-table coverage, and boundaries with existing `OPSEL`, literal, and VOP3SD metadata gaps. |
| 15.3.8-15.3.9 DPP16/DPP8 | Audited statically | Checked DPP16/DPP8 extension-DWORD fields, lane-selection fields, row/bank/bounds/fetch-invalid fields, and applicable VALU extension records against XML DPP encodings and existing DPP semantic gaps. |
| 15.4.1 VINTERP | Audited statically | Checked VINTERP field positions, `WAITEXP`, `OPSEL`, `CLMP`, source selectors, no-literal/no-DPP boundary, and opcode table against XML `ENC_VINTERP` and VINTERP instruction entries. |
| 15.10 EXP format | Audited statically | Checked EXP field positions and target inventory against XML `ENC_EXP`, `OPR_TGT`, and `EXP` instruction entries. |
| 15.2.1 SMEM | Audited statically | Checked SMEM field positions, opcode inventory, and scalar/buffer source operand classes against XML `ENC_SMEM` field records and SMEM instruction entries. |
| 15.5-15.6 LDSDIR and DS formats | Audited statically | Checked LDSDIR and DS field positions, opcode tables, `GDS`, offset, `ADDR`/`DATA`/`VDST`, and opcode inventory against XML `ENC_LDSDIR`, `ENC_DS`, and DS/LDSDIR instruction entries. |
| 15.7.1-15.7.2 MTBUF/MUBUF | Audited statically | Checked buffer field positions and opcode inventories against XML `ENC_MTBUF`/`ENC_MUBUF` records and opcode tables. |
| 15.8.1 MIMG | Audited statically | Checked MIMG normal and NSA field positions, opcode table shape, and address-register extension fields against XML `ENC_MIMG` and `MIMG_NSA1` records. |
| 15.9 Flat Formats | Audited statically | Checked FLAT, GLOBAL, and SCRATCH field positions, `OFFSET`/`SEG`/`SADDR`/`SVE` prose, and opcode tables against XML `ENC_FLAT`, `ENC_FLAT_GLOBAL`, and `ENC_FLAT_SCRATCH` records. |
| 15.3.6 VOP3P | Audited for field inventory only | Checked generic VOP3P field positions, `OPSEL_HI2`, and opcode table coverage for the packed-math slice. |
| 15.3.7 VOPD | Audited statically | Checked VOPD field positions, source selector enumerations, literal extension sizing, `VDSTY` parity reconstruction, and OPX/OPY opcode inventories. |
| Instruction definitions 0-26 | Audited statically | Checked VOP3P packed I16/U16/F16 and DOT rows for opcode inventory, operand formats, DOT signedness/modifier prose, BF16/F16 data formats, inline-literal interaction, and overlap with existing packed-math XML gaps. |
| Instruction definitions 32-34 | Audited for MIX slice only | Checked FMA_MIX fused semantics, selector prose, and `NEG_HI` modifier wording. |
| Instruction definitions 64-69 | Audited statically | Checked WMMA definition rows for opcode inventory, matrix-operation pseudocode, full-`EXEC` execution override, and overlap with Chapter 7.9 WMMA operand/modifier/layout rules. |
| 16.1 SOP2 Instructions | Audited statically | Checked scalar two-source arithmetic, SCC, shifts, BFE/BFM, min/max, multiply, conditional select, and pack definitions against XML SOP2 opcode shells and existing SALU semantic gaps. |
| 16.2 SOPK Instructions | Audited statically | Checked SIMM16 sign/zero extension, `S_VERSION`, conditional move/compare, add/multiply, HWREG, call, and split wait-count definitions against XML SOPK opcode shells; found a split wait-count threshold-expression gap. |
| 16.3 SOP1 Instructions | Audited statically | Checked scalar move/conditional move, bit-reverse/count/bitset/bitreplicate, EXEC save/write forms, relative SGPR moves, direct PC, trap return, and return-message definitions against XML SOP1 opcode shells and existing state/control gaps. |
| 16.4 SOPC Instructions | Audited statically | Checked signed/unsigned compare and bit-compare definitions against XML SOPC opcode shells and existing SCC/bit-index notes. |
| 16.5 SOPP Instructions | Audited statically | Checked NOP, wave-control, prefetch, clause, delay, wait, trap, branch, end-program, wakeup, priority, message, perf, cache-invalidate, and barrier definitions against XML SOPP opcode shells and existing control-flow gaps. |
| 16.7 VOP2 Instructions | Audited statically | Checked compact VOP2 arithmetic, carry, literal-only FMA, packed F16 MAC, VOP3/VOP3SD aliases, and min/max edge prose against XML VALU opcode shells and existing Chapter 7 semantic gaps. |
| 16.8 VOP1 Instructions | Audited statically | Checked VOP1 moves/conversions/transcendentals, relative-indexed moves, pipeflush, swap/permlane, true16 forms, and VOP3 aliases against XML VALU opcode shells and existing Chapter 7 semantic gaps. |
| 16.9 VOPC Instructions | Audited statically | Checked compact compare inventory, VCC/EXEC result semantics, compare/class variants, literals, DPP/SDWA interactions, and VOP3 aliases against XML VOPC/VOP3 opcode shells and operand metadata. |
| 16.12 VOP3 & VOP3SD Instructions | Audited statically | Checked VOP3/VOP3SD opcode inventory, VOP3SD opcode restriction, compare/CMPX result operands, DPP/literal alternatives, modifier/OPSEL metadata, min/max edge prose, and overlap with earlier VALU metadata gaps. |
| 16.13 VINTERP Instructions | Audited statically | Checked F32/F16 interpolation formulas, fixed DPP8 source selection, OPSEL notes, and RTZ variants against XML VINTERP opcode shells and operand-width metadata. |
| 16.6 SMEM Instructions | Audited statically | Checked scalar load, scalar buffer load, and scalar/cache invalidation instruction definitions against XML opcode shells and operand metadata. |
| 16.14-16.15 LDSDIR and LDS/GDS Instructions | Audited statically | Checked LDS parameter/direct load definitions, DS load/store/atomic definitions, ADDTID, append/consume, permute/swizzle, GDS/GWS, GS streamout, and DS instruction limitations against XML opcode shells, operand sizes, aliases, and functional groups. |
| 16.16-16.17 MUBUF/MTBUF Instructions | Audited statically | Checked typed/formatted/raw/D16/cache/atomic buffer definitions against XML instruction descriptions, operand sizes, functional groups, and opcode aliases. |
| 16.18 MIMG Instructions | Audited statically | Checked image load/store/atomic/query/MSAA/BVH/sample/gather instruction definitions against XML opcode shells, operand sizes, descriptor formats, and functional groups. |
| 16.19 EXPORT Instructions | Audited statically | Checked the singleton EXPORT definition, pixel `VM`/valid-mask and `DONE` obligations, color/depth/NULL target wording, vertex position/parameter obligations, and overlap with Chapter 14/15.10 export gaps. |
| 16.20 FLAT, Scratch and Global Instructions | Audited statically | Checked FLAT/SCRATCH/GLOBAL load, store, D16, ADDTID, and atomic instruction definitions against XML opcode shells, operand sizes, aliases, and functional groups. |
| 16.11 VOPD Instructions | Audited statically | Checked the VOPD X/Y instruction definitions for MOV/CNDMASK/DOT2ACC behavior and ordinary slot semantics referenced by Chapter 7.6. |
| Remaining RDNA3 manual sections | Not started | Full chapter-by-chapter audit remains outside the rows above. |

## Gaps

### RDNA3-XML-001: Packed VOP3P OPSEL legality restrictions are prose-only

Manual evidence:

- RDNA3 section 7.5 says packed math uses VOP3P and defines `OPSEL`/`OPSEL_HI`
  restrictions at `rdna3/README.md:2577` through `:2628`.
- `OPSEL` must be zero when the corresponding source or destination is 32-bit,
  except for MIX instructions; inline constants also require `OPSEL` zero
  because their value exists only in the low 16 bits.
- `OPSEL_HI` must be zero when the corresponding source or destination is
  32-bit or is a constant, except for MIX instructions.

XML evidence:

- The generic VOP3P field descriptions define `OP_SEL` and `OP_SEL_HI` as
  lower/upper 16-bit source selectors at
  `amdgpu_isa_rdna3.xml:3363` through `:3384`.
- Packed instruction entries such as `V_PK_FMA_F16` carry operand sizes and
  formats at `amdgpu_isa_rdna3.xml:118641` through `:118904`, but do not encode
  the per-source selector legality rules, the inline-constant restriction, or
  the MIX exception.

Impact:

An XML-only decoder or legality fuzzer can see the selector bit positions, but
cannot derive which selector values are legal for each packed source class.

### RDNA3-XML-002: Packed inline-constant behavior and DOT exceptions are prose-only

Manual evidence:

- Section 7.5.1 says inline constants used with packed math produce a value
  only in the low 16 bits, and float 16-bit sources receive an F16 constant
  value, at `rdna3/README.md:2648` through `:2656`.
- The DOT exception table at `rdna3/README.md:2660` through `:2672` gives
  per-opcode behavior for 8-bit, 4-bit, F16, and BF16 DOT forms, including
  which forms ignore `OPSEL` and which duplicate or derive upper halves.

XML evidence:

- The XML has inline-source operand classes and VOP3P literal extension
  encodings such as `VOP3P_INST_LITERAL` at `amdgpu_isa_rdna3.xml:14914`.
- Packed entries use formats such as `FMT_NUM_PK2_F16`, but the entries do not
  encode the low-16-only inline value rule, the F16-inline conversion rule, the
  BF16 upper-16 selection rule, or the DOT exception table.

Impact:

Generated semantics need manual-derived special cases for packed inline
constants. Operand size and data format alone do not distinguish all of these
behaviors.

### RDNA3-XML-003: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 7.5 says `V_FMA_MIX_*` uses VOP3P but is not packed math at
  `rdna3/README.md:2598` through `:2606`.
- The MIX selector note says `{OPSEL_HI[i], OPSEL[i]}` is a 2-bit source
  selector choosing between full FP32, low FP16, and high FP16 inputs at
  `rdna3/README.md:2632` through `:2640`.
- The MIX definitions say `NEG_HI` acts as an absolute-value modifier at
  `rdna3/README.md:15422` through `:15480`.

XML evidence:

- The generic VOP3P field description says `NEG_HI` negates the high operation
  at `amdgpu_isa_rdna3.xml:3344`, and says `OP_SEL`/`OP_SEL_HI` choose lower or
  upper 16-bit inputs at `:3363` through `:3384`.
- `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` are present at
  `amdgpu_isa_rdna3.xml:121083`, `:121409`, and `:121735`, but their entries do
  not encode the MIX-only selector mapping or `NEG_HI` absolute-value behavior.

Impact:

The generic XML field descriptions imply the wrong interpretation for MIX. XML
consumers need a hard-coded MIX override or manual-derived metadata.

### RDNA3-XML-004: `DS_BVH_STACK_RTN_B32` stack-state semantics are prose-only

Manual evidence:

- Section 12.5.3 defines `DS_BVH_STACK_RTN_B32` as an LDS-only per-thread BVH
  stack instruction and gives the field contract at `rdna3/README.md:4919`
  through `:4940`: `GDS` must be zero, `OFFSET0` is unused, `OFFSET1[5:4]`
  carries stack size, and `ADDR` is a packed in-out stack address.
- The instruction definition repeats that `OFFSET1[5:4]` selects stack sizes
  `{8, 16, 32, 64}`, describes `DATA0` as the last visited node pointer and
  `DATA1` as four candidate pointers, and gives the push/pop pseudocode at
  `rdna3/README.md:23530` through `:23579`.

XML evidence:

- The generic `ENC_DS` field descriptions only say `OFFSET0` and `OFFSET1`
  are DS offsets whose meanings are instruction-specific at
  `amdgpu_isa_rdna3.xml:3878` through `:3897`.
- The `DS_BVH_STACK_RTN_B32` XML entry has the opcode and explicit operand
  widths, plus implicit DSMEM input/output operands, at
  `amdgpu_isa_rdna3.xml:24285` through `:24333`; it does not structure
  `GDS == 0`, the `OFFSET1[5:4]` stack-size field, the packed `ADDR` layout,
  `DATA_VALID`, stack wrap/exhaustion, memory invalidation, or push/pop order.

Impact:

An XML-only consumer can decode the instruction and see that DSMEM is touched,
but cannot reconstruct the state machine or field invariants needed for
execution, legality checking, or focused test generation.

### RDNA3-XML-005: Chapter 1 architecture and host-control overview is not machine-readable

Manual evidence:

- Chapter 1 defines the manual's scope as the RDNA3 instruction set and
  shader-program accessible state at `rdna3/README.md:335`, then introduces
  RDNA3 as a parallel micro-architecture at `rdna3/README.md:337`.
- The terminology table defines dispatch, work-group, wave, WGP, CU, SIMD32,
  LDS, GDS, VMEM, sampler, texture resource, and buffer resource terms at
  `rdna3/README.md:366` through `:400`.
- Chapter 1.2 says the device includes a processor array, command processor,
  and memory controller; the command processor reads host-written
  memory-mapped registers and sends hardware-generated completion interrupts,
  while the memory controller has direct access to device and host-specified
  system memory, at `rdna3/README.md:459`.
- Chapter 1.2.1 through 1.2.3 describe independent WGP pipelines, automatic
  instruction fetch into on-chip caches, hardware FP-exception interrupts, LDS
  geometry and the 64KiB per-workgroup allocation limit, 4KiB GDS with
  append/consume support, cache hierarchy, acknowledgments, and relaxed
  consistency at `rdna3/README.md:475`, `:477`, `:479`, `:495`, `:499`, and
  `:507` through `:514`.

XML evidence:

- The top-level XML architecture metadata records only `AMD RDNA 3` and
  architecture ID `8` at `amdgpu_isa_rdna3.xml:11` through `:12` before
  entering instruction encoding records.
- Searching XML for command processor, memory controller, hardware-generated
  host interrupts, 128KiB LDS/WGP geometry, 4KiB GDS, append/consume GDS
  support, FP exceptions, and relaxed/serial consistency finds no
  architecture-topology or host-control schema.
- Related narrow records exist only as instruction or data-format shells: data
  formats such as BF16/F16/F32/F64 are represented at
  `amdgpu_isa_rdna3.xml:170171` through `:170285`, and top-level functional
  groups such as SALU, SMEM, VALU, VMEM, EXPORT, MESSAGE, WAVE_CONTROL, and
  TRAP are listed at `amdgpu_isa_rdna3.xml:186868` through `:186905`.

Impact:

XML consumers cannot derive the Chapter 1 device topology, host launch and
completion relationship, FP-exception interrupt capability, memory hierarchy,
LDS/GDS topology, or relaxed-consistency overview from XML alone. Validators,
emulators, and documentation generators need manual prose or separate
architecture configuration for that layer, even when instruction-level field
decode comes from XML.

### RDNA3-XML-006: Chapter 2 wave32/wave64 issue model is prose-only

Manual evidence:

- Chapter 2.1 says both wave32 and wave64 are supported for all operations, and
  that shader programs are compiled for one fixed wave size at
  `rdna3/README.md:541` through `:543`.
- Wave32 issues each instruction once, while wave64 typically issues VALU and
  vector-memory instructions, including LDS, texture, buffer, and flat, twice;
  scalar ALU, scalar memory, branches, messages, and exports issue once at
  `rdna3/README.md:545`.
- Chapter 2.1 records wave64 skip-half rules for zero `EXEC` halves, the VMEM
  outstanding-operation exception, and the rule that VALU instructions writing
  SGPRs are not half-skipped at `rdna3/README.md:547` through `:551`.
- The same section says both wave64 passes use the pre-instruction wave state,
  the second pass increments selected carry/divergence inputs and outputs, and
  wave32 ignores upper `EXEC`/`VCC` bits at `rdna3/README.md:553` through
  `:563`.

XML evidence:

- XML encodes instruction families and operands, and one mask operand
  description notes that 64-bit masks may be truncated to 32 bits in wave32 at
  `amdgpu_isa_rdna3.xml:170484`.
- Searches found no architecture record for fixed shader wave size, the
  functional-family issue-once/issue-twice split, wave64 half-skip rules, the
  pre-instruction-state rule for both wave64 passes, or the second-pass scalar
  input/output increments.
- Some instruction entries mention wave32/wave64 in local descriptions, such as
  a DPP wave64 half-swap at `amdgpu_isa_rdna3.xml:69852` and WMMA matrix
  replication text at `:171720`, but these do not structure the Chapter 2
  general issue model.

Impact:

XML-only consumers can identify instruction families and operands, but cannot
derive RDNA3's wave32/wave64 issue scheduling, skipped-half behavior, or
wave64 scalar-input/pass interaction from XML alone.

### RDNA3-XML-007: Shader-type launch modes are absent from XML

Manual evidence:

- Chapter 2.2.1 describes compute shaders as dispatch programs over a 1D, 2D,
  or 3D grid, with the processor walking the grid, creating waves, and
  initializing each work-item with a unique grid address/index at
  `rdna3/README.md:567` through `:569`.
- Chapter 2.2.2 defines pixel, geometry, and hull shader waves, says normal NGG
  geometry-engine launch initializes VGPRs with primitive/index and
  vertex-buffer data, and describes mesh-shader plus amplification-shader
  launch modes at `rdna3/README.md:571` through `:584`.

XML evidence:

- Searches found no shader-stage launch schema, compute grid-walk model,
  work-item initialization model, geometry-engine launch payload, mesh-shader
  mode, or amplification-shader mode in XML.
- XML contains only narrow stage-adjacent instruction names or message
  descriptions, such as `MSG_HS_TESSFACTOR` and `MSG_GS_ALLOC_REQ` in the
  message table at `amdgpu_isa_rdna3.xml:175213` and `:175238`, not a
  machine-readable launch taxonomy or per-stage VGPR initialization contract.

Impact:

XML cannot drive or validate compute-vs-graphics shader wave creation, normal
geometry-engine VGPR setup, mesh-shader launch conversion, or amplification
shader control without manual prose or another launch ABI source.

### RDNA3-XML-008: Workgroup and CU/WGP mode constraints are prose-only

Manual evidence:

- Chapter 2.3 says a workgroup's waves share LDS, synchronize at barriers, are
  issued to the same WGP, and can run on any of that WGP's four SIMD32 units at
  `rdna3/README.md:586` through `:588`.
- The same section states the WGP supports up to 32 workgroups and at most
  1024 work-items per workgroup; single-wave workgroups do not count against
  the 32-workgroup limit, do not allocate a barrier resource, and treat barrier
  operations as `S_NOP` at `rdna3/README.md:588` through `:590`.
- CU mode and WGP mode change LDS sharing and placement: CU mode splits LDS
  into upper/lower halves and keeps all workgroup waves resident within one CU,
  while WGP mode exposes one contiguous WGP LDS and may distribute workgroup
  waves across both CUs. `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` are not
  supported in WGP mode at `rdna3/README.md:592` through `:604`.

XML evidence:

- Searches found no XML record for WGP workgroup capacity, the 1024-work-item
  workgroup limit, single-wave barrier-resource elision, CU-mode LDS
  split/isolation, WGP-mode LDS sharing, mixed CU/WGP residency, or WGP-mode
  exclusion for LDSDIR instructions.
- XML does contain `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` instruction shells at
  `amdgpu_isa_rdna3.xml:28897` and `:28945`, but those entries do not encode
  the Chapter 2 WGP-mode legality condition.

Impact:

An XML-only validator can decode the LDSDIR instructions and identify
workgroup-related instruction families, but cannot infer the RDNA3 placement,
capacity, barrier-resource, or CU/WGP-mode legality contracts.

### RDNA3-XML-009: Shader padding and prefetch-distance requirements are prose-only

Manual evidence:

- Chapter 2.4 says shaders must be padded with 64 extra DWORDs, or 256 bytes,
  beyond the end of the shader and recommends `S_CODE_END` padding to avoid
  prefetching uninitialized or unmapped memory at `rdna3/README.md:608` through
  `:610`.
- The same section says the shader can prefetch 1, 2, or 3 64-byte cachelines
  ahead of the current PC, controlled by wave-launch state or
  `S_SET_INST_PREFETCH_DISTANCE`, at `rdna3/README.md:612`.

XML evidence:

- XML models `S_SET_INST_PREFETCH_DISTANCE` and its `S_INST_PREFETCH` alias as
  an instruction that changes instruction-prefetch mode at
  `amdgpu_isa_rdna3.xml:56436` through `:56445`.
- XML models `S_CODE_END` as an instruction that causes an illegal-instruction
  interrupt and marks the end of a shader buffer for debug tools at
  `amdgpu_isa_rdna3.xml:56752` through `:56753`, and models `S_ENDPGM` as a
  program terminator at `amdgpu_isa_rdna3.xml:57159` through `:57160`.
- Searches found no XML representation of the 64-DWORD/256-byte padding length,
  the code-object validation rule, the 1/2/3-cacheline legal prefetch-distance
  set, or the wave-launch state relation.

Impact:

XML consumers can decode the prefetch and padding-related instructions, but
cannot validate the Chapter 2 shader-buffer padding requirement or prefetch
distance contract from XML alone.

### RDNA3-XML-010: Chapter 3.1 wave-state inventory is only partially structured

Manual evidence:

- Chapter 3 opens by saying each wave has a private copy of visible shader
  state unless otherwise specified at `rdna3/README.md:616`.
- The state table lists PC, VGPRs, SGPRs, LDS, `EXEC`, `EXECZ`, `VCC`, `VCCZ`,
  `SCC`, `FLAT_SCRATCH`, `STATUS`, `MODE`, `M0`, `TRAPSTS`, `TBA`, `TMA`,
  `TTMP0`-`TTMP15`, and the VM/VScnt/EXP/LGKM wait counters with sizes and
  ownership notes at `rdna3/README.md:620` through `:651`.

XML evidence:

- XML has nominal operand records for several pieces of state: `OPR_PC` is a
  program-counter implied operand at `amdgpu_isa_rdna3.xml:174486` through
  `:174494`; `FMT_NUM_M64` describes 64-bit lane masks and wave32 truncation at
  `:170483` through `:170484`; `OPR_HWREG` lists hardware-register IDs such as
  `HW_REG_MODE`, `HW_REG_STATUS`, `HW_REG_TRAPSTS`, TBA, and flat scratch at
  `:174307` through `:174440`; `OPR_WAITCNT` structures exp/lgkm/vm fields at
  `:186752` through `:186785`.
- These records do not form a complete Chapter 3.1 state table: they do not
  encode the full visible-state inventory, per-wave versus shared ownership,
  shader read/write access policy, `TBA` bit-63/address replication wording,
  all counter names and sizes as wave state, or the relationship between raw
  `STATUS` bits and live `EXEC`/`VCC`.

Impact:

An XML-only state model can discover individual operand names and some encoded
fields, but cannot reconstruct the Chapter 3.1 architectural wave-state
contract without manual prose.

### RDNA3-XML-011: PC control-state rules are prose-only

Manual evidence:

- Chapter 3.2.1 says PC is a DWORD-aligned byte address, initialized to the
  first shader instruction, with the two low bits forced to zero at
  `rdna3/README.md:657`.
- The same section says `S_GETPC_B64`, `S_SETPC_B64`, `S_CALL_B64`,
  `S_RFE_B64`, and `S_SWAPPC_B64` transfer PC to and from an even-aligned SGPR
  pair sign-extended, that ordinary branches target the next instruction plus
  `offset * 4`, and that `S_TRAP` saves the PC of the trap instruction itself
  at `rdna3/README.md:659` through `:661`.
- The debug wording says a read PC points to the next instruction to issue and
  earlier instructions may not have completed at `rdna3/README.md:663`.

XML evidence:

- XML carries individual instruction entries for `S_GETPC_B64`,
  `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_RFE_B64` at
  `amdgpu_isa_rdna3.xml:45879` through `:46018`, and for `S_CALL_B64` at
  `:56108` through `:56136`.
- XML marks `S_TRAP` as a `TRAP` functional-group instruction but describes it
  only as entering the trap handler at `amdgpu_isa_rdna3.xml:56656` through
  `:56678`.
- Those entries do not encode the forced low-zero PC bits, sign-extension and
  even-SGPR-pair transfer details, debugger PC visibility, or the `S_TRAP`
  current-instruction-PC save rule.

Impact:

XML can identify the PC-manipulating instructions, but an XML-derived emulator,
validator, or fuzzer needs manual text to enforce PC alignment, PC save
semantics, and direct-PC operand invariants.

### RDNA3-XML-012: `EXEC==0` instruction skip policy is absent

Manual evidence:

- Chapter 3.2.2 defines `EXEC` as affecting vector ALU, vector memory, LDS,
  GDS, and export instructions, but not scalar execution or branches, and says
  wave32 only uses bits 31:0 with `EXECZ` reflecting that low half at
  `rdna3/README.md:667` through `:671`.
- Chapter 3.2.3 defines timing-visible `EXEC==0` skip behavior and exceptions:
  ordinary VALU skip except SGPR/VCC writers and WMMA/SWMMA, listed lane/control
  instructions and buffer invalidations never skip, selected SGPR/VCC-writing
  VALU issue twice in wave64, export skip depends on `Done`, `POS0`, and
  `SKIP_EXPORT`, and VMEM/LDS skipping depends on outstanding VM/VScnt or
  LGKM counters at `rdna3/README.md:673` through `:694`.

XML evidence:

- XML's `FMT_NUM_M64` notes mask truncation in wave32 at
  `amdgpu_isa_rdna3.xml:170483` through `:170484`, and branch entries for
  `S_CBRANCH_VCCZ` and `S_CBRANCH_EXECZ` reference the implicit mask operands
  at `:56883` through `:56975`.
- Some encoding descriptions say vector compares respect `EXEC`, for example
  `ENC_VOPC` at `amdgpu_isa_rdna3.xml:1091`.
- Searches found instruction names such as `V_NOP`, `V_READLANE_B32`, and
  `BUFFER_GL1_INV`, but no machine-readable per-family zero-`EXEC` skip table,
  no no-skip exception list, and no counter-dependent VMEM/LDS/export skip
  metadata.

Impact:

XML-only consumers can model mask operands and some per-instruction compare
effects, but cannot derive the Chapter 3 scheduler/issue/skip behavior or its
wait-counter and export exceptions.

### RDNA3-XML-013: SGPR region-crossing, out-of-range, and TTMP privilege behavior is incomplete

Manual evidence:

- Section 3.3.1.3 says 64-bit SGPR data must be even-aligned, greater-than-64-bit
  SGPR data and some scalar-memory data GPRs must be quad-aligned, and hardware
  enforces SGPR alignment by ignoring low address bits at `rdna3/README.md:714`
  through `:731`.
- Section 3.3.1.4 defines scalar selector regions, disallows multi-DWORD operands
  and GPR indexing that cross SGPR/VCC, TTMP, and other scalar-source regions,
  defines source-zero/write-ignore out-of-range behavior, and defines TTMP
  privilege plus failed-TTMP SCC preservation at `rdna3/README.md:735` through
  `:750`.
- The same section defines SMEM result range behavior and `S_MOVREL` source and
  destination range rules at `rdna3/README.md:755` through `:771`.

XML evidence:

- `OPR_SDST` and `OPR_SREG` enumerate the scalar selector names and include a
  generic alignment description at `amdgpu_isa_rdna3.xml:174497` through
  `:175150` and `:181440` through `:182077`.
- `S_MOVRELS_B32`, `S_MOVRELS_B64`, `S_MOVRELD_B32`, `S_MOVRELD_B64`, and
  `S_MOVRELSD_2_B32` expose explicit scalar operands and implicit `M0` at
  `amdgpu_isa_rdna3.xml:45616` through `:45860`.
- The XML records selector names, operand widths, and a broad alignment note, but
  does not encode the manual's low-bit masking behavior, cross-region legality,
  out-of-range read/write behavior, TTMP privilege gate, failed-TTMP SCC rule, or
  `S_MOVREL` base-range rules.

Impact:

XML-driven decoders and fuzzers can name many scalar operands, but need manual
semantics to decide which multi-word/indexed scalar references are illegal,
which accesses read zero or suppress writes, and when TTMP access is privileged.

### RDNA3-XML-014: VGPR allocation, deallocation, and out-of-range behavior is incomplete

Manual evidence:

- Section 3.3.2.1 says VGPRs are allocated in blocks of 16 for wave32 or 8 for
  wave64, that a shader may use up to 256 VGPRs, that a wave cannot be created
  with zero VGPRs, and that 1536-VGPR-per-SIMD devices use 24/12-register
  blocks at `rdna3/README.md:775` through `:779`.
- The same section says a wave may deallocate all VGPRs with `S_SENDMSG`, may
  not reallocate them afterward, and should only terminate afterward at
  `rdna3/README.md:779`.
- Section 3.3.2.2 defines the generic VGPR out-of-range predicate, special
  `V_MOVREL` indexed operand checks, destination-NOP behavior, `V_SWAPREL`
  discard behavior, multi-destination suppression, VMEM/export and VALU source
  fallback to `VGPR0`, and the VOPD modulo-4 source exception at
  `rdna3/README.md:783` through `:815`.
- Chapter 7.8 spells out the indexed VALU formulas, including unsigned
  `M0[31:0]` indexing for `V_MOVRELD_B32`, `V_MOVRELS_B32`, and
  `V_MOVRELSD_B32`, plus separate `M0[9:0]` source and `M0[25:16]`
  destination indexes for `V_MOVRELSD_2_B32` and `V_SWAPREL_B32`, at
  `rdna3/README.md:2839` through `:2856`.

XML evidence:

- `OPR_SRC_VGPR` and `OPR_VGPR` enumerate vector register operands and mention
  `NUM_VGPR` at `amdgpu_isa_rdna3.xml:178415` through `:178421` and
  `:185466` through `:185472`.
- The scalar message field includes `MSG_DEALLOC_VGPRS` and a free-form
  description of deallocation at `amdgpu_isa_rdna3.xml:175218` through
  `:175219`.
- `V_MOVRELD_B32`, `V_MOVRELS_B32`, `V_MOVRELSD_B32`,
  `V_MOVRELSD_2_B32`, and `V_SWAPREL_B32` expose their explicit VGPR operands
  and implicit `M0` operand at `amdgpu_isa_rdna3.xml:65484` through `:65508`,
  `:65695` through `:65720`, `:65858` through `:65883`, `:66021` through
  `:66040`, and `:69889` through `:69913`.
- These records do not structure the wave32/wave64 physical allocation block
  sizes, zero-VGPR wave creation rule, 1536-VGPR/SIMD exception, generic
  `Vs`/`Ve` range predicate, per-family out-of-range consequences, per-op `M0`
  bit slices for the relative VGPR instructions, multi-destination write
  suppression, memory/export per-dword `VGPR0` fallback, or VOPD modulo-4
  source rule. Swap operand direction is tracked separately in `RDNA3-XML-068`.

Impact:

XML-only consumers can decode VGPR operands and discover the deallocation
message, but they still need manual prose to model visible VGPR size, indexed
register legality, and the fallback/NOP rules that determine execution results.

### RDNA3-XML-015: Memory alignment and memory-operation out-of-range side effects are incomplete

Manual evidence:

- Section 3.3.3 defines the behavior for out-of-range source/destination GPRs
  and memory addresses for LDS, GDS, buffer, global, flat, and scratch memory
  accesses at `rdna3/README.md:818`.
- The same section says out-of-range source VGPRs or SGPRs make the data
  undefined, out-of-range destination VGPRs nullify the memory operation as if
  `EXEC` were zero, the destination test covers every VGPR that could be
  returned, and the test includes the extra PRT return VGPR whether the texture
  system actually returns it or not at `rdna3/README.md:820` through `:828`.
- Section 3.3.3 also says image loads and stores consider `DMASK` bits for the
  out-of-bounds determination, that `SH_MEM_CONFIG.alignment_mode` controls
  VMEM texture alignment and affects LDS/flat/scratch/global operations, that
  formatted buffer loads have 1/2/4-byte alignment requirements by format width,
  and that misaligned atomics trigger `MEMVIOL` at `rdna3/README.md:827`
  through `:842`.

XML evidence:

- XML exposes the DS address/data/GDS/VDST fields at
  `amdgpu_isa_rdna3.xml:2477` through `:2565`, the generic DSMEM and GPUMEM
  implicit operand shells at `:174264` through `:174303`, and representative
  formatted buffer load operands at `:36719` through `:36755`.
- XML has field text for `TFE`/partially resident textures at
  `amdgpu_isa_rdna3.xml:2765` through `:2766`, and image `DMASK` component
  selection text at `:3134` through `:3149`.
- These records do not structure the manual's source-GPR undefined-data rule,
  destination-VGPR `EXEC=0` nullification rule, all-possible-return-VGPR test,
  extra PRT return-VGPR test, atomic return nullification, `DMASK`-driven
  out-of-bounds determination, `SH_MEM_CONFIG.alignment_mode`, formatted-buffer
  alignment table, or atomic data-size alignment/MEMVIOL rule.

Impact:

An XML-only memory executor or fuzzer can identify memory operands and several
field names, but cannot derive the cross-cutting range/nullification and
alignment behavior that determines whether a memory operation executes, returns
data, or raises a memory violation.

### RDNA3-XML-016: LDS allocation, side placement, and LDS/GDS alignment/OOR policy are incomplete

Manual evidence:

- Section 3.3.4 says waves may allocate 0-64 KiB of LDS per wave or workgroup,
  the allocation is shared by all waves in a workgroup, the allocation is in
  1024-byte blocks, and all LDS accesses are restricted to the allocated space
  at `rdna3/README.md:846` through `:847`.
- The same section describes the internal two-64KiB-block LDS layout, CU-mode
  same-side placement and no-cross/no-wrap rule, WGP-mode placement or
  straddling across the 64KiB boundary, and pixel-shader LDS parameter placement
  at `rdna3/README.md:848` through `:855`.
- Section 3.3.4.1 defines DS/LDS/GDS alignment behavior, `LDS_CONFIG` address
  out-of-range reporting, STRICT/DWORD_STRICT `MEMVIOL`, read-zero and
  write-discard behavior, multi-dword read zeroing when any part is out of
  range, source VGPR `VGPR0` fallback, destination VGPR nullification, and
  native alignment masks at `rdna3/README.md:856` through `:890`.

XML evidence:

- XML exposes the generic DS field map, including `ADDR`, `DATA0`, `DATA1`,
  `GDS`, offsets, and `VDST`, at `amdgpu_isa_rdna3.xml:2477` through `:2565`.
- XML has representative DS load/store instruction shells such as
  `DS_STORE_B32` and `DS_LOAD_B128` at `amdgpu_isa_rdna3.xml:12926` through
  `:12963` and `:18556` through `:18586`, plus `OPR_DSMEM` and `HW_REG_LDS_ALLOC`
  names at `:174264` through `:174270` and `:174343` through `:174345`.
- XML also exposes LDSDIR instruction shells such as `LDS_PARAM_LOAD` and
  `LDS_DIRECT_LOAD` at `amdgpu_isa_rdna3.xml:28897` through `:28929` and
  `:28945` through `:28969`.
- These records do not encode per-workgroup LDS allocation granularity and
  bounds, CU/WGP side placement, pixel-parameter side constraints,
  `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT `MEMVIOL`,
  native LDS/GDS alignment masks, source/destination VGPR range consequences, or
  the multi-dword read rule.

Impact:

An XML-only DS/LDS consumer can decode LDS memory operands and the LDS/GDS
selector bit, but cannot derive the manual's allocation, placement, alignment,
or violation policy that determines whether an LDS operation reaches storage,
returns zero, is nullified, or reports a memory violation.

### RDNA3-XML-017: STATUS and MODE field semantics are prose-only

Manual evidence:

- Section 3.4 lists the wave state registers readable or writable through
  `S_GETREG` and `S_SETREG`, including `MODE`, `STATUS`, `TRAPSTS`,
  `FLUSH_IB`, `SH_MEM_BASES`, flat scratch, hardware ID registers, and
  `SHADER_CYCLES`, with read/write, read-only, write-only, or privileged access
  qualifiers at `rdna3/README.md:892` through `:910`.
- Section 3.4.1 defines STATUS fields, bit positions, and privileged-write
  permissions for `SCC`, wave priorities, `PRIV`, `TRAP_EN`, `EXPORT_RDY`,
  `EXECZ`, `VCCZ`, `IN_WG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`,
  `SKIP_EXPORT`, `PERF_EN`, conditional-debug bits, `FATAL_HALT`, `NO_VGPRS`,
  `LDS_PARAM_RDY`, `MUST_GS_ALLOC`, `MUST_EXPORT`, `IDLE`, and `SCRATCH_EN` at
  `rdna3/README.md:912` through `:948`.
- Section 3.4.2 defines MODE fields and their effects for `FP_ROUND`,
  `FP_DENORM`, `DX10_CLAMP`, `IEEE`, `LOD_CLAMPED`, `TRAP_AFTER_INST`,
  `EXCP_EN`, `FP16_OVFL`, and `DISABLE_PERF` at `rdna3/README.md:952` through
  `:973`.

XML evidence:

- XML exposes `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` as
  HWREG-access instructions at `amdgpu_isa_rdna3.xml:55994` through `:56090`.
- XML's `OPR_HWREG` predefined values name `HW_REG_MODE`, `HW_REG_STATUS`,
  `HW_REG_TRAPSTS`, `HW_REG_FLUSH_IB`, `HW_REG_SH_MEM_BASES`,
  `HW_REG_SHADER_FLAT_SCRATCH_LO/HI`, `HW_REG_HW_ID1/2`, and
  `HW_REG_SHADER_CYCLES` at `amdgpu_isa_rdna3.xml:174323` through `:174440`.
- Searches found no XML records for the STATUS field names or MODE field names
  above, their bit positions, read/write/privileged access policy, initialization
  rules, export/trap/liveness effects, floating-point mode effects, or exception
  enable behavior.

Impact:

An XML-only state-register model can find the HWREG selector names and the
get/set instruction forms, but cannot reconstruct the STATUS/MODE layout or the
field-level behavior needed to model export waits, trap state, priority, live
summary bits, floating-point modes, or exception controls.

### RDNA3-XML-018: The complete M0 use table is prose-only

Manual evidence:

- Section 3.4.3 says each wave has one 32-bit `M0` register and lists distinct
  layouts or meanings for `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, LDS ADDTID,
  Global Data Share, GDS ordered count, global wave sync, `S/V_MOVREL`,
  `S_SENDMSG`/`_RTN`, mesh-shader exports, SMEM address offset, and temporary
  use at `rdna3/README.md:975` through `:995`.
- The same table gives field-level constraints such as
  `LDS_PARAM_LOAD` byte-offset alignment, wave32 new-primitive-mask layout,
  `LDS_DIRECT_LOAD` data-type bits, LDS ADDTID 4-byte alignment, and GDS
  base/size packing.

XML evidence:

- XML exposes `M0` as an operand selector and scalar destination subtype at
  `amdgpu_isa_rdna3.xml:174497` through `:174503` and
  `amdgpu_isa_rdna3.xml:175138` through `:175176`.
- `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` include implicit `OPR_SDST_M0`
  operands and local prose for the direct-load `M0` address/data-type fields at
  `amdgpu_isa_rdna3.xml:28897` through `:28969`.
- Searches found no structured XML table for the rest of the M0 consumers or
  for the cross-instruction layouts and alignment constraints listed in
  Section 3.4.3.

Impact:

An XML-only consumer can decode uses of the `M0` operand and recover a few
instruction-local descriptions, but cannot derive the full per-consumer M0
interpretation needed for legality checks, operand-state modeling, or targeted
test generation.

### RDNA3-XML-019: FLAT_SCRATCH and hardware-internal register behavior is prose-only

Manual evidence:

- Section 3.4 lists the S_GETREG/S_SETREG hardware-register target IDs and
  access permissions, including `FLUSH_IB`, `SH_MEM_BASES`,
  `FLAT_SCRATCH_LO/HI`, `HW_ID1/2`, and `SHADER_CYCLES`, at
  `rdna3/README.md:892` through `:910`.
- Section 3.4.7 says `FLAT_SCRATCH` is a per-wave scratch base initialized by
  wave-launch hardware when scratch is allocated, read-only outside trap
  handling, zero when no scratch is allocated, and a 256-byte-aligned byte
  address computed from `scratch_base + spi_scratch_offset` at
  `rdna3/README.md:1042` through `:1052`.
- Section 3.4.8 defines read-only hardware internal registers, the
  unpredictable/context-switch caveat for `HW_ID` and `*_BASE` values, the
  `HW_ID1` and `HW_ID2` bit layouts, the `FLUSH_IB` write side effect,
  `SH_MEM_BASES` private/shared aperture bits, the warning against reading PC
  via `S_GETREG`, and the TMA/TBA access path at `rdna3/README.md:1054`
  through `:1089`.

XML evidence:

- XML has an `OPR_FLAT_SCRATCH` operand whose predefined value is only
  `flat_scratch[63:0]` at `amdgpu_isa_rdna3.xml:174286` through `:174294`.
- XML's `OPR_HWREG` operand records the HWREG selector field and names
  `HW_REG_FLUSH_IB`, `HW_REG_SH_MEM_BASES`,
  `HW_REG_SHADER_FLAT_SCRATCH_LO/HI`, and `HW_REG_HW_ID1/2`, but their
  descriptions are `N/A` at `amdgpu_isa_rdna3.xml:174307` through `:174420`.
- XML exposes the `S_GETREG_B32` and `S_SETREG_B32` instruction shells at
  `amdgpu_isa_rdna3.xml:55994` and `:56032`, and many scratch instruction
  entries carry implicit `OPR_FLAT_SCRATCH` operands, but searches found no
  structured FLAT_SCRATCH launch formula, 256-byte alignment rule, no-scratch
  zero behavior, trap-only write policy, HW_ID bit layouts, SH_MEM_BASES
  private/shared bit layout, FLUSH_IB side effect, PC read warning, or TMA/TBA
  access note.

Impact:

An XML-only consumer can identify the flat-scratch operand and the HWREG
selector names, but cannot reconstruct the launch, access-control, alignment,
field-layout, side-effect, or unpredictability rules needed for emulation and
legality testing.

### RDNA3-XML-020: Trap-status and exception-control behavior is prose-only

Manual evidence:

- Section 3.4.9 says TTMP writes are privileged, user-mode TTMP reads return
  zero, user-mode TTMP writes are ignored, and TMA/TBA are read-only through
  `S_SENDMSG_RTN` at `rdna3/README.md:1091` through `:1102`.
- The same section says trap entry loads `{TTMP1, TTMP0}` with host-trap,
  trap-ID, and PC payload fields, and defines host/user/exception trap-ID
  behavior at `rdna3/README.md:1103` through `:1109`.
- `STATUS.TRAP_EN` controls whether traps are taken at all and converts
  `S_TRAP` to a hardware NOP when no handler is present at
  `rdna3/README.md:1111` through `:1115`.
- `MODE.EXCP_EN` defines independently enabled exception sources, with MEMVIOL
  and illegal instruction unmaskable, and the table lists exception causes and
  architectural results at `rdna3/README.md:1117` through `:1133`.
- The `TRAPSTS` table defines writable sticky `EXCP` bits and the
  `SAVECTX`, `ILLEGAL_INST`, `ADDR_WATCH1-3`, `BUFFER_OOB`, `HOST_TRAP`,
  `WAVE_START`, `WAVE_END`, and `TRAP_AFTER_INST` status bits at
  `rdna3/README.md:1135` through `:1162`.

XML evidence:

- XML names `HW_REG_TRAPSTS` in the HWREG selector table, but the description is
  `N/A` at `amdgpu_isa_rdna3.xml:174333` through `:174335`.
- XML describes `S_TRAP` only as entering the trap handler and records the
  immediate operand and `TRAP` functional group at
  `amdgpu_isa_rdna3.xml:56656` through `:56678`.
- XML describes `S_RFE_B64` as clearing `PRIV` and jumping to the scalar input
  at `amdgpu_isa_rdna3.xml:46001` through `:46018`, and names TMA/TBA return
  messages at `amdgpu_isa_rdna3.xml:175258` through `:175278`.
- These records do not structure TTMP privilege side effects, trap-entry TTMP
  payload packing, `STATUS.TRAP_EN` gating and `S_TRAP`-to-NOP behavior,
  exception-enable bits, unmaskable exception policy, exception result table, or
  the `TRAPSTS` sticky/status bit layout.

Impact:

XML-only trap handling can identify the relevant instruction and selector
names, but cannot derive when a trap is taken, what state is saved, which
exceptions accumulate or jump to the handler, or how `TRAPSTS` should be read
and cleared.

### RDNA3-XML-021: Time-counter clock-domain and wait semantics are prose-only

Manual evidence:

- Section 3.4.10 distinguishes `TIME` from `REALTIME`: `TIME` measures graphics
  core clock cycles as a 20-bit counter, while `REALTIME` is a 64-bit
  fixed-frequency clock that usually runs at 100 MHz and continues regardless
  of shader or memory clock speed at `rdna3/README.md:1163` through `:1168`
  and `:1172`.
- The manual says `SHADER_CYCLES` is read through `S_GETREG`, is not
  synchronized across different SIMDs, should be used only for within-wave
  deltas, and has typical SALU latency around 8 cycles at
  `rdna3/README.md:1170`.
- The manual's `REALTIME` sequence uses `S_SENDMSG_RTN_B64` followed by
  `S_WAITCNT LGKMcnt == 0` at `rdna3/README.md:1176` through `:1179`.

XML evidence:

- XML names `HW_REG_SHADER_CYCLES` but gives it an `N/A` description at
  `amdgpu_isa_rdna3.xml:174437` through `:174440`.
- XML names `MSG_RTN_GET_REALTIME` and says it returns the constant-frequency
  REFCLK time counter for 32-bit or 64-bit requests at
  `amdgpu_isa_rdna3.xml:175263` through `:175265`.
- XML exposes the `S_GETREG_B32` and `S_SENDMSG_RTN_B64` instruction shells at
  `amdgpu_isa_rdna3.xml:55994` through `:56018` and
  `amdgpu_isa_rdna3.xml:46076` through `:46104`, but does not encode the
  20-bit width, per-SIMD synchronization caveat, SALU latency, fixed-frequency
  clock rate expectation, idle-clock independence, or required LGKM wait
  dependency.

Impact:

XML-only consumers can find the selector and message names, but cannot recover
the time-counter domains, widths, synchronization limits, or dependency behavior
needed for timing-sensitive emulation and tests.

### RDNA3-XML-022: Initial `EXEC` and `FLAT_SCRATCH` launch state is prose-only

Manual evidence:

- Section 3.5 says wave state is initialized from state data and dynamic
  wave-launch state before execution, and that some state is common across a
  draw while some is unique per wave, at `rdna3/README.md:1181` through
  `:1189`.
- Section 3.5.1 says `EXEC` normally starts as the active-thread mask, but some
  launch cases initialize `EXEC == 0` for "Null waves" that should do no work
  and exit immediately, at `rdna3/README.md:1191` through `:1193`.
- Section 3.5.2 says waves with scratch memory are initialized with
  `FLAT_SCRATCH` pointing at global memory, while waves without scratch get
  zero, at `rdna3/README.md:1195` through `:1197`.

XML evidence:

- XML records `OPR_EXEC` as the 64-bit vector execute mask and names the
  `exec` selector at `amdgpu_isa_rdna3.xml:174274` through `:174283`.
- XML records `OPR_FLAT_SCRATCH` as an implied 64-bit flat-scratch operand and
  names the `flat_scratch` selector at `amdgpu_isa_rdna3.xml:174286` through
  `:174294`.
- Scratch instructions reference `OPR_FLAT_SCRATCH` as an implicit operand, but
  XML does not encode wave-launch state, active-lane mask derivation,
  null-wave exit behavior, or scratch/no-scratch initialization policy.

Impact:

XML-only consumers can identify the architectural registers and implicit
scratch operand, but cannot derive the initial `EXEC`/`FLAT_SCRATCH` values a
wave should see at launch. That launch-state model has to come from manual
prose or another non-XML source.

### RDNA3-XML-023: SGPR launch payload tables are prose-only

Manual evidence:

- Section 3.5.3 says SGPRs are initialized from SPI/COMPUTE program-resource
  register settings, only enabled values are loaded, enabled values are packed
  consecutively, and no SGPRs are skipped for alignment at
  `rdna3/README.md:1199` through `:1203`.
- The PS table defines user-data, `{bc_optimize, prim_mask, lds_offset}`,
  `{ps_wave_id, ps_wave_index}`, and provoking-vertex SGPR payloads at
  `rdna3/README.md:1205` through `:1224`.
- The GS and HS sections define combined-stage launch behavior, first-eight
  automatically initialized SGPRs, program-address payloads, off-chip LDS and
  stage-specific packed fields, FAST_LAUNCH variants, and user-SGPR ranges at
  `rdna3/README.md:1226` through `:1278`.
- The CS table defines up-to-16 user-data SGPRs, optional
  `work_group_id0/1/2` SGPRs, and a packed `tg_size_en` payload containing
  `first_wave`, `wave_id_in_group`, `ordered_append_term`, and
  `work_group_size_in_waves` at `rdna3/README.md:1280` through `:1290`.

XML evidence:

- XML has scalar operand classes such as `OPR_SDST`, `OPR_SREG`, and
  `OPR_SSRC` that enumerate SGPR selector values and alignment notes at
  `amdgpu_isa_rdna3.xml:174497` through `:175044`,
  `amdgpu_isa_rdna3.xml:181440` through `:181984`, and
  `amdgpu_isa_rdna3.xml:183177` through `:183726`.
- Searches for `COMPUTE_PGM_RSRC2`, `SPI_SHADER_PGM_RSRC`,
  `SPI_SHADER_USER_DATA`, `tg_size`, `wave_id_in_group`,
  `ordered_append`, `work_group_size_in_waves`, `PS_wave_id`,
  `FAST_LAUNCH`, and `LOAD_PROVOKING_VTX` found no RDNA3 XML records.
- The scalar operand records do not encode shader-stage SGPR initialization
  order, enable-field names, no-alignment packing, graphics combined-stage
  payloads, compute workgroup-id payloads, or the packed `TG_SIZE` system SGPR.

Impact:

XML-only consumers can decode instructions that reference SGPRs, but cannot
derive which SGPRs are preloaded for PS/GS/HS/CS launch or how enabled launch
payloads are packed. Emulators and launch-state tests need the manual or a
separate launch ABI source.

### RDNA3-XML-024: VGPR launch payload and PS input routing tables are prose-only

Manual evidence:

- Section 3.5.4 says pre-launch VGPR initialization is selected by
  `COMPUTE_PGM_RSRC*` or `SPI_SHADER_PGM_RSRC*` control registers at
  `rdna3/README.md:1294` through `:1297`.
- The initial VGPR table maps HS(+LS) and GS(+ES) combined-stage payloads into
  `VGPR0` through `VGPR8`, including HS patch/control-point IDs, LS vertex
  indices and optional user VGPR, GS offsets, primitive/payload data, RT
  index/edge flags/instance ID, ES patch/parameter/vertex data, and GS
  Fast Launch 1/2 variants at `rdna3/README.md:1298` through `:1306`.
- Section 3.5.4.1 says pixel shader VGPR input loading is routed by a CAM from
  VS outputs to PS inputs, and lists the pixel input terms in load order at
  `rdna3/README.md:1308` through `:1321`.
- The manual defines `SPI_PS_INPUT_ENA` and `SPI_PS_INPUT_ADDR`: `ENA`
  controls which terms are calculated or loaded, `ADDR` controls destination
  placement, `ADDR` bits may be set without `ENA`, and `ENA` requires the
  corresponding `ADDR` bit at `rdna3/README.md:1323` through `:1325`.
- The PSR field table maps full-load pixel terms into `VGPR0` through
  `VGPR23`, including perspective/linear I/J terms, line stipple, position
  floats, front face, ancillary packed bits, sample coverage, and fixed
  position at `rdna3/README.md:1329` through `:1354`, and examples show packed
  and skipped destinations at `:1356` through `:1417`.

XML evidence:

- XML records ordinary VGPR instruction operands such as `OPR_VGPR`, which
  names `v0` and says values increment through `NUM_VGPR`, at
  `amdgpu_isa_rdna3.xml:185466` through `:185485`.
- Searches for `SPI_PS_INPUT`, `PS_INPUT`, `PERSP_SAMPLE`, `PERSP_CENTER`,
  `PERSP_CENTROID`, `LINEAR_SAMPLE`, `LINEAR_CENTER`, `LINEAR_CENTROID`,
  `POS_X_FLOAT`, `FRONT_FACE`, `ANCILLARY`, `SAMPLE_COVERAGE`, `POS_FIXED`,
  `VGPR_COMP_CNT`, `FAST_LAUNCH`, and `TIDIG_COMP_CNT` found no RDNA3 XML
  launch-state records.
- The VGPR operand records do not encode initial VGPR payloads by shader stage,
  GS fast-launch variants, pixel input CAM ordering, PSR bit layouts, PS
  `ENA`/`ADDR` legality, or destination packing/skipping behavior.

Impact:

XML-only consumers can identify VGPR instruction operands, but cannot derive
which VGPRs hardware preloads for graphics stages or how pixel shader inputs
are packed and routed. That prevents XML-only validation of graphics launch
state and PS barycentric/position input setup.

### RDNA3-XML-025: Pixel-shader LDS launch preload is prose-only

Manual evidence:

- Section 3.5.5 says only pixel shader waves have LDS pre-initialized before
  launch, and that PS LDS is preloaded with vertex parameter data that can be
  interpolated using barycentrics I/J to compute per-pixel parameters, at
  `rdna3/README.md:1419` through `:1421`.

XML evidence:

- XML has `LDS_PARAM_LOAD` in `ENC_LDSDIR`, whose description transfers
  parameter data from LDS to VGPRs and expands it using `NewPrimMask` in `M0`,
  at `amdgpu_isa_rdna3.xml:28897` through `:28928`.
- XML has `LDS_DIRECT_LOAD` in the same encoding family at
  `amdgpu_isa_rdna3.xml:28945` through `:28969`.
- These entries describe instructions that consume LDS contents; they do not
  define which pixel-shader launches preload LDS, how vertex parameter data is
  laid out in LDS, or how launch state supplies the barycentric source data.

Impact:

XML-only consumers can discover LDSDIR instructions, but cannot construct or
validate the initial LDS contents required for pixel-shader parameter
interpolation.

### RDNA3-XML-026: Inline-constant materialization rules are only partially structured

Manual evidence:

- Chapter 4.1 defines inline constants as selector values 128 through 248 and
  says float constants work with single, double, and 16-bit float operations;
  when used by non-float instructions, the data remains the float bit pattern,
  at `rdna3/README.md:1444` through `:1448`.
- The same section says float inline constants are encoded according to the
  source operand size, with 16-bit operations using the 16-bit float value in
  the low bits and zeros in the high bits, and integer constants sign-extend
  for 64-bit sources at `rdna3/README.md:1448` through `:1450`.

XML evidence:

- XML enumerates integer and float inline source selector values in operand
  classes such as `OPR_SRC` and `OPR_SSRC`; representative float entries are at
  `amdgpu_isa_rdna3.xml:177085` through `:177096` and
  `amdgpu_isa_rdna3.xml:184239` through `:184282`.
- The special `1 / (2 * PI)` selector carries per-format bit patterns, but the
  ordinary float entries only name the numeric value, and searches for
  `source operand size`, `zero-extended 32-bit`, `remains a float`, and
  `sign-extended for 64-bit` found only the special `1 / (2 * PI)` wording and
  literal-source wording.

Impact:

An XML-only consumer can recover the selector numbers, but still needs manual
rules to materialize most inline constants for 16-bit, 32-bit, 64-bit, and
non-float consumers.

### RDNA3-XML-027: SLC/GLC/DLC cache policy tables are incomplete and partly inconsistent

Manual evidence:

- Chapter 4.1.1 says SLC, GLC, and DLC control cache behavior for scalar and
  vector memory instructions, with GLC controlling the graphics first-level
  cache, SLC the graphics L2 cache, and DLC the MALL at
  `rdna3/README.md:1495` through `:1505`.
- The load and store/atomic tables define the resulting MALL, GL2, GL1, texture
  L0, scope, and non-temporal policies for SRD `llc_noalloc`, ISA DLC, SLC,
  and GLC combinations at `rdna3/README.md:1507` through `:1544`.
- The prose after the tables says GLC is a load-scope bit (`0=CU`,
  `1=DEVICE`), stores/atomics are device-scope, SLC is the graphics-client
  temporal hint, DLC is the Infinity Cache temporal hint, and atomic GLC
  selects whether the pre-operation value is returned at `rdna3/README.md:1546`
  through `:1565`.

XML evidence:

- XML records field positions and short descriptions for SMEM `DLC` and `GLC`
  at `amdgpu_isa_rdna3.xml:615` through `:636`, and for MUBUF `DLC`, `GLC`,
  and `SLC` at `amdgpu_isa_rdna3.xml:2661` through `:2682` and
  `:2731` through `:2732`.
- Those field descriptions do not encode the Chapter 4 policy tables, the SRD
  `llc_noalloc` interaction, the MALL/GL2/GL1/Tex(L0) outcomes, the load-scope
  rule, or the atomic return rule. They also use coherence-oriented descriptions
  such as `DLC` being L1-coherent and `SLC` being System Level Coherent, which
  do not match the Chapter 4 SLC/DLC temporal-hint prose.

Impact:

XML-only memory modeling cannot derive RDNA3 cache scope, temporal-hint, or
atomic-return behavior from field positions alone, and the short XML field
descriptions can steer an implementation toward the wrong GFX11 interpretation.

### RDNA3-XML-028: Common unused-field canonicalization is missing

Manual evidence:

- The Chapter 4 introduction says not every instruction uses every field in its
  encoding; unused SGPR-capable source/destination fields are typically set to
  `NULL`, and other unused fields are typically set to zero, at
  `rdna3/README.md:1440`.

XML evidence:

- XML encodings define the physical fields for each format, for example SMEM at
  `amdgpu_isa_rdna3.xml:586` through `:650` and MUBUF at
  `amdgpu_isa_rdna3.xml:2657` through `:2738`.
- Searches for `unused`, `set to zero`, `zero when unused`, and unused-field
  `NULL` canonicalization did not find a matching general rule in RDNA3 XML.

Impact:

Encoders, disassemblers, and legality fuzzers using only XML cannot distinguish
canonical unused-field values from merely decodable field values.

### RDNA3-XML-029: Chapter 5 control-op side effects are only partially structured

Manual evidence:

- Section 5.1 defines program-control behavior for `S_TRAP`, `S_RFE_B64`,
  `S_SETKILL`, `S_SETHALT`, `S_NOP`, `S_SLEEP`, `S_WAKEUP`, `S_SETPRIO`,
  `S_SENDMSG`, `S_SENDMSG_RTN_B32/B64`, `S_SENDMSGHALT`, `S_CODE_END`,
  `S_ICACHE_INV`, and `S_BARRIER` at `rdna3/README.md:1571` through `:1617`.
- The table includes details such as trap-entry TTMP payload, `PC = TBA`,
  `PRIV = 1`, trap-handler-only `S_RFE_B64`, `S_SETHALT` privilege and fatal
  halt bits, `S_NOP` repeat count, `S_SLEEP` duration and wakeup interaction,
  the priority formula, illegal `S_CODE_END`, and WGP-scoped instruction-cache
  invalidation.
- Chapter 16.5 says `S_ENDPGM`, `S_ENDPGM_SAVED`, and
  `S_ENDPGM_ORDERED_PS_DONE` implicitly execute `S_WAITCNT 0` and
  `S_WAITCNT_VSCNT 0`, and says the ordered-PS-done form combines
  `S_SENDMSG(MSG_ORDERED_PS_DONE)` with program termination, at
  `rdna3/README.md:9909` through `:9928`.

XML evidence:

- XML records the opcode shells and short descriptions, including `S_TRAP` at
  `amdgpu_isa_rdna3.xml:56656` through `:56678`, `S_SETHALT` and `S_SLEEP` at
  `:56372` through `:56418`, `S_SETPRIO` at `:57259` through `:57273`,
  `S_SENDMSG`/`S_SENDMSGHALT` at `:57291` through `:57347`, `S_CODE_END` at
  `:56752` through `:56759`, `S_ENDPGM*` at `:57159` through `:57210`, and
  `S_ICACHE_INV` at `:57492` through `:57499`.
- Those entries use generic immediate or message operands and do not encode the
  trap payload layout, handler-only legality, privilege/STATUS effects,
  sleep-duration formula, wakeup target rules, priority formula, illegal
  instruction behavior, implicit end-program waits, ordered-PS-done message
  behavior, or WGP-scoped I-cache effect.

Impact:

XML consumers can decode the control opcodes, but must import manual prose to
model side effects, privilege legality, scheduling, cache/debug behavior,
termination waits, and ordered-PS-done messaging.

### RDNA3-XML-030: Instruction-clause membership, start, and break rules are prose-only

Manual evidence:

- Section 5.1 says `S_CLAUSE` length is `SIMM16[5:0] + 1`, valid programmed
  values are `1-62`, and `SIMM16[11:8]` defines regular breaks at
  `rdna3/README.md:1603` through `:1606`.
- Section 5.2 says a clause is an uninterrupted same-type instruction sequence
  whose type is defined by the instruction immediately after `S_CLAUSE`, then
  lists valid clause types, clause-internal instructions, illegal instruction
  classes, sub-clause rules, `EXEC==0` start/mid-clause behavior, `S_DELAY_ALU`
  ordering, skipped-first-instruction behavior, and clause-break causes at
  `rdna3/README.md:1619` through `:1687`.

XML evidence:

- `S_CLAUSE` is an SOPP opcode with an `OPR_CLAUSE` operand at
  `amdgpu_isa_rdna3.xml:56471` through `:56483`.
- `OPR_CLAUSE` structures `BREAK_SPAN` and `LENGTH`, including the `[1, 62]`
  programmed-length range, at `amdgpu_isa_rdna3.xml:174042` through `:174066`.
- Searches for clause type names and rule phrases such as `clause type`,
  `illegal in a clause`, `first instruction`, `EXEC==0`, and `break a clause`
  found no machine-readable clause membership or stream-state rule in RDNA3 XML.

Impact:

The XML is enough to decode the immediate fields, but not enough to validate or
model whether a clause starts, what instruction type it binds to, or when it
must end.

### RDNA3-XML-031: Send-message legality and `LGKMcnt` return protocol are partial

Manual evidence:

- Section 5.1 says `S_SENDMSG_RTN_B32/B64` use `LGKMcnt`, return data to an
  SGPR or aligned SGPR pair, treat `SSRC0` as an enum rather than an SGPR, and
  leave `VCCZ` undefined if the destination is VCC at `rdna3/README.md:1614`
  through `:1615`.
- Section 5.3 says `S_SENDMSG` uses `SIMM16`, `S_SENDMSG_RTN` uses `SSRC0`,
  payloads come from `M0`, completion is tracked with `LGKMcnt`, return
  messages increment `LGKMcnt` by 2 and decrement once on send plus once on
  data return, and unlisted message codes are reserved/illegal at
  `rdna3/README.md:1688` through `:1727`.

XML evidence:

- `S_SENDMSG_RTN_B32/B64` use `OPR_SENDMSG` at
  `amdgpu_isa_rdna3.xml:46038` through `:46104`, while `S_SENDMSG` and
  `S_SENDMSGHALT` use `OPR_SENDMSG` plus implicit `M0` at
  `amdgpu_isa_rdna3.xml:57291` through `:57347`.
- `OPR_SENDMSG` lists the named non-return and return message values and some
  payload descriptions at `amdgpu_isa_rdna3.xml:175193` through `:175287`.
- The XML does not connect return messages to `LGKMcnt`, encode the two-step
  decrement protocol, represent the `VCCZ` undefined side effect, or provide a
  structured legality rule for unlisted numeric message codes.

Impact:

The XML can recover many message names and payload hints, but not the
wait-counter protocol or full legality/side-effect contract required for
execution modeling.

### RDNA3-XML-032: Wait-counter producers, events, and `S_DELAY_ALU` stream semantics are partial

Manual evidence:

- Section 5.6 defines `S_WAITCNT`, the per-counter `S_WAITCNT_*` forms,
  `S_WAIT_EVENT`, and `S_DELAY_ALU`, then assigns operation groups to
  `VMcnt`, `VScnt`, `LGKMcnt`, and `EXPcnt`; notably FLAT instructions use
  both `LGKMcnt` and either `VMcnt` or `VScnt`, messages use `LGKMcnt`, and
  LDS parameter/direct loads use `EXPcnt`, at `rdna3/README.md:1762` through
  `:1811`.
- The same table says `S_WAIT_EVENT` bit 0 selects export-ready waiting and that
  exceptions wait for this to complete at `rdna3/README.md:1779`.
- Section 5.7 defines `S_DELAY_ALU` stream-history semantics: `INSTID` counts
  previously issued VALU instructions, branched-over instructions do not count,
  VALU instructions skipped due to `EXEC==0` do count, `SKIP` counts all
  instruction types, and a later unconsumed `S_DELAY_ALU` replaces an earlier
  one at `rdna3/README.md:1813` through `:1855`.

XML evidence:

- `S_WAITCNT` and `OPR_WAITCNT` encode `EXP`, `LGKM`, and `VM` fields at
  `amdgpu_isa_rdna3.xml:56567` through `:56579` and
  `amdgpu_isa_rdna3.xml:186752` through `:186787`.
- `S_WAITCNT_DEPCTR` and `OPR_WAITCNT_DEPCTR` encode dependency-counter fields
  at `amdgpu_isa_rdna3.xml:56535` through `:56547` and
  `amdgpu_isa_rdna3.xml:186790` through `:186830`.
- `S_WAIT_EVENT` uses a generic `OPR_SIMM16` at
  `amdgpu_isa_rdna3.xml:56624` through `:56636`; no bit-level event table or
  exception-wait rule was found.
- `S_DELAY_ALU` uses `OPR_DELAY` at `amdgpu_isa_rdna3.xml:56503` through
  `:56515`, and `OPR_DELAY` enumerates `INSTID0`, `INSTID1`, and `INSTSKIP`
  fields at `amdgpu_isa_rdna3.xml:174070` through `:174260`. It does not encode
  the cross-instruction history, branch/skipped-instruction counting, or
  replacement rules.

Impact:

XML consumers can decode wait and delay operands, but cannot infer which
producers feed each counter, how `S_WAIT_EVENT` should block, or how to apply
`S_DELAY_ALU` to an instruction stream.

### RDNA3-XML-033: Signed SALU add/sub descriptions say carry-out instead of signed overflow

Manual evidence:

- Chapter 6.3 defines SCC for signed add/sub as signed overflow at
  `rdna3/README.md:1965`.
- Chapter 6.4 marks `S_ADD_I32`, `S_SUB_I32`, and `S_ADDK_I32` as overflow
  producers and says their SCC result is overflow at `rdna3/README.md:1974`,
  `:1977`, and `:1985`.

XML evidence:

- `S_ADD_I32` says the signed 32-bit add stores the carry-out bit into SCC at
  `amdgpu_isa_rdna3.xml:46392` through `:46393`.
- `S_SUB_I32` says the signed 32-bit subtract stores the carry-out bit into SCC
  at `amdgpu_isa_rdna3.xml:46531` through `:46532`.
- `S_ADDK_I32` says the signed 16-bit-immediate add stores the carry-out bit
  into SCC at `amdgpu_isa_rdna3.xml:55913` through `:55914`.

Impact:

An XML-only semantic generator can implement unsigned carry-out for signed
forms whose manual contract is signed overflow. The difference is observable:
`0x7fffffff + 1` has signed overflow without unsigned carry-out, while
`0xffffffff + 1` has unsigned carry-out without signed overflow.

### RDNA3-XML-034: Memory aperture selector formulas are prose-only

Manual evidence:

- Chapter 6.9 says `PRIVATE_BASE`, `PRIVATE_LIMIT`, `SHARED_BASE`, and
  `SHARED_LIMIT` originate from `SH_MEM_BASES` and are usable by SALU and VALU
  operations as 64-bit unsigned integers at `rdna3/README.md:2087` through
  `:2104`.
- The same section gives the `ptr32` formulas for base/limit values and defines
  the illegal-address "Hole" predicate at `rdna3/README.md:2104` through
  `:2106`.

XML evidence:

- XML enumerates the aperture source selectors but gives each `N/A`
  descriptions in both representative 9-bit and scalar source classes at
  `amdgpu_isa_rdna3.xml:177104` through `:177119` and
  `amdgpu_isa_rdna3.xml:184290` through `:184305`.
- XML names `HW_REG_SH_MEM_BASES`, but its predefined-value description is
  also `N/A` at `amdgpu_isa_rdna3.xml:174373` through `:174376`.
- Searches found no structured XML record for the `ptr32` split, the
  `SH_MEM_BASES` bitfield source of the four constants, the zero-base aperture
  disable rule, or the address-hole predicate.

Impact:

An XML-only consumer can decode the selector numbers but cannot reconstruct
the aperture base/limit values or flat-address validity rules without the
manual prose.

### RDNA3-XML-035: VOP3SD `SDST` field metadata conflicts with the manual

Manual evidence:

- Chapter 7.1 says VOP3SD has `SDST` instead of `OPSEL` and `ABS`, is used only
  for carry-out/div-scale/mad/dot2acc forms, and is not used for `V_CMP*`, at
  `rdna3/README.md:2148` through `:2155`.
- The VALU field table says `VDST` is a scalar destination for `V_READLANE` and
  `V_CMP`, and that it cannot be `M0` or `EXEC`; the same table says `SDST` is
  the scalar result of operations that produce scalar output, cannot be `M0` or
  `EXEC`, supports `NULL`, and is not used for `V_CMP`, at
  `rdna3/README.md:2168` through `:2169`.

XML evidence:

- The generic `VOP3_SDST_ENC` field description labels `SDST` as the
  destination operand for a compare result at `amdgpu_isa_rdna3.xml:8630`
  through `:8631`, even though the manual says VOP3SD is not used for
  `V_CMP*`.
- Concrete VOP3SD instruction entries do carry the expected scalar destination
  shape. For example, `V_ADD_CO_CI_U32` uses `SDST` with `FMT_NUM_M64` and
  `OPR_SREG` at `amdgpu_isa_rdna3.xml:77139` through `:77143`, and `OPR_SREG`
  is the scalar-register class including `VCC` and `NULL` at
  `amdgpu_isa_rdna3.xml:181440` through `:181444`.

Impact:

Instruction-specific operands are mostly recoverable, but format-level XML
consumers get a misleading VOP3SD `SDST` role and cannot rely on the generic
field description for the manual's non-compare contract.

### RDNA3-XML-036: Generic VOP3 modifier and `OPSEL` legality is prose-only

Manual evidence:

- Chapter 7.1 says VOP3 adds `NEG`, `ABS`, `OMOD`, `CLAMP`, and `OPSEL`, but
  limits `NEG`, `ABS`, and `OMOD` to floating-point use at
  `rdna3/README.md:2136` through `:2141` and `rdna3/README.md:2169` through
  `:2177`.
- Section 7.2.2.1 further says input modifiers are undefined for non-FP inputs
  except selected move/cndmask forms, and are not supported for readlane,
  readfirstlane, writelane, integer/bitwise, permlane, or QSAD, at
  `rdna3/README.md:2264` through `:2276`.
- Section 7.2.2.4 says VOP3 `OPSEL` is usable only for a subset of VOP3 and
  promoted VOP1/VOP2/VOPC instructions, with the allow-list beginning at
  `rdna3/README.md:2308` through `:2318`.
- Section 7.5 says `V_DOT4...IU...`, `V_DOT8...IU...`, and integer WMMA
  forms repurpose `NEG[1:0]` as signed/unsigned selectors for source 0 and
  source 1, with `NEG[2]` undefined and `NEG_HI` required zero, at
  `rdna3/README.md:2644` through `:2646`.

XML evidence:

- The generic VOP3 field descriptions define `ABS`, `CLAMP`, `NEG`, `OMOD`,
  and `OP_SEL`, but only refer to non-structured `OPF_NO*` lists or generic
  16-bit selection semantics at `amdgpu_isa_rdna3.xml:6529` through `:6606`.
- The VOP3P field descriptions similarly define `NEG` and `NEG_HI` only as
  floating-point negation of the low/high operation sources at
  `amdgpu_isa_rdna3.xml:10375` through `:10386` and
  `:10654` through `:10665`.
- Searches for `OPF_NOABS`, `OPF_NONEG`, `OPF_NOOMOD`, and `OPF_NOCLAMP` found
  only these field-description references, not structured per-instruction
  legality records.
- The XML `OP_SEL` field text gives the basic 16-bit high/low selector
  behavior, but not the manual's VOP3 allow-list or the requirement that
  non-16-bit operands/results use zero selector bits.

Impact:

An XML-only validator can decode the modifier bits, but cannot decide whether a
particular VOP3/VOP3P instruction may legally use them, or whether selected
integer DOT opcodes interpret `NEG[1:0]` as signedness rather than negation,
without manual-derived opcode metadata.

### RDNA3-XML-037: VALU source-combination legality is prose-only

Manual evidence:

- Section 7.2.2.3 says not every expressible source combination is legal, caps
  each instruction at two scalar values, permits one literal extension, counts
  implicit VCC uses for add/sub carry, div-fmas, and cndmask, gives special
  64-bit shift rules, and defines same-SGPR/same-literal size accounting at
  `rdna3/README.md:2288` through `:2306`.
- Section 7.2.5 identifies VALU forms that use SGPRs as masks or carry values
  and clarifies how VOP3 `SRC2`/`SDST` stand in for VCC, at
  `rdna3/README.md:2398` through `:2418`.

XML evidence:

- `OPR_SRC` is a broad scalar-or-vector source class at
  `amdgpu_isa_rdna3.xml:175990` through `:176000`; it does not encode
  cross-operand scalar-source counts or implicit VCC accounting.
- VOP3 literal encoding conditions check which raw source fields equal the
  literal selector, including shared-extension cases such as
  `has_lit_0_has_lit_1`, `has_lit_0_has_lit_2`, and
  `has_lit_0_has_lit_1_has_lit_2` at `amdgpu_isa_rdna3.xml:6200` through
  `:6462`. Those conditions size the extension word but do not encode the
  manual's source-combination legality rules.
- Representative VOP3SD entries expose the explicit `SDST` and `SRC2` mask
  operands, for example `V_ADD_CO_CI_U32` at
  `amdgpu_isa_rdna3.xml:77139` through `:77161`, but not the scalar-count or
  mask-sharing restrictions across all operands.

Impact:

Operand classes and literal-extension formats are available, but an XML-only
consumer still needs manual-derived cross-operand validation for scalar source
counts, literals, implicit VCC, and mask/carry operands.

### RDNA3-XML-038: `V_READLANE`/`V_WRITELANE` lane masking is not encoded

Manual evidence:

- Section 7.2.1 says the readlane lane-select is limited to the valid wave
  lanes by ignoring upper bits: 0-31 for wave32 and 0-63 for wave64, at
  `rdna3/README.md:2239` through `:2247`.

XML evidence:

- `V_READLANE_B32` uses `OPR_SSRC_LANESEL` for `SRC1` at
  `amdgpu_isa_rdna3.xml:113641` through `:113665`.
- `V_WRITELANE_B32` also uses `OPR_SSRC_LANESEL` for `SRC1` at
  `amdgpu_isa_rdna3.xml:113685` through `:113709`.
- The operand type says lane-select operands are generally scalar integer
  values in the range 0..63 at `amdgpu_isa_rdna3.xml:184321` through
  `:184326`, but it does not encode the wave32 high-bit masking behavior.

Impact:

The XML can identify lane-select operands, but it does not tell a decoder or
emulator that upper lane-number bits are ignored rather than illegal or used as
part of the lane index.

### RDNA3-XML-039: Wave64 SGPR mask-pass rules are prose-only

Manual evidence:

- Section 7.2.2.3 says SGPR source rules must be met for both passes of a
  wave64, and that mask sources increment the SGPR address for the second pass
  and may not be shared with other sources, at `rdna3/README.md:2306`.
- Section 7.2.6 says ordinary SGPR data is broadcast to all 64 work-items, but
  mask-style SGPR data such as carry-in or cndmask uses two consecutive SGPRs
  so each work-item receives a separate bit, at `rdna3/README.md:2420`
  through `:2422`.

XML evidence:

- XML entries expose the affected mask/carry operands, such as `V_ADD_CO_CI_U32`
  `SDST`/`SRC2` at `amdgpu_isa_rdna3.xml:77139` through `:77161`, and the
  readlane/writelane scalar operands cited above.
- The XML operand records do not encode the second-pass SGPR increment, the
  mask-source sharing restriction, or the distinction between broadcast SGPR
  data and per-lane mask bits in wave64.

Impact:

Wave64 validators or emulators need manual-derived rules for mask/carry SGPR
pairing and sharing. The XML operand widths alone do not describe the per-pass
behavior.

### RDNA3-XML-040: VOP3 output modifier and clamp edge rules are only partially structured

Manual evidence:

- Section 7.2.3.1 defines `OMOD` scaling, says integer and packed-F16 results
  ignore `OMOD`, describes the output-denorm/IEEE-mode interactions, and gives
  the opcodes that do not support output modifiers at `rdna3/README.md:2343`
  through `:2351`.
- The same section says `CLAMP` is signaling-compare control for `V_CMP`, clamps
  floating-point outputs to `[0, 1]`, clamps integer outputs to the signed or
  unsigned representable range, and lists opcode families where clamp is ignored
  at `rdna3/README.md:2353` through `:2371`.

XML evidence:

- The generic VOP3 `CLAMP` field has prose for compare, floating-point, signed
  integer, and unsigned integer meanings, and references `OPF_NOCLAMP` at
  `amdgpu_isa_rdna3.xml:6539` through `:6540`.
- The generic VOP3 `OMOD` field has prose for floating-point-only use,
  denormal/IEEE interactions, and references `OPF_NOOMOD` at
  `amdgpu_isa_rdna3.xml:6579` through `:6586`.
- Searches for `OPF_NOOMOD` and `OPF_NOCLAMP` found only the field-description
  references, not structured per-instruction allow/deny metadata.

Impact:

XML consumers can decode the raw fields and recover some generic semantics, but
cannot derive the manual's opcode-specific unsupported lists or the full
instruction-family policy from structured XML data.

### RDNA3-XML-041: Wave64 VALU SGPR destination/source overlap rule is prose-only

Manual evidence:

- Section 7.2.3.2 says a wave64 VALU instruction may issue as two wave32
  instructions and therefore may not both read and write the same SGPR value,
  because the first pass can overwrite a scalar value needed by the second pass,
  at `rdna3/README.md:2373` through `:2375`.

XML evidence:

- XML entries expose scalar read/write operands for affected VALU forms, such as
  `V_ADD_CO_CI_U32` `SDST` and `SRC2` at `amdgpu_isa_rdna3.xml:77139` through
  `:77161`.
- The XML has no wave64-pass or cross-operand hazard metadata that says scalar
  destinations must not overlap scalar sources in wave64 VALU instructions.

Impact:

An XML-only validator can see the individual scalar operands but cannot reject
or warn on same-SGPR read/write tuples that the manual calls unpredictable for
wave64 execution.

### RDNA3-XML-042: `PERMLANE`-after-`CMPX` hazard is absent

Manual evidence:

- Section 7.2.8 says `V_PERMLANE` may not occur immediately after `V_CMPX` and
  requires inserting another VALU opcode such as `V_NOP`, at
  `rdna3/README.md:2432` through `:2434`.

XML evidence:

- The XML instruction entries for `V_PERMLANE16_B32`,
  `V_PERMLANEX16_B32`, and `V_PERMLANE64_B32` describe the standalone
  gather/permutation operations at `amdgpu_isa_rdna3.xml:101888` through
  `:102022` and `amdgpu_isa_rdna3.xml:69851` through `:69880`.
- `V_CMPX_*` and `V_PERMLANE*` are ordinary instruction entries in the XML; no
  instruction-stream adjacency hazard or padding requirement is encoded.

Impact:

An XML-only scheduler or validator cannot discover that a `V_CMPX` followed
immediately by `V_PERMLANE*` is forbidden by the ISA.

### RDNA3-XML-043: Compact true16 VGPR half-addressing is prose-only

Manual evidence:

- Section 7.4 says non-packed 16-bit VALU instructions can separately address
  the two halves of a 32-bit VGPR and names them `V0.L` and `V0.H` at
  `rdna3/README.md:2551` through `:2559`.
- For VOP1, VOP2, and VOPC encodings, the manual defines
  `SRC/DST[6:0]` as the 32-bit VGPR address, `SRC/DST[7]` as the high/low
  half selector, and says only 256 16-bit VGPRs are addressable at
  `rdna3/README.md:2561` through `:2563`.
- For VOP3, VOP3P, and VINTERP encodings, the manual says `SRC/DST[7:0]`
  remains the 32-bit VGPR address and `OPSEL` selects the half, giving a wave
  access to 512 16-bit VGPRs at `rdna3/README.md:2565` through `:2567`.

XML evidence:

- The `ENC_VOP1` field table exposes `SRC0` and `VDST` widths/offsets, but the
  descriptions are generic operand labels and do not split bit 7 into a true16
  high-half selector or state the 256 16-bit VGPR limit:
  `amdgpu_isa_rdna3.xml:845` through `:860`.
- The `ENC_VOPC` and `ENC_VOP2` field tables likewise expose 8-bit `VSRC1` and
  `VDST` fields as generic operands, without the Chapter 7.4 true16 split:
  `amdgpu_isa_rdna3.xml:1115` through `:1130` and `:1237` through `:1262`.
- XML does contain generic OPSEL text for VOP3 and VINTERP true16
  half-selection and destination preservation at `amdgpu_isa_rdna3.xml:1844`
  through `:1845` and `:2190` through `:2191`, plus VOP3P low/high source
  selector descriptions at `:2060` through `:2071`, but it does not encode the
  compact VOP1/VOP2/VOPC bit-7 rule or the 256-versus-512 16-bit VGPR reach.

Impact:

Consumers using XML field tables alone cannot reconstruct the compact true16
VGPR namespace, disassembly names such as `.l`/`.h`, or the addressability
change between compact encodings and OPSEL-based encodings.

### RDNA3-XML-044: VOPD literal-length conditions omit `SRCY0 == 255`

Manual evidence:

- Section 7.6 says each VOPD operation has a source 0 that may be a VGPR, SGPR,
  or constant, and the instruction-field table says both `src0X` and `src0Y`
  may be literal constants at `rdna3/README.md:2680` through `:2681` and
  `rdna3/README.md:2716` through `:2717`.
- Section 7.6 restricts the pair to at most one literal constant, or a shared
  literal, at `rdna3/README.md:2701` through `:2702`.
- Section 15.3.7 says VOPD can be followed by a 32-bit literal constant at
  `rdna3/README.md:6573`, and its field table gives `SRCX0 == 255` as literal
  and says `SRCY0` uses the same enumeration at `rdna3/README.md:6596`
  through `:6620`.

XML evidence:

- The 64-bit `VOPDXY` encoding condition checks only `SRCX0 != 255` at
  `amdgpu_isa_rdna3.xml:11957` through `:11978`.
- The 96-bit `VOPDXY_INST_LITERAL` conditions likewise key only on
  `SRCX0 == 255` / `SRCX0 != 255` at `amdgpu_isa_rdna3.xml:12121` through
  `:12160`.
- No corresponding condition ties the 96-bit form to `SRCY0 == 255`, even
  though the XML field table does expose `SRCY0` as the Y source-0 field at
  `amdgpu_isa_rdna3.xml:12024` and `:12217`.

Impact:

An XML-derived decoder can mis-size a valid VOPD pair whose Y operation is the
only operation using a literal source, treating a 96-bit instruction as 64 bits.

### RDNA3-XML-045: VOPD pair legality and source-cache rules are prose-only

Manual evidence:

- Section 7.6 says VOPD is legal only for wave32, must not be used by wave64,
  and hardware does not function correctly if the restrictions are not met at
  `rdna3/README.md:2676`.
- The restrictions include per-slot VGPR counts, per-slot SGPR/literal limits,
  `SRC0` source classes, `VSRC1` VGPR-only use, source-cache bank and read-port
  limits, `FMAMK` source-port behavior, `SRC2` even/odd pairing, one literal,
  one even and one odd destination, instruction independence, no DPP, and
  wave32-only at `rdna3/README.md:2687` through `:2706`.
- Section 7.6 also says simultaneous paired operations read old VGPR values
  when one operation reads a VGPR written by the other at
  `rdna3/README.md:2683`.

XML evidence:

- The XML VOPD encoding records carry raw fields for `OPX`, `OPY`, `SRCX0`,
  `SRCY0`, `VSRCX1`, `VSRCY1`, `VDSTX`, and `VDSTY` at
  `amdgpu_isa_rdna3.xml:11993` through `:12073` and `:12193` through `:12268`.
- Searching the RDNA3 XML for the source-cache, bank, port, wave32, DPP,
  independence, or hard-rule VOPD prose finds no structured equivalent around
  the VOPD encoding or instruction entries.

Impact:

An XML-only validator can reconstruct VOPD fields and opcode names, but cannot
derive the hardware legality contract that distinguishes valid pairs from
encodings the manual says do not function correctly.

### RDNA3-XML-046: VOPD destination-accumulator inputs are not operand-modeled

Manual evidence:

- Section 7.6 says VOPD `FMAC_F32` uses the destination operand as `SRC2`, and
  `DOT2ACC_F32_F16` / `DOT2ACC_F32_BF16` also use the destination operand as a
  `SRC2` input at `rdna3/README.md:2699`.
- Chapter 16.11 repeats that `V_DUAL_DOT2ACC_F32_F16` and
  `V_DUAL_DOT2ACC_F32_BF16` accumulate with the destination and use the initial
  destination value as `S2` at `rdna3/README.md:15591` through `:15593` and
  `rdna3/README.md:15643` through `:15646`.

XML evidence:

- The `V_DUAL_DOT2ACC_F32_F16` XML description mentions that the initial value
  in `D` is used as `S2`, but the `VDSTX` and `VDSTY` operands are marked only
  `Input="false" Output="true"` and no operand record models the old
  destination value as an input at `amdgpu_isa_rdna3.xml:169471` through
  `:169580`.
- The `V_DUAL_FMAC_F32` XML entries similarly model `VDSTX`/`VDSTY` as output
  operands while the description says the operation accumulates into the
  destination at `amdgpu_isa_rdna3.xml:167999` through `:168109`.

Impact:

XML-derived def-use, dependency, or source-cache validation can miss the
old-destination read for accumulator-style VOPD slots unless it supplements the
operand records with manual-derived special cases.

### RDNA3-XML-047: VOPD paired-exception coalescing is missing from XML

Manual evidence:

- Section 7.6 says VOPD instruction pairs generate only a single exception if
  either or both paired operations raise an exception at
  `rdna3/README.md:2727`.

XML evidence:

- The VOPD encoding descriptions at `amdgpu_isa_rdna3.xml:11980` and
  `amdgpu_isa_rdna3.xml:12162` only describe the X/Y opcode encoding and
  promotion relationship.
- Searching the RDNA3 XML for the manual's single-exception wording finds no
  VOPD paired-exception record.

Impact:

An XML-only simulator or trap/exception validator cannot derive the VOPD
exception coalescing rule from the machine-readable source.

### RDNA3-XML-048: Scalar-buffer descriptor range semantics are prose-only

Manual evidence:

- Section 8.1.2 says `S_BUFFER_LOAD_*` uses a buffer resource descriptor for
  `base_address`, `stride`, and `num_records`, ignores other descriptor fields,
  uses `stride` only for bounds checking, requires nonnegative `inst_offset`,
  and gives the address and size formula at `rdna3/README.md:2962` through
  `:2985`.
- Section 8.4.1 says buffer address out-of-range is determined by
  `offset >= ((stride == 0 ? 1 : stride) * num_records)`, and out-of-range
  DWORDs in scalar buffer loads return zero, including partial multi-DWORD
  requests, at `rdna3/README.md:3031` through `:3041`.

XML evidence:

- The XML `ENC_SMEM` field text records that `OFFSET` is signed and that
  negative buffer offsets produce `MEMVIOL`, and says `SBASE` carries a base
  address or buffer descriptor with only `BASE`, `STRIDE`, and `NUM_RECORDS`
  used at `amdgpu_isa_rdna3.xml:645` through `:666`.
- The `S_BUFFER_LOAD_B32` through `S_BUFFER_LOAD_B512` entries use 128-bit
  `FMT_RSRC_SCALAR` `SBASE` operands and implicit `OPR_GPUMEM` return payloads
  at `amdgpu_isa_rdna3.xml:41324` through `:41565`.
- Those records do not structure the descriptor bit extraction formula,
  `stride == 0 ? 1 : stride` size rule, per-DWORD out-of-range zeroing, or
  partial multi-DWORD return behavior.

Impact:

An XML-only scalar-buffer executor or fuzzer can identify that a resource
descriptor is used, but cannot reconstruct the address/range contract that
determines which DWORDs load real memory and which return zero.

### RDNA3-XML-049: SMEM dependency-counter increments are prose-only

Manual evidence:

- Section 8.2 says scalar memory loads can return out-of-order or partially
  when a load crosses cache lines, and assigns `LGKMcnt` increments by width:
  `+1` for single-DWORD fetches or cache invalidates, `+2` for fetches of two
  or more DWORDs, with equal decrements on completion at
  `rdna3/README.md:2995` through `:3001`.
- The same section says cache invalidates are not known complete until
  `LGKMcnt == 0` at `rdna3/README.md:3003` through `:3005`.

XML evidence:

- XML exposes `S_WAITCNT_LGKMCNT` and says `LGKMCNT` tracks outstanding LDS,
  GDS, scalar memory, and message events at
  `amdgpu_isa_rdna3.xml:56270` through `:56271`.
- The SMEM load entries record payload widths and the SMEM functional group at
  `amdgpu_isa_rdna3.xml:41064` through `:41625`, but no field or instruction
  metadata encodes the Chapter 8 `+1` versus `+2` increment rule, partial
  cache-line return behavior, or invalidate completion dependency.

Impact:

XML consumers can see that SMEM belongs to the scalar-memory family and that
`LGKMCNT` exists, but cannot derive the width-sensitive wait-counter accounting
required to model `S_WAITCNT LGKMcnt 0` precisely.

### RDNA3-XML-050: SMEM invalidate group restrictions are prose-only

Manual evidence:

- Section 8.3 defines scalar memory groups and says `INV` must be in a group by
  itself and may not be in a clause at `rdna3/README.md:3007` through `:3015`.

XML evidence:

- XML records `S_GL1_INV` and `S_DCACHE_INV` as operandless SMEM opcodes at
  `amdgpu_isa_rdna3.xml:41584` through `:41616`.
- `S_CLAUSE` has a structured clause operand elsewhere in XML, but searches
  around the SMEM instruction entries found no group-state or invalidate-only
  clause exclusion metadata.

Impact:

An XML-only stream validator can decode scalar cache invalidations, but cannot
tell that they must be isolated from other SMEM instructions and excluded from
`S_CLAUSE` sequences.

### RDNA3-XML-051: Buffer resource descriptor bitfields and validity semantics are opaque

Manual evidence:

- Section 9.6 defines the four-SGPR buffer resource descriptor fields,
  including base address, stride, swizzle enable, `Num_records`, `Dst_sel`,
  resource format, index stride, add-tid enable, reserved bits, `OOB_SELECT`,
  and resource type, at `rdna3/README.md:3534` through `:3558`.
- The same section says all-zero resources force loads to return zero and
  stores to be ignored, and says resource/instruction type mismatches ignore
  the instruction at `rdna3/README.md:3561` through `:3568`.

XML evidence:

- XML resource formats such as `FMT_RSRC_SCRATCH`,
  `FMT_RSRC_SCRATCH_BYTE`, `FMT_RSRC_TYPED`, and `FMT_RSRC_VECTOR` describe a
  128-bit buffer resource constant but expose a single 128-bit `Descriptor`
  field at `amdgpu_isa_rdna3.xml:171512` through `:171635`.
- Representative buffer instruction operands use those descriptor formats, but
  the XML entries do not break out descriptor bitfields or encode the
  unbound-resource and resource/instruction mismatch behavior.

Impact:

XML consumers can identify that an operand is a resource descriptor, but cannot
derive descriptor-field layout, validity, unbound-resource behavior, or
resource/instruction mismatch semantics without the manual.

### RDNA3-XML-052: Buffer addressing, swizzle, and `OOB_SELECT` formulas are prose-only

Manual evidence:

- Section 9.4 describes buffer addresses as base plus SGPR offset plus a
  buffer-offset built from instruction offset, `IDXEN`/`OFFEN`, VGPR index,
  VGPR offset, stride, add-tid, and optional swizzle fields at
  `rdna3/README.md:3363` through `:3412`.
- Section 9.4.1 defines the two-bit `OOB_SELECT` modes, payload-inclusive
  bounds checks, all-or-nothing formatted/atomic behavior, and per-component
  MUBUF versus MTBUF B64/B96/B128 behavior at `rdna3/README.md:3414` through
  `:3432`.
- Sections 9.4.1.1 through 9.4.2 give the structured, raw, scratch, scalar, and
  swizzled address formulas at `rdna3/README.md:3434` through `:3501`.

XML evidence:

- XML `ENC_MUBUF` and `ENC_MTBUF` field records expose `OFFSET`, `SLC`, `DLC`,
  `GLC`, `OP`, `FORMAT`, `VADDR`, `VDATA`, `SRSRC`, `TFE`, `OFFEN`, `IDXEN`,
  and `SOFFSET` bit positions at `amdgpu_isa_rdna3.xml:2571` through `:2975`.
- The XML field descriptions do not encode the manual's address formulas,
  swizzle equation, add-tid behavior, two-bit `OOB_SELECT`, payload-inclusive
  bounds rules, or MUBUF/MTBUF component-level range differences.

Impact:

An XML-only emulator or fuzzer can decode the address-related fields, but cannot
compute hardware buffer addresses or range behavior for nontrivial descriptor
settings.

### RDNA3-XML-053: Buffer data format, `DST_SEL`, D16 packing, and mismatch behavior are partial

Manual evidence:

- Section 9.2 defines VGPR address/data usage and the raw buffer load/store data
  formats, including sign/zero extension and D16/D16_HI half placement, at
  `rdna3/README.md:3246` through `:3312`.
- Section 9.3 says data format can come from the MTBUF instruction, MUBUF
  resource, or opcode-derived type, and defines `DST_SEL` identity/resource
  selection at `rdna3/README.md:3313` through `:3342`.
- Section 9.3.1 defines D16 packing, preservation, and conversion rounding, and
  section 9.3.2 defines LOAD/STORE_FORMAT component-count versus data-format
  mismatch behavior at `rdna3/README.md:3343` through `:3362`.

XML evidence:

- XML instruction entries carry opcode names, operand widths, and some natural
  language descriptions for typed, formatted, raw, D16, cache, and atomic
  buffer forms at `amdgpu_isa_rdna3.xml:35799` through `:41032`.
- Those entries do not provide structured metadata for `DST_SEL`, resource
  versus instruction versus opcode-derived format selection, D16 packing and
  preservation, conversion rounding, or format/opcode component-count mismatch
  behavior.

Impact:

Generated decoders can size many operands from XML, but execution and
oracle-generation still need manual-derived rules for formatted conversion,
destination channel selection, D16 packing, and mismatch cases.

### RDNA3-XML-054: Image resource, sampler, and BVH descriptor bitfields are opaque

Manual evidence:

- Section 10.5 defines the image resource (`T#`) as four or eight SGPRs that
  carry base address, dimensions, tiling, format, `dst_sel`, level, array,
  PRT/min-LOD, metadata, and compression fields at `rdna3/README.md:3918`
  through `:3964`.
- The same section says an all-zero resource is unbound, returns zeros, and
  generates no memory transaction at `rdna3/README.md:3962`.
- Section 10.6 defines sampler (`S#`) clamp, filter, compare, LOD, border-color,
  and anisotropy fields at `rdna3/README.md:3964` through `:3988`.
- Section 10.9 and the BVH definitions use a separate 128-bit BVH resource and
  impose `R128=1` for BVH ray tracing at `rdna3/README.md:4081` through `:4180`
  and `rdna3/README.md:25026` through `:25046`.

XML evidence:

- `FMT_IMG` is a 256-bit image resource constant but exposes a single 256-bit
  `Descriptor` field at `amdgpu_isa_rdna3.xml:169950` through `:169969`.
- `FMT_IMG_BVH` is a 128-bit BVH resource constant but exposes a single 128-bit
  `Descriptor` field at `amdgpu_isa_rdna3.xml:169970` through `:169989`.
- `FMT_SAMP` is a 128-bit sampler constant but exposes a single 128-bit
  `Descriptor` field at `amdgpu_isa_rdna3.xml:171697` through `:171715`.

Impact:

XML-only consumers can identify descriptor operands, but cannot derive image or
sampler field layouts, unbound-resource behavior, format/dimension metadata, or
BVH resource restrictions without the manual.

### RDNA3-XML-055: Image address tables and VGPR packing rules are prose-only

Manual evidence:

- Sections 10.2 and 10.3 define no-sampler and sampler address tables by opcode,
  dimensionality, A16 state, MSAA usage, cube face encoding, and suffix-driven
  address components at `rdna3/README.md:3694` through `:3838`.
- Section 10.4 defines offset, bias, z-compare, derivative, body, and ACNT-based
  VGPR packing, including bias/derivative mutual exclusion and 32-bit versus
  16-bit derivative layouts at `rdna3/README.md:3838` through `:3870`.
- Section 10.1.5 defines NSA grouping and the rule that paired 16-bit address
  halves cannot be split across VGPRs at `rdna3/README.md:3676` through `:3692`.

XML evidence:

- `ENC_MIMG` and `MIMG_NSA1` expose the raw `DIM`, `A16`, `NSA`, `VADDR`, and
  `VADDRA` through `VADDRD` fields at `amdgpu_isa_rdna3.xml:2978` through
  `:3295` and `:3971` through `:4333`.
- Instruction entries expose broad operand sizes, for example `IMAGE_LOAD`
  `VADDR` as 128 bits at `amdgpu_isa_rdna3.xml:28987` through `:29038`, but they
  do not encode the manual's per-op/per-DIM address tables, cube face formulas,
  suffix ordering, derivative packing, or NSA group semantics.

Impact:

XML can decode which address register fields are present, but an emulator,
validator, or fuzzer still needs manual-derived tables to determine which VGPR
components are consumed for each image form.

### RDNA3-XML-056: Image TFE/LWE status protocol is only partially represented

Manual evidence:

- Section 10.1.1 says TFE and LWE write an extra VGPR after all image data
  returns, and defines the status payload as texture-fail, LOD-warning, and LOD
  fields at `rdna3/README.md:3640` through `:3659`.
- The same section says texture-fail and LOD-warning are mutually exclusive, with
  texture-fail winning precedence, and describes the NACK behavior for TFE.

XML evidence:

- The MIMG field records name `TFE` as texture-fail enable and `LWE` as LOD
  warning enable at `amdgpu_isa_rdna3.xml:3177` through `:3185` and `:3255`
  through `:3263`, and duplicate those names for `MIMG_NSA1` at `:4170` through
  `:4257`.
- The `DMASK` field description includes the special `DMASK==0` override and
  says no TFE status is generated when the fetch is dropped at
  `amdgpu_isa_rdna3.xml:3134` through `:3149`, but the XML does not encode the
  status word layout, extra destination window, TFE/LWE precedence, or sampler-only
  LWE restriction.

Impact:

XML consumers can see the enable bits, but cannot derive destination register
pressure or the returned status value for partially resident texture behavior.

### RDNA3-XML-057: Image data-format, store-fill, sampler, and atomic legality rules are partial

Manual evidence:

- Section 10.4 says loads expand texture data to canonical RGBA using
  `T#.dst_sel`, stores write whole image elements with missing components filled
  as zero, D16 packs load/store halves, image atomic `DMASK` values are restricted
  by 32-bit/64-bit and compare-swap form, and sampler operations flush denormals
  at `rdna3/README.md:3872` through `:3904`.
- Section 10.4.1 defines the VGPR-side data formats for SINT, UINT, other
  formats, atomics, and ASTC at `rdna3/README.md:3906` through `:3917`.

XML evidence:

- MIMG instruction entries carry opcode names, broad operand formats, and some
  natural-language descriptions for loads, stores, atomics, samples, gathers,
  and queries at `amdgpu_isa_rdna3.xml:28987` through `:35752`.
- The `DMASK` field text includes component routing and D16 write packing at
  `amdgpu_isa_rdna3.xml:3134` through `:3149`, but XML does not provide
  structured metadata for `T#.dst_sel`, store whole-element fill/ignore behavior,
  sampler denormal flushing, atomics' legal `DMASK` sets, or VGPR data-format
  conversion by resource format.

Impact:

Generated decoders can size operands and classify image opcodes, but image
execution and oracle generation still need manual rules for format conversion,
component fill/selection, sampler numeric behavior, and atomic legality.

### RDNA3-XML-058: BVH encoding restrictions and return-order behavior are prose-only

Manual evidence:

- Section 10.9 and Chapter 16.18 define 32-bit and 64-bit BVH ray address
  layouts, A16 packed ray-direction forms, node-result contents, and NSA grouping
  at `rdna3/README.md:4081` through `:4180` and `rdna3/README.md:24973` through
  `:25095`.
- Chapter 16.18 says BVH opcodes require `DMASK=0xf`, `D16=0`, `R128=1`,
  `UNRM=1`, `DIM=0`, `LWE=0`, and `TFE=0`; those restrictions are
  compiler-respected and not hardware-enforced, and the hardware ignores return
  order settings for BVH fetches at `rdna3/README.md:25026` through `:25050`.

XML evidence:

- XML has `IMAGE_BVH_INTERSECT_RAY` and `IMAGE_BVH64_INTERSECT_RAY` entries with
  352-bit and 384-bit `VADDR` operands, 128-bit `FMT_IMG_BVH` resources, opcode
  numbers, and `BVH` functional subgroup metadata at `amdgpu_isa_rdna3.xml:30985`
  through `:31116`.
- Those entries do not encode the required field values, undefined-behavior
  contract for violations, return-order override, or detailed A16/NSA ray-data
  layouts.

Impact:

XML can distinguish the two BVH opcodes and broad operand widths, but it cannot
validate legal BVH encodings or reproduce the manual's ray-data and ordering
contract.

### RDNA3-XML-059: FLAT/GLOBAL/SCRATCH address formulas and aperture rules are prose-only

Manual evidence:

- Chapter 11 says FLAT/GLOBAL/SCRATCH instructions do not use SRDs, that flat
  per-thread addresses may resolve to global, scratch/private, LDS/shared, or
  invalid apertures, and that scratch uses `FLAT_SCRATCH` as an implicit per-wave
  base at `rdna3/README.md:4216` through `:4230`.
- The field table gives SEG-specific `ADDR`, `SADDR`, `SVE`, and `OFFSET`
  meanings, including the flat-offset rule that the MSB is ignored/forced zero,
  at `rdna3/README.md:4241` through `:4253`; Chapter 15.9 repeats that FLAT uses
  a 12-bit unsigned offset while Scratch/Global use a 13-bit signed offset at
  `rdna3/README.md:7137` through `:7148`.
- Section 11.2 gives the global, scratch, and flat equations, says flat aperture
  selection ignores `INST_OFFSET`, defines the scratch `SWIZZLE(offset,TID)`
  formula, and lists the SADDR/SVE addressing modes at `rdna3/README.md:4368`
  through `:4468`.

XML evidence:

- XML exposes `ENC_FLAT`, `ENC_FLAT_GLOBAL`, and `ENC_FLAT_SCRATCH` raw fields,
  including `ADDR`, `DATA`, cache bits, `OFFSET`, `OP`, `SADDR`, `SEG`, `SVE`,
  and `VDST`, at `amdgpu_isa_rdna3.xml:3422` through `:3969`.
- Those records describe `OFFSET` generically as a 13-bit signed offset and
  `SEG`/`SVE` only at a high level at `amdgpu_isa_rdna3.xml:3540` through
  `:3599`, `:3735` through `:3794`, and `:3897` through `:3956`.
- Representative instruction entries carry operand shells such as `FLAT_LOAD_U8`
  with `ADDR`, implicit `OPR_FLAT_SCRATCH`, and implicit `OPR_GPUMEM` at
  `amdgpu_isa_rdna3.xml:18668` through `:18709`, but do not structure the
  SEG-specific address equations, aperture-base/limit comparisons, flat
  before-offset aperture test, scratch swizzle, or flat 12-bit offset exception.

Impact:

XML-only consumers can decode FLAT/GLOBAL/SCRATCH bitfields and broad operand
classes, but cannot derive the address that hardware will access, validate
offset modes, or distinguish flat/private/shared/invalid lanes without manual
prose.

### RDNA3-XML-060: FLAT/GLOBAL/SCRATCH wait, ordering, memory-error, and special atomic semantics are partial

Manual evidence:

- Section 11.1.1 says FLAT instructions execute as both LDS and global
  instructions, increment both `VMcnt`/`VScnt` and `LGKMcnt`, support 4-byte
  scratch atomics but return `MEMVIOL` for 8-byte scratch atomics, perform the
  aperture check before `inst_offset`, and can complete out of order with respect
  to the separate counter paths at `rdna3/README.md:4320` through `:4347`.
- Sections 11.1.2 and 11.1.3 say GLOBAL/SCRATCH instructions use only
  `VMcnt`/`VScnt`, that GLOBAL-to-LDS attempts return `MEMVIOL`, and that SCRATCH
  skips aperture checks at `rdna3/README.md:4349` through `:4366`.
- Section 11.3 defines invalid aperture, read-only write, misalignment, LDS
  out-of-range, bad-address load/store behavior, and `TRAPSTS.MEMVIOL` side
  effects at `rdna3/README.md:4470` through `:4483`.
- Chapter 11 and Chapter 15.9 list `GLOBAL_ATOMIC_CSUB_U32` as a GLOBAL-only
  opcode 55 and say `GLC` must be set to 1 at `rdna3/README.md:4316` and
  `rdna3/README.md:7195`.

XML evidence:

- Representative flat and global instruction entries identify broad functional
  groups and memory operands, for example `FLAT_LOAD_U8` is `VMEM`/`FLAT` at
  `amdgpu_isa_rdna3.xml:18668` through `:18709`, `FLAT_ATOMIC_SUB_U32` is
  `VMEM`/`ATOMIC`/`FLAT` at `amdgpu_isa_rdna3.xml:19979` through `:20032`, and
  `GLOBAL_ATOMIC_CSUB_U32` carries a natural-language clamp description at
  `amdgpu_isa_rdna3.xml:24041` through `:24086`.
- The entries do not encode dual-counter FLAT completion, flat ordering hazards,
  GLOBAL-to-LDS or 8-byte-flat-scratch-atomic `MEMVIOL`, bad-address zero/no-write
  policy, `TRAPSTS.MEMVIOL` updates, or the `GLOBAL_ATOMIC_CSUB_U32` `GLC=1`
  requirement as structured semantics.

Impact:

Generated metadata can classify these operations as vector memory instructions,
but executable models, validators, and fuzzers still need manual-derived rules
for wait-counter synchronization, per-lane error behavior, and trap side effects.

### RDNA3-XML-061: LDSDIR parameter/direct-load execution semantics are partial

Manual evidence:

- Section 12.2.1 says `LDS_PARAM_LOAD` is LDS-only, CU-mode-only, reads
  `P0`/`P10`/`P20` for one 32-bit attribute or two packed 16-bit attributes,
  derives primitive layout from `M0`, requires the parameter offset bits
  `[6:0]` to be zero, and writes whole quads when any pixel in the quad is
  active at `rdna3/README.md:4573` through `:4616`.
- Section 12.2.1.2 says LDSDIR reads use `EXPcnt`, have `WAITVDST`
  write-after-read hazard control, skip when `EXEC==0` and `EXPcnt==0`, and can
  complete out of order with exports at `rdna3/README.md:4620` through `:4632`.
- Section 16.14 repeats that `LDS_PARAM_LOAD` expands parameter data as
  `{P0, P10, P20, 0.0}`, packs FP16 attributes as `2*ATTR`/`2*ATTR+1`, and that
  both `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` run in whole-quad mode at
  `rdna3/README.md:22029` through `:22051`.

XML evidence:

- XML has the `ENC_LDSDIR` field map and `WAIT_VDST` field description at
  `amdgpu_isa_rdna3.xml:2253` through `:2328`.
- XML has `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` entries with `VDST`,
  `OPR_DSMEM`, and implicit `M0` operands at `amdgpu_isa_rdna3.xml:28897`
  through `:28969`.
- The XML entries do not structure the `P0`/`P10`/`P20` lane placement,
  `NewPrimMask` primitive derivation, FP16 attribute packing, whole-quad
  `EXEC` behavior, `EXPcnt` participation, `EXEC==0` skip condition, export
  ordering caveat, or CU-mode-only legality.

Impact:

An XML-only LDSDIR consumer can decode operands and the raw wait field, but
cannot faithfully emulate parameter expansion, direct-load write masks, or
wait-counter/legality behavior without manual-derived rules.

### RDNA3-XML-062: DS address, permute, GDS, and streamout semantics are partial

Manual evidence:

- Section 12.5 defines single-address DS offset concatenation, double-address
  scaling, ADDTID's `M0 + TID*4` formula, LDS atomic read-modify-write ordering
  per lane, and lane-permute wave64 half-wave/`EXEC`/index-wrap behavior at
  `rdna3/README.md:4765` through `:4915`.
- Section 12.6 says GDS uses the DS format but supports only one active lane
  per instruction, uses `M0[31:16]` as base and `M0[15:0]` as size, and that
  `DS_ADD_GS_REG_RTN`/`DS_SUB_GS_REG_RTN` target a streamout-register table
  selected by `OFFSET0[5:2]` at `rdna3/README.md:4942` through `:4962`.
- Section 16.15 and 16.15.1 provide the DS instruction definitions and list the
  DS opcodes that are GDS-only at `rdna3/README.md:22053` through `:22072` and
  `:23765` through `:23825`.

XML evidence:

- XML has the `ENC_DS` field map, including `GDS`, `OFFSET0`, `OFFSET1`,
  `ADDR`, `DATA0`, `DATA1`, and `VDST`, at `amdgpu_isa_rdna3.xml:2477`
  through `:2565`.
- XML entries for `DS_STORE_ADDTID_B32`, `DS_LOAD_ADDTID_B32`,
  `DS_PERMUTE_B32`, and `DS_BPERMUTE_B32` provide opcode shells and operands at
  `amdgpu_isa_rdna3.xml:18240` through `:18395`, and several GDS/GWS/GS-register
  entries have one-line descriptions at `:13364` through `:13535` and
  `:17678` through `:17740`.
- These records do not structure the ADDTID address formula, double-address
  scale factors, stride64 scaling, GDS first-active-lane rule, `M0` base/size
  interpretation, streamout register index table and 32/64-bit return split,
  permute wave64 half-wave behavior, disabled-lane zeroing, or the GDS-only
  limitation table.

Impact:

XML can identify the DS opcodes and broad operands, but targeted legality,
execution, and test generation still need manual-only rules for the
instruction-specific address, lane, GDS, and streamout contracts.

### RDNA3-XML-063: Float memory atomic numeric edge semantics are prose-only

Manual evidence:

- Chapter 13 says floating-point atomics can be LDS, Buffer, and
  Flat/Global/Scratch instructions, and that float-atomic-add rounding is fixed
  to round-to-nearest-even while `MODE.round` is ignored, at
  `rdna3/README.md:5011` through `:5017`.
- Section 13.2 gives LDS and cache-atomic denormal rules: LDS add uses input and
  output denormal controls, LDS compare/min/max use input-denormal control,
  cache min/max and compare-store use the low `MODE.fp_denorm` bit, and memory
  atomic add always flushes input denormals, at `rdna3/README.md:5021` through
  `:5053`.
- Section 13.3 defines SNaN quieting, DS add NaN propagation, min/max
  signed-zero and NaN ordering, compare-swap's `+0 == -0` equality rule, and
  float-add infinity/NaN/zero special cases at `rdna3/README.md:5064` through
  `:5104`.

XML evidence:

- XML exposes representative LDS float atomic entries such as `DS_ADD_F32` at
  `amdgpu_isa_rdna3.xml:13315` through `:13355`, and LDS F64 compare/min/max
  entries at `amdgpu_isa_rdna3.xml:16184` through `:16330`.
- XML exposes cache float atomic opcode shells such as
  `FLAT_ATOMIC_CMPSWAP_F32`, `FLAT_ATOMIC_MIN_F32`,
  `FLAT_ATOMIC_MAX_F32`, and `FLAT_ATOMIC_ADD_F32` at
  `amdgpu_isa_rdna3.xml:21428` through `:21668`, the corresponding global
  forms at `amdgpu_isa_rdna3.xml:26456` through `:26864`, and the corresponding
  buffer forms at `amdgpu_isa_rdna3.xml:40811` through `:41055`.
- These records state the broad operation and operand formats, but do not encode
  fixed add rounding, denormal flush/control tables, SNaN-to-QNaN conversion,
  NaN payload selection, signed-zero ordering, compare-swap's floating equality
  rule, or the float-add special cases.

Impact:

An XML-only decoder can discover the float atomic instruction set and operand
formats, but a semantic model or edge-case fuzzer cannot derive the numeric
behavior that distinguishes RDNA3 float atomics from host/library `fmin`,
`fmax`, `+`, or integer compare-swap semantics.

### RDNA3-XML-064: GWS and ordered-count synchronization semantics are partial

Manual evidence:

- Section 13.4 says GWS instructions use `LGKMcnt`, must be issued in a
  single-instruction clause bracketed by `S_WAITCNT LGKMcnt==0`, must avoid a
  following `S_ENDPGM`, and require distinct source/destination VGPRs for
  replay after a NACK, at `rdna3/README.md:5106` through `:5122`.
- Section 13.4.2 says GDS/GWS is single-lane, chooses the least-significant
  active lane when multiple `EXEC` bits are set, and defines special
  `EXEC==0` behavior for `ORDERED_COUNT`, `GWS_INIT`, `SEMA_BR`, and
  `GWS_BARRIER`, at `rdna3/README.md:5124` through `:5132`.
- Sections 13.4.3 and 13.4.4 define ordered-count queueing in wave-creation
  order, ordered-count field reinterpretation, target restrictions,
  append/consume full-`EXEC` counting, 64 GWS resources, `M0[21:16]` resource
  virtualization, resource-ID clamping/NACK behavior, and GWS instruction
  pseudocode at `rdna3/README.md:5134` through `:5251`.

XML evidence:

- XML has opcode shells and short descriptions for `DS_GWS_SEMA_RELEASE_ALL`,
  `DS_GWS_INIT`, `DS_GWS_SEMA_V`, `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and
  `DS_GWS_BARRIER` at `amdgpu_isa_rdna3.xml:13364` through `:13526`.
- XML has a `DS_ORDERED_COUNT` entry with `VDST`, `ADDR`, implicit DSMEM, and
  implicit `M0` operands, plus a short ordered-append-module description, at
  `amdgpu_isa_rdna3.xml:15280` through `:15323`.
- These entries do not structure the single-instruction-clause rule, required
  waits, NACK/replay behavior, `EXEC==0` exceptions, least-active-lane
  selection, ordered-count field reinterpretation, target legality, append and
  consume full-`EXEC` counting, GWS resource state, resource-ID formula,
  clamping/NACK policy, or barrier/semaphore queue semantics.

Impact:

An XML-only validator can find the GWS and ordered-count opcodes, but cannot
derive the synchronization, replay, lane-selection, or resource-state behavior
needed to validate legal programs or emulate GWS progress.

### RDNA3-XML-065: Export valid-mask, target, dependency, and dual-source rules are partial

Manual evidence:

- Chapter 14 says exports use `EXEC`, copy 32-bit data or packed 16-bit pairs
  from VGPRs, each target may be exported only once, `DONE` marks the last pixel
  or position export, primitive exports must set `DONE`, and mesh position/prim
  row exports use `M0` at `rdna3/README.md:5253` through `:5288`.
- Section 14.1 requires every pixel shader to have at least one export, applies
  `EXEC` to all exports, requires the last export to set `DONE`, constrains
  depth/MRT0/sample-mask ordering, and defines dual-source blend back-to-back
  requirements plus lane-mask formulas for targets 21 and 22 at
  `rdna3/README.md:5290` through `:5317`.
- Sections 14.2 and 14.3 require vertex position output, define the two-phase
  export/`EXPCNT` dependency model, describe same-type export ordering versus
  cross-type out-of-order completion, and say `STATUS.SKIP_EXPORT` makes export
  instructions act as NOPs at `rdna3/README.md:5319` through `:5333`.
- Section 15.10.1 gives the EXP field positions and target ranges at
  `rdna3/README.md:7249` through `:7271`.
- Chapter 16.19 summarizes `EXPORT` and says every pixel shader must export to
  a color, depth, or `NULL` target with the `VM` bit set to communicate the
  pixel-valid mask; only one pixel export type may set `DONE`; vertex shaders
  must have one or more position exports and at least one parameter export; and
  the final position export must set `DONE` at `rdna3/README.md:25211` through
  `:25213`.

XML evidence:

- XML `ENC_EXP` carries raw fields for `DONE`, `EN`, `ROW_EN`, `TGT`, and
  `VSRC0` through `VSRC3` at `amdgpu_isa_rdna3.xml:4733` through `:4865`.
- XML `ENC_EXP` prose says `EXEC` carries the valid mask and shows disabled
  channels as `off`, but the field map has no named `VM` bit between `DONE`
  bit 11 and `ROW_EN` bit 13 at `amdgpu_isa_rdna3.xml:4753` through `:4815`;
  the only nearby XML `VM` field is the wait-count field at
  `amdgpu_isa_rdna3.xml:187596` through `:187604`.
- XML `OPR_TGT` enumerates MRT, Z, position, primitive, and dual-source blend
  targets but no `NULL` export target at `amdgpu_isa_rdna3.xml:186118`
  through `:186207`, and the `EXP` instruction entry exposes `TGT`, four VGPR
  sources, implicit `EXEC`, and implicit `M0` at
  `amdgpu_isa_rdna3.xml:24689` through `:24743`.
- XML also exposes `S_WAITCNT_EXPCNT` and generic wait-counter records, but the
  export records do not structure the two-phase `EXPCNT` producer/completion
  model, same-type ordering, `SKIP_EXPORT`, target-once validation, `VM`
  valid-mask bit requirement, last-`DONE` requirements, primitive `DONE` rule,
  sample-mask/depth/MRT0 ordering, dual-source blend adjacency or lane-mask
  transforms, or graphics-stage output obligations.
- XML's `ENC_EXP.EN` description says `EN` must not be zero at
  `amdgpu_isa_rdna3.xml:4783` through `:4790`, while Chapter 14 lists `0x0` as
  a valid 16-bit component-enable value at `rdna3/README.md:5273`.

Impact:

XML has enough data to decode the base `EXP` opcode and its ordinary
MRT/Z/POS/PRIM targets, but a validator, disassembler, or executor cannot
reconstruct RDNA3 export valid-mask signalling, `NULL` target naming, ordering,
liveness, dependency, dual-source blend, or graphics-pipeline legality rules
from XML alone. The `EN=0` mismatch also means XML-only legality can reject a
value the manual explicitly lists for 16-bit exports.

### RDNA3-XML-066: Split SOPK wait-count threshold expressions are prose-only

Manual evidence:

- Chapter 16.2 defines `S_WAITCNT_VSCNT`, `S_WAITCNT_VMCNT`, and
  `S_WAITCNT_LGKMCNT` thresholds as `S0.u[5:0] + S1.u[5:0]`, and says the
  six-bit comparison applies with no clamping on add overflow, at
  `rdna3/README.md:8092` through `:8124` and `:8145` through `:8160`.
- The same section defines `S_WAITCNT_EXPCNT` as
  `S0.u[2:0] + S1.u[2:0]`, with a three-bit comparison and no clamping, at
  `rdna3/README.md:8128` through `:8142`.
- The manual also says literal-only waits use `NULL` for the GPR argument, and
  that these split wait opcodes may only appear inside a clause when the SGPR
  operand is `NULL`, at `rdna3/README.md:8105` through `:8107`, `:8122`
  through `:8124`, `:8139` through `:8141`, and `:8158` through `:8160`.

XML evidence:

- XML exposes the four split wait opcodes with `OPR_SDST_NULL` and
  `OPR_SIMM16` operands at `amdgpu_isa_rdna3.xml:56156` through `:56295`.
- The XML descriptions state which counter each opcode waits on, but do not
  encode the low-bit `SDST + SIMM16` threshold expressions, the no-clamp add
  overflow behavior, or the instruction-specific clause restriction.

Impact:

XML consumers can identify the split wait instructions and their two operand
fields, but cannot reconstruct the actual wait threshold for non-`NULL` SGPR
operands or validate the clause-only-when-`NULL` rule from XML alone.

### RDNA3-XML-067: DPP execution semantics are only partially structured

Manual evidence:

- Chapter 7.7 says DPP is limited to selected VOP1/VOP2/VOPC/VOP3/VOP3P
  instructions, excluding packed math, VOPD, VINTERP, 64-bit opcodes, and
  several special opcodes, at `rdna3/README.md:2741` through `:2762`.
- The same section says `V_CMP`/`V_CMPX` write a full mask under DPP, `FI=1`
  does not re-enable inactive `V_CMPX` lanes, and VOPC row/bank-disabled lanes
  receive zero result bits at `rdna3/README.md:2764` and `:2779` through
  `:2788`.
- DPP16 `BC`/`FI` behavior distinguishes out-of-range lanes from in-range but
  inactive lanes, and `BC=0/FI=0` disables the destination write for both
  invalid cases, at `rdna3/README.md:2791` through `:2815`.
- The manual enumerates the DPP16 `DPP_CTRL` control-word ranges and says DPP8
  follows DPP16 `BC=1` while assuming all source lanes are in range at
  `rdna3/README.md:2794` through `:2803` and `:2817` through `:2835`.

XML evidence:

- XML exposes DPP8/DPP16 encoding records, conditions, and field maps. For
  example, VOP1 DPP8 is selected by source selectors `233`/`234` at
  `amdgpu_isa_rdna3.xml:5171` through `:5265`, VOP1 DPP16 carries
  `BANK_MASK`, `BOUND_CTRL`, `DPP_CTRL`, `FI`, and `ROW_MASK` at
  `amdgpu_isa_rdna3.xml:5402` through `:5618`, and VOPC DPP16 records the same
  fields at `amdgpu_isa_rdna3.xml:11776` through `:11845`.
- Those field descriptions capture broad row/bank enable masks, invalid shared
  data behavior, and fetch-inactive behavior, but `DPP_CTRL` remains only a
  generic field description, DPP8's `BC=1`/all-in-range contract is not
  represented, and the VOPC-specific zero-result behavior for row/bank masks,
  compare full-mask writes, and `CMPX` fetch-inactive interaction are not
  structured in the XML.

Impact:

XML consumers can decode DPP extension words and recover much of instruction
availability from per-instruction encoding records, but a validator or executor
still needs the manual prose to implement DPP16 lane-write behavior, DPP8
inactive-lane semantics, compare-mask zeroing, and `DPP_CTRL` control meanings.

### RDNA3-XML-068: Swap-family operands omit read dependencies

Manual evidence:

- Chapter 7.8 defines `V_SWAPREL_B32` as a true swap: it reads
  `VGPR[src + M0[9:0]]` into a temporary, writes the source-indexed VGPR from
  `VGPR[dst + M0[25:16]]`, then writes the destination-indexed VGPR from the
  temporary at `rdna3/README.md:2851` through `:2854`.
- The Chapter 7.3 inventory also lists `V_SWAP_B16`, `V_SWAP_B32`, and
  `V_SWAPREL_B32` as VALU operations at `rdna3/README.md:2530` through
  `:2532`, and the XML instruction descriptions call the non-relative and
  relative forms swaps.

XML evidence:

- `V_SWAP_B32` is described as swapping two vector-register values, but both
  explicit operands are marked `Input="false" Output="true"` at
  `amdgpu_isa_rdna3.xml:69775` through `:69795`.
- `V_SWAPREL_B32` is likewise described as swapping two relatively indexed
  vector-register values, but its `VDST` and `SRC0` operands are both marked
  output-only; only the implicit `M0` operand is marked as input at
  `amdgpu_isa_rdna3.xml:69889` through `:69913`.

Impact:

XML-only def-use consumers can see the writebacks for swap instructions, but
miss the read dependencies on the old source and destination VGPR values. That
can make liveness, scheduling, or data-dependency validation treat swaps as
pure writes even though their result depends on both prior register contents.

### RDNA3-XML-069: WMMA modifier, rounding, layout, and scheduling semantics are only partially structured

Manual evidence:

- Chapter 7.9 says WMMA uses VOP3P encoding, performs matrix `A * B + C => D`,
  and lists the six RDNA3 WMMA opcodes at `rdna3/README.md:2860` through
  `:2883`.
- Chapter 16.10's WMMA definition rows repeat those six VOP3P opcodes and show
  each operation saving `EXEC`, forcing `EXEC` to all ones, evaluating the
  matrix operation, and restoring `EXEC` at `rdna3/README.md:15485` through
  `:15550`.
- Chapter 7.9 also says WMMA does not generate ALU exceptions, float WMMA uses
  only round-to-nearest-even, inline constants can only be used for the C
  matrix with F16/BF16 values replicated into both DWORD halves, and dependent
  back-to-back WMMAs require one `V_NOP` or independent VALU when the first D
  overlaps the second A/B, at `rdna3/README.md:2870` and `:2891` through
  `:2895`.
- Chapter 7.9 repurposes `NEG`/`NEG_HI`: integer IU8/IU4 WMMA uses `NEG[1:0]`
  for A/B signedness with other modifier bits required zero, while F16/BF16
  WMMA uses low/high A/B negation bits and `{NEG_HI[2], NEG[2]}` as C
  `{ABS, NEG}`, at `rdna3/README.md:2872`.
- The manual describes required A/B lane replication and points to external
  matrix-calculator mappings for detailed element-to-register layout at
  `rdna3/README.md:2868` and `:2887` through `:2905`.

XML evidence:

- XML carries the six WMMA instruction entries as `ENC_VOP3P` with explicit
  256-bit D operands, VGPR-only A/B operands, and `OPR_SRC_VGPR_OR_INLINE` C
  operands at `amdgpu_isa_rdna3.xml:119685` through `:119976`.
- XML's WMMA data-format records describe the broad A/B packing, OPSEL ignore
  behavior for A/B, wave32/wave64 replication count, D/C per-lane element
  storage, D/C OPSEL behavior for 16-bit data, and wave-size-dependent D/C VGPR
  counts at `amdgpu_isa_rdna3.xml:171717` through `:173773`.
- Those records do not encode the instruction-specific `NEG`/`NEG_HI`
  overloads and zero requirements, RNE-only and no-ALU-exception behavior,
  full-`EXEC` execution override, C-only inline replication details, the
  dependent-WMMA adjacency hazard, or the full element-to-register mapping
  needed without the manual/external calculator.

Impact:

XML-only consumers can enumerate and broadly type the RDNA3 WMMA instructions,
but cannot implement or validate their complete modifier, rounding, exception,
full-`EXEC`, constant, layout, and instruction-stream legality contracts from
XML alone.

## No-Gap Notes

- Chapter 15.1 scalar/control field narrow match: XML carries the SOP1, SOPC,
  SOPP, SOPK, and SOP2 32-bit microcode records with the expected `SSRC*`,
  `SDST`, `SIMM16`, `OP`, and `ENCODING` bit positions at
  `amdgpu_isa_rdna3.xml:16` through `:138`, `:141` through `:252`, `:255`
  through `:340`, `:344` through `:424`, and `:428` through `:583`. The
  scalar-selector, literal, wait, and side-effect semantics remain covered by
  earlier Chapter 5/6 gaps rather than a new Chapter 15 field gap.
- Chapter 16.1 SOP2 opcode-shell narrow match: XML has entries for the audited
  SOP2 arithmetic, shift, bitfield, min/max, multiply, conditional-select, and
  pack definitions with the expected explicit operands plus implicit SCC where
  applicable. The remaining gaps are the signed-overflow wording issue in
  `RDNA3-XML-033` and the broader scalar-selector/out-of-range rules already
  tracked in Chapter 3/4/6 rows.
- Chapter 16.2 SOPK opcode-shell boundary: XML records the audited `S_VERSION`,
  conditional move/compare, add/multiply, HWREG, call, and split wait-count
  opcode shells. The split wait-count gap is limited to the dynamic threshold
  expression and clause restriction, not to the existence of the instructions
  or their raw operand fields.
- Chapter 16.3 SOP1 opcode-shell boundary: XML records the audited scalar move,
  bit-manipulation/count, EXEC save/write, relative-move, direct-PC, trap-return,
  and return-message instruction shells. The missing pieces remain the
  structured `M0` relative-index contract, direct-PC/trap state, and message
  return protocol already tracked under Chapter 3/5 gaps.
- Chapter 16.4 SOPC opcode-shell narrow match: XML records the signed/unsigned
  compare and bit-compare instructions with explicit scalar source operands and
  implicit SCC output. Existing SCC and scalar-selector gaps cover the semantic
  boundaries outside these concrete opcode shells.
- Chapter 16.5 SOPP opcode-shell boundary: XML records the audited SOPP control,
  branch, wait, end-program, message, cache, and barrier instruction shells.
  Existing gaps cover side-effect structure, clause membership, wait-counter
  semantics, and STATUS/debug predicate metadata rather than raw opcode
  presence.
- Chapter 16.10 VOP3P packed/DOT opcode-shell narrow match: XML records the
  audited packed rows 0-18 and DOT rows 19/22-26 with matching opcode numbers
  and broad packed data formats. Representative entries include
  `V_PK_MAD_I16` at `amdgpu_isa_rdna3.xml:114261` through `:114293`,
  `V_PK_LSHLREV_B16` at `:114885` through `:114911`,
  `V_DOT4_I32_IU8` at `:117295` through `:117330`, and
  `V_DOT2_F32_BF16` at `:118369` through `:118399`. Remaining packed/DOT
  semantics are covered by `RDNA3-XML-001`, `RDNA3-XML-002`,
  `RDNA3-XML-036`, and `RDNA3-XML-040` rather than missing opcode rows.
- Chapter 16.7-16.8 VOP2/VOP1 opcode-shell boundary: XML records the audited
  compact VALU opcodes, VOP3/VOP3SD aliases, literal-only FMA forms, `V_PIPEFLUSH`,
  relative-indexed moves, swap/permlane forms, and concrete operand widths. The
  remaining differences are semantic details already tracked under Chapter 7
  gaps plus the runtime-side `V_PIPEFLUSH` issue in `RDNA3-RJ-091`.
- Chapter 16.9 VOPC opcode-shell boundary: XML records the compact and VOP3
  compare opcode inventory, result-mask operand classes on VOP3 aliases, literal
  forms, and concrete source widths for the audited compare/class definitions.
  The compact rocjitsu metadata issue is recorded in `RDNA3-RJ-092`; the XML
  surface itself is not missing the affected compare instruction records.
- Chapter 16.12 VOP3/VOP3SD opcode-shell boundary: XML records the audited
  VOP3/VOP3SD instruction inventory, the VOP3SD scalar-destination forms,
  VOP3 compare/CMPX result operand classes, literal and DPP alternative
  encodings, and concrete operand widths. Existing XML gaps cover modifier,
  OPSEL, source-combination, and VOP3SD semantic metadata; the compare-DPP
  result-mask issue is runtime-side and recorded in `RDNA3-RJ-093`.
- Chapter 16.13 VINTERP opcode-shell boundary: XML records all six VINTERP
  opcode shells with interpolation descriptions and operand-width metadata at
  `amdgpu_isa_rdna3.xml:57542` through `:57810`. The Chapter 16.13 formulas,
  fixed DPP8 source-lane selections, and OPSEL/RTZ notes are largely prose
  details layered on those opcode shells and overlap with existing Chapter 7/12
  VINTERP notes rather than missing instruction records.
- Chapter 15.3.1-15.3.5 VALU field narrow match: XML carries compact VOP1,
  VOPC, VOP2, VOP3, and VOP3SD records with the expected raw field positions at
  `amdgpu_isa_rdna3.xml:702` through `:865`, `:868` through `:1135`, `:1138`
  through `:1267`, `:1270` through `:1895`, and `:8485` through `:8681`.
  Existing gaps already track the semantic pieces that are not reducible to
  these fields, including generic modifier/`OPSEL` legality, source-combination
  rules, compact true16 addressing, and VOP3SD field-description drift.
- Chapter 15.3.8-15.3.9 DPP field narrow match: XML has separate DPP8 and
  DPP16 extension records for VOP1, VOP3, VOP2, VOP3SD, VOP3P, and VOPC,
  beginning at `amdgpu_isa_rdna3.xml:5171`, `:5402`, `:6658`, `:7263`,
  `:8042`, `:8260`, `:9279`, `:9561`, `:10472`, `:10761`, `:11332`, and
  `:11624`. The DPP execution and lane-mask semantics remain covered by
  `RDNA3-XML-067`.
- Chapter 7.7 sampled DPP availability narrow match: XML per-instruction
  encodings match representative entries from the manual's DPP support table:
  packed `V_PK_ADD_F16` exposes only base/literal VOP3P encodings at
  `amdgpu_isa_rdna3.xml:116490` through `:116593`; `V_DOT4_I32_IU8` likewise
  has base/literal encodings but no DPP forms at `amdgpu_isa_rdna3.xml:117295`
  through `:117549`; allowed `V_DOT2_F32_F16` and `V_FMA_MIX_F32` expose
  `VOP3P_VOP_DPP8` and `VOP3P_VOP_DPP16` at
  `amdgpu_isa_rdna3.xml:117218` through `:117249` and
  `:118950` through `:118981`.
- Chapter 15.4 VINTERP field narrow match: XML `ENC_VINTERP` records the 64-bit
  field layout, `WAIT_EXP`, `OP_SEL`, `CLAMP`, source fields, and
  no-DPP/no-literal description at `amdgpu_isa_rdna3.xml:2127` through `:2250`,
  matching the Chapter 15.4 field table. Interpolation arithmetic, packed
  parameter, and true16 half-selection semantics remain covered by the
  Chapter 12/7 gaps.
- Chapter 14 / 15.10 / 16.19 export opcode-shell narrow match: XML `ENC_EXP`,
  `OPR_TGT`, and the `EXP` instruction entry carry the singleton export opcode,
  the documented base field positions, named MRT/Z/POS/PRIM/dual-source target
  values, VGPR sources, and implicit `EXEC`/`M0` operands. `RDNA3-XML-065` is
  about missing export behavior, `VM`/`NULL` target metadata, and a narrow
  `EN=0` legality conflict, not absence of the export encoding.
- Chapter 13.4 GWS/ordered-count opcode-shell narrow match: XML contains the
  audited GWS and `DS_ORDERED_COUNT` opcode entries with broad operand metadata
  and one-line descriptions. `RDNA3-XML-064` is about the manual-only
  synchronization, lane, resource, and replay contracts layered on top of those
  opcode shells.
- Chapter 13 float-atomic opcode-shell narrow match: XML lists the audited LDS
  F32/F64, FLAT/GLOBAL/SCRATCH F32, and buffer F32 float-atomic instruction
  shells with broad operand sizes, memory operands, aliases, and functional
  groups. `RDNA3-XML-063` is about missing numeric edge semantics, not absence
  of those opcodes.
- Chapter 12 / 15.5 LDSDIR field narrow match: XML `ENC_LDSDIR` carries the
  expected raw `VDST`, `ATTR_CHAN`, `ATTR`, `WAIT_VDST`, `OP`, and encoding
  fields at `amdgpu_isa_rdna3.xml:2253` through `:2328`. `RDNA3-XML-061` is
  about missing LDSDIR execution, counter, and whole-quad semantics, not absence
  of the microcode field map.
- Chapter 12 / 15.6 DS field and opcode-shell narrow match: XML `ENC_DS` has
  the expected raw DS fields and the audited opcode entries for ordinary
  load/store/atomic, ADDTID, permute, GDS/GWS, GS-register, BVH, B96/B128, and
  D16 forms. `RDNA3-XML-062` records the missing address/permute/GDS/streamout
  behavioral rules, while `RDNA3-XML-004`, `RDNA3-XML-016`, and
  `RDNA3-XML-018` continue to cover BVH stack state, LDS allocation/alignment,
  and the broader `M0` use table.
- Chapter 11 / 15.9 FLAT field narrow match: XML `ENC_FLAT`,
  `ENC_FLAT_GLOBAL`, and `ENC_FLAT_SCRATCH` carry the expected raw field
  positions for `ADDR`, `DATA`, cache bits, `OFFSET`, `OP`, `SADDR`, `SEG`,
  `SVE`, and `VDST` at `amdgpu_isa_rdna3.xml:3422` through `:3969`. The gaps
  above are about SEG-specific interpretation and execution rules, not absence of
  the field map.
- Chapter 16.20 FLAT/GLOBAL/SCRATCH opcode-shell narrow match: XML lists the
  audited load, store, D16, ADDTID, and atomic instruction entries with opcode
  numbers, aliases, broad operand sizes, implicit memory operands, and functional
  group metadata. Notably `GLOBAL_ATOMIC_CSUB_U32` is present with a clamp
  description at `amdgpu_isa_rdna3.xml:24041` through `:24086`; missing
  executable wait/error and special-legality semantics remain tracked under
  `RDNA3-XML-059` and `RDNA3-XML-060`.
- Chapter 10 / 15.8.1 MIMG field narrow match: XML `ENC_MIMG` and `MIMG_NSA1`
  carry the expected raw MIMG field positions for `NSA`, `DIM`, `UNORM`, `DMASK`,
  cache flags, `R128`, `A16`, `D16`, `OP`, `VADDR`, `VDATA`, `SRSRC`, `TFE`,
  `LWE`, `SSAMP`, and NSA `VADDRA` through `VADDRD` at
  `amdgpu_isa_rdna3.xml:2978` through `:3295` and `:3971` through `:4333`.
- Chapter 10 DMASK/UNORM field-text narrow match: the XML MIMG field descriptions
  preserve unusually detailed prose for component routing, D16 write packing,
  `DMASK==0`, gather4 output counts, and normalized-versus-unnormalized sampler
  addressing at `amdgpu_isa_rdna3.xml:3134` through `:3149` and `:3265` through
  `:3279`. The gaps above are the remaining resource, status, address-table, and
  conversion semantics.
- Chapter 16.18 MIMG opcode-shell narrow match: XML lists the audited image
  load/store/atomic/query/MSAA/BVH/sample/gather instruction entries with opcode
  numbers, functional groups, operand directions, descriptor formats, and broad
  operand sizes. `RDNA3-XML-054` through `RDNA3-XML-058` are about missing
  structured semantics, not missing MIMG opcode shells.
- Chapter 10 BVH operand-shell narrow match: XML distinguishes the 32-bit and
  64-bit BVH forms with different `VADDR` widths and uses the separate
  `FMT_IMG_BVH` descriptor format, matching the broad manual distinction before
  the restrictions recorded under `RDNA3-XML-058`.
- Chapter 9 / 15.7 buffer field narrow match: XML `ENC_MTBUF` and `ENC_MUBUF`
  carry the same basic instruction bit positions for `OFFSET`, cache flags,
  opcode, `FORMAT` for MTBUF, `VADDR`, `VDATA`, `SRSRC`, `TFE`, `OFFEN`,
  `IDXEN`, and `SOFFSET` as the manual field tables at `rdna3/README.md:6900`
  through `:6977`.
- Chapter 9 SOFFSET selector-shell narrow match: XML uses
  `OPR_SREG_M0_INL` for buffer `SOFFSET`, and that operand type enumerates
  SGPRs, `NULL`, `M0`, and inline constants at
  `amdgpu_isa_rdna3.xml:182080` and nearby predefined values. The gaps above
  are descriptor, address, and data semantics layered on top of that selector.
- Chapter 16.16/16.17 opcode-shell narrow match: XML lists the audited MUBUF
  and MTBUF instruction definitions with opcode numbers, aliases, functional
  groups, operand directions, and operand sizes. `RDNA3-XML-051` through
  `RDNA3-XML-053` are about missing structured semantics, not missing opcode
  shells.
- Chapter 8 / 15.2.1 SMEM field narrow match: XML `ENC_SMEM` exposes the
  expected 64-bit scalar-memory format with `SBASE`, `SDATA`, `DLC`, `GLC`,
  `OP`, `OFFSET`, and `SOFFSET` fields at `amdgpu_isa_rdna3.xml:586` through
  `:699`. The generated rocjitsu bitfield struct also matches the XML bit
  positions, so the markdown manual's odd `DLC`/`GLC` table text was not
  treated as a confirmed XML gap.
- Chapter 8 / 16.6 SMEM opcode narrow match: XML has all audited RDNA3 scalar
  load, scalar buffer load, `S_GL1_INV`, and `S_DCACHE_INV` opcode entries at
  `amdgpu_isa_rdna3.xml:41064` through `:41625`, with the expected load widths
  and operandless invalidate forms.
- Chapter 8 SMEM offset narrow match: XML records `SOFFSET` as an SGPR/M0/NULL
  scalar-memory offset operand through `OPR_SMEM_OFFSET`, and the `OFFSET`
  field description includes the signed-byte-offset and negative-buffer-offset
  `MEMVIOL` rule at `amdgpu_isa_rdna3.xml:645` through `:690` and
  `amdgpu_isa_rdna3.xml:175329` through `:175335`. The remaining scalar-buffer
  descriptor and range formula is tracked under `RDNA3-XML-048`.
- Chapter 7.6 / 15.3.7 VOPD field narrow match: XML has 64-bit and 96-bit VOPD
  microcode records with the expected raw `OPX`, `OPY`, `SRCX0`, `SRCY0`,
  `VSRCX1`, `VSRCY1`, `VDSTX`, `VDSTY`, and `LITERAL` fields, and the `VDSTY`
  description preserves the manual's reconstructed low-bit parity rule.
  `RDNA3-XML-044` is limited to the missing `SRCY0 == 255` literal-length
  condition.
- Chapter 7.6 / 15.3.7 VOPD opcode-inventory narrow match: the XML VOPD
  encoding identifiers and instruction entries cover the audited OPX opcode
  values `0..13` and OPY opcode values `0..13,16..18`, including the Y-only
  integer/bitwise slots.
- Chapter 7.6 CNDMASK VCC narrow match: XML models the VOPD
  `V_DUAL_CNDMASK_B32` implicit VCC input with `OPR_VCC` operands on both X
  and Y entries at `amdgpu_isa_rdna3.xml:169094` through `:169217`; the missing
  pieces are the pair-level source-count and legality rules tracked under
  `RDNA3-XML-045`.
- Chapter 7.6 MOV ignored-source narrow match: XML `V_DUAL_MOV_B32` entries
  expose only the destination and `SRC0`/literal operands and omit `VSRC1`,
  matching the manual's statement that `vsrc1X`/`vsrc1Y` are ignored for MOV at
  `amdgpu_isa_rdna3.xml:168999` through `:169075`.
- Chapter 7.4 OPSEL narrow match: XML carries generic VOP3 and VINTERP
  `OP_SEL` text for true16 source selection and destination-half preservation,
  and VOP3P carries distinct low/high operation selector fields. `RDNA3-XML-043`
  is about the missing compact VOP1/VOP2/VOPC bit-7 namespace split and
  addressability distinction, not total absence of OPSEL metadata.
- Chapter 3 state hook narrow match: XML has `OPR_PC`, `OPR_HWREG`,
  `OPR_WAITCNT`, `OPR_SDST_EXEC`, `OPR_VCC`, and `FMT_NUM_M64` records, plus
  TTMP, `VCC_LO`/`VCC_HI`, and `EXEC_LO`/`EXEC_HI` selector names in scalar
  operand classes. The gaps above are about missing cross-state semantics and
  policy, not total absence of these names.
- Chapter 3 PC/branch instruction-shell narrow match: XML has the direct-PC
  instruction entries and marks the branch forms with `OPR_LABEL`, `OPR_PC`,
  and branch flags. `RDNA3-XML-011` is about the manual-only alignment,
  sign-extension, debug, and trap-save details.
- Chapter 3 mask operand narrow match: XML's `FMT_NUM_M64` records the low-dword
  storage order and wave32 truncation, and `S_CBRANCH_EXECZ`/`VCCZ` entries use
  implicit mask operands. `RDNA3-XML-012` is about the absent zero-`EXEC`
  skip/issue policy, not the existence of mask operands.
- Chapter 3.3.1 VCC operand narrow match: representative XML entries expose VCC
  operands on `V_CNDMASK_B32`, `V_ADD_CO_CI_U32`, `V_SUB_CO_CI_U32`,
  `V_DIV_FMAS_F32`, and `V_DIV_FMAS_F64` at
  `amdgpu_isa_rdna3.xml:70445` through `:70505`, `:76985` through `:77020`,
  `:77366` through `:77401`, `:92366` through `:92402`, and `:92673` through
  `:92709`. The SGPR gap above is about selector-region behavior, not total
  absence of VCC operand metadata.
- Chapter 3.3.2 VGPR operand narrow match: XML has separate ordinary source and
  destination VGPR operand classes, carries the `MSG_DEALLOC_VGPRS` predefined
  value, and exposes `V_MOVREL*`/`V_SWAPREL` explicit operands. `RDNA3-XML-014`
  is about the missing cross-instruction allocation and out-of-range semantics,
  not total absence of VGPR names or these opcode shells.
- Chapter 3.3.3 memory operand narrow match: XML has the DS, MUBUF/MTBUF,
  MIMG, FLAT, DSMEM, and GPUMEM operand/field shells needed to decode memory
  instructions, including `DMASK`, `TFE`, formatted buffer opcodes, and
  destination/source operand widths. `RDNA3-XML-015` is about missing
  cross-instruction range, nullification, and alignment semantics, not total
  absence of memory instruction metadata.
- Chapter 3.3.4 LDS instruction-shell narrow match: XML has the `ENC_DS` field
  map, `GDS` field, `OPR_DSMEM`, `HW_REG_LDS_ALLOC`, representative DS
  load/store opcode shells, and LDSDIR parameter/direct load entries.
  `RDNA3-XML-016` is about missing LDS allocation, side-placement,
  alignment/OOR, and violation semantics, not total absence of LDS instruction
  metadata.
- Chapter 3.4.1/3.4.2 HWREG narrow match: XML has the HWREG selector names and
  the `S_GETREG_B32`/`S_SETREG_B32`/`S_SETREG_IMM32_B32` instruction shells.
  `RDNA3-XML-017` is about the missing STATUS/MODE field layout, access
  permissions, initialization, and behavioral effects.
- Chapter 3.4.3-3.4.6 scalar-condition/operand narrow match: XML records
  `NULL` read-zero/write-discard behavior, `M0` and `VCC_LO`/`VCC_HI`
  selectors, `OPR_SSRC_SPECIAL_SCC`, `OPR_VCC`, and VOPC's `EXEC`-masked
  compare result rule at `amdgpu_isa_rdna3.xml:1091`,
  `:175123` through `:175140`, `:175182` through `:175188`,
  `:185288` through `:185296`, and `:185390` through `:185398`.
  `RDNA3-XML-018` is about the missing complete M0 use table, not absence of
  these scalar and condition-code operands.
- Chapter 3.4.7-3.4.8 FLAT_SCRATCH/HWREG selector narrow match: XML records
  the `OPR_FLAT_SCRATCH` implied operand, scratch instruction implicit
  operands, `OPR_HWREG` selector encoding, and the relevant HWREG selector
  names. `RDNA3-XML-019` is about the missing launch, field-layout, access,
  and side-effect behavior, not absence of the operand or selector shells.
- Chapter 3.4.9 trap instruction-shell narrow match: XML records the `S_TRAP`
  and `S_RFE_B64` instruction entries, `TRAP` functional group, `HW_REG_TRAPSTS`
  selector name, and TMA/TBA return-message IDs. `RDNA3-XML-020` is about the
  missing trap-state machine, TTMP payload, exception-enable, and `TRAPSTS`
  field semantics, not absence of the opcodes or selector names.
- Chapter 3.4.10 time selector narrow match: XML records `HW_REG_SHADER_CYCLES`,
  `MSG_RTN_GET_REALTIME`, `S_GETREG_B32`, and `S_SENDMSG_RTN_B64`.
  `RDNA3-XML-021` is about the missing clock-domain, width, synchronization,
  latency, and LGKM wait-counter semantics.
- Chapter 3.5.1-3.5.2 register-name narrow match: XML records the `OPR_EXEC`
  and `OPR_FLAT_SCRATCH` operand/register shells, and flat/scratch instruction
  entries reference the scratch operand where needed. `RDNA3-XML-022` is about
  the missing launch-time initialization contract, not absence of those
  architectural names.
- Chapter 3.5.3 scalar-register narrow match: XML records ordinary scalar
  source/destination selector classes and alignment metadata for instruction
  operands. `RDNA3-XML-023` is about the missing shader-launch preload tables,
  not the ability to name SGPR operands in instructions.
- Chapter 3.5.4 VGPR operand narrow match: XML records ordinary VGPR selector
  classes for instruction operands. `RDNA3-XML-024` is about launch-time VGPR
  initialization and pixel input routing, not absence of VGPR operand names.
- Chapter 3.5.5 LDSDIR instruction-shell narrow match: XML records
  `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, implicit DSMEM operands, and the M0
  dependency for LDSDIR forms. `RDNA3-XML-025` is about prelaunch PS LDS
  contents and vertex-parameter layout, not absence of LDSDIR opcodes.
- Chapter 4 instruction-group narrow match: XML functional-group metadata names
  the broad instruction roles, including scalar ALU/memory, vector ALU/memory,
  DS/LDS, export, branch, message, wave-control, and trap categories. The gaps
  above are about common-field materialization and cache-policy semantics, not
  absence of all instruction grouping metadata.
- Chapter 4.1 source-selector narrow match: XML carries the basic scalar and
  vector source selector map, including SGPRs, VCC, TTMP, `NULL`, `M0`, `EXEC`,
  integer inline constants, float inline constants, `SCC`, aperture selectors,
  literal constants, and VGPR selectors. The gaps above are about
  source-size-dependent inline materialization, cache policy, and
  unused-field canonicalization.
- Chapter 4.1 literal-source narrow match: XML `SRC_LITERAL` descriptions do
  include 16-bit low-half use, signed/unsigned 64-bit extension, and F64
  high-half placement at `amdgpu_isa_rdna3.xml:178403` through `:178410` and
  `amdgpu_isa_rdna3.xml:184309` through `:184316`. This is more complete than
  the ordinary inline-constant records audited in `RDNA3-XML-026`.
- Chapter 5 opcode-shell narrow match: XML records all main Chapter 5 RDNA3
  control, branch, wait, clause, message, barrier, and direct-PC opcode shells
  inspected in this slice. The Chapter 5 XML gaps above are about missing
  side-effect, legality, scheduling, and stream-state semantics, not opcode
  absence.
- Chapter 5 branch narrow match: XML records branch flags plus `OPR_LABEL`,
  implicit `SCC`, `VCC`, and `EXEC` operands for representative branch forms at
  `amdgpu_isa_rdna3.xml:56777` through `:57149`, and records direct-PC SOP1
  operands for `S_GETPC_B64`, `S_SETPC_B64`, and `S_SWAPPC_B64` at
  `amdgpu_isa_rdna3.xml:45879` through `:45983`.
- Chapter 5 clause/delay field narrow match: XML `OPR_CLAUSE` carries the
  `BREAK_SPAN` and `LENGTH` bit layouts, and XML `OPR_DELAY` carries the raw
  `INSTID0`, `INSTID1`, and `INSTSKIP` bit layouts and enumerated values. The
  gaps are the cross-instruction legality and stream-history rules.
- Chapter 6.1/6.2 narrow match: XML records the SOP1/SOP2/SOPK/SOPC/SOPP
  format fields needed for scalar decode, and the concrete
  `S_SETREG_IMM32_B32` entry carries both the `OPR_HWREG` selector and
  `LITERAL` source at `amdgpu_isa_rdna3.xml:56070` through `:56090`, matching
  the manual's single SOPK literal exception.
- Chapter 6.2 selector narrow match: XML scalar source classes enumerate SGPRs,
  VCC, TTMP, `NULL`, `M0`, `EXEC`, integer inline constants, float inline
  constants, `SCC`, literal constants, and aperture selectors. The gaps above
  are in source-value materialization and aperture formulas, not absence of the
  selector codes.
- Chapter 6.3-6.7 SCC metadata narrow match: XML exposes SCC as implicit input
  or output on representative arithmetic, conditional, compare, WREXEC, and
  bit-compare entries, for example `S_ADD_I32` at
  `amdgpu_isa_rdna3.xml:46399` through `:46430`, `S_CMOVK_I32` at
  `:55354` through `:55382`, `S_BITCMP0_B32` at `:54594` through `:54623`, and
  `S_AND_NOT0_WREXEC_B32` at `:45256` through `:45288`. The signed-SCC gap
  above is limited to the prose description on the signed add/sub rows.
- Chapter 6.7 WREXEC destination narrow match: XML uses `OPR_SREG` for the
  explicit WREXEC destination at `amdgpu_isa_rdna3.xml:45270`, and `OPR_SREG`
  includes SGPRs, TTMPs, VCC, and `NULL` but not `M0` or `EXEC` at
  `amdgpu_isa_rdna3.xml:181440` through `:181475`; the manual's "D cannot be
  EXEC" restriction is therefore structurally recoverable from XML.
- Chapter 6.8 access-instruction narrow match: XML has `S_GETREG_B32`,
  `S_SETREG_B32`, and `S_SETREG_IMM32_B32` with `OPR_HWREG` operands and the
  literal source for `S_SETREG_IMM32_B32`; the missing HWREG bitfields and side
  effects remain tracked under `RDNA3-XML-017` and `RDNA3-XML-019`.
- Chapter 7.1 encoding-shell narrow match: XML has core VOP1, VOP2, VOPC,
  VOP3, VOP3P, VOP3SD, and VOPD microcode records with the expected raw field
  positions. The Chapter 7 gaps above are about missing legality, semantic
  policy, and one misleading VOP3SD field description, not complete absence of
  VALU encodings.
- Chapter 7.2 operand-shell narrow match: XML records the representative VALU
  operand classes for VOP3 sources, VOP3SD carry destinations, readlane,
  writelane, and lane-select operands. The gaps above are about cross-operand
  constraints and lane/mask semantics that are not recoverable from those
  operand shells alone.
- Chapter 7.2 output/mode edge boundary: XML carries generic VOP3 `CLAMP` and
  `OMOD` field prose, VOPC/`V_CMPX` output operands, VOP3SD carry operands, and
  the `S_ROUND_MODE`/`S_DENORM_MODE` instruction shells. The Chapter 7.2 gaps
  are about missing structured opcode-policy and stream/wave64 hazard metadata;
  the detailed MODE bitfield and behavioral effects remain tracked under
  `RDNA3-XML-017`, and generic VALU out-of-range VGPR behavior remains under
  `RDNA3-XML-014`.
- Chapter 7.3 inventory narrow match: the XML contains every explicit
  non-VOP3P VALU mnemonic listed in the Chapter 7.3 inventory table after
  accounting for XML aliases such as `V_CVT_FLR_I32_F32`,
  `V_CVT_PKRTZ_F16_F32`, `V_CVT_PKNORM_*`, `V_FMAC_LEGACY_F32`, and
  `V_FMA_LEGACY_F32`.
- Chapter 7.3 classification narrow match: table classes are recoverable from
  concrete XML encoding names. VOP1 and VOP2 entries expose `ENC_VOP1` or
  `ENC_VOP2`, VOP3 entries expose `ENC_VOP3` or `VOP3_SDST_ENC`, and the
  always-literal `FMAMK`/`FMAAK` VOP2 forms expose `VOP2_INST_LITERAL`.
- Chapter 7.3 compare summary boundary: the compact compare-family rows are a
  high-level summary, while the detailed Chapter 15 opcode table and XML carry
  the concrete compare names. The XML follows the detailed table for integer
  `NE` predicates and for the absence of compact `F`/`T` predicates on I16/U16.
- Chapter 2 instruction-role narrow match: XML has top-level functional group
  descriptions for SALU, SMEM, VALU, VMEM, EXPORT, BRANCH, MESSAGE,
  WAVE_CONTROL, and TRAP at `amdgpu_isa_rdna3.xml:186868` through `:186905`,
  matching the broad processor-component roles listed in Chapter 2. The gaps
  are the cross-instruction execution, launch, wave-scheduling, and placement
  contracts recorded in `RDNA3-XML-006` through `RDNA3-XML-009`.
- Chapter 2 instruction-shell narrow match: XML has entries for
  `LDS_PARAM_LOAD`, `LDS_DIRECT_LOAD`, `S_SET_INST_PREFETCH_DISTANCE`,
  `S_CODE_END`, and `S_ENDPGM`. The gaps are the Chapter 2 relationships around
  WGP-mode legality, shader padding length, and launch/prefetch state, not the
  existence of those opcodes.
- Chapter 1 narrow match: XML preserves the RDNA3 architecture identity and
  basic floating data-format bit layouts that correspond to several suffix
  definitions. Examples include BF16 at `amdgpu_isa_rdna3.xml:170171`, F16 at
  `:170209`, F32 at `:170247`, and F64 at `:170285`. The gap is the
  surrounding architecture, topology, host-control, and memory-system prose
  recorded in `RDNA3-XML-005`, not those basic format records.
- Chapter 1 instruction-role narrow match: XML has top-level functional group
  descriptions for SALU, SMEM, VALU, VMEM, EXPORT, BRANCH, MESSAGE,
  WAVE_CONTROL, and TRAP at `amdgpu_isa_rdna3.xml:186868` through `:186905`.
  Those groups match broad instruction roles, but not the cross-instruction
  topology, launch, cache, GDS, or consistency contracts in Chapter 1.
- The RDNA3 XML has a distinct `OP_SEL_HI_2` field at
  `amdgpu_isa_rdna3.xml:3383`, matching the VOP3P split high-selector field in
  the manual at `rdna3/README.md:6500` and `:6544`.
- The XML carries explicit packed data formats and packed VOP3P entries for the
  audited ordinary packed-F16 forms, including `V_PK_FMA_F16` at
  `amdgpu_isa_rdna3.xml:118641`.
- `V_PK_FMAC_F16` is not missing from VOP3P: the XML lists it as an `ENC_VOP2`
  instruction at `amdgpu_isa_rdna3.xml:84100`, matching its accumulator-style
  packed-F16 contract.
- RDNA3 MIX descriptions in both manual and XML say fused multiply-add. This is
  different from the CDNA3/CDNA4 ambiguity where some detailed sources say
  multiply-add.
- The manual's DPP table says `V_PK_*` and WMMA do not support DPP, while
  `V_FMA_MIX_*` and `V_DOT2_F32_{BF16,F16}` do, at
  `rdna3/README.md:2736` through `:2762`. The inspected `V_PK_FMA_F16` XML
  entry exposes base and literal VOP3P encodings, not DPP encodings.
- Chapter 7.9 WMMA opcode/operand narrow match: XML lists the same six RDNA3
  WMMA opcodes as the manual, all as `ENC_VOP3P`, and records A/B as VGPR-only
  operands with C as `OPR_SRC_VGPR_OR_INLINE` at
  `amdgpu_isa_rdna3.xml:119685` through `:119976`. The remaining WMMA concerns
  are semantic details recorded in `RDNA3-XML-069`, not missing opcode rows.
- Chapter 16.10 WMMA definition-row narrow match: the XML rows for opcodes
  64-69 match the manual's six WMMA definition names and result/source matrix
  types at `amdgpu_isa_rdna3.xml:119685` through `:119976`; the missing
  full-`EXEC` pseudocode and other semantic details are covered by
  `RDNA3-XML-069`.
- The XML does carry the RDNA3 `DS_BVH_STACK_RTN_B32` opcode 173, the expected
  explicit operands (`VDST`, in-out `ADDR`, `DATA0`, `DATA1`), implicit DSMEM
  operands, and `VMEM`/`DATA_SHARE` functional group metadata at
  `amdgpu_isa_rdna3.xml:24285` through `:24333`.
