# rj_isa_diff implementation map

Inspection of the RocJITsu checkout used to build `rj_isa_diff`, the incremental
AMDGPU ISA-to-ISA differential tester. Every API named here was checked against
this checkout. Behaviour marked *measured* was observed by decoding or executing
real gfx950 encodings, not inferred from comments.

## Provenance

| item | value |
|---|---|
| repository | `ROCm/rocm-systems`, `emulation/rocjitsu` |
| base revision | `origin/develop` @ `90ddc8aae5` (2026-11-14) |
| branch | `users/lialan/rj/incremental-isa-diff` |
| build host | Ubuntu 22.04, 32 cores (remote build host) |
| compiler | g++-13 (gcc-11 lacks `<format>`) |
| configure | `CC=gcc-13 CXX=g++-13 cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DROCJITSU_ENABLE_VFIO=OFF -DROCJITSU_ENABLE_PERFSIM_PLUGIN=OFF -DROCJITSU_HOST_CPU_BASELINE=x86-64-v3 ` plus an include-path shim when the system libdrm headers predate the bundled ones |
| LLVM used only to author fixtures | ROCm 7.2.3 `llvm-mc`/`clang`/`ld.lld`, AMD LLVM 22.0.0git |

Two targets on develop do not build in this environment, and neither has anything to do
with this work. `hsa_init_test` fails to link because the installed ROCm HSA runtime lacks
`hsa_amd_queue_create`. `interposer_dup_test` trips a gcc-13
`-Wstringop-overflow` false positive under `-Werror`. Every other target builds,
and `ScalarSccTest.*:*Sop*:*Scalar*` passes 1162/1162 on the base.

## Target and execution

| need | real symbol | notes |
|---|---|---|
| gfx950 | `ROCJITSU_CODE_ARCH_CDNA4`, `ROCJITSU_CODE_TARGET_GFX950` | `isa/arch/amdgpu/cdna4/target_descriptor.h` |
| decoder | `Decoder::create(rj_code_arch_t)`, `Decoder::decode_window(span<const uint32_t>, src_loc)` | `isa/decoder.h`; returns `FailureOr<unique_ptr<Instruction>>` |
| execution backend | link `rocjitsu_amdgpu_isa_registry` (via `RJ_TOOL_OBJECT_LIBS`) | The model-only registry (`rocjitsu_amdgpu_isa_model_registry`) leaves `Instruction::execute == nullptr`. We check this per instruction and never infer it from a successful decode. |
| per-instruction execution | `ComputeUnitCore::execute_instruction(Instruction*, Wavefront&, std::string*)` | `vm/amdgpu/compute_unit.h:1545`. Returns `util::Result`. Unimplemented stubs and register-access failures come back as a failure carrying `InstructionExecutionError`, and host exceptions can still escape. It does not advance PC. |
| CU / wave | `ComputeUnitCore::create(name, Config, GpuMemory*, L2Cache*, ExecMode)`, `dispatch_wf(wg, pc, nsgpr, nvgpr, wave_size)` | Uses the same recipe as `tests/scalar_scc_test.cpp` and `tests/cdna_integer_arith_test.cpp`: one wave slot, 104 SGPRs, wave64. |
| SGPR state | `ComputeUnitCore::read_sgpr_storage(phys)`, `write_sgpr(phys, v)`, with `phys = wf.sgpr_alloc().base + n` | Raw access, no observer callbacks. |
| specials | `Wavefront::vcc()/set_vcc_raw`, `exec_raw()/set_exec_raw`, `m0()/set_m0`, `read_scc()/write_scc`, `status_raw/set_status_raw`, `mode_raw/set_mode_raw`, `trapsts`, `ttmp/set_ttmp`, `scratch_base`, `wait_counters()`, `in_trap_handler`, `sleep_cycles`, `pc` | On CDNA4, VCC is wave-local and is not aliased into the SGPR block (only CDNA3 binds it; see `compute_unit.cpp:556`). EXECZ and VCCZ are derived when read, not stored, so they cannot contradict EXEC or VCC. |
| UB detection | `ComputeUnitCore::isa_diagnostic_count()` | Read the delta around each instruction. `report_undefined_behavior` only logs and bumps this counter. |
| arithmetic mode | `ComputeUnitCore::Config::arithmetic` | Recorded in every contract. Integer SALU does not depend on it. |
| VCC aliasing (assumption) | -- | On CDNA hardware VCC occupies the top of the kernel's SGPR allocation. RocJITsu's CDNA4 model keeps VCC wave-local, and RocJITsu execution is the trusted semantics here, so P0 tracks VCC_LO/HI separately from s0..s101. Compiled code never names the aliased SGPRs together with `vcc`. |

## Instruction model (*measured* on gfx950 with a throwaway decode/operand probe)

- `Instruction`: `mnemonic()`, `size()` (bytes, literal included), `flags()`
  (`InstFlags`), `src_operand(i)`/`dst_operand(i)`, `src_loc()`, `raw_encoding()`, `data()`.
- SCC is an explicit *fieldless* operand: `src_scc`, `to_special_reg_class()==SCC`,
  `encoding_value()==253`, `size_bits()==1`. Carry-in shows up as a source and
  SCC results as a destination (`s_add_u32`, `s_addc_u32`, `s_cselect_*`, `s_cmp*`).
- `s_mov_b32 sN, src_scc` reads SCC through the generic selector (253, 32 bits) and is
  **not** fieldless.
- SGPR operands: `to_register_ref() = {SGPR, n, width}`, with width 2 for pairs.
- EXEC by selector: `to_register_ref() = {EXEC, 0|1, width}` (lo/hi/pair).
- VCC_LO/HI (106/107) and M0 (124) by selector: `to_register_ref()` is `nullopt`, so
  they have to be classified from `encoding_value()` + `size_bits()`.
- FLAT_SCRATCH (102/103) and XNACK_MASK (104/105) by selector are also `nullopt`. They are
  outside P0 and rejected.
- Constants: `const_value()` holds both inline constants and literals.
- `InstDefUse` (`code/analysis/def_use_chain.h`) reported **no special effects
  at all** on gfx950, not even SCC, for every SALU instruction probed. It is not used for
  specials. The tool builds its own effect summary (`effects.cpp`).
- The `SALU` flag is also set on `s_nop`, `s_waitcnt`, `s_cbranch_*`, `s_getpc_b64`,
  and `s_endpgm`. P0 therefore uses an explicit mnemonic allowlist, *and* every operand
  must classify into a modelled location.

## CFG, liveness, ingestion

| need | real symbol | reuse decision |
|---|---|---|
| code object | `AmdGpuCodeObject(const uint8_t*, size_t)`, `is_valid()`, `text_sections()`, `target_id()` | reused |
| kernels | `scan_kernel_descriptors(image, text_offset, text_size, text_section_index)` → `KernelDescriptorInfo{kernel_name, entry_text_offset, descriptor}` | reused. The kernel byte range comes from the ELF symbol `st_size`, read via `read_elf_table<Elf64_Sym>`. |
| raw stream | Custom `CodeObject` subclass with one `.text` section | Same pattern as `tests/analysis/liveness_test.cpp:155` |
| CFG | `BasicBlock::build_reachable(co, decoder, arch, entries, emit, permitted_ranges)` | Reused. The blocks own the decoded, executable `Instruction`s, so each kernel is decoded once. Any `successor_issue() != None` or call edge makes everything live. |
| liveness | `LivenessAnalysis` tracks ordinary registers only; SCC, EXEC, VCC and M0 are not part of its dataflow (see the header comment) | **Not reused.** The handoff requires SCC-precise liveness (live and dead SCC must be told apart). `kernel_model.cpp` runs a backward dataflow over `LocSet`. |
| EXEC full/unknown | `ExecMaskAnalysis` | not needed in P0 (SALU only) |
| encoders | `code/builders/instruction_builder.h`, generated `cdna4::build_*` | Not used. Fixtures are authored as `.s`, assembled once with `llvm-mc -mcpu=gfx950`, and kept as words with the asm beside them. |
| JSON | none in-tree (config uses schema-driven FlatBuffers) | Small tool-local reader/writer (`json.cpp`). Case files are hand-editable. |
| SHA-256 | `util/sha256.h` | reused for artifact and contract hashes |
| existing differential, snapshot or symbolic work | none: no region runner, no per-wave snapshot, no SMT. SemaAST is Python codegen only, and the VM checkpoint is whole-VM. `fuzz/execution` is a crash fuzzer. | new code |
| assembler / LLVM MC | none in-tree | **Deferred to the separate LLVM plugin task** (see README) |

## P0 supported subset

- Target: gfx950 / CDNA4, wave64, `ArithmeticSemantics` default.
- Region instructions must be on the allowlist in `effects.cpp` (integer SOP1/SOP2/SOPC/SOPK:
  moves, add/sub/carry, min/max, bitwise, shifts, bfe/bfm, mul, compare/select,
  `s_cmov*`, `*_saveexec`, `s_nop`). Each operand must classify into one of:
  s0..s101, VCC_LO/HI, EXEC_LO/HI, M0, SCC, a constant, or a literal.
- The following are rejected with a reason in a region (a kernel may still contain them):
  VALU/VMEM/SMEM/LDS, MFMA, branches,
  waitcnt/barrier, getpc/setpc/swappc, getreg/setreg, movrel, sendmsg/trap,
  FLAT_SCRATCH/XNACK/TTMP/aperture selectors, any instruction with `data()` state,
  `execute == nullptr`, and PC-derived values (forward taint from `s_getpc_b64`).
- Frame state that P0 neither reads nor writes is fixed and recorded in the contract:
  MODE, STATUS except SCC, TRAPSTS, TTMP, scratch base, wait counters, trap state.
  Each one is checked for preservation after every case.
