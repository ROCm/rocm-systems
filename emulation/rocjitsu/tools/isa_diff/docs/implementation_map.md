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
| LLVM for fixtures, the plugin and the real pairs | ROCm 7.2.3 `/opt/rocm/llvm/bin`: `AMD LLVM version 22.0.0git`, `roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e` |
| other LLVM checked | upstream `llc` (LLVM 24, `684dade7d978`) on the build host: no diff against ROCm `llc` for the candidate kernels, so it was not used for the recorded pairs |

Two targets on develop do not build in this environment, and neither has anything to do
with this work. `hsa_init_test` fails to link because the installed ROCm HSA runtime lacks
`hsa_amd_queue_create`. `interposer_dup_test` trips a gcc-13
`-Wstringop-overflow` false positive under `-Werror`. Every other target builds,
and `ScalarSccTest.*:*Sop*:*Scalar*` passes 1162/1162 on the base.

## Target and execution

| need | real symbol | notes |
|---|---|---|
| gfx950 | `ROCJITSU_CODE_ARCH_CDNA4`, `ROCJITSU_CODE_TARGET_GFX950` | `isa/arch/amdgpu/cdna4/target_descriptor.h` |
| every target | `rj_code_arch_t` per family (CDNA1-5, RDNA1-4); `target.h` maps GPU names and `EF_AMDGPU_MACH` values to it | `AmdGpuCodeObject::target_id()` knows 8 GPUs only, so the machine number is read from `e_flags` directly |
| wave size | `kernel_wavefront_size(arch, desc)` (`code/kernel_descriptor_scan.h`), `dispatch_wf(..., wave_size)` | wave32 on RDNA when `ENABLE_WAVEFRONT_SIZE32`; gfx1250 wave32 only |
| wave masks | `Operand::is_wave_mask()`, `wave_sized_register(op, ref, wave_size)` (`isa/operand.h`) | RDNA decoders size wave-mask SGPR operands for wave64; fieldless VCC/EXEC stay unmarked, so `effects.cpp` narrows those by the target's wave size |
| user SGPRs | `kernel_descriptor_user_sgpr_count(arch, desc)` | gfx1250 has a 6-bit field |
| VGPR granule | `descriptor_vgpr_granularity_for_wavefront(arch, wave)` | used by the descriptor allocation check |
| waits | `WaitCounterPolicy::explicit_wait_fields(inst, arch)`, `waitcnt_model(arch)`, `canonical_wait_counter(type, model)`, `Instruction::amdgpu_memory_issue_info()->counter_obligations()` | `PendingState` keeps one counter per `WaitCounterKind` |
| MODE per generation | `initial_mode_from_compute_pgm_rsrc1` (DX10/IEEE/DEBUG only before GFX12) | mirrored in `artifact.cpp` `initial_mode(descriptor, target)` |
| decoder | `Decoder::create(rj_code_arch_t)`, `Decoder::decode_window(span<const uint32_t>, src_loc)` | `isa/decoder.h`; returns `FailureOr<unique_ptr<Instruction>>` |
| execution backend | link `rocjitsu_amdgpu_isa_registry` (via `RJ_TOOL_OBJECT_LIBS`) | The model-only registry (`rocjitsu_amdgpu_isa_model_registry`) leaves `Instruction::execute == nullptr`. We check this per instruction and never infer it from a successful decode. |
| per-instruction execution | `ComputeUnitCore::execute_instruction(Instruction*, Wavefront&, std::string*)` | `vm/amdgpu/compute_unit.h:1545`. Returns `util::Result`. Unimplemented stubs and register-access failures come back as a failure carrying `InstructionExecutionError`, and host exceptions can still escape. It does not advance PC. |
| CU / wave | `ComputeUnitCore::create(name, Config, GpuMemory*, L2Cache*, ExecMode)`, `dispatch_wf(wg, pc, nsgpr, nvgpr, wave_size)` | Uses the same recipe as `tests/scalar_scc_test.cpp` and `tests/cdna_integer_arith_test.cpp`: one wave slot, 104 SGPRs, wave64. |
| SGPR state | `ComputeUnitCore::read_sgpr_storage(phys)`, `write_sgpr(phys, v)`, with `phys = wf.sgpr_alloc().base + n` | Raw access, no observer callbacks. |
| specials | `Wavefront::vcc()/set_vcc_raw`, `exec_raw()/set_exec_raw`, `m0()/set_m0`, `read_scc()/write_scc`, `status_raw/set_status_raw`, `mode_raw/set_mode_raw`, `trapsts`, `ttmp/set_ttmp`, `scratch_base`, `wait_counters()`, `in_trap_handler`, `sleep_cycles`, `pc` | On CDNA4, VCC is wave-local and is not aliased into the SGPR block (only CDNA3 binds it; see `compute_unit.cpp:556`). EXECZ and VCCZ are derived when read, not stored, so they cannot contradict EXEC or VCC. |
| VGPR state (VALU support) | `ComputeUnitCore::write_vgpr(phys, lane, v)`, `read_vgpr_storage(phys, lane)` with `phys = wf.vgpr_alloc().base + n`; `dispatch_wf(..., nvgpr, ...)` | The CU is configured with a 256-VGPR file. The wave allocates exactly the region's tracked range, so an access beyond it fails as an invalid register access. |
| initial MODE | `CommandProcessor` sets MODE from `compute_pgm_rsrc1` (`initial_mode_from_compute_pgm_rsrc1`, internal) | Reimplemented in `artifact.cpp` `initial_mode()` with the same fields. *Measured*: RocJITsu's `v_cvt_f32_f16` honours the f16 denormal mode, which a per-side MODE test relies on. |
| memory instructions (memory support) | `ComputeUnitCore::step()`; region code bytes written into `GpuMemory` at their PCs | One step issues the single wave's instruction through fetch, execute and route. A functional access completes within the step. HALTED or MEMORY_TRAP after a step is a fault. |
| memory backing | `L2Cache::set_backing_memory(GpuMemory*)` | *Measured*: without it every global access is malformed and the wave halts with PC 0 and no error. |
| caches | `ComputeUnitCore::flush_all()`, `flush_l1()`, `L2Cache::invalidate_all()` | L1 scalar/vector caches are write-through and the L2 is write-back. The runner writes back and invalidates before every memory step and at load/capture. |
| access observation | `ExecutionPlugin::observes_memory_routing()`, `onAmdgpuMemoryAccessRouted(MemoryAccessObservation)`; `ExecutionPluginGroup` + `ComputeUnitCore::set_plugin_group` | Fires after routing and before the pipeline access, with per-lane addresses, element size, lane mask, load/store, atomic op and secondary addresses. The memory model seeds touched words and records stores here. |
| wait checker | `Config::memory_wait_diagnostics = MemoryWaitDiagnostics::Off` | The CU's scoreboard keeps state across cases that re-issue the same PCs. The planner's pending analysis replaces it. |
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
- `Instruction::implicit_uses/implicit_defs` (`RegisterSet`, iterated with `for_each` as
  `RegisterRef{cls, index, width}`) is *measured* as follows.
  - `v_mov_b32`, `v_add_co_u32`, `s_mul_hi_u32`, `s_load_dword`, `s_store_dword` and
    `s_waitcnt` report nothing.
  - `global_store_dwordx2` reports its SGPR base pair (`s6`, `s7`) as implicit uses.
  - Outside P0, the tool translates SGPR/EXEC/VCC/SCC/M0 entries into tracked locations
    and ignores VGPR/AGPR. TTMP, FLAT_SCRATCH and PC remain unknown effects.
  - Before this translation, every global store made all locations live, and the
    renamed dead SALU temporaries of the real `-enable-misched=false` pair showed up as
    observable mismatches.
  - P0 instructions keep the strict rule: any hidden effect rejects them.
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
| assembler / LLVM MC | none in-tree | The separate plugin library `llvm/` (`rj_isa_diff_llvm`) runs the external `llc`/`llvm-mc`/`ld.lld`/`llvm-objcopy` with `posix_spawn` and records provenance. The core does not link LLVM. |
| relocations | none in the code-object reader (`AmdGpuCodeObject` exposes sections, not relocations) | `artifact.cpp` reads `SHT_REL`/`SHT_RELA` entries targeting `.text`: static `.rela.text` (`sh_info` = .text, kept by `ld.lld --emit-relocs`, `r_offset` a vaddr) and dynamic entries whose `r_offset` falls in `.text`. *Measured*: plain `ld.lld -shared` resolves `R_AMDGPU_GOTPCREL32_LO/HI` and leaves no trace in `.text`, while `--emit-relocs` keeps both entries at the literal offsets of `s_add_u32`/`s_addc_u32` after `s_getpc_b64`. |
| AMDHSA metadata | `amdhsa::MetadataDocument::parse(image)` (`code/amdhsa_metadata.h`), `MetadataValue::{kind,map,array,field,text_field_is}` | Reused. The kernel entry is selected by `.name` or `.symbol == name + ".kd"` and flattened to key/value pairs; `.args[i]` records are grouped. String views borrow the image, so values are copied out before the image goes away. |
| build fingerprint | none | `result_cache.cpp`: SHA-256 over `/proc/self/exe` and every executable file mapping in `/proc/self/maps`, via `util/sha256.h` |

## P0 supported subset

- Target: gfx950 / CDNA4, wave64, `ArithmeticSemantics` default.
- Region instructions must be on the allowlist in `effects.cpp` (integer SOP1/SOP2/SOPC/SOPK:
  moves, add/sub/carry, min/max, bitwise, shifts, bfe/bfm, mul, compare/select,
  `s_cmov*`, `*_saveexec`, `s_nop`). Each operand must classify into one of:
  s0..s101, VCC_LO/HI, EXEC_LO/HI, M0, SCC, a constant, or a literal.
- The following are rejected with a reason in a region (a kernel may still contain them).
  VALU support later admitted plain VALU, and memory support admitted `global_load_*`,
  `global_store_*`, `s_load_dword*` and `s_waitcnt`:
  VALU/VMEM/SMEM/LDS, MFMA, branches,
  waitcnt/barrier, getpc/setpc/swappc, getreg/setreg, movrel, sendmsg/trap,
  FLAT_SCRATCH/XNACK/TTMP/aperture selectors, any instruction with `data()` state,
  `execute == nullptr`, and PC-derived values (forward taint from `s_getpc_b64`).
- Frame state that P0 neither reads nor writes is fixed and recorded in the contract:
  MODE, STATUS except SCC, TRAPSTS, TTMP, scratch base, wait counters, trap state.
  Each one is checked for preservation after every case.
- Instructions whose bytes are patched by a relocation are rejected in a region.

## Handoff section to code

| handoff | code |
|---|---|
| 3 contract, 3.1 mapping | `contract.{h,cpp}` (`RegionContract`, `register_map`, `map_forward/inverse`, `parse_register_map`) |
| 3.2 observable union, preservation | `region_planner.cpp` `build_region` (live-in/live-out unions), `differential.cpp` `compare_pair` (effect model, frame, R_out) |
| 3.3 constraints and sources | `contract.h` `Constraint::source`; `region_planner.cpp` known-bits (`analyse_known_bits`, `derive_constraints`); `case_io.cpp` unrestricted run |
| 4 artifacts, relocations, metadata | `artifact.{h,cpp}`, `diff.cpp` (descriptor fields, metadata fields, relocations at matched instructions) |
| 4.1 LLVM (plugin) | `llvm/llvm_frontend.{h,cpp}`, `rj_isa_diff_main.cpp` `load_side` / `llvm-pair` |
| 5 alignment and classification | `diff.{h,cpp}` |
| 5.1/5.2 regions, merge, context, composition | `region_planner.{h,cpp}` (steps 1-8), `case_io.cpp` (variants after a mismatch) |
| 5.3 stop rules, diagnostic path | `PlanOptions::{max_region_insts,max_expansions,max_analysis_us,diagnostic_entry_prefix}` |
| 6 state model, 6.1 pending effects | `location.*`, `effects.*`, `machine_state.*`, `kernel_model.cpp` `compute_pending` |
| 7 execution | `region_runner.{h,cpp}` |
| 8 inputs and learning hooks | `boundary.*`, `roles.*`, `input_policy.*`, `differential.cpp` (log, novelty) |
| 9 result model | `differential.h`, `case_io.cpp` `aggregate_results` |
| 10 counterexamples, replay, reduction | `differential.cpp` (`counterexample_json`, `reduce_mismatch`, `replay_counterexample`), `mutation.*` |
| 11 caches and cost | `result_cache.{h,cpp}`, `region.cpp` `compute_prepared_key`, timing fields in `case_io.cpp` / `differential.cpp` |
| 12 CLI and schemas | `rj_isa_diff_main.cpp`, `docs/schema/` |
| 13 tests | `tests/isa_diff/isa_diff_test.cpp`, `tests/isa_diff/llvm_plugin_test.cpp`, `docs/schema/validate_outputs.py` |
| callable functions (Phase G) | `artifact.cpp` (`artifact_from_code_object`: function symbol without descriptor; `initial_mode(const Artifact&)`), `kernel_model.cpp` (`BlockInfo::returns`, `return_observable_locs/vgprs`), `rj_isa_diff_main.cpp` (`--return-vgprs`, `--function-mode`) |
| inline asm (Phase G) | `llvm/llvm_frontend.cpp` (`mark_inline_asm`), `artifact.cpp` (`read_inline_asm`), `effects.cpp` (`parse_inline_asm`, `classify_inline_asm`, `inline_asm_shape`), `kernel_model.cpp` (pseudo-instructions), `diff.cpp` (`key_of`, shape notes), `region_runner.cpp` (`run_inline_asm`, `inline_asm_word`), `differential.cpp` (`asm_use_difference`, inline-asm seed) |
| one command for an LLVM change (Phase G) | `llvm/rj_isa_diff_pr.py` (copied to the build tree as `rj_isa_diff_pr`) |
| NaN equivalence (Phase H) | `effects.cpp` (`fp_result_kind`, `fp_source_kind`, `nan_residual`), `region_runner.cpp` (`plan_tags`, `apply_tags`, memory-model `record_fp_units`), `differential.cpp` (`compare_pair`, `nan_unit_equal`) |
| undefined values and narrow returns (Phase H) | `kernel_model.cpp` (`compute_undefined`, `vundef_before`, `vgpr_liveness`), `region_planner.cpp` (`vundef_old/new`, `vexit_masks`), `differential.cpp` (`compute_undef_mask`, `check_new_undef`) |
| NaN choice and undefined scalars (Phase I) | `region_runner.cpp` (`choose_nans`, `set_nan_choice`), `kernel_model.cpp` (`undef_before`, scalar may-written analysis), `differential.cpp` (`compute_undef_mask`, `check_new_undef`, `note_mask`), `llvm/rj_isa_diff_pr.py` (`argument_sgprs`) |
| poison filter (Phase H) | `effects.cpp` (`poison_check_kind`, `parse_poison_kinds`), `region_runner.cpp` (`poison_before`, `poison_after`), `llvm/rj_isa_diff_pr.py` (`poison_kinds`, `poison_triage`) |
