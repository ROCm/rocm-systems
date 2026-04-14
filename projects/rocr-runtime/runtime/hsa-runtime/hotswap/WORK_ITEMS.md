# HotSwap MLIR Pipeline – Work Items

## Done

- [x] Integrate waveasm MLIR dialect into build
- [x] Binary lifter: MCInst → waveasm ops (VALU, SALU, SMEM, VMEM, DS, SOPP)
- [x] Branch/label resolution in lifter
- [x] SSA construction pass (physical → virtual registers)
- [x] Cross-target mnemonic mapping (GFX12 → GFX9)
- [x] Wave width translation (saveexec_b32 expansion, exec_hi clear, v_cmpx, VCC width)
- [x] Assembly emitter (waveasm IR → text)
- [x] LLVM MC assembler integration (text → bytes)
- [x] Pipeline orchestration (lift → retarget → widen → emit → assemble)
- [x] Unit tests for all passes (18 tests passing)
- [x] SMEM / global memory / VOPC assembly emission fixes
- [x] Code object builder (ELF extraction, assembly splicing, llvm-mc + ld.lld rebuild)
- [x] GPU round-trip MVE: vecadd kernel gfx942→gfx942, 292 instructions, zero fallbacks, GPU PASS

## P0 – Production integration

- [ ] Wire `runPipeline` into `TranspileCodeObject` as dual-pipeline (run both, compare, fall back to old)
- [ ] Kernel descriptor parsing/preservation (256-byte aligned KDs in .text)
- [ ] ELF .text replacement and section resizing
- [ ] MSGPACK metadata patching (register counts, ISA name)
- [ ] Kernel descriptor patching for wave64 (RSRC1/RSRC2 fields)

## P1 – Instruction coverage

- [ ] WMMA → MFMA matrix instruction translation
- [ ] DPP format conversion (DPP8 → DPP16)
- [ ] `s_add_nc_u64` → 32-bit pair expansion
- [ ] Encoding suffix handling (`_e32`, `_e64`, `_dpp`)
- [ ] Register spill/save sequences for workgroup IDs
- [ ] SGPR replacement (TTMP → VGPR shadows)
- [ ] GPU end-to-end validation with real kernels

## P2 – Advanced transforms

- [ ] Structured control flow recovery (saveexec/cbranch → waveasm.if/loop)
- [ ] Register allocation pass (SSA → physical)
- [ ] Instruction scheduling on SSA IR

## P3 – Cleanup

- [ ] Retire string-based transpiler once MLIR pipeline has full parity
