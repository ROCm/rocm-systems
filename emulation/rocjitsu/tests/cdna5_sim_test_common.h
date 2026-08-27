// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "aql_queue.h"
#include "decode_test_util.h"
#include "halt_snapshot_plugin.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vds.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vglobal.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vimage.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "rocjitsu/vm/soc.h"
#include "util/except.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include "rocjitsu/vm/amdgpu/register_access.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
namespace rocjitsu::test::cdna5 {

inline const std::string kGfx1250ConfigPath = std::string(CONFIG_DIR) + "/gfx1250_mi455x.json";

inline constexpr uint32_t S_ENDPGM_GFX12 = 0xBFB00000u;
inline constexpr uint32_t S_WAIT_KMCNT_0_GFX12 = 0xBFC70000u;
inline constexpr uint32_t S_SET_VGPR_MSB = 0xBF860000u;
// LLVM references for these gfx1250 register capabilities:
// - llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:
//   getAddressableNumSGPRs() returns 106 ordinary SGPRs for GFX10+.
// - llvm/lib/Target/AMDGPU/SIRegisterInfo.td:
//   VCC, TTMP0-15, null, M0, and EXEC occupy scalar selector values 106-127.
// - llvm/lib/Target/AMDGPU/AMDGPU.td:
//   FeatureISAVersion12_50_Common includes Feature1024AddressableVGPRs.
// - llvm/test/MC/AMDGPU/hsa-gfx1250-v4.s:
//   max_vgprs accepts .amdhsa_next_free_vgpr 1024 for Wave32 gfx1250.
// - llvm/lib/Target/AMDGPU/AMDGPULowerVGPREncoding.cpp:
//   high VGPRs still use the encoded v0-v255 operand window; s_set_vgpr_msb
//   supplies two high address bits per operand role.
// - llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:
//   GFX12.5 reports four SIMD units per CU.
inline constexpr uint32_t kGfx1250ScalarSlots = 128;
inline constexpr uint32_t kGfx1250Wave32VgprAllocation = 1024;
inline constexpr uint32_t kGfx1250VgprEncodingGranule = 16;
inline constexpr uint32_t kGfx1250SimdsPerCu = 4;
inline constexpr uint32_t kGfx1250MaxWavesPerSimd = 16;
inline constexpr uint32_t kGfx1250WaveSlotsPerCu = kGfx1250SimdsPerCu * kGfx1250MaxWavesPerSimd;
inline constexpr uint32_t kGfx1250LdsSizeKb = 320;
inline constexpr uint64_t kGfx1250HbmBytes = 432ull * 1024 * 1024 * 1024;
inline constexpr uint32_t kGfx1250HbmWidthBits = 12 * 2048;
inline constexpr uint32_t kGfx1250VectorCacheSizeKb = 64;
inline constexpr uint32_t kGfx1250L2SizeKb = 192 * 1024;
struct Gfx1250Sim {
  // Direct-construction tests intentionally bypass Decoder. Select the same
  // immutable backend that a full gfx1250 decoder would inject while building
  // their generated instruction and operand objects.
  ScopedIsaExecutionBackend execution_backend_scope{&::rocjitsu::cdna5::execution_backend()};
  config::LoadedConfig loaded;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  test::HaltSnapshotPlugin *snapshot = nullptr;

  Gfx1250Sim() : loaded(config::load_config(kGfx1250ConfigPath, rocjitsu::kEmbeddedSchema)) {
    build();
  }

  explicit Gfx1250Sim(const std::string &config_json)
      : loaded(config::load_config_from_string(config_json, rocjitsu::kEmbeddedSchema)) {
    build();
  }

  void build() {
    soc = loaded.soc();
    memory = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();
    // Capture every wavefront's final register state at halt. A wave frees its
    // registers at s_endpgm, so tests that run a kernel to completion read the
    // result from the snapshot rather than the (freed) slot.
    plugin_group = test::make_halt_snapshot_group(&snapshot);
    soc->set_plugin_group(plugin_group);
  }

  amdgpu::Xcd *xcd(uint32_t idx = 0) { return soc->xcd(idx); }
  amdgpu::CommandProcessor *cp(uint32_t idx = 0) { return xcd(idx)->command_processor(); }
  amdgpu::ComputeUnitCore *cu(uint32_t idx = 0) {
    return xcd()->shader_engine(0)->compute_unit(idx);
  }

  /// Place a single resident wavefront on CU 0 without running it to s_endpgm, for
  /// tests that drive individual instructions and inspect/modify live wave state.
  amdgpu::Wavefront *dispatch_scratch_wf(uint32_t vgprs = kGfx1250Wave32VgprAllocation) {
    return cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, vgprs);
  }

  uint64_t write_kernel(uint64_t addr, const uint32_t *words, size_t num_words,
                        uint32_t sgprs = 104, uint32_t vgprs = 32, uint32_t user_sgprs = 2,
                        bool enable_wg_id_x = false, bool enable_wg_id_y = false,
                        bool enable_wg_id_z = false, uint32_t kernel_code_properties = 0,
                        uint32_t kernarg_size = 0, uint32_t kernarg_preload_length = 0,
                        uint32_t kernarg_preload_offset = 0, uint32_t enable_vgpr_workitem_id = 0,
                        uint32_t named_barrier_blocks = 0) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    kd.kernarg_size = kernarg_size;
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((vgprs / kGfx1250VgprEncodingGranule) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((sgprs / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT, user_sgprs);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    enable_wg_id_x);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    enable_wg_id_y);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    enable_wg_id_z);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID,
                    enable_vgpr_workitem_id);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT,
                    named_barrier_blocks);
    kd.kernel_code_properties = kernel_code_properties;
    AMDHSA_BITS_SET(kd.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH, kernarg_preload_length);
    AMDHSA_BITS_SET(kd.kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET, kernarg_preload_offset);

    memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    memory->load_image(reinterpret_cast<const uint8_t *>(words), num_words * sizeof(uint32_t),
                       addr + sizeof(kernel_descriptor_t));
    return addr;
  }
};
// Drive the engine until the CU has no resident wavefronts. A wave frees itself at
// s_endpgm, so the kernel is complete once the CU reports idle (after having had
// work). Final wavefront state should be captured via HaltSnapshotPlugin.
inline void step_until_halted(simdojo::SimulationEngine &engine, amdgpu::ComputeUnitCore &cu,
                              uint32_t max_steps = 10000) {
  bool saw_work = false;
  for (uint32_t i = 0; i < max_steps && engine.step(); ++i) {
    if (cu.has_active_wfs())
      saw_work = true;
    else if (saw_work)
      return;
  }
}

inline void step_until_xcd_halted(Gfx1250Sim &sim, uint32_t max_steps = 10000) {
  auto any_active = [&]() {
    for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
      auto *se = sim.xcd()->shader_engine(se_idx);
      for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx)
        if (se->compute_unit(cu_idx)->has_active_wfs())
          return true;
    }
    return false;
  };

  bool saw_work = false;
  for (uint32_t i = 0; i < max_steps && sim.engine->step(); ++i) {
    if (any_active()) {
      saw_work = true;
      continue;
    }
    if (saw_work)
      return;
  }
}

// Runs a one-wave kernel to s_endpgm and returns the wave's final-state snapshot
// (captured at halt, before its registers are freed), or nullptr if THIS dispatch
// produced no new halt. Snapshots accumulate across calls in the fixture plugin, so
// compare the count before/after and only return a pointer when this dispatch
// appended one — otherwise a regression that fails to run/halt would silently return
// a stale snapshot from a previous dispatch.
inline const test::WavefrontSnapshot *dispatch_one_wave(Gfx1250Sim &sim, const uint32_t *code,
                                                        size_t num_words, uint32_t vgprs = 32) {
  const size_t before = sim.snapshot->snapshots().size();
  uint64_t kernel_object = sim.write_kernel(0x10000, code, num_words, 104, vgprs);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());
  if (sim.snapshot->snapshots().size() == before)
    return nullptr;
  return &sim.snapshot->snapshots().back();
}

constexpr uint32_t make_vmov_b32(uint8_t vdst) {
  return 0x7E0002FFu | (static_cast<uint32_t>(vdst) << 17);
}

constexpr std::array<uint32_t, 2> make_vmov_b32_literal(uint8_t vdst, uint32_t literal) {
  return {make_vmov_b32(vdst), literal};
}

constexpr std::array<uint32_t, 2> make_s_load_b32_scaled_imm(uint8_t sdata, uint8_t sbase_pair,
                                                             uint32_t scaled_offset) {
  constexpr uint32_t kSmemEncoding = 0x3Du << 26;
  constexpr uint32_t kSoffsetNull = 0x7Cu;
  return {kSmemEncoding | ((static_cast<uint32_t>(sdata) & 0x7Fu) << 6) |
              (static_cast<uint32_t>(sbase_pair) & 0x3Fu),
          (scaled_offset & 0x00FF'FFFFu) | (1u << 24) | (kSoffsetNull << 25)};
}

constexpr uint16_t vopd_src0_vgpr(uint16_t reg) { return 256 + reg; }

enum class VopdOp : uint16_t {
  FmamkF32 = 2,
  MulF32 = 3,
  MulDx9ZeroF32 = 7,
  MovB32 = 8,
  CndmaskB32 = 9,
  FmaF32 = 19,
  FmaF64 = 32,
};

struct VopdSlot {
  VopdOp op;
  uint16_t src0;
  uint8_t src1;
  uint8_t src2;
  uint8_t dst;
};

constexpr std::array<uint32_t, 3> make_vopd3_pair(VopdSlot x, VopdSlot y, uint8_t negx = 0,
                                                  uint8_t negy = 0) {
  return {
      0xCF000000u | ((static_cast<uint32_t>(x.op) & 0x3Fu) << 18) |
          ((static_cast<uint32_t>(y.op) & 0x3Fu) << 12) | (static_cast<uint32_t>(x.src0) & 0x1FFu),
      (static_cast<uint32_t>(y.src0) & 0x1FFu) | ((static_cast<uint32_t>(negx) & 0x7u) << 9) |
          ((static_cast<uint32_t>(negy) & 0x7u) << 12) | (static_cast<uint32_t>(x.src1) << 16) |
          (static_cast<uint32_t>(x.src2) << 24),
      static_cast<uint32_t>(x.dst) | (static_cast<uint32_t>(y.src1) << 8) |
          (static_cast<uint32_t>(y.src2) << 16) | (static_cast<uint32_t>(y.dst) << 24),
  };
}

template <size_t N>
void append_instruction(std::vector<uint32_t> &code, const std::array<uint32_t, N> &words) {
  code.insert(code.end(), words.begin(), words.end());
}

inline void append_instruction(std::vector<uint32_t> &code, uint32_t word) { code.push_back(word); }

inline void write_wave_sgpr(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf, uint32_t reg,
                            uint32_t value) {
  cu.write_sgpr(wf.sgpr_alloc().base + reg, value);
}

inline uint32_t read_wave_sgpr(const amdgpu::ComputeUnitCore &cu, const amdgpu::Wavefront &wf,
                               uint32_t reg) {
  return cu.read_sgpr(wf.sgpr_alloc().base + reg);
}

inline void write_global_u32(amdgpu::GpuMemory &memory, uint64_t addr, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte)
    memory.write8(addr + byte, static_cast<uint8_t>(value >> (byte * 8)));
}

inline uint32_t read_global_u32(amdgpu::GpuMemory &memory, uint64_t addr) {
  uint32_t value = 0;
  for (uint32_t byte = 0; byte < 4; ++byte)
    value |= static_cast<uint32_t>(memory.read8(addr + byte)) << (byte * 8);
  return value;
}

inline std::unique_ptr<Instruction> decode_gfx1250(const std::array<uint32_t, 3> &words,
                                                   std::string_view expected_mnemonic) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  if (!decoder) {
    ADD_FAILURE() << "Decoder::create() returned nullptr for gfx1250";
    return nullptr;
  }

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  if (!inst) {
    ADD_FAILURE() << "decode() returned nullptr for gfx1250 instruction";
    return nullptr;
  }

  EXPECT_EQ(inst->mnemonic(), expected_mnemonic);
  EXPECT_EQ(inst->size(), static_cast<int>(words.size() * sizeof(words[0])));
  return inst;
}

} // namespace rocjitsu::test::cdna5
