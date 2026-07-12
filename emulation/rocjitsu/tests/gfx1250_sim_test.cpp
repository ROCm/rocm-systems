// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vds.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vglobal.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vimage.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop3.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "rocjitsu/vm/soc.h"
#include "util/except.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string kGfx1250ConfigPath = std::string(CONFIG_DIR) + "/gfx1250.json";

constexpr uint32_t S_ENDPGM_GFX12 = 0xBFB00000u;
constexpr uint32_t S_WAIT_KMCNT_0_GFX12 = 0xBFC70000u;
constexpr uint32_t S_SET_VGPR_MSB = 0xBF860000u;
// LLVM references for these gfx1250 register capacities:
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
//   GFX12.5 reports four SIMDs per CU, 20 waves per SIMD, and 160 KiB of
//   addressable local memory.
constexpr uint32_t kGfx1250ScalarSlots = 128;
constexpr uint32_t kGfx1250Wave32VgprAllocation = 1024;
constexpr uint32_t kGfx1250VgprEncodingGranule = 16;
constexpr uint32_t kGfx1250SimdsPerCu = 4;
constexpr uint32_t kGfx1250MaxWavesPerSimd = 20;
constexpr uint32_t kGfx1250WaveSlotsPerCu = kGfx1250SimdsPerCu * kGfx1250MaxWavesPerSimd;
constexpr uint32_t kGfx1250LdsSizeKb = 160;
constexpr uint32_t kSdmaOpCopy = 1;
constexpr uint32_t kSdmaOpFence = 5;
constexpr uint32_t kSdmaOpPollRegmem = 8;
constexpr uint32_t kSdmaOpConstFill = 11;
constexpr uint32_t kSdmaOpTimestamp = 13;
constexpr uint32_t kSdmaOpGcr = 17;
constexpr uint32_t kSdmaSubopCopyLinear = 0;
constexpr uint32_t kSdmaSubopFence64 = 2;
constexpr uint32_t kSdmaSubopPollMem64 = 5;

class TestMemoryInstruction : public Instruction {
public:
  explicit TestMemoryInstruction(std::unique_ptr<DynamicInstState> state)
      : Instruction("test_mem", nullptr) {
    flags_ |= MEMORY_OP;
    set_data(std::move(state));
  }
};

class DeferredClusterLdsMulticastEngine : public amdgpu::ClusterLdsMulticastEngine {
public:
  amdgpu::ClusterLdsMulticastResult
  submit(amdgpu::ClusterLdsMulticastTransaction submitted,
         amdgpu::ClusterLdsMulticastCompletion submitted_completion) override {
    txn = std::move(submitted);
    completion = std::move(submitted_completion);
    return amdgpu::ClusterLdsMulticastResult::Deferred;
  }

  amdgpu::ClusterLdsMulticastTransaction txn;
  amdgpu::ClusterLdsMulticastCompletion completion;
};

struct Gfx1250Sim {
  config::LoadedConfig loaded;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine;

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
    engine->build();
  }

  amdgpu::Xcd *xcd(uint32_t idx = 0) { return soc->xcd(idx); }
  amdgpu::CommandProcessor *cp(uint32_t idx = 0) { return xcd(idx)->command_processor(); }
  amdgpu::ComputeUnitCore *cu(uint32_t idx = 0) {
    return xcd()->shader_engine(0)->compute_unit(idx);
  }

  uint64_t write_kernel(uint64_t addr, const uint32_t *words, size_t num_words,
                        uint32_t sgprs = 104, uint32_t vgprs = 32, uint32_t user_sgprs = 2,
                        bool enable_wg_id_x = false, bool enable_wg_id_y = false,
                        bool enable_wg_id_z = false, uint32_t kernel_code_properties = 0,
                        uint32_t kernarg_size = 0, uint32_t kernarg_preload_length = 0,
                        uint32_t kernarg_preload_offset = 0, uint32_t enable_vgpr_workitem_id = 0) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    kd.kernarg_size = kernarg_size;
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((vgprs / kGfx1250VgprEncodingGranule) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((sgprs / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, user_sgprs);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    enable_wg_id_x);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    enable_wg_id_y);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    enable_wg_id_z);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID,
                    enable_vgpr_workitem_id);
    kd.kernel_code_properties = kernel_code_properties;
    AMDHSA_BITS_SET(kd.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH, kernarg_preload_length);
    AMDHSA_BITS_SET(kd.kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET, kernarg_preload_offset);

    memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    memory->load_image(reinterpret_cast<const uint8_t *>(words), num_words * sizeof(uint32_t),
                       addr + sizeof(kernel_descriptor_t));
    return addr;
  }
};

std::string make_single_se_gfx1250_config(uint32_t num_cus) {
  std::string cu_range = "cu[0:" + std::to_string(num_cus) + "]";
  std::string links;
  for (uint32_t i = 0; i < num_cus; ++i) {
    if (i > 0)
      links += ",";
    links += R"({"src":"xcd0.cp.req_)" + std::to_string(i) + R"(","dst":"xcd0.se0.cu)" +
             std::to_string(i) + R"(.cpl","latency":1,"weight":2})";
    links += R"(,{"src":"xcd0.se0.cu)" + std::to_string(i) + R"(.req","dst":"xcd0.l2.cpl_)" +
             std::to_string(i) + R"(","latency":1,"weight":10})";
  }

  return R"({"max_ticks":10000,"num_threads":1,"vm":{"arch":"gfx1250"},)"
         R"("topology":{"root":{"name":"soc","type":"soc","children":[)"
         R"({"name":"vram","type":"gpu_memory"},)"
         R"({"name":"xcd0","type":"xcd","children":[)"
         R"({"name":"l2","type":"l2_cache"},)"
         R"({"name":"cp","type":"command_processor"},)"
         R"({"name":"se0","type":"shader_engine","children":[)"
         R"({"name":")" +
         cu_range +
         R"(","type":"compute_unit","config":[)"
         R"({"key":"num_wf_slots","value":"80"},)"
         R"({"key":"sgprs_per_wf","value":"128"},)"
         R"({"key":"vgprs_per_wf","value":"1024"},)"
         R"({"key":"lds_size_kb","value":"160"})"
         R"(]}]}]}]},"links":[)" +
         links + R"(]}})";
}

class HostSdmaQueueForTest {
public:
  explicit HostSdmaQueueForTest(Gfx1250Sim &sim) : sim_(sim) {
    sim_.memory->set_passthrough(true);

    amdgpu::HwQueue queue{};
    queue.process_id = kProcessId;
    queue.queue_id = kQueueId;
    queue.ring_base_va = reinterpret_cast<uint64_t>(ring_.data());
    queue.ring_size = static_cast<uint32_t>(ring_.size() * sizeof(uint32_t));
    queue.read_ptr_va = reinterpret_cast<uint64_t>(&read_idx_);
    queue.write_ptr_va = reinterpret_cast<uint64_t>(&write_idx_);
    queue.doorbell_base = doorbells_.data();
    queue.doorbell_offset = 0;
    queue.host_accessible = true;
    queue.is_sdma = true;
    sim_.cp()->register_queue(std::move(queue));
  }

  ~HostSdmaQueueForTest() { sim_.cp()->unregister_queue(kQueueId, kProcessId); }

  uint32_t *ring() { return ring_.data(); }

  void submit(uint32_t dwords) {
    uint64_t write_idx = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(write_idx_).store(write_idx, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_idx, std::memory_order_release);
    sim_.engine->schedule_event_now(sim_.cp()->doorbell_event());
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(read_idx_).load(std::memory_order_acquire);
  }

private:
  static constexpr uint32_t kProcessId = 0;
  static constexpr uint32_t kQueueId = 1250;

  Gfx1250Sim &sim_;
  std::array<uint32_t, 64> ring_{};
  alignas(8) uint64_t read_idx_ = 0;
  alignas(8) uint64_t write_idx_ = 0;
  std::array<uint64_t, 1> doorbells_{};
};

class TranslatedSdmaQueueForTest {
public:
  explicit TranslatedSdmaQueueForTest(Gfx1250Sim &sim) : sim_(sim), process_(kProcessId) {
    sim_.memory->register_process(kProcessId, &process_.page_table_, &process_.page_table_mutex_);
    process_.map_pages(kRingVa, ring_.data(), ring_.size() * sizeof(ring_[0]));
    process_.map_pages(kQueueStateVa, queue_state_.data(),
                       queue_state_.size() * sizeof(queue_state_[0]));
    process_.map_pages(kSrcVa, src_.data(), src_.size());
    process_.map_pages(kDstVa, dst_.data(), dst_.size());
    process_.map_pages(kSignalVa, signal_.data(), signal_.size() * sizeof(signal_[0]));
    process_.map_pages(kPollVa, poll_.data(), poll_.size() * sizeof(poll_[0]));

    amdgpu::HwQueue queue{};
    queue.process_id = kProcessId;
    queue.queue_id = kQueueId;
    queue.ring_base_va = kRingVa;
    queue.ring_size = static_cast<uint32_t>(ring_.size() * sizeof(ring_[0]));
    queue.read_ptr_va = kQueueStateVa;
    queue.write_ptr_va = kQueueStateVa + sizeof(queue_state_[0]);
    queue.doorbell_base = doorbells_.data();
    queue.doorbell_offset = 0;
    queue.host_accessible = true;
    queue.is_sdma = true;
    sim_.cp()->register_queue(std::move(queue));
  }

  ~TranslatedSdmaQueueForTest() {
    sim_.cp()->unregister_queue(kQueueId, kProcessId);
    sim_.memory->unregister_process(kProcessId);
  }

  uint32_t *ring() { return ring_.data(); }
  uint8_t *src() { return src_.data(); }
  uint8_t *dst() { return dst_.data(); }
  int64_t &signal_value() { return signal_[0]; }
  uint64_t &poll_value() { return poll_[0]; }

  uint64_t src_va() const { return kSrcVa; }
  uint64_t dst_va() const { return kDstVa; }
  uint64_t signal_va() const { return kSignalVa; }
  uint64_t poll_va() const { return kPollVa; }

  void submit(uint32_t dwords) {
    uint64_t write_idx = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(queue_state_[1]).store(write_idx, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_idx, std::memory_order_release);
    sim_.engine->schedule_event_now(sim_.cp()->doorbell_event());
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(queue_state_[0]).load(std::memory_order_acquire);
  }

private:
  static constexpr uint32_t kProcessId = 1251;
  static constexpr uint32_t kQueueId = 1251;
  static constexpr uint64_t kRingVa = 0x1000'0000'0000ULL;
  static constexpr uint64_t kQueueStateVa = 0x1000'0000'1000ULL;
  static constexpr uint64_t kSrcVa = 0x1000'0000'2000ULL;
  static constexpr uint64_t kDstVa = 0x1000'0000'3000ULL;
  static constexpr uint64_t kSignalVa = 0x1000'0000'4000ULL;
  static constexpr uint64_t kPollVa = 0x1000'0000'5000ULL;

  Gfx1250Sim &sim_;
  KfdProcess process_;
  alignas(4096) std::array<uint32_t, 1024> ring_{};
  alignas(4096) std::array<uint64_t, 512> queue_state_{};
  alignas(4096) std::array<uint8_t, 4096> src_{};
  alignas(4096) std::array<uint8_t, 4096> dst_{};
  alignas(4096) std::array<int64_t, 512> signal_{};
  alignas(4096) std::array<uint64_t, 512> poll_{};
  std::array<uint64_t, 1> doorbells_{};
};

void write_sdma_qword_va(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, uint64_t va) {
  packet[lo_dw] = static_cast<uint32_t>(va) & ~0x7u;
  packet[hi_dw] = static_cast<uint32_t>(va >> 32);
}

void write_sdma_qword_address(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, const void *addr) {
  write_sdma_qword_va(packet, lo_dw, hi_dw, reinterpret_cast<uintptr_t>(addr));
}

void step_until_halted(simdojo::SimulationEngine &engine, amdgpu::ComputeUnitCore &cu,
                       uint32_t max_steps = 10000) {
  for (uint32_t i = 0; i < max_steps && engine.step(); ++i) {
    if (cu.num_wfs() == 0)
      continue;
    bool all_halted = true;
    for (uint32_t w = 0; w < cu.num_wfs(); ++w) {
      if (cu.wf(w) && !cu.wf(w)->is_halted()) {
        all_halted = false;
        break;
      }
    }
    if (all_halted)
      return;
  }
}

void step_until_xcd_halted(Gfx1250Sim &sim, uint32_t max_steps = 10000) {
  auto all_halted = [&]() {
    for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
      auto *se = sim.xcd()->shader_engine(se_idx);
      for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
        auto *cu = se->compute_unit(cu_idx);
        for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
          auto *wf = cu->wf(wf_idx);
          if (wf && wf->sgpr_alloc().count > 0 && !wf->is_halted())
            return false;
        }
      }
    }
    return true;
  };

  for (uint32_t i = 0; i < max_steps && sim.engine->step(); ++i) {
    if (all_halted())
      return;
  }
}

amdgpu::Wavefront *dispatch_one_wave(Gfx1250Sim &sim, const uint32_t *code, size_t num_words,
                                     uint32_t vgprs = 32) {
  uint64_t kernel_object = sim.write_kernel(0x10000, code, num_words, 104, vgprs);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());
  if (sim.cu()->num_wfs() == 0)
    return nullptr;
  return sim.cu()->wf(0);
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
  MulF32 = 3,
  MulDx9ZeroF32 = 7,
  CndmaskB32 = 9,
  FmaF32 = 19,
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

void append_instruction(std::vector<uint32_t> &code, uint32_t word) { code.push_back(word); }

void write_wave_sgpr(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf, uint32_t reg,
                     uint32_t value) {
  cu.write_sgpr(wf.sgpr_alloc().base + reg, value);
}

uint64_t read_wave_sgpr64(const amdgpu::ComputeUnitCore &cu, const amdgpu::Wavefront &wf,
                          uint32_t reg) {
  uint32_t lo = cu.read_sgpr(wf.sgpr_alloc().base + reg);
  uint32_t hi = cu.read_sgpr(wf.sgpr_alloc().base + reg + 1);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

uint32_t read_wave_sgpr(const amdgpu::ComputeUnitCore &cu, const amdgpu::Wavefront &wf,
                        uint32_t reg) {
  return cu.read_sgpr(wf.sgpr_alloc().base + reg);
}

void write_tensor_dma_d0(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wf, uint32_t reg,
                         uint64_t global_addr, uint32_t lds_base = 0) {
  write_wave_sgpr(cu, wf, reg + 0, 1);
  write_wave_sgpr(cu, wf, reg + 1, lds_base);
  write_wave_sgpr(cu, wf, reg + 2, static_cast<uint32_t>(global_addr));
  write_wave_sgpr(cu, wf, reg + 3,
                  static_cast<uint32_t>((global_addr >> 32) & 0x01ffffffu) | 0x80000000u);
}

void write_global_u32(amdgpu::GpuMemory &memory, uint64_t addr, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte)
    memory.write8(addr + byte, static_cast<uint8_t>(value >> (byte * 8)));
}

uint32_t read_global_u32(amdgpu::GpuMemory &memory, uint64_t addr) {
  uint32_t value = 0;
  for (uint32_t byte = 0; byte < 4; ++byte)
    value |= static_cast<uint32_t>(memory.read8(addr + byte)) << (byte * 8);
  return value;
}

std::unique_ptr<Instruction> decode_gfx1250(const std::array<uint32_t, 3> &words,
                                            std::string_view expected_mnemonic) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  if (!decoder) {
    ADD_FAILURE() << "Decoder::create() returned nullptr for gfx1250";
    return nullptr;
  }

  std::unique_ptr<Instruction> inst(decoder->decode(words.data()));
  if (!inst) {
    ADD_FAILURE() << "decode() returned nullptr for gfx1250 instruction";
    return nullptr;
  }

  EXPECT_EQ(inst->mnemonic(), expected_mnemonic);
  EXPECT_EQ(inst->size(), static_cast<int>(words.size() * sizeof(words[0])));
  return inst;
}

template <typename T> void append_bytes(std::vector<uint8_t> &bytes, const T &value) {
  auto *src = reinterpret_cast<const uint8_t *>(&value);
  bytes.insert(bytes.end(), src, src + sizeof(T));
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<uint8_t> make_minimal_gfx1250_elf() {
  constexpr uint8_t text[] = {0x00, 0x00, 0xB0, 0xBF};
  constexpr char shstrtab[] = "\0.text\0.shstrtab\0";
  constexpr uint32_t text_name = 1;
  constexpr uint32_t shstrtab_name = 7;

  std::vector<uint8_t> image(sizeof(Elf64_Ehdr), 0);
  const size_t text_offset = image.size();
  image.insert(image.end(), std::begin(text), std::end(text));
  const size_t shstrtab_offset = image.size();
  image.insert(image.end(), std::begin(shstrtab), std::end(shstrtab));
  image.resize(align_up(image.size(), alignof(Elf64_Shdr)), 0);
  const size_t shoff = image.size();

  Elf64_Shdr null_shdr{};
  append_bytes(image, null_shdr);

  Elf64_Shdr text_shdr{};
  text_shdr.sh_name = text_name;
  text_shdr.sh_type = SHT_PROGBITS;
  text_shdr.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  text_shdr.sh_offset = text_offset;
  text_shdr.sh_size = sizeof(text);
  text_shdr.sh_addralign = alignof(uint32_t);
  append_bytes(image, text_shdr);

  Elf64_Shdr shstrtab_shdr{};
  shstrtab_shdr.sh_name = shstrtab_name;
  shstrtab_shdr.sh_type = SHT_STRTAB;
  shstrtab_shdr.sh_offset = shstrtab_offset;
  shstrtab_shdr.sh_size = sizeof(shstrtab);
  shstrtab_shdr.sh_addralign = 1;
  append_bytes(image, shstrtab_shdr);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = 1;
  ehdr.e_ident[EI_VERSION] = 1;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 3;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  return image;
}

TEST(Gfx1250ConfigTest, ConfigLoadsTopology) {
  auto loaded = config::load_config(kGfx1250ConfigPath, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  ASSERT_NE(soc, nullptr);
  EXPECT_EQ(soc->arch(), ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(config::parse_arch("gfx1250"), ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_STREQ(config::arch_to_string(ROCJITSU_CODE_ARCH_GFX1250), "gfx1250");

  EXPECT_TRUE(loaded.device.present);
  EXPECT_EQ(loaded.device.gfx_target_version, 120500u);
  EXPECT_EQ(loaded.device.simd_count, 1024u);
  EXPECT_EQ(loaded.device.max_waves_per_simd, kGfx1250MaxWavesPerSimd);
  EXPECT_EQ(loaded.device.num_shader_arrays_per_engine, 2u);
  EXPECT_EQ(loaded.device.num_cu_per_sh, 4u);
  EXPECT_EQ(loaded.device.simd_per_cu, kGfx1250SimdsPerCu);
  EXPECT_EQ(loaded.device.wave_front_size, 32u);
  EXPECT_EQ(loaded.device.lds_size_kb, kGfx1250LdsSizeKb);
  EXPECT_EQ(loaded.device.mem_width, 8192u);
  EXPECT_EQ(loaded.device.marketing_name, "gfx1250");

  EXPECT_EQ(soc->num_xcds(), 8u);
  EXPECT_EQ(soc->num_iods(), 2u);
  EXPECT_EQ(soc->xcd(0)->num_shader_engines(), 4u);
  EXPECT_EQ(soc->xcd(0)->shader_engine(0)->num_compute_units(), 8u);
  auto *cu = soc->xcd(0)->shader_engine(0)->compute_unit(0);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->wf_size(), 32u);
  EXPECT_EQ(cu->config().num_wf_slots, kGfx1250WaveSlotsPerCu);
  EXPECT_EQ(cu->config().sgprs_per_wf, kGfx1250ScalarSlots);
  EXPECT_EQ(cu->config().vgprs_per_wf, kGfx1250Wave32VgprAllocation);
  EXPECT_EQ(cu->config().lds_size_kb, kGfx1250LdsSizeKb);
  EXPECT_EQ(soc->xcd(0)->command_processor()->vgpr_granularity(), kGfx1250VgprEncodingGranule);
  EXPECT_EQ(soc->xcd(0)->command_processor()->sdma_packet_dialect(),
            amdgpu::SdmaPacketDialect::Gfx1250);
}

TEST(Gfx1250SdmaTest, PollMem64WaitsForFull64BitCondition) {
  Gfx1250Sim sim;
  HostSdmaQueueForTest queue(sim);
  alignas(8) uint64_t value = 1;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28); // 64-bit equal poll.
  write_sdma_qword_address(packet, 1, 2, &value);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(value).store(0, std::memory_order_release);
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 8u * sizeof(uint32_t));
}

TEST(Gfx1250SdmaTest, Fence64WritesFull64BitValue) {
  Gfx1250Sim sim;
  HostSdmaQueueForTest queue(sim);
  alignas(8) uint64_t value = 0;
  constexpr uint64_t kFenceValue = 0x12345678ABCDEF01ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpFence | (kSdmaSubopFence64 << 8) | (3u << 16);
  write_sdma_qword_address(packet, 1, 2, &value);
  packet[3] = static_cast<uint32_t>(kFenceValue);
  packet[4] = static_cast<uint32_t>(kFenceValue >> 32);

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 5u * sizeof(uint32_t));
  EXPECT_EQ(std::atomic_ref<uint64_t>(value).load(std::memory_order_acquire), kFenceValue);
}

// A wrong GCR packet size silently desyncs the SDMA ring read pointer and
// corrupts the following packet. Emit OP_GCR followed by a 32-bit FENCE and
// assert both the read pointer advance and that the FENCE decoded at the right
// boundary (its sentinel lands). gfx11/12 GCR is 5 dwords; gfx1250 is 6.
TEST(Gfx1250SdmaTest, GcrPacketSizeMatchesDialectAndKeepsRingInSync) {
  constexpr uint32_t kGcrLegacySize = 5;
  constexpr uint32_t kGcrGfx1250Size = 6;
  constexpr uint32_t kFenceSize = 4;
  constexpr uint32_t kFenceSentinel = 0xC0FFEE11u;
  // GL2 invalidate control bit position differs by dialect; setting it exercises
  // a realistic invalidate GCR but does not affect the decoded packet size.
  constexpr uint32_t kLegacyGl2InvControlDw = 2;
  constexpr uint32_t kLegacyGl2InvBit = 1u << 30;
  constexpr uint32_t kGfx1250Gl2InvControlDw = 3;
  constexpr uint32_t kGfx1250Gl2InvBit = 1u << 14;

  auto run_dialect = [kFenceSentinel](amdgpu::SdmaPacketDialect dialect, uint32_t gcr_size,
                                      uint32_t control_dw, uint32_t control_bit) {
    Gfx1250Sim sim;
    sim.cp()->set_sdma_packet_dialect(dialect);
    HostSdmaQueueForTest queue(sim);
    alignas(8) uint32_t fence_value = 0;

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    packet[control_dw] = control_bit;

    uint32_t *fence = packet + gcr_size;
    fence[0] = kSdmaOpFence; // 32-bit fence (sub_op 0).
    write_sdma_qword_address(fence, 1, 2, &fence_value);
    fence[3] = kFenceSentinel;

    queue.submit(gcr_size + kFenceSize);
    ASSERT_TRUE(sim.engine->step());
    EXPECT_EQ(queue.read_idx(), (gcr_size + kFenceSize) * sizeof(uint32_t));
    EXPECT_EQ(std::atomic_ref<uint32_t>(fence_value).load(std::memory_order_acquire),
              kFenceSentinel);
  };

  run_dialect(amdgpu::SdmaPacketDialect::Gfx11Plus, kGcrLegacySize, kLegacyGl2InvControlDw,
              kLegacyGl2InvBit);
  run_dialect(amdgpu::SdmaPacketDialect::Gfx1250, kGcrGfx1250Size, kGfx1250Gl2InvControlDw,
              kGfx1250Gl2InvBit);
}

// SDMA writes go straight to backing while L2 may still hold a dirty line that
// overlaps the destination (e.g. left by a prior K$ writeback). The post-write
// cache maintenance must not write that stale line back over the SDMA result.
// The fix flushes the caches before the direct write, so the dirty line is
// published first and the SDMA data supersedes it. Regression for that ordering.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  // The config-driven topology build wires the XCD's L2 into the CP, so the SDMA
  // cache maintenance operates on the same L2 instance we dirty below.
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Seed a dirty L2 line overlapping the destination, without touching backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, static_cast<int>(kStaleWord & 0xFF), sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  // CONST_FILL the destination line with a different byte pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());

  // Backing must reflect the SDMA fill, not the stale cached line.
  EXPECT_EQ(sim.memory->read32(queue.dst_va(), kProcessId), kFillWord);
  EXPECT_NE(sim.memory->read32(queue.dst_va(), kProcessId), kStaleWord);
}

// Same ordering hazard as above, but for a dirty scalar L1 (K$) line rather than
// an L2 line. A CU can hold a dirty K$ line overlapping an SDMA destination. The
// pre-write maintenance must write the K$ line back (through L2 to backing)
// before the direct SDMA write, so the SDMA result is not later clobbered when
// the stale scalar line is flushed. Regression for K$ inclusion in the flush.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingDirtyScalarL1Line) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  ASSERT_NE(cu, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Dirty a K$ line overlapping the SDMA destination via a scalar store. This
  // leaves the line dirty in K$ (write-back), not yet in L2 or backing.
  cu->l1_scalar().store(queue.dst_va(), /*num_dwords=*/1, &kStaleWord, kProcessId);

  // CONST_FILL the destination line with a different pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());

  // Force any still-resident dirty K$ line out to backing, mimicking a later
  // acquire/release flush. With the fix the K$ line was already published and
  // invalidated before the SDMA write, so this does not resurrect stale data.
  cu->flush_l1(kProcessId);
  if (auto *l2 = sim.xcd()->l2_cache())
    l2->flush_all();

  // Backing must reflect the SDMA fill, not the stale scalar line.
  EXPECT_EQ(sim.memory->read32(queue.dst_va(), kProcessId), kFillWord);
  EXPECT_NE(sim.memory->read32(queue.dst_va(), kProcessId), kStaleWord);
}

// OP_TIMESTAMP is a direct backing-store write like COPY/FENCE/CONST_FILL, so it
// has the same clobber hazard: a dirty cached line overlapping the timestamp
// address must be published before the store, not written out over it by a later
// flush. Seed a dirty L2 line at the timestamp address, issue OP_TIMESTAMP, then
// force a flush; the stored timestamp must survive (stale word gone, value set).
TEST(Gfx1250SdmaTest, TimestampSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint64_t kStaleQword = 0x1111111111111111ULL;

  // Seed a dirty L2 line overlapping the timestamp destination, without touching
  // backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, 0x11, sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpTimestamp;
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());

  queue.submit(3); // TIMESTAMP is 3 dwords.
  ASSERT_TRUE(sim.engine->step());

  // Force any still-resident dirty line out, mimicking a later flush.
  if (auto *xl2 = sim.xcd()->l2_cache())
    xl2->flush_all();

  // The timestamp value is nondeterministic, but it must not be the stale word
  // and must be a plausible nonzero nanosecond count.
  const uint64_t stored = sim.memory->read64(queue.dst_va(), kProcessId);
  EXPECT_NE(stored, kStaleQword);
  EXPECT_NE(stored, 0u);
}

// The GCR decoder now distinguishes GL2 writeback (publish dirty lines),
// invalidate/discard (drop without writeback), and no-op (no GL2 bits). This is
// the data-loss distinction the PR protects. Dirty an L2 line, then issue each
// GCR flavor and observe whether the dirty data reaches backing.
TEST(Gfx1250SdmaTest, GcrWritebackPublishesInvalidateDropsNoopKeeps) {
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kDirtyWord = 0x33333333u;
  constexpr uint32_t kBackingWord = 0x44444444u;
  // gfx1250 GCR control dword (DW3) bit positions.
  constexpr uint32_t kControlDw = 3;
  constexpr uint32_t kGl2InvBit = 1u << 14;
  constexpr uint32_t kGl2WbBit = 1u << 15;

  // Outcome of a GCR flavor: the value in backing (read directly through the
  // page table) and the value seen through L2 (which returns the resident dirty
  // line if still present, or re-fetches backing if the line was dropped).
  struct GcrOutcome {
    uint32_t backing = 0;
    uint32_t via_l2 = 0;
  };

  enum class GcrKind { WritebackOnly, InvalidateOnly, Noop };
  // Void return so a missing L2 is a fatal guard (ASSERT_*) before we deref it.
  auto run = [&](GcrKind kind, GcrOutcome &out) {
    Gfx1250Sim sim;
    auto *l2 = sim.xcd()->l2_cache();
    ASSERT_NE(l2, nullptr);
    TranslatedSdmaQueueForTest queue(sim);

    // Put a known value in backing, then a different dirty value in L2 on top.
    for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
      sim.memory->write8(queue.dst_va() + i, static_cast<uint8_t>((kBackingWord >> (i * 8)) & 0xFF),
                         kProcessId);
    uint8_t dirty_line[amdgpu::L2Cache::LINE_SIZE];
    std::memset(dirty_line, static_cast<int>(kDirtyWord & 0xFF), sizeof(dirty_line));
    l2->writeback_line(queue.dst_va(), dirty_line, amdgpu::Mtype::RW, kProcessId);

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    if (kind == GcrKind::WritebackOnly)
      packet[kControlDw] = kGl2WbBit;
    else if (kind == GcrKind::InvalidateOnly)
      packet[kControlDw] = kGl2InvBit;
    else
      packet[kControlDw] = 0; // no GL2 bits: no-op.

    queue.submit(6); // gfx1250 GCR is 6 dwords.
    EXPECT_TRUE(sim.engine->step());

    out.backing = sim.memory->read32(queue.dst_va(), kProcessId);
    // Read back through L2: a still-resident dirty line returns kDirtyWord; a
    // dropped line re-fetches from backing on the miss.
    uint32_t l2_word = 0;
    l2->read(queue.dst_va(), reinterpret_cast<uint8_t *>(&l2_word), sizeof(l2_word),
             amdgpu::Mtype::RW, kProcessId);
    out.via_l2 = l2_word;
  };

  // Writeback publishes the dirty line to backing.
  GcrOutcome wb;
  run(GcrKind::WritebackOnly, wb);
  EXPECT_EQ(wb.backing, kDirtyWord);
  // Invalidate/discard drops the dirty line without writeback; backing keeps its
  // original value and the line is no longer resident.
  GcrOutcome inv;
  run(GcrKind::InvalidateOnly, inv);
  EXPECT_EQ(inv.backing, kBackingWord);
  EXPECT_EQ(inv.via_l2, kBackingWord); // line dropped → L2 re-fetches backing.
  // No GL2 bits: no cache maintenance at all. Backing is untouched and the dirty
  // line stays resident in L2 (this is what distinguishes no-op from
  // invalidate-only: an incorrect invalidate would drop the line here too).
  GcrOutcome noop;
  run(GcrKind::Noop, noop);
  EXPECT_EQ(noop.backing, kBackingWord);
  EXPECT_EQ(noop.via_l2, kDirtyWord); // dirty line still resident in L2.
}

TEST(Gfx1250SdmaTest, CopyWaitSignalResolvesTranslatedAddresses) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x5a);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, queue.signal_va());
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 19u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 4);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedWaitAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedWaitVa = 0x2000'0000'0000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xa5);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, kUnmappedWaitVa);
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());

  queue.submit(19);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedDstDoesNotAdvanceOrSignal) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'2000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x3c);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, kUnmappedDstVa);
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, queue.signal_va());
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedSignalDoesNotAdvanceOrCopy) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedSignalVa = 0x2000'0000'4000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x69);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, kUnmappedSignalVa);
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyLinearUnresolvedDstDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'3000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xc3);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kUnmappedDstVa);

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearNpdBitDoesNotDecodeAsBroadcast) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x4d);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 28);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, PollMem64ResolvesTranslatedAddress) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.poll_value() = 1;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, queue.poll_va());
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(queue.poll_value()).store(0, std::memory_order_release);
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 8u * sizeof(uint32_t));
}

TEST(Gfx1250SdmaTest, PollMem64UnresolvedAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint64_t kUnmappedPollVa = 0x2000'0000'1000ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, kUnmappedPollVa);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250ExecutionTest, DivScaleWritesExplicitSdstMask) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  constexpr uint32_t kOne = 0x3f800000u;
  constexpr uint32_t kTwoTo8 = 0x43800000u;
  constexpr uint32_t kTwoTo100 = 0x71800000u;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto write_vgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_vgpr(vgpr_base + reg, kLane, value);
  };
  auto read_vgpr = [&](uint32_t reg) { return cu->read_vgpr(vgpr_base + reg, kLane); };
  auto write_sgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  };

  write_vgpr(1, kOne);
  write_vgpr(2, kTwoTo100);
  wf->set_vcc(0x5a5a5a5au);
  const std::array<uint32_t, 2> null_sdst_words = {
      0xd6fc7c00u, 0x040a0301u}; // v_div_scale_f32 v0, null, v1, v1, v2
  gfx1250::VDivScaleF32Vop3SdstEnc null_sdst(null_sdst_words.data());
  null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x5a5a5a5au);
  EXPECT_EQ(read_vgpr(0), 0x5f800000u); // 2^64

  write_sgpr(7, kTwoTo8);
  write_vgpr(3, kOne);
  wf->set_vcc(0xa5a5a5a5u);
  const std::array<uint32_t, 2> normal_null_sdst_words = {
      0xd6fc7c09u, 0x040c0e07u}; // v_div_scale_f32 v9, null, s7, s7, v3
  gfx1250::VDivScaleF32Vop3SdstEnc normal_null_sdst(normal_null_sdst_words.data());
  normal_null_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0xa5a5a5a5u);
  EXPECT_EQ(read_vgpr(9), kTwoTo8);

  write_vgpr(4, kOne);
  write_vgpr(5, kTwoTo100);
  wf->set_vcc(0);
  const std::array<uint32_t, 2> vcc_sdst_words = {
      0xd6fc6a03u, 0x04160904u}; // v_div_scale_f32 v3, vcc_lo, v4, v4, v5
  gfx1250::VDivScaleF32Vop3SdstEnc vcc_sdst(vcc_sdst_words.data());
  vcc_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 1u);
  EXPECT_EQ(read_vgpr(3), 0x5f800000u);

  write_vgpr(7, kOne);
  write_vgpr(8, kTwoTo100);
  write_sgpr(3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);
  const std::array<uint32_t, 2> sgpr_sdst_words = {
      0xd6fc0206u, 0x04220f07u}; // v_div_scale_f32 v6, s2, v7, v7, v8
  gfx1250::VDivScaleF32Vop3SdstEnc sgpr_sdst(sgpr_sdst_words.data());
  sgpr_sdst.execute_impl(*wf);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x12345679u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(read_vgpr(6), 0x5f800000u);
}

TEST(Gfx1250ExecutionTest, VMovB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F023880u}; // v_mov_b16_e32 v1.h, 0
  gfx1250::VMovB16Vop1 high_half_mov(words.data());
  high_half_mov.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x00005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VNotB16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x000000FFu);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x7F02D300u}; // v_not_b16_e32 v1.h, v0.l
  gfx1250::VNotB16Vop1 high_half_not(words.data());
  high_half_not.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0xFF005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, VAddF16HighVdstMergesIntoLowPhysicalVgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0xAAAA5555u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0xDEADBEEFu);

  const std::array<uint32_t, 1> words = {0x65020500u}; // v_add_f16_e32 v1.h, v0.l, v2.l
  gfx1250::VAddF16Vop2 high_half_add(words.data());
  high_half_add.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x40005555u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250ExecutionTest, IreeF16ReductionTailKeepsLane31Sum) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffffffffu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  const uint32_t packed_1_2 = 0x40003c00u;
  const uint32_t packed_3_4 = 0x44004200u;
  const uint32_t packed_5_6 = 0x46004500u;
  const uint32_t packed_7_8 = 0x48004700u;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vgpr_base + 10, lane, packed_7_8);
    cu->write_vgpr(vgpr_base + 11, lane, packed_1_2);
    cu->write_vgpr(vgpr_base + 16, lane, packed_5_6);
    cu->write_vgpr(vgpr_base + 17, lane, packed_3_4);
  }

  const std::array<std::array<uint32_t, 3>, 20> words = {{
      {0x64021680u, 0, 0},                     // v_add_f16_e32 v1, 0, v11
      {0x32041690u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v11
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042290u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v17
      {0x64022301u, 0, 0},                     // v_add_f16_e32 v1, v1, v17
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32042090u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v16
      {0x64022101u, 0, 0},                     // v_add_f16_e32 v1, v1, v16
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0x32041490u, 0, 0},                     // v_lshrrev_b32_e32 v2, 16, v10
      {0x64021501u, 0, 0},                     // v_add_f16_e32 v1, v1, v10
      {0x64020501u, 0, 0},                     // v_add_f16_e32 v1, v1, v2
      {0xd5320001u, 0x000202fau, 0xff08b101u}, // quad_perm:[1,0,3,2]
      {0xd5320001u, 0x000202fau, 0xff084e01u}, // quad_perm:[2,3,0,1]
      {0xd5320001u, 0x000202fau, 0xff094101u}, // row_half_mirror
      {0xd5320001u, 0x000202fau, 0xff094001u}, // row_mirror
      {0xd65c0802u, 0x03058301u, 0},           // v_permlanex16_b32 v2, v1, -1, -1
      {0x64020302u, 0, 0},                     // v_add_f16_e32 v1, v2, v1
      {0xd7600000u, 0x02013f01u, 0},           // v_readlane_b32 s0, v1, 31
      {0xa4808000u, 0, 0},                     // s_add_f16 s0, s0, 0
  }};

  for (const auto &inst_words : words) {
    std::unique_ptr<Instruction> inst(decoder->decode(inst_words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  }

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 0) & 0xffffu, 0x6480u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop3HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);

  const std::array<uint32_t, 2> words = {
      0xD5364001u, // v_fmac_f16 v1.h, v0.l, v2.l
      0x02020500u,
  };
  gfx1250::VFmacF16Vop3 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
}

TEST(Gfx1250ExecutionTest, VFmacF16Vop2HighVdstUsesHighHalfAddend) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 0, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 1, kLane, 0x40003C00u);
  cu->write_vgpr(vgpr_base + 2, kLane, 0x00003C00u);
  cu->write_vgpr(vgpr_base + 129, kLane, 0x3C003C00u);

  const std::array<uint32_t, 1> words = {0x6D020500u}; // v_fmac_f16_e32 v1.h, v0.l, v2.l
  gfx1250::VFmacF16Vop2 high_half_fmac(words.data());
  high_half_fmac.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 1, kLane), 0x42003C00u);
  EXPECT_EQ(cu->read_vgpr(vgpr_base + 129, kLane), 0x3C003C00u);
}

TEST(Gfx1250ExecutionTest, VMadU32LiteralTimesScalarAddsVector) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kLane = 0;
  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  write_wave_sgpr(*cu, *wf, 3, 1);
  cu->write_vgpr(vgpr_base + 4, kLane, 0x24u);

  const std::array<uint32_t, 3> words = {
      0xD6350004u, // v_mad_u32 v4, 0x48, s3, v4
      0x041006FFu,
      0x00000048u,
  };
  gfx1250::VMadU32Vop3 mad(words.data());
  mad.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr(vgpr_base + 4, kLane), 0x6Cu);
}

TEST(Gfx1250ExecutionTest, VCmpGtU32Wave32ExplicitSdstPreservesHighSgpr) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base + 4, 0, 3u);
  cu->write_vgpr(vgpr_base + 4, 1, 5u);
  write_wave_sgpr(*cu, *wf, 2, 0xaaaaaaaau);
  write_wave_sgpr(*cu, *wf, 3, 0xfefefefeu);
  wf->set_vcc(0x12345678u);

  const std::array<uint32_t, 2> words = {
      0xD44C0002u, // v_cmp_gt_u32_e64 s2, 4, v4
      0x02020884u,
  };
  gfx1250::VCmpGtU32Vop3 cmp(words.data());
  cmp.execute_impl(*wf);

  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 2), 0x1u);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf, 3), 0xfefefefeu);
  EXPECT_EQ(wf->vcc(), 0x12345678u);
}

TEST(Gfx1250ExecutionTest, Wave32ScalarVccHiWritePreservesUpperHalf) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0xffff0000u);
  wf->set_vcc(0);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const uint32_t words[] = {0x8c6b7e6bu, 0}; // s_or_b32 vcc_hi, vcc_hi, exec_lo
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_or_b32");
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(wf->vcc(), 0xffff0000'00000000ULL);
}

TEST(Gfx1250ExecutionTest, ClusterLoadsDecodeAndPopulateVectorMemState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_m0(0xffff0003u);

  constexpr uint64_t kGlobalBase = 0x180000u;
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  auto write_vgpr_lane_offsets = [&](uint32_t reg, uint32_t stride) {
    cu->write_vgpr(vgpr_base + reg, 0, 0);
    cu->write_vgpr(vgpr_base + reg, 1, stride);
  };
  auto write_vgpr_lane_indices = [&](uint32_t reg) {
    cu->write_vgpr(vgpr_base + reg, 0, 0);
    cu->write_vgpr(vgpr_base + reg, 1, 1);
  };

  struct LoadCase {
    std::array<uint32_t, 3> words;
    std::string_view mnemonic;
    uint32_t num_elems;
    uint32_t dst_reg;
    bool scale_offset;
  };

  const LoadCase load_cases[] = {
      {{0xEE19C000u, 0x00000001u, 0x00000000u}, "cluster_load_b32", 1, 1, false},
      {{0xEE1A0000u, 0x00000000u, 0x00000002u}, "cluster_load_b64", 2, 0, false},
      {{0xEE1A4000u, 0x00010002u, 0x00000000u}, "cluster_load_b128", 4, 2, true},
  };

  for (const LoadCase &tc : load_cases) {
    if (tc.scale_offset)
      write_vgpr_lane_indices(0);
    else
      write_vgpr_lane_offsets(tc.words[2] & 0xffu, 16);

    auto inst = decode_gfx1250(tc.words, tc.mnemonic);
    ASSERT_NE(inst, nullptr);
    inst->execute(*inst, wf);
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
    EXPECT_EQ(state->dst_reg_base, vgpr_base + tc.dst_reg);
    EXPECT_EQ(state->elem_size, 4u);
    EXPECT_EQ(state->num_elems, tc.num_elems);
    EXPECT_TRUE(state->is_load);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::LOADCNT);
    EXPECT_FALSE(state->lds_dst);
    EXPECT_FALSE(state->cluster_multicast);
    EXPECT_TRUE(state->request_force_l1_bypass);
    EXPECT_EQ(state->lane_mask, 0x3u);
    EXPECT_EQ(state->per_lane_addr[0], kGlobalBase);
    EXPECT_EQ(state->per_lane_addr[1], kGlobalBase + 16);
  }

  struct AsyncCase {
    std::array<uint32_t, 3> words;
    std::string_view mnemonic;
    uint32_t elem_size;
    uint32_t num_elems;
    uint32_t lds_addr_reg;
    uint32_t global_stride;
    uint32_t lds_stride;
    uint32_t offset;
    bool scale_offset;
    bool cluster_multicast;
  };

  const AsyncCase async_cases[] = {
      {{0xEE1A8000u, 0x00000000u, 0x00000000u},
       "cluster_load_async_to_lds_b8",
       1,
       1,
       0,
       1,
       1,
       0,
       false,
       true},
      {{0xEE180000u, 0x00000000u, 0x00000800u},
       "global_load_async_to_lds_b32",
       4,
       1,
       0,
       16,
       16,
       8,
       false,
       false},
      {{0xEE1AC000u, 0x00000000u, 0x00000800u},
       "cluster_load_async_to_lds_b32",
       4,
       1,
       0,
       16,
       16,
       8,
       false,
       true},
      {{0xEE1B0000u, 0x00000004u, 0x00000004u},
       "cluster_load_async_to_lds_b64",
       4,
       2,
       4,
       16,
       16,
       0,
       false,
       true},
      {{0xEE1B4000u, 0x00010001u, 0x00000000u},
       "cluster_load_async_to_lds_b128",
       4,
       4,
       1,
       16,
       16,
       0,
       true,
       true},
  };

  for (const AsyncCase &tc : async_cases) {
    if (tc.scale_offset) {
      write_vgpr_lane_indices(0);
      write_vgpr_lane_offsets(tc.lds_addr_reg, 16);
    } else {
      write_vgpr_lane_offsets(tc.lds_addr_reg, tc.global_stride);
    }

    auto inst = decode_gfx1250(tc.words, tc.mnemonic);
    ASSERT_NE(inst, nullptr);
    inst->execute(*inst, wf);
    auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
    EXPECT_EQ(state->elem_size, tc.elem_size);
    EXPECT_EQ(state->num_elems, tc.num_elems);
    EXPECT_TRUE(state->is_load);
    EXPECT_TRUE(state->lds_dst);
    EXPECT_TRUE(state->lds_per_lane_addr);
    EXPECT_EQ(state->lds_base, wf->lds_base());
    EXPECT_EQ(state->cluster_multicast, tc.cluster_multicast);
    EXPECT_EQ(state->cluster_mcast_mask, tc.cluster_multicast ? 0x3u : 0u);
    EXPECT_EQ(state->request_force_l1_bypass, tc.cluster_multicast);
    EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
    EXPECT_EQ(state->lane_mask, 0x3u);
    EXPECT_EQ(state->per_lane_addr[0], kGlobalBase + tc.offset);
    EXPECT_EQ(state->per_lane_addr[1], kGlobalBase + tc.offset + tc.global_stride);
    EXPECT_EQ(state->per_lane_lds_addr[0], wf->lds_base());
    EXPECT_EQ(state->per_lane_lds_addr[1], wf->lds_base() + tc.lds_stride);
  }
}

TEST(Gfx1250ExecutionTest, GlobalStoreAsyncFromLdsKeepsIoffsetOnGlobalDestination) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalBase = 0x190000u;
  constexpr uint32_t kLane0Value = 0x12345678u;
  constexpr uint32_t kLane1Value = 0xabcdef01u;
  constexpr uint32_t kIoffset = 8;
  write_wave_sgpr(*cu, *wf, 0, static_cast<uint32_t>(kGlobalBase));
  write_wave_sgpr(*cu, *wf, 1, static_cast<uint32_t>(kGlobalBase >> 32));

  const uint32_t vgpr_base = wf->vgpr_alloc().base;
  cu->write_vgpr(vgpr_base, 0, 0);
  cu->write_vgpr(vgpr_base, 1, 16);
  cu->lds().write32(wf->lds_base(), kLane0Value);
  cu->lds().write32(wf->lds_base() + 16, kLane1Value);

  auto inst =
      decode_gfx1250({0xEE190000u, 0x00000000u, 0x00000800u}, "global_store_async_from_lds_b32");
  ASSERT_NE(inst, nullptr);
  inst->execute(*inst, wf);
  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->tag(), amdgpu::GLOBAL_MEM);
  EXPECT_FALSE(state->is_load);
  EXPECT_EQ(state->wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  EXPECT_EQ(state->lane_mask, 0x3u);
  EXPECT_EQ(state->per_lane_addr[0], kGlobalBase + kIoffset);
  EXPECT_EQ(state->per_lane_addr[1], kGlobalBase + 16 + kIoffset);
  ASSERT_GE(state->store_data.size(), 8u);

  uint32_t lane0_value = 0;
  uint32_t lane1_value = 0;
  std::memcpy(&lane0_value, &state->store_data[0], sizeof(lane0_value));
  std::memcpy(&lane1_value, &state->store_data[4], sizeof(lane1_value));
  EXPECT_EQ(lane0_value, kLane0Value);
  EXPECT_EQ(lane1_value, kLane1Value);
}

TEST(Gfx1250ExecutionTest, DispatchEntryClusterMathCoversMultiDimensionalShapes) {
  amdgpu::DispatchEntry entry{};
  entry.grid_wgs_x = 4;
  entry.grid_wgs_y = 4;
  entry.grid_wgs_z = 4;
  entry.cluster_count_x = 2;
  entry.cluster_count_y = 2;
  entry.cluster_count_z = 2;
  entry.cluster_size_x = 2;
  entry.cluster_size_y = 2;
  entry.cluster_size_z = 2;

  EXPECT_TRUE(entry.cluster_grid_is_complete());
  EXPECT_EQ(entry.cluster_size(), 8u);
  EXPECT_EQ(entry.cluster_rank_for_local_wg(0), 0u);
  EXPECT_EQ(entry.cluster_rank_for_local_wg(5), 3u);
  EXPECT_EQ(entry.cluster_rank_for_local_wg(21), 7u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 0), 0u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 1), 1u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 2), 4u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 3), 5u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 4), 16u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(0, 7), 21u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(42, 0), 42u);
  EXPECT_EQ(entry.cluster_peer_local_wg_id(42, 7), 63u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(0), 0u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(1), 2u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(2), 8u);
  EXPECT_EQ(entry.cluster_base_local_wg_id_for_ordinal(4), 32u);

  entry.grid_wgs_x = 3;
  entry.grid_wgs_y = 2;
  entry.grid_wgs_z = 1;
  entry.cluster_count_x = 2;
  entry.cluster_count_y = 2;
  entry.cluster_count_z = 1;
  entry.cluster_size_x = 2;
  entry.cluster_size_y = 1;
  entry.cluster_size_z = 1;
  EXPECT_FALSE(entry.cluster_grid_is_complete());
}

TEST(Gfx1250ExecutionTest, ClusterLdsMulticastTransactionCapturesRemapState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(9, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(7);
  wf->set_cluster_info(/*rank=*/1, /*size=*/4);
  wf->set_lds_base(0x100);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.elem_size = 4;
  state.num_elems = 2;
  state.wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state.lds_base = wf->lds_base();
  state.lds_per_lane_addr = true;
  state.cluster_multicast = true;
  state.cluster_mcast_mask = 0xa;
  state.wf_size = 32;
  state.lane_mask = 0x3;
  state.per_lane_addr[0] = 0x8000;
  state.per_lane_addr[1] = 0x8020;
  state.per_lane_lds_addr[0] = state.lds_base + 0x10;
  state.per_lane_lds_addr[1] = state.lds_base + 0x24;
  state.response_data.resize(state.wf_size * state.num_elems * state.elem_size);

  std::vector<amdgpu::ClusterLdsTarget> targets = {{cu, /*wg_id=*/11, /*lds_base=*/0x400,
                                                    /*cluster_rank=*/3}};
  auto txn = amdgpu::make_cluster_lds_multicast_transaction(state, *wf, std::move(targets));

  EXPECT_EQ(txn.dispatch_id, 7u);
  EXPECT_EQ(txn.source_wg_id, 9u);
  EXPECT_EQ(txn.source_cluster_rank, 1u);
  EXPECT_EQ(txn.source_lds_base, 0x100u);
  EXPECT_EQ(txn.mcast_mask, 0xau);
  EXPECT_EQ(txn.wait_counter_type, amdgpu::WaitCounterType::ASYNCCNT);
  EXPECT_EQ(txn.bytes_per_lane, 8u);
  // Retained for deferred/timing backends that model global request coalescing.
  EXPECT_EQ(txn.per_lane_global_addr[0], 0x8000u);
  EXPECT_EQ(txn.per_lane_global_addr[1], 0x8020u);
  ASSERT_EQ(txn.targets.size(), 1u);
  EXPECT_EQ(txn.targets[0].wg_id, 11u);
  EXPECT_EQ(amdgpu::cluster_lds_lane_addr(txn, 0, txn.targets[0].lds_base), 0x410u);
  EXPECT_EQ(amdgpu::cluster_lds_lane_addr(txn, 1, txn.targets[0].lds_base), 0x424u);
  EXPECT_THROW((void)amdgpu::remap_cluster_lds_addr(0x100, 0x400, 0xfc), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ClusterLdsSourceRankSelectionCoversDefaultAndMasks) {
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_cluster_rank = 2;

  txn.mcast_mask = 0;
  EXPECT_TRUE(amdgpu::cluster_lds_source_rank_selected(txn));

  txn.mcast_mask = amdgpu::cluster_multicast_rank_mask(2);
  EXPECT_TRUE(amdgpu::cluster_lds_source_rank_selected(txn));

  txn.mcast_mask = amdgpu::cluster_multicast_rank_mask(1);
  EXPECT_FALSE(amdgpu::cluster_lds_source_rank_selected(txn));
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineWritesOnlyIssuingParticipant) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = 1;
  txn.source_cluster_rank = 1;
  txn.source_lds_base = 0x300;
  txn.mcast_mask = 0x3;
  txn.bytes_per_lane = 4;
  txn.wf_size = 4;
  txn.lane_mask = 0x5;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0x304;
  txn.per_lane_lds_addr[2] = 0x30c;
  txn.payload.resize(txn.wf_size * txn.bytes_per_lane);
  const uint32_t lane0 = 0x11223344;
  const uint32_t lane2 = 0xaabbccdd;
  std::memcpy(&txn.payload[0 * txn.bytes_per_lane], &lane0, sizeof(lane0));
  std::memcpy(&txn.payload[2 * txn.bytes_per_lane], &lane2, sizeof(lane2));
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0},
                 {cu, /*wg_id=*/1, /*lds_base=*/0x300, /*cluster_rank=*/1}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  bool deferred_callback_called = false;
  EXPECT_EQ(engine.submit(std::move(txn), [&]() { deferred_callback_called = true; }),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_FALSE(deferred_callback_called);
  EXPECT_EQ(cu->lds().read32(0x204), 0u);
  EXPECT_EQ(cu->lds().read32(0x20c), 0u);
  EXPECT_EQ(cu->lds().read32(0x304), lane0);
  EXPECT_EQ(cu->lds().read32(0x30c), lane2);
  EXPECT_EQ(cu->lds().read32(0x208), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineSkipsUnissuedSelectedPeer) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  constexpr uint32_t kValue = 0x55667788;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = 1;
  txn.source_cluster_rank = 1;
  txn.source_lds_base = 0x300;
  txn.mcast_mask = 0x1; // Selects rank 0, not the issuing rank 1.
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0x310;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(cu->lds().read32(0x210), 0u);
  EXPECT_EQ(cu->lds().read32(0x310), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineUsesRecipientOwnedDestinations) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  cu->clear_lds();

  constexpr uint32_t kWg0Value = 0x11112222;
  constexpr uint32_t kWg1Value = 0x33334444;

  auto make_txn = [&](uint32_t wg_id, uint32_t rank, uint32_t lds_base, uint32_t lds_offset,
                      uint32_t value) {
    amdgpu::ClusterLdsMulticastTransaction txn{};
    txn.source_wg_id = wg_id;
    txn.source_cluster_rank = rank;
    txn.source_lds_base = lds_base;
    txn.mcast_mask = 0x3;
    txn.bytes_per_lane = sizeof(value);
    txn.wf_size = 1;
    txn.lane_mask = 0x1;
    txn.per_lane_addr = true;
    txn.per_lane_lds_addr[0] = lds_base + lds_offset;
    txn.payload.resize(sizeof(value));
    std::memcpy(txn.payload.data(), &value, sizeof(value));
    txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x100, /*cluster_rank=*/0},
                   {cu, /*wg_id=*/1, /*lds_base=*/0x200, /*cluster_rank=*/1}};
    return txn;
  };

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(make_txn(/*wg_id=*/0, /*rank=*/0, /*lds_base=*/0x100,
                                   /*lds_offset=*/0x10, kWg0Value),
                          []() {}),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(engine.submit(make_txn(/*wg_id=*/1, /*rank=*/1, /*lds_base=*/0x200,
                                   /*lds_offset=*/0x30, kWg1Value),
                          []() {}),
            amdgpu::ClusterLdsMulticastResult::Complete);

  EXPECT_EQ(cu->lds().read32(0x110), kWg0Value);
  EXPECT_EQ(cu->lds().read32(0x230), kWg1Value);
  EXPECT_EQ(cu->lds().read32(0x210), 0u);
  EXPECT_EQ(cu->lds().read32(0x130), 0u);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineRejectsUndersizedPayload) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.bytes_per_lane = 4;
  txn.wf_size = 2;
  txn.lane_mask = 0x3;
  txn.payload.resize(4);
  txn.targets = {{cu, /*wg_id=*/0, /*lds_base=*/0x200, /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_THROW((void)engine.submit(std::move(txn), []() {}), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ImmediateClusterLdsMulticastEngineRejectsOutOfRangeTarget) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_lds_base = 0;
  txn.bytes_per_lane = 4;
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = 0;
  txn.payload.resize(4);
  txn.targets = {{cu, /*wg_id=*/0, static_cast<uint32_t>(cu->lds().size_bytes()) - 2,
                  /*cluster_rank=*/0}};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_THROW((void)engine.submit(std::move(txn), []() {}), std::runtime_error);
}

TEST(Gfx1250ExecutionTest, ClusterLdsPinPreventsAllocatorReuseUntilClusterCompletes) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();

  EXPECT_EQ(cu->allocate_lds(257), 0u);
  cu->pin_lds_until_cluster_retired(7);
  cu->retire_halted_wfs();

  EXPECT_EQ(cu->allocate_lds(257), 512u);
  cu->unpin_lds_for_cluster(7);
  cu->retire_halted_wfs();
  EXPECT_EQ(cu->allocate_lds(257), 0u);
}

TEST(Gfx1250ExecutionTest, OrdinaryLdsDstLoadWritesDirectlyAndCompletesAsyncCounter) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(3);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0x9000;
  constexpr uint32_t kLoadedValue = 0x12345678;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x20;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  EXPECT_FALSE(static_cast<bool>(deferred_engine.completion));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x20), kLoadedValue);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250ExecutionTest, NonClusterClusterLdsLoadDowngradesToOrdinaryAsyncToLds) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(5);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0x9080;
  constexpr uint32_t kLoadedValue = 0x78563412;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->cluster_multicast = true;
  state->cluster_mcast_mask = 0x2; // Excludes rank 0, but non-clustered loads downgrade.
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x24;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  EXPECT_FALSE(static_cast<bool>(deferred_engine.completion));
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x24), kLoadedValue);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250ExecutionTest, ClusterLoadRequestBypassesStaleL1VectorLine) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(6);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobalAddr = 0xa000;
  constexpr uint32_t kOldValue = 0x11111111;
  constexpr uint32_t kNewValue = 0x22222222;
  for (uint32_t byte = 0; byte < sizeof(kOldValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte, static_cast<uint8_t>((kOldValue >> (byte * 8)) & 0xffu));
  }

  uint64_t addrs[64] = {};
  addrs[0] = kGlobalAddr;
  uint8_t l1_fill[64 * sizeof(kOldValue)] = {};
  cu->l1_vector().load(addrs, /*lane_mask=*/0x1, /*elem_size=*/4, /*num_elems=*/1, l1_fill,
                       amdgpu::Mtype::RW, /*non_temporal=*/false,
                       /*request_l1_bypass=*/false, /*vmid=*/0);
  uint32_t filled_value = 0;
  std::memcpy(&filled_value, l1_fill, sizeof(filled_value));
  ASSERT_EQ(filled_value, kOldValue);

  cu->l2()->write(kGlobalAddr, reinterpret_cast<const uint8_t *>(&kNewValue), sizeof(kNewValue),
                  amdgpu::Mtype::RW, /*vmid=*/0);

  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());

  auto ordinary = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  ordinary->elem_size = 4;
  ordinary->num_elems = 1;
  ordinary->is_load = true;
  ordinary->wait_counter_type = amdgpu::WaitCounterType::LOADCNT;
  ordinary->wf_size = 32;
  ordinary->lane_mask = 0x1;
  ordinary->exec_mask = 0x1;
  ordinary->dst_reg_base = wf->vgpr_alloc().base;
  ordinary->per_lane_addr[0] = kGlobalAddr;
  pipeline.issue(new TestMemoryInstruction(std::move(ordinary)), *wf);
  EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, 0), kOldValue);

  auto cluster = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  cluster->elem_size = 4;
  cluster->num_elems = 1;
  cluster->is_load = true;
  cluster->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  cluster->request_force_l1_bypass = true;
  cluster->lds_dst = true;
  cluster->lds_per_lane_addr = true;
  cluster->lds_base = wf->lds_base();
  cluster->wf_size = 32;
  cluster->lane_mask = 0x1;
  cluster->exec_mask = 0x1;
  cluster->cluster_multicast = true;
  cluster->cluster_mcast_mask = 0x1;
  cluster->per_lane_addr[0] = kGlobalAddr;
  cluster->per_lane_lds_addr[0] = wf->lds_base() + 0x40;
  pipeline.issue(new TestMemoryInstruction(std::move(cluster)), *wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0x40), kNewValue);
}

TEST(Gfx1250ExecutionTest, ClusterLdsFallbackSkipsSelfWhenMaskExcludesSource) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/3, /*pc=*/0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(17);
  wf->set_lds_base(cu->allocate_lds(256));
  wf->set_cluster_info(/*rank=*/1, /*size=*/2);

  constexpr uint64_t kGlobalAddr = 0x9100;
  constexpr uint32_t kLoadedValue = 0xabcdef01;
  for (uint32_t byte = 0; byte < sizeof(kLoadedValue); ++byte) {
    sim.memory->write8(kGlobalAddr + byte,
                       static_cast<uint8_t>((kLoadedValue >> (byte * 8)) & 0xffu));
  }

  auto state = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  state->elem_size = 4;
  state->num_elems = 1;
  state->is_load = true;
  state->wait_counter_type = amdgpu::WaitCounterType::ASYNCCNT;
  state->lds_dst = true;
  state->lds_per_lane_addr = true;
  state->lds_base = wf->lds_base();
  state->wf_size = 32;
  state->lane_mask = 0x1;
  state->exec_mask = 0x1;
  state->cluster_multicast = true;
  state->cluster_mcast_mask = 0x1; // Selects rank 0, not the source rank 1.
  state->per_lane_addr[0] = kGlobalAddr;
  state->per_lane_lds_addr[0] = wf->lds_base() + 0x20;

  DeferredClusterLdsMulticastEngine deferred_engine;
  cu->set_cluster_lds_multicast_engine(&deferred_engine);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(wf->wait_counters().asynccnt, 1u);
  ASSERT_TRUE(static_cast<bool>(deferred_engine.completion));
  EXPECT_TRUE(deferred_engine.txn.targets.empty());

  deferred_engine.completion();
  EXPECT_EQ(wf->wait_counters().asynccnt, 0u);
  cu->set_cluster_lds_multicast_engine(nullptr);
}

TEST(Gfx1250SimulationTest, ClusterLdsTargetsCoversMasksAndLifetime) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);

  ASSERT_TRUE(sim.engine->step());

  auto all = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(all.size(), 2u);
  EXPECT_EQ(all[0].wg_id, 0u);
  EXPECT_EQ(all[1].wg_id, 1u);

  auto self = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x1);
  ASSERT_EQ(self.size(), 1u);
  EXPECT_EQ(self[0].wg_id, 0u);

  auto peer = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x2);
  ASSERT_EQ(peer.size(), 1u);
  EXPECT_EQ(peer[0].wg_id, 1u);

  auto peer_from_rank1 =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/1, /*mcast_mask=*/0x1);
  ASSERT_EQ(peer_from_rank1.size(), 1u);
  EXPECT_EQ(peer_from_rank1[0].wg_id, 0u);

  auto zero_mask = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0);
  ASSERT_EQ(zero_mask.size(), 1u);
  EXPECT_EQ(zero_mask[0].wg_id, 0u);

  auto out_of_range =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x4);
  EXPECT_TRUE(out_of_range.empty());

  auto mixed_range =
      sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x5);
  ASSERT_EQ(mixed_range.size(), 1u);
  EXPECT_EQ(mixed_range[0].wg_id, 0u);

  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/999, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());

  sim.engine->run();
  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());
}

TEST(Gfx1250SimulationTest, ClusterLdsTargetsUseMultiDimensionalClusterPlacement) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);

  amdgpu::AmdExtKernelDispatchPacket pkt{};
  pkt.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  pkt.amd_format = amdgpu::kHsaAmdPacketTypeExtKernelDispatch;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.cluster_count_x = 2;
  pkt.cluster_count_y = 2;
  pkt.cluster_count_z = 1;
  pkt.cluster_size_x = 2;
  pkt.cluster_size_y = 2;
  pkt.cluster_size_z = 1;
  pkt.group_segment_size = lds_bytes_per_wg;
  pkt.kernel_object = kernel_object;

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.submit(pkt);
  ASSERT_TRUE(sim.engine->step());

  auto cluster0 = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0xf);
  ASSERT_EQ(cluster0.size(), 4u);
  EXPECT_EQ(cluster0[0].wg_id, 0u);
  EXPECT_EQ(cluster0[1].wg_id, 1u);
  EXPECT_EQ(cluster0[2].wg_id, 4u);
  EXPECT_EQ(cluster0[3].wg_id, 5u);

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/5, /*mcast_mask=*/0x5);
  ASSERT_EQ(targets.size(), 2u);
  EXPECT_EQ(targets[0].wg_id, 0u);
  EXPECT_EQ(targets[1].wg_id, 4u);

  auto source = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/5, /*mcast_mask=*/0x8);
  ASSERT_EQ(source.size(), 1u);
  EXPECT_EQ(source[0].wg_id, 5u);

  constexpr uint32_t kValue = 0xfeed1234;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = source[0].wg_id;
  txn.source_cluster_rank = source[0].cluster_rank;
  txn.source_lds_base = source[0].lds_base;
  txn.mcast_mask = 0x8;
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = source[0].lds_base + 0x10;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = source;

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  bool deferred_callback_called = false;
  EXPECT_EQ(engine.submit(std::move(txn), [&]() { deferred_callback_called = true; }),
            amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_FALSE(deferred_callback_called);
  EXPECT_EQ(targets[0].cu->lds().read32(targets[0].lds_base + 0x10), 0u);
  EXPECT_EQ(targets[1].cu->lds().read32(targets[1].lds_base + 0x10), 0u);
  EXPECT_EQ(source[0].cu->lds().read32(source[0].lds_base + 0x10), kValue);

  sim.engine->run();
}

TEST(Gfx1250SimulationTest, ClusterLdsDoesNotRemapIntoNonParticipatingPeerLdsBase) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint32_t lds_bytes_per_wg = 256;
  constexpr uint32_t cluster_size = 8;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim(make_single_se_gfx1250_config(/*num_cus=*/4));
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, cluster_size,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  struct RemapCase {
    amdgpu::ClusterLdsTarget source;
    amdgpu::ClusterLdsTarget target;
  };
  std::optional<RemapCase> remap_case;
  for (uint32_t wg_id = 0; wg_id < cluster_size && !remap_case; ++wg_id) {
    auto source = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, wg_id, /*mcast_mask=*/0);
    ASSERT_EQ(source.size(), 1u);
    ASSERT_NE(source[0].cu, nullptr);

    auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, wg_id,
                                                 /*mcast_mask=*/(1u << cluster_size) - 1);
    for (const auto &target : targets) {
      if (target.wg_id == wg_id)
        continue;
      if (target.cu == source[0].cu && target.lds_base != source[0].lds_base) {
        remap_case = RemapCase{source[0], target};
        break;
      }
    }
  }
  ASSERT_TRUE(remap_case.has_value());
  const auto source = remap_case->source;
  const auto target = remap_case->target;
  ASSERT_NE(target.cu, nullptr);
  EXPECT_EQ(target.cu, source.cu);
  EXPECT_NE(target.lds_base, source.lds_base);

  constexpr uint32_t kValue = 0x13579bdf;
  amdgpu::ClusterLdsMulticastTransaction txn{};
  txn.source_wg_id = source.wg_id;
  txn.source_cluster_rank = source.cluster_rank;
  txn.source_lds_base = source.lds_base;
  txn.mcast_mask = (1u << cluster_size) - 1;
  txn.bytes_per_lane = sizeof(kValue);
  txn.wf_size = 1;
  txn.lane_mask = 0x1;
  txn.per_lane_addr = true;
  txn.per_lane_lds_addr[0] = source.lds_base + 0x20;
  txn.payload.resize(sizeof(kValue));
  std::memcpy(txn.payload.data(), &kValue, sizeof(kValue));
  txn.targets = {target};

  amdgpu::ImmediateClusterLdsMulticastEngine engine;
  EXPECT_EQ(engine.submit(std::move(txn), []() {}), amdgpu::ClusterLdsMulticastResult::Complete);
  EXPECT_EQ(target.cu->lds().read32(target.lds_base + 0x20), 0u);
  EXPECT_EQ(source.cu->lds().read32(source.lds_base + 0x20), 0u);

  sim.engine->run();
}

TEST(Gfx1250SimulationTest, RejectsUnsupportedClusterSize) {
  constexpr uint64_t kernel_addr = 0x10000;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1,
                           /*cluster_size_x=*/amdgpu::kClusterMulticastMaskBits + 1,
                           /*workgroup_size_x=*/32);

  EXPECT_THROW((void)sim.engine->step(), std::runtime_error);
}

TEST(Gfx1250SimulationTest, RejectsIncompleteClusterGrid) {
  constexpr uint64_t kernel_addr = 0x10000;
  const uint32_t code[] = {S_ENDPGM_GFX12};

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/0);

  EXPECT_THROW((void)sim.engine->step(), std::runtime_error);
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsDoesNotWriteMaskExcludedParticipant) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2000;
  constexpr uint32_t lds_bytes_per_wg = 256;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr), // s_mov_b32 s0, input_addr
      0xBE810080u,                                       // s_mov_b32 s1, 0
      0xBEFD0081u,                                       // s_mov_b32 m0, 0x1
      0x30000082u,                                       // v_lshlrev_b32_e32 v0, 2, v0
      0xEE1AC000u,    0x00000000u,
      0x00000000u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1]
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t byte = 0; byte < 256; ++byte)
    sim.memory->write8(input_addr + byte, static_cast<uint8_t>(0x40u + byte));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_NE(targets[0].cu, nullptr);
  ASSERT_NE(targets[1].cu, nullptr);
  EXPECT_NE(targets[0].cu, targets[1].cu);
  sim.engine->run();
  EXPECT_TRUE(sim.cp()
                  ->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0,
                                        /*mcast_mask=*/0x3)
                  .empty());

  auto read_unaligned_input = [&](uint32_t byte_offset) {
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
      value |= static_cast<uint32_t>(sim.memory->read8(input_addr + byte_offset + byte))
               << (byte * 8);
    return value;
  };

  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t value0 = targets[0].cu->lds().read32(targets[0].lds_base + lane * 4);
    uint32_t value1 = targets[1].cu->lds().read32(targets[1].lds_base + lane * 4);
    uint32_t wg0_value = read_unaligned_input(lane * 4);
    EXPECT_EQ(value0, wg0_value) << "lane " << lane;
    EXPECT_EQ(value1, 0u) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, ClusterLoadAsyncToLdsWritesEachIssuingParticipant) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t input_addr = 0x2400;
  constexpr uint32_t lds_bytes_per_wg = 256;

  const uint32_t code[] = {
      0xBE8000FFu,    static_cast<uint32_t>(input_addr), // s_mov_b32 s0, input_addr
      0xBE810080u,                                       // s_mov_b32 s1, 0
      0xBEFD0083u,                                       // s_mov_b32 m0, 0x3
      0x30000082u,                                       // v_lshlrev_b32_e32 v0, 2, v0
      0xEE1AC000u,    0x00000000u,
      0x00000000u, // cluster_load_async_to_lds_b32 v0, v0, s[0:1]
      0xBFCA0000u, // s_wait_asynccnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t byte = 0; byte < 256; ++byte)
    sim.memory->write8(input_addr + byte, static_cast<uint8_t>(0x80u + byte));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32, /*kernarg_addr=*/0, lds_bytes_per_wg);
  ASSERT_TRUE(sim.engine->step());

  auto targets = sim.cp()->cluster_lds_targets(/*dispatch_id=*/1, /*wg_id=*/0, /*mcast_mask=*/0x3);
  ASSERT_EQ(targets.size(), 2u);
  ASSERT_NE(targets[0].cu, nullptr);
  ASSERT_NE(targets[1].cu, nullptr);
  EXPECT_NE(targets[0].cu, targets[1].cu);
  sim.engine->run();

  auto read_unaligned_input = [&](uint32_t byte_offset) {
    uint32_t value = 0;
    for (uint32_t byte = 0; byte < 4; ++byte)
      value |= static_cast<uint32_t>(sim.memory->read8(input_addr + byte_offset + byte))
               << (byte * 8);
    return value;
  };

  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t value0 = targets[0].cu->lds().read32(targets[0].lds_base + lane * 4);
    uint32_t value1 = targets[1].cu->lds().read32(targets[1].lds_base + lane * 4);
    uint32_t expected_value = read_unaligned_input(lane * 4);
    EXPECT_EQ(value0, expected_value) << "lane " << lane;
    EXPECT_EQ(value1, expected_value) << "lane " << lane;
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaD2CopiesGlobalAndLds) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint32_t kElements = 16;
  constexpr uint64_t kLoadGlobal = 0x100000;
  constexpr uint64_t kStoreGlobal = 0x110000;
  auto write_sgpr = [&](uint32_t reg, uint32_t value) {
    cu->write_sgpr(wf->sgpr_alloc().base + reg, value);
  };
  auto write_d0 = [&](uint32_t reg, uint64_t global_addr) {
    write_sgpr(reg + 0, 1);
    write_sgpr(reg + 1, 0);
    write_sgpr(reg + 2, static_cast<uint32_t>(global_addr));
    write_sgpr(reg + 3, static_cast<uint32_t>((global_addr >> 32) & 0x01ffffffu) | 0x80000000u);
  };

  write_d0(0, kLoadGlobal);
  write_d0(8, kStoreGlobal);
  write_sgpr(12, 2u << 16);        // i32 elements.
  write_sgpr(13, kElements << 16); // tensor dim0.
  write_sgpr(14, 0);
  write_sgpr(15, kElements << 16); // tile dim0.
  write_sgpr(16, 0);
  write_sgpr(17, 0);
  write_sgpr(18, 0);
  write_sgpr(19, 0);

  for (uint32_t i = 0; i < kElements; ++i) {
    const uint32_t value = 0x11000000u + i * 0x101u;
    for (uint32_t byte = 0; byte < 4; ++byte)
      sim.memory->write8(kLoadGlobal + i * 4 + byte, static_cast<uint8_t>(value >> (byte * 8)));
  }

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);
  for (uint32_t i = 0; i < kElements; ++i)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), 0x11000000u + i * 0x101u);

  for (uint32_t i = 0; i < kElements; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, 0x22000000u + i * 0x303u);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c08u};
  gfx1250::TensorStoreFromLdsVimage store_inst(store_words.data());
  store_inst.execute_impl(*wf);
  for (uint32_t i = 0; i < kElements; ++i) {
    const uint32_t actual = read_global_u32(*sim.memory, kStoreGlobal + i * 4);
    EXPECT_EQ(actual, 0x22000000u + i * 0x303u);
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaDecodeExecuteCoversLoadStoreD2D4Forms) {
  constexpr uint32_t kNullSgpr = 124;
  constexpr std::array<uint32_t, 3> kLoadD2 = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  constexpr std::array<uint32_t, 3> kStoreD2 = {0xd0714001u, 0x7c000000u, 0x7c7c0c08u};
  constexpr std::array<uint32_t, 3> kLoadD4 = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  constexpr std::array<uint32_t, 3> kStoreD4 = {0xd0714001u, 0x7c000000u, 0x18140c08u};

  {
    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));

    constexpr uint32_t kElements = 4;
    constexpr uint64_t kLoadGlobal = 0x1a0000;
    constexpr uint64_t kStoreGlobal = 0x1b0000;

    write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
    write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16);        // i32 elements.
    write_wave_sgpr(*cu, *wf, 13, kElements << 16); // Tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, 0);
    write_wave_sgpr(*cu, *wf, 15, kElements << 16); // Tile dim0.
    write_wave_sgpr(*cu, *wf, 16, 0);
    write_wave_sgpr(*cu, *wf, 17, 0);
    write_wave_sgpr(*cu, *wf, 18, 0);
    write_wave_sgpr(*cu, *wf, 19, 0);

    for (uint32_t i = 0; i < kElements; ++i)
      write_global_u32(*sim.memory, kLoadGlobal + i * 4, 0x71000000u + i);

    auto load = decode_gfx1250(kLoadD2, "tensor_load_to_lds");
    ASSERT_NE(load, nullptr);
    ASSERT_EQ(load->num_src_operands(), 4);
    EXPECT_EQ(load->src_operand(2)->encoding_value(), static_cast<int>(kNullSgpr));
    EXPECT_EQ(load->src_operand(3)->encoding_value(), static_cast<int>(kNullSgpr));
    load->execute(*load, wf);

    for (uint32_t i = 0; i < kElements; ++i)
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + i * 4), 0x71000000u + i);

    for (uint32_t i = 0; i < kElements; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, 0x72000000u + i);

    auto store = decode_gfx1250(kStoreD2, "tensor_store_from_lds");
    ASSERT_NE(store, nullptr);
    ASSERT_EQ(store->num_src_operands(), 4);
    EXPECT_EQ(store->src_operand(2)->encoding_value(), static_cast<int>(kNullSgpr));
    EXPECT_EQ(store->src_operand(3)->encoding_value(), static_cast<int>(kNullSgpr));
    store->execute(*store, wf);

    for (uint32_t i = 0; i < kElements; ++i)
      EXPECT_EQ(read_global_u32(*sim.memory, kStoreGlobal + i * 4), 0x72000000u + i);
  }

  {
    Gfx1250Sim sim;
    auto *cu = sim.cu();
    auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_lds_base(cu->allocate_lds(256));

    constexpr uint32_t kCols = 2;
    constexpr uint32_t kRows = 2;
    constexpr uint32_t kTileRows = 3;
    constexpr uint32_t kDepth = 2;
    constexpr uint32_t kPlaneStride = kTileRows * kCols;
    constexpr uint64_t kLoadGlobal = 0x1c0000;
    constexpr uint64_t kStoreGlobal = 0x1d0000;
    constexpr uint32_t kSentinel = 0x7e7e7e7eu;

    write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
    write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
    write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
    write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
    write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
    write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
    write_wave_sgpr(*cu, *wf, 16, kTileRows | (kDepth << 16));
    write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
    write_wave_sgpr(*cu, *wf, 18, kPlaneStride << 16);
    write_wave_sgpr(*cu, *wf, 19, 0);
    write_wave_sgpr(*cu, *wf, 20, kDepth); // Tensor dim2, from D2.
    write_wave_sgpr(*cu, *wf, 21, 0);
    write_wave_sgpr(*cu, *wf, 22, 0);
    write_wave_sgpr(*cu, *wf, 23, 0);
    write_wave_sgpr(*cu, *wf, 24, 0);
    write_wave_sgpr(*cu, *wf, 25, 0);
    write_wave_sgpr(*cu, *wf, 26, 0);
    write_wave_sgpr(*cu, *wf, 27, 0);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint64_t offset = static_cast<uint64_t>(z * kPlaneStride + y * kCols + x) * 4;
          const uint32_t value = 0x73000000u + z * 0x100u + y * 0x10u + x;
          write_global_u32(*sim.memory, kLoadGlobal + offset, value);
          write_global_u32(*sim.memory, kStoreGlobal + offset, kSentinel);
        }
      }
    }

    auto load = decode_gfx1250(kLoadD4, "tensor_load_to_lds");
    ASSERT_NE(load, nullptr);
    ASSERT_EQ(load->num_src_operands(), 4);
    EXPECT_EQ(load->src_operand(2)->encoding_value(), 20);
    EXPECT_EQ(load->src_operand(3)->encoding_value(), 24);
    load->execute(*load, wf);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint32_t lds_index = z * kTileRows * kCols + y * kCols + x;
          const uint32_t expected = (y < kRows) ? (0x73000000u + z * 0x100u + y * 0x10u + x) : 0u;
          EXPECT_EQ(cu->lds().read32(wf->lds_base() + lds_index * 4), expected);
        }
      }
    }

    for (uint32_t i = 0; i < kDepth * kTileRows * kCols; ++i)
      cu->lds().write32(wf->lds_base() + i * 4, 0x74000000u + i);

    auto store = decode_gfx1250(kStoreD4, "tensor_store_from_lds");
    ASSERT_NE(store, nullptr);
    ASSERT_EQ(store->num_src_operands(), 4);
    EXPECT_EQ(store->src_operand(2)->encoding_value(), 20);
    EXPECT_EQ(store->src_operand(3)->encoding_value(), 24);
    store->execute(*store, wf);

    for (uint32_t z = 0; z < kDepth; ++z) {
      for (uint32_t y = 0; y < kTileRows; ++y) {
        for (uint32_t x = 0; x < kCols; ++x) {
          const uint64_t offset = static_cast<uint64_t>(z * kPlaneStride + y * kCols + x) * 4;
          const uint32_t lds_index = z * kTileRows * kCols + y * kCols + x;
          const uint32_t expected = (y < kRows) ? (0x74000000u + lds_index) : kSentinel;
          EXPECT_EQ(read_global_u32(*sim.memory, kStoreGlobal + offset), expected);
        }
      }
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaLoadAppliesLdsPadding) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x120000;
  constexpr uint32_t kSentinel = 0xCAFECAFEu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, pad enabled.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 4u << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 4; ++i) {
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x33000000u + i);
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);
  }
  cu->lds().write32(wf->lds_base() + 4 * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x33000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x33000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), 0x33000002u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * 4), 0x33000003u);
}

TEST(Gfx1250ExecutionTest, TensorDmaByteLoadUsesDwordPaddingUnits) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

  constexpr uint64_t kGlobal = 0x121000;
  constexpr uint8_t kSentinel = 0xA5u;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12,
                  (1u << 20) | (5u << 22) | (3u << 25)); // u8, pad 4 dwords per 64 dwords.
  write_wave_sgpr(*cu, *wf, 13, 260u << 16);             // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 260u << 16); // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  for (uint32_t i = 0; i < 260; ++i)
    sim.memory->write8(kGlobal + i, static_cast<uint8_t>(i));
  for (uint32_t i = 0; i < 300; ++i)
    cu->lds().write8(wf->lds_base() + i, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 63), 63u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 64), 64u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 255), 255u);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(cu->lds().read8(wf->lds_base() + 256 + i), kSentinel);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 272), 0u);
  EXPECT_EQ(cu->lds().read8(wf->lds_base() + 275), 3u);
}

TEST(Gfx1250ExecutionTest, TensorDmaPaddedStoreIsRejected) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  write_tensor_dma_d0(*cu, *wf, 0, 0x121000);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 20)); // i32, pad enabled.
  write_wave_sgpr(*cu, *wf, 13, 4u << 16);
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 4u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x7c7c0c00u};
  gfx1250::TensorStoreFromLdsVimage store_inst(store_words.data());
  EXPECT_THROW(store_inst.execute_impl(*wf), util::UnimplementedInst);
}

TEST(Gfx1250ExecutionTest, TensorDmaIterateCopiesMultipleTiles) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x130000;
  constexpr uint32_t kSentinel = 0xFEEDFEEDu;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 19)); // i32, iterate enabled.
  write_wave_sgpr(*cu, *wf, 13, 2u << 16);                // tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, 2u << 16);                // tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);                // tile dim0.
  write_wave_sgpr(*cu, *wf, 16, 1u);                      // tile dim1.
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, 0);
  write_wave_sgpr(*cu, *wf, 21, 4);        // LDS increment in elements.
  write_wave_sgpr(*cu, *wf, 22, 2);        // Global increment in elements.
  write_wave_sgpr(*cu, *wf, 23, 1u << 16); // iteration_count - 1.

  for (uint32_t i = 0; i < 4; ++i)
    write_global_u32(*sim.memory, kGlobal + i * 4, 0x44000000u + i);
  for (uint32_t i = 0; i < 8; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c140c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x44000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x44000001u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 2 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 3 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 4 * 4), 0x44000002u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 5 * 4), 0x44000003u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 6 * 4), kSentinel);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 7 * 4), kSentinel);
}

TEST(Gfx1250ExecutionTest, TensorDmaAtomicBarrierArrivesAfterCopy) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint64_t kGlobal = 0x140000;
  constexpr uint32_t kBarrierLdsAddr = 64;
  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 12, (2u << 16) | (1u << 18)); // i32, atomic barrier enabled.
  write_wave_sgpr(*cu, *wf, 13, (2u << 16) | (kBarrierLdsAddr >> 3));
  write_wave_sgpr(*cu, *wf, 14, 0);
  write_wave_sgpr(*cu, *wf, 15, 2u << 16);
  write_wave_sgpr(*cu, *wf, 16, 0);
  write_wave_sgpr(*cu, *wf, 17, 0);
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);

  write_global_u32(*sim.memory, kGlobal + 0 * 4, 0x55000000u);
  write_global_u32(*sim.memory, kGlobal + 1 * 4, 0x55000001u);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, 0);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x7c7c0c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x55000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x55000001u);
  const uint64_t state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(state, 0xffffull << 16);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
}

TEST(Gfx1250ExecutionTest, SBarrierWaitEntersBarrierState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->state(), amdgpu::WfState::RUNNING);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const std::array<uint32_t, 2> wait_words = {0xBF940000u, 0u};
  std::unique_ptr<Instruction> wait_inst(decoder->decode(wait_words.data()));
  ASSERT_NE(wait_inst, nullptr);
  ASSERT_EQ(std::string_view(wait_inst->mnemonic()), "s_barrier_wait");

  cu->execute_instruction(wait_inst.get(), *wf);

  EXPECT_EQ(wf->state(), amdgpu::WfState::BARRIER);
}

TEST(Gfx1250ExecutionTest, SBarrierWaitReleasesOnlyAfterAllSiblingsWait) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  constexpr uint64_t kEndPgmPc = 0x150000;
  const std::array<uint32_t, 1> endpgm_words = {S_ENDPGM_GFX12};
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(endpgm_words.data()), sizeof(uint32_t),
                         kEndPgmPc);

  auto *wf0 = cu->dispatch_wf(0, kEndPgmPc, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, kEndPgmPc, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const std::array<uint32_t, 2> wait_words = {0xBF940000u, 0u};
  std::unique_ptr<Instruction> wait_inst(decoder->decode(wait_words.data()));
  ASSERT_NE(wait_inst, nullptr);
  ASSERT_EQ(std::string_view(wait_inst->mnemonic()), "s_barrier_wait");

  cu->execute_instruction(wait_inst.get(), *wf0);
  wf1->wait_counters().increment(amdgpu::WaitCounterType::TENSORCNT);
  wf1->set_wait_target_tensorcnt(0);
  wf1->set_state(amdgpu::WfState::WAITCNT);

  ASSERT_TRUE(cu->step());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::WAITCNT);

  wf1->release_wait_counter(amdgpu::WaitCounterType::TENSORCNT);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  cu->execute_instruction(wait_inst.get(), *wf1);
  ASSERT_EQ(wf1->state(), amdgpu::WfState::BARRIER);

  EXPECT_FALSE(cu->step());
  EXPECT_TRUE(wf0->is_halted());
  EXPECT_TRUE(wf1->is_halted());
}

TEST(Gfx1250ExecutionTest, ReleaseWaitCounterWakesWaitcntWhenTargetIsSatisfied) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  wf->wait_counters().increment(amdgpu::WaitCounterType::TENSORCNT);
  wf->set_wait_target_tensorcnt(0);
  ASSERT_EQ(wf->state(), amdgpu::WfState::WAITCNT);

  wf->release_wait_counter(amdgpu::WaitCounterType::TENSORCNT);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_EQ(wf->state(), amdgpu::WfState::RUNNING);
}

TEST(Gfx1250ExecutionTest, ReleaseWaitCounterHaltsEndingWaveWhenLastCounterRetires) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  wf->wait_counters().increment(amdgpu::WaitCounterType::TENSORCNT);
  wf->end();
  ASSERT_EQ(wf->state(), amdgpu::WfState::ENDING);

  wf->release_wait_counter(amdgpu::WaitCounterType::TENSORCNT);

  EXPECT_TRUE(wf->wait_counters().empty());
  EXPECT_TRUE(wf->is_halted());
}

TEST(Gfx1250ExecutionTest, DsAtomicAsyncBarrierArriveFlipsRawBarrierPhase) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint32_t kBarrierLdsAddr = 64;
  cu->write_vgpr(wf->vgpr_alloc().base + 0, 0, kBarrierLdsAddr);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, amdgpu::lds_barrier_cell_init_state(1));

  const std::array<uint32_t, 2> words = {0xd9580000u, 0x00000000u};
  auto *arrive_inst = new gfx1250::DsAtomicAsyncBarrierArriveB64Vds(words.data());
  arrive_inst->execute_impl(*wf);
  amdgpu::LocalMemPipeline local_pipeline(&cu->lds());
  local_pipeline.issue(arrive_inst, *wf);

  const uint64_t state = cu->lds().read64(wf->lds_base() + kBarrierLdsAddr);
  EXPECT_EQ(state, 0xffffull << 16);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, LocalMemPipelineUsesInjectedBarrierDecrementPayload) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);
  wf->set_lds_base(cu->allocate_lds(256));

  constexpr uint32_t kBarrierLdsAddr = 64;
  cu->write_vgpr(wf->vgpr_alloc().base + 0, 0, kBarrierLdsAddr);
  cu->lds().write64(wf->lds_base() + kBarrierLdsAddr, amdgpu::lds_barrier_cell_init_state(2));

  const std::array<uint32_t, 2> words = {0xd9580000u, 0x00000000u};
  auto *arrive_inst = new gfx1250::DsAtomicAsyncBarrierArriveB64Vds(words.data());
  arrive_inst->execute_impl(*wf);

  auto *state = arrive_inst->data_as<amdgpu::VectorMemState>();
  const uint64_t decrement = 2;
  state->store_data.resize(static_cast<size_t>(wf->wf_size()) * sizeof(decrement));
  std::memcpy(state->store_data.data(), &decrement, sizeof(decrement));

  amdgpu::LocalMemPipeline local_pipeline(&cu->lds());
  local_pipeline.issue(arrive_inst, *wf);

  const uint64_t expected =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2), decrement);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), expected);
  EXPECT_EQ(expected, (1ull << 32) | (0xffffull << 16) | 1ull);
  EXPECT_TRUE(wf->wait_counters().empty());
}

TEST(Gfx1250ExecutionTest, LdsBarrierCellHandlesSingleAndBatchedArrivals) {
  uint64_t state = amdgpu::lds_barrier_cell_init_state(2);
  EXPECT_EQ(state, (1ull << 32) | 1ull);

  state = amdgpu::lds_barrier_cell_update_arrive(state);
  EXPECT_EQ(state, 1ull << 32);
  EXPECT_FALSE(amdgpu::lds_barrier_cell_phase_parity(state));

  state = amdgpu::lds_barrier_cell_update_arrive(state);
  EXPECT_EQ(state, (1ull << 32) | (0xffffull << 16) | 1ull);
  EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));

  const uint64_t drained =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2));
  ASSERT_EQ(amdgpu::lds_barrier_cell_pending_count(drained), 0ull);
  EXPECT_EQ(amdgpu::lds_barrier_cell_update_arrive(drained, 0), drained);
  EXPECT_EQ(amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(3), 0),
            amdgpu::lds_barrier_cell_init_state(3));

  uint64_t iterated = amdgpu::lds_barrier_cell_init_state(2);
  for (int i = 0; i < 5; ++i)
    iterated = amdgpu::lds_barrier_cell_update_arrive(iterated);
  const uint64_t batched =
      amdgpu::lds_barrier_cell_update_arrive(amdgpu::lds_barrier_cell_init_state(2), 5);
  EXPECT_EQ(batched, iterated);
  EXPECT_EQ(batched, (1ull << 32) | (0xfffeull << 16));

  for (uint32_t arrivals_per_phase : {0u, 1u}) {
    state = amdgpu::lds_barrier_cell_init_state(arrivals_per_phase);
    EXPECT_EQ(amdgpu::lds_barrier_cell_pending_count(state), 0ull);
    state = amdgpu::lds_barrier_cell_update_arrive(state);
    EXPECT_EQ(amdgpu::lds_barrier_cell_pending_count(state), 0ull);
    EXPECT_EQ(amdgpu::lds_barrier_cell_phase(state), amdgpu::kLdsBarrierCellPhaseMask);
    EXPECT_TRUE(amdgpu::lds_barrier_cell_phase_parity(state));
  }

  const uint64_t reserved = 0xabcdull << 48;
  const uint64_t reserved_state =
      amdgpu::lds_barrier_cell_update_arrive(reserved | amdgpu::lds_barrier_cell_init_state(2));
  EXPECT_EQ(reserved_state & amdgpu::kLdsBarrierCellReservedMask, reserved);
  EXPECT_EQ(amdgpu::lds_barrier_cell_init_count(reserved_state), 1ull);
}

TEST(Gfx1250ExecutionTest, TensorDmaDenseDescriptorCopiesDenseRows) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

  constexpr uint64_t kLoadGlobal = 0x150000;
  constexpr uint64_t kStoreGlobal = 0x160000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 4;
  constexpr uint32_t kTileDim1RowCount = 3;
  constexpr uint32_t kSentinel = 0xABCDABCDu;
  constexpr uint32_t kIgnoredD2 = 0xFFFFu;

  write_tensor_dma_d0(*cu, *wf, 0, kLoadGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u);
  write_tensor_dma_d0(*cu, *wf, 8, kStoreGlobal);
  write_wave_sgpr(*cu, *wf, 8, 1u);
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, kTileDim1RowCount);
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 21, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 22, kIgnoredD2);
  write_wave_sgpr(*cu, *wf, 23, 0);
  write_wave_sgpr(*cu, *wf, 24, 0);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      write_global_u32(*sim.memory, kLoadGlobal + (row * kCols + col) * 4,
                       0x66000000u + row * 0x100u + col);
      write_global_u32(*sim.memory, kStoreGlobal + (row * kCols + col) * 4, kSentinel);
    }
  }
  for (uint32_t i = 0; i < (kTileDim1RowCount + 1) * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kTileDim1RowCount; ++row_idx) {
    const uint32_t src_row = row_idx;
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x66000000u + src_row * 0x100u + col);
    }
  }
  for (uint32_t col = 0; col < kCols; ++col)
    EXPECT_EQ(cu->lds().read32(wf->lds_base() + (kTileDim1RowCount * kCols + col) * 4), kSentinel);

  for (uint32_t i = 0; i < kTileDim1RowCount * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, 0x77000000u + i);

  const std::array<uint32_t, 3> store_words = {0xd0714001u, 0x7c000000u, 0x18140c08u};
  gfx1250::TensorStoreFromLdsVimage store_inst(store_words.data());
  store_inst.execute_impl(*wf);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col) {
      const uint32_t actual = read_global_u32(*sim.memory, kStoreGlobal + (row * kCols + col) * 4);
      if (row < kTileDim1RowCount)
        EXPECT_EQ(actual, 0x77000000u + row * kCols + col);
      else
        EXPECT_EQ(actual, kSentinel);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherSupportsI32Indices) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

  constexpr uint64_t kGlobal = 0x170000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 2;
  constexpr std::array<uint32_t, 5> kIndices = {7, 0, 3, 6, 1};
  constexpr uint32_t kSentinel = 0xBCADBCADu;

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30) | (1u << 31));
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, static_cast<uint32_t>(kIndices.size()));
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIndices[0]);
  write_wave_sgpr(*cu, *wf, 21, kIndices[1]);
  write_wave_sgpr(*cu, *wf, 22, kIndices[2]);
  write_wave_sgpr(*cu, *wf, 23, kIndices[3]);
  write_wave_sgpr(*cu, *wf, 24, kIndices[4]);
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x68000000u + row * 0x100u + col);
  }
  for (uint32_t i = 0; i < kIndices.size() * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kIndices.size(); ++row_idx) {
    const uint32_t src_row = kIndices[row_idx];
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x68000000u + src_row * 0x100u + col);
    }
  }
}

TEST(Gfx1250ExecutionTest, TensorDmaGatherSupportsI16Indices) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_lds_base(cu->allocate_lds(512));

  constexpr uint64_t kGlobal = 0x180000;
  constexpr uint32_t kRows = 8;
  constexpr uint32_t kCols = 2;
  constexpr std::array<uint32_t, 10> kIndices = {7, 0, 3, 6, 1, 4, 2, 5, 7, 3};
  constexpr uint32_t kSentinel = 0xCDAECDAEu;

  write_tensor_dma_d0(*cu, *wf, 0, kGlobal);
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 31));
  write_wave_sgpr(*cu, *wf, 12, 2u << 16);    // i32 elements.
  write_wave_sgpr(*cu, *wf, 13, kCols << 16); // Tensor dim0.
  write_wave_sgpr(*cu, *wf, 14, kRows << 16); // Tensor dim1.
  write_wave_sgpr(*cu, *wf, 15, kCols << 16); // Tile dim0.
  write_wave_sgpr(*cu, *wf, 16, static_cast<uint32_t>(kIndices.size()));
  write_wave_sgpr(*cu, *wf, 17, kCols); // Tensor dim0 stride.
  write_wave_sgpr(*cu, *wf, 18, 0);
  write_wave_sgpr(*cu, *wf, 19, 0);
  write_wave_sgpr(*cu, *wf, 20, kIndices[0] | (kIndices[1] << 16));
  write_wave_sgpr(*cu, *wf, 21, kIndices[2] | (kIndices[3] << 16));
  write_wave_sgpr(*cu, *wf, 22, kIndices[4] | (kIndices[5] << 16));
  write_wave_sgpr(*cu, *wf, 23, kIndices[6] | (kIndices[7] << 16));
  write_wave_sgpr(*cu, *wf, 24, kIndices[8] | (kIndices[9] << 16));
  write_wave_sgpr(*cu, *wf, 25, 0);
  write_wave_sgpr(*cu, *wf, 26, 0);
  write_wave_sgpr(*cu, *wf, 27, 0);

  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t col = 0; col < kCols; ++col)
      write_global_u32(*sim.memory, kGlobal + (row * kCols + col) * 4,
                       0x69000000u + row * 0x100u + col);
  }
  for (uint32_t i = 0; i < kIndices.size() * kCols; ++i)
    cu->lds().write32(wf->lds_base() + i * 4, kSentinel);

  const std::array<uint32_t, 3> load_words = {0xd0710001u, 0x7c000000u, 0x18140c00u};
  gfx1250::TensorLoadToLdsVimage load_inst(load_words.data());
  load_inst.execute_impl(*wf);

  for (uint32_t row_idx = 0; row_idx < kIndices.size(); ++row_idx) {
    const uint32_t src_row = kIndices[row_idx];
    for (uint32_t col = 0; col < kCols; ++col) {
      EXPECT_EQ(cu->lds().read32(wf->lds_base() + (row_idx * kCols + col) * 4),
                0x69000000u + src_row * 0x100u + col);
    }
  }
}

TEST(Gfx1250CodeObjectTest, MachineFlagMapsToTarget) {
  auto image = make_minimal_gfx1250_elf();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  EXPECT_EQ(co.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_EQ(co.text_sections().size(), 1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  const auto *text = co.text_sections()[0];
  auto *words = reinterpret_cast<const uint32_t *>(text->data());
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_endpgm");
}

TEST(Gfx1250DecodeTest, SMovB64Literal64ConsumesThreeDwords) {
  const uint32_t words[] = {
      0xBEB801FEu, // s_mov_b64 s[56:57], literal64
      0xFFFFFF80u,
      0xFFFFFFFFu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_mov_b64");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(Gfx1250DecodeTest, Vop3LiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD6570001u, // v_and_or_b32 v1, 0xf8, v1, v2
      0x040A02FFu,
      0x000000F8u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_and_or_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(Gfx1250DecodeTest, Vop3SdstLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xD7020001u, // v_subrev_co_u32 v1, s0, 0x60, s12
      0x020018FFu,
      0x00000060u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_subrev_co_u32");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(Gfx1250DecodeTest, SWaitXcntHasWaitcntMetadata) {
  const uint32_t words[] = {
      0xBFC50000u, // s_wait_xcnt 0
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_wait_xcnt");
  EXPECT_TRUE(inst->is_waitcnt());
  EXPECT_EQ(inst->disassemble(), "s_wait_xcnt 0");
}

TEST(Gfx1250DecodeTest, BufferOffenUsesSingleVaddrRegister) {
  const uint32_t words[] = {
      0xC405C07Cu, // buffer_load_b128 v[32:35], v7, s[4:7], s0 offen
      0x40800820u,
      0x00000007u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "buffer_load_b128");
  ASSERT_GE(inst->num_src_operands(), 1);

  const Operand *vaddr = inst->src_operand(0);
  ASSERT_NE(vaddr, nullptr);
  EXPECT_EQ(vaddr->size_bits(), 32);
  ASSERT_TRUE(vaddr->to_register_ref().has_value());
  EXPECT_EQ(*vaddr->to_register_ref(), (RegisterRef{RegClass::VGPR, 7, 1}));
  EXPECT_NE(inst->disassemble().find("v7"), std::string::npos);
}

TEST(Gfx1250DecodeTest, BufferWithoutIdxenOffenDoesNotExposeVaddrRegister) {
  const uint32_t words[] = {
      0xC405C07Cu, // buffer_load_b128 v[32:35], s[4:7], s0
      0x00800820u,
      0x00000007u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "buffer_load_b128");
  ASSERT_GE(inst->num_src_operands(), 1);

  const Operand *vaddr = inst->src_operand(0);
  ASSERT_NE(vaddr, nullptr);
  EXPECT_EQ(vaddr->size_bits(), 0);
  EXPECT_FALSE(vaddr->to_register_ref().has_value());
  EXPECT_EQ(inst->disassemble().find("v7"), std::string::npos);
}

TEST(Gfx1250DecodeTest, WmmaF8f6f4UsesMatrixFormatOperandWidths) {
  const uint32_t words[] = {
      0xCC336010u,
      0x04421100u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_wmma_f32_16x16x128_f8f6f4 v[16:23], v[0:7], v[8:15], v[16:23] "
                                 "matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScaleF8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x02020900u,
      0xCC330006u,
      0x02026912u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33], v[52:67], 0, v0, v4");
}

TEST(Gfx1250DecodeTest, WmmaScale16F8f6f4ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC3A0000u,
      0x0202391Au,
      0xCC336012u,
      0x02021502u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale16_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale16_f32_16x16x128_f8f6f4 v[18:25], v[2:9], v[10:17], 0, "
            "v[26:27], v[28:29] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4");
}

TEST(Gfx1250DecodeTest, WmmaScaleF4_32x16x128ConsumesVop3px2Pair) {
  const uint32_t words[] = {
      0xCC350000u,
      0x02025328u,
      0xCC884000u,
      0x1A024110u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_wmma_scale_f32_32x16x128_f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(),
            "v_wmma_scale_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], 0, v40, v41");
}

TEST(Gfx1250DecodeTest, SwmmacPrintsIndexKeyAndReuseModifiers) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  const uint32_t index_key_words[] = {
      0xCC65081Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> index_key_inst(decoder->decode(index_key_words));
  ASSERT_NE(index_key_inst, nullptr);
  EXPECT_EQ(index_key_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 index_key:1");

  const uint32_t matrix_a_reuse_words[] = {
      0xCC65201Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_a_reuse_inst(decoder->decode(matrix_a_reuse_words));
  ASSERT_NE(matrix_a_reuse_inst, nullptr);
  EXPECT_EQ(matrix_a_reuse_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 matrix_a_reuse");

  const uint32_t matrix_b_reuse_words[] = {
      0xCC65401Au,
      0x1C8E0112u,
  };
  std::unique_ptr<Instruction> matrix_b_reuse_inst(decoder->decode(matrix_b_reuse_words));
  ASSERT_NE(matrix_b_reuse_inst, nullptr);
  EXPECT_EQ(matrix_b_reuse_inst->disassemble(),
            "v_swmmac_f32_16x16x64_f16 v[26:33], v[18:25], v[0:15], v35 matrix_b_reuse");
}

TEST(Gfx1250DecodeTest, VopdXyConsumesTwoDwords) {
  const uint32_t words[] = {
      0xCA500501u, // v_dual_cndmask_b32 v2, v1, v2 :: v_dual_mov_b32 v1, 0
      0x02000080u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_cndmask_b32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("v_dual_cndmask_b32"), std::string::npos);
  EXPECT_NE(inst->disassemble().find("v_dual_mov_b32"), std::string::npos);
}

TEST(Gfx1250DecodeTest, Vopd3ConsumesThreeDwords) {
  const uint32_t words[] = {
      0xCF455083u, // v_dual_lshlrev_b32 v1, 3, v0 :: v_dual_lshrrev_b32 v10, 6, v0
      0x00000086u,
      0x0A000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_lshlrev_b32 :: v_dual_lshrrev_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("v_dual_lshlrev_b32"), std::string::npos);
  EXPECT_NE(inst->disassemble().find("v_dual_lshrrev_b32"), std::string::npos);
}

TEST(Gfx1250DecodeTest, VopdLiteralConsumesThreeDwords) {
  const uint32_t words[] = {
      0xC8D006FFu, // v_dual_mul_f32 v4, 0x4f7ffffe, v3 :: v_dual_mov_b32 v3, 0
      0x04020080u,
      0x4F7FFFFEu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_mul_f32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_NE(inst->disassemble().find("0x4f7ffffe"), std::string::npos);
}

TEST(Gfx1250DecodeTest, VopdSourceOperandsFollowPrintedSlots) {
  const uint32_t words[] = {
      0xCF448082u, // v_dual_lshlrev_b32 v17, 2, v9 :: v_dual_mov_b32 v9, s11
      0x0009000Bu,
      0x09000011u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_dual_lshlrev_b32 :: v_dual_mov_b32");
  EXPECT_EQ(inst->num_src_operands(), 3);
  ASSERT_NE(inst->src_operand(2), nullptr);
  EXPECT_EQ(inst->src_operand(2)->name(), "s11");
  ASSERT_TRUE(inst->src_operand(2)->to_register_ref().has_value());
  EXPECT_EQ(*inst->src_operand(2)->to_register_ref(), (RegisterRef{RegClass::SGPR, 11, 1}));
}

TEST(Gfx1250SimulationTest, DispatchesEndpgmThroughConfig) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  EXPECT_EQ(sim.cp()->dispatched_count(), 1u);
  ASSERT_EQ(sim.cu()->num_wfs(), 1u);
  EXPECT_EQ(sim.cu()->wf(0)->wf_size(), 32u);
  EXPECT_TRUE(sim.cu()->wf(0)->is_halted());
}

TEST(Gfx1250SimulationTest, MultiWaveDispatchPacksWorkitemIdsInV0) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 104, 32, 2, false,
                                            false, false, 0, 0, 0, 0, 1);

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 4;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 32;
  pkt.grid_size_y = 4;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  std::vector<uint32_t> lane0_values;
  std::vector<uint32_t> lane31_values;
  for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
    auto *se = sim.xcd()->shader_engine(se_idx);
    for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
      auto *cu = se->compute_unit(cu_idx);
      for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
        auto *wf = cu->wf(wf_idx);
        if (!wf || wf->sgpr_alloc().count == 0)
          continue;
        const uint32_t vbase = wf->vgpr_alloc().base;
        lane0_values.push_back(cu->read_vgpr(vbase, 0));
        lane31_values.push_back(cu->read_vgpr(vbase, 31));
      }
    }
  }

  std::sort(lane0_values.begin(), lane0_values.end());
  std::sort(lane31_values.begin(), lane31_values.end());
  const std::vector<uint32_t> expected_lane0{0u, 1u << 10, 2u << 10, 3u << 10};
  const std::vector<uint32_t> expected_lane31{31u, 31u | (1u << 10), 31u | (2u << 10),
                                              31u | (3u << 10)};
  EXPECT_EQ(lane0_values, expected_lane0);
  EXPECT_EQ(lane31_values, expected_lane31);
}

TEST(Gfx1250SimulationTest, PartialWorkgroupMasksTailWaveExec) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 33, 33);
  step_until_xcd_halted(sim);

  std::vector<uint64_t> exec_masks;
  for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
    auto *se = sim.xcd()->shader_engine(se_idx);
    for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
      auto *cu = se->compute_unit(cu_idx);
      for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
        auto *wf = cu->wf(wf_idx);
        if (wf && wf->sgpr_alloc().count > 0)
          exec_masks.push_back(wf->exec());
      }
    }
  }

  std::sort(exec_masks.begin(), exec_masks.end());
  const std::vector<uint64_t> expected{1ULL, 0xFFFFFFFFULL};
  EXPECT_EQ(exec_masks, expected);
}

// Count the distinct workgroup ids across all wavefronts activated for the
// (single) dispatch. dispatched_count() counts dispatch packets, not
// workgroups, so it cannot observe the rounding; wavefront wg_ids can.
uint32_t count_dispatched_workgroups(Gfx1250Sim &sim) {
  std::set<uint32_t> wg_ids;
  for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
    auto *se = sim.xcd()->shader_engine(se_idx);
    for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
      auto *cu = se->compute_unit(cu_idx);
      for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
        auto *wf = cu->wf(wf_idx);
        if (wf && wf->sgpr_alloc().count > 0)
          wg_ids.insert(wf->wg_id());
      }
    }
  }
  return static_cast<uint32_t>(wg_ids.size());
}

// The grid-to-workgroup count must round up: a partial final workgroup still
// counts. With grid_size_x=65 and workgroup_size_x=32, HW dispatches 3 WGs
// (ceil(65/32)); the old floor division dropped the tail WG and dispatched 2.
TEST(Gfx1250SimulationTest, PartialFinalWorkgroupRoundsUpDispatchCount) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, /*grid_size_x=*/65, /*workgroup_size_x=*/32);
  step_until_xcd_halted(sim);

  EXPECT_EQ(count_dispatched_workgroups(sim), 3u);
}

// Same rounding rule in 2D: grid 65x33 with workgroups 32x16 dispatches
// ceil(65/32) * ceil(33/16) = 3 * 3 = 9 workgroups, not floor's 2 * 2 = 4.
TEST(Gfx1250SimulationTest, PartialFinalWorkgroupRoundsUpDispatchCount2D) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2; // 2D grid.
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 16;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 65;
  pkt.grid_size_y = 33;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  EXPECT_EQ(count_dispatched_workgroups(sim), 9u);
}

TEST(Gfx1250SimulationTest, DispatchPreloadsKernargDwordsIntoUserSgprs) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  struct Args {
    uint32_t skip;
    uint32_t first;
    uint32_t second;
  } args{0x11111111u, 0x22222222u, 0x33333333u};

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kKernargAddr);
  uint64_t kernel_object =
      sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 4, false, false, false,
                       kernel_code_properties, sizeof(args), 2, 1);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.cu()->num_wfs(), 1u);
  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  uint32_t sbase = wf->sgpr_alloc().base;
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 0), kKernargAddr);
  EXPECT_EQ(sim.cu()->read_sgpr(sbase + 2), args.first);
  EXPECT_EQ(sim.cu()->read_sgpr(sbase + 3), args.second);
}

TEST(Gfx1250SimulationTest, DispatchPreloadsKernargWhenDescriptorSizeIsUnknown) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  const std::array<uint32_t, 3> args{0x11111111u, 0x22222222u, 0x33333333u};

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(args.data()),
                         args.size() * sizeof(args[0]), kKernargAddr);
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 4, false,
                                            false, false, kernel_code_properties, 0, 2, 1);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.cu()->num_wfs(), 1u);
  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  uint32_t sbase = wf->sgpr_alloc().base;
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 0), kKernargAddr);
  EXPECT_EQ(sim.cu()->read_sgpr(sbase + 2), args[1]);
  EXPECT_EQ(sim.cu()->read_sgpr(sbase + 3), args[2]);
}

TEST(Gfx1250SimulationTest, SLoadB32ScalesImmediateOffset) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint64_t kKernargAddr = 0x400000;
  constexpr uint32_t kExpected = 0x12345678u;

  std::vector<uint32_t> code;
  append_instruction(code, make_s_load_b32_scaled_imm(4, 0, 1));
  append_instruction(code, S_WAIT_KMCNT_0_GFX12);
  append_instruction(code, S_ENDPGM_GFX12);

  uint32_t kernel_code_properties = 0;
  AMDHSA_BITS_SET(kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);

  Gfx1250Sim sim;
  write_global_u32(*sim.memory, kKernargAddr + 4, kExpected);
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size(), 104, 32, 2,
                                            false, false, false, kernel_code_properties, 16);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.cu()->num_wfs(), 1u);
  auto *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(read_wave_sgpr(*sim.cu(), *wf, 4), kExpected);
}

TEST(Gfx1250SimulationTest, TtmpWorkgroupIdsUseGridCoordinatesFor2DDispatch) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  uint64_t kernel_object =
      sim.write_kernel(0x10000, code, std::size(code), 104, 32, 2, true, false, false);

  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 2;
  pkt.workgroup_size_x = 32;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = 96;
  pkt.grid_size_y = 2;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kernel_object;
  queue.submit(pkt);
  step_until_xcd_halted(sim);

  amdgpu::Wavefront *target = nullptr;
  amdgpu::ComputeUnitCore *target_cu = nullptr;
  for (uint32_t se_idx = 0; se_idx < sim.xcd()->num_shader_engines(); ++se_idx) {
    auto *se = sim.xcd()->shader_engine(se_idx);
    for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
      auto *cu = se->compute_unit(cu_idx);
      for (uint32_t wf_idx = 0; wf_idx < cu->num_wf_slots(); ++wf_idx) {
        auto *wf = cu->wf(wf_idx);
        if (wf && wf->sgpr_alloc().count > 0 && wf->wg_id() == 4) {
          target = wf;
          target_cu = cu;
          break;
        }
      }
      if (target)
        break;
    }
    if (target)
      break;
  }

  ASSERT_NE(target, nullptr);
  const uint32_t sbase = target->sgpr_alloc().base;
  ASSERT_NE(target_cu, nullptr);
  EXPECT_EQ(target_cu->read_sgpr(sbase + 117), 1u);
  EXPECT_EQ(target_cu->read_sgpr(sbase + 115), 1u);
}

TEST(Gfx1250SimulationTest, SGetPcI64ReturnsNextInstructionAddress) {
  constexpr uint64_t kKernelAddr = 0x10000;
  const uint32_t code[] = {
      0xBE844700u, // s_get_pc_i64 s[4:5]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code, std::size(code));
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  amdgpu::Wavefront *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 4), entry_pc + sizeof(uint32_t));
}

TEST(Gfx1250SimulationTest, SAddPcI64SkipsRelativeToNextPc) {
  const uint32_t code[] = {
      0xBE804B84u, // s_add_pc_i64 4
      0xBE840081u, // s_mov_b32 s4, 1
      0xBE840082u, // s_mov_b32 s4, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 4), 2u);
}

TEST(Gfx1250SimulationTest, SSetPcI64JumpsToScalarAddress) {
  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint32_t kTargetWord = 5;
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  uint64_t target_pc = entry_pc + kTargetWord * sizeof(uint32_t);
  std::vector<uint32_t> code = {
      0xBE8400FFu,    static_cast<uint32_t>(target_pc), // s_mov_b32 s4, target_pc[31:0]
      0xBE850080u,                                      // s_mov_b32 s5, 0
      0xBE804804u,                                      // s_set_pc_i64 s[4:5]
      0xBE860081u,                                      // s_mov_b32 s6, 1
      0xBE860082u,                                      // s_mov_b32 s6, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size());
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  amdgpu::Wavefront *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 6), 2u);
}

TEST(Gfx1250SimulationTest, SSwapPcI64StoresReturnAddressAndJumps) {
  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint32_t kReturnWord = 4;
  constexpr uint32_t kTargetWord = 5;
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  uint64_t return_pc = entry_pc + kReturnWord * sizeof(uint32_t);
  uint64_t target_pc = entry_pc + kTargetWord * sizeof(uint32_t);
  std::vector<uint32_t> code = {
      0xBE8400FFu,    static_cast<uint32_t>(target_pc), // s_mov_b32 s4, target_pc[31:0]
      0xBE850080u,                                      // s_mov_b32 s5, 0
      0xBE864904u,                                      // s_swap_pc_i64 s[6:7], s[4:5]
      0xBE880081u,                                      // s_mov_b32 s8, 1
      0xBE880082u,                                      // s_mov_b32 s8, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size());
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  amdgpu::Wavefront *wf = sim.cu()->wf(0);
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 6), return_pc);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 8), 2u);
}

TEST(Gfx1250SimulationTest, SBitreplicateB64B32DuplicatesEachSourceBit) {
  const uint32_t code[] = {
      0xBE8200FFu,
      0x80000001u, // s_mov_b32 s2, 0x80000001
      0xBE841402u, // s_bitreplicate_b64_b32 s[4:5], s2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 4), 0xC000000000000003ULL);
}

TEST(Gfx1250SimulationTest, SGetShaderCyclesU64ReadsSimulationTime) {
  const uint32_t code[] = {
      0xBE840600u, // s_get_shader_cycles_u64 s[4:5]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  sim.engine->schedule_event_async(sim.cp()->doorbell_event(), 17);
  ASSERT_TRUE(sim.engine->step());
  ASSERT_EQ(sim.engine->global_time(), 17u);

  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  const auto observed_time = read_wave_sgpr64(*sim.cu(), *wf, 4);
  EXPECT_GE(observed_time, 17u);
  EXPECT_LE(observed_time, sim.engine->global_time());
}

TEST(Gfx1250SimulationTest, SSendmsgRtnB64ReadsRealtimeAndB32UsesPlaceholder) {
  const uint32_t code[] = {
      0xBE844D83u, // s_sendmsg_rtn_b64 s[4:5], sendmsg(MSG_RTN_GET_REALTIME)
      0xBE864C80u, // s_sendmsg_rtn_b32 s6, sendmsg(MSG_RTN_GET_DOORBELL)
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  sim.engine->schedule_event_async(sim.cp()->doorbell_event(), 23);
  ASSERT_TRUE(sim.engine->step());
  ASSERT_EQ(sim.engine->global_time(), 23u);

  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  const auto observed_time = read_wave_sgpr64(*sim.cu(), *wf, 4);
  EXPECT_GE(observed_time, 23u);
  EXPECT_LE(observed_time, sim.engine->global_time());
  EXPECT_EQ(read_wave_sgpr(*sim.cu(), *wf, 6), 0u);
}

TEST(Gfx1250SimulationTest, SMovrelsReadsM0IndexedScalarSources) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8200FFu,    0x11111111u, // s_mov_b32 s2, 0x11111111
      0xBE8300FFu,    0x22222222u, // s_mov_b32 s3, 0x22222222
      0xBE8400FFu,    0x33333333u, // s_mov_b32 s4, 0x33333333
      0xBE8500FFu,    0x44444444u, // s_mov_b32 s5, 0x44444444
      0xBE884002u,                 // s_movrels_b32 s8, s2
      0xBE8A4102u,                 // s_movrels_b64 s[10:11], s[2:3]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 8), 0x22222222u);
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 10), 0x4444444433333333ULL);
}

TEST(Gfx1250SimulationTest, SMovreldWritesM0IndexedScalarDestinations) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8200FFu,    0x55555555u, // s_mov_b32 s2, 0x55555555
      0xBE8300FFu,    0x66666666u, // s_mov_b32 s3, 0x66666666
      0xBE884202u,                 // s_movreld_b32 s8, s2
      0xBE8A4302u,                 // s_movreld_b64 s[10:11], s[2:3]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 9), 0x55555555u);
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 12), 0x6666666655555555ULL);
}

TEST(Gfx1250SimulationTest, SMovrelsd2B32UsesSeparatePackedM0Offsets) {
  const uint32_t code[] = {
      0xBEFD00FFu,    0x00000201u, // s_mov_b32 m0, 0x201
      0xBE8200FFu,    0x77777777u, // s_mov_b32 s2, 0x77777777
      0xBE8300FFu,    0x88888888u, // s_mov_b32 s3, 0x88888888
      0xBE884402u,                 // s_movrelsd_2_b32 s8, s2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 10), 0x88888888u);
}

TEST(Gfx1250SimulationTest, SplitNamedBarrierOpsReportIdleState) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8400FFu,    0xFFFFFFFFu, // s_mov_b32 s4, -1
      0xBE80517Du,                 // s_barrier_init m0
      0xBE80527Du,                 // s_barrier_join m0
      0xBE80577Du,                 // s_wakeup_barrier m0
      0xBE84507Du,                 // s_get_barrier_state s4, m0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(sim.cu()->read_sgpr(wf->sgpr_alloc().base + 4), 0u);
}

TEST(Gfx1250SimulationTest, VgprMsbModeTracksModeRegisterLayout) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);

  constexpr uint8_t kSetLayout = 0xB9;  // src0=1, src1=2, src2=3, dst=2.
  constexpr uint8_t kModeLayout = 0xE6; // dst,src0,src1,src2 packed in MODE.
  static_assert(amdgpu::set_vgpr_msb_to_mode_layout(kSetLayout) == kModeLayout);
  static_assert(amdgpu::mode_layout_to_set_vgpr_msb(kModeLayout) == kSetLayout);
  for (uint32_t layout = 0; layout <= 0xFF; ++layout) {
    EXPECT_EQ(amdgpu::mode_layout_to_set_vgpr_msb(
                  amdgpu::set_vgpr_msb_to_mode_layout(static_cast<uint8_t>(layout))),
              layout);
    EXPECT_EQ(amdgpu::set_vgpr_msb_to_mode_layout(
                  amdgpu::mode_layout_to_set_vgpr_msb(static_cast<uint8_t>(layout))),
              layout);
  }

  wf->set_vgpr_msb_mode(kSetLayout);
  EXPECT_EQ(wf->vgpr_msb_mode(), kSetLayout);
  EXPECT_EQ((wf->mode_raw() & amdgpu::VGPR_MSB_MODE_MASK) >> amdgpu::VGPR_MSB_MODE_SHIFT,
            kModeLayout);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src0), 1u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src1), 2u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src2), 3u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Dst), 2u);

  wf->set_mode_raw(kModeLayout << amdgpu::VGPR_MSB_MODE_SHIFT);
  EXPECT_EQ(wf->vgpr_msb_mode(), kSetLayout);
}

TEST(Gfx1250SimulationTest, SSetVgprMsbUpdatesWavefrontMode) {
  constexpr uint8_t kSetLayout = 0xB9;
  constexpr uint8_t kModeLayout = amdgpu::set_vgpr_msb_to_mode_layout(kSetLayout);
  Gfx1250Sim sim;
  const uint32_t code[] = {S_SET_VGPR_MSB | kSetLayout, S_ENDPGM_GFX12};

  amdgpu::Wavefront *wf =
      dispatch_one_wave(sim, code, std::size(code), kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  EXPECT_EQ(wf->vgpr_msb_mode(), kSetLayout);
  EXPECT_EQ((wf->mode_raw() & amdgpu::VGPR_MSB_MODE_MASK) >> amdgpu::VGPR_MSB_MODE_SHIFT,
            kModeLayout);
}

TEST(Gfx1250SimulationTest, VgprMsbRolesSelectHighVgprBanks) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  amdgpu::Wavefront *wf =
      dispatch_one_wave(sim, code, std::size(code), kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kLane = 0;
  constexpr uint8_t kSetLayout = 0xB9; // src0=1, src1=2, src2=3, dst=2.
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(kSetLayout);
  cu.write_vgpr(vb + 5, kLane, 0xDEADBEEFu);
  cu.write_vgpr(vb + 1 * 256 + 2, kLane, 0x11111111u);
  cu.write_vgpr(vb + 2 * 256 + 3, kLane, 0x22222222u);
  cu.write_vgpr(vb + 3 * 256 + 4, kLane, 0x33333333u);

  gfx1250::Operand src0(32, gfx1250::OperandType::OPR_SRC, 256 + 2);
  gfx1250::Operand src1(32, gfx1250::OperandType::OPR_VGPR, 3);
  gfx1250::Operand src2(32, gfx1250::OperandType::OPR_SRC_VGPR, 256 + 4);
  gfx1250::Operand dst(32, gfx1250::OperandType::OPR_VGPR, 5);
  src0.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
  src1.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);
  src2.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);
  dst.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);

  EXPECT_EQ(src0.read_lane(*wf, kLane), 0x11111111u);
  EXPECT_EQ(src1.read_lane(*wf, kLane), 0x22222222u);
  EXPECT_EQ(src2.read_lane(*wf, kLane), 0x33333333u);

  dst.write_lane(*wf, kLane, 0x44444444u);
  EXPECT_EQ(cu.read_vgpr(vb + 2 * 256 + 5, kLane), 0x44444444u);
  EXPECT_EQ(cu.read_vgpr(vb + 5, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250SimulationTest, PackedTrue16SourcesHonorGprIdx) {
  Gfx1250Sim sim;
  const uint32_t code[] = {S_ENDPGM_GFX12};
  amdgpu::Wavefront *wf =
      dispatch_one_wave(sim, code, std::size(code), kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kLane = 0;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_mode_raw(amdgpu::Wavefront::GPR_IDX_EN_BIT);
  wf->set_m0((1u << 8u) | 16u);
  cu.write_vgpr(vb + 2, kLane, 0xAAAA1111u);
  cu.write_vgpr(vb + 18, kLane, 0xBBBB2222u);

  gfx1250::Operand lo(16, gfx1250::OperandType::OPR_VGPR, 2, true);
  gfx1250::Operand hi(16, gfx1250::OperandType::OPR_VGPR, 128 + 2, true);
  EXPECT_EQ(lo.read_lane(*wf, kLane), 0x2222u);
  EXPECT_EQ(hi.read_lane(*wf, kLane), 0xBBBBu);
}

TEST(Gfx1250SimulationTest, VMovrelsReadsM0RelativeVgpr) {
  const uint32_t code[] = {
      0xBEFD0082u, // s_mov_b32 m0, 2
      0x7E1402FFu,
      0x00000063u, // v_mov_b32_e32 v10, 99
      0x7E028708u, // v_movrels_b32_e32 v1, v8
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code), 16);
  ASSERT_NE(wf, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 1, lane), 99u) << "lane " << lane;
}

TEST(Gfx1250SimulationTest, VopdMulDx9ZeroOverridesNanProducts) {
  constexpr auto dx9_mul = make_vopd3_pair(
      {.op = VopdOp::MulDx9ZeroF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 0, .dst = 4},
      {.op = VopdOp::MulDx9ZeroF32, .src0 = vopd_src0_vgpr(0), .src1 = 2, .src2 = 0, .dst = 5});
  constexpr auto ieee_mul = make_vopd3_pair(
      {.op = VopdOp::MulF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 0, .dst = 6},
      {.op = VopdOp::MulF32, .src0 = vopd_src0_vgpr(0), .src1 = 2, .src2 = 0, .dst = 7});

  std::vector<uint32_t> code;
  append_instruction(code, make_vmov_b32_literal(0, 0x7FC00000u)); // quiet NaN
  append_instruction(code, make_vmov_b32_literal(1, 0x00000000u)); // +0.0f
  append_instruction(code, make_vmov_b32_literal(2, 0x80000000u)); // -0.0f
  append_instruction(code, dx9_mul);
  append_instruction(code, ieee_mul);
  append_instruction(code, S_ENDPGM_GFX12);

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code.data(), code.size(), 16);
  ASSERT_NE(wf, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 4, lane), 0x00000000u) << "lane " << lane;
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 5, lane), 0x00000000u) << "lane " << lane;
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(sim.cu()->read_vgpr(vb + 6, lane))))
        << "lane " << lane;
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(sim.cu()->read_vgpr(vb + 7, lane))))
        << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VopdFmaUsesSingleRounding) {
  constexpr uint32_t kSrc0 = 0x3F800001u;
  constexpr uint32_t kSrc1 = 0x3F7FFFFFu;
  constexpr uint32_t kSrc2 = 0xBF800000u;
  constexpr auto fma = make_vopd3_pair(
      {.op = VopdOp::FmaF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 2, .dst = 4},
      {.op = VopdOp::FmaF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 2, .dst = 5});
  const uint32_t expected = std::bit_cast<uint32_t>(std::fma(
      std::bit_cast<float>(kSrc0), std::bit_cast<float>(kSrc1), std::bit_cast<float>(kSrc2)));
  ASSERT_EQ(expected, 0x337FFFFEu);

  std::vector<uint32_t> code;
  append_instruction(code, make_vmov_b32_literal(0, kSrc0));
  append_instruction(code, make_vmov_b32_literal(1, kSrc1));
  append_instruction(code, make_vmov_b32_literal(2, kSrc2));
  append_instruction(code, fma);
  append_instruction(code, S_ENDPGM_GFX12);

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code.data(), code.size(), 16);
  ASSERT_NE(wf, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 4, lane), expected) << "lane " << lane;
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 5, lane), expected) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VopdFmacUsesDestinationAccumulator) {
  const uint32_t code[] = {
      0x7E0002FFu,
      0x40000000u, // v_mov_b32_e32 v0, 2.0f
      0x7E1202FFu,
      0x3F800000u, // v_mov_b32_e32 v9, 1.0f
      0x7E1402FFu,
      0x40400000u, // v_mov_b32_e32 v10, 3.0f
      0x7E1C02FFu,
      0x40800000u, // v_mov_b32_e32 v14, 4.0f
      0xC900150Eu,    0x0A0800FFu,
      0x3F000000u, // v_dual_add_f32 v10, v14, v10 :: v_dual_fmac_f32 v9, 0.5f, v0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = dispatch_one_wave(sim, code, std::size(code), 16);
  ASSERT_NE(wf, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 10, lane), 0x40E00000u) << "lane " << lane;
    EXPECT_EQ(sim.cu()->read_vgpr(vb + 9, lane), 0x40000000u) << "lane " << lane;
  }
}

TEST(Gfx1250ExecutionTest, Vopd3CndmaskAppliesB32NegModifiers) {
  constexpr auto cndmask = make_vopd3_pair(
      {.op = VopdOp::CndmaskB32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 106, .dst = 2},
      {.op = VopdOp::CndmaskB32, .src0 = vopd_src0_vgpr(3), .src1 = 4, .src2 = 106, .dst = 5},
      /*negx=*/0x1, /*negy=*/0x2);

  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0x3u);
  wf->set_vcc(0x1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(cndmask.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_dual_cndmask_b32 :: v_dual_cndmask_b32");

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + 0, 0, 0x3F800000u);
  cu->write_vgpr(vb + 0, 1, 0x3F000000u);
  cu->write_vgpr(vb + 1, 0, 0x11223344u);
  cu->write_vgpr(vb + 1, 1, 0x55667788u);
  cu->write_vgpr(vb + 3, 0, 0x40000000u);
  cu->write_vgpr(vb + 3, 1, 0x40400000u);
  cu->write_vgpr(vb + 4, 0, 0x3F800000u);
  cu->write_vgpr(vb + 4, 1, 0x3F000000u);

  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr(vb + 2, 0), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr(vb + 2, 1), 0xBF000000u);
  EXPECT_EQ(cu->read_vgpr(vb + 5, 0), 0xBF800000u);
  EXPECT_EQ(cu->read_vgpr(vb + 5, 1), 0x40400000u);
}

TEST(Gfx1250SimulationTest, GlobalStoreWritesVisibleMemory) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t output_addr = 0x2000;

  const uint32_t code[] = {
      0xBE8400FFu,    static_cast<uint32_t>(output_addr), // s_mov_b32 s4, output_addr
      0xBE850080u,                                        // s_mov_b32 s5, 0
      0x30000082u,                                        // v_lshlrev_b32_e32 v0, 2, v0
      0x7E020281u,                                        // v_mov_b32_e32 v1, 1
      0xEE068004u,    0x00800000u,
      0x00000000u, // global_store_b32 v0, v1, s[4:5]
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code));
  for (uint32_t lane = 0; lane < 32; ++lane)
    sim.memory->write32(output_addr + lane * sizeof(uint32_t), 0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(sim.memory->read32(output_addr + lane * sizeof(uint32_t)), 1u) << "lane " << lane;
}

TEST(Gfx1250SimulationTest, BufferStoreUsesM0Soffset) {
  constexpr uint64_t kernel_addr = 0x10000;
  constexpr uint64_t output_addr = 0x2000;

  const uint32_t code[] = {
      0xBE8400FFu,    static_cast<uint32_t>(output_addr), // s_mov_b32 s4, output_addr
      0xBE850080u,                                        // s_mov_b32 s5, 0
      0xBE8600FFu,    0x1000u,                            // s_mov_b32 s6, num_records
      0xBEFD0090u,                                        // s_mov_b32 m0, 16
      0x300A0085u,                                        // v_lshlrev_b32_e32 v5, 5, v0
      0x7E000287u,                                        // v_mov_b32_e32 v0, 7
      0xC406807Du,    0x40800800u,
      0x00000005u, // buffer_store_b32 v0, v5, s[4:7], m0 offen
      0xBFC10000u, // s_wait_storecnt 0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kernel_addr, code, std::size(code), 128);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    sim.memory->write32(output_addr + lane * 32, 0);
    sim.memory->write32(output_addr + 16 + lane * 32, 0);
  }

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(sim.memory->read32(output_addr + lane * 32), 0u) << "lane " << lane;
    EXPECT_EQ(sim.memory->read32(output_addr + 16 + lane * 32), 7u) << "lane " << lane;
  }
}

} // namespace
