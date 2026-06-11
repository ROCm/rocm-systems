// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vimage.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vop3.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
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
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string kGfx1250ConfigPath = std::string(CONFIG_DIR) + "/amdgpu_gfx1250.json";

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
constexpr uint32_t kSdmaOpFence = 5;
constexpr uint32_t kSdmaOpPollRegmem = 8;
constexpr uint32_t kSdmaSubopFence64 = 2;
constexpr uint32_t kSdmaSubopPollMem64 = 5;

struct Gfx1250Sim {
  config::LoadedConfig loaded;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine;

  Gfx1250Sim() : loaded(config::load_config(kGfx1250ConfigPath, rocjitsu::kEmbeddedSchema)) {
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

void write_sdma_qword_address(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, const void *addr) {
  auto value = reinterpret_cast<uintptr_t>(addr);
  packet[lo_dw] = static_cast<uint32_t>(value) & ~0x7u;
  packet[hi_dw] = static_cast<uint32_t>(value >> 32);
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

std::string shell_quote(std::string_view value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted += c;
  }
  quoted += "'";
  return quoted;
}

std::optional<std::filesystem::path> real_kernel_path(const char *name) {
  const char *dir = std::getenv("ROCJITSU_GFX1250_KERNEL_DIR");
  if (!dir)
    return std::nullopt;
  return std::filesystem::path(dir) / (std::string(name) + ".o");
}

void expect_gfx1250_code_object_decodes(const CodeObject &co) {
  ASSERT_FALSE(co.text_sections().empty());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  size_t decoded = 0;
  for (const auto *sec : co.text_sections()) {
    ASSERT_EQ(sec->size() % sizeof(uint32_t), 0u) << sec->name();
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    for (size_t pc = 0; pc < words;) {
      std::unique_ptr<Instruction> inst;
      try {
        inst.reset(decoder->decode(&data[pc]));
      } catch (const std::exception &e) {
        FAIL() << "section " << sec->name() << " word " << pc << " raw 0x" << std::hex << data[pc]
               << ": " << e.what();
      }
      ASSERT_NE(inst, nullptr) << "section " << sec->name() << " word " << pc;
      size_t inst_words = inst->size() / sizeof(uint32_t);
      ASSERT_GT(inst_words, 0u) << inst->mnemonic();
      pc += inst_words;
      ++decoded;
    }
  }
  EXPECT_GT(decoded, 0u);
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

  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 0 * 4), 0x55000000u);
  EXPECT_EQ(cu->lds().read32(wf->lds_base() + 1 * 4), 0x55000001u);
  EXPECT_EQ(cu->lds().read64(wf->lds_base() + kBarrierLdsAddr), 7ull << 29);
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
  write_wave_sgpr(*cu, *wf, 0, 1u | (1u << 30));
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

TEST(Gfx1250CodeObjectTest, LlvmMcObjectMapsToTarget) {
  const char *llvm_mc = std::getenv("ROCJITSU_LLVM_MC");
  if (!llvm_mc)
    GTEST_SKIP() << "set ROCJITSU_LLVM_MC to an llvm-mc executable";

  auto dir = std::filesystem::temp_directory_path() / "rocjitsu-gfx1250-llvm-smoke";
  std::filesystem::create_directories(dir);
  auto asm_path = dir / "s_endpgm.s";
  auto obj_path = dir / "s_endpgm.o";
  {
    std::ofstream asm_file(asm_path);
    asm_file << ".text\ns_endpgm\n";
  }

  std::string cmd = shell_quote(llvm_mc) +
                    " -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 "
                    "-filetype=obj -o " +
                    shell_quote(obj_path.string()) + " " + shell_quote(asm_path.string());
  ASSERT_EQ(std::system(cmd.c_str()), 0);

  AmdGpuCodeObject co(obj_path.string());
  ASSERT_TRUE(co.is_valid());
  EXPECT_EQ(co.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
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
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 4), 17u);
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
  EXPECT_EQ(read_wave_sgpr64(*sim.cu(), *wf, 4), 23u);
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

TEST(Gfx1250RealKernelTest, VectorAddLoadsAndDecodes) {
  auto path = real_kernel_path("vector_add");
  if (!path)
    GTEST_SKIP() << "set ROCJITSU_GFX1250_KERNEL_DIR to gfx1250 HIP object directory";
  ASSERT_TRUE(std::filesystem::exists(*path)) << path->string();

  Executable exec(path->string());
  ASSERT_TRUE(exec.is_valid()) << "failed to load " << path->string();
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX1250), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_NE(co->kernel_descriptor_offset("vector_add"), 0u);

  expect_gfx1250_code_object_decodes(*co);
}

TEST(Gfx1250RealKernelTest, MatmulNaiveLoadsAndDecodes) {
  auto path = real_kernel_path("matmul_naive");
  if (!path)
    GTEST_SKIP() << "set ROCJITSU_GFX1250_KERNEL_DIR to gfx1250 HIP object directory";
  ASSERT_TRUE(std::filesystem::exists(*path)) << path->string();

  Executable exec(path->string());
  ASSERT_TRUE(exec.is_valid()) << "failed to load " << path->string();
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX1250), 0u);
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_NE(co->kernel_descriptor_offset("matmul_naive"), 0u);

  expect_gfx1250_code_object_decodes(*co);
}

TEST(Gfx1250RealKernelTest, VectorAddExecutesGolden) {
  auto path = real_kernel_path("vector_add");
  if (!path)
    GTEST_SKIP() << "set ROCJITSU_GFX1250_KERNEL_DIR to gfx1250 HIP object directory";
  ASSERT_TRUE(std::filesystem::exists(*path)) << path->string();

  Executable exec(path->string());
  ASSERT_TRUE(exec.is_valid()) << "failed to load " << path->string();
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(co, nullptr);

  constexpr uint64_t kd_addr = 0x10000;
  constexpr uint64_t a_addr = 0x100000;
  constexpr uint64_t b_addr = 0x200000;
  constexpr uint64_t c_addr = 0x300000;
  constexpr uint64_t kernarg_addr = 0x400000;
  constexpr uint32_t n = 64;

  Gfx1250Sim sim;
  co->load_to_memory(sim.memory, kd_addr);
  uint64_t kd_offset = co->kernel_descriptor_offset("vector_add");
  ASSERT_NE(kd_offset, 0u);
  uint64_t kernel_object = kd_addr + kd_offset;

  std::vector<float> a(n), b(n), expected(n), zeros(n, 0.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.25f;
    b[i] = static_cast<float>(i % 7) * -0.5f;
    expected[i] = a[i] + b[i];
  }
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(a.data()), n * sizeof(float), a_addr);
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(b.data()), n * sizeof(float), b_addr);
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), n * sizeof(float),
                         c_addr);

  struct {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint32_t n;
  } args = {a_addr, b_addr, c_addr, n};
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kernarg_addr);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, n, 64, kernarg_addr);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t i = 0; i < n; ++i) {
    float actual = std::bit_cast<float>(sim.memory->read32(c_addr + i * sizeof(float)));
    EXPECT_FLOAT_EQ(actual, expected[i]) << "element " << i;
  }
}

TEST(Gfx1250RealKernelTest, MatmulNaiveExecutesGolden) {
  auto path = real_kernel_path("matmul_naive");
  if (!path)
    GTEST_SKIP() << "set ROCJITSU_GFX1250_KERNEL_DIR to gfx1250 HIP object directory";
  ASSERT_TRUE(std::filesystem::exists(*path)) << path->string();

  Executable exec(path->string());
  ASSERT_TRUE(exec.is_valid()) << "failed to load " << path->string();
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(co, nullptr);

  constexpr uint64_t kd_addr = 0x10000;
  constexpr uint64_t a_addr = 0x100000;
  constexpr uint64_t b_addr = 0x200000;
  constexpr uint64_t c_addr = 0x300000;
  constexpr uint64_t kernarg_addr = 0x400000;
  constexpr uint32_t n = 4;
  constexpr uint32_t elements = n * n;

  Gfx1250Sim sim;
  co->load_to_memory(sim.memory, kd_addr);
  uint64_t kd_offset = co->kernel_descriptor_offset("matmul_naive");
  ASSERT_NE(kd_offset, 0u);
  uint64_t kernel_object = kd_addr + kd_offset;

  std::vector<float> a(elements), b(elements), expected(elements, 0.0f), zeros(elements, 0.0f);
  for (uint32_t i = 0; i < elements; ++i) {
    a[i] = static_cast<float>((i % 5) + 1);
    b[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  for (uint32_t row = 0; row < n; ++row)
    for (uint32_t col = 0; col < n; ++col)
      for (uint32_t k = 0; k < n; ++k)
        expected[row * n + col] += a[row * n + k] * b[k * n + col];

  sim.memory->load_image(reinterpret_cast<const uint8_t *>(a.data()), elements * sizeof(float),
                         a_addr);
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(b.data()), elements * sizeof(float),
                         b_addr);
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), elements * sizeof(float),
                         c_addr);

  struct {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint32_t n;
  } args = {a_addr, b_addr, c_addr, n};
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kernarg_addr);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 64, 64, kernarg_addr);
  sim.engine->run();
  sim.soc->flush_all();

  for (uint32_t i = 0; i < elements; ++i) {
    float actual = std::bit_cast<float>(sim.memory->read32(c_addr + i * sizeof(float)));
    EXPECT_NEAR(actual, expected[i], 1e-5f) << "element " << i;
  }
}

} // namespace
