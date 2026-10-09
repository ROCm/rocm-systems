// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amdgpu_elf_test_support.h"
#include "aql_queue.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

enum class Gfx1251E2eSetup {
  PackedLshlAddU64,
  PackedAddU64,
  PackedSubU64,
  PackedF64Binary,
  PackedF64Fma,
  F64Wmma,
};

struct Gfx1251E2eCase {
  std::string_view name;
  std::string_view mnemonic;
  Gfx1251E2eSetup setup;
  std::array<uint32_t, 2> instruction;
  std::array<uint32_t, 4> expected_words;
  uint8_t store_vgpr;
};

class MnemonicExecutionCounter final : public ExecutionPlugin {
public:
  explicit MnemonicExecutionCounter(std::string_view expected)
      : ExecutionPlugin("gfx1251_e2e_mnemonic_counter"), expected_(expected) {}

  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst,
                                        amdgpu::Wavefront &) override {
    if (inst.mnemonic() == expected_)
      count_.fetch_add(1, std::memory_order_relaxed);
  }

  uint32_t count() const { return count_.load(std::memory_order_relaxed); }

private:
  std::string expected_;
  std::atomic<uint32_t> count_{0};
};

class Gfx1251SimulatorInstructionTest : public testing::TestWithParam<Gfx1251E2eCase> {};

TEST_P(Gfx1251SimulatorInstructionTest, DispatchesCodeObjectAndVerifiesExactOutput) {
  using namespace rocr::llvm::amdhsa;

  const Gfx1251E2eCase &test_case = GetParam();
  constexpr uint64_t kCodeObjectBase = 0x100000;
  constexpr uint64_t kOutputAddress = 0x2000;
  constexpr uint64_t kCompletionSignal = 0x4000;
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kWaveSize = 32;

  std::vector<uint32_t> code;
  const auto append = [&code](const auto &words) {
    code.insert(code.end(), words.begin(), words.end());
  };
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 255, .sdst = 2}));
  code.push_back(static_cast<uint32_t>(kOutputAddress));
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 3}));
  // Preserve the work-item ID before the WMMA case reuses v0 as matrix A.
  append(cdna5::build_vop2(cdna5::kVLshlrevB32Vop2, {.src0 = 132, .vsrc1 = 0, .vdst = 24}));

  const auto append_literal_vmov = [&append, &code](uint8_t vdst, uint32_t value) {
    append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 255, .vdst = vdst}));
    code.push_back(value);
  };
  const auto append_u64_pair = [&append_literal_vmov](uint8_t vdst,
                                                      std::array<uint64_t, 2> values) {
    for (uint32_t element = 0; element < values.size(); ++element) {
      append_literal_vmov(vdst + 2 * element, static_cast<uint32_t>(values[element]));
      append_literal_vmov(vdst + 2 * element + 1, static_cast<uint32_t>(values[element] >> 32));
    }
  };

  switch (test_case.setup) {
  case Gfx1251E2eSetup::PackedLshlAddU64:
    append_u64_pair(8, {2u, 3u});
    append_literal_vmov(12, 4u);
    append_literal_vmov(13, 4u);
    append_u64_pair(16, {7u, 11u});
    break;
  case Gfx1251E2eSetup::PackedAddU64:
    append_u64_pair(8, {2u, 3u});
    append_u64_pair(12, {7u, 11u});
    break;
  case Gfx1251E2eSetup::PackedSubU64:
    append_u64_pair(8, {9u, 14u});
    append_u64_pair(12, {7u, 11u});
    break;
  case Gfx1251E2eSetup::PackedF64Binary:
    append_u64_pair(8, {0x4000000000000000ULL, 0x4008000000000000ULL});  // 2.0, 3.0
    append_u64_pair(12, {0x4010000000000000ULL, 0x4014000000000000ULL}); // 4.0, 5.0
    break;
  case Gfx1251E2eSetup::PackedF64Fma:
    append_u64_pair(8, {0x4000000000000000ULL, 0x4008000000000000ULL});  // 2.0, 3.0
    append_u64_pair(12, {0x4010000000000000ULL, 0x4014000000000000ULL}); // 4.0, 5.0
    append_u64_pair(16, {0x401c000000000000ULL, 0x4026000000000000ULL}); // 7.0, 11.0
    break;
  case Gfx1251E2eSetup::F64Wmma:
    // Uniform inputs make every modeled result 4 * 2.0 * 3.0 - 5.0 =
    // 19.0, so this end-to-end case does not depend on element placement.
    append_u64_pair(0, {0x4000000000000000ULL, 0x4000000000000000ULL});
    append_u64_pair(4, {0x4008000000000000ULL, 0x4008000000000000ULL});
    for (uint8_t reg = 8; reg < 24; reg += 4)
      append_u64_pair(reg, {0xc014000000000000ULL, 0xc014000000000000ULL});
    break;
  }

  // Every case uses the exact base-form bytes from the public LLVM gfx1251 MC tests.
  append(test_case.instruction);
  append(cdna5::build_vglobal(cdna5::kGlobalStoreB128Vglobal,
                              {.saddr = 2, .vsrc = test_case.store_vgpr, .vaddr = 24}));
  append(cdna5::build_sopp(cdna5::kSWaitStorecntSopp));
  append(cdna5::build_sopp(cdna5::kSEndpgmSopp));

  test_support::TestKernelDescriptor descriptor{};
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  3);
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  15);
  const std::vector<uint8_t> image =
      test_support::make_minimal_amdgpu_kernel_elf(code, EF_AMDGPU_MACH_AMDGCN_GFX1251, descriptor);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  ASSERT_EQ(code_object.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
  const uint64_t descriptor_offset = code_object.kernel_descriptor_offset("kernel");
  ASSERT_NE(descriptor_offset, 0u);

  auto loaded =
      config::load_config(test::config_path("gfx1251_synthetic.json"), rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.target, code_object.target_id());
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  ASSERT_NE(soc, nullptr);
  ASSERT_NE(memory, nullptr);
  auto plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto mnemonic_counter = std::make_unique<MnemonicExecutionCounter>(test_case.mnemonic);
  MnemonicExecutionCounter *mnemonic_counter_ptr = mnemonic_counter.get();
  ASSERT_TRUE(plugin_group->add(std::move(mnemonic_counter)));
  soc->set_plugin_group(std::move(plugin_group));
  simdojo::SimulationEngine engine(loaded.engine_config);
  engine.topology().set_root(loaded.take_root());
  loaded.wire_links(engine.topology());
  engine.create();

  code_object.load_to_memory(memory, kCodeObjectBase);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < test_case.expected_words.size(); ++word)
      memory->write32(kOutputAddress + lane * 16 + word * sizeof(uint32_t), 0xDEADBEEFu);
  memory->write64(kCompletionSignal + kSignalValueOffset, 1);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.setup = 1;
  packet.workgroup_size_x = kWaveSize;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = kWaveSize;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.kernel_object = kCodeObjectBase + descriptor_offset;
  packet.completion_signal.handle = kCompletionSignal;

  auto *cp = soc->xcd(0)->command_processor();
  test::AqlQueue queue(memory, cp);
  queue.submit(packet);
  engine.run();
  soc->flush_all();

  EXPECT_EQ(cp->dispatched_count(), 1u);
  EXPECT_EQ(memory->read64(test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
  EXPECT_EQ(memory->read64(kCompletionSignal + kSignalValueOffset), 0u);
  EXPECT_EQ(mnemonic_counter_ptr->count(), 1u);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < test_case.expected_words.size(); ++word)
      EXPECT_EQ(memory->read32(kOutputAddress + lane * 16 + word * sizeof(uint32_t)),
                test_case.expected_words[word])
          << "lane " << lane << ", word " << word;
}

INSTANTIATE_TEST_SUITE_P(AllNineInstructions, Gfx1251SimulatorInstructionTest,
                         testing::Values(
                             // Exact public LLVM gfx1251_asm_vop3p.s base-form encodings.
                             Gfx1251E2eCase{"PkLshlAddU64",
                                            "v_pk_lshl_add_u64",
                                            Gfx1251E2eSetup::PackedLshlAddU64,
                                            {0xCC7E4004u, 0x1C421908u},
                                            {39u, 0u, 59u, 0u},
                                            4},
                             Gfx1251E2eCase{"PkAddNcU64",
                                            "v_pk_add_nc_u64",
                                            Gfx1251E2eSetup::PackedAddU64,
                                            {0xCC4C4004u, 0x1A021908u},
                                            {9u, 0u, 14u, 0u},
                                            4},
                             Gfx1251E2eCase{"PkSubNcU64",
                                            "v_pk_sub_nc_u64",
                                            Gfx1251E2eSetup::PackedSubU64,
                                            {0xCC4D4004u, 0x1A021908u},
                                            {2u, 0u, 3u, 0u},
                                            4},
                             Gfx1251E2eCase{"PkAddF64",
                                            "v_pk_add_f64",
                                            Gfx1251E2eSetup::PackedF64Binary,
                                            {0xCC4B4004u, 0x1A021908u},
                                            {0u, 0x40180000u, 0u, 0x40200000u},
                                            4},
                             Gfx1251E2eCase{"PkMulF64",
                                            "v_pk_mul_f64",
                                            Gfx1251E2eSetup::PackedF64Binary,
                                            {0xCC3C4004u, 0x1A021908u},
                                            {0u, 0x40200000u, 0u, 0x402e0000u},
                                            4},
                             Gfx1251E2eCase{"PkMaxNumF64",
                                            "v_pk_max_num_f64",
                                            Gfx1251E2eSetup::PackedF64Binary,
                                            {0xCC4E4004u, 0x1A021908u},
                                            {0u, 0x40100000u, 0u, 0x40140000u},
                                            4},
                             Gfx1251E2eCase{"PkMinNumF64",
                                            "v_pk_min_num_f64",
                                            Gfx1251E2eSetup::PackedF64Binary,
                                            {0xCC4F4004u, 0x1A021908u},
                                            {0u, 0x40000000u, 0u, 0x40080000u},
                                            4},
                             Gfx1251E2eCase{"PkFmaF64",
                                            "v_pk_fma_f64",
                                            Gfx1251E2eSetup::PackedF64Fma,
                                            {0xCC3B4004u, 0x1C421908u},
                                            {0u, 0x402e0000u, 0u, 0x403a0000u},
                                            4},
                             // Exact public LLVM gfx1251_asm_wmma_w32.s base-form encoding.
                             Gfx1251E2eCase{"WmmaF6416x16x4F64",
                                            "v_wmma_f64_16x16x4_f64",
                                            Gfx1251E2eSetup::F64Wmma,
                                            {0xCC5B0008u, 0x1C220900u},
                                            {0u, 0x40330000u, 0u, 0x40330000u},
                                            8}),
                         [](const testing::TestParamInfo<Gfx1251E2eCase> &info) {
                           return std::string(info.param.name);
                         });

TEST(Gfx1251SimulatorTest, DispatchesTargetSpecificSetregSemantics) {
  using namespace rocr::llvm::amdhsa;

  constexpr uint64_t kCodeObjectBase = 0x100000;
  constexpr uint64_t kOutputAddress = 0x2000;
  constexpr uint64_t kCompletionSignal = 0x4000;
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kWaveSize = 32;

  // Exact public LLVM gfx1251 MC encodings for MODE[19:12], plus the
  // target-specific immediately-adjacent S_SET_VGPR_MSB sequence.
  constexpr uint32_t kSetregB32ModeVgprMsb = 0xB9043B01u;
  constexpr uint32_t kSetregImm32ModeVgprMsb = 0xB9803B01u;
  constexpr uint32_t kGetregModeVgprMsb = 0xB8843B01u;
  constexpr uint32_t kSetVgprMsbZero = 0xBF860000u;
  constexpr uint32_t kSetVgprMsb41 = 0xBF860041u;

  std::vector<uint32_t> code;
  const auto append = [&code](const auto &words) {
    code.insert(code.end(), words.begin(), words.end());
  };
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 255, .sdst = 2}));
  code.push_back(static_cast<uint32_t>(kOutputAddress));
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 3}));
  append(cdna5::build_vop2(cdna5::kVLshlrevB32Vop2, {.src0 = 132, .vsrc1 = 0, .vdst = 24}));

  const auto append_literal_smov = [&append, &code](uint8_t sdst, uint32_t value) {
    append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 255, .sdst = sdst}));
    code.push_back(value);
  };
  const auto capture_mode_vgpr_msb = [&](uint8_t vdst, uint32_t offset) {
    code.push_back(kGetregModeVgprMsb);
    // MODE.VGPR_MSB also selects high vector-register banks. Clear it only
    // after the scalar snapshot so the output move and store use local VGPRs.
    code.push_back(kSetVgprMsbZero);
    append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 4, .vdst = vdst}));
    append(cdna5::build_vglobal(cdna5::kGlobalStoreB32Vglobal,
                                {.saddr = 2, .vsrc = vdst, .vaddr = 24, .ioffset = offset}));
  };

  append_literal_smov(/*sdst=*/4, 0xA5u);
  code.push_back(kSetregB32ModeVgprMsb);
  capture_mode_vgpr_msb(/*vdst=*/4, /*offset=*/0);

  code.push_back(kSetregImm32ModeVgprMsb);
  code.push_back(0xC3u);
  capture_mode_vgpr_msb(/*vdst=*/5, /*offset=*/4);

  code.push_back(kSetregImm32ModeVgprMsb);
  code.push_back(0xE7u);
  code.push_back(kSetVgprMsb41);
  capture_mode_vgpr_msb(/*vdst=*/6, /*offset=*/8);

  append(cdna5::build_sopp(cdna5::kSWaitStorecntSopp));
  append(cdna5::build_sopp(cdna5::kSEndpgmSopp));

  test_support::TestKernelDescriptor descriptor{};
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  3);
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  15);
  const std::vector<uint8_t> image =
      test_support::make_minimal_amdgpu_kernel_elf(code, EF_AMDGPU_MACH_AMDGCN_GFX1251, descriptor);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  ASSERT_EQ(code_object.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
  const uint64_t descriptor_offset = code_object.kernel_descriptor_offset("kernel");
  ASSERT_NE(descriptor_offset, 0u);

  auto loaded =
      config::load_config(test::config_path("gfx1251_synthetic.json"), rocjitsu::kEmbeddedSchema);
  ASSERT_EQ(loaded.target, code_object.target_id());
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  ASSERT_NE(soc, nullptr);
  ASSERT_NE(memory, nullptr);
  simdojo::SimulationEngine engine(loaded.engine_config);
  engine.topology().set_root(loaded.take_root());
  loaded.wire_links(engine.topology());
  engine.create();

  code_object.load_to_memory(memory, kCodeObjectBase);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < 3; ++word)
      memory->write32(kOutputAddress + lane * 16 + word * sizeof(uint32_t), 0xDEADBEEFu);
  memory->write64(kCompletionSignal + kSignalValueOffset, 1);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.setup = 1;
  packet.workgroup_size_x = kWaveSize;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = kWaveSize;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.kernel_object = kCodeObjectBase + descriptor_offset;
  packet.completion_signal.handle = kCompletionSignal;

  auto *cp = soc->xcd(0)->command_processor();
  test::AqlQueue queue(memory, cp);
  queue.submit(packet);
  engine.run();
  soc->flush_all();

  EXPECT_EQ(cp->dispatched_count(), 1u);
  EXPECT_EQ(memory->read64(test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
  EXPECT_EQ(memory->read64(kCompletionSignal + kSignalValueOffset), 0u);
  constexpr std::array<uint32_t, 3> kExpected{0xA5u, 0xC3u, 0x05u};
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    for (uint32_t word = 0; word < kExpected.size(); ++word)
      EXPECT_EQ(memory->read32(kOutputAddress + lane * 16 + word * sizeof(uint32_t)),
                kExpected[word])
          << "lane " << lane << ", word " << word;
}

} // namespace
