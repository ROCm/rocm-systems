// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_tensor_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include <gtest/gtest.h>

namespace rocjitsu::consan {
namespace {
namespace tdm = amdgpu::tensor_dma_detail;

struct TensorCase {
  std::array<uint32_t, 4> d0{1, 4096, 0, 0};
  std::array<uint32_t, 8> d1{};
  std::array<uint32_t, 4> d2{};
  std::array<uint32_t, 4> d3{};
  bool null2 = false;
  bool null3 = false;
};

TensorCase tile(std::array<uint32_t, 5> dimensions) {
  TensorCase result;
  result.d1[0] = (1u << 20) | (5u << 22) | (3u << 25); // Byte elements, dword padding.
  result.d1[3] = dimensions[0] << 16;
  result.d1[4] = dimensions[1] | (dimensions[2] << 16);
  result.d2[3] = dimensions[3] << 16;
  result.d3[2] = dimensions[4] << 16;
  return result;
}

TEST(ConSanTensor, SelectedAddressesMatchEnumeratedDenseGatherAndRepeatedTransfers) {
  std::vector<TensorCase> cases{tile({3, 4, 0, 0, 0}), tile({2, 3, 2, 2, 2}), tile({3, 0, 2, 0, 0}),
                                tile({3, 2, 0, 0, 2}), tile({})};
  auto repeated = tile({3, 2, 0, 0, 0});
  repeated.d1[0] |= 1u << 19;
  repeated.d2[1] = 16;
  repeated.d2[3] = 2u << 16; // Three iterations, separated LDS tiles.
  cases.push_back(repeated);
  repeated.d2[1] = 0; // Aliased repeated tiles must still select valid addresses.
  cases.push_back(repeated);
  for (const bool wide_indices : {false, true}) {
    auto gather = tile({3, 4, 0, 0, 0});
    gather.d0[0] |= (1u << 31) | (wide_indices ? 1u << 30 : 0u);
    gather.d1[0] |= 1u << 19;    // Gather ignores the iterate bit.
    gather.d2.fill(0xffffffffu); // Indices, not higher tile dimensions.
    gather.d3.fill(0xffffffffu);
    cases.push_back(gather);
  }
  auto null_optional = tile({2, 3, 2, 0, 0});
  null_optional.null2 = null_optional.null3 = true;
  cases.push_back(null_optional);
  auto inactive = tile({3, 4, 0, 0, 0});
  inactive.d0[0] = 0;
  cases.push_back(inactive);

  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    SCOPED_TRACE(case_index);
    const auto &c = cases[case_index];
    amdgpu::GpuMemory memory("tensor_selector_mem");
    amdgpu::L2Cache l2("tensor_selector_l2");
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 64;
    config.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("tensor_selector", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
    ASSERT_NE(wave, nullptr);
    const auto vb = wave->vgpr_alloc().base;
    const auto sb = wave->sgpr_alloc().base;
    for (uint32_t reg = 0; reg < 106; ++reg)
      cu->write_sgpr(sb + reg, 0xbeef0000u + reg);
    const auto write_group = [&](uint32_t base, const auto &values) {
      for (size_t i = 0; i < values.size(); ++i)
        cu->write_sgpr(sb + base + i, values[i]);
    };
    write_group(8, c.d0);
    write_group(20, c.d1);
    write_group(40, c.d2);
    write_group(50, c.d3);
    std::array<uint32_t, 106> scalar_before{};
    for (uint32_t reg = 0; reg < 106; ++reg)
      scalar_before[reg] = cu->read_sgpr(sb + reg);
    ProgramSite site;
    site.origin = AccessOrigin::TensorLds;
    site.kind = LdsAccessKind::Write;
    site.operands.tensor_descriptor_sgprs =
        std::array<uint16_t, 4>{8, 20, uint16_t(c.null2 ? 124 : 40), uint16_t(c.null3 ? 124 : 50)};
    std::vector<uint32_t> words;
    ASSERT_TRUE(
        detail::append_select_tensor_load_element(words, site, 10, 11, 20, 21, 30, config.arch));
    ASSERT_TRUE(
        detail::append_materialize_tensor_load_lds_address(words, site, 20, 22, 35, config.arch));
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(i * 4, words[i]);
    const auto desc = tdm::parse_descriptor(c.d0, c.d1, c.null2 ? std::array<uint32_t, 4>{} : c.d2,
                                            c.null3 ? std::array<uint32_t, 4>{} : c.d3);
    const tdm::TensorDmaLayout layout(desc);
    tdm::TensorDmaState reference(desc, false);
    auto visit = [&](uint64_t global, uint64_t lds, bool in_bounds) {
      tdm::append_copy(reference, *wave, global, lds, in_bounds);
    };
    if (desc.active()) {
      if (desc.gather)
        tdm::for_each_gather_tensor_element(desc, layout, visit);
      else
        tdm::for_each_dense_tensor_element(desc, layout, visit);
    }
    const uint32_t iterations = desc.iterate ? desc.iteration_count : 1u;
    const uint32_t count = reference.elements.size() / iterations;
    for (const uint32_t exec : {0xffffffffu, 0x80010005u, 0u}) {
      std::array<uint32_t, 32> hashes{}, iteration_hashes{};
      for (uint32_t lane = 0; lane < 32; ++lane) {
        hashes[lane] = lane == 31 ? 0xffffffffu : lane * 0x9e3779b9u;
        iteration_hashes[lane] = lane % 2 ? 0xffffffffu : lane * 0x85ebca6bu;
        for (uint32_t reg = 10; reg < 38; ++reg)
          cu->write_vgpr(vb + reg, lane, 0xabc00000u + reg * 64u + lane);
        cu->write_vgpr(vb + 10, lane, hashes[lane]);
        cu->write_vgpr(vb + 11, lane, iteration_hashes[lane]);
      }
      wave->pc = 0;
      wave->set_exec(exec);
      wave->write_scc(true);
      size_t steps = 0;
      while (wave->pc < words.size() * 4) {
        ASSERT_LT(steps++, words.size());
        cu->step();
      }
      cu->flush_all();
      for (uint32_t lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(cu->read_vgpr(vb + 10, lane), hashes[lane]);
        EXPECT_EQ(cu->read_vgpr(vb + 11, lane), iteration_hashes[lane]);
        if ((exec >> lane) & 1u) {
          EXPECT_EQ(cu->read_vgpr(vb + 21, lane), count);
          if (count != 0) {
            const size_t selected = (uint64_t{hashes[lane]} * count) >> 32;
            const size_t iter = (uint64_t{iteration_hashes[lane]} * iterations) >> 32;
            EXPECT_EQ(cu->read_vgpr(vb + 22, lane),
                      reference.elements[iter * count + selected].lds_address - wave->lds_base());
          }
        } else {
          for (uint32_t reg = 20; reg < 38; ++reg)
            EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xabc00000u + reg * 64u + lane);
        }
      }
      for (uint32_t reg = 0; reg < 106; ++reg)
        EXPECT_EQ(cu->read_sgpr(sb + reg), scalar_before[reg]);
      EXPECT_EQ(wave->exec(), exec);
      EXPECT_TRUE(wave->read_scc());
    }
    wave->halt();
  }
}

TEST(ConSanTensor, RejectsDescriptorAndScratchAliasesBeforeEmission) {
  ProgramSite site;
  site.origin = AccessOrigin::TensorLds;
  site.kind = LdsAccessKind::Write;
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 20, 124, 124};
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  for (const std::array<uint16_t, 5> regs : {std::array<uint16_t, 5>{10, 11, 20, 20, 30},
                                             {10, 11, 10, 21, 30},
                                             {10, 11, 20, 11, 30},
                                             {10, 11, 20, 21, 252},
                                             {30, 11, 20, 21, 30},
                                             {10, 11, 20, 34, 30},
                                             {256, 11, 20, 21, 30}}) {
    EXPECT_FALSE(detail::append_select_tensor_load_element(
        words, site, regs[0], regs[1], regs[2], regs[3], regs[4], ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(words, prefix);
  }
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 20, 103, 124};
  EXPECT_FALSE(detail::append_select_tensor_load_element(words, site, 10, 11, 20, 21, 30,
                                                         ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(words, prefix);
}
} // namespace
} // namespace rocjitsu::consan
