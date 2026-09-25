// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_tensor_access.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
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

TEST(ConSanTensor, DefaultProbeExecutesDmaAndPublishesRuntimeWidthWithGuestStatePreserved) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  const auto tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 0, .vaddr1 = 12, .vaddr2 = 124, .vaddr3 = 124});
  std::vector<uint32_t> guest(tensor.begin(), tensor.end());
  guest.push_back(build_s_endpgm(arch));
  const auto bytes = make_gfx1250_code_object(guest, "tensor_default_emission");
  const auto inventory = test_lower_consan(bytes, test_options());
  ASSERT_EQ(inventory.program_inventory.access_sites().size(), 1u);
  const auto &site = inventory.program_inventory.access_sites().front();
  const Candidate candidate(site);
  detail::AccessEmissionPlan plan;
  plan.tensor_full_wave_resources = true;
  plan.supercollider_report_buffer_address = 0x200000;
  plan.report_generation = 1;
  plan.exec_save_sgpr = 88;
  plan.persistent_sgprs.set_owner_epoch(80, 81);
  plan.owner_epoch_vgprs = {.owner = 31, .epoch = 32};
  plan.dispatch_id.literal = 5;
  plan.workgroup_sources.x = WorkgroupSource::scalar(82);
  plan.scratch_vgpr = 20;
  plan.base_scratch_vgpr_count = 8;
  plan.scratch_vgpr_count = 13;
  plan.watchpoints_offset = 0x1000;
  plan.causal_windows_offset = 0x2000;
  plan.pending_acquires_offset = 0x3000;
  plan.pending_acquire_owner_bank_count = 1;
  std::vector<std::string> errors;
  uint32_t guest_offset = 0, guest_size = 0;
  const auto probe = detail::build_direct_watchpoint_words(bytes, candidate, 0, plan, nullptr, arch,
                                                           errors, &guest_offset, &guest_size);
  ASSERT_TRUE(probe) << testing::PrintToString(errors);
  EXPECT_EQ(guest_size, 3u);
  ASSERT_LE(guest_offset / 4 + guest_size, probe->size());
  EXPECT_TRUE(std::equal(tensor.begin(), tensor.end(), probe->begin() + guest_offset / 4));
  SpillManager manager(0, 256);
  const auto spill = build_vgpr_spill_sequence(manager, 20, 13, arch);
  ASSERT_TRUE(spill);
  const auto full_spill = detail::tensor_full_wave_spill(*spill, 96, arch);
  ASSERT_TRUE(full_spill);
  auto words = full_spill->save_words;
  words.insert(words.end(), probe->begin(), probe->end());
  words.insert(words.end(), full_spill->restore_words.begin(), full_spill->restore_words.end());
  for (const auto [active, global_in_bounds, atomic_barrier] :
       {std::array<bool, 3>{true, true, false}, std::array<bool, 3>{false, true, false},
        std::array<bool, 3>{true, false, false}, std::array<bool, 3>{true, true, true},
        std::array<bool, 3>{false, true, true}}) {
    for (uint32_t size_log2 = 0; size_log2 < 4; ++size_log2) {
      for (uint32_t exec : {0xffffffffu, 0x80000001u, 0u}) {
        SCOPED_TRACE(atomic_barrier);
        SCOPED_TRACE(global_in_bounds);
        SCOPED_TRACE(active);
        SCOPED_TRACE(size_log2);
        SCOPED_TRACE(exec);
        amdgpu::GpuMemory memory("tensor_probe_mem");
        amdgpu::L2Cache l2("tensor_probe_l2");
        l2.set_backing_memory(&memory);
        amdgpu::ComputeUnitCore::Config config{};
        config.arch = arch;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 106;
        config.vgprs_per_wf = 64;
        config.lds_size_kb = 4;
        auto cu = amdgpu::ComputeUnitCore::create("tensor_probe", config, &memory, &l2);
        ASSERT_NE(cu, nullptr);
        auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
        ASSERT_NE(wave, nullptr);
        const auto lds_base = cu->allocate_lds(4096);
        ASSERT_NE(lds_base, UINT32_MAX);
        wave->set_lds_base(lds_base);
        wave->set_scratch_base(0x400000);
        wave->set_scratch_lane_size(256);
        const auto sb = wave->sgpr_alloc().base;
        const auto vb = wave->vgpr_alloc().base;
        cu->write_sgpr(sb, active ? 1u : 0u);
        cu->write_sgpr(sb + 1, 0);
        cu->write_sgpr(sb + 2, 0x100000);
        cu->write_sgpr(sb + 3, 0x80000000u);
        cu->write_sgpr(sb + 12, (size_log2 << 16) | (atomic_barrier ? 1u << 18 : 0u));
        cu->write_sgpr(sb + 13,
                       (global_in_bounds ? 16u << 16 : 0u) | (atomic_barrier ? 512u >> 3 : 0u));
        cu->write_sgpr(sb + 14, 0);
        cu->write_sgpr(sb + 15, 16u << 16);
        for (uint32_t reg = 16; reg < 20; ++reg)
          cu->write_sgpr(sb + reg, 0);
        cu->lds().write64(wave->lds_base() + 512, 0);
        cu->write_sgpr(sb + 80, 2);
        cu->write_sgpr(sb + 81, 3);
        cu->write_sgpr(sb + 82, 4);
        for (uint32_t byte = 0; byte < (16u << size_log2); ++byte) {
          memory.write8(0x100000 + byte, static_cast<uint8_t>(byte + 1));
          cu->lds().write8(wave->lds_base() + byte, 0x5a);
        }
        for (uint32_t byte = 0; byte < 0x4000; byte += 4)
          memory.write32(0x200000 + byte, 0);
        for (uint32_t reg = 20; reg < 33; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            cu->write_vgpr(vb + reg, lane, 0xaaaa0000u + reg * 32 + lane);
        for (size_t i = 0; i < words.size(); ++i)
          memory.write32(i * 4, words[i]);
        wave->set_exec(exec);
        wave->set_vcc(0x12345678);
        wave->write_scc(true);
        wave->pc = 0;
        size_t steps = 0;
        while (wave->pc < words.size() * 4 && steps++ < words.size() * 100)
          cu->step();
        cu->flush_all();
        ASSERT_EQ(wave->pc, words.size() * 4);
        const auto entry = decode_watchpoint_entry(memory.read64(0x201000));
        EXPECT_EQ(entry.valid, active && !atomic_barrier);
        if (entry.valid) {
          EXPECT_EQ(entry.kind, ShadowAccessKind::Write);
          EXPECT_EQ(entry.owner_id, 2u);
          EXPECT_EQ(entry.epoch, 3u);
          EXPECT_EQ(entry.byte_count, 1u << size_log2);
          EXPECT_LT(entry.start_byte, 16u << size_log2);
          EXPECT_EQ(entry.start_byte % (1u << size_log2), 0u);
        }
        EXPECT_EQ(memory.read32(0x200000 + offsetof(ReportHeader, unsupported_sync_count)) != 0,
                  active && atomic_barrier);
        EXPECT_EQ(cu->lds().read64(wave->lds_base() + 512) != 0, active && atomic_barrier);
        for (uint32_t byte = 0; byte < (16u << size_log2); ++byte)
          EXPECT_EQ(cu->lds().read8(wave->lds_base() + byte),
                    active ? (global_in_bounds ? static_cast<uint8_t>(byte + 1) : uint8_t{0})
                           : uint8_t{0x5a});
        EXPECT_EQ(wave->exec(), exec);
        EXPECT_EQ(wave->vcc(), 0x12345678u);
        EXPECT_TRUE(wave->read_scc());
        for (uint32_t reg = 20; reg < 33; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xaaaa0000u + reg * 32 + lane);
        EXPECT_EQ(cu->read_sgpr(sb), active ? 1u : 0u);
        EXPECT_EQ(cu->read_sgpr(sb + 12), (size_log2 << 16) | (atomic_barrier ? 1u << 18 : 0u));
        EXPECT_EQ(cu->read_sgpr(sb + 80), 2u);
        EXPECT_EQ(cu->read_sgpr(sb + 81), 3u);
        wave->halt();
      }
    }
  }
  plan.tensor_full_wave_resources = false;
  EXPECT_FALSE(
      detail::build_direct_watchpoint_words(bytes, candidate, 0, plan, nullptr, arch, errors));
  plan.tensor_full_wave_resources = true;
  plan.workgroup_sources.x = WorkgroupSource::vector(1);
  EXPECT_FALSE(
      detail::build_direct_watchpoint_words(bytes, candidate, 0, plan, nullptr, arch, errors));
}

TEST(ConSanTensor, FullWaveScratchSpillPreservesInactiveAndEmptyExecLanes) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  SpillManager manager(0, 256);
  const auto ordinary = build_vgpr_spill_sequence(manager, 20, 4, arch);
  ASSERT_TRUE(ordinary);
  const auto full_wave = detail::tensor_full_wave_spill(*ordinary, 96, arch);
  ASSERT_TRUE(full_wave);
  EXPECT_EQ(full_wave->slot_offsets, ordinary->slot_offsets);
  EXPECT_EQ(full_wave->total_private_bytes, ordinary->total_private_bytes);
  amdgpu::GpuMemory memory("tensor_full_wave_spill_mem");
  amdgpu::L2Cache l2("tensor_full_wave_spill_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 64;
  auto cu = amdgpu::ComputeUnitCore::create("tensor_full_wave_spill", config, &memory, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
  ASSERT_NE(wave, nullptr);
  wave->set_scratch_base(0x100000);
  wave->set_scratch_lane_size(256);
  const auto vb = wave->vgpr_alloc().base;
  const auto execute = [&](const std::vector<uint32_t> &words, uint64_t pc) {
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(pc + i * 4, words[i]);
    wave->pc = pc;
    size_t steps = 0;
    while (wave->pc < pc + words.size() * 4 && steps++ < words.size() * 100)
      cu->step();
    cu->flush_all();
    EXPECT_EQ(wave->pc, pc + words.size() * 4);
  };
  for (const uint32_t exec : {0xffffffffu, 0x80000001u, 0u}) {
    for (uint32_t reg = 20; reg < 24; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0xa5a50000u + 64u * reg + lane);
    wave->set_exec(exec);
    wave->set_vcc(0x12345678u);
    wave->write_scc(true);
    execute(full_wave->save_words, 0);
    EXPECT_EQ(wave->exec(), exec);
    // Model a wave-wide tensor probe clobbering its entire scratch allocation.
    for (uint32_t reg = 20; reg < 24; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0xdeadbeefu);
    execute(full_wave->restore_words, 4096);
    for (uint32_t reg = 20; reg < 24; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xa5a50000u + 64u * reg + lane);
    EXPECT_EQ(wave->exec(), exec);
    EXPECT_EQ(wave->vcc(), 0x12345678u);
    EXPECT_TRUE(wave->read_scc());
  }
  EXPECT_FALSE(detail::tensor_full_wave_spill(*ordinary, 105, arch));
  auto dynamic = *ordinary;
  dynamic.uses_dynamic_stack_frame = true;
  EXPECT_FALSE(detail::tensor_full_wave_spill(dynamic, 96, arch));
  wave->halt();
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
