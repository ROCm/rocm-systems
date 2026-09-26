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

TEST(ConSanTensor, SuperColliderPlansFullWaveSpillAndDescriptorSafeScalarState) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (const auto [alias, bank] : {std::pair{false, 0u}, std::pair{true, 0u},
                                   std::pair{false, 0x55u}, std::pair{true, 0x55u}}) {
    SCOPED_TRACE(alias);
    SCOPED_TRACE(bank);
    const auto tensor = cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                                            {.vaddr4 = 124,
                                             .vaddr0 = 0,
                                             .vaddr1 = static_cast<uint8_t>(alias ? 1 : 12),
                                             .vaddr2 = 124,
                                             .vaddr3 = 124});
    std::vector<uint32_t> guest;
    if (bank != 0u) {
      const auto select = instrumentation::build_s_set_vgpr_msb_transition(0u, bank, arch);
      ASSERT_TRUE(select);
      guest.push_back(*select);
    }
    guest.insert(guest.end(), tensor.begin(), tensor.end());
    guest.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
    TestOptions options = test_options();
    options.mode = Mode::SuperCollider;
    options.probe_lds_check_trap = true;
    options.supercollider_report_buffer_address = 0x200000;
    const auto result =
        test_lower_consan(make_gfx1250_code_object(guest, "tensor_compare"), options);
    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.patches.size(), 1u);
    const auto &patch = result.patches.front();
    EXPECT_EQ(patch.kind, PatchKind::LdsStoreCheckTrap);
    EXPECT_EQ(patch.spilled_vgpr_count, detail::kTensorLoadCompareScratchVgprs);
    EXPECT_GE(patch.required_private_segment_size, detail::kTensorLoadCompareScratchVgprs * 4u);
    EXPECT_GE(patch.required_sgpr_count, alias ? 12u : 4u);
    ASSERT_TRUE(patch.relocated_guest_instruction_offset.has_value());
    AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    const auto *text = patched.text_sections().front();
    const auto body = patched_words_at_file_offset(
        result, text->sectionOffset() + patch.trampoline_offset, patch.trampoline_size);
    ASSERT_FALSE(body.empty());
    if (bank != 0u) {
      const auto low = instrumentation::build_s_set_vgpr_msb_transition(bank, 0u, arch);
      const auto restore = instrumentation::build_s_set_vgpr_msb_transition(0u, bank, arch);
      ASSERT_TRUE(low);
      ASSERT_TRUE(restore);
      EXPECT_EQ(body.front(), *low);
      EXPECT_EQ(body.back(), *restore);
    }
    const auto guest_index = *patch.relocated_guest_instruction_offset / sizeof(uint32_t);
    ASSERT_LE(guest_index + tensor.size(), body.size());
    EXPECT_EQ(body[guest_index], tensor[0]);
    EXPECT_EQ(body[guest_index + 1u], tensor[1]);
    // Aliases rewrite only the descriptor-copy operand of the original DMA.
    EXPECT_EQ(body[guest_index + 2u] & ~0xff00u, tensor[2] & ~0xff00u);
  }
}

TEST(ConSanTensor, SuperColliderRejectsUnsafeSpillOrMisalignedExplicitScratch) {
  const auto tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 0, .vaddr1 = 12, .vaddr2 = 124, .vaddr3 = 124});
  std::vector<uint32_t> guest(tensor.begin(), tensor.end());
  guest.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  for (const bool dynamic_stack : {false, true}) {
    SCOPED_TRACE(dynamic_stack);
    TestOptions options = test_options();
    options.mode = Mode::SuperCollider;
    options.probe_lds_check_trap = true;
    if (!dynamic_stack)
      options.scratch_vgpr = 1u;
    const auto result = test_lower_consan(
        make_gfx1250_code_object(guest, "tensor_unavailable_resources", 15u, true, dynamic_stack),
        options);
    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
    EXPECT_FALSE(result.modified());
    EXPECT_EQ(access_lowering_count(result, LoweringOutcomeKind::ResourceRejected), 1u);
  }
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

TEST(ConSanTensor, TensorOwnerBarrierAdvancesEpochWithEmptyExecAndPreservesAllLanes) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  std::vector<uint32_t> guest{build_s_nop(0, arch), 0xd8340000u, 0x00000100u};
  const auto tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 0, .vaddr1 = 12, .vaddr2 = 124, .vaddr3 = 124});
  guest.insert(guest.end(), tensor.begin(), tensor.end());
  guest.push_back(0xbe804ec1u); // s_barrier_signal -1
  guest.push_back(0xbf94ffffu); // s_barrier_wait -1
  guest.resize(512, build_s_nop(0, arch));
  guest.push_back(build_s_endpgm(arch));
  for (int identity_mode : {0, 1, 2, 3}) {
    const bool private_identity = identity_mode != 0;
    const bool lane_scalar_spill = identity_mode == 2;
    const bool scalar_spill = identity_mode >= 2;
    SCOPED_TRACE(identity_mode);
    TestOptions options = test_options();
    options.exec_save_sgpr = 88;
    if (private_identity) {
      options.test_force_private_epoch = true;
    } else {
      options.persistent_sgprs.set_owner_epoch(80, 81);
      options.persistent_sgprs.exact_workgroup = PersistentWorkgroupRegisters{82, 83, 84};
    }
    options.test_force_vgpr_spill = !lane_scalar_spill;
    if (lane_scalar_spill)
      options.scratch_vgpr = 20;
    if (scalar_spill) {
      options.automatic_scalar_spill_layout = ScalarSpillLayout::Compact;
      options.scalar_spill_setup =
          ScalarSpillSetup{.temporaries = ScalarSpillTemporaries{96u, 98u}};
    }
    options.track_barriers = true;
    options.track_atomics = false;
    options.report_buffer_address = 0x200000;
    options.report_buffer_size = direct_report_bytes(8);
    const auto result =
        test_lower_consan(make_gfx1250_code_object(guest, "tensor_barrier"), options);
    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.warnings);
    const auto barrier =
        std::ranges::find(result.patches, PatchKind::TrampolineSyncMetadata, &PatchInfo::kind);
    ASSERT_NE(barrier, result.patches.end()) << testing::PrintToString(result.warnings);
    EXPECT_EQ(test_persistent_sgpr_state(result).complete(), !private_identity);
    ASSERT_EQ(barrier->private_state_layout.has_value(), private_identity);
    ASSERT_TRUE(barrier->scratch_vgpr);
    EXPECT_EQ(barrier->spilled_vgpr_count, lane_scalar_spill ? 0u : (scalar_spill ? 11u : 10u));
    const auto owners = detail::tensor_execution_owner_kernels(result.program_inventory);
    ASSERT_EQ(owners.size(), 1u);
    const auto cave = emitted_patch_words(result, *barrier);
    ASSERT_FALSE(cave.empty());
    const uint64_t begin = barrier->trampoline_offset;
    const uint64_t end = begin + cave.size() * 4;
    for (uint32_t exec : {0xffffffffu, 0x80000001u, 0u}) {
      SCOPED_TRACE(exec);
      amdgpu::GpuMemory memory("tensor_barrier_mem");
      amdgpu::L2Cache l2("tensor_barrier_l2");
      l2.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 106;
      config.vgprs_per_wf = 64;
      config.lds_size_kb = 4;
      auto cu = amdgpu::ComputeUnitCore::create("tensor_barrier", config, &memory, &l2);
      ASSERT_NE(cu, nullptr);
      auto *wave = cu->dispatch_wf(0, begin, 106, 64, 32);
      ASSERT_NE(wave, nullptr);
      wave->set_scratch_base(0x400000);
      wave->set_scratch_lane_size(256);
      const auto sb = wave->sgpr_alloc().base;
      const auto vb = wave->vgpr_alloc().base;
      for (uint16_t reg = 0; reg < 106; ++reg)
        cu->write_sgpr(sb + reg, 0);
      for (uint16_t reg = 88; reg < 96; ++reg)
        cu->write_sgpr(sb + reg, 0xcafe0000u + reg);
      const auto &state = test_persistent_sgpr_state(result);
      if (!private_identity) {
        cu->write_sgpr(sb + *state.owner(), 0);
        cu->write_sgpr(sb + *state.epoch(), 7);
      }
      for (uint16_t reg = 0; reg < 64; ++reg)
        for (uint32_t lane = 0; lane < 32; ++lane)
          cu->write_vgpr(vb + reg, lane, 0xa5a50000u + reg * 32 + lane);
      for (uint64_t offset = 0; offset < options.report_buffer_size; offset += 4)
        memory.write32(0x200000 + offset, 0);
      for (size_t i = 0; i < cave.size(); ++i)
        memory.write32(begin + i * 4, cave[i]);
      const auto execute_private = [&](const std::vector<uint32_t> &code) {
        constexpr uint64_t pc = 0x80000;
        for (size_t i = 0; i < code.size(); ++i)
          memory.write32(pc + i * 4, code[i]);
        wave->pc = pc;
        wave->set_exec(0xffffffffu);
        size_t steps = 0;
        while (wave->pc < pc + code.size() * 4 && steps++ < code.size() * 100)
          cu->step();
        cu->flush_all();
        EXPECT_EQ(wave->pc, pc + code.size() * 4);
      };
      if (private_identity) {
        const auto &layout = *barrier->private_state_layout;
        for (uint32_t lane = 0; lane < 32; ++lane) {
          cu->write_vgpr(vb + 60, lane, 0);
          cu->write_vgpr(vb + 61, lane, 7);
        }
        std::vector<uint32_t> seed;
        for (uint32_t offset = 0; offset < layout.persistent_state_end; offset += 4) {
          const auto store = instrumentation::build_private_store_b32(
              offset == layout.epoch_offset ? 61 : 60, offset, arch);
          ASSERT_TRUE(store);
          seed.insert(seed.end(), store->begin(), store->end());
        }
        seed.push_back(*instrumentation::build_s_wait_private_store0(arch));
        execute_private(seed);
      }
      wave->pc = begin;
      wave->set_exec(exec);
      wave->set_vcc(0x12345678);
      wave->write_scc(true);
      size_t steps = 0;
      while (wave->pc >= begin && wave->pc < end && steps++ < cave.size() * 100)
        cu->step();
      cu->flush_all();
      EXPECT_TRUE(wave->pc < begin || wave->pc >= end);
      if (!private_identity) {
        EXPECT_EQ(cu->read_sgpr(sb + *state.epoch()), 8u);
      }
      EXPECT_EQ(wave->exec(), exec);
      EXPECT_EQ(wave->vcc(), 0x12345678u);
      EXPECT_TRUE(wave->read_scc());
      if (scalar_spill) {
        for (uint16_t reg = 88; reg < 96; ++reg)
          EXPECT_EQ(cu->read_sgpr(sb + reg), 0xcafe0000u + reg);
      }
      if (!lane_scalar_spill) {
        for (uint16_t reg = *barrier->scratch_vgpr;
             reg < *barrier->scratch_vgpr + barrier->spilled_vgpr_count; ++reg)
          for (uint32_t lane = 0; lane < 32; ++lane)
            EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xa5a50000u + reg * 32 + lane);
      }
      if (private_identity) {
        auto read = instrumentation::build_private_load_b32(
            60, barrier->private_state_layout->epoch_offset, arch);
        ASSERT_TRUE(read);
        read->push_back(*instrumentation::build_s_wait_private_load0(arch));
        execute_private(*read);
        for (uint32_t lane = 0; lane < 32; ++lane)
          EXPECT_EQ(cu->read_vgpr(vb + 60, lane), 8u);
      }
      wave->halt();
    }
  }
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

TEST(ConSanTensor, PrivateIdentityReplicationPreservesInactiveRegistersAndSpecialState) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  const PrivateStateLayout layout{.epoch_offset = 0u,
                                  .owner_offset = 4u,
                                  .dispatch_id_offset = 12u,
                                  .exact_workgroup_offsets =
                                      PersistentWorkgroupPrivateOffsets{20u, 24u, 28u, 32u},
                                  .persistent_state_end = 36u,
                                  .ephemeral_base = 48u};
  SpillManager manager(64, 256);
  const auto spill = build_vgpr_spill_sequence(manager, 20, 2, arch);
  ASSERT_TRUE(spill);
  std::vector<uint32_t> words;
  ASSERT_TRUE(detail::append_full_wave_private_identity(words, layout, 20, 96, *spill, arch));
  amdgpu::GpuMemory memory("tensor_identity_mem");
  amdgpu::L2Cache l2("tensor_identity_l2");
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 64;
  auto cu = amdgpu::ComputeUnitCore::create("tensor_identity", config, &memory, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
  ASSERT_NE(wave, nullptr);
  wave->set_scratch_base(0x100000);
  wave->set_scratch_lane_size(256);
  const auto vb = wave->vgpr_alloc().base;
  const auto execute = [&](const std::vector<uint32_t> &code, uint64_t pc) {
    for (size_t i = 0; i < code.size(); ++i)
      memory.write32(pc + i * 4, code[i]);
    wave->pc = pc;
    size_t steps = 0;
    while (wave->pc < pc + code.size() * 4 && steps++ < code.size() * 100)
      cu->step();
    cu->flush_all();
    EXPECT_EQ(wave->pc, pc + code.size() * 4);
  };
  const std::array<uint32_t, 8> offsets{0, 4, 12, 16, 20, 24, 28, 32};
  std::vector<uint32_t> seed, read;
  for (size_t i = 0; i < offsets.size(); ++i) {
    const auto store = instrumentation::build_private_store_b32(30 + i, offsets[i], arch);
    const auto load = instrumentation::build_private_load_b32(30 + i, offsets[i], arch);
    ASSERT_TRUE(store);
    ASSERT_TRUE(load);
    seed.insert(seed.end(), store->begin(), store->end());
    read.insert(read.end(), load->begin(), load->end());
  }
  seed.push_back(*instrumentation::build_s_wait_private_store0(arch));
  read.push_back(*instrumentation::build_s_wait_private_load0(arch));
  for (uint32_t exec : {0xffffffffu, 1u, 0x80000001u, 0u}) {
    SCOPED_TRACE(exec);
    for (uint32_t reg = 20; reg < 38; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0xa5000000u + reg * 64 + lane);
    wave->set_exec(0xffffffffu);
    execute(seed, 0);
    wave->set_exec(exec);
    wave->set_vcc(0x12345678);
    wave->write_scc(true);
    execute(words, 4096);
    EXPECT_EQ(wave->exec(), exec);
    EXPECT_EQ(wave->vcc(), 0x12345678u);
    EXPECT_TRUE(wave->read_scc());
    for (uint32_t reg = 20; reg < 22; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xa5000000u + reg * 64 + lane);
    wave->set_exec(0xffffffffu);
    execute(read, 8192);
    for (uint32_t reg = 30; reg < 38; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xa5000000u + reg * 64);
  }
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

TEST(ConSanTensor, CoordinateDivisionMatchesIntegerOracleAndPreservesGuestState) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (unsigned width : {1u, 2u}) {
    SCOPED_TRACE(width);
    amdgpu::GpuMemory memory("tensor_division_mem");
    amdgpu::L2Cache l2("tensor_division_l2");
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 64;
    config.lds_size_kb = 4;
    auto cu = amdgpu::ComputeUnitCore::create("tensor_division", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
    ASSERT_NE(wave, nullptr);
    const auto vb = wave->vgpr_alloc().base;
    const auto sb = wave->sgpr_alloc().base;
    for (uint32_t reg = 0; reg < 106; ++reg)
      cu->write_sgpr(sb + reg, 0xabcd0000u + reg);
    std::vector<uint32_t> words;
    if (width == 1)
      ASSERT_TRUE(detail::append_tensor_divmod_u32(words, 10, 12, 20, 22, 30, arch));
    else
      ASSERT_TRUE(detail::append_tensor_divmod_u64(words, 10, 12, 20, 22, 30, arch));
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(i * 4, words[i]);
    // Tensor halfword and 48-bit stride limits, unsigned boundaries, zero,
    // and reproducible random pairs. Host integer arithmetic is the oracle.
    constexpr std::array<uint64_t, 18> boundary{0,
                                                1,
                                                2,
                                                3,
                                                7,
                                                65534,
                                                65535,
                                                65536,
                                                0x7fffffffu,
                                                0x80000000u,
                                                0xfffffffeu,
                                                0xffffffffu,
                                                0x100000000ull,
                                                0xffffffffffffull,
                                                0x7fffffffffffffffull,
                                                0x8000000000000000ull,
                                                UINT64_MAX - 1,
                                                UINT64_MAX};
    uint64_t random = 0x6a09e667f3bcc909ull;
    const auto next = [&] {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      return random;
    };
    for (uint32_t exec : {0xffffffffu, 0x80010005u, 0u}) {
      for (size_t batch = 0; batch < 40; ++batch) {
        std::array<uint64_t, 32> dividends{}, divisors{};
        for (uint32_t lane = 0; lane < 32; ++lane) {
          const size_t index = batch * 32 + lane;
          dividends[lane] = index < boundary.size() * boundary.size()
                                ? boundary[index / boundary.size()]
                                : next();
          divisors[lane] = index < boundary.size() * boundary.size()
                               ? boundary[index % boundary.size()]
                               : next();
          if (width == 1) {
            dividends[lane] &= UINT32_MAX;
            divisors[lane] &= UINT32_MAX;
          }
          for (uint32_t reg = 0; reg < 64; ++reg)
            cu->write_vgpr(vb + reg, lane, 0xabc00000u + reg * 64u + lane);
          for (unsigned word = 0; word < width; ++word) {
            cu->write_vgpr(vb + 10 + word, lane, dividends[lane] >> (word * 32));
            cu->write_vgpr(vb + 12 + word, lane, divisors[lane] >> (word * 32));
          }
        }
        wave->pc = 0;
        wave->set_exec(exec);
        wave->write_scc(batch % 2 != 0);
        size_t steps = 0;
        while (wave->pc < words.size() * 4) {
          ASSERT_LT(steps++, words.size() * 100);
          cu->step();
        }
        cu->flush_all();
        for (uint32_t lane = 0; lane < 32; ++lane) {
          const bool active = (exec >> lane) & 1u;
          const uint64_t numerator = dividends[lane], denominator = divisors[lane];
          const uint64_t quotient = denominator ? numerator / denominator : UINT64_MAX;
          const uint64_t remainder = denominator ? numerator % denominator : numerator;
          for (uint32_t reg = 0; reg < 64; ++reg) {
            if (active && reg >= 30 && reg < 30 + width + 2)
              continue;
            uint32_t expected = 0xabc00000u + reg * 64u + lane;
            if (reg >= 10 && reg < 10 + width)
              expected = numerator >> ((reg - 10) * 32);
            if (reg >= 12 && reg < 12 + width)
              expected = denominator >> ((reg - 12) * 32);
            if (active && reg >= 20 && reg < 20 + width)
              expected = quotient >> ((reg - 20) * 32);
            if (active && reg >= 22 && reg < 22 + width)
              expected = remainder >> ((reg - 22) * 32);
            EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected)
                << "batch=" << batch << " lane=" << lane << " register=" << reg
                << " numerator=" << numerator << " denominator=" << denominator;
          }
        }
        EXPECT_EQ(wave->exec(), exec);
        EXPECT_EQ(wave->read_scc(), batch % 2 != 0);
        for (uint32_t reg = 0; reg < 106; ++reg)
          EXPECT_EQ(cu->read_sgpr(sb + reg), 0xabcd0000u + reg);
      }
    }
    wave->halt();
  }
}

TEST(ConSanTensor, IterationOriginsMatchStorageOrderedDescriptorInverse) {
  const auto make_case = [](std::array<uint32_t, 3> dims, uint64_t stride1, uint64_t stride2,
                            bool rank_three) {
    auto result = tile({2, 2, rank_three ? 2u : 0u, 0, 0});
    result.d1[0] = 1u << 19;
    const auto put = [](auto &words, unsigned offset, unsigned width, uint64_t value) {
      for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned pos = offset + bit;
        words[pos / 32] =
            (words[pos / 32] & ~(1u << (pos % 32))) | (((value >> bit) & 1u) << (pos % 32));
      }
    };
    put(result.d1, 48, 32, dims[0]);
    put(result.d1, 80, 32, dims[1]);
    put(result.d1, 160, 48, stride1);
    put(result.d1, 208, 48, stride2);
    result.d2[0] = dims[2];
    result.d2[1] = 16;
    result.d2[2] = 1;
    result.d2[3] = 1u << 16;
    return result;
  };
  const std::vector<TensorCase> cases{make_case({4, 3, 2}, 4, 32, true),
                                      make_case({4, 3, 2}, 32, 4, true),
                                      make_case({4, 1, 2}, 0xffffffffffffull, 4, true),
                                      make_case({4, 3, 1}, 4, 0xffffffffffffull, true),
                                      make_case({1, 3, 2}, 8, 64, true),
                                      make_case({1, 1, 1}, 9, 3, true),
                                      make_case({4, 3, 17}, 1ull << 40, 1, false),
                                      make_case({4, 3, 2}, 1ull << 40, 1ull << 44, true)};
  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    SCOPED_TRACE(case_index);
    const auto &c = cases[case_index];
    const auto desc = tdm::parse_descriptor(c.d0, c.d1, c.d2, c.d3);
    const tdm::TensorDmaLayout layout(desc);
    ASSERT_FALSE(layout.empty());
    ASSERT_TRUE(layout.validate_iteration_inverse().succeeded());
    amdgpu::GpuMemory memory("tensor_origin_mem");
    amdgpu::L2Cache l2("tensor_origin_l2");
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 64;
    config.lds_size_kb = 4;
    auto cu = amdgpu::ComputeUnitCore::create("tensor_origin", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
    ASSERT_NE(wave, nullptr);
    const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
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
    site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 20, 40, 50};
    std::vector<uint32_t> words;
    ASSERT_TRUE(detail::append_tensor_iteration_origin(words, site, 10, 20, 30, config.arch));
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(i * 4, words[i]);
    for (uint32_t exec : {0xffffffffu, 0x80010005u, 0u}) {
      std::array<uint64_t, 32> offsets{};
      for (uint32_t lane = 0; lane < 32; ++lane) {
        offsets[lane] = lane < 2 ? lane : lane == 2 ? UINT64_MAX : lane * 0x9e3779b97f4a7c15ull;
        for (uint32_t reg = 0; reg < 64; ++reg)
          cu->write_vgpr(vb + reg, lane, 0xabc00000u + reg * 64u + lane);
        cu->write_vgpr(vb + 10, lane, offsets[lane]);
        cu->write_vgpr(vb + 11, lane, offsets[lane] >> 32);
      }
      wave->pc = 0;
      wave->set_exec(exec);
      wave->write_scc(true);
      size_t steps = 0;
      while (wave->pc < words.size() * 4) {
        ASSERT_LT(steps++, words.size() * 100);
        cu->step();
      }
      cu->flush_all();
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const bool active = (exec >> lane) & 1u;
        const auto origin = layout.origin_from_linear_offset(offsets[lane]);
        for (uint32_t reg = 0; reg < 64; ++reg) {
          if (active && reg >= 30 && reg < 48)
            continue;
          uint32_t expected = 0xabc00000u + reg * 64u + lane;
          if (reg == 10)
            expected = offsets[lane];
          if (reg == 11)
            expected = offsets[lane] >> 32;
          if (active && reg >= 20 && reg < 26)
            expected = origin[(reg - 20) / 2] >> (((reg - 20) % 2) * 32);
          EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected)
              << "lane=" << lane << " register=" << reg << " offset=" << offsets[lane];
        }
      }
      EXPECT_EQ(wave->exec(), exec);
      EXPECT_TRUE(wave->read_scc());
      for (uint32_t reg = 0; reg < 106; ++reg)
        EXPECT_EQ(cu->read_sgpr(sb + reg), scalar_before[reg]);
    }
    wave->halt();
  }
}

TEST(ConSanTensor, IterationOriginsRejectAliasesAndInvalidDescriptorsWithoutEmission) {
  ProgramSite site;
  site.origin = AccessOrigin::TensorLds;
  site.kind = LdsAccessKind::Write;
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 20, 40, 50};
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  for (const std::array<uint16_t, 3> regs : {std::array<uint16_t, 3>{10, 11, 30},
                                             {10, 20, 25},
                                             {29, 20, 30},
                                             {255, 20, 30},
                                             {10, 251, 30},
                                             {10, 20, 239}}) {
    EXPECT_FALSE(detail::append_tensor_iteration_origin(words, site, regs[0], regs[1], regs[2],
                                                        ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(words, prefix);
  }
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 99, 40, 50};
  EXPECT_FALSE(
      detail::append_tensor_iteration_origin(words, site, 10, 20, 30, ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(words, prefix);
}

TEST(ConSanTensor, GlobalSourcesMatchFinalDenseGatherAndRepeatedDmaWrites) {
  const auto put = [](auto &words, unsigned offset, unsigned width, uint64_t value) {
    for (unsigned bit = 0; bit < width; ++bit) {
      const unsigned pos = offset + bit;
      words[pos / 32] =
          (words[pos / 32] & ~(1u << (pos % 32))) | (((value >> bit) & 1u) << (pos % 32));
    }
  };
  const auto dense = [&](std::array<uint32_t, 5> tiles, std::array<uint32_t, 5> dims,
                         std::array<uint64_t, 4> strides) {
    auto c = tile(tiles);
    c.d0[2] = 0xabcde000u;
    c.d0[3] = 0xf1234567u; // Reserved upper bits must not become address bits.
    put(c.d1, 48, 32, dims[0]);
    put(c.d1, 80, 32, dims[1]);
    put(c.d2, 0, 32, dims[2]);
    put(c.d2, 32, 32, dims[3]);
    put(c.d3, 48, 32, dims[4]);
    put(c.d1, 160, 48, strides[0]);
    put(c.d1, 208, 48, strides[1]);
    put(c.d2, 64, 48, strides[2]);
    put(c.d3, 0, 48, strides[3]);
    return c;
  };
  std::vector<TensorCase> cases{
      dense({9, 0, 0, 0, 0}, {7, 0, 0, 0, 0}, {}),
      dense({3, 4, 0, 0, 0}, {2, 3, 0, 0, 0}, {7, 0, 0, 0}),
      dense({2, 2, 2, 2, 2}, {2, 2, 1, 2, 2}, {4, 16, 64, 256}),
      dense({3, 2, 2, 0, 0}, {3, 2, 2, 0, 0}, {1ull << 40, 1ull << 44, 0, 0}),
      dense({3, 2, 0, 0, 0}, {0, 2, 0, 0, 0}, {3, 0, 0, 0}),
      dense({3, 0, 2, 0, 0}, {3, 2, 2, 0, 0}, {4, 16, 0, 0}),
      dense({}, {}, {})};
  auto inactive = cases[0];
  inactive.d0[0] = 0;
  cases.push_back(inactive);
  auto null_optional = cases[1];
  null_optional.null2 = null_optional.null3 = true;
  cases.push_back(null_optional);
  for (bool wide_indices : {false, true}) {
    auto c = dense({3, 4, 0, 0, 0}, {2, 3, 0, 0, 0}, {7, 0, 0, 0});
    c.d0[0] |= (1u << 31) | (wide_indices ? 1u << 30 : 0u);
    c.d1[0] |= 1u << 19; // Ignored for gather.
    c.d2.fill(0xffffffffu);
    c.d3.fill(0xffffffffu);
    for (unsigned index = 0; index < 4; ++index)
      put(c.d2, index * (wide_indices ? 32 : 16), wide_indices ? 32 : 16,
          std::array<uint32_t, 4>{2, 5, 0, 1}[index]);
    cases.push_back(c);
    const unsigned index_count = wide_indices ? 8u : 16u;
    put(c.d1, 128, 16, index_count);
    for (unsigned index = 0; index < index_count; ++index) {
      const unsigned per_group = index_count / 2;
      auto &group = index < per_group ? c.d2 : c.d3;
      put(group, (index % per_group) * (wide_indices ? 32 : 16), wide_indices ? 32 : 16,
          index + 1 == index_count ? UINT32_MAX : index % 5);
    }
    cases.push_back(c);
  }
  for (uint32_t increment : {0u, 2u, 8u}) {
    auto c = dense({3, 2, 0, 0, 0}, {8, 4, 0, 0, 0}, {8, 0, 0, 0});
    c.d1[0] |= 1u << 19;
    c.d2[1] = increment;
    c.d2[2] = 6;
    c.d2[3] = 2u << 16;
    cases.push_back(c);
    if (increment == 0) {
      c.d2[2] = 20; // Final overlapping iteration is masked and must zero-fill.
      cases.push_back(c);
    }
  }
  auto reordered = dense({2, 2, 2, 0, 0}, {4, 3, 2, 0, 0}, {32, 4, 0, 0});
  reordered.d1[0] |= 1u << 19;
  reordered.d2[1] = 4;
  reordered.d2[2] = 13;
  reordered.d2[3] = 2u << 16;
  cases.push_back(reordered);

  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    for (unsigned size_log2 = 0; size_log2 < 4; ++size_log2) {
      SCOPED_TRACE(case_index);
      SCOPED_TRACE(size_log2);
      auto c = cases[case_index];
      put(c.d1, 16, 2, size_log2);
      const auto desc =
          tdm::parse_descriptor(c.d0, c.d1, c.null2 ? std::array<uint32_t, 4>{} : c.d2,
                                c.null3 ? std::array<uint32_t, 4>{} : c.d3);
      const tdm::TensorDmaLayout layout(desc);
      struct Element {
        uint64_t global;
        uint64_t lds;
        bool in_bounds;
      };
      std::vector<Element> reference;
      if (desc.active())
        tdm::for_each_tensor_element(desc, layout, [&](uint64_t global, uint64_t lds, bool bounds) {
          reference.push_back({desc.global_base + (global << size_log2), lds, bounds});
        });
      const uint32_t iterations = desc.iterate ? desc.iteration_count : 1;
      const uint32_t count = reference.size() / iterations;
      amdgpu::GpuMemory memory("tensor_source_mem");
      amdgpu::L2Cache l2("tensor_source_l2");
      l2.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = ROCJITSU_CODE_ARCH_CDNA5;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 106;
      config.vgprs_per_wf = 64;
      config.lds_size_kb = 4;
      auto cu = amdgpu::ComputeUnitCore::create("tensor_source", config, &memory, &l2);
      ASSERT_NE(cu, nullptr);
      auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
      ASSERT_NE(wave, nullptr);
      const auto vb = wave->vgpr_alloc().base, sb = wave->sgpr_alloc().base;
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
      site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{
          8, 20, uint16_t(c.null2 ? 124 : 40), uint16_t(c.null3 ? 124 : 50)};
      std::vector<uint32_t> words;
      ASSERT_TRUE(
          detail::append_select_tensor_load_element(words, site, 10, 11, 12, 13, 20, config.arch));
      ASSERT_TRUE(detail::append_materialize_tensor_load_source(words, site, 12, 13, 14, 16, 32,
                                                                config.arch));
      for (size_t i = 0; i < words.size(); ++i)
        memory.write32(i * 4, words[i]);
      for (uint32_t exec : {0xffffffffu, 0x80010005u, 0u}) {
        std::array<uint32_t, 32> hashes{}, iteration_hashes{};
        for (uint32_t lane = 0; lane < 32; ++lane) {
          hashes[lane] = lane == 31 ? UINT32_MAX : lane * 0x9e3779b9u;
          iteration_hashes[lane] = lane % 2 ? UINT32_MAX : lane * 0x85ebca6bu;
          for (uint32_t reg = 0; reg < 64; ++reg)
            cu->write_vgpr(vb + reg, lane, 0xabc00000u + reg * 64u + lane);
          cu->write_vgpr(vb + 10, lane, hashes[lane]);
          cu->write_vgpr(vb + 11, lane, iteration_hashes[lane]);
        }
        wave->pc = 0;
        wave->set_exec(exec);
        wave->write_scc(case_index % 2 != 0);
        size_t steps = 0;
        while (wave->pc < words.size() * 4) {
          ASSERT_LT(steps++, words.size() * 100);
          cu->step();
        }
        cu->flush_all();
        for (uint32_t lane = 0; lane < 32; ++lane) {
          const bool active = (exec >> lane) & 1u;
          for (uint32_t reg = 0; reg < 64; ++reg) {
            if (active &&
                ((reg >= 12 && reg <= 16) || (reg >= 20 && reg < 25) || (reg >= 32 && reg < 62)))
              continue;
            uint32_t expected = 0xabc00000u + reg * 64u + lane;
            if (reg == 10)
              expected = hashes[lane];
            if (reg == 11)
              expected = iteration_hashes[lane];
            EXPECT_EQ(cu->read_vgpr(vb + reg, lane), expected);
          }
          if (!active)
            continue;
          EXPECT_EQ(cu->read_vgpr(vb + 13, lane), count);
          if (!count) {
            EXPECT_EQ(cu->read_vgpr(vb + 16, lane), 0u);
            continue;
          }
          const size_t selected = (uint64_t{hashes[lane]} * count) >> 32;
          const size_t iteration = (uint64_t{iteration_hashes[lane]} * iterations) >> 32;
          const uint64_t lds = reference[iteration * count + selected].lds;
          EXPECT_EQ(cu->read_vgpr(vb + 12, lane), lds);
          auto last = std::find_if(reference.rbegin(), reference.rend(),
                                   [&](const Element &element) { return element.lds == lds; });
          ASSERT_NE(last, reference.rend());
          EXPECT_EQ(cu->read_vgpr(vb + 16, lane), last->in_bounds ? 1u : 0u);
          if (last->in_bounds) {
            const uint64_t actual =
                cu->read_vgpr(vb + 14, lane) | (uint64_t{cu->read_vgpr(vb + 15, lane)} << 32);
            EXPECT_EQ(actual, last->global) << "lane=" << lane << " lds=" << lds;
          }
        }
        EXPECT_EQ(wave->exec(), exec);
        EXPECT_EQ(wave->read_scc(), case_index % 2 != 0);
        for (uint32_t reg = 0; reg < 106; ++reg)
          EXPECT_EQ(cu->read_sgpr(sb + reg), scalar_before[reg]);
      }
      wave->halt();
    }
  }
}

TEST(ConSanTensor, GlobalSourceRejectsAliasesWithoutEmission) {
  ProgramSite site;
  site.origin = AccessOrigin::TensorLds;
  site.kind = LdsAccessKind::Write;
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{8, 20, 40, 50};
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  for (const std::array<uint16_t, 5> regs : {std::array<uint16_t, 5>{12, 12, 14, 16, 32},
                                             {12, 13, 11, 16, 32},
                                             {12, 13, 14, 15, 32},
                                             {12, 13, 31, 16, 32},
                                             {12, 13, 14, 32, 32},
                                             {12, 13, 14, 16, 227},
                                             {12, 13, 255, 16, 32}}) {
    EXPECT_FALSE(detail::append_materialize_tensor_load_source(
        words, site, regs[0], regs[1], regs[2], regs[3], regs[4], ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(words, prefix);
  }
}

TEST(ConSanTensor, ValueComparisonPreservesStateAndDefersExactlyOneCompletionArrival) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  for (bool alias : {false, true}) {
    SCOPED_TRACE(alias);
    constexpr uint16_t scratch = 8;
    constexpr uint16_t state = 88;
    const uint16_t spill_exec = alias ? 100 : 96;
    const uint16_t d1 = alias ? 1 : 12;
    constexpr uint16_t report_scratch = scratch + detail::kTensorLoadCompareWorkspaceOffset;
    const auto tensor =
        cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage, {.vaddr4 = 124,
                                                            .vaddr0 = 0,
                                                            .vaddr1 = static_cast<uint8_t>(d1),
                                                            .vaddr2 = 124,
                                                            .vaddr3 = 124});
    ProgramSite site;
    site.origin = AccessOrigin::TensorLds;
    site.kind = LdsAccessKind::Write;
    site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{0, d1, 124, 124};
    SpillManager manager(0, 256);
    const auto spill =
        build_vgpr_spill_sequence(manager, scratch, detail::kTensorLoadCompareScratchVgprs, arch);
    ASSERT_TRUE(spill);
    const auto full_spill = detail::tensor_full_wave_spill(*spill, spill_exec, arch);
    ASSERT_TRUE(full_spill);
    std::vector<uint32_t> mismatch;
    const auto append = [&](const auto &optional) {
      if (!optional)
        return false;
      if constexpr (std::is_integral_v<typename std::decay_t<decltype(optional)>::value_type>)
        mismatch.push_back(*optional);
      else
        mismatch.insert(mismatch.end(), optional->begin(), optional->end());
      return true;
    };
    ASSERT_TRUE(append(instrumentation::build_v_mov_b32_literal(report_scratch, 0x200000, arch)));
    ASSERT_TRUE(append(instrumentation::build_v_mov_b32_literal(report_scratch + 1, 0, arch)));
    ASSERT_TRUE(append(instrumentation::build_v_mov_b32_literal(report_scratch + 2, 1, arch)));
    ASSERT_TRUE(
        append(instrumentation::build_flat_store_b32(report_scratch, report_scratch + 2, arch)));
    ASSERT_TRUE(append(instrumentation::build_s_wait_global_store0(arch)));
    auto words = full_spill->save_words;
    uint32_t guest_offset = UINT32_MAX;
    ASSERT_TRUE(detail::append_tensor_load_compare(words, site, tensor, scratch, state, {},
                                                   mismatch, guest_offset, arch));
    ASSERT_LE(guest_offset + tensor.size(), words.size());
    if (!alias)
      EXPECT_TRUE(std::equal(tensor.begin(), tensor.end(), words.begin() + guest_offset));
    else {
      EXPECT_EQ(words[guest_offset], tensor[0]);
      EXPECT_EQ(words[guest_offset + 1], tensor[1]);
      EXPECT_EQ(words[guest_offset + 2], (tensor[2] & ~0xff00u) | ((state + 4u) << 8));
    }
    words.insert(words.end(), full_spill->restore_words.begin(), full_spill->restore_words.end());
    const uint64_t after_dma_wait_pc = (guest_offset + tensor.size() + 1) * 4;
    for (unsigned size_log2 = 0; size_log2 < 4; ++size_log2) {
      if (alias && size_log2 != 0)
        continue;
      for (bool active : {false, true}) {
        for (bool in_bounds : {false, true}) {
          for (bool atomic : {false, true}) {
            for (bool corrupt : {false, true}) {
              for (uint32_t exec : {0xffffffffu, 0x80000001u, 0u}) {
                SCOPED_TRACE(testing::Message() << "size=" << size_log2 << " active=" << active
                                                << " bounds=" << in_bounds << " atomic=" << atomic
                                                << " corrupt=" << corrupt << " exec=" << exec);
                amdgpu::GpuMemory memory("tensor_compare_mem");
                amdgpu::L2Cache l2("tensor_compare_l2");
                l2.set_backing_memory(&memory);
                amdgpu::ComputeUnitCore::Config config{};
                config.arch = arch;
                config.num_wf_slots = 1;
                config.sgprs_per_wf = 106;
                config.vgprs_per_wf = 64;
                config.lds_size_kb = alias ? 320 : 4;
                auto cu = amdgpu::ComputeUnitCore::create("tensor_compare", config, &memory, &l2);
                ASSERT_NE(cu, nullptr);
                auto *wave = cu->dispatch_wf(0, 0, 106, 64, 32);
                ASSERT_NE(wave, nullptr);
                wave->set_lds_base(cu->allocate_lds(config.lds_size_kb * 1024));
                ASSERT_NE(wave->lds_base(), UINT32_MAX);
                wave->set_scratch_base(0x400000);
                wave->set_scratch_lane_size(256);
                const auto sb = wave->sgpr_alloc().base, vb = wave->vgpr_alloc().base;
                for (uint32_t reg = 0; reg < 106; ++reg)
                  cu->write_sgpr(sb + reg, 0xbeef0000u + reg);
                cu->write_sgpr(sb, active ? 1 : 0);
                if (alias) {
                  // D1[0] is also D0's LDS base. Clearing its completion bit in
                  // place would redirect the DMA from byte262144 to byte0.
                  cu->write_sgpr(sb + 1, atomic ? 1u << 18 : 0);
                  cu->write_sgpr(sb + 2, (in_bounds ? 16u << 16 : 0) | (512u >> 3));
                  cu->write_sgpr(sb + 3, 0);
                  cu->write_sgpr(sb + 4, 16u << 16);
                  for (uint32_t reg = 5; reg < 9; ++reg)
                    cu->write_sgpr(sb + reg, 0);
                } else {
                  cu->write_sgpr(sb + 1, 0);
                  cu->write_sgpr(sb + 2, 0x100000);
                  cu->write_sgpr(sb + 3, 0x80000000u);
                  cu->write_sgpr(sb + 12, (size_log2 << 16) | (atomic ? 1u << 18 : 0));
                  cu->write_sgpr(sb + 13, (in_bounds ? 16u << 16 : 0) | (512u >> 3));
                  cu->write_sgpr(sb + 14, 0);
                  cu->write_sgpr(sb + 15, 16u << 16);
                  for (uint32_t reg = 16; reg < 20; ++reg)
                    cu->write_sgpr(sb + reg, 0);
                }
                const uint32_t source_base = cu->read_sgpr(sb + 2);
                const uint32_t payload_base = wave->lds_base() + cu->read_sgpr(sb + 1);
                std::array<uint32_t, 106> scalar_before{};
                for (uint32_t reg = 0; reg < 106; ++reg)
                  scalar_before[reg] = cu->read_sgpr(sb + reg);
                const uint64_t barrier_before =
                    0xabcd000000000000ull | amdgpu::lds_barrier_cell_init_state(corrupt ? 2 : 1);
                cu->lds().write64(wave->lds_base() + 512, barrier_before);
                for (uint32_t byte = 0; byte < (16u << size_log2); ++byte) {
                  if (!alias || in_bounds)
                    memory.write8(source_base + byte, static_cast<uint8_t>(byte + 1));
                  cu->lds().write8(payload_base + byte, 0x5a);
                }
                memory.write32(0x200000, 0);
                for (uint32_t reg = 0; reg < 64; ++reg)
                  for (uint32_t lane = 0; lane < 32; ++lane)
                    cu->write_vgpr(vb + reg, lane, 0xa5000000u + reg * 32 + lane);
                for (size_t i = 0; i < words.size(); ++i)
                  memory.write32(i * 4, words[i]);
                wave->set_exec(exec);
                wave->set_vcc(0x12345678);
                wave->write_scc(in_bounds);
                wave->pc = 0;
                bool observed_dma = false;
                size_t steps = 0;
                while (wave->pc < words.size() * 4) {
                  ASSERT_LT(steps++, words.size() * 100);
                  if (!observed_dma && wave->pc == after_dma_wait_pc) {
                    observed_dma = true;
                    EXPECT_EQ(cu->lds().read64(wave->lds_base() + 512), barrier_before);
                    for (uint32_t byte = 0; byte < (16u << size_log2); ++byte)
                      EXPECT_EQ(cu->lds().read8(payload_base + byte),
                                active ? (in_bounds ? static_cast<uint8_t>(byte + 1) : uint8_t{0})
                                       : uint8_t{0x5a});
                    if (active && corrupt)
                      cu->lds().write8(payload_base, 0xa5);
                  }
                  cu->step();
                }
                cu->flush_all();
                EXPECT_TRUE(observed_dma);
                EXPECT_EQ(memory.read32(0x200000), active && corrupt ? 1u : 0u);
                EXPECT_EQ(cu->lds().read64(wave->lds_base() + 512),
                          active && atomic ? amdgpu::lds_barrier_cell_update_arrive(barrier_before)
                                           : barrier_before);
                EXPECT_TRUE(wave->wait_counters().empty());
                EXPECT_EQ(wave->exec(), exec);
                EXPECT_EQ(wave->vcc(), 0x12345678u);
                EXPECT_EQ(wave->read_scc(), in_bounds);
                for (uint32_t reg = 0; reg < 64; ++reg)
                  for (uint32_t lane = 0; lane < 32; ++lane)
                    EXPECT_EQ(cu->read_vgpr(vb + reg, lane), 0xa5000000u + reg * 32 + lane);
                for (uint32_t reg = 0; reg < 106; ++reg) {
                  if ((reg >= state && reg < static_cast<uint32_t>(state) +
                                                 detail::tensor_load_compare_state_sgprs(site)) ||
                      (reg >= spill_exec && reg < spill_exec + 2u))
                    continue;
                  EXPECT_EQ(cu->read_sgpr(sb + reg), scalar_before[reg]) << reg;
                }
                wave->halt();
              }
            }
          }
        }
      }
    }
  }
}

TEST(ConSanTensor, ValueComparisonRejectsDescriptorScalarOverlapWithoutEmission) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  ProgramSite site;
  site.origin = AccessOrigin::TensorLds;
  site.kind = LdsAccessKind::Write;
  site.operands.tensor_descriptor_sgprs = std::array<uint16_t, 4>{0, 12, 124, 124};
  const auto tensor =
      cdna5::build_vimage(cdna5::kTensorLoadToLdsVimage,
                          {.vaddr4 = 124, .vaddr0 = 0, .vaddr1 = 12, .vaddr2 = 124, .vaddr3 = 124});
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  uint32_t guest_offset = 77;
  for (uint16_t state : {0, 2, 10, 12, 18, 89, 104}) {
    EXPECT_FALSE(detail::append_tensor_load_compare(words, site, tensor, 8, state, {}, {},
                                                    guest_offset, arch));
    EXPECT_EQ(words, prefix);
    EXPECT_EQ(guest_offset, 77u);
  }
  EXPECT_FALSE(
      detail::append_tensor_load_compare(words, site, tensor, 211, 88, {}, {}, guest_offset, arch));
  EXPECT_EQ(words, prefix);
}

TEST(ConSanTensor, CoordinateDivisionRejectsRegisterAliasesWithoutEmission) {
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  for (const std::array<uint16_t, 5> regs : {std::array<uint16_t, 5>{10, 11, 20, 20, 30},
                                             {10, 10, 20, 21, 30},
                                             {10, 11, 10, 21, 30},
                                             {10, 11, 20, 11, 30},
                                             {10, 11, 20, 21, 254},
                                             {30, 11, 20, 21, 30},
                                             {10, 11, 20, 32, 30},
                                             {256, 11, 20, 21, 30}}) {
    EXPECT_FALSE(detail::append_tensor_divmod_u32(words, regs[0], regs[1], regs[2], regs[3],
                                                  regs[4], ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(words, prefix);
  }
  EXPECT_FALSE(
      detail::append_tensor_divmod_u32(words, 10, 11, 20, 21, 30, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words, prefix);
}

TEST(ConSanTensor, CoordinateDivisionRejectsOverlappingPairsWithoutEmission) {
  const std::vector<uint32_t> prefix{0xabcdef01u};
  auto words = prefix;
  for (const std::array<uint16_t, 5> regs : {std::array<uint16_t, 5>{10, 11, 20, 22, 30},
                                             {10, 12, 11, 22, 30},
                                             {10, 12, 20, 21, 30},
                                             {10, 12, 20, 255, 30},
                                             {10, 12, 20, 22, 254},
                                             {29, 12, 20, 22, 30},
                                             {10, 12, 20, 32, 30}}) {
    EXPECT_FALSE(detail::append_tensor_divmod_u64(words, regs[0], regs[1], regs[2], regs[3],
                                                  regs[4], ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(words, prefix);
  }
  EXPECT_FALSE(
      detail::append_tensor_divmod_u64(words, 10, 12, 20, 22, 30, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(words, prefix);
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
