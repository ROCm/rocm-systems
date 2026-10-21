// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <gtest/gtest.h>
#include <memory>

namespace rocjitsu {
namespace {
class ScalarCacheDiscard : public testing::TestWithParam<rj_code_arch_t> {};

TEST_P(ScalarCacheDiscard, AddressedLinesAndWaitMetadataIgnoreExec) {
  amdgpu::GpuMemory memory{"discard_memory"};
  amdgpu::L2Cache l2{"discard_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = GetParam();
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 102;
  cfg.vgprs_per_wf = 4;
  auto cu = amdgpu::ComputeUnitCore::create("discard", cfg, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 102, 4);
  ASSERT_NE(wave, nullptr);
  auto decoder = Decoder::create(GetParam());
  auto &cache = cu->l1_scalar();
  const unsigned base = wave->sgpr_alloc().base;
  cu->write_sgpr(base, 0x4000);
  cu->write_sgpr(base + 1, 0);
  for (unsigned address_offset : {4u, 60u, 68u})
    for (unsigned lines : {1u, 2u})
      for (unsigned mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(testing::Message() << "lines=" << lines << " address_mode=" << mode);
        cache.invalidate_all();
        for (unsigned i = 0; i < 4; ++i) {
          uint32_t value = 100 + i;
          ASSERT_EQ(cache.store(0x4000 + i * 64, 1, &value), amdgpu::VmAccessOutcome::Complete);
          value = 200 + i;
          ASSERT_EQ(
              l2.write(0x4000 + i * 64, reinterpret_cast<const uint8_t *>(&value), sizeof(value)),
              amdgpu::VmAccessOutcome::Complete);
        }
        // X2 always discards the next line, even when eight bytes at the
        // unaligned address would fit entirely within the first line.
        const bool imm = mode != 1;
        const bool soe = mode == 2;
        cu->write_sgpr(base + 4, address_offset - (mode == 2 ? address_offset / 2 + 2 : 0));
        const unsigned offset = mode == 0 ? address_offset : mode == 2 ? address_offset / 2 + 2 : 4;
        const std::array<uint32_t, 2> words{0xc0000000u | ((39 + lines) << 18) |
                                                (uint32_t{imm} << 17) | (uint32_t{soe} << 14),
                                            offset | (4u << 25)};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        auto *info = inst->amdgpu_memory_issue_info();
        ASSERT_NE(info, nullptr);
        EXPECT_FALSE(info->exec_masked);
        ASSERT_EQ(info->counter_obligations().size(), 1);
        EXPECT_EQ(info->counter_obligations()[0].wait_counter_type(),
                  amdgpu::WaitCounterType::LGKMCNT);
        EXPECT_EQ(info->counter_obligations()[0].counter_increment(), lines);
        wave->set_exec(0);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
        for (unsigned i = 0; i < 4; ++i) {
          uint32_t value = 0;
          ASSERT_EQ(cache.load(0x4000 + i * 64, 1, &value), amdgpu::VmAccessOutcome::Complete);
          const unsigned first_line = address_offset / 64;
          EXPECT_EQ(value, (i >= first_line && i < first_line + lines ? 200 : 100) + i);
        }
      }
  wave->halt();
}

TEST(ScalarCacheDiscardCache, ProcessTagsAndAbsentLinesDoNotReadOrWriteMemory) {
  test::LegacyGpuMemoryFixture memory{"discard_memory"};
  amdgpu::L2Cache l2{"discard_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::L1ScalarCache cache{&l2};
  std::array<uint8_t, amdgpu::kLegacyPageSize> first{}, second{};
  amdgpu::LegacyPageTable pages1, pages2;
  util::DistributedSharedMutex mutex;
  pages1[4] = {first.data(), amdgpu::Mtype::RW};
  pages2[4] = {second.data(), amdgpu::Mtype::RW};
  memory.register_process(1, &pages1, &mutex);
  memory.register_process(2, &pages2, &mutex);
  l2.set_gpu_vm(&memory.gpu_vm());
  cache.set_gpu_vm(&memory.gpu_vm());
  uint32_t value = 123;
  ASSERT_EQ(cache.store(0x4000, 1, &value, 1), amdgpu::VmAccessOutcome::Complete);
  value = 456;
  ASSERT_EQ(cache.store(0x4000, 1, &value, 2), amdgpu::VmAccessOutcome::Complete);
  value = 789;
  ASSERT_EQ(l2.write(0x4000, reinterpret_cast<const uint8_t *>(&value), sizeof(value),
                     amdgpu::Mtype::RW, 1),
            amdgpu::VmAccessOutcome::Complete);
  cache.discard(0x403f, 1, 1);
  ASSERT_EQ(cache.load(0x4000, 1, &value, 1), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(value, 789);
  ASSERT_EQ(cache.load(0x4000, 1, &value, 2), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(value, 456);
  // A miss only changes cache residency, even for an address that has no backing storage.
  amdgpu::L1ScalarCache empty{&l2};
  empty.discard(0xffffffffffffffff, 2, 1);
  ASSERT_EQ(cache.load(0x4000, 1, &value, 2), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(value, 456);
}

INSTANTIATE_TEST_SUITE_P(Architectures, ScalarCacheDiscard,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4));
} // namespace
} // namespace rocjitsu
