// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace {

using namespace rocjitsu;

class ScalarCachePolicyTest : public ::testing::Test {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 102;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("scalar_cache", cfg, &memory, &l2);
    wave = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    decoder = Decoder::create(cfg.arch);
  }

  void TearDown() override { wave->halt(); }

  void issue(uint32_t op, bool glc, uint64_t address, uint32_t records = 0xffffffffu) {
    const auto base = wave->sgpr_alloc().base;
    cu->write_sgpr(base, static_cast<uint32_t>(address));
    cu->write_sgpr(base + 1, static_cast<uint32_t>(address >> 32));
    cu->write_sgpr(base + 2, records);
    cu->write_sgpr(base + 3, 0);
    // CDNA4 SMEM, immediate zero offset, s[0:1] base (s[0:3] for buffers),
    // and s32 onward as the load destination or store source.
    const std::array<uint32_t, 2> words{
        0xc0000000u | (op << 18) | (1u << 17) | (uint32_t{glc} << 16) | (32u << 6), 0};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    ASSERT_EQ(inst->data_as<amdgpu::ScalarMemState>()->mtype,
              glc ? amdgpu::Mtype::CC : amdgpu::Mtype::RW);
    ASSERT_EQ(pipeline.issue(inst.release(), *wave), amdgpu::VmAccessOutcome::Complete);
  }

  uint32_t result(uint32_t word) { return cu->read_sgpr(wave->sgpr_alloc().base + 32 + word); }

  amdgpu::GpuMemory memory{"scalar_memory"};
  amdgpu::L2Cache l2{"scalar_l2"};
  amdgpu::L1ScalarCache l1{&l2};
  amdgpu::ScalarMemPipeline pipeline{&l1};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wave = nullptr;
  std::unique_ptr<Decoder> decoder;
};

TEST_F(ScalarCachePolicyTest, GlcLoadsRefetchWarmScalarAndL2Lines) {
  // CDNA4 ISA 9.1.10.1: GLC loads miss both caches. Each external update
  // deliberately leaves their cached copies stale, as in a polling consumer.
  for (uint32_t op = 0; op <= 4; ++op) {
    SCOPED_TRACE(op);
    const uint32_t words = 1u << op;
    const uint64_t address = 0x103c + op * 0x100;
    for (uint32_t i = 0; i < words; ++i)
      memory.write32(address + 4 * i, 100 + i);
    ASSERT_NO_FATAL_FAILURE(issue(op, false, address));
    for (uint32_t i = 0; i < words; ++i)
      EXPECT_EQ(result(i), 100 + i);

    for (uint32_t version : {200u, 300u}) {
      for (uint32_t i = 0; i < words; ++i)
        memory.write32(address + 4 * i, version + i);
      if (version == 200) {
        ASSERT_NO_FATAL_FAILURE(issue(op, false, address));
        for (uint32_t i = 0; i < words; ++i)
          EXPECT_EQ(result(i), 100 + i);
      }
      ASSERT_NO_FATAL_FAILURE(issue(op, true, address));
      for (uint32_t i = 0; i < words; ++i)
        EXPECT_EQ(result(i), version + i);
    }
  }
}

TEST_F(ScalarCachePolicyTest, GlcBufferLoadsRefreshOnlyInBoundsWords) {
  constexpr uint64_t address = 0x203c;
  for (uint32_t i = 0; i < 4; ++i)
    memory.write32(address + 4 * i, 100 + i);
  ASSERT_NO_FATAL_FAILURE(issue(/*s_buffer_load_dwordx4=*/10, false, address, /*records=*/12));
  for (uint32_t i = 0; i < 4; ++i)
    memory.write32(address + 4 * i, 200 + i);
  ASSERT_NO_FATAL_FAILURE(issue(10, true, address, 12));
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(result(i), i < 3 ? 200 + i : 0);

  const auto transactions = l2.backing_read_transactions();
  ASSERT_NO_FATAL_FAILURE(issue(10, true, /*unmapped base=*/0, /*records=*/0));
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(result(i), 0u);
  EXPECT_EQ(l2.backing_read_transactions(), transactions);
}

TEST_F(ScalarCachePolicyTest, GlcStoresDoNotLeavePersistentScalarCopies) {
  // GLC stores reach L2 without retaining their values in the scalar cache.
  // A later update through L2 must be visible to a normal scalar load.
  for (uint32_t op = 0; op <= 2; ++op) {
    for (bool glc : {false, true}) {
      SCOPED_TRACE(testing::Message() << "op=" << op << " glc=" << glc);
      const uint32_t words = 1u << op;
      const uint64_t address = 0x303c + op * 0x200 + uint32_t{glc} * 0x100;
      for (uint32_t i = 0; i < words; ++i)
        memory.write32(address + 4 * i, 100 + i);
      ASSERT_NO_FATAL_FAILURE(issue(op, false, address));
      for (uint32_t i = 0; i < words; ++i)
        cu->write_sgpr(wave->sgpr_alloc().base + 32 + i, 200 + i);
      ASSERT_NO_FATAL_FAILURE(issue(/*s_store_dword[xN]=*/16 + op, glc, address));
      for (uint32_t i = 0; i < words; ++i) {
        EXPECT_EQ(memory.read32(address + 4 * i), 200 + i);
        const uint32_t next = 300 + i;
        ASSERT_EQ(l2.write(address + 4 * i, reinterpret_cast<const uint8_t *>(&next), sizeof(next),
                           amdgpu::Mtype::RW),
                  amdgpu::VmAccessOutcome::Complete);
      }
      ASSERT_NO_FATAL_FAILURE(issue(op, false, address));
      for (uint32_t i = 0; i < words; ++i)
        EXPECT_EQ(result(i), (glc ? 300 : 200) + i);
    }
  }
}

TEST(ScalarCacheMtypeTest, InstructionAndPageRestrictionsApplyAcrossPageBoundaries) {
  constexpr uint32_t vmid = 7;
  constexpr uint64_t address = 0x1ffe;
  for (auto instruction_mtype : {amdgpu::Mtype::RW, amdgpu::Mtype::CC}) {
    for (auto page_mtype : {amdgpu::Mtype::RW, amdgpu::Mtype::CC, amdgpu::Mtype::UC}) {
      for (bool byte_load : {false, true}) {
        SCOPED_TRACE(testing::Message() << "instruction=" << int(instruction_mtype)
                                        << " page=" << int(page_mtype) << " bytes=" << byte_load);
        test::LegacyGpuMemoryFixture memory("page_memory");
        amdgpu::L2Cache l2("page_l2");
        amdgpu::L1ScalarCache l1(&l2);
        l2.set_backing_memory(&memory);
        l2.set_gpu_vm(&memory.gpu_vm());
        l1.set_gpu_vm(&memory.gpu_vm());
        std::array<uint8_t, amdgpu::kLegacyPageSize> first{}, second{};
        first.fill(0x11);
        second.fill(0x11);
        amdgpu::LegacyPageTable pages;
        util::DistributedSharedMutex mutex;
        pages[1] = {first.data(), amdgpu::Mtype::RW};
        pages[2] = {second.data(), amdgpu::Mtype::RW};
        memory.register_process(vmid, &pages, &mutex);

        std::array<uint32_t, 2> words{};
        ASSERT_EQ(l1.load(address, words.size(), words.data(), vmid),
                  amdgpu::VmAccessOutcome::Complete);
        ASSERT_EQ(words[0], 0x11111111u);
        ASSERT_EQ(words[1], 0x11111111u);
        first.fill(0x22);
        second.fill(0x22);
        {
          std::unique_lock lock(mutex);
          pages[2].mtype = page_mtype;
        }
        if (byte_load) {
          ASSERT_EQ(l1.load_bytes(address, sizeof(words), reinterpret_cast<uint8_t *>(words.data()),
                                  vmid, instruction_mtype),
                    amdgpu::VmAccessOutcome::Complete);
        } else {
          ASSERT_EQ(l1.load(address, words.size(), words.data(), vmid, instruction_mtype),
                    amdgpu::VmAccessOutcome::Complete);
        }
        const auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
        for (size_t i = 0; i < sizeof(words); ++i) {
          const bool refetched =
              instruction_mtype == amdgpu::Mtype::CC || (i >= 2 && page_mtype != amdgpu::Mtype::RW);
          EXPECT_EQ(bytes[i], refetched ? 0x22 : 0x11) << "byte=" << i;
        }
        memory.unregister_process(vmid);
      }
    }
  }
}

TEST(ScalarCacheMtypeTest, NarrowPipelineLoadsForwardDecodedPolicy) {
  // Narrow loads use load_bytes rather than the DWORD path. RDNA4's decoded
  // device-scope policy must reach that same cache interface.
  for (uint32_t op : {8u, 9u, 10u, 11u}) {
    SCOPED_TRACE(op);
    amdgpu::GpuMemory memory("narrow_memory");
    amdgpu::L2Cache l2("narrow_l2");
    l2.set_backing_memory(&memory);
    amdgpu::L1ScalarCache l1(&l2);
    amdgpu::ScalarMemPipeline pipeline(&l1);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("narrow_scalar", cfg, &memory, &l2);
    auto *wave = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    auto decoder = Decoder::create(cfg.arch);
    const auto base = wave->sgpr_alloc().base;
    constexpr uint64_t address = 0x4000;
    cu->write_sgpr(base, address);
    cu->write_sgpr(base + 1, 0);
    memory.write32(address, 0);
    uint32_t warm = 0;
    ASSERT_EQ(l1.load(address, 1, &warm), amdgpu::VmAccessOutcome::Complete);
    memory.write32(address, 0xfedcba98);
    // s_load_{i8,u8,i16,u16} s8, s[0:1], 0, scope:DEVICE.
    const std::array<uint32_t, 2> words{0xf4000200u | (op << 13) | (2u << 21), 0xf8000000u};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    ASSERT_EQ(inst->data_as<amdgpu::ScalarMemState>()->mtype, amdgpu::Mtype::UC);
    ASSERT_EQ(pipeline.issue(inst.release(), *wave), amdgpu::VmAccessOutcome::Complete);
    const uint32_t expected = op == 8    ? 0xffffff98u
                              : op == 9  ? 0x98u
                              : op == 10 ? 0xffffba98u
                                         : 0xba98u;
    EXPECT_EQ(cu->read_sgpr(base + 8), expected);
    wave->halt();
  }
}

} // namespace
