// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu {
namespace {

class Cdna5Prefetch : public testing::Test {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
    cfg.num_wf_slots = 1;
    cfg.vgprs_per_wf = 8;
    cfg.sgprs_per_wf = 32;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "prefetch", cfg, &memory, &l2));
    wave = cu->dispatch_wf(0, 0, 32, 8);
    ASSERT_NE(wave, nullptr);
    ASSERT_EQ(wave->wf_size(), 32u);
    for (uint32_t offset = 0; offset < 4096; offset += 64)
      memory.write32(kBase + offset, kMarker + offset);
  }
  void TearDown() override {
    if (wave)
      wave->halt();
  }
  template <size_t N> void execute(const std::array<uint32_t, N> &words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->num_dst_operands(), 0);
    EXPECT_EQ(inst->amdgpu_memory_issue_info(), nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
    EXPECT_EQ(inst->data(), nullptr);
    EXPECT_EQ(wave->wait_counters().kmcnt, 0u);
    EXPECT_EQ(wave->wait_counters().vmcnt, 0u);
  }
  template <class Cache> bool resident(Cache &cache, uint64_t address) {
    std::array<uint8_t, 1> byte{};
    return cache.try_read_peer_cached(address, byte, 0, 0, 0);
  }
  bool instruction_resident(uint64_t address) {
    uint32_t word = 0;
    const bool hit = cu->instruction_cache().peek_word(address, 0, word);
    if (hit)
      EXPECT_EQ(word, kMarker + address - kBase);
    return hit;
  }
  static constexpr uint64_t kBase = 0x4000;
  static constexpr uint32_t kMarker = 0x12340000;
  amdgpu::GpuMemory memory{"prefetch_ram"};
  amdgpu::L2Cache l2{"prefetch_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wave = nullptr;
  decltype(Decoder::create(ROCJITSU_CODE_ARCH_CDNA5)) decoder =
      Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
};

TEST_F(Cdna5Prefetch, ScalarDataLengthAndDisabledModeIgnoreExec) {
  wave->set_exec(0);
  const auto s = wave->sgpr_alloc().base;
  cu->write_sgpr(s + 4, kBase + 129);
  cu->write_sgpr(s + 5, 0);
  cu->write_sgpr(s + 7, 32);
  const auto words = cdna5::build_smem(cdna5::kSPrefetchDataSmem,
                                       {.sbase = 2, .sdata = 1, .ioffset = 0xffffff, .soffset = 7});
  execute(words);
  EXPECT_FALSE(resident(l2, kBase));
  wave->set_mode_raw(1u << 24);
  execute(words);
  // Base and -1 are separately DWORD-aligned, so the request starts at kBase.
  for (uint32_t offset : {0u, 64u, 128u, 192u})
    EXPECT_TRUE(resident(cu->l1_scalar(), kBase + offset)) << offset;
  EXPECT_FALSE(resident(cu->l1_scalar(), kBase + 256));
  EXPECT_EQ(cu->read_sgpr(s + 7), 32u);
}

TEST_F(Cdna5Prefetch, ScalarBufferUsesLow48AndDropsNegativeOffset) {
  wave->set_mode_raw(1u << 24);
  const auto s = wave->sgpr_alloc().base;
  cu->write_sgpr(s + 4, kBase);
  cu->write_sgpr(s + 5, 0xfea50000);
  cu->write_sgpr(s + 6, 0);
  cu->write_sgpr(s + 7, 0);
  wave->set_m0(31);
  auto fields =
      cdna5::SmemBuilderFields{.sbase = 2, .sdata = 1, .ioffset = 0xfffffc, .soffset = 125};
  execute(cdna5::build_smem(cdna5::kSBufferPrefetchDataSmem, fields));
  EXPECT_EQ(l2.backing_read_transactions(), 0u);
  fields.ioffset = 128;
  execute(cdna5::build_smem(cdna5::kSBufferPrefetchDataSmem, fields));
  EXPECT_TRUE(resident(cu->l1_scalar(), kBase + 128));
  EXPECT_FALSE(resident(cu->l1_scalar(), kBase + 256));
}

TEST_F(Cdna5Prefetch, PcRelativeUsesNextPcAndInstructionCacheIgnoresDataPolicy) {
  wave->set_mode_raw(1u << 24);
  wave->pc = kBase + 256 - 4;
  const auto fields =
      cdna5::SmemBuilderFields{.scope = 3, .th = 3, .ioffset = 0xfffffc, .soffset = 124};
  execute(cdna5::build_smem(cdna5::kSPrefetchInstPcRelSmem, fields));
  EXPECT_TRUE(instruction_resident(kBase + 256));
  EXPECT_TRUE(instruction_resident(kBase + 320));
  EXPECT_FALSE(instruction_resident(kBase + 192));
  EXPECT_FALSE(resident(cu->l1_scalar(), kBase + 256));
  execute(cdna5::build_smem(cdna5::kSPrefetchDataPcRelSmem, {.ioffset = 0xfffffc, .soffset = 124}));
  EXPECT_TRUE(resident(cu->l1_scalar(), kBase + 256));
  EXPECT_FALSE(resident(cu->l1_scalar(), kBase + 192));
}

TEST_F(Cdna5Prefetch, ScalarInstructionBaseSkipsMissingPagesAndDebugger) {
  wave->set_mode_raw(1u << 24);
  const auto s = wave->sgpr_alloc().base;
  cu->write_sgpr(s + 4, kBase + 128);
  cu->write_sgpr(s + 5, 0);
  auto words = cdna5::build_smem(cdna5::kSPrefetchInstSmem,
                                 {.sbase = 2, .ioffset = 0xffff80, .soffset = 124});
  cu->set_debug_active(true);
  execute(words);
  EXPECT_FALSE(instruction_resident(kBase));
  cu->set_debug_active(false);
  execute(words);
  EXPECT_TRUE(instruction_resident(kBase));
  cu->write_sgpr(s + 4, kBase + 8192 + 128);
  execute(words);
  EXPECT_FALSE(instruction_resident(kBase + 8192));
}

TEST_F(Cdna5Prefetch, FetchAndIssueAdvanceAcrossBothInstructionSizes) {
  wave->set_mode_raw(1u << 24);
  wave->set_exec(1);
  wave->pc = 0x1000;
  const auto scalar = cdna5::build_smem(cdna5::kSPrefetchDataPcRelSmem,
                                        {.ioffset = kBase - 0x1008, .soffset = 124});
  const auto vector =
      cdna5::build_vglobal(cdna5::kGlobalPrefetchB8Vglobal, {.saddr = 124, .vaddr = 2});
  uint64_t pc = wave->pc;
  for (uint32_t word : scalar) {
    memory.write32(pc, word);
    pc += 4;
  }
  for (uint32_t word : vector) {
    memory.write32(pc, word);
    pc += 4;
  }
  cu->write_vgpr(wave->vgpr_alloc().base + 2, 0, kBase + 256);
  cu->write_vgpr(wave->vgpr_alloc().base + 3, 0, 0);
  cu->step();
  EXPECT_EQ(wave->pc, 0x1008u);
  EXPECT_TRUE(resident(cu->l1_scalar(), kBase));
  cu->step();
  EXPECT_EQ(wave->pc, 0x1014u);
  EXPECT_TRUE(resident(cu->l1_vector(), kBase + 256));
  EXPECT_EQ(wave->wait_counters().kmcnt, 0u);
  EXPECT_EQ(wave->wait_counters().vmcnt, 0u);
}

TEST_F(Cdna5Prefetch, GrantedRamWarmsTranslatedInstructionsAndData) {
  constexpr uint32_t vmid = 19;
  alignas(4096) std::array<uint8_t, 4096> bytes{};
  bytes.fill(0x56);
  amdgpu::LegacyPageTable pages;
  util::DistributedSharedMutex mutex;
  pages[kBase / 4096] = {bytes.data(), amdgpu::Mtype::RW};
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory);
  const auto space = adapter.register_address_space(
      vmid, {.page_table = &pages,
             .page_table_mutex = &mutex,
             .request_mutex = {},
             .local_grant_epoch = std::make_shared<std::atomic<uint64_t>>(1),
             .local_grant = [](uint64_t, size_t) {
               return amdgpu::VmLocalGrant{kBase, kBase + 4096, 1};
             }});
  ASSERT_TRUE(space);
  l2.set_gpu_vm(&vm);
  cu->set_gpu_vm(&vm);
  wave->set_process_id(vmid);
  wave->set_address_space(space);
  auto access = wave->snapshot_vm_access();
  ASSERT_TRUE(access);
  wave->set_mode_raw(1u << 24);
  cu->write_sgpr(wave->sgpr_alloc().base + 4, kBase);
  cu->write_sgpr(wave->sgpr_alloc().base + 5, 0);
  auto fields = cdna5::SmemBuilderFields{.sbase = 2, .soffset = 124};
  execute(cdna5::build_smem(cdna5::kSPrefetchInstSmem, fields));
  uint32_t word = 0;
  EXPECT_TRUE(cu->instruction_cache().peek_word(*access, kBase + 64, word));
  EXPECT_EQ(word, 0x56565656u);
  EXPECT_FALSE(cu->instruction_cache().peek_word(kBase, 0, word));
  const auto backing_reads = l2.backing_read_transactions();
  execute(cdna5::build_smem(cdna5::kSPrefetchDataSmem, fields));
  uint8_t first = 0;
  EXPECT_TRUE(
      cu->l1_scalar().try_read_peer_cached(kBase, {&first, 1}, vmid, 1, access->cache_identity()));
  EXPECT_EQ(first, 0x56u);
  EXPECT_EQ(l2.backing_read_transactions(), backing_reads + 1);
  execute(cdna5::build_smem(cdna5::kSPrefetchDataSmem, fields));
  EXPECT_EQ(l2.backing_read_transactions(), backing_reads + 1);
  wave->halt();
  wave = nullptr;
  cu->set_gpu_vm(nullptr);
  l2.set_gpu_vm(nullptr);
}

TEST_F(Cdna5Prefetch, GlobalSignedLaneOffsetsAndScope) {
  const auto s = wave->sgpr_alloc().base;
  cu->write_sgpr(s + 4, kBase + 256);
  cu->write_sgpr(s + 5, 0);
  const auto v = wave->vgpr_alloc().base;
  cu->write_vgpr(v + 2, 0, uint32_t(-255));
  cu->write_vgpr(v + 2, 1, 1);
  wave->set_exec(1);
  auto fields =
      cdna5::VglobalBuilderFields{.saddr = 4, .scope = 2, .vaddr = 2, .ioffset = 0xffffff};
  execute(cdna5::build_vglobal(cdna5::kGlobalPrefetchB8Vglobal, fields));
  EXPECT_TRUE(resident(l2, kBase));
  EXPECT_FALSE(resident(cu->l1_vector(), kBase));
  EXPECT_FALSE(resident(l2, kBase + 256));
  fields.scope = 0;
  execute(cdna5::build_vglobal(cdna5::kGlobalPrefetchB8Vglobal, fields));
  EXPECT_TRUE(resident(cu->l1_vector(), kBase));
  EXPECT_EQ(cu->read_vgpr(v + 2, 0), uint32_t(-255));
}

TEST_F(Cdna5Prefetch, FlatExcludesLdsAndInactiveLanes) {
  cu->set_apertures(kBase + 256, kBase + 383, 0, 0);
  const auto v = wave->vgpr_alloc().base;
  for (unsigned lane = 0; lane < 3; ++lane) {
    cu->write_vgpr(v + 2, lane, kBase + lane * 256 - 1);
    cu->write_vgpr(v + 3, lane, 0);
  }
  auto words =
      cdna5::build_vflat(cdna5::kFlatPrefetchB8Vflat, {.saddr = 124, .vaddr = 2, .ioffset = 1});
  wave->set_exec(0);
  execute(words);
  EXPECT_EQ(l2.backing_read_transactions(), 0u);
  wave->set_exec(3);
  execute(words);
  EXPECT_TRUE(resident(cu->l1_vector(), kBase));
  EXPECT_FALSE(resident(l2, kBase + 256));
  EXPECT_FALSE(resident(l2, kBase + 512));
}
} // namespace
} // namespace rocjitsu
