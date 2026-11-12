// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "counting_gpu_memory.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace rocjitsu {
namespace {
class Rdna1Memory : public testing::Test {
protected:
  virtual rj_code_arch_t arch() const { return ROCJITSU_CODE_ARCH_RDNA1; }

  void SetUp() override {
    config_.arch = arch();
    config_.num_wf_slots = 2;
    config_.sgprs_per_wf = 106;
    config_.vgprs_per_wf = 32;
    config_.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    ASSERT_NO_FATAL_FAILURE(
        test::initialize_compute_unit(cu_, "rdna1_cu", config_, &memory_, &cache_));
    decoder_ = Decoder::create(config_.arch);
    wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
    ASSERT_TRUE(wf_);
    wf_->set_exec(1);
    wf_->set_lds_base(512);
    wf_->set_lds_size(512);
  }
  void TearDown() override {
    if (wf_)
      wf_->halt();
  }
  void execute(std::array<uint32_t, 2> words) {
    inst_.reset(decode_valid(*decoder_, words.data()));
    ASSERT_NE(inst_, nullptr);
    ASSERT_TRUE(cu_->execute_instruction(inst_.get(), *wf_).succeeded());
  }
  void issue(std::array<uint32_t, 2> words) {
    ASSERT_NO_FATAL_FAILURE(execute(words));
    ASSERT_NE(inst_->data(), nullptr);
    if (inst_->data()->tag() == amdgpu::LOCAL_MEM)
      ASSERT_EQ(local_.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    else {
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      ASSERT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    }
  }
  void scalar(unsigned reg, uint32_t value) {
    cu_->write_sgpr(wf_->sgpr_alloc().base + reg, value);
  }
  void vector(unsigned reg, unsigned lane, uint32_t value) {
    cu_->write_vgpr(wf_->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t vector(unsigned reg, unsigned lane) const {
    return cu_->read_vgpr(wf_->vgpr_alloc().base + reg, lane);
  }
  amdgpu::ComputeUnitCore::Config config_{};
  amdgpu::GpuMemory memory_{"rdna1_memory"};
  amdgpu::L2Cache cache_{"rdna1_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wf_ = nullptr;
  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<Instruction> inst_;
  amdgpu::LocalMemPipeline local_;
};

TEST_F(Rdna1Memory, ScalarScratchScalesOnlyAlignedRegisterOffset) {
  scalar(0, 0x4003);
  scalar(1, 0);
  scalar(2, 7);
  for (unsigned op : {rdna1::kSLoadDword, rdna1::kSScratchLoadDword, rdna1::kSScratchLoadDwordx2,
                      rdna1::kSScratchLoadDwordx4}) {
    ASSERT_NO_FATAL_FAILURE(
        execute(rdna1::build_smem(op, {.sdata = 8, .offset = 7, .soffset = 2})));
    EXPECT_EQ(inst_->data_as<amdgpu::ScalarMemState>()->addr,
              op == rdna1::kSLoadDword ? 0x4008u : 0x4104u);
  }
}

TEST_F(Rdna1Memory, RawBuffersRespectUnboundResourcesAlignmentAndLdsExtent) {
  scalar(0, 0x4000);
  scalar(1, 0);
  scalar(2, 256);
  memory_.write32(0x4000, 0x44332211);
  memory_.write32(0x4004, 0x88776655);
  const auto load =
      rdna1::build_mubuf(rdna1::kBufferLoadDword, {.offset = 1, .vdata = 8, .soffset = 128});
  scalar(3, 3u << 28);
  vector(8, 0, ~0u);
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 0u);
  scalar(3, (3u << 28) | (20u << 12));
  ASSERT_NO_FATAL_FAILURE(issue(load));
  EXPECT_EQ(vector(8, 0), 0x44332211u);
  // M0[15:0] points at the last DWORD. Lane 1 would reach the next group's LDS.
  wf_->set_exec(3);
  wf_->set_m0(0x10000 + 508);
  const auto load_lds =
      rdna1::build_mubuf(rdna1::kBufferLoadUbyte, {.lds = 1, .vdata = 8, .soffset = 128});
  wf_->lds().write32(1020, ~0u);
  wf_->lds().write32(1024, 0xcccccccc);
  ASSERT_NO_FATAL_FAILURE(issue(load_lds));
  EXPECT_EQ(wf_->lds().read32(1020), 0x11u);
  EXPECT_EQ(wf_->lds().read32(1024), 0xccccccccu);
}

class Gfx10MemoryAllocation : public Rdna1Memory,
                              public testing::WithParamInterface<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const override { return GetParam(); }

  void bind_single_texel_image(uint64_t address) {
    const bool gfx1010 = arch() == ROCJITSU_CODE_ARCH_RDNA1;
    scalar(0, address >> 8);
    scalar(1, (20u << 20) | (gfx1010 ? 1u << 29 : 0u));
    scalar(3, (9u << 28) | 4u);
    // GFX10.3 encodes linear pitch in texels minus one. A DWORD row spans 256 bytes.
    scalar(4, gfx1010 ? 0u : 63u);
  }
};

INSTANTIATE_TEST_SUITE_P(Gfx10, Gfx10MemoryAllocation,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2));

TEST_P(Gfx10MemoryAllocation, BufferLdsLoadsExtendNarrowValuesAtM0PlusOffset) {
  constexpr uint32_t address = 0x4000;
  constexpr uint32_t offset = 4;
  constexpr uint32_t m0_offset = 32;
  constexpr uint32_t first = 0x123480ab;
  constexpr uint32_t second = 0x987681f5;
  constexpr uint32_t sentinel = 0xcccccccc;
  scalar(0, address);
  scalar(1, 0);
  scalar(2, 256);
  scalar(3, (3u << 28) | (1u << 24) | (20u << 12) | 4u); // 32_UINT, select X.
  wf_->set_exec(5);
  wf_->set_m0(0x10000 + m0_offset);
  vector(2, 0, 0);
  vector(2, 2, 8);
  memory_.write32(address + offset, first);
  memory_.write32(address + offset + 8, second);
  const auto destination = wf_->lds_base() + m0_offset + offset;
  for (const auto &[op, lane0, lane2] :
       {std::tuple{rdna1::kBufferLoadSbyte, 0xffffffabu, 0xfffffff5u},
        std::tuple{rdna1::kBufferLoadSshort, 0xffff80abu, 0xffff81f5u},
        std::tuple{rdna1::kBufferLoadFormatX, first, second}}) {
    SCOPED_TRACE(op);
    for (unsigned i = 0; i < 5; ++i)
      wf_->lds().write32(destination - 4 + 4 * i, sentinel);
    ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_mubuf(
        op, {.offset = offset, .offen = 1, .lds = 1, .vaddr = 2, .vdata = 8, .soffset = 128})));
    EXPECT_EQ(wf_->lds().read32(destination - 4), sentinel);
    EXPECT_EQ(wf_->lds().read32(destination), lane0);
    EXPECT_EQ(wf_->lds().read32(destination + 4), sentinel);
    EXPECT_EQ(wf_->lds().read32(destination + 8), lane2);
    EXPECT_EQ(wf_->lds().read32(destination + 12), sentinel);
  }
}

TEST_P(Gfx10MemoryAllocation, GenericFlatIgnoresImmediateBitEleven) {
  constexpr uint32_t pointer = 0x100000;
  vector(2, 0, pointer);
  vector(3, 0, 0);
  for (const auto &[word, address] : {std::pair{0xdc300800u, pointer},
                                      {0xdc3007fcu, pointer + 2044},
                                      {0xdc308800u, pointer - 2048}}) {
    const std::array words{word, 0x017d0002u};
    ASSERT_NO_FATAL_FAILURE(execute(words));
    const auto *data = inst_->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->per_lane_addr[0], address);
    if (word != 0xdc308800u) {
      const auto lanes = amdgpu::MemoryWaitScoreboard::flat_lanes(*inst_, *wf_, address, address);
      EXPECT_EQ(lanes.shared, 1u);
    }
  }
}

TEST_P(Gfx10MemoryAllocation, FlatWaitUsesTheSharedApertureAtIssue) {
  constexpr uint16_t sh_mem_bases = 15 | (31 << 11);
  const auto setreg = rdna1::build_sopk(rdna1::kSSetregB32, {.simm16 = sh_mem_bases, .sdst = 2});
  const auto load = rdna1::build_flat(rdna1::kFlatLoadDword, {.saddr = 125, .vdst = 8});
  const auto vm_wait = rdna1::build_sopp(rdna1::kSWaitcnt, {.simm16 = 0x3f70});
  const auto read = rdna1::build_vop1(rdna1::kVMovB32Vop1, {.src0 = 256 + 8, .vdst = 9});
  const auto full_wait = rdna1::build_sopp(rdna1::kSWaitcnt, {.simm16 = 0});
  constexpr uint64_t pc = 0x1000;
  const std::array code{setreg[0], load[0], load[1], vm_wait[0], read[0], full_wait[0], read[0]};
  for (size_t i = 0; i < code.size(); ++i)
    memory_.write32(pc + 4 * i, code[i]);

  for (bool ptr32 : {false, true}) {
    if (ptr32 && arch() != ROCJITSU_CODE_ARCH_RDNA1)
      continue;
    const uint64_t launch = uint64_t{1} << (ptr32 ? 16 : 48);
    for (bool relocate : {false, true}) {
      SCOPED_TRACE(ptr32);
      SCOPED_TRACE(relocate);
      wf_->halt();
      cu_->set_apertures(launch, launch + 0xffff, 0, 0, ptr32);
      wf_ = cu_->dispatch_wf(0, relocate ? pc : pc + 4, 106, 16, 64);
      ASSERT_NE(wf_, nullptr);
      wf_->set_exec(1);
      wf_->set_lds_size(512);
      wf_->lds().write32(wf_->lds_base() + 4, 0x12345678);
      scalar(2, 2u << 16);
      const uint64_t address = (relocate ? 2 * launch : launch) + 4;
      vector(0, 0, static_cast<uint32_t>(address));
      vector(1, 0, address >> 32);
      vector(9, 0, 0);

      const auto warnings = cu_->memory_wait_diagnostic_count();
      if (relocate) {
        ASSERT_TRUE(cu_->step()); // s_setreg_b32 SH_MEM_BASES, s2
      }
      ASSERT_TRUE(cu_->step()); // flat_load_dword v8, v[0:1]
      ASSERT_TRUE(cu_->step()); // s_waitcnt vmcnt(0)
      ASSERT_TRUE(cu_->step()); // v_mov_b32 v9, v8
      EXPECT_EQ(vector(9, 0), 0x12345678u);
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), warnings + 1);
      ASSERT_TRUE(cu_->step()); // s_waitcnt 0
      ASSERT_TRUE(cu_->step()); // v_mov_b32 v9, v8
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), warnings + 1);
    }
  }
}

TEST_P(Gfx10MemoryAllocation, UnallocatedLastWordSuppressesWholeLdsTuple) {
  wf_->set_m0(8);
  vector(0, 0, 4);
  vector(15, 0, 0x1515);
  vector(16, 0, 0x1616);
  wf_->lds().write32(wf_->lds_base() + 4, 41);
  ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_ds(rdna1::kDsRead2B32, {.addr = 0, .vdst = 15})));
  EXPECT_EQ(vector(15, 0), 0x1515u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(issue(rdna1::build_ds(rdna1::kDsRead2B32, {.addr = 0, .vdst = 14})));
  EXPECT_EQ(vector(14, 0), 41u);
  EXPECT_EQ(vector(15, 0), 41u);
}

TEST_P(Gfx10MemoryAllocation, ConditionalAtomicResultsAndLdsRedirectUseActualDestination) {
  constexpr uint32_t address = 0x4000;
  memory_.write32(address, 9);
  vector(0, 0, address);
  vector(1, 0, 0);
  vector(2, 0, 3);
  vector(16, 0, 0x1616);
  const auto add = [](uint8_t glc) {
    return rdna1::build_flat(rdna1::kFlatAtomicAdd,
                             {.seg = 2, .glc = glc, .data = 2, .saddr = 125, .vdst = 16});
  };
  ASSERT_NO_FATAL_FAILURE(issue(add(1)));
  EXPECT_EQ(memory_.read32(address), 9u);
  EXPECT_EQ(vector(16, 0), 0x1616u);
  ASSERT_NO_FATAL_FAILURE(issue(add(0)));
  EXPECT_EQ(memory_.read32(address), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  scalar(0, address);
  scalar(1, 0);
  scalar(2, 256);
  scalar(3, (3u << 28) | (20u << 12));
  wf_->set_m0(0);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_mubuf(rdna1::kBufferLoadDword, {.lds = 1, .vdata = 16, .soffset = 128})));
  EXPECT_EQ(wf_->lds().read32(wf_->lds_base()), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  // CMPSWAP consumes two VGPRs but returns only the old dword. V15 is a
  // legal destination even though the second source is outside the allocation.
  const auto cmp =
      rdna1::build_mubuf(rdna1::kBufferAtomicCmpswap, {.glc = 1, .vdata = 15, .soffset = 128});
  ASSERT_NO_FATAL_FAILURE(issue(cmp));
  EXPECT_EQ(vector(15, 0), 12u);
  EXPECT_EQ(vector(16, 0), 0x1616u);

  // In a non-returning MUBUF atomic, VDATA is only an input. A source with
  // an undefined value must still issue the transaction with the active lanes.
  ASSERT_NO_FATAL_FAILURE(
      execute(rdna1::build_mubuf(rdna1::kBufferAtomicAdd, {.vdata = 16, .soffset = 128})));
  const auto *state = inst_->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->lane_mask, 1u);
  EXPECT_FALSE(state->is_load);
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
  EXPECT_EQ(pipeline.issue(inst_.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(vector(16, 0), 0x1616u);
}

TEST_F(Rdna1Memory, DualDsHonorsIndependentBoundsAndAliasing) {
  vector(0, 0, 508);
  vector(1, 0, 0x12345678);
  vector(2, 0, 0xabcdef01);
  wf_->lds().write32(1024, 0xcccccccc);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsWrite2B32, {.offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2})));
  EXPECT_EQ(wf_->lds().read32(1020), 0x12345678u);
  EXPECT_EQ(wf_->lds().read32(1024), 0xccccccccu);
  ASSERT_NO_FATAL_FAILURE(issue(
      rdna1::build_ds(rdna1::kDsWrxchg2RtnB32, {.addr = 0, .data0 = 2, .data1 = 1, .vdst = 4})));
  EXPECT_EQ(wf_->lds().read32(1020), 0xabcdef01u);
  EXPECT_EQ(vector(4, 0), 0x12345678u);
  EXPECT_EQ(vector(5, 0), 0x12345678u);
}

TEST_F(Rdna1Memory, ScratchBoundsApplyPerLaneAndFlatOffsetIsUnsigned) {
  wf_->set_exec(3);
  wf_->set_scratch_lane_size(16);
  wf_->set_scratch_base(0x10000);
  vector(0, 0, 16);
  vector(0, 1, 12);
  memory_.write32(0x10400, 0xfeedface);
  memory_.write32(0x10304, 0x12345678);
  ASSERT_NO_FATAL_FAILURE(
      issue(rdna1::build_flat(rdna1::kFlatLoadDword, {.seg = 1, .saddr = 125, .vdst = 8})));
  EXPECT_EQ(vector(8, 0), 0u);
  EXPECT_EQ(vector(8, 1), 0x12345678u);
  wf_->set_exec(1);
  vector(0, 0, 0x4000);
  vector(1, 0, 0);
  ASSERT_NO_FATAL_FAILURE(execute(
      rdna1::build_flat(rdna1::kFlatLoadDword, {.offset = 0x800, .saddr = 125, .vdst = 8})));
  EXPECT_EQ(inst_->data_as<amdgpu::VectorMemState>()->per_lane_addr[0], 0x4000u);
}

TEST_F(Rdna1Memory, Ptr32IsCapturedAtLaunchAndReadsOneAddressRegister) {
  wf_->halt();
  cu_->set_apertures(0x10000, 0x1ffff, 0x20000, 0x2ffff, /*ptr32=*/true);
  wf_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_TRUE(wf_);
  wf_->set_exec(7);
  wf_->set_lds_size(512);
  wf_->set_scratch_base(0x40000);
  wf_->set_scratch_lane_size(16);
  constexpr uint16_t sh_mem_bases = 15 | (31 << 11);
  uint32_t bases = 0;
  ASSERT_EQ(amdgpu::read_hwreg_field(*wf_, sh_mem_bases, bases),
            amdgpu::HwregAccessResult::Success);
  EXPECT_EQ(bases, 0x10002u);
  ASSERT_EQ(amdgpu::write_hwreg_field(*wf_, sh_mem_bases, 0x60005),
            amdgpu::HwregAccessResult::Success);
  vector(15, 0, 0x60004);
  vector(15, 1, 0x50008);
  vector(15, 2, 0x30008);
  wf_->lds().write32(4, 17);
  memory_.write32(0x40000 + 2 * 64 * 4 + 4, 23);
  memory_.write32(0x30008, 31);
  const auto load = rdna1::build_flat(rdna1::kFlatLoadDword, {.addr = 15, .saddr = 125, .vdst = 8});
  ASSERT_NO_FATAL_FAILURE(execute(load));
  const auto &state = *inst_->data_as<amdgpu::VectorMemState>();
  EXPECT_EQ(state.scratch_lane_mask, 2u);
  EXPECT_EQ(state.per_lane_addr[0], 0x60004u);
  EXPECT_EQ(state.per_lane_addr[1], 0x40204u);
  EXPECT_EQ(state.per_lane_addr[2], 0x30008u);
  memory_.write32(wf_->pc, load[0]);
  memory_.write32(wf_->pc + sizeof(uint32_t), load[1]);
  ASSERT_TRUE(cu_->step());
  EXPECT_EQ(vector(8, 0), 17u);
  EXPECT_EQ(vector(8, 1), 23u);
  EXPECT_EQ(vector(8, 2), 31u);
}

TEST_F(Rdna1Memory, Wave64PermutesWithinEachPhysicalHalf) {
  wf_->set_exec(~uint64_t{0});
  for (unsigned lane = 0; lane < 64; ++lane) {
    vector(0, lane, 0);
    vector(1, lane, 100 + lane);
  }
  ASSERT_NO_FATAL_FAILURE(
      execute(rdna1::build_ds(rdna1::kDsBpermuteB32, {.addr = 0, .data0 = 1, .vdst = 2})));
  for (unsigned lane = 0; lane < 64; ++lane)
    EXPECT_EQ(vector(2, lane), lane < 32 ? 100u : 132u);
}

TEST_F(Rdna1Memory, LdsStoreExecutesAndIndexedGdsIsRejectedBeforeWriteback) {
  vector(0, 0, 4);
  vector(1, 0, 9);
  const auto normal = rdna1::build_ds(rdna1::kDsWriteB32, {.addr = 0, .data0 = 1});
  ASSERT_NO_FATAL_FAILURE(issue(normal));
  EXPECT_EQ(wf_->lds().read32(wf_->lds_base() + 4), 9u);

  vector(1, 0, 42);
  const auto indexed = rdna1::build_ds(rdna1::kDsWriteB32, {.gds = 1, .addr = 0, .data0 = 1});
  auto decoded = decoder_->decode_window(indexed);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_EQ(decoded.value()->data(), nullptr);
  EXPECT_EQ(wf_->lds().read32(wf_->lds_base() + 4), 9u);
}
} // namespace
} // namespace rocjitsu
