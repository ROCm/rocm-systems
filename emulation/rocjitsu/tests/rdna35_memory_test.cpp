// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

#include <array>
#include <utility>

namespace rocjitsu {
namespace {
using namespace amdgpu;

class Rdna35MemoryTest : public testing::Test {
protected:
  static constexpr uint64_t kShared = 0x1000100000000ULL;
  static constexpr uint64_t kPrivate = 0x1000000000000ULL;

  virtual rj_code_arch_t architecture() const { return ROCJITSU_CODE_ARCH_RDNA3_5; }

  void make_cu(std::unique_ptr<ComputeUnitCore> &unit) {
    ComputeUnitCore::Config cfg{};
    cfg.arch = architecture();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    // Leave physical registers beyond the 32 allocated to the guest wave.
    cfg.vgprs_per_wf = 64;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(unit, "rdna35", cfg, &memory_, &l2_));
    ASSERT_TRUE(unit->set_gws_device(device_));
  }

  void SetUp() override {
    decoder_ = Decoder::create(architecture());
    ASSERT_NE(decoder_, nullptr);
    l2_.set_backing_memory(&memory_);
    ASSERT_NO_FATAL_FAILURE(make_cu(cu_));
    cu_->set_apertures(kShared, kShared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
    wf_ = cu_->dispatch_wf(0, 0, 106, 32, architecture() == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32);
    ASSERT_NE(wf_, nullptr);
    wf_->set_apertures(kShared, kShared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX,
                       /*override_launch=*/true);
    wf_->set_lds_base(1024);
    wf_->set_lds_size(1024);
    wf_->set_scratch_base(0x20000);
    wf_->set_scratch_lane_size(256);
    wf_->set_exec(1);
  }

  void TearDown() override {
    if (wf_)
      wf_->halt();
  }

  void issue(uint32_t word0, uint32_t word1, VmAccessOutcome expected = VmAccessOutcome::Complete) {
    issue_on(*cu_, *wf_, word0, word1, expected);
  }

  void issue_on(ComputeUnitCore &unit, Wavefront &wave, uint32_t word0, uint32_t word1,
                VmAccessOutcome expected = VmAccessOutcome::Complete) {
    const std::array words{word0, word1};
    auto decoded = decoder_->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(unit.execute_instruction(inst.get(), wave).succeeded()) << inst->mnemonic();
    ASSERT_NE(inst->data(), nullptr);
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(unit, inst.release(), wave), expected);
  }

  void write_vector(unsigned reg, uint32_t value, unsigned lane = 0) {
    cu_->write_vgpr(wf_->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t read_vector(unsigned reg = 0, unsigned lane = 0) {
    return cu_->read_vgpr(wf_->vgpr_alloc().base + reg, lane);
  }
  void write_scalar(unsigned reg, uint32_t value) {
    cu_->write_sgpr(wf_->sgpr_alloc().base + reg, value);
  }

  std::shared_ptr<GwsDevice> device_ = std::make_shared<GwsDevice>();
  GpuMemory memory_{"rdna35_memory"};
  L2Cache l2_{"rdna35_l2"};
  std::unique_ptr<ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  Wavefront *wf_ = nullptr;
};

class Gfx11MemoryTest : public Rdna35MemoryTest,
                        public testing::WithParamInterface<rj_code_arch_t> {
  rj_code_arch_t architecture() const override { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11MemoryTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11MemoryTest, ScratchAndFlatPrivateShareDwordInterleaving) {
  wf_->set_exec(2);
  memory_.write32(0x20000 + 132, 0x12345678);
  memory_.write32(0x20000 + 260, 0xdeadbeef);
  write_vector(2, 4, 1);
  issue(0xdc510000, 0x00fc0002); // scratch_load_b32 v0, v2, off
  EXPECT_EQ(read_vector(0, 1), 0x12345678u);
  write_vector(3, kPrivate >> 32, 1);
  issue(0xdc500000, 0x007c0002); // flat_load_b32 v0, v[2:3]
  EXPECT_EQ(read_vector(0, 1), 0x12345678u);

  // Adjacent lanes own adjacent DWORDs; a lane's next DWORD is a wave stride away.
  constexpr uint64_t pair_base = 0x25000;
  wf_->set_scratch_base(pair_base);
  wf_->set_exec(3);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    write_vector(2, 0, lane);
    write_vector(3, kPrivate >> 32, lane);
    memory_.write32(pair_base + lane * 4, 10 + lane);
    memory_.write32(pair_base + 32 * 4 + lane * 4, 20 + lane);
  }
  issue(0xdc540000, 0x007c0002); // flat_load_b64 v[0:1], v[2:3]
  for (uint32_t lane = 0; lane < 2; ++lane) {
    EXPECT_EQ(read_vector(0, lane), 10u + lane);
    EXPECT_EQ(read_vector(1, lane), 20u + lane);
  }

  wf_->set_exec(3);
  write_vector(2, 0x40000);
  write_vector(3, 0);
  write_vector(4, 1);
  write_vector(5, 0);
  write_vector(2, 0, 1);
  memory_.write32(0x40000, 41);
  ASSERT_EQ(wf_->trapsts() & (1u << 8), 0u);
  issue(0xdd0c4000, 0x007c0402); // flat_atomic_add_u64 glc
  EXPECT_NE(wf_->trapsts() & (1u << 8), 0u);
  EXPECT_EQ(memory_.read32(0x40000), 41u);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    EXPECT_EQ(memory_.read32(pair_base + lane * 4), 10u + lane);
    EXPECT_EQ(memory_.read32(pair_base + 32 * 4 + lane * 4), 20u + lane);
    EXPECT_EQ(read_vector(0, lane), 10u + lane);
    EXPECT_EQ(read_vector(1, lane), 20u + lane);
  }
}

class ReturningDsAtomicRouteTest : public Rdna35MemoryTest,
                                   public testing::WithParamInterface<rj_code_arch_t> {
  rj_code_arch_t architecture() const override { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(Memory, ReturningDsAtomicRouteTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                         ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(ReturningDsAtomicRouteTest, DirectRouteQualifiesReturningAtomicBeforeWritingLds) {
  write_vector(2, 8);
  write_vector(3, 1);
  const auto address = wf_->lds_base() + 8;
  const bool cdna4 = GetParam() == ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t ordinary_vgprs = cdna4 ? 16 : 32;
  if (cdna4)
    wf_->set_accum_offset(ordinary_vgprs);
  for (uint32_t dst : {ordinary_vgprs - 1, ordinary_vgprs}) {
    SCOPED_TRACE(dst);
    wf_->lds().write32(address, 41);
    // ds_add_rtn_u32 vdst, v2, v3
    const uint32_t opcode = cdna4 ? 0xd8400000 : 0xd8800000;
    const std::array words{opcode, (dst << 24) | 0x0302u};
    auto inst = decoder_->decode_window(words);
    ASSERT_TRUE(inst.succeeded());
    inst.value()->execute(*inst.value(), wf_);
    ASSERT_NE(inst.value()->data(), nullptr);
    const bool masked_at_issue = cdna4 && dst == ordinary_vgprs;
    ASSERT_EQ(inst.value()->data_as<VectorMemState>()->lane_mask, masked_at_issue ? 0u : 1u);
    if (masked_at_issue) {
      // Capture a live request, then give the direct router the original, narrower allocation.
      inst = decoder_->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      wf_->set_accum_offset(wf_->num_vgprs());
      inst.value()->execute(*inst.value(), wf_);
      wf_->set_accum_offset(ordinary_vgprs);
      ASSERT_NE(inst.value()->data(), nullptr);
      ASSERT_EQ(inst.value()->data_as<VectorMemState>()->lane_mask, 1u);
    }
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(
                  *cu_, std::move(inst).value().release(), *wf_),
              VmAccessOutcome::Complete);
    EXPECT_EQ(wf_->lds().read32(address), dst < ordinary_vgprs ? 42u : 41u);
  }
}

TEST_P(Gfx11MemoryTest, LdsAllocationRoundsEachWorkgroupInKiBUnits) {
  EXPECT_EQ(cu_->allocate_lds(1), 0u);
  EXPECT_EQ(cu_->allocate_lds(513), 1024u);
  EXPECT_EQ(cu_->allocate_lds(1025), 2048u);
  EXPECT_EQ(cu_->allocate_lds(1), 4096u);
}

TEST_P(Gfx11MemoryTest, DsReadsStopAtTheWindowInsideTheReservedKiB) {
  wf_->set_lds_size(lds_workgroup_window_bytes(wf_->cu().arch(), 4));
  for (uint32_t offset : {508u, 512u, 1020u}) {
    SCOPED_TRACE(offset);
    wf_->lds().write32(wf_->lds_base() + offset, 0x12345678);
    write_vector(2, offset);
    issue(0xd8d80000, 2); // ds_load_b32 v0, v2
    EXPECT_EQ(read_vector(), offset < 512 ? 0x12345678u : 0u);
  }
}

TEST_F(Rdna35MemoryTest, GuardLdsAllocationAndReturningAtomicDestinations) {
  wf_->lds().write32(2048, 41);
  write_vector(2, 1024);
  issue(0xd8d80000, 2); // ds_load_b32 v0, v2
  EXPECT_EQ(read_vector(), 0u);
  write_vector(3, kShared >> 32);
  issue(0xdc500000, 0x007c0002);
  EXPECT_EQ(read_vector(), 0u);
  wf_->lds().write32(1032, 41);
  write_vector(2, 8);
  write_vector(3, 1);
  issue(0xd8800000, 0x20000302); // ds_add_rtn_u32 v32, v2, v3
  EXPECT_EQ(wf_->lds().read32(1032), 41u);
  memory_.write32(0x40000, 41);
  write_vector(2, 0x40000);
  write_vector(3, 0);
  write_vector(4, 1);
  issue(0xdcd64000, 0x207c0402); // global_atomic_add_u32 v32, v[2:3], v4, off glc
  EXPECT_EQ(memory_.read32(0x40000), 41u);
  wf_->set_exec(3);
  write_vector(2, 1, 1);
  write_vector(3, kShared >> 32, 1);
  write_vector(4, 7, 1);
  issue(0xdcd44000, 0x007c0402, VmAccessOutcome::Faulted); // flat_atomic_add_u32 glc
  EXPECT_EQ(memory_.read32(0x40000), 41u); // Preflight the mixed operation before any RMW.
}

TEST_P(Gfx11MemoryTest, SixBitBufferFormatControlsMemoryAndAddressDependencies) {
  write_scalar(0, 0x40000);
  write_scalar(1, 4u << 16);
  write_scalar(2, 4);
  write_vector(4, 4);
  memory_.write32(0x40004, 0x11223344);
  const auto words = rdna3::build_mubuf(rdna3::kBufferLoadB32Mubuf,
                                        {.vaddr = 4, .vdata = 8, .offen = 1, .soffset = 128});
  for (bool bound : {false, true}) {
    SCOPED_TRACE(bound);
    write_scalar(3, bound ? 0x20054000 : 0x20040000); // Bit 18 is outside FORMAT[17:12].
    write_vector(8, 0xdeadbeef);
    auto decoded = decoder_->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    auto &wait = wf_->ensure_memory_wait_scoreboard();
    unsigned hazards = 0;
    wait.clear();
    wait.bind(0x200, &hazards, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    wait.add({wait.issue(WaitCounterKind::Load),
              0x100,
              1,
              {RegClass::VGPR, 4, 1},
              WaitCounterKind::Load,
              0xf});
    wait.check_instruction(*inst, *wf_);
    EXPECT_EQ(hazards, unsigned(bound));
    wait.clear();

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    EXPECT_EQ(inst->data_as<VectorMemState>()->lane_mask, uint64_t(bound));
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu_, inst.release(), *wf_),
              VmAccessOutcome::Complete);
    EXPECT_EQ(read_vector(8), bound ? 0x11223344u : 0u);
  }
}

TEST_P(Gfx11MemoryTest, DistinguishUnboundBufferFromWrongResourceTypeAndMissingStatusVgpr) {
  write_scalar(0, 0x40000);
  write_scalar(1, 4u << 16);
  write_scalar(2, 4);
  write_scalar(3, 2u << 28); // FORMAT=0, ADD_TID=0
  memory_.write32(0x40000, 0x11223344);
  write_vector(0, 0xdeadbeef);
  issue(0xe0500000, 0x80000000); // buffer_load_b32 v0, off, s[0:3], 0
  EXPECT_EQ(read_vector(), 0u);
  // ADD_TID indexes lane 1 only when the descriptor is bound.
  constexpr uint32_t kBoundFormatWithThreadId = (2u << 28) | (1u << 23) | (20u << 12);
  wf_->set_exec(2);
  for (bool bound : {false, true}) {
    write_scalar(3, bound ? kBoundFormatWithThreadId : (2u << 28) | (1u << 23));
    memory_.write32(0x40004, 0x11223345);
    write_vector(0, 0xdeadbeef, 1);
    issue(0xe0500000, 0x80000000); // buffer_load_b32
    EXPECT_EQ(read_vector(0, 1), bound ? 0x11223345u : 0u);
    write_vector(0, 0x76543210, 1);
    issue(0xe0680000, 0x80000000); // buffer_store_b32
    EXPECT_EQ(memory_.read32(0x40004), bound ? 0x76543210u : 0x11223345u);
    EXPECT_EQ(memory_.read32(0x40000), 0x11223344u);
  }
  wf_->set_exec(1);
  write_scalar(3, (8u << 28) | (20u << 12)); // 1D image descriptor type
  write_vector(0, 0xdeadbeef);
  issue(0xe0500000, 0x80000000);
  EXPECT_EQ(read_vector(), 0xdeadbeefu);
  write_scalar(3, (2u << 28) | (20u << 12));
  write_vector(31, 0xdeadbeef);
  issue(0xe0500000, 0x80201f00); // buffer_load_b32 v[31:32] ... tfe
  EXPECT_EQ(read_vector(31), 0xdeadbeefu);
  issue(0xe0500000, 0x80000000);
  EXPECT_EQ(read_vector(), 0x11223344u);
}

class ResidentBelowBufferBoundary final : public AddressSpaceTranslator {
public:
  void publish_hole(bool sparse, bool clear_after_snapshot = false) {
    clear_after_snapshot_ = clear_after_snapshot;
    ranges_ = sparse ? std::make_shared<const PrtRanges>(PrtRanges{{0x50000, 0x51000}}) : nullptr;
  }
  std::shared_ptr<const PrtRanges> nonresident_prt_ranges() const override {
    return clear_after_snapshot_ ? std::exchange(ranges_, nullptr) : ranges_;
  }

private:
  bool clear_after_snapshot_ = false;
  mutable std::shared_ptr<const PrtRanges> ranges_;
  IdentityAddressSpaceTranslator identity;
  VmTranslationResult translate(uint64_t address, size_t size, VmAccessKind kind) const override {
    if (address >= 0x50000)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    auto result = identity.translate(address, size, kind);
    result.translation.contiguous_bytes =
        std::min<uint64_t>(result.translation.contiguous_bytes, 0x50000 - address);
    return result;
  }
};

TEST_P(Gfx11MemoryTest, DebuggerProbesOnlyResidentBufferTfeElements) {
  constexpr uint32_t limit = 0x50000, kernel = 0x1000, sentinel = 0xdeadbeef;
  GpuVm vm;
  const auto translator = std::make_shared<ResidentBelowBufferBoundary>();
  const auto handle =
      vm.register_address_space(1, translator, std::make_shared<GpuMemoryPhysicalAccess>(memory_));
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  write_scalar(1, 0);
  write_scalar(2, 32);
  write_scalar(3, (2u << 28) | (20u << 12));
  write_vector(4, 0);
  memory_.write32(limit - 8, 11);
  memory_.write32(limit - 4, 17);
  memory_.write32(kernel, 0xe0540000); // buffer_load_b64 v[0:2],v4,s[0:3],0 offen tfe
  memory_.write32(kernel + 4, 0x80600004);
  memory_.write32(kernel + 8, 0xe0540000); // The same load without TFE.
  memory_.write32(kernel + 12, 0x80400004);
  std::vector<std::pair<uint64_t, uint32_t>> watches;
  unsigned faults = 0;
  cu_->set_debug_active(true);
  cu_->set_watchpoint_handler([&](Wavefront &, uint64_t addr, uint32_t bytes, bool, bool) {
    watches.emplace_back(addr, bytes);
    return false;
  });
  cu_->set_memory_violation_handler([&](Wavefront &, uint64_t, bool, uint64_t) {
    ++faults;
    return true;
  });
  for (bool sparse : {false, true}) {
    for (bool tfe : {false, true}) {
      for (unsigned resident : {0u, 1u, 2u}) {
        SCOPED_TRACE(testing::Message() << sparse << "/" << tfe << "/" << resident);
        // Release the partial reservation after the debugger takes its snapshot,
        // before the same instruction reaches the memory pipeline.
        translator->publish_hole(sparse, sparse && tfe && resident == 1);
        const uint32_t address = limit - 4 * resident;
        write_scalar(0, address);
        for (unsigned reg = 0; reg < 3; ++reg)
          write_vector(reg, sentinel);
        wf_->pc = tfe ? kernel : kernel + 8;
        watches.clear();
        faults = 0;
        ASSERT_TRUE(cu_->step());
        const bool claimed = (!tfe || !sparse) && resident < 2;
        EXPECT_EQ(faults, claimed ? 1u : 0u);
        EXPECT_EQ(read_vector(), claimed ? sentinel : resident ? memory_.read32(address) : 0u);
        EXPECT_EQ(read_vector(1), claimed ? sentinel : resident == 2 ? 17u : 0u);
        EXPECT_EQ(read_vector(2), tfe && sparse && resident != 2 ? 1u : sentinel);
        uint32_t watched_bytes = 0;
        for (const auto &[watch_address, watch_bytes] : watches) {
          EXPECT_GE(watch_address, address);
          EXPECT_LE(watch_address + watch_bytes, limit);
          watched_bytes += watch_bytes;
        }
        EXPECT_EQ(watched_bytes, claimed ? 0u : resident * 4u);
      }
    }
  }
  cu_->set_debug_active(false);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, BufferTfePreservesResidentDwordsAndReportsMissingPages) {
  GpuVm vm;
  unsigned faults = 0;
  const auto translator = std::make_shared<ResidentBelowBufferBoundary>();
  const auto handle =
      vm.register_address_space(1, translator, std::make_shared<GpuMemoryPhysicalAccess>(memory_),
                                [&](uint64_t, VmAccessKind) { ++faults; });
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_exec(3);
  write_scalar(0, 0x4fff8);
  write_scalar(1, 0);
  write_scalar(2, 32);
  write_scalar(3, (2u << 28) | (20u << 12));
  memory_.write32(0x4fff8, 11);
  memory_.write32(0x4fffc, 17);
  write_vector(4, 0);
  write_vector(4, 4, 1);
  write_vector(2, 0xdeadbeef);
  write_vector(2, 0xdeadbeef, 1);
  // Without a sparse reservation the same unmapped DWORD raises a VM fault.
  issue(0xe0540000, 0x80600004, VmAccessOutcome::Faulted);
  EXPECT_EQ(faults, 1u);
  EXPECT_EQ(read_vector(2), 0xdeadbeefu);
  EXPECT_EQ(read_vector(2, 1), 0xdeadbeefu);
  translator->publish_hole(true);
  issue(0xe0540000, 0x80600004); // buffer_load_b64 v[0:2],v4,s[0:3],0 offen tfe
  EXPECT_EQ(read_vector(), 11u);
  EXPECT_EQ(read_vector(1), 17u);
  EXPECT_EQ(read_vector(2), 0xdeadbeefu);
  EXPECT_EQ(read_vector(0, 1), 17u);
  EXPECT_EQ(read_vector(1, 1), 0u);
  EXPECT_EQ(read_vector(2, 1), 1u);
  EXPECT_EQ(faults, 1u);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, AlignScalarAddressComponentsAndFaultNegativeBufferImmediate) {
  write_scalar(0, 0x60001);
  write_scalar(1, 0);
  write_scalar(2, 32);
  write_scalar(5, 0);
  memory_.write32(0x60000, 0x66666666);
  memory_.write32(0x60004, 0x77777777);
  issue(0xf4000100, 0x0a000003); // s_load_b32 s4, s[0:1], s5 offset:3
  EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 0x66666666u);
  issue(0xf4200100, 0x0a000003); // s_buffer_load_b32 s4, s[0:3], s5 offset:3
  EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 0x66666666u);
  write_scalar(4, 0xdeadbeef);
  write_scalar(5, 4);
  issue(0xf4200100, 0x0a1ffffc, VmAccessOutcome::Faulted); // imm=-4, combined offset=0
  EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 0xdeadbeefu);
}

TEST_F(Rdna35MemoryTest, FlatIgnoresOffsetBitTwelve) {
  write_vector(2, 0x70000);
  write_vector(3, 0);
  memory_.write32(0x70000, 41);
  memory_.write32(0x6f000, 42);
  issue(0xdc501000, 0x007c0002); // flat_load_b32 v0, v[2:3], raw offset 0x1000
  EXPECT_EQ(read_vector(), 41u);
}
} // namespace
} // namespace rocjitsu
