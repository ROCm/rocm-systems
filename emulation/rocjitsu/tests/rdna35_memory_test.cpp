// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
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

  static constexpr uint32_t gds(uint32_t opcode, uint16_t offsets = 0) {
    return 0xd8020000u | (opcode << 18) | offsets;
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

  write_scalar(0, 0x50000);
  issue(0xf4880000, 0x0a000000); // Unmapped ATC probe drops without a VM fault.
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

TEST_F(Rdna35MemoryTest, AtcProbesWarmResidentScalarLinesWithoutReadingBackingOnMiss) {
  constexpr uint64_t address = 0x80040;
  uint32_t a = 17;
  memory_.write32(address, a);
  std::array<uint8_t, 64> line{};
  ASSERT_EQ(l2_.read(address, line.data(), line.size(), Mtype::RW), VmAccessOutcome::Complete);
  const uint64_t reads = l2_.backing_read_transactions();
  write_scalar(0, address - 64);
  write_scalar(1, 0);
  write_scalar(2, 128);
  write_scalar(5, 64);
  issue(0xf4880000, 0x0a000000); // s_atc_probe 0, s[0:1], s5
  EXPECT_EQ(l2_.backing_read_transactions(), reads);
  a = 29;
  ASSERT_EQ(l2_.write(address, reinterpret_cast<uint8_t *>(&a), sizeof(a), Mtype::RW),
            VmAccessOutcome::Complete);
  issue(0xf4000100, 0x0a000000); // s_load_b32 s4, s[0:1], s5
  EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 17u);

  cu_->l1_scalar().invalidate_all();
  issue(0xf48c0000, 0x0a000000); // s_atc_probe_buffer 0, s[0:3], s5
  a = 43;
  ASSERT_EQ(l2_.write(address, reinterpret_cast<uint8_t *>(&a), sizeof(a), Mtype::RW),
            VmAccessOutcome::Complete);
  issue(0xf4000100, 0x0a000000);
  EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 29u);

  write_scalar(5, 0x40000);
  const uint64_t before = l2_.backing_read_transactions();
  issue(0xf4880000, 0x0a000000);
  issue(0xf48c0000, 0x0a000000);
  EXPECT_EQ(l2_.backing_read_transactions(), before);
}

TEST_F(Rdna35MemoryTest, GdsSharesBytesAcrossCusAndUsesFirstLaneExceptAppend) {
  std::unique_ptr<ComputeUnitCore> peer_cu;
  ASSERT_NO_FATAL_FAILURE(make_cu(peer_cu));
  auto *peer = peer_cu->dispatch_wf(1, 0, 106, 32, 32);
  ASSERT_NE(peer, nullptr);
  wf_->set_exec(0xa);
  wf_->set_m0((64u << 16) | 8);
  peer->set_m0(wf_->m0());
  peer->set_exec(1);
  write_vector(2, 0, 1);
  write_vector(2, 4, 3);
  write_vector(3, 41, 1);
  write_vector(3, 99, 3);
  wf_->lds().write32(64, 17);
  EXPECT_FALSE(device_->has_gds_data());
  device_->access_gds(cu_->arch(), wf_->process_id(), [](Lds &gds) { gds.write32(64, 73); });
  EXPECT_TRUE(device_->has_gds_data());
  issue(gds(54), 2);
  EXPECT_EQ(read_vector(0, 1), 73u);
  issue(gds(13), 0x302);                 // ds_store_b32 v2, v3 gds
  issue_on(*peer_cu, *peer, gds(54), 2); // ds_load_b32 v0, v2 gds
  EXPECT_EQ(peer_cu->read_vgpr(peer->vgpr_alloc().base, 0), 41u);
  EXPECT_EQ(wf_->lds().read32(64), 17u);
  device_->access_gds(cu_->arch(), wf_->process_id(),
                      [](Lds &gds) { EXPECT_EQ(gds.read32(64), 41u); });
  peer_cu->write_vgpr(peer->vgpr_alloc().base + 2, 0, 4);
  issue_on(*peer_cu, *peer, gds(54), 2);
  EXPECT_EQ(peer_cu->read_vgpr(peer->vgpr_alloc().base, 0), 0u);

  write_vector(0, 0xdeadbeef, 3);
  issue(gds(54), 2);
  EXPECT_EQ(read_vector(0, 1), 41u);
  EXPECT_EQ(read_vector(0, 3), 0xdeadbeefu); // Ordinary GDS writes only its first enabled lane.
  issue(gds(62), 0); // ds_append v0 gds, counts both EXEC bits and broadcasts old value.
  EXPECT_EQ(read_vector(0, 1), 41u);
  EXPECT_EQ(read_vector(0, 3), 41u);
  write_vector(3, 1, 1);
  issue(gds(32), 0x302); // ds_add_rtn_u32 v0,v2,v3 gds
  EXPECT_EQ(read_vector(0, 1), 43u);
  wf_->set_m0((64u << 16) | 3);
  issue(gds(54), 2);
  EXPECT_EQ(read_vector(0, 1), 0u); // Only three bytes in the M0 allocation.
  wf_->set_m0((64u << 16) | 8);
  write_vector(2, 1, 1);
  issue(gds(32), 0x302, VmAccessOutcome::Faulted);

  peer->set_process_id(wf_->process_id() + 1);
  peer_cu->write_vgpr(peer->vgpr_alloc().base + 2, 0, 0);
  issue_on(*peer_cu, *peer, gds(54), 2);
  EXPECT_EQ(peer_cu->read_vgpr(peer->vgpr_alloc().base, 0), 0u);
  peer->halt();
}

TEST_F(Rdna35MemoryTest, StreamoutRegistersPreserveIssueLanesWithoutReportingGdsBytes) {
  using test::execution_plugin::MemoryObservationPlugin;
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<MemoryObservationPlugin>(true);
  auto *observer = plugin.get();
  ASSERT_TRUE(group->add(std::move(plugin)));
  cu_->set_plugin_group(group);
  wf_->set_exec(0xa);
  write_vector(3, 7, 1);
  for (auto [opcode, offset] : {std::pair{122u, 4u}, {123u, 32u}}) {
    observer->accesses.clear();
    ASSERT_NO_FATAL_FAILURE(issue(gds(opcode, offset), 0x08000300));
    ASSERT_EQ(observer->accesses.size(), 1u);
    const auto &access = observer->accesses.front();
    EXPECT_EQ(access.decoded_space, DecodedMemorySpace::GDS);
    EXPECT_EQ(access.active_lane_mask, 2u);
    EXPECT_EQ(access.architectural_exec_lane_mask, 0xau);
    EXPECT_EQ(access.wait_counter, WaitCounterType::DSCNT);
    EXPECT_EQ(access.valid_lane_mask, 0u);
    EXPECT_EQ(access.request_lane_mask, 0u);
    EXPECT_EQ(access.element_size_bytes, 0u);
    EXPECT_TRUE(access.addresses.empty());
  }
  observer->accesses.clear();
  wf_->set_m0(32);
  write_vector(2, 16, 1);
  ASSERT_NO_FATAL_FAILURE(issue(gds(54), 0x08000002)); // ds_read_b32 gds
  ASSERT_EQ(observer->accesses.size(), 1u);
  const auto &byte_access = observer->accesses.front();
  EXPECT_EQ(byte_access.request_lane_mask, 2u);
  ASSERT_GT(byte_access.addresses.size(), 1u);
  EXPECT_EQ(byte_access.addresses[1], 16u);
}

TEST_F(Rdna35MemoryTest, StreamoutRegisterAtomicsUseUnsignedSourcesAndSelectedWidth) {
  write_vector(3, UINT32_MAX);
  issue(gds(122, 32), 0x300); // ds_add_gs_reg_rtn v[0:1], v3, register 8
  EXPECT_EQ(read_vector(), 0u);
  EXPECT_EQ(read_vector(1), 0u);
  write_vector(3, 2);
  issue(gds(122, 32), 0x300);
  EXPECT_EQ(read_vector(), UINT32_MAX);
  issue(gds(123, 32), 0x300);
  EXPECT_EQ(read_vector(), 1u);
  EXPECT_EQ(read_vector(1), 1u);
  write_vector(1, 0xdeadbeef);
  issue(gds(123, 16), 0x300); // Lower bank returns a single VGPR.
  EXPECT_EQ(read_vector(), 0u);
  EXPECT_EQ(read_vector(1), 0xdeadbeefu);
  write_vector(3, 0);
  issue(gds(122, 16), 0x300);
  EXPECT_EQ(read_vector(), UINT32_MAX - 1);
  write_vector(3, 9);
  issue(gds(122, 32), 0x1f000300); // v[31:32] is outside this wave.
  issue(gds(122, 32), 0x300);
  EXPECT_EQ(read_vector(), UINT32_MAX);
  EXPECT_EQ(read_vector(1), 0u);
}

TEST_F(Rdna35MemoryTest, OrderedCountReturnsOneWordFromWideStreamoutCounters) {
  for (bool m0_selects_bank : {false, true}) {
    SCOPED_TRACE(m0_selects_bank);
    constexpr uint32_t offset = 8u * sizeof(uint32_t);
    write_vector(3, UINT32_MAX);
    issue(gds(122, offset), 0x300); // Seed the wide counter through DS_ADD_GS_REG_RTN.
    wf_->set_m0(m0_selects_bank ? offset << 16 : 0);
    write_vector(2, 2);
    write_vector(9, 0xdeadbeef);
    const std::array words{gds(63, m0_selects_bank ? 0 : offset), 0x08000002u};
    auto decoded = decoder_->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    const InstDefUse def_use(*decoded.value(), 32);
    EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, 1}));
    EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, 9, 1}));

    constexpr uint32_t wait_ds = 0xbf89fc07u; // s_waitcnt lgkmcnt(0)
    constexpr uint32_t read_v9 = 0x7e140309u; // v_mov_b32 v10, v9
    const std::array program{wait_ds, words[0], words[1], read_v9, wait_ds, read_v9};
    wf_->pc = m0_selects_bank ? 0x9000 : 0x8000;
    memory_.load_image(reinterpret_cast<const uint8_t *>(program.data()), sizeof(program), wf_->pc);
    auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
    ASSERT_TRUE(cu_->step());
    ASSERT_EQ(scoreboard.outstanding(WaitCounterKind::Ds), 0u);
    ASSERT_TRUE(cu_->step());
    ASSERT_EQ(scoreboard.outstanding(WaitCounterKind::Ds), 1u);
    EXPECT_EQ(read_vector(8), UINT32_MAX);
    const auto diagnostics = cu_->memory_wait_diagnostic_count();
    for (bool waited : {false, true}) {
      if (waited) {
        ASSERT_TRUE(cu_->step());
        ASSERT_EQ(scoreboard.outstanding(WaitCounterKind::Ds), 0u);
      }
      write_vector(10, 0);
      ASSERT_TRUE(cu_->step());
      EXPECT_EQ(read_vector(10), 0xdeadbeefu);
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), diagnostics);
    }
    write_vector(3, 0);
    issue(gds(122, offset), 0x300);
    EXPECT_EQ(read_vector(), 1u);
    EXPECT_EQ(read_vector(1), 1u); // The ordered addition carried into the high counter word.
    // Restore the same counter before selecting it through M0.
    write_vector(3, UINT32_MAX);
    issue(gds(123, offset), 0x300);
    write_vector(3, 2);
    issue(gds(123, offset), 0x300);
  }
}

TEST_F(Rdna35MemoryTest, OrderedCountScoreboardTracksOnlyTheReturnedLane) {
  const std::array words{gds(63), 0x08000002u};
  auto decoded = decoder_->decode_window(words);
  ASSERT_TRUE(decoded.succeeded());
  for (uint64_t exec : {0xau, 0u}) {
    wf_->set_exec(exec);
    const unsigned result_lane = exec ? 1 : 0;
    EXPECT_EQ(MemoryWaitScoreboard::result_lanes(*decoded.value(), *wf_), 1ull << result_lane);
    for (unsigned lane = 0; lane < 4; ++lane)
      write_vector(8, 0xdeadbeef, lane);
    issue(words[0], words[1]);
    for (unsigned lane = 0; lane < 4; ++lane)
      EXPECT_EQ(read_vector(8, lane), lane == result_lane ? 0u : 0xdeadbeefu);
  }
}

TEST_F(Rdna35MemoryTest, EmptyOrderedCountKeepsCounterAndAdvancesWalker) {
  unsigned id = 0;
  const auto ordered = [&](uint16_t opcode, uint64_t exec, uint32_t value) {
    wf_->set_m0(id++);
    wf_->set_exec(exec);
    write_vector(2, value);
    issue(gds(63, 0x310 | (opcode << 12)), 2);
    EXPECT_EQ(wf_->state(), WfState::RUNNING);
  };
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 17));
  for (const auto opcode : {0u, 1u, 3u}) {
    ASSERT_NO_FATAL_FAILURE(ordered(opcode, 0, 91));
    ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
    EXPECT_EQ(read_vector(), 17u);
  }
  ASSERT_NO_FATAL_FAILURE(ordered(1, 1, 9));
  EXPECT_EQ(read_vector(), 17u);
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
  EXPECT_EQ(read_vector(), 9u);
  wf_->set_m0(id);
  wf_->set_exec(1);
  const std::array wrap{gds(63, 0x3310), 2u};
  auto decoded = decoder_->decode_window(wrap);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
  EXPECT_EQ(wf_->instruction_execution_error(), InstructionExecutionError::UnsupportedOperandValue);
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
  EXPECT_EQ(read_vector(), 9u);
}

TEST_F(Rdna35MemoryTest, DebuggerTreatsAtcProbesAsNonfaultingHints) {
  constexpr uint32_t kernel = 0x1000, readable = 0x40000, hole = 0x50000;
  class ReadOnlyData final : public AddressSpaceTranslator {
    IdentityAddressSpaceTranslator identity;
    VmTranslationResult translate(uint64_t address, size_t size,
                                  VmAccessKind access) const override {
      if (address >= 0x50000 || (address >= 0x40000 && access == VmAccessKind::Write))
        return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      return identity.translate(address, size, access);
    }
  };
  GpuVm vm;
  const auto handle = vm.register_address_space(1, std::make_shared<ReadOnlyData>(),
                                                std::make_shared<GpuMemoryPhysicalAccess>(memory_));
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  const auto snapshot = wf_->snapshot_vm_access();
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->query_access(readable, 4, VmAccessKind::Write), VmAccessOutcome::Faulted);
  write_scalar(1, 0);
  write_scalar(2, 64);
  write_scalar(3, 0);
  write_scalar(5, 0);
  memory_.write32(readable, 42);
  const std::array code{0xf4880000u, 0x0a000000u,  // s_atc_probe
                        0xf48c0000u, 0x0a000000u,  // s_atc_probe_buffer
                        0xf4000100u, 0x0a000000u}; // s_load_b32 s4
  memory_.load_image(reinterpret_cast<const uint8_t *>(code.data()), sizeof(code), kernel);
  std::vector<std::pair<uint64_t, bool>> faults;
  std::vector<uint64_t> watches;
  cu_->set_debug_active(true);
  cu_->set_memory_violation_handler([&](Wavefront &, uint64_t addr, bool write, uint64_t) {
    faults.emplace_back(addr, write);
    return true;
  });
  cu_->set_watchpoint_handler([&](Wavefront &, uint64_t addr, uint32_t, bool, bool) {
    watches.push_back(addr);
    return false;
  });
  for (const uint32_t address : {readable, hole}) {
    write_scalar(0, address);
    for (size_t i = 0; i < code.size(); i += 2) {
      SCOPED_TRACE(testing::Message() << address << "/" << i);
      faults.clear();
      watches.clear();
      wf_->pc = kernel + 4 * i;
      ASSERT_TRUE(cu_->step());
      const bool load = i == 4;
      EXPECT_EQ(faults.size(), load && address == hole ? 1u : 0u);
      if (!faults.empty()) {
        EXPECT_EQ(faults.front(), std::make_pair(uint64_t{address}, false));
      }
      EXPECT_EQ(watches.size(), load && address == readable ? 1u : 0u);
      if (!watches.empty()) {
        EXPECT_EQ(watches.front(), address);
      }
      if (load && address == readable) {
        EXPECT_EQ(cu_->read_sgpr(wf_->sgpr_alloc().base + 4), 42u);
      }
    }
  }
  cu_->set_debug_active(false);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, OrderedCountDefersYoungerWaveAndCommitsOnce) {
  std::unique_ptr<ComputeUnitCore> peer_cu;
  ASSERT_NO_FATAL_FAILURE(make_cu(peer_cu));
  auto *older = peer_cu->dispatch_wf(1, 0, 106, 32, 32);
  ASSERT_NE(older, nullptr);
  older->set_exec(0);
  older->set_m0(0);
  wf_->set_m0(1); // Logical wave 1 arrives before logical wave 0.
  write_vector(2, 7);
  write_vector(0, 0xdeadbeef);
  wf_->set_exec(0xf);
  write_vector(2, 99, 1);
  write_vector(0, 0xcafefeed, 1);
  issue(gds(63, 0xc310), 2); // GFX11 ignores the old GFX10 DWORD-count bits.
  EXPECT_EQ(wf_->state(), WfState::VM_RETRY);
  EXPECT_EQ(read_vector(), 0xdeadbeefu);
  peer_cu->write_vgpr(older->vgpr_alloc().base + 2, 0, 99);
  issue_on(*peer_cu, *older, gds(63, 0x310), 2); // Empty EXEC adds zero and advances.
  test::ComputeUnitTestAccess::tick_memory_pipelines(*cu_);
  EXPECT_EQ(wf_->state(), WfState::RUNNING);
  EXPECT_EQ(read_vector(), 0u);
  EXPECT_EQ(read_vector(0, 1), 0xcafefeedu);
  write_vector(3, 0);
  issue(gds(122, 16), 0x300);
  EXPECT_EQ(read_vector(), 7u);

  wf_->set_m0(3); // Wave 2 never arrives: cancel 3 while its write is pending.
  issue(gds(63, 0x310), 2);
  EXPECT_EQ(wf_->state(), WfState::VM_RETRY);
  const uint32_t pid = wf_->process_id();
  const uint32_t qid = wf_->queue_id();
  const uint32_t dispatch = wf_->dispatch_id();
  cu_->abort_dispatch(dispatch);
  wf_ = nullptr;
  device_->retire_ordered_dispatch(pid, qid, dispatch);
  test::ComputeUnitTestAccess::tick_memory_pipelines(*cu_);
  older->set_exec(1);
  older->set_m0(0);
  peer_cu->write_vgpr(older->vgpr_alloc().base + 2, 0, 9);
  issue_on(*peer_cu, *older, gds(63, 0x1310), 2); // Exchange after counter-walker reset.
  EXPECT_EQ(older->state(), WfState::RUNNING);
  EXPECT_EQ(peer_cu->read_vgpr(older->vgpr_alloc().base, 0), 7u);
  older->halt();
  device_->release_process(pid);
  auto *replacement = peer_cu->dispatch_wf(1, 0, 106, 32, 32);
  ASSERT_NE(replacement, nullptr);
  replacement->set_exec(1);
  peer_cu->write_vgpr(replacement->vgpr_alloc().base + 3, 0, 0);
  issue_on(*peer_cu, *replacement, gds(122, 16), 0x300);
  EXPECT_EQ(peer_cu->read_vgpr(replacement->vgpr_alloc().base, 0), 0u);
  replacement->halt();
}

} // namespace
} // namespace rocjitsu
