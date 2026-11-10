// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <array>

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
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(unit, "rdna35", cfg, &memory, &l2));
    ASSERT_TRUE(unit->set_gws_device(device));
  }

  void SetUp() override {
    decoder = Decoder::create(architecture());
    ASSERT_NE(decoder, nullptr);
    l2.set_backing_memory(&memory);
    ASSERT_NO_FATAL_FAILURE(make_cu(cu));
    cu->set_apertures(kShared, kShared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
    wf = cu->dispatch_wf(0, 0, 106, 32, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_apertures(kShared, kShared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
    wf->set_lds_base(1024);
    wf->set_lds_size(1024);
    wf->set_scratch_base(0x20000);
    wf->set_scratch_lane_size(256);
    wf->set_exec(1);
  }

  void TearDown() override {
    if (wf)
      wf->halt();
  }

  void issue(uint32_t word0, uint32_t word1, VmAccessOutcome expected = VmAccessOutcome::Complete) {
    issue_on(*cu, *wf, word0, word1, expected);
  }

  void issue_on(ComputeUnitCore &unit, Wavefront &wave, uint32_t word0, uint32_t word1,
                VmAccessOutcome expected = VmAccessOutcome::Complete) {
    const std::array words{word0, word1};
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(unit.execute_instruction(inst.get(), wave).succeeded()) << inst->mnemonic();
    ASSERT_NE(inst->data(), nullptr);
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(unit, inst.release(), wave), expected);
  }

  void vector(unsigned reg, uint32_t value, unsigned lane = 0) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t result(unsigned reg = 0, unsigned lane = 0) {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }
  void scalar(unsigned reg, uint32_t value) { cu->write_sgpr(wf->sgpr_alloc().base + reg, value); }

  static constexpr uint32_t gds(uint32_t opcode, uint16_t offsets = 0) {
    return 0xd8020000u | (opcode << 18) | offsets;
  }

  std::shared_ptr<GwsDevice> device = std::make_shared<GwsDevice>();
  GpuMemory memory{"rdna35_memory"};
  L2Cache l2{"rdna35_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  Wavefront *wf = nullptr;
};

class Gfx11MemoryTest : public Rdna35MemoryTest,
                        public testing::WithParamInterface<rj_code_arch_t> {
  rj_code_arch_t architecture() const override { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11MemoryTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11MemoryTest, ScratchAndFlatPrivateShareDwordInterleaving) {
  wf->set_exec(2);
  memory.write32(0x20000 + 132, 0x12345678);
  memory.write32(0x20000 + 260, 0xdeadbeef);
  vector(2, 4, 1);
  issue(0xdc510000, 0x00fc0002); // scratch_load_b32 v0, v2, off
  EXPECT_EQ(result(0, 1), 0x12345678u);
  vector(3, kPrivate >> 32, 1);
  issue(0xdc500000, 0x007c0002); // flat_load_b32 v0, v[2:3]
  EXPECT_EQ(result(0, 1), 0x12345678u);

  // Adjacent lanes own adjacent DWORDs; a lane's next DWORD is a wave stride away.
  constexpr uint64_t pair_base = 0x25000;
  wf->set_scratch_base(pair_base);
  wf->set_exec(3);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    vector(2, 0, lane);
    vector(3, kPrivate >> 32, lane);
    memory.write32(pair_base + lane * 4, 10 + lane);
    memory.write32(pair_base + 32 * 4 + lane * 4, 20 + lane);
  }
  issue(0xdc540000, 0x007c0002); // flat_load_b64 v[0:1], v[2:3]
  for (uint32_t lane = 0; lane < 2; ++lane) {
    EXPECT_EQ(result(0, lane), 10u + lane);
    EXPECT_EQ(result(1, lane), 20u + lane);
  }

  wf->set_exec(3);
  vector(2, 0x40000);
  vector(3, 0);
  vector(4, 1);
  vector(5, 0);
  vector(2, 0, 1);
  memory.write32(0x40000, 41);
  issue(0xdd0c4000, 0x007c0402, VmAccessOutcome::Faulted); // flat_atomic_add_u64 glc
  EXPECT_EQ(memory.read32(0x40000), 41u); // The global lane must also remain untouched.
}

TEST_P(Gfx11MemoryTest, LdsAllocationRoundsEachWorkgroupInKiBUnits) {
  EXPECT_EQ(cu->allocate_lds(1), 0u);
  EXPECT_EQ(cu->allocate_lds(513), 1024u);
  EXPECT_EQ(cu->allocate_lds(1025), 2048u);
  EXPECT_EQ(cu->allocate_lds(1), 4096u);
}

TEST_P(Gfx11MemoryTest, DsReadsStopAtTheWindowInsideTheReservedKiB) {
  wf->set_lds_size(lds_workgroup_window_bytes(wf->cu().arch(), 4));
  for (uint32_t offset : {508u, 512u, 1020u}) {
    SCOPED_TRACE(offset);
    wf->lds().write32(wf->lds_base() + offset, 0x12345678);
    vector(2, offset);
    issue(0xd8d80000, 2); // ds_load_b32 v0, v2
    EXPECT_EQ(result(), offset < 512 ? 0x12345678u : 0u);
  }
}

TEST_F(Rdna35MemoryTest, GuardLdsAllocationAndReturningAtomicDestinations) {
  wf->lds().write32(2048, 41);
  vector(2, 1024);
  issue(0xd8d80000, 2); // ds_load_b32 v0, v2
  EXPECT_EQ(result(), 0u);
  vector(3, kShared >> 32);
  issue(0xdc500000, 0x007c0002);
  EXPECT_EQ(result(), 0u);
  wf->lds().write32(1032, 41);
  vector(2, 8);
  vector(3, 1);
  issue(0xd8800000, 0x20000302); // ds_add_rtn_u32 v32, v2, v3
  EXPECT_EQ(wf->lds().read32(1032), 41u);
  memory.write32(0x40000, 41);
  vector(2, 0x40000);
  vector(3, 0);
  vector(4, 1);
  issue(0xdcd64000, 0x207c0402); // global_atomic_add_u32 v32, v[2:3], v4, off glc
  EXPECT_EQ(memory.read32(0x40000), 41u);
  wf->set_exec(3);
  vector(2, 1, 1);
  vector(3, kShared >> 32, 1);
  vector(4, 7, 1);
  issue(0xdcd44000, 0x007c0402, VmAccessOutcome::Faulted); // flat_atomic_add_u32 glc
  EXPECT_EQ(memory.read32(0x40000), 41u); // Preflight the mixed operation before any RMW.
}

TEST_P(Gfx11MemoryTest, DistinguishUnboundBufferFromWrongResourceTypeAndMissingStatusVgpr) {
  scalar(0, 0x40000);
  scalar(1, 4u << 16);
  scalar(2, 4);
  scalar(3, 2u << 28); // FORMAT=0, ADD_TID=0
  memory.write32(0x40000, 0x11223344);
  vector(0, 0xdeadbeef);
  issue(0xe0500000, 0x80000000); // buffer_load_b32 v0, off, s[0:3], 0
  EXPECT_EQ(result(), 0u);
  // ADD_TID indexes lane 1 only when the descriptor is bound.
  constexpr uint32_t kBoundFormatWithThreadId = (2u << 28) | (1u << 23) | (20u << 12);
  wf->set_exec(2);
  for (bool bound : {false, true}) {
    scalar(3, bound ? kBoundFormatWithThreadId : (2u << 28) | (1u << 23));
    memory.write32(0x40004, 0x11223345);
    vector(0, 0xdeadbeef, 1);
    issue(0xe0500000, 0x80000000); // buffer_load_b32
    EXPECT_EQ(result(0, 1), bound ? 0x11223345u : 0u);
    vector(0, 0x76543210, 1);
    issue(0xe0680000, 0x80000000); // buffer_store_b32
    EXPECT_EQ(memory.read32(0x40004), bound ? 0x76543210u : 0x11223345u);
    EXPECT_EQ(memory.read32(0x40000), 0x11223344u);
  }
  wf->set_exec(1);
  scalar(3, (8u << 28) | (20u << 12)); // 1D image descriptor type
  vector(0, 0xdeadbeef);
  issue(0xe0500000, 0x80000000);
  EXPECT_EQ(result(), 0xdeadbeefu);
  scalar(3, (2u << 28) | (20u << 12));
  vector(31, 0xdeadbeef);
  issue(0xe0500000, 0x80201f00); // buffer_load_b32 v[31:32] ... tfe
  EXPECT_EQ(result(31), 0xdeadbeefu);
  issue(0xe0500000, 0x80000000);
  EXPECT_EQ(result(), 0x11223344u);
}

class ResidentBelowBufferBoundary final : public AddressSpaceTranslator {
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
  const auto handle = vm.register_address_space(1, std::make_shared<ResidentBelowBufferBoundary>(),
                                                std::make_shared<GpuMemoryPhysicalAccess>(memory));
  ASSERT_TRUE(handle);
  cu->set_gpu_vm(&vm);
  wf->set_address_space(handle);
  scalar(1, 0);
  scalar(2, 32);
  scalar(3, (2u << 28) | (20u << 12));
  vector(4, 0);
  memory.write32(limit - 8, 11);
  memory.write32(limit - 4, 17);
  memory.write32(kernel, 0xe0540000); // buffer_load_b64 v[0:2],v4,s[0:3],0 offen tfe
  memory.write32(kernel + 4, 0x80600004);
  memory.write32(kernel + 8, 0xe0540000); // The same load without TFE.
  memory.write32(kernel + 12, 0x80400004);
  std::vector<std::pair<uint64_t, uint32_t>> watches;
  unsigned faults = 0;
  cu->set_debug_active(true);
  cu->set_watchpoint_handler([&](Wavefront &, uint64_t addr, uint32_t bytes, bool, bool) {
    watches.emplace_back(addr, bytes);
    return false;
  });
  cu->set_memory_violation_handler([&](Wavefront &, uint64_t, bool, uint64_t) {
    ++faults;
    return true;
  });
  for (bool tfe : {false, true}) {
    for (unsigned resident : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << tfe << "/" << resident);
      const uint32_t address = limit - 4 * resident;
      scalar(0, address);
      for (unsigned reg = 0; reg < 3; ++reg)
        vector(reg, sentinel);
      wf->pc = tfe ? kernel : kernel + 8;
      watches.clear();
      faults = 0;
      ASSERT_TRUE(cu->step());
      const bool claimed = !tfe && resident < 2;
      EXPECT_EQ(faults, claimed ? 1u : 0u);
      EXPECT_EQ(result(), claimed ? sentinel : resident ? memory.read32(address) : 0u);
      EXPECT_EQ(result(1), claimed ? sentinel : resident == 2 ? 17u : 0u);
      EXPECT_EQ(result(2), tfe && resident != 2 ? 1u : sentinel);
      uint32_t watched_bytes = 0;
      for (const auto &[watch_address, watch_bytes] : watches) {
        EXPECT_GE(watch_address, address);
        EXPECT_LE(watch_address + watch_bytes, limit);
        watched_bytes += watch_bytes;
      }
      EXPECT_EQ(watched_bytes, claimed ? 0u : resident * 4u);
    }
  }
  cu->set_debug_active(false);
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, BufferTfePreservesResidentDwordsAndReportsMissingPages) {
  GpuVm vm;
  unsigned faults = 0;
  const auto handle = vm.register_address_space(1, std::make_shared<ResidentBelowBufferBoundary>(),
                                                std::make_shared<GpuMemoryPhysicalAccess>(memory),
                                                [&](uint64_t, VmAccessKind) { ++faults; });
  ASSERT_TRUE(handle);
  cu->set_gpu_vm(&vm);
  wf->set_address_space(handle);
  wf->set_exec(3);
  scalar(0, 0x4fff8);
  scalar(1, 0);
  scalar(2, 32);
  scalar(3, (2u << 28) | (20u << 12));
  memory.write32(0x4fff8, 11);
  memory.write32(0x4fffc, 17);
  vector(4, 0);
  vector(4, 4, 1);
  vector(2, 0xdeadbeef);
  vector(2, 0xdeadbeef, 1);
  issue(0xe0540000, 0x80600004); // buffer_load_b64 v[0:2],v4,s[0:3],0 offen tfe
  EXPECT_EQ(result(), 11u);
  EXPECT_EQ(result(1), 17u);
  EXPECT_EQ(result(2), 0xdeadbeefu);
  EXPECT_EQ(result(0, 1), 17u);
  EXPECT_EQ(result(1, 1), 0u);
  EXPECT_EQ(result(2, 1), 1u);
  scalar(0, 0x50000);
  issue(0xf4880000, 0x0a000000); // Unmapped ATC probe drops without a VM fault.
  EXPECT_EQ(faults, 0u);
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, AlignScalarAddressComponentsAndFaultNegativeBufferImmediate) {
  scalar(0, 0x60001);
  scalar(1, 0);
  scalar(2, 32);
  scalar(5, 0);
  memory.write32(0x60000, 0x66666666);
  memory.write32(0x60004, 0x77777777);
  issue(0xf4000100, 0x0a000003); // s_load_b32 s4, s[0:1], s5 offset:3
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 0x66666666u);
  issue(0xf4200100, 0x0a000003); // s_buffer_load_b32 s4, s[0:3], s5 offset:3
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 0x66666666u);
  scalar(4, 0xdeadbeef);
  scalar(5, 4);
  issue(0xf4200100, 0x0a1ffffc, VmAccessOutcome::Faulted); // imm=-4, combined offset=0
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 0xdeadbeefu);
}

TEST_F(Rdna35MemoryTest, FlatIgnoresOffsetBitTwelve) {
  vector(2, 0x70000);
  vector(3, 0);
  memory.write32(0x70000, 41);
  memory.write32(0x6f000, 42);
  issue(0xdc501000, 0x007c0002); // flat_load_b32 v0, v[2:3], raw offset 0x1000
  EXPECT_EQ(result(), 41u);
}
TEST_F(Rdna35MemoryTest, AtcProbesWarmResidentScalarLinesWithoutReadingBackingOnMiss) {
  constexpr uint64_t address = 0x80040;
  uint32_t a = 17;
  memory.write32(address, a);
  std::array<uint8_t, 64> line{};
  ASSERT_EQ(l2.read(address, line.data(), line.size(), Mtype::RW), VmAccessOutcome::Complete);
  const uint64_t reads = l2.backing_read_transactions();
  scalar(0, address - 64);
  scalar(1, 0);
  scalar(2, 128);
  scalar(5, 64);
  issue(0xf4880000, 0x0a000000); // s_atc_probe 0, s[0:1], s5
  EXPECT_EQ(l2.backing_read_transactions(), reads);
  a = 29;
  ASSERT_EQ(l2.write(address, reinterpret_cast<uint8_t *>(&a), sizeof(a), Mtype::RW),
            VmAccessOutcome::Complete);
  issue(0xf4000100, 0x0a000000); // s_load_b32 s4, s[0:1], s5
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 17u);

  cu->l1_scalar().invalidate_all();
  issue(0xf48c0000, 0x0a000000); // s_atc_probe_buffer 0, s[0:3], s5
  a = 43;
  ASSERT_EQ(l2.write(address, reinterpret_cast<uint8_t *>(&a), sizeof(a), Mtype::RW),
            VmAccessOutcome::Complete);
  issue(0xf4000100, 0x0a000000);
  EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 29u);

  scalar(5, 0x40000);
  const uint64_t before = l2.backing_read_transactions();
  issue(0xf4880000, 0x0a000000);
  issue(0xf48c0000, 0x0a000000);
  EXPECT_EQ(l2.backing_read_transactions(), before);
}

TEST_F(Rdna35MemoryTest, GdsSharesBytesAcrossCusAndUsesFirstLaneExceptAppend) {
  std::unique_ptr<ComputeUnitCore> peer_cu;
  ASSERT_NO_FATAL_FAILURE(make_cu(peer_cu));
  auto *peer = peer_cu->dispatch_wf(1, 0, 106, 32, 32);
  ASSERT_NE(peer, nullptr);
  wf->set_exec(0xa);
  wf->set_m0((64u << 16) | 8);
  peer->set_m0(wf->m0());
  peer->set_exec(1);
  vector(2, 0, 1);
  vector(2, 4, 3);
  vector(3, 41, 1);
  vector(3, 99, 3);
  wf->lds().write32(64, 17);
  issue(gds(13), 0x302);                 // ds_store_b32 v2, v3 gds
  issue_on(*peer_cu, *peer, gds(54), 2); // ds_load_b32 v0, v2 gds
  EXPECT_EQ(peer_cu->read_vgpr(peer->vgpr_alloc().base, 0), 41u);
  EXPECT_EQ(wf->lds().read32(64), 17u);
  peer_cu->write_vgpr(peer->vgpr_alloc().base + 2, 0, 4);
  issue_on(*peer_cu, *peer, gds(54), 2);
  EXPECT_EQ(peer_cu->read_vgpr(peer->vgpr_alloc().base, 0), 0u);

  vector(0, 0xdeadbeef, 3);
  issue(gds(54), 2);
  EXPECT_EQ(result(0, 1), 41u);
  EXPECT_EQ(result(0, 3), 0xdeadbeefu); // Ordinary GDS writes only its first enabled lane.
  issue(gds(62), 0); // ds_append v0 gds, counts both EXEC bits and broadcasts old value.
  EXPECT_EQ(result(0, 1), 41u);
  EXPECT_EQ(result(0, 3), 41u);
  vector(3, 1, 1);
  issue(gds(32), 0x302); // ds_add_rtn_u32 v0,v2,v3 gds
  EXPECT_EQ(result(0, 1), 43u);
  wf->set_m0((64u << 16) | 3);
  issue(gds(54), 2);
  EXPECT_EQ(result(0, 1), 0u); // Only three bytes in the M0 allocation.
  wf->set_m0((64u << 16) | 8);
  vector(2, 1, 1);
  issue(gds(32), 0x302, VmAccessOutcome::Faulted);

  peer->set_process_id(wf->process_id() + 1);
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
  cu->set_plugin_group(group);
  wf->set_exec(0xa);
  vector(3, 7, 1);
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
  wf->set_m0(32);
  vector(2, 16, 1);
  ASSERT_NO_FATAL_FAILURE(issue(gds(54), 0x08000002)); // ds_read_b32 gds
  ASSERT_EQ(observer->accesses.size(), 1u);
  const auto &byte_access = observer->accesses.front();
  EXPECT_EQ(byte_access.request_lane_mask, 2u);
  ASSERT_GT(byte_access.addresses.size(), 1u);
  EXPECT_EQ(byte_access.addresses[1], 16u);
}

TEST_F(Rdna35MemoryTest, StreamoutRegisterAtomicsUseUnsignedSourcesAndSelectedWidth) {
  vector(3, UINT32_MAX);
  issue(gds(122, 32), 0x300); // ds_add_gs_reg_rtn v[0:1], v3, register 8
  EXPECT_EQ(result(), 0u);
  EXPECT_EQ(result(1), 0u);
  vector(3, 2);
  issue(gds(122, 32), 0x300);
  EXPECT_EQ(result(), UINT32_MAX);
  issue(gds(123, 32), 0x300);
  EXPECT_EQ(result(), 1u);
  EXPECT_EQ(result(1), 1u);
  vector(1, 0xdeadbeef);
  issue(gds(123, 16), 0x300); // Lower bank returns a single VGPR.
  EXPECT_EQ(result(), 0u);
  EXPECT_EQ(result(1), 0xdeadbeefu);
  vector(3, 0);
  issue(gds(122, 16), 0x300);
  EXPECT_EQ(result(), UINT32_MAX - 1);
  vector(3, 9);
  issue(gds(122, 32), 0x1f000300); // v[31:32] is outside this wave.
  issue(gds(122, 32), 0x300);
  EXPECT_EQ(result(), UINT32_MAX);
  EXPECT_EQ(result(1), 0u);
}

TEST_F(Rdna35MemoryTest, EmptyOrderedCountKeepsCounterAndAdvancesWalker) {
  unsigned id = 0;
  const auto ordered = [&](uint16_t opcode, uint64_t exec, uint32_t value) {
    wf->set_m0(id++);
    wf->set_exec(exec);
    vector(2, value);
    issue(gds(63, 0x310 | (opcode << 12)), 2);
    EXPECT_EQ(wf->state(), WfState::RUNNING);
  };
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 17));
  for (const auto opcode : {0u, 1u, 3u}) {
    ASSERT_NO_FATAL_FAILURE(ordered(opcode, 0, 91));
    ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
    EXPECT_EQ(result(), 17u);
  }
  ASSERT_NO_FATAL_FAILURE(ordered(1, 1, 9));
  EXPECT_EQ(result(), 17u);
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
  EXPECT_EQ(result(), 9u);
  wf->set_m0(id);
  wf->set_exec(1);
  const std::array wrap{gds(63, 0x3310), 2u};
  auto decoded = decoder->decode_window(wrap);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu->execute_instruction(decoded.value().get(), *wf).failed());
  EXPECT_EQ(wf->instruction_execution_error(), InstructionExecutionError::UnsupportedOperandValue);
  ASSERT_NO_FATAL_FAILURE(ordered(0, 1, 0));
  EXPECT_EQ(result(), 9u);
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
                                                std::make_shared<GpuMemoryPhysicalAccess>(memory));
  ASSERT_TRUE(handle);
  cu->set_gpu_vm(&vm);
  wf->set_address_space(handle);
  const auto snapshot = wf->snapshot_vm_access();
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->query_access(readable, 4, VmAccessKind::Write), VmAccessOutcome::Faulted);
  scalar(1, 0);
  scalar(2, 64);
  scalar(3, 0);
  scalar(5, 0);
  memory.write32(readable, 42);
  const std::array code{0xf4880000u, 0x0a000000u,  // s_atc_probe
                        0xf48c0000u, 0x0a000000u,  // s_atc_probe_buffer
                        0xf4000100u, 0x0a000000u}; // s_load_b32 s4
  memory.load_image(reinterpret_cast<const uint8_t *>(code.data()), sizeof(code), kernel);
  std::vector<std::pair<uint64_t, bool>> faults;
  std::vector<uint64_t> watches;
  cu->set_debug_active(true);
  cu->set_memory_violation_handler([&](Wavefront &, uint64_t addr, bool write, uint64_t) {
    faults.emplace_back(addr, write);
    return true;
  });
  cu->set_watchpoint_handler([&](Wavefront &, uint64_t addr, uint32_t, bool, bool) {
    watches.push_back(addr);
    return false;
  });
  for (const uint32_t address : {readable, hole}) {
    scalar(0, address);
    for (size_t i = 0; i < code.size(); i += 2) {
      SCOPED_TRACE(testing::Message() << address << "/" << i);
      faults.clear();
      watches.clear();
      wf->pc = kernel + 4 * i;
      ASSERT_TRUE(cu->step());
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
        EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4), 42u);
      }
    }
  }
  cu->set_debug_active(false);
  wf->set_address_space({});
  cu->set_gpu_vm(nullptr);
}

TEST_F(Rdna35MemoryTest, OrderedCountDefersYoungerWaveAndCommitsOnce) {
  std::unique_ptr<ComputeUnitCore> peer_cu;
  ASSERT_NO_FATAL_FAILURE(make_cu(peer_cu));
  auto *older = peer_cu->dispatch_wf(1, 0, 106, 32, 32);
  ASSERT_NE(older, nullptr);
  older->set_exec(0);
  older->set_m0(0);
  wf->set_m0(1); // Logical wave 1 arrives before logical wave 0.
  vector(2, 7);
  vector(0, 0xdeadbeef);
  wf->set_exec(0xf);
  vector(2, 99, 1);
  vector(0, 0xcafefeed, 1);
  issue(gds(63, 0xc310), 2); // GFX11 ignores the old GFX10 DWORD-count bits.
  EXPECT_EQ(wf->state(), WfState::VM_RETRY);
  EXPECT_EQ(result(), 0xdeadbeefu);
  peer_cu->write_vgpr(older->vgpr_alloc().base + 2, 0, 99);
  issue_on(*peer_cu, *older, gds(63, 0x310), 2); // Empty EXEC adds zero and advances.
  test::ComputeUnitTestAccess::tick_memory_pipelines(*cu);
  EXPECT_EQ(wf->state(), WfState::RUNNING);
  EXPECT_EQ(result(), 0u);
  EXPECT_EQ(result(0, 1), 0xcafefeedu);
  vector(3, 0);
  issue(gds(122, 16), 0x300);
  EXPECT_EQ(result(), 7u);

  wf->set_m0(3); // Wave 2 never arrives: cancel 3 while its write is pending.
  issue(gds(63, 0x310), 2);
  EXPECT_EQ(wf->state(), WfState::VM_RETRY);
  const uint32_t pid = wf->process_id();
  const uint32_t qid = wf->queue_id();
  const uint32_t dispatch = wf->dispatch_id();
  cu->abort_dispatch(dispatch);
  wf = nullptr;
  device->retire_ordered_dispatch(pid, qid, dispatch);
  test::ComputeUnitTestAccess::tick_memory_pipelines(*cu);
  older->set_exec(1);
  older->set_m0(0);
  peer_cu->write_vgpr(older->vgpr_alloc().base + 2, 0, 9);
  issue_on(*peer_cu, *older, gds(63, 0x1310), 2); // Exchange after counter-walker reset.
  EXPECT_EQ(older->state(), WfState::RUNNING);
  EXPECT_EQ(peer_cu->read_vgpr(older->vgpr_alloc().base, 0), 7u);
  older->halt();
  device->release_process(pid);
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
