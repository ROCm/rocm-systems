// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

namespace rocjitsu {
namespace {
using namespace amdgpu;

class Rdna4MemoryTest : public testing::Test {
protected:
  static constexpr uint64_t kPrivate = 0x1000000000000ull;
  GpuMemory memory{"memory"};
  L2Cache l2{"l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  Wavefront *wf = nullptr;
  void SetUp() override {
    l2.set_backing_memory(&memory);
    ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 64;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "cu", cfg, &memory, &l2));
    wf = cu->dispatch_wf(0, 0, 106, 32, 32);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    wf->set_lds_size(1024);
    wf->set_scratch_base(0x20000);
    wf->set_scratch_lane_size(256);
    const auto shared = kPrivate + (1ull << 32);
    cu->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
    wf->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
  }
  void TearDown() override {
    if (wf)
      wf->halt();
  }
  void set(unsigned reg, uint32_t word, unsigned lane = 0) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, word);
  }
  uint32_t get(unsigned reg = 0, unsigned lane = 0) {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }
  void issue(std::array<uint32_t, 3> words, VmAccessOutcome expected = VmAccessOutcome::Complete) {
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded()) << inst->mnemonic();
    ASSERT_NE(inst->data(), nullptr);
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wf), expected);
  }
};

TEST_F(Rdna4MemoryTest, OrderedAddMatchesIdUpdatesTwoWordsAndReturnsOldPair) {
  // gfx1201 global_atomic_ordered_add_b64 v[0:1], v[2:3], v[4:5], off th:TH_ATOMIC_RETURN
  constexpr std::array words{0xee1cc07cu, 0x02100000u, 0x00000002u};
  set(2, 0x10000);
  set(3, 0);
  memory.write32(0x10000, 4095);
  memory.write32(0x10004, UINT32_MAX);
  set(4, 3);
  set(5, 2);
  issue(words);
  EXPECT_EQ(get(0), 4095u);
  EXPECT_EQ(get(1), UINT32_MAX);
  set(4, 4095);
  issue(words);
  EXPECT_EQ(get(0), 4095u);
  EXPECT_EQ(get(1), UINT32_MAX);
  issue({0xee05407c, 0, 2}); // global_load_b64
  EXPECT_EQ(get(0), 0u);
  EXPECT_EQ(get(1), 1u);
}

TEST_F(Rdna4MemoryTest, MisalignedFlatAtomicPreservesSharedAndGlobalLanes) {
  constexpr uint64_t shared = kPrivate + (uint64_t{1} << 32), global = 0x50000;
  constexpr uint32_t old = 0x11223344, sentinel = 0xdeadbeef;
  wf->set_exec(3);
  for (unsigned offset : {4u, 5u}) {
    const bool aligned = offset == 4;
    wf->lds().write32(4, old);
    wf->lds().write32(8, 0x55667788);
    memory.write32(global, 37);
    for (unsigned lane = 0; lane < 2; ++lane) {
      const uint64_t addr = lane == 0 ? shared + offset : global;
      set(0, sentinel, lane);
      set(2, uint32_t(addr), lane);
      set(3, addr >> 32, lane);
      set(4, lane + 1, lane);
    }
    issue({0xec0d407c, 0x02100000, 0x00000002},
          aligned ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted);
    EXPECT_EQ(wf->lds().read32(4), aligned ? old + 1 : old);
    EXPECT_EQ(memory.read32(global), aligned ? 39u : 37u);
    EXPECT_EQ(get(0, 0), aligned ? old : sentinel);
    EXPECT_EQ(get(0, 1), aligned ? 37u : sentinel);
  }
}

TEST_F(Rdna4MemoryTest, FlatSelectsItsSegmentBeforeAddingTheImmediate) {
  constexpr uint64_t shared = kPrivate + (uint64_t{2} << 32);
  cu->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
  wf->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
  memory.write32(shared, 0x11112222);
  memory.write32(kPrivate, 0x33334444);
  memory.write32(wf->scratch_base(), 0x55556666);
  wf->lds().write32(0, 0x77778888);
  const std::array words{0xec05007cu, 0x00000000u, 0x00000402u};
  for (const auto &[pointer, lds, expected] : {std::tuple{shared - 4, false, 0x11112222u},
                                               {kPrivate - 4, false, 0x33334444u},
                                               {kPrivate + 0xfffffffc, false, 0x55556666u},
                                               {shared + 0xfffffffc, true, 0x77778888u}}) {
    set(2, static_cast<uint32_t>(pointer));
    set(3, static_cast<uint32_t>(pointer >> 32));
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    const auto lanes = amdgpu::MemoryWaitScoreboard::flat_lanes(*decoded.value(), *wf, shared,
                                                                shared + UINT32_MAX);
    EXPECT_EQ(lanes.shared, lds ? 1u : 0u);
    ASSERT_NO_FATAL_FAILURE(issue(words));
    EXPECT_EQ(get(0), expected);
  }
}

TEST_F(Rdna4MemoryTest, PrivateScratchInterleavesDwordsAndRejectsMixedB64Atomic) {
  wf->set_exec(2);
  memory.write32(0x20000 + 132, 0x12345678);
  memory.write32(0x20000 + 260, 0xdeadbeef);
  set(2, 4, 1);
  issue({0xed05007c, 0x00020000, 2}); // scratch_load_b32
  EXPECT_EQ(get(0, 1), 0x12345678u);
  set(3, kPrivate >> 32, 1);
  issue({0xec05007c, 0, 2}); // flat_load_b32
  EXPECT_EQ(get(0, 1), 0x12345678u);
  wf->set_exec(3);
  set(2, 0x30000);
  set(3, 0);
  set(4, 1);
  set(5, 0);
  set(2, 0, 1);
  memory.write32(0x30000, 41);
  issue({0xec10c07c, 0x02100000, 2}, VmAccessOutcome::Faulted); // flat_atomic_add_u64
  EXPECT_EQ(memory.read32(0x30000), 41u);
}

TEST_F(Rdna4MemoryTest, AllocatesLdsInKibBlocks) {
  EXPECT_EQ(cu->allocate_lds(1), 0u);
  EXPECT_EQ(cu->allocate_lds(512), 1024u);
  EXPECT_EQ(cu->allocate_lds(1025), 2048u);
  EXPECT_EQ(cu->allocate_lds(1), 4096u);
}

TEST_F(Rdna4MemoryTest, GuardLdsRangeAndReturningAtomicResult) {
  wf->set_lds_base(1024);
  wf->lds().write32(2048, 41);
  set(2, 1024);
  issue({0xd8d80000, 2}); // ds_load_b32
  EXPECT_EQ(get(), 0u);
  wf->lds().write32(1028, 41);
  set(2, 4);
  set(3, 1);
  issue({0xd8800000, 0x20000302}); // ds_add_rtn_u32 v32
  EXPECT_EQ(wf->lds().read32(1028), 41u);
}

TEST_F(Rdna4MemoryTest, DirectRouteSuppressesUnallocatedAtomicBeforeLdsUpdate) {
  // The direct route also accepts payloads from generated execution without
  // ComputeUnitCore::execute_instruction (instrumentation and embedded callers).
  wf->lds().write32(8, 41);
  set(2, 8);
  set(3, 1);
  constexpr std::array words{0xd8800000u, 0x20000302u}; // ds_add_rtn_u32 v32, v2, v3
  auto decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  auto inst = std::move(decoded).value();
  inst->execute(*inst, wf);
  ASSERT_NE(inst->data(), nullptr);
  ASSERT_EQ(inst->data_as<VectorMemState>()->lane_mask, 1u);
  EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu, inst.release(), *wf),
            VmAccessOutcome::Complete);
  EXPECT_EQ(wf->lds().read32(8), 41u);
  EXPECT_EQ(wf->exec(), 1u);
}

TEST_F(Rdna4MemoryTest, TransposeReadsOnlyItsAddressLanesAndWritesTheWholeWave) {
  using namespace test::execution_plugin;
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<OrderingPlugin>();
  auto *seen = observer.get();
  group->add(std::move(observer));
  cu->set_plugin_group(group);
  group->onInit();
  for (unsigned wave_size : {32u, 64u}) {
    wf->halt();
    wf = cu->dispatch_wf(0, 0x8000, 106, 32, wave_size);
    ASSERT_NE(wf, nullptr);
    const uint64_t live = wave_size == 64 ? UINT64_MAX : UINT32_MAX;
    wf->set_exec(live);
    for (auto op : {rdna4::kGlobalLoadTrB64Vglobal, rdna4::kGlobalLoadTrB128Vglobal,
                    rdna4::kGlobalLoadB32Vglobal}) {
      const bool compact = op != rdna4::kGlobalLoadB32Vglobal;
      const unsigned words = op == rdna4::kGlobalLoadTrB128Vglobal ? 4u : compact ? 2u : 1u;
      const auto dst = static_cast<uint8_t>(32 - (compact && wave_size == 64 ? words / 2 : words));
      const uint64_t address_lanes = compact ? UINT32_MAX : live;
      for (bool scalar_base : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << wave_size << " opcode=" << op << " saddr=" << scalar_base);
        auto inst = decoder->decode_window(rdna4::build_vglobal(
            op, {.saddr = static_cast<uint8_t>(scalar_base ? 0 : 124), .vdst = dst, .vaddr = 2}));
        ASSERT_TRUE(inst.succeeded());
        auto &wait = wf->ensure_memory_wait_scoreboard();
        for (uint16_t reg : {2u, 3u}) {
          for (unsigned lane : {0u, wave_size - 1}) {
            unsigned reports = 0;
            wait.clear();
            wait.bind(0x200, &reports,
                      [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
            constexpr auto counter = WaitCounterKind::Load;
            wait.add({wait.issue(counter),
                      0x100,
                      uint64_t{1} << lane,
                      {RegClass::VGPR, reg, 1},
                      counter,
                      0xf});
            wait.check_instruction(*inst.value(), *wf);
            EXPECT_EQ(reports, unsigned((reg == 2 || !scalar_base) &&
                                        (address_lanes & (uint64_t{1} << lane))));
          }
        }
        wait.clear();
        seen->events.clear();
        ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wf).succeeded());
        const auto reads = vgpr_read_events(*seen);
        ASSERT_EQ(reads.size(), scalar_base ? 1u : 2u);
        for (const auto &read : reads)
          EXPECT_EQ(read.lane_mask, address_lanes);
        ASSERT_NE(inst.value()->data(), nullptr);
        EXPECT_EQ(inst.value()->data_as<VectorMemState>()->lane_mask, live);
        for (unsigned lane : {0u, wave_size - 1}) {
          wait.clear();
          cu->track_memory_wait(*inst.value(), *wf);
          unsigned reports = 0;
          wait.bind(0x200, &reports,
                    [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
          wait.access({RegClass::VGPR, dst, 1}, uint64_t{1} << lane, 0xf, false);
          EXPECT_EQ(reports, 1u);
        }
      }
    }
  }
}

TEST_F(Rdna4MemoryTest, GlobalTransposeAllocatesOnlyTheReturnedWaveWords) {
  constexpr uint32_t sentinel = 0xdeadbeef;
  constexpr uint64_t pc = 0x8000;
  for (unsigned wave_size : {32u, 64u}) {
    wf->halt();
    wf = cu->dispatch_wf(0, pc, 106, 32, wave_size);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(wave_size == 64 ? UINT64_MAX : UINT32_MAX);
    for (unsigned word = 0; word < 256; ++word)
      memory.write32(0x20000 + word * 4, word + 1);
    for (unsigned lane = 0; lane < wave_size; ++lane) {
      set(2, 0x20000 + 16 * lane, lane);
      set(3, 0, lane);
    }
    for (const auto [op, width] : {std::pair{0xee15c07cu, 4u}, std::pair{0xee16007cu, 2u}}) {
      const unsigned output_words = wave_size == 64 ? width / 2 : width;
      for (unsigned overflow : {0u, 1u}) {
        SCOPED_TRACE(testing::Message() << wave_size << ' ' << width << ' ' << overflow);
        const unsigned dst = 32 - output_words + overflow;
        for (unsigned reg = dst; reg < dst + width; ++reg)
          set(reg, sentinel);
        const std::array<uint32_t, 3> words{op, dst, 2u};
        auto decoded = decoder->decode(words.data());
        ASSERT_TRUE(decoded.succeeded());
        auto *inst = decoded.value().get();
        auto &wait = wf->ensure_memory_wait_scoreboard();
        wait.clear();
        cu->track_memory_wait(*inst, *wf);
        EXPECT_EQ(wait.outstanding(WaitCounterKind::Load), 1u);
        unsigned reports = 0;
        wait.bind(pc + 12, &reports,
                  [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
        wait.access({RegClass::VGPR, static_cast<uint16_t>(dst), 1}, 1, 0xf, false);
        EXPECT_EQ(reports, unsigned(!overflow));
        ASSERT_NO_FATAL_FAILURE(issue(words));
        for (unsigned word = 0; word < output_words; ++word) {
          EXPECT_EQ(get(dst + word) != sentinel, !overflow);
        }
      }
    }
  }
}

TEST_F(Rdna4MemoryTest, WaitingForAnUnallocatedLoadResultHasNoRegisterDependency) {
  wf->set_exec(1);
  set(2, 0x20000);
  set(3, 0);
  const auto s = wf->sgpr_alloc().base;
  cu->write_sgpr(s, 0x20000);
  cu->write_sgpr(s + 1, 0);
  cu->write_sgpr(s + 2, 64);
  cu->write_sgpr(s + 3, (2u << 28) | (20u << 12));
  for (bool tfe : {false, true}) {
    for (uint8_t dst : {uint8_t{30}, uint8_t{31}}) {
      SCOPED_TRACE(testing::Message() << tfe << ' ' << unsigned(dst));
      auto words =
          tfe ? rdna4::build_vbuffer(rdna4::kBufferLoadB32Vbuffer,
                                     {.soffset = kModernNullSelector, .tfe = true, .vdata = dst})
              : rdna4::build_vglobal(rdna4::kGlobalLoadB64Vglobal,
                                     {.saddr = kModernNullSelector, .vdst = dst, .vaddr = 2});
      auto decoded = decoder->decode(words.data());
      ASSERT_TRUE(decoded.succeeded());
      auto &wait = wf->ensure_memory_wait_scoreboard();
      wait.clear();
      cu->track_memory_wait(*decoded.value(), *wf);
      EXPECT_EQ(wait.outstanding(WaitCounterKind::Load), 1u);
      unsigned reports = 0;
      wait.bind(0x100, &reports, [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
      // For VBUFFER the extra allocation word is reserved even when unwritten.
      wait.access({RegClass::VGPR, dst, 1}, 1, 0xf, false);
      EXPECT_EQ(reports, unsigned(dst == 30));
    }
  }
}

TEST_F(Rdna4MemoryTest, WideLoadsRequireTheWholeDestinationTuple) {
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (unsigned dst : {30u, 31u}) {
    SCOPED_TRACE(dst);
    const bool fits = dst == 30;
    for (uint32_t opcode : {0xd8dc0100u, 0xd9d80000u}) {
      set(2, 0);
      wf->lds().write32(0, 11);
      wf->lds().write32(4, 17);
      set(dst, sentinel);
      set(dst + 1, sentinel);
      issue({opcode, (dst << 24) | 2u, 0});
      EXPECT_EQ(get(dst), fits ? 11u : sentinel);
      EXPECT_EQ(get(dst + 1), fits ? 17u : sentinel);
    }
    set(2, 0x10000);
    set(3, 0);
    memory.write32(0x10000, 23);
    memory.write32(0x10004, 29);
    set(dst, sentinel);
    set(dst + 1, sentinel);
    issue({0xee05407c, dst, 2}); // global_load_b64
    EXPECT_EQ(get(dst), fits ? 23u : sentinel);
    EXPECT_EQ(get(dst + 1), fits ? 29u : sentinel);
  }
  const auto base = wf->sgpr_alloc().base;
  cu->write_sgpr(base, 0x10000);
  cu->write_sgpr(base + 1, 0);
  cu->write_sgpr(base + 2, 64);
  cu->write_sgpr(base + 3, (2u << 28) | (20u << 12));
  for (bool tfe : {false, true}) {
    set(31, sentinel);
    const auto words = rdna4::build_vbuffer(rdna4::kBufferLoadB32Vbuffer,
                                            {.soffset = 124, .tfe = tfe, .vdata = 31});
    issue(words);
    EXPECT_EQ(get(31), tfe ? sentinel : 23u);
    EXPECT_EQ(wf->exec(), 1u);
  }
}

TEST_F(Rdna4MemoryTest, DualAddressExchangeRequiresBothDestinationTuples) {
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (const unsigned width : {1u, 2u}) {
    // DS_STOREXCHG_2ADDR_RTN_B32/B64, offset0:0 offset1:1.
    const uint32_t encoding = width == 1 ? 0xd8b80100u : 0xd9b80100u;
    for (unsigned dst = wf->num_vgprs() - 2 * width; dst <= wf->num_vgprs(); ++dst) {
      SCOPED_TRACE(testing::Message() << "destination " << dst << " width " << width);
      set(2, 0);
      for (unsigned i = 0; i < 2 * width; ++i) {
        wf->lds().write32(4 * i, 11 + i);
        set((i < width ? 3 : 5) + i % width, 21 + i);
        set(dst + i, sentinel);
      }
      issue({encoding, (dst << 24) | 0x00050302u, 0});
      const bool fits = dst + 2 * width <= wf->num_vgprs();
      for (unsigned i = 0; i < 2 * width; ++i) {
        EXPECT_EQ(wf->lds().read32(4 * i), (fits ? 21u : 11u) + i);
        EXPECT_EQ(get(dst + i), fits ? 11u + i : sentinel);
      }
      EXPECT_EQ(wf->exec(), 1u);
    }
  }
}

TEST_F(Rdna4MemoryTest, UnboundAndWrongTypeHaveDifferentWriteback) {
  auto s = wf->sgpr_alloc().base;
  cu->write_sgpr(s, 0x40000);
  cu->write_sgpr(s + 1, 4u << 16);
  cu->write_sgpr(s + 2, 4);
  memory.write32(0x40000, 0x44444444);
  constexpr std::array words{0xc405007cu, 0x00800000u, 0u}; // buffer_load_b32
  cu->write_sgpr(s + 3, 2u << 28);
  set(0, 0x55555555);
  issue(words);
  EXPECT_EQ(get(), 0u);
  constexpr uint32_t kBoundFormatWithThreadId = (2u << 28) | (1u << 23) | (20u << 12);
  constexpr std::array store{0xc406807cu, 0x00800000u, 0u}; // buffer_store_b32
  wf->set_exec(2);
  for (bool bound : {false, true}) {
    cu->write_sgpr(s + 3, bound ? kBoundFormatWithThreadId : (2u << 28) | (1u << 23));
    memory.write32(0x40004, 0x44444445);
    set(0, 0x55555555, 1);
    issue(words);
    EXPECT_EQ(get(0, 1), bound ? 0x44444445u : 0u);
    set(0, 0x76543210, 1);
    issue(store);
    EXPECT_EQ(memory.read32(0x40004), bound ? 0x76543210u : 0x44444445u);
    EXPECT_EQ(memory.read32(0x40000), 0x44444444u);
  }
  wf->set_exec(1);
  cu->write_sgpr(s + 3, (8u << 28) | (20u << 12));
  set(0, 0x55555555);
  issue(words);
  EXPECT_EQ(get(), 0x55555555u);
  cu->write_sgpr(s + 3, (2u << 28) | (20u << 12));
  issue(words);
  EXPECT_EQ(get(), 0x44444444u);
}

} // namespace
} // namespace rocjitsu
