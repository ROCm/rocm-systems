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
  GpuMemory memory_{"memory"};
  L2Cache l2_{"l2"};
  std::unique_ptr<ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  Wavefront *wf_ = nullptr;
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 64;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu_, "cu", cfg, &memory_, &l2_));
    wf_ = cu_->dispatch_wf(0, 0, 106, 32, 32);
    ASSERT_NE(wf_, nullptr);
    wf_->set_exec(1);
    wf_->set_lds_size(1024);
    wf_->set_scratch_base(0x20000);
    wf_->set_scratch_lane_size(256);
    const auto shared = kPrivate + (1ull << 32);
    cu_->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
    wf_->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX,
                       /*override_launch=*/true);
  }
  void TearDown() override {
    if (wf_)
      wf_->halt();
  }
  void write_vector(unsigned reg, uint32_t word, unsigned lane = 0) {
    cu_->write_vgpr(wf_->vgpr_alloc().base + reg, lane, word);
  }
  uint32_t read_vector(unsigned reg = 0, unsigned lane = 0) {
    return cu_->read_vgpr(wf_->vgpr_alloc().base + reg, lane);
  }
  void issue(std::array<uint32_t, 3> words, VmAccessOutcome expected = VmAccessOutcome::Complete) {
    auto decoded = decoder_->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded()) << inst->mnemonic();
    ASSERT_NE(inst->data(), nullptr);
    EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu_, std::move(inst), *wf_),
              expected);
  }
};

TEST_F(Rdna4MemoryTest, OrderedAddMatchesIdUpdatesTwoWordsAndReturnsOldPair) {
  // gfx1201 global_atomic_ordered_add_b64 v[0:1], v[2:3], v[4:5], off th:TH_ATOMIC_RETURN
  constexpr std::array words{0xee1cc07cu, 0x02100000u, 0x00000002u};
  write_vector(2, 0x10000);
  write_vector(3, 0);
  memory_.write32(0x10000, 4095);
  memory_.write32(0x10004, UINT32_MAX);
  write_vector(4, 3);
  write_vector(5, 2);
  issue(words);
  EXPECT_EQ(read_vector(0), 4095u);
  EXPECT_EQ(read_vector(1), UINT32_MAX);
  write_vector(4, 4095);
  issue(words);
  EXPECT_EQ(read_vector(0), 4095u);
  EXPECT_EQ(read_vector(1), UINT32_MAX);
  issue({0xee05407c, 0, 2}); // global_load_b64
  EXPECT_EQ(read_vector(0), 0u);
  EXPECT_EQ(read_vector(1), 1u);
}

TEST_F(Rdna4MemoryTest, MisalignedFlatAtomicPreservesSharedAndGlobalLanes) {
  constexpr uint64_t shared = kPrivate + (uint64_t{1} << 32), global = 0x50000;
  constexpr uint32_t old = 0x11223344, sentinel = 0xdeadbeef;
  wf_->set_exec(3);
  for (unsigned offset : {4u, 5u}) {
    const bool aligned = offset == 4;
    wf_->lds().write32(4, old);
    wf_->lds().write32(8, 0x55667788);
    memory_.write32(global, 37);
    for (unsigned lane = 0; lane < 2; ++lane) {
      const uint64_t addr = lane == 0 ? shared + offset : global;
      write_vector(0, sentinel, lane);
      write_vector(2, uint32_t(addr), lane);
      write_vector(3, addr >> 32, lane);
      write_vector(4, lane + 1, lane);
    }
    issue({0xec0d407c, 0x02100000, 0x00000002},
          aligned ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted);
    EXPECT_EQ(wf_->lds().read32(4), aligned ? old + 1 : old);
    EXPECT_EQ(memory_.read32(global), aligned ? 39u : 37u);
    EXPECT_EQ(read_vector(0, 0), aligned ? old : sentinel);
    EXPECT_EQ(read_vector(0, 1), aligned ? 37u : sentinel);
  }
}

TEST_F(Rdna4MemoryTest, FlatSelectsItsSegmentBeforeAddingTheImmediate) {
  constexpr uint64_t shared = kPrivate + (uint64_t{2} << 32);
  cu_->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX);
  wf_->set_apertures(shared, shared + UINT32_MAX, kPrivate, kPrivate + UINT32_MAX,
                     /*override_launch=*/true);
  memory_.write32(shared, 0x11112222);
  memory_.write32(kPrivate, 0x33334444);
  memory_.write32(wf_->scratch_base(), 0x55556666);
  wf_->lds().write32(0, 0x77778888);
  const std::array words{0xec05007cu, 0x00000000u, 0x00000402u};
  for (const auto &[pointer, lds, expected] : {std::tuple{shared - 4, false, 0x11112222u},
                                               {kPrivate - 4, false, 0x33334444u},
                                               {kPrivate + 0xfffffffc, false, 0x55556666u},
                                               {shared + 0xfffffffc, true, 0x77778888u}}) {
    write_vector(2, static_cast<uint32_t>(pointer));
    write_vector(3, static_cast<uint32_t>(pointer >> 32));
    auto decoded = decoder_->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    const auto lanes = amdgpu::MemoryWaitScoreboard::flat_lanes(*decoded.value(), *wf_, shared,
                                                                shared + UINT32_MAX);
    EXPECT_EQ(lanes.shared, lds ? 1u : 0u);
    ASSERT_NO_FATAL_FAILURE(issue(words));
    EXPECT_EQ(read_vector(0), expected);
  }
}

TEST_F(Rdna4MemoryTest, PrivateScratchInterleavesDwordsAndRejectsMixedB64Atomic) {
  wf_->set_exec(2);
  memory_.write32(0x20000 + 132, 0x12345678);
  memory_.write32(0x20000 + 260, 0xdeadbeef);
  write_vector(2, 4, 1);
  issue({0xed05007c, 0x00020000, 2}); // scratch_load_b32
  EXPECT_EQ(read_vector(0, 1), 0x12345678u);
  write_vector(3, kPrivate >> 32, 1);
  issue({0xec05007c, 0, 2}); // flat_load_b32
  EXPECT_EQ(read_vector(0, 1), 0x12345678u);
  wf_->set_exec(3);
  write_vector(2, 0x30000);
  write_vector(3, 0);
  write_vector(4, 1);
  write_vector(5, 0);
  write_vector(2, 0, 1);
  memory_.write32(0x30000, 41);
  issue({0xec10c07c, 0x02100000, 2}); // flat_atomic_add_u64
  EXPECT_NE(wf_->trapsts() & (1u << 8), 0u);
  EXPECT_EQ(memory_.read32(0x30000), 41u);
}

TEST_F(Rdna4MemoryTest, AllocatesLdsInKibBlocks) {
  EXPECT_EQ(cu_->allocate_lds(1), 0u);
  EXPECT_EQ(cu_->allocate_lds(512), 1024u);
  EXPECT_EQ(cu_->allocate_lds(1025), 2048u);
  EXPECT_EQ(cu_->allocate_lds(1), 4096u);
}

TEST_F(Rdna4MemoryTest, GuardLdsRangeAndReturningAtomicResult) {
  wf_->set_lds_base(1024);
  wf_->lds().write32(2048, 41);
  write_vector(2, 1024);
  issue({0xd8d80000, 2}); // ds_load_b32
  EXPECT_EQ(read_vector(), 0u);
  wf_->lds().write32(1028, 41);
  write_vector(2, 4);
  write_vector(3, 1);
  issue({0xd8800000, 0x20000302}); // ds_add_rtn_u32 v32
  EXPECT_EQ(wf_->lds().read32(1028), 41u);
}

TEST_F(Rdna4MemoryTest, DirectRouteSuppressesUnallocatedAtomicBeforeLdsUpdate) {
  // The direct route also accepts payloads from generated execution without
  // ComputeUnitCore::execute_instruction (instrumentation and embedded callers).
  wf_->lds().write32(8, 41);
  write_vector(2, 8);
  write_vector(3, 1);
  constexpr std::array words{0xd8800000u, 0x20000302u}; // ds_add_rtn_u32 v32, v2, v3
  auto decoded = decoder_->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  auto inst = std::move(decoded).value();
  inst->execute(*inst, wf_);
  ASSERT_NE(inst->data(), nullptr);
  ASSERT_EQ(inst->data_as<VectorMemState>()->lane_mask, 1u);
  EXPECT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu_, std::move(inst), *wf_),
            VmAccessOutcome::Complete);
  EXPECT_EQ(wf_->lds().read32(8), 41u);
  EXPECT_EQ(wf_->exec(), 1u);
}

TEST_F(Rdna4MemoryTest, TransposeReadsOnlyItsAddressLanesAndWritesTheWholeWave) {
  using namespace test::execution_plugin;
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<OrderingPlugin>();
  auto *seen = observer.get();
  group->add(std::move(observer));
  cu_->set_plugin_group(group);
  group->onInit();
  for (unsigned wave_size : {32u, 64u}) {
    wf_->halt();
    wf_ = cu_->dispatch_wf(0, 0x8000, 106, 32, wave_size);
    ASSERT_NE(wf_, nullptr);
    const uint64_t live = wave_size == 64 ? UINT64_MAX : UINT32_MAX;
    wf_->set_exec(live);
    for (auto op : {rdna4::kGlobalLoadTrB64Vglobal, rdna4::kGlobalLoadTrB128Vglobal,
                    rdna4::kGlobalLoadB32Vglobal}) {
      const bool compact = op != rdna4::kGlobalLoadB32Vglobal;
      const unsigned words = op == rdna4::kGlobalLoadTrB128Vglobal ? 4u : compact ? 2u : 1u;
      const auto dst = static_cast<uint8_t>(32 - (compact && wave_size == 64 ? words / 2 : words));
      const uint64_t address_lanes = compact ? UINT32_MAX : live;
      for (bool scalar_base : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << wave_size << " opcode=" << op << " saddr=" << scalar_base);
        auto inst = decoder_->decode_window(rdna4::build_vglobal(
            op, {.saddr = static_cast<uint8_t>(scalar_base ? 0 : 124), .vdst = dst, .vaddr = 2}));
        ASSERT_TRUE(inst.succeeded());
        auto &wait = wf_->ensure_memory_wait_scoreboard();
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
            wait.check_instruction(*inst.value(), *wf_);
            EXPECT_EQ(reports, unsigned((reg == 2 || !scalar_base) &&
                                        (address_lanes & (uint64_t{1} << lane))));
          }
        }
        wait.clear();
        seen->events.clear();
        ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
        const auto reads = vgpr_read_events(*seen);
        ASSERT_EQ(reads.size(), scalar_base ? 1u : 2u);
        for (const auto &read : reads)
          EXPECT_EQ(read.lane_mask, address_lanes);
        ASSERT_NE(inst.value()->data(), nullptr);
        EXPECT_EQ(inst.value()->data_as<VectorMemState>()->lane_mask, live);
        for (unsigned lane : {0u, wave_size - 1}) {
          wait.clear();
          cu_->track_memory_wait(*inst.value(), *wf_);
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
    wf_->halt();
    wf_ = cu_->dispatch_wf(0, pc, 106, 32, wave_size);
    ASSERT_NE(wf_, nullptr);
    wf_->set_exec(wave_size == 64 ? UINT64_MAX : UINT32_MAX);
    for (unsigned word = 0; word < 256; ++word)
      memory_.write32(0x20000 + word * 4, word + 1);
    for (unsigned lane = 0; lane < wave_size; ++lane) {
      write_vector(2, 0x20000 + 16 * lane, lane);
      write_vector(3, 0, lane);
    }
    for (const auto &[op, width] : {std::pair{0xee15c07cu, 4u}, std::pair{0xee16007cu, 2u}}) {
      const unsigned output_words = wave_size == 64 ? width / 2 : width;
      for (unsigned overflow : {0u, 1u}) {
        SCOPED_TRACE(testing::Message() << wave_size << ' ' << width << ' ' << overflow);
        const unsigned dst = 32 - output_words + overflow;
        for (unsigned reg = dst; reg < dst + width; ++reg)
          write_vector(reg, sentinel);
        const std::array<uint32_t, 3> words{op, dst, 2u};
        auto decoded = decoder_->decode(words.data());
        ASSERT_TRUE(decoded.succeeded());
        auto *inst = decoded.value().get();
        auto &wait = wf_->ensure_memory_wait_scoreboard();
        wait.clear();
        cu_->track_memory_wait(*inst, *wf_);
        EXPECT_EQ(wait.outstanding(WaitCounterKind::Load), 1u);
        unsigned reports = 0;
        wait.bind(pc + 12, &reports,
                  [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
        wait.access({RegClass::VGPR, static_cast<uint16_t>(dst), 1}, 1, 0xf, false);
        EXPECT_EQ(reports, unsigned(!overflow));
        ASSERT_NO_FATAL_FAILURE(issue(words));
        for (unsigned word = 0; word < output_words; ++word) {
          EXPECT_EQ(read_vector(dst + word) != sentinel, !overflow);
        }
      }
    }
  }
}

TEST_F(Rdna4MemoryTest, WaitingForAnUnallocatedLoadResultHasNoRegisterDependency) {
  wf_->set_exec(1);
  write_vector(2, 0x20000);
  write_vector(3, 0);
  const auto sgpr_base = wf_->sgpr_alloc().base;
  cu_->write_sgpr(sgpr_base, 0x20000);
  cu_->write_sgpr(sgpr_base + 1, 0);
  cu_->write_sgpr(sgpr_base + 2, 64);
  cu_->write_sgpr(sgpr_base + 3, (2u << 28) | (20u << 12));
  for (bool tfe : {false, true}) {
    for (uint8_t dst : {uint8_t{30}, uint8_t{31}}) {
      SCOPED_TRACE(testing::Message() << tfe << ' ' << unsigned(dst));
      auto words =
          tfe ? rdna4::build_vbuffer(rdna4::kBufferLoadB32Vbuffer,
                                     {.soffset = kModernNullSelector, .tfe = true, .vdata = dst})
              : rdna4::build_vglobal(rdna4::kGlobalLoadB64Vglobal,
                                     {.saddr = kModernNullSelector, .vdst = dst, .vaddr = 2});
      auto decoded = decoder_->decode(words.data());
      ASSERT_TRUE(decoded.succeeded());
      auto &wait = wf_->ensure_memory_wait_scoreboard();
      wait.clear();
      cu_->track_memory_wait(*decoded.value(), *wf_);
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
      write_vector(2, 0);
      wf_->lds().write32(0, 11);
      wf_->lds().write32(4, 17);
      write_vector(dst, sentinel);
      write_vector(dst + 1, sentinel);
      issue({opcode, (dst << 24) | 2u, 0});
      EXPECT_EQ(read_vector(dst), fits ? 11u : sentinel);
      EXPECT_EQ(read_vector(dst + 1), fits ? 17u : sentinel);
    }
    write_vector(2, 0x10000);
    write_vector(3, 0);
    memory_.write32(0x10000, 23);
    memory_.write32(0x10004, 29);
    write_vector(dst, sentinel);
    write_vector(dst + 1, sentinel);
    issue({0xee05407c, dst, 2}); // global_load_b64
    EXPECT_EQ(read_vector(dst), fits ? 23u : sentinel);
    EXPECT_EQ(read_vector(dst + 1), fits ? 29u : sentinel);
  }
  const auto base = wf_->sgpr_alloc().base;
  cu_->write_sgpr(base, 0x10000);
  cu_->write_sgpr(base + 1, 0);
  cu_->write_sgpr(base + 2, 64);
  cu_->write_sgpr(base + 3, (2u << 28) | (20u << 12));
  for (bool tfe : {false, true}) {
    write_vector(31, sentinel);
    const auto words = rdna4::build_vbuffer(rdna4::kBufferLoadB32Vbuffer,
                                            {.soffset = 124, .tfe = tfe, .vdata = 31});
    issue(words);
    EXPECT_EQ(read_vector(31), tfe ? sentinel : 23u);
    EXPECT_EQ(wf_->exec(), 1u);
  }
}

TEST_F(Rdna4MemoryTest, DualAddressExchangeRequiresBothDestinationTuples) {
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (const unsigned width : {1u, 2u}) {
    // DS_STOREXCHG_2ADDR_RTN_B32/B64, offset0:0 offset1:1.
    const uint32_t encoding = width == 1 ? 0xd8b80100u : 0xd9b80100u;
    for (unsigned dst = wf_->num_vgprs() - 2 * width; dst <= wf_->num_vgprs(); ++dst) {
      SCOPED_TRACE(testing::Message() << "destination " << dst << " width " << width);
      write_vector(2, 0);
      for (unsigned i = 0; i < 2 * width; ++i) {
        wf_->lds().write32(4 * i, 11 + i);
        write_vector((i < width ? 3 : 5) + i % width, 21 + i);
        write_vector(dst + i, sentinel);
      }
      issue({encoding, (dst << 24) | 0x00050302u, 0});
      const bool fits = dst + 2 * width <= wf_->num_vgprs();
      for (unsigned i = 0; i < 2 * width; ++i) {
        EXPECT_EQ(wf_->lds().read32(4 * i), (fits ? 21u : 11u) + i);
        EXPECT_EQ(read_vector(dst + i), fits ? 11u + i : sentinel);
      }
      EXPECT_EQ(wf_->exec(), 1u);
    }
  }
}

TEST_F(Rdna4MemoryTest, UnboundAndWrongTypeHaveDifferentWriteback) {
  const auto sgpr_base = wf_->sgpr_alloc().base;
  cu_->write_sgpr(sgpr_base, 0x40000);
  cu_->write_sgpr(sgpr_base + 1, 4u << 16);
  cu_->write_sgpr(sgpr_base + 2, 4);
  memory_.write32(0x40000, 0x44444444);
  constexpr std::array words{0xc405007cu, 0x00800000u, 0u}; // buffer_load_b32
  cu_->write_sgpr(sgpr_base + 3, 2u << 28);
  write_vector(0, 0x55555555);
  issue(words);
  EXPECT_EQ(read_vector(), 0u);
  constexpr uint32_t kBoundFormatWithThreadId = (2u << 28) | (1u << 23) | (20u << 12);
  constexpr std::array store{0xc406807cu, 0x00800000u, 0u}; // buffer_store_b32
  wf_->set_exec(2);
  for (bool bound : {false, true}) {
    cu_->write_sgpr(sgpr_base + 3, bound ? kBoundFormatWithThreadId : (2u << 28) | (1u << 23));
    memory_.write32(0x40004, 0x44444445);
    write_vector(0, 0x55555555, 1);
    issue(words);
    EXPECT_EQ(read_vector(0, 1), bound ? 0x44444445u : 0u);
    write_vector(0, 0x76543210, 1);
    issue(store);
    EXPECT_EQ(memory_.read32(0x40004), bound ? 0x76543210u : 0x44444445u);
    EXPECT_EQ(memory_.read32(0x40000), 0x44444444u);
  }
  wf_->set_exec(1);
  cu_->write_sgpr(sgpr_base + 3, (8u << 28) | (20u << 12));
  write_vector(0, 0x55555555);
  issue(words);
  EXPECT_EQ(read_vector(), 0x55555555u);
  cu_->write_sgpr(sgpr_base + 3, (2u << 28) | (20u << 12));
  issue(words);
  EXPECT_EQ(read_vector(), 0x44444444u);
}

} // namespace
} // namespace rocjitsu
