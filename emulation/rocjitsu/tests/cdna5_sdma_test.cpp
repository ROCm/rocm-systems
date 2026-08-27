// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

#include <limits>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

constexpr uint32_t kSdmaOpCopy = 1;
constexpr uint32_t kSdmaOpFence = 5;
constexpr uint32_t kSdmaOpPollRegmem = 8;
constexpr uint32_t kSdmaOpConstFill = 11;
constexpr uint32_t kSdmaOpTimestamp = 13;
constexpr uint32_t kSdmaOpGcr = 17;
constexpr uint32_t kSdmaSubopCopyLinear = 0;
constexpr uint32_t kSdmaSubopFence64 = 2;
constexpr uint32_t kSdmaSubopPollMem64 = 5;

class HostSdmaQueueForTest {
public:
  explicit HostSdmaQueueForTest(Gfx1250Sim &sim, uint64_t initial_doorbell = 0,
                                uint64_t last_doorbell = 0)
      : sim_(sim), doorbells_{initial_doorbell} {
    sim_.memory->set_passthrough(true);

    amdgpu::HwQueue queue{};
    queue.process_id = kProcessId;
    queue.queue_id = kQueueId;
    queue.ring_base_va = reinterpret_cast<uint64_t>(ring_.data());
    queue.ring_size = static_cast<uint32_t>(ring_.size() * sizeof(uint32_t));
    queue.read_ptr_va = reinterpret_cast<uint64_t>(&read_idx_);
    queue.write_ptr_va = reinterpret_cast<uint64_t>(&write_idx_);
    queue.doorbell_base = doorbells_.data();
    queue.doorbell_offset = 0;
    queue.last_doorbell = last_doorbell;
    queue.host_accessible = true;
    queue.is_sdma = true;
    sim_.cp()->register_queue(std::move(queue));
  }

  ~HostSdmaQueueForTest() { sim_.cp()->unregister_queue(kQueueId, kProcessId); }

  uint32_t *ring() { return ring_.data(); }

  void submit(uint32_t dwords) {
    uint64_t write_idx = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(write_idx_).store(write_idx, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_idx, std::memory_order_release);
    sim_.engine->schedule_event_now(sim_.cp()->doorbell_event());
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(read_idx_).load(std::memory_order_acquire);
  }

private:
  static constexpr uint32_t kProcessId = 0;
  static constexpr uint32_t kQueueId = 1250;

  Gfx1250Sim &sim_;
  std::array<uint32_t, 64> ring_{};
  alignas(8) uint64_t read_idx_ = 0;
  alignas(8) uint64_t write_idx_ = 0;
  std::array<uint64_t, 1> doorbells_{};
};

class TranslatedSdmaQueueForTest {
public:
  explicit TranslatedSdmaQueueForTest(Gfx1250Sim &sim) : sim_(sim), process_(kProcessId) {
    sim_.memory->register_process(kProcessId, &process_.page_table_, &process_.page_table_mutex_,
                                  process_.page_table_generation());
    process_.map_pages(kRingVa, ring_.data(), ring_.size() * sizeof(ring_[0]));
    process_.map_pages(kQueueStateVa, queue_state_.data(),
                       queue_state_.size() * sizeof(queue_state_[0]));
    process_.map_pages(kSrcVa, src_.data(), src_.size());
    process_.map_pages(kDstVa, dst_.data(), dst_.size());
    process_.map_pages(kDst2Va, dst2_.data(), dst2_.size());
    process_.map_pages(kSignalVa, signal_.data(), signal_.size() * sizeof(signal_[0]));
    process_.map_pages(kPollVa, poll_.data(), poll_.size() * sizeof(poll_[0]));

    amdgpu::HwQueue queue{};
    queue.process_id = kProcessId;
    queue.queue_id = kQueueId;
    queue.ring_base_va = kRingVa;
    queue.ring_size = static_cast<uint32_t>(ring_.size() * sizeof(ring_[0]));
    queue.read_ptr_va = kQueueStateVa;
    queue.write_ptr_va = kQueueStateVa + sizeof(queue_state_[0]);
    queue.doorbell_base = doorbells_.data();
    queue.doorbell_offset = 0;
    queue.host_accessible = true;
    queue.is_sdma = true;
    sim_.cp()->register_queue(std::move(queue));
  }

  ~TranslatedSdmaQueueForTest() {
    sim_.cp()->unregister_queue(kQueueId, kProcessId);
    sim_.memory->unregister_process(kProcessId);
  }

  uint32_t *ring() { return ring_.data(); }
  uint8_t *src() { return src_.data(); }
  uint8_t *dst() { return dst_.data(); }
  uint8_t *dst2() { return dst2_.data(); }
  int64_t &signal_value() { return signal_[0]; }
  uint64_t &poll_value() { return poll_[0]; }

  uint64_t src_va() const { return kSrcVa; }
  uint64_t dst_va() const { return kDstVa; }
  uint64_t dst2_va() const { return kDst2Va; }
  uint64_t signal_va() const { return kSignalVa; }
  uint64_t poll_va() const { return kPollVa; }

  void clip_dst_mapping(size_t size) {
    process_.unmap_pages(kDstVa, dst_.size());
    process_.map_pages(kDstVa, dst_.data(), size);
  }

  void unmap_src_tail_page() {
    process_.unmap_pages(kSrcVa + KfdProcess::kPageSize, KfdProcess::kPageSize);
  }

  void unmap_dst_tail_page() {
    process_.unmap_pages(kDstVa + KfdProcess::kPageSize, KfdProcess::kPageSize);
  }

  void remap_dst_tail_page() {
    process_.map_pages(kDstVa + KfdProcess::kPageSize, dst_.data() + KfdProcess::kPageSize,
                       KfdProcess::kPageSize);
  }

  void submit(uint32_t dwords) {
    uint64_t write_idx = static_cast<uint64_t>(dwords) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(queue_state_[1]).store(write_idx, std::memory_order_release);
    std::atomic_ref<uint64_t>(doorbells_[0]).store(write_idx, std::memory_order_release);
    sim_.engine->schedule_event_now(sim_.cp()->doorbell_event());
  }

  uint64_t read_idx() const {
    return std::atomic_ref<const uint64_t>(queue_state_[0]).load(std::memory_order_acquire);
  }

private:
  static constexpr uint32_t kProcessId = 1251;
  static constexpr uint32_t kQueueId = 1251;
  static constexpr uint64_t kRingVa = 0x1000'0000'0000ULL;
  static constexpr uint64_t kQueueStateVa = 0x1000'0000'1000ULL;
  static constexpr uint64_t kSrcVa = 0x1000'0000'2000ULL;
  static constexpr uint64_t kDstVa = 0x1000'0000'4000ULL;
  static constexpr uint64_t kDst2Va = 0x1000'0000'6000ULL;
  static constexpr uint64_t kSignalVa = 0x1000'0000'8000ULL;
  static constexpr uint64_t kPollVa = 0x1000'0000'9000ULL;

  Gfx1250Sim &sim_;
  KfdProcess process_;
  alignas(4096) std::array<uint32_t, 1024> ring_{};
  alignas(4096) std::array<uint64_t, 512> queue_state_{};
  alignas(4096) std::array<uint8_t, 8192> src_{};
  alignas(4096) std::array<uint8_t, 8192> dst_{};
  alignas(4096) std::array<uint8_t, 8192> dst2_{};
  alignas(4096) std::array<int64_t, 512> signal_{};
  alignas(4096) std::array<uint64_t, 512> poll_{};
  std::array<uint64_t, 1> doorbells_{};
};

void write_sdma_qword_va(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, uint64_t va) {
  packet[lo_dw] = static_cast<uint32_t>(va) & ~0x7u;
  packet[hi_dw] = static_cast<uint32_t>(va >> 32);
}

void write_sdma_qword_address(uint32_t *packet, uint32_t lo_dw, uint32_t hi_dw, const void *addr) {
  write_sdma_qword_va(packet, lo_dw, hi_dw, reinterpret_cast<uintptr_t>(addr));
}

TEST(Gfx1250SdmaTest, UnrungDoorbellSentinelDoesNotAdvanceAnEmptyQueue) {
  Gfx1250Sim sim;
  constexpr uint64_t kUnrungDoorbell = std::numeric_limits<uint64_t>::max();
  HostSdmaQueueForTest queue(sim, kUnrungDoorbell, kUnrungDoorbell);
  // Make accidental packet consumption terminate quickly instead of scanning
  // the all-ones producer range indefinitely.
  queue.ring()[0] = 0xFFu;

  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, PollMem64WaitsForFull64BitCondition) {
  Gfx1250Sim sim;
  HostSdmaQueueForTest queue(sim);
  alignas(8) uint64_t value = 1;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28); // 64-bit equal poll.
  write_sdma_qword_address(packet, 1, 2, &value);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(value).store(0, std::memory_order_release);
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 8u * sizeof(uint32_t));
}

TEST(Gfx1250SdmaTest, Fence64WritesFull64BitValue) {
  Gfx1250Sim sim;
  HostSdmaQueueForTest queue(sim);
  alignas(8) uint64_t value = 0;
  constexpr uint64_t kFenceValue = 0x12345678ABCDEF01ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpFence | (kSdmaSubopFence64 << 8) | (3u << 16);
  write_sdma_qword_address(packet, 1, 2, &value);
  packet[3] = static_cast<uint32_t>(kFenceValue);
  packet[4] = static_cast<uint32_t>(kFenceValue >> 32);

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 5u * sizeof(uint32_t));
  EXPECT_EQ(std::atomic_ref<uint64_t>(value).load(std::memory_order_acquire), kFenceValue);
}

// A wrong GCR packet size silently desyncs the SDMA ring read pointer and
// corrupts the following packet. Emit OP_GCR followed by a 32-bit FENCE and
// assert both the read pointer advance and that the FENCE decoded at the right
// boundary (its sentinel lands). gfx11/12 GCR is 5 dwords; gfx1250 is 6.
TEST(Gfx1250SdmaTest, GcrPacketSizeMatchesDialectAndKeepsRingInSync) {
  constexpr uint32_t kGcrLegacySize = 5;
  constexpr uint32_t kGcrGfx1250Size = 6;
  constexpr uint32_t kFenceSize = 4;
  constexpr uint32_t kFenceSentinel = 0xC0FFEE11u;
  // GL2 invalidate control bit position differs by dialect; setting it exercises
  // a realistic invalidate GCR but does not affect the decoded packet size.
  constexpr uint32_t kLegacyGl2InvControlDw = 2;
  constexpr uint32_t kLegacyGl2InvBit = 1u << 30;
  constexpr uint32_t kGfx1250Gl2InvControlDw = 3;
  constexpr uint32_t kGfx1250Gl2InvBit = 1u << 14;

  auto run_dialect = [kFenceSentinel](amdgpu::SdmaPacketDialect dialect, uint32_t gcr_size,
                                      uint32_t control_dw, uint32_t control_bit) {
    Gfx1250Sim sim;
    sim.cp()->set_sdma_packet_dialect(dialect);
    HostSdmaQueueForTest queue(sim);
    alignas(8) uint32_t fence_value = 0;

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    packet[control_dw] = control_bit;

    uint32_t *fence = packet + gcr_size;
    fence[0] = kSdmaOpFence; // 32-bit fence (sub_op 0).
    write_sdma_qword_address(fence, 1, 2, &fence_value);
    fence[3] = kFenceSentinel;

    queue.submit(gcr_size + kFenceSize);
    ASSERT_TRUE(sim.engine->step());
    EXPECT_EQ(queue.read_idx(), (gcr_size + kFenceSize) * sizeof(uint32_t));
    EXPECT_EQ(std::atomic_ref<uint32_t>(fence_value).load(std::memory_order_acquire),
              kFenceSentinel);
  };

  run_dialect(amdgpu::SdmaPacketDialect::Gfx11Plus, kGcrLegacySize, kLegacyGl2InvControlDw,
              kLegacyGl2InvBit);
  run_dialect(amdgpu::SdmaPacketDialect::Gfx1250, kGcrGfx1250Size, kGfx1250Gl2InvControlDw,
              kGfx1250Gl2InvBit);
}

// SDMA writes go straight to backing while L2 may still hold a dirty line that
// overlaps the destination. Seed that state explicitly with writeback_line().
// The fix flushes the caches before the direct write, so the dirty line is
// published first and the SDMA data supersedes it. Regression for that ordering.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  // The config-driven topology build wires the XCD's L2 into the CP, so the SDMA
  // cache maintenance operates on the same L2 instance we dirty below.
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Seed a dirty L2 line overlapping the destination, without touching backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, static_cast<int>(kStaleWord & 0xFF), sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  // CONST_FILL the destination line with a different byte pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());

  // Backing must reflect the SDMA fill, not the stale cached line.
  EXPECT_EQ(sim.memory->read32(queue.dst_va(), kProcessId), kFillWord);
  EXPECT_NE(sim.memory->read32(queue.dst_va(), kProcessId), kStaleWord);
}

TEST(Gfx1250SdmaTest, ConstFillWritesMappedPrefixAndAdvances) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr size_t kFillBytes = 128;
  constexpr size_t kMappedBytes = 64;
  constexpr uint8_t kInitialByte = 0xa5;
  constexpr uint32_t kFillWord = 0x44332211;
  queue.clip_dst_mapping(kMappedBytes);
  std::fill_n(queue.dst(), kFillBytes, kInitialByte);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = kFillBytes - 1;

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 5u * sizeof(uint32_t));

  std::array<uint8_t, sizeof(kFillWord)> pattern{};
  std::memcpy(pattern.data(), &kFillWord, sizeof(kFillWord));
  for (size_t i = 0; i < kMappedBytes; ++i)
    EXPECT_EQ(queue.dst()[i], pattern[i % pattern.size()]);
  EXPECT_TRUE(std::all_of(queue.dst() + kMappedBytes, queue.dst() + kFillBytes,
                          [](uint8_t value) { return value == kInitialByte; }));
}

TEST(Gfx1250SdmaTest, ConstFillUnmappedTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr size_t kFillBytes = 8192;
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30);
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = 0x44332211;
  packet[4] = kFillBytes - 1;

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

// A scalar L1 (K$) can retain a clean snapshot overlapping an SDMA destination.
// The pre-write maintenance invalidates that snapshot, and later K$ maintenance
// must not publish it over the direct SDMA result.
TEST(Gfx1250SdmaTest, ConstFillSupersedesOverlappingScalarL1Line) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  ASSERT_NE(cu, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kStaleWord = 0x11111111u;
  constexpr uint32_t kFillWord = 0x22222222u;

  // Populate a K$ line overlapping the SDMA destination via a write-through
  // scalar store. K$ retains a clean snapshot of the pre-fill value.
  cu->l1_scalar().store(queue.dst_va(), /*num_dwords=*/1, &kStaleWord, kProcessId);

  // CONST_FILL the destination line with a different pattern.
  auto *packet = queue.ring();
  packet[0] = kSdmaOpConstFill | (0x2u << 30); // fillsize=2 (dword granularity).
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());
  packet[3] = kFillWord;
  packet[4] = amdgpu::L2Cache::LINE_SIZE - 1; // count-1 bytes.

  queue.submit(5);
  ASSERT_TRUE(sim.engine->step());

  // The SDMA pre-write maintenance must have invalidated the old K$ snapshot,
  // so the first scalar reload observes the fill rather than kStaleWord.
  uint32_t scalar_value = 0;
  cu->l1_scalar().load(queue.dst_va(), /*num_dwords=*/1, &scalar_value, kProcessId);
  EXPECT_EQ(scalar_value, kFillWord);

  // Mimic later acquire/release maintenance. The clean K$ snapshot must not
  // resurrect stale data over the SDMA fill.
  cu->flush_l1(kProcessId);
  if (auto *l2 = sim.xcd()->l2_cache())
    l2->flush_all();

  // Backing must reflect the SDMA fill, not the stale scalar line.
  EXPECT_EQ(sim.memory->read32(queue.dst_va(), kProcessId), kFillWord);
  EXPECT_NE(sim.memory->read32(queue.dst_va(), kProcessId), kStaleWord);
}

// OP_TIMESTAMP is a direct backing-store write like COPY/FENCE/CONST_FILL, so it
// has the same clobber hazard: a dirty cached line overlapping the timestamp
// address must be published before the store, not written out over it by a later
// flush. Seed a dirty L2 line at the timestamp address, issue OP_TIMESTAMP, then
// force a flush; the stored timestamp must survive (stale word gone, value set).
TEST(Gfx1250SdmaTest, TimestampSupersedesOverlappingDirtyL2Line) {
  Gfx1250Sim sim;
  auto *l2 = sim.xcd()->l2_cache();
  ASSERT_NE(l2, nullptr);

  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint64_t kStaleQword = 0x1111111111111111ULL;

  // Seed a dirty L2 line overlapping the timestamp destination, without touching
  // backing.
  uint8_t stale_line[amdgpu::L2Cache::LINE_SIZE];
  std::memset(stale_line, 0x11, sizeof(stale_line));
  l2->writeback_line(queue.dst_va(), stale_line, amdgpu::Mtype::RW, kProcessId);

  auto *packet = queue.ring();
  packet[0] = kSdmaOpTimestamp;
  write_sdma_qword_va(packet, 1, 2, queue.dst_va());

  queue.submit(3); // TIMESTAMP is 3 dwords.
  ASSERT_TRUE(sim.engine->step());

  // Force any still-resident dirty line out, mimicking a later flush.
  if (auto *xl2 = sim.xcd()->l2_cache())
    xl2->flush_all();

  // The timestamp value is nondeterministic, but it must not be the stale word
  // and must be a plausible nonzero nanosecond count.
  const uint64_t stored = sim.memory->read64(queue.dst_va(), kProcessId);
  EXPECT_NE(stored, kStaleQword);
  EXPECT_NE(stored, 0u);
}

// The GCR decoder now distinguishes GL2 writeback (publish dirty lines),
// invalidate/discard (drop without writeback), and no-op (no GL2 bits). This is
// the data-loss distinction the PR protects. Dirty an L2 line, then issue each
// GCR flavor and observe whether the dirty data reaches backing.
TEST(Gfx1250SdmaTest, GcrWritebackPublishesInvalidateDropsNoopKeeps) {
  constexpr uint32_t kProcessId = 1251; // matches TranslatedSdmaQueueForTest.
  constexpr uint32_t kDirtyWord = 0x33333333u;
  constexpr uint32_t kBackingWord = 0x44444444u;
  // gfx1250 GCR control dword (DW3) bit positions.
  constexpr uint32_t kControlDw = 3;
  constexpr uint32_t kGl2InvBit = 1u << 14;
  constexpr uint32_t kGl2WbBit = 1u << 15;

  // Outcome of a GCR flavor: the value in backing (read directly through the
  // page table) and the value seen through L2 (which returns the resident dirty
  // line if still present, or re-fetches backing if the line was dropped).
  struct GcrOutcome {
    uint32_t backing = 0;
    uint32_t via_l2 = 0;
  };

  enum class GcrKind { WritebackOnly, InvalidateOnly, Noop };
  // Void return so a missing L2 is a fatal guard (ASSERT_*) before we deref it.
  auto run = [&](GcrKind kind, GcrOutcome &out) {
    Gfx1250Sim sim;
    auto *l2 = sim.xcd()->l2_cache();
    ASSERT_NE(l2, nullptr);
    TranslatedSdmaQueueForTest queue(sim);

    // Put a known value in backing, then a different dirty value in L2 on top.
    for (uint32_t i = 0; i < sizeof(uint32_t); ++i)
      sim.memory->write8(queue.dst_va() + i, static_cast<uint8_t>((kBackingWord >> (i * 8)) & 0xFF),
                         kProcessId);
    uint8_t dirty_line[amdgpu::L2Cache::LINE_SIZE];
    std::memset(dirty_line, static_cast<int>(kDirtyWord & 0xFF), sizeof(dirty_line));
    l2->writeback_line(queue.dst_va(), dirty_line, amdgpu::Mtype::RW, kProcessId);

    auto *packet = queue.ring();
    packet[0] = kSdmaOpGcr;
    if (kind == GcrKind::WritebackOnly)
      packet[kControlDw] = kGl2WbBit;
    else if (kind == GcrKind::InvalidateOnly)
      packet[kControlDw] = kGl2InvBit;
    else
      packet[kControlDw] = 0; // no GL2 bits: no-op.

    queue.submit(6); // gfx1250 GCR is 6 dwords.
    EXPECT_TRUE(sim.engine->step());

    out.backing = sim.memory->read32(queue.dst_va(), kProcessId);
    // Read back through L2: a still-resident dirty line returns kDirtyWord; a
    // dropped line re-fetches from backing on the miss.
    uint32_t l2_word = 0;
    l2->read(queue.dst_va(), reinterpret_cast<uint8_t *>(&l2_word), sizeof(l2_word),
             amdgpu::Mtype::RW, kProcessId);
    out.via_l2 = l2_word;
  };

  // Writeback publishes the dirty line to backing.
  GcrOutcome wb;
  run(GcrKind::WritebackOnly, wb);
  EXPECT_EQ(wb.backing, kDirtyWord);
  // Invalidate/discard drops the dirty line without writeback; backing keeps its
  // original value and the line is no longer resident.
  GcrOutcome inv;
  run(GcrKind::InvalidateOnly, inv);
  EXPECT_EQ(inv.backing, kBackingWord);
  EXPECT_EQ(inv.via_l2, kBackingWord); // line dropped → L2 re-fetches backing.
  // No GL2 bits: no cache maintenance at all. Backing is untouched and the dirty
  // line stays resident in L2 (this is what distinguishes no-op from
  // invalidate-only: an incorrect invalidate would drop the line here too).
  GcrOutcome noop;
  run(GcrKind::Noop, noop);
  EXPECT_EQ(noop.backing, kBackingWord);
  EXPECT_EQ(noop.via_l2, kDirtyWord); // dirty line still resident in L2.
}

TEST(Gfx1250SdmaTest, CopyWaitSignalResolvesTranslatedAddresses) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.poll_value() = 0;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x5a);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30) | (1u << 31);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, queue.poll_va());
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());
  packet[14] = 0x70;
  write_sdma_qword_va(packet, 15, 16, queue.signal_va());
  packet[17] = 1;
  packet[18] = 0;

  queue.submit(19);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 19u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 4);
}

TEST(Gfx1250SdmaTest, CompactCopyWaitPacketCopies) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.poll_value() = 0;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xb7);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, queue.poll_va());
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());

  queue.submit(14);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 14u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CompactCopySignalPacketCopiesAndSignals) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xd3);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 12u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 4);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedWaitAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedWaitVa = 0x2000'0000'0000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xa5);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 30);
  packet[1] = 3;
  write_sdma_qword_va(packet, 2, 3, kUnmappedWaitVa);
  packet[4] = 0;
  packet[5] = 0;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0xFFFFFFFFu;
  packet[8] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 10, 11, queue.src_va());
  write_sdma_qword_va(packet, 12, 13, queue.dst_va());

  queue.submit(14);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedDstDoesNotAdvanceOrSignal) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'2000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x3c);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kUnmappedDstVa);
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, queue.signal_va());
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyWaitSignalUnresolvedSignalDoesNotAdvanceOrCopy) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedSignalVa = 0x2000'0000'4000ULL;
  queue.signal_value() = 5;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x69);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 31);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  packet[7] = 0x70;
  write_sdma_qword_va(packet, 8, 9, kUnmappedSignalVa);
  packet[10] = 1;
  packet[11] = 0;

  queue.submit(12);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(queue.signal_value(), 5);
}

TEST(Gfx1250SdmaTest, CopyLinearUnresolvedDstDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint64_t kUnmappedDstVa = 0x2000'0000'3000ULL;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0xc3);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, kUnmappedDstVa);

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
  EXPECT_NE(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearUnmappedTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, CopyLinearResumesAfterTailMappingIsInstalled) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 31 + 11) & 0xff);
    queue.dst()[i] = 0;
  }
  queue.unmap_dst_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);

  queue.remap_dst_tail_page();
  // A page-table update is followed by the same doorbell recheck a real queue
  // receives from its host-side monitor.
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  (void)sim.engine->step();
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearUnmappedSourceTailDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  queue.unmap_src_tail_page();

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

TEST(Gfx1250SdmaTest, CopyLinearClippedDstAdvancesDeterministically) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  constexpr uint32_t kMappedBytes = 64;
  queue.clip_dst_mapping(kMappedBytes);
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x6d);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kMappedBytes), 0);
  EXPECT_TRUE(std::all_of(queue.dst() + kMappedBytes, queue.dst() + kCopyBytes,
                          [](uint8_t value) { return value == 0; }));
}

TEST(Gfx1250SdmaTest, CopyLinearTransfersMultipleScratchChunks) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, BroadcastCopyTransfersMultipleScratchChunks) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 8192;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>((i * 29 + 7) & 0xff);
    queue.dst()[i] = 0;
    queue.dst2()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 27);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());
  write_sdma_qword_va(packet, 7, 8, queue.dst2_va());

  queue.submit(9);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 9u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
  EXPECT_EQ(std::memcmp(queue.dst2(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, CopyLinearNpdBitDoesNotDecodeAsBroadcast) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint32_t kCopyBytes = 128;
  for (uint32_t i = 0; i < kCopyBytes; ++i) {
    queue.src()[i] = static_cast<uint8_t>(i ^ 0x4d);
    queue.dst()[i] = 0;
  }

  auto *packet = queue.ring();
  packet[0] = kSdmaOpCopy | (kSdmaSubopCopyLinear << 8) | (1u << 28);
  packet[1] = kCopyBytes - 1;
  write_sdma_qword_va(packet, 3, 4, queue.src_va());
  write_sdma_qword_va(packet, 5, 6, queue.dst_va());

  queue.submit(7);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 7u * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(queue.dst(), queue.src(), kCopyBytes), 0);
}

TEST(Gfx1250SdmaTest, PollMem64ResolvesTranslatedAddress) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  queue.poll_value() = 1;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, queue.poll_va());
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);

  std::atomic_ref<uint64_t>(queue.poll_value()).store(0, std::memory_order_release);
  sim.engine->schedule_event_now(sim.cp()->doorbell_event());
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 8u * sizeof(uint32_t));
}

TEST(Gfx1250SdmaTest, PollMem64UnresolvedAddressDoesNotAdvance) {
  Gfx1250Sim sim;
  TranslatedSdmaQueueForTest queue(sim);
  constexpr uint64_t kUnmappedPollVa = 0x2000'0000'1000ULL;

  auto *packet = queue.ring();
  packet[0] = kSdmaOpPollRegmem | (kSdmaSubopPollMem64 << 8) | (3u << 28);
  write_sdma_qword_va(packet, 1, 2, kUnmappedPollVa);
  packet[3] = 0;
  packet[4] = 0;
  packet[5] = 0xFFFFFFFFu;
  packet[6] = 0xFFFFFFFFu;
  packet[7] = 0;

  queue.submit(8);
  ASSERT_TRUE(sim.engine->step());
  EXPECT_EQ(queue.read_idx(), 0u);
}

} // namespace
