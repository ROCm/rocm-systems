// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/log_buffer.h"

#include "rocjitsu/code/patch/log_abi.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// Simulate what a device producer does: write a record into the slot at the
// current write_ptr, publish valid, then advance write_ptr. Host-side stand-in
// used by the CPU-only tests (there is no atomic arbitration to model here
// because these tests are single-threaded and post-completion).
void publish_record(LogBuffer &buf, uint32_t record_type, uint32_t site, uint64_t payload) {
  RjLogBufferHeader *hdr = buf.header();
  RjLogRecord *recs = buf.records();
  const uint64_t index = hdr->write_ptr;
  RjLogRecord &rec = recs[index & (buf.slot_count() - 1)];
  rec.abi_version = kRjLogAbiVersion;
  rec.record_size = kRjLogRecordSize;
  rec.record_type = record_type;
  rec.site = site;
  rec.payload = payload;
  rec.writer_lane = 0;
  rec.valid = 1;
  hdr->write_ptr = index + 1;
}

std::vector<RjLogRecord> drain_all(LogBuffer &buf, DrainStats *stats_out) {
  std::vector<RjLogRecord> out;
  DrainStats stats = buf.drain([&](const RjLogRecord &r) { out.push_back(r); });
  if (stats_out)
    *stats_out = stats;
  return out;
}

TEST(LogAbiTest, StructSizesAndAlignment) {
  EXPECT_EQ(sizeof(RjLogBufferHeader), 256u);
  EXPECT_EQ(alignof(RjLogBufferHeader), 64u);
  EXPECT_EQ(sizeof(RjLogRecord), 64u);
  EXPECT_EQ(alignof(RjLogRecord), 64u);
  EXPECT_EQ(kRjLogRecordSize, 64u);
}

TEST(LogBufferTest, RejectsNonPowerOfTwoSlotCount) {
  for (uint32_t bad : {0u, 3u, 6u, 7u, 100u}) {
    std::string err;
    auto buf = LogBuffer::create_for_host_tests(bad, &err);
    EXPECT_EQ(buf, nullptr) << "slot_count=" << bad;
    EXPECT_FALSE(err.empty()) << "slot_count=" << bad;
  }
}

TEST(LogBufferTest, AcceptsPowerOfTwoSlotCount) {
  for (uint32_t good : {1u, 2u, 8u, 1024u}) {
    std::string err;
    auto buf = LogBuffer::create_for_host_tests(good, &err);
    ASSERT_NE(buf, nullptr) << "slot_count=" << good << " err=" << err;
    EXPECT_EQ(buf->slot_count(), good);
    EXPECT_TRUE(err.empty());
  }
}

TEST(LogBufferTest, HeaderInitialized) {
  auto buf = LogBuffer::create_for_host_tests(8);
  ASSERT_NE(buf, nullptr);
  const RjLogBufferHeader *hdr = buf->header();
  EXPECT_EQ(hdr->magic, kRjLogMagic);
  EXPECT_EQ(hdr->abi_version, kRjLogAbiVersion);
  EXPECT_EQ(hdr->header_size, sizeof(RjLogBufferHeader));
  EXPECT_EQ(hdr->record_size, kRjLogRecordSize);
  EXPECT_EQ(hdr->slot_count, 8u);
  EXPECT_EQ(hdr->flags, 0u);
  EXPECT_EQ(hdr->write_ptr, 0u);
  EXPECT_EQ(hdr->read_ptr, 0u);
  EXPECT_EQ(hdr->overflow_count, 0u);
}

TEST(LogBufferTest, RecordsRegionFollowsHeader) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  const auto *hdr_end = reinterpret_cast<const char *>(buf->header()) + sizeof(RjLogBufferHeader);
  EXPECT_EQ(reinterpret_cast<const char *>(buf->records()), hdr_end);
  EXPECT_EQ(buf->total_bytes(), sizeof(RjLogBufferHeader) + 4u * sizeof(RjLogRecord));
}

TEST(LogBufferTest, SyntheticDrainInOrder) {
  auto buf = LogBuffer::create_for_host_tests(8);
  ASSERT_NE(buf, nullptr);
  constexpr uint32_t kN = 5;
  for (uint32_t i = 0; i < kN; ++i)
    publish_record(*buf, /*record_type=*/1, /*site=*/0x100 + i, /*payload=*/i);

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), kN);
  EXPECT_EQ(stats.records_drained, kN);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(stats.overflow_count, 0u);
  for (uint32_t i = 0; i < kN; ++i) {
    EXPECT_EQ(drained[i].record_type, 1u);
    EXPECT_EQ(drained[i].site, 0x100u + i);
    EXPECT_EQ(drained[i].payload, i);
  }
  // read_ptr caught up to write_ptr; a second drain yields nothing.
  EXPECT_EQ(buf->header()->read_ptr, buf->header()->write_ptr);
  auto again = drain_all(*buf, nullptr);
  EXPECT_TRUE(again.empty());
}

TEST(LogBufferTest, DrainClearsValidFlags) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);
  drain_all(*buf, nullptr);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(buf->records()[i].valid, 0u) << "slot " << i;
}

TEST(LogBufferTest, WraparoundReusesSlots) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);

  // First fill: indices 0..3 -> slots 0..3.
  for (uint32_t i = 0; i < 4; ++i)
    publish_record(*buf, 1, 0x200 + i, i);
  auto first = drain_all(*buf, nullptr);
  ASSERT_EQ(first.size(), 4u);

  // Second fill: indices 4..7 -> slots 0..3 again.
  for (uint32_t i = 0; i < 4; ++i)
    publish_record(*buf, 2, 0x300 + i, 100 + i);
  DrainStats stats;
  auto second = drain_all(*buf, &stats);
  ASSERT_EQ(second.size(), 4u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(buf->header()->write_ptr, 8u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(second[i].record_type, 2u);
    EXPECT_EQ(second[i].site, 0x300u + i);
    EXPECT_EQ(second[i].payload, 100u + i);
  }
}

TEST(LogBufferTest, OverflowCountSurfaced) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  // Simulate producers that dropped records on a full ring.
  buf->header()->overflow_count = 7;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.overflow_count, 7u);
  EXPECT_EQ(buf->overflow_count(), 7u);
}

TEST(LogBufferTest, InvalidAdvertisedSlotIsDiagnosedNotSpun) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  // write_ptr advertises two records, but only one was actually published.
  publish_record(*buf, 1, 0x10, 42);
  buf->header()->write_ptr = 2; // second slot left with valid == 0

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.invalid_slots, 1u);
  EXPECT_EQ(buf->header()->read_ptr, 2u); // still advances; no spin
}

TEST(LogBufferTest, ReinitializeResetsForReuse) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);
  buf->header()->overflow_count = 3;
  drain_all(*buf, nullptr);

  buf->reinitialize();
  EXPECT_EQ(buf->header()->write_ptr, 0u);
  EXPECT_EQ(buf->header()->read_ptr, 0u);
  EXPECT_EQ(buf->overflow_count(), 0u);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(buf->records()[i].valid, 0u);
  // Header control fields survive reinitialization.
  EXPECT_EQ(buf->header()->magic, kRjLogMagic);
  EXPECT_EQ(buf->header()->slot_count, 4u);

  // Buffer is usable again.
  publish_record(*buf, 9, 0xabc, 5);
  auto drained = drain_all(*buf, nullptr);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].record_type, 9u);
}

TEST(LogBufferTest, EmptyDrainOnFreshBuffer) {
  auto buf = LogBuffer::create_for_host_tests(8);
  ASSERT_NE(buf, nullptr);
  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_TRUE(drained.empty());
  EXPECT_EQ(stats.records_drained, 0u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(stats.overflow_count, 0u);
  EXPECT_EQ(buf->header()->read_ptr, 0u);
  EXPECT_EQ(buf->header()->write_ptr, 0u);
}

TEST(LogBufferTest, RecordsRegionIs64ByteAligned) {
  auto buf = LogBuffer::create_for_host_tests(8);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf->header()) % 64u, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf->records()) % 64u, 0u);
}

TEST(LogBufferTest, NullCallbackStillDrainsAndClears) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);

  DrainStats stats = buf->drain(LogRecordCallback{}); // empty callback
  EXPECT_EQ(stats.records_drained, 2u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(buf->header()->read_ptr, buf->header()->write_ptr);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(buf->records()[i].valid, 0u) << "slot " << i;
}

TEST(LogBufferTest, InvalidThenValidSlotBothHandled) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  // Leave index 0 unpublished (valid == 0), publish a real record at index 1.
  buf->records()[0].valid = 0;
  buf->records()[1] = RjLogRecord{};
  buf->records()[1].record_type = 7;
  buf->records()[1].site = 0x55;
  buf->records()[1].valid = 1;
  buf->header()->write_ptr = 2;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.invalid_slots, 1u);
  EXPECT_EQ(drained[0].record_type, 7u); // the later valid record is delivered
  EXPECT_EQ(drained[0].site, 0x55u);
  EXPECT_EQ(buf->header()->read_ptr, 2u);
}

TEST(LogBufferTest, SingleSlotRingDrainsAcrossCycles) {
  auto buf = LogBuffer::create_for_host_tests(1);
  ASSERT_NE(buf, nullptr);
  for (uint32_t cycle = 0; cycle < 5; ++cycle) {
    publish_record(*buf, 1, 0x100 + cycle, cycle);
    DrainStats stats;
    auto drained = drain_all(*buf, &stats);
    ASSERT_EQ(drained.size(), 1u) << "cycle " << cycle;
    EXPECT_EQ(drained[0].site, 0x100u + cycle);
    EXPECT_EQ(stats.invalid_slots, 0u);
  }
  EXPECT_EQ(buf->header()->write_ptr, 5u);
}

TEST(LogBufferTest, ExactFullDrainAtCapacity) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  for (uint32_t i = 0; i < 4; ++i) // write_ptr - read_ptr == slot_count
    publish_record(*buf, 1, 0x10 + i, i);
  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 4u);
  EXPECT_EQ(stats.invalid_slots, 0u);
}

TEST(LogBufferTest, Uint64IndexWraparound) {
  auto buf = LogBuffer::create_for_host_tests(4);
  ASSERT_NE(buf, nullptr);
  // Position the monotonic counters two below the 64-bit boundary so the third
  // published record wraps write_ptr through zero. Drain must still walk all
  // three in order (count-based iteration is wrap-safe).
  constexpr uint64_t kNearMax = 0xFFFFFFFFFFFFFFFEULL;
  buf->header()->write_ptr = kNearMax;
  buf->header()->read_ptr = kNearMax;
  for (uint32_t i = 0; i < 3; ++i)
    publish_record(*buf, 1, 0xA0 + i, i);
  EXPECT_EQ(buf->header()->write_ptr, 1u); // wrapped: FE -> FF -> 00 -> 01

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 3u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  for (uint32_t i = 0; i < 3; ++i)
    EXPECT_EQ(drained[i].site, 0xA0u + i);
  EXPECT_EQ(buf->header()->read_ptr, 1u);
}

TEST(LogBufferTest, FormatLogLine) {
  RjLogRecord rec{};
  rec.record_type = 1;
  rec.site = 0x140;
  rec.wave_id = 0xabc;
  rec.writer_lane = 0;
  rec.payload = 0xdead;

  const std::string line = format_log_line(rec);
  EXPECT_NE(line.find("rj-log"), std::string::npos) << line;
  EXPECT_NE(line.find("record_type=1"), std::string::npos) << line;
  EXPECT_NE(line.find("site=0x140"), std::string::npos) << line;
  EXPECT_NE(line.find("wave=0xabc"), std::string::npos) << line;
  EXPECT_NE(line.find("lane=0"), std::string::npos) << line;
  EXPECT_NE(line.find("payload=0xdead"), std::string::npos) << line;
}

TEST(LogBufferTest, FormatLogLineMaxValues) {
  RjLogRecord rec{};
  rec.record_type = 0xffffffffu;
  rec.site = 0xffffffffu;
  rec.wave_id = 0xffffffffu;
  rec.writer_lane = 0xffffffffu;
  rec.payload = 0xffffffffffffffffULL;
  // Exact-string check locks the field mapping, hex widths, and buffer size.
  EXPECT_EQ(format_log_line(rec), "rj-log record_type=4294967295 site=0xffffffff wave=0xffffffff "
                                  "lane=4294967295 payload=0xffffffffffffffff");
}

} // namespace
} // namespace rocjitsu
