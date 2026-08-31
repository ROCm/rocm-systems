// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/log_buffer.h"

#include "rocjitsu/code/patch/log_abi.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace rocjitsu {

// Test-only accessor. LogBuffer keeps its factory and raw header/record pointers
// private because no production consumer defines their contract yet; the tests
// reach those seams through this friend shim rather than through a public API
// that would freeze prematurely. Must live in namespace rocjitsu (not the
// anonymous namespace below) so it names the same type as the friend
// declaration in LogBuffer.
struct LogBufferTestAccess {
  static std::unique_ptr<LogBuffer> create(uint32_t slot_count, std::string *error_out = nullptr) {
    return LogBuffer::create(slot_count, error_out);
  }
  static RjLogBufferHeader *header(LogBuffer &buf) { return buf.header(); }
  static RjLogRecord *records(LogBuffer &buf) { return buf.records(); }
};

namespace {

// Simulate what a device producer does: write a record into the slot at the
// current write_ptr, publish valid, then advance write_ptr. Host-side stand-in
// used by the CPU-only tests (there is no atomic arbitration to model here
// because these tests are single-threaded and post-completion).
void publish_record(LogBuffer &buf, uint32_t record_type, uint32_t site, uint64_t payload) {
  RjLogBufferHeader *hdr = LogBufferTestAccess::header(buf);
  RjLogRecord *recs = LogBufferTestAccess::records(buf);
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
    auto buf = LogBufferTestAccess::create(bad, &err);
    EXPECT_EQ(buf, nullptr) << "slot_count=" << bad;
    EXPECT_FALSE(err.empty()) << "slot_count=" << bad;
  }
}

TEST(LogBufferTest, AcceptsPowerOfTwoSlotCount) {
  for (uint32_t good : {1u, 2u, 8u, 1024u}) {
    std::string err;
    auto buf = LogBufferTestAccess::create(good, &err);
    ASSERT_NE(buf, nullptr) << "slot_count=" << good << " err=" << err;
    EXPECT_EQ(buf->slot_count(), good);
    EXPECT_TRUE(err.empty());
  }
}

TEST(LogBufferTest, HeaderInitialized) {
  auto buf = LogBufferTestAccess::create(8);
  ASSERT_NE(buf, nullptr);
  const RjLogBufferHeader *hdr = LogBufferTestAccess::header(*buf);
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
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  const auto *hdr_end =
      reinterpret_cast<const char *>(LogBufferTestAccess::header(*buf)) + sizeof(RjLogBufferHeader);
  EXPECT_EQ(reinterpret_cast<const char *>(LogBufferTestAccess::records(*buf)), hdr_end);
  EXPECT_EQ(buf->total_bytes(), sizeof(RjLogBufferHeader) + 4u * sizeof(RjLogRecord));
}

TEST(LogBufferTest, SyntheticDrainInOrder) {
  auto buf = LogBufferTestAccess::create(8);
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
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr,
            LogBufferTestAccess::header(*buf)->write_ptr);
  auto again = drain_all(*buf, nullptr);
  EXPECT_TRUE(again.empty());
}

TEST(LogBufferTest, DrainClearsValidFlags) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);
  drain_all(*buf, nullptr);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(LogBufferTestAccess::records(*buf)[i].valid, 0u) << "slot " << i;
}

TEST(LogBufferTest, WraparoundReusesSlots) {
  auto buf = LogBufferTestAccess::create(4);
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
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->write_ptr, 8u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(second[i].record_type, 2u);
    EXPECT_EQ(second[i].site, 0x300u + i);
    EXPECT_EQ(second[i].payload, 100u + i);
  }
}

TEST(LogBufferTest, OverflowCountSurfaced) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  // Simulate producers that dropped records on a full ring.
  LogBufferTestAccess::header(*buf)->overflow_count = 7;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.overflow_count, 7u);
  EXPECT_EQ(buf->overflow_count(), 7u);
}

TEST(LogBufferTest, InvalidAdvertisedSlotIsDiagnosedNotSpun) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  // write_ptr advertises two records, but only one was actually published.
  publish_record(*buf, 1, 0x10, 42);
  LogBufferTestAccess::header(*buf)->write_ptr = 2; // second slot left with valid == 0

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.invalid_slots, 1u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr, 2u); // still advances; no spin
}

TEST(LogBufferTest, ReinitializeResetsForReuse) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);
  LogBufferTestAccess::header(*buf)->overflow_count = 3;
  drain_all(*buf, nullptr);

  buf->reinitialize();
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->write_ptr, 0u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr, 0u);
  EXPECT_EQ(buf->overflow_count(), 0u);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(LogBufferTestAccess::records(*buf)[i].valid, 0u);
  // Header control fields survive reinitialization.
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->magic, kRjLogMagic);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->slot_count, 4u);

  // Buffer is usable again.
  publish_record(*buf, 9, 0xabc, 5);
  auto drained = drain_all(*buf, nullptr);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].record_type, 9u);
}

TEST(LogBufferTest, EmptyDrainOnFreshBuffer) {
  auto buf = LogBufferTestAccess::create(8);
  ASSERT_NE(buf, nullptr);
  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_TRUE(drained.empty());
  EXPECT_EQ(stats.records_drained, 0u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(stats.overflow_count, 0u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr, 0u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->write_ptr, 0u);
}

TEST(LogBufferTest, RecordsRegionIs64ByteAligned) {
  auto buf = LogBufferTestAccess::create(8);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(LogBufferTestAccess::header(*buf)) % 64u, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(LogBufferTestAccess::records(*buf)) % 64u, 0u);
}

TEST(LogBufferTest, NullCallbackStillDrainsAndClears) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 0);
  publish_record(*buf, 1, 0x20, 0);

  DrainStats stats = buf->drain(LogRecordCallback{}); // empty callback
  EXPECT_EQ(stats.records_drained, 2u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr,
            LogBufferTestAccess::header(*buf)->write_ptr);
  for (uint32_t i = 0; i < buf->slot_count(); ++i)
    EXPECT_EQ(LogBufferTestAccess::records(*buf)[i].valid, 0u) << "slot " << i;
}

TEST(LogBufferTest, InvalidThenValidSlotBothHandled) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  RjLogRecord *recs = LogBufferTestAccess::records(*buf);
  // Leave index 0 unpublished (valid == 0), publish a real record at index 1.
  recs[0].valid = 0;
  recs[1] = RjLogRecord{};
  recs[1].abi_version = kRjLogAbiVersion;
  recs[1].record_size = kRjLogRecordSize;
  recs[1].record_type = 7;
  recs[1].site = 0x55;
  recs[1].valid = 1;
  LogBufferTestAccess::header(*buf)->write_ptr = 2;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.invalid_slots, 1u);
  EXPECT_EQ(drained[0].record_type, 7u); // the later valid record is delivered
  EXPECT_EQ(drained[0].site, 0x55u);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr, 2u);
}

TEST(LogBufferTest, SingleSlotRingDrainsAcrossCycles) {
  auto buf = LogBufferTestAccess::create(1);
  ASSERT_NE(buf, nullptr);
  for (uint32_t cycle = 0; cycle < 5; ++cycle) {
    publish_record(*buf, 1, 0x100 + cycle, cycle);
    DrainStats stats;
    auto drained = drain_all(*buf, &stats);
    ASSERT_EQ(drained.size(), 1u) << "cycle " << cycle;
    EXPECT_EQ(drained[0].site, 0x100u + cycle);
    EXPECT_EQ(stats.invalid_slots, 0u);
  }
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->write_ptr, 5u);
}

TEST(LogBufferTest, ExactFullDrainAtCapacity) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  for (uint32_t i = 0; i < 4; ++i) // write_ptr - read_ptr == slot_count
    publish_record(*buf, 1, 0x10 + i, i);
  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(drained.size(), 4u);
  EXPECT_EQ(stats.invalid_slots, 0u);
}

TEST(LogBufferTest, Uint64IndexWraparound) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  // Position the monotonic counters two below the 64-bit boundary so the third
  // published record wraps write_ptr through zero. Drain must still walk all
  // three in order (count-based iteration is wrap-safe).
  constexpr uint64_t kNearMax = 0xFFFFFFFFFFFFFFFEULL;
  LogBufferTestAccess::header(*buf)->write_ptr = kNearMax;
  LogBufferTestAccess::header(*buf)->read_ptr = kNearMax;
  for (uint32_t i = 0; i < 3; ++i)
    publish_record(*buf, 1, 0xA0 + i, i);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->write_ptr, 1u); // wrapped: FE -> FF -> 00 -> 01

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 3u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  for (uint32_t i = 0; i < 3; ++i)
    EXPECT_EQ(drained[i].site, 0xA0u + i);
  EXPECT_EQ(LogBufferTestAccess::header(*buf)->read_ptr, 1u);
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

TEST(LogBufferTest, ValidateAcceptsFreshBuffer) {
  auto buf = LogBufferTestAccess::create(8);
  ASSERT_NE(buf, nullptr);
  std::string err;
  EXPECT_TRUE(buf->validate(&err)) << err;
}

TEST(LogBufferTest, ValidateRejectsEachCorruptedSelfDescribingField) {
  // One case per validate() rejection branch. Each corrupts a single
  // self-describing header field on an otherwise-fresh buffer and checks both
  // that validate() fails and that the diagnostic names the offending field.
  // No-capture lambdas so the table entries stay plain function pointers.
  struct Case {
    const char *name;
    void (*corrupt)(RjLogBufferHeader *);
    const char *diagnostic_substr;
  };
  const Case cases[] = {
      {"magic", [](RjLogBufferHeader *h) { h->magic = 0xdeadbeef; }, "magic"},
      {"abi_version", [](RjLogBufferHeader *h) { h->abi_version = kRjLogAbiVersion + 1; },
       "abi_version"},
      {"header_size", [](RjLogBufferHeader *h) { h->header_size = sizeof(RjLogBufferHeader) + 1; },
       "header_size"},
      {"record_size", [](RjLogBufferHeader *h) { h->record_size = kRjLogRecordSize + 1; },
       "record_size"},
      {"slot_count", [](RjLogBufferHeader *h) { h->slot_count += 1; }, "slot_count"},
  };

  for (const Case &c : cases) {
    auto buf = LogBufferTestAccess::create(8);
    ASSERT_NE(buf, nullptr) << c.name;
    c.corrupt(LogBufferTestAccess::header(*buf)); // as if written by an incompatible producer
    std::string err;
    EXPECT_FALSE(buf->validate(&err)) << c.name;
    EXPECT_NE(err.find(c.diagnostic_substr), std::string::npos) << c.name << ": " << err;
  }
}

TEST(LogBufferTest, DrainClampsProducerOverrun) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  for (uint32_t i = 0; i < 4; ++i)
    publish_record(*buf, 1, 0x10 + i, i);
  // Simulate a buggy/untrusted producer that broke the drop-newest contract and
  // advanced write_ptr past read_ptr + slot_count. Without clamping, indices 4
  // and 5 would alias slots 0 and 1 and deliver those records a second time.
  // This checks only the clamp's guarantees -- no double-delivery and the excess
  // reported in dropped_overrun. Survivor ordering after a contract violation is
  // undefined by design and deliberately not asserted here.
  LogBufferTestAccess::header(*buf)->write_ptr = 6; // pending = 6 > slot_count (4)

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  EXPECT_EQ(stats.dropped_overrun, 2u);
  EXPECT_EQ(stats.records_drained, 4u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  ASSERT_EQ(drained.size(), 4u);
  // Assert contents order-independently
  std::set<uint32_t> sites;
  for (const RjLogRecord &record : drained)
    sites.insert(record.site);
  EXPECT_EQ(sites.size(), 4u);
  EXPECT_EQ(sites, (std::set<uint32_t>{0x10u, 0x11u, 0x12u, 0x13u}));
}

TEST(LogBufferTest, DrainSkipsIncompatibleAbiVersion) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 1);
  publish_record(*buf, 2, 0x20, 2);
  // A probe built against a newer ABI publishes a record the host cannot decode.
  LogBufferTestAccess::records(*buf)[1].abi_version = kRjLogAbiVersion + 1;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].site, 0x10u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.incompatible_records, 1u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  // Incompatible slots are cleared so a reused ring does not re-see them.
  EXPECT_EQ(LogBufferTestAccess::records(*buf)[1].valid, 0u);
}

TEST(LogBufferTest, DrainSkipsIncompatibleRecordSize) {
  auto buf = LogBufferTestAccess::create(4);
  ASSERT_NE(buf, nullptr);
  publish_record(*buf, 1, 0x10, 1);
  publish_record(*buf, 2, 0x20, 2);
  LogBufferTestAccess::records(*buf)[1].record_size = kRjLogRecordSize * 2;

  DrainStats stats;
  auto drained = drain_all(*buf, &stats);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].site, 0x10u);
  EXPECT_EQ(stats.records_drained, 1u);
  EXPECT_EQ(stats.incompatible_records, 1u);
  EXPECT_EQ(stats.invalid_slots, 0u);
  EXPECT_EQ(LogBufferTestAccess::records(*buf)[1].valid, 0u);
}

} // namespace
} // namespace rocjitsu
