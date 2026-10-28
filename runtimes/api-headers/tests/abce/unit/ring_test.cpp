/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// The reserve / write / commit ring, driven over plain memory with a fake engine.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "abce_test_harness.hpp"

namespace {

using abce_test::FakeRing;
using abce_test::RingOptions;

// Every ring test runs against both queue-control backends.
class Ring : public ::testing::TestWithParam<bool> {
 protected:
  RingOptions WithBackend(RingOptions options = RingOptions()) const {
    options.use_queue_ops = GetParam();
    return options;
  }
};

INSTANTIATE_TEST_SUITE_P(Backends, Ring, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "QueueOps" : "MappedWords";
                         });

abce_ring_reservation_t AcquireAndRelease(FakeRing& fake, uint32_t bytes, uint8_t fill_byte) {
  abce_ring_reservation_t reservation{};
  char* buffer = nullptr;
  EXPECT_EQ(abce_ring_acquire(fake.ring(), bytes, &reservation, &buffer), ABCE_STATUS_OK);
  std::memset(buffer, fill_byte, bytes);
  EXPECT_EQ(abce_ring_release(fake.ring(), &reservation), ABCE_STATUS_OK);
  return reservation;
}

//===----------------------------------------------------------------------===//
// Single producer
//===----------------------------------------------------------------------===//

TEST_P(Ring, PublishAdvancesTheWritePointerAndRingsTheDoorbell) {
  FakeRing fake(WithBackend());
  AcquireAndRelease(fake, 64, 0xaa);
  EXPECT_EQ(fake.write_index(), 64u);
  EXPECT_EQ(fake.doorbell(), 64u);
  AcquireAndRelease(fake, 32, 0xbb);
  EXPECT_EQ(fake.write_index(), 96u);
  EXPECT_EQ(fake.doorbell_history(), (std::vector<uint64_t>{64, 96}));
}

TEST_P(Ring, AcquiredRegionIsZeroedEvenOverDirtyRingMemory) {
  FakeRing fake(WithBackend());
  std::memset(fake.base(), 0xab, fake.size_bytes());
  abce_ring_reservation_t reservation{};
  char* buffer = nullptr;
  ASSERT_EQ(abce_ring_acquire(fake.ring(), 128, &reservation, &buffer), ABCE_STATUS_OK);
  for (uint32_t byte_idx = 0; byte_idx < 128; ++byte_idx)
    ASSERT_EQ(buffer[byte_idx], 0) << "byte " << byte_idx
                                   << " leaked stale ring content into a builder's zeroed buffer";
  ASSERT_EQ(abce_ring_release(fake.ring(), &reservation), ABCE_STATUS_OK);
}

TEST_P(Ring, AttachingToAQueueInUseSeedsTheCursorsFromItsWritePointer) {
  RingOptions options;
  options.size_bytes = 256;
  options.initial_write_index = 1000;
  FakeRing fake(WithBackend(options));
  abce_ring_reservation_t reservation{};
  char* buffer = nullptr;
  ASSERT_EQ(abce_ring_acquire(fake.ring(), 16, &reservation, &buffer), ABCE_STATUS_OK);
  EXPECT_EQ(reservation.start, 1000u) << "handed out bytes the engine may still own";
  EXPECT_EQ(buffer, fake.base() + (1000 & 255));
  ASSERT_EQ(abce_ring_release(fake.ring(), &reservation), ABCE_STATUS_OK);
  EXPECT_EQ(fake.write_index(), 1016u);
}

//===----------------------------------------------------------------------===//
// Sizing
//===----------------------------------------------------------------------===//

TEST_P(Ring, PaddedSizeAppliesTheMinimumAndTheSixtyFourByteRounding) {
  RingOptions options;
  options.min_submission_size = 64;
  FakeRing minimum_only(WithBackend(options));
  uint32_t reserve_bytes = 0;
  ASSERT_EQ(abce_ring_padded_size(minimum_only.ring(), 8, &reserve_bytes), ABCE_STATUS_OK);
  EXPECT_EQ(reserve_bytes, 64u);
  ASSERT_EQ(abce_ring_padded_size(minimum_only.ring(), 100, &reserve_bytes), ABCE_STATUS_OK);
  EXPECT_EQ(reserve_bytes, 100u);

  options.min_submission_size = 0;
  options.align64 = true;
  FakeRing rounded(WithBackend(options));
  ASSERT_EQ(abce_ring_padded_size(rounded.ring(), 65, &reserve_bytes), ABCE_STATUS_OK);
  EXPECT_EQ(reserve_bytes, 128u);
  ASSERT_EQ(abce_ring_padded_size(rounded.ring(), 64, &reserve_bytes), ABCE_STATUS_OK);
  EXPECT_EQ(reserve_bytes, 64u);
}

TEST_P(Ring, AnEmptyPayloadHasNoPaddedSize) {
  FakeRing fake(WithBackend());
  uint32_t reserve_bytes = 0;
  EXPECT_EQ(abce_ring_padded_size(fake.ring(), 0, &reserve_bytes), ABCE_STATUS_INVALID_ARGUMENT);
}

TEST_P(Ring, ARequestAtOrAboveTheRingSizeCanNeverFit) {
  RingOptions options;
  options.size_bytes = 256;
  FakeRing fake(WithBackend(options));
  uint32_t reserve_bytes = 0;
  EXPECT_EQ(abce_ring_padded_size(fake.ring(), 256, &reserve_bytes), ABCE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ(abce_ring_padded_size(fake.ring(), 255, &reserve_bytes), ABCE_STATUS_OK);

  abce_ring_reservation_t reservation{};
  char* buffer = nullptr;
  EXPECT_EQ(abce_ring_acquire(fake.ring(), 256, &reservation, &buffer), ABCE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_TRUE(fake.doorbell_history().empty()) << "a rejected request must not move the ring";
}

//===----------------------------------------------------------------------===//
// Wrap
//===----------------------------------------------------------------------===//

// README section 8, verified there on the same 256-byte ring.
TEST_P(Ring, APayloadThatWouldStraddleTheEndIsPrecededByASeparateNopPad) {
  RingOptions options;
  options.size_bytes = 256;
  FakeRing fake(WithBackend(options));
  AcquireAndRelease(fake, 200, 0xaa);
  const abce_ring_reservation_t wrapped = AcquireAndRelease(fake, 100, 0xbb);

  EXPECT_EQ(wrapped.start, 256u) << "payload lands at the start of the ring, offset 0";
  EXPECT_EQ(fake.write_index(), 356u);
  EXPECT_EQ(fake.doorbell(), 356u);
  EXPECT_EQ(fake.doorbell_history(), (std::vector<uint64_t>{200, 256, 356}))
      << "the pad is published as its own region before the payload";

  const std::vector<uint8_t>& stream = fake.stream();
  ASSERT_EQ(stream.size(), 356u);
  for (size_t byte_idx = 0; byte_idx < 200; ++byte_idx) ASSERT_EQ(stream[byte_idx], 0xaa);
  for (size_t byte_idx = 200; byte_idx < 256; ++byte_idx)
    ASSERT_EQ(stream[byte_idx], 0x00) << "the 56-byte pad must be NOPs";
  for (size_t byte_idx = 256; byte_idx < 356; ++byte_idx) ASSERT_EQ(stream[byte_idx], 0xbb);
}

TEST_P(Ring, ExactlyFillingTheRingDoesNotPad) {
  RingOptions options;
  options.size_bytes = 256;
  FakeRing fake(WithBackend(options));
  AcquireAndRelease(fake, 128, 0x11);
  AcquireAndRelease(fake, 128, 0x22);
  EXPECT_EQ(fake.doorbell_history(), (std::vector<uint64_t>{128, 256}));
}

//===----------------------------------------------------------------------===//
// Back-pressure
//===----------------------------------------------------------------------===//

TEST_P(Ring, AProducerWaitsForTheEngineToFreeSpaceAndThenProceeds) {
  RingOptions options;
  options.size_bytes = 256;
  options.engine_consumes_instantly = false;
  FakeRing fake(WithBackend(options));
  AcquireAndRelease(fake, 200, 0xaa);

  std::atomic<bool> second_acquired{false};
  std::thread producer([&] {
    AcquireAndRelease(fake, 100, 0xbb);
    second_acquired.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(second_acquired.load()) << "no space is free until the engine consumes";

  fake.EngineConsumeAll();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!second_acquired.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_TRUE(second_acquired.load());
  fake.EngineConsumeAll();
  producer.join();
  EXPECT_EQ(fake.write_index(), 356u);
}

TEST_P(Ring, DrainReturnsOnceTheEngineHasConsumedEverythingPublished) {
  RingOptions options;
  options.engine_consumes_instantly = false;
  FakeRing fake(WithBackend(options));
  AcquireAndRelease(fake, 64, 0x33);

  std::atomic<bool> drained{false};
  std::thread waiter([&] {
    EXPECT_EQ(abce_ring_drain(fake.ring()), ABCE_STATUS_OK);
    drained.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(drained.load());
  fake.EngineConsumeAll();
  waiter.join();
  EXPECT_TRUE(drained.load());
}

//===----------------------------------------------------------------------===//
// Multi-producer
//===----------------------------------------------------------------------===//

// Producers write disjoint regions with no lock but must publish in reservation
// order. The checks are the ring's contract: the doorbell only ever advances,
// every record arrives exactly once and untorn, and a producer's own records
// stay in order.
TEST_P(Ring, ConcurrentProducersPublishEveryRecordOnceAndInOrder) {
#if defined(__SANITIZE_THREAD__)
  if (!GetParam())
    GTEST_SKIP() << "Ring memory is reused on the engine's say-so, and ThreadSanitizer can only "
                    "follow that through the queue-ops callbacks, where the harness annotates it.";
#endif  // __SANITIZE_THREAD__
  constexpr uint32_t kNumProducers = 4;
  constexpr uint32_t kRecordsPerProducer = 3000;
  constexpr uint32_t kRecordHeaderDwords = 3;
  constexpr uint32_t kProducerMagic = 0xab000000u;

  RingOptions options;
  options.size_bytes = 4096;  // small enough that wraps and back-pressure both occur
  FakeRing fake(WithBackend(options));

  std::vector<std::thread> producers;
  for (uint32_t producer_id = 0; producer_id < kNumProducers; ++producer_id) {
    producers.emplace_back([&fake, producer_id] {
      for (uint32_t record_idx = 0; record_idx < kRecordsPerProducer; ++record_idx) {
        // Eight sizes, 16..128 bytes, so reservations are uneven.
        const uint32_t payload_dwords = 4 + 4 * ((record_idx + producer_id) % 8);
        const uint32_t payload_bytes = payload_dwords * sizeof(uint32_t);
        abce_ring_reservation_t reservation{};
        char* buffer = nullptr;
        ASSERT_EQ(abce_ring_acquire(fake.ring(), payload_bytes, &reservation, &buffer),
                  ABCE_STATUS_OK);
        uint32_t* words = reinterpret_cast<uint32_t*>(buffer);
        words[0] = kProducerMagic | producer_id;
        words[1] = record_idx;
        words[2] = payload_dwords;
        for (uint32_t dword_idx = kRecordHeaderDwords; dword_idx < payload_dwords; ++dword_idx)
          words[dword_idx] = record_idx * 2654435761u + dword_idx;
        ASSERT_EQ(abce_ring_release(fake.ring(), &reservation), ABCE_STATUS_OK);
      }
    });
  }
  for (std::thread& producer : producers) producer.join();

  const std::vector<uint64_t>& doorbells = fake.doorbell_history();
  ASSERT_FALSE(doorbells.empty());
  for (size_t ring_idx = 1; ring_idx < doorbells.size(); ++ring_idx)
    ASSERT_LT(doorbells[ring_idx - 1], doorbells[ring_idx]) << "doorbell went backwards at " << ring_idx;
  EXPECT_EQ(doorbells.back(), fake.write_index());

  // Walk the stream: a zero dword is wrap-pad NOP, anything else starts a record.
  const std::vector<uint8_t>& stream = fake.stream();
  ASSERT_EQ(stream.size() % sizeof(uint32_t), 0u);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(stream.data());
  const size_t total_dwords = stream.size() / sizeof(uint32_t);
  std::vector<uint32_t> next_expected_record(kNumProducers, 0);
  size_t cursor = 0;
  size_t records_seen = 0;
  while (cursor < total_dwords) {
    if (words[cursor] == 0) {
      ++cursor;
      continue;
    }
    ASSERT_EQ(words[cursor] & 0xff000000u, kProducerMagic) << "torn or misplaced record at dword " << cursor;
    const uint32_t producer_id = words[cursor] & 0xffffffu;
    ASSERT_LT(producer_id, kNumProducers);
    const uint32_t record_idx = words[cursor + 1];
    const uint32_t payload_dwords = words[cursor + 2];
    ASSERT_EQ(record_idx, next_expected_record[producer_id]) << "producer " << producer_id << " reordered";
    ASSERT_LE(cursor + payload_dwords, total_dwords);
    for (uint32_t dword_idx = kRecordHeaderDwords; dword_idx < payload_dwords; ++dword_idx)
      ASSERT_EQ(words[cursor + dword_idx], record_idx * 2654435761u + dword_idx)
          << "payload torn in record " << record_idx << " of producer " << producer_id;
    ++next_expected_record[producer_id];
    ++records_seen;
    cursor += payload_dwords;
  }
  EXPECT_EQ(records_seen, size_t{kNumProducers} * kRecordsPerProducer);
  for (uint32_t producer_id = 0; producer_id < kNumProducers; ++producer_id)
    EXPECT_EQ(next_expected_record[producer_id], kRecordsPerProducer);
}

}  // namespace
