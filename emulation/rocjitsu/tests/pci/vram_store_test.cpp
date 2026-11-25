// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/vram_store.h"

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <latch>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kMemoryBytes = 0x2000;
constexpr uint64_t kApertureBytes = 0x1000;

/// @brief Bytes that differ from their neighbours, from other seeds, and from
///        the zeroes memory starts with.
/// @details Never zero, so that a store which silently skipped a byte of fresh
/// memory cannot pass for one that wrote it.
std::vector<std::byte> patterned_bytes(std::size_t length, uint32_t seed) {
  std::vector<std::byte> bytes(length);
  for (std::size_t index = 0; index < length; ++index) {
    bytes[index] =
        static_cast<std::byte>((seed * 131u + static_cast<uint32_t>(index) * 7u) % 255u + 1u);
  }
  return bytes;
}

/// @brief Record in @p model what a store of @p bytes at @p at should leave.
void overlay(std::vector<std::byte> &model, std::span<const std::byte> bytes, uint64_t at) {
  std::ranges::copy(bytes, model.begin() + static_cast<std::ptrdiff_t>(at));
}

/// @brief Compare two byte ranges, naming the first offset at which they differ.
/// @details A whole-memory comparison that printed its operands would show the
/// first few bytes of each and hide where they part.
::testing::AssertionResult same_bytes(std::span<const std::byte> observed,
                                      std::span<const std::byte> expected) {
  if (observed.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << std::format("{} bytes instead of {}", observed.size(), expected.size());
  }
  for (std::size_t offset = 0; offset < expected.size(); ++offset) {
    if (observed[offset] != expected[offset]) {
      return ::testing::AssertionFailure()
             << std::format("byte {:#x} is {:#04x} instead of {:#04x}", offset,
                            std::to_integer<uint32_t>(observed[offset]),
                            std::to_integer<uint32_t>(expected[offset]));
    }
  }
  return ::testing::AssertionSuccess();
}

/// @brief A usable store, half of which lies past the mapped window.
/// @details Memory twice the size of the aperture gives every test a range
/// inside the window, a range past it, and the boundary between the two.
class VramStorePlainAccess : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(vram_.usable());
    void *mapped =
        ::mmap(nullptr, kApertureBytes, PROT_READ | PROT_WRITE, MAP_SHARED, vram_.fd(), 0);
    ASSERT_NE(mapped, MAP_FAILED);
    peer_window_ = std::span(static_cast<std::byte *>(mapped), kApertureBytes);
  }

  void TearDown() override {
    if (!peer_window_.empty()) {
      ::munmap(peer_window_.data(), peer_window_.size());
    }
  }

  /// @brief Read memory through the shared descriptor, as a transport's client
  ///        that was handed it would.
  [[nodiscard]] bool read_descriptor(std::span<std::byte> bytes, uint64_t at) const {
    return ::pread(vram_.fd(), bytes.data(), bytes.size(), static_cast<off_t>(at)) ==
           static_cast<ssize_t>(bytes.size());
  }

  /// @brief Store into memory through the shared descriptor.
  [[nodiscard]] bool write_descriptor(std::span<const std::byte> bytes, uint64_t at) const {
    return ::pwrite(vram_.fd(), bytes.data(), bytes.size(), static_cast<off_t>(at)) ==
           static_cast<ssize_t>(bytes.size());
  }

  /// @brief Check all of memory against @p model by both routes a peer has to
  ///        it: the descriptor, and the mapped window.
  ::testing::AssertionResult memory_matches(std::span<const std::byte> model) const {
    std::vector<std::byte> through_descriptor(kMemoryBytes);
    if (!read_descriptor(through_descriptor, 0)) {
      return ::testing::AssertionFailure() << "cannot read memory through the descriptor";
    }
    ::testing::AssertionResult descriptor_result = same_bytes(through_descriptor, model);
    if (!descriptor_result) {
      return descriptor_result << ", through the descriptor";
    }
    ::testing::AssertionResult mapping_result =
        same_bytes(peer_window_, model.first(kApertureBytes));
    if (!mapping_result) {
      return mapping_result << ", through the mapping";
    }
    return ::testing::AssertionSuccess();
  }

  rocjitsu::VramStore vram_{"plain-access-vram", kMemoryBytes, kApertureBytes};
  /// @brief This test's own mapping of the aperture, as a peer handed the
  ///        descriptor would make. Only for use while no other thread runs.
  std::span<std::byte> peer_window_;
};

TEST_F(VramStorePlainAccess, WordsInsideTheApertureRoundTripAtEveryAlignment) {
  std::vector<std::byte> model(kMemoryBytes);
  constexpr uint64_t kFirstWord = 0x340;
  constexpr uint64_t kSpacing = 0x10;
  for (uint64_t alignment = 0; alignment < sizeof(uint64_t); ++alignment) {
    const uint64_t offset = kFirstWord + alignment * kSpacing + alignment;
    const uint32_t value = 0x89abcdefu + static_cast<uint32_t>(alignment) * 0x01010101u;
    ASSERT_TRUE(vram_.write(offset, value)) << "alignment " << alignment;
    overlay(model, std::as_bytes(std::span(&value, 1)), offset);

    uint32_t read_back = 0;
    ASSERT_TRUE(vram_.read(offset, read_back)) << "alignment " << alignment;
    EXPECT_EQ(read_back, value) << "alignment " << alignment;
  }

  constexpr uint32_t kFirstValue = 0x0badf00d;
  constexpr uint32_t kLastValue = 0x600dcafe;
  constexpr uint64_t kLastWordInAperture = kApertureBytes - sizeof(uint32_t);
  ASSERT_TRUE(vram_.write(0, kFirstValue));
  ASSERT_TRUE(vram_.write(kLastWordInAperture, kLastValue));
  overlay(model, std::as_bytes(std::span(&kFirstValue, 1)), 0);
  overlay(model, std::as_bytes(std::span(&kLastValue, 1)), kLastWordInAperture);
  uint32_t read_back = 0;
  ASSERT_TRUE(vram_.read(0, read_back));
  EXPECT_EQ(read_back, kFirstValue);
  ASSERT_TRUE(vram_.read(kLastWordInAperture, read_back));
  EXPECT_EQ(read_back, kLastValue);

  EXPECT_TRUE(memory_matches(model));
}

TEST_F(VramStorePlainAccess, BuffersOfEveryShortLengthAndAlignmentRoundTripInsideTheAperture) {
  constexpr uint64_t kRegion = 0x200;
  constexpr std::size_t kLongestBuffer = 40;
  // Read back this far either side of the region, to see that a store of any
  // shape leaves what lies beside it alone.
  constexpr std::size_t kMargin = 16;
  constexpr std::size_t kWindowBytes = kLongestBuffer + sizeof(uint64_t) + 2 * kMargin;

  std::vector<std::byte> model(kMemoryBytes);
  std::vector<std::byte> window(kWindowBytes);
  uint32_t seed = 0;
  for (uint64_t alignment = 0; alignment < sizeof(uint64_t); ++alignment) {
    for (std::size_t length = 0; length <= kLongestBuffer; ++length) {
      const uint64_t at = kRegion + alignment;
      const std::vector<std::byte> stored = patterned_bytes(length, ++seed);
      ASSERT_TRUE(vram_.write_all(stored, at)) << "alignment " << alignment << " length " << length;
      overlay(model, stored, at);

      ASSERT_TRUE(vram_.read_all(window, kRegion - kMargin));
      ASSERT_TRUE(same_bytes(window, std::span(model).subspan(kRegion - kMargin, kWindowBytes)))
          << "alignment " << alignment << " length " << length;
    }
  }
  EXPECT_TRUE(memory_matches(model));
}

TEST_F(VramStorePlainAccess, ReadsInsideTheApertureObserveStoresMadeByAPeer) {
  constexpr uint64_t kMappedStoreAt = 0x503;
  const std::vector<std::byte> mapped_store = patterned_bytes(37, 1);
  std::ranges::copy(mapped_store, peer_window_.begin() + kMappedStoreAt);
  std::vector<std::byte> observed(mapped_store.size());
  ASSERT_TRUE(vram_.read_all(observed, kMappedStoreAt));
  EXPECT_TRUE(same_bytes(observed, mapped_store));

  constexpr uint64_t kDescriptorStoreAt = 0x905;
  const std::vector<std::byte> descriptor_store = patterned_bytes(37, 2);
  ASSERT_TRUE(write_descriptor(descriptor_store, kDescriptorStoreAt));
  ASSERT_TRUE(vram_.read_all(observed, kDescriptorStoreAt));
  EXPECT_TRUE(same_bytes(observed, descriptor_store));

  uint32_t expected_word = 0;
  std::memcpy(&expected_word, descriptor_store.data() + 1, sizeof(expected_word));
  uint32_t observed_word = 0;
  ASSERT_TRUE(vram_.read(kDescriptorStoreAt + 1, observed_word));
  EXPECT_EQ(observed_word, expected_word);
}

TEST_F(VramStorePlainAccess, WholeApertureAndWholeMemoryRoundTrip) {
  std::vector<std::byte> model(kMemoryBytes);
  const std::vector<std::byte> whole_aperture = patterned_bytes(kApertureBytes, 5);
  ASSERT_TRUE(vram_.write_all(whole_aperture, 0));
  overlay(model, whole_aperture, 0);
  std::vector<std::byte> observed_aperture(kApertureBytes);
  ASSERT_TRUE(vram_.read_all(observed_aperture, 0));
  EXPECT_TRUE(same_bytes(observed_aperture, whole_aperture));
  EXPECT_TRUE(memory_matches(model));

  const std::vector<std::byte> whole_memory = patterned_bytes(kMemoryBytes, 6);
  ASSERT_TRUE(vram_.write_all(whole_memory, 0));
  std::vector<std::byte> observed_memory(kMemoryBytes);
  ASSERT_TRUE(vram_.read_all(observed_memory, 0));
  EXPECT_TRUE(same_bytes(observed_memory, whole_memory));
  EXPECT_TRUE(memory_matches(whole_memory));
}

TEST_F(VramStorePlainAccess, RangesStraddlingTheApertureEndReachBothSidesOfIt) {
  std::vector<std::byte> model(kMemoryBytes);

  constexpr uint64_t kBufferAt = kApertureBytes - 11;
  const std::vector<std::byte> straddling = patterned_bytes(24, 7);
  ASSERT_TRUE(vram_.write_all(straddling, kBufferAt));
  overlay(model, straddling, kBufferAt);
  std::vector<std::byte> observed(straddling.size());
  ASSERT_TRUE(vram_.read_all(observed, kBufferAt));
  EXPECT_TRUE(same_bytes(observed, straddling));
  EXPECT_TRUE(memory_matches(model));

  constexpr uint64_t kWordAt = kApertureBytes - 2;
  constexpr uint32_t kWord = 0x5a6b7c8d;
  ASSERT_TRUE(vram_.write(kWordAt, kWord));
  overlay(model, std::as_bytes(std::span(&kWord, 1)), kWordAt);
  uint32_t read_back = 0;
  ASSERT_TRUE(vram_.read(kWordAt, read_back));
  EXPECT_EQ(read_back, kWord);
  EXPECT_TRUE(memory_matches(model));
}

TEST_F(VramStorePlainAccess, RangesInsideAndAcrossTheApertureEndObserveEachOthersStores) {
  constexpr uint64_t kLastWordInAperture = kApertureBytes - sizeof(uint32_t);
  constexpr uint32_t kInside = 0x11223344;
  constexpr uint32_t kOutside = 0x55667788;
  ASSERT_TRUE(vram_.write(kLastWordInAperture, kInside));
  ASSERT_TRUE(vram_.write(kApertureBytes, kOutside));
  std::array<uint32_t, 2> across{};
  ASSERT_TRUE(vram_.read_all(std::as_writable_bytes(std::span(across)), kLastWordInAperture));
  EXPECT_EQ(across[0], kInside);
  EXPECT_EQ(across[1], kOutside);

  constexpr std::array<uint32_t, 2> kReplacement{0x99aabbcc, 0xddeeff00};
  ASSERT_TRUE(vram_.write_all(std::as_bytes(std::span(kReplacement)), kLastWordInAperture));
  uint32_t inside = 0;
  uint32_t outside = 0;
  ASSERT_TRUE(vram_.read(kLastWordInAperture, inside));
  ASSERT_TRUE(vram_.read(kApertureBytes, outside));
  EXPECT_EQ(inside, kReplacement[0]);
  EXPECT_EQ(outside, kReplacement[1]);
}

TEST_F(VramStorePlainAccess, RangesBeyondTheApertureRoundTripAndAreSharedWithTheDescriptor) {
  std::vector<std::byte> model(kMemoryBytes);

  constexpr uint64_t kLastWordInMemory = kMemoryBytes - sizeof(uint32_t);
  constexpr std::array<uint64_t, 4> kWordOffsets{kApertureBytes, 0x1801, 0x1c06, kLastWordInMemory};
  for (const uint64_t offset : kWordOffsets) {
    const uint32_t value = 0xc0de0000u + static_cast<uint32_t>(offset);
    ASSERT_TRUE(vram_.write(offset, value)) << std::format("offset {:#x}", offset);
    overlay(model, std::as_bytes(std::span(&value, 1)), offset);
    uint32_t read_back = 0;
    ASSERT_TRUE(vram_.read(offset, read_back)) << std::format("offset {:#x}", offset);
    EXPECT_EQ(read_back, value) << std::format("offset {:#x}", offset);
  }

  constexpr uint64_t kBufferAt = 0x1403;
  const std::vector<std::byte> buffer = patterned_bytes(53, 8);
  ASSERT_TRUE(vram_.write_all(buffer, kBufferAt));
  overlay(model, buffer, kBufferAt);
  std::vector<std::byte> observed(buffer.size());
  ASSERT_TRUE(vram_.read_all(observed, kBufferAt));
  EXPECT_TRUE(same_bytes(observed, buffer));

  const std::vector<std::byte> up_to_the_end = patterned_bytes(19, 9);
  const uint64_t tail_at = kMemoryBytes - up_to_the_end.size();
  ASSERT_TRUE(vram_.write_all(up_to_the_end, tail_at));
  overlay(model, up_to_the_end, tail_at);
  std::vector<std::byte> observed_tail(up_to_the_end.size());
  ASSERT_TRUE(vram_.read_all(observed_tail, tail_at));
  EXPECT_TRUE(same_bytes(observed_tail, up_to_the_end));

  // Nothing above stored inside the aperture, which the model leaves at zero.
  EXPECT_TRUE(memory_matches(model));

  constexpr uint64_t kDescriptorStoreAt = 0x1a07;
  const std::vector<std::byte> descriptor_store = patterned_bytes(buffer.size(), 10);
  ASSERT_TRUE(write_descriptor(descriptor_store, kDescriptorStoreAt));
  ASSERT_TRUE(vram_.read_all(observed, kDescriptorStoreAt));
  EXPECT_TRUE(same_bytes(observed, descriptor_store));
}

TEST_F(VramStorePlainAccess, EmptyRangesAreAcceptedAnywhereInMemoryAndStoreNothing) {
  // A caller with nothing to transfer hands over a span with no storage behind
  // it, so the store may not touch the pointer either.
  constexpr std::array<uint64_t, 5> kOffsetsInMemory{0, 0x7ff, kApertureBytes, kApertureBytes + 1,
                                                     kMemoryBytes};
  for (const uint64_t at : kOffsetsInMemory) {
    EXPECT_TRUE(vram_.read_all(std::span<std::byte>{}, at)) << std::format("at {:#x}", at);
    EXPECT_TRUE(vram_.write_all(std::span<const std::byte>{}, at)) << std::format("at {:#x}", at);
  }
  EXPECT_FALSE(vram_.read_all(std::span<std::byte>{}, kMemoryBytes + 1));
  EXPECT_FALSE(vram_.write_all(std::span<const std::byte>{}, kMemoryBytes + 1));

  EXPECT_TRUE(memory_matches(std::vector<std::byte>(kMemoryBytes)));
}

TEST_F(VramStorePlainAccess, RangesLeavingMemoryAreRefusedAndStoreNothing) {
  std::vector<std::byte> model(kMemoryBytes);
  const std::vector<std::byte> last_bytes = patterned_bytes(16, 11);
  const uint64_t last_bytes_at = kMemoryBytes - last_bytes.size();
  ASSERT_TRUE(vram_.write_all(last_bytes, last_bytes_at));
  overlay(model, last_bytes, last_bytes_at);

  constexpr uint64_t kWrappingOffset = std::numeric_limits<uint64_t>::max() - 1;
  constexpr std::array<uint64_t, 4> kRefusedWordOffsets{kMemoryBytes - 3, kMemoryBytes,
                                                        kMemoryBytes + 0x1000, kWrappingOffset};
  constexpr uint32_t kRefusedValue = 0xdeadbeef;
  for (const uint64_t offset : kRefusedWordOffsets) {
    uint32_t value = 0;
    EXPECT_FALSE(vram_.read(offset, value)) << std::format("offset {:#x}", offset);
    EXPECT_FALSE(vram_.write(offset, kRefusedValue)) << std::format("offset {:#x}", offset);
  }

  const std::vector<std::byte> refused = patterned_bytes(32, 12);
  std::vector<std::byte> destination(refused.size());
  constexpr std::array<uint64_t, 3> kRefusedBufferOffsets{kMemoryBytes - 31, kMemoryBytes,
                                                          kWrappingOffset};
  for (const uint64_t at : kRefusedBufferOffsets) {
    EXPECT_FALSE(vram_.read_all(destination, at)) << std::format("at {:#x}", at);
    EXPECT_FALSE(vram_.write_all(refused, at)) << std::format("at {:#x}", at);
  }

  // A range that begins inside the aperture and runs on past the end of memory.
  const std::vector<std::byte> oversized = patterned_bytes(kMemoryBytes, 13);
  std::vector<std::byte> oversized_destination(oversized.size());
  EXPECT_FALSE(vram_.read_all(oversized_destination, 1));
  EXPECT_FALSE(vram_.write_all(oversized, 1));

  EXPECT_TRUE(memory_matches(model));
}

TEST_F(VramStorePlainAccess, PlainAndAtomicAccessorsAddressTheSameMemory) {
  constexpr std::array<uint64_t, 2> kWordOffsets{0x600, 0x1600};
  constexpr uint32_t kPlain = 0x11223344;
  constexpr uint32_t kExchanged = 0x55667788;
  constexpr uint32_t kAtomic = 0x99aabbcc;
  for (const uint64_t offset : kWordOffsets) {
    ASSERT_TRUE(vram_.write(offset, kPlain));
    const std::optional<rocjitsu::VramAtomicLoadResult> loaded =
        vram_.atomic_load(offset, sizeof(uint32_t));
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->value, kPlain) << std::format("offset {:#x}", offset);

    const std::optional<rocjitsu::VramAtomicCompareExchangeResult> exchange =
        vram_.compare_exchange(offset, sizeof(uint32_t), kPlain, kExchanged);
    ASSERT_TRUE(exchange.has_value());
    EXPECT_TRUE(exchange->exchanged) << std::format("offset {:#x}", offset);
    uint32_t read_back = 0;
    ASSERT_TRUE(vram_.read(offset, read_back));
    EXPECT_EQ(read_back, kExchanged) << std::format("offset {:#x}", offset);

    ASSERT_TRUE(vram_.atomic_store(offset, sizeof(uint32_t), kAtomic));
    ASSERT_TRUE(vram_.read(offset, read_back));
    EXPECT_EQ(read_back, kAtomic) << std::format("offset {:#x}", offset);
  }
}

TEST(VramStoreUnusable, RefusesEveryPlainAccess) {
  struct Sizing {
    uint64_t bytes;
    uint64_t aperture_bytes;
  };
  constexpr std::array<Sizing, 3> kUnusableSizings{{{0x2000, 0}, {0x1000, 0x2000}, {0, 0}}};
  for (const Sizing &sizing : kUnusableSizings) {
    rocjitsu::VramStore vram("unusable-vram", sizing.bytes, sizing.aperture_bytes);
    ASSERT_FALSE(vram.usable());
    uint32_t value = 0;
    std::array<std::byte, 8> buffer{};
    EXPECT_FALSE(vram.read(0, value));
    EXPECT_FALSE(vram.write(0, 1));
    EXPECT_FALSE(vram.read_all(buffer, 0));
    EXPECT_FALSE(vram.write_all(buffer, 0));
    EXPECT_FALSE(vram.read_all(std::span<std::byte>{}, 0));
    EXPECT_FALSE(vram.write_all(std::span<const std::byte>{}, 0));
  }
}

/// @brief Rounds in a race whose store is exactly the contended word.
constexpr uint32_t kExchangeRaceRounds = 20000;

/// @brief Rounds in a race whose store runs on past the contended word.
/// @details A store that repeated a write is caught only in a round where the
/// exchange lands between the two writes, which is a small fraction of the
/// rounds even with the word placed to favour it. There are enough rounds that
/// a whole run seldom misses.
constexpr uint32_t kLongStoreExchangeRaceRounds = 100000;

/// @brief How long a race goes on for when its rounds are not finished sooner.
/// @details On an idle host they are, many times over. Two threads that share
/// their processors with other busy processes hand a round over only once per
/// scheduler time slice, and the same rounds would then take minutes.
constexpr std::chrono::seconds kExchangeRaceTimeLimit{5};

/// @brief Rounds a race has to get through in that time to have tested anything.
constexpr uint32_t kFewestExchangeRaceRounds = 16;

/// @brief Offset of a four-byte word that is the last of its cache line.
/// @details Where the races below put the contended word when the store under
/// test is longer than the word. A store that repeated a write would then
/// repeat one that crosses into the next line. Such a store needs both lines,
/// so the exchanging processor, which needs only the first, gets in between two
/// of them far more often than between two stores that stay inside one line. A
/// repeat inside one line is so rarely separated that a test relying on it
/// passes nearly every run against a store that is wrong.
constexpr uint64_t last_word_of_the_line_at(uint64_t line_offset) {
  constexpr uint64_t kCacheLineBytes = 64;
  return line_offset + kCacheLineBytes - sizeof(uint32_t);
}

/// @brief Set by the exchanging thread on the value a plain store left.
constexpr uint64_t kExchangedMark = 1;

/// @brief The value the plain store of @p round leaves in the contended word.
/// @details Different in every round and never carrying the mark, so that an
/// exchange can only succeed against the store of its own round and what it
/// leaves is told apart from what the store left.
constexpr uint64_t plain_value_of(uint32_t round, uint32_t width) {
  const uint64_t low_half = uint64_t{round} << 1;
  return width == sizeof(uint64_t) ? (low_half << 32) | low_half : low_half;
}

struct ExchangeRaceTally {
  /// Plain stores that were refused.
  uint32_t refused_stores = 0;
  /// Rounds in which no exchange found the plain value, though it began after
  /// the plain store had returned.
  uint32_t missed_exchanges = 0;
  /// Rounds that did not settle on the exchanged value.
  uint32_t undone_exchanges = 0;
};

/// @brief Race one plain store per round against a compare/exchange on a word
///        that the store covers.
///
/// @details Each round the calling thread stores a fresh value and a second
/// thread, already retrying, exchanges that value for a marked one as soon as
/// it appears. Nothing else writes the word, so the round has to settle on the
/// marked value. A plain store that wrote the word more than once would, when
/// the exchange lands between two of its writes, put the unmarked value back.
///
/// A correct store passes every time. One that repeats a write is caught only
/// in the rounds where the exchange happens to land between the writes, which
/// is why there are many rounds. The race ends early, with the rounds it got
/// through, once @ref kExchangeRaceTimeLimit has passed, and fails if those
/// are fewer than @ref kFewestExchangeRaceRounds.
///
/// The threads hand rounds to each other through atomics alone and, the plain
/// store under test aside, reach the word only through the store's atomic
/// accessors while both are running. Every other check waits for the join.
///
/// @param[in] vram Store holding the contended word.
/// @param[in] word_offset Offset of the contended word, inside the aperture.
/// @param[in] width Width of the contended word in bytes, four or eight.
/// @param[in] rounds Number of rounds to race.
/// @param[in] store_plain Stores the given value over the contended word with
///            a plain write, returning what the store returned.
ExchangeRaceTally
race_plain_store_against_exchange(rocjitsu::VramStore &vram, uint64_t word_offset, uint32_t width,
                                  uint32_t rounds,
                                  const std::function<bool(uint64_t)> &store_plain) {
  // Retrying without a pause keeps an attempt in flight when the plain store
  // lands; yielding now and then keeps a single-processor host from spending a
  // whole time slice on retries that cannot succeed until the other thread runs.
  constexpr uint32_t kAttemptsBetweenYields = 64;

  std::atomic<uint32_t> stored_round{0};
  std::atomic<uint32_t> exchanged_round{0};
  std::atomic<uint32_t> missed_exchanges{0};
  std::atomic<bool> race_is_over{false};

  std::jthread exchanger([&] {
    for (uint32_t round = 1; round <= rounds; ++round) {
      const uint64_t plain = plain_value_of(round, width);
      bool exchanged = false;
      bool store_had_returned = false;
      uint32_t attempts = 0;
      while (!exchanged && !store_had_returned && !race_is_over.load()) {
        // Sampled before the attempt: only an attempt that begins after the
        // plain store has returned is certain to find its value, so only such
        // an attempt failing ends the round as a miss.
        store_had_returned = stored_round.load() >= round;
        const std::optional<rocjitsu::VramAtomicCompareExchangeResult> result =
            vram.compare_exchange(word_offset, width, plain, plain | kExchangedMark);
        exchanged = result.has_value() && result->exchanged;
        if (!exchanged && ++attempts % kAttemptsBetweenYields == 0) {
          std::this_thread::yield();
        }
      }
      // Stopped while waiting for a store that is not going to be made.
      if (!exchanged && !store_had_returned) {
        break;
      }
      if (!exchanged) {
        missed_exchanges.fetch_add(1);
      }
      exchanged_round.store(round);
    }
  });

  ExchangeRaceTally tally;
  uint32_t rounds_raced = 0;
  const auto deadline = std::chrono::steady_clock::now() + kExchangeRaceTimeLimit;
  for (uint32_t round = 1; round <= rounds && std::chrono::steady_clock::now() < deadline;
       ++round) {
    const uint64_t plain = plain_value_of(round, width);
    if (!store_plain(plain)) {
      ++tally.refused_stores;
    }
    stored_round.store(round);
    while (exchanged_round.load() < round) {
      std::this_thread::yield();
    }
    const std::optional<rocjitsu::VramAtomicLoadResult> settled =
        vram.atomic_load(word_offset, width);
    if (!settled.has_value() || settled->value != (plain | kExchangedMark)) {
      ++tally.undone_exchanges;
    }
    rounds_raced = round;
  }
  race_is_over.store(true);
  exchanger.join();
  ::testing::Test::RecordProperty("rounds", std::to_string(rounds_raced));
  EXPECT_GE(rounds_raced, kFewestExchangeRaceRounds)
      << "too few rounds to have tested the store against a concurrent compare/exchange";
  tally.missed_exchanges = missed_exchanges.load();
  return tally;
}

TEST_F(VramStorePlainAccess, PlainWordStoreDoesNotUndoAConcurrentCompareExchange) {
  constexpr uint64_t kWordOffset = 0x100;
  const ExchangeRaceTally tally = race_plain_store_against_exchange(
      vram_, kWordOffset, sizeof(uint32_t), kExchangeRaceRounds,
      [this](uint64_t plain) { return vram_.write(kWordOffset, static_cast<uint32_t>(plain)); });
  EXPECT_EQ(tally.refused_stores, 0u);
  EXPECT_EQ(tally.missed_exchanges, 0u);
  EXPECT_EQ(tally.undone_exchanges, 0u);
}

TEST_F(VramStorePlainAccess, PlainQuadwordStoreDoesNotUndoAConcurrentCompareExchange) {
  constexpr uint64_t kWordOffset = 0x180;
  const ExchangeRaceTally tally = race_plain_store_against_exchange(
      vram_, kWordOffset, sizeof(uint64_t), kExchangeRaceRounds, [this](uint64_t plain) {
        return vram_.write_all(std::as_bytes(std::span(&plain, 1)), kWordOffset);
      });
  EXPECT_EQ(tally.refused_stores, 0u);
  EXPECT_EQ(tally.missed_exchanges, 0u);
  EXPECT_EQ(tally.undone_exchanges, 0u);
}

TEST_F(VramStorePlainAccess, PlainStoreAcrossACacheLineDoesNotUndoACompareExchangeOnItsFirstWord) {
  // Eight bytes that begin on the contended word and end in the next line: the
  // store is as long as a quadword but aligned only for a word.
  constexpr uint64_t kWordOffset = last_word_of_the_line_at(0x100);
  const ExchangeRaceTally tally = race_plain_store_against_exchange(
      vram_, kWordOffset, sizeof(uint32_t), kLongStoreExchangeRaceRounds, [this](uint64_t plain) {
        const std::array<uint32_t, 2> buffer{static_cast<uint32_t>(plain), 0x3333cccc};
        return vram_.write_all(std::as_bytes(std::span(buffer)), kWordOffset);
      });
  EXPECT_EQ(tally.refused_stores, 0u);
  EXPECT_EQ(tally.missed_exchanges, 0u);
  EXPECT_EQ(tally.undone_exchanges, 0u);
}

TEST_F(VramStorePlainAccess, PlainBufferStoreDoesNotUndoACompareExchangeOnAWordItCovers) {
  // The buffer carries one word before the contended one and two after it, so
  // that word is stored along with its neighbours rather than on its own, and
  // the buffer crosses from the contended word's cache line into the next.
  constexpr uint64_t kWordOffset = last_word_of_the_line_at(0x200);
  constexpr uint32_t kWordBefore = 0xaaaa5555;
  constexpr uint32_t kWordAfter = 0x3333cccc;
  const ExchangeRaceTally tally = race_plain_store_against_exchange(
      vram_, kWordOffset, sizeof(uint32_t), kLongStoreExchangeRaceRounds, [this](uint64_t plain) {
        const std::array<uint32_t, 4> buffer{kWordBefore, static_cast<uint32_t>(plain), kWordAfter,
                                             kWordAfter};
        return vram_.write_all(std::as_bytes(std::span(buffer)), kWordOffset - sizeof(uint32_t));
      });
  EXPECT_EQ(tally.refused_stores, 0u);
  EXPECT_EQ(tally.missed_exchanges, 0u);
  EXPECT_EQ(tally.undone_exchanges, 0u);

  std::array<uint32_t, 4> settled{};
  ASSERT_TRUE(
      vram_.read_all(std::as_writable_bytes(std::span(settled)), kWordOffset - sizeof(uint32_t)));
  EXPECT_EQ(settled[0], kWordBefore);
  EXPECT_EQ(settled[2], kWordAfter);
  EXPECT_EQ(settled[3], kWordAfter);
}

TEST_F(VramStorePlainAccess, PlainStoreAcrossTheApertureEndDoesNotUndoACompareExchangeInsideIt) {
  // The contended word is the last of the aperture. The store begins on the
  // word before it and runs on into memory that is reached by file offset, so
  // it is served in two parts, and the part inside the aperture has to be
  // stored as a store wholly inside it is.
  constexpr uint64_t kWordOffset = kApertureBytes - sizeof(uint32_t);
  constexpr uint32_t kWordBefore = 0xaaaa5555;
  constexpr uint32_t kWordAfter = 0x3333cccc;
  const ExchangeRaceTally tally = race_plain_store_against_exchange(
      vram_, kWordOffset, sizeof(uint32_t), kLongStoreExchangeRaceRounds, [this](uint64_t plain) {
        const std::array<uint32_t, 4> buffer{kWordBefore, static_cast<uint32_t>(plain), kWordAfter,
                                             kWordAfter};
        return vram_.write_all(std::as_bytes(std::span(buffer)), kWordOffset - sizeof(uint32_t));
      });
  EXPECT_EQ(tally.refused_stores, 0u);
  EXPECT_EQ(tally.missed_exchanges, 0u);
  EXPECT_EQ(tally.undone_exchanges, 0u);

  std::array<uint32_t, 4> settled{};
  ASSERT_TRUE(
      vram_.read_all(std::as_writable_bytes(std::span(settled)), kWordOffset - sizeof(uint32_t)));
  EXPECT_EQ(settled[0], kWordBefore);
  EXPECT_EQ(settled[2], kWordAfter);
  EXPECT_EQ(settled[3], kWordAfter);
}

TEST_F(VramStorePlainAccess, PlainStoresBesideAWordDoNotDisturbCompareExchangesOnIt) {
  // The counter is the low half of an eight-byte aligned word. The store after
  // it therefore shares that word with it, and the store before it ends on the
  // byte before it, at an address no wider store is aligned for.
  constexpr uint64_t kCounterOffset = 0x408;
  constexpr uint32_t kIncrements = 50000;
  const std::vector<std::byte> before = patterned_bytes(7, 14);
  const std::vector<std::byte> after = patterned_bytes(7, 15);
  const uint64_t before_at = kCounterOffset - before.size();
  const uint64_t after_at = kCounterOffset + sizeof(uint32_t);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> neighbour_passes{0};
  std::atomic<uint64_t> refused_stores{0};
  std::jthread neighbour([&] {
    while (!stop.load()) {
      const bool stored_before = vram_.write_all(before, before_at);
      const bool stored_after = vram_.write_all(after, after_at);
      if (!stored_before || !stored_after) {
        refused_stores.fetch_add(1);
      }
      neighbour_passes.fetch_add(1);
    }
  });
  while (neighbour_passes.load() == 0) {
    std::this_thread::yield();
  }

  // Only this thread ever writes the counter, so each exchange must find the
  // value the previous one left. One that does not was disturbed from outside.
  uint32_t disturbed_exchanges = 0;
  for (uint32_t count = 0; count < kIncrements; ++count) {
    const std::optional<rocjitsu::VramAtomicCompareExchangeResult> result =
        vram_.compare_exchange(kCounterOffset, sizeof(uint32_t), count, count + 1);
    if (!result.has_value() || !result->exchanged) {
      ++disturbed_exchanges;
    }
  }
  stop.store(true);
  neighbour.join();

  EXPECT_EQ(disturbed_exchanges, 0u);
  EXPECT_EQ(refused_stores.load(), 0u);
  uint32_t counter = 0;
  ASSERT_TRUE(vram_.read(kCounterOffset, counter));
  EXPECT_EQ(counter, kIncrements);
  std::vector<std::byte> observed(before.size());
  ASSERT_TRUE(vram_.read_all(observed, before_at));
  EXPECT_TRUE(same_bytes(observed, before));
  ASSERT_TRUE(vram_.read_all(observed, after_at));
  EXPECT_TRUE(same_bytes(observed, after));
}

/// @brief Seed of the pattern that the thread owning range @p owner stores in
///        @p round, different for every owner and round.
constexpr uint32_t range_seed(std::size_t owner, uint32_t round) {
  return static_cast<uint32_t>(owner) * 0x10001u + round;
}

TEST_F(VramStorePlainAccess, ThreadsOnDisjointRangesCompleteAndAgree) {
  constexpr std::size_t kRangeBytes = 61;
  constexpr uint32_t kRounds = 2000;
  // Four ranges packed end to end at an odd address inside the aperture, so
  // that each shares machine words with the next; then one across the aperture
  // end and one past it, which keeps the file path busy at the same time.
  constexpr uint64_t kPackedStart = 0x7f3;
  constexpr uint64_t kPackedEnd = kPackedStart + 4 * kRangeBytes;
  constexpr std::array<uint64_t, 6> kRangeStarts{kPackedStart,
                                                 kPackedStart + kRangeBytes,
                                                 kPackedStart + 2 * kRangeBytes,
                                                 kPackedStart + 3 * kRangeBytes,
                                                 kApertureBytes - 30,
                                                 0x1803};
  // Bytes touching each end of the packed ranges that no thread stores to,
  // read throughout by threads of their own.
  constexpr std::size_t kGuardBytes = 5;
  constexpr std::array<uint64_t, 2> kGuardStarts{kPackedStart - kGuardBytes, kPackedEnd};
  const std::vector<std::byte> guard = patterned_bytes(kGuardBytes, 16);

  std::vector<std::byte> model(kMemoryBytes);
  for (const uint64_t guard_start : kGuardStarts) {
    ASSERT_TRUE(vram_.write_all(guard, guard_start));
    overlay(model, guard, guard_start);
  }

  std::latch start{1};
  std::atomic<uint32_t> refused_accesses{0};
  std::atomic<uint32_t> disagreements{0};
  std::vector<std::jthread> threads;
  for (std::size_t owner = 0; owner < kRangeStarts.size(); ++owner) {
    threads.emplace_back([&, owner] {
      const uint64_t range_start = kRangeStarts[owner];
      std::vector<std::byte> observed(kRangeBytes);
      start.wait();
      for (uint32_t round = 1; round <= kRounds; ++round) {
        const std::vector<std::byte> stored =
            patterned_bytes(kRangeBytes, range_seed(owner, round));
        // The head goes in word by word and the rest as one buffer, with the
        // split moving each round so that every store shape meets every
        // alignment while the neighbouring ranges are being written.
        const std::size_t word_bytes =
            (round % (kRangeBytes / sizeof(uint32_t))) * sizeof(uint32_t);
        bool accepted = true;
        for (std::size_t done = 0; done < word_bytes; done += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, stored.data() + done, sizeof(word));
          accepted = vram_.write(range_start + done, word) && accepted;
        }
        accepted =
            vram_.write_all(std::span(stored).subspan(word_bytes), range_start + word_bytes) &&
            accepted;

        accepted = vram_.read_all(observed, range_start) && accepted;
        const std::size_t probe = round % (kRangeBytes - sizeof(uint32_t));
        uint32_t expected_word = 0;
        std::memcpy(&expected_word, stored.data() + probe, sizeof(expected_word));
        uint32_t observed_word = 0;
        accepted = vram_.read(range_start + probe, observed_word) && accepted;

        if (!accepted) {
          refused_accesses.fetch_add(1);
        }
        if (observed != stored || observed_word != expected_word) {
          disagreements.fetch_add(1);
        }
      }
    });
  }
  for (const uint64_t guard_start : kGuardStarts) {
    threads.emplace_back([&, guard_start] {
      std::vector<std::byte> observed(kGuardBytes);
      start.wait();
      for (uint32_t round = 1; round <= kRounds; ++round) {
        if (!vram_.read_all(observed, guard_start)) {
          refused_accesses.fetch_add(1);
        }
        if (observed != guard) {
          disagreements.fetch_add(1);
        }
      }
    });
  }
  start.count_down();
  for (std::jthread &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(refused_accesses.load(), 0u);
  EXPECT_EQ(disagreements.load(), 0u);
  for (std::size_t owner = 0; owner < kRangeStarts.size(); ++owner) {
    overlay(model, patterned_bytes(kRangeBytes, range_seed(owner, kRounds)), kRangeStarts[owner]);
  }
  EXPECT_TRUE(memory_matches(model));
}

// Dispatch threads reach the same aperture bytes with every accessor at once,
// and at whatever widths the guest's code uses. Every accessor is meant to go
// through util/shared_memory.h, which makes no plain access. A plain load or
// store put back on the mapped path races the other accessors here, and a
// ThreadSanitizer build reports it. In any build the test also checks what an
// aligned access promises: that it never sees part of a store.
TEST_F(VramStorePlainAccess, OverlappingAccessesOfEveryKindAndWidthNeverSplitAnAlignedWord) {
  constexpr uint64_t kQuadword = 0x100;
  constexpr uint64_t kHighWord = kQuadword + sizeof(uint32_t);
  constexpr uint32_t kRounds = 20000;
  // Every value stored repeats one 16-bit pattern in each of its 16-bit lanes,
  // and every store covers whole aligned 32-bit words. So a 32-bit word that
  // was loaded whole has two equal lanes whichever stores it fell between.
  const auto lanes16 = [](uint32_t round) {
    return uint64_t{round & 0xffffu} * 0x0001000100010001u;
  };
  const auto word_is_whole = [](uint64_t word) {
    return (word & 0xffffu) == ((word >> 16) & 0xffffu);
  };
  const auto quadword_is_whole = [&](uint64_t quadword) {
    return word_is_whole(quadword & 0xffffffffu) && word_is_whole(quadword >> 32);
  };

  std::atomic<bool> stop{false};
  std::atomic<uint32_t> refused{0};
  std::atomic<uint32_t> split{0};
  const auto count_unless = [](std::atomic<uint32_t> &counter, bool ok) {
    if (!ok) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::jthread> threads;
  // Plain 32-bit reads of both words, and a misaligned one that promises nothing.
  threads.emplace_back([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      uint32_t low = 0;
      uint32_t high = 0;
      uint32_t straddling = 0;
      count_unless(refused, vram_.read(kQuadword, low) && vram_.read(kHighWord, high) &&
                                vram_.read(kQuadword + 1, straddling));
      count_unless(split, word_is_whole(low) && word_is_whole(high));
    }
  });
  // Buffer reads: the quadword alone, and a longer span that covers it.
  threads.emplace_back([&] {
    std::array<uint64_t, 3> wide{};
    while (!stop.load(std::memory_order_relaxed)) {
      uint64_t quadword = 0;
      count_unless(refused,
                   vram_.read_all(std::as_writable_bytes(std::span(&quadword, 1)), kQuadword) &&
                       vram_.read_all(std::as_writable_bytes(std::span(wide)),
                                      kQuadword - sizeof(uint64_t)));
      count_unless(split, quadword_is_whole(quadword) && quadword_is_whole(wide[1]));
    }
  });
  // Atomic loads at each width.
  threads.emplace_back([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      const auto half = vram_.atomic_load(kQuadword + 2, sizeof(uint16_t));
      const auto word = vram_.atomic_load(kHighWord, sizeof(uint32_t));
      const auto quadword = vram_.atomic_load(kQuadword, sizeof(uint64_t));
      count_unless(refused, half && word && quadword);
      if (word && quadword) {
        count_unless(split, word_is_whole(word->value) && quadword_is_whole(quadword->value));
      }
    }
  });
  // Compare/exchange at both widths, always to another whole value.
  threads.emplace_back([&] {
    uint32_t round = 0x4000;
    while (!stop.load(std::memory_order_relaxed)) {
      ++round;
      const auto word = vram_.atomic_load(kHighWord, sizeof(uint32_t));
      const auto quadword = vram_.atomic_load(kQuadword, sizeof(uint64_t));
      if (!word || !quadword) {
        count_unless(refused, false);
        continue;
      }
      count_unless(refused, vram_.compare_exchange(kHighWord, sizeof(uint32_t), word->value,
                                                   lanes16(round) & 0xffffffffu) &&
                                vram_.compare_exchange(kQuadword, sizeof(uint64_t), quadword->value,
                                                       lanes16(round + 1)));
    }
  });
  // Atomic stores at both widths.
  threads.emplace_back([&] {
    uint32_t round = 0x8000;
    while (!stop.load(std::memory_order_relaxed)) {
      ++round;
      count_unless(refused, vram_.atomic_store(kQuadword, sizeof(uint32_t), lanes16(round)) &&
                                vram_.atomic_store(kQuadword, sizeof(uint64_t), lanes16(round)));
    }
  });

  // Plain stores: a word, the quadword as a buffer, and a longer buffer that
  // starts in the word before it.
  for (uint32_t round = 1; round <= kRounds; ++round) {
    const uint64_t value = lanes16(round);
    const std::array<uint32_t, 4> wide{static_cast<uint32_t>(value), static_cast<uint32_t>(value),
                                       static_cast<uint32_t>(value), static_cast<uint32_t>(value)};
    count_unless(refused,
                 vram_.write(kHighWord, static_cast<uint32_t>(value)) &&
                     vram_.write_all(std::as_bytes(std::span(&value, 1)), kQuadword) &&
                     vram_.write_all(std::as_bytes(std::span(wide)), kQuadword - sizeof(uint32_t)));
  }
  stop.store(true);
  threads.clear();

  EXPECT_EQ(refused.load(), 0u);
  EXPECT_EQ(split.load(), 0u) << "an aligned load saw part of a store";
}

// A cache line fill reads a whole line while other dispatch threads exchange,
// atomically store and plainly store single words of it, and holds no lock in
// common with any of them. The words the fill is not waiting on are loaded
// all the same, so a plain copy on the mapped path races each of those
// writers here, and a ThreadSanitizer build reports it. In any build the test
// checks what the fill may rely on with no lock: the words nobody stores stay
// intact, each stored word is one of the values its writer stored, in the
// order it stored them, and no store or exchange is lost.
TEST_F(VramStorePlainAccess, LineFillsRaceExchangesAndStoresOfOtherWordsInTheLine) {
  constexpr uint64_t kLine = 0x400;
  constexpr std::size_t kLineBytes = 128;
  constexpr std::size_t kExchangedQuadword = 0x18;
  // The low half of a quadword whose high half nobody stores: a fill loads
  // both halves at once, and an exchange of one must leave the other alone.
  constexpr std::size_t kExchangedWord = 0x30;
  constexpr std::size_t kAtomicallyStored = 0x48;
  constexpr std::size_t kPlainWord = 0x60;
  constexpr std::size_t kPlainQuadword = 0x68;
  constexpr uint32_t kRounds = 20000;
  // The round in the low half and its complement in the high half, so that a
  // value made of two different stores is told apart from either.
  const auto quadword_of = [](uint32_t round) { return (uint64_t{~round} << 32) | round; };
  const auto word_of = [](uint32_t round) {
    return ((~round & 0xffffu) << 16) | (round & 0xffffu);
  };
  const auto round_of_quadword = [](uint64_t quadword) -> std::optional<uint32_t> {
    const uint32_t round = static_cast<uint32_t>(quadword);
    if (static_cast<uint32_t>(quadword >> 32) != ~round) {
      return std::nullopt;
    }
    return round;
  };
  const auto round_of_word = [](uint32_t word) -> std::optional<uint32_t> {
    if ((word >> 16) != (~word & 0xffffu)) {
      return std::nullopt;
    }
    return word & 0xffffu;
  };
  struct Stored {
    std::size_t offset;
    std::size_t width;
  };
  constexpr std::array<Stored, 5> kStored{
      Stored{kExchangedQuadword, sizeof(uint64_t)}, Stored{kExchangedWord, sizeof(uint32_t)},
      Stored{kAtomicallyStored, sizeof(uint64_t)}, Stored{kPlainWord, sizeof(uint32_t)},
      Stored{kPlainQuadword, sizeof(uint64_t)}};
  const auto line_at_round = [&](uint32_t round) {
    std::vector<std::byte> line = patterned_bytes(kLineBytes, 23);
    for (const Stored &stored : kStored) {
      const uint64_t value = stored.width == sizeof(uint64_t) ? quadword_of(round) : word_of(round);
      std::memcpy(line.data() + stored.offset, &value, stored.width);
    }
    return line;
  };
  const std::vector<std::byte> initial = line_at_round(0);
  ASSERT_TRUE(vram_.write_all(initial, kLine));

  std::latch start{1};
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> refused{0};
  std::atomic<uint32_t> lost{0};
  std::atomic<uint32_t> disturbed{0};
  std::atomic<uint32_t> torn{0};
  std::atomic<uint32_t> reordered{0};
  const auto count_unless = [](std::atomic<uint32_t> &counter, bool ok) {
    if (!ok) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::jthread> writers;
  writers.emplace_back([&] {
    start.wait();
    for (uint32_t round = 1; round <= kRounds; ++round) {
      const auto quadword = vram_.compare_exchange(kLine + kExchangedQuadword, sizeof(uint64_t),
                                                   quadword_of(round - 1), quadword_of(round));
      const auto word = vram_.compare_exchange(kLine + kExchangedWord, sizeof(uint32_t),
                                               word_of(round - 1), word_of(round));
      count_unless(refused, quadword && word);
      // Nothing else stores these words, so an exchange can only fail to find
      // its own last value if something put an older one back.
      count_unless(lost, quadword && quadword->exchanged && word && word->exchanged);
    }
  });
  writers.emplace_back([&] {
    start.wait();
    for (uint32_t round = 1; round <= kRounds; ++round) {
      count_unless(refused, vram_.atomic_store(kLine + kAtomicallyStored, sizeof(uint64_t),
                                               quadword_of(round)));
    }
  });
  writers.emplace_back([&] {
    start.wait();
    for (uint32_t round = 1; round <= kRounds; ++round) {
      const uint64_t quadword = quadword_of(round);
      count_unless(refused, vram_.write(kLine + kPlainWord, word_of(round)) &&
                                vram_.write_all(std::as_bytes(std::span(&quadword, 1)),
                                                kLine + kPlainQuadword));
    }
  });
  std::jthread filler([&] {
    std::vector<std::byte> line(kLineBytes);
    std::array<uint32_t, kStored.size()> last_round{};
    start.wait();
    // Sampled before the fill, so that one more fill follows the last store.
    for (bool last = false; !last;) {
      last = stop.load();
      if (!vram_.read_all(line, kLine)) {
        count_unless(refused, false);
        continue;
      }
      std::vector<std::byte> unstored = line;
      for (std::size_t index = 0; index < kStored.size(); ++index) {
        const Stored &stored = kStored[index];
        uint64_t value = 0;
        std::memcpy(&value, line.data() + stored.offset, stored.width);
        const std::optional<uint32_t> round = stored.width == sizeof(uint64_t)
                                                  ? round_of_quadword(value)
                                                  : round_of_word(static_cast<uint32_t>(value));
        count_unless(torn, round.has_value());
        if (round) {
          count_unless(reordered, *round >= last_round[index]);
          last_round[index] = *round;
        }
        std::memcpy(unstored.data() + stored.offset, initial.data() + stored.offset, stored.width);
      }
      count_unless(disturbed, unstored == initial);
    }
  });
  start.count_down();
  writers.clear();
  stop.store(true);
  filler.join();

  EXPECT_EQ(refused.load(), 0u);
  EXPECT_EQ(lost.load(), 0u) << "an exchange did not find the value the one before it left";
  EXPECT_EQ(disturbed.load(), 0u) << "a fill returned a changed byte that no thread stores";
  EXPECT_EQ(torn.load(), 0u) << "a fill returned a word that no single store left";
  EXPECT_EQ(reordered.load(), 0u) << "a fill returned an older value of a word than one before it";
  std::vector<std::byte> model(kMemoryBytes);
  overlay(model, line_at_round(kRounds), kLine);
  EXPECT_TRUE(memory_matches(model));
}

} // namespace
