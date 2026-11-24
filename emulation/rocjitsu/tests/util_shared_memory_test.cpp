// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_shared_memory_test.cpp
/// @brief Unit tests for access to memory that other threads and another
///        process access at the same time, at other widths.

#include "util/shared_memory.h"

#include <gtest/gtest.h>

#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <utility>

namespace {

constexpr std::size_t kWidestPiece = sizeof(uint64_t);
constexpr std::size_t kLongestLength = 33;
constexpr std::size_t kCacheLineBytes = 64;

// An aligned 2-, 4- or 8-byte copy is promised to be one atomic access, which
// holds only if the whole copy is selected as a single piece.
static_assert(util::detail::shared_piece_width(0x1002, sizeof(uint16_t)) == sizeof(uint16_t));
static_assert(util::detail::shared_piece_width(0x1004, sizeof(uint32_t)) == sizeof(uint32_t));
static_assert(util::detail::shared_piece_width(0x1008, sizeof(uint64_t)) == sizeof(uint64_t));

using CopyFunction = void(std::byte *, const std::byte *, std::size_t);

// The callers that matter pass a length known only at run time, so they run
// the out-of-line copy loop. A constant length inlined into a test would
// instead be folded into one move, and would be even if store_once() were
// nothing but a libc memcpy. Calling through a pointer that has to be reloaded
// keeps every concurrent test below on the path those callers take.
void store_out_of_line(std::byte *shared_destination, const std::byte *source, std::size_t length) {
  CopyFunction *volatile reloaded_store_once = util::store_once;
  reloaded_store_once(shared_destination, source, length);
}

void load_out_of_line(std::byte *destination, const std::byte *shared_source, std::size_t length) {
  CopyFunction *volatile reloaded_load_once = util::load_once;
  reloaded_load_once(destination, shared_source, length);
}

// The shared words of these tests are bytes of a std::byte array and are
// reached only through util/shared_memory.h, so that no test stores through
// one type what it loads or exchanges through another.
template <typename WordType> std::byte *word_at(std::byte *storage, std::size_t index) {
  return storage + index * sizeof(WordType);
}

template <typename WordType> void store_word(std::byte *shared, WordType value) {
  store_out_of_line(shared, reinterpret_cast<const std::byte *>(&value), sizeof(value));
}

// A thread that finds nothing new from the other one gives the processor away
// after this many looks, so that a single-processor host alternates the two
// threads instead of spending a whole time slice on every round.
constexpr std::size_t kLooksPerYield = 64;

// Looks after which a thread stops waiting for the other one. A count rather
// than a time, so that a loaded host cannot fail a test by being slow; it is
// reached only when the other thread has stopped making progress.
constexpr std::size_t kMostLooks = std::size_t{1} << 28;

// How long a race goes on for when its rounds are not finished sooner. On an
// idle host they are, many times over. Two threads that share their processors
// with other busy processes hand a round over only once per scheduler time
// slice, and the same rounds would then take minutes.
constexpr std::chrono::seconds kRaceTimeLimit{5};

// Rounds a race has to get through in that time to have tested anything.
constexpr std::size_t kFewestRaceRounds = 16;

using RaceClock = std::chrono::steady_clock;

RaceClock::time_point race_deadline() { return RaceClock::now() + kRaceTimeLimit; }

bool race_time_is_up(RaceClock::time_point deadline) { return RaceClock::now() >= deadline; }

template <typename WordType> constexpr WordType kClaimedBit = 1;

template <typename WordType>
constexpr unsigned kHalfWordBits = std::numeric_limits<WordType>::digits / 2;

// Round tags that fit in half a word beside the claim bit.
template <typename WordType>
constexpr std::size_t kRoundTagCount = std::size_t{1} << (kHalfWordBits<WordType> - 1);

// Both halves carry the round tag, so a load that mixes the bytes of two
// rounds is recognisable. The lowest bit is left clear for the claim. Tags
// repeat, which is harmless because the claimer below sees every round.
template <typename WordType> constexpr WordType unclaimed_value(std::size_t round) {
  const std::size_t tag = round % kRoundTagCount<WordType>;
  return static_cast<WordType>((tag << kHalfWordBits<WordType>) | (tag << 1));
}

template <typename WordType> constexpr std::size_t round_tag(WordType observed) {
  return observed >> kHalfWordBits<WordType>;
}

template <typename WordType> constexpr bool is_whole_value(WordType observed) {
  return round_tag(observed) < kRoundTagCount<WordType> &&
         static_cast<WordType>(observed & ~kClaimedBit<WordType>) ==
             unclaimed_value<WordType>(round_tag(observed));
}

// Guaranteed: store_once() stores only the bytes of its destination range, so
// a neighbouring word that another thread advances by compare/exchange loses
// no update. A wider load-modify-store would pass a single-threaded guard-byte
// check and still fail here.
//
// The storer keeps going until the counter has advanced during a fixed number
// of its rounds or the time for the race is up, so the two threads are known
// to have run against each other, and it gives up with a failure if the
// counter stops advancing.
template <typename WordType> void expect_stores_beside_a_counter_lose_no_increment() {
  constexpr std::size_t kCounterIndex = 1;
  constexpr std::size_t kOverlappingRounds = 20000;
  constexpr std::size_t kIncrementsPerYield = 1024;
  alignas(kCacheLineBytes) std::array<std::byte, 3 * sizeof(WordType)> storage{};
  std::byte *const counter_word = word_at<WordType>(storage.data(), kCounterIndex);
  std::byte *const word_before = word_at<WordType>(storage.data(), kCounterIndex - 1);
  std::byte *const word_after = word_at<WordType>(storage.data(), kCounterIndex + 1);
  std::atomic<bool> storing_finished = false;
  std::size_t increments = 0;
  std::size_t overlapping_rounds = 0;
  WordType last_stored = 0;
  {
    std::jthread counter([&] {
      do {
        WordType expected = util::shared_load<WordType>(counter_word);
        while (!util::shared_compare_exchange<WordType>(counter_word, expected,
                                                        static_cast<WordType>(expected + 1))) {
        }
        ++increments;
        if (increments % kIncrementsPerYield == 0) {
          std::this_thread::yield();
        }
      } while (!storing_finished.load());
    });
    std::jthread storer([&] {
      const RaceClock::time_point deadline = race_deadline();
      WordType last_seen_count = util::shared_load<WordType>(counter_word);
      std::size_t looks_without_progress = 0;
      bool time_is_up = false;
      while (overlapping_rounds < kOverlappingRounds && looks_without_progress < kMostLooks &&
             !time_is_up) {
        ++last_stored;
        store_word<WordType>(word_before, last_stored);
        store_word<WordType>(word_after, static_cast<WordType>(~last_stored));
        // A counter that moved between two looks was incremented while the
        // stores between those looks were made.
        const WordType seen_count = util::shared_load<WordType>(counter_word);
        if (seen_count != last_seen_count) {
          last_seen_count = seen_count;
          ++overlapping_rounds;
          looks_without_progress = 0;
          time_is_up = race_time_is_up(deadline);
        } else if (++looks_without_progress % kLooksPerYield == 0) {
          std::this_thread::yield();
          time_is_up = race_time_is_up(deadline);
        }
      }
      storing_finished.store(true);
    });
  }
  ASSERT_GE(overlapping_rounds, kFewestRaceRounds)
      << "the counter stopped advancing while the stores ran, so the stores were not tested "
         "against concurrent increments";
  EXPECT_EQ(util::shared_load<WordType>(counter_word), static_cast<WordType>(increments));
  EXPECT_EQ(util::shared_load<WordType>(word_before), last_stored);
  EXPECT_EQ(util::shared_load<WordType>(word_after), static_cast<WordType>(~last_stored));
}

// Guaranteed for a copy of whole words to a destination aligned for that word:
// each word changes from its old value to the new one in a single store. A
// concurrent atomic load therefore never sees a mixture of the two, and a
// compare/exchange that succeeds against the new value is not overwritten by
// the call that stored it.
//
// Not guaranteed, and not asserted: that updates made before a store survive
// it, that a misaligned copy stores a word in one piece, or that a copy of
// several words is atomic as a whole.
//
// Each round the storer copies a fresh value over the whole run and then waits
// until the claimer has set the claim bit in every word of it, so the claimer
// sees the settled state of every round. A second store of a word would clear
// a claim again, and the claimer would then find a round it has already
// claimed waiting to be claimed.
//
// The copy is free to store a piece wider than the words the claimer
// exchanges, and does whenever two of them make up an aligned quadword.
//
// Every claim is one alternation of the two threads, so the number of claims
// measures how much they overlapped. It is asserted exactly against the rounds
// that fitted in the time for the race, and a claimer that stops answering
// fails the test instead of letting it pass untested.
template <typename WordType>
void expect_no_claim_on_stored_words_is_undone(std::size_t run_word_count, std::size_t rounds) {
  constexpr std::size_t kLongestRun = 9;
  constexpr std::size_t kFirstRunIndex = 1;
  constexpr std::size_t kWordCount = kFirstRunIndex + kLongestRun + 1;
  constexpr std::size_t kIdlePassesPerYield = 1024;
  ASSERT_LE(run_word_count, kLongestRun);
  // The run starts at the second word, which is aligned for its own width and
  // no further: the least the guarantee asks for. The first word is the
  // storer's own and shares a cache line with the start of the run. The words
  // after the run must come through untouched.
  alignas(kCacheLineBytes) std::array<std::byte, kWordCount * sizeof(WordType)> storage;
  const auto word = [&](std::size_t index) { return word_at<WordType>(storage.data(), index); };
  for (std::size_t index = 0; index < kWordCount; ++index) {
    util::shared_store<WordType>(word(index), unclaimed_value<WordType>(0) | kClaimedBit<WordType>);
  }
  std::byte *const line_holder = word(kFirstRunIndex - 1);
  util::shared_store<WordType>(line_holder, 0);
  std::byte *const run = word(kFirstRunIndex);
  std::atomic<bool> storing_finished = false;
  std::size_t torn_observations = 0;
  std::size_t undone_claims = 0;
  std::size_t refused_claims = 0;
  std::size_t claims = 0;
  std::size_t unexpected_values = 0;
  std::size_t claims_seen_at_first_look = 0;
  std::size_t rounds_claimed = 0;
  bool claimer_is_answering = true;
  {
    std::jthread claimer([&] {
      std::array<std::size_t, kLongestRun> last_claimed_tag{};
      std::size_t idle_passes = 0;
      while (!storing_finished.load()) {
        bool claimed_any = false;
        for (std::size_t index = 0; index < run_word_count; ++index) {
          std::byte *const shared_word = word_at<WordType>(run, index);
          WordType observed = util::shared_load<WordType>(shared_word);
          if (!is_whole_value(observed)) {
            ++torn_observations;
            continue;
          }
          if ((observed & kClaimedBit<WordType>) != 0) {
            continue;
          }
          if (round_tag(observed) == last_claimed_tag[index]) {
            ++undone_claims;
          }
          // The storer leaves an unclaimed word alone, so this fails only if
          // the copy that stored the word is still changing it.
          if (util::shared_compare_exchange<WordType>(
                  shared_word, observed, static_cast<WordType>(observed | kClaimedBit<WordType>))) {
            last_claimed_tag[index] = round_tag(observed);
            ++claims;
            claimed_any = true;
          } else {
            ++refused_claims;
          }
        }
        if (claimed_any) {
          idle_passes = 0;
        } else if (++idle_passes % kIdlePassesPerYield == 0) {
          std::this_thread::yield();
        }
      }
    });
    std::jthread storer([&] {
      std::array<WordType, kLongestRun> fresh_values;
      const RaceClock::time_point deadline = race_deadline();
      for (std::size_t round = 1;
           round <= rounds && claimer_is_answering && !race_time_is_up(deadline); ++round) {
        fresh_values.fill(unclaimed_value<WordType>(round));
        // Taking the cache line first, with a locked exchange that waits for
        // it, puts the claimer's request to have it back in flight while the
        // copy runs. Left to the copy itself, that request would arrive after
        // both of two adjacent stores, and a word stored twice would go
        // unnoticed several times as often.
        WordType held_rounds = static_cast<WordType>(round - 1);
        static_cast<void>(util::shared_compare_exchange<WordType>(line_holder, held_rounds,
                                                                  static_cast<WordType>(round)));
        store_out_of_line(run, reinterpret_cast<const std::byte *>(fresh_values.data()),
                          run_word_count * sizeof(WordType));
        const WordType claimed_value = fresh_values.front() | kClaimedBit<WordType>;
        for (std::size_t index = 0; index < run_word_count && claimer_is_answering; ++index) {
          const std::byte *const shared_word = word_at<WordType>(run, index);
          WordType observed = util::shared_load<WordType>(shared_word);
          if (observed == claimed_value) {
            ++claims_seen_at_first_look;
          }
          std::size_t looks = 1;
          while ((observed & kClaimedBit<WordType>) == 0 && looks < kMostLooks) {
            if (++looks % kLooksPerYield == 0) {
              std::this_thread::yield();
            }
            observed = util::shared_load<WordType>(shared_word);
          }
          if ((observed & kClaimedBit<WordType>) == 0) {
            claimer_is_answering = false;
          } else if (observed != claimed_value) {
            ++unexpected_values;
          }
        }
        if (claimer_is_answering) {
          rounds_claimed = round;
        }
      }
      storing_finished.store(true);
    });
  }
  // How many claims had already landed when the copy returned. Zero is
  // legitimate on a single processor, so it is reported and not asserted.
  ::testing::Test::RecordProperty("claims_seen_at_first_look",
                                  std::to_string(claims_seen_at_first_look));
  ::testing::Test::RecordProperty("rounds", std::to_string(rounds_claimed));
  ASSERT_TRUE(claimer_is_answering) << "the claimer stopped claiming";
  ASSERT_GE(rounds_claimed, kFewestRaceRounds)
      << "too few rounds to have tested the copies against a concurrent compare/exchange";
  EXPECT_EQ(torn_observations, 0u);
  EXPECT_EQ(undone_claims, 0u);
  EXPECT_EQ(refused_claims, 0u);
  EXPECT_EQ(unexpected_values, 0u);
  EXPECT_EQ(claims, rounds_claimed * run_word_count);
  EXPECT_EQ(util::shared_load<WordType>(line_holder), static_cast<WordType>(rounds_claimed));
  for (std::size_t index = kFirstRunIndex; index < kWordCount; ++index) {
    const std::size_t last_round = index < kFirstRunIndex + run_word_count ? rounds_claimed : 0;
    EXPECT_EQ(util::shared_load<WordType>(word(index)),
              static_cast<WordType>(unclaimed_value<WordType>(last_round) | kClaimedBit<WordType>))
        << "word " << index;
  }
}

// Guaranteed for a copy out of one word from a source aligned for it: the word
// is loaded once, so the copy returns a value that some store left there and
// never part of one store and part of another.
//
// The storer writes whole values until the loader has seen the value change a
// fixed number of times or the time for the race is up, so the loads are known
// to have run against the stores, and the loader gives up with a failure if
// the value stops changing.
// Both give the processor away regularly, so that a single-processor host
// alternates them often enough to get through those changes.
template <typename WordType> void expect_aligned_loads_see_whole_stores() {
  constexpr std::size_t kChangesSeen = 20000;
  alignas(kCacheLineBytes) std::array<std::byte, 2 * sizeof(WordType)> storage{};
  // Aligned for its own width and no further.
  std::byte *const shared_word = word_at<WordType>(storage.data(), 1);
  std::atomic<bool> loading_finished = false;
  std::size_t torn_observations = 0;
  std::size_t changes_seen = 0;
  {
    std::jthread storer([&] {
      for (std::size_t round = 1; !loading_finished.load(); ++round) {
        store_word<WordType>(shared_word, unclaimed_value<WordType>(round));
        if (round % kLooksPerYield == 0) {
          std::this_thread::yield();
        }
      }
    });
    std::jthread loader([&] {
      const RaceClock::time_point deadline = race_deadline();
      WordType last_observed = 0;
      std::size_t looks_without_change = 0;
      bool time_is_up = false;
      while (changes_seen < kChangesSeen && looks_without_change < kMostLooks && !time_is_up) {
        WordType observed = 0;
        load_out_of_line(reinterpret_cast<std::byte *>(&observed), shared_word, sizeof(observed));
        if (!is_whole_value(observed)) {
          ++torn_observations;
        }
        if (observed != last_observed) {
          last_observed = observed;
          ++changes_seen;
          looks_without_change = 0;
          time_is_up = race_time_is_up(deadline);
        } else if (++looks_without_change % kLooksPerYield == 0) {
          std::this_thread::yield();
          time_is_up = race_time_is_up(deadline);
        }
      }
      loading_finished.store(true);
    });
  }
  ASSERT_GE(changes_seen, kFewestRaceRounds)
      << "the stored value stopped changing, so the loads were not tested against concurrent "
         "stores";
  EXPECT_EQ(torn_observations, 0u);
}

// A libc copy of up to a few words stores a word twice with adjacent
// instructions, and a claim lands between those only rarely.
constexpr std::size_t kShortCopyRounds = 100000;

// A longer copy that stores words twice does so with overlapping vector stores
// further apart, and is caught in a large share of its rounds.
constexpr std::size_t kLongCopyRounds = 20000;

constexpr std::size_t kGuardLength = 2 * kWidestPiece;
constexpr std::byte kGuard{0xEE};
using GuardedBuffer =
    std::array<std::byte, kGuardLength + kWidestPiece + kLongestLength + kGuardLength>;

// Bytes that differ from their neighbours and from the guard, so that one that
// is skipped, misplaced or written outside its range is visible.
template <std::size_t Size> constexpr std::array<std::byte, Size> distinct_bytes() {
  std::array<std::byte, Size> bytes{};
  for (std::size_t index = 0; index < Size; ++index) {
    bytes[index] = static_cast<std::byte>(index + 1);
  }
  return bytes;
}

TEST(SharedMemoryStoreOnceTest, CopiesEveryLengthAtEveryAlignmentAndNothingElse) {
  alignas(uint64_t) const auto source = distinct_bytes<kWidestPiece + kLongestLength>();
  for (std::size_t length = 0; length <= kLongestLength; ++length) {
    for (std::size_t source_misalignment = 0; source_misalignment < kWidestPiece;
         ++source_misalignment) {
      for (std::size_t destination_misalignment = 0; destination_misalignment < kWidestPiece;
           ++destination_misalignment) {
        alignas(uint64_t) GuardedBuffer destination;
        destination.fill(kGuard);
        GuardedBuffer expected = destination;
        const std::size_t first_stored = kGuardLength + destination_misalignment;
        std::copy_n(source.begin() + source_misalignment, length, expected.begin() + first_stored);

        util::store_once(destination.data() + first_stored, source.data() + source_misalignment,
                         length);

        ASSERT_EQ(destination, expected)
            << "length " << length << ", source misalignment " << source_misalignment
            << ", destination misalignment " << destination_misalignment;
      }
    }
  }
}

TEST(SharedMemoryStoreOnceTest, ZeroLengthAcceptsNullPointers) {
  // Callers pass the data() of an empty span, which may be null.
  util::store_once(nullptr, nullptr, 0);
}

TEST(SharedMemoryLoadOnceTest, CopiesEveryLengthAtEveryAlignmentAndNothingElse) {
  alignas(uint64_t) const auto shared_source = distinct_bytes<kWidestPiece + kLongestLength>();
  for (std::size_t length = 0; length <= kLongestLength; ++length) {
    for (std::size_t source_misalignment = 0; source_misalignment < kWidestPiece;
         ++source_misalignment) {
      for (std::size_t destination_misalignment = 0; destination_misalignment < kWidestPiece;
           ++destination_misalignment) {
        alignas(uint64_t) GuardedBuffer destination;
        destination.fill(kGuard);
        GuardedBuffer expected = destination;
        const std::size_t first_filled = kGuardLength + destination_misalignment;
        std::copy_n(shared_source.begin() + source_misalignment, length,
                    expected.begin() + first_filled);

        util::load_once(destination.data() + first_filled,
                        shared_source.data() + source_misalignment, length);

        ASSERT_EQ(destination, expected)
            << "length " << length << ", source misalignment " << source_misalignment
            << ", destination misalignment " << destination_misalignment;
      }
    }
  }
}

TEST(SharedMemoryLoadOnceTest, ZeroLengthAcceptsNullPointers) {
  util::load_once(nullptr, nullptr, 0);
}

TEST(SharedMemoryLoadOnceTest, AnAligned16BitLoadSeesAWholeStore) {
  expect_aligned_loads_see_whole_stores<uint16_t>();
}

TEST(SharedMemoryLoadOnceTest, AnAligned32BitLoadSeesAWholeStore) {
  expect_aligned_loads_see_whole_stores<uint32_t>();
}

TEST(SharedMemoryLoadOnceTest, AnAligned64BitLoadSeesAWholeStore) {
  expect_aligned_loads_see_whole_stores<uint64_t>();
}

TEST(SharedMemoryStoreOnceTest, EveryPieceIsTheWidestAlignedAccessThatFits) {
  constexpr std::uintptr_t kFirstAddress = 0x1000;
  for (std::uintptr_t address = kFirstAddress; address < kFirstAddress + 2 * kWidestPiece;
       ++address) {
    for (std::size_t remaining = 1; remaining <= kLongestLength; ++remaining) {
      const std::size_t width = util::detail::shared_piece_width(address, remaining);
      const std::size_t wider = 2 * width;
      const bool wider_piece_was_available =
          wider <= kWidestPiece && wider <= remaining && address % wider == 0;
      SCOPED_TRACE(testing::Message() << "width " << width << " at address " << address << " with "
                                      << remaining << " bytes remaining");
      ASSERT_TRUE(std::has_single_bit(width));
      ASSERT_LE(width, kWidestPiece);
      ASSERT_LE(width, remaining);
      ASSERT_EQ(address % width, 0u);
      ASSERT_FALSE(wider_piece_was_available);
    }
  }
}

TEST(SharedMemoryStoreOnceTest, StoresBesideA16BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint16_t>();
}

TEST(SharedMemoryStoreOnceTest, StoresBesideA32BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint32_t>();
}

TEST(SharedMemoryStoreOnceTest, StoresBesideA64BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint64_t>();
}

TEST(SharedMemoryStoreOnceTest, CompareExchangeOnAStored16BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint16_t>(1, kShortCopyRounds);
}

TEST(SharedMemoryStoreOnceTest, CompareExchangeOnAStored32BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint32_t>(1, kShortCopyRounds);
}

TEST(SharedMemoryStoreOnceTest, CompareExchangeOnAStored64BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint64_t>(1, kShortCopyRounds);
}

TEST(SharedMemoryStoreOnceTest, CompareExchangeInsideAStoredRunOf32BitWordsIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint32_t>(3, kShortCopyRounds);
}

TEST(SharedMemoryStoreOnceTest, CompareExchangeInsideAStoredRunOf64BitWordsIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint64_t>(9, kLongCopyRounds);
}

constexpr std::array<std::size_t, 4> kWordWidths{1, 2, 4, 8};

constexpr uint64_t low_bytes(uint64_t value, std::size_t width) {
  return width == sizeof(uint64_t) ? value : value & ((uint64_t{1} << (8 * width)) - 1);
}

// What a guarded buffer should hold once the low @p width bytes of @p value
// have been stored at @p offset of it.
GuardedBuffer guarded_buffer_holding(uint64_t value, std::size_t width, std::size_t offset) {
  GuardedBuffer buffer;
  buffer.fill(kGuard);
  std::memcpy(buffer.data() + offset, &value, width);
  return buffer;
}

TEST(SharedMemoryWordTest, TypedFormsLoadStoreAndExchangeOneWord) {
  const auto round_trip = []<util::SharedWord WordType>(WordType first, WordType second) {
    alignas(uint64_t) GuardedBuffer storage;
    storage.fill(kGuard);
    std::byte *const shared = storage.data() + kGuardLength;

    util::shared_store<WordType>(shared, first);
    EXPECT_EQ(storage, guarded_buffer_holding(first, sizeof(WordType), kGuardLength));
    EXPECT_EQ(util::shared_load<WordType>(shared), first);

    WordType expected = second;
    EXPECT_FALSE(util::shared_compare_exchange<WordType>(shared, expected, second));
    EXPECT_EQ(expected, first) << "a refused exchange reports the value it found";
    EXPECT_EQ(util::shared_load<WordType>(shared), first);

    EXPECT_TRUE(util::shared_compare_exchange<WordType>(shared, expected, second));
    EXPECT_EQ(expected, first);
    EXPECT_EQ(storage, guarded_buffer_holding(second, sizeof(WordType), kGuardLength));
  };
  round_trip(uint8_t{0xa5}, uint8_t{0x5a});
  round_trip(uint16_t{0xa55a}, uint16_t{0x1234});
  round_trip(uint32_t{0xa55a0ff0}, uint32_t{0x12345678});
  round_trip(uint64_t{0xa55a0ff0c33c9669}, uint64_t{0x0123456789abcdef});
}

TEST(SharedMemoryWordTest, OnlyAnAlignedWidthOfOneTwoFourOrEightIsASharedWord) {
  alignas(uint64_t) std::array<std::byte, 2 * kWidestPiece> storage{};
  for (std::size_t offset = 0; offset < kWidestPiece; ++offset) {
    for (std::size_t width = 0; width <= 2 * kWidestPiece; ++width) {
      const bool is_word_width = width == 1 || width == 2 || width == 4 || width == 8;
      EXPECT_EQ(util::is_shared_word(storage.data() + offset, width),
                is_word_width && offset % width == 0)
          << "width " << width << " at offset " << offset;
    }
  }
}

TEST(SharedMemoryWordTest, RunTimeWidthFormsAccessExactlyTheWidthGiven) {
  // Set above every width narrower than eight bytes, to show that an operand's
  // high bytes are ignored and a result's are zero.
  constexpr uint64_t kFirst = 0xa55a0ff0c33c9669;
  constexpr uint64_t kSecond = 0x0123456789abcdef;
  for (const std::size_t width : kWordWidths) {
    for (std::size_t offset = 0; offset < 2 * kWidestPiece; offset += width) {
      SCOPED_TRACE(testing::Message() << "width " << width << " at offset " << offset);
      alignas(uint64_t) GuardedBuffer storage;
      storage.fill(kGuard);
      const std::size_t at = kGuardLength + offset;
      std::byte *const shared = storage.data() + at;

      util::shared_store_word(shared, width, kFirst);
      ASSERT_EQ(storage, guarded_buffer_holding(kFirst, width, at));
      EXPECT_EQ(util::shared_load_word(shared, width), low_bytes(kFirst, width));

      uint64_t expected = kSecond;
      EXPECT_FALSE(util::shared_compare_exchange_word(shared, width, expected, kSecond));
      EXPECT_EQ(expected, low_bytes(kFirst, width))
          << "a refused exchange reports the value it found, zero-extended";
      ASSERT_EQ(storage, guarded_buffer_holding(kFirst, width, at));

      expected = kFirst;
      EXPECT_TRUE(util::shared_compare_exchange_word(shared, width, expected, kSecond));
      EXPECT_EQ(expected, low_bytes(kFirst, width));
      ASSERT_EQ(storage, guarded_buffer_holding(kSecond, width, at));
      EXPECT_EQ(util::shared_load_word(shared, width), low_bytes(kSecond, width));
    }
  }
}

#if GTEST_HAS_DEATH_TEST
// The single-word operations stop the process on a width or address that no
// shared word can have, in every build type: there is no narrower or wider
// access they could make instead that the caller asked for.
class SharedMemoryWordDeathTest : public testing::Test {
protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }

  alignas(uint64_t) std::array<std::byte, 2 * kWidestPiece> storage_{};
};

TEST_F(SharedMemoryWordDeathTest, RunTimeWidthFormsRejectAWidthThatIsNotAWordWidth) {
  constexpr std::array<std::size_t, 5> kRejectedWidths{0, 3, 5, 7, 16};
  for (const std::size_t width : kRejectedWidths) {
    SCOPED_TRACE(testing::Message() << "width " << width);
    EXPECT_DEATH(static_cast<void>(util::shared_load_word(storage_.data(), width)), "");
    EXPECT_DEATH(util::shared_store_word(storage_.data(), width, 0), "");
    uint64_t expected = 0;
    EXPECT_DEATH(
        static_cast<void>(util::shared_compare_exchange_word(storage_.data(), width, expected, 1)),
        "");
  }
}

TEST_F(SharedMemoryWordDeathTest, RunTimeWidthFormsRejectAnAddressNotAlignedForTheWidth) {
  constexpr std::array<std::pair<std::size_t, std::size_t>, 3> kMisalignedWords{
      {{sizeof(uint16_t), 1}, {sizeof(uint32_t), 2}, {sizeof(uint64_t), 4}}};
  for (const auto &[width, offset] : kMisalignedWords) {
    SCOPED_TRACE(testing::Message() << "width " << width << " at offset " << offset);
    std::byte *const shared = storage_.data() + offset;
    EXPECT_DEATH(static_cast<void>(util::shared_load_word(shared, width)), "");
    EXPECT_DEATH(util::shared_store_word(shared, width, 0), "");
    uint64_t expected = 0;
    EXPECT_DEATH(static_cast<void>(util::shared_compare_exchange_word(shared, width, expected, 1)),
                 "");
  }
}

TEST_F(SharedMemoryWordDeathTest, TypedFormsRejectAnAddressNotAlignedForTheWord) {
  std::byte *const shared = storage_.data() + 2;
  EXPECT_DEATH(static_cast<void>(util::shared_load<uint32_t>(shared)), "");
  EXPECT_DEATH(util::shared_store<uint32_t>(shared, 0), "");
  uint64_t expected = 0;
  EXPECT_DEATH(static_cast<void>(util::shared_compare_exchange<uint64_t>(shared, expected, 1)), "");
}
#endif

// What the peer process below does to the shared words is written out here and
// not taken from the header under test: the peer stands for a guest, whose
// code knows nothing of that header and uses its processor's own atomic
// instructions at whatever width suits it.
typedef uint8_t __attribute__((__may_alias__)) PeerU8;
typedef uint16_t __attribute__((__may_alias__)) PeerU16;
typedef uint32_t __attribute__((__may_alias__)) PeerU32;

bool peer_compare_exchange_32(std::byte *word, uint32_t expected, uint32_t desired) {
  return __atomic_compare_exchange_n(reinterpret_cast<PeerU32 *>(word), &expected, desired,
                                     /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

/// @brief Pages shared with a process forked after they were mapped.
class SharedPages {
public:
  explicit SharedPages(std::size_t length) : length_(length) {
    void *mapped =
        ::mmap(nullptr, length_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
      bytes_ = static_cast<std::byte *>(mapped);
    }
  }
  ~SharedPages() {
    if (bytes_ != nullptr) {
      ::munmap(bytes_, length_);
    }
  }
  SharedPages(const SharedPages &) = delete;
  SharedPages &operator=(const SharedPages &) = delete;

  [[nodiscard]] std::byte *data() const { return bytes_; }

private:
  std::size_t length_;
  std::byte *bytes_ = nullptr;
};

/// @brief What the two processes tell each other, in a shared page of its own.
/// @details Lock-free atomics are address-free, so each process may use them
/// through its own mapping of the page.
struct PeerControl {
  std::atomic<uint32_t> go{0};
  std::atomic<uint32_t> stop{0};
  std::atomic<uint32_t> written_round{0};
  std::atomic<uint32_t> exchanged_round{0};
  std::atomic<uint32_t> missed_exchanges{0};
  std::atomic<uint32_t> peer_lost_stores{0};
};
static_assert(std::atomic<uint32_t>::is_always_lock_free);

/// @brief A forked child that runs one function and exits.
/// @details The child touches only the shared pages and its own stack, and
/// leaves with _exit(), so it runs none of the test program's atexit handlers
/// and reports nothing of its own; what it found is read from the shared pages.
/// It is killed when the test process goes, however that happens, so a test
/// that is itself killed leaves no child waiting for a round that never comes.
class PeerProcess {
public:
  template <typename Body> explicit PeerProcess(Body body) : parent_(::getpid()), pid_(::fork()) {
    if (pid_ == 0) {
      // The parent may have gone before the request took effect.
      if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != parent_) {
        ::_exit(1);
      }
      body();
      ::_exit(0);
    }
  }
  ~PeerProcess() {
    if (pid_ > 0 && !reaped_) {
      ::kill(pid_, SIGKILL);
      ::waitpid(pid_, nullptr, 0);
    }
  }
  PeerProcess(const PeerProcess &) = delete;
  PeerProcess &operator=(const PeerProcess &) = delete;

  [[nodiscard]] bool started() const { return pid_ > 0; }

  /// @brief Whether the child has already gone, without waiting for it.
  [[nodiscard]] bool has_exited() {
    if (!reaped_ && ::waitpid(pid_, &status_, WNOHANG) == pid_) {
      reaped_ = true;
    }
    return reaped_;
  }

  /// @brief Wait for the child, and say whether it ran to its end.
  [[nodiscard]] bool finished_cleanly() {
    if (!reaped_ && ::waitpid(pid_, &status_, 0) == pid_) {
      reaped_ = true;
    }
    return reaped_ && WIFEXITED(status_) && WEXITSTATUS(status_) == 0;
  }

private:
  pid_t parent_;
  pid_t pid_;
  int status_ = 0;
  bool reaped_ = false;
};

class SharedMemoryCrossProcessTest : public testing::Test {
protected:
  static constexpr std::size_t kPageBytes = 4096;

  void SetUp() override {
    ASSERT_NE(words_.data(), nullptr);
    ASSERT_NE(control_page_.data(), nullptr);
    control_ = new (control_page_.data()) PeerControl;
  }

  SharedPages words_{kPageBytes};
  SharedPages control_page_{kPageBytes};
  PeerControl *control_ = nullptr;
};

// One quadword holds three counters that the two processes advance at three
// widths, each with a single atomic instruction of its own choosing:
//
//   bytes 0-3  this process, by 32-bit compare/exchange, and the peer, by its
//              own 32-bit compare/exchange;
//   bytes 4-5  the peer, by 16-bit exchange-and-add;
//   bytes 6-7  this process, by 64-bit compare/exchange of the whole quadword.
//
// The next quadword holds bytes each process stores plainly and reads back:
// three from this process, through store_once() and load_once(), then one and
// four from the peer.
//
// Every counter must come out exact and no plain store may be lost, which is
// so only if each access of either process is indivisible against the other's
// accesses of other widths, and touches no byte outside its own range. The
// processes are not made to interleave in any particular way, so nothing here
// depends on timing; how often an exchange had to be retried is recorded as a
// measure of how much they did.
TEST_F(SharedMemoryCrossProcessTest, CountersAdvancedByAPeerProcessAtOtherWidthsStayExact) {
  constexpr uint32_t kRounds = 200000;
  constexpr uint64_t kHighCounterOne = uint64_t{1} << 48;
  std::byte *const quadword = words_.data() + 2 * kCacheLineBytes;
  std::byte *const peer_counter = quadword + 4;
  std::byte *const own_bytes = quadword + 8;
  std::byte *const peer_byte = quadword + 11;
  std::byte *const peer_word = quadword + 12;
  constexpr std::size_t kOwnByteCount = 3;

  PeerControl *const control = control_;
  PeerProcess peer([=] {
    while (control->go.load() == 0) {
      ::sched_yield();
    }
    uint32_t lost_stores = 0;
    for (uint32_t round = 1; round <= kRounds; ++round) {
      uint32_t count = __atomic_load_n(reinterpret_cast<PeerU32 *>(quadword), __ATOMIC_SEQ_CST);
      while (!peer_compare_exchange_32(quadword, count, count + 1)) {
        count = __atomic_load_n(reinterpret_cast<PeerU32 *>(quadword), __ATOMIC_SEQ_CST);
      }
      __atomic_fetch_add(reinterpret_cast<PeerU16 *>(peer_counter), 1, __ATOMIC_SEQ_CST);

      auto *const plain_word = reinterpret_cast<volatile PeerU32 *>(peer_word);
      auto *const plain_byte = reinterpret_cast<volatile PeerU8 *>(peer_byte);
      *plain_word = round;
      *plain_byte = static_cast<uint8_t>(round);
      if (*plain_word != round || *plain_byte != static_cast<uint8_t>(round)) {
        ++lost_stores;
      }
    }
    control->peer_lost_stores.store(lost_stores);
  });
  ASSERT_TRUE(peer.started());

  control_->go.store(1);
  uint64_t exchange_retries = 0;
  uint32_t lost_stores = 0;
  uint32_t counts_running_backwards = 0;
  uint32_t last_count = 0;
  for (uint32_t round = 1; round <= kRounds; ++round) {
    uint32_t count = util::shared_load<uint32_t>(quadword);
    while (!util::shared_compare_exchange<uint32_t>(quadword, count, count + 1)) {
      ++exchange_retries;
    }

    uint64_t whole = util::shared_load<uint64_t>(quadword);
    while (!util::shared_compare_exchange<uint64_t>(quadword, whole, whole + kHighCounterOne)) {
      ++exchange_retries;
    }

    const std::array<std::byte, kOwnByteCount> stored{static_cast<std::byte>(round),
                                                      static_cast<std::byte>(round >> 8),
                                                      static_cast<std::byte>(round >> 16)};
    store_out_of_line(own_bytes, stored.data(), stored.size());
    std::array<std::byte, kOwnByteCount> loaded{};
    load_out_of_line(loaded.data(), own_bytes, loaded.size());
    if (loaded != stored) {
      ++lost_stores;
    }

    // One aligned load of the counter both processes advance: it only grows.
    uint32_t seen_count = 0;
    load_out_of_line(reinterpret_cast<std::byte *>(&seen_count), quadword, sizeof(seen_count));
    if (seen_count < last_count) {
      ++counts_running_backwards;
    }
    last_count = seen_count;
  }
  ASSERT_TRUE(peer.finished_cleanly());
  RecordProperty("exchange_retries", std::to_string(exchange_retries));

  EXPECT_EQ(util::shared_load<uint32_t>(quadword), 2 * kRounds);
  EXPECT_EQ(util::shared_load<uint16_t>(peer_counter), static_cast<uint16_t>(kRounds));
  EXPECT_EQ(util::shared_load<uint16_t>(quadword + 6), static_cast<uint16_t>(kRounds));
  EXPECT_EQ(lost_stores, 0u);
  EXPECT_EQ(counts_running_backwards, 0u);
  EXPECT_EQ(control_->peer_lost_stores.load(), 0u);
  EXPECT_EQ(util::shared_load<uint8_t>(peer_byte), static_cast<uint8_t>(kRounds));
  EXPECT_EQ(util::shared_load<uint32_t>(peer_word), kRounds);
}

// Each round this process copies a fresh value over a word and the peer,
// already retrying, exchanges that value for a marked one with its own 32-bit
// compare/exchange. Nothing else writes the word, so the round has to settle on
// the marked value. A copy that stored the word twice would, when the peer's
// exchange lands between the two stores, put the unmarked value back.
//
// The word is the last of its cache line and the copy runs on into the next
// line. A repeated store would then be one that needs both lines, and the
// peer, which needs only the first, gets in between two of those far more
// often than between two stores inside one line.
//
// The processes hand rounds to each other through the control page, so a slow
// or single-processor host only gets through fewer rounds in the time for the
// race.
TEST_F(SharedMemoryCrossProcessTest, StoreOnceDoesNotUndoAPeerProcessCompareExchange) {
  constexpr uint32_t kRounds = 100000;
  constexpr uint32_t kExchangedMark = 1;
  constexpr uint32_t kAttemptsPerYield = 64;
  constexpr uint32_t kLooksPerLivenessCheck = 4096;
  std::byte *const word = words_.data() + 4 * kCacheLineBytes - sizeof(uint32_t);

  PeerControl *const control = control_;
  PeerProcess peer([=] {
    for (uint32_t round = 1; round <= kRounds; ++round) {
      const uint32_t plain = round << 1;
      bool exchanged = false;
      bool copy_had_returned = false;
      uint32_t attempts = 0;
      while (!exchanged && !copy_had_returned && control->stop.load() == 0) {
        // Sampled before the attempt: only an attempt that begins after the
        // copy has returned is certain to find its value.
        copy_had_returned = control->written_round.load() >= round;
        exchanged = peer_compare_exchange_32(word, plain, plain | kExchangedMark);
        if (!exchanged && ++attempts % kAttemptsPerYield == 0) {
          ::sched_yield();
        }
      }
      // Stopped while waiting for a copy that is not going to be made.
      if (!exchanged && !copy_had_returned) {
        break;
      }
      if (!exchanged) {
        control->missed_exchanges.fetch_add(1);
      }
      control->exchanged_round.store(round);
    }
  });
  ASSERT_TRUE(peer.started());

  uint32_t undone_exchanges = 0;
  uint32_t rounds_raced = 0;
  bool peer_is_answering = true;
  const RaceClock::time_point deadline = race_deadline();
  for (uint32_t round = 1; round <= kRounds && peer_is_answering && !race_time_is_up(deadline);
       ++round) {
    const uint32_t plain = round << 1;
    const std::array<uint32_t, 2> buffer{plain, 0x3333cccc};
    store_out_of_line(word, reinterpret_cast<const std::byte *>(buffer.data()), sizeof(buffer));
    control_->written_round.store(round);
    for (uint32_t looks = 1; control_->exchanged_round.load() < round; ++looks) {
      ::sched_yield();
      if (looks % kLooksPerLivenessCheck == 0 && peer.has_exited()) {
        peer_is_answering = control_->exchanged_round.load() >= round;
        break;
      }
    }
    if (!peer_is_answering) {
      break;
    }
    if (util::shared_load<uint32_t>(word) != (plain | kExchangedMark)) {
      ++undone_exchanges;
    }
    rounds_raced = round;
  }
  control_->stop.store(1);
  RecordProperty("rounds", std::to_string(rounds_raced));
  ASSERT_TRUE(peer_is_answering) << "the peer process stopped before the last round";
  ASSERT_TRUE(peer.finished_cleanly());
  ASSERT_GE(rounds_raced, kFewestRaceRounds)
      << "too few rounds to have tested the copy against the peer's compare/exchange";

  EXPECT_EQ(control_->missed_exchanges.load(), 0u);
  EXPECT_EQ(undone_exchanges, 0u)
      << "a copy stored a word again after the peer's compare/exchange had replaced it";
}

} // namespace
