// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_store_once_test.cpp
/// @brief Unit tests for copying into memory that other threads update atomically.

#include "util/store_once.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <thread>

namespace {

constexpr std::size_t kWidestPiece = sizeof(uint64_t);
constexpr std::size_t kLongestLength = 33;

// An aligned 2-, 4- or 8-byte copy is promised to be one atomic store, which
// holds only if the whole copy is selected as a single piece.
static_assert(util::detail::store_piece_width(0x1002, sizeof(uint16_t)) == sizeof(uint16_t));
static_assert(util::detail::store_piece_width(0x1004, sizeof(uint32_t)) == sizeof(uint32_t));
static_assert(util::detail::store_piece_width(0x1008, sizeof(uint64_t)) == sizeof(uint64_t));

using StoreFunction = void(std::byte *, const std::byte *, std::size_t);

// The callers that matter pass a length known only at run time, so they run
// the out-of-line copy loop. A constant length inlined into a test would
// instead be folded into one move, and would be even if store_once() were
// nothing but a libc memcpy. Calling through a pointer that has to be reloaded
// keeps every concurrent test below on the path those callers take.
void store_out_of_line(std::byte *destination, const std::byte *source, std::size_t length) {
  StoreFunction *volatile reloaded_store_once = util::store_once;
  reloaded_store_once(destination, source, length);
}

template <typename WordType>
void store_words(std::span<WordType> destination, std::span<const WordType> source) {
  store_out_of_line(std::as_writable_bytes(destination).data(), std::as_bytes(source).data(),
                    destination.size_bytes());
}

// A thread that finds nothing new from the other one gives the processor away
// after this many looks, so that a single-processor host alternates the two
// threads instead of spending a whole time slice on every round.
constexpr std::size_t kLooksPerYield = 64;

// Looks after which a thread stops waiting for the other one. A count rather
// than a time, so that a loaded host cannot fail a test by being slow; it is
// reached only when the other thread has stopped making progress.
constexpr std::size_t kMostLooks = std::size_t{1} << 28;

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
// a neighbouring word that another thread advances by compare-exchange loses
// no update. A wider load-modify-store would pass a single-threaded guard-byte
// check and still fail here.
//
// The storer keeps going until the counter has advanced during a fixed number
// of its rounds, so the two threads are known to have run against each other,
// and it gives up with a failure if the counter stops advancing.
template <typename WordType> void expect_stores_beside_a_counter_lose_no_increment() {
  constexpr std::size_t kCounterIndex = 1;
  constexpr std::size_t kOverlappingRounds = 20000;
  constexpr std::size_t kIncrementsPerYield = 1024;
  alignas(uint64_t) std::array<WordType, 3> words{};
  std::atomic<bool> storing_finished = false;
  std::size_t increments = 0;
  std::size_t overlapping_rounds = 0;
  WordType last_stored = 0;
  {
    std::jthread counter([&] {
      std::atomic_ref<WordType> counter_word(words[kCounterIndex]);
      do {
        WordType expected = counter_word.load();
        bool was_incremented = false;
        while (!was_incremented) {
          was_incremented =
              counter_word.compare_exchange_weak(expected, static_cast<WordType>(expected + 1));
        }
        ++increments;
        if (increments % kIncrementsPerYield == 0) {
          std::this_thread::yield();
        }
      } while (!storing_finished.load());
    });
    std::jthread storer([&] {
      std::atomic_ref<WordType> counter_word(words[kCounterIndex]);
      WordType last_seen_count = counter_word.load();
      std::size_t looks_without_progress = 0;
      while (overlapping_rounds < kOverlappingRounds && looks_without_progress < kMostLooks) {
        ++last_stored;
        const WordType inverse = static_cast<WordType>(~last_stored);
        store_words(std::span(&words[kCounterIndex - 1], 1),
                    std::span<const WordType>(&last_stored, 1));
        store_words(std::span(&words[kCounterIndex + 1], 1), std::span(&inverse, 1));
        // A counter that moved between two looks was incremented while the
        // stores between those looks were made.
        const WordType seen_count = counter_word.load();
        if (seen_count != last_seen_count) {
          last_seen_count = seen_count;
          ++overlapping_rounds;
          looks_without_progress = 0;
        } else if (++looks_without_progress % kLooksPerYield == 0) {
          std::this_thread::yield();
        }
      }
      storing_finished.store(true);
    });
  }
  ASSERT_EQ(overlapping_rounds, kOverlappingRounds)
      << "the counter stopped advancing while the stores ran, so the stores were not tested "
         "against concurrent increments";
  EXPECT_EQ(words[kCounterIndex], static_cast<WordType>(increments));
  EXPECT_EQ(words[kCounterIndex - 1], last_stored);
  EXPECT_EQ(words[kCounterIndex + 1], static_cast<WordType>(~last_stored));
}

// Guaranteed for a copy of whole words to a destination aligned for that word:
// each word changes from its old value to the new one in a single store. A
// concurrent atomic load therefore never sees a mixture of the two, and a
// compare-exchange that succeeds against the new value is not overwritten by
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
// Every claim is one alternation of the two threads, so the number of claims
// measures how much they overlapped. It is asserted exactly, and a claimer
// that stops answering fails the test instead of letting it pass untested.
template <typename WordType>
void expect_no_claim_on_stored_words_is_undone(std::size_t run_word_count, std::size_t rounds) {
  constexpr std::size_t kLongestRun = 9;
  constexpr std::size_t kFirstRunIndex = 1;
  constexpr std::size_t kCacheLineBytes = 64;
  constexpr std::size_t kIdlePassesPerYield = 1024;
  ASSERT_LE(run_word_count, kLongestRun);
  // The run starts at the second element, which is aligned for its own width
  // and no further: the least the guarantee asks for. The first element is the
  // storer's own and shares a cache line with the start of the run. The
  // elements after the run must come through untouched.
  alignas(kCacheLineBytes) std::array<WordType, kFirstRunIndex + kLongestRun + 1> words;
  words.fill(unclaimed_value<WordType>(0) | kClaimedBit<WordType>);
  WordType &line_holder = words[kFirstRunIndex - 1];
  line_holder = 0;
  const std::span<WordType> run(&words[kFirstRunIndex], run_word_count);
  std::atomic<bool> storing_finished = false;
  std::size_t torn_observations = 0;
  std::size_t undone_claims = 0;
  std::size_t refused_claims = 0;
  std::size_t claims = 0;
  std::size_t unexpected_values = 0;
  std::size_t claims_seen_at_first_look = 0;
  std::size_t rounds_claimed = 0;
  {
    std::jthread claimer([&] {
      std::array<std::size_t, kLongestRun> last_claimed_tag{};
      std::size_t idle_passes = 0;
      while (!storing_finished.load()) {
        bool claimed_any = false;
        for (std::size_t index = 0; index < run.size(); ++index) {
          std::atomic_ref<WordType> shared_word(run[index]);
          WordType observed = shared_word.load();
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
          if (shared_word.compare_exchange_strong(
                  observed, static_cast<WordType>(observed | kClaimedBit<WordType>))) {
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
      bool claimer_is_answering = true;
      for (std::size_t round = 1; round <= rounds && claimer_is_answering; ++round) {
        fresh_values.fill(unclaimed_value<WordType>(round));
        // Taking the cache line first puts the claimer's request to have it
        // back in flight while the copy runs. Left to the copy itself, that
        // request would arrive after both of two adjacent stores, and a word
        // stored twice would go unnoticed several times as often.
        std::atomic_ref<WordType>(line_holder).fetch_add(1);
        store_words(run, std::span<const WordType>(fresh_values.data(), run.size()));
        const WordType claimed_value = fresh_values.front() | kClaimedBit<WordType>;
        for (std::size_t index = 0; index < run.size() && claimer_is_answering; ++index) {
          std::atomic_ref<WordType> shared_word(run[index]);
          WordType observed = shared_word.load();
          if (observed == claimed_value) {
            ++claims_seen_at_first_look;
          }
          std::size_t looks = 1;
          while ((observed & kClaimedBit<WordType>) == 0 && looks < kMostLooks) {
            if (++looks % kLooksPerYield == 0) {
              std::this_thread::yield();
            }
            observed = shared_word.load();
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
  ASSERT_EQ(rounds_claimed, rounds)
      << "the claimer stopped claiming, so the copies were not tested against a concurrent "
         "compare-exchange";
  EXPECT_EQ(torn_observations, 0u);
  EXPECT_EQ(undone_claims, 0u);
  EXPECT_EQ(refused_claims, 0u);
  EXPECT_EQ(unexpected_values, 0u);
  EXPECT_EQ(claims, rounds * run_word_count);
  EXPECT_EQ(line_holder, static_cast<WordType>(rounds));
  for (std::size_t index = kFirstRunIndex; index < words.size(); ++index) {
    const std::size_t last_round = index < kFirstRunIndex + run_word_count ? rounds : 0;
    EXPECT_EQ(words[index],
              static_cast<WordType>(unclaimed_value<WordType>(last_round) | kClaimedBit<WordType>))
        << "word " << index;
  }
}

// A libc copy of up to a few words stores a word twice with adjacent
// instructions, and a claim lands between those only tens to hundreds of times
// in a million rounds.
constexpr std::size_t kShortCopyRounds = 100000;

// A longer copy that stores words twice does so with overlapping vector stores
// further apart, and is caught in a large share of its rounds.
constexpr std::size_t kLongCopyRounds = 20000;

TEST(StoreOnceTest, CopiesEveryLengthAtEveryAlignmentAndNothingElse) {
  constexpr std::size_t kGuardLength = 2 * kWidestPiece;
  constexpr std::byte kGuard{0xEE};
  using GuardedBuffer =
      std::array<std::byte, kGuardLength + kWidestPiece + kLongestLength + kGuardLength>;

  // Every source byte differs from its neighbours and from the guard, so a
  // byte that is skipped, misplaced or written outside the range is visible.
  alignas(uint64_t) std::array<std::byte, kWidestPiece + kLongestLength> source;
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<std::byte>(index + 1);
  }

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

TEST(StoreOnceTest, ZeroLengthAcceptsNullPointers) {
  // Callers pass the data() of an empty span, which may be null.
  util::store_once(nullptr, nullptr, 0);
}

TEST(StoreOnceTest, EveryPieceIsTheWidestAlignedStoreThatFits) {
  constexpr std::uintptr_t kFirstAddress = 0x1000;
  for (std::uintptr_t address = kFirstAddress; address < kFirstAddress + 2 * kWidestPiece;
       ++address) {
    for (std::size_t remaining = 1; remaining <= kLongestLength; ++remaining) {
      const std::size_t width = util::detail::store_piece_width(address, remaining);
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

TEST(StoreOnceTest, StoresBesideA16BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint16_t>();
}

TEST(StoreOnceTest, StoresBesideA32BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint32_t>();
}

TEST(StoreOnceTest, StoresBesideA64BitCompareExchangeCounterLoseNoIncrement) {
  expect_stores_beside_a_counter_lose_no_increment<uint64_t>();
}

TEST(StoreOnceTest, CompareExchangeOnAStored16BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint16_t>(1, kShortCopyRounds);
}

TEST(StoreOnceTest, CompareExchangeOnAStored32BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint32_t>(1, kShortCopyRounds);
}

TEST(StoreOnceTest, CompareExchangeOnAStored64BitWordIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint64_t>(1, kShortCopyRounds);
}

TEST(StoreOnceTest, CompareExchangeInsideAStoredRunOf32BitWordsIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint32_t>(3, kShortCopyRounds);
}

TEST(StoreOnceTest, CompareExchangeInsideAStoredRunOf64BitWordsIsNotUndone) {
  expect_no_claim_on_stored_words_is_undone<uint64_t>(9, kLongCopyRounds);
}

} // namespace
