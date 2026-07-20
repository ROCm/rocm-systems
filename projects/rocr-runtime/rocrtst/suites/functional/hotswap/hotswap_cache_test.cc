//===- hotswap_cache_test.cc - HotSwap cache tests -----------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/inc/hotswap.hpp"
#include "gtest/gtest.h"

namespace {

using rocr::hotswap::ContentRetargetCache;
using rocr::hotswap::RetargetCacheKey;
using rocr::hotswap::RetargetCacheMetrics;
using rocr::hotswap::RetargetError;
using rocr::hotswap::RetargetOperationResult;
using rocr::hotswap::RetargetResultSource;
using rocr::hotswap::SourceSnapshotRef;

constexpr size_t kSourceSize = 1024;
constexpr size_t kTestElfSize = 4096;

RetargetCacheKey MakeKey(size_t index = 0) {
  return {"amdgcn-amd-amdhsa--gfx1250:source-" + std::to_string(index),
          "amdgcn-amd-amdhsa--gfx1250:target-" + std::to_string(index), false, false};
}

std::vector<unsigned char> MakeSource(unsigned char fill = 0x21, size_t size = kSourceSize) {
  return std::vector<unsigned char>(size, fill);
}

RetargetOperationResult MakeElf(const SourceSnapshotRef& source, size_t size = kTestElfSize,
                                unsigned char fill = 0x5a) {
  rocr::hotswap::OwnedElfBuffer bytes(std::malloc(size), &std::free);
  if (!bytes) return {{}, RetargetError::kOutOfResources};
  std::memset(bytes.get(), fill, size);
  return {std::make_shared<const rocr::hotswap::RetargetedElf>(std::move(bytes), size, source),
          RetargetError::kNone};
}

RetargetOperationResult GetOrCompute(ContentRetargetCache* cache,
                                     const std::vector<unsigned char>& source,
                                     uint64_t reader_id, const RetargetCacheKey& key,
                                     const ContentRetargetCache::Producer& producer) {
  return cache->GetOrCompute(source.data(), source.size(), reader_id, key, producer);
}

bool WaitUntil(const std::function<bool()>& predicate,
               std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return predicate();
}

uint32_t NextRandom(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

TEST(HotswapCache, ConcurrentMissHasOneProducerAndOnePayload) {
  constexpr size_t kThreadCount = 16;
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();
  std::atomic<size_t> producer_calls{0};
  std::atomic<bool> release_producer{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] =
          GetOrCompute(&cache, source, 0, key, [&](const SourceSnapshotRef& snapshot) {
            producer_calls.fetch_add(1, std::memory_order_relaxed);
            while (!release_producer.load(std::memory_order_acquire)) std::this_thread::yield();
            return MakeElf(snapshot);
          });
    });
  }

  const bool all_waiters = WaitUntil([&] {
    return cache.WaiterCountForTesting(source.data(), source.size(), key) == kThreadCount - 1;
  });
  release_producer.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(all_waiters);
  ASSERT_EQ(producer_calls.load(std::memory_order_relaxed), 1u);
  size_t computed = 0;
  size_t coalesced = 0;
  for (const auto& result : results) {
    ASSERT_TRUE(result.succeeded());
    EXPECT_EQ(result.elf.get(), results.front().elf.get());
    computed += result.source == RetargetResultSource::kComputed;
    coalesced += result.source == RetargetResultSource::kCoalesced;
  }

  const RetargetCacheMetrics metrics = cache.SnapshotMetrics();
  EXPECT_EQ(computed, 1u);
  EXPECT_EQ(coalesced, kThreadCount - 1);
  EXPECT_EQ(metrics.producer_calls, 1u);
  EXPECT_EQ(metrics.coalesced_results, kThreadCount - 1);
  EXPECT_EQ(metrics.source_snapshot_allocations, 1u);
  EXPECT_EQ(metrics.live_source_snapshot_bytes, source.size());
  EXPECT_EQ(metrics.live_output_bytes, kTestElfSize);
  EXPECT_GT(metrics.hash_bytes, 0u);
  EXPECT_GT(metrics.exact_compare_bytes, 0u);
  EXPECT_GT(metrics.wait_nanoseconds, 0u);

  results.clear();
  EXPECT_EQ(cache.SnapshotMetrics().live_source_snapshot_bytes, 0u);
  EXPECT_EQ(cache.SnapshotMetrics().live_output_bytes, 0u);
}

TEST(HotswapCache, DistinctReadersWithIdenticalContentShareOneResult) {
  ContentRetargetCache cache;
  const auto first_view = MakeSource();
  const auto second_view = first_view;
  const RetargetCacheKey key = MakeKey();
  constexpr uint64_t kFirstReaderId = 1;
  constexpr uint64_t kSecondReaderId = 2;
  size_t producer_calls = 0;

  const auto first =
      GetOrCompute(&cache, first_view, kFirstReaderId, key,
                   [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot);
      });
  const auto second = GetOrCompute(&cache, second_view, kSecondReaderId, key,
                                   [&](const SourceSnapshotRef& snapshot) {
                                     ++producer_calls;
                                     return MakeElf(snapshot, kTestElfSize, 0x99);
                                   });

  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  EXPECT_EQ(producer_calls, 1u);
  EXPECT_EQ(second.source, RetargetResultSource::kReadyCache);
  EXPECT_EQ(first.elf.get(), second.elf.get());
  EXPECT_EQ(cache.SnapshotMetrics().cross_reader_results, 1u);
}

TEST(HotswapCache, MutationBetweenLoadsCreatesANewExactEntry) {
  ContentRetargetCache cache;
  auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();
  constexpr uint64_t kReaderId = 1;
  size_t producer_calls = 0;

  const auto first =
      GetOrCompute(&cache, source, kReaderId, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot, kTestElfSize, 0x11);
      });
  source[source.size() / 2] ^= 0xff;
  const auto second =
      GetOrCompute(&cache, source, kReaderId, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot, kTestElfSize, 0x22);
      });

  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  EXPECT_EQ(producer_calls, 2u);
  EXPECT_NE(first.elf.get(), second.elf.get());
  EXPECT_EQ(*static_cast<const unsigned char*>(first.elf->data()), 0x11);
  EXPECT_EQ(*static_cast<const unsigned char*>(second.elf->data()), 0x22);
}

TEST(HotswapCache, ForcedHashCollisionCannotAliasDifferentContent) {
  ContentRetargetCache cache([](const void*, size_t) { return 7; });
  const auto first_source = MakeSource(0x31);
  const auto second_source = MakeSource(0x32);
  const RetargetCacheKey key = MakeKey();
  size_t producer_calls = 0;

  const auto first =
      GetOrCompute(&cache, first_source, 0, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot, kTestElfSize, 0x41);
      });
  const auto second =
      GetOrCompute(&cache, second_source, 0, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot, kTestElfSize, 0x42);
      });
  const auto first_again =
      GetOrCompute(&cache, first_source, 0, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot, kTestElfSize, 0x43);
      });

  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  ASSERT_TRUE(first_again.succeeded());
  EXPECT_EQ(producer_calls, 2u);
  EXPECT_NE(first.elf.get(), second.elf.get());
  EXPECT_EQ(first_again.elf.get(), first.elf.get());
  EXPECT_EQ(cache.ReadyEntryCountForTesting(), 2u);
  EXPECT_GE(cache.SnapshotMetrics().exact_compare_bytes, 2 * kSourceSize);
}

TEST(HotswapCache, CollidingContentCanProduceConcurrently) {
  ContentRetargetCache cache([](const void*, size_t) { return 7; });
  const auto first_source = MakeSource(0x51);
  const auto second_source = MakeSource(0x52);
  const RetargetCacheKey key = MakeKey();
  std::atomic<size_t> active_producers{0};
  std::atomic<bool> release_producers{false};
  std::vector<RetargetOperationResult> results(2);
  std::vector<std::thread> threads;

  const auto run = [&](size_t index, const std::vector<unsigned char>& source) {
    results[index] =
        GetOrCompute(&cache, source, 0, key, [&](const SourceSnapshotRef& snapshot) {
          active_producers.fetch_add(1, std::memory_order_acq_rel);
          while (!release_producers.load(std::memory_order_acquire)) std::this_thread::yield();
          active_producers.fetch_sub(1, std::memory_order_acq_rel);
          return MakeElf(snapshot, kTestElfSize, static_cast<unsigned char>(index + 1));
        });
  };
  threads.emplace_back(run, 0, std::cref(first_source));
  threads.emplace_back(run, 1, std::cref(second_source));

  const bool both_active =
      WaitUntil([&] { return active_producers.load(std::memory_order_acquire) == 2; });
  release_producers.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(both_active);
  ASSERT_TRUE(results[0].succeeded());
  ASSERT_TRUE(results[1].succeeded());
  EXPECT_NE(results[0].elf.get(), results[1].elf.get());
  EXPECT_EQ(cache.SnapshotMetrics().source_snapshot_allocations, 2u);
}

TEST(HotswapCache, FinalOwnerReleaseExpiresPayloadAndSourceSnapshot) {
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();
  size_t producer_calls = 0;
  std::weak_ptr<const rocr::hotswap::RetargetedElf> weak;

  {
    const auto result =
        GetOrCompute(&cache, source, 0, key, [&](const SourceSnapshotRef& snapshot) {
          ++producer_calls;
          return MakeElf(snapshot);
        });
    ASSERT_TRUE(result.succeeded());
    weak = result.elf;
    EXPECT_EQ(cache.SnapshotMetrics().live_source_snapshot_bytes, source.size());
  }

  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(cache.ReadyEntryCountForTesting(), 0u);
  EXPECT_EQ(cache.SnapshotMetrics().live_source_snapshot_bytes, 0u);
  EXPECT_EQ(cache.SnapshotMetrics().live_output_bytes, 0u);

  const auto retry =
      GetOrCompute(&cache, source, 0, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        return MakeElf(snapshot);
      });
  EXPECT_TRUE(retry.succeeded());
  EXPECT_EQ(producer_calls, 2u);
}

TEST(HotswapCache, ConcurrentFailureWakesEveryWaiterAndRetries) {
  constexpr size_t kThreadCount = 16;
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();
  std::atomic<size_t> producer_calls{0};
  std::atomic<bool> release_producer{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = GetOrCompute(&cache, source, 0, key,
                                [&](const SourceSnapshotRef&) -> RetargetOperationResult {
                                  producer_calls.fetch_add(1, std::memory_order_relaxed);
                                  while (!release_producer.load(std::memory_order_acquire))
                                    std::this_thread::yield();
                                  throw std::runtime_error("injected producer failure");
                                });
    });
  }

  const bool all_waiters = WaitUntil([&] {
    return cache.WaiterCountForTesting(source.data(), source.size(), key) == kThreadCount - 1;
  });
  release_producer.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(all_waiters);
  EXPECT_EQ(producer_calls.load(std::memory_order_relaxed), 1u);
  for (const auto& result : results) {
    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.error, RetargetError::kComgrFailure);
  }

  const auto retry =
      GetOrCompute(&cache, source, 0, key, [&](const SourceSnapshotRef& snapshot) {
        producer_calls.fetch_add(1, std::memory_order_relaxed);
        return MakeElf(snapshot);
      });
  EXPECT_TRUE(retry.succeeded());
  EXPECT_EQ(producer_calls.load(std::memory_order_relaxed), 2u);
  EXPECT_EQ(cache.SnapshotMetrics().producer_failures, 1u);
  EXPECT_EQ(cache.SnapshotMetrics().in_flight_entries, 0u);
}

TEST(HotswapCache, SameThreadReentrantRequestIsRejectedWithoutWaiting) {
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();
  size_t producer_calls = 0;
  RetargetOperationResult nested;

  const auto outer = GetOrCompute(
      &cache, source, 1, key, [&](const SourceSnapshotRef& snapshot) {
        ++producer_calls;
        nested = GetOrCompute(
            &cache, source, 1, key, [&](const SourceSnapshotRef& nested_snapshot) {
              ++producer_calls;
              return MakeElf(nested_snapshot);
            });
        return MakeElf(snapshot);
      });

  ASSERT_TRUE(outer.succeeded());
  EXPECT_FALSE(nested.succeeded());
  EXPECT_EQ(nested.error, RetargetError::kReentrantRequest);
  EXPECT_EQ(producer_calls, 1u);
  const auto metrics = cache.SnapshotMetrics();
  EXPECT_EQ(metrics.reentrant_rejections, 1u);
  EXPECT_EQ(metrics.in_flight_entries, 0u);
}

TEST(HotswapCache, AllocationFailureIsTypedAndRetryable) {
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const RetargetCacheKey key = MakeKey();

  const auto failed = GetOrCompute(
      &cache, source, 0, key,
      [](const SourceSnapshotRef&) -> RetargetOperationResult { throw std::bad_alloc(); });
  const auto retry =
      GetOrCompute(&cache, source, 0, key,
                   [](const SourceSnapshotRef& snapshot) { return MakeElf(snapshot); });

  EXPECT_FALSE(failed.succeeded());
  EXPECT_EQ(failed.error, RetargetError::kOutOfResources);
  EXPECT_TRUE(retry.succeeded());
  EXPECT_EQ(cache.SnapshotMetrics().producer_failures, 1u);
}

TEST(HotswapCache, DifferentContentComputesOutsideOtherBucketLocks) {
  constexpr size_t kThreadCount = 8;
  ContentRetargetCache cache;
  std::vector<std::vector<unsigned char>> sources;
  std::atomic<size_t> active_producers{0};
  std::atomic<size_t> peak_producers{0};
  std::atomic<bool> release_producers{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    sources.push_back(MakeSource(static_cast<unsigned char>(i + 1)));
  }

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = GetOrCompute(
          &cache, sources[i], 0, MakeKey(i), [&](const SourceSnapshotRef& snapshot) {
            const size_t active = active_producers.fetch_add(1, std::memory_order_acq_rel) + 1;
            size_t peak = peak_producers.load(std::memory_order_relaxed);
            while (peak < active &&
                   !peak_producers.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {
            }
            while (!release_producers.load(std::memory_order_acquire)) std::this_thread::yield();
            active_producers.fetch_sub(1, std::memory_order_acq_rel);
            return MakeElf(snapshot, kTestElfSize, static_cast<unsigned char>(i));
          });
    });
  }

  const bool all_active =
      WaitUntil([&] { return active_producers.load(std::memory_order_acquire) == kThreadCount; });
  release_producers.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(all_active);
  EXPECT_EQ(peak_producers.load(std::memory_order_relaxed), kThreadCount);
  for (const auto& result : results) EXPECT_TRUE(result.succeeded());
}

TEST(HotswapCache, DifferentTransformsOfOneContentProduceConcurrently) {
  constexpr size_t kThreadCount = 8;
  ContentRetargetCache cache;
  const auto source = MakeSource();
  std::atomic<size_t> active_producers{0};
  std::atomic<size_t> peak_producers{0};
  std::atomic<bool> release_producers{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = GetOrCompute(
          &cache, source, 0, MakeKey(i), [&](const SourceSnapshotRef& snapshot) {
            const size_t active = active_producers.fetch_add(1, std::memory_order_acq_rel) + 1;
            size_t peak = peak_producers.load(std::memory_order_relaxed);
            while (peak < active &&
                   !peak_producers.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {
            }
            while (!release_producers.load(std::memory_order_acquire)) std::this_thread::yield();
            active_producers.fetch_sub(1, std::memory_order_acq_rel);
            return MakeElf(snapshot, kTestElfSize, static_cast<unsigned char>(i));
          });
    });
  }

  const bool all_active =
      WaitUntil([&] { return active_producers.load(std::memory_order_acquire) == kThreadCount; });
  release_producers.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(all_active);
  EXPECT_EQ(peak_producers.load(std::memory_order_relaxed), kThreadCount);
  for (const auto& result : results) EXPECT_TRUE(result.succeeded());
  const auto metrics = cache.SnapshotMetrics();
  EXPECT_EQ(metrics.producer_calls, kThreadCount);
  EXPECT_EQ(metrics.source_snapshot_allocations, 1u);
}

TEST(HotswapCache, EveryTransformKeyFieldSeparatesReadyEntries) {
  ContentRetargetCache cache;
  const auto source = MakeSource();
  const std::vector<RetargetCacheKey> keys{
      {"source", "target", false, false},       {"other-source", "target", false, false},
      {"source", "other-target", false, false}, {"source", "target", true, false},
      {"source", "target", false, true},
  };
  std::vector<RetargetOperationResult> results;

  for (size_t i = 0; i < keys.size(); ++i) {
    results.push_back(
        GetOrCompute(&cache, source, 0, keys[i], [i](const SourceSnapshotRef& snapshot) {
          return MakeElf(snapshot, kTestElfSize, static_cast<unsigned char>(i + 1));
        }));
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto hit = GetOrCompute(&cache, source, 0, keys[i], [](const SourceSnapshotRef&) {
      return RetargetOperationResult{{}, RetargetError::kComgrFailure};
    });
    ASSERT_TRUE(hit.succeeded());
    EXPECT_EQ(hit.source, RetargetResultSource::kReadyCache);
    EXPECT_EQ(hit.elf.get(), results[i].elf.get());
  }
  const auto metrics = cache.SnapshotMetrics();
  EXPECT_EQ(metrics.ready_entries, keys.size());
  EXPECT_EQ(metrics.source_snapshot_allocations, 1u);
}

TEST(HotswapCache, RandomizedStressPreservesExactContentSingleFlight) {
  constexpr size_t kKeyCount = 4;
  constexpr size_t kCallsPerThread = 64;
  const size_t thread_counts[] = {1, 2, 4, 8, 16, 32, 64};

  for (const size_t thread_count : thread_counts) {
    SCOPED_TRACE(thread_count);
    ContentRetargetCache cache;
    std::vector<RetargetCacheKey> keys;
    std::vector<std::vector<unsigned char>> sources;
    std::vector<std::atomic<size_t>> producer_calls(kKeyCount);
    std::vector<std::vector<RetargetOperationResult>> results(thread_count);
    std::vector<std::thread> threads;
    for (size_t key = 0; key < kKeyCount; ++key) {
      keys.push_back(MakeKey(key));
      sources.push_back(MakeSource(static_cast<unsigned char>(key + 1), 256));
      producer_calls[key].store(0, std::memory_order_relaxed);
    }

    for (size_t thread = 0; thread < thread_count; ++thread) {
      threads.emplace_back([&, thread] {
        uint32_t random = 0x9e3779b9u ^ static_cast<uint32_t>(thread + 1);
        results[thread].reserve(kCallsPerThread);
        for (size_t call = 0; call < kCallsPerThread; ++call) {
          const size_t key = (call + thread) % kKeyCount;
          const uint32_t delay = NextRandom(&random) & 7u;
          if (delay < 4) std::this_thread::yield();
          if (delay == 7) std::this_thread::sleep_for(std::chrono::microseconds(10));
          results[thread].push_back(
              GetOrCompute(&cache, sources[key], 0, keys[key],
                           [&, key](const SourceSnapshotRef& snapshot) {
                             producer_calls[key].fetch_add(1, std::memory_order_relaxed);
                             std::this_thread::sleep_for(std::chrono::microseconds(50));
                             return MakeElf(snapshot, 256, static_cast<unsigned char>(key));
                           }));
        }
      });
    }
    for (auto& thread : threads) thread.join();

    for (size_t key = 0; key < kKeyCount; ++key) {
      EXPECT_EQ(producer_calls[key].load(std::memory_order_relaxed), 1u);
    }
    for (size_t thread = 0; thread < thread_count; ++thread) {
      for (size_t call = 0; call < kCallsPerThread; ++call) {
        const size_t key = (call + thread) % kKeyCount;
        const auto& result = results[thread][call];
        ASSERT_TRUE(result.succeeded());
        EXPECT_EQ(*static_cast<const unsigned char*>(result.elf->data()),
                  static_cast<unsigned char>(key));
      }
    }

    const auto metrics = cache.SnapshotMetrics();
    EXPECT_EQ(metrics.producer_calls, kKeyCount);
    EXPECT_EQ(metrics.source_snapshot_allocations, kKeyCount);
    EXPECT_EQ(metrics.ready_hits + metrics.coalesced_results,
              thread_count * kCallsPerThread - kKeyCount);
  }
}

}  // namespace
