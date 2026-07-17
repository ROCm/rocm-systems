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

using rocr::hotswap::ReaderRetargetCache;
using rocr::hotswap::RetargetCacheKey;
using rocr::hotswap::RetargetError;
using rocr::hotswap::RetargetOperationResult;
using rocr::hotswap::RetargetResultSource;

constexpr size_t kTestElfSize = 4096;

RetargetCacheKey MakeKey(size_t index = 0) {
  return {"amdgcn-amd-amdhsa--gfx1250:source-" + std::to_string(index),
          "amdgcn-amd-amdhsa--gfx1250:target-" + std::to_string(index), false, false};
}

RetargetOperationResult MakeElf(size_t size = kTestElfSize, unsigned char fill = 0x5a) {
  rocr::hotswap::OwnedElfBuffer bytes(std::malloc(size), &std::free);
  if (!bytes) return {{}, RetargetError::kOutOfResources};
  std::memset(bytes.get(), fill, size);
  return {std::make_shared<const rocr::hotswap::RetargetedElf>(std::move(bytes), size),
          RetargetError::kNone};
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
  ReaderRetargetCache cache;
  const RetargetCacheKey key = MakeKey();
  std::atomic<size_t> producer_calls{0};
  std::atomic<bool> release_producer{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = cache.GetOrCompute(key, [&] {
        producer_calls.fetch_add(1, std::memory_order_relaxed);
        while (!release_producer.load(std::memory_order_acquire)) std::this_thread::yield();
        return MakeElf();
      });
    });
  }

  const bool all_waiters =
      WaitUntil([&] { return cache.WaiterCountForTesting(key) == kThreadCount - 1; });
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

  const ReaderRetargetCache::Metrics metrics = cache.MetricsForTesting();
  EXPECT_EQ(computed, 1u);
  EXPECT_EQ(coalesced, kThreadCount - 1);
  EXPECT_EQ(metrics.producer_calls, 1u);
  EXPECT_EQ(metrics.coalesced_results, kThreadCount - 1);
  EXPECT_EQ(metrics.produced_output_bytes, kTestElfSize);
  EXPECT_EQ(metrics.live_output_bytes, kTestElfSize);
  EXPECT_EQ(metrics.peak_live_output_bytes, kTestElfSize);
  EXPECT_GT(metrics.wait_nanoseconds, 0u);
  EXPECT_GT(metrics.lock_hold_nanoseconds, 0u);

  results.clear();
  EXPECT_EQ(cache.MetricsForTesting().live_output_bytes, 0u);
}

TEST(HotswapCache, ConcurrentFailureWakesEveryWaiterAndRetries) {
  constexpr size_t kThreadCount = 16;
  ReaderRetargetCache cache;
  const RetargetCacheKey key = MakeKey();
  std::atomic<size_t> producer_calls{0};
  std::atomic<bool> release_producer{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = cache.GetOrCompute(key, [&]() -> RetargetOperationResult {
        producer_calls.fetch_add(1, std::memory_order_relaxed);
        while (!release_producer.load(std::memory_order_acquire)) std::this_thread::yield();
        throw std::runtime_error("injected producer failure");
      });
    });
  }

  const bool all_waiters =
      WaitUntil([&] { return cache.WaiterCountForTesting(key) == kThreadCount - 1; });
  release_producer.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  ASSERT_TRUE(all_waiters);
  EXPECT_EQ(producer_calls.load(std::memory_order_relaxed), 1u);
  for (const auto& result : results) {
    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.error, RetargetError::kComgrFailure);
  }

  const RetargetOperationResult retry = cache.GetOrCompute(key, [&] {
    producer_calls.fetch_add(1, std::memory_order_relaxed);
    return MakeElf();
  });
  ASSERT_TRUE(retry.succeeded());
  EXPECT_EQ(producer_calls.load(std::memory_order_relaxed), 2u);

  const ReaderRetargetCache::Metrics metrics = cache.MetricsForTesting();
  EXPECT_EQ(metrics.producer_calls, 2u);
  EXPECT_EQ(metrics.producer_failures, 1u);
  EXPECT_EQ(metrics.coalesced_results, kThreadCount - 1);
  EXPECT_EQ(metrics.in_flight_entries, 0u);
}

TEST(HotswapCache, AllocationFailureIsTypedAndRetryable) {
  ReaderRetargetCache cache;
  const RetargetCacheKey key = MakeKey();

  const RetargetOperationResult failed =
      cache.GetOrCompute(key, []() -> RetargetOperationResult { throw std::bad_alloc(); });
  const RetargetOperationResult retry = cache.GetOrCompute(key, [] { return MakeElf(); });

  EXPECT_FALSE(failed.succeeded());
  EXPECT_EQ(failed.error, RetargetError::kOutOfResources);
  EXPECT_TRUE(retry.succeeded());
  EXPECT_EQ(cache.MetricsForTesting().producer_failures, 1u);
}

TEST(HotswapCache, DifferentKeysComputeOutsideTheCacheMutex) {
  constexpr size_t kThreadCount = 8;
  ReaderRetargetCache cache;
  std::atomic<size_t> active_producers{0};
  std::atomic<size_t> peak_producers{0};
  std::atomic<bool> release_producers{false};
  std::vector<RetargetOperationResult> results(kThreadCount);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      results[i] = cache.GetOrCompute(MakeKey(i), [&, i] {
        const size_t active = active_producers.fetch_add(1, std::memory_order_acq_rel) + 1;
        size_t peak = peak_producers.load(std::memory_order_relaxed);
        while (peak < active &&
               !peak_producers.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {
        }
        while (!release_producers.load(std::memory_order_acquire)) std::this_thread::yield();
        active_producers.fetch_sub(1, std::memory_order_acq_rel);
        return MakeElf(kTestElfSize, static_cast<unsigned char>(i));
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
  EXPECT_EQ(cache.MetricsForTesting().producer_calls, kThreadCount);
}

TEST(HotswapCache, EveryTransformKeyFieldSeparatesReadyEntries) {
  ReaderRetargetCache cache;
  const std::vector<RetargetCacheKey> keys{
      {"source", "target", false, false},
      {"other-source", "target", false, false},
      {"source", "other-target", false, false},
      {"source", "target", true, false},
      {"source", "target", false, true},
  };
  std::vector<RetargetOperationResult> results;

  for (size_t i = 0; i < keys.size(); ++i) {
    results.push_back(cache.GetOrCompute(keys[i], [i] {
      return MakeElf(kTestElfSize, static_cast<unsigned char>(i + 1));
    }));
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    const RetargetOperationResult hit = cache.GetOrCompute(keys[i], [] {
      return RetargetOperationResult{{}, RetargetError::kComgrFailure};
    });
    ASSERT_TRUE(hit.succeeded());
    EXPECT_EQ(hit.source, RetargetResultSource::kReadyCache);
    EXPECT_EQ(hit.elf.get(), results[i].elf.get());
    EXPECT_EQ(*static_cast<const unsigned char*>(hit.elf->data()),
              static_cast<unsigned char>(i + 1));
  }

  const ReaderRetargetCache::Metrics metrics = cache.MetricsForTesting();
  EXPECT_EQ(metrics.producer_calls, keys.size());
  EXPECT_EQ(metrics.ready_hits, keys.size());
  EXPECT_EQ(metrics.ready_entries, keys.size());
}

TEST(HotswapCache, RandomizedStressPreservesSingleFlightAndExactKeys) {
  constexpr size_t kKeyCount = 4;
  constexpr size_t kCallsPerThread = 64;
  constexpr size_t kPayloadSize = 256;
  const size_t thread_counts[] = {1, 2, 4, 8, 16, 32, 64};

  for (const size_t thread_count : thread_counts) {
    SCOPED_TRACE(thread_count);
    ReaderRetargetCache cache;
    std::vector<RetargetCacheKey> keys;
    std::vector<std::atomic<size_t>> producer_calls(kKeyCount);
    std::vector<std::vector<RetargetOperationResult>> results(thread_count);
    std::vector<std::thread> threads;
    for (size_t key = 0; key < kKeyCount; ++key) keys.push_back(MakeKey(key));
    for (auto& calls : producer_calls) calls.store(0, std::memory_order_relaxed);

    for (size_t thread = 0; thread < thread_count; ++thread) {
      threads.emplace_back([&, thread] {
        uint32_t random = 0x9e3779b9u ^ static_cast<uint32_t>(thread + 1);
        results[thread].reserve(kCallsPerThread);
        for (size_t call = 0; call < kCallsPerThread; ++call) {
          const size_t key = (call + thread) % kKeyCount;
          const uint32_t delay = NextRandom(&random) & 7u;
          if (delay < 4) {
            std::this_thread::yield();
          } else if (delay == 7) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          results[thread].push_back(cache.GetOrCompute(keys[key], [&, key] {
            producer_calls[key].fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            return MakeElf(kPayloadSize, static_cast<unsigned char>(key));
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

    const ReaderRetargetCache::Metrics metrics = cache.MetricsForTesting();
    EXPECT_EQ(metrics.producer_calls, kKeyCount);
    EXPECT_EQ(metrics.produced_output_bytes, kKeyCount * kPayloadSize);
    EXPECT_EQ(metrics.live_output_bytes, kKeyCount * kPayloadSize);
    EXPECT_EQ(metrics.peak_live_output_bytes, kKeyCount * kPayloadSize);
    EXPECT_EQ(metrics.ready_hits + metrics.coalesced_results,
              thread_count * kCallsPerThread - kKeyCount);
  }
}

}  // namespace
