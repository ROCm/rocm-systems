////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/hotswap.hpp"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef ROCR_HOTSWAP_TESTING
#include <atomic>
#include <chrono>
#endif

namespace rocr {
namespace hotswap {
namespace {

struct RetargetCacheKeyHash {
  size_t operator()(const RetargetCacheKey& key) const {
    size_t hash = std::hash<std::string>{}(key.source_isa);
    hash ^= std::hash<std::string>{}(key.target_isa) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.entry_trampolines) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.strict_mode) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct RetargetFlight {
  bool completed = false;
  size_t waiter_count = 0;
  RetargetOperationResult result;
  std::condition_variable completed_cv;
};

RetargetOperationResult RunRetargetProducer(
    const std::function<RetargetOperationResult()>& producer) {
  try {
    RetargetOperationResult result = producer();
    if (result.succeeded()) {
      result.error = RetargetError::kNone;
    } else if (result.error == RetargetError::kNone) {
      result.error = RetargetError::kComgrFailure;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return {{}, RetargetError::kOutOfResources};
  } catch (...) {
    return {{}, RetargetError::kComgrFailure};
  }
}

#ifdef ROCR_HOTSWAP_TESTING
using MetricsClock = std::chrono::steady_clock;

void AddElapsed(std::atomic<uint64_t>* destination, MetricsClock::time_point start) {
  destination->fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(MetricsClock::now() - start).count(),
      std::memory_order_relaxed);
}

void UpdateMaximum(std::atomic<uint64_t>* destination, uint64_t candidate) {
  uint64_t current = destination->load(std::memory_order_relaxed);
  while (current < candidate &&
         !destination->compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
  }
}
#endif

}  // namespace

struct ReaderRetargetCache::Impl {
  mutable std::mutex mutex;
  std::unordered_map<RetargetCacheKey, std::weak_ptr<const RetargetedElf>, RetargetCacheKeyHash>
      ready;
  std::unordered_map<RetargetCacheKey, std::shared_ptr<RetargetFlight>, RetargetCacheKeyHash>
      in_flight;
#ifdef ROCR_HOTSWAP_TESTING
  std::atomic<uint64_t> producer_calls{0};
  std::atomic<uint64_t> producer_failures{0};
  std::atomic<uint64_t> ready_hits{0};
  std::atomic<uint64_t> coalesced_results{0};
  std::atomic<uint64_t> wait_nanoseconds{0};
  std::atomic<uint64_t> lock_hold_nanoseconds{0};
  std::atomic<uint64_t> produced_output_bytes{0};
  std::atomic<uint64_t> peak_live_output_bytes{0};
#endif
};

ReaderRetargetCache::ReaderRetargetCache() : impl_(new Impl()) {}
ReaderRetargetCache::~ReaderRetargetCache() = default;

RetargetOperationResult ReaderRetargetCache::GetOrCompute(
    const RetargetCacheKey& key, const std::function<RetargetOperationResult()>& producer) {
  std::shared_ptr<RetargetFlight> flight;
  const auto run_producer = [&] {
#ifdef ROCR_HOTSWAP_TESTING
    impl_->producer_calls.fetch_add(1, std::memory_order_relaxed);
#endif
    RetargetOperationResult result = RunRetargetProducer(producer);
#ifdef ROCR_HOTSWAP_TESTING
    if (result.succeeded()) {
      impl_->produced_output_bytes.fetch_add(result.elf->size(), std::memory_order_relaxed);
    } else {
      impl_->producer_failures.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    return result;
  };

  try {
    std::unique_lock<std::mutex> lock(impl_->mutex);
#ifdef ROCR_HOTSWAP_TESTING
    auto lock_acquired = MetricsClock::now();
#endif
    auto ready = impl_->ready.find(key);
    if (ready != impl_->ready.end()) {
      RetargetedElfRef elf = ready->second.lock();
      if (elf) {
#ifdef ROCR_HOTSWAP_TESTING
        impl_->ready_hits.fetch_add(1, std::memory_order_relaxed);
        AddElapsed(&impl_->lock_hold_nanoseconds, lock_acquired);
#endif
        return {std::move(elf), RetargetError::kNone, RetargetResultSource::kReadyCache};
      }
      impl_->ready.erase(ready);
    }

    auto in_flight = impl_->in_flight.find(key);
    if (in_flight != impl_->in_flight.end()) {
      flight = in_flight->second;
      ++flight->waiter_count;
#ifdef ROCR_HOTSWAP_TESTING
      AddElapsed(&impl_->lock_hold_nanoseconds, lock_acquired);
      const auto wait_started = MetricsClock::now();
#endif
      flight->completed_cv.wait(lock, [&] { return flight->completed; });
#ifdef ROCR_HOTSWAP_TESTING
      impl_->wait_nanoseconds.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(MetricsClock::now() - wait_started)
              .count(),
          std::memory_order_relaxed);
      lock_acquired = MetricsClock::now();
#endif
      --flight->waiter_count;
      RetargetOperationResult result = flight->result;
      result.source = RetargetResultSource::kCoalesced;
#ifdef ROCR_HOTSWAP_TESTING
      impl_->coalesced_results.fetch_add(1, std::memory_order_relaxed);
      AddElapsed(&impl_->lock_hold_nanoseconds, lock_acquired);
#endif
      return result;
    }

    flight = std::make_shared<RetargetFlight>();
    impl_->in_flight.emplace(key, flight);
#ifdef ROCR_HOTSWAP_TESTING
    AddElapsed(&impl_->lock_hold_nanoseconds, lock_acquired);
#endif
  } catch (const std::bad_alloc&) {
    // The cache is an optimization. Resource pressure may disable coalescing,
    // but it must not prevent an otherwise valid rewrite.
    return run_producer();
  }

  RetargetOperationResult result = run_producer();
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef ROCR_HOTSWAP_TESTING
    const auto lock_acquired = MetricsClock::now();
#endif
    flight->result = result;
    flight->completed = true;
    impl_->in_flight.erase(key);
    if (result.succeeded()) {
      try {
        for (auto entry = impl_->ready.begin(); entry != impl_->ready.end();) {
          if (entry->second.expired()) {
            entry = impl_->ready.erase(entry);
          } else {
            ++entry;
          }
        }
        impl_->ready.emplace(key, result.elf);
#ifdef ROCR_HOTSWAP_TESTING
        uint64_t live_bytes = 0;
        for (const auto& entry : impl_->ready) {
          if (RetargetedElfRef elf = entry.second.lock()) live_bytes += elf->size();
        }
        UpdateMaximum(&impl_->peak_live_output_bytes, live_bytes);
#endif
      } catch (const std::bad_alloc&) {
        // Active callers and loaded executables still own the result.
      }
    }
#ifdef ROCR_HOTSWAP_TESTING
    AddElapsed(&impl_->lock_hold_nanoseconds, lock_acquired);
#endif
  }
  flight->completed_cv.notify_all();
  return result;
}

#ifdef ROCR_HOTSWAP_TESTING
size_t ReaderRetargetCache::ReadyEntryCountForTesting() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  size_t count = 0;
  for (const auto& entry : impl_->ready) {
    if (!entry.second.expired()) ++count;
  }
  return count;
}

size_t ReaderRetargetCache::WaiterCountForTesting(const RetargetCacheKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto flight = impl_->in_flight.find(key);
  return flight == impl_->in_flight.end() ? 0 : flight->second->waiter_count;
}

ReaderRetargetCache::Metrics ReaderRetargetCache::MetricsForTesting() const {
  Metrics metrics;
  metrics.producer_calls = impl_->producer_calls.load(std::memory_order_relaxed);
  metrics.producer_failures = impl_->producer_failures.load(std::memory_order_relaxed);
  metrics.ready_hits = impl_->ready_hits.load(std::memory_order_relaxed);
  metrics.coalesced_results = impl_->coalesced_results.load(std::memory_order_relaxed);
  metrics.wait_nanoseconds = impl_->wait_nanoseconds.load(std::memory_order_relaxed);
  metrics.lock_hold_nanoseconds = impl_->lock_hold_nanoseconds.load(std::memory_order_relaxed);
  metrics.produced_output_bytes = impl_->produced_output_bytes.load(std::memory_order_relaxed);
  metrics.peak_live_output_bytes = impl_->peak_live_output_bytes.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(impl_->mutex);
  metrics.in_flight_entries = impl_->in_flight.size();
  for (const auto& entry : impl_->ready) {
    if (RetargetedElfRef elf = entry.second.lock()) {
      ++metrics.ready_entries;
      metrics.live_output_bytes += elf->size();
    }
  }
  return metrics;
}

void ReaderRetargetCache::ResetMetricsForTesting() {
  impl_->producer_calls.store(0, std::memory_order_relaxed);
  impl_->producer_failures.store(0, std::memory_order_relaxed);
  impl_->ready_hits.store(0, std::memory_order_relaxed);
  impl_->coalesced_results.store(0, std::memory_order_relaxed);
  impl_->wait_nanoseconds.store(0, std::memory_order_relaxed);
  impl_->lock_hold_nanoseconds.store(0, std::memory_order_relaxed);
  impl_->produced_output_bytes.store(0, std::memory_order_relaxed);

  const Metrics metrics = MetricsForTesting();
  impl_->peak_live_output_bytes.store(metrics.live_output_bytes, std::memory_order_relaxed);
}
#endif

}  // namespace hotswap
}  // namespace rocr
