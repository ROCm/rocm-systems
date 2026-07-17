//===- hotswap_cache_benchmark.cc - HotSwap cache benchmark --------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/inc/hotswap.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using rocr::hotswap::ContentRetargetCache;
using rocr::hotswap::RetargetCacheKey;
using rocr::hotswap::RetargetError;
using rocr::hotswap::RetargetOperationResult;
using rocr::hotswap::SourceSnapshotRef;

enum class Phase { kCold, kWarm };

struct Options {
  std::vector<size_t> thread_counts{1, 2, 4, 8};
  std::vector<size_t> payload_sizes{64 * 1024, 1024 * 1024, 8 * 1024 * 1024};
  size_t producer_microseconds = 2000;
  // Models the cost of a cross-process disk-cache read (I/O + validation) that
  // the disk tier pays on a warm start in place of the full COMGR retarget.
  // Much smaller than producer_microseconds by design.
  size_t disk_read_microseconds = 200;
  size_t iterations = 3;
};

struct Outcome {
  uint64_t elapsed_microseconds = 0;
  uint64_t producer_calls = 0;
  uint64_t payload_allocations = 0;
  uint64_t peak_payload_bytes = 0;
  uint64_t retained_payload_bytes = 0;
  uint64_t source_snapshot_allocations = 0;
  uint64_t peak_source_snapshot_bytes = 0;
  uint64_t retained_source_snapshot_bytes = 0;
  uint64_t ready_hits = 0;
  uint64_t coalesced_results = 0;
};

class StartGate {
 public:
  explicit StartGate(size_t expected) : expected_(expected) {}

  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (++arrived_ == expected_) {
      open_ = true;
      condition_.notify_all();
      return;
    }
    condition_.wait(lock, [&] { return open_; });
  }

 private:
  const size_t expected_;
  std::mutex mutex_;
  std::condition_variable condition_;
  size_t arrived_ = 0;
  bool open_ = false;
};

class PayloadTracker {
 public:
  void Allocate(size_t size) {
    allocations_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t live = live_bytes_.fetch_add(size, std::memory_order_relaxed) + size;
    uint64_t peak = peak_bytes_.load(std::memory_order_relaxed);
    while (peak < live &&
           !peak_bytes_.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
  }

  void Release(size_t size) { live_bytes_.fetch_sub(size, std::memory_order_relaxed); }

  void ResetMeasurements() {
    allocations_.store(0, std::memory_order_relaxed);
    peak_bytes_.store(live_bytes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  uint64_t allocations() const { return allocations_.load(std::memory_order_relaxed); }
  uint64_t live_bytes() const { return live_bytes_.load(std::memory_order_relaxed); }
  uint64_t peak_bytes() const { return peak_bytes_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> allocations_{0};
  std::atomic<uint64_t> live_bytes_{0};
  std::atomic<uint64_t> peak_bytes_{0};
};

class Payload {
 public:
  Payload(size_t size, PayloadTracker* tracker, unsigned char fill)
      : bytes_(size, fill), tracker_(tracker) {
    tracker_->Allocate(bytes_.size());
  }

  ~Payload() { tracker_->Release(bytes_.size()); }

  Payload(const Payload&) = delete;
  Payload& operator=(const Payload&) = delete;

  const unsigned char* data() const { return bytes_.data(); }
  unsigned char* data() { return bytes_.data(); }
  size_t size() const { return bytes_.size(); }

 private:
  std::vector<unsigned char> bytes_;
  PayloadTracker* tracker_;
};

std::vector<size_t> ParseList(const std::string& value) {
  std::vector<size_t> values;
  size_t start = 0;
  while (start < value.size()) {
    const size_t comma = value.find(',', start);
    values.push_back(std::stoull(value.substr(start, comma - start)));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--threads=", 0) == 0) {
      options.thread_counts = ParseList(argument.substr(std::strlen("--threads=")));
    } else if (argument.rfind("--bytes=", 0) == 0) {
      options.payload_sizes = ParseList(argument.substr(std::strlen("--bytes=")));
    } else if (argument.rfind("--producer-us=", 0) == 0) {
      options.producer_microseconds = std::stoull(argument.substr(std::strlen("--producer-us=")));
    } else if (argument.rfind("--disk-read-us=", 0) == 0) {
      options.disk_read_microseconds = std::stoull(argument.substr(std::strlen("--disk-read-us=")));
    } else if (argument.rfind("--iterations=", 0) == 0) {
      options.iterations = std::stoull(argument.substr(std::strlen("--iterations=")));
    } else if (argument == "--help") {
      std::cout << "Usage: hotswap_cache_benchmark [--threads=1,2,4,8] "
                   "[--bytes=65536,1048576] [--producer-us=2000] "
                   "[--disk-read-us=200] [--iterations=3]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      std::exit(2);
    }
  }
  const auto contains_zero = [](const std::vector<size_t>& values) {
    return std::any_of(values.begin(), values.end(), [](size_t value) { return value == 0; });
  };
  if (options.thread_counts.empty() || options.payload_sizes.empty() || options.iterations == 0 ||
      contains_zero(options.thread_counts) || contains_zero(options.payload_sizes)) {
    std::cerr << "Thread counts, payload sizes, and iterations must be nonzero\n";
    std::exit(2);
  }
  return options;
}

std::shared_ptr<Payload> MakePayload(size_t size, PayloadTracker* tracker,
                                     unsigned char fill = 0x5a) {
  return std::make_shared<Payload>(size, tracker, fill);
}

Outcome RunNoCache(size_t thread_count, size_t payload_size, size_t producer_microseconds) {
  PayloadTracker tracker;
  std::atomic<uint64_t> producer_calls{0};
  std::vector<std::shared_ptr<Payload>> results(thread_count);
  std::vector<std::thread> threads;
  StartGate start(thread_count);

  const auto started = Clock::now();
  for (size_t i = 0; i < thread_count; ++i) {
    threads.emplace_back([&, i] {
      start.ArriveAndWait();
      producer_calls.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::microseconds(producer_microseconds));
      results[i] = MakePayload(payload_size, &tracker);
    });
  }
  for (auto& thread : threads) thread.join();

  return {
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count()),
      producer_calls.load(std::memory_order_relaxed),
      tracker.allocations(),
      tracker.peak_bytes(),
      tracker.live_bytes(),
      0,
      0,
      0,
      0,
      0};
}

Outcome RunCopyCache(Phase phase, size_t thread_count, size_t payload_size,
                     size_t producer_microseconds) {
  PayloadTracker tracker;
  std::mutex cache_mutex;
  std::shared_ptr<Payload> cached;
  if (phase == Phase::kWarm) cached = MakePayload(payload_size, &tracker);
  tracker.ResetMeasurements();

  std::atomic<uint64_t> producer_calls{0};
  std::atomic<uint64_t> ready_hits{0};
  std::vector<std::shared_ptr<Payload>> results(thread_count);
  std::vector<std::thread> threads;
  StartGate start(thread_count);
  StartGate cold_miss_gate(thread_count);

  const auto started = Clock::now();
  for (size_t i = 0; i < thread_count; ++i) {
    threads.emplace_back([&, i] {
      start.ArriveAndWait();
      std::shared_ptr<Payload> cached_result;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cached_result = cached;
      }

      if (cached_result) {
        ready_hits.fetch_add(1, std::memory_order_relaxed);
        results[i] = MakePayload(payload_size, &tracker, 0);
        std::memcpy(results[i]->data(), cached_result->data(), payload_size);
        return;
      }

      // Force every cold caller to observe the same miss, matching the race
      // that a lookup-only cache cannot coalesce.
      cold_miss_gate.ArriveAndWait();
      producer_calls.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::microseconds(producer_microseconds));
      results[i] = MakePayload(payload_size, &tracker);

      std::shared_ptr<Payload> cache_candidate = MakePayload(payload_size, &tracker, 0);
      std::memcpy(cache_candidate->data(), results[i]->data(), payload_size);
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (!cached) cached = std::move(cache_candidate);
    });
  }
  for (auto& thread : threads) thread.join();

  return {
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count()),
      producer_calls.load(std::memory_order_relaxed),
      tracker.allocations(),
      tracker.peak_bytes(),
      tracker.live_bytes(),
      0,
      0,
      0,
      ready_hits.load(std::memory_order_relaxed),
      0};
}

RetargetOperationResult MakeRetargetedElf(const SourceSnapshotRef& source, size_t size,
                                          size_t producer_microseconds) {
  std::this_thread::sleep_for(std::chrono::microseconds(producer_microseconds));
  rocr::hotswap::OwnedElfBuffer bytes(std::malloc(size), &std::free);
  if (!bytes) return {{}, RetargetError::kOutOfResources};
  std::memset(bytes.get(), 0x5a, size);
  return {std::make_shared<const rocr::hotswap::RetargetedElf>(std::move(bytes), size, source),
          RetargetError::kNone};
}

Outcome RunSingleFlight(Phase phase, size_t thread_count, size_t payload_size,
                        size_t producer_microseconds) {
  ContentRetargetCache cache;
  const RetargetCacheKey key{"source", "target", false, false};
  const std::vector<unsigned char> source(payload_size, 0x21);
  rocr::hotswap::RetargetedElfRef warm_owner;
  if (phase == Phase::kWarm) {
    warm_owner = cache
                     .GetOrCompute(source.data(), source.size(), 0, key,
                                   [&](const SourceSnapshotRef& snapshot) {
                                     return MakeRetargetedElf(snapshot, payload_size, 0);
                                   })
                     .elf;
  }
  cache.ResetMetricsForTesting();

  std::vector<RetargetOperationResult> results(thread_count);
  std::vector<std::thread> threads;
  StartGate start(thread_count);

  const auto started = Clock::now();
  for (size_t i = 0; i < thread_count; ++i) {
    threads.emplace_back([&, i] {
      start.ArriveAndWait();
      results[i] = cache.GetOrCompute(
          source.data(), source.size(), 0, key, [&](const SourceSnapshotRef& snapshot) {
            return MakeRetargetedElf(snapshot, payload_size, producer_microseconds);
          });
    });
  }
  for (auto& thread : threads) thread.join();

  const auto metrics = cache.SnapshotMetrics();
  for (const auto& result : results) {
    if (!result.succeeded()) {
      std::cerr << "single-flight allocation failed\n";
      std::exit(1);
    }
  }
  return {
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count()),
      metrics.producer_calls,
      metrics.produced_output_bytes / payload_size,
      metrics.peak_live_output_bytes,
      metrics.live_output_bytes,
      metrics.source_snapshot_allocations,
      metrics.peak_live_source_snapshot_bytes,
      metrics.live_source_snapshot_bytes,
      metrics.ready_hits,
      metrics.coalesced_results};
}

// Models the disk-persistent tier layered on the single-flight cache: on a
// cross-process warm start the on-disk artifact already exists, so the
// single-flight producer pays a cheap disk read (disk_read_microseconds) plus
// a copy-out instead of the full COMGR retarget (producer_microseconds). Uses
// the real ContentRetargetCache so single-flight coalescing is identical to the
// in-memory-only strategy; only the producer's cost differs. kWarm additionally
// pre-populates the in-memory tier (same-process second launch).
Outcome RunSingleFlightDisk(Phase phase, size_t thread_count, size_t payload_size,
                            size_t disk_read_microseconds) {
  ContentRetargetCache cache;
  const RetargetCacheKey key{"source", "target", false, false};
  const std::vector<unsigned char> source(payload_size, 0x21);
  rocr::hotswap::RetargetedElfRef warm_owner;
  if (phase == Phase::kWarm) {
    warm_owner = cache
                     .GetOrCompute(source.data(), source.size(), 0, key,
                                   [&](const SourceSnapshotRef& snapshot) {
                                     return MakeRetargetedElf(snapshot, payload_size, 0);
                                   })
                     .elf;
  }
  cache.ResetMetricsForTesting();

  std::vector<RetargetOperationResult> results(thread_count);
  std::vector<std::thread> threads;
  StartGate start(thread_count);

  const auto started = Clock::now();
  for (size_t i = 0; i < thread_count; ++i) {
    threads.emplace_back([&, i] {
      start.ArriveAndWait();
      results[i] = cache.GetOrCompute(
          source.data(), source.size(), 0, key, [&](const SourceSnapshotRef& snapshot) {
            // Disk hit: cheap read + copy-out, NOT the full COMGR retarget.
            return MakeRetargetedElf(snapshot, payload_size, disk_read_microseconds);
          });
    });
  }
  for (auto& thread : threads) thread.join();

  const auto metrics = cache.SnapshotMetrics();
  for (const auto& result : results) {
    if (!result.succeeded()) {
      std::cerr << "single-flight+disk allocation failed\n";
      std::exit(1);
    }
  }
  return {
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count()),
      metrics.producer_calls,
      metrics.produced_output_bytes / payload_size,
      metrics.peak_live_output_bytes,
      metrics.live_output_bytes,
      metrics.source_snapshot_allocations,
      metrics.peak_live_source_snapshot_bytes,
      metrics.live_source_snapshot_bytes,
      metrics.ready_hits,
      metrics.coalesced_results};
}

template <typename Runner> Outcome MedianOutcome(size_t iterations, Runner&& runner) {
  std::vector<Outcome> outcomes;
  outcomes.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) outcomes.push_back(runner());
  std::sort(outcomes.begin(), outcomes.end(), [](const Outcome& left, const Outcome& right) {
    return left.elapsed_microseconds < right.elapsed_microseconds;
  });
  return outcomes[outcomes.size() / 2];
}

void PrintOutcome(const char* strategy, Phase phase, size_t thread_count, size_t payload_size,
                  size_t producer_microseconds, const Outcome& outcome) {
  std::cout << strategy << ',' << (phase == Phase::kCold ? "cold" : "warm") << ',' << thread_count
            << ',' << payload_size << ',' << producer_microseconds << ','
            << outcome.elapsed_microseconds << ',' << outcome.producer_calls << ','
            << outcome.payload_allocations << ',' << outcome.peak_payload_bytes << ','
            << outcome.retained_payload_bytes << ',' << outcome.source_snapshot_allocations << ','
            << outcome.peak_source_snapshot_bytes << ',' << outcome.retained_source_snapshot_bytes
            << ',' << outcome.ready_hits << ',' << outcome.coalesced_results << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = ParseOptions(argc, argv);
  std::cout << "strategy,phase,threads,payload_bytes,producer_us,latency_us,producer_calls,"
               "payload_allocations,peak_payload_bytes,retained_payload_bytes,"
               "source_snapshot_allocations,peak_source_snapshot_bytes,"
               "retained_source_snapshot_bytes,ready_hits,coalesced_results\n";

  for (const size_t payload_size : options.payload_sizes) {
    for (const size_t thread_count : options.thread_counts) {
      for (const Phase phase : {Phase::kCold, Phase::kWarm}) {
        PrintOutcome("no-cache", phase, thread_count, payload_size, options.producer_microseconds,
                     MedianOutcome(options.iterations, [&] {
                       return RunNoCache(thread_count, payload_size, options.producer_microseconds);
                     }));
        PrintOutcome("pr8274-copy-cache", phase, thread_count, payload_size,
                     options.producer_microseconds, MedianOutcome(options.iterations, [&] {
                       return RunCopyCache(phase, thread_count, payload_size,
                                           options.producer_microseconds);
                     }));
        PrintOutcome("content-single-flight", phase, thread_count, payload_size,
                     options.producer_microseconds, MedianOutcome(options.iterations, [&] {
                       return RunSingleFlight(phase, thread_count, payload_size,
                                              options.producer_microseconds);
                     }));
        PrintOutcome("single-flight+disk", phase, thread_count, payload_size,
                     options.disk_read_microseconds, MedianOutcome(options.iterations, [&] {
                       return RunSingleFlightDisk(phase, thread_count, payload_size,
                                                  options.disk_read_microseconds);
                     }));
      }
    }
  }
  return 0;
}
