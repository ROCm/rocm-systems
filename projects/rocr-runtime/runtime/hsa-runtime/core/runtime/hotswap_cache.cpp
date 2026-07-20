/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/hotswap.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocr {
namespace hotswap {

namespace {

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

uint64_t RotateLeft(uint64_t value, unsigned int count) {
  return (value << count) | (value >> (64 - count));
}

uint64_t Read64(const unsigned char* bytes) {
  uint64_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

uint32_t Read32(const unsigned char* bytes) {
  uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

constexpr uint64_t kHashPrime1 = 11400714785074694791ULL;
constexpr uint64_t kHashPrime2 = 14029467366897019727ULL;
constexpr uint64_t kHashPrime3 = 1609587929392839161ULL;
constexpr uint64_t kHashPrime4 = 9650029242287828579ULL;
constexpr uint64_t kHashPrime5 = 2870177450012600261ULL;

uint64_t HashRound(uint64_t accumulator, uint64_t input) {
  accumulator += input * kHashPrime2;
  accumulator = RotateLeft(accumulator, 31);
  return accumulator * kHashPrime1;
}

uint64_t MergeHashRound(uint64_t accumulator, uint64_t value) {
  accumulator ^= HashRound(0, value);
  return accumulator * kHashPrime1 + kHashPrime4;
}

// xxHash64 selects a collision bucket. Exact byte comparison determines
// identity, so neither correctness nor isolation depends on hash strength.
uint64_t HashContent(const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  const unsigned char* cursor = bytes;
  const unsigned char* const end = bytes + size;
  constexpr uint64_t kSeed = 0x4f1bbcdc6762f36bULL;
  uint64_t hash = 0;

  if (size >= 32) {
    const unsigned char* const block_end = end - 32;
    uint64_t lane1 = kSeed + kHashPrime1 + kHashPrime2;
    uint64_t lane2 = kSeed + kHashPrime2;
    uint64_t lane3 = kSeed;
    uint64_t lane4 = kSeed - kHashPrime1;
    do {
      lane1 = HashRound(lane1, Read64(cursor));
      cursor += 8;
      lane2 = HashRound(lane2, Read64(cursor));
      cursor += 8;
      lane3 = HashRound(lane3, Read64(cursor));
      cursor += 8;
      lane4 = HashRound(lane4, Read64(cursor));
      cursor += 8;
    } while (cursor <= block_end);

    hash =
        RotateLeft(lane1, 1) + RotateLeft(lane2, 7) + RotateLeft(lane3, 12) + RotateLeft(lane4, 18);
    hash = MergeHashRound(hash, lane1);
    hash = MergeHashRound(hash, lane2);
    hash = MergeHashRound(hash, lane3);
    hash = MergeHashRound(hash, lane4);
  } else {
    hash = kSeed + kHashPrime5;
  }

  hash += size;
  while (static_cast<size_t>(end - cursor) >= 8) {
    hash ^= HashRound(0, Read64(cursor));
    hash = RotateLeft(hash, 27) * kHashPrime1 + kHashPrime4;
    cursor += 8;
  }
  if (static_cast<size_t>(end - cursor) >= 4) {
    hash ^= static_cast<uint64_t>(Read32(cursor)) * kHashPrime1;
    hash = RotateLeft(hash, 23) * kHashPrime2 + kHashPrime3;
    cursor += 4;
  }
  while (cursor != end) {
    hash ^= static_cast<uint64_t>(*cursor++) * kHashPrime5;
    hash = RotateLeft(hash, 11) * kHashPrime1;
  }

  hash ^= hash >> 33;
  hash *= kHashPrime2;
  hash ^= hash >> 29;
  hash *= kHashPrime3;
  return hash ^ (hash >> 32);
}

size_t CombineHash(size_t hash, size_t value) {
  return hash ^ (value + 0x9e3779b9U + (hash << 6) + (hash >> 2));
}

struct SourceBucketKey {
  uint64_t content_hash = 0;
  size_t source_size = 0;

  bool operator==(const SourceBucketKey& other) const {
    return content_hash == other.content_hash && source_size == other.source_size;
  }
};

struct SourceBucketKeyHash {
  size_t operator()(const SourceBucketKey& key) const {
    return CombineHash(std::hash<uint64_t>{}(key.content_hash),
                       std::hash<size_t>{}(key.source_size));
  }
};

struct RetargetCacheBucketKey {
  RetargetCacheKey transform;
  uint64_t content_hash = 0;
  size_t source_size = 0;

  bool operator==(const RetargetCacheBucketKey& other) const {
    return transform == other.transform && content_hash == other.content_hash &&
        source_size == other.source_size;
  }
};

struct RetargetCacheBucketKeyHash {
  size_t operator()(const RetargetCacheBucketKey& key) const {
    size_t hash = std::hash<std::string>{}(key.transform.source_isa);
    hash = CombineHash(hash, std::hash<std::string>{}(key.transform.target_isa));
    hash = CombineHash(hash, std::hash<bool>{}(key.transform.entry_trampolines));
    hash = CombineHash(hash, std::hash<bool>{}(key.transform.strict_mode));
    hash = CombineHash(hash, std::hash<uint64_t>{}(key.content_hash));
    return CombineHash(hash, std::hash<size_t>{}(key.source_size));
  }
};

struct RetargetReadyEntry {
  std::weak_ptr<const RetargetedElf> elf;
  uint64_t producer_reader_id = 0;
};

struct RetargetFlight {
  RetargetFlight(SourceSnapshotRef source_snapshot, uint64_t reader_id)
      : source(std::move(source_snapshot)),
        producer_reader_id(reader_id),
        producer_thread_id(std::this_thread::get_id()) {}

  SourceSnapshotRef source;
  uint64_t producer_reader_id = 0;
  std::thread::id producer_thread_id;
  bool completed = false;
  size_t waiter_count = 0;
  RetargetOperationResult result;
  std::condition_variable completed_cv;
};

struct RetargetCacheBucket {
  mutable std::mutex mutex;
  std::vector<RetargetReadyEntry> ready;
  std::vector<std::shared_ptr<RetargetFlight>> in_flight;
};

struct SourceBucket {
  mutable std::mutex mutex;
  std::vector<std::weak_ptr<const SourceSnapshot>> snapshots;
  uint64_t generation = 0;
};

RetargetOperationResult RunRetargetProducer(const ContentRetargetCache::Producer& producer,
                                            const SourceSnapshotRef& source) {
  try {
    RetargetOperationResult result = producer(source);
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

bool IsCrossReader(uint64_t producer_reader_id, uint64_t reader_id) {
  return producer_reader_id != 0 && reader_id != 0 && producer_reader_id != reader_id;
}

}  // namespace

class RetargetCacheMemoryTracker final {
 public:
  std::atomic<uint64_t> producer_calls{0};
  std::atomic<uint64_t> producer_failures{0};
  std::atomic<uint64_t> ready_hits{0};
  std::atomic<uint64_t> cross_reader_results{0};
  std::atomic<uint64_t> coalesced_results{0};
  std::atomic<uint64_t> reentrant_rejections{0};
  std::atomic<uint64_t> hash_bytes{0};
  std::atomic<uint64_t> hash_nanoseconds{0};
  std::atomic<uint64_t> exact_compare_bytes{0};
  std::atomic<uint64_t> exact_compare_nanoseconds{0};
  std::atomic<uint64_t> wait_nanoseconds{0};
  std::atomic<uint64_t> lock_hold_nanoseconds{0};
  std::atomic<uint64_t> source_snapshot_allocations{0};
  std::atomic<uint64_t> source_snapshot_bytes{0};
  std::atomic<uint64_t> live_source_snapshot_bytes{0};
  std::atomic<uint64_t> peak_live_source_snapshot_bytes{0};
  std::atomic<uint64_t> produced_output_bytes{0};
  std::atomic<uint64_t> live_output_bytes{0};
  std::atomic<uint64_t> peak_live_output_bytes{0};
};

SourceSnapshot::SourceSnapshot(std::unique_ptr<unsigned char[]> bytes, size_t size,
                               std::shared_ptr<RetargetCacheMemoryTracker> tracker)
    : bytes_(std::move(bytes)), size_(size), tracker_(std::move(tracker)) {
  tracker_->source_snapshot_allocations.fetch_add(1, std::memory_order_relaxed);
  tracker_->source_snapshot_bytes.fetch_add(size_, std::memory_order_relaxed);
  const uint64_t live =
      tracker_->live_source_snapshot_bytes.fetch_add(size_, std::memory_order_relaxed) + size_;
  UpdateMaximum(&tracker_->peak_live_source_snapshot_bytes, live);
}

SourceSnapshot::~SourceSnapshot() {
  tracker_->live_source_snapshot_bytes.fetch_sub(size_, std::memory_order_relaxed);
}

bool SourceSnapshot::Equals(const void* data, size_t size) const {
  return data != nullptr && size == size_ && std::memcmp(bytes_.get(), data, size_) == 0;
}

RetargetedElf::RetargetedElf(OwnedElfBuffer bytes, size_t size, SourceSnapshotRef source)
    : bytes_(std::move(bytes)), size_(size), source_(std::move(source)) {
  if (!source_) return;
  tracker_ = source_->tracker_;
  tracker_->produced_output_bytes.fetch_add(size_, std::memory_order_relaxed);
  const uint64_t live =
      tracker_->live_output_bytes.fetch_add(size_, std::memory_order_relaxed) + size_;
  UpdateMaximum(&tracker_->peak_live_output_bytes, live);
}

RetargetedElf::~RetargetedElf() {
  if (tracker_) tracker_->live_output_bytes.fetch_sub(size_, std::memory_order_relaxed);
}

struct ContentRetargetCache::Impl {
  using HashFunction = std::function<uint64_t(const void*, size_t)>;

  explicit Impl(HashFunction function)
      : hash_function(function ? std::move(function) : HashFunction(HashContent)),
        tracker(std::make_shared<RetargetCacheMemoryTracker>()) {}

  std::shared_ptr<SourceBucket> GetOrCreateSourceBucket(const SourceBucketKey& key) {
    std::lock_guard<std::mutex> lock(table_mutex);
    const auto lock_acquired = MetricsClock::now();
    const auto existing = source_buckets.find(key);
    if (existing != source_buckets.end()) {
      AddElapsed(&tracker->lock_hold_nanoseconds, lock_acquired);
      return existing->second;
    }

    auto bucket = std::make_shared<SourceBucket>();
    source_buckets.emplace(key, bucket);
    MaybeSweepExpiredBucketsLocked();
    AddElapsed(&tracker->lock_hold_nanoseconds, lock_acquired);
    return bucket;
  }

  std::shared_ptr<RetargetCacheBucket> GetOrCreateRetargetBucket(
      const RetargetCacheBucketKey& key) {
    std::lock_guard<std::mutex> lock(table_mutex);
    const auto lock_acquired = MetricsClock::now();
    const auto existing = retarget_buckets.find(key);
    if (existing != retarget_buckets.end()) {
      AddElapsed(&tracker->lock_hold_nanoseconds, lock_acquired);
      return existing->second;
    }

    auto bucket = std::make_shared<RetargetCacheBucket>();
    retarget_buckets.emplace(key, bucket);
    MaybeSweepExpiredBucketsLocked();
    AddElapsed(&tracker->lock_hold_nanoseconds, lock_acquired);
    return bucket;
  }

  void MaybeSweepExpiredBucketsLocked() {
    if ((++insertions_since_sweep & 63U) != 0) return;

    for (auto entry = source_buckets.begin(); entry != source_buckets.end();) {
      if (entry->second.use_count() != 1) {
        ++entry;
        continue;
      }
      std::unique_lock<std::mutex> bucket_lock(entry->second->mutex, std::try_to_lock);
      if (!bucket_lock.owns_lock()) {
        ++entry;
        continue;
      }
      auto& snapshots = entry->second->snapshots;
      snapshots.erase(std::remove_if(snapshots.begin(), snapshots.end(),
                                     [](const auto& value) { return value.expired(); }),
                      snapshots.end());
      if (snapshots.empty()) {
        entry = source_buckets.erase(entry);
      } else {
        ++entry;
      }
    }

    for (auto entry = retarget_buckets.begin(); entry != retarget_buckets.end();) {
      if (entry->second.use_count() != 1) {
        ++entry;
        continue;
      }

      std::unique_lock<std::mutex> bucket_lock(entry->second->mutex, std::try_to_lock);
      if (!bucket_lock.owns_lock() || !entry->second->in_flight.empty()) {
        ++entry;
        continue;
      }
      auto& ready = entry->second->ready;
      ready.erase(std::remove_if(ready.begin(), ready.end(),
                                 [](const auto& value) { return value.elf.expired(); }),
                  ready.end());
      if (ready.empty()) {
        entry = retarget_buckets.erase(entry);
      } else {
        ++entry;
      }
    }
  }

  uint64_t ComputeHash(const void* data, size_t size, bool account) const {
    const auto started = MetricsClock::now();
    const uint64_t hash = hash_function(data, size);
    if (account) {
      tracker->hash_bytes.fetch_add(size, std::memory_order_relaxed);
      AddElapsed(&tracker->hash_nanoseconds, started);
    }
    return hash;
  }

  bool ExactMatch(const SourceSnapshot& snapshot, const void* source_data, size_t source_size,
                  bool account) const {
    const auto started = MetricsClock::now();
    const bool matches = snapshot.Equals(source_data, source_size);
    if (account && snapshot.size() == source_size) {
      tracker->exact_compare_bytes.fetch_add(source_size, std::memory_order_relaxed);
      AddElapsed(&tracker->exact_compare_nanoseconds, started);
    }
    return matches;
  }

  mutable std::mutex table_mutex;
  std::unordered_map<SourceBucketKey, std::shared_ptr<SourceBucket>, SourceBucketKeyHash>
      source_buckets;
  std::unordered_map<RetargetCacheBucketKey, std::shared_ptr<RetargetCacheBucket>,
                     RetargetCacheBucketKeyHash>
      retarget_buckets;
  HashFunction hash_function;
  std::shared_ptr<RetargetCacheMemoryTracker> tracker;
  size_t insertions_since_sweep = 0;
};

ContentRetargetCache::ContentRetargetCache() : impl_(new Impl({})) {}
ContentRetargetCache::~ContentRetargetCache() = default;

#ifdef ROCR_HOTSWAP_TESTING
ContentRetargetCache::ContentRetargetCache(ContentHashFunction hash_function)
    : impl_(new Impl(std::move(hash_function))) {}
#endif

RetargetOperationResult ContentRetargetCache::GetOrCompute(const void* source_data,
                                                           size_t source_size,
                                                           uint64_t reader_id,
                                                           const RetargetCacheKey& key,
                                                           const Producer& producer) {
  const auto run_producer = [&](const SourceSnapshotRef& source) {
    impl_->tracker->producer_calls.fetch_add(1, std::memory_order_relaxed);
    RetargetOperationResult result = RunRetargetProducer(producer, source);
    if (!result.succeeded()) {
      impl_->tracker->producer_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
  };

  if (!source_data || source_size == 0) {
    return {{}, RetargetError::kInvalidArgument};
  }

  uint64_t content_hash = 0;
  try {
    content_hash = impl_->ComputeHash(source_data, source_size, true);
  } catch (...) {
    return run_producer({});
  }
  const SourceBucketKey source_bucket_key{content_hash, source_size};
  SourceSnapshotRef source;
  try {
    std::shared_ptr<SourceBucket> source_bucket = impl_->GetOrCreateSourceBucket(source_bucket_key);

    while (!source) {
      SourceSnapshotRef first_candidate;
      std::vector<SourceSnapshotRef> colliding_candidates;
      uint64_t observed_generation = 0;
      {
        std::lock_guard<std::mutex> lock(source_bucket->mutex);
        const auto lock_acquired = MetricsClock::now();
        observed_generation = source_bucket->generation;
        for (auto snapshot = source_bucket->snapshots.begin();
             snapshot != source_bucket->snapshots.end();) {
          SourceSnapshotRef candidate = snapshot->lock();
          if (!candidate) {
            snapshot = source_bucket->snapshots.erase(snapshot);
            continue;
          }
          if (!first_candidate) {
            first_candidate = std::move(candidate);
          } else {
            colliding_candidates.push_back(std::move(candidate));
          }
          ++snapshot;
        }
        AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
      }

      if (first_candidate && impl_->ExactMatch(*first_candidate, source_data, source_size, true)) {
        source = std::move(first_candidate);
      } else {
        for (auto& candidate : colliding_candidates) {
          if (impl_->ExactMatch(*candidate, source_data, source_size, true)) {
            source = std::move(candidate);
            break;
          }
        }
      }
      if (source) break;

      std::lock_guard<std::mutex> lock(source_bucket->mutex);
      const auto lock_acquired = MetricsClock::now();
      if (source_bucket->generation == observed_generation) {
        std::unique_ptr<unsigned char[]> snapshot_bytes(new unsigned char[source_size]);
        std::memcpy(snapshot_bytes.get(), source_data, source_size);
        source.reset(new SourceSnapshot(std::move(snapshot_bytes), source_size, impl_->tracker));
        source_bucket->snapshots.push_back(source);
        ++source_bucket->generation;
      }
      AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
    }
  } catch (const std::bad_alloc&) {
    return run_producer({});
  }

  const RetargetCacheBucketKey bucket_key{key, content_hash, source_size};
  std::shared_ptr<RetargetCacheBucket> bucket;
  try {
    bucket = impl_->GetOrCreateRetargetBucket(bucket_key);
  } catch (const std::bad_alloc&) {
    return run_producer(source);
  }

  std::shared_ptr<RetargetFlight> flight;
  try {
    std::unique_lock<std::mutex> lock(bucket->mutex);
    auto lock_acquired = MetricsClock::now();

    for (auto ready = bucket->ready.begin(); ready != bucket->ready.end();) {
      RetargetedElfRef elf = ready->elf.lock();
      if (!elf) {
        ready = bucket->ready.erase(ready);
        continue;
      }
      if (elf->source().get() == source.get()) {
        impl_->tracker->ready_hits.fetch_add(1, std::memory_order_relaxed);
        if (IsCrossReader(ready->producer_reader_id, reader_id)) {
          impl_->tracker->cross_reader_results.fetch_add(1, std::memory_order_relaxed);
        }
        AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
        return {std::move(elf), RetargetError::kNone, RetargetResultSource::kReadyCache};
      }
      ++ready;
    }

    for (const auto& candidate : bucket->in_flight) {
      if (candidate->source.get() != source.get()) continue;
      // A producer cannot wait for the flight that only it can complete.
      if (candidate->producer_thread_id == std::this_thread::get_id()) {
        impl_->tracker->reentrant_rejections.fetch_add(1, std::memory_order_relaxed);
        AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
        return {{}, RetargetError::kReentrantRequest};
      }

      flight = candidate;
      ++flight->waiter_count;
      AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
      const auto wait_started = MetricsClock::now();
      flight->completed_cv.wait(lock, [&] { return flight->completed; });
      AddElapsed(&impl_->tracker->wait_nanoseconds, wait_started);
      lock_acquired = MetricsClock::now();
      --flight->waiter_count;
      RetargetOperationResult result = flight->result;
      result.source = RetargetResultSource::kCoalesced;
      impl_->tracker->coalesced_results.fetch_add(1, std::memory_order_relaxed);
      if (IsCrossReader(flight->producer_reader_id, reader_id)) {
        impl_->tracker->cross_reader_results.fetch_add(1, std::memory_order_relaxed);
      }
      AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
      return result;
    }

    flight = std::make_shared<RetargetFlight>(source, reader_id);
    bucket->in_flight.push_back(flight);
    AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
  } catch (const std::bad_alloc&) {
    return run_producer(source);
  }

  RetargetOperationResult result = run_producer(flight->source);
  if (result.succeeded() && result.elf->source().get() != flight->source.get()) {
    result = {{}, RetargetError::kComgrFailure};
    impl_->tracker->producer_failures.fetch_add(1, std::memory_order_relaxed);
  }

  {
    std::lock_guard<std::mutex> lock(bucket->mutex);
    const auto lock_acquired = MetricsClock::now();
    flight->result = result;
    flight->completed = true;
    bucket->in_flight.erase(std::remove(bucket->in_flight.begin(), bucket->in_flight.end(), flight),
                            bucket->in_flight.end());
    if (result.succeeded()) {
      try {
        bucket->ready.push_back({result.elf, flight->producer_reader_id});
      } catch (const std::bad_alloc&) {
        // Active callers and loaded executables still own the result.
      }
    }
    AddElapsed(&impl_->tracker->lock_hold_nanoseconds, lock_acquired);
  }
  flight->completed_cv.notify_all();
  return result;
}

RetargetCacheMetrics ContentRetargetCache::SnapshotMetrics() const {
  RetargetCacheMetrics metrics;
  const auto& tracker = *impl_->tracker;
  metrics.producer_calls = tracker.producer_calls.load(std::memory_order_relaxed);
  metrics.producer_failures = tracker.producer_failures.load(std::memory_order_relaxed);
  metrics.ready_hits = tracker.ready_hits.load(std::memory_order_relaxed);
  metrics.cross_reader_results = tracker.cross_reader_results.load(std::memory_order_relaxed);
  metrics.coalesced_results = tracker.coalesced_results.load(std::memory_order_relaxed);
  metrics.reentrant_rejections = tracker.reentrant_rejections.load(std::memory_order_relaxed);
  metrics.hash_bytes = tracker.hash_bytes.load(std::memory_order_relaxed);
  metrics.hash_nanoseconds = tracker.hash_nanoseconds.load(std::memory_order_relaxed);
  metrics.exact_compare_bytes = tracker.exact_compare_bytes.load(std::memory_order_relaxed);
  metrics.exact_compare_nanoseconds =
      tracker.exact_compare_nanoseconds.load(std::memory_order_relaxed);
  metrics.wait_nanoseconds = tracker.wait_nanoseconds.load(std::memory_order_relaxed);
  metrics.lock_hold_nanoseconds = tracker.lock_hold_nanoseconds.load(std::memory_order_relaxed);
  metrics.source_snapshot_allocations =
      tracker.source_snapshot_allocations.load(std::memory_order_relaxed);
  metrics.source_snapshot_bytes = tracker.source_snapshot_bytes.load(std::memory_order_relaxed);
  metrics.live_source_snapshot_bytes =
      tracker.live_source_snapshot_bytes.load(std::memory_order_relaxed);
  metrics.peak_live_source_snapshot_bytes =
      tracker.peak_live_source_snapshot_bytes.load(std::memory_order_relaxed);
  metrics.produced_output_bytes = tracker.produced_output_bytes.load(std::memory_order_relaxed);
  metrics.live_output_bytes = tracker.live_output_bytes.load(std::memory_order_relaxed);
  metrics.peak_live_output_bytes = tracker.peak_live_output_bytes.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> table_lock(impl_->table_mutex);
  metrics.content_bucket_entries = impl_->source_buckets.size();
  metrics.transform_bucket_entries = impl_->retarget_buckets.size();
  metrics.bucket_entries = metrics.content_bucket_entries + metrics.transform_bucket_entries;
  for (const auto& entry : impl_->retarget_buckets) {
    std::lock_guard<std::mutex> bucket_lock(entry.second->mutex);
    metrics.in_flight_entries += entry.second->in_flight.size();
    for (const auto& ready : entry.second->ready) {
      if (!ready.elf.expired()) ++metrics.ready_entries;
    }
  }
  return metrics;
}

#ifdef ROCR_HOTSWAP_TESTING
size_t ContentRetargetCache::ReadyEntryCountForTesting() const {
  return SnapshotMetrics().ready_entries;
}

size_t ContentRetargetCache::WaiterCountForTesting(const void* source_data, size_t source_size,
                                                   const RetargetCacheKey& key) const {
  if (!source_data || source_size == 0) return 0;
  const uint64_t content_hash = impl_->ComputeHash(source_data, source_size, false);
  SourceSnapshotRef source;
  std::shared_ptr<SourceBucket> source_bucket;
  {
    std::lock_guard<std::mutex> table_lock(impl_->table_mutex);
    const auto entry = impl_->source_buckets.find({content_hash, source_size});
    if (entry == impl_->source_buckets.end()) return 0;
    source_bucket = entry->second;
  }
  {
    std::lock_guard<std::mutex> source_lock(source_bucket->mutex);
    for (const auto& candidate : source_bucket->snapshots) {
      SourceSnapshotRef snapshot = candidate.lock();
      if (snapshot && impl_->ExactMatch(*snapshot, source_data, source_size, false)) {
        source = std::move(snapshot);
        break;
      }
    }
  }
  if (!source) return 0;

  const RetargetCacheBucketKey bucket_key{key, content_hash, source_size};
  std::shared_ptr<RetargetCacheBucket> bucket;
  {
    std::lock_guard<std::mutex> table_lock(impl_->table_mutex);
    const auto entry = impl_->retarget_buckets.find(bucket_key);
    if (entry == impl_->retarget_buckets.end()) return 0;
    bucket = entry->second;
  }

  std::lock_guard<std::mutex> bucket_lock(bucket->mutex);
  for (const auto& flight : bucket->in_flight) {
    if (flight->source.get() == source.get()) {
      return flight->waiter_count;
    }
  }
  return 0;
}

void ContentRetargetCache::ResetMetricsForTesting() {
  auto& tracker = *impl_->tracker;
  tracker.producer_calls.store(0, std::memory_order_relaxed);
  tracker.producer_failures.store(0, std::memory_order_relaxed);
  tracker.ready_hits.store(0, std::memory_order_relaxed);
  tracker.cross_reader_results.store(0, std::memory_order_relaxed);
  tracker.coalesced_results.store(0, std::memory_order_relaxed);
  tracker.reentrant_rejections.store(0, std::memory_order_relaxed);
  tracker.hash_bytes.store(0, std::memory_order_relaxed);
  tracker.hash_nanoseconds.store(0, std::memory_order_relaxed);
  tracker.exact_compare_bytes.store(0, std::memory_order_relaxed);
  tracker.exact_compare_nanoseconds.store(0, std::memory_order_relaxed);
  tracker.wait_nanoseconds.store(0, std::memory_order_relaxed);
  tracker.lock_hold_nanoseconds.store(0, std::memory_order_relaxed);
  tracker.source_snapshot_allocations.store(0, std::memory_order_relaxed);
  tracker.source_snapshot_bytes.store(0, std::memory_order_relaxed);
  tracker.produced_output_bytes.store(0, std::memory_order_relaxed);
  tracker.peak_live_source_snapshot_bytes.store(
      tracker.live_source_snapshot_bytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  tracker.peak_live_output_bytes.store(tracker.live_output_bytes.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
}
#endif

namespace {
std::once_flag g_process_retarget_cache_once;
std::unique_ptr<ContentRetargetCache> g_process_retarget_cache;
}  // namespace

ContentRetargetCache& GetProcessRetargetCache() {
  std::call_once(g_process_retarget_cache_once,
                 [] { g_process_retarget_cache.reset(new ContentRetargetCache()); });
  return *g_process_retarget_cache;
}

}  // namespace hotswap
}  // namespace rocr
