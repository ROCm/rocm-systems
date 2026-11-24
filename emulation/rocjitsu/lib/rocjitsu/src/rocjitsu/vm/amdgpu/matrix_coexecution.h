// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file matrix_coexecution.h
/// @brief Helper publication, completion, and VM-owned execution resources.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/arch/amdgpu/shared/matrix_input_snapshot.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/helper_error_sink.h"
#include "util/host_target.h"
#include "util/result.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace rocjitsu::amdgpu::matrix_coexecution {

// Bound the warm handoff period before falling back to a blocking wait. An
// idle helper must stay awake across the gap between one wave's MMA blocks
// (memory steps, other waves on the CU); otherwise every handoff pays a futex
// wakeup. The issuer's completion wait only has to cover one chain.
// RJ_MMA_SPIN=<pauses> overrides both, e.g. for ceiling measurements.
inline unsigned spin_override() {
  static const unsigned pauses = [] {
    const char *value = std::getenv("RJ_MMA_SPIN");
    return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 0)) : 0u;
  }();
  return pauses;
}
inline unsigned idle_spin_count() {
  const unsigned pauses = spin_override();
  return pauses ? pauses : 1u << 13;
}
inline unsigned completion_spin_count() {
  const unsigned pauses = spin_override();
  return pauses ? pauses : 1u << 12;
}
inline void relax_cpu() {
#if RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN
  __builtin_ia32_pause();
#elif RJ_HOST_ARCH_AARCH64
  asm volatile("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline bool async_candidate(const Instruction &inst) { return async_mma_policy::candidate(inst); }

// A fork child has none of the parent's helper threads. Disable offload there
// with a load, avoiding a getpid syscall on every issue. Register before the
// first helper starts; failure to register also leaves execution synchronous.
inline constinit std::atomic<bool> helpers_disabled{false};
static_assert(std::atomic<bool>::is_always_lock_free);
inline bool helpers_enabled() { return !helpers_disabled.load(std::memory_order_relaxed); }
inline void register_helper_fork_guard() {
  if (!helpers_enabled())
    return;
#if defined(__linux__)
  static const int status = pthread_atfork(
      nullptr, nullptr, [] { helpers_disabled.store(true, std::memory_order_relaxed); });
  if (status != 0)
    helpers_disabled.store(true, std::memory_order_relaxed);
#endif
}

// Borrowed helper capacity. With a budget (RJ_MMA_BORROW=1: the process's
// allowed CPUs; RJ_MMA_BORROW=<n>: n), helpers only use host threads that
// engines and CU dispatch leave idle. Engine threads count as always busy:
// they meet at a barrier every epoch and must get a core the moment it
// completes. One more core stays with the host process (the application
// waits for the dispatch by spinning). Each command processor publishes the
// dispatch workers its current quantum adds to its engine thread in its own
// cache line; a helper claim sums those shards and the claimed helpers.
// Dispatch is never gated. Without a budget, helpers are a separate reservation.
struct alignas(64) OccupancyShard {
  std::atomic<int> threads{0};
};
inline constexpr unsigned kMaxOccupancyShards = 256;
inline constinit std::array<std::atomic<OccupancyShard *>, kMaxOccupancyShards> occupancy_shards{};
inline constinit std::atomic<unsigned> occupancy_shard_end{0};
inline constinit std::atomic<int> reserved_threads{0};
inline int borrow_budget() {
  static const int budget = [] {
    const char *value = std::getenv("RJ_MMA_BORROW");
    if (!value || !*value || *value == '0')
      return 0;
    const long requested = std::strtol(value, nullptr, 0);
    if (requested > 1)
      return int(requested);
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
      return CPU_COUNT(&allowed);
#endif
    return int(std::thread::hardware_concurrency());
  }();
  return budget;
}
/// @brief Registers one command processor's occupancy for its lifetime.
class ScopedOccupancyShard {
public:
  ScopedOccupancyShard() {
    for (unsigned i = 0; i != kMaxOccupancyShards; ++i) {
      OccupancyShard *expected = nullptr;
      if (occupancy_shards[i].compare_exchange_strong(expected, &shard_,
                                                      std::memory_order_acq_rel)) {
        index_ = i;
        unsigned end = occupancy_shard_end.load(std::memory_order_relaxed);
        while (end <= i && !occupancy_shard_end.compare_exchange_weak(end, i + 1,
                                                                      std::memory_order_release)) {
        }
        return;
      }
    }
  }
  ~ScopedOccupancyShard() {
    if (index_ != kMaxOccupancyShards)
      occupancy_shards[index_].store(nullptr, std::memory_order_release);
  }
  ScopedOccupancyShard(const ScopedOccupancyShard &) = delete;
  ScopedOccupancyShard &operator=(const ScopedOccupancyShard &) = delete;
  /// Plain stores, once at the start and end of a quantum; no shared read-modify-write.
  void occupy(int threads) {
    if (index_ != kMaxOccupancyShards)
      shard_.threads.store(threads, std::memory_order_relaxed);
  }

private:
  OccupancyShard shard_;
  unsigned index_ = kMaxOccupancyShards;
};
inline int occupied_threads() {
  int threads = 0;
  const unsigned end = occupancy_shard_end.load(std::memory_order_acquire);
  for (unsigned i = 0; i != end; ++i)
    if (const OccupancyShard *shard = occupancy_shards[i].load(std::memory_order_acquire))
      threads += shard->threads.load(std::memory_order_relaxed);
  return threads;
}

/// @brief Last-level-cache domains of the allowed host CPUs.
/// @details A helper reads and writes the issuer's register tiles; across a
/// CCD boundary each line crosses the fabric. Helpers are pinned to one domain
/// and issuers claim helpers of their own domain.
struct CacheDomains {
  std::vector<std::vector<int>> cpus;  ///< Allowed CPUs per domain.
  std::vector<int8_t> domain_of_cpu;   ///< -1 when not allowed.
  static CacheDomains detect() {
    CacheDomains result;
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
      return result;
    std::vector<int> ids;
    for (int cpu = 0; cpu != CPU_SETSIZE; ++cpu) {
      if (!CPU_ISSET(cpu, &allowed))
        continue;
      // Shared L3 list identifies the domain; fall back to one domain.
      char path[96];
      std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index3/id", cpu);
      int id = 0;
      if (FILE *file = std::fopen(path, "r")) {
        if (std::fscanf(file, "%d", &id) != 1)
          id = 0;
        std::fclose(file);
      }
      auto it = std::find(ids.begin(), ids.end(), id);
      if (it == ids.end()) {
        ids.push_back(id);
        result.cpus.emplace_back();
        it = ids.end() - 1;
      }
      const auto domain = it - ids.begin();
      result.cpus[domain].push_back(cpu);
      if (result.domain_of_cpu.size() <= size_t(cpu))
        result.domain_of_cpu.resize(cpu + 1, -1);
      result.domain_of_cpu[cpu] = int8_t(domain);
    }
#endif
    if (result.cpus.size() > 64)
      result = {};
    return result;
  }
};
/// RJ_MMA_LOCAL_ONLY=1 never lets an issuer use another cache domain's helpers.
inline bool local_helpers_only() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_LOCAL_ONLY");
    return value && *value && *value != '0';
  }();
  return enabled;
}

// Persistent helpers belong to a VM-owned pool, constructed lazily.
// Helpers run in the default host floating-point environment, like the issuer.
// Callbacks must return normally; admitted guest errors are rejected before publication.
// Instructions remain allocated and destroyed by the original decoder thread.
// The caller validates disjoint register accesses and materializes destinations
// before publication. All jobs complete before the CU can be rescheduled.
/// @brief Persistent worker executing one published instruction at a time.
class Helper {
public:
  Helper() : thread_([this] { work(); }) {}
  ~Helper() {
    publish(-1);
    thread_.join();
  }

  static constexpr unsigned kMaxChain = 8;

  void submit(Instruction &instruction, void *wave) {
    Instruction *chain = &instruction;
    submit(&chain, 1, wave);
  }
  /// Execute same-wave instructions back to back in program order with one handoff.
  void submit(Instruction *const *chain, unsigned count, void *wave,
              const MatrixInputSnapshot *const *snapshots = nullptr) {
    assert(count != 0 && count <= kMaxChain);
    std::copy_n(chain, count, chain_.begin());
    if (snapshots)
      std::copy_n(snapshots, count, snapshots_.begin());
    else
      std::fill_n(snapshots_.begin(), count, nullptr);
    count_ = count;
    wave_ = wave;
    publish(1);
  }

  void wait() {
    while (state_.load(std::memory_order_acquire) != 0)
      wait_while(1);
  }

  bool ready() const { return state_.load(std::memory_order_acquire) == 0; }
  void pin(const std::vector<int> &cpus) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus)
      CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
#else
    (void)cpus;
#endif
  }
  /// The first failing chain member and its error; valid after wait().
  unsigned failed_member() const { return failed_; }
  const HelperErrorSink &error() const { return error_; }

private:
  void publish(int value) {
    // A release store followed by notification is sufficient for this protocol
    // under the C++ atomic wait/notify rules. The seq_cst store is a library
    // workaround for libstdc++'s implementation.
    // In libstdc++'s int-sized futex path, notify_one checks the bucket's
    // waiter count before deciding whether to wake. Order publication before
    // that check: on x86 a release store can remain buffered while the check
    // sees no waiter and a newly registered waiter still sees the old state.
    // Apply the ordering to job, completion, and shutdown publication.
    state_.store(value, std::memory_order_seq_cst);
    state_.notify_one();
  }

  void wait_while(int expected) {
    const unsigned spins = expected == 0 ? idle_spin_count() : completion_spin_count();
    for (unsigned i = 0; i != spins; ++i) {
      if (state_.load(std::memory_order_acquire) != expected)
        return;
      relax_cpu();
    }
    // There is one waiter per helper: the issuer during execution, or the
    // helper while idle. Atomic waiting rechecks the value before sleeping.
    state_.wait(expected, std::memory_order_acquire);
  }

  void work() {
    for (;;) {
      int state = state_.load(std::memory_order_acquire);
      if (state == -1)
        return;
      if (state == 0) {
        wait_while(0);
        continue;
      }
      HelperErrorSink sink;
      helper_error_sink = &sink;
      failed_ = count_;
      for (unsigned i = 0; i != count_; ++i) {
        matrix_input_snapshot = snapshots_[i];
        chain_[i]->execute(*chain_[i], wave_);
        if (sink.error) {
          // Later members depend on this result; they do not execute.
          failed_ = i;
          break;
        }
      }
      matrix_input_snapshot = nullptr;
      helper_error_sink = nullptr;
      error_ = sink;
      publish(0);
    }
  }

  alignas(64) std::atomic<int> state_{0};
  alignas(64) std::array<Instruction *, kMaxChain> chain_{};
  std::array<const MatrixInputSnapshot *, kMaxChain> snapshots_{};
  unsigned failed_ = 0;
  HelperErrorSink error_;
  unsigned count_ = 0;
  void *wave_ = nullptr;
  std::thread thread_;
};

// Map the issuer hash into a domain's helpers before selecting a free bit.
constexpr unsigned preferred_helper_start(uint32_t issuer_hash, std::size_t capacity) {
  return static_cast<unsigned>((uint64_t{issuer_hash} * capacity) >> 32) & 63;
}

/// @brief Factor copies for the chain a claimed helper executes.
/// @details Owned by the helper slot, so only chains that really run on a
/// helper copy anything; reused after the slot is joined.
struct SnapshotBuffer {
  std::array<MatrixInputSnapshot, Helper::kMaxChain> entries{};
  std::array<const MatrixInputSnapshot *, Helper::kMaxChain> pointers{};
  std::vector<uint32_t> data;
};

// Each claim covers publication through joining, so another issuer cannot
// reuse a helper's payload while its previous owner still reads completion.
// There is no pending-job queue and no wait for capacity. Two bitmap words
// cover the entire pool; an exhausted pool needs no atomic RMW.
/// @brief Shared CPU budget with bounded nonblocking reservation.
class SharedPool {
  struct Slot {
    Helper helper;
    unsigned index;
    SnapshotBuffer snapshot;
    explicit Slot(unsigned i) : index(i) {}
  };

public:
  struct Ticket {
    Slot *slot = nullptr;
    explicit operator bool() const { return slot != nullptr; }
  };

  /// Claim a helper and publish one instruction, or return an empty ticket.
  Ticket submit(Instruction &instruction, void *context) {
    Instruction *chain = &instruction;
    return submit(&chain, 1, context);
  }
  Ticket submit(Instruction *const *chain, unsigned count, void *context,
                const MatrixInputSnapshot *const *snapshots = nullptr) {
    const Ticket ticket = claim();
    if (ticket)
      start(ticket, chain, count, context, snapshots);
    return ticket;
  }
  /// Reserve a helper without publishing; start() must follow on success.
  Ticket claim() { return {try_claim()}; }
  static SnapshotBuffer &snapshot_buffer(Ticket ticket) { return ticket.slot->snapshot; }
  static void start(Ticket ticket, Instruction *const *chain, unsigned count, void *context,
                    const MatrixInputSnapshot *const *snapshots = nullptr) {
    ticket.slot->helper.submit(chain, count, context, snapshots);
  }
  /// A true result permits finish() without waiting for the helper.
  static bool ready(Ticket ticket) { return !helpers_enabled() || ticket.slot->helper.ready(); }
  /// Join a claimed job and release its slot; consume each ticket exactly once.
  /// Returns failure in a fork child where the inherited helper no longer exists.
  util::Result finish(Ticket ticket) { return finish(ticket, nullptr, nullptr); }
  /// Also reports the first execution error of the job, if any, and its chain member.
  util::Result finish(Ticket ticket, HelperErrorSink *error, unsigned *member) {
    if (!helpers_enabled())
      return util::Result::failure();
    ticket.slot->helper.wait();
    if (error) {
      *error = ticket.slot->helper.error();
      *member = ticket.slot->helper.failed_member();
    }
    release(ticket.slot);
    return util::Result::success();
  }

  explicit SharedPool(unsigned count) : domains_(CacheDomains::detect()) {
    assert(count <= 128);
    register_helper_fork_guard();
    if (!helpers_enabled())
      count = 0;
    slots_.reserve(count);
    for (unsigned i = 0; i != count; ++i)
      slots_.push_back(std::make_unique<Slot>(i));
    for (unsigned word = 0; word != 2; ++word) {
      const unsigned bits = count > word * 64 ? std::min(64u, count - word * 64) : 0;
      free_[word].store(bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1,
                        std::memory_order_relaxed);
    }
    // Split helpers evenly over cache domains, contiguous per domain, and pin them.
    const size_t domains = domains_.cpus.size();
    if (domains > 1 && count) {
      domain_masks_.assign(domains, {});
      unsigned next = 0;
      for (size_t d = 0; d != domains; ++d) {
        const unsigned share = unsigned(count / domains + (d < count % domains));
        for (unsigned k = 0; k != share; ++k, ++next) {
          domain_masks_[d][next / 64] |= uint64_t{1} << (next % 64);
          slots_[next]->helper.pin(domains_.cpus[d]);
        }
      }
    }
  }

  ~SharedPool() {
    // Inherited std::thread objects cannot be joined or destroyed in the child.
    // Leave their storage to process teardown; the parent's ownership is intact.
    if (!helpers_enabled())
      for (auto &slot : slots_)
        (void)slot.release();
  }

  /// Cheap advisory capacity check. A following claim() may still lose a race.
  bool available() const {
    if (!helpers_enabled() || !within_budget())
      return false;
    return free_[0].load(std::memory_order_relaxed) != 0 ||
           (slots_.size() > 64 && free_[1].load(std::memory_order_relaxed) != 0);
  }

private:
  bool within_budget() const {
    const int budget = borrow_budget();
    if (!budget)
      return true;
    int claimed = int(slots_.size());
    for (const auto &word : free_)
      claimed -= std::popcount(word.load(std::memory_order_relaxed));
    return reserved_threads.load(std::memory_order_relaxed) + occupied_threads() + claimed <
           budget;
  }

  /// Helpers of the issuer's cache domain, or none with a single domain.
  std::array<uint64_t, 2> local_mask() const {
#if defined(__linux__)
    if (!domain_masks_.empty()) {
      const int cpu = sched_getcpu();
      if (cpu >= 0 && size_t(cpu) < domains_.domain_of_cpu.size() &&
          domains_.domain_of_cpu[cpu] >= 0)
        return domain_masks_[domains_.domain_of_cpu[cpu]];
    }
#endif
    return {0, 0};
  }

  Slot *claim_from(const std::array<uint64_t, 2> &mask, unsigned start) {
    for (unsigned word = 0; word != (slots_.size() > 64 ? 2u : 1u); ++word) {
      auto free = free_[word].load(std::memory_order_relaxed);
      for (unsigned attempt = 0; (free & mask[word]) && attempt != 2; ++attempt) {
        const uint64_t candidates = free & mask[word];
        const unsigned bit = (std::countr_zero(std::rotr(candidates, int(start))) + start) & 63;
        if (!free_[word].compare_exchange_strong(free, free & ~(uint64_t{1} << bit),
                                                 std::memory_order_acquire,
                                                 std::memory_order_relaxed))
          continue;
        return slots_[word * 64 + bit].get();
      }
    }
    return nullptr;
  }

  Slot *try_claim() {
    if (!helpers_enabled() || !within_budget())
      return nullptr;
    // Bound claims and fall back inline on a miss: one or two loads detect an
    // exhausted pool, and after two failed CAS attempts per word the chain
    // runs on the issuer. A stable per-issuer start keeps reusing an awake
    // helper. Helpers sharing the issuer's last-level cache come first.
    static thread_local const uint32_t issuer_hash =
        static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const unsigned start = preferred_helper_start(issuer_hash, slots_.size());
    const auto local = local_mask();
    if (local[0] | local[1])
      if (Slot *slot = claim_from(local, start))
        return slot;
    if ((local[0] | local[1]) && local_helpers_only())
      return nullptr;
    return claim_from({~uint64_t{0}, ~uint64_t{0}}, start);
  }
  void release(Slot *slot) {
    free_[slot->index / 64].fetch_or(uint64_t{1} << (slot->index % 64), std::memory_order_release);
  }
  alignas(64) std::array<std::atomic<uint64_t>, 2> free_{};

  CacheDomains domains_;
  std::vector<std::array<uint64_t, 2>> domain_masks_;
  std::vector<std::unique_ptr<Slot>> slots_;
};

/// Shared by all CUs and GPUs in one VM. Config parsing does not start helpers;
/// the first eligible instruction creates the pool.
class ExecutionResources {
public:
  explicit ExecutionResources(unsigned helpers, unsigned engines = 1) : helpers_(helpers) {
    reserved_threads.store(int(engines) + 1, std::memory_order_relaxed);
  }
  unsigned helpers() const { return helpers_; }
  SharedPool &pool() {
    if (!helpers_enabled()) {
      // Avoid an inherited once_flag if another thread forked during startup.
      static SharedPool disabled(0);
      return disabled;
    }
    std::call_once(once_, [&] { pool_ = std::make_unique<SharedPool>(helpers_); });
    return *pool_;
  }

private:
  unsigned helpers_;
  std::once_flag once_;
  std::unique_ptr<SharedPool> pool_;
};

} // namespace rocjitsu::amdgpu::matrix_coexecution
