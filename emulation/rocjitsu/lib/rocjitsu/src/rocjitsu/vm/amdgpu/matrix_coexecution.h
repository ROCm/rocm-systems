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
inline int borrow_budget();
inline unsigned idle_spin_count() {
  static const unsigned count = [] {
    if (const char *override_idle = std::getenv("RJ_MMA_IDLE_SPIN");
        override_idle && *override_idle)
      return static_cast<unsigned>(std::strtoul(override_idle, nullptr, 0));
    const unsigned pauses = spin_override();
    if (pauses)
      return pauses;
    // A borrowed core belongs to dispatch the moment the helper is idle.
    // Dedicated helpers stay awake across K-loop gaps to avoid CU-path futex
    // wakeups while still sleeping during long non-MMA phases.
    return borrow_budget() ? 1u << 10 : 1u << 16;
  }();
  return count;
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
// engines and CU dispatch leave idle. Engine threads count as always busy
// even while they sleep at the epoch barrier: every epoch they wake and must
// get a core at once, or the barrier, and with it every engine, waits for the
// scheduler to displace a helper (measured as a 2-3x slowdown when helpers
// borrowed the cores of sleeping engines). One more core stays with the host
// process (the application waits for the dispatch by spinning). Each command
// processor publishes the dispatch workers its current quantum adds to its
// engine thread in its own cache line; a helper claim sums those shards and
// the claimed helpers. Dispatch is never gated. Without a budget, helpers are
// a separate reservation.
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
/// RJ_MMA_LOCAL_ONLY=1 never lets an issuer use another cache domain's helpers;
/// RJ_MMA_LOCAL_ONLY=0 forces cross-domain fallback even for lightweight MMAs.
inline int local_helpers_only_override() {
  static const int mode = [] {
    const char *value = std::getenv("RJ_MMA_LOCAL_ONLY");
    if (!value || !*value)
      return -1;
    return *value != '0' ? 1 : 0;
  }();
  return mode;
}
inline bool local_helpers_only() { return local_helpers_only_override() == 1; }

struct PoolStats {
  alignas(64) std::atomic<uint64_t> claims{0};
  std::atomic<uint64_t> warm_claims{0};
  std::atomic<uint64_t> wake_submits{0};
  std::atomic<uint64_t> wait_ready{0};
  std::atomic<uint64_t> wait_spun{0};
  std::atomic<uint64_t> wait_slept{0};
  alignas(64) std::atomic<uint64_t> helper_sleeps{0};
};

inline bool pool_stats_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_ASYNC_STATS");
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
  static constexpr int kStop = -1;
  static constexpr int kIdle = 0;
  static constexpr int kBusy = 1;
  static constexpr int kIdleSleeping = 2;
  static constexpr int kBusyWaiting = 3;

  /// `awake` counts helpers not blocked in the kernel; `awake_word`/`awake_bit`
  /// publish this helper's bit so borrowers reuse warm spinning helpers before
  /// waking sleeping ones.
  explicit Helper(std::atomic<int> *awake = nullptr,
                  std::atomic<uint64_t> *awake_word = nullptr, uint64_t awake_bit = 0,
                  PoolStats *stats = nullptr)
      : awake_(awake), awake_word_(awake_word), awake_bit_(awake_bit), stats_(stats) {
    mark_awake();
    thread_ = std::thread([this] { work(); });
  }
  ~Helper() {
    if (state_.exchange(kStop, std::memory_order_seq_cst) == kIdleSleeping)
      state_.notify_one();
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
    // Exchange orders publication before any notify_one() bucket check in
    // libstdc++, while skipping notify_one() entirely when the helper is warm.
    if (state_.exchange(kBusy, std::memory_order_seq_cst) == kIdleSleeping) {
      if (stats_) [[unlikely]]
        stats_->wake_submits.fetch_add(1, std::memory_order_relaxed);
      state_.notify_one();
    }
  }

  static constexpr bool is_idle(int state) { return state == kIdle || state == kIdleSleeping; }

  void wait() {
    if (is_idle(state_.load(std::memory_order_acquire))) {
      if (stats_) [[unlikely]]
        stats_->wait_ready.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const unsigned spins = completion_spin_count();
    for (unsigned i = 0; i != spins; ++i) {
      if (is_idle(state_.load(std::memory_order_acquire))) {
        if (stats_) [[unlikely]]
          stats_->wait_spun.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      relax_cpu();
    }
    int expected = kBusy;
    if (state_.compare_exchange_strong(expected, kBusyWaiting, std::memory_order_seq_cst,
                                       std::memory_order_acquire)) {
      if (stats_) [[unlikely]]
        stats_->wait_slept.fetch_add(1, std::memory_order_relaxed);
      while (state_.load(std::memory_order_acquire) == kBusyWaiting)
        state_.wait(kBusyWaiting, std::memory_order_acquire);
    } else if (stats_) [[unlikely]] {
      stats_->wait_spun.fetch_add(1, std::memory_order_relaxed);
    }
  }

  bool ready() const { return is_idle(state_.load(std::memory_order_acquire)); }
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
  void mark_awake() {
    if (awake_) {
      awake_->fetch_add(1, std::memory_order_relaxed);
      if (awake_word_)
        awake_word_->fetch_or(awake_bit_, std::memory_order_relaxed);
    }
  }
  void mark_asleep() {
    if (awake_) {
      if (awake_word_)
        awake_word_->fetch_and(~awake_bit_, std::memory_order_relaxed);
      awake_->fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void idle_wait() {
    const unsigned spins = idle_spin_count();
    for (unsigned i = 0; i != spins; ++i) {
      if (state_.load(std::memory_order_acquire) != kIdle)
        return;
      relax_cpu();
    }
    mark_asleep();
    int expected = kIdle;
    if (state_.compare_exchange_strong(expected, kIdleSleeping, std::memory_order_seq_cst,
                                       std::memory_order_acquire)) {
      if (stats_) [[unlikely]]
        stats_->helper_sleeps.fetch_add(1, std::memory_order_relaxed);
      while (state_.load(std::memory_order_acquire) == kIdleSleeping)
        state_.wait(kIdleSleeping, std::memory_order_acquire);
    }
    mark_awake();
  }

  void work() {
    for (;;) {
      int state = state_.load(std::memory_order_acquire);
      if (state == kStop) {
        mark_asleep();
        return;
      }
      if (state == kIdle) {
        idle_wait();
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
      if (state_.exchange(kIdle, std::memory_order_seq_cst) == kBusyWaiting)
        state_.notify_one();
    }
  }

  std::atomic<int> *awake_ = nullptr;
  std::atomic<uint64_t> *awake_word_ = nullptr;
  uint64_t awake_bit_ = 0;
  PoolStats *stats_ = nullptr;
  alignas(64) std::atomic<int> state_{kIdle};
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
// There is no pending-job queue and no wait for capacity. Each LLC domain owns
// its own cache-line-isolated bitmap shard so local claims and releases never
// bounce cache lines across CCD boundaries.
/// @brief Shared CPU budget with bounded nonblocking reservation.
class SharedPool {
  struct alignas(64) Shard {
    std::atomic<uint64_t> free{0};
    std::atomic<uint64_t> awake_mask{0};
    std::atomic<int> awake{0};
    unsigned base = 0;
    unsigned count = 0;
    unsigned active_rank = 0;
  };
  struct Slot {
    Helper helper;
    unsigned index;
    unsigned shard;
    unsigned bit;
    SnapshotBuffer snapshot;
    Slot(unsigned i, unsigned s, unsigned b, Shard &owner, PoolStats *stats)
        : helper(&owner.awake, &owner.awake_mask, uint64_t{1} << b, stats), index(i), shard(s),
          bit(b) {}
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
  Ticket claim() { return {try_claim(within_budget() ? kUnlimited : 0, local_helpers_only())}; }
  /// Reserve a helper while fewer than `cap` are claimed pool-wide. The cap is
  /// the caller's share of the borrowing budget, computed once per quantum by
  /// borrow_cap(): summing every occupancy shard on each claim bounced those
  /// lines between cores on every block.
  Ticket claim(unsigned cap, bool local_only = false) {
    const int override_mode = local_helpers_only_override();
    const bool restrict_local = override_mode >= 0 ? (override_mode != 0) : local_only;
    return {try_claim(cap, restrict_local)};
  }
  static constexpr unsigned kUnlimited = ~0u;
  /// Helpers this issuer may hold right now: unlimited without a budget,
  /// otherwise the budget minus reserved and occupied threads (zero when the
  /// host is saturated, so the caller issues synchronously). Warm spinning
  /// helpers are reused via each shard's awake_mask without shrinking the cap.
  unsigned borrow_cap() const { return borrow_cap_for(borrow_budget()); }
  unsigned borrow_cap_for(int budget) const {
    if (!budget)
      return kUnlimited;
    const int reserved = reserved_threads.load(std::memory_order_relaxed);
    const int occupied = occupied_threads();
    const int room = budget - reserved - occupied;
    if (room <= 0)
      return 0u;
    // When dispatch workers are already scaling across CUs (occupied > 0),
    // borrowing a tiny helper pool (< 0.5 helpers per active CU) thrashes the
    // shared bitmap and cores without improving the critical path.
    const int active_cus = std::max(1, reserved - 1) + occupied;
    if (occupied > 0 && room * 2 < active_cus)
      return 0u;
    return unsigned(room);
  }
  unsigned awake() const {
    int total = 0;
    for (size_t s = 0; s != num_shards_; ++s)
      total += shards_[s].awake.load(std::memory_order_relaxed);
    return total > 0 ? unsigned(total) : 0u;
  }
  unsigned claimed() const {
    unsigned free_count = 0;
    for (size_t s = 0; s != num_shards_; ++s)
      free_count += unsigned(std::popcount(shards_[s].free.load(std::memory_order_relaxed)));
    return unsigned(slots_.size()) - free_count;
  }
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

  explicit SharedPool(unsigned count)
      : domains_(CacheDomains::detect()), stats_(pool_stats_enabled() ? &stats_storage_ : nullptr) {
    assert(count <= 128);
    register_helper_fork_guard();
    if (!helpers_enabled())
      count = 0;
    const size_t domains = domains_.cpus.size();
    multi_domain_ = domains > 1 && count > 0;
    if (multi_domain_) {
      // One cache-line-isolated shard per LLC domain (up to 64 helpers/domain).
      // Split helpers evenly starting from the constructing issuer's domain so
      // small pools (< domains) always place helpers locally.
      size_t start_domain = 0;
#if defined(__linux__)
      const int cpu = sched_getcpu();
      if (cpu >= 0 && size_t(cpu) < domains_.domain_of_cpu.size() &&
          domains_.domain_of_cpu[cpu] >= 0)
        start_domain = size_t(domains_.domain_of_cpu[cpu]);
#endif
      shards_ = std::make_unique<Shard[]>(domains);
      num_shards_ = domains;
      std::vector<unsigned> shares(domains, 0);
      for (size_t i = 0; i != domains; ++i)
        shares[(start_domain + i) % domains] = unsigned(count / domains + (i < count % domains));
      slots_.reserve(count);
      unsigned next = 0;
      for (size_t d = 0; d != domains; ++d) {
        const unsigned share = shares[d];
        assert(share <= 64);
        shards_[d].base = next;
        shards_[d].count = share;
        if (share)
          shards_[d].active_rank = active_shards_++;
        shards_[d].free.store(share == 64 ? ~uint64_t{0} : (uint64_t{1} << share) - 1,
                              std::memory_order_relaxed);
        for (unsigned b = 0; b != share; ++b, ++next) {
          slots_.push_back(std::make_unique<Slot>(next, unsigned(d), b, shards_[d], stats_));
          slots_.back()->helper.pin(domains_.cpus[d]);
        }
      }
    } else if (count) {
      const unsigned num_shards = (count + 63u) / 64u;
      shards_ = std::make_unique<Shard[]>(num_shards);
      num_shards_ = num_shards;
      slots_.reserve(count);
      unsigned next = 0;
      for (unsigned s = 0; s != num_shards; ++s) {
        const unsigned share = std::min(64u, count - next);
        shards_[s].base = next;
        shards_[s].count = share;
        if (share)
          shards_[s].active_rank = active_shards_++;
        shards_[s].free.store(share == 64 ? ~uint64_t{0} : (uint64_t{1} << share) - 1,
                              std::memory_order_relaxed);
        for (unsigned b = 0; b != share; ++b, ++next)
          slots_.push_back(std::make_unique<Slot>(next, s, b, shards_[s], stats_));
      }
    }
  }

  void flush_stats() {
    if (!stats_)
      return;
    const uint64_t claims = stats_->claims.exchange(0, std::memory_order_relaxed);
    if (!claims)
      return;
    const uint64_t warm = stats_->warm_claims.exchange(0, std::memory_order_relaxed);
    const uint64_t wakes = stats_->wake_submits.exchange(0, std::memory_order_relaxed);
    const uint64_t ready = stats_->wait_ready.exchange(0, std::memory_order_relaxed);
    const uint64_t spun = stats_->wait_spun.exchange(0, std::memory_order_relaxed);
    const uint64_t slept = stats_->wait_slept.exchange(0, std::memory_order_relaxed);
    const uint64_t hsleeps = stats_->helper_sleeps.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "RJ_POOL helpers=%zu domains=%zu claims=%llu warm_claims=%llu "
                 "wake_submits=%llu wait(ready=%llu spun=%llu slept=%llu) helper_sleeps=%llu\n",
                 slots_.size(), domains_.cpus.size(),
                 static_cast<unsigned long long>(claims),
                 static_cast<unsigned long long>(warm),
                 static_cast<unsigned long long>(wakes),
                 static_cast<unsigned long long>(ready),
                 static_cast<unsigned long long>(spun),
                 static_cast<unsigned long long>(slept),
                 static_cast<unsigned long long>(hsleeps));
  }

  ~SharedPool() {
    flush_stats();
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
    for (size_t s = 0; s != num_shards_; ++s)
      if (shards_[s].free.load(std::memory_order_relaxed) != 0)
        return true;
    return false;
  }

private:
  bool within_budget() const { return claimed() < borrow_cap(); }

  /// Current issuer's LLC domain index, or -1 on a single-domain host.
  int local_domain() const {
#if defined(__linux__)
    if (multi_domain_) {
      const int cpu = sched_getcpu();
      if (cpu >= 0 && size_t(cpu) < domains_.domain_of_cpu.size())
        return domains_.domain_of_cpu[cpu];
    }
#endif
    return -1;
  }

  Slot *claim_shard(unsigned shard_idx, uint64_t mask, unsigned start) {
    Shard &shard = shards_[shard_idx];
    if (!shard.count || !mask)
      return nullptr;
    const unsigned local_start = start % shard.count;
    uint64_t free = shard.free.load(std::memory_order_relaxed);
    for (unsigned attempt = 0; (free & mask) && attempt != 2; ++attempt) {
      const uint64_t candidates = free & mask;
      const unsigned bit =
          (std::countr_zero(std::rotr(candidates, int(local_start))) + local_start) & 63;
      if (!shard.free.compare_exchange_strong(free, free & ~(uint64_t{1} << bit),
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed))
        continue;
      return slots_[shard.base + bit].get();
    }
    return nullptr;
  }

  Slot *try_claim(unsigned cap, bool local_only) {
    if (!helpers_enabled() || num_shards_ == 0 || cap == 0)
      return nullptr;
    // Bound claims and fall back inline on a miss: one load detects an
    // exhausted local shard, and after two failed CAS attempts the chain runs
    // on the issuer. Dedicated and shard-budgeted local claims touch only the
    // issuer's own LLC shard — zero cross-CCD cache-line traffic.
    const bool unlimited = cap >= slots_.size();
    static thread_local const uint32_t issuer_hash =
        static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const unsigned start = preferred_helper_start(issuer_hash, slots_.size());
    const int local_d = local_domain();
    const bool has_local = local_d >= 0 && shards_[size_t(local_d)].count != 0;
    const bool forbid_remote = multi_domain_ && local_only;
    const auto record = [&](Slot *slot, bool warm) -> Slot * {
      if (stats_) [[unlikely]] {
        stats_->claims.fetch_add(1, std::memory_order_relaxed);
        if (warm)
          stats_->warm_claims.fetch_add(1, std::memory_order_relaxed);
      }
      return slot;
    };
    const auto claim_warm_in = [&](unsigned s) -> Slot * {
      const uint64_t w = shards_[s].awake_mask.load(std::memory_order_relaxed);
      return w ? claim_shard(s, w, start) : nullptr;
    };
    // Finite-budget local-only claims enforce a per-shard cap whose sum across
    // active shards equals `cap`, avoiding cross-CCD shard scans on every claim.
    if (forbid_remote && !unlimited && has_local && active_shards_ > 0 && cap >= active_shards_) {
      const Shard &ls = shards_[size_t(local_d)];
      const unsigned local_cap = std::min(
          ls.count, cap / active_shards_ + (ls.active_rank < cap % active_shards_ ? 1u : 0u));
      const unsigned local_claimed =
          ls.count - unsigned(std::popcount(ls.free.load(std::memory_order_relaxed)));
      if (local_claimed >= local_cap)
        return nullptr;
      if (Slot *slot = claim_warm_in(unsigned(local_d)))
        return record(slot, true);
      if (ls.awake.load(std::memory_order_relaxed) < int(local_cap))
        if (Slot *slot = claim_shard(unsigned(local_d), ~uint64_t{0}, start))
          return record(slot, false);
      return nullptr;
    }
    if (!unlimited && claimed() >= cap)
      return nullptr;
    if (has_local) {
      if (Slot *slot = claim_warm_in(unsigned(local_d)))
        return record(slot, true);
      if (unlimited || awake() < cap)
        if (Slot *slot = claim_shard(unsigned(local_d), ~uint64_t{0}, start))
          return record(slot, false);
    }
    if (forbid_remote)
      return nullptr;
    // Single-domain or remote fallback: prefer warm helpers first.
    for (size_t s = 0; s != num_shards_; ++s) {
      if (int(s) == local_d)
        continue;
      if (Slot *slot = claim_warm_in(unsigned(s)))
        return record(slot, true);
    }
    if (!unlimited && awake() >= cap)
      return nullptr;
    for (size_t s = 0; s != num_shards_; ++s) {
      if (int(s) == local_d)
        continue;
      if (Slot *slot = claim_shard(unsigned(s), ~uint64_t{0}, start))
        return record(slot, false);
    }
    return nullptr;
  }
  void release(Slot *slot) {
    shards_[slot->shard].free.fetch_or(uint64_t{1} << slot->bit, std::memory_order_release);
  }
  CacheDomains domains_;
  bool multi_domain_ = false;
  size_t num_shards_ = 0;
  unsigned active_shards_ = 0;
  std::unique_ptr<Shard[]> shards_;
  std::vector<std::unique_ptr<Slot>> slots_;
  PoolStats stats_storage_{};
  PoolStats *stats_ = nullptr;
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
