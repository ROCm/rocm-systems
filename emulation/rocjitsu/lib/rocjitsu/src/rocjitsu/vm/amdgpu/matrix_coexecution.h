// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file matrix_coexecution.h
/// @brief Helper publication, completion, and VM-owned execution resources.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cfenv>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rocjitsu::amdgpu::matrix_coexecution {

// Bound the warm handoff period before falling back to a blocking wait.
inline constexpr unsigned spin_count() { return 512; }
inline constexpr unsigned idle_spin_count() { return spin_count(); }
inline constexpr unsigned completion_spin_count() { return spin_count(); }
inline void relax_cpu() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline bool async_candidate(std::string_view name) { return async_mma_policy::candidate(name); }

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

// Persistent helpers belong to a VM-owned pool, constructed lazily.
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

  void submit(Instruction &instruction, void *wave) {
    instruction_ = &instruction;
    wave_ = wave;
    error_ = nullptr;
    std::fegetenv(&environment_);
    publish(1);
  }

  std::exception_ptr wait() {
    while (state_.load(std::memory_order_acquire) != 0)
      wait_while(1);
    return error_;
  }

  bool ready() const { return state_.load(std::memory_order_acquire) == 0; }

private:
  void publish(int value) {
    state_.store(value, std::memory_order_release);
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
      std::fesetenv(&environment_);
      try {
        instruction_->execute(*instruction_, wave_);
      } catch (...) {
        error_ = std::current_exception();
      }
      publish(0);
    }
  }

  alignas(64) std::atomic<int> state_{0};
  alignas(64) Instruction *instruction_ = nullptr;
  void *wave_ = nullptr;
  std::fenv_t environment_{};
  std::exception_ptr error_;
  std::thread thread_;
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
    explicit Slot(unsigned i) : index(i) {}
  };

public:
  struct Ticket {
    Slot *slot = nullptr;
    explicit operator bool() const { return slot != nullptr; }
  };

  /// Claim a helper without waiting, or return an empty ticket for inline execution.
  Ticket submit(Instruction &instruction, void *context) {
    return {try_submit(instruction, context)};
  }
  /// A true result permits finish() without waiting for the helper.
  static bool ready(Ticket ticket) { return !helpers_enabled() || ticket.slot->helper.ready(); }
  /// Join a claimed job and release its slot; consume each ticket exactly once.
  std::exception_ptr finish(Ticket ticket) {
    if (!helpers_enabled())
      return std::make_exception_ptr(std::runtime_error("MMA helper unavailable after fork"));
    auto error = ticket.slot->helper.wait();
    release(ticket.slot);
    return error;
  }

  explicit SharedPool(unsigned count) {
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
  }

  ~SharedPool() {
    // Inherited std::thread objects cannot be joined or destroyed in the child.
    // Leave their storage to process teardown; the parent's ownership is intact.
    if (!helpers_enabled())
      for (auto &slot : slots_)
        (void)slot.release();
  }

  /// Cheap advisory capacity check. A following submit() may still lose a race.
  bool available() const {
    if (!helpers_enabled())
      return false;
    return free_[0].load(std::memory_order_relaxed) != 0 ||
           (slots_.size() > 64 && free_[1].load(std::memory_order_relaxed) != 0);
  }

private:
  Slot *try_submit(Instruction &instruction, void *wave) {
    if (!helpers_enabled())
      return nullptr;
    // One or two loads detect an exhausted pool. Reservation is bounded even
    // under contention: after two failed CAS attempts per word, run inline.
    // A conservative miss is allowed; waiting for worker capacity is not.
    for (unsigned word = 0; word != (slots_.size() > 64 ? 2u : 1u); ++word) {
      auto free = free_[word].load(std::memory_order_relaxed);
      for (unsigned attempt = 0; free && attempt != 2; ++attempt) {
        thread_local unsigned next = std::hash<std::thread::id>{}(std::this_thread::get_id()) & 63;
        const unsigned bit = (std::countr_zero(std::rotr(free, int(next))) + next) & 63;
        if (!free_[word].compare_exchange_strong(free, free & ~(uint64_t{1} << bit),
                                                 std::memory_order_acquire,
                                                 std::memory_order_relaxed))
          continue;
        next = (bit + 1) & 63;
        auto *slot = slots_[word * 64 + bit].get();
        slot->helper.submit(instruction, wave);
        return slot;
      }
    }
    return nullptr;
  }
  void release(Slot *slot) {
    free_[slot->index / 64].fetch_or(uint64_t{1} << (slot->index % 64), std::memory_order_release);
  }
  alignas(64) std::array<std::atomic<uint64_t>, 2> free_{};

  std::vector<std::unique_ptr<Slot>> slots_;
};

/// Shared by all CUs and GPUs in one VM. Config parsing does not start helpers;
/// the first eligible instruction creates the pool.
class ExecutionResources {
public:
  explicit ExecutionResources(unsigned helpers) : helpers_(helpers) {}
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
