// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/instruction.h"

#include <array>
#include <atomic>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace rocjitsu::amdgpu::matrix_coexecution {

// Experimental controls: 0 = ordinary issue, 1 = scan only, 2 = serial batch,
// 3 = parallel batch. Each process keeps its selected mode for the entire run.
inline int mode() {
  static const int value = [] {
    const char *text = std::getenv("RJ_MATRIX_COEXEC");
    return text ? std::atoi(text) : 0;
  }();
  return value;
}

inline unsigned width() {
  static const unsigned value = [] {
    const char *text = std::getenv("RJ_MMA_HELPERS");
    const int helpers = text ? std::atoi(text) : 1;
    return 1u + static_cast<unsigned>(helpers < 0 ? 0 : helpers > 7 ? 7 : helpers);
  }();
  return value;
}

// Zero uses the standard library's atomic wait policy. A positive value adds
// a bounded spin before sleeping, on both sides of the handoff. This is an
// experiment control, not a CPU-count-aware production scheduling policy.
inline unsigned spin_count() {
  static const unsigned value = [] {
    const char *text = std::getenv("RJ_MMA_SPINS");
    const int spins = text ? std::atoi(text) : 0;
    return static_cast<unsigned>(spins < 0 ? 0 : spins);
  }();
  return value;
}

inline bool private_futex() {
#if defined(__linux__)
  static const bool value = [] {
    const char *text = std::getenv("RJ_MMA_WAIT");
    return !text || std::atoi(text) == 1;
  }();
  return value;
#else
  return false;
#endif
}

inline void relax_cpu() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline bool candidate(std::string_view mnemonic) {
  return ((mnemonic.starts_with("v_wmma_f32_16x16x64_") ||
           mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
          (mnemonic.ends_with("fp8_fp8") || mnemonic.ends_with("fp8_bf8") ||
           mnemonic.ends_with("bf8_fp8") || mnemonic.ends_with("bf8_bf8"))) ||
         mnemonic == "v_wmma_f32_32x16x128_f4";
}

struct Range {
  uint32_t base = 0;
  uint32_t count = 0;
  bool overlaps(Range other) const {
    return count && other.count && base < other.base + other.count && other.base < base + count;
  }
};

struct Footprint {
  Range output;
  std::array<Range, 3> inputs;
};

inline bool independent(const Footprint &a, const Footprint &b) {
  if (a.output.overlaps(b.output))
    return false;
  for (const auto &input : b.inputs)
    if (a.output.overlaps(input))
      return false;
  for (const auto &input : a.inputs)
    if (b.output.overlaps(input))
      return false;
  return true;
}

struct Stats {
  uint64_t candidates = 0;
  uint64_t adjacent = 0;
  uint64_t conflicts = 0;
  uint64_t batches = 0;
  uint64_t covered = 0;
  std::array<uint64_t, 9> widths{};
  void flush() {
    if (candidates)
      std::fprintf(
          stderr,
          "RJ_COEXEC candidates=%llu adjacent=%llu conflicts=%llu batches=%llu "
          "covered=%llu width2=%llu width3=%llu width4=%llu width5=%llu width6=%llu "
          "width7=%llu width8=%llu\n",
          static_cast<unsigned long long>(candidates), static_cast<unsigned long long>(adjacent),
          static_cast<unsigned long long>(conflicts), static_cast<unsigned long long>(batches),
          static_cast<unsigned long long>(covered), static_cast<unsigned long long>(widths[2]),
          static_cast<unsigned long long>(widths[3]), static_cast<unsigned long long>(widths[4]),
          static_cast<unsigned long long>(widths[5]), static_cast<unsigned long long>(widths[6]),
          static_cast<unsigned long long>(widths[7]), static_cast<unsigned long long>(widths[8]));
    candidates = adjacent = conflicts = batches = covered = 0;
    widths.fill(0);
  }
  ~Stats() { flush(); }
};
inline thread_local Stats stats;

// Persistent helpers belong to the issuing host thread and are constructed lazily.
// Instructions remain allocated and destroyed by the original decoder thread.
// The caller validates disjoint register accesses and materializes destinations
// before publication. Both operations complete before the CU can be rescheduled.
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
    while ((state_.load(std::memory_order_acquire) & ~kSleeping) != 0)
      wait_while(1);
    return error_;
  }

  void execute_pair(Instruction &a, Instruction &b, void *wave) {
    submit(a, wave);
    std::exception_ptr local_error;
    try {
      b.execute(b, wave);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (auto error = wait())
      std::rethrow_exception(error);
    if (local_error)
      std::rethrow_exception(local_error);
  }

private:
  // There is exactly one waiter: the owner while a job is running, or the
  // helper while idle. Encoding its sleep intent in the same atomic word as
  // the job state prevents a missed wake without a shared waiter registry.
  static constexpr int kSleeping = 2;

  void publish(int value) {
#if defined(__linux__)
    if (private_futex()) {
      const int previous = state_.exchange(value, std::memory_order_release);
      if (previous & kSleeping)
        syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
      return;
    }
#endif
    state_.store(value, std::memory_order_release);
    state_.notify_one();
  }

  void wait_while(int expected) {
    for (unsigned i = 0; i != spin_count(); ++i) {
      if ((state_.load(std::memory_order_acquire) & ~kSleeping) != expected)
        return;
      relax_cpu();
    }
#if defined(__linux__)
    if (private_futex()) {
      int observed = expected;
      if (state_.compare_exchange_strong(observed, expected | kSleeping, std::memory_order_acq_rel,
                                         std::memory_order_acquire) ||
          observed == (expected | kSleeping))
        syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, expected | kSleeping, nullptr, nullptr, 0);
      return; // The caller rechecks after a spurious wake, EINTR or EAGAIN.
    }
#endif
    state_.wait(expected, std::memory_order_acquire);
  }

  void work() {
    for (;;) {
      int state = state_.load(std::memory_order_acquire);
      if (state == -1)
        return;
      if ((state & ~kSleeping) == 0) {
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

  static_assert(sizeof(std::atomic<int>) == 4 && std::atomic<int>::is_always_lock_free);
  alignas(64) std::atomic<int> state_{0};
  alignas(64) Instruction *instruction_ = nullptr;
  void *wave_ = nullptr;
  std::fenv_t environment_{};
  std::exception_ptr error_;
  std::thread thread_;
};

inline void execute_batch(std::span<Instruction *> instructions, void *wave) {
  thread_local std::vector<std::unique_ptr<Helper>> helpers;
  while (helpers.size() + 1 < instructions.size())
    helpers.push_back(std::make_unique<Helper>());
  for (size_t i = 0; i + 1 < instructions.size(); ++i)
    helpers[i]->submit(*instructions[i], wave);
  std::exception_ptr local_error;
  try {
    instructions.back()->execute(*instructions.back(), wave);
  } catch (...) {
    local_error = std::current_exception();
  }
  std::exception_ptr first_error;
  for (size_t i = 0; i + 1 < instructions.size(); ++i)
    if (auto error = helpers[i]->wait(); error && !first_error)
      first_error = error;
  if (first_error)
    std::rethrow_exception(first_error);
  if (local_error)
    std::rethrow_exception(local_error);
}

} // namespace rocjitsu::amdgpu::matrix_coexecution
