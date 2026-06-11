// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cpu_dispatch_pool.h
/// @brief Host CPU worker pool that drives CU wavefront execution in parallel.

#ifndef ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
#define ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Pool of host threads executing one functional quantum per active CU.
///
/// @details run() distributes one ComputeUnitCore::run_quantum() call per CU
/// across the calling thread plus up to N-1 workers. Each CU is executed by
/// exactly one thread per run() (no intra-CU parallelism). run() returns when
/// all CUs have completed their quantum.
///
/// Task hand-out is lock-free: workers and the calling thread claim CUs with a
/// single atomic fetch_add on @ref next_task_, and signal completion by
/// decrementing @ref remaining_. The mutex is held only for the wakeup/teardown
/// condition-variable predicates, never on the per-CU hot path. This keeps
/// scaling from collapsing into lock contention when many short quanta retire.
class CpuDispatchPool {
public:
  explicit CpuDispatchPool(uint32_t threads) {
    uint32_t worker_count = threads > 1 ? threads - 1 : 0;
    workers_.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; ++i)
      workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop); });
  }

  ~CpuDispatchPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    work_cv_.notify_all();
  }

  uint32_t thread_count() const { return static_cast<uint32_t>(workers_.size() + 1); }

  void run(std::span<ComputeUnitCore *> tasks, uint32_t threads) {
    if (tasks.empty())
      return;

    threads = std::clamp<uint32_t>(threads, 1, static_cast<uint32_t>(tasks.size()));
    uint32_t worker_goal =
        std::min<uint32_t>(threads > 1 ? threads - 1 : 0, static_cast<uint32_t>(workers_.size()));

    if (worker_goal == 0) {
      for (auto *cu : tasks)
        cu->run_quantum();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.assign(tasks.begin(), tasks.end());
      task_count_ = tasks_.size();
      next_task_.store(0, std::memory_order_relaxed);
      remaining_.store(task_count_, std::memory_order_relaxed);
      ++generation_;
    }
    for (uint32_t i = 0; i < worker_goal; ++i)
      work_cv_.notify_one();

    // The calling thread participates as one of the workers.
    drain_tasks();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
    tasks_.clear();
    task_count_ = 0;
  }

private:
  /// @brief Claim and execute CUs until the task queue is drained.
  ///
  /// Lock-free: each claim is one atomic fetch_add; the last completion wakes
  /// the thread blocked in run() via done_cv_.
  void drain_tasks() {
    while (true) {
      size_t i = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (i >= task_count_)
        return;
      tasks_[i]->run_quantum();
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_cv_.notify_one();
      }
    }
  }

  void worker_loop(std::stop_token stop) {
    uint64_t seen_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [this, &stop, seen_generation]() {
        return stopping_ || stop.stop_requested() || generation_ != seen_generation;
      });
      if (stopping_ || stop.stop_requested())
        return;
      seen_generation = generation_;
      lock.unlock();

      // Lock-free task draining; extra woken workers simply observe an empty
      // queue and loop back to wait.
      drain_tasks();
    }
  }

  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::jthread> workers_;
  std::vector<ComputeUnitCore *> tasks_;
  size_t task_count_ = 0;
  std::atomic<size_t> next_task_ = 0;
  std::atomic<size_t> remaining_ = 0;
  uint64_t generation_ = 0;
  bool stopping_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
