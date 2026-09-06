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
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class CpuDispatchPoolTestAccess;

/// @brief Pool of host threads executing one functional quantum per active CU.
///
/// @details run() distributes one ComputeUnitCore::run_quantum() call per CU
/// across the calling thread plus up to N-1 workers. Each CU is executed by
/// exactly one thread per run() (no intra-CU parallelism). run() returns when
/// all CUs have completed their quantum. The output-span overload preserves
/// each CU's result so its owner can schedule the next quantum independently.
///
/// Task hand-out is lock-free: workers and the calling thread claim CUs with a
/// single atomic fetch_add on @ref next_task_, and signal completion by
/// decrementing @ref remaining_. The mutex is held only for the wakeup/teardown
/// condition-variable predicates, never on the per-CU hot path. This keeps
/// scaling from collapsing into lock contention when many short quanta retire.
class CpuDispatchPool {
public:
  explicit CpuDispatchPool(uint32_t threads) : CpuDispatchPool(threads, std::nullopt) {}

  ~CpuDispatchPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    for (auto &worker : workers_)
      worker.request_stop();
    work_cv_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable())
        worker.join();
  }

  uint32_t thread_count() const { return static_cast<uint32_t>(workers_.size() + 1); }

  FunctionalQuantumResult run(std::span<ComputeUnitCore *> tasks, uint32_t threads) {
    std::vector<FunctionalQuantumResult> results(tasks.size());
    return run(tasks, threads, results);
  }

  FunctionalQuantumResult run(std::span<ComputeUnitCore *> tasks, uint32_t threads,
                              std::span<FunctionalQuantumResult> results) {
    if (tasks.empty())
      return {};
    if (results.size() != tasks.size())
      throw std::invalid_argument("dispatch result count must match task count");

    std::lock_guard<std::mutex> run_lock(run_mutex_);
    std::fill(results.begin(), results.end(), FunctionalQuantumResult{});

    threads = std::clamp<uint32_t>(threads, 1, static_cast<uint32_t>(tasks.size()));
    uint32_t worker_goal =
        std::min<uint32_t>(threads > 1 ? threads - 1 : 0, static_cast<uint32_t>(workers_.size()));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.assign(tasks.begin(), tasks.end());
      task_data_.store(tasks_.data(), std::memory_order_release);
      result_data_.store(results.data(), std::memory_order_release);
      task_count_.store(tasks_.size(), std::memory_order_release);
      next_task_.store(0, std::memory_order_relaxed);
      remaining_.store(tasks_.size(), std::memory_order_relaxed);
      worker_tickets_ = worker_goal;
      first_exception_ = nullptr;
    }
    for (uint32_t i = 0; i < worker_goal; ++i)
      work_cv_.notify_one();

    // The calling thread participates as one of the workers.
    drain_tasks();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
    done_cv_.wait(lock, [this]() { return worker_tickets_ == 0 && active_workers_ == 0; });
    task_count_.store(0, std::memory_order_release);
    task_data_.store(nullptr, std::memory_order_release);
    result_data_.store(nullptr, std::memory_order_release);
    tasks_.clear();
    std::exception_ptr first_exception = first_exception_;
    first_exception_ = nullptr;
    lock.unlock();
    if (first_exception)
      std::rethrow_exception(first_exception);
    FunctionalQuantumResult result;
    for (const auto &task_result : results)
      result.merge(task_result);
    return result;
  }

private:
  friend class CpuDispatchPoolTestAccess;

  CpuDispatchPool(uint32_t threads, std::optional<uint32_t> fail_after) {
    threads = std::max(threads, 1u);
    uint32_t worker_count = threads > 1 ? threads - 1 : 0;
    workers_.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; ++i) {
      if (fail_after && i == *fail_after)
        throw std::runtime_error("injected worker construction failure");
      workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop); });
    }
  }
  /// @brief Claim and execute CUs until the task queue is drained.
  ///
  /// Lock-free: each claim is one atomic fetch_add; the last completion wakes
  /// the thread blocked in run() via done_cv_.
  void drain_tasks() {
    ComputeUnitCore **tasks = task_data_.load(std::memory_order_acquire);
    FunctionalQuantumResult *results = result_data_.load(std::memory_order_acquire);
    size_t task_count = task_count_.load(std::memory_order_acquire);
    while (true) {
      size_t i = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (i >= task_count)
        return;
      try {
        results[i] = tasks[i]->run_quantum();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_exception_)
          first_exception_ = std::current_exception();
      }
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_cv_.notify_one();
      }
    }
  }

  void worker_loop(std::stop_token stop) {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, stop, [this]() { return stopping_ || worker_tickets_ != 0; });
      if (stopping_ || stop.stop_requested())
        return;
      --worker_tickets_;
      ++active_workers_;
      lock.unlock();

      // Lock-free task draining; extra woken workers simply observe an empty
      // queue and loop back to wait.
      drain_tasks();

      lock.lock();
      --active_workers_;
      if (worker_tickets_ == 0 && active_workers_ == 0)
        done_cv_.notify_one();
    }
  }

  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable_any work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::jthread> workers_;
  std::vector<ComputeUnitCore *> tasks_;
  std::atomic<ComputeUnitCore **> task_data_ = nullptr;
  std::atomic<FunctionalQuantumResult *> result_data_ = nullptr;
  std::atomic<size_t> task_count_ = 0;
  std::atomic<size_t> next_task_ = 0;
  std::atomic<size_t> remaining_ = 0;
  // Protected by mutex_; keeps the task vector alive until ticketed workers leave drain_tasks().
  size_t worker_tickets_ = 0;
  size_t active_workers_ = 0;
  std::exception_ptr first_exception_;
  bool stopping_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
