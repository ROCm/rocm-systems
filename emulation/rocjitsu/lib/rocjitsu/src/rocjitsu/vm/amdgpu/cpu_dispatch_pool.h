// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cpu_dispatch_pool.h
/// @brief Host CPU worker pool that drives CU wavefront execution in parallel.

#ifndef ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
#define ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <algorithm>
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
      next_task_ = 0;
      active_workers_ = 0;
      started_workers_ = 0;
      target_workers_ = worker_goal;
      ++generation_;
    }
    for (uint32_t i = 0; i < worker_goal; ++i)
      work_cv_.notify_one();

    while (auto *cu = take_task()) {
      cu->run_quantum();
      finish_task();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return next_task_ >= tasks_.size() && active_workers_ == 0; });
    tasks_.clear();
    target_workers_ = 0;
  }

private:
  ComputeUnitCore *take_task() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_task_ >= tasks_.size())
      return nullptr;
    ++active_workers_;
    return tasks_[next_task_++];
  }

  void finish_task() {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_workers_;
    if (next_task_ >= tasks_.size() && active_workers_ == 0)
      done_cv_.notify_one();
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
      if (started_workers_ >= target_workers_)
        continue;
      ++started_workers_;
      lock.unlock();

      // Same task-grab/accounting protocol as the main thread in run();
      // finish_task() notifies done_cv_ when the last task retires.
      while (auto *cu = take_task()) {
        cu->run_quantum();
        finish_task();
      }

      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::jthread> workers_;
  std::vector<ComputeUnitCore *> tasks_;
  size_t next_task_ = 0;
  size_t active_workers_ = 0;
  uint32_t started_workers_ = 0;
  uint32_t target_workers_ = 0;
  uint64_t generation_ = 0;
  bool stopping_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
