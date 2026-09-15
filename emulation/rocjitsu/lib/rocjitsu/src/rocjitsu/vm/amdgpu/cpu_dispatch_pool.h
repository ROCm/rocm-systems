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
/// This is host acceleration machinery, not a modeled GPU resource: changing
/// its width must preserve the observable result of race-free workloads.
///
/// Each run() owns its submission state and contributes up to its requested
/// number of work lanes to one shared queue. Workers claim a lane, then claim
/// that submission's CUs with a single atomic fetch_add. This allows independent
/// command processors to use otherwise-idle pool capacity without sharing
/// result, completion, or exception state. The pool still bounds execution to N
/// host threads: N-1 persistent workers plus at most one participating caller.
class CpuDispatchPool {
  struct Submission;

  struct WorkLane {
    Submission *submission = nullptr;
    WorkLane *next = nullptr;
  };

  struct Submission {
    Submission(std::span<ComputeUnitCore *> submitted_tasks,
               std::span<FunctionalQuantumResult> submitted_results, uint32_t lane_count)
        : tasks(submitted_tasks), results(submitted_results), lanes(lane_count),
          remaining_lanes(lane_count) {
      for (auto &lane : lanes)
        lane.submission = this;
    }

    std::span<ComputeUnitCore *> tasks;
    std::span<FunctionalQuantumResult> results;
    std::vector<WorkLane> lanes;
    std::atomic<size_t> next_task = 0;
    // Protected by CpuDispatchPool::mutex_. Completion of the final lane makes
    // all task results and the exception visible to the submitting thread.
    size_t remaining_lanes;
    std::exception_ptr first_exception;
  };

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

    std::fill(results.begin(), results.end(), FunctionalQuantumResult{});

    threads = std::clamp<uint32_t>(threads, 1, static_cast<uint32_t>(tasks.size()));
    uint32_t lane_count = std::min(threads, thread_count());
    Submission submission(tasks, results, lane_count);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto &lane : submission.lanes)
        enqueue_lane(&lane);
    }
    work_cv_.notify_all();

    help_until_done(submission);

    std::exception_ptr first_exception;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      first_exception = submission.first_exception;
    }
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

  /// @brief Claim and execute CUs for one submission work lane.
  void drain_submission(Submission &submission) {
    while (true) {
      size_t i = submission.next_task.fetch_add(1, std::memory_order_relaxed);
      if (i >= submission.tasks.size())
        break;
      try {
        submission.results[i] = submission.tasks[i]->run_quantum();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!submission.first_exception)
          submission.first_exception = std::current_exception();
      }
    }

    bool done;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done = --submission.remaining_lanes == 0;
    }
    if (done)
      work_cv_.notify_all();
  }

  /// @brief Let one submitting thread at a time provide the pool's Nth lane.
  void help_until_done(Submission &own_submission) {
    while (true) {
      WorkLane *lane;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this, &own_submission]() {
          return own_submission.remaining_lanes == 0 || (!caller_active_ && ready_head_ != nullptr);
        });
        if (own_submission.remaining_lanes == 0)
          return;
        caller_active_ = true;
        lane = dequeue_lane();
      }

      drain_submission(*lane->submission);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        caller_active_ = false;
      }
      work_cv_.notify_all();
    }
  }

  void worker_loop(std::stop_token stop) {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, stop, [this]() { return stopping_ || ready_head_ != nullptr; });
      if (stopping_ || stop.stop_requested())
        return;
      WorkLane *lane = dequeue_lane();
      lock.unlock();

      drain_submission(*lane->submission);
    }
  }

  // Requires mutex_. WorkLane storage is owned by the corresponding run()'s
  // stack submission and remains alive until every one of its lanes completes.
  void enqueue_lane(WorkLane *lane) {
    if (ready_tail_)
      ready_tail_->next = lane;
    else
      ready_head_ = lane;
    ready_tail_ = lane;
  }

  // Requires mutex_.
  WorkLane *dequeue_lane() {
    WorkLane *lane = ready_head_;
    ready_head_ = lane->next;
    if (!ready_head_)
      ready_tail_ = nullptr;
    lane->next = nullptr;
    return lane;
  }

  std::mutex mutex_;
  std::condition_variable_any work_cv_;
  std::vector<std::jthread> workers_;
  WorkLane *ready_head_ = nullptr;
  WorkLane *ready_tail_ = nullptr;
  bool caller_active_ = false;
  bool stopping_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
