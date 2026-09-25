// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cpu_dispatch_pool.h
/// @brief Host CPU worker pool that drives CU wavefront execution in parallel.

#ifndef ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
#define ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class CpuDispatchPoolTestAccess;

/// @brief Pool of host threads executing one functional quantum per active CU.
///
/// @details This is host acceleration, not a modeled GPU resource. Changing the
/// width must preserve the observable result of a race-free workload.
/// For a pool constructed with N threads, run() uses its caller and up to N-1
/// shared workers, capped by that call's requested thread count.
/// Concurrent submissions must own disjoint CUs and result storage. Each CU runs
/// exactly once per submission; results and exceptions belong to that submission.
/// The caller keeps its spans alive until run() returns, and the pool must outlive
/// all run() calls.
///
/// Explicit affinity domains assign each CU to the same execution lane and each
/// lane to the same persistent worker on every submission. Besides improving
/// host-cache locality, this keeps thread-local execution observers in program
/// order without serializing their hot callbacks. Work is sent directly to the
/// selected workers: idle workers are not awakened to scan unrelated submissions.
/// Callers execute lane zero and join only the workers assigned to their own
/// submission. The pool retains N-1 workers total, independent of the number of
/// callers; no additional worker pool is created per XCD.
class CpuDispatchPool {
public:
  explicit CpuDispatchPool(uint32_t threads) : CpuDispatchPool(threads, std::nullopt) {}

  ~CpuDispatchPool() {
    for (auto &worker : workers_)
      worker.request_stop();
    for (auto &queue : worker_queues_)
      queue->cv.notify_all();
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
                              std::span<FunctionalQuantumResult> results,
                              const void *affinity_domain = nullptr) {
    if (tasks.empty())
      return {};
    if (results.size() != tasks.size())
      throw std::invalid_argument("dispatch result count must match task count");
    std::fill(results.begin(), results.end(), FunctionalQuantumResult{});
    const uint32_t lane_count =
        std::clamp<uint32_t>(threads, 1, static_cast<uint32_t>(workers_.size() + 1));
    const bool stable_affinity = affinity_domain != nullptr;
    Submission submission(tasks, results, lane_count, stable_affinity);
    if (lane_count > 1) {
      const uint32_t base = affinity_base(affinity_domain, lane_count);
      enqueue_worker_lanes(submission, base);
    }

    drain_tasks(submission, /*lane=*/0);

    std::unique_lock lock(submission.mutex);
    // Every nonempty worker lane owns tasks the caller must not steal: moving a
    // CU between host threads would fragment thread-local observer streams. The
    // final assigned worker notifies while holding mutex_, before this stack
    // object can be destroyed by its caller.
    submission.done_cv.wait(lock, [&] { return submission.pending_workers == 0; });
    auto first_exception = submission.first_exception;
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

  struct Submission {
    Submission(std::span<ComputeUnitCore *> tasks, std::span<FunctionalQuantumResult> results,
               uint32_t lane_count, bool stable_affinity)
        : tasks(tasks), results(results), lane_count(lane_count), stable_affinity(stable_affinity),
          lane_has_tasks(lane_count, false) {
      for (size_t index = 0; index != tasks.size(); ++index)
        lane_has_tasks[task_lane(tasks[index], index, lane_count, stable_affinity)] = true;
    }

    const std::span<ComputeUnitCore *> tasks;
    const std::span<FunctionalQuantumResult> results;
    const uint32_t lane_count;
    const bool stable_affinity;
    std::vector<bool> lane_has_tasks;
    std::mutex mutex;
    std::condition_variable done_cv;
    std::exception_ptr first_exception;
    uint32_t pending_workers = 0;
  };

  struct WorkerAssignment {
    Submission *submission = nullptr;
    uint32_t lane = 0;
  };

  struct WorkerQueue {
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<WorkerAssignment> assignments;
  };

  static uint32_t task_lane(const ComputeUnitCore *task, size_t task_index, uint32_t lane_count,
                            bool stable_affinity) {
    return stable_affinity ? static_cast<uint32_t>(task->id() % lane_count)
                           : static_cast<uint32_t>(task_index % lane_count);
  }

  CpuDispatchPool(uint32_t threads, std::optional<uint32_t> fail_after) {
    const uint32_t worker_count = std::max(threads, 1u) - 1;
    worker_queues_.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; ++i)
      worker_queues_.push_back(std::make_unique<WorkerQueue>());
    workers_.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; ++i) {
      if (fail_after && i == *fail_after)
        throw std::runtime_error("injected worker construction failure");
      workers_.emplace_back([this, i](std::stop_token stop) { worker_loop(stop, i); });
    }
  }

  uint32_t affinity_base(const void *affinity_domain, uint32_t lane_count) {
    assert(lane_count > 1 && !workers_.empty());
    std::lock_guard lock(affinity_mutex_);
    const uint32_t stride = lane_count - 1;
    if (!affinity_domain) {
      const uint32_t base = next_affinity_base_;
      next_affinity_base_ = (next_affinity_base_ + stride) % workers_.size();
      return base;
    }
    const auto [it, inserted] = affinity_bases_.try_emplace(affinity_domain, next_affinity_base_);
    if (inserted)
      next_affinity_base_ = (next_affinity_base_ + stride) % workers_.size();
    return it->second % workers_.size();
  }

  void enqueue_worker_lanes(Submission &submission, uint32_t base) {
    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(submission.lane_count - 1);
    for (uint32_t lane = 1; lane < submission.lane_count; ++lane) {
      if (!submission.lane_has_tasks[lane])
        continue;
      assignments.emplace_back((base + lane - 1) % static_cast<uint32_t>(worker_queues_.size()),
                               lane);
    }
    {
      std::lock_guard lock(submission.mutex);
      submission.pending_workers = static_cast<uint32_t>(assignments.size());
    }
    for (const auto [worker, lane] : assignments) {
      auto &queue = *worker_queues_[worker];
      {
        std::lock_guard lock(queue.mutex);
        queue.assignments.push_back({&submission, lane});
      }
      queue.cv.notify_one();
    }
  }

  void drain_tasks(Submission &submission, uint32_t lane) {
    for (size_t i = 0; i < submission.tasks.size(); ++i) {
      if (task_lane(submission.tasks[i], i, submission.lane_count, submission.stable_affinity) !=
          lane)
        continue;
      try {
        submission.results[i] = submission.tasks[i]->run_quantum();
      } catch (...) {
        std::lock_guard lock(submission.mutex);
        if (!submission.first_exception)
          submission.first_exception = std::current_exception();
      }
    }
  }

  void worker_loop(std::stop_token stop, uint32_t worker) {
    WorkerQueue &queue = *worker_queues_[worker];
    while (true) {
      std::unique_lock lock(queue.mutex);
      if (!queue.cv.wait(lock, stop, [&] { return !queue.assignments.empty(); }))
        return;
      const WorkerAssignment assignment = queue.assignments.front();
      queue.assignments.pop_front();
      lock.unlock();
      drain_tasks(*assignment.submission, assignment.lane);
      {
        std::lock_guard done_lock(assignment.submission->mutex);
        assert(assignment.submission->pending_workers != 0);
        if (--assignment.submission->pending_workers == 0)
          assignment.submission->done_cv.notify_one();
      }
    }
  }

  std::mutex affinity_mutex_;
  std::unordered_map<const void *, uint32_t> affinity_bases_;
  uint32_t next_affinity_base_ = 0;
  std::vector<std::unique_ptr<WorkerQueue>> worker_queues_;
  // Destroy jthreads before the state they inspect if construction throws.
  std::vector<std::jthread> workers_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
