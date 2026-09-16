// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cpu_dispatch_pool.h
/// @brief Host CPU worker pool that drives CU wavefront execution in parallel.

#ifndef ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
#define ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <algorithm>
#include <atomic>
#include <cassert>
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
/// @details This is host acceleration, not a modeled GPU resource. Changing the
/// width must preserve the observable result of a race-free workload.
/// For a pool constructed with N threads, run() uses its caller and up to N-1
/// shared workers, capped by that call's requested thread count.
/// Concurrent submissions must own disjoint CUs and result storage. Each CU runs
/// exactly once per submission; results and exceptions belong to that submission.
/// The caller keeps its spans alive until run() returns, and the pool must outlive
/// all run() calls.
///
/// Workers take submission assignments under a short shared lock, then claim CUs
/// with a submission-local atomic counter. Callers always drain their own work.
/// Each submission joins only its assigned workers, so an unrelated slow batch
/// cannot hold up its completion. The pool retains N-1 workers total, independent
/// of the number of callers; no additional worker pool is created per XCD.
class CpuDispatchPool {
public:
  explicit CpuDispatchPool(uint32_t threads) : CpuDispatchPool(threads, std::nullopt) {}

  ~CpuDispatchPool() {
    {
      std::lock_guard lock(mutex_);
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
    const uint32_t worker_goal =
        std::min<uint32_t>(threads - 1, static_cast<uint32_t>(workers_.size()));
    Submission submission(tasks, results, worker_goal);
    if (worker_goal != 0) {
      {
        std::lock_guard lock(mutex_);
        enqueue(submission);
      }
      for (uint32_t i = 0; i < worker_goal; ++i)
        work_cv_.notify_one();
    }

    drain_tasks(submission);

    std::unique_lock lock(mutex_);
    // The caller has exhausted the task index, so every CU has been claimed.
    // Cancel unused worker assignments and join only workers holding this batch.
    // The final worker notifies while holding mutex_, before this stack object
    // can be destroyed by its caller.
    if (submission.queued)
      unlink(submission);
    submission.worker_tickets = 0;
    submission.done_cv.wait(lock, [&] { return submission.active_workers == 0; });
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
               uint32_t tickets)
        : tasks(tasks), results(results), worker_tickets(tickets) {}

    const std::span<ComputeUnitCore *> tasks;
    const std::span<FunctionalQuantumResult> results;
    std::atomic<size_t> next_task{0};
    // Remaining fields are protected by the pool mutex.
    std::condition_variable done_cv;
    std::exception_ptr first_exception;
    uint32_t worker_tickets;
    uint32_t active_workers = 0;
    Submission *previous = nullptr;
    Submission *next = nullptr;
    bool queued = false;
  };

  CpuDispatchPool(uint32_t threads, std::optional<uint32_t> fail_after) {
    const uint32_t worker_count = std::max(threads, 1u) - 1;
    workers_.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; ++i) {
      if (fail_after && i == *fail_after)
        throw std::runtime_error("injected worker construction failure");
      workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop); });
    }
  }

  // Intrusive links keep publication and cancellation allocation-free. The
  // queue contains only submissions with unclaimed worker assignments.
  void enqueue(Submission &submission) {
    assert(!submission.queued && submission.worker_tickets != 0);
    submission.previous = ready_tail_;
    submission.next = nullptr;
    if (ready_tail_)
      ready_tail_->next = &submission;
    else
      ready_head_ = &submission;
    ready_tail_ = &submission;
    submission.queued = true;
  }

  void unlink(Submission &submission) {
    assert(submission.queued);
    if (submission.previous)
      submission.previous->next = submission.next;
    else
      ready_head_ = submission.next;
    if (submission.next)
      submission.next->previous = submission.previous;
    else
      ready_tail_ = submission.previous;
    submission.previous = submission.next = nullptr;
    submission.queued = false;
  }

  void drain_tasks(Submission &submission) {
    while (true) {
      const size_t i = submission.next_task.fetch_add(1, std::memory_order_relaxed);
      if (i >= submission.tasks.size())
        return;
      try {
        submission.results[i] = submission.tasks[i]->run_quantum();
      } catch (...) {
        std::lock_guard lock(mutex_);
        if (!submission.first_exception)
          submission.first_exception = std::current_exception();
      }
    }
  }

  void worker_loop(std::stop_token stop) {
    while (true) {
      std::unique_lock lock(mutex_);
      work_cv_.wait(lock, stop, [this] { return stopping_ || ready_head_ != nullptr; });
      if (stopping_ || stop.stop_requested())
        return;
      Submission &submission = *ready_head_;
      unlink(submission);
      --submission.worker_tickets;
      ++submission.active_workers;
      // Rotate pending assignments so arrivals can share free workers.
      if (submission.worker_tickets != 0)
        enqueue(submission);
      lock.unlock();
      drain_tasks(submission);
      lock.lock();
      if (--submission.active_workers == 0)
        submission.done_cv.notify_one();
    }
  }

  std::mutex mutex_;
  std::condition_variable_any work_cv_;
  Submission *ready_head_ = nullptr;
  Submission *ready_tail_ = nullptr;
  bool stopping_ = false;
  // Destroy jthreads before the state they inspect if construction throws.
  std::vector<std::jthread> workers_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CPU_DISPATCH_POOL_H_
