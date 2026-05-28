/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <thread>
#include <atomic>
#include "CollTraceEvent.h"
#include "CollTraceUtils.h"
#include "EventQueue.h"

struct ncclComm;

namespace latency_profiler {

struct CollStats;

// CollTrace Workflow
// 1) measure collective latency by adding cuda/hip event before and after
// a kernel launch in RCCL.

// 2) Started a separate worker thread to collect latency data into
// its local ring buffer.

// 3) Perform CPU level all gather to exchange latency data
// between threads (200 to 300us per 100 latency data exchange)

// 4) Report to scuba when results buffer is full or every few minutes

class CollTrace {
 public:
  CollTrace(ncclComm* comm);
  __attribute__((visibility("default"))) ~CollTrace();

  void enqueueEvent(std::unique_ptr<CollTraceEvent> event);

  std::unique_ptr<CollTraceEvent> createEvent(
      CollTraceEvent::EventType type = CollTraceEvent::EventType::COMM);

  void recordCurCollResult(int rank, float latencyMs);

  void reportIfNeeded(bool checkInterval);

 private:
  EventQueue<CollTraceEvent> eventQueue_;
  std::thread profilingWorkerThread_;
  void* collTraceThreadFn(int cudaDev);
  std::atomic<uint64_t> curCollId_{0};
  std::unique_ptr<CollTraceEvent> curEvent_;
  std::deque<std::unique_ptr<CollTraceInfo>> pastColls_;

  ncclComm* comm_{nullptr};
  std::string commHash_;
  int rank_{-1};
  std::deque<std::vector<CollStats>> stats_;
  std::chrono::time_point<std::chrono::steady_clock> lastReportTime_;
};
} // namespace latency_profiler
