/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "latency_profiler/CollTrace.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "param.h"

NCCL_PARAM(ColltraceRecordMax, "COLLTRACE_RECORD_MAX", 100);
NCCL_PARAM(ColltraceMaxDumpSize, "COLLTRACE_MAX_DUMP_SIZE", 20);
NCCL_PARAM(ColltraceDumpIntervalSec, "COLLTRACE_DUMP_INTERVAL_SEC", 300);

constexpr int RANKS_PER_HOST = 8;

namespace latency_profiler {

namespace {
CudaEventPtr getCudaEventPtr() {
  cudaEvent_t newEvent = nullptr;
  CUDACHECKIGNORE(cudaEventCreate(&newEvent));
  CudaEventPtr item(newEvent);
  return item;
}
} // namespace

CollTrace::CollTrace(ncclComm* comm)
    : comm_(comm),
      commHash_(std::to_string(comm->commHash)),
      rank_(comm->rank) {
  profilingWorkerThread_ =
      std::thread{[this]() { return collTraceThreadFn(comm_->cudaDev); }};
}

CollTrace::~CollTrace() {
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: commHash %s rank %d - Destroy START",
        commHash_.c_str(),
        rank_);
    eventQueue_.push(std::unique_ptr<CollTraceEvent>(
        new CollTraceEvent(CollTraceEvent::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    if (rank_ == 0) {
      reportIfNeeded(false);
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: commHash %s rank %d - Destroy COMPLETE",
        commHash_.c_str(),
        rank_);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: commHash %s rank %d - Destroy FAILED: %s",
        commHash_.c_str(),
        rank_,
        e.what());
  }
}

void* CollTrace::collTraceThreadFn(int cudaDev) {
  INFO(NCCL_INIT, "CollTrace thread started for cudaDev %d", cudaDev);
  auto err = cudaSetDevice(cudaDev);
  if (err != cudaSuccess) {
    WARN("Cuda failure '%s'", cudaGetErrorString(err));
    return nullptr;
  }

  lastReportTime_ = std::chrono::steady_clock::now();

  INFO(
      NCCL_INIT,
      "COLLTRACE: commHash %s rank %d - worker thread STARTED",
      commHash_.c_str(),
      rank_);
  while (true) {
    curEvent_ = eventQueue_.waitPop();
    if (curEvent_->eventType == CollTraceEvent::EventType::TERMINATE) {
      break;
    }
    curEvent_->start->waitEventFinish();
    auto ncclRes = curEvent_->stop->waitEventFinish();
    float latency = -1;

    if (ncclRes == ncclSuccess) {
      auto latencyMaybe =
          curEvent_->stop->getElapsedTimeSinceEvent(curEvent_->start.get());
      // latencyMaybe could be nullopt when cudaEventElapsedTime failed
      // this could happen when events are not recorded or stream is not valid
      if (latencyMaybe == nullptr) {
        WARN(
            "CollTrace: getElapsedTimeSinceEvent failed, aborting worker thread");
        return nullptr;
      }
      latency = *latencyMaybe;
    }

    recordCurCollResult(cudaDev, latency);
    curEvent_.reset();
  }

  INFO(
      NCCL_INIT,
      "COLLTRACE: commHash %s rank %d - worker thread TERMINATE",
      commHash_.c_str(),
      rank_);
  return nullptr;
}

void CollTrace::enqueueEvent(std::unique_ptr<CollTraceEvent> event) {
  event->coll.collId = curCollId_.fetch_add(1);
  eventQueue_.push(std::move(event));
}

std::unique_ptr<CollTraceEvent> CollTrace::createEvent(
    CollTraceEvent::EventType type) {
  auto eventInfo = std::make_unique<CollTraceEvent>(type);
  eventInfo->start = std::make_unique<CudaWaitEvent>(getCudaEventPtr());
  eventInfo->stop = std::make_unique<CudaWaitEvent>(getCudaEventPtr());

  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<CollTraceEvent> nullCollTraceEvent(nullptr);
    return nullCollTraceEvent;
  }
  return eventInfo;
}

bool shouldAggregateRingBuffer(int collId) {
  const int NCCL_COLLTRACE_RECORD_MAX = ncclParamColltraceRecordMax();
  return ((collId + 1) % NCCL_COLLTRACE_RECORD_MAX == 0);
}

void CollTrace::reportIfNeeded(bool checkInterval = true) {
  auto now = std::chrono::steady_clock::now();
  auto secs_passed =
      std::chrono::duration_cast<std::chrono::seconds>(now - lastReportTime_)
          .count();
  if (checkInterval) {
    if (secs_passed < ncclParamColltraceDumpIntervalSec() &&
        stats_.size() < ncclParamColltraceMaxDumpSize()) {
      return;
    }
  }

  INFO(
      NCCL_COLL,
      "CollTrace: %ld seconds passed since last report, stats size = %zu, checkInterval = %d",
      secs_passed,
      stats_.size(),
      checkInterval);

// reportToScuba is a placeholder for oss environment.
// meta production reports to scuba instead of file, which enables
// filering, aggregation and visualization.
#ifdef ENABLE_SCUBA_LOGGING
  reportToScuba(stats_, commHash_);
#else
  reportToFile(stats_, commHash_);
#endif
  lastReportTime_ = std::chrono::steady_clock::now();
  stats_.clear();
}

void CollTrace::recordCurCollResult(int rank, float latency) {
  const int NCCL_COLLTRACE_RECORD_MAX = ncclParamColltraceRecordMax();

  auto result = std::make_unique<CollTraceInfo>(curEvent_->coll);
  auto collId = result->collId;
  result->latencyMs = latency;

  pastColls_.push_back(std::move(result));
  if (pastColls_.size() > NCCL_COLLTRACE_RECORD_MAX) {
    pastColls_.pop_front();
  }

  if (shouldAggregateRingBuffer(collId) && pastColls_.size() > 0) {
    std::vector<float> latencyAllGather;
    latencyAllGather.resize(RANKS_PER_HOST * NCCL_COLLTRACE_RECORD_MAX, 0);
    int start = (comm_->localRank) * NCCL_COLLTRACE_RECORD_MAX;
    for (int i = start; i < start + NCCL_COLLTRACE_RECORD_MAX; i++) {
      latencyAllGather[i] = pastColls_[i - start]->latencyMs;
    }
    auto before = std::chrono::high_resolution_clock::now();
    auto ncclResult = bootstrapIntraNodeAllGather(
        comm_->bootstrap,
        comm_->localRankToRank,
        comm_->localRank,
        comm_->localRanks,
        latencyAllGather.data(),
        NCCL_COLLTRACE_RECORD_MAX * sizeof(float));
    auto after = std::chrono::high_resolution_clock::now();
    auto interval_us =
        std::chrono::duration_cast<std::chrono::microseconds>(after - before)
            .count();
    if (ncclResult != ncclSuccess) {
      WARN("CollTrace: All gather exchange latency data failed");
      return;
    }

    if (rank == 0) {
      INFO(NCCL_COLL, "latency metrics all gather takes %ld us", interval_us);
      try {
        auto stats = aggregateResults(
            pastColls_,
            latencyAllGather,
            RANKS_PER_HOST,
            NCCL_COLLTRACE_RECORD_MAX);
        stats_.push_back(stats);
        if (stats_.size() > ncclParamColltraceMaxDumpSize()) {
          stats_.pop_front();
        }
        reportIfNeeded();
      } catch (const std::exception& e) {
        WARN("Aggregating error: %s", e.what());
      }
    }
  }
}

} // namespace latency_profiler
