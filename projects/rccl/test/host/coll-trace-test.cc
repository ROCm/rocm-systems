/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the RCCL_LATENCY_PROFILER collective tracer in src/misc/latency_profiler/
// (CollTrace.cc, CollTraceEvent.cc, CollTraceFunc.cc, CollTraceUtils.cc) and its EventQueue.h. HIP
// events, the intra-node all-gather and the parameters are seamed here; the worker thread is real.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "comm.h"
#include "latency_profiler/CollTraceFunc.h"
#include "fakes/debug_fakes.h"

namespace {
std::mutex g_mutex;  // the worker thread reaches most of these seams
std::map<std::string, int64_t> g_params;
std::map<std::string, std::string> g_env;
uintptr_t g_nextEvent = 0x100;
hipError_t g_eventCreateResult = hipSuccess;
std::vector<hipEvent_t> g_destroyedEvents;
std::deque<hipError_t> g_queryResults;  // then hipSuccess
int g_queries = 0;
std::deque<float> g_elapsed;  // then 1.0
hipError_t g_elapsedResult = hipSuccess;
std::vector<std::pair<hipEvent_t, hipStream_t>> g_recordedEvents;
hipError_t g_recordResult = hipSuccess;
std::vector<int> g_setDevices;
hipError_t g_setDeviceResult = hipSuccess;
ncclResult_t g_capturingResult = ncclSuccess;
bool g_capturing = false;

struct AllGatherCall {
  int* ranks;
  int rank, nranks, size;
  std::vector<float> mine;  // what the caller put in its own slice
};
std::vector<AllGatherCall> g_allGathers;
ncclResult_t g_allGatherResult = ncclSuccess;
float g_peerLatency = 5.0f;  // every other rank reports this for every collective
}  // namespace

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t, int64_t*, int8_t*) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
}
const char* ncclGetEnv(const char* name) {
  auto it = g_env.find(name);
  return it == g_env.end() ? nullptr : it->second.c_str();
}

hipError_t hipEventCreate(hipEvent_t* event) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_eventCreateResult != hipSuccess) return g_eventCreateResult;
  *event = reinterpret_cast<hipEvent_t>(g_nextEvent++);
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_destroyedEvents.push_back(event);
  return hipSuccess;
}
hipError_t hipEventQuery(hipEvent_t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ++g_queries;
  if (g_queryResults.empty()) return hipSuccess;
  hipError_t res = g_queryResults.front();
  g_queryResults.pop_front();
  return res;
}
hipError_t hipEventElapsedTime(float* ms, hipEvent_t, hipEvent_t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_elapsedResult != hipSuccess) return g_elapsedResult;
  *ms = 1.0f;
  if (!g_elapsed.empty()) {
    *ms = g_elapsed.front();
    g_elapsed.pop_front();
  }
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_recordedEvents.push_back({event, stream});
  return g_recordResult;
}
hipError_t hipSetDevice(int dev) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_setDevices.push_back(dev);
  return g_setDeviceResult;
}
hipError_t hipGetLastError() { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, hipStream_t, int) {
  graph->graph = g_capturing ? reinterpret_cast<hipGraph_t>(0x6a) : nullptr;
  return g_capturingResult;
}

ncclResult_t bootstrapIntraNodeAllGather(void*, int* ranks, int rank, int nranks, void* allData, int size) {
  std::lock_guard<std::mutex> lock(g_mutex);
  float* data = static_cast<float*>(allData);
  int per = size / (int)sizeof(float);
  g_allGathers.push_back({ranks, rank, nranks, size, std::vector<float>(data + rank * per, data + (rank + 1) * per)});
  if (g_allGatherResult != ncclSuccess) return g_allGatherResult;
  for (int r = 0; r < nranks; ++r) {
    if (r == rank) continue;
    for (int i = 0; i < per; ++i) data[r * per + i] = g_peerLatency;
  }
  return ncclSuccess;
}

const char* ncclFuncToString(ncclFunc_t fn) { return fn == ncclFuncAllReduce ? "AllReduce" : "Other"; }
const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclUint8: return "ncclUint8";
  default: return "unknown";
  }
}

#include COLL_TRACE_UTILS_CC_PATH
#include COLL_TRACE_EVENT_CC_PATH
#include COLL_TRACE_CC_PATH
#include COLL_TRACE_FUNC_CC_PATH

namespace {

using namespace latency_profiler;

size_t AllGatherCount() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_allGathers.size();
}

int QueryCount() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_queries;
}

template <typename Pred>
bool WaitUntil(Pred pred) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!pred()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

class CollTraceMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::lock_guard<std::mutex> lock(g_mutex);
    ResetDebugLog();
    g_params = {{"NCCL_COLLTRACE_CHECK_INTERVAL_MS", 0}};
    g_env.clear();
    g_eventCreateResult = hipSuccess;
    g_destroyedEvents.clear();
    g_queryResults.clear();
    g_queries = 0;
    g_elapsed.clear();
    g_elapsedResult = hipSuccess;
    g_recordedEvents.clear();
    g_recordResult = hipSuccess;
    g_setDevices.clear();
    g_setDeviceResult = hipSuccess;
    g_capturingResult = ncclSuccess;
    g_capturing = false;
    g_allGathers.clear();
    g_allGatherResult = ncclSuccess;
    g_peerLatency = 5.0f;
    comm_ = std::make_unique<ncclComm>();
    comm_->commHash = 0xabc;
    comm_->rank = 0;
    comm_->cudaDev = 0;
    comm_->localRank = 1;
    comm_->localRanks = 8;  // CollTrace aggregates exactly eight local ranks
    comm_->localRankToRank = localRankToRank_;
    comm_->bootstrap = reinterpret_cast<void*>(0xb0);
  }

  void TearDown() override { delete comm_->ctrace; }

  void SetParam(const char* key, int64_t value) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_params[key] = value;
  }

  // Enqueues one AllReduce of `count` floats whose stop event reports `latencyMs`.
  void Enqueue(CollTrace* trace, float latencyMs, int64_t count = 1024, const char* dataType = "ncclFloat32") {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_elapsed.push_back(latencyMs);
    }
    auto event = trace->createEvent();
    event->coll.opName = "AllReduce";
    event->coll.dataType = dataType;
    event->coll.count = count;
    trace->enqueueEvent(std::move(event));
  }

  std::unique_ptr<ncclComm> comm_;
  int localRankToRank_[8] = {4, 6, 8, 10, 12, 14, 16, 18};
};

TEST_F(CollTraceMicrotest, Utils_SizeMbScalesByElementWidth) {
  const int n = 1 << 20;
  for (const char* t : {"ncclInt8", "ncclFp8E4M3", "ncclFp8E5M2"}) EXPECT_FLOAT_EQ(getSizeMb(t, n), 1.0f) << t;
  for (const char* t : {"ncclFloat16", "ncclBfloat16"}) EXPECT_FLOAT_EQ(getSizeMb(t, n), 2.0f) << t;
  for (const char* t : {"ncclInt32", "ncclUint32", "ncclFloat32"}) EXPECT_FLOAT_EQ(getSizeMb(t, n), 4.0f) << t;
  for (const char* t : {"ncclInt64", "ncclUint64", "ncclFloat64"}) EXPECT_FLOAT_EQ(getSizeMb(t, n), 8.0f) << t;
}

// The size table lacks ncclDatatypeToString's FP8 and Uint8 names; a fix flips this pin.
TEST_F(CollTraceMicrotest, Utils_KnownBug_SizeMbRejectsUnknownTypes) {
  for (const char* t : {"ncclUint8", "ncclFloat8e4m3", "ncclFloat8e5m2"}) {
    try {
      getSizeMb(t, 1);
      ADD_FAILURE() << t;
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()), std::string("CollTrace: unsupported data type ") + t);
    }
  }
}

TEST_F(CollTraceMicrotest, Utils_AggregateTakesMinAndMaxAcrossRanks) {
  std::deque<std::unique_ptr<CollTraceInfo>> info;
  info.push_back(std::make_unique<CollTraceInfo>(CollTraceInfo{7, "AllReduce", "ncclFloat32", 10}));
  info.push_back(std::make_unique<CollTraceInfo>(CollTraceInfo{8, "Other", "ncclInt8", 20}));
  // 3 ranks x 2 collectives: collective 0 sees 2,4,3 ms; collective 1 sees 1,1,1.5 ms.
  std::vector<float> lat = {2, 1, 4, 1, 3, 1.5};

  std::vector<CollStats> stats = aggregateResults(info, lat, 3, 2);

  ASSERT_EQ(stats.size(), 2u);
  EXPECT_EQ(stats[0].collId, 7);
  EXPECT_EQ(stats[0].percent, 100);
  EXPECT_FLOAT_EQ(stats[0].minLatencyUs, 2000);
  EXPECT_FLOAT_EQ(stats[0].maxLatencyUs, 4000);
  EXPECT_EQ(stats[0].opName, "AllReduce");
  EXPECT_EQ(stats[0].dataType, "ncclFloat32");
  EXPECT_EQ(stats[0].count, 10);
  EXPECT_EQ(stats[1].collId, 8);
  EXPECT_EQ(stats[1].percent, 50);
  EXPECT_FLOAT_EQ(stats[1].minLatencyUs, 1000);
  EXPECT_FLOAT_EQ(stats[1].maxLatencyUs, 1500);
  EXPECT_EQ(stats[1].count, 20);
}

TEST_F(CollTraceMicrotest, Utils_AggregateReportsOnlyTheRecordedCollectives) {
  std::deque<std::unique_ptr<CollTraceInfo>> info;
  info.push_back(std::make_unique<CollTraceInfo>(CollTraceInfo{1, "AllReduce", "ncclFloat32", 1}));
  std::vector<float> lat = {1, 1, 2, 2};

  EXPECT_EQ(aggregateResults(info, lat, 2, 2).size(), 1u);
}

TEST_F(CollTraceMicrotest, Utils_AggregateRejectsAZeroLatency) {
  std::deque<std::unique_ptr<CollTraceInfo>> info;
  std::vector<float> lat = {1, 1, 1, 0};

  EXPECT_THROW(aggregateResults(info, lat, 2, 2), std::runtime_error);
}

TEST_F(CollTraceMicrotest, Utils_ReportLogsOneLinePerCollective) {
  std::deque<std::vector<CollStats>> stats(2);
  stats[0].emplace_back(3, 25, 1.5f, 2.5f, "AllReduce", "ncclFloat64", 1 << 17);
  stats[1].emplace_back(4, 0, 1.0f, 1.0f, "Other", "ncclInt8", 1 << 20);

  reportToFile(stats, "hash");

  std::vector<std::string> lines = DebugLogMatching("coll_id");
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "coll_id 3, percent 25, min_latency_us 1.500000, max_latency_us 2.500000, op_name AllReduce, "
                      "data_type ncclFloat64, count 131072, message_size_MB 1.000000, comm_hash hash");
  EXPECT_NE(lines[1].find("coll_id 4, percent 0,"), std::string::npos);
}

TEST_F(CollTraceMicrotest, Event_WaitPollsUntilTheEventIsReady) {
  g_queryResults = {hipErrorNotReady, hipErrorNotReady};
  CudaWaitEvent event(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x10)));

  EXPECT_EQ(event.waitEventFinish(), ncclSuccess);

  EXPECT_EQ(g_queries, 3);
}

TEST_F(CollTraceMicrotest, Event_WaitSleepsTheCheckIntervalBetweenPolls) {
  SetParam("NCCL_COLLTRACE_CHECK_INTERVAL_MS", 20);
  g_queryResults = {hipErrorNotReady, hipErrorNotReady};
  CudaWaitEvent event(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x10)));
  auto start = std::chrono::steady_clock::now();

  EXPECT_EQ(event.waitEventFinish(), ncclSuccess);

  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(40));
}

TEST_F(CollTraceMicrotest, Event_WaitFailsOnAnyOtherError) {
  g_queryResults = {hipErrorNotReady, hipErrorInvalidValue};
  CudaWaitEvent event(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x10)));

  EXPECT_EQ(event.waitEventFinish(), ncclUnhandledCudaError);

  EXPECT_EQ(g_queries, 2);
}

TEST_F(CollTraceMicrotest, Event_ElapsedTimeIsMeasuredFromTheStartEvent) {
  g_elapsed = {2.5f};
  CudaWaitEvent start(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x10)));
  CudaWaitEvent stop(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x11)));

  std::shared_ptr<float> ms = stop.getElapsedTimeSinceEvent(&start);

  ASSERT_NE(ms, nullptr);
  EXPECT_FLOAT_EQ(*ms, 2.5f);
  g_elapsedResult = hipErrorInvalidValue;
  EXPECT_EQ(stop.getElapsedTimeSinceEvent(&start), nullptr);
  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "get elapsed time failed error: fake hip error"));
}

TEST_F(CollTraceMicrotest, Event_DestroysItsHipEvent) {
  { CudaWaitEvent event(CudaEventPtr(reinterpret_cast<hipEvent_t>(0x12))); }

  EXPECT_EQ(g_destroyedEvents, std::vector<hipEvent_t>{reinterpret_cast<hipEvent_t>(0x12)});
}

TEST_F(CollTraceMicrotest, EventQueue_IsFifoAndWaitPopBlocksForAPush) {
  EventQueue<CollTraceEvent> queue;
  queue.push(std::make_unique<CollTraceEvent>(CollTraceEvent::EventType::COMM));
  queue.push(std::make_unique<CollTraceEvent>(CollTraceEvent::EventType::TERMINATE));
  EXPECT_EQ(queue.waitPop()->eventType, CollTraceEvent::EventType::COMM);
  EXPECT_EQ(queue.waitPop()->eventType, CollTraceEvent::EventType::TERMINATE);

  std::atomic<bool> popped{false};
  std::thread consumer([&] {
    auto event = queue.waitPop();
    popped = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(popped);
  queue.push(std::make_unique<CollTraceEvent>());
  consumer.join();
  EXPECT_TRUE(popped);
}

TEST_F(CollTraceMicrotest, Event_DefaultsToACommEventAndMoves) {
  CollTraceEvent event;
  event.coll.count = 9;
  CollTraceEvent moved(std::move(event));
  EXPECT_EQ(moved.eventType, CollTraceEvent::EventType::COMM);
  EXPECT_EQ(moved.coll.count, 9);
  EXPECT_FLOAT_EQ(moved.coll.latencyMs, -1);
}

TEST_F(CollTraceMicrotest, Trace_WorkerRunsOnTheCommDeviceUntilDestroyed) {
  comm_->cudaDev = 3;
  auto trace = std::make_unique<CollTrace>(comm_.get());
  ASSERT_TRUE(WaitUntil([] { return DebugLogged(NCCL_LOG_INFO, "worker thread STARTED"); }));

  trace.reset();

  EXPECT_EQ(g_setDevices, std::vector<int>{3});
  EXPECT_TRUE(DebugLogged(NCCL_LOG_INFO, "COLLTRACE: commHash 2748 rank 0 - worker thread TERMINATE"));
  EXPECT_TRUE(DebugLogged(NCCL_LOG_INFO, "COLLTRACE: commHash 2748 rank 0 - Destroy COMPLETE"));
}

TEST_F(CollTraceMicrotest, Trace_WorkerExitsWhenTheDeviceCannotBeSet) {
  g_setDeviceResult = hipErrorInvalidDevice;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  trace.reset();

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "Cuda failure 'fake hip error'"));
  EXPECT_FALSE(DebugLogged(NCCL_LOG_INFO, "worker thread STARTED"));
}

TEST_F(CollTraceMicrotest, Trace_CreateEventMakesStartAndStopEvents) {
  CollTrace trace(comm_.get());

  auto event = trace.createEvent(CollTraceEvent::EventType::COMM);

  ASSERT_NE(event, nullptr);
  ASSERT_NE(event->start, nullptr);
  ASSERT_NE(event->stop, nullptr);
  EXPECT_NE(event->start->getCudaEvent(), nullptr);
  EXPECT_NE(event->start->getCudaEvent(), event->stop->getCudaEvent());
}

// The event is still handed out, with null HIP events; a fix flips this pin.
TEST_F(CollTraceMicrotest, Trace_KnownBug_EventCreateFailureIsOnlyLogged) {
  CollTrace trace(comm_.get());
  g_eventCreateResult = hipErrorOutOfMemory;

  auto event = trace.createEvent();

  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->start->getCudaEvent(), nullptr);
  EXPECT_TRUE(DebugLogged(NCCL_LOG_INFO, "Cuda failure 'fake hip error'"));
}

TEST_F(CollTraceMicrotest, Trace_RingBoundaryAllGathersAndReports) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 2);
  SetParam("NCCL_COLLTRACE_MAX_DUMP_SIZE", 1);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f, 1 << 18);
  Enqueue(trace.get(), 10.0f, 1 << 18);
  trace.reset();

  ASSERT_EQ(g_allGathers.size(), 1u);
  EXPECT_EQ(g_allGathers[0].ranks, localRankToRank_);
  EXPECT_EQ(g_allGathers[0].rank, 1);
  EXPECT_EQ(g_allGathers[0].nranks, 8);
  EXPECT_EQ(g_allGathers[0].size, (int)(2 * sizeof(float)));
  EXPECT_EQ(g_allGathers[0].mine, (std::vector<float>{2.0f, 10.0f}));
  std::vector<std::string> lines = DebugLogMatching("coll_id");
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "coll_id 0, percent 150, min_latency_us 2000.000000, max_latency_us 5000.000000, op_name "
                      "AllReduce, data_type ncclFloat32, count 262144, message_size_MB 1.000000, comm_hash 2748");
  EXPECT_NE(lines[1].find("coll_id 1, percent 100, min_latency_us 5000.000000, max_latency_us 10000.000000"),
            std::string::npos)
    << lines[1];
}

TEST_F(CollTraceMicrotest, Trace_RingBufferKeepsTheLastRecordMaxCollectives) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 2);
  SetParam("NCCL_COLLTRACE_MAX_DUMP_SIZE", 1);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  for (float ms : {1.0f, 2.0f, 3.0f, 4.0f, 6.0f}) Enqueue(trace.get(), ms);
  trace.reset();

  ASSERT_EQ(g_allGathers.size(), 2u);
  EXPECT_EQ(g_allGathers[1].mine, (std::vector<float>{3.0f, 4.0f}));
  std::vector<std::string> lines = DebugLogMatching("coll_id");
  ASSERT_EQ(lines.size(), 4u);
  EXPECT_EQ(lines[2].find("coll_id 2,"), 0u);
  EXPECT_EQ(lines[3].find("coll_id 3,"), 0u);
}

// Gated on the worker's CUDA device, not on the rank; a fix flips this pin.
TEST_F(CollTraceMicrotest, Trace_KnownBug_OnlyDeviceZeroAggregates) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  SetParam("NCCL_COLLTRACE_MAX_DUMP_SIZE", 1);
  comm_->cudaDev = 1;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  EXPECT_EQ(g_allGathers.size(), 1u);
  EXPECT_TRUE(DebugLogMatching("coll_id").empty());
}

TEST_F(CollTraceMicrotest, Trace_StatsWaitForTheIntervalOrDumpSize) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  SetParam("NCCL_COLLTRACE_MAX_DUMP_SIZE", 3);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  Enqueue(trace.get(), 2.0f);
  ASSERT_TRUE(WaitUntil([] { return AllGatherCount() == 2; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_TRUE(DebugLogMatching("coll_id").empty());
  Enqueue(trace.get(), 2.0f);
  ASSERT_TRUE(WaitUntil([] { return DebugLogMatching("coll_id").size() == 3; }));
  trace.reset();

  EXPECT_EQ(DebugLogMatching("coll_id").size(), 3u);
}

TEST_F(CollTraceMicrotest, Trace_ZeroIntervalReportsEveryAggregation) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  SetParam("NCCL_COLLTRACE_DUMP_INTERVAL_SEC", 0);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  ASSERT_TRUE(WaitUntil([] { return DebugLogMatching("coll_id").size() == 1; }));
  trace.reset();
}

TEST_F(CollTraceMicrotest, Trace_DestroyReportsWhatIsLeftOnRankZeroOnly) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  auto trace = std::make_unique<CollTrace>(comm_.get());
  Enqueue(trace.get(), 2.0f);
  trace.reset();
  EXPECT_EQ(DebugLogMatching("coll_id").size(), 1u);

  comm_->rank = 1;
  trace = std::make_unique<CollTrace>(comm_.get());
  Enqueue(trace.get(), 2.0f);
  trace.reset();
  EXPECT_EQ(DebugLogMatching("coll_id").size(), 1u);
}

TEST_F(CollTraceMicrotest, Trace_ZeroDumpSizeDropsEveryAggregation) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  SetParam("NCCL_COLLTRACE_MAX_DUMP_SIZE", 0);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  EXPECT_EQ(g_allGathers.size(), 1u);
  EXPECT_TRUE(DebugLogMatching("coll_id").empty());
  EXPECT_FALSE(DebugLogMatching("seconds passed since last report, stats size = 0").empty());
}

TEST_F(CollTraceMicrotest, Trace_AllGatherFailureSkipsTheAggregation) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  g_allGatherResult = ncclSystemError;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "CollTrace: All gather exchange latency data failed"));
  EXPECT_TRUE(DebugLogMatching("coll_id").empty());
}

TEST_F(CollTraceMicrotest, Trace_AggregationErrorIsLogged) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  g_peerLatency = 0;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "Aggregating error: CollTrace: latency value cannot be zero"));
}

// Aggregation reads eight ranks' slices, so the unfilled ones count as a failed all-gather; a fix flips this pin.
TEST_F(CollTraceMicrotest, Trace_KnownBug_FewerThanEightLocalRanksReportNothing) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  comm_->localRanks = 2;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  ASSERT_EQ(g_allGathers.size(), 1u);
  EXPECT_EQ(g_allGathers[0].nranks, 2);
  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "Aggregating error: CollTrace: latency value cannot be zero"));
  EXPECT_TRUE(DebugLogMatching("coll_id").empty());
}

TEST_F(CollTraceMicrotest, Trace_ReportErrorAtDestroyIsLogged) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f, 8, "ncclUint8");
  trace.reset();

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "COLLTRACE: commHash 2748 rank 0 - Destroy FAILED: CollTrace: unsupported data "
                                    "type ncclUint8"));
}

TEST_F(CollTraceMicrotest, Trace_FailedStopEventRecordsMinusOne) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  g_queryResults = {hipSuccess, hipErrorInvalidValue};  // start, stop
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  trace.reset();

  ASSERT_EQ(g_allGathers.size(), 1u);
  EXPECT_EQ(g_allGathers[0].mine, std::vector<float>{-1.0f});
}

TEST_F(CollTraceMicrotest, Trace_ElapsedTimeFailureStopsTheWorker) {
  SetParam("NCCL_COLLTRACE_RECORD_MAX", 1);
  g_elapsedResult = hipErrorInvalidValue;
  auto trace = std::make_unique<CollTrace>(comm_.get());

  Enqueue(trace.get(), 2.0f);
  Enqueue(trace.get(), 2.0f);
  trace.reset();

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "getElapsedTimeSinceEvent failed, aborting worker thread"));
  EXPECT_TRUE(g_allGathers.empty());
  EXPECT_FALSE(DebugLogged(NCCL_LOG_INFO, "worker thread TERMINATE"));
}

TEST_F(CollTraceMicrotest, Func_InitNeedsTheEnvSetToOne) {
  EXPECT_EQ(collTraceInit(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->ctrace, nullptr);
  g_env["RCCL_LATENCY_PROFILER"] = "0";
  EXPECT_EQ(collTraceInit(comm_.get()), ncclSuccess);
  EXPECT_EQ(comm_->ctrace, nullptr);
  EXPECT_TRUE(DebugLogged(NCCL_LOG_INFO, "RCCL_LATENCY_PROFILER set by environment to 0."));

  g_env["RCCL_LATENCY_PROFILER"] = "1";
  EXPECT_EQ(collTraceInit(comm_.get()), ncclSuccess);
  CollTrace* first = comm_->ctrace;
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(collTraceInit(comm_.get()), ncclSuccess);  // replaces the previous tracer
  EXPECT_NE(comm_->ctrace, nullptr);
  EXPECT_EQ(DebugLogMatching("Destroy COMPLETE").size(), 1u);
}

TEST_F(CollTraceMicrotest, Func_DestroyDeletesTheTracer) {
  EXPECT_EQ(collTraceDestroy(comm_.get()), ncclSuccess);
  comm_->ctrace = new CollTrace(comm_.get());

  EXPECT_EQ(collTraceDestroy(comm_.get()), ncclSuccess);

  EXPECT_EQ(comm_->ctrace, nullptr);
  EXPECT_EQ(DebugLogMatching("Destroy COMPLETE").size(), 1u);
}

TEST_F(CollTraceMicrotest, Func_RecordEventsOnlyWithATracerAndEvent) {
  auto stream = reinterpret_cast<hipStream_t>(0x57);
  auto owned = std::make_unique<CollTrace>(comm_.get());
  CollTrace* trace = owned.get();
  auto event = trace->createEvent();
  hipEvent_t start = event->start->getCudaEvent(), stop = event->stop->getCudaEvent();

  EXPECT_EQ(collTraceRecordStartEvent(comm_.get(), stream, event.get()), ncclSuccess);  // no tracer yet
  EXPECT_EQ(collTraceRecordEndEvent(comm_.get(), nullptr, stream, trace->createEvent()), ncclSuccess);
  EXPECT_TRUE(g_recordedEvents.empty());
  comm_->ctrace = owned.release();
  EXPECT_EQ(collTraceRecordStartEvent(comm_.get(), stream, nullptr), ncclSuccess);
  EXPECT_EQ(collTraceRecordEndEvent(comm_.get(), nullptr, stream, nullptr), ncclSuccess);
  EXPECT_TRUE(g_recordedEvents.empty());

  EXPECT_EQ(collTraceRecordStartEvent(comm_.get(), stream, event.get()), ncclSuccess);
  EXPECT_EQ(collTraceRecordEndEvent(comm_.get(), nullptr, stream, std::move(event)), ncclSuccess);

  ASSERT_EQ(g_recordedEvents.size(), 2u);
  EXPECT_EQ(g_recordedEvents[0], std::make_pair(start, stream));
  EXPECT_EQ(g_recordedEvents[1], std::make_pair(stop, stream));
  EXPECT_TRUE(WaitUntil([] { return QueryCount() >= 2; }));  // the worker took the event
}

TEST_F(CollTraceMicrotest, Func_RecordFailuresAreReturned) {
  comm_->ctrace = new CollTrace(comm_.get());
  auto event = comm_->ctrace->createEvent();
  g_recordResult = hipErrorInvalidValue;

  EXPECT_EQ(collTraceRecordStartEvent(comm_.get(), nullptr, event.get()), ncclUnhandledCudaError);
  EXPECT_EQ(collTraceRecordEndEvent(comm_.get(), nullptr, nullptr, std::move(event)), ncclUnhandledCudaError);
}

TEST_F(CollTraceMicrotest, Func_CollInfoComesFromTheTask) {
  ncclTaskColl task = {};
  task.func = ncclFuncAllReduce;
  task.datatype = ncclFloat64;
  task.count = 77;

  CollTraceInfo info = parseCollInfoFromCollTask(task);

  EXPECT_EQ(info.opName, "AllReduce");
  EXPECT_EQ(info.dataType, "ncclFloat64");
  EXPECT_EQ(info.count, 77);
}

class CollTraceAcquireMicrotest : public CollTraceMicrotest {
 protected:
  void SetUp() override {
    CollTraceMicrotest::SetUp();
    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->comm = comm_.get();
    task_.func = ncclFuncAllReduce;
    task_.datatype = ncclFloat32;
    task_.count = 5;
  }
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclTaskColl task_ = {};
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x57);
};

TEST_F(CollTraceAcquireMicrotest, NoTracerOrCommGivesNothing) {
  EXPECT_EQ(collTraceAquireEventBaseline(plan_.get(), stream_), nullptr);
  plan_->comm = nullptr;
  EXPECT_EQ(collTraceAquireEventBaseline(plan_.get(), stream_), nullptr);
  EXPECT_EQ(collTraceAquireEventCommon(comm_.get(), CollTraceEvent::EventType::COMM, stream_), nullptr);
  EXPECT_FALSE(DebugLogged(NCCL_LOG_WARN, "no coll task"));
}

TEST_F(CollTraceAcquireMicrotest, EmptyPlanGivesNothing) {
  comm_->ctrace = new CollTrace(comm_.get());

  EXPECT_EQ(collTraceAquireEventBaseline(plan_.get(), stream_), nullptr);

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "CollTrace: no coll task in this plan, this plan is empty"));
}

TEST_F(CollTraceAcquireMicrotest, EventCarriesTheFirstTasksInfo) {
  comm_->ctrace = new CollTrace(comm_.get());
  ncclTaskColl second = {};
  second.func = ncclFuncBroadcast;
  ncclIntruQueueEnqueue(&plan_->collTaskQueue, &task_);
  ncclIntruQueueEnqueue(&plan_->collTaskQueue, &second);

  auto event = collTraceAquireEventBaseline(plan_.get(), stream_);

  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->eventType, CollTraceEvent::EventType::COMM);
  EXPECT_EQ(event->coll.opName, "AllReduce");
  EXPECT_EQ(event->coll.dataType, "ncclFloat32");
  EXPECT_EQ(event->coll.count, 5);
  EXPECT_NE(event->start, nullptr);
  EXPECT_NE(event->stop, nullptr);
}

TEST_F(CollTraceAcquireMicrotest, GraphCaptureIsNotTraced) {
  comm_->ctrace = new CollTrace(comm_.get());
  ncclIntruQueueEnqueue(&plan_->collTaskQueue, &task_);
  g_capturing = true;

  EXPECT_EQ(collTraceAquireEventBaseline(plan_.get(), stream_), nullptr);

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "COLLTRACE: does not support cuda graph. Collectives from comm abc will be skipped"));
  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "COLLTRACE: failed to aquire event"));
}

TEST_F(CollTraceAcquireMicrotest, CaptureQueryFailureIsNotTraced) {
  comm_->ctrace = new CollTrace(comm_.get());
  ncclIntruQueueEnqueue(&plan_->collTaskQueue, &task_);
  g_capturingResult = ncclInternalError;

  EXPECT_EQ(collTraceAquireEventBaseline(plan_.get(), stream_), nullptr);

  EXPECT_TRUE(DebugLogged(NCCL_LOG_WARN, "Internal error: ncclCudaGetCapturingGraph failed by 3"));
}

}  // namespace
