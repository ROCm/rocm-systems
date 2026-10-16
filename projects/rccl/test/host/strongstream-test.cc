/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/strongstream.cc: CUDA-context tracking, strong-stream
// acquire/release across live and captured work, and the graph-capture helpers. The HIP stream,
// event and graph calls it makes are recording seams defined in this TU.

#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "nccl.h"
#include "debug.h"
#include "checks.h"
#include "param.h"
#include "rocmwrap.h"
#include "strongstream.h"

// --- Logging: this binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;

bool Logged(ncclDebugLogLevel level, const char* needle) {
  for (const auto& m : g_logged) {
    if (m.level == level && m.text.find(needle) != std::string::npos) return true;
  }
  return false;
}

std::string g_paramEnv;
int64_t g_paramDefault = 0;
bool g_raceFatalSet = false;
int64_t g_raceFatal = 0;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

// Never fills the cache, so each ncclParamLaunchRaceFatal() call reads the current setting.
int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t, int64_t*, int8_t*) {
  g_paramEnv = env;
  g_paramDefault = deftVal;
  return g_raceFatalSet ? g_raceFatal : deftVal;
}

// --- HIP seams. Handles are opaque counters; every call is recorded and any one can be failed.
namespace {
struct Call {
  std::string fn;
  const void* a = nullptr;
  const void* b = nullptr;
  const void* c = nullptr;
  uint64_t n = 0;
  uint64_t flags = 0;
};
std::vector<Call> g_calls;
std::map<std::string, int> g_callCount;
std::map<std::string, int> g_failAt;  // fn -> 1-based call that returns an error
std::function<void(const std::string&)> g_onCall;

struct CaptureInfo {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long id = 0;
  hipGraph_t graph = nullptr;
  std::vector<hipGraphNode_t> deps;
};
std::map<hipStream_t, CaptureInfo> g_capture;
CaptureInfo g_defaultCapture;  // for streams created inside the call under test
std::set<const void*> g_liveStreams, g_liveEvents;
uintptr_t g_nextHandle = 0;
int g_device = 0;

template <class T>
T NewHandle(uintptr_t base) {
  return reinterpret_cast<T>(base + ++g_nextHandle);
}

// Fails the nth call to `fn` from now on.
void FailNext(const std::string& fn, int nth = 1) { g_failAt = {{fn, g_callCount[fn] + nth}}; }

hipError_t Hit(Call call) {
  std::string fn = call.fn;
  g_calls.push_back(std::move(call));
  if (g_onCall) g_onCall(fn);
  auto it = g_failAt.find(fn);
  return (++g_callCount[fn] == (it == g_failAt.end() ? 0 : it->second)) ? hipErrorInvalidValue : hipSuccess;
}

const CaptureInfo& InfoFor(hipStream_t s) {
  auto it = g_capture.find(s);
  return it == g_capture.end() ? g_defaultCapture : it->second;
}
}  // namespace

hipError_t hipGetDevice(int* device) {
  hipError_t err = Hit({"hipGetDevice"});
  if (err == hipSuccess) *device = g_device;
  return err;
}
hipError_t hipGetLastError() { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  hipStream_t s = NewHandle<hipStream_t>(0x10000);
  hipError_t err = Hit({"hipStreamCreateWithFlags", s, nullptr, nullptr, 0, flags});
  if (err == hipSuccess) {
    *stream = s;
    g_liveStreams.insert(s);
  }
  return err;
}
hipError_t hipStreamDestroy(hipStream_t stream) {
  g_liveStreams.erase(stream);
  return Hit({"hipStreamDestroy", stream});
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  hipEvent_t e = NewHandle<hipEvent_t>(0x20000);
  hipError_t err = Hit({"hipEventCreateWithFlags", e, nullptr, nullptr, 0, flags});
  if (err == hipSuccess) {
    *event = e;
    g_liveEvents.insert(e);
  }
  return err;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  g_liveEvents.erase(event);
  return Hit({"hipEventDestroy", event});
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) { return Hit({"hipEventRecord", event, stream}); }
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  return Hit({"hipStreamWaitEvent", stream, event, nullptr, 0, flags});
}
hipError_t hipStreamSynchronize(hipStream_t stream) { return Hit({"hipStreamSynchronize", stream}); }
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* status) {
  *status = InfoFor(stream).status;
  return Hit({"hipStreamIsCapturing", stream});
}
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* status, unsigned long long* id,
                                      hipGraph_t* graph, const hipGraphNode_t** deps, size_t* numDeps) {
  const CaptureInfo& info = InfoFor(stream);
  *status = info.status;
  if (id) *id = info.id;
  if (graph) *graph = info.graph;
  if (deps) *deps = info.deps.data();
  if (numDeps) *numDeps = info.deps.size();
  return Hit({"hipStreamGetCaptureInfo_v2", stream});
}
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* deps, size_t numDeps,
                                              unsigned int flags) {
  return Hit({"hipStreamUpdateCaptureDependencies", stream, deps, numDeps ? deps[0] : nullptr, numDeps, flags});
}
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t*,
                                      size_t numDeps, hipEvent_t event) {
  *node = NewHandle<hipGraphNode_t>(0x30000);
  return Hit({"hipGraphAddEventRecordNode", graph, event, *node, numDeps});
}
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to,
                                   size_t numDeps) {
  return Hit({"hipGraphAddDependencies", *from, *to, graph, numDeps});
}
hipError_t hipUserObjectCreate(hipUserObject_t* object, void* ptr, hipHostFn_t destroy, unsigned int refcount,
                               unsigned int flags) {
  *object = NewHandle<hipUserObject_t>(0x40000);
  return Hit({"hipUserObjectCreate", ptr, reinterpret_cast<const void*>(destroy), *object, refcount, flags});
}
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags) {
  return Hit({"hipGraphRetainUserObject", graph, object, nullptr, count, flags});
}

// calloc can be failed; free is counted so dropped capture entries are seen to be released.
namespace {
bool g_failCalloc = false;
std::vector<void*> g_callocs;
int g_frees = 0;
void* test_calloc(size_t n, size_t size) {
  if (g_failCalloc) return nullptr;
  void* p = std::calloc(n, size);
  g_callocs.push_back(p);
  return p;
}
void test_free(void* p) {
  ++g_frees;
  std::free(p);
}
}  // namespace

#define calloc test_calloc
#define free test_free
#include STRONGSTREAM_CC_PATH
#undef calloc
#undef free

namespace {

const hipStream_t kOrigin = reinterpret_cast<hipStream_t>(0x777);
const hipGraph_t kGraph = reinterpret_cast<hipGraph_t>(0x888);

int Count(const std::string& fn) {
  int n = 0;
  for (const auto& c : g_calls) n += c.fn == fn;
  return n;
}
std::vector<Call> CallsTo(const std::string& fn) {
  std::vector<Call> out;
  for (const auto& c : g_calls) {
    if (c.fn == fn) out.push_back(c);
  }
  return out;
}

// True when another thread cannot take `m` right now.
bool HeldElsewhere(std::mutex& m) {
  bool got = false;
  std::thread([&] {
    got = m.try_lock();
    if (got) m.unlock();
  }).join();
  return !got;
}

// Runs `fn` while `m` is held by this thread: true when fn ran to completion without the lock.
template <class F>
bool CompletesWhileLocked(std::mutex& m, F fn) {
  m.lock();
  auto done = std::async(std::launch::async, fn);
  bool finished = done.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
  m.unlock();
  done.get();
  return finished;
}

ncclCudaGraph Live(int mode = 2) { return ncclCudaGraphNone(mode); }
ncclCudaGraph Captured(unsigned long long id, int mode = 2) {
  ncclCudaGraph g;
  g.origin = kOrigin;
  g.graph = kGraph;
  g.graphId = id;
  g.graphUsageMode = mode;
  return g;
}

class StrongstreamMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_logged.clear();
    g_calls.clear();
    g_callCount.clear();
    g_failAt.clear();
    g_onCall = nullptr;
    g_capture.clear();
    g_defaultCapture = CaptureInfo{};
    g_liveStreams.clear();
    g_liveEvents.clear();
    g_device = 0;
    g_raceFatalSet = false;
    g_failCalloc = false;
    g_callocs.clear();
    g_frees = 0;
  }
  void TearDown() override { g_onCall = nullptr; }

  void Construct(ncclStrongStream* s) {
    ASSERT_EQ(ncclStrongStreamConstruct(s), ncclSuccess);
    g_calls.clear();
  }
  // Acquire for graph `id` and mark its capture stream as actively capturing.
  hipStream_t AcquireCapture(ncclStrongStream* s, unsigned long long id, int mode = 2) {
    hipStream_t work = nullptr;
    EXPECT_EQ(ncclStrongStreamAcquire(Captured(id, mode), s, false, &work), ncclSuccess);
    g_capture[work].status = hipStreamCaptureStatusActive;
    return work;
  }

  ncclStrongStream ss_;
};

// Context tracking

TEST_F(StrongstreamMicrotest, ContextTrack_FirstUseCreatesAContextWithItsLaunchOrderStream) {
  g_device = 3;
  ncclCudaContext* cxt = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&cxt, 1, 0x42), ncclSuccess);
  ASSERT_NE(cxt, nullptr);
  EXPECT_EQ(cxt->hcontext, 3);
  EXPECT_EQ(cxt->refCount, 1);
  EXPECT_TRUE(cxt->launchOrderImplicitEverEnabled);
  EXPECT_FALSE(cxt->launchOrderImplicitEverDisabled);
  EXPECT_EQ(Count("hipStreamCreateWithFlags"), 1);
  EXPECT_EQ(cxt->launchOrder.liveStream, CallsTo("hipStreamCreateWithFlags")[0].a);
  ncclCudaContextDrop(cxt);
  EXPECT_TRUE(g_liveStreams.empty());
  EXPECT_TRUE(g_liveEvents.empty());
}

// Only launchOrderImplicit == 1 enables it; any other value counts as disabled.
TEST_F(StrongstreamMicrotest, ContextTrack_OnlyOneEnablesImplicitLaunchOrder) {
  ncclCudaContext* cxt = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&cxt, 2, 0), ncclSuccess);
  EXPECT_FALSE(cxt->launchOrderImplicitEverEnabled);
  EXPECT_TRUE(cxt->launchOrderImplicitEverDisabled);
  ncclCudaContext* again = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&again, 1, 0), ncclSuccess);
  EXPECT_TRUE(cxt->launchOrderImplicitEverEnabled);
  ncclCudaContextDrop(again);
  ncclCudaContextDrop(cxt);
}

TEST_F(StrongstreamMicrotest, ContextTrack_SameDeviceSharesTheContext) {
  ncclCudaContext* a = nullptr;
  ncclCudaContext* b = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&a, 0, 0), ncclSuccess);
  ASSERT_EQ(ncclCudaContextTrack(&b, 0, 0), ncclSuccess);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a->refCount, 2);
  EXPECT_EQ(Count("hipStreamCreateWithFlags"), 1);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "opposite value"));

  ncclCudaContextDrop(a);
  EXPECT_EQ(b->refCount, 1);
  EXPECT_EQ(Count("hipStreamDestroy"), 0);
  ncclCudaContextDrop(b);
  EXPECT_EQ(Count("hipStreamDestroy"), 1);
}

TEST_F(StrongstreamMicrotest, ContextTrack_MixingLaunchOrderSettingsIsReported) {
  ncclCudaContext* a = nullptr;
  ncclCudaContext* b = nullptr;
  ncclCudaContext* c = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&a, 1, 0), ncclSuccess);
  ASSERT_EQ(ncclCudaContextTrack(&b, 0, 0xabc), ncclSuccess);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "commHash 0xabc uses config launchOrderImplicit=0"));
  EXPECT_TRUE(a->launchOrderImplicitEverEnabled);
  EXPECT_TRUE(a->launchOrderImplicitEverDisabled);

  g_logged.clear();
  ASSERT_EQ(ncclCudaContextTrack(&c, 1, 0xdef), ncclSuccess);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "commHash 0xdef uses config launchOrderImplicit=1"));
  ncclCudaContextDrop(a);
  ncclCudaContextDrop(b);
  ncclCudaContextDrop(c);
}

TEST_F(StrongstreamMicrotest, ContextTrack_SettingsReportedOnlyAgainstTheOppositeValue) {
  ncclCudaContext* a = nullptr;
  ncclCudaContext* b = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&a, 1, 0), ncclSuccess);
  ASSERT_EQ(ncclCudaContextTrack(&b, 1, 0), ncclSuccess);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "opposite value"));
  EXPECT_FALSE(a->launchOrderImplicitEverDisabled);
  ncclCudaContextDrop(a);
  ncclCudaContextDrop(b);
}

// Contexts for three devices; the middle one is found and unlinked without disturbing the others.
TEST_F(StrongstreamMicrotest, ContextTrack_DistinctDevicesKeepSeparateContexts) {
  ncclCudaContext* cxt[3];
  for (int d = 0; d < 3; d++) {
    g_device = d;
    ASSERT_EQ(ncclCudaContextTrack(&cxt[d], 0, 0), ncclSuccess);
  }
  EXPECT_NE(cxt[0], cxt[1]);
  EXPECT_NE(cxt[1], cxt[2]);

  ncclCudaContextDrop(cxt[1]);
  for (int d : {0, 2}) {
    g_device = d;
    ncclCudaContext* again = nullptr;
    ASSERT_EQ(ncclCudaContextTrack(&again, 0, 0), ncclSuccess);
    EXPECT_EQ(again, cxt[d]);
    ncclCudaContextDrop(again);
  }
  g_device = 1;
  ncclCudaContext* fresh = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&fresh, 0, 0), ncclSuccess);
  EXPECT_EQ(fresh->refCount, 1);
  ncclCudaContextDrop(fresh);
  ncclCudaContextDrop(cxt[0]);
  ncclCudaContextDrop(cxt[2]);
  EXPECT_TRUE(g_liveStreams.empty());
}

TEST_F(StrongstreamMicrotest, ContextTrack_GetDeviceFailureIsAHipError) {
  FailNext("hipGetDevice");
  ncclCudaContext* cxt = reinterpret_cast<ncclCudaContext*>(0x1);
  EXPECT_EQ(ncclCudaContextTrack(&cxt, 0, 0), ncclUnhandledCudaError);
  EXPECT_EQ(cxt, reinterpret_cast<ncclCudaContext*>(0x1));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "HIP failure"));
}

// Observed: a failed launch-order stream construction is logged but still reported as success,
// and the half-built context stays tracked.
TEST_F(StrongstreamMicrotest, ContextTrack_ConstructFailureIsSwallowed) {
  FailNext("hipStreamCreateWithFlags");
  ncclCudaContext* cxt = nullptr;
  EXPECT_EQ(ncclCudaContextTrack(&cxt, 0, 0), ncclSuccess);
  ASSERT_NE(cxt, nullptr);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "HIP failure"));
  ncclCudaContext* again = nullptr;
  ASSERT_EQ(ncclCudaContextTrack(&again, 0, 0), ncclSuccess);
  EXPECT_EQ(again, cxt);
  ncclCudaContextDrop(again);
  ncclCudaContextDrop(cxt);
}

// Graph capture helpers

TEST_F(StrongstreamMicrotest, GetCapturingGraph_NotCapturingIsNoGraph) {
  ncclCudaGraph g = Captured(5);
  ASSERT_EQ(ncclCudaGetCapturingGraph(&g, kOrigin, 1), ncclSuccess);
  EXPECT_EQ(g.origin, nullptr);
  EXPECT_EQ(g.graph, nullptr);
  EXPECT_EQ(g.graphId, ULLONG_MAX);
  EXPECT_EQ(g.graphUsageMode, 1);
  EXPECT_EQ(CallsTo("hipStreamGetCaptureInfo_v2")[0].a, kOrigin);
}

TEST_F(StrongstreamMicrotest, GetCapturingGraph_ActiveCaptureReturnsItsGraph) {
  g_capture[kOrigin] = {hipStreamCaptureStatusActive, 9, kGraph, {}};
  ncclCudaGraph g = Live();
  ASSERT_EQ(ncclCudaGetCapturingGraph(&g, kOrigin, 2), ncclSuccess);
  EXPECT_EQ(g.origin, kOrigin);
  EXPECT_EQ(g.graph, kGraph);
  EXPECT_EQ(g.graphId, 9u);
  EXPECT_EQ(g.graphUsageMode, 2);
}

// An invalidated capture is not a graph to capture into.
TEST_F(StrongstreamMicrotest, GetCapturingGraph_InvalidatedCaptureIsNoGraph) {
  g_capture[kOrigin] = {hipStreamCaptureStatusInvalidated, 9, kGraph, {}};
  ncclCudaGraph g = Live();
  ASSERT_EQ(ncclCudaGetCapturingGraph(&g, kOrigin, 0), ncclSuccess);
  EXPECT_FALSE(ncclCudaGraphValid(g));
}

TEST_F(StrongstreamMicrotest, GetCapturingGraph_QueryFailureIsAHipError) {
  FailNext("hipStreamGetCaptureInfo_v2");
  ncclCudaGraph g = Live();
  EXPECT_EQ(ncclCudaGetCapturingGraph(&g, kOrigin, 0), ncclUnhandledCudaError);
}

void DestructorFn(void*) {}

// The user object is handed to the graph: created with one reference that the graph then takes.
TEST_F(StrongstreamMicrotest, GraphAddDestructor_MovesAUserObjectIntoTheGraph) {
  int arg;
  ASSERT_EQ(ncclCudaGraphAddDestructor(Captured(1), DestructorFn, &arg), ncclSuccess);
  auto create = CallsTo("hipUserObjectCreate");
  ASSERT_EQ(create.size(), 1u);
  EXPECT_EQ(create[0].a, &arg);
  EXPECT_EQ(create[0].b, reinterpret_cast<const void*>(&DestructorFn));
  EXPECT_EQ(create[0].n, 1u);
  EXPECT_EQ(create[0].flags, (uint64_t)hipUserObjectNoDestructorSync);
  auto retain = CallsTo("hipGraphRetainUserObject");
  ASSERT_EQ(retain.size(), 1u);
  EXPECT_EQ(retain[0].a, kGraph);
  EXPECT_EQ(retain[0].b, create[0].c);
  EXPECT_EQ(retain[0].n, 1u);
  EXPECT_EQ(retain[0].flags, (uint64_t)hipGraphUserObjectMove);
}

TEST_F(StrongstreamMicrotest, GraphAddDestructor_HipFailuresPropagate) {
  for (const char* fn : {"hipUserObjectCreate", "hipGraphRetainUserObject"}) {
    SCOPED_TRACE(fn);
    FailNext(fn);
    EXPECT_EQ(ncclCudaGraphAddDestructor(Captured(1), DestructorFn, nullptr), ncclUnhandledCudaError);
  }
  EXPECT_EQ(Count("hipGraphRetainUserObject"), 1);
}

// Construct / destruct / synchronize

TEST_F(StrongstreamMicrotest, Construct_CreatesANonBlockingStreamAndAnUntimedEvent) {
  ss_.everCaptured = true;
  ss_.graphOriginCaptured = true;
  ss_.captureHead = reinterpret_cast<ncclStrongStreamCapture*>(0x1);
  ASSERT_EQ(ncclStrongStreamConstruct(&ss_), ncclSuccess);
  auto streams = CallsTo("hipStreamCreateWithFlags");
  ASSERT_EQ(streams.size(), 1u);
  EXPECT_EQ(streams[0].flags, (uint64_t)hipStreamNonBlocking);
  EXPECT_EQ(ss_.liveStream, streams[0].a);
  auto events = CallsTo("hipEventCreateWithFlags");
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].flags, (uint64_t)hipEventDisableTiming);
  EXPECT_EQ(ss_.serialEvent, events[0].a);
  EXPECT_FALSE(ss_.everCaptured);
  EXPECT_FALSE(ss_.graphOriginCaptured);
  EXPECT_EQ(ss_.captureHead, nullptr);
  ASSERT_EQ(ncclStrongStreamDestruct(&ss_), ncclSuccess);
}

TEST_F(StrongstreamMicrotest, Construct_HipFailuresPropagate) {
  FailNext("hipStreamCreateWithFlags");
  EXPECT_EQ(ncclStrongStreamConstruct(&ss_), ncclUnhandledCudaError);
  EXPECT_EQ(Count("hipEventCreateWithFlags"), 0);
  FailNext("hipEventCreateWithFlags");
  EXPECT_EQ(ncclStrongStreamConstruct(&ss_), ncclUnhandledCudaError);
}

TEST_F(StrongstreamMicrotest, Destruct_ReleasesEveryCaptureStream) {
  Construct(&ss_);
  AcquireCapture(&ss_, 1);
  AcquireCapture(&ss_, 2);
  ASSERT_EQ(g_liveStreams.size(), 3u);
  g_frees = 0;
  ASSERT_EQ(ncclStrongStreamDestruct(&ss_), ncclSuccess);
  EXPECT_TRUE(g_liveStreams.empty());
  EXPECT_TRUE(g_liveEvents.empty());
  EXPECT_EQ(g_frees, 2);
}

TEST_F(StrongstreamMicrotest, Destruct_HipFailuresPropagate) {
  for (int failing : {1, 2}) {  // live stream, then the capture stream
    SCOPED_TRACE(failing);
    Construct(&ss_);
    AcquireCapture(&ss_, 1);
    g_calls.clear();
    FailNext("hipStreamDestroy", failing);
    EXPECT_EQ(ncclStrongStreamDestruct(&ss_), ncclUnhandledCudaError);
    EXPECT_EQ(Count("hipEventDestroy"), 0);
    std::free(ss_.captureHead);  // the early return skips the entry's free
    g_failAt.clear();
  }
  Construct(&ss_);
  FailNext("hipEventDestroy");
  EXPECT_EQ(ncclStrongStreamDestruct(&ss_), ncclUnhandledCudaError);
}

TEST_F(StrongstreamMicrotest, Synchronize_WaitsOnSerialEventThenSyncsTheLiveStream) {
  Construct(&ss_);
  ASSERT_EQ(ncclStrongStreamSynchronize(&ss_), ncclSuccess);
  ASSERT_EQ(g_calls.size(), 2u);
  EXPECT_EQ(g_calls[0].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[0].a, ss_.liveStream);
  EXPECT_EQ(g_calls[0].b, ss_.serialEvent);
  EXPECT_EQ(g_calls[1].fn, "hipStreamSynchronize");
  EXPECT_EQ(g_calls[1].a, ss_.liveStream);

  FailNext("hipStreamWaitEvent");
  EXPECT_EQ(ncclStrongStreamSynchronize(&ss_), ncclUnhandledCudaError);
  FailNext("hipStreamSynchronize");
  EXPECT_EQ(ncclStrongStreamSynchronize(&ss_), ncclUnhandledCudaError);
  ncclStrongStreamDestruct(&ss_);
}

// Acquire / release of live work

TEST_F(StrongstreamMicrotest, AcquireLive_ReturnsTheLiveStream) {
  Construct(&ss_);
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Live(), &ss_, false, &work), ncclSuccess);
  EXPECT_EQ(work, ss_.liveStream);
  EXPECT_EQ(ss_.liveAcquiredBy, localThreadId());
  EXPECT_TRUE(g_calls.empty());  // never captured: nothing to order against
  ASSERT_EQ(ncclStrongStreamAcquiredWorkStream(Live(), &ss_, false, &work), ncclSuccess);
  EXPECT_EQ(work, ss_.liveStream);
  ncclStrongStreamDestruct(&ss_);
}

// Once a graph has captured the stream, live work in mixing mode is ordered after it.
TEST_F(StrongstreamMicrotest, AcquireLive_AfterACaptureWaitsOnTheSerialEventWhenMixing) {
  Construct(&ss_);
  ss_.everCaptured = true;
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Live(2), &ss_, false, &work), ncclSuccess);
  auto waits = CallsTo("hipStreamWaitEvent");
  ASSERT_EQ(waits.size(), 1u);
  EXPECT_EQ(waits[0].a, ss_.liveStream);
  EXPECT_EQ(waits[0].b, ss_.serialEvent);

  g_calls.clear();
  ASSERT_EQ(ncclStrongStreamAcquire(Live(1), &ss_, false, &work), ncclSuccess);
  EXPECT_TRUE(g_calls.empty());

  FailNext("hipStreamWaitEvent");
  EXPECT_EQ(ncclStrongStreamAcquire(Live(2), &ss_, false, &work), ncclUnhandledCudaError);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, ReleaseLive_RecordsTheSerialEventOnlyAfterACapture) {
  Construct(&ss_);
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Live(), &ss_, false, &work), ncclSuccess);
  ASSERT_EQ(ncclStrongStreamRelease(Live(), &ss_, false), ncclSuccess);
  EXPECT_TRUE(g_calls.empty());

  ss_.everCaptured = true;
  ASSERT_EQ(ncclStrongStreamRelease(Live(), &ss_, false), ncclSuccess);
  auto records = CallsTo("hipEventRecord");
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].a, ss_.serialEvent);
  EXPECT_EQ(records[0].b, ss_.liveStream);

  FailNext("hipEventRecord");
  EXPECT_EQ(ncclStrongStreamRelease(Live(), &ss_, false), ncclUnhandledCudaError);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, Release_WithoutMixingDoesNothing) {
  Construct(&ss_);
  ss_.everCaptured = true;
  ss_.liveAcquiredBy = nullptr;  // would be a race in mixing mode
  EXPECT_EQ(ncclStrongStreamRelease(Live(1), &ss_, false), ncclSuccess);
  EXPECT_EQ(ncclStrongStreamRelease(Captured(4, 0), &ss_, false), ncclSuccess);
  EXPECT_TRUE(g_calls.empty());
  ncclStrongStreamDestruct(&ss_);
}

// Release from a thread other than the acquirer is fatal by default (NCCL_LAUNCH_RACE_FATAL=1).
TEST_F(StrongstreamMicrotest, ReleaseLive_FromAnotherThreadIsALaunchRace) {
  Construct(&ss_);
  std::thread([&] {
    hipStream_t work;
    ASSERT_EQ(ncclStrongStreamAcquire(Live(), &ss_, false, &work), ncclSuccess);
  }).join();
  EXPECT_EQ(ncclStrongStreamRelease(Live(), &ss_, false), ncclInvalidUsage);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "host threads racing to launch NCCL on same device"));
  EXPECT_EQ(g_paramEnv, "NCCL_LAUNCH_RACE_FATAL");
  EXPECT_EQ(g_paramDefault, 1);

  g_raceFatalSet = true;
  g_raceFatal = 0;
  EXPECT_EQ(ncclStrongStreamRelease(Live(), &ss_, false), ncclSuccess);
  ncclStrongStreamDestruct(&ss_);
}

// Acquire / release of captured work

TEST_F(StrongstreamMicrotest, AcquireCapture_JoinsTheGraphOnANewCaptureStream) {
  Construct(&ss_);
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Captured(7), &ss_, false, &work), ncclSuccess);
  EXPECT_TRUE(ss_.everCaptured);
  ASSERT_NE(ss_.captureHead, nullptr);
  EXPECT_EQ(ss_.captureHead->graphId, 7u);
  EXPECT_EQ(ss_.captureHead->captureStream, work);
  EXPECT_EQ(ss_.captureHead->acquiredBy, localThreadId());
  EXPECT_EQ(ss_.captureHead->next, nullptr);
  EXPECT_NE(work, ss_.liveStream);

  // The capture stream enters the graph via an event recorded on the origin, then drops that
  // dependency; on first capture the live stream's position is published and waited on.
  ASSERT_EQ(g_calls.size(), 8u);
  EXPECT_EQ(g_calls[0].fn, "hipStreamCreateWithFlags");
  EXPECT_EQ(g_calls[0].flags, (uint64_t)hipStreamNonBlocking);
  EXPECT_EQ(g_calls[1].fn, "hipEventCreateWithFlags");
  EXPECT_EQ(g_calls[1].flags, (uint64_t)hipEventDisableTiming);
  const void* scratch = g_calls[1].a;
  EXPECT_EQ(g_calls[2].fn, "hipEventRecord");
  EXPECT_EQ(g_calls[2].a, scratch);
  EXPECT_EQ(g_calls[2].b, kOrigin);
  EXPECT_EQ(g_calls[3].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[3].a, work);
  EXPECT_EQ(g_calls[3].b, scratch);
  EXPECT_EQ(g_calls[4].fn, "hipEventDestroy");
  EXPECT_EQ(g_calls[4].a, scratch);
  EXPECT_EQ(g_calls[5].fn, "hipStreamUpdateCaptureDependencies");
  EXPECT_EQ(g_calls[5].a, work);
  EXPECT_EQ(g_calls[5].n, 0u);
  EXPECT_EQ(g_calls[5].flags, (uint64_t)hipStreamSetCaptureDependencies);
  EXPECT_EQ(g_calls[6].fn, "hipEventRecord");
  EXPECT_EQ(g_calls[6].a, ss_.serialEvent);
  EXPECT_EQ(g_calls[6].b, ss_.liveStream);
  EXPECT_EQ(g_calls[7].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[7].a, work);
  EXPECT_EQ(g_calls[7].b, ss_.serialEvent);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, AcquireCapture_SerialEventIsPublishedOnlyOnTheFirstCapture) {
  Construct(&ss_);
  AcquireCapture(&ss_, 1);
  g_calls.clear();
  AcquireCapture(&ss_, 2);
  auto records = CallsTo("hipEventRecord");
  ASSERT_EQ(records.size(), 1u);
  EXPECT_NE(records[0].a, ss_.serialEvent);
  EXPECT_EQ(CallsTo("hipStreamWaitEvent").back().b, ss_.serialEvent);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, AcquireCapture_WithoutMixingSkipsTheSerialEvent) {
  Construct(&ss_);
  AcquireCapture(&ss_, 1, /*mode=*/1);
  EXPECT_TRUE(ss_.everCaptured);
  for (const auto& c : g_calls) {
    EXPECT_NE(c.a, ss_.serialEvent) << c.fn;
    EXPECT_NE(c.b, ss_.serialEvent) << c.fn;
  }
  EXPECT_EQ(Count("hipEventRecord"), 1);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, AcquireCapture_SameGraphReusesItsCaptureStream) {
  Construct(&ss_);
  hipStream_t first = AcquireCapture(&ss_, 1);
  hipStream_t second = AcquireCapture(&ss_, 2);
  g_calls.clear();

  hipStream_t work = nullptr;
  std::thread([&] { ASSERT_EQ(ncclStrongStreamAcquire(Captured(1), &ss_, false, &work), ncclSuccess); }).join();
  EXPECT_EQ(work, first);
  EXPECT_EQ(ss_.captureHead->captureStream, second);
  EXPECT_NE(ss_.captureHead->next->acquiredBy, localThreadId());  // re-acquired by the other thread
  EXPECT_EQ(CallsTo("hipStreamIsCapturing").size(), 1u);
  EXPECT_EQ(Count("hipStreamCreateWithFlags"), 0);

  ASSERT_EQ(ncclStrongStreamAcquiredWorkStream(Captured(1), &ss_, false, &work), ncclSuccess);
  EXPECT_EQ(work, first);
  ncclStrongStreamDestruct(&ss_);
}

// Entries whose capture ended are dropped; one is kept as a spare for the new graph.
TEST_F(StrongstreamMicrotest, AcquireCapture_EndedCapturesAreRecycled) {
  Construct(&ss_);
  hipStream_t a = AcquireCapture(&ss_, 1);
  hipStream_t b = AcquireCapture(&ss_, 2);
  hipStream_t c = AcquireCapture(&ss_, 3);
  g_capture[c].status = hipStreamCaptureStatusNone;
  g_capture[a].status = hipStreamCaptureStatusInvalidated;
  g_calls.clear();
  g_frees = 0;

  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Captured(4), &ss_, false, &work), ncclSuccess);
  EXPECT_EQ(work, c);
  EXPECT_EQ(Count("hipStreamCreateWithFlags"), 0);
  auto destroyed = CallsTo("hipStreamDestroy");
  ASSERT_EQ(destroyed.size(), 1u);
  EXPECT_EQ(destroyed[0].a, a);
  EXPECT_EQ(g_frees, 1);

  ASSERT_EQ(ss_.captureHead->graphId, 4u);
  ASSERT_EQ(ss_.captureHead->next->graphId, 2u);
  EXPECT_EQ(ss_.captureHead->next->captureStream, b);
  EXPECT_EQ(ss_.captureHead->next->next, nullptr);
  ncclStrongStreamDestruct(&ss_);
}

// A failed destroy of a surplus ended capture is only logged.
TEST_F(StrongstreamMicrotest, AcquireCapture_DestroyFailureOfAnEndedCaptureIsIgnored) {
  Construct(&ss_);
  hipStream_t a = AcquireCapture(&ss_, 1);
  hipStream_t b = AcquireCapture(&ss_, 2);
  g_capture[a].status = hipStreamCaptureStatusNone;
  g_capture[b].status = hipStreamCaptureStatusNone;
  FailNext("hipStreamDestroy");
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Captured(3), &ss_, false, &work), ncclSuccess);
  EXPECT_EQ(work, b);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Cuda failure"));
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, AcquireCapture_CaptureQueryFailureLeavesTheListAlone) {
  Construct(&ss_);
  AcquireCapture(&ss_, 1);
  ncclStrongStreamCapture* head = ss_.captureHead;
  FailNext("hipStreamIsCapturing");
  hipStream_t work = nullptr;
  EXPECT_EQ(ncclStrongStreamAcquire(Captured(2), &ss_, true, &work), ncclUnhandledCudaError);
  EXPECT_EQ(work, nullptr);
  EXPECT_EQ(ss_.captureHead, head);
  EXPECT_FALSE(HeldElsewhere(ss_.mutex));
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, AcquireCapture_AllocationFailureIsASystemError) {
  Construct(&ss_);
  g_failCalloc = true;
  hipStream_t work = nullptr;
  EXPECT_EQ(ncclStrongStreamAcquire(Captured(1), &ss_, true, &work), ncclSystemError);
  EXPECT_EQ(ss_.captureHead, nullptr);
  EXPECT_EQ(work, nullptr);
  EXPECT_FALSE(HeldElsewhere(ss_.mutex));
  ncclStrongStreamDestruct(&ss_);
}

// Observed: the calloc'd entry is not freed when its capture stream cannot be created.
TEST_F(StrongstreamMicrotest, AcquireCapture_CaptureStreamCreateFailurePropagates) {
  Construct(&ss_);
  FailNext("hipStreamCreateWithFlags");
  hipStream_t work = nullptr;
  EXPECT_EQ(ncclStrongStreamAcquire(Captured(1), &ss_, false, &work), ncclUnhandledCudaError);
  EXPECT_EQ(ss_.captureHead, nullptr);
  ASSERT_EQ(g_callocs.size(), 1u);
  EXPECT_EQ(g_frees, 0);
  std::free(g_callocs[0]);
  ncclStrongStreamDestruct(&ss_);
}

// Each HIP call after the entry is linked propagates; the entry stays for the next acquire.
TEST_F(StrongstreamMicrotest, AcquireCapture_JoinFailuresPropagate) {
  const std::vector<std::pair<std::string, int>> failures = {
      {"hipEventCreateWithFlags", 1}, {"hipEventRecord", 1},  {"hipStreamWaitEvent", 1}, {"hipEventDestroy", 1},
      {"hipStreamUpdateCaptureDependencies", 1}, {"hipEventRecord", 2}, {"hipStreamWaitEvent", 2}};
  for (const auto& f : failures) {
    SCOPED_TRACE(f.first + " #" + std::to_string(f.second));
    Construct(&ss_);
    FailNext(f.first, f.second);
    hipStream_t work = nullptr;
    EXPECT_EQ(ncclStrongStreamAcquire(Captured(1), &ss_, false, &work), ncclUnhandledCudaError);
    ASSERT_NE(ss_.captureHead, nullptr);
    EXPECT_EQ(work, ss_.captureHead->captureStream);
    g_failAt.clear();
    ncclStrongStreamDestruct(&ss_);
  }
}

// The list is walked under the strong stream's mutex, which is dropped before the stream joins.
TEST_F(StrongstreamMicrotest, AcquireCapture_ConcurrentHoldsTheMutexOnlyForTheList) {
  Construct(&ss_);
  hipStream_t a = AcquireCapture(&ss_, 1);
  std::map<std::string, bool> held;
  g_onCall = [&](const std::string& fn) {
    if (!held.count(fn)) held[fn] = HeldElsewhere(ss_.mutex);
  };
  hipStream_t work = nullptr;
  ASSERT_EQ(ncclStrongStreamAcquire(Captured(2), &ss_, true, &work), ncclSuccess);
  EXPECT_TRUE(held["hipStreamIsCapturing"]);
  EXPECT_TRUE(held["hipStreamCreateWithFlags"]);
  EXPECT_FALSE(held["hipEventCreateWithFlags"]);

  held.clear();
  g_capture[a].status = hipStreamCaptureStatusNone;
  ASSERT_EQ(ncclStrongStreamAcquire(Captured(3), &ss_, false, &work), ncclSuccess);
  EXPECT_FALSE(held["hipStreamIsCapturing"]);
  g_onCall = nullptr;
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, ConcurrentLookupsWaitForTheMutex) {
  Construct(&ss_);
  AcquireCapture(&ss_, 1);
  hipStream_t work = nullptr;
  g_raceFatalSet = true;  // Release runs on the async thread, not the acquirer
  g_raceFatal = 0;
  for (bool concurrent : {true, false}) {
    SCOPED_TRACE(concurrent);
    EXPECT_EQ(CompletesWhileLocked(ss_.mutex, [&] { ncclStrongStreamAcquire(Captured(1), &ss_, concurrent, &work); }),
              !concurrent);
    EXPECT_EQ(CompletesWhileLocked(ss_.mutex,
                                   [&] { ncclStrongStreamAcquiredWorkStream(Captured(1), &ss_, concurrent, &work); }),
              !concurrent);
    EXPECT_EQ(CompletesWhileLocked(ss_.mutex,
                                   [&] { ncclStrongStreamRelease(Captured(1), &ss_, concurrent); }),
              !concurrent);
  }
  ncclStrongStreamDestruct(&ss_);
}

// Release adds a serialEvent record node after the capture stream's frontier and moves the
// frontier onto it.
TEST_F(StrongstreamMicrotest, ReleaseCapture_RecordsTheSerialEventIntoTheGraph) {
  Construct(&ss_);
  hipStream_t first = AcquireCapture(&ss_, 1);
  AcquireCapture(&ss_, 2);
  hipGraphNode_t n1 = reinterpret_cast<hipGraphNode_t>(0x51), n2 = reinterpret_cast<hipGraphNode_t>(0x52);
  g_capture[first].deps = {n1, n2};
  g_calls.clear();

  std::map<std::string, bool> held;
  g_onCall = [&](const std::string& fn) { held[fn] = HeldElsewhere(ss_.mutex); };
  ASSERT_EQ(ncclStrongStreamRelease(Captured(1), &ss_, true), ncclSuccess);
  g_onCall = nullptr;
  EXPECT_FALSE(held["hipGraphAddEventRecordNode"]);

  auto node = CallsTo("hipGraphAddEventRecordNode");
  ASSERT_EQ(node.size(), 1u);
  EXPECT_EQ(node[0].a, kGraph);
  EXPECT_EQ(node[0].b, ss_.serialEvent);
  EXPECT_EQ(node[0].n, 0u);
  EXPECT_EQ(CallsTo("hipStreamGetCaptureInfo_v2")[0].a, first);
  auto deps = CallsTo("hipGraphAddDependencies");
  ASSERT_EQ(deps.size(), 2u);
  EXPECT_EQ(deps[0].a, n1);
  EXPECT_EQ(deps[1].a, n2);
  for (const auto& d : deps) {
    EXPECT_EQ(d.b, node[0].c);
    EXPECT_EQ(d.c, kGraph);
    EXPECT_EQ(d.n, 1u);
  }
  auto update = CallsTo("hipStreamUpdateCaptureDependencies");
  ASSERT_EQ(update.size(), 1u);
  EXPECT_EQ(update[0].a, first);
  EXPECT_EQ(update[0].c, node[0].c);
  EXPECT_EQ(update[0].n, 1u);
  EXPECT_EQ(update[0].flags, (uint64_t)hipStreamSetCaptureDependencies);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, ReleaseCapture_FromAnotherThreadIsALaunchRace) {
  Construct(&ss_);
  std::thread([&] { AcquireCapture(&ss_, 1); }).join();
  EXPECT_EQ(ncclStrongStreamRelease(Captured(1), &ss_, false), ncclInvalidUsage);
  EXPECT_TRUE(Logged(NCCL_LOG_ERROR, "host threads racing"));
  EXPECT_EQ(Count("hipStreamUpdateCaptureDependencies"), 2);

  g_raceFatalSet = true;
  g_raceFatal = 0;
  EXPECT_EQ(ncclStrongStreamRelease(Captured(1), &ss_, false), ncclSuccess);
  ncclStrongStreamDestruct(&ss_);
}

TEST_F(StrongstreamMicrotest, ReleaseCapture_RecordFailuresPropagate) {
  const std::vector<std::pair<std::string, int>> failures = {{"hipGraphAddEventRecordNode", 1},
                                                             {"hipStreamGetCaptureInfo_v2", 1},
                                                             {"hipGraphAddDependencies", 1},
                                                             {"hipStreamUpdateCaptureDependencies", 1}};
  Construct(&ss_);
  hipStream_t work = AcquireCapture(&ss_, 1);
  g_capture[work].deps = {reinterpret_cast<hipGraphNode_t>(0x51)};
  for (const auto& f : failures) {
    SCOPED_TRACE(f.first);
    FailNext(f.first, f.second);
    EXPECT_EQ(ncclStrongStreamRelease(Captured(1), &ss_, false), ncclUnhandledCudaError);
  }
  ncclStrongStreamDestruct(&ss_);
}

// ncclCudaGraphRecordEvent records like Release but must leave the origin's frontier alone.
TEST_F(StrongstreamMicrotest, GraphRecordEvent_AddsANodeWithoutMovingTheFrontier) {
  hipEvent_t ev = reinterpret_cast<hipEvent_t>(0x99);
  hipGraphNode_t n1 = reinterpret_cast<hipGraphNode_t>(0x51);
  g_capture[kOrigin].deps = {n1};
  ASSERT_EQ(ncclCudaGraphRecordEvent(Captured(1, 0), ev, kOrigin), ncclSuccess);
  auto node = CallsTo("hipGraphAddEventRecordNode");
  ASSERT_EQ(node.size(), 1u);
  EXPECT_EQ(node[0].b, ev);
  auto deps = CallsTo("hipGraphAddDependencies");
  ASSERT_EQ(deps.size(), 1u);
  EXPECT_EQ(deps[0].a, n1);
  EXPECT_EQ(Count("hipStreamUpdateCaptureDependencies"), 0);

  g_calls.clear();
  ASSERT_EQ(ncclCudaGraphRecordEvent(Live(), ev, kOrigin), ncclSuccess);
  EXPECT_TRUE(g_calls.empty());

  FailNext("hipGraphAddEventRecordNode");
  EXPECT_EQ(ncclCudaGraphRecordEvent(Captured(1, 0), ev, kOrigin), ncclUnhandledCudaError);
}

// Stream ordering helpers

TEST_F(StrongstreamMicrotest, StreamWaitStream_RecordsOnBThenAWaits) {
  hipStream_t a = reinterpret_cast<hipStream_t>(0xa), b = reinterpret_cast<hipStream_t>(0xb);
  hipEvent_t e = reinterpret_cast<hipEvent_t>(0xe);
  ASSERT_EQ(ncclStreamWaitStream(a, b, e), ncclSuccess);
  ASSERT_EQ(g_calls.size(), 2u);
  EXPECT_EQ(g_calls[0].fn, "hipEventRecord");
  EXPECT_EQ(g_calls[0].a, e);
  EXPECT_EQ(g_calls[0].b, b);
  EXPECT_EQ(g_calls[1].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[1].a, a);
  EXPECT_EQ(g_calls[1].b, e);
  EXPECT_EQ(g_calls[1].flags, 0u);

  FailNext("hipEventRecord");
  EXPECT_EQ(ncclStreamWaitStream(a, b, e), ncclUnhandledCudaError);
  FailNext("hipStreamWaitEvent");
  EXPECT_EQ(ncclStreamWaitStream(a, b, e), ncclUnhandledCudaError);
}

TEST_F(StrongstreamMicrotest, AdvanceToEvent_LiveJustWaits) {
  hipEvent_t e = reinterpret_cast<hipEvent_t>(0xe);
  ASSERT_EQ(ncclStreamAdvanceToEvent(Live(), kOrigin, e), ncclSuccess);
  ASSERT_EQ(g_calls.size(), 1u);
  EXPECT_EQ(g_calls[0].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[0].a, kOrigin);
  EXPECT_EQ(g_calls[0].b, e);
  FailNext("hipStreamWaitEvent");
  EXPECT_EQ(ncclStreamAdvanceToEvent(Live(), kOrigin, e), ncclUnhandledCudaError);
}

// Captured: `s` takes over the dependency set a scratch stream gets from waiting on `e`.
TEST_F(StrongstreamMicrotest, AdvanceToEvent_CapturedReplacesTheFrontierWithTheEvents) {
  hipEvent_t e = reinterpret_cast<hipEvent_t>(0xe);
  hipGraphNode_t n1 = reinterpret_cast<hipGraphNode_t>(0x51), n2 = reinterpret_cast<hipGraphNode_t>(0x52);
  g_defaultCapture.deps = {n1, n2};
  ASSERT_EQ(ncclStreamAdvanceToEvent(Captured(1), kOrigin, e), ncclSuccess);
  ASSERT_EQ(g_calls.size(), 5u);
  EXPECT_EQ(g_calls[0].fn, "hipStreamCreateWithFlags");
  EXPECT_EQ(g_calls[0].flags, (uint64_t)hipStreamNonBlocking);
  const void* tmp = g_calls[0].a;
  EXPECT_EQ(g_calls[1].fn, "hipStreamWaitEvent");
  EXPECT_EQ(g_calls[1].a, tmp);
  EXPECT_EQ(g_calls[1].b, e);
  EXPECT_EQ(g_calls[2].fn, "hipStreamGetCaptureInfo_v2");
  EXPECT_EQ(g_calls[2].a, tmp);
  EXPECT_EQ(g_calls[3].fn, "hipStreamUpdateCaptureDependencies");
  EXPECT_EQ(g_calls[3].a, kOrigin);
  EXPECT_EQ(g_calls[3].b, g_defaultCapture.deps.data());
  EXPECT_EQ(g_calls[3].n, 2u);
  EXPECT_EQ(g_calls[3].flags, (uint64_t)hipStreamSetCaptureDependencies);
  EXPECT_EQ(g_calls[4].fn, "hipStreamDestroy");
  EXPECT_EQ(g_calls[4].a, tmp);
  EXPECT_TRUE(g_liveStreams.empty());
}

TEST_F(StrongstreamMicrotest, AdvanceToEvent_CapturedHipFailuresPropagate) {
  for (const char* fn : {"hipStreamCreateWithFlags", "hipStreamWaitEvent", "hipStreamGetCaptureInfo_v2",
                         "hipStreamUpdateCaptureDependencies", "hipStreamDestroy"}) {
    SCOPED_TRACE(fn);
    FailNext(fn);
    EXPECT_EQ(ncclStreamAdvanceToEvent(Captured(1), kOrigin, nullptr), ncclUnhandledCudaError);
  }
}

}  // namespace
