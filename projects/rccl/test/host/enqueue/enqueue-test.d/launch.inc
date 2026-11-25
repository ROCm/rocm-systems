/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rcclAddonLaunchBegin/End, graph capture ordering, ncclLaunchFinish under capture and getImplicitOrder.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// rcclAddonLaunchBegin (enqueue.cc:2092) / rcclAddonLaunchEnd (:2125) -- the
// bracket an addon collective wraps its own launch in.
//
// AICOMRCCL-2184: the epilogue records comm->doneEvent unless a kernel took it as
// its stop event. The production launches that take nothing are the CE two-shot
// AllReduce and the hierarchical ReduceScatter, which need CE-capable hardware or
// eight nodes, so the arm is only assertable here.
//
// In the two cases that must not record, the hipEventRecord hook fails every
// call: an unexpected call then shows up twice, in the hook's call count and in
// the result the bracket returns.
//
// Each case installs the capture answer it relies on instead of inheriting it: a host binary has no
// graph and can only answer "not capturing", and a case that took that answer without asking for it
// would look covered while never reaching capture at all.
// ===========================================================================

namespace {
// Opaque handles. Every HIP entry point this path reaches is faked, so nothing dereferences them.
hipEvent_t const  kAddonDoneEvent = reinterpret_cast<hipEvent_t>(0xD0E);
hipStream_t const kAddonStream    = reinterpret_cast<hipStream_t>(0x57A);

// ncclComm is far too large for the stack (see CostComm). cudaDev matches the seam's current
// device, which keeps the prologue's hipSetDevice and the epilogue's restore out of the way.
struct AddonComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  AddonComm() {
    comm->cudaDev   = g_currentDevice;
    comm->doneEvent = kAddonDoneEvent;
  }
  ncclComm* c() { return comm.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, AddonLaunch_NothingTookTheEvent_EpilogueRecordsIt) {
  AddonComm ac;
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  hipEvent_t recordedEvent = nullptr;
  hipStream_t recordedStream = nullptr;
  ScopedHook record(g_hipEventRecord, [&](hipEvent_t event, hipStream_t stream) {
    recordedEvent = event;
    recordedStream = stream;
    return hipSuccess;
  });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclSuccess; }));

  ASSERT_EQ(1, record.calls);
  EXPECT_EQ(kAddonDoneEvent, recordedEvent);
  EXPECT_EQ(kAddonStream, recordedStream);
  EXPECT_EQ(ncclStreamTag(kAddonStream), ac.c()->lastStreamTag);
}

TEST_F(EnqueueMicrotest, AddonLaunch_KernelTookTheEvent_EpilogueSkipsTheRecord) {
  AddonComm ac;
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });

  hipEvent_t stopEvent = nullptr;
  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    stopEvent = rcclTakeAddonStopEvent(ac.c());
    return ncclSuccess;
  }));

  EXPECT_EQ(kAddonDoneEvent, stopEvent);
  EXPECT_EQ(0, record.calls);
  // The tag advances however the event reached the stream, so the next collective on another
  // stream still gets its edge.
  EXPECT_EQ(ncclStreamTag(kAddonStream), ac.c()->lastStreamTag);
}

TEST_F(EnqueueMicrotest, AddonLaunch_LaunchFailed_RecordsNothingAndLeavesTheTag) {
  AddonComm ac;
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclInternalError,
            rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclInternalError; }));

  EXPECT_EQ(0, record.calls);
  EXPECT_EQ(0u, ac.c()->lastStreamTag);
}

// Begin fails, so launch() must not run. cudaDev 5 != current 3 makes the set and the restore observable.
TEST_F(EnqueueMicrotest, AddonLaunch_SetDeviceFailed_SkipsLaunchAndRestoresTheDevice) {
  AddonComm ac;
  ac.c()->cudaDev        = 5;
  ac.c()->addonStopEvent = kAddonDoneEvent;  // Begin clears it, so a stale event can't leak through
  g_currentDevice        = 3;
  std::vector<int> setDevices;
  ScopedHook setDevice(g_hipSetDevice, [&](int dev) {
    setDevices.push_back(dev);
    if (dev == 5) return hipErrorInvalidDevice;
    g_currentDevice = dev;
    return hipSuccess;
  });
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });

  int launches = 0;
  ncclResult_t res = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] {
    res = rcclAddonLaunch(ac.c(), kAddonStream, [&] {
      ++launches;
      return ncclSuccess;
    });
  });

  EXPECT_EQ(ncclUnhandledCudaError, res);
  EXPECT_EQ(0, launches);
  EXPECT_EQ(0, notCapturing.calls);  // Begin stopped at the failed set
  EXPECT_EQ(0, record.calls);
  EXPECT_EQ(0u, ac.c()->lastStreamTag);
  EXPECT_EQ(nullptr, ac.c()->addonStopEvent);
  EXPECT_EQ(std::vector<int>({5, 3}), setDevices);  // the failed set, then the restore
  EXPECT_EQ(3, g_currentDevice);
}

TEST_F(EnqueueMicrotest, AddonLaunch_CaptureQueryFailed_SkipsLaunchAndRestoresTheDevice) {
  AddonComm ac;
  ac.c()->cudaDev        = 5;
  ac.c()->addonStopEvent = kAddonDoneEvent;  // Begin clears it, so a stale event can't leak through
  g_currentDevice        = 3;
  // A tag from another stream: a launch that ran anyway would overwrite it with kAddonStream's.
  hipStream_t const otherStream = reinterpret_cast<hipStream_t>(0x0B5);
  ac.c()->lastStreamTag = ncclStreamTag(otherStream);
  std::vector<int> setDevices;
  ScopedHook setDevice(g_hipSetDevice, [&](int dev) {
    setDevices.push_back(dev);
    g_currentDevice = dev;
    return hipSuccess;
  });
  ScopedHook queryFails(g_cudaGetCapturingGraph,
                        [](struct ncclCudaGraph*, hipStream_t, int) { return ncclInternalError; });
  ScopedHook waitEvent(g_hipStreamWaitEvent, [](hipStream_t, hipEvent_t, unsigned int) { return hipSuccess; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });

  int launches = 0;
  EXPECT_EQ(ncclInternalError, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
              ++launches;
              return ncclSuccess;
            }));

  EXPECT_EQ(1, queryFails.calls);
  EXPECT_EQ(0, launches);
  EXPECT_EQ(0, waitEvent.calls);
  EXPECT_EQ(0, record.calls);
  EXPECT_EQ(ncclStreamTag(otherStream), ac.c()->lastStreamTag);
  EXPECT_EQ(nullptr, ac.c()->addonStopEvent);
  EXPECT_EQ(std::vector<int>({5, 3}), setDevices);  // set by the prologue, restored by the epilogue
  EXPECT_EQ(3, g_currentDevice);
}

// ROCM-33092: an addon collective never reaches ncclEnqueueCheck, so the info overload is what keeps
// comm->opCount in step with the native path.
TEST_F(EnqueueMicrotest, AddonLaunch_WithInfo_AdvancesOpCountOnSuccess) {
  AddonComm ac;
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });
  int localRankToRank[1] = {0};
  ac.c()->localRankToRank = localRankToRank;
  ac.c()->opCount = 7;

  struct ncclInfo info = {};
  info.opName = "ReduceScatter";
  info.comm = ac.c();
  info.stream = kAddonStream;
  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(info, [] { return ncclSuccess; }));
  EXPECT_EQ(8u, ac.c()->opCount);
}

TEST_F(EnqueueMicrotest, AddonLaunch_WithInfo_LaunchFailedLeavesOpCount) {
  AddonComm ac;
  ScopedHook notCapturing(g_cudaGetCapturingGraph,
                          [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
                            if (graph) *graph = ncclCudaGraphNone(mode);
                            return ncclSuccess;
                          });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });
  int localRankToRank[1] = {0};
  ac.c()->localRankToRank = localRankToRank;
  ac.c()->opCount = 7;

  struct ncclInfo info = {};
  info.opName = "ReduceScatter";
  info.comm = ac.c();
  info.stream = kAddonStream;
  EXPECT_EQ(ncclInternalError, rcclAddonLaunch(info, [] { return ncclInternalError; }));
  EXPECT_EQ(7u, ac.c()->opCount);
}

// ===========================================================================
// rcclAddonLaunch under graph capture (AICOMRCCL-2184). A captured addon collective joins the
// per-communicator chain that ncclLaunchPrepare / ncclLaunchFinish capture native collectives into:
// sharedRes->deviceStream when graphStreamOrdering is on, its serialEvent when it is off.
// ===========================================================================

namespace {
hipStream_t const        kAddonDeviceStream = reinterpret_cast<hipStream_t>(0xD57);
hipStream_t const        kAddonLiveStream   = reinterpret_cast<hipStream_t>(0x11E);
hipEvent_t const         kAddonScratchEvent = reinterpret_cast<hipEvent_t>(0x5C8);
hipEvent_t const         kAddonSerialEvent  = reinterpret_cast<hipEvent_t>(0x5E1);
unsigned long long const kAddonGraphId      = 42;

struct CapturingAddonComm : AddonComm {
  std::unique_ptr<ncclSharedResources> sharedRes{new ncclSharedResources{}};
  explicit CapturingAddonComm(int graphStreamOrdering) {
    comm->config.graphStreamOrdering    = graphStreamOrdering;
    comm->sharedRes                     = sharedRes.get();
    sharedRes->scratchEvent             = kAddonScratchEvent;
    sharedRes->deviceStream.liveStream  = kAddonLiveStream;
    sharedRes->deviceStream.serialEvent = kAddonSerialEvent;
  }
};

ncclResult_t CapturingGraph(struct ncclCudaGraph* graph, hipStream_t, int mode) {
  if (graph) {
    *graph         = ncclCudaGraphNone(mode);
    graph->graphId = kAddonGraphId;
  }
  return ncclSuccess;
}
}  // namespace

TEST_F(EnqueueMicrotest, AddonLaunch_CapturedWithStreamOrdering_JoinsTheDeviceStreamChain) {
  CapturingAddonComm ac(/*graphStreamOrdering=*/1);
  struct ncclStrongStream* const deviceStream = &ac.c()->sharedRes->deviceStream;
  std::vector<std::string> calls;
  ScopedHook capturing(g_cudaGetCapturingGraph, CapturingGraph);
  ScopedHook acquire(g_strongStreamAcquire,
                     [&](struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool, hipStream_t* workStream) {
                       EXPECT_EQ(kAddonGraphId, graph.graphId);
                       EXPECT_EQ(deviceStream, ss);
                       *workStream = kAddonDeviceStream;
                       calls.push_back("acquire");
                       return ncclSuccess;
                     });
  ScopedHook waitStream(g_ncclStreamWaitStream, [&](hipStream_t waiter, hipStream_t waited, hipEvent_t event) {
    EXPECT_EQ(kAddonStream, waiter);
    EXPECT_EQ(kAddonDeviceStream, waited);
    EXPECT_EQ(kAddonScratchEvent, event);
    calls.push_back("wait");
    return ncclSuccess;
  });
  ScopedHook record(g_hipEventRecord, [&](hipEvent_t event, hipStream_t stream) {
    EXPECT_EQ(kAddonStream, stream);
    calls.push_back(event == kAddonDoneEvent      ? "recordDone"
                    : event == kAddonScratchEvent ? "recordScratch"
                                                  : "recordOther");
    return hipSuccess;
  });
  ScopedHook advanceToEvent(g_ncclStreamAdvanceToEvent, [](struct ncclCudaGraph, hipStream_t, hipEvent_t) {
    return ncclInternalError;
  });
  ScopedHook advance(g_ncclStreamAdvanceToStream, [&](hipStream_t stream, hipStream_t source) {
    EXPECT_EQ(kAddonDeviceStream, stream);
    EXPECT_EQ(kAddonStream, source);
    calls.push_back("advance");
    return ncclSuccess;
  });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool) {
    EXPECT_EQ(kAddonGraphId, graph.graphId);
    EXPECT_EQ(deviceStream, ss);
    calls.push_back("release");
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    EXPECT_EQ(nullptr, rcclTakeAddonStopEvent(ac.c()));
    calls.push_back("launch");
    return ncclSuccess;
  }));

  EXPECT_EQ((std::vector<std::string>{"acquire", "wait", "launch", "recordDone", "advance", "release"}), calls);
  EXPECT_EQ(0, advanceToEvent.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_CapturedWithoutStreamOrdering_WaitsOnAndRecordsTheSerialEvent) {
  CapturingAddonComm ac(/*graphStreamOrdering=*/0);
  ScopedHook capturing(g_cudaGetCapturingGraph, CapturingGraph);
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclInternalError; });
  int bootstrapRecords = 0;
  ScopedHook record(g_hipEventRecord, [&](hipEvent_t event, hipStream_t stream) {
    if (event == kAddonSerialEvent) {
      EXPECT_EQ(kAddonLiveStream, stream);
      ++bootstrapRecords;
    }
    return hipSuccess;
  });
  ScopedHook wait(g_hipStreamWaitEvent, [](hipStream_t stream, hipEvent_t event, unsigned int flags) {
    EXPECT_EQ(kAddonStream, stream);
    EXPECT_EQ(kAddonSerialEvent, event);
    EXPECT_EQ(static_cast<unsigned int>(hipEventWaitExternal), flags);
    return hipSuccess;
  });
  ScopedHook graphRecord(g_ncclCudaGraphRecordEvent, [](struct ncclCudaGraph graph, hipEvent_t event, hipStream_t stream) {
    EXPECT_EQ(kAddonGraphId, graph.graphId);
    EXPECT_EQ(kAddonSerialEvent, event);
    EXPECT_EQ(kAddonStream, stream);
    return ncclSuccess;
  });

  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclSuccess; }));
  }

  EXPECT_EQ(0, acquire.calls);
  EXPECT_EQ(0, release.calls);
  // The live-stream record only bootstraps the first capture's external wait.
  EXPECT_EQ(1, bootstrapRecords);
  EXPECT_EQ(2, wait.calls);
  EXPECT_EQ(2, graphRecord.calls);
  EXPECT_TRUE(ac.c()->sharedRes->deviceStream.graphOriginCaptured);
  EXPECT_TRUE(ac.c()->sharedRes->deviceStream.everCaptured);
}

TEST_F(EnqueueMicrotest, AddonLaunch_NotCapturing_LeavesTheDeviceStreamAlone) {
  AddonComm ac;  // no sharedRes: touching the capture chain would crash, not just miscount
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclInternalError; });
  ScopedHook waitStream(g_ncclStreamWaitStream, [](hipStream_t, hipStream_t, hipEvent_t) { return ncclInternalError; });
  ScopedHook graphRecord(g_ncclCudaGraphRecordEvent,
                         [](struct ncclCudaGraph, hipEvent_t, hipStream_t) { return ncclInternalError; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclSuccess; }));

  EXPECT_EQ(0, acquire.calls);
  EXPECT_EQ(0, release.calls);
  EXPECT_EQ(0, waitStream.calls);
  EXPECT_EQ(0, graphRecord.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_CapturedLaunchFailed_StillReleasesTheDeviceStream) {
  CapturingAddonComm ac(/*graphStreamOrdering=*/1);
  ScopedHook capturing(g_cudaGetCapturingGraph, CapturingGraph);
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                       *workStream = kAddonDeviceStream;
                       return ncclSuccess;
                     });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook advance(g_ncclStreamAdvanceToStream, [](hipStream_t, hipStream_t) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclSuccess; });

  EXPECT_EQ(ncclInternalError, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclInternalError; }));

  EXPECT_EQ(1, acquire.calls);
  EXPECT_EQ(1, release.calls);
  EXPECT_EQ(0, record.calls);
  EXPECT_EQ(0, advance.calls);
  EXPECT_EQ(0u, ac.c()->lastStreamTag);
}

// ===========================================================================
// rcclAddonLaunch under NCCL_LAUNCH_ORDER_IMPLICIT=1. The bracket joins context->launchOrder through the
// helpers ncclLaunchPrepare / ncclLaunchFinish use for native plans: the addon stream first waits on
// launchOrder (the previous launch on this device, from any communicator), then launchOrder waits on an
// event recorded after the addon work (Serial order; an addon launch has no launch-completion event).
// ===========================================================================

namespace {
hipStream_t const kAddonLaunchOrderStream = reinterpret_cast<hipStream_t>(0x10D);
hipStream_t const kAddonGraphOrigin       = reinterpret_cast<hipStream_t>(0x0F1);

struct ImplicitOrderAddonComm : CapturingAddonComm {
  std::unique_ptr<ncclCudaContext> context{new ncclCudaContext{}};
  explicit ImplicitOrderAddonComm(int graphStreamOrdering) : CapturingAddonComm(graphStreamOrdering) {
    comm->config.launchOrderImplicit = 1;
    comm->context                    = context.get();
  }
  struct ncclStrongStream* launchOrder() { return &context->launchOrder; }
  struct ncclStrongStream* deviceStream() { return &sharedRes->deviceStream; }
};

const char* AddonStreamName(hipStream_t s) {
  if (s == kAddonStream) return "addon";
  if (s == kAddonLaunchOrderStream) return "launchOrder";
  if (s == kAddonDeviceStream) return "device";
  if (s == kAddonGraphOrigin) return "origin";
  return "other";
}

const char* AddonEventName(hipEvent_t e) {
  if (e == kAddonDoneEvent) return "Done";
  if (e == kAddonScratchEvent) return "Scratch";
  if (e == kAddonSerialEvent) return "Serial";
  return "Other";
}
}  // namespace

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_NotCapturing_JoinsTheLaunchOrderChain) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  std::vector<std::string> calls;
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) {
                       EXPECT_EQ(ac.launchOrder(), ss);
                       EXPECT_FALSE(concurrent);  // uncaptured launches keep host program order
                       *workStream = kAddonLaunchOrderStream;
                       calls.push_back("acquire:launchOrder");
                       return ncclSuccess;
                     });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) {
                        EXPECT_EQ(ac.launchOrder(), ss);
                        EXPECT_FALSE(concurrent);
                        *workStream = kAddonLaunchOrderStream;
                        return ncclSuccess;
                      });
  ScopedHook waitStream(g_ncclStreamWaitStream, [&](hipStream_t waiter, hipStream_t waited, hipEvent_t event) {
    EXPECT_EQ(kAddonScratchEvent, event);
    calls.push_back(std::string("wait:") + AddonStreamName(waiter) + "<-" + AddonStreamName(waited));
    return ncclSuccess;
  });
  ScopedHook record(g_hipEventRecord, [&](hipEvent_t event, hipStream_t stream) {
    calls.push_back(std::string("record") + AddonEventName(event) + ":" + AddonStreamName(stream));
    return hipSuccess;
  });
  ScopedHook waitEvent(g_hipStreamWaitEvent, [&](hipStream_t stream, hipEvent_t event, unsigned int) {
    calls.push_back(std::string("wait:") + AddonStreamName(stream) + "<-" + AddonEventName(event));
    return hipSuccess;
  });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool concurrent) {
    EXPECT_EQ(ac.launchOrder(), ss);
    EXPECT_FALSE(concurrent);
    calls.push_back("release:launchOrder");
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    calls.push_back("launch");
    return ncclSuccess;
  }));

  EXPECT_EQ((std::vector<std::string>{"acquire:launchOrder", "wait:addon<-launchOrder", "launch", "recordDone:addon",
                                      "recordScratch:addon", "wait:launchOrder<-Scratch", "release:launchOrder"}),
            calls);
  EXPECT_EQ(1, acquired.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_CapturedWithStreamOrdering_JoinsBothChains) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  std::vector<std::string> calls;
  ScopedHook capturing(g_cudaGetCapturingGraph, CapturingGraph);
  ScopedHook acquire(g_strongStreamAcquire,
                     [&](struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) {
                       EXPECT_EQ(kAddonGraphId, graph.graphId);
                       if (ss == ac.launchOrder()) {
                         EXPECT_TRUE(concurrent);  // as ncclLaunchPrepare: captured launchOrder access is concurrent
                         *workStream = kAddonLaunchOrderStream;
                         calls.push_back("acquire:launchOrder");
                       } else {
                         EXPECT_EQ(ac.deviceStream(), ss);
                         EXPECT_FALSE(concurrent);
                         *workStream = kAddonDeviceStream;
                         calls.push_back("acquire:device");
                       }
                       return ncclSuccess;
                     });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [&](struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) {
                        EXPECT_EQ(kAddonGraphId, graph.graphId);
                        EXPECT_EQ(ac.launchOrder(), ss);
                        EXPECT_TRUE(concurrent);
                        *workStream = kAddonLaunchOrderStream;
                        return ncclSuccess;
                      });
  ScopedHook waitStream(g_ncclStreamWaitStream, [&](hipStream_t waiter, hipStream_t waited, hipEvent_t) {
    calls.push_back(std::string("wait:") + AddonStreamName(waiter) + "<-" + AddonStreamName(waited));
    return ncclSuccess;
  });
  ScopedHook record(g_hipEventRecord, [&](hipEvent_t event, hipStream_t stream) {
    calls.push_back(std::string("record") + AddonEventName(event) + ":" + AddonStreamName(stream));
    return hipSuccess;
  });
  ScopedHook waitEvent(g_hipStreamWaitEvent, [&](hipStream_t stream, hipEvent_t event, unsigned int) {
    calls.push_back(std::string("wait:") + AddonStreamName(stream) + "<-" + AddonEventName(event));
    return hipSuccess;
  });
  ScopedHook advance(g_ncclStreamAdvanceToStream, [&](hipStream_t stream, hipStream_t source) {
    calls.push_back(std::string("advance:") + AddonStreamName(stream) + "<-" + AddonStreamName(source));
    return ncclSuccess;
  });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool concurrent) {
    if (ss == ac.launchOrder()) {
      EXPECT_TRUE(concurrent);
      calls.push_back("release:launchOrder");
    } else {
      EXPECT_EQ(ac.deviceStream(), ss);
      calls.push_back("release:device");
    }
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    calls.push_back("launch");
    return ncclSuccess;
  }));

  // launchOrder is released before deviceStream, as in ncclLaunchFinish.
  EXPECT_EQ((std::vector<std::string>{"acquire:device", "wait:addon<-device", "acquire:launchOrder",
                                      "wait:addon<-launchOrder", "launch", "recordDone:addon",
                                      "advance:device<-addon", "recordScratch:addon", "wait:launchOrder<-Scratch",
                                      "release:launchOrder", "release:device"}),
            calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_CapturedWithoutStreamOrdering_UsesTheGraphOrigin) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/0);
  ScopedHook capturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) {
      *graph         = ncclCudaGraphNone(mode);
      graph->graphId = kAddonGraphId;
      graph->origin  = kAddonGraphOrigin;
    }
    return ncclSuccess;
  });
  // As ncclLaunchPrepare with GRAPH_STREAM_ORDERING=0: the graph origin stands in for launchOrder.
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclInternalError; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });
  std::vector<std::string> waits;
  ScopedHook wait(g_hipStreamWaitEvent, [&](hipStream_t stream, hipEvent_t event, unsigned int) {
    waits.push_back(std::string(AddonStreamName(stream)) + "<-" + AddonEventName(event));
    return hipSuccess;
  });
  ScopedHook graphRecord(g_ncclCudaGraphRecordEvent,
                         [](struct ncclCudaGraph, hipEvent_t, hipStream_t) { return ncclSuccess; });
  ScopedHook waitStream(g_ncclStreamWaitStream, [&](hipStream_t waiter, hipStream_t waited, hipEvent_t) {
    waits.push_back(std::string(AddonStreamName(waiter)) + "<-" + AddonStreamName(waited));
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclSuccess; }));

  EXPECT_EQ((std::vector<std::string>{"addon<-Serial", "addon<-origin", "origin<-Scratch"}), waits);
  EXPECT_EQ(0, acquire.calls);
  EXPECT_EQ(0, acquired.calls);
  EXPECT_EQ(0, release.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_LaunchFailed_StillReleasesTheLaunchOrder) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                       *workStream = kAddonLaunchOrderStream;
                       return ncclSuccess;
                     });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook waitStream(g_ncclStreamWaitStream, [](hipStream_t, hipStream_t, hipEvent_t) { return ncclSuccess; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool) {
    EXPECT_EQ(ac.launchOrder(), ss);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclInternalError, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclInternalError; }));

  EXPECT_EQ(1, acquire.calls);
  EXPECT_EQ(1, waitStream.calls);  // only the prologue's wait; a failed launch adds no order edge
  EXPECT_EQ(0, acquired.calls);
  EXPECT_EQ(1, release.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_PrologueWaitFailed_SkipsLaunchAndReleases) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                       *workStream = kAddonLaunchOrderStream;
                       return ncclSuccess;
                     });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook waitStream(g_ncclStreamWaitStream, [](hipStream_t, hipStream_t, hipEvent_t) { return ncclInternalError; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool) {
    EXPECT_EQ(ac.launchOrder(), ss);
    return ncclSuccess;
  });

  int launches = 0;
  EXPECT_EQ(ncclInternalError, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    ++launches;
    return ncclSuccess;
  }));

  EXPECT_EQ(0, launches);
  EXPECT_EQ(1, acquire.calls);
  EXPECT_EQ(1, waitStream.calls);
  EXPECT_EQ(0, acquired.calls);
  EXPECT_EQ(0, record.calls);
  EXPECT_EQ(1, release.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrder_EpilogueWaitFailed_StillReleasesTheLaunchOrder) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                       *workStream = kAddonLaunchOrderStream;
                       return ncclSuccess;
                     });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                        *workStream = kAddonLaunchOrderStream;
                        return ncclSuccess;
                      });
  // The prologue's wait (addon <- launchOrder) succeeds; the epilogue's (launchOrder <- addon work) fails.
  ScopedHook waitStream(g_ncclStreamWaitStream, [](hipStream_t, hipStream_t, hipEvent_t) { return ncclSuccess; });
  ScopedHook waitEvent(g_hipStreamWaitEvent, [](hipStream_t stream, hipEvent_t, unsigned int) {
    EXPECT_EQ(kAddonLaunchOrderStream, stream);
    return hipErrorInvalidValue;
  });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });
  ScopedHook release(g_strongStreamRelease, [&](struct ncclCudaGraph, struct ncclStrongStream* ss, bool) {
    EXPECT_EQ(ac.launchOrder(), ss);
    return ncclSuccess;
  });

  int launches = 0;
  EXPECT_NE(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [&] {
    ++launches;
    return ncclSuccess;
  }));

  EXPECT_EQ(1, launches);
  EXPECT_EQ(1, waitStream.calls);
  EXPECT_EQ(1, waitEvent.calls);
  EXPECT_EQ(1, acquired.calls);
  EXPECT_EQ(1, release.calls);
}

TEST_F(EnqueueMicrotest, AddonLaunch_ImplicitOrderUnset_LeavesTheLaunchOrderAlone) {
  ImplicitOrderAddonComm ac(/*graphStreamOrdering=*/1);
  ac.c()->config.launchOrderImplicit = NCCL_CONFIG_UNDEF_INT;  // init default when neither env nor config sets it
  ScopedHook notCapturing(g_cudaGetCapturingGraph, [](struct ncclCudaGraph* graph, hipStream_t, int mode) {
    if (graph) *graph = ncclCudaGraphNone(mode);
    return ncclSuccess;
  });
  ScopedHook acquire(g_strongStreamAcquire,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t*) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclInternalError; });
  ScopedHook waitStream(g_ncclStreamWaitStream, [](hipStream_t, hipStream_t, hipEvent_t) { return ncclInternalError; });
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });

  EXPECT_EQ(ncclSuccess, rcclAddonLaunch(ac.c(), kAddonStream, [] { return ncclSuccess; }));

  EXPECT_EQ(0, acquire.calls);
  EXPECT_EQ(0, release.calls);
  EXPECT_EQ(0, waitStream.calls);
}

// ===========================================================================
// ncclLaunchFinish fast-forward of deviceStream under capture: a single user stream reads the
// launch stream's frontier directly, several user streams go through finishedEvent.
// ===========================================================================

namespace {
hipStream_t const kSecondUserStream = reinterpret_cast<hipStream_t>(0x2B5);

struct CapturedLaunchFinish {
  CapturingAddonComm ac{/*graphStreamOrdering=*/1};
  std::unique_ptr<ncclKernelPlan> plan{new ncclKernelPlan{}};
  ncclCudaStreamList streams[2] = {};
  explicit CapturedLaunchFinish(int numStreams) {
    struct ncclKernelPlanner* planner = &ac.c()->planner;
    planner->capturingGraph = ncclCudaGraphNone(ac.c()->config.graphUsageMode);
    planner->capturingGraph.graphId = kAddonGraphId;
    streams[0].stream = kAddonStream;
    if (numStreams == 2) {
      streams[0].next = &streams[1];
      streams[1].stream = kSecondUserStream;
    }
    planner->streams = &streams[0];
    planner->numStreams = numStreams;
    ncclIntruQueueEnqueue(&planner->planQueue, plan.get());
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, LaunchFinish_CapturedSingleStream_AdvancesDeviceStreamToTheLaunchStream) {
  CapturedLaunchFinish lf(/*numStreams=*/1);
  ScopedHook record(g_hipEventRecord, [](hipEvent_t event, hipStream_t stream) {
    EXPECT_EQ(kAddonScratchEvent, event);
    EXPECT_EQ(kAddonStream, stream);
    return hipSuccess;
  });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph graph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                        EXPECT_EQ(kAddonGraphId, graph.graphId);
                        *workStream = kAddonDeviceStream;
                        return ncclSuccess;
                      });
  ScopedHook toStream(g_ncclStreamAdvanceToStream, [](hipStream_t stream, hipStream_t source) {
    EXPECT_EQ(kAddonDeviceStream, stream);
    EXPECT_EQ(kAddonStream, source);
    return ncclSuccess;
  });
  ScopedHook toEvent(g_ncclStreamAdvanceToEvent,
                     [](struct ncclCudaGraph, hipStream_t, hipEvent_t) { return ncclInternalError; });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, ncclLaunchFinish(lf.ac.c()));

  EXPECT_EQ(1, toStream.calls);
  EXPECT_EQ(0, toEvent.calls);
  EXPECT_EQ(1, release.calls);
}

TEST_F(EnqueueMicrotest, LaunchFinish_CapturedTwoStreams_AdvancesDeviceStreamToTheFinishedEvent) {
  CapturedLaunchFinish lf(/*numStreams=*/2);
  ScopedHook record(g_hipEventRecord, [](hipEvent_t, hipStream_t) { return hipSuccess; });
  ScopedHook acquired(g_ncclStrongStreamAcquiredWorkStream,
                      [](struct ncclCudaGraph, struct ncclStrongStream*, bool, hipStream_t* workStream) {
                        *workStream = kAddonDeviceStream;
                        return ncclSuccess;
                      });
  ScopedHook toStream(g_ncclStreamAdvanceToStream, [](hipStream_t, hipStream_t) { return ncclInternalError; });
  ScopedHook toEvent(g_ncclStreamAdvanceToEvent, [](struct ncclCudaGraph graph, hipStream_t stream, hipEvent_t event) {
    EXPECT_EQ(kAddonGraphId, graph.graphId);
    EXPECT_EQ(kAddonDeviceStream, stream);
    EXPECT_EQ(kAddonScratchEvent, event);
    return ncclSuccess;
  });
  ScopedHook wait(g_hipStreamWaitEvent, [](hipStream_t stream, hipEvent_t event, unsigned int) {
    EXPECT_EQ(kSecondUserStream, stream);
    EXPECT_EQ(kAddonScratchEvent, event);
    return hipSuccess;
  });
  ScopedHook release(g_strongStreamRelease,
                     [](struct ncclCudaGraph, struct ncclStrongStream*, bool) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, ncclLaunchFinish(lf.ac.c()));

  EXPECT_EQ(0, toStream.calls);
  EXPECT_EQ(1, toEvent.calls);
  EXPECT_EQ(1, wait.calls);
  EXPECT_EQ(1, release.calls);
}

// ===========================================================================
// getImplicitOrder (enqueue.cc:2091)
// Reads comm->config.launchOrderImplicit (env is applied at init). On AMD the
// CUDA driver-version arm is #if'd out, so only two arms are reachable:
// config==1 -> Serial, anything else (0 / UNDEF) -> None. Pinning that the AMD
// build cannot return ncclImplicitOrderLaunch is the useful assertion.
// ncclComm is too large for the stack (see BatchPlanComm).
// ===========================================================================

namespace {
std::unique_ptr<ncclComm> MakeLaunchOrderComm(int launchOrderImplicit) {
  auto comm = std::unique_ptr<ncclComm>(new ncclComm{});
  comm->config.launchOrderImplicit = launchOrderImplicit;
  return comm;
}
}  // namespace

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamDisabled_IsNone) {
  auto comm = MakeLaunchOrderComm(0);
  auto mode = ncclImplicitOrderLaunch;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, comm.get(), /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderNone, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_UndefDefault_IsNone) {
  auto comm = MakeLaunchOrderComm(NCCL_CONFIG_UNDEF_INT);
  auto mode = ncclImplicitOrderLaunch;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, comm.get(), /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderNone, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamEnabled_IsSerialOnAmd) {
  auto comm = MakeLaunchOrderComm(1);
  auto mode = ncclImplicitOrderNone;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, comm.get(), /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderSerial, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_CapturingIsIrrelevantOnAmd) {
  // HONEST SCOPE: this pins that `capturing` does not change the answer; it does
  // NOT prove the AMD arm is what produced it. Under the seam's driver 12000 the
  // CUDA arm returns Serial for both values too, so an #if change would not fail
  // here. The optional driver argument is what separates the CUDA arms.
  auto comm = MakeLaunchOrderComm(1);
  auto a = ncclImplicitOrderNone;
  auto b = ncclImplicitOrderNone;
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&a, comm.get(), /*capturing=*/true));
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&b, comm.get(), /*capturing=*/false));
  EXPECT_EQ(a, b);
  EXPECT_EQ(ncclImplicitOrderSerial, a);
}
