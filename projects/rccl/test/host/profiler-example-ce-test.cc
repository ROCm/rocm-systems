/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for plugins/profiler/example/profiler_plugin_ce.cc, compiled unhipified against
// the fake CUDA runtime in fakes/profiler_example_cuda/. The CE events are timed with event records on the
// collective's stream, which must not happen while that stream is being captured into a graph.

#include <gtest/gtest.h>

#include "../../plugins/profiler/example/profiler_plugin_ce.cc"

// plugin.cc defines these; this binary does not link it.
double gettime(void) { return 0.0; }
double getProfilerStartTime(void) { return 0.0; }
void debugEvent(void*, const char*) {}

namespace {

class ProfilerExampleCeTest : public ::testing::Test {
 protected:
  static constexpr int kPoolSize = 4;
  struct ceColl pool[kPoolSize] = {};
  struct context ctx = {};
  ncclProfilerEventDescr_v6_t descr = {};

  void SetUp() override {
    g_fakeCuda = {};
    ctx.ceCollPoolSize = kPoolSize;
    ctx.ceCollPool = pool;
    pthread_mutex_init(&ctx.ceEvents.mutex, nullptr);
    descr.type = ncclProfileCeColl;
    descr.ceColl.stream = reinterpret_cast<void*>(0x1234);
  }
  void TearDown() override { pthread_mutex_destroy(&ctx.ceEvents.mutex); }
};

// Eager stream: the event is created, recorded on the stream and linked for the poller.
TEST_F(ProfilerExampleCeTest, EagerStreamStartsCeCollEvent) {
  g_fakeCuda.captureStatus = cudaStreamCaptureStatusNone;
  void* handle = nullptr;
  ASSERT_EQ(ceProfilerStartCeCollEvent(&ctx, &handle, &descr, 0.0), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(g_fakeCuda.eventsRecorded, 1);
  EXPECT_EQ(ctx.ceEvents.ceCollHead, static_cast<struct ceColl*>(handle));
  ASSERT_EQ(ceProfilerStopCeCollEvent(handle), ncclSuccess);
  EXPECT_EQ(g_fakeCuda.eventsRecorded, 2);
}

// Capturing stream (active or invalidated): no event, no record into the graph, no pool slot taken.
TEST_F(ProfilerExampleCeTest, CapturingStreamSkipsCeCollEvent) {
  for (cudaStreamCaptureStatus status : {cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated}) {
    SCOPED_TRACE(static_cast<int>(status));
    g_fakeCuda = {};
    g_fakeCuda.captureStatus = status;
    void* handle = reinterpret_cast<void*>(0x1);
    ASSERT_EQ(ceProfilerStartCeCollEvent(&ctx, &handle, &descr, 0.0), ncclSuccess);
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(g_fakeCuda.eventsCreated, 0);
    EXPECT_EQ(g_fakeCuda.eventsRecorded, 0);
    EXPECT_EQ(ctx.ceEvents.ceCollHead, nullptr);
    EXPECT_EQ(ctx.ceCollPoolIndex, 0);
  }
}

}  // namespace
