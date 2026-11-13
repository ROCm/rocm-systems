/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/progress_monitor.cc: the per-device worker that mirrors each registered
// communicator's device progress counters into host memory.

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "comm.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "WaitUntil.h"

#include RAS_PROGRESS_MONITOR_CC_PATH

namespace {

hipStream_t const kSideStream = reinterpret_cast<hipStream_t>(0x51);
constexpr int kCallerDev = 3;

class RasProgressMonitorMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetHipFakes();
    // The 1 ms poll is clamped to the 50 ms minimum.
    g_loadParam = ParamOverride({{"PROGRESS_COUNTER_MONITOR_POLL_MS", 1}});
    g_hipGetDevice = [](int* dev) {
      *dev = kCallerDev;
      return hipSuccess;
    };
    g_hipSetDevice = [this](int dev) {
      lastSetDevice_ = dev;
      return setDeviceResult_.load();
    };
    g_hipThreadExchangeStreamCaptureMode = [this](hipStreamCaptureMode*) { return captureModeResult_.load(); };
    g_hipStreamCreateWithFlags = [this](hipStream_t* stream, unsigned) {
      streamsCreated_++;
      *stream = kSideStream;
      return hipSuccess;
    };
    g_hipStreamSynchronize = [this](hipStream_t stream) {
      if (stream == kSideStream) streamSyncs_++;
      return hipSuccess;
    };
    g_hipStreamDestroy = [this](hipStream_t stream) {
      if (stream == kSideStream) streamsDestroyed_++;
      return hipSuccess;
    };
    g_hipMemcpyAsync = [this](void* dst, const void* src, size_t n, hipMemcpyKind kind, hipStream_t stream) {
      if (kind != hipMemcpyDeviceToHost || stream != kSideStream) return hipErrorInvalidValue;
      copies_++;
      if (memcpyResult_ != hipSuccess) return memcpyResult_.load();
      std::lock_guard<std::mutex> lock(countersMutex_);
      std::memcpy(dst, src, n);
      return hipSuccess;
    };
    g_hipEventRecord = [this](hipEvent_t, hipStream_t stream) {
      if (stream == kSideStream) eventRecords_++;
      return hipSuccess;
    };
    g_hipEventCreateResult = hipSuccess;
    g_hipAsyncOpsResult = hipSuccess;
  }

  void TearDown() override {
    // Unregister everything first so no worker still runs on the hooks reset below.
    for (auto& c : comms_) ncclProgressCounterMonitorDestroy(c->comm.get());
    comms_.clear();
    ResetHipFakes();
    ResetNcclFakes();
  }

  struct TestComm {
    std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
    ncclProgressCountersBlock device{};
    ncclProgressCountersBlock host{};
  };

  ncclComm* AddComm(int cudaDev, uint64_t marker) {
    comms_.push_back(std::make_unique<TestComm>());
    TestComm& c = *comms_.back();
    c.comm->cudaDev = cudaDev;
    c.device.completedWorkCount[0] = marker;
    c.comm->deviceCountersBlock = &c.device;
    c.comm->hostCountersBlock = &c.host;
    return c.comm.get();
  }
  // The worker copies under countersMutex_, standing in for the DMA engine.
  uint64_t HostMarker(size_t i) {
    std::lock_guard<std::mutex> lock(countersMutex_);
    return comms_[i]->host.completedWorkCount[0];
  }
  void SetDeviceMarker(size_t i, uint64_t marker) {
    std::lock_guard<std::mutex> lock(countersMutex_);
    comms_[i]->device.completedWorkCount[0] = marker;
  }

  std::mutex countersMutex_;
  std::vector<std::unique_ptr<TestComm>> comms_;
  std::atomic<int> lastSetDevice_{-1};
  std::atomic<hipError_t> setDeviceResult_{hipSuccess};
  std::atomic<hipError_t> captureModeResult_{hipSuccess};
  std::atomic<hipError_t> memcpyResult_{hipSuccess};
  std::atomic<int> streamsCreated_{0};
  std::atomic<int> streamSyncs_{0};
  std::atomic<int> streamsDestroyed_{0};
  std::atomic<int> copies_{0};
  std::atomic<int> eventRecords_{0};
};

TEST_F(RasProgressMonitorMicrotest, InitWithoutDeviceCountersRegistersNothing) {
  ncclComm* comm = AddComm(0, 7);
  comm->deviceCountersBlock = nullptr;
  EXPECT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(comm));
  EXPECT_EQ(0, streamsCreated_);
}

TEST_F(RasProgressMonitorMicrotest, InitRejectsAnOutOfRangeDevice) {
  EXPECT_EQ(ncclInvalidArgument, ncclProgressCounterMonitorInit(AddComm(-1, 7)));
  EXPECT_EQ(ncclInvalidArgument, ncclProgressCounterMonitorInit(AddComm(kRasMaxCudaDevices, 7)));
  EXPECT_EQ(0, streamsCreated_);
}

TEST_F(RasProgressMonitorMicrotest, DestroyOfAnUnregisteredCommIsANoOp) {
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(0, 7)));
  EXPECT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(AddComm(0, 8)));
  EXPECT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(AddComm(-1, 9)));
  EXPECT_EQ(0, streamSyncs_);
  // The registered comm keeps its worker.
  SetDeviceMarker(0, 10);
  EXPECT_TRUE(waitUntil([&] { return HostMarker(0) == 10; }));
}

TEST_F(RasProgressMonitorMicrotest, WorkerMirrorsDeviceCountersIntoHostMemory) {
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(1, 42)));
  EXPECT_EQ(kCallerDev, lastSetDevice_);  // setup restores the caller's device
  EXPECT_TRUE(waitUntil([&] { return HostMarker(0) == 42; }));
  EXPECT_TRUE(waitUntil([&] { return eventRecords_ > 0; }));
  // The device keeps counting; later polls pick that up.
  SetDeviceMarker(0, 43);
  EXPECT_TRUE(waitUntil([&] { return HostMarker(0) == 43; }));
}

TEST_F(RasProgressMonitorMicrotest, CommsOnOneDeviceShareAWorker) {
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(2, 10)));
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(2, 20)));
  EXPECT_EQ(1, streamsCreated_);
  EXPECT_TRUE(waitUntil([&] { return HostMarker(0) == 10 && HostMarker(1) == 20; }));
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(4, 30)));
  EXPECT_EQ(2, streamsCreated_);
}

TEST_F(RasProgressMonitorMicrotest, CommWithoutHostCountersIsSkipped) {
  ncclComm* comm = AddComm(1, 42);
  comm->hostCountersBlock = nullptr;
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(comm));
  ncclComm* other = AddComm(1, 43);
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(other));
  EXPECT_TRUE(waitUntil([&] { return HostMarker(1) == 43; }));
  // Joins the worker so the call log is stable.
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(comm));
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(other));
  EXPECT_FALSE(g_hipMemcpyAsyncArgs.empty());
  for (const HipMemcpyAsyncRecord& copy : g_hipMemcpyAsyncArgs) EXPECT_EQ(other->hostCountersBlock, copy.dst);
}

TEST_F(RasProgressMonitorMicrotest, DestroyDrainsTheSideStreamAndStopsTheLastWorker) {
  ncclComm* a = AddComm(1, 10);
  ncclComm* b = AddComm(1, 20);
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(a));
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(b));

  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(a));
  EXPECT_EQ(1, streamSyncs_);
  EXPECT_EQ(0, streamsDestroyed_);  // b still uses the worker
  EXPECT_EQ(kCallerDev, lastSetDevice_);

  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorDestroy(b));
  EXPECT_EQ(2, streamSyncs_);
  EXPECT_EQ(1, streamsDestroyed_);
  EXPECT_EQ(kCallerDev, lastSetDevice_);

  // A later registration on the device starts a fresh worker.
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(1, 30)));
  EXPECT_EQ(2, streamsCreated_);
  EXPECT_TRUE(waitUntil([&] { return HostMarker(2) == 30; }));
}

TEST_F(RasProgressMonitorMicrotest, InitFailsWhenTheDeviceCannotBeSelected) {
  setDeviceResult_ = hipErrorInvalidDevice;
  EXPECT_EQ(ncclUnhandledCudaError, ncclProgressCounterMonitorInit(AddComm(1, 10)));
  EXPECT_EQ(0, streamsCreated_);
  // Nothing was left registered: a later Init on the device creates its worker.
  setDeviceResult_ = hipSuccess;
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(1, 20)));
  EXPECT_EQ(1, streamsCreated_);
}

TEST_F(RasProgressMonitorMicrotest, InitFailsWhenTheWorkerCannotStart) {
  captureModeResult_ = hipErrorUnknown;
  EXPECT_EQ(ncclSystemError, ncclProgressCounterMonitorInit(AddComm(1, 10)));
  EXPECT_EQ(1, streamsDestroyed_);
}

TEST_F(RasProgressMonitorMicrotest, FailedCopyLaunchRecordsNoCompletionEvent) {
  memcpyResult_ = hipErrorInvalidValue;
  ASSERT_EQ(ncclSuccess, ncclProgressCounterMonitorInit(AddComm(1, 10)));
  // Two polls' worth of failed launches.
  EXPECT_TRUE(waitUntil([&] { return copies_ >= 2; }));
  EXPECT_EQ(0, eventRecords_);
}

}  // namespace
