/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler.cc's ncclProfilerStartTaskEvents().
//
// NCCL 2.32.3: a symmetric-kernel plan runs on plan->channelMask while its tasks keep
// nChannels at 0, so the head task's Coll event advertises countOneBits(channelMask)
// to every Coll subscriber. Before, it did so only for plugins that had also enabled
// the per-channel ncclProfileKernelCh events.
//
// ncclProfiler is file-static, so this TU #includes the hipified profiler.cc
// (PROFILER_CC_PATH) and points it at a fake ncclProfiler_t. No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include PROFILER_CC_PATH

// sym_kernels.cc: names the kernel variant a symmetric Coll event reports.
const char* ncclSymkKernelIdToString(int) { return "AGxLL_R"; }

// The rest of profiler.cc (plugin loading, the KernelCh thread, proxy events) links
// against these; the tests below never reach them.
ncclProfiler_t* getNcclProfiler_v1(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v2(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v3(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v4(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v5(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v6(void*) { return nullptr; }
ncclProfiler_t* getNcclProfiler_v7(void*) { return nullptr; }
const char* ncclGetEnv(const char* name) { return std::getenv(name); }
void* ncclOpenProfilerPluginLib(const char*) { return nullptr; }
void* ncclGetNetPluginLib(enum ncclPluginType) { return nullptr; }
ncclResult_t ncclClosePluginLib(void*, enum ncclPluginType) { return ncclSuccess; }
char* ncclPluginLibPaths[6] = {};
int64_t rcclParamEnableProxyTrace() { return 0; }
uint64_t ncclOsGetPid() { return 0; }
thread_local int ncclGroupDepth = 0;
struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {};
char ncclLastError[1024] = {};
void ncclMemoryStackDestruct(struct ncclMemoryStack*) {}
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack*, size_t, size_t) { return nullptr; }
hipError_t hipSetDevice(int) { return hipSuccess; }
hipError_t hipGetLastError(void) { return hipSuccess; }
const char* hipGetErrorString(hipError_t) { return ""; }

namespace {

constexpr int kPlanChannels = 4;
std::vector<ncclProfilerEventDescr_t> g_collStarts;

ncclResult_t RecordCollStart(void* /*context*/, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  if (eDescr->type == ncclProfileColl) g_collStarts.push_back(*eDescr);
  *eHandle = nullptr;
  return ncclSuccess;
}

// A one-task symmetric plan on kPlanChannels channels. As symmetric_sched.cc leaves it,
// the task's own nChannels stays 0: only plan->channelMask carries the count.
class ProfilerMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_collStarts.clear();
    plugin_.name = "ProfilerMicrotest";
    plugin_.startEvent = RecordCollStart;
    ncclProfiler = &plugin_;
    plan_->comm = comm_.get();
    plan_->isSymColl = true;
    plan_->channelMask.masks[0] = (1ull << kPlanChannels) - 1;
    task_.func = ncclFuncAllGather;
    task_.datatype = ncclFloat32;
    ncclIntruQueueEnqueue(&plan_->collTaskQueue, &task_);
  }
  void TearDown() override { ncclProfiler = nullptr; }

  // nChannels of the Coll event the plugin is handed for the plan's task.
  int StartedCollNChannels(int eActivationMask) {
    task_.eActivationMask = eActivationMask;
    EXPECT_EQ(ncclProfilerStartTaskEvents(plan_.get()), ncclSuccess);
    if (g_collStarts.size() != 1) {
      ADD_FAILURE() << "expected one Coll start event, got " << g_collStarts.size();
      return -1;
    }
    EXPECT_TRUE(g_collStarts[0].coll.isSymColl);
    return g_collStarts[0].coll.nChannels;
  }

  ncclProfiler_t plugin_{};
  std::unique_ptr<ncclComm> comm_{new ncclComm{}};
  std::unique_ptr<ncclKernelPlan> plan_{new ncclKernelPlan{}};
  ncclTaskColl task_{};
};

}  // namespace

// KernelCh adds an event per channel per collective; a plugin that subscribes to Coll
// alone must still learn how many channels the symmetric kernel ran on.
TEST_F(ProfilerMicrotest, SymCollWithoutKernelChAdvertisesPlanChannels) {
  EXPECT_EQ(StartedCollNChannels(ncclProfileColl), kPlanChannels);
}

// With KernelCh enabled as well, the count was advertised before NCCL 2.32.3 too.
TEST_F(ProfilerMicrotest, SymCollWithKernelChAdvertisesPlanChannels) {
  EXPECT_EQ(StartedCollNChannels(ncclProfileColl | ncclProfileKernelCh), kPlanChannels);
}
