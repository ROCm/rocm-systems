/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Checks that the GPU nodes of a real topology carry the SMI index as `dev`, distinct per GPU.
// A check can only fail where HIP order differs from SMI order, so it runs once with the host's
// order and once with the first two GPUs swapped; one of the two orders differs on any host.

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <set>
#include <vector>

#include "comm.h"
#include "graph/topo.h"
#include "../common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

namespace {

void checkGpuDevsAreNvmlDevs() {
  int numDevices = 0;
  ASSERT_EQ(hipGetDeviceCount(&numDevices), hipSuccess);
  if (numDevices < 2) GTEST_SKIP() << "Needs at least 2 GPUs";

  std::vector<ncclComm_t> comms(numDevices, nullptr);
  ASSERT_EQ(ncclCommInitAll(comms.data(), numDevices, nullptr), ncclSuccess);

  for (ncclComm_t comm : comms) {
    const struct ncclTopoNodeSet& gpus = comm->topo->nodes[GPU];
    int ownDev = -1;
    for (int gpuIndex = 0; gpuIndex < gpus.count; gpuIndex++) {
      if (gpus.nodes[gpuIndex].gpu.rank == comm->rank) ownDev = gpus.nodes[gpuIndex].gpu.dev;
    }
    EXPECT_EQ(ownDev, comm->nvmlDev) << "rank " << comm->rank << " (HIP device " << comm->cudaDev << ")";
  }

  const struct ncclTopoNodeSet& gpus = comms[0]->topo->nodes[GPU];
  std::set<int> devs;
  for (int gpuIndex = 0; gpuIndex < gpus.count; gpuIndex++) devs.insert(gpus.nodes[gpuIndex].gpu.dev);
  EXPECT_EQ(static_cast<int>(devs.size()), gpus.count) << "GPU nodes share a dev index";

  for (ncclComm_t comm : comms) EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

} // namespace

TEST(TopoGpuDev, OwnGpuDevIsNvmlDevAndUnique)
{
  RUN_ISOLATED_TEST("TopoGpuDev_OwnGpuDevIsNvmlDevAndUnique", []() { checkGpuDevsAreNvmlDevs(); });
}

TEST(TopoGpuDev, OwnGpuDevIsNvmlDevWithSwappedVisibility)
{
  RUN_ISOLATED_TEST_WITH_ENV(
    "TopoGpuDev_OwnGpuDevIsNvmlDevWithSwappedVisibility", []() { checkGpuDevsAreNvmlDevs(); },
    {{"HIP_VISIBLE_DEVICES", "1,0"}});
}

} // namespace RcclUnitTesting
