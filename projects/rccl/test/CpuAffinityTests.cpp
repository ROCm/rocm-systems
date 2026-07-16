/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include "StandaloneUtils.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  /**
   * \brief Verify the calling thread's CPU affinity is restored after ncclCommInitRank.
   *
   * Regression guard for NCCL issue #2033 / AICOMRCCL-1537: initTransportsRank() must
   * restore the mask it saved before applying the GPU-local one. Single-rank
   * ncclCommInitRank runs initTransportsRank on the calling thread, so a leak is
   * observable here. The check only bites when the GPU-local CPU set is a strict
   * subset of the process mask (multi-NUMA hosts); elsewhere the invariant still holds.
   * ******************************************************************************************/
  TEST(CpuAffinity, RestoredAfterInitRank)
  {
    RUN_ISOLATED_TEST("CpuAffinity_RestoredAfterInitRank", []()
    {
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 1) {
        GTEST_SKIP() << "No devices available.";
      }

      // Widen to the full online CPU set so the GPU-local subset is more likely to be
      // strictly narrower, then record what the kernel actually granted.
      long numCpus = sysconf(_SC_NPROCESSORS_ONLN);
      ASSERT_GT(numCpus, 0);

      cpu_set_t fullMask;
      CPU_ZERO(&fullMask);

      int maxCpus = static_cast<int>(std::min<long>(numCpus, CPU_SETSIZE));
      for (int cpu = 0; cpu < maxCpus; ++cpu) {
        CPU_SET(cpu, &fullMask);
      }

      if (sched_setaffinity(0, sizeof(fullMask), &fullMask) != 0) {
        GTEST_SKIP() << "Could not widen CPU affinity, cannot exercise the "
                     << "GPU-local subset case: " << strerror(errno);
      }

      cpu_set_t before;
      CPU_ZERO(&before);
      ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0)
          << "sched_getaffinity failed: " << strerror(errno);

      ncclComm_t comm;
      ncclUniqueId id;
      NCCLCHECK(ncclGetUniqueId(&id));
      HIPCALL(hipSetDevice(0));
      NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

      cpu_set_t after;
      CPU_ZERO(&after);
      ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0)
          << "sched_getaffinity failed: " << strerror(errno);

      NCCLCHECK(ncclCommDestroy(comm));

      ASSERT_TRUE(CPU_EQUAL(&before, &after))
          << "CPU affinity was not restored after ncclCommInitRank "
          << "(NCCL issue #2033 / AICOMRCCL-1537 regression)";
    });
  }
}
