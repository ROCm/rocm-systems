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
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include "StandaloneUtils.hpp"
#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  namespace
  {
    // CPUs in the NUMA node the device hangs off, or -1 when sysfs does not expose it.
    // RCCL narrows the calling thread to exactly this set, so the regression is only
    // observable when the set is a strict subset of the process mask.
    int GetDeviceLocalCpuCount(int device)
    {
      char busId[32];
      if (hipDeviceGetPCIBusId(busId, sizeof(busId), device) != hipSuccess) {
        return -1;
      }
      for (char* c = busId; *c != '\0'; ++c) {
        *c = static_cast<char>(tolower(static_cast<unsigned char>(*c)));
      }

      char path[128];
      snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", busId);
      FILE* file = fopen(path, "r");
      if (file == nullptr) {
        return -1;
      }
      int numaNode = -1;
      if (fscanf(file, "%d", &numaNode) != 1) {
        numaNode = -1;
      }
      fclose(file);
      // The topology parser attaches devices that report no NUMA node to node 0.
      if (numaNode < 0) {
        numaNode = 0;
      }

      snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpumap", numaNode);
      file = fopen(path, "r");
      if (file == nullptr) {
        return -1;
      }

      // Hex bitmap in comma-separated 32-bit groups, e.g. "00000000,0000ffff".
      int count = 0;
      for (int c = fgetc(file); c != EOF; c = fgetc(file)) {
        int nibble = -1;
        if (c >= '0' && c <= '9') nibble = c - '0';
        else if (c >= 'a' && c <= 'f') nibble = c - 'a' + 10;
        if (nibble >= 0) {
          count += __builtin_popcount(static_cast<unsigned>(nibble));
        }
      }
      fclose(file);
      return count;
    }
  }

  /**
   * \brief Verify the calling thread's CPU affinity is restored after ncclCommInitRank.
   *
   * Regression guard for NCCL issue #2033 / AICOMRCCL-1537: initTransportsRank() must
   * restore the mask it saved before applying the GPU-local one. Single-rank
   * ncclCommInitRank runs initTransportsRank on the calling thread, so a leak is
   * observable here. The check only bites when the GPU-local CPU set is a strict
   * subset of the process mask (multi-NUMA hosts); elsewhere the invariant still holds
   * but the regression is not exercised, which the test reports so that a pass is not
   * mistaken for coverage.
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

      // cpu_set_t is fixed at CPU_SETSIZE, so on larger hosts the widening is partial.
      // That only narrows the test's reach; dynamic CPU_ALLOC sets are out of scope.
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

      int localCpus = GetDeviceLocalCpuCount(0);
      if (localCpus < 0) {
        TEST_WARN("Could not read the NUMA-local CPU set of GPU 0 from sysfs, cannot tell "
                  "whether this host narrows the affinity mask.");
      } else if (localCpus >= CPU_COUNT(&before)) {
        TEST_WARN("GPU 0 is local to all %d CPUs of the process mask, so RCCL never narrows it "
                  "here: the invariant is still checked, but the regression is not exercised. "
                  "Real coverage requires a multi-NUMA host.", CPU_COUNT(&before));
      }

      ncclComm_t comm;
      ncclUniqueId id;
      NCCLCHECK(ncclGetUniqueId(&id));
      HIPCALL(hipSetDevice(0));
      NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

      cpu_set_t after;
      CPU_ZERO(&after);
      // Destroy the communicator before asserting, and keep errno across it.
      int getAffinityStatus = sched_getaffinity(0, sizeof(after), &after);
      int getAffinityErrno  = errno;

      NCCLCHECK(ncclCommDestroy(comm));

      ASSERT_EQ(getAffinityStatus, 0)
          << "sched_getaffinity failed: " << strerror(getAffinityErrno);
      ASSERT_TRUE(CPU_EQUAL(&before, &after))
          << "CPU affinity was not restored after ncclCommInitRank "
          << "(NCCL issue #2033 / AICOMRCCL-1537 regression)";
    });
  }
}
