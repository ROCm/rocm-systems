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
    // Fill 'mask' with the CPUs in the NUMA node the device hangs off, and return the
    // number of CPUs set, or -1 when sysfs does not expose it. RCCL narrows the calling
    // thread to the intersection of the process mask and this set, so the behavior is
    // only observable when this set is a strict subset of the process mask.
    int GetDeviceLocalCpuMask(int device, cpu_set_t* mask)
    {
      CPU_ZERO(mask);

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

      // Hex bitmap in comma-separated 32-bit groups, most-significant group first,
      // e.g. "00000000,0000ffff". Collect the nibbles first, then assign bit positions
      // from the least-significant (rightmost) nibble upward.
      char nibbles[CPU_SETSIZE / 4];
      int numNibbles = 0;
      for (int c = fgetc(file); c != EOF; c = fgetc(file)) {
        int value = -1;
        if (c >= '0' && c <= '9') value = c - '0';
        else if (c >= 'a' && c <= 'f') value = c - 'a' + 10;
        if (value >= 0 && numNibbles < static_cast<int>(sizeof(nibbles))) {
          nibbles[numNibbles++] = static_cast<char>(value);
        }
      }
      fclose(file);

      int count = 0;
      for (int i = 0; i < numNibbles; ++i) {
        int value = nibbles[numNibbles - 1 - i];
        for (int bit = 0; bit < 4; ++bit) {
          if (value & (1 << bit)) {
            int cpu = i * 4 + bit;
            if (cpu < CPU_SETSIZE) {
              CPU_SET(cpu, mask);
              ++count;
            }
          }
        }
      }
      return count;
    }
  }

  /**
   * \brief Verify the calling thread stays pinned to the GPU-local NUMA CPUs after ncclCommInitRank.
   *
   * Guards the intentional RCCL 2.29.7 comm-init behavior restored in commit
   * 1f36e555 (AICOMRCCL-1537): initTransportsRank() saves the caller's mask, pins the
   * calling thread to the GPU-local NUMA CPUs so RCCL's host buffers are allocated
   * locally, and on the exit path re-applies that GPU-local mask rather than restoring
   * the original process mask. Single-rank ncclCommInitRank runs initTransportsRank on
   * the calling thread, so the pinning is observable here. The check only bites when the
   * GPU-local CPU set is a strict subset of the process mask (multi-NUMA hosts);
   * elsewhere the invariant still holds but the behavior is not exercised, which the test
   * reports so that a pass is not mistaken for coverage.
   * ******************************************************************************************/
  TEST(CpuAffinity, PinnedToLocalAfterInitRank)
  {
    RUN_ISOLATED_TEST("CpuAffinity_PinnedToLocalAfterInitRank", []()
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

      cpu_set_t localMask;
      int localCpus = GetDeviceLocalCpuMask(0, &localMask);

      // RCCL pins the calling thread to (process mask) AND (GPU-local mask). Predict that
      // set from what we can read, so we can assert the exact mask below.
      cpu_set_t expected;
      CPU_ZERO(&expected);
      bool haveExpected = false;
      if (localCpus < 0) {
        TEST_WARN("Could not read the NUMA-local CPU set of GPU 0 from sysfs, cannot tell "
                  "whether this host narrows the affinity mask.");
      } else {
        CPU_AND(&expected, &before, &localMask);
        haveExpected = true;
        if (CPU_COUNT(&expected) == 0) {
          // Empty intersection: RCCL's ncclOsCpuCount() guard skips the affinity call, so
          // the calling thread keeps the process mask.
          CPU_ZERO(&expected);
          memcpy(&expected, &before, sizeof(before));
          TEST_WARN("GPU 0 has no NUMA-local CPU in the process mask, so RCCL never pins "
                    "here: the invariant is still checked, but the behavior is not exercised.");
        } else if (CPU_COUNT(&expected) >= CPU_COUNT(&before)) {
          TEST_WARN("GPU 0 is local to all %d CPUs of the process mask, so RCCL never narrows it "
                    "here: the invariant is still checked, but the behavior is not exercised. "
                    "Real coverage requires a multi-NUMA host.", CPU_COUNT(&before));
        }
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

      // RCCL only ever narrows the mask, never widens it: 'after' must be a subset of 'before'.
      cpu_set_t afterOutsideBefore;
      CPU_ZERO(&afterOutsideBefore);
      CPU_XOR(&afterOutsideBefore, &after, &before);
      CPU_AND(&afterOutsideBefore, &afterOutsideBefore, &after);
      ASSERT_EQ(CPU_COUNT(&afterOutsideBefore), 0)
          << "CPU affinity gained CPUs outside the original process mask after ncclCommInitRank";

      if (haveExpected) {
        // Core regression guard (AICOMRCCL-1537): the calling thread stays pinned to the
        // GPU-local NUMA CPUs rather than being restored to the original process mask.
        ASSERT_TRUE(CPU_EQUAL(&expected, &after))
            << "Calling thread was not left pinned to the GPU-local NUMA CPUs after "
            << "ncclCommInitRank (expected " << CPU_COUNT(&expected) << " CPUs, got "
            << CPU_COUNT(&after) << "): RCCL 2.29.7 comm-init affinity behavior "
            << "(AICOMRCCL-1537) regressed";
      }
    });
  }
}
