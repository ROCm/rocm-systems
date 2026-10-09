/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include "StandaloneUtils.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  /**
   * \brief Verify that ncclCommInitAll + AllReduce succeeds regardless of
   *        whether the GPUs support P2P/IPC (DDA IPC must be skipped
   *        gracefully on non-P2P topologies like RDNA3 over PCIe).
   */
  TEST(DdaIpc, CommInitDoesNotCrashWithoutP2p)
  {
    RUN_ISOLATED_TEST("CommInitDoesNotCrashWithoutP2p", []()
    {
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      // Cap at 8 GPUs (DDA IPC is hardcoded for exactly 8 ranks)
      int numRanks = std::min(numDevices, 8);

      // Create communicators -- this is where the crash occurred before the fix
      std::vector<ncclComm_t> comms(numRanks);
      ncclResult_t res = ncclCommInitAll(comms.data(), numRanks, nullptr);
      ASSERT_EQ(res, ncclSuccess) << "ncclCommInitAll failed: " << ncclGetErrorString(res);

      // Run a small AllReduce to verify the communicator is functional
      std::vector<float*> sendbuffs(numRanks);
      std::vector<float*> recvbuffs(numRanks);
      std::vector<hipStream_t> streams(numRanks);
      const int numElements = 1024;

      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipMalloc(&sendbuffs[i], numElements * sizeof(float)));
        HIPCALL(hipMalloc(&recvbuffs[i], numElements * sizeof(float)));
        HIPCALL(hipStreamCreate(&streams[i]));
        HIPCALL(hipMemset(sendbuffs[i], 0, numElements * sizeof(float)));
        HIPCALL(hipMemset(recvbuffs[i], 0, numElements * sizeof(float)));
      }

      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        NCCLCHECK(ncclAllReduce(sendbuffs[i], recvbuffs[i], numElements,
                                ncclFloat32, ncclSum, comms[i], streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipStreamSynchronize(streams[i]));
      }

      // Cleanup
      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipFree(sendbuffs[i]));
        HIPCALL(hipFree(recvbuffs[i]));
        HIPCALL(hipStreamDestroy(streams[i]));
      }
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    });
  }

  /**
   * \brief Verify that DDA IPC is explicitly disabled via RCCL_DDA_ENABLE=0
   *        and AllReduce still works using standard algorithms.
   */
  TEST(DdaIpc, AllReduceWorksWithDdaDisabled)
  {
    RUN_ISOLATED_TEST_WITH_ENV("AllReduceWorksWithDdaDisabled", []()
    {
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      int numRanks = std::min(numDevices, 8);

      std::vector<ncclComm_t> comms(numRanks);
      ncclResult_t res = ncclCommInitAll(comms.data(), numRanks, nullptr);
      ASSERT_EQ(res, ncclSuccess) << "ncclCommInitAll failed: " << ncclGetErrorString(res);

      std::vector<float*> sendbuffs(numRanks);
      std::vector<float*> recvbuffs(numRanks);
      std::vector<hipStream_t> streams(numRanks);
      const int numElements = 1024;

      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipMalloc(&sendbuffs[i], numElements * sizeof(float)));
        HIPCALL(hipMalloc(&recvbuffs[i], numElements * sizeof(float)));
        HIPCALL(hipStreamCreate(&streams[i]));

        // Fill send buffer: each rank sends (rank + 1) for all elements
        std::vector<float> hostData(numElements, static_cast<float>(i + 1));
        HIPCALL(hipMemcpy(sendbuffs[i], hostData.data(),
                          numElements * sizeof(float), hipMemcpyHostToDevice));
        HIPCALL(hipMemset(recvbuffs[i], 0, numElements * sizeof(float)));
      }

      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        NCCLCHECK(ncclAllReduce(sendbuffs[i], recvbuffs[i], numElements,
                                ncclFloat32, ncclSum, comms[i], streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      // Verify results: sum of (1 + 2 + ... + numRanks)
      float expectedSum = static_cast<float>(numRanks * (numRanks + 1) / 2);
      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipStreamSynchronize(streams[i]));

        std::vector<float> hostResult(numElements);
        HIPCALL(hipMemcpy(hostResult.data(), recvbuffs[i],
                          numElements * sizeof(float), hipMemcpyDeviceToHost));

        for (int e = 0; e < numElements; e++) {
          ASSERT_FLOAT_EQ(hostResult[e], expectedSum)
              << "Mismatch at rank " << i << " element " << e;
        }
      }

      // Cleanup
      for (int i = 0; i < numRanks; i++) {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipFree(sendbuffs[i]));
        HIPCALL(hipFree(recvbuffs[i]));
        HIPCALL(hipStreamDestroy(streams[i]));
      }
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    },
    {{"RCCL_DDA_ENABLE", "0"}});
  }
}
