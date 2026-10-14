/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// In-place AllGather and ReduceScatter on symmetric windows, with per-rank slices that are and are
// not a multiple of 16 bytes.
//
// In place, one operand sits rank * sliceBytes into the shared window, so the relative input/output
// offset differs by rank. Every rank must still pick the same symmetric kernel: ranks that disagree
// launch grids that never meet at the LSA barrier, which shows up here as the isolated child timing
// out. NCCL_SYM_TMA_ENABLE=2 narrows the choice to the Tensor Data Mover kernels wherever they are
// valid, so on gfx1250 a 4 MiB slice reaches them. Elsewhere the cases still check in-place results.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/SymmetricBufferHelpers.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{
  namespace
  {
    constexpr int kMaxRanks = 8;

    // 4 MiB per rank, and 4 MiB + 12 B, which leaves every rank but 0 off the 16 B grid.
    constexpr size_t kAlignedSlice   = 1 << 20;
    constexpr size_t kUnalignedSlice = kAlignedSlice + 3;

    constexpr std::chrono::seconds kTimeout{240};

    // Prime, so a slice-aligned misplacement still changes the values. Sums stay exact in float.
    constexpr size_t kPeriod = 1021;

    float Value(int rank, size_t idx)
    {
      return static_cast<float>(rank + 1) + static_cast<float>(idx % kPeriod);
    }

    float SumOverRanks(int nRanks, size_t idx)
    {
      return static_cast<float>(nRanks * (nRanks + 1)) / 2.0f + static_cast<float>(nRanks) * (idx % kPeriod);
    }

    void RunInPlace(ncclFunc_t coll, size_t sliceCount)
    {
      int numDevices = 0;
      HIP_CHECK(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) GTEST_SKIP() << "This test requires at least 2 GPUs.";
      const int    nRanks     = std::min(numDevices, kMaxRanks);
      const size_t totalCount = sliceCount * nRanks;
      const size_t totalBytes = totalCount * sizeof(float);

      std::vector<ncclComm_t> comms(nRanks);
      ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), nRanks, nullptr));

      std::vector<hipStream_t>             streams(nRanks);
      std::vector<RCCLTestHelpers::SymBuf> bufs(nRanks);
      bool                                 allocOk = true;
      for (int i = 0; i < nRanks; i++)
      {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        bufs[i].comm = comms[i];
        allocOk      = ncclMemAlloc(&bufs[i].ptr, totalBytes) == ncclSuccess && allocOk;
      }

      // Window registration is collective, so every rank registers inside one group.
      ncclResult_t regRes = allocOk ? ncclSuccess : ncclSystemError;
      if (allocOk)
      {
        std::vector<ncclResult_t> rr(nRanks, ncclSuccess);
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          rr[i] = ncclCommWindowRegister(comms[i], bufs[i].ptr, totalBytes, &bufs[i].win, NCCL_WIN_COLL_SYMMETRIC);
        }
        regRes = ncclGroupEnd();
        for (ncclResult_t r : rr)
          if (r != ncclSuccess) regRes = r;
      }

      std::string skip;
      if (regRes != ncclSuccess)
      {
        skip = std::string("symmetric windows unavailable (cuMem/VMM off or unsupported): ") +
               ncclGetErrorString(regRes);
      }
      else
      {
        std::vector<float> host(totalCount);
        for (int i = 0; i < nRanks; i++)
        {
          // ReduceScatter reads every rank's whole buffer; AllGather only each rank's own slice.
          for (size_t k = 0; k < totalCount; k++)
          {
            const bool own = static_cast<int>(k / sliceCount) == i;
            host[k] = coll == ncclFuncReduceScatter ? Value(i, k) : own ? Value(i, k % sliceCount) : -1.0f;
          }
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipMemcpy(bufs[i].ptr, host.data(), totalBytes, hipMemcpyHostToDevice));
        }

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          float* full  = static_cast<float*>(bufs[i].ptr);
          float* slice = full + i * sliceCount;
          if (coll == ncclFuncReduceScatter)
            ASSERT_EQ(ncclSuccess, ncclReduceScatter(full, slice, sliceCount, ncclFloat32, ncclSum, comms[i], streams[i]));
          else
            ASSERT_EQ(ncclSuccess, ncclAllGather(slice, full, sliceCount, ncclFloat32, comms[i], streams[i]));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipStreamSynchronize(streams[i]));
        }

        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipMemcpy(host.data(), bufs[i].ptr, totalBytes, hipMemcpyDeviceToHost));
          const size_t lo = coll == ncclFuncReduceScatter ? i * sliceCount : 0;
          const size_t hi = coll == ncclFuncReduceScatter ? lo + sliceCount : totalCount;
          size_t       wrong = 0, firstWrong = hi;
          for (size_t k = lo; k < hi; k++)
          {
            const float expected = coll == ncclFuncReduceScatter ? SumOverRanks(nRanks, k)
                                                                 : Value(static_cast<int>(k / sliceCount), k % sliceCount);
            if (host[k] != expected && wrong++ == 0) firstWrong = k;
          }
          EXPECT_EQ(0u, wrong) << "rank " << i << ": " << wrong << " wrong elements, first at " << firstWrong;
        }
      }

      for (auto& b : bufs) b.release();
      for (int i = 0; i < nRanks; i++)
      {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamDestroy(streams[i]));
      }
      for (auto& c : comms) ASSERT_EQ(ncclSuccess, ncclCommDestroy(c));
      if (!skip.empty()) GTEST_SKIP() << skip;
    }

    ProcessIsolatedTestRunner::TestConfig InPlaceCase(const char* name, ncclFunc_t coll, size_t sliceCount)
    {
      return ProcessIsolatedTestRunner::TestConfig(name, [coll, sliceCount]() { RunInPlace(coll, sliceCount); })
        .withEnvironment({{"NCCL_SYM_TMA_ENABLE", "2"}, {"RCCL_IGNORE_ARCH_TABLE", "1"}})
        .withTimeout(kTimeout);
    }
  }  // namespace

  TEST(ReduceScatter, SymmetricInPlaceSlices)
  {
    RUN_ISOLATED_TESTS(InPlaceCase("ReduceScatterAlignedSlice", ncclFuncReduceScatter, kAlignedSlice),
                       InPlaceCase("ReduceScatterUnalignedSlice", ncclFuncReduceScatter, kUnalignedSlice));
  }

  TEST(AllGather, SymmetricInPlaceSlices)
  {
    RUN_ISOLATED_TESTS(InPlaceCase("AllGatherAlignedSlice", ncclFuncAllGather, kAlignedSlice),
                       InPlaceCase("AllGatherUnalignedSlice", ncclFuncAllGather, kUnalignedSlice));
  }
}  // namespace RcclUnitTesting
