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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/EnvVars.hpp"
#include "common/IsolatedChildSkip.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/SymmetricBufferHelpers.hpp"
#include "common/SymmetricKernelLog.hpp"
#include "common/TestChecks.hpp"
#include "device/tdm/tdmCopy.h"  // TDM_TOOLCHAIN_AVAILABLE
#include "rccl_common.h"  // rcclGetCollImplInfo, rcclAddonAlgos_t

namespace RcclUnitTesting
{
  // Reopens NCCL_DEBUG_FILE on the next log. Exported from src/debug.cc.
  extern "C" void ncclResetDebugInitInternal();

  namespace
  {
    constexpr int kMaxRanks = 8;

    // 4 MiB per rank, and 4 MiB + 8 B, which puts every odd rank off the 16 B grid.
    constexpr size_t kAlignedSlice   = 1 << 20;
    constexpr size_t kUnalignedSlice = kAlignedSlice + 2;

    constexpr char kSkipFileEnv[] = "RCCL_UT_SYM_INPLACE_SKIP_FILE";

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

    // EnvVars reports 0 GPUs in a re-exec'd child, and a skipped child scores as a pass, so gate in the parent.
    bool TooFewGpus()
    {
      if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr)
      {
        return false;
      }
      return EnvVars().GetNumDetectedGpus() < 2;
    }

    bool IsGfx1250()
    {
      hipDeviceProp_t prop;
      return hipGetDeviceProperties(&prop, 0) == hipSuccess && std::strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
    }

    // Kernel named by the last symmetric TUNING line rank 0 logged for func, or "" if none ran.
    std::string SymmetricKernelFromLog(const std::string& path, const std::string& func)
    {
      std::ifstream f(path);
      return LastSymmetricKernel(f, func);
    }

    void RunInPlace(ncclFunc_t coll, size_t sliceCount)
    {
      int numDevices = 0;
      HIP_CHECK(hipGetDeviceCount(&numDevices));
      ASSERT_GE(numDevices, 2) << "the parent gate saw at least 2 GPUs";
      const bool        gfx1250  = IsGfx1250();
      const std::string funcName = coll == ncclFuncReduceScatter ? "ReduceScatter" : "AllGather";
      char              logPath[256];
      snprintf(logPath, sizeof(logPath), "/tmp/rccl_sym_inplace_%d_%s.log", (int)getpid(), funcName.c_str());
      remove(logPath);
      setenv("NCCL_DEBUG_FILE", logPath, 1);
      ncclResetDebugInitInternal();
      const int    nRanks     = std::min(numDevices, kMaxRanks);
      const size_t totalCount = sliceCount * nRanks;
      const size_t totalBytes = totalCount * sizeof(float);

      std::vector<ncclComm_t> comms(nRanks);
      ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), nRanks, nullptr));

      std::vector<hipStream_t> streams(nRanks);
      for (int i = 0; i < nRanks; i++)
      {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
      }

      std::vector<RCCLTestHelpers::SymBuf> bufs;
      const ncclResult_t regRes = RCCLTestHelpers::ncclSymBufAllocPerRank(comms, totalBytes, bufs);

      std::string skip;
      if (regRes != ncclSuccess)
      {
        // cuMem is on by default on gfx1250, so a failed registration there means the cases below never ran.
        EXPECT_FALSE(gfx1250) << "symmetric window registration failed: " << ncclGetErrorString(regRes);
        skip = std::string("symmetric windows unavailable (cuMem/VMM off or unsupported): ") +
               ncclGetErrorString(regRes);
      }
      else
      {
        // Every rank must report the same symmetric choice for its own in-place operands.
        std::vector<int> algos(nRanks, -1);
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          float* full     = static_cast<float*>(bufs[i].ptr);
          float* slice    = full + i * sliceCount;
          int    protocol = 0, channels = 0;
          ASSERT_EQ(ncclSuccess, rcclGetCollImplInfo(comms[i], coll, sliceCount, ncclFloat32, ncclSum,
                                                     coll == ncclFuncReduceScatter ? full : slice,
                                                     coll == ncclFuncReduceScatter ? slice : full,
                                                     /*graphCapturing=*/0, &algos[i], &protocol, &channels));
          EXPECT_EQ(algos[0], algos[i]) << "rank " << i << " reports a different algo than rank 0";
        }
        const int kSym    = static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC);
        const int kSymTdm = static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM);
        if (gfx1250 && sliceCount * sizeof(float) % 16 == 0)
        {
          // The build's own SDK check, not the library's probe, so a lost TDM path cannot pass as vector.
          EXPECT_EQ(TDM_TOOLCHAIN_AVAILABLE ? kSymTdm : kSym, algos[0]) << "aligned slice";
        }
        else if (gfx1250)
        {
          EXPECT_TRUE(algos[0] == kSym || algos[0] == kSymTdm) << "unaligned slice reports algo " << algos[0];
        }

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

        const std::string kernel = SymmetricKernelFromLog(logPath, funcName);
        if (gfx1250)
        {
          EXPECT_FALSE(kernel.empty()) << "no " << funcName << " [Symmetric] line in " << logPath;
        }
        if (!kernel.empty())
        {
          EXPECT_EQ(algos[0] == kSymTdm, IsTdmSymmetricKernel(kernel))
            << "rank 0 ran " << kernel << ", reported " << algos[0];
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
      unsetenv("NCCL_DEBUG_FILE");
      ncclResetDebugInitInternal();
      if (!::testing::Test::HasFailure())
      {
        remove(logPath);
      }
      if (!skip.empty())
      {
        SkipOrHandToParent(kSkipFileEnv, skip);
      }
    }

    ProcessIsolatedTestRunner::TestConfig InPlaceCase(const std::string& name, ncclFunc_t coll, size_t sliceCount,
                                                      const std::string& skipFile)
    {
      return ProcessIsolatedTestRunner::TestConfig(name, [coll, sliceCount]() { RunInPlace(coll, sliceCount); })
        .withEnvironment({{"NCCL_SYM_TMA_ENABLE", "2"},
                          {"RCCL_IGNORE_ARCH_TABLE", "1"},
                          {"NCCL_DEBUG", "INFO"},
                          {"NCCL_DEBUG_SUBSYS", "TUNING"},
                          {kSkipFileEnv, skipFile}})
        .clearVariable("NCCL_ALGO")
        .withTimeout(kTimeout);
    }

    // Runs both slices in children; a child skip comes back through skipFile and is raised here.
    void RunBothSlices(const std::string& funcName, ncclFunc_t coll)
    {
      if (TooFewGpus())
      {
        GTEST_SKIP() << "This test requires at least 2 GPUs.";
      }
      const std::string skipFile = "/tmp/rccl_sym_inplace_skip_" + std::to_string(getpid());
      std::remove(skipFile.c_str());
      RUN_ISOLATED_TESTS(InPlaceCase(funcName + "AlignedSlice", coll, kAlignedSlice, skipFile),
                         InPlaceCase(funcName + "UnalignedSlice", coll, kUnalignedSlice, skipFile));
      const std::string reason = TakeHandedSkip(skipFile);
      if (!reason.empty())
      {
        GTEST_SKIP() << reason;
      }
    }
  }  // namespace

  TEST(ReduceScatter, SymmetricInPlaceSlices)
  {
    RunBothSlices("ReduceScatter", ncclFuncReduceScatter);
  }

  TEST(AllGather, SymmetricInPlaceSlices)
  {
    RunBothSlices("AllGather", ncclFuncAllGather);
  }
}  // namespace RcclUnitTesting
