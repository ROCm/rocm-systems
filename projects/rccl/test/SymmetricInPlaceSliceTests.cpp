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
//
// AllGather.SymmetricTmaStSkewedHead forces AllGather_TmaST on skewed buffers; it runs only on gfx1250.

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
      std::string   line, kernel;
      while (std::getline(f, line))
      {
        const std::string onLine = SymmetricKernelOnLine(line, func);
        if (!onLine.empty())
        {
          kernel = onLine;
        }
      }
      return kernel;
    }

    // AllReduce_RSxTmaLD_AGxTmaST on k x nRanks x CTAs deep chunks. A chunk is 4 warp tiles and a block 16
    // warps, so a warp runs k / 4 deep-loop iterations: past the first it prefetches its next tile, and with
    // k % 4 != 0 the last wave is short. kPeriod is prime, so a tile from the wrong iteration changes the sum.
    constexpr int    kArTdmCtas      = 1;
    constexpr size_t kArTdmChunk     = ncclSymkDeepBytePerChunk(sizeof(float));
    constexpr size_t kArTdmWaves[]   = {5, 11};
    constexpr size_t kArTdmGuard     = 4096;  // floats after recv that must keep the sentinel
    constexpr float  kArTdmSentinel  = -5.0f;
    constexpr float  kArTdmSendPad   = -3.0f;
    constexpr char   kArTdmSkipFileEnv[] = "RCCL_UT_SYM_AR_TDM_SKIP_FILE";

    float ReduceValue(int rank, size_t idx)
    {
      return static_cast<float>((rank + 1) * 4096 + static_cast<int>(idx % kPeriod));
    }

    float ReduceSum(int nRanks, size_t idx)
    {
      return static_cast<float>(nRanks * (nRanks + 1) / 2 * 4096 + nRanks * static_cast<int>(idx % kPeriod));
    }

    void RunAllReduceTdm()
    {
      if (!IsGfx1250() || !TDM_TOOLCHAIN_AVAILABLE)
      {
        SkipOrHandToParent(kArTdmSkipFileEnv, "AllReduce_RSxTmaLD_AGxTmaST needs gfx1250 and a TDM-capable build");
        return;
      }
      int numDevices = 0;
      HIP_CHECK(hipGetDeviceCount(&numDevices));
      ASSERT_GE(numDevices, 2) << "the parent gate saw at least 2 GPUs";
      const int nRanks = std::min(numDevices, kMaxRanks);
      char      logPath[256];
      snprintf(logPath, sizeof(logPath), "/tmp/rccl_sym_ar_tdm_%d.log", (int)getpid());
      remove(logPath);
      setenv("NCCL_DEBUG_FILE", logPath, 1);
      ncclResetDebugInitInternal();

      // The larger size also leaves kPeriod floats past the last full wave for the per-element tail.
      std::vector<size_t> counts;
      for (size_t k : kArTdmWaves) counts.push_back(k * nRanks * kArTdmCtas * kArTdmChunk / sizeof(float));
      counts.back() += kPeriod;
      const size_t maxCount  = *std::max_element(counts.begin(), counts.end());
      const size_t winFloats = maxCount + kArTdmGuard;
      const size_t winBytes  = winFloats * sizeof(float);

      std::vector<ncclComm_t> comms(nRanks);
      ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), nRanks, nullptr));
      std::vector<hipStream_t> streams(nRanks);
      for (int i = 0; i < nRanks; i++)
      {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
      }

      std::vector<RCCLTestHelpers::SymBuf> send, recv;
      ncclResult_t regRes = RCCLTestHelpers::ncclSymBufAllocPerRank(comms, winBytes, send);
      if (regRes == ncclSuccess)
      {
        regRes = RCCLTestHelpers::ncclSymBufAllocPerRank(comms, winBytes, recv);
      }
      // cuMem is on by default on gfx1250, so a failed registration is a failure, not a skip.
      EXPECT_EQ(ncclSuccess, regRes) << ncclGetErrorString(regRes);

      int                nCalls = 0;
      std::vector<float> host(winFloats);
      for (size_t count : counts)
      {
        if (regRes != ncclSuccess || ::testing::Test::HasFailure()) break;
        SCOPED_TRACE("count " + std::to_string(count));
        for (int i = 0; i < nRanks; i++)
        {
          int algo = -1, protocol = 0, channels = 0;
          ASSERT_EQ(ncclSuccess, rcclGetCollImplInfo(comms[i], ncclFuncAllReduce, count, ncclFloat32, ncclSum,
                                                     send[i].ptr, recv[i].ptr, /*graphCapturing=*/0, &algo, &protocol,
                                                     &channels));
          EXPECT_EQ(static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM), algo) << "rank " << i;
        }

        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          std::fill(host.begin(), host.end(), kArTdmSendPad);
          for (size_t k = 0; k < count; k++) host[k] = ReduceValue(i, k);
          HIP_CHECK(hipMemcpy(send[i].ptr, host.data(), winBytes, hipMemcpyHostToDevice));
          std::fill(host.begin(), host.end(), kArTdmSentinel);
          HIP_CHECK(hipMemcpy(recv[i].ptr, host.data(), winBytes, hipMemcpyHostToDevice));
        }

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          ASSERT_EQ(ncclSuccess,
                    ncclAllReduce(send[i].ptr, recv[i].ptr, count, ncclFloat32, ncclSum, comms[i], streams[i]));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        nCalls++;
        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipStreamSynchronize(streams[i]));
        }

        for (int i = 0; i < nRanks; i++)
        {
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipMemcpy(host.data(), recv[i].ptr, winBytes, hipMemcpyDeviceToHost));
          size_t wrong = 0, firstWrong = 0;
          for (size_t k = 0; k < winFloats; k++)
          {
            const float expected = k < count ? ReduceSum(nRanks, k) : kArTdmSentinel;
            if (host[k] != expected && wrong++ == 0) firstWrong = k;
          }
          EXPECT_EQ(0u, wrong) << "rank " << i << ": " << wrong << " wrong floats, first at " << firstWrong
                               << " (count " << count << ")";
        }
      }

      // Rank 0 logs one line per launch; each must be the forced kernel on the pinned block count.
      int           logged = 0;
      std::ifstream log(logPath);
      for (std::string line; std::getline(log, line);)
      {
        const std::string kernel = SymmetricKernelOnLine(line, "AllReduce");
        if (kernel.empty()) continue;
        logged++;
        EXPECT_EQ("AllReduce_RSxTmaLD_AGxTmaST", kernel) << line;
        EXPECT_EQ(std::to_string(kArTdmCtas), TokenAfter(line, "nchannels ")) << line;
      }
      EXPECT_EQ(nCalls, logged) << "symmetric AllReduce lines in " << logPath;

      send.clear();
      recv.clear();
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

    // AllGather_TmaST with send and recv both skewed by one head, so the 256 B tier peels a nonzero head
    // and starts its tiles mid-slice. Out of place at an odd rank count and in place at an even one leave
    // a partial peer pass for any UnrollPeers of 2 or 4.
    constexpr int    kTmaCtas  = 4;
    constexpr size_t kTmaSpan  = kTmaCtas * static_cast<size_t>(ncclSymkAlign256BDeepBytePerChunk);
    constexpr size_t kMaxSkew  = 256;
    constexpr size_t kGuard    = 64 << 10;
    constexpr float  kSentinel = -7.0f;
    constexpr float  kSendPad  = -3.0f;  // send-window bytes outside the slice, so an over-read shows up
    // Per-rank bytes, all multiples of 256 so the tier engages on every rank: an exact span multiple
    // (the head pushes one wave to the lower tiers), just under one, and one past it by 43 KiB.
    constexpr size_t kTmaSlices[] = {3 * kTmaSpan, 4 * kTmaSpan - 256, 3 * kTmaSpan + (43 << 10)};
    constexpr size_t kTmaSkews[]  = {16, 112, 240};

    constexpr char kTmaSkipFileEnv[] = "RCCL_UT_SYM_TMAST_SKIP_FILE";

    float GatherValue(int rank, size_t idx)
    {
      return static_cast<float>((rank + 1) * 4096 + static_cast<int>(idx % kPeriod));
    }

    enum class TmaRanks
    {
      kAll,
      kOdd,
      kEven
    };

    void RunTmaStSkewed(TmaRanks ranks, bool inPlace)
    {
      if (!IsGfx1250() || !TDM_TOOLCHAIN_AVAILABLE)
      {
        SkipOrHandToParent(kTmaSkipFileEnv, "AllGather_TmaST needs gfx1250 and a TDM-capable build");
        return;
      }
      int numDevices = 0;
      HIP_CHECK(hipGetDeviceCount(&numDevices));
      ASSERT_GE(numDevices, 3) << "the parent gate saw at least 3 GPUs";
      const int all    = std::min(numDevices, kMaxRanks);
      const int nRanks = ranks == TmaRanks::kAll ? all : ranks == TmaRanks::kOdd ? all - 1 + all % 2 : all - all % 2;
      char logPath[256];
      snprintf(logPath, sizeof(logPath), "/tmp/rccl_sym_tmast_%d.log", (int)getpid());
      remove(logPath);
      setenv("NCCL_DEBUG_FILE", logPath, 1);
      ncclResetDebugInitInternal();

      const size_t maxSlice = *std::max_element(std::begin(kTmaSlices), std::end(kTmaSlices));
      const size_t recvBytes = kMaxSkew + maxSlice * nRanks + kGuard;
      const size_t sendBytes = kMaxSkew + maxSlice + kGuard;

      std::vector<ncclComm_t> comms(nRanks);
      ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), nRanks, nullptr));
      std::vector<hipStream_t> streams(nRanks);
      for (int i = 0; i < nRanks; i++)
      {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
      }

      std::vector<RCCLTestHelpers::SymBuf> recv, send;
      ncclResult_t regRes = RCCLTestHelpers::ncclSymBufAllocPerRank(comms, recvBytes, recv);
      if (regRes == ncclSuccess && !inPlace)
      {
        regRes = RCCLTestHelpers::ncclSymBufAllocPerRank(comms, sendBytes, send);
      }
      // cuMem is on by default on gfx1250, so a failed registration is a failure, not a skip.
      EXPECT_EQ(ncclSuccess, regRes) << ncclGetErrorString(regRes);

      int nCalls = 0;
      for (size_t sliceBytes : kTmaSlices)
      {
        for (size_t skew : kTmaSkews)
        {
          if (regRes != ncclSuccess || ::testing::Test::HasFailure()) break;
          SCOPED_TRACE("slice " + std::to_string(sliceBytes) + " B, skew " + std::to_string(skew) + " B");
          const size_t sliceCount = sliceBytes / sizeof(float);
          const size_t skewCount  = skew / sizeof(float);
          const size_t totalCount = sliceCount * nRanks;
          auto sendOf = [&](int i) {
            return inPlace ? static_cast<float*>(recv[i].ptr) + skewCount + i * sliceCount
                           : static_cast<float*>(send[i].ptr) + skewCount;
          };

          std::vector<int> algos(nRanks, -1);
          for (int i = 0; i < nRanks; i++)
          {
            int protocol = 0, channels = 0;
            ASSERT_EQ(ncclSuccess, rcclGetCollImplInfo(comms[i], ncclFuncAllGather, sliceCount, ncclFloat32, ncclSum,
                                                       sendOf(i), static_cast<float*>(recv[i].ptr) + skewCount,
                                                       /*graphCapturing=*/0, &algos[i], &protocol, &channels));
            EXPECT_EQ(static_cast<int>(rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM), algos[i]) << "rank " << i;
          }

          std::vector<float> host(recvBytes / sizeof(float));
          for (int i = 0; i < nRanks; i++)
          {
            HIP_CHECK(hipSetDevice(i));
            std::fill(host.begin(), host.end(), inPlace ? kSentinel : kSendPad);
            float* own = inPlace ? host.data() + skewCount + i * sliceCount : host.data() + skewCount;
            for (size_t k = 0; k < sliceCount; k++) own[k] = GatherValue(i, k);
            if (inPlace)
            {
              HIP_CHECK(hipMemcpy(recv[i].ptr, host.data(), recvBytes, hipMemcpyHostToDevice));
            }
            else
            {
              HIP_CHECK(hipMemcpy(send[i].ptr, host.data(), sendBytes, hipMemcpyHostToDevice));
              std::fill(host.begin(), host.end(), kSentinel);
              HIP_CHECK(hipMemcpy(recv[i].ptr, host.data(), recvBytes, hipMemcpyHostToDevice));
            }
          }

          ASSERT_EQ(ncclSuccess, ncclGroupStart());
          for (int i = 0; i < nRanks; i++)
          {
            HIP_CHECK(hipSetDevice(i));
            ASSERT_EQ(ncclSuccess, ncclAllGather(sendOf(i), static_cast<float*>(recv[i].ptr) + skewCount, sliceCount,
                                                 ncclFloat32, comms[i], streams[i]));
          }
          ASSERT_EQ(ncclSuccess, ncclGroupEnd());
          nCalls++;
          for (int i = 0; i < nRanks; i++)
          {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamSynchronize(streams[i]));
          }

          // The whole window: sentinel before the skewed base and after the gathered span, data between.
          for (int i = 0; i < nRanks; i++)
          {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMemcpy(host.data(), recv[i].ptr, recvBytes, hipMemcpyDeviceToHost));
            size_t wrong = 0, firstWrong = 0;
            for (size_t k = 0; k < host.size(); k++)
            {
              const bool  inData   = k >= skewCount && k < skewCount + totalCount;
              const size_t d       = k - skewCount;
              const float expected = inData ? GatherValue(static_cast<int>(d / sliceCount), d % sliceCount) : kSentinel;
              if (host[k] != expected && wrong++ == 0) firstWrong = k;
            }
            EXPECT_EQ(0u, wrong) << "rank " << i << ": " << wrong << " wrong floats, first at window float "
                                 << firstWrong << " (data starts at " << skewCount << ")";
          }
        }
      }

      // Rank 0 logs one line per launch; each must be the forced kernel on the pinned block count.
      int  logged = 0;
      std::ifstream log(logPath);
      for (std::string line; std::getline(log, line);)
      {
        const std::string kernel = SymmetricKernelOnLine(line, "AllGather");
        if (kernel.empty()) continue;
        logged++;
        EXPECT_EQ("AllGather_TmaST", kernel) << line;
        EXPECT_EQ(std::to_string(kTmaCtas), TokenAfter(line, "nchannels ")) << line;
      }
      EXPECT_EQ(nCalls, logged) << "symmetric AllGather lines in " << logPath;

      recv.clear();
      send.clear();
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
    }

    ProcessIsolatedTestRunner::TestConfig TmaStCase(const std::string& name, TmaRanks ranks, bool inPlace,
                                                    const std::string& skipFile)
    {
      return ProcessIsolatedTestRunner::TestConfig(name, [ranks, inPlace]() { RunTmaStSkewed(ranks, inPlace); })
        .withEnvironment({{"NCCL_SYM_KERNEL", "AllGather_TmaST"},
                          {"NCCL_SYM_CTAS", std::to_string(kTmaCtas)},
                          {"RCCL_IGNORE_ARCH_TABLE", "1"},
                          {"NCCL_DEBUG", "INFO"},
                          {"NCCL_DEBUG_SUBSYS", "TUNING"},
                          {kTmaSkipFileEnv, skipFile}})
        .clearVariable("NCCL_ALGO")
        .clearVariable("NCCL_SYM_TMA_ENABLE")
        .withTimeout(kTimeout);
    }
  }  // namespace

  TEST(ReduceScatter, SymmetricInPlaceSlices)
  {
    RunBothSlices("ReduceScatter", ncclFuncReduceScatter);
  }

  // gfx1250 only: forces the TDM AllReduce so each warp runs more than one deep-loop iteration.
  TEST(AllReduce, SymmetricTmaLdPrefetch)
  {
    if (TooFewGpus())
    {
      GTEST_SKIP() << "This test requires at least 2 GPUs.";
    }
    const std::string skipFile = "/tmp/rccl_sym_ar_tdm_skip_" + std::to_string(getpid());
    std::remove(skipFile.c_str());
    RUN_ISOLATED_TESTS(ProcessIsolatedTestRunner::TestConfig("AllReduceTmaLdPrefetch", []() { RunAllReduceTdm(); })
                         .withEnvironment({{"NCCL_SYM_KERNEL", "AllReduce_RSxTmaLD_AGxTmaST"},
                                           {"NCCL_SYM_CTAS", std::to_string(kArTdmCtas)},
                                           {"RCCL_IGNORE_ARCH_TABLE", "1"},
                                           {"NCCL_DEBUG", "INFO"},
                                           {"NCCL_DEBUG_SUBSYS", "TUNING"},
                                           {kArTdmSkipFileEnv, skipFile}})
                         .clearVariable("NCCL_ALGO")
                         .clearVariable("NCCL_SYM_TMA_ENABLE")
                         .withTimeout(kTimeout));
    const std::string reason = TakeHandedSkip(skipFile);
    if (!reason.empty())
    {
      GTEST_SKIP() << reason;
    }
  }

  TEST(AllGather, SymmetricInPlaceSlices)
  {
    RunBothSlices("AllGather", ncclFuncAllGather);
  }

  // gfx1250 only. Odd ranks out of place and even ranks in place, so the tile tier's last peer pass is partial.
  TEST(AllGather, SymmetricTmaStSkewedHead)
  {
    if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) == nullptr && EnvVars().GetNumDetectedGpus() < 3)
    {
      GTEST_SKIP() << "This test requires at least 3 GPUs.";
    }
    const std::string skipFile = "/tmp/rccl_sym_tmast_skip_" + std::to_string(getpid());
    std::remove(skipFile.c_str());
    RUN_ISOLATED_TESTS(TmaStCase("OutOfPlaceAllRanks", TmaRanks::kAll, false, skipFile),
                       TmaStCase("OutOfPlaceOddRanks", TmaRanks::kOdd, false, skipFile),
                       TmaStCase("InPlaceEvenRanks", TmaRanks::kEven, true, skipFile));
    const std::string reason = TakeHandedSkip(skipFile);
    if (!reason.empty())
    {
      GTEST_SKIP() << reason;
    }
  }
}  // namespace RcclUnitTesting
