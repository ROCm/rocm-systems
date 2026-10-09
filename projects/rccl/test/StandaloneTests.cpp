/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "TestBed.hpp"
#include "StandaloneUtils.hpp"
#include "VerifiableFp8.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  // HIP must stay out of the gtest parent (later tests fork without exec), so device checks
  // run in an isolated child and the GPU count comes from EnvVars' out-of-process probe.
  // Gate in the parent only: EnvVars reports 0 GPUs in a re-exec'd child, and a skipped
  // child is scored as a pass.
  static bool ShouldSkipForNoGpu()
  {
    if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr) return false;
    return EnvVars().maxGpus < 1;
  }

  // Non-reduce FP8 collectives (e.g. AllToAll) compare against one reference, bit-exactly:
  // values one FP8 step apart, inside the former absolute bound (< 9e-2), must not match.
  TEST(Fp8Validation, HostCompareIsExact)
  {
    auto matches = [](ncclDataType_t const dataType, float const actualValue, float const expectedValue)
    {
      uint8_t actualStorage = 0;
      uint8_t expectedStorage = 0;
      PtrUnion actual;
      PtrUnion expected;
      EXPECT_EQ(actual.Attach(&actualStorage), TEST_SUCCESS);
      EXPECT_EQ(expected.Attach(&expectedStorage), TEST_SUCCESS);
      EXPECT_EQ(actual.Set(dataType, 0, 0, actualValue), TEST_SUCCESS);
      EXPECT_EQ(expected.Set(dataType, 0, 0, expectedValue), TEST_SUCCESS);

      bool isMatch = false;
      EXPECT_EQ(actual.IsEqual(dataType, 1, expected, false, isMatch), TEST_SUCCESS);
      return isMatch;
    };
    for (ncclDataType_t const dataType : {ncclFloat8e4m3, ncclFloat8e5m2})
    {
      EXPECT_TRUE(matches(dataType, 1.5f, 1.5f));
      EXPECT_FALSE(matches(dataType, 1.75f, 1.5f));
    }
    EXPECT_FALSE(matches(ncclFloat8e4m3, 0.5f, 0.5625f));
    EXPECT_FALSE(matches(ncclFloat8e5m2, 0.25f, 0.3125f));
  }

  TEST(Fp8Validation, DeviceCompareIsExact)
  {
    if (ShouldSkipForNoGpu())
      GTEST_SKIP() << "Requires a GPU";

    RUN_ISOLATED_TEST("Fp8Validation_DeviceCompareIsExact", []()
    {
      for (ncclDataType_t const dataType : {ncclFloat8e4m3, ncclFloat8e5m2})
      {
        PtrUnion actual;
        PtrUnion expected;
        ASSERT_EQ(actual.AllocateGpuMem(1), TEST_SUCCESS);
        ASSERT_EQ(expected.AllocateGpuMem(1), TEST_SUCCESS);
        ASSERT_EQ(actual.ClearGpuMem(1), TEST_SUCCESS);
        ASSERT_EQ(expected.ClearGpuMem(1), TEST_SUCCESS);

        size_t mismatches = 0;
        ASSERT_EQ(PtrUnion::IsEqualDevice(dataType, 1, actual.ptr, expected.ptr, mismatches, false),
                  TEST_SUCCESS);
        EXPECT_EQ(mismatches, 0);

        ASSERT_EQ(hipMemset(expected.ptr, 0xFF, 1), hipSuccess);
        ASSERT_EQ(PtrUnion::IsEqualDevice(dataType, 1, actual.ptr, expected.ptr, mismatches, false),
                  TEST_SUCCESS);
        EXPECT_EQ(mismatches, 1);

        EXPECT_EQ(actual.FreeGpuMem(), TEST_SUCCESS);
        EXPECT_EQ(expected.FreeGpuMem(), TEST_SUCCESS);
      }
    });
  }

  // FP8 reductions validate against the verifiable generator's single reference. It must accept
  // its own expected values, report each corrupted element, and give every rank distinct inputs
  // so that a missing or duplicated rank changes the result (for Min/Max too).
  TEST(Fp8Validation, VerifiableReferenceDetectsErrors)
  {
    if (ShouldSkipForNoGpu())
      GTEST_SKIP() << "Requires a GPU";

    RUN_ISOLATED_TEST("Fp8Validation_VerifiableReferenceDetectsErrors", []()
    {
      ASSERT_EQ(hipSetDevice(0), hipSuccess);

      size_t const numElements = 4096;
      int    const rankCounts[] = {3, 8};
      uint64_t const seed = 0x5eed;
      uint8_t* bufGpu = nullptr;
      uint8_t* otherGpu = nullptr;
      int64_t* badEltN = nullptr;
      ASSERT_EQ(hipMalloc(&bufGpu, numElements), hipSuccess);
      ASSERT_EQ(hipMalloc(&otherGpu, numElements), hipSuccess);
      ASSERT_EQ(hipHostMalloc((void**)&badEltN, sizeof(int64_t)), hipSuccess);
      std::vector<uint8_t> rank0(numElements), rank1(numElements);

      for (ncclDataType_t const dataType : {ncclFloat8e4m3, ncclFloat8e5m2})
      for (ncclRedOp_t const redOp : {ncclSum, ncclProd, ncclMax, ncclMin})
      for (int const rankN : rankCounts)
      {
        SCOPED_TRACE(std::string(ncclDataTypeNames[dataType]) + " " + ncclRedOpNames[redOp] +
                     " ranks=" + std::to_string(rankN));
        auto verify = [&]() {
          *badEltN = -1;
          EXPECT_EQ(VerifiableFp8Verify(bufGpu, numElements, dataType, redOp, rankN,
                                        seed, 0, badEltN, nullptr), hipSuccess);
          EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
          return *badEltN;
        };

        ASSERT_EQ(VerifiableFp8PrepareExpected(bufGpu, numElements, dataType, redOp, rankN,
                                               seed, 0, nullptr), hipSuccess);
        EXPECT_EQ(verify(), 0);

        // Flip an exponent bit: the value changes by far more than any tolerance
        uint8_t byte = 0;
        ASSERT_EQ(hipMemcpy(&byte, bufGpu + 17, 1, hipMemcpyDeviceToHost), hipSuccess);
        byte ^= 0x08;
        ASSERT_EQ(hipMemcpy(bufGpu + 17, &byte, 1, hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(verify(), 1);

        ASSERT_EQ(VerifiableFp8PrepareInput(bufGpu, numElements, dataType, redOp, rankN, 0,
                                            seed, 0, nullptr), hipSuccess);
        ASSERT_EQ(VerifiableFp8PrepareInput(otherGpu, numElements, dataType, redOp, rankN, 1,
                                            seed, 0, nullptr), hipSuccess);
        ASSERT_EQ(hipMemcpy(rank0.data(), bufGpu, numElements, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(rank1.data(), otherGpu, numElements, hipMemcpyDeviceToHost), hipSuccess);
        size_t differing = 0;
        for (size_t i = 0; i < numElements; ++i) differing += (rank0[i] != rank1[i]);
        EXPECT_GT(differing, numElements / 2);
      }

      EXPECT_EQ(hipFree(bufGpu), hipSuccess);
      EXPECT_EQ(hipFree(otherGpu), hipSuccess);
      EXPECT_EQ(hipHostFree(badEltN), hipSuccess);
    });
  }

  /**
   * \brief Verify that each device is assigned to the right rank using ncclCommSplit API.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_RankCheck)
  {
    RUN_ISOLATED_TEST("SplitComms_RankCheck", []()
    {
      // Check for multi-gpu
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      // Initialize the original comms
      std::vector<ncclComm_t> comms(numDevices);
      NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

      // Split into new comms (round-robin)
      std::vector<ncclComm_t> subComms(numDevices);
      int numSubComms = 2;

      std::map<int, int> mapCounter;
      NCCLCHECK(ncclGroupStart());
      for (int localRank = 0; localRank < numDevices; localRank++) {
        NCCLCHECK(ncclCommSplit(comms[localRank], localRank % numSubComms, localRank, &subComms[localRank], NULL));
        mapCounter[localRank % numSubComms]++;
      }
      NCCLCHECK(ncclGroupEnd());

      // Check that new comms have correct subranks / ranks
      for (int i = 0; i < numDevices; i++) {
        int subCommRank, subCommNRank;
        NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
        NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));

        ASSERT_EQ(subCommRank, i / numSubComms);
        ASSERT_EQ(subCommNRank, mapCounter[i % numSubComms]);
      }

      // Clean up comms
      for (auto& subComm : subComms)
        NCCLCHECK(ncclCommDestroy(subComm));
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    });
  }

  /**
   * \brief Creates a communicator for each device and gathers them all in one rank.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_OneColor)
  {
    RUN_ISOLATED_TEST("SplitComms_OneColor", []()
    {
      // Check for multi-gpu
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      // Initialize the original comms
      std::vector<ncclComm_t> comms(numDevices);
      NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

      // Split into new comms (all of the same color)
      std::vector<ncclComm_t> subComms(numDevices);
      NCCLCHECK(ncclGroupStart());
      for (int localRank = 0; localRank < numDevices; localRank++)
        NCCLCHECK(ncclCommSplit(comms[localRank], 0, localRank, &subComms[localRank], NULL));
      NCCLCHECK(ncclGroupEnd());

      // Validate results
      for (int i = 0; i < numDevices; i++) {
        int originalRank, originalNRank;
        NCCLCHECK(ncclCommUserRank(comms[i], &originalRank));
        NCCLCHECK(ncclCommCount(comms[i], &originalNRank));

        int subCommRank, subCommNRank;
        NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
        NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));

        ASSERT_EQ(originalRank, subCommRank);
        ASSERT_EQ(originalNRank, subCommNRank);
      }

      // Clean up comms
      for (auto& subComm : subComms)
        NCCLCHECK(ncclCommDestroy(subComm));
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    });
  }

  /**
   * \brief Creates a communicator for each device and reduces them into (numDevices / 2) ranks.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_Reduce)
  {
    RUN_ISOLATED_TEST("SplitComms_Reduce", []()
    {
      // Check for multi-gpu
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      // Initialize the original comms
      std::vector<ncclComm_t> comms(numDevices);
      NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

      // Split into new comms
      int numReducedRanks = numDevices / 2;
      std::vector<ncclComm_t> subComms(numDevices);
      NCCLCHECK(ncclGroupStart());
      for (int localRank = 0; localRank < numDevices; localRank++)
        NCCLCHECK(ncclCommSplit(comms[localRank],
              localRank < numReducedRanks ? 0 : NCCL_SPLIT_NOCOLOR,
              localRank, &subComms[localRank], NULL));
      NCCLCHECK(ncclGroupEnd());

      // Validate results
      for (int i = 0; i < numDevices; i++) {
        int originalRank, originalNRank;
        NCCLCHECK(ncclCommUserRank(comms[i], &originalRank));
        NCCLCHECK(ncclCommCount(comms[i], &originalNRank));

        if (i < numReducedRanks) {
          int subCommRank, subCommNRank;
          NCCLCHECK(ncclCommUserRank(subComms[i], &subCommRank));
          NCCLCHECK(ncclCommCount(subComms[i], &subCommNRank));

          ASSERT_EQ(originalRank, subCommRank);
          ASSERT_EQ(subCommNRank, numReducedRanks);
        } else {
          ASSERT_EQ(subComms[i], nullptr);
        }
      }

      // Cleanup comms
      for (auto& subComm : subComms)
        NCCLCHECK(ncclCommDestroy(subComm));
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    });
  }

  /**
   * \brief Verify there is no regression in timing for each protocol [LL, LL128, Simple]
   * ******************************************************************************************/
  TEST(Standalone, RegressionTiming)
  {
    RUN_ISOLATED_TEST("RegressionTiming", []()
    {
      // timing
      using namespace std::chrono;
      using Clock = std::chrono::high_resolution_clock;
      int usElapsed, numIterations = 20, numWarmups = 5;

      // Check for 2 GPUs
      int numGpus;
      HIPCALL(hipGetDeviceCount(&numGpus));
      if (numGpus < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }
      hipDeviceProp_t devProp;
      HIPCALL(hipGetDeviceProperties(&devProp, 0));
      // Initialize RCCL
      constexpr int numRanks = 2;
      std::vector<ncclComm_t> comms(numRanks);
      std::vector<int*> gpuInput(numRanks);
      std::vector<int*> gpuOutput(numRanks);
      std::vector<hipStream_t> stream(numRanks);

      char *proto = std::getenv("NCCL_PROTO");
      const char* protocolList[3] = {"LL", "LL128", "Simple"};

      for (auto p : protocolList)
      {
        usElapsed = 0;
        if(strncmp("gfx12",devProp.gcnArchName,5) == 0) {
          setenv("NCCL_PROTO", "Simple", 1);
        } else {
          setenv("NCCL_PROTO", p, 1);
        }

        NCCLCHECK(ncclCommInitAll(comms.data(), numRanks, nullptr));

        // Prepare CPU data arrays
        int N = 1250;
        std::vector<int> cpuInput(N);
        std::vector<int> cpuExpected(N);
        for (int i = 0; i < N; i++) {
          cpuInput[i]    = i;
          cpuExpected[i] = 2 * i;
        }

        // Prepare GPU data arrays
        for (int rank = 0; rank < numRanks; rank++) {
          HIPCALL(hipSetDevice(rank));
          HIPCALL(hipStreamCreate(&stream[rank]));
          HIPCALL(hipMalloc((void**)&gpuInput[rank], N * sizeof(int)));
          HIPCALL(hipMalloc((void**)&gpuOutput[rank], N * sizeof(int)));
          HIPCALL(hipMemcpy(gpuInput[rank], cpuInput.data(), N * sizeof(int), hipMemcpyHostToDevice));
          HIPCALL(hipMemset(gpuOutput[rank], 0, N * sizeof(int)));
          HIPCALL(hipDeviceSynchronize());
        }

        for (int iter = -numWarmups; iter < numIterations; iter++) {

          for (int rank = 0; rank < numRanks; rank++) {
            HIPCALL(hipSetDevice(rank));
            HIPCALL(hipMemset(gpuOutput[rank], 0, N * sizeof(int)));
            HIPCALL(hipDeviceSynchronize());
          }

          // Initiate the allreduce
          NCCLCHECK(ncclGroupStart());
          for (int rank = 0; rank < numRanks; rank++)
            NCCLCHECK(ncclAllReduce(gpuInput[rank], gpuOutput[rank], N, ncclInt, ncclSum, comms[rank], stream[rank]));
          ncclResult_t res = ncclGroupEnd();

          if (res != ncclSuccess) continue;

          const auto start = Clock::now();

          // Wait for completion
          for (int rank = 0; rank < numRanks; rank++) {
            HIPCALL(hipStreamSynchronize(stream[rank]));
          }

          if (iter >= 0)
            usElapsed += duration_cast<microseconds>(Clock::now() - start).count();

          // Check results
          std::vector<int> cpuOutput(N);
          for (int rank = 0; rank < numRanks; rank++) {
            HIPCALL(hipMemcpy(cpuOutput.data(), gpuOutput[rank], N * sizeof(int), hipMemcpyDeviceToHost));
            HIPCALL(hipDeviceSynchronize());
            for (int i = 0; i < N; i++)
              ASSERT_EQ(cpuOutput[i], cpuExpected[i]);
          }
        }

        EXPECT_LT(usElapsed/(double)numIterations, 5000);
        printf("[ INFO     ] protocol: %s, average runtime: %f microseconds\n", p, usElapsed/(double)numIterations);
        // Release resources
        for (int rank = 0; rank < numRanks; rank++){
          HIPCALL(hipFree(gpuInput[rank]));
          HIPCALL(hipFree(gpuOutput[rank]));
          HIPCALL(hipStreamDestroy(stream[rank]));
          NCCLCHECK(ncclCommDestroy(comms[rank]));
        }
      }
      if (proto)
        setenv("NCCL_PROTO", proto, 1);
      else
        unsetenv("NCCL_PROTO");
    });
  }

  /**
   * \brief Verify rccl generic kernel stack size for each gfx architecture is less than the
   * expected MAX_STACK_SIZE.
   * ******************************************************************************************/
  TEST(Standalone, StackSize) {
    const char* mainKernel = "ncclDevKernel";

    // Look for the .co files
    std::vector<std::string> coFileList = splitString(executeCommand("find ../ -type f -name \"librccl*.co\""), '\n');

    // Check if the .co files exist in the build directory
    if (coFileList.empty())
      GTEST_SKIP() << "Skipping... Could not found required files in the build directory.";

    for (const auto& file : coFileList) {
      // Store the output in a list
      std::string cmd = std::string(ROCM_PATH) + "/llvm/bin/llvm-readelf --notes " + file;
      std::vector<std::string> metadata = splitString(executeCommand(cmd.c_str()), '\n');

      // Skip if llvm is not installed
      if (metadata.empty())
        GTEST_SKIP() << "Skipping... llvm is not found.";

      // Parse metadata from file and store it for each arch
      ArchInfo archInfo = parseMetadata(metadata);

      // iterate over each archs kernels
      for (const auto& kernel : archInfo.kernels) {
        if (kernel.name.find(mainKernel) != std::string::npos) {
          // Kernel stack size should be less than or equal to the maxStackSize value
          printf("[ INFO     ] Arch: %s Kernel: %s Size: %d\n", archInfo.archName.c_str(), kernel.name.c_str(), kernel.privateSegmentFixedSize);
          EXPECT_LE(kernel.privateSegmentFixedSize, archInfo.archName == "gfx90a" ? MAX_STACK_SIZE_gfx90a : MAX_STACK_SIZE);
        }
      }
    }
  }
  /**
   * \brief Verify the device associated with communicator in both single and multi-device scenarios
   * ******************************************************************************************/
  TEST(Standalone, CommCuDevice_Check)
  {
    RUN_ISOLATED_TEST("CommCuDevice_Check", []()
    {
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 1) {
        GTEST_SKIP() << "No devices available.";
      }

      // Test single comm initialization
      ncclComm_t comm;
      ncclUniqueId id;
      NCCLCHECK(ncclGetUniqueId(&id));
      HIPCALL(hipSetDevice(0));
      NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

      // Verify device assignment
      int device;
      NCCLCHECK(ncclCommCuDevice(comm, &device));
      ASSERT_EQ(device, 0);
      NCCLCHECK(ncclCommDestroy(comm));

      // Test multi-device scenario if available
      if (numDevices > 1) {
        std::vector<ncclComm_t> comms(numDevices);

        // Initialize all communicators at once
        NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

        // Verify device assignments
        for (int i = 0; i < numDevices; i++) {
          int assignedDevice;
          NCCLCHECK(ncclCommCuDevice(comms[i], &assignedDevice));
          ASSERT_EQ(assignedDevice, i);
        }

        // Clean up
        for (int i = 0; i < numDevices; i++) {
          NCCLCHECK(ncclCommDestroy(comms[i]));
        }
      }
    });
  }

  /**
   * \brief verifies that ncclCommUserRank correctly fails when provided with an invalid (null) communicator handle
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_RankCheck_Basic_Failure) {
    RUN_ISOLATED_TEST("SplitComms_RankCheck_Basic_Failure", []()
    {
      // Check for multi-gpu
      int numDevices;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
      }

      // Initialize the original comms
      std::vector<ncclComm_t> comms(numDevices);
      NCCLCHECK(ncclCommInitAll(comms.data(), numDevices, nullptr));

      // Create an invalid comm handle that will cause a failure
      ncclComm_t invalidComm = nullptr;

      // This NCCL_CHECK will fail because we're trying to query rank from a null communicator
      int rank;
      NCCLCHECK(ncclCommUserRank(invalidComm, &rank));

      // Clean up comms
      for (auto& comm : comms)
        NCCLCHECK(ncclCommDestroy(comm));
    });
  }

  static std::string nvlsLogPath;
  static void RemoveNvlsLog()
  {
    if (!nvlsLogPath.empty())
      std::remove(nvlsLogPath.c_str());
  }
  struct NvlsLogRemover
  {
    ~NvlsLogRemover() { RemoveNvlsLog(); }
  };

  // Init + AllReduce across all devices, then check the RCCL log (NCCL_DEBUG_FILE) for NVLS activity.
  static void RunAllReduceAndCheckNoNvls()
  {
    // A unique file created here, before RCCL reads NCCL_DEBUG_FILE at first init, so another
    // local user cannot pre-create the log and make the checks below read planted content.
    char logTemplate[] = "/tmp/rccl_nvls_enable_XXXXXX";
    int const logFd = mkstemp(logTemplate);
    ASSERT_GE(logFd, 0) << "mkstemp failed";
    close(logFd);
    std::string const logPath = logTemplate;
    nvlsLogPath = logPath;
    // The guard covers ASSERT_* early returns; HIPCALL failures exit(-1), which skips local
    // destructors but still runs atexit handlers.
    std::atexit(RemoveNvlsLog);
    NvlsLogRemover removeLog;
    ASSERT_EQ(setenv("NCCL_DEBUG_FILE", logPath.c_str(), 1), 0);

    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));

    std::vector<ncclComm_t> comms(numDevices);
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), numDevices, nullptr));

    constexpr int N = 1024;
    std::vector<int*> gpuInput(numDevices);
    std::vector<int*> gpuOutput(numDevices);
    std::vector<hipStream_t> stream(numDevices);
    for (int rank = 0; rank < numDevices; rank++) {
      std::vector<int> cpuInput(N, rank + 1);
      HIPCALL(hipSetDevice(rank));
      HIPCALL(hipStreamCreate(&stream[rank]));
      HIPCALL(hipMalloc((void**)&gpuInput[rank], N * sizeof(int)));
      HIPCALL(hipMalloc((void**)&gpuOutput[rank], N * sizeof(int)));
      HIPCALL(hipMemcpy(gpuInput[rank], cpuInput.data(), N * sizeof(int), hipMemcpyHostToDevice));
      HIPCALL(hipMemset(gpuOutput[rank], 0, N * sizeof(int)));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int rank = 0; rank < numDevices; rank++) {
      ASSERT_EQ(ncclSuccess,
                ncclAllReduce(gpuInput[rank], gpuOutput[rank], N, ncclInt, ncclSum, comms[rank], stream[rank]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    int const expected = numDevices * (numDevices + 1) / 2;
    for (int rank = 0; rank < numDevices; rank++) {
      HIPCALL(hipSetDevice(rank));
      HIPCALL(hipStreamSynchronize(stream[rank]));
      std::vector<int> cpuOutput(N);
      HIPCALL(hipMemcpy(cpuOutput.data(), gpuOutput[rank], N * sizeof(int), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; i++)
        ASSERT_EQ(cpuOutput[i], expected) << "rank " << rank << " element " << i;
      ncclResult_t asyncErr;
      ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[rank], &asyncErr));
      EXPECT_EQ(asyncErr, ncclSuccess);
    }

    for (int rank = 0; rank < numDevices; rank++) {
      HIPCALL(hipSetDevice(rank));
      HIPCALL(hipFree(gpuInput[rank]));
      HIPCALL(hipFree(gpuOutput[rank]));
      HIPCALL(hipStreamDestroy(stream[rank]));
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[rank]));
    }

    std::ifstream logFile(logPath);
    ASSERT_TRUE(logFile.is_open()) << "RCCL wrote no log to " << logPath;
    std::stringstream log;
    log << logFile.rdbuf();
    EXPECT_NE(log.str().find("Init COMPLETE"), std::string::npos) << "log did not capture communicator init";
    // Tripwires: both strings exist only in nvls.cc's `#if CUDART_VERSION >= 12010` branch, which
    // RCCL does not compile today. They fail only if a future change enables that branch.
    // NCCL's multicast bind failure, a WARN that fails initialization:
    EXPECT_EQ(log.str().find("Failed to bind NVLink SHARP"), std::string::npos);
    // Printed by NCCL's NVLS init whenever NCCL_NVLS_ENABLE is nonzero:
    EXPECT_EQ(log.str().find("NVLS multicast support is"), std::string::npos);
  }

  /**
   * \brief Verify NCCL_NVLS_ENABLE has no effect: RCCL does not implement NVLS, so no value
   * attempts a multicast bind or fails communicator init the way NCCL 2.29+ can.
   * ******************************************************************************************/
  TEST(Standalone, NvlsEnable_NoEffect)
  {
    // Gated here rather than inside the isolated body: RUN_ISOLATED_TESTS ends in
    // EXPECT_TRUE(), so a GTEST_SKIP() in the child is reported as a pass by the parent.
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    using Config = ProcessIsolatedTestRunner::TestConfig;
    auto nvlsEnableIs = [](const char* value) {
      return Config(std::string("NvlsEnable_NoEffect_") + value, RunAllReduceAndCheckNoNvls)
        .withEnvironment({{"NCCL_NVLS_ENABLE", value},
                          {"NCCL_DEBUG", "INFO"},
                          {"NCCL_DEBUG_SUBSYS", "INIT,NVLS,REG"}})
        .withTimeout(std::chrono::seconds(60));
    };
    RUN_ISOLATED_TESTS(nvlsEnableIs("0"), nvlsEnableIs("1"), nvlsEnableIs("2"));
  }
}
