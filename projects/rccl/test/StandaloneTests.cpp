/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "TestBed.hpp"
#include "StandaloneUtils.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  // FP8 reductions can round after every operation or accumulate in FP32 and round once on store.
  // Verify both modeled results are accepted exactly, while values admitted only by the high tolerance are rejected.
  TEST(Fp8Validation, AcceptsOnlyModeledAccumulationResults)
  {
    auto matchesEither = [](ncclDataType_t const dataType, float const actualValue,
                            float const stepwiseValue, float const fp32Value)
    {
      uint8_t actualStorage = 0;
      uint8_t stepwiseStorage = 0;
      uint8_t fp32Storage = 0;
      PtrUnion actual;
      PtrUnion stepwise;
      PtrUnion fp32;
      EXPECT_EQ(actual.Attach(&actualStorage), TEST_SUCCESS);
      EXPECT_EQ(stepwise.Attach(&stepwiseStorage), TEST_SUCCESS);
      EXPECT_EQ(fp32.Attach(&fp32Storage), TEST_SUCCESS);
      EXPECT_EQ(actual.Set(dataType, 0, 0, actualValue), TEST_SUCCESS);
      EXPECT_EQ(stepwise.Set(dataType, 0, 0, stepwiseValue), TEST_SUCCESS);
      EXPECT_EQ(fp32.Set(dataType, 0, 0, fp32Value), TEST_SUCCESS);

      bool isMatch = false;
      EXPECT_EQ(actual.IsEqual(dataType, 1, stepwise, &fp32, false, isMatch), TEST_SUCCESS);
      return isMatch;
    };

    // Both accumulation models are valid.
    EXPECT_TRUE(matchesEither(ncclFloat8e4m3, 1.75f, 1.75f, 1.625f));
    EXPECT_TRUE(matchesEither(ncclFloat8e4m3, 1.625f, 1.75f, 1.625f));
    EXPECT_TRUE(matchesEither(ncclFloat8e5m2, 1.75f, 1.75f, 1.5f));
    EXPECT_TRUE(matchesEither(ncclFloat8e5m2, 1.5f, 1.75f, 1.5f));

    // The former relative bounds accepted these missing-rank examples.
    EXPECT_FALSE(matchesEither(ncclFloat8e4m3, 6.0f, 8.0f, 8.0f));
    EXPECT_FALSE(matchesEither(ncclFloat8e5m2, 4.0f, 8.0f, 8.0f));
  }

  // GPU validation compares output against stepwise-FP8 and FP32-accumulation references.
  // Verify one intact reference is accepted, but corrupting both produces a mismatch.
  TEST(Fp8Validation, DeviceCorruptionMustAffectBothReferences)
  {
    for (ncclDataType_t const dataType : {ncclFloat8e4m3, ncclFloat8e5m2})
    {
      PtrUnion actual;
      PtrUnion stepwise;
      PtrUnion fp32;
      ASSERT_EQ(actual.AllocateGpuMem(1), TEST_SUCCESS);
      ASSERT_EQ(stepwise.AllocateGpuMem(1), TEST_SUCCESS);
      ASSERT_EQ(fp32.AllocateGpuMem(1), TEST_SUCCESS);
      ASSERT_EQ(actual.ClearGpuMem(1), TEST_SUCCESS);
      ASSERT_EQ(fp32.ClearGpuMem(1), TEST_SUCCESS);
      ASSERT_EQ(hipMemset(stepwise.ptr, 0xFF, 1), hipSuccess);

      // A damaged primary reference must not reject a valid alternative.
      size_t mismatches = 0;
      ASSERT_EQ(PtrUnion::IsEqualDevice(dataType, 1, actual.ptr, stepwise.ptr,
                                        fp32.ptr, mismatches, false),
                TEST_SUCCESS);
      EXPECT_EQ(mismatches, 0);

      // UT_DEVICE_DATA_FAULT corrupts both references, so neither can mask the fault.
      ASSERT_EQ(hipMemset(fp32.ptr, 0xFF, 1), hipSuccess);
      ASSERT_EQ(PtrUnion::IsEqualDevice(dataType, 1, actual.ptr, stepwise.ptr,
                                        fp32.ptr, mismatches, false),
                TEST_SUCCESS);
      EXPECT_EQ(mismatches, 1);

      EXPECT_EQ(actual.FreeGpuMem(), TEST_SUCCESS);
      EXPECT_EQ(stepwise.FreeGpuMem(), TEST_SUCCESS);
      EXPECT_EQ(fp32.FreeGpuMem(), TEST_SUCCESS);
    }
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
}
