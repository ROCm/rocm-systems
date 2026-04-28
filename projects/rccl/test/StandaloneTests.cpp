/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <thread>

#include "TestBed.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{
  /**
   * \brief Verify that each device is assigned to the right rank using ncclCommSplit API.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_RankCheck)
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
  }

  /**
   * \brief Creates a communicator for each device and gathers them all in one rank.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_OneColor)
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
  }

  /**
   * \brief Creates a communicator for each device and reduces them into (numDevices / 2) ranks.
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_Reduce)
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
  }

  /**
   * \brief Verify there is no regression in timing for each protocol [LL, LL128, Simple]
   * ******************************************************************************************/
  TEST(Standalone, RegressionTiming)
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
  }

  /**
   * \brief verifies that ncclCommUserRank correctly fails when provided with an invalid (null) communicator handle
   * ******************************************************************************************/
  TEST(Standalone, SplitComms_RankCheck_Basic_Failure) {
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
  }

  /**
   * \brief Grow a 1-rank communicator to 2 ranks and verify both newcomms work.
   *        Existing rank 0 and new rank 1 rendezvous via ncclCommGetUniqueId,
   *        perform an allreduce on the grown comm, then destroy cleanly.
   * ******************************************************************************************/
  TEST(Standalone, Grow_OneToTwo_CleanLifecycle)
  {
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    ncclUniqueId initId, growId;
    ncclComm_t comm0 = nullptr, newcomm0 = nullptr, newcomm1 = nullptr;
    ncclResult_t res0 = ncclSuccess, res1 = ncclSuccess;

    NCCLCHECK(ncclGetUniqueId(&initId));
    NCCLCHECK(ncclCommInitRank(&comm0, 1, initId, 0));
    NCCLCHECK(ncclCommGetUniqueId(comm0, &growId));

    std::thread t1([&]() {
      HIPCALL(hipSetDevice(1));
      res1 = ncclCommGrow(nullptr, 2, &growId, 1, &newcomm1, nullptr);
    });

    HIPCALL(hipSetDevice(0));
    res0 = ncclCommGrow(comm0, 2, &growId, -1, &newcomm0, nullptr);
    t1.join();

    ASSERT_EQ(res0, ncclSuccess);
    ASSERT_EQ(res1, ncclSuccess);
    ASSERT_NE(newcomm0, nullptr);
    ASSERT_NE(newcomm1, nullptr);

    // Verify the grown comm can perform collectives.
    float *sbuf0, *rbuf0, *sbuf1, *rbuf1;
    hipStream_t stream0, stream1;
    HIPCALL(hipSetDevice(0));
    HIPCALL(hipMalloc((void**)&sbuf0, sizeof(float)));
    HIPCALL(hipMalloc((void**)&rbuf0, sizeof(float)));
    HIPCALL(hipStreamCreate(&stream0));
    HIPCALL(hipSetDevice(1));
    HIPCALL(hipMalloc((void**)&sbuf1, sizeof(float)));
    HIPCALL(hipMalloc((void**)&rbuf1, sizeof(float)));
    HIPCALL(hipStreamCreate(&stream1));

    NCCLCHECK(ncclGroupStart());
    ASSERT_EQ(ncclAllReduce(sbuf0, rbuf0, 1, ncclFloat, ncclSum, newcomm0, stream0), ncclSuccess);
    ASSERT_EQ(ncclAllReduce(sbuf1, rbuf1, 1, ncclFloat, ncclSum, newcomm1, stream1), ncclSuccess);
    NCCLCHECK(ncclGroupEnd());

    HIPCALL(hipSetDevice(0));
    HIPCALL(hipStreamSynchronize(stream0));
    HIPCALL(hipStreamDestroy(stream0));
    HIPCALL(hipFree(sbuf0)); HIPCALL(hipFree(rbuf0));
    HIPCALL(hipSetDevice(1));
    HIPCALL(hipStreamSynchronize(stream1));
    HIPCALL(hipStreamDestroy(stream1));
    HIPCALL(hipFree(sbuf1)); HIPCALL(hipFree(rbuf1));

    ASSERT_EQ(ncclCommDestroy(newcomm0), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(newcomm1), ncclSuccess);
    ASSERT_EQ(ncclCommDestroy(comm0), ncclSuccess);
  }

  /**
   * \brief Grow a 4-rank communicator to 5 ranks. Existing ranks 1 and 2 are
   *        non-boundary in the parent — they MUST take the bootstrapInit
   *        nHandles=0 + parent path that rendezvous via the parent ring.
   *        Boundary ranks 0 and 3 receive the grow handle via bcastGrowHandle.
   *        After grow, every rank performs an allreduce and verifies sum.
   * ******************************************************************************************/
  TEST(Standalone, Grow_FourToFive_NonBoundaryPath)
  {
    constexpr int kParent = 4;
    constexpr int kChild  = 5;

    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < kChild) {
      GTEST_SKIP() << "This test requires at least " << kChild << " devices.";
    }

    ncclUniqueId initId, growId;
    NCCLCHECK(ncclGetUniqueId(&initId));

    // Spin up the 4-rank parent comm, one thread per rank.
    ncclComm_t parents[kParent] = {nullptr, nullptr, nullptr, nullptr};
    ncclResult_t initRes[kParent] = {ncclSuccess, ncclSuccess, ncclSuccess, ncclSuccess};
    std::thread initThreads[kParent];
    for (int r = 0; r < kParent; ++r) {
      initThreads[r] = std::thread([r, &parents, &initRes, &initId]() {
        HIPCALL(hipSetDevice(r));
        initRes[r] = ncclCommInitRank(&parents[r], kParent, initId, r);
      });
    }
    for (int r = 0; r < kParent; ++r) initThreads[r].join();
    for (int r = 0; r < kParent; ++r) ASSERT_EQ(initRes[r], ncclSuccess) << "rank " << r;

    // Rank 0 generates the grow handle and broadcasts to boundary ranks via
    // bcastGrowHandle (driven inside ncclCommGetUniqueId / ncclCommGrow).
    NCCLCHECK(ncclCommGetUniqueId(parents[0], &growId));

    // All 5 ranks call ncclCommGrow concurrently. Existing ranks 1 and 2 are
    // non-boundary; they pass nId=0 to bootstrapInit and rendezvous via the
    // parent bootstrap ring.
    ncclComm_t children[kChild] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    ncclResult_t growRes[kChild] = {ncclSuccess, ncclSuccess, ncclSuccess, ncclSuccess, ncclSuccess};
    std::thread growThreads[kChild];
    for (int r = 0; r < kChild; ++r) {
      growThreads[r] = std::thread([r, &parents, &children, &growRes, &growId]() {
        HIPCALL(hipSetDevice(r));
        if (r < kParent) {
          // Existing rank: pass parent comm, rank=-1.
          growRes[r] = ncclCommGrow(parents[r], kChild, &growId, -1, &children[r], nullptr);
        } else {
          // New rank: comm=NULL, rank=its own.
          growRes[r] = ncclCommGrow(nullptr, kChild, &growId, r, &children[r], nullptr);
        }
      });
    }
    for (int r = 0; r < kChild; ++r) growThreads[r].join();
    for (int r = 0; r < kChild; ++r) ASSERT_EQ(growRes[r], ncclSuccess) << "rank " << r;
    for (int r = 0; r < kChild; ++r) ASSERT_NE(children[r], nullptr) << "rank " << r;

    // Run an allreduce on the grown comm: each rank contributes its index,
    // expected sum = 0+1+2+3+4 = 10.
    float* sbuf[kChild] = {nullptr};
    float* rbuf[kChild] = {nullptr};
    hipStream_t streams[kChild] = {0};
    for (int r = 0; r < kChild; ++r) {
      HIPCALL(hipSetDevice(r));
      HIPCALL(hipMalloc((void**)&sbuf[r], sizeof(float)));
      HIPCALL(hipMalloc((void**)&rbuf[r], sizeof(float)));
      float v = static_cast<float>(r);
      HIPCALL(hipMemcpy(sbuf[r], &v, sizeof(float), hipMemcpyHostToDevice));
      HIPCALL(hipStreamCreate(&streams[r]));
    }

    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < kChild; ++r) {
      ASSERT_EQ(ncclAllReduce(sbuf[r], rbuf[r], 1, ncclFloat, ncclSum, children[r], streams[r]),
                ncclSuccess) << "rank " << r;
    }
    NCCLCHECK(ncclGroupEnd());

    constexpr float kExpected = 0.0f + 1.0f + 2.0f + 3.0f + 4.0f;
    for (int r = 0; r < kChild; ++r) {
      HIPCALL(hipSetDevice(r));
      HIPCALL(hipStreamSynchronize(streams[r]));
      float got = -1.0f;
      HIPCALL(hipMemcpy(&got, rbuf[r], sizeof(float), hipMemcpyDeviceToHost));
      ASSERT_FLOAT_EQ(got, kExpected) << "rank " << r;
      HIPCALL(hipStreamDestroy(streams[r]));
      HIPCALL(hipFree(sbuf[r]));
      HIPCALL(hipFree(rbuf[r]));
    }

    for (int r = 0; r < kChild; ++r) ASSERT_EQ(ncclCommDestroy(children[r]), ncclSuccess);
    for (int r = 0; r < kParent; ++r) ASSERT_EQ(ncclCommDestroy(parents[r]), ncclSuccess);
  }

  /**
   * \brief Passing newcomm=NULL must be rejected with ncclInvalidArgument
   *        before any state is mutated. Does not require a GPU.
   * ******************************************************************************************/
  TEST(Standalone, Grow_NullNewcomm_Rejected)
  {
    ASSERT_EQ(ncclCommGrow(nullptr, 2, nullptr, 1, nullptr, nullptr), ncclInvalidArgument);
  }

  /**
   * \brief An existing rank (comm != NULL) must pass rank=-1.
   *        Passing any non-negative rank must be rejected with ncclInvalidArgument.
   * ******************************************************************************************/
  TEST(Standalone, Grow_ExistingRank_NonNegativeRank_Rejected)
  {
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
      GTEST_SKIP() << "This test requires at least 1 device.";
    }

    ncclUniqueId initId;
    ncclComm_t comm = nullptr, newcomm = nullptr;
    NCCLCHECK(ncclGetUniqueId(&initId));
    NCCLCHECK(ncclCommInitRank(&comm, 1, initId, 0));

    ncclUniqueId growId;
    NCCLCHECK(ncclCommGetUniqueId(comm, &growId));

    // rank=0 instead of -1 — must be rejected
    ASSERT_EQ(ncclCommGrow(comm, 2, &growId, 0, &newcomm, nullptr), ncclInvalidArgument);

    NCCLCHECK(ncclCommDestroy(comm));
  }

  /**
   * \brief nRanks must be strictly greater than comm->nRanks for an existing rank.
   *        Passing nRanks <= current nRanks must be rejected with ncclInvalidArgument.
   * ******************************************************************************************/
  TEST(Standalone, Grow_NRanksTooSmall_Rejected)
  {
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
      GTEST_SKIP() << "This test requires at least 1 device.";
    }

    ncclUniqueId initId;
    ncclComm_t comm = nullptr, newcomm = nullptr;
    NCCLCHECK(ncclGetUniqueId(&initId));
    NCCLCHECK(ncclCommInitRank(&comm, 1, initId, 0));

    ncclUniqueId growId;
    NCCLCHECK(ncclCommGetUniqueId(comm, &growId));

    // nRanks=1 equals current nRanks — must be rejected
    ASSERT_EQ(ncclCommGrow(comm, 1, &growId, -1, &newcomm, nullptr), ncclInvalidArgument);

    NCCLCHECK(ncclCommDestroy(comm));
  }

  /**
   * \brief A new rank (comm=NULL) must provide a non-NULL uniqueId.
   *        Passing uniqueId=NULL must be rejected with ncclInvalidArgument.
   *        Does not require any rendezvous with an existing rank.
   * ******************************************************************************************/
  TEST(Standalone, Grow_NewRank_NullUniqueId_Rejected)
  {
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
      GTEST_SKIP() << "This test requires at least 1 device.";
    }

    ncclComm_t newcomm = nullptr;
    // comm=NULL (new rank), uniqueId=NULL — must be rejected before any rendezvous
    ASSERT_EQ(ncclCommGrow(nullptr, 2, nullptr, 1, &newcomm, nullptr), ncclInvalidArgument);
  }
}
