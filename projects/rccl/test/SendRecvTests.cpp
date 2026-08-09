/*************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(SendRecv, SinglePairs)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& testDataTypes   = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024, 0};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }
    if (testBed.ev.maxGpus < 2) {
      GTEST_SKIP() << "Skipping... SendRecv.SinglePairs requires at least 2 GPUs (detected " << testBed.ev.maxGpus << ")";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank; //self sendrecv group has two coll
          int recvId      = sendRank == recvRank; //where recv will be second coll
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            //set up the collArg slot to make sure AllocateMem is called once and correctly
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);
          options.root = sendRank;

          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Shared sweep body for the LL128 latency-bound P2P send/recv regression tests.
  //
  // RCCL's P2P send/recv uses the LL128 protocol (in place of the legacy LL) for per-channel
  // payloads <= NCCL_P2P_LL_THRESHOLD, and SIMPLE above it. The switch requires converting between
  // LL128 "wire" chunk sizes (data lines carry an extra flag word) and "data" chunk sizes in the
  // enqueue path, plus matching proxy byte accounting. Those conversions are size-sensitive, so a
  // bug typically corrupts data only at specific sizes near the protocol boundary or at sizes that
  // are not a multiple of the LL128 line/data granularity.
  //
  // This sweeps awkward element counts straddling the LL128/SIMPLE threshold using datatypes of
  // different element sizes (so the per-channel byte payload crosses the threshold at different
  // element counts, exercising both protocol selections and the wire<->data conversion), and
  // validates end-to-end correctness for every send/recv pair. The caller pins
  // NCCL_ALLOC_P2P_NET_LL_BUFFERS and, to force intranode send/recv onto the network transport on a
  // single node, disables the P2P and SHM transports (disabling P2P alone routes via SHM; only
  // disabling both P2P and SHM falls through to NET/IB, where NCCL_ALLOC_P2P_NET_LL_BUFFERS actually
  // governs LL/LL128 staging-buffer allocation).
  static void RunLL128BoundarySweep(TestBed& testBed)
  {
    // Configuration: int8 (1B) and float32 (4B) so the 16 KiB/channel LL128 threshold is crossed at
    // 16384 and 4096 elements respectively; sizes below pick LL128, sizes above pick SIMPLE.
    std::vector<ncclDataType_t> const& testDataTypes = {ncclInt8, ncclFloat32};
    // Awkward counts around the boundary (non-power-of-2 / prime), plus a clearly-SIMPLE large size.
    // Kept small because these variants force transfers over the (slower) network loopback path.
    std::vector<int>            const  numElements   = {262144, 16385, 4097, 1};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }
    if (testBed.ev.maxGpus < 2) {
      GTEST_SKIP() << "Skipping... SendRecv.LL128LatencyBoundarySizes requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg = 0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank; //self sendrecv group has two coll
          int recvId      = sendRank == recvRank; //where recv will be second coll
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s LL128 boundary SendReceive Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);
          options.root = sendRank;

          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=0: no dedicated LL/LL128 net staging buffers are allocated for P2P
  // network connections, so send/recv falls back to SIMPLE over the network. Verifies correctness of
  // the guarded allocation path when P2P net LL buffers are disabled.
  TEST(SendRecv, LL128NetBuffersDisabled)
  {
    setenv("NCCL_P2P_DISABLE", "1", 1);              // disable P2P/IPC so send/recv does not use it
    setenv("NCCL_SHM_DISABLE", "1", 1);              // disable SHM so send/recv falls through to NET
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0", 1); // disabled case
    TestBed testBed;
    RunLL128BoundarySweep(testBed);
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    unsetenv("NCCL_SHM_DISABLE");
    unsetenv("NCCL_P2P_DISABLE");
  }

  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=1: dedicated LL/LL128 net staging buffers are allocated for P2P
  // (shared) network connections, enabling LL128 for latency-bound send/recv over the network.
  // Verifies correctness of the changed net allocation + LL128 dispatch path when enabled (and, with
  // the accompanying fix, that ring/tree collective connections do not allocate these buffers).
  TEST(SendRecv, LL128NetBuffersEnabled)
  {
    setenv("NCCL_P2P_DISABLE", "1", 1);              // disable P2P/IPC so send/recv does not use it
    setenv("NCCL_SHM_DISABLE", "1", 1);              // disable SHM so send/recv falls through to NET
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "1", 1); // enabled case
    TestBed testBed;
    RunLL128BoundarySweep(testBed);
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    unsetenv("NCCL_SHM_DISABLE");
    unsetenv("NCCL_P2P_DISABLE");
  }

  TEST(SendRecv, UserBufferRegister)
  {
    setenv("RCCL_ENABLE_INTRANET", "1", 1);
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& testDataTypes   = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = true;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank;
          int recvId      = sendRank == recvRank;
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank, userRegistered);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank, userRegistered);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);

          options.root = sendRank;
          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank, userRegistered);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
    unsetenv("RCCL_ENABLE_INTRANET");
  }
}
