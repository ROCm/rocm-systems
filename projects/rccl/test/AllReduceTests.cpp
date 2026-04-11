/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat32, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216, 384});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat16, ncclFloat64, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclMin});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({12888});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclInt32, ncclInt8});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclProd});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({384});
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclInt32, ncclFloat8e4m3, ncclFloat8e5m2});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclMax});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({393216, 12888, 384});
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclUint8, ncclUint64});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({2500});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, Channels)
  {
    TestBed testBed;
    if(testBed.ev.maxGpus >= 8) {
      if(testBed.ev.isGfx94) {
        // Configuration
        std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
        std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclBfloat16});
        std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
        std::vector<int>            const roots           = {0};
        std::vector<int>            const numElements     = testBed.ev.GetElements({64 * 1024 * 1024, 1024});
        std::vector<bool>           const inPlaceList     = {false};
        std::vector<bool>           const managedMemList  = {false};
        std::vector<bool>           const useHipGraphList = {false, true};
        std::vector<const char *>   const channelList     = {"84", "112"};
        bool                        const enableSweep     = false;
        for (auto channel : channelList) {
          setenv("NCCL_MIN_NCHANNELS", channel, 1);
          testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                inPlaceList, managedMemList, useHipGraphList, enableSweep);
          testBed.Finalize();
          unsetenv("NCCL_MIN_NCHANNELS");
        }
      }
    }
  }

  TEST(AllReduce, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.ev.GetDataTypes({ncclFloat64, ncclBfloat16});
    std::vector<ncclRedOp_t>    const redOps          = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = testBed.ev.GetElements({4314});
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  // This tests using custom pre-mult scalars reductions
  TEST(AllReduce, PreMultScalar)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                   const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t>  const  dataTypes     = testBed.ev.GetDataTypes({ncclFloat32, ncclFloat16, ncclBfloat16});
    std::vector<ncclRedOp_t>     const  redOps        = testBed.ev.GetRedOps({ncclSum});
    std::vector<int>             const  numElements   = testBed.ev.GetElements({384 * 1024, 384 * 32, 384});
    bool                         const  inPlace       = false;
    bool                         const  useManagedMem = false;

    OptionalColArgs options;

    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int redOpIdx = 0; redOpIdx < redOps.size() && isCorrect; ++redOpIdx)
      {
        ncclDataType_t const dataType = dataTypes[dataIdx];
        ncclRedOp_t    const redOp    = redOps[redOpIdx];

        // Set scalars per rank
        PtrUnion scalarsPerRank;
        scalarsPerRank.AllocateCpuMem(totalRanks * DataTypeToBytes(dataType));
        for (int i = 0;  i < totalRanks; i++)
        {
          double F = i;
          scalarsPerRank.Set(dataType, i, i, F);
        }
        int const numBytes = totalRanks * DataTypeToBytes(dataType);
        memcpy(options.scalarTransport.ptr, scalarsPerRank.ptr, numBytes);
        scalarsPerRank.FreeCpuMem();

        // Test various scalar residence modes
        for (int scalarMode = 0; scalarMode <= 1 && isCorrect; ++scalarMode)
        {
          if (testBed.ev.showNames)
            TEST_INFO("%s %d-ranks AllReduce (custom-scalar Mode %d %s)",
                 isMultiProcess ? "MP" : "SP",
                 totalRanks, scalarMode, ncclDataTypeNames[dataType]);

          for (int i = 0; i < numElements.size() && isCorrect; ++i)
          {
            options.scalarMode = scalarMode;
            options.redOp = redOp;
            testBed.SetCollectiveArgs(funcType, dataType,
                                      numElements[i], numElements[i],
                                      options);
            // For performance, only allocate and prepare data on largest size
            if (i == 0)
            {
              testBed.AllocateMem(inPlace, useManagedMem);
              testBed.PrepareData();
            }
            testBed.ExecuteCollectives();
            testBed.ValidateResults(isCorrect);
          }
          testBed.DeallocateMem();
        }
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  TEST(AllReduce, UserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(count, 0);
    std::vector<int> expected(count, 0);

    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
        expected[i] = i * nranks;
    }
    callCollectiveForked(nranks, ncclCollAllReduce, sendBuff, recvBuff, expected);
  }

  TEST(AllReduce, ManagedMemUserBufferRegistration)
  {
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(count, 0);
    std::vector<int> expected(count, 0);
    const bool use_managed_mem = true;
    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
        expected[i] = i * nranks;
    }
    callCollectiveForked(nranks, ncclCollAllReduce, sendBuff, recvBuff, expected, use_managed_mem);
  }

  TEST(AllReduce, ROCTX)
  {
    // Set RCCL_LOG_ROCTX=1 to enable ROCTX logging
    // Verify that ROCTX logging doesn't break functionality when enabled
    setenv("RCCL_LOG_ROCTX", "1", 1);

    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(count, 0);
    std::vector<int> expected(count, 0);

    for (int i = 0; i < count; ++i) {
        sendBuff[i] = i;
        expected[i] = i * nranks;
    }
    callCollectiveForked(nranks, ncclCollAllReduce, sendBuff, recvBuff, expected);

    unsetenv("RCCL_LOG_ROCTX");
  }

  // Regression test for "one-rank avg reduction missing elements"
  // (NVIDIA/nccl GitHub issue #1950, fixed upstream in NCCL v2.29.7-1; AICOMRCCL-1110).
  //
  // On a single-rank communicator, ncclAvg on floating-point types lowers to a
  // PreMulSum reduction (scalar 1/nRanks), which is the only out-of-place path
  // that is serviced by the oneRankReduce kernel (src/device/onerank.cu) rather
  // than a plain memcpy. The kernel divided work across blocks using floor
  // division (nElts/bn) instead of divUp(nElts, bn); when the element count was
  // not a multiple of the launched block count, the trailing (nElts % bn)
  // elements were never written, silently corrupting the tail of the output.
  //
  // Each count below is constructed so that floor(count/numBlocks) is already
  // pack-aligned and the buffer is large enough to launch the maximum of 32
  // blocks. This guarantees the pre-fix kernel under-covers the output by a
  // fixed, non-zero remainder, so ValidateResults observes the corrupted tail.
  // The test fails on the pre-fix kernel and passes once divUp is used.
  TEST(AllReduce, OneRankAvgTailElements)
  {
    TestBed testBed;

    if (testBed.numDevicesAvailable < 1)
      GTEST_SKIP() << "Requires at least 1 GPU";

    ncclFunc_t                  const funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const dataTypes     = testBed.ev.GetDataTypes({ncclFloat32, ncclFloat64, ncclBfloat16});
    bool                        const inPlace       = false; // out-of-place: tail of separate output buffer must be written
    bool                        const useManagedMem = false;

    // oneRankReduce only executes for a single-rank communicator.
    testBed.InitComms(1);

    OptionalColArgs options;
    options.redOp = ncclAvg;

    bool isCorrect = true;
    for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
    {
      ncclDataType_t const dataType  = dataTypes[dataIdx];
      size_t         const eltSize   = DataTypeToBytes(dataType);
      // Mirror the kernel: EltPerPack = 16 / sizeof(T), grid.x capped at 32 blocks.
      size_t         const eltPerPack = 16 / eltSize;
      size_t         const numBlocks  = 32;
      // Pack-aligned per-block segment, then add a remainder that is not a
      // multiple of numBlocks so the pre-fix floor division drops the tail.
      size_t         const perBlock   = eltPerPack * 4096;
      size_t         const numElems   = numBlocks * perBlock + 17;

      if (testBed.ev.showNames)
        TEST_INFO("SP 1-rank AllReduce (avg) %s count %zu",
                  ncclDataTypeNames[dataType], numElems);

      testBed.SetCollectiveArgs(funcType, dataType, numElems, numElems, options);
      testBed.AllocateMem(inPlace, useManagedMem);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
    }
    testBed.DestroyComms();
    testBed.Finalize();
  }
}
