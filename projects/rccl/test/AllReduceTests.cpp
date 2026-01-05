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
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
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
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat64, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {12888};
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
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {384};
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
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclFloat8e4m3, ncclFloat8e5m2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 12888, 384};
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
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {2500};
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
        std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16};
        std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
        std::vector<int>            const roots           = {0};
        std::vector<int>            const numElements     = {64 * 1024 * 1024, 1024};
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
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {4314};
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
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclFloat32};
    ncclRedOp_t                 const  redOp         = ncclSum;
    std::vector<int>            const  numElements   = {384 * 1024, 384 * 32, 384};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

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
      {
        ncclDataType_t const dataType = dataTypes[dataIdx];

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

        // Test various scalar residence modes
        for (int scalarMode = 0; scalarMode <= 1 && isCorrect; ++scalarMode)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks AllReduce (custom-scalar Mode %d %s)\n",
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

#ifdef RCCL_ALLREDUCE_WITH_BIAS
  // Note: All bias tests require:
  // nRanks >= 2 (bias NOT supported for single rank)

  // Named constants for bias test configuration
  namespace BiasTestConstants
  {
  // Element counts for different operations
  constexpr std::initializer_list<int> STANDARD_ELEM_COUNTS    = {2048, 384}; // For Sum/Max/Min
  constexpr std::initializer_list<int> PROD_ELEM_COUNTS_MEDIUM = {32}; // For Int32/Uint32 Prod
  constexpr std::initializer_list<int> PROD_ELEM_COUNTS_LARGE  = {64}; // For Int8/Uint8/Int64/Uint64/Float Prod

  // Bias and input pattern constants
  constexpr int BIAS_CONSTANT_ONE = 1; // Use constant bias value of 1 (prevents overflow)
  constexpr int BIAS_INCREMENTAL_PATTERN
      = -1; // Use incremental pattern: bias[i] = i (more thorough testing)
  constexpr int INPUT_RANK_BASED_PATTERN
      = -1; // Use rank-based pattern: input[rank][i] = (rank+i)%256
  constexpr int INPUT_CONSTANT_ONE = 1; // Use constant input value of 1 (prevents overflow)
  } // namespace BiasTestConstants

  /*
   * @brief Helper function for running bias tests with specific datatype and redOp
   * @param dataType Data type
   * @param redOp Reduction operation
   * @param numElements Number of elements
   * @param biasConstVal Bias constant value, -1 for incremental bias
   * @param inputConstVal Input constant value, -1 for rank-based input
   */
  void RunBiasTest(ncclDataType_t   dataType,
                   ncclRedOp_t      redOp,
                   std::vector<int> numElements,
                   int              biasConstVal  = BiasTestConstants::BIAS_INCREMENTAL_PATTERN,
                   int              inputConstVal = BiasTestConstants::INPUT_RANK_BASED_PATTERN)
  {
      // Create TestBed first (doesn't create child processes yet)
      TestBed testBed;

      // Check if architecture is gfx94 (covers gfx942) or gfx95 (covers gfx950)
      if (!testBed.ev.isGfx94 && !testBed.ev.isGfx95)
      {
          INFO("SKIPPED: AllReduce with Bias is only supported on gfx942 or gfx950 architectures.\n");
          return;
      }

      bool const inPlace       = false;
      bool const useManagedMem = false;
      bool const useHipGraph   = false;

      OptionalColArgs options;
      options.useBias            = true;
      options.redOp              = redOp;
      options.biasConstantValue  = biasConstVal;
      options.inputConstantValue = inputConstVal;

      bool isCorrect = true;

      for(int totalRanks : testBed.ev.GetNumGpusList())
      {
          if(totalRanks < 2)
              continue;

          int const               numProcesses     = totalRanks;
          bool const              isMultiProcess   = true;
          const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
          testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

          for(auto numElem : numElements)
          {
              if(!isCorrect)
                  break;

              if(testBed.ev.showNames)
              {
                  std::string name = testBed.GetTestCaseName(totalRanks,
                                                             isMultiProcess,
                                                             ncclCollAllReduce,
                                                             dataType,
                                                             redOp,
                                                             -1,
                                                             inPlace,
                                                             useManagedMem,
                                                             useHipGraph);
                  INFO("  %s (with bias, count=%d)\n", name.c_str(), numElem);
              }

              options.biasNumElements = numElem;

              testBed.SetCollectiveArgs(ncclCollAllReduce,
                                        dataType,
                                        numElem,
                                        numElem,
                                        options,
                                        -1,
                                        0,
                                        -1);
              testBed.AllocateMem(inPlace, useManagedMem);
              testBed.PrepareData();
              testBed.ExecuteCollectives({}, useHipGraph);
              testBed.ValidateResults(isCorrect);
              testBed.DeallocateMem();
          }
          testBed.DestroyComms();
      }
      testBed.Finalize();
  }

  // Int8 Tests
  TEST(AllReduce, BiasInt8_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt8,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt8_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt8,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt8_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt8,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt8_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt8,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_CONSTANT_ONE,
                  INPUT_CONSTANT_ONE);
  }

  // Uint8 Tests
  TEST(AllReduce, BiasUint8_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint8,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint8_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint8,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint8_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint8,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint8_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint8,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_CONSTANT_ONE,
                  INPUT_CONSTANT_ONE);
  }

  // Int32 Tests
  TEST(AllReduce, BiasInt32_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt32,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt32_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt32,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt32_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt32,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt32_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt32,
                  ncclProd,
                  PROD_ELEM_COUNTS_MEDIUM,
                  BIAS_CONSTANT_ONE,
                  INPUT_CONSTANT_ONE);
  }

  // Uint32 Tests
  TEST(AllReduce, BiasUint32_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint32,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint32_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint32,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint32_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint32,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_CONSTANT_ONE,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint32_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint32,
                  ncclProd,
                  PROD_ELEM_COUNTS_MEDIUM,
                  BIAS_CONSTANT_ONE,
                  INPUT_CONSTANT_ONE);
  }

  // Int64 Tests
  TEST(AllReduce, BiasInt64_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt64,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt64_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt64,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt64_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt64,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasInt64_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclInt64,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_CONSTANT_ONE);
  }

  // Uint64 Tests
  TEST(AllReduce, BiasUint64_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint64,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint64_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint64,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint64_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint64,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasUint64_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclUint64,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_CONSTANT_ONE);
  }

  // Float32 Tests
  TEST(AllReduce, BiasFloat32_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat32,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat32_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat32,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat32_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat32,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat32_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat32,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_CONSTANT_ONE);
  }

  // Float64 Tests
  TEST(AllReduce, BiasFloat64_Sum)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat64,
                  ncclSum,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat64_Max)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat64,
                  ncclMax,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat64_Min)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat64,
                  ncclMin,
                  STANDARD_ELEM_COUNTS,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_RANK_BASED_PATTERN);
  }

  TEST(AllReduce, BiasFloat64_Prod)
  {
      using namespace BiasTestConstants;
      RunBiasTest(ncclFloat64,
                  ncclProd,
                  PROD_ELEM_COUNTS_LARGE,
                  BIAS_INCREMENTAL_PATTERN,
                  INPUT_CONSTANT_ONE);
  }

#else
  // If RCCL_ALLREDUCE_WITH_BIAS is not defined, skip all bias tests
  TEST(AllReduce, BiasNotAvailable)
  {
      INFO("SKIPPED: RCCL_ALLREDUCE_WITH_BIAS not defined - bias tests skipped\n");
      return;
  }
#endif
}
