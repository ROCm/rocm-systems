/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

// Correctness coverage for the reduceCopyPacksWithBias() restructuring in
// src/device/common_kernel.h (load/reduce fusion + full-hunk accPtr advance).
// That path backs ncclAllReduceWithBias (useAcc kernels). The existing
// AllReduceTests.cpp bias cases already cover the integer and fp32/fp64 types;
// this file adds the narrow-float types the reduce/scratch work specifically
// targets (fp16, bf16, fp8 e4m3/e5m2), which had no bias coverage.
//
// Inputs and bias are held at a constant 1 so the reduced result is exactly
// representable in every narrow format and is independent of the CI rank count
// (Sum -> nRanks+1, Prod/Max/Min -> 1); this isolates the code-structure change
// from fp8 rounding/overflow semantics.

namespace RcclUnitTesting
{
#ifdef RCCL_ALLREDUCE_WITH_BIAS
  namespace
  {
    constexpr int          BIAS_CONSTANT_ONE  = 1;
    constexpr int          INPUT_CONSTANT_ONE = 1;
    // 2048 exercises the unrolled big-pack aligned path (multiple hunks/warps,
    // where the accPtr advance matters); 384 exercises a smaller/remainder size.
    const std::vector<int> kElemCounts        = {2048, 384};

    void RunNarrowBiasTest(ncclDataType_t dataType, ncclRedOp_t redOp)
    {
      TestBed testBed;

      // ncclAllReduceWithBias is only supported on gfx942/gfx950.
      if (!testBed.ev.isGfx94 && !testBed.ev.isGfx95)
      {
        TEST_INFO("SKIPPED: AllReduce with Bias is only supported on gfx942 or gfx950 architectures.");
        return;
      }

      bool const inPlace       = false;
      bool const useManagedMem = false;
      bool const useHipGraph   = false;

      OptionalColArgs options;
      options.useBias            = true;
      options.redOp              = redOp;
      options.biasConstantValue  = BIAS_CONSTANT_ONE;
      options.inputConstantValue = INPUT_CONSTANT_ONE;

      bool isCorrect = true;

      for (int totalRanks : testBed.ev.GetNumGpusList())
      {
        int const               numProcesses     = totalRanks;
        bool const              isMultiProcess   = true;
        const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
        testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

        for (auto numElem : kElemCounts)
        {
          if (!isCorrect)
            break;

          if (testBed.ev.showNames)
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
            TEST_INFO("  %s (with bias, count=%d)", name.c_str(), numElem);
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
  } // namespace

  // fp16
  TEST(AllReduceBias, Float16_Sum)  { RunNarrowBiasTest(ncclFloat16, ncclSum);  }
  TEST(AllReduceBias, Float16_Prod) { RunNarrowBiasTest(ncclFloat16, ncclProd); }
  TEST(AllReduceBias, Float16_Max)  { RunNarrowBiasTest(ncclFloat16, ncclMax);  }
  TEST(AllReduceBias, Float16_Min)  { RunNarrowBiasTest(ncclFloat16, ncclMin);  }

  // bf16
  TEST(AllReduceBias, Bfloat16_Sum)  { RunNarrowBiasTest(ncclBfloat16, ncclSum);  }
  TEST(AllReduceBias, Bfloat16_Prod) { RunNarrowBiasTest(ncclBfloat16, ncclProd); }
  TEST(AllReduceBias, Bfloat16_Max)  { RunNarrowBiasTest(ncclBfloat16, ncclMax);  }
  TEST(AllReduceBias, Bfloat16_Min)  { RunNarrowBiasTest(ncclBfloat16, ncclMin);  }

  // fp8 e4m3 / e5m2: Prod/Max/Min keep the result at exactly 1 for any rank
  // count, so they stay bit-exact in both fp8 formats (Sum is left to the wider
  // types above to avoid fp8 overflow/rounding semantics).
  TEST(AllReduceBias, Float8e4m3_Prod) { RunNarrowBiasTest(ncclFloat8e4m3, ncclProd); }
  TEST(AllReduceBias, Float8e4m3_Max)  { RunNarrowBiasTest(ncclFloat8e4m3, ncclMax);  }
  TEST(AllReduceBias, Float8e4m3_Min)  { RunNarrowBiasTest(ncclFloat8e4m3, ncclMin);  }

  TEST(AllReduceBias, Float8e5m2_Prod) { RunNarrowBiasTest(ncclFloat8e5m2, ncclProd); }
  TEST(AllReduceBias, Float8e5m2_Max)  { RunNarrowBiasTest(ncclFloat8e5m2, ncclMax);  }
  TEST(AllReduceBias, Float8e5m2_Min)  { RunNarrowBiasTest(ncclFloat8e5m2, ncclMin);  }
#else
  TEST(AllReduceBias, NotAvailable)
  {
    TEST_INFO("SKIPPED: RCCL_ALLREDUCE_WITH_BIAS not defined - bias tests skipped");
    return;
  }
#endif
} // namespace RcclUnitTesting
