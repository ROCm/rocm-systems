/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SymmetricMemoryTests.cpp
 * @brief TestBed sweeps that exercise the symmetric-memory kernel path
 *
 * Each test passes useSymmetric=true on RunSimpleSweep (or sets it manually
 * on AllocateMem). That flag tells the TestBed to allocate input/output via
 * ncclMemAlloc and register them with ncclCommWindowRegister(
 * NCCL_WIN_COLL_SYMMETRIC); RCCL then picks the symmetric-memory kernel
 * automatically when supported.
 *
 * Symmetric tests set NCCL_CUMEM_ENABLE=-2 (auto-detect) for the process for
 * the duration of each test. You can still override from the shell if needed.
 *
 * Run:
 *   ./rccl-UnitTests --gtest_filter=SymmetricMemory.*
 */

#include "TestBed.hpp"
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstring>

namespace RcclUnitTesting
{
  static std::vector<ncclDataType_t> SymSupportedTypes()
  {
    return {ncclFloat32, ncclFloat16, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
  }

  // Regression for upstream NCCL (2.28.9) fix: data corruption with built-in symmetric
  // kernels when total message size has granularity under 8 bytes. Exercises
  // small element counts so total byte sizes land on non-8B boundaries (e.g.
  // ncclFloat16 with 1/3/5 elements -> 2/6/10 bytes; ncclFloat32 with 1/3
  // elements -> 4/12 bytes; ncclFloat8e4m3/e5m2 with 3/5/7 -> 3/5/7 bytes).
  //
  // Validation accuracy caveat:
  //   PtrUnion::IsEqual (test/common/PtrUnion.cpp) uses tolerance 9e-2 for
  //   fp16/bf16/fp8e4m3/fp8e5m2 and 1e-5 for fp32. The default fill pattern
  //   (DefaultPrepData_Reduce + FillPattern) seeds inputs with small
  //   reciprocals 1/((rank+i)%256 + 1) in [1/256, 1], so AllReduce sums stay
  //   bounded ~N. With small magnitudes and a loose 9e-2 tolerance, single-
  //   cell bit-level corruption in fp16/bf16/fp8 may slip through; corruption
  //   at the magnitude of a real bug (whole-word stomps, off-by-one cells,
  //   etc.) will still trigger a mismatch. fp32 is the strictest detector
  //   here and is the most reliable signal of an actual regression.
  TEST(SymmetricMemory, SmallMessageGranularity)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce, ncclCollReduceScatter};
    std::vector<ncclDataType_t> const dataTypes       = SymSupportedTypes();
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1, 3, 5, 7, 9, 11, 17};
    std::vector<bool>           const inPlaceList     = {false, true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList,
                           /*enableSweep=*/true, /*useSymmetric=*/true);
    testBed.Finalize();
  }

  // Regression for upstream NCCL fix: data corruption when multiple symmetric
  // operations are aggregated in a single group.
  TEST(SymmetricMemory, GroupedAggregation)
  {
    TestBed testBed;

    // Each entry is one collective in the group.
    struct GroupColl {
      ncclFunc_t funcType;
      int        elementCount;
    };
    std::vector<GroupColl> const groupColls = {
      {ncclCollAllReduce,     3},
      {ncclCollAllGather,     7},
      {ncclCollReduceScatter, 11},
      {ncclCollAllReduce,     4096},
    };

    int  const numCollPerGroup = static_cast<int>(groupColls.size());
    bool const inPlace         = false;
    bool const useManagedMem   = false;
    bool const useSymmetric    = true;

    OptionalColArgs options;
    options.redOp = ncclSum;

    // Auto-detect CUMEM support
    setenv("NCCL_CUMEM_ENABLE", "-2", 1);

    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    for (ncclDataType_t dataType : {ncclFloat32, ncclBfloat16, ncclFloat8e4m3})
    {
      if (totalRanks < 2)
        continue;

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder),
                        numCollPerGroup);

      bool hasSymmetricSupport = false;
      testBed.HasSymmetricSupport(hasSymmetricSupport);
      if (!hasSymmetricSupport)
      {
        testBed.DestroyComms();
        unsetenv("NCCL_CUMEM_ENABLE");
        GTEST_SKIP() << "Skipping... symmetric memory not supported on this configuration "
                     << "(totalRanks=" << totalRanks
                     << ", isMultiProcess=" << isMultiProcess
                     << "). Requires NCCL_CUMEM_ENABLE=-2 (auto) or 1 and a compatible platform.";
      }

      if (testBed.ev.showNames)
        TEST_INFO("%s %d-ranks SymmetricMemory GroupedAggregation dtype=%s",
                  isMultiProcess ? "MP" : "SP", totalRanks,
                  ncclDataTypeNames[dataType]);

      for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
      {
        int numInputElements  = 0;
        int numOutputElements = 0;
        CollectiveArgs::GetNumElementsForFuncType(groupColls[collIdx].funcType,
                                                  groupColls[collIdx].elementCount,
                                                  totalRanks,
                                                  &numInputElements,
                                                  &numOutputElements);
        testBed.SetCollectiveArgs(groupColls[collIdx].funcType,
                                  dataType,
                                  numInputElements,
                                  numOutputElements,
                                  options,
                                  collIdx);
      }

      testBed.AllocateMem(inPlace, useManagedMem,
                          /*groupId=*/-1, /*collId=*/-1, /*rank=*/-1,
                          /*userRegistered=*/false, useSymmetric);
      testBed.PrepareData();
      testBed.ExecuteCollectives();
      testBed.ValidateResults(isCorrect);
      testBed.DeallocateMem();
      testBed.DestroyComms();

      if (!isCorrect || testing::Test::HasFailure()) break;
    }
    testBed.Finalize();
    unsetenv("NCCL_CUMEM_ENABLE");
  }
}  // namespace RcclUnitTesting
