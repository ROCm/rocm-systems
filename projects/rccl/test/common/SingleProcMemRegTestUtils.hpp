/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "ScopedEnvVar.hpp"
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  enum class SingleProcMemRegMode
  {
    Disabled,
    Enabled
  };

  struct SingleProcMemRegTestConfig
  {
    SingleProcMemRegMode mode;
    std::vector<ncclFunc_t> funcTypes;
    std::vector<ncclDataType_t> dataTypes;
    std::vector<ncclRedOp_t> redOps;
    std::vector<int> roots;
    std::vector<int> numElements;
    std::vector<bool> inPlaceList;
    std::vector<bool> useHipGraphList;
  };

  inline void RunSingleProcMemRegTest(SingleProcMemRegTestConfig const& config)
  {
    char const* const mode =
      config.mode == SingleProcMemRegMode::Enabled ? "1" : "0";
    ScopedEnvVar pool("UT_COMM_POOL", "0");
    ScopedEnvVar singleProcMemReg("NCCL_SINGLE_PROC_MEM_REG_ENABLE", mode);
    ScopedEnvVar cuMem("NCCL_CUMEM_ENABLE", "1");
    TestBed testBed;
    int const hipRuntimeVer = testBed.ev.GetHipRuntimeVersion();
    if (hipRuntimeVer < 71260540)
    {
      GTEST_SKIP() << "Skipping single-process memory registration test: "
                   << "HIP runtime version (" << hipRuntimeVer
                   << ") is lower than 71260540";
    }

    std::vector<bool> const managedMemList = {false};
    testBed.RunSimpleSweep(config.funcTypes, config.dataTypes, config.redOps,
                           config.roots, config.numElements, config.inPlaceList,
                           managedMemList, config.useHipGraphList, true,
                           MEM_ALLOC_SYMMETRIC_WIN);
    testBed.Finalize();
  }

  inline void RunSingleProcMemRegDisabledSmoke(ncclFunc_t const funcType,
                                                ncclDataType_t const dataType,
                                                bool const supportsInPlace)
  {
    // Compact flag-off coverage: one unaligned count, both graph modes, and
    // both buffer placements when the collective supports in-place operation.
    SingleProcMemRegTestConfig config;
    config.mode = SingleProcMemRegMode::Disabled;
    config.funcTypes = {funcType};
    config.dataTypes = {dataType};
    config.redOps = {ncclSum};
    config.roots = {0};
    config.numElements = {4314};
    config.inPlaceList =
      supportsInPlace ? std::vector<bool>{true, false} : std::vector<bool>{false};
    config.useHipGraphList = {true, false};
    RunSingleProcMemRegTest(config);
  }
}
