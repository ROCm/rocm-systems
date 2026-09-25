/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "ScopedEnvVar.hpp"
#include "TestBed.hpp"
#include "rocmwrap.h"

namespace RcclUnitTesting
{
  enum class SingleProcMemRegMode
  {
    Disabled,
    Enabled
  };

  struct SingleProcMemRegTestConfig
  {
    SingleProcMemRegMode mode = SingleProcMemRegMode::Disabled;
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
    ScopedEnvVar singleProcMemReg("NCCL_SINGLE_PROC_MEM_REG_ENABLE", mode);
    // Symmetric window registration additionally needs cuMem, NCCL_WIN_ENABLE
    // and an LSA team; pin them so an inherited environment cannot make NCCL
    // decline the registration, which the worker reports as a hard failure.
    ScopedEnvVar cuMem("NCCL_CUMEM_ENABLE", "1");
    ScopedEnvVar winEnable("NCCL_WIN_ENABLE", "1");
    ScopedEnvVar lsaTeamSize("NCCL_LSA_TEAM_SIZE", "0");
    TestBed testBed;
    int const hipRuntimeVer = testBed.ev.GetHipRuntimeVersion();
    if (!NCCL_CUMEM_VERSION_SUPPORTED(hipRuntimeVer))
    {
      GTEST_SKIP() << "Skipping single-process memory registration test: "
                   << "HIP runtime version (" << hipRuntimeVer
                   << ") does not support cuMem";
    }
    if (!testBed.ev.IsCuMemRuntimeSupported())
    {
      GTEST_SKIP() << "Skipping single-process memory registration test: "
                   << "cuMem is not supported on this platform "
                   << "(requires Linux kernel >= 6.8 and VMM support)";
    }

    std::vector<bool> const managedMemList = {false};
    testBed.RunSimpleSweep(config.funcTypes, config.dataTypes, config.redOps,
                           config.roots, config.numElements, config.inPlaceList,
                           managedMemList, config.useHipGraphList,
                           /*enableSweep=*/true, MEM_ALLOC_SYMMETRIC_WIN);
    testBed.Finalize();
  }

  // Flag-off control: runs the same symmetric-window path with
  // NCCL_SINGLE_PROC_MEM_REG_ENABLE=0 to check that the fallback still produces
  // correct results. The flag's effect on registration is asserted host-side in
  // test/host/init-test.cc. Kept compact: one unaligned count, both graph modes,
  // and both buffer placements when the collective supports in-place.
  inline void RunSingleProcMemRegDisabledTest(ncclFunc_t const funcType,
                                              ncclDataType_t const dataType,
                                              bool const supportsInPlace,
                                              std::vector<int> const& roots = {0})
  {
    SingleProcMemRegTestConfig config;
    config.mode = SingleProcMemRegMode::Disabled;
    config.funcTypes = {funcType};
    config.dataTypes = {dataType};
    config.redOps = {ncclSum};
    config.roots = roots;
    config.numElements = {4314};
    config.inPlaceList =
      supportsInPlace ? std::vector<bool>{true, false} : std::vector<bool>{false};
    config.useHipGraphList = {true, false};
    RunSingleProcMemRegTest(config);
  }
}
