/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once
#include <hsa/hsa.h>
#include <vector>
#include "rccl/rccl.h"

namespace RcclUnitTesting
{
  // Helper function to count the number of GPUs on system
  static hsa_status_t CountGpus(hsa_agent_t agent, void* data);

  // Helper class to track environment variables that affect the unit tests
  class EnvVars
  {
  public:
    bool debugPause;     // Pause for debugger attach              [UT_DEBUG_PAUSE]
    bool showNames;      // List test case names during run        [UT_SHOW_NAMES]
    int  minGpus;        // Set the minimum number of GPUs to use  [UT_MIN_GPUS]
    int  maxGpus;        // Set the maximum number of GPUs to use  [UT_MAX_GPUS]
    bool onlyPow2Gpus;   // Only allow power-of-2 # of GPUs        [UT_POW2_GPUS]
    int  processMask;    // Filter single/multi process            [UT_PROCESS_MASK]
    bool verbose;        // Show verbose TestBed output for debug  [UT_VERBOSE]
    int  printValues;    // Print out input/output/expected arrays [UT_PRINT_VALUES]
    int  maxRanksPerGpu; // Number of ranks using the same GPU     [UT_MAX_RANKS_PER_GPU]
    bool showTiming;     // Show timing per case at end            [UT_SHOW_TIMING]
    bool useInteractive; // Run in interactive mode                [UT_INTERACTIVE]
    int  timeoutUs;      // Set timeout for child in microseconds  [UT_TIMEOUT_US]
    bool useMultithreading; // Multi-thread single-process ranks   [UT_MULTITHREAD]

    bool isGfx94;        // Detects if architecture is gfx94
    bool isGfx12;        // Detects if architecture is gfx12
    bool isGfx90;        // Detects if architecture is gfx90

    // Constructor that parses and collects environment variables
    EnvVars();

    std::vector<ncclRedOp_t>    const& GetAllSupportedRedOps();
    std::vector<ncclDataType_t> const& GetAllSupportedDataTypes();

    std::vector<int>            const& GetNumGpusList();
    std::vector<int>            const& GetIsMultiProcessList();
    std::vector<int>            const& GetGpuPriorityOrder();   // Orders the gpus based on the associativity of them with OAM with higher gpus linked.
    void ShowConfig();

  protected:
    std::vector<ncclRedOp_t>    redOps;             // Supported reduction ops [UT_REDOPS]
    std::vector<ncclDataType_t> dataTypes;          // Support datatypes       [UT_DATATYPES]
    std::vector<int>            numGpusList;        // List of # Gpus to use   [UT_MIN_GPUS/UT_MAX_GPUS/UT_POW2_GPUS]
    std::vector<int>            isMultiProcessList; // Single or multi process [UT_PROCESS_MASK]
    int                         numDetectedGpus;
    std::vector<int>            gpuPriorityOrder;   // Orders the gpus based on the associativity of them with OAM with higher gpus linked.

    // Helper functions to parse environment variables
    int GetEnvVar(std::string const varname, int defaultValue);
    std::vector<std::string> GetEnvVarsList(std::string const varname);
  };
}
