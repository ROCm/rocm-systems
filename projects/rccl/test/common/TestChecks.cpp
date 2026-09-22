/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file TestChecks.cpp
 * @brief Implementation file for TestChecks.hpp
 *
 * Provides definitions for variables used by test logging macros.
 */

#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

#include <cstdlib>
#include <cstring>

// Define and initialize rcclTestDebugLevel for TEST_* macros

// This matches RCCL's debug level parsing logic from src/debug.cc
// Values correspond to ncclDebugLogLevel enum in nccl_common.h:
// - -1 = Uninitialized (treated as ERROR level)
// - 0 = NCCL_LOG_NONE
// - 1 = NCCL_LOG_VERSION
// - 2 = NCCL_LOG_WARN
// - 3 = NCCL_LOG_INFO
// - 4 = NCCL_LOG_ABORT
// - 5 = NCCL_LOG_TRACE
int rcclTestDebugLevel = []() -> int {
    const char* env = std::getenv("NCCL_DEBUG");

    // Default to ERROR level if not set (matches RCCL behavior)
    if (!env) return -1;

    // Match RCCL's case-insensitive string comparison
    if (strcasecmp(env, "NONE") == 0) return 0;
    if (strcasecmp(env, "VERSION") == 0) return 1;
    if (strcasecmp(env, "WARN") == 0) return 2;
    if (strcasecmp(env, "INFO") == 0) return 3;
    if (strcasecmp(env, "ABORT") == 0) return 4;
    if (strcasecmp(env, "TRACE") == 0) return 5;

    // Unknown value, default to ERROR level
    return -1;
}();

#endif // MPI_TESTS_ENABLED


