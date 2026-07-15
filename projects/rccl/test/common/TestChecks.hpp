/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file TestChecks.hpp
 * @brief Centralized test error checking and logging macros
 *
 * Common (always available):
 * - HIP error checking (HIP_CHECK, HIP_EXPECT, HIP_TEST_CHECK_GTEST_FAIL,
 *                        HIPCHECK, HIP_TEST_CHECK)
 * - NCCL error checking (RCCL_TEST_CHECK, RCCL_TEST_CHECK_GTEST_FAIL)
 *
 * MPI-only (requires MPI_TESTS_ENABLED):
 * - MPI error checking (MPICHECK with 3 overload variants)
 * - MPI-aware assertions (ASSERT_MPI_*)
 * - Debug logging (TEST_WARN, TEST_INFO, TEST_ABORT, TEST_TRACE)
 */

#ifndef RCCL_TEST_CHECKS_HPP
#define RCCL_TEST_CHECKS_HPP

// ============================================================================
// Common macros — available to all test binaries (no MPI dependency)
// ============================================================================

#include "gtest/gtest.h"
#include <cstdio>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

// HIP Error Checking Macros (GTest-native)

/**
 * @def HIP_CHECK
 * @brief Fatal HIP assertion for test bodies and void-returning helpers
 *
 * Consumes the [[nodiscard]] hipError_t and fails the current test on error.
 * Uses ASSERT_EQ internally, so it aborts the current function on failure.
 *
 * @note Not suitable for destructors (use HIP_EXPECT there)
 */
#ifndef HIP_CHECK
#define HIP_CHECK(cmd) ASSERT_EQ((cmd), hipSuccess)
#endif

/**
 * @def HIP_EXPECT
 * @brief Non-fatal HIP check safe for destructors and non-void contexts
 *
 * Reports failure but continues execution.  Use in destructors where
 * fatal assertions (ASSERT_*) must not be used.
 */
#ifndef HIP_EXPECT
#define HIP_EXPECT(cmd) EXPECT_EQ((cmd), hipSuccess)
#endif

/**
 * @def HIP_TEST_CHECK_GTEST_FAIL
 * @brief HIP error checking for GTest test bodies with detailed error output
 *
 * Checks HIP function calls and fails the test if an error occurs.
 * Use in TEST_F/TEST_P test bodies.
 *
 * Behavior:
 * - Checks HIP function result
 * - Prints error with file/line to stdout
 * - Calls FAIL() to mark test as failed
 *
 * @note For infrastructure code, use HIPCHECK or HIP_TEST_CHECK instead
 */
#define HIP_TEST_CHECK_GTEST_FAIL(cmd)                                                       \
    do                                                                                       \
    {                                                                                        \
        hipError_t err = cmd;                                                                \
        if(err != hipSuccess)                                                                \
        {                                                                                    \
            printf("HIP Error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            FAIL() << "HIP Error: " << hipGetErrorString(err);                               \
        }                                                                                    \
    }                                                                                        \
    while(0)

// HIP Error Checking Macros (return-based, for ncclResult_t-returning functions)

/**
 * @def HIP_TEST_CHECK
 * @brief HIP error checking macro for test infrastructure code
 *
 * Checks HIP function calls and returns ncclUnhandledCudaError if failed.
 * Use in test setup/teardown and infrastructure code that returns ncclResult_t.
 *
 * Behavior:
 * - Checks HIP function result
 * - Logs error to stderr
 * - Returns ncclUnhandledCudaError to caller
 *
 * @note Requires enclosing function to return ncclResult_t
 * @note For test bodies, use HIP_CHECK or HIP_TEST_CHECK_GTEST_FAIL instead
 */
#define HIP_TEST_CHECK(cmd)                                      \
    do                                                           \
    {                                                            \
        hipError_t err = cmd;                                    \
        if(err != hipSuccess)                                    \
        {                                                        \
            fprintf(stderr,                                      \
                    "HIP Error at %s:%d - %s (hipError_t=%d)\n", \
                    __FILE__,                                    \
                    __LINE__,                                    \
                    hipGetErrorString(err),                      \
                    static_cast<int>(err));                      \
            return ncclUnhandledCudaError;                       \
        }                                                        \
    }                                                            \
    while(0)

/**
 * @def HIPCHECK
 * @brief HIP error checking macro (library-style)
 *
 * Similar to RCCL library's CUDACHECK macro. Returns ncclUnhandledCudaError on error.
 * Use in any code that returns ncclResult_t.
 *
 * Behavior:
 * - Checks HIP function result
 * - Logs error to stderr
 * - Returns ncclUnhandledCudaError to caller
 *
 * @note Requires enclosing function to return ncclResult_t
 * @note For GTest test bodies, use HIP_CHECK or HIP_TEST_CHECK_GTEST_FAIL instead
 * @note This mirrors the library's CUDACHECK behavior
 */
#ifndef HIPCHECK
    #define HIPCHECK(cmd)                                            \
        do                                                           \
        {                                                            \
            hipError_t err = cmd;                                    \
            if(err != hipSuccess)                                    \
            {                                                        \
                fprintf(stderr,                                      \
                        "HIP Error at %s:%d - %s (hipError_t=%d)\n", \
                        __FILE__,                                    \
                        __LINE__,                                    \
                        hipGetErrorString(err),                      \
                        static_cast<int>(err));                      \
                return ncclUnhandledCudaError;                       \
            }                                                        \
        }                                                            \
        while(0)
#endif // HIPCHECK

// NCCL Error Checking Macros

/**
 * @def RCCL_TEST_CHECK
 * @brief NCCL error checking macro for test infrastructure code
 *
 * Checks NCCL function calls and returns error code if failed.
 * Use in test setup/teardown and infrastructure code that returns ncclResult_t.
 *
 * Behavior:
 * - Checks NCCL function result
 * - Logs error to stderr
 * - Returns the error code to caller
 *
 * @note For GTest test bodies, use RCCL_TEST_CHECK_GTEST_FAIL instead
 */
#define RCCL_TEST_CHECK(cmd)                            \
    do                                                  \
    {                                                   \
        ncclResult_t res = cmd;                         \
        if(res != ncclSuccess && res != ncclInProgress) \
        {                                               \
            fprintf(stderr,                             \
                    "RCCL Error at %s:%d - %s\n",       \
                    __FILE__,                           \
                    __LINE__,                           \
                    ncclGetErrorString(res));           \
            return res;                                 \
        }                                               \
    }                                                   \
    while(0)

/**
 * @def RCCL_TEST_CHECK_GTEST_FAIL
 * @brief RCCL error checking macro for GTest test bodies
 *
 * Checks NCCL function calls and fails the test if an error occurs.
 * Use in TEST_F/TEST_P test bodies.
 *
 * Behavior:
 * - Checks NCCL function result
 * - Prints error to stdout
 * - Calls FAIL() to mark test as failed
 *
 * @note For infrastructure code (setup/teardown), use RCCL_TEST_CHECK instead
 */
#define RCCL_TEST_CHECK_GTEST_FAIL(cmd)                                                        \
    do                                                                                         \
    {                                                                                          \
        ncclResult_t res = cmd;                                                                \
        if(res != ncclSuccess)                                                                 \
        {                                                                                      \
            printf("RCCL Error at %s:%d - %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            FAIL() << "RCCL Error: " << ncclGetErrorString(res);                               \
        }                                                                                      \
    }                                                                                          \
    while(0)

// ============================================================================
// MPI-only macros — require MPI_TESTS_ENABLED
// ============================================================================

#ifdef MPI_TESTS_ENABLED

#include <cstring>
#include <mpi.h>
#include "utils.h" // For RCCL's getHostName utility

// Forward declaration of MPIEnvironment class (defined in MPIEnvironment.hpp)
class MPIEnvironment;

// Forward declarations for helper functions
extern int         rcclTestDebugLevel;
inline int         getTestDebugLevel();
inline int         getTestMpiRank();
inline const char* getTestHostname();
inline bool        isMultiNodeTest();

// Helper function implementations
inline int getTestDebugLevel()
{
    return rcclTestDebugLevel;
}

inline int getTestMpiRank()
{
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline const char* getTestHostname()
{
    static char hostname[256] = {0};
    static bool initialized   = false;

    if(!initialized)
    {
        // Use RCCL's getHostName utility to get short hostname (delimited by '.')
        if(getHostName(hostname, sizeof(hostname), '.') != ncclSuccess)
        {
            strncpy(hostname, "unknown", sizeof(hostname) - 1);
        }
        initialized = true;
    }
    return hostname;
}

// Forward declaration of helper function to access MPIEnvironment state
// (Defined in MPIEnvironment.cpp to avoid circular dependency)
int getMPIEnvironmentCachedMultiNodeResult();

inline bool isMultiNodeTest()
{
    // Return cached result from global environment
    // If not yet computed (== -1), assume single node to be safe
    return getMPIEnvironmentCachedMultiNodeResult() == 1;
}

// Debug Logging Macros (TEST_*)

/**
 * @def TEST_WARN
 * @brief Warning-level logging macro
 *
 * Prints warning messages when NCCL_DEBUG=WARN or higher.
 * Automatically includes rank and hostname prefixes.
 */
#define TEST_WARN(...)                                                 \
    do                                                                 \
    {                                                                  \
        if(getTestDebugLevel() >= 2)                                   \
        {                                                              \
            int rank = getTestMpiRank();                               \
            if(isMultiNodeTest())                                      \
            {                                                          \
                printf("%s:[%d] TEST WARN ", getTestHostname(), rank); \
            }                                                          \
            else                                                       \
            {                                                          \
                printf("[%d] TEST WARN ", rank);                       \
            }                                                          \
            printf(__VA_ARGS__);                                       \
            printf("\n");                                              \
            fflush(stdout);                                            \
        }                                                              \
    }                                                                  \
    while(0)

/**
 * @def TEST_INFO
 * @brief Info-level logging macro
 *
 * Prints informational messages when NCCL_DEBUG=INFO or higher.
 * Automatically includes rank and hostname prefixes.
 */
#define TEST_INFO(...)                                                 \
    do                                                                 \
    {                                                                  \
        if(getTestDebugLevel() >= 3)                                   \
        {                                                              \
            int rank = getTestMpiRank();                               \
            if(isMultiNodeTest())                                      \
            {                                                          \
                printf("%s:[%d] TEST INFO ", getTestHostname(), rank); \
            }                                                          \
            else                                                       \
            {                                                          \
                printf("[%d] TEST INFO ", rank);                       \
            }                                                          \
            printf(__VA_ARGS__);                                       \
            printf("\n");                                              \
            fflush(stdout);                                            \
        }                                                              \
    }                                                                  \
    while(0)

/**
 * @def TEST_ABORT
 * @brief Abort-level logging macro
 *
 * Prints abort-level messages when NCCL_DEBUG=ABORT or higher.
 * Automatically includes rank and hostname prefixes.
 */
#define TEST_ABORT(...)                                                 \
    do                                                                  \
    {                                                                   \
        if(getTestDebugLevel() >= 4)                                    \
        {                                                               \
            int rank = getTestMpiRank();                                \
            if(isMultiNodeTest())                                       \
            {                                                           \
                printf("%s:[%d] TEST ABORT ", getTestHostname(), rank); \
            }                                                           \
            else                                                        \
            {                                                           \
                printf("[%d] TEST ABORT ", rank);                       \
            }                                                           \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
            fflush(stdout);                                             \
        }                                                               \
    }                                                                   \
    while(0)

/**
 * @def TEST_TRACE
 * @brief Trace-level logging macro
 *
 * Prints trace messages when NCCL_DEBUG=TRACE.
 * Automatically includes rank and hostname prefixes.
 */
    #define TEST_TRACE(...)                                                 \
        do                                                                  \
        {                                                                   \
            if(getTestDebugLevel() >= 5)                                    \
            {                                                               \
                int rank = getTestMpiRank();                                \
                if(isMultiNodeTest())                                       \
                {                                                           \
                    printf("%s:[%d] TEST TRACE ", getTestHostname(), rank); \
                }                                                           \
                else                                                        \
                {                                                           \
                    printf("[%d] TEST TRACE ", rank);                       \
                }                                                           \
                printf(__VA_ARGS__);                                        \
                printf("\n");                                               \
                fflush(stdout);                                             \
            }                                                               \
        }                                                                   \
        while(0)

    // MPI-Aware Assertion Macros (ASSERT_MPI_*)
    //
    // These macros coordinate across MPI ranks using MPI_Allreduce so that
    // if ANY rank's assertion fails, ALL ranks exit together (preventing
    // deadlocks in subsequent collective operations).
    //
    // Failure semantics:
    //   - Failing rank:  FAIL()       -> test marked FAILED with error details
    //   - Passing ranks: GTEST_SKIP() -> clean exit to avoid MPI deadlock

/**
 * @def ASSERT_MPI_TRUE
 * @brief MPI-aware version of ASSERT_TRUE
 *
 * Checks condition on all ranks. If ANY rank fails:
 * - Failing rank calls FAIL() (test marked FAILED)
 * - Passing ranks call GTEST_SKIP() (clean coordinated exit)
 *
 * @param condition The condition to test
 */
#define ASSERT_MPI_TRUE(condition)                                                            \
    do                                                                                        \
    {                                                                                         \
        bool _local_pass    = static_cast<bool>(condition);                                   \
        int  _local_status  = _local_pass ? 1 : 0;                                            \
        int  _global_status = 0;                                                              \
        MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                              \
        if(_global_status == 0)                                                               \
        {                                                                                     \
            if(!_local_pass)                                                                  \
            {                                                                                 \
                FAIL()                                                                        \
                    << "Rank " << MPIEnvironment::world_rank                                  \
                    << ": ASSERT_MPI_TRUE(" << #condition << ") failed";                      \
            }                                                                                 \
            GTEST_SKIP()                                                                      \
                << "Rank " << MPIEnvironment::world_rank                                      \
                << ": Aborting - assertion failed on another rank";                            \
        }                                                                                     \
    }                                                                                         \
    while(0)

/**
 * @def ASSERT_MPI_FALSE
 * @brief MPI-aware version of ASSERT_FALSE
 */
#define ASSERT_MPI_FALSE(condition) ASSERT_MPI_TRUE(!(condition))

/**
 * @def ASSERT_MPI_EQ
 * @brief MPI-aware version of ASSERT_EQ
 *
 * Checks if val1 == val2 on all ranks. If ANY rank fails:
 * - Failing rank calls FAIL() (test marked FAILED)
 * - Passing ranks call GTEST_SKIP() (clean coordinated exit)
 *
 * @param val1 First value
 * @param val2 Second value
 */
#define ASSERT_MPI_EQ(val1, val2)                                                             \
    do                                                                                        \
    {                                                                                         \
        auto _v1            = (val1);                                                         \
        auto _v2            = (val2);                                                         \
        bool _local_pass    = (_v1 == _v2);                                                   \
        int  _local_status  = _local_pass ? 1 : 0;                                            \
        int  _global_status = 0;                                                              \
        MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                              \
        if(_global_status == 0)                                                               \
        {                                                                                     \
            if(!_local_pass)                                                                  \
            {                                                                                 \
                FAIL()                                                                        \
                    << "Rank " << MPIEnvironment::world_rank                                  \
                    << ": ASSERT_MPI_EQ(" << #val1 << ", " << #val2 << ") failed";            \
            }                                                                                 \
            GTEST_SKIP()                                                                      \
                << "Rank " << MPIEnvironment::world_rank                                      \
                << ": Aborting - assertion failed on another rank";                            \
        }                                                                                     \
    }                                                                                         \
    while(0)

/**
 * @def ASSERT_MPI_NE
 * @brief MPI-aware version of ASSERT_NE
 *
 * @param val1 First value
 * @param val2 Second value
 */
#define ASSERT_MPI_NE(val1, val2)                                                             \
    do                                                                                        \
    {                                                                                         \
        auto _v1            = (val1);                                                         \
        auto _v2            = (val2);                                                         \
        bool _local_pass    = (_v1 != _v2);                                                   \
        int  _local_status  = _local_pass ? 1 : 0;                                            \
        int  _global_status = 0;                                                              \
        MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                              \
        if(_global_status == 0)                                                               \
        {                                                                                     \
            if(!_local_pass)                                                                  \
            {                                                                                 \
                FAIL()                                                                        \
                    << "Rank " << MPIEnvironment::world_rank                                  \
                    << ": ASSERT_MPI_NE(" << #val1 << ", " << #val2 << ") failed";            \
            }                                                                                 \
            GTEST_SKIP()                                                                      \
                << "Rank " << MPIEnvironment::world_rank                                      \
                << ": Aborting - assertion failed on another rank";                            \
        }                                                                                     \
    }                                                                                         \
    while(0)

/**
 * @def ASSERT_MPI_GT
 * @brief MPI-aware version of ASSERT_GT
 *
 * Checks if val1 > val2 on all ranks. If ANY rank fails:
 * - Failing rank calls FAIL() (test marked FAILED)
 * - Passing ranks call GTEST_SKIP() (clean coordinated exit)
 *
 * @param val1 First value (expected to be greater)
 * @param val2 Second value
 */
#define ASSERT_MPI_GT(val1, val2)                                                             \
    do                                                                                        \
    {                                                                                         \
        auto _v1            = (val1);                                                         \
        auto _v2            = (val2);                                                         \
        bool _local_pass    = (_v1 > _v2);                                                    \
        int  _local_status  = _local_pass ? 1 : 0;                                            \
        int  _global_status = 0;                                                              \
        MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                              \
        if(_global_status == 0)                                                               \
        {                                                                                     \
            if(!_local_pass)                                                                  \
            {                                                                                 \
                FAIL()                                                                        \
                    << "Rank " << MPIEnvironment::world_rank                                  \
                    << ": ASSERT_MPI_GT(" << #val1 << ", " << #val2 << ") failed";            \
            }                                                                                 \
            GTEST_SKIP()                                                                      \
                << "Rank " << MPIEnvironment::world_rank                                      \
                << ": Aborting - assertion failed on another rank";                            \
        }                                                                                     \
    }                                                                                         \
    while(0)

/**
 * @def ASSERT_MPI_SUCCESS
 * @brief MPI-aware assertion for MPI operations
 *
 * Checks if MPI operation succeeded on all ranks. If ANY rank fails:
 * - Failing rank calls FAIL() with MPI error string
 * - Passing ranks call GTEST_SKIP() (clean coordinated exit)
 *
 * @param expr Expression that returns an MPI error code
 */
#define ASSERT_MPI_SUCCESS(expr)                                                               \
    do                                                                                         \
    {                                                                                          \
        int  _result        = (expr);                                                          \
        bool _local_pass    = (_result == MPI_SUCCESS);                                        \
        int  _local_status  = _local_pass ? 1 : 0;                                             \
        int  _global_status = 0;                                                               \
        MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);   \
                                                                                               \
        if(_global_status == 0)                                                                \
        {                                                                                      \
            if(!_local_pass)                                                                   \
            {                                                                                  \
                char _error_string[MPI_MAX_ERROR_STRING];                                      \
                int  _len;                                                                     \
                MPI_Error_string(_result, _error_string, &_len);                               \
                FAIL() << "Rank " << MPIEnvironment::world_rank                                \
                       << ": MPI operation failed: " << _error_string;                         \
            }                                                                                  \
            GTEST_SKIP() << "Rank " << MPIEnvironment::world_rank                              \
                         << ": Aborting - MPI failure on another rank";                        \
        }                                                                                      \
    }                                                                                          \
    while(0)

// MPI Error Checking Macros (MPICHECK)

/**
 * @def MPICHECK
 * @brief Context-aware MPI error checking macro with overloaded behavior
 *
 * Provides three usage modes depending on context:
 *
 * @par Usage Modes:
 * - `MPICHECK(cmd)` - Normal test code: Fails test with FAIL() on error
 * - `MPICHECK(cmd, rank)` - Cleanup code: Calls MPI_Abort() on error
 * - `MPICHECK(cmd, rank, true)` - MPI_Finalize: Calls std::exit() on error
 *
 * @par Example:
 * @code
 * // In test body
 * MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
 *
 * // In cleanup code
 * MPICHECK(MPI_Barrier(MPI_COMM_WORLD), world_rank);
 *
 * // During finalization
 * MPICHECK(MPI_Finalize(), world_rank, true);
 * @endcode
 *
 * @note Prints detailed error message including file, line, and MPI error string
 */

// Helper macros for argument counting
#define MPICHECK_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define MPICHECK(...) \
    MPICHECK_GET_MACRO(__VA_ARGS__, MPICHECK_3, MPICHECK_2, MPICHECK_1)(__VA_ARGS__)

/**
 * @def MPICHECK_1
 * @brief 1-argument version: Normal test code (uses FAIL())
 * @hideinitializer
 */
#define MPICHECK_1(cmd)                                                            \
    do                                                                             \
    {                                                                              \
        int err = cmd;                                                             \
        if(err != MPI_SUCCESS)                                                     \
        {                                                                          \
            char error_string[MPI_MAX_ERROR_STRING];                               \
            int  length;                                                           \
            MPI_Error_string(err, error_string, &length);                          \
            printf("MPI Error at %s:%d - %s\n", __FILE__, __LINE__, error_string); \
            FAIL() << "MPI Error: " << error_string;                               \
        }                                                                          \
    }                                                                              \
    while(0)

/**
 * @def MPICHECK_2
 * @brief 2-argument version: Cleanup code (uses MPI_Abort())
 * @hideinitializer
 */
#define MPICHECK_2(cmd, rank)                                  \
    do                                                         \
    {                                                          \
        int err = cmd;                                         \
        if(err != MPI_SUCCESS)                                 \
        {                                                      \
            char error_string[MPI_MAX_ERROR_STRING];           \
            int  length;                                       \
            MPI_Error_string(err, error_string, &length);      \
            std::fprintf(stderr,                               \
                         "Rank %d: MPI Error at %s:%d - %s\n", \
                         rank,                                 \
                         __FILE__,                             \
                         __LINE__,                             \
                         error_string);                        \
            std::fflush(stderr);                               \
            MPI_Abort(MPI_COMM_WORLD, err);                    \
        }                                                      \
    }                                                          \
    while(0)

/**
 * @def MPICHECK_3
 * @brief 3-argument version: MPI_Finalize (uses std::exit())
 * @hideinitializer
 */
#define MPICHECK_3(cmd, rank, is_finalize)                              \
    do                                                                  \
    {                                                                   \
        int err = cmd;                                                  \
        if(err != MPI_SUCCESS)                                          \
        {                                                               \
            char error_string[MPI_MAX_ERROR_STRING];                    \
            int  length;                                                \
            MPI_Error_string(err, error_string, &length);               \
            std::fprintf(stderr,                                        \
                         "Rank %d: MPI_Finalize Error at %s:%d - %s\n", \
                         rank,                                          \
                         __FILE__,                                      \
                         __LINE__,                                      \
                         error_string);                                 \
            std::fflush(stderr);                                        \
            std::exit(err);                                             \
        }                                                               \
    }                                                                   \
    while(0)

#endif // MPI_TESTS_ENABLED

#endif // RCCL_TEST_CHECKS_HPP
