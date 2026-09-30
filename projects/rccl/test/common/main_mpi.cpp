/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file main_mpi.cpp
 * @brief Main entry point for Google Test-based MPI tests
 *
 * This file provides the main() function for running GTest-based MPI tests.
 * For standalone tests (performance benchmarks, etc.), each test should have
 * its own main() function and use MPIHelpers for common functionality.
 */

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <gtest/gtest.h>

#ifdef MPI_TESTS_ENABLED

    #include "MPIHelpers.hpp"
    #include "MPITestBase.hpp"
    #include "MPIEnvironment.hpp"

namespace
{
struct NetIbThreadConfig
{
    int         nThreads{1};
    bool        valid{true};
    bool        clamped{false};
    std::string error;
};

/**
 * @brief Parse and strip --net_ib_nthreads=N from argv before GTest sees it
 *
 * Mirrors rccl-tests' own -t/--nthreads handling (parsed in its hand-written
 * main() before any collective/threading logic runs): a custom flag must be
 * consumed here since GTest's InitGoogleTest() does not know about it.
 *
 * Parsing is deliberately strict: accepting a malformed value as one worker
 * would make a multithread configuration silently run single-threaded.
 */
NetIbThreadConfig parseAndStripNThreads(int* argc, char** argv)
{
    constexpr const char* kFlagPrefix = "--net_ib_nthreads=";
    const size_t          kPrefixLen  = std::strlen(kFlagPrefix);
    NetIbThreadConfig     config;
    bool                  sawFlag = false;
    // Conflict detection compares the value as written on the command line,
    // before clamping. Comparing post-clamp values would let two distinct
    // out-of-range requests (e.g. 99 and 100) both collapse to kMaxThreads and
    // pass as non-conflicting, so the reject-conflicting-values contract would
    // not hold above the cap.
    int                   rawSeen = 0;

    int writeIdx = 1;
    for(int readIdx = 1; readIdx < *argc; ++readIdx)
    {
        if(std::strncmp(argv[readIdx], kFlagPrefix, kPrefixLen) != 0)
        {
            argv[writeIdx++] = argv[readIdx];
            continue;
        }

        const char* value = argv[readIdx] + kPrefixLen;
        const char* end   = value + std::strlen(value);
        int         parsed = 0;
        auto [ptr, ec] = std::from_chars(value, end, parsed);
        if(value == end || ec != std::errc{} || ptr != end || parsed < 1)
        {
            config.valid = false;
            if(config.error.empty())
                config.error = "expected a positive integer for --net_ib_nthreads";
            continue;
        }

        const int raw = parsed;

        if(sawFlag && raw != rawSeen)
        {
            config.valid = false;
            if(config.error.empty())
                config.error = "conflicting --net_ib_nthreads values";
            continue;
        }

        if(parsed > MPIEnvironment::kMaxThreads)
        {
            parsed         = MPIEnvironment::kMaxThreads;
            config.clamped = true;
        }

        sawFlag         = true;
        rawSeen         = raw;
        config.nThreads = parsed;
    }

    *argc = writeIdx;
    // C/POSIX guarantee argv[argc] == NULL; MPI_Init_thread() receives this
    // argv, so restore the terminator after compacting.
    argv[writeIdx] = nullptr;
    return config;
}

bool validateNThreadsAcrossRanks(const NetIbThreadConfig& config, const MPIHelpers::MPIContext& mpiCtx)
{
    int localValid  = config.valid ? 1 : 0;
    int allValid    = 0;
    int minNThreads = 0;
    int maxNThreads = 0;
    MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&config.nThreads, &minNThreads, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&config.nThreads, &maxNThreads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if(allValid && minNThreads == maxNThreads) return true;

    if(mpiCtx.world_rank == 0)
    {
        if(!allValid)
        {
            std::fprintf(stderr,
                         "ERROR: invalid --net_ib_nthreads configuration on at least one MPI rank");
            if(!config.error.empty()) std::fprintf(stderr, " (%s)", config.error.c_str());
            std::fprintf(stderr, "\n");
        }
        if(minNThreads != maxNThreads)
        {
            std::fprintf(stderr,
                         "ERROR: --net_ib_nthreads must resolve to one value on all ranks "
                         "(observed %d through %d)\n",
                         minNThreads,
                         maxNThreads);
        }
    }
    return false;
}
} // namespace

int main(int argc, char* argv[])
{
    // Parse our own custom flag before anything else touches argv
    const NetIbThreadConfig threadConfig = parseAndStripNThreads(&argc, argv);

    // Initialize MPI using shared helper
    auto mpi_ctx = MPIHelpers::initializeMPI(&argc, &argv);

    if(!validateNThreadsAcrossRanks(threadConfig, mpi_ctx))
    {
        MPI_Finalize();
        MPIEnvironment::mpi_initialized = false;
        return EXIT_FAILURE;
    }
    MPIEnvironment::nThreads = threadConfig.nThreads;

    const auto world_rank = mpi_ctx.world_rank;
    const auto world_size = mpi_ctx.world_size;

    // Setup per-rank logging using shared helper
    auto       rank_log_config          = MPIHelpers::setupRankLogging(world_rank);
    const auto per_rank_logging_enabled = rank_log_config && rank_log_config->logging_enabled;

    // Print initialization message
    if(world_rank == 0 && !per_rank_logging_enabled)
    {
        TEST_INFO("MPI initialized - World size: %d, Thread support: %d",
                  world_size,
                  mpi_ctx.thread_support);
        if(threadConfig.clamped)
            TEST_INFO("--net_ib_nthreads exceeds maximum %d; using %d workers per rank",
                      MPIEnvironment::kMaxThreads,
                      MPIEnvironment::nThreads);
    }

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Suppress GTest output for non-zero ranks (unless per-rank logging is enabled)
    // This is done by deleting GTest listeners for non-zero ranks
    // Note: stdout/stderr are already redirected for non-zero ranks by setupRankLogging
    if(world_rank != 0 && !per_rank_logging_enabled)
    {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        delete listeners.Release(listeners.default_xml_generator());
    }

    // Set up the RCCL MPI environment for all tests
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment());

    // Run all tests
    const auto ret_code = RUN_ALL_TESTS();

    // Restore original output if per-rank logging was enabled
    if(rank_log_config)
    {
        MPIHelpers::restoreRankLogging(*rank_log_config);
    }

    // MPI_Finalize is called by:
    // 1. MPIEnvironment::TearDown() -> cleanup_mpi() (normal case)
    // 2. MPIEnvironment destructor (safety net if TearDown fails or no tests match)
    return ret_code;
}

#else // MPI_TESTS_ENABLED not defined

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    std::fprintf(stderr,
                 "ERROR: MPI tests are not enabled. Please build with ENABLE_MPI_TESTS=ON\n");
    std::fprintf(stderr, "Usage: cmake -DENABLE_MPI_TESTS=ON -DMPI_PATH=/path/to/mpi ..\n");
    return 1;
}

#endif // MPI_TESTS_ENABLED
