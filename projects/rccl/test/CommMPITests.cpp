/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "TestChecks.hpp"
#include "ResourceGuards.hpp"

#include <cstdlib>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

/**
 * @class ConfigCommMPITestBase
 * @brief Shared fixture that builds the test communicator via
 *        ncclCommInitRankConfig() with RAII cleanup. Subclasses fill in the
 *        ncclConfig_t fields they want to exercise by overriding applyConfig().
 */
class ConfigCommMPITestBase : public MPITestBase
{
protected:
    virtual void applyConfig(ncclConfig_t& config) = 0;

    // Human-readable description of the config under test, for diagnostic logs.
    virtual std::string configLabel() const = 0;

    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
        {
            TEST_INFO("Creating test-specific communicator with %s", configLabel().c_str());
        }

        // Rank 0 generates unique ID
        if(world_rank == 0)
        {
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        }

        // Broadcast ID to all ranks
        MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Let the subclass populate the ncclConfig_t fields under test.
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        applyConfig(config);

        // Initialize NCCL communicator with automatic cleanup on error
        RCCL_TEST_CHECK(ncclGroupStart());

        // RAII guard: Automatically calls ncclGroupEnd() if subsequent operations fail
        auto group_guard = makeScopeGuard([]() { (void)ncclGroupEnd(); });

        RCCL_TEST_CHECK(ncclCommInitRankConfig(&test_comm_, world_size, nccl_id_, world_rank, &config));

        // RAII guard: Automatically destroys test_comm_ if subsequent operations fail
        auto comm_guard = makeScopeGuard(
            [this]()
            {
                if(test_comm_)
                {
                    (void)ncclCommDestroy(test_comm_);
                    test_comm_ = nullptr;
                }
            });

        RCCL_TEST_CHECK(ncclGroupEnd());
        group_guard.dismiss(); // ncclGroupEnd succeeded, don't call it again

        // Create HIP stream - if this fails, comm_guard automatically cleans up test_comm_
        HIP_TEST_CHECK(hipStreamCreate(&test_stream_));

        // RAII guard: Automatically destroys test_stream_ if subsequent operations fail
        auto stream_guard = makeScopeGuard(
            [this]()
            {
                if(test_stream_)
                {
                    (void)hipStreamDestroy(test_stream_);
                    test_stream_ = nullptr;
                }
            });

        MPI_Barrier(MPI_COMM_WORLD);

        // All succeeded - dismiss guards to keep resources
        comm_guard.dismiss();
        stream_guard.dismiss();

        if(world_rank == 0)
        {
            TEST_INFO("Test-specific communicator created successfully");
        }

        return ncclSuccess;
    }
};

/**
 * @class TrafficClassMPITest
 * @brief Test fixture for Traffic Class (QoS) configuration via ncclConfig_t.
 */
class TrafficClassMPITest : public ConfigCommMPITestBase
{
protected:
    int configured_traffic_class_ = NCCL_CONFIG_UNDEF_INT;

    void applyConfig(ncclConfig_t& config) override
    {
        config.trafficClass = configured_traffic_class_;
    }

    std::string configLabel() const override
    {
        return "trafficClass=" + std::to_string(configured_traffic_class_);
    }
};

/**
 * @test TrafficClassMPITest.ConfiguredTrafficClass
 * @brief Verify traffic class in communicator and in NCCL debug output
 *
 * Uses MPIHelpers::TestLogAssertionContext with makeCombinedAssertionLogOptions():
 * - Sets NCCL_DEBUG_FILE for this scope (before communicator init) for NCCL-native logs.
 * - Optionally matches the same line in rccl_test_rank_<r>.log when
 *   RCCL_MPI_LOG_ALL_RANKS=1 (stderr/tee). Either sink may contain the substring.
 *
 * Requires NCCL_DEBUG=INFO (or higher) for the log line to exist.
 */
TEST_F(TrafficClassMPITest, ConfiguredTrafficClass)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kTestTrafficClass = 46;
    configured_traffic_class_ = kTestTrafficClass;

    MPIHelpers::TestLogAssertionContext log_ctx(
        MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    // Verify trafficClass in communicator
    ASSERT_MPI_EQ(getActiveCommunicator()->config.trafficClass, kTestTrafficClass);

    static constexpr const char* kTrafficClassLogNeedle = "Traffic class set to 46";
    const std::string            from_nccl               = log_ctx.readNcclDebugLog();
    const std::string            from_rank_log           = log_ctx.readPerRankStderrLog();
    const bool hit_nccl   = from_nccl.find(kTrafficClassLogNeedle) != std::string::npos;
    const bool hit_stderr = from_rank_log.find(kTrafficClassLogNeedle) != std::string::npos;
    const bool found_line = hit_nccl || hit_stderr;

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("Expected NCCL log line \"%s\": %s",
                  kTrafficClassLogNeedle,
                  found_line ? "passed" : "failed");
    }

    ASSERT_MPI_TRUE(found_line);
}

/**
 * @class CtaConfigMPITest
 * @brief Fixture for the CTA override paths on AMD GPUs. Injects minCTAs/maxCTAs
 *        through ncclCommInitRankConfig() and inspects the resulting comm->config
 *        and comm->nChannels.
 */
class CtaConfigMPITest : public ConfigCommMPITestBase
{
protected:
    int configured_min_ctas_ = NCCL_CONFIG_UNDEF_INT;
    int configured_max_ctas_ = NCCL_CONFIG_UNDEF_INT;

    void applyConfig(ncclConfig_t& config) override
    {
        config.minCTAs = configured_min_ctas_;
        config.maxCTAs = configured_max_ctas_;
    }

    std::string configLabel() const override
    {
        return "minCTAs=" + std::to_string(configured_min_ctas_)
             + " maxCTAs=" + std::to_string(configured_max_ctas_);
    }
};

/**
 * @test CtaConfigMPITest.ConfigOverrideAppliesMinMaxCTAs
 * @brief ncclConfig_t minCTAs/maxCTAs land in comm->config and clamp
 *        comm->nChannels into [minCTAs, maxCTAs].
 */
TEST_F(CtaConfigMPITest, ConfigOverrideAppliesMinMaxCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    constexpr int kMinCTAs = 2;
    constexpr int kMaxCTAs = 4;
    configured_min_ctas_   = kMinCTAs;
    configured_max_ctas_   = kMaxCTAs;

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Config-override path: ncclConfig_t values are accepted into comm->config.
    ASSERT_MPI_EQ(comm->config.minCTAs, kMinCTAs);
    ASSERT_MPI_EQ(comm->config.maxCTAs, kMaxCTAs);

    ASSERT_MPI_TRUE(comm->nChannels >= kMinCTAs);
    ASSERT_MPI_TRUE(comm->nChannels <= kMaxCTAs);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("minCTAs=%d maxCTAs=%d -> nChannels=%d",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.EnvKnobsApplyMinMaxCTAs
 * @brief NCCL_MIN_CTAS / NCCL_MAX_CTAS env knobs override comm->config; skips when unset.
 */
TEST_F(CtaConfigMPITest, EnvKnobsApplyMinMaxCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    const char* min_env = std::getenv("NCCL_MIN_CTAS");
    const char* max_env = std::getenv("NCCL_MAX_CTAS");
    if(min_env == nullptr && max_env == nullptr)
    {
        GTEST_SKIP() << "NCCL_MIN_CTAS / NCCL_MAX_CTAS not set; run under the CTA env CI tier.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Env-knob path: NCCL_MIN_CTAS / NCCL_MAX_CTAS override comm->config.
    if(min_env != nullptr)
    {
        const int expected_min = std::atoi(min_env);
        ASSERT_MPI_EQ(comm->config.minCTAs, expected_min);
        ASSERT_MPI_TRUE(comm->nChannels >= expected_min);
    }
    if(max_env != nullptr)
    {
        const int expected_max = std::atoi(max_env);
        ASSERT_MPI_EQ(comm->config.maxCTAs, expected_max);
        ASSERT_MPI_TRUE(comm->nChannels <= expected_max);
    }

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("env NCCL_MIN_CTAS=%s NCCL_MAX_CTAS=%s -> config min=%d max=%d nChannels=%d",
                  min_env ? min_env : "(unset)",
                  max_env ? max_env : "(unset)",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.EnvKnobsIgnoreNonPositiveCTAs
 * @brief Negative test: env values <= 0 are rejected and config resolves to its positive default; skips when no knob is non-positive.
 */
TEST_F(CtaConfigMPITest, EnvKnobsIgnoreNonPositiveCTAs)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    const char* min_env     = std::getenv("NCCL_MIN_CTAS");
    const char* max_env     = std::getenv("NCCL_MAX_CTAS");
    const bool  min_nonpos  = (min_env != nullptr && std::atoi(min_env) <= 0);
    const bool  max_nonpos  = (max_env != nullptr && std::atoi(max_env) <= 0);
    if(!min_nonpos && !max_nonpos)
    {
        GTEST_SKIP() << "Negative CTA env test: set NCCL_MIN_CTAS and/or NCCL_MAX_CTAS <= 0.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    if(min_nonpos)
    {
        ASSERT_MPI_TRUE(comm->config.minCTAs > 0);
    }
    if(max_nonpos)
    {
        ASSERT_MPI_TRUE(comm->config.maxCTAs > 0);
    }

    // Comm still initialized with a usable channel count (not clamped to 0).
    ASSERT_MPI_TRUE(comm->nChannels >= 1);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("rejected non-positive CTAs -> config min=%d max=%d nChannels=%d",
                  comm->config.minCTAs,
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

/**
 * @test CtaConfigMPITest.DefaultConfigDoesNotClampChannels
 * @brief Guard: with no min/maxCTAs (config or env), the maxCTAs clamp is a no-op and does not reduce nChannels; skips when the env knobs are set.
 */
TEST_F(CtaConfigMPITest, DefaultConfigDoesNotClampChannels)
{
    ASSERT_MPI_TRUE(validateTestPrerequisites(kMinProcessesForMPI));

    if(std::getenv("NCCL_MIN_CTAS") != nullptr || std::getenv("NCCL_MAX_CTAS") != nullptr)
    {
        GTEST_SKIP() << "NCCL_MIN_CTAS / NCCL_MAX_CTAS set; this test validates the default (unset) path.";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();
    ASSERT_MPI_TRUE(comm != nullptr);

    // Default maxCTAs (MAXCHANNELS) must be >= nChannels, i.e. the cap did not reduce the channel count.
    ASSERT_MPI_TRUE(comm->nChannels >= 1);
    ASSERT_MPI_TRUE(comm->config.maxCTAs >= comm->nChannels);

    if(getTestMpiRank() == 0)
    {
        TEST_INFO("default config -> maxCTAs=%d nChannels=%d (cap is a no-op)",
                  comm->config.maxCTAs,
                  comm->nChannels);
    }
}

#endif // MPI_TESTS_ENABLED
