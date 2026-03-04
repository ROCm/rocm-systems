/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <cstdlib>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace ImplicitLaunchOrderConstants
{
constexpr size_t kBufferElements    = 64 * 1024;  // Larger buffers = longer kernel execution
constexpr size_t kBufferSize        = kBufferElements * sizeof(float);
constexpr int    kChainLength       = 8;          // More operations = more race opportunities
constexpr int    kIterations        = 20;         // More iterations = higher chance to catch races
constexpr float  kValidationEpsilon = 1e-3f;
} // namespace ImplicitLaunchOrderConstants

using namespace ImplicitLaunchOrderConstants;

class ImplicitLaunchOrderMPITest : public MPITestBase
{
protected:
    std::vector<hipStream_t> streams_;
    std::vector<void*>       buffers_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        streams_.clear();
        buffers_.clear();
    }

    void TearDown() override
    {
        for(auto& buf : buffers_)
        {
            if(buf)
            {
                hipFree(buf);
                buf = nullptr;
            }
        }

        for(auto& stream : streams_)
        {
            if(stream)
            {
                hipStreamDestroy(stream);
                stream = nullptr;
            }
        }

        MPITestBase::TearDown();
    }

    bool allocateStreams(int num_streams)
    {
        streams_.resize(num_streams);
        for(int i = 0; i < num_streams; i++)
        {
            if(hipStreamCreate(&streams_[i]) != hipSuccess)
            {
                return false;
            }
        }
        return true;
    }

    bool allocateBuffers(int num_buffers)
    {
        buffers_.resize(num_buffers);
        for(int i = 0; i < num_buffers; i++)
        {
            if(hipMalloc(&buffers_[i], kBufferSize) != hipSuccess)
            {
                return false;
            }
        }
        return true;
    }

    static bool isImplicitLaunchOrderEnabled()
    {
        const char* env = getenv("NCCL_LAUNCH_ORDER_IMPLICIT");
        return env != nullptr && atoi(env) != 0;
    }

    bool runDependentChain(int nranks, float& actual_value)
    {
        hipError_t err = initializeBufferWithPattern<float>(
            buffers_[0],
            kBufferElements,
            [rank = MPIEnvironment::world_rank](size_t) {
                return static_cast<float>(rank + 1);
            });
        if(err != hipSuccess) return false;

        for(int i = 1; i <= kChainLength; i++)
        {
            if(hipMemset(buffers_[i], 0, kBufferSize) != hipSuccess)
                return false;
        }

        if(hipDeviceSynchronize() != hipSuccess) return false;
        MPI_Barrier(MPI_COMM_WORLD);

        // Launch all operations as fast as possible without waiting
        for(int i = 0; i < kChainLength; i++)
        {
            ncclResult_t result = ncclAllReduce(buffers_[i],
                                                 buffers_[i + 1],
                                                 kBufferElements,
                                                 ncclFloat,
                                                 ncclSum,
                                                 getActiveCommunicator(),
                                                 streams_[i]);
            if(result != ncclSuccess) return false;
        }

        for(int i = 0; i < kChainLength; i++)
        {
            if(hipStreamSynchronize(streams_[i]) != hipSuccess)
                return false;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(hipMemcpy(&actual_value,
                     buffers_[kChainLength],
                     sizeof(float),
                     hipMemcpyDeviceToHost) != hipSuccess)
            return false;

        return true;
    }
};

TEST_F(ImplicitLaunchOrderMPITest, DependentOperationsChain)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met";

    bool implicit_order_enabled = isImplicitLaunchOrderEnabled();

    if(MPIEnvironment::world_rank == 0)
    {
        TEST_INFO("NCCL_LAUNCH_ORDER_IMPLICIT=%s", implicit_order_enabled ? "1" : "0");
        TEST_INFO("Chain length: %d, Buffer: %zu KB, Iterations: %d",
                  kChainLength, kBufferSize / 1024, kIterations);
    }

    ASSERT_TRUE(allocateStreams(kChainLength));
    ASSERT_TRUE(allocateBuffers(kChainLength + 1));
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int nranks = MPIEnvironment::world_size;

    // Expected: sum(1..nranks) * nranks^(chainLength-1)
    double expected_value = static_cast<double>(nranks * (nranks + 1) / 2);
    for(int i = 1; i < kChainLength; i++)
    {
        expected_value *= static_cast<double>(nranks);
    }

    int   correct_count = 0;
    int   wrong_count   = 0;
    bool  all_same      = true;
    float first_result  = 0.0f;

    for(int iter = 0; iter < kIterations; iter++)
    {
        float actual_value = 0.0f;
        ASSERT_TRUE(runDependentChain(nranks, actual_value));

        bool correct = (std::abs(actual_value - expected_value) < kValidationEpsilon * expected_value);
        if(correct)
            correct_count++;
        else
            wrong_count++;

        if(iter == 0)
            first_result = actual_value;
        else if(std::abs(actual_value - first_result) > kValidationEpsilon * expected_value)
            all_same = false;
    }

    if(MPIEnvironment::world_rank == 0)
    {
        TEST_INFO("Expected: %.0f, Correct: %d/%d, Consistent: %s",
                  expected_value, correct_count, kIterations, all_same ? "yes" : "no");
    }

    if(implicit_order_enabled)
    {
        EXPECT_EQ(correct_count, kIterations)
            << "With NCCL_LAUNCH_ORDER_IMPLICIT=1, all iterations should be correct";
        EXPECT_TRUE(all_same)
            << "With NCCL_LAUNCH_ORDER_IMPLICIT=1, results should be consistent";
    }
    else
    {
        if(MPIEnvironment::world_rank == 0)
        {
            if(wrong_count > 0 || !all_same)
            {
                TEST_INFO("Race detected: %d wrong, %s",
                          wrong_count, all_same ? "consistent" : "inconsistent");
            }
            else
            {
                TEST_INFO("No race detected (non-deterministic)");
            }
        }
    }
}

#endif // MPI_TESTS_ENABLED
