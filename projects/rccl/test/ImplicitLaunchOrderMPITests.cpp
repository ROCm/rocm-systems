/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
constexpr size_t kBufferElements    = 64 * 1024;
constexpr size_t kBufferSize        = kBufferElements * sizeof(float);
constexpr int    kNumCommunicators  = 4;
constexpr int    kIterations        = 20;
constexpr float  kValidationEpsilon = 1e-3f;
} // namespace ImplicitLaunchOrderConstants

using namespace ImplicitLaunchOrderConstants;

class ImplicitLaunchOrderMPITest : public MPITestBase
{
protected:
    std::vector<NcclCommAutoGuard>     comm_guards_;
    std::vector<HipStreamAutoGuard>    stream_guards_;
    std::vector<DeviceBufferAutoGuard> buffer_guards_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        comm_guards_.clear();
        stream_guards_.clear();
        buffer_guards_.clear();
    }

    void TearDown() override
    {
        // Destroy child comms before parent (cleaned up by base class)
        comm_guards_.clear();
        buffer_guards_.clear();
        stream_guards_.clear();
        MPITestBase::TearDown();
    }

    ncclResult_t allocateStreams(int num_streams)
    {
        stream_guards_.reserve(num_streams);
        for(int i = 0; i < num_streams; i++)
        {
            hipStream_t stream{};
            HIPCHECK(hipStreamCreate(&stream));
            stream_guards_.push_back(makeStreamAutoGuard(stream));
        }
        return ncclSuccess;
    }

    ncclResult_t allocateBuffers(int num_buffers)
    {
        buffer_guards_.reserve(num_buffers);
        for(int i = 0; i < num_buffers; i++)
        {
            void* buf = nullptr;
            HIPCHECK(hipMalloc(&buf, kBufferSize));
            buffer_guards_.push_back(makeDeviceBufferAutoGuard(buf));
        }
        return ncclSuccess;
    }

    ncclResult_t splitCommunicators(int num_comms)
    {
        comm_guards_.reserve(num_comms);
        ncclComm_t parent = getActiveCommunicator();
        int        rank   = MPIEnvironment::world_rank;

        for(int i = 0; i < num_comms; i++)
        {
            ncclComm_t comm{};
            RCCL_TEST_CHECK(ncclCommSplit(parent, 0, rank, &comm, nullptr));
            comm_guards_.push_back(makeCommAutoGuard(comm));
        }
        return ncclSuccess;
    }

    static bool isImplicitLaunchOrderEnabled()
    {
        const char* env = getenv("NCCL_LAUNCH_ORDER_IMPLICIT");
        return env != nullptr && atoi(env) != 0;
    }

    ncclResult_t runMultiCommChain()
    {
        HIPCHECK(initializeBufferWithPattern<float>(
            buffer_guards_[0].get(),
            kBufferElements,
            [rank = MPIEnvironment::world_rank](size_t) {
                return static_cast<float>(rank + 1);
            }));

        for(int i = 1; i <= kNumCommunicators; i++)
        {
            HIPCHECK(zeroInitializeBuffer<float>(buffer_guards_[i].get(), kBufferElements));
        }

        for(int i = 0; i < kNumCommunicators; i++)
        {
            RCCL_TEST_CHECK(ncclAllReduce(buffer_guards_[i].get(),
                                          buffer_guards_[i + 1].get(),
                                          kBufferElements,
                                          getNcclDataType<float>(),
                                          ncclSum,
                                          comm_guards_[i].get(),
                                          stream_guards_[i].get()));
        }

        for(int i = 0; i < kNumCommunicators; i++)
        {
            HIPCHECK(hipStreamSynchronize(stream_guards_[i].get()));
        }

        return ncclSuccess;
    }
};

TEST_F(ImplicitLaunchOrderMPITest, MultiCommunicatorChain)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met";

    bool implicit_order_enabled = isImplicitLaunchOrderEnabled();

    TEST_INFO("NCCL_LAUNCH_ORDER_IMPLICIT=%s", implicit_order_enabled ? "1" : "0");
    TEST_INFO("Communicators: %d, Buffer: %zu KB, Iterations: %d",
              kNumCommunicators, kBufferSize / 1024, kIterations);

    ASSERT_MPI_EQ(ncclSuccess, allocateStreams(kNumCommunicators));
    ASSERT_MPI_EQ(ncclSuccess, allocateBuffers(kNumCommunicators + 1));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess, splitCommunicators(kNumCommunicators));

    int nranks = MPIEnvironment::world_size;

    // Expected: sum(1..nranks) * nranks^(numComms-1)
    double expected_value = static_cast<double>(nranks * (nranks + 1) / 2);
    for(int i = 1; i < kNumCommunicators; i++)
    {
        expected_value *= static_cast<double>(nranks);
    }

    float expected_f = static_cast<float>(expected_value);

    int   correct_count = 0;
    int   wrong_count   = 0;
    bool  all_same      = true;
    float first_result  = 0.0f;

    for(int iter = 0; iter < kIterations; iter++)
    {
        ASSERT_MPI_EQ(ncclSuccess, runMultiCommChain());

        bool correct = verifyBufferData<float>(
            buffer_guards_[kNumCommunicators].get(),
            kBufferElements,
            [expected_f](size_t) { return expected_f; },
            0,
            static_cast<double>(kValidationEpsilon * expected_value));

        if(correct)
            correct_count++;
        else
            wrong_count++;

        auto [dl_err, host_data] = downloadBuffer<float>(
            buffer_guards_[kNumCommunicators].get(), 1);
        ASSERT_MPI_EQ(dl_err, hipSuccess);

        if(iter == 0)
            first_result = host_data[0];
        else if(std::abs(host_data[0] - first_result) > kValidationEpsilon * expected_value)
            all_same = false;
    }

    TEST_INFO("Expected: %.0f, Correct: %d/%d, Consistent: %s",
              expected_value, correct_count, kIterations, all_same ? "yes" : "no");

    if(implicit_order_enabled)
    {
        EXPECT_EQ(correct_count, kIterations)
            << "With NCCL_LAUNCH_ORDER_IMPLICIT=1, all iterations should be correct";
        EXPECT_TRUE(all_same)
            << "With NCCL_LAUNCH_ORDER_IMPLICIT=1, results should be consistent";
    }
    else
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

#endif // MPI_TESTS_ENABLED
