/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "rccl_common.h"

#include <cstdlib>
#include <mpi.h>
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
// Small enough that every chained AllReduce takes the DDA IPC addon path on an 8-rank gfx942/gfx950 node.
constexpr size_t kDdaChainElements  = 4 * 1024;
constexpr int    kDdaIpcRanks       = 8;
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

    // Independent top-level communicators. Children of ncclCommSplit skip ncclDdaIpcCommInit, so only
    // these can take the DDA IPC addon path.
    ncclResult_t initCommunicators(int num_comms)
    {
        comm_guards_.reserve(num_comms);
        const int rank = MPIEnvironment::world_rank;
        for(int i = 0; i < num_comms; i++)
        {
            // Broadcast rank 0's result with the id, so a failed ncclGetUniqueId fails every rank
            // instead of leaving the others in MPI_Bcast.
            struct
            {
                ncclResult_t result;
                ncclUniqueId id;
            } root{ncclSuccess, {}};
            if(rank == 0)
                root.result = ncclGetUniqueId(&root.id);
            if(MPI_Bcast(&root, sizeof(root), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
                return ncclSystemError;
            RCCL_TEST_CHECK(root.result);
            ncclComm_t   comm{};
            ncclResult_t init_result
                = ncclCommInitRank(&comm, MPIEnvironment::world_size, root.id, rank);
            if(init_result == ncclSuccess)
                comm_guards_.push_back(makeCommAutoGuard(comm));
            // Agree on the result before the next MPI_Bcast, so a rank-local init failure
            // returns every rank together.
            int local_ok = init_result == ncclSuccess ? 1 : 0;
            int all_ok   = 0;
            if(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
                return ncclSystemError;
            if(!all_ok)
                return init_result != ncclSuccess ? init_result : ncclInternalError;
        }
        return ncclSuccess;
    }

    static bool isImplicitLaunchOrderEnabled()
    {
        const char* env = getenv("NCCL_LAUNCH_ORDER_IMPLICIT");
        return env != nullptr && atoi(env) != 0;
    }

    ncclResult_t runMultiCommChain(size_t elements = kBufferElements)
    {
        HIPCHECK(initializeBufferWithPattern<float>(
            buffer_guards_[0].get(),
            elements,
            [rank = MPIEnvironment::world_rank](size_t) {
                return static_cast<float>(rank + 1);
            }));

        for(int i = 1; i <= kNumCommunicators; i++)
        {
            HIPCHECK(zeroInitializeBuffer<float>(buffer_guards_[i].get(), elements));
        }

        for(int i = 0; i < kNumCommunicators; i++)
        {
            RCCL_TEST_CHECK(ncclAllReduce(buffer_guards_[i].get(),
                                          buffer_guards_[i + 1].get(),
                                          elements,
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
    SKIP_UNLESS_MPI_PREREQS(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kRequireSingleNode);

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

// Same chain at a DDA IPC size: DDA stays selected under NCCL_LAUNCH_ORDER_IMPLICIT=1 and keeps the order.
TEST_F(ImplicitLaunchOrderMPITest, MultiCommunicatorChainDda)
{
    SKIP_UNLESS_MPI_PREREQS(kDdaIpcRanks, kDdaIpcRanks, kNoPowerOfTwoRequired, 1, kRequireSingleNode);

    const bool implicit_order_enabled = isImplicitLaunchOrderEnabled();
    TEST_INFO("NCCL_LAUNCH_ORDER_IMPLICIT=%s", implicit_order_enabled ? "1" : "0");

    ASSERT_MPI_EQ(ncclSuccess, allocateStreams(kNumCommunicators));
    ASSERT_MPI_EQ(ncclSuccess, allocateBuffers(kNumCommunicators + 1));
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    ASSERT_MPI_EQ(ncclSuccess, initCommunicators(kNumCommunicators));

    // Only a DDA IPC selection exercises the addon bracket; RCCL_DDA_ENABLE=0, NCCL_ALGO, an arch
    // without DDA IPC or a check mode route elsewhere and leave nothing to assert here. Check modes
    // divert the call through ncclEnqueueCheck even though rcclGetCollImplInfo still reports DDA IPC.
    bool        notDda   = false;
    const char* algoName = "?";
    for(int i = 0; i < kNumCommunicators; i++)
    {
        int algo = -1, protocol = -1, maxChannels = -1;
        ASSERT_MPI_EQ(ncclSuccess,
                      rcclGetCollImplInfo(comm_guards_[i].get(), ncclFuncAllReduce, kDdaChainElements,
                                          getNcclDataType<float>(), ncclSum, buffer_guards_[i].get(),
                                          buffer_guards_[i + 1].get(), 0, &algo, &protocol, &maxChannels));
        if(algo != static_cast<int>(RCCL_DDA_IPC) || comm_guards_[i].get()->checkMode != ncclCheckModeDefault)
        {
            notDda = true;
            rcclGetAlgoName(algo, &algoName);
        }
    }
    const std::string skip = mpiCoordinatedSkipReason(notDda, "chained AllReduce does not run on DDA IPC (algorithm or check mode)");
    if(!skip.empty())
        GTEST_SKIP() << skip << " (selected " << algoName << ")";

    const int nranks         = MPIEnvironment::world_size;
    double    expected_value = static_cast<double>(nranks * (nranks + 1) / 2);
    for(int i = 1; i < kNumCommunicators; i++)
        expected_value *= static_cast<double>(nranks);
    const float expected_f = static_cast<float>(expected_value);

    int correct_count = 0;
    for(int iter = 0; iter < kIterations; iter++)
    {
        ASSERT_MPI_EQ(ncclSuccess, runMultiCommChain(kDdaChainElements));
        if(verifyBufferData<float>(buffer_guards_[kNumCommunicators].get(),
                                   kDdaChainElements,
                                   [expected_f](size_t) { return expected_f; },
                                   0,
                                   static_cast<double>(kValidationEpsilon * expected_value)))
            correct_count++;
    }
    TEST_INFO("DDA IPC chain: expected %.0f, correct %d/%d", expected_value, correct_count, kIterations);

    if(implicit_order_enabled)
        EXPECT_EQ(correct_count, kIterations)
            << "With NCCL_LAUNCH_ORDER_IMPLICIT=1, chained DDA AllReduces must observe launch order";
}

#endif // MPI_TESTS_ENABLED
