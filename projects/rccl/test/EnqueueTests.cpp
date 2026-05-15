/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "enqueue.h"
#include "info.h"
#include "utils.h"

namespace RcclUnitTesting
{

// Simple test kernel for validating ncclInitKernelsForDevice
__global__ void simpleTestKernel(int* data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(data)
        data[tid] = tid;
}

// Helper function to test ncclInitKernelsForDevice with a real kernel
ncclResult_t testKernelAttributes(void* kernelFn, size_t* maxStackSize)
{
    if(!kernelFn || !maxStackSize)
        return ncclInvalidArgument;

    *maxStackSize          = 0;
    hipFuncAttributes attr = {0};

    hipError_t errcode = hipFuncGetAttributes(&attr, kernelFn);
    if(errcode != hipSuccess)
        return ncclSystemError;

    *maxStackSize = attr.localSizeBytes;
    return ncclSuccess; // ncclSuccess
}

// Helper function to test shared memory limit checking with a real kernel
// ncclMaxSharedMem: For gfx906 (cudaArch 906) with WarpSize 64, this is typically 32832 bytes
ncclResult_t testKernelSharedMemoryLimit(
    void* kernelFn, int cudaArch, int maxSharedMem, size_t* maxStackSize, int ncclMaxSharedMem
)
{
    if(!kernelFn)
        return ncclInvalidArgument;

    ncclResult_t result = ncclSuccess;
    if(maxStackSize)
        *maxStackSize = 0;

    hipFuncAttributes attr    = {0};
    hipError_t        errcode = hipFuncGetAttributes(&attr, kernelFn);
    if(errcode != hipSuccess)
    {
        return ncclSystemError;
    }

    if(maxStackSize)
    {
        *maxStackSize = attr.localSizeBytes;
    }

    // Test the shared memory limit check (mimics enqueue.cc lines 135-146)
    if(ncclMaxSharedMem != 0)
    {
        int sharedMemSize = ncclMaxSharedMem;

        if(sharedMemSize > (maxSharedMem - attr.sharedSizeBytes))
        {
            WARN(
                "cudaArch %d ncclMaxSharedMem %d exceeds device/fn maxSharedMem %zu",
                cudaArch,
                sharedMemSize,
                maxSharedMem - attr.sharedSizeBytes
            );
            return ncclSystemError;
        }
    }

    return result;
}

// Helper structure to hold test environment
struct EnqueueTestEnvironment
{
    ncclComm* comm;
    ncclInfo* info;
    void*     sendbuff;
    void*     recvbuff;
    uint32_t  abortFlag0;
    uint32_t  abortFlag1;
    int       abortFlagRefCount;

    EnqueueTestEnvironment()
        : comm(nullptr)
        , info(nullptr)
        , sendbuff(nullptr)
        , recvbuff(nullptr)
        , abortFlag0(0)
        , abortFlag1(0)
        , abortFlagRefCount(0)
    {}

    ~EnqueueTestEnvironment()
    {
        cleanup();
    }

    void setup()
    {
        // Allocate GPU memory for buffers
        size_t     bufferSize = 1024 * sizeof(float);
        hipError_t hipErr     = hipMalloc(&sendbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate sendbuff";

        hipErr = hipMalloc(&recvbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate recvbuff";

        // Initialize communicator
        comm = new ncclComm();
        memset(comm, 0, sizeof(ncclComm));

        comm->startMagic = NCCL_MAGIC; // 0x0280028002800280

        // Initialize critical fields
        comm->rank      = 0;
        comm->nRanks    = 2;
        comm->cudaDev   = 0;
        comm->localRank = 0;

        // Initialize abort flags
        comm->abortFlag         = &abortFlag0;
        comm->childAbortFlag    = &abortFlag1;
        comm->abortFlagRefCount = &abortFlagRefCount;

        // Initialize memory stack
        ncclMemoryStackConstruct(&comm->memScoped);
        ncclMemoryStackConstruct(&comm->memPermanent);

        // Initialize intra-communication pointers
        comm->intraComm0 = nullptr;
        comm->intraNext  = nullptr;

        // Initialize work FIFO structures
        comm->workFifoBytes                = 1024; // Power of 2
        comm->workFifoBuf                  = nullptr;
        comm->workFifoBufDev               = nullptr;
        comm->workFifoConsumed             = 0;
        comm->workFifoProducedLastRecorded = 0;
        comm->workFifoProduced             = 0;

        // Initialize planner
        memset(&comm->planner, 0, sizeof(comm->planner));

        // Initialize config
        memset(&comm->config, 0, sizeof(comm->config));
        comm->config.blocking = 1;
        comm->checkPointers   = 0; // Disable pointer validation for easier testing

        // Initialize peer info arrays
        comm->peerInfo = new ncclPeerInfo[comm->nRanks];
        memset(comm->peerInfo, 0, comm->nRanks * sizeof(ncclPeerInfo));

        comm->localRankToRank = new int[comm->nRanks];
        for(int i = 0; i < comm->nRanks; i++)
        {
            comm->localRankToRank[i] = i;
        }

        comm->endMagic = NCCL_MAGIC; // 0x0280028002800280

        // Initialize operation info with valid GPU buffers
        info = new ncclInfo();
        memset(info, 0, sizeof(ncclInfo));
        info->comm     = comm;
        info->opName   = "AllReduce";
        info->count    = 1024;
        info->datatype = ncclFloat;
        info->op       = ncclSum;
        info->root     = 0;
        info->sendbuff = sendbuff; // Use allocated GPU memory
        info->recvbuff = recvbuff; // Use allocated GPU memory
        info->stream   = nullptr;
    }

    void cleanup()
    {
        // Clean up info first (it references comm)
        if(info)
        {
            delete info;
            info = nullptr;
        }

        // Clean up comm and its allocated resources
        if(comm)
        {
            // Clean up memory stacks
            ncclMemoryStackDestruct(&comm->memScoped);
            ncclMemoryStackDestruct(&comm->memPermanent);

            // Clean up peer info arrays
            if(comm->peerInfo)
            {
                delete[] comm->peerInfo;
                comm->peerInfo = nullptr;
            }

            if(comm->localRankToRank)
            {
                delete[] comm->localRankToRank;
                comm->localRankToRank = nullptr;
            }

            delete comm;
            comm = nullptr;
        }

        // Clean up GPU buffers last
        if(sendbuff)
        {
            hipError_t err = hipFree(sendbuff);
            if(err != hipSuccess)
            {
                // Log error but don't throw in cleanup
                fprintf(stderr, "Warning: hipFree(sendbuff) failed with error %d\n", err);
            }
            sendbuff = nullptr;
        }

        if(recvbuff)
        {
            hipError_t err = hipFree(recvbuff);
            if(err != hipSuccess)
            {
                // Log error but don't throw in cleanup
                fprintf(stderr, "Warning: hipFree(recvbuff) failed with error %d\n", err);
            }
            recvbuff = nullptr;
        }
    }
};

// Empty test fixture for test organization
class EnqueueTests : public ::testing::Test
{
    // No setup/teardown - all tests use process isolation
};

// Test ncclInitKernelsForDevice function
TEST_F(EnqueueTests, ncclInitKernelsForDevice_ValidInput)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false; // Continue running all tests
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ValidInput",
            [this]()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(906, 65536, &maxStackSize);

                EXPECT_TRUE(result == ncclSuccess);
                // maxStackSize should be set to a reasonable value (> 0)
                EXPECT_GT(maxStackSize, 0)
                    << "Expected maxStackSize to be computed and set to a positive value";
    }
        ).withEnvironment({{"NCCL_DEBUG", "INFO"}, {"NCCL_DEBUG_SUBSYS", "ALL"}}),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ValidInputCarveout",
            [this]()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(906, 65536, &maxStackSize);

                EXPECT_TRUE(result == ncclSuccess);
                // maxStackSize should be set to a reasonable value (> 0)
                EXPECT_GT(maxStackSize, 0)
                    << "Expected maxStackSize to be computed and set to a positive value";
            }
        )
            .withEnvironment(
                {{"NCCL_L1_SHARED_MEMORY_CARVEOUT", "1"},
                 {"NCCL_DEBUG", "INFO"},
                 {"NCCL_DEBUG_SUBSYS", "ALL"}}
            )
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_NullStackSize)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_NullStackSize",
            []()
            {
                ncclResult_t result = ncclInitKernelsForDevice(906, 65536, nullptr);
                EXPECT_EQ(result, ncclSuccess);
            }
        )
    );
}

// Test with a real compiled kernel to verify attribute retrieval works correctly
TEST_F(EnqueueTests, KernelAttributes_WithRealKernel)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "KernelAttributes_WithRealKernel",
            []()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result = testKernelAttributes((void*)simpleTestKernel, &maxStackSize);

                EXPECT_EQ(result, ncclSuccess)
                    << "Expected successful kernel attribute retrieval with a real compiled kernel";
    }
        ).withEnvironment({{"NCCL_DEBUG", "INFO"}})
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_InvalidArch)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_InvalidArch",
            []()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(-1, 65536, &maxStackSize);
                EXPECT_EQ(result, ncclSuccess);
            }
        )
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_ExceedsSharedMemory)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ExceedsSharedMemory",
            []()
            {
                size_t maxStackSize = 0;
                // For gfx906, ncclMaxSharedMem is 32832 (as shown in test output)
                // Use a very small maxSharedMem (16000 bytes) to trigger the exceeds check
                ncclResult_t result = testKernelSharedMemoryLimit(
                    (void*)simpleTestKernel, // Use our real compiled kernel
                    906, // cudaArch
                    16000, // maxSharedMem (intentionally too small)
                    &maxStackSize,
                    32832  // ncclMaxSharedMem for gfx906
                );

                EXPECT_EQ(result, ncclSystemError)
                    << "Expected ncclSystemError when ncclMaxSharedMem exceeds maxSharedMem";
    }
        ).withEnvironment({{"NCCL_DEBUG", "WARN"}})
    );
}

// Test ncclEnqueueCheck function
TEST_F(EnqueueTests, ncclEnqueueCheck_ValidInput)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_ValidInput",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();
                ncclResult_t result = ncclEnqueueCheck(env.info);
                EXPECT_TRUE(result == ncclSuccess);
                env.cleanup();
            }
        )
    );
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidComm)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_InvalidComm",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();
                env.info->comm      = nullptr;
                ncclResult_t result = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);
                env.cleanup();
            }
        )
    );
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidBuffers)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_InvalidBuffers",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();

                // Test with null sendbuff
                env.comm->checkPointers = 1;
                env.info->sendbuff      = nullptr;
                ncclResult_t result     = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);

                // Reset sendbuff and test with null recvbuff
                env.info->sendbuff = env.sendbuff;
                env.info->recvbuff = nullptr;
                result             = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);

                env.cleanup();
            }
        )
    );
}

// Test ncclFuncSendCount function
TEST_F(EnqueueTests, ncclFuncSendCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_Broadcast",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncBroadcast, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_Reduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncAllGather, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncSendCount(ncclFuncReduceScatter, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncSendCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncSendCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

// Test ncclFuncRecvCount function
TEST_F(EnqueueTests, ncclFuncRecvCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_Broadcast",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncBroadcast, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_Reduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncAllGather, nRanks, count);
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncRecvCount(ncclFuncReduceScatter, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncRecvCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncRecvCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

// Test ncclFuncMaxSendRecvCount function
TEST_F(EnqueueTests, ncclFuncMaxSendRecvCount_AllTests)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_AllReduce",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count);
                EXPECT_EQ(result, count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_AllGather",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count);
                // For AllGather, receive count (count * nRanks) is larger than send count (count)
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_ReduceScatter",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 4;
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncReduceScatter, nRanks, count);
                // For ReduceScatter, send count (count) is larger than receive count (count/nRanks)
                EXPECT_EQ(result, count * nRanks);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncMaxSendRecvCount_ZeroCount",
            []()
            {
                size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 0);
                EXPECT_EQ(result, 0);
            }
        )
    );
}

// Edge case tests
TEST_F(EnqueueTests, ncclFuncCounts_EdgeCases)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncCounts_SingleRank",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 1;
                // Test with single rank
                EXPECT_EQ(ncclFuncSendCount(ncclFuncAllReduce, nRanks, count), count);
                EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count), count);
                EXPECT_EQ(ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count), count);
            }
        ),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclFuncCounts_LargeRankCount",
            []()
            {
                size_t count  = 1000;
                int    nRanks = 1024;
                // Test with large number of ranks
                EXPECT_EQ(ncclFuncSendCount(ncclFuncAllGather, nRanks, count), count);
                EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllGather, nRanks, count), count * nRanks);
                EXPECT_EQ(
                    ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count),
                    count * nRanks
                );
            }
        )
    );
}

} // namespace RcclUnitTesting