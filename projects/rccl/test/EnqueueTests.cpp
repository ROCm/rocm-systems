/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <cstring>
#include <hip/hip_runtime.h>

#include "comm.h"
#include "info.h"
#include "enqueue.h"
#include "utils.h"

class EnqueueTests : public ::testing::Test {
protected:
    ncclComm* comm;
    ncclInfo* info;
    void* sendbuff;
    void* recvbuff;
    static uint32_t abortFlag0, abortFlag1;
    static int abortFlagRefCount;

    void SetUp() override {
        // Allocate GPU memory for buffers
        size_t bufferSize = 1024 * sizeof(float);
        hipError_t hipErr = hipMalloc(&sendbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate sendbuff";

        hipErr = hipMalloc(&recvbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate recvbuff";

        // Initialize communicator
        comm = new ncclComm();
        memset(comm, 0, sizeof(ncclComm));

        comm->startMagic = NCCL_MAGIC;  // 0x0280028002800280

        // Initialize critical fields
        comm->rank = 0;
        comm->nRanks = 2;
        comm->cudaDev = 0;
        comm->localRank = 0;

        // Initialize abort flags
        comm->abortFlag = &abortFlag0;
        comm->childAbortFlag = &abortFlag1;
        comm->abortFlagRefCount = &abortFlagRefCount;

        // Initialize memory stack
        ncclMemoryStackConstruct(&comm->memScoped);
        ncclMemoryStackConstruct(&comm->memPermanent);

        // Initialize intra-communication pointers
        comm->intraComm0 = nullptr;
        comm->intraNext = nullptr;

        // Initialize work FIFO structures
        comm->workFifoBytes = 1024;  // Power of 2
        comm->workFifoBuf = nullptr;
        comm->workFifoBufDev = nullptr;
        comm->workFifoConsumed = 0;
        comm->workFifoProducedLastRecorded = 0;
        comm->workFifoProduced = 0;

        // Initialize planner
        memset(&comm->planner, 0, sizeof(comm->planner));

        // Initialize config
        memset(&comm->config, 0, sizeof(comm->config));
        comm->config.blocking = 1;
        comm->checkPointers = 0;  // Disable pointer validation for easier testing

        // Initialize peer info arrays
        comm->peerInfo = new ncclPeerInfo[comm->nRanks];
        memset(comm->peerInfo, 0, comm->nRanks * sizeof(ncclPeerInfo));

        comm->localRankToRank = new int[comm->nRanks];
        for (int i = 0; i < comm->nRanks; i++) {
            comm->localRankToRank[i] = i;
        }

        comm->endMagic = NCCL_MAGIC;    // 0x0280028002800280

        // Initialize operation info with valid GPU buffers
        info = new ncclInfo();
        memset(info, 0, sizeof(ncclInfo));
        info->comm = comm;
        info->opName = "AllReduce";
        info->count = 1024;
        info->datatype = ncclFloat;
        info->op = ncclSum;
        info->root = 0;
        info->sendbuff = sendbuff;  // Use allocated GPU memory
        info->recvbuff = recvbuff;  // Use allocated GPU memory
        info->stream = nullptr;
    }

    void TearDown() override {
        if (sendbuff) {
            hipFree(sendbuff);
        }
        if (recvbuff) {
            hipFree(recvbuff);
        }
        if (comm) {
            ncclMemoryStackDestruct(&comm->memScoped);
            ncclMemoryStackDestruct(&comm->memPermanent);
            delete[] comm->peerInfo;
            delete[] comm->localRankToRank;
            delete comm;
        }
        if (info) {
            delete info;
        }
    }
};

// Static member definitions
uint32_t EnqueueTests::abortFlag0 = 0;
uint32_t EnqueueTests::abortFlag1 = 0;
int EnqueueTests::abortFlagRefCount = 0;

// Test ncclInitKernelsForDevice function
TEST_F(EnqueueTests, ncclInitKernelsForDevice_ValidInput) {
    size_t maxStackSize = 0;
    ncclResult_t result = ncclInitKernelsForDevice(906, 65536, &maxStackSize);

    EXPECT_TRUE(result == ncclSuccess);
    EXPECT_GT(maxStackSize, 0);
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_NullStackSize) {
    ncclResult_t result = ncclInitKernelsForDevice(906, 65536, nullptr);

    EXPECT_EQ(result, ncclSuccess);
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_InvalidArch) {
    size_t maxStackSize = 0;
    ncclResult_t result = ncclInitKernelsForDevice(-1, 65536, &maxStackSize);
    EXPECT_EQ(result, ncclSuccess);

}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_ExceedsSharedMemory) {
    size_t maxStackSize = 0;

    ncclResult_t result = ncclInitKernelsForDevice(906, 32832, &maxStackSize);
    EXPECT_TRUE(result == ncclSystemError);
}

// Test ncclEnqueueCheck function
TEST_F(EnqueueTests, ncclEnqueueCheck_ValidInput) {
    ncclResult_t result = ncclEnqueueCheck(info);
    EXPECT_TRUE(result == ncclSuccess);
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidComm) {
    info->comm = nullptr;
    ncclResult_t result = ncclEnqueueCheck(info);
    EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidBuffers) {
    // Test with null sendbuff
    comm->checkPointers = 1;
    info->sendbuff = nullptr;
    ncclResult_t result = ncclEnqueueCheck(info);
    EXPECT_EQ(result, ncclInvalidArgument);

    // Reset sendbuff and test with null recvbuff
    info->sendbuff = sendbuff;
    info->recvbuff = nullptr;
    result = ncclEnqueueCheck(info);
    EXPECT_EQ(result, ncclInvalidArgument);
}

// Test ncclFuncSendCount function
TEST_F(EnqueueTests, ncclFuncSendCount_AllReduce) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncSendCount(ncclFuncAllReduce, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncSendCount_Broadcast) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncSendCount(ncclFuncBroadcast, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncSendCount_Reduce) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncSendCount(ncclFuncReduce, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncSendCount_AllGather) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncSendCount(ncclFuncAllGather, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncSendCount_ReduceScatter) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncSendCount(ncclFuncReduceScatter, nRanks, count);
    EXPECT_EQ(result, count * nRanks);
}

TEST_F(EnqueueTests, ncclFuncSendCount_ZeroCount) {
    size_t result = ncclFuncSendCount(ncclFuncAllReduce, 4, 0);
    EXPECT_EQ(result, 0);
}

// Test ncclFuncRecvCount function
TEST_F(EnqueueTests, ncclFuncRecvCount_AllReduce) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncRecvCount_Broadcast) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncRecvCount(ncclFuncBroadcast, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncRecvCount_Reduce) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncRecvCount(ncclFuncReduce, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncRecvCount_AllGather) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncRecvCount(ncclFuncAllGather, nRanks, count);
    EXPECT_EQ(result, count * nRanks);
}

TEST_F(EnqueueTests, ncclFuncRecvCount_ReduceScatter) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncRecvCount(ncclFuncReduceScatter, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncRecvCount_ZeroCount) {
    size_t result = ncclFuncRecvCount(ncclFuncAllReduce, 4, 0);
    EXPECT_EQ(result, 0);
}

// Test ncclFuncMaxSendRecvCount function
TEST_F(EnqueueTests, ncclFuncMaxSendRecvCount_AllReduce) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count);
    EXPECT_EQ(result, count);
}

TEST_F(EnqueueTests, ncclFuncMaxSendRecvCount_AllGather) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count);
    // For AllGather, receive count (count * nRanks) is larger than send count (count)
    EXPECT_EQ(result, count * nRanks);
}

TEST_F(EnqueueTests, ncclFuncMaxSendRecvCount_ReduceScatter) {
    size_t count = 1000;
    int nRanks = 4;

    size_t result = ncclFuncMaxSendRecvCount(ncclFuncReduceScatter, nRanks, count);
    // For ReduceScatter, send count (count) is larger than receive count (count/nRanks)
    EXPECT_EQ(result, count * nRanks);
}

TEST_F(EnqueueTests, ncclFuncMaxSendRecvCount_ZeroCount) {
    size_t result = ncclFuncMaxSendRecvCount(ncclFuncAllReduce, 4, 0);
    EXPECT_EQ(result, 0);
}

// Edge case tests
TEST_F(EnqueueTests, ncclFuncCounts_SingleRank) {
    size_t count = 1000;
    int nRanks = 1;

    // Test with single rank
    EXPECT_EQ(ncclFuncSendCount(ncclFuncAllReduce, nRanks, count), count);
    EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllReduce, nRanks, count), count);
    EXPECT_EQ(ncclFuncMaxSendRecvCount(ncclFuncAllReduce, nRanks, count), count);
}

TEST_F(EnqueueTests, ncclFuncCounts_LargeRankCount) {
    size_t count = 1000;
    int nRanks = 1024;

    // Test with large number of ranks
    EXPECT_EQ(ncclFuncSendCount(ncclFuncAllGather, nRanks, count), count);
    EXPECT_EQ(ncclFuncRecvCount(ncclFuncAllGather, nRanks, count), count * nRanks);
    EXPECT_EQ(ncclFuncMaxSendRecvCount(ncclFuncAllGather, nRanks, count), count * nRanks);
}
