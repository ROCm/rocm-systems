/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "argcheck.h"
#include "comm.h"
#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

// Helper struct for ArgCheck tests (NOT a fixture - used inside isolated tests)
struct ArgCheckTestEnvironment
{
    ncclComm_t       comm;
    struct ncclInfo* info;
    int*             sendDevicePtr = nullptr;
    int*             recvDevicePtr = nullptr;

    // Helper function to set up valid ncclInfo for boundary testing
    void SetupValidInfo()
    {
        // Set up valid info structure
        info->comm     = comm;
        info->root     = 0;                 // Valid root
        info->datatype = (ncclDataType_t)0; // Valid datatype
        info->op       = (ncclRedOp_t)0;    // Valid reduction operation
        info->coll     = ncclFuncBroadcast; // Valid collective operation
        info->sendbuff = nullptr;           // Will be set per test if needed
        info->recvbuff = nullptr;           // Will be set per test if needed
        info->count    = 10;                // Valid count
        info->opName   = "TestOp";          // Valid operation name
    }

    // Helper function for tests requiring device memory
    void SetupValidBufferWithDeviceMemory()
    {
        // Set the active device to match comm->cudaDev
        hipError_t errSetDevice = hipSetDevice(comm->cudaDev);
        ASSERT_EQ(errSetDevice, hipSuccess);

        // Allocate device memory
        hipError_t errSend = hipMalloc(&sendDevicePtr, sizeof(int));
        ASSERT_EQ(errSend, hipSuccess);
        hipError_t errRecv = hipMalloc(&recvDevicePtr, sizeof(int));
        ASSERT_EQ(errRecv, hipSuccess);

        // Set device pointers
        info->sendbuff = sendDevicePtr;
        info->recvbuff = recvDevicePtr;
    }

    // Helper to clean up device memory
    void CleanupDeviceMemory()
    {
        if(sendDevicePtr)
        {
            hipFree(sendDevicePtr);
            sendDevicePtr = nullptr;
        }
        if(recvDevicePtr)
        {
            hipFree(recvDevicePtr);
            recvDevicePtr = nullptr;
        }
    }

    void setup()
    {
        // Allocate and zero-initialize ncclComm as a pointer
        comm = (struct ncclComm*)calloc(1, sizeof(struct ncclComm));
        ASSERT_NE(comm, nullptr) << "Failed to allocate ncclComm";

        // Initialize the communicator with required fields
        comm->cudaDev       = 0;
        comm->nRanks        = 4;
        comm->checkPointers = true;
        comm->rank          = 0;

        comm->startMagic = NCCL_MAGIC;
        comm->endMagic   = NCCL_MAGIC;

        // Verify the magic values were set correctly
        ASSERT_EQ(comm->startMagic, NCCL_MAGIC) << "startMagic not set correctly";
        ASSERT_EQ(comm->endMagic, NCCL_MAGIC) << "endMagic not set correctly";

        // Allocate and zero-initialize ncclInfo as a pointer
        info = (ncclInfo*)calloc(1, sizeof(ncclInfo));
        ASSERT_NE(info, nullptr) << "Failed to allocate ncclInfo";

        SetupValidInfo();

        SetupValidBufferWithDeviceMemory();
    }

    void cleanup()
    {
        // Free the allocated memory
        CleanupDeviceMemory();
        if(info)
        {
            free(info);
            info = nullptr;
        }
        if(comm)
        {
            free(comm);
            comm = nullptr;
        }
    }
};

TEST(ArgCheckTest, CudaPtrCheck_ValidPointer)
{
    RUN_ISOLATED_TEST(
        "CudaPtrCheck_ValidPointer",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            int*       devicePtr = nullptr;
            hipError_t err       = hipMalloc(&devicePtr, sizeof(int));
            ASSERT_EQ(err, hipSuccess);

            ncclResult_t result = CudaPtrCheck(devicePtr, env.comm, "devicePtr", "TestOp");
            EXPECT_EQ(result, ncclSuccess);

            hipFree(devicePtr);
            env.cleanup();
            INFO("Test 'CudaPtrCheck_ValidPointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CudaPtrCheck_NullPointer)
{
    RUN_ISOLATED_TEST(
        "CudaPtrCheck_NullPointer",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            ncclResult_t result = CudaPtrCheck(nullptr, env.comm, "invalidPtr", "TestOp");
            EXPECT_EQ(result, ncclInvalidArgument);

            env.cleanup();
            INFO("Test 'CudaPtrCheck_NullPointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CudaPtrCheck_DifferentDevicePointer)
{
    RUN_ISOLATED_TEST(
        "CudaPtrCheck_DifferentDevicePointer",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            int* devicePtr = nullptr;
            hipSetDevice(1);
            hipError_t err = hipMalloc(&devicePtr, sizeof(int));
            ASSERT_EQ(err, hipSuccess);

            ncclResult_t result = CudaPtrCheck(devicePtr, env.comm, "devicePtr", "TestOp");
            EXPECT_EQ(result, ncclInvalidArgument);

            hipFree(devicePtr);
            hipSetDevice(env.comm->cudaDev);

            env.cleanup();
            INFO("Test 'CudaPtrCheck_DifferentDevicePointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CudaPtrCheck_HostMemoryPointer)
{
    RUN_ISOLATED_TEST(
        "CudaPtrCheck_HostMemoryPointer",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Test with host memory instead of device memory
            int* hostPtr = (int*)malloc(sizeof(int));
            ASSERT_NE(hostPtr, nullptr) << "Failed to allocate host memory";

            *hostPtr = 42; // Initialize the memory

            // This should fail because host memory is not device memory
            ncclResult_t result = CudaPtrCheck(hostPtr, env.comm, "hostPtr", "TestOp");

            // Host memory should be rejected by CudaPtrCheck
            EXPECT_EQ(result, ncclInvalidArgument)
                << "Host memory should be rejected by CudaPtrCheck";

            free(hostPtr);

            env.cleanup();
            INFO("Test 'CudaPtrCheck_HostMemoryPointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, PtrCheck_ValidPointer)
{
    RUN_ISOLATED_TEST(
        "PtrCheck_ValidPointer",
        []()
        {
            int          value  = 42;
            ncclResult_t result = PtrCheck(&value, "TestOp", "value");
            ASSERT_EQ(result, ncclSuccess);
            INFO("Test 'PtrCheck_ValidPointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, PtrCheck_NullPointer)
{
    RUN_ISOLATED_TEST(
        "PtrCheck_NullPointer",
        []()
        {
            ncclResult_t result = PtrCheck(nullptr, "TestOp", "value");
            ASSERT_EQ(result, ncclInvalidArgument);
            INFO("Test 'PtrCheck_NullPointer' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CommCheck_ValidComm)
{
    RUN_ISOLATED_TEST(
        "CommCheck_ValidComm",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.comm->startMagic = NCCL_MAGIC;
            env.comm->endMagic   = NCCL_MAGIC;

            // Verify magic values are still correct (should be set in setup())
            ASSERT_EQ(env.comm->startMagic, NCCL_MAGIC) << "startMagic was corrupted";
            ASSERT_EQ(env.comm->endMagic, NCCL_MAGIC) << "endMagic was corrupted";

            // Call CommCheck and verify the result
            ncclResult_t result = CommCheck(env.comm, "TestOp", "testComm");
            EXPECT_EQ(result, ncclSuccess) << "Failed for valid communicator";

            env.cleanup();
            INFO("Test 'CommCheck_ValidComm' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CommCheck_NullComm)
{
    RUN_ISOLATED_TEST(
        "CommCheck_NullComm",
        []()
        {
            ncclResult_t result = CommCheck(nullptr, "TestOp", "comm");
            ASSERT_EQ(result, ncclInvalidArgument);
            INFO("Test 'CommCheck_NullComm' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CommCheck_CorruptedStartMagic)
{
    RUN_ISOLATED_TEST(
        "CommCheck_CorruptedStartMagic",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Corrupt only startMagic, keep endMagic valid
            env.comm->startMagic = 1;          // Corrupt startMagic
            env.comm->endMagic   = NCCL_MAGIC; // Keep endMagic valid

            // Call CommCheck and verify the result
            ncclResult_t result = CommCheck(env.comm, "TestOp", "comm");
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for corrupted startMagic";

            env.cleanup();
            INFO("Test 'CommCheck_CorruptedStartMagic' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CommCheck_CorruptedEndMagic)
{
    RUN_ISOLATED_TEST(
        "CommCheck_CorruptedEndMagic",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Keep startMagic valid, corrupt only endMagic
            env.comm->startMagic = NCCL_MAGIC; // Keep startMagic valid
            env.comm->endMagic   = 1;          // Corrupt endMagic

            // Call CommCheck and verify the result
            ncclResult_t result = CommCheck(env.comm, "TestOp", "comm");
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for corrupted endMagic";

            env.cleanup();
            INFO("Test 'CommCheck_CorruptedEndMagic' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, CommCheck_CorruptedBothMagics)
{
    RUN_ISOLATED_TEST(
        "CommCheck_CorruptedBothMagics",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Corrupt both startMagic and endMagic
            env.comm->startMagic = 1; // Corrupt startMagic
            env.comm->endMagic   = 1; // Corrupt endMagic

            // Call CommCheck and verify the result
            ncclResult_t result = CommCheck(env.comm, "TestOp", "comm");
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for corrupted both magic values";

            env.cleanup();
            INFO("Test 'CommCheck_CorruptedBothMagics' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidRoot_NegativeValue)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidRoot_NegativeValue",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->root = -1; // Invalid root (< 0)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid root < 0";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidRoot_NegativeValue' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidRoot_ExceedsNRanks)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidRoot_ExceedsNRanks",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->root = env.comm->nRanks; // Invalid root (>= nRanks)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid root >= nRanks";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidRoot_ExceedsNRanks' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidDatatype_NegativeValue)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidDatatype_NegativeValue",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->datatype = (ncclDataType_t)-1; // Invalid datatype (< 0)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid datatype < 0";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidDatatype_NegativeValue' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidDatatype_ExceedsMaxValue)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidDatatype_ExceedsMaxValue",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->datatype = (ncclDataType_t)ncclNumTypes; // Invalid datatype (>= ncclNumTypes)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid datatype >= ncclNumTypes";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidDatatype_ExceedsMaxValue' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidReductionOperation_NegativeValue)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidReductionOperation_NegativeValue",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->op = (ncclRedOp_t)-1; // Invalid reduction operation (< 0)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid reduction operation < 0";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidReductionOperation_NegativeValue' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidReductionOperation_ExceedsMaxValue)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidReductionOperation_ExceedsMaxValue",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->op = (ncclRedOp_t)ncclNumOps; // Invalid reduction operation (>= ncclNumOps)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument)
                << "Failed for invalid reduction operation >= ncclNumOps";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidReductionOperation_ExceedsMaxValue' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidCommunicatorPointers)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidCommunicatorPointers",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->op = (ncclRedOp_t)0; // Valid reduction operation
            if(env.info->sendbuff)
            {
                hipFree((void*)env.info->sendbuff);
                env.info->sendbuff = nullptr; // Invalid send buffer
            }
            if(env.info->recvbuff)
            {
                hipFree((void*)env.info->recvbuff);
                env.info->recvbuff = nullptr; // Invalid receive buffer
            }

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid communicator pointers";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidCommunicatorPointers' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_InvalidReductionOperationOutOfRange)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_InvalidReductionOperationOutOfRange",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->op = (ncclRedOp_t)5; // Invalid reduction operation (out of range)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid reduction operation";

            env.cleanup();
            INFO("Test 'ArgsCheck_InvalidReductionOperationOutOfRange' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_UserDefinedReductionOperationInvalid)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_UserDefinedReductionOperationInvalid",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Test case: User-defined reduction operation with freeNext != -1
            env.info->op
                = (ncclRedOp_t)(ncclNumOps + 1); // Set op to a user-defined reduction operation

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclInvalidArgument)
                << "Failed for user-defined reduction operation with freeNext != -1";

            env.cleanup();
            INFO("Test 'ArgsCheck_UserDefinedReductionOperationInvalid' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_SendAndRecvFunction)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_SendAndRecvFunction",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.info->recvbuff
                = env.recvDevicePtr; // Use allocated device pointer for receive buffer

            // Test both ncclFuncSend and ncclFuncRecv
            for(auto coll : {ncclFuncSend, ncclFuncRecv})
            {
                env.info->coll = coll; // Set the collective operation

                // Call ArgsCheck and verify the result
                ncclResult_t result = ArgsCheck(env.info);
                ASSERT_EQ(result, ncclSuccess) << "Failed for coll = " << coll;
            }

            env.cleanup();
            INFO("Test 'ArgsCheck_SendAndRecvFunction' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_CollNotReduce)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_CollNotReduce",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Case: env.info->coll != ncclFuncReduce
            env.info->coll = ncclFuncBroadcast; // Set coll to ncclFuncBroadcast

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclSuccess) << "Failed for coll != ncclFuncReduce";

            env.cleanup();
            INFO("Test 'ArgsCheck_CollNotReduce' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_ReduceCollWithRootRank)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_ReduceCollWithRootRank",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            // Case: env.info->coll == ncclFuncReduce and env.info->env.comm->rank == env.info->root
            env.info->coll = ncclFuncReduce; // Set coll to ncclFuncReduce

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclSuccess) << "Failed for coll == ncclFuncReduce and rank == root";

            env.cleanup();
            INFO("Test 'ArgsCheck_ReduceCollWithRootRank' PASSED\n");
        }
    );
}

TEST(ArgCheckTest, ArgsCheck_ReduceCollWithNonRootRank)
{
    RUN_ISOLATED_TEST(
        "ArgsCheck_ReduceCollWithNonRootRank",
        []()
        {
            ArgCheckTestEnvironment env;
            env.setup();

            env.comm->rank = 1; // Set rank to 1 (non-root)

            ncclResult_t result = ArgsCheck(env.info);
            EXPECT_EQ(result, ncclSuccess) << "Failed for coll == ncclFuncReduce and rank != root";

            env.cleanup();
            INFO("Test 'ArgsCheck_ReduceCollWithNonRootRank' PASSED\n");
        }
    );
}
