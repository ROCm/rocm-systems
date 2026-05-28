/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>

#include "argcheck.h"
#include "comm.h"
#include <hip/hip_runtime.h>

class ArgCheckTest : public ::testing::Test {
protected:
  ncclComm_t comm;
  struct ncclInfo *info;
  int *sendDevicePtr = nullptr;
  int *recvDevicePtr = nullptr;

  // Helper function to set up valid ncclInfo for boundary testing
  void SetupValidInfo() {
    // Set up valid info structure
    info->comm = comm;
    info->root = 0;                     // Valid root
    info->datatype = (ncclDataType_t)0; // Valid datatype
    info->op = (ncclRedOp_t)0;          // Valid reduction operation
    info->coll = ncclFuncBroadcast;     // Valid collective operation
    info->sendbuff = nullptr;           // Will be set per test if needed
    info->recvbuff = nullptr;           // Will be set per test if needed
    info->count = 10;                   // Valid count
    info->opName = "TestOp";            // Valid operation name
  }

  // Helper function for tests requiring device memory
  void SetupValidBufferWithDeviceMemory() {
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
  void CleanupDeviceMemory() {
    if (sendDevicePtr) {
      hipFree(sendDevicePtr);
      sendDevicePtr = nullptr;
    }
    if (recvDevicePtr) {
      hipFree(recvDevicePtr);
      recvDevicePtr = nullptr;
    }
  }

  void SetUp() override {
    // Allocate and zero-initialize ncclComm as a pointer
    comm = (struct ncclComm *)calloc(1, sizeof(struct ncclComm));
    ASSERT_NE(comm, nullptr) << "Failed to allocate ncclComm";

    // Initialize the communicator with required fields
    comm->cudaDev = 0;
    comm->nRanks = 4;
    comm->checkPointers = true;
    comm->rank = 0;

    comm->startMagic = NCCL_MAGIC;
    comm->endMagic = NCCL_MAGIC;

    // Verify the magic values were set correctly
    ASSERT_EQ(comm->startMagic, NCCL_MAGIC) << "startMagic not set correctly";
    ASSERT_EQ(comm->endMagic, NCCL_MAGIC) << "endMagic not set correctly";

    // Allocate and zero-initialize ncclInfo as a pointer
    info = (ncclInfo *)calloc(1, sizeof(ncclInfo));
    ASSERT_NE(info, nullptr) << "Failed to allocate ncclInfo";

    SetupValidInfo();

    SetupValidBufferWithDeviceMemory();
  }

  void TearDown() override {
    // Free the allocated memory
    CleanupDeviceMemory();
    if (info) {
      free(info);
      info = nullptr;
    }
    if (comm) {
      free(comm);
      comm = nullptr;
    }
  }
};

TEST_F(ArgCheckTest, CudaPtrCheck_ValidPointer) {
  int *devicePtr = nullptr;
  hipError_t err = hipMalloc(&devicePtr, sizeof(int));
  ASSERT_EQ(err, hipSuccess);

  ncclResult_t result = CudaPtrCheck(devicePtr, comm, "devicePtr", "TestOp");
  EXPECT_EQ(result, ncclSuccess);

  hipFree(devicePtr);
}

TEST_F(ArgCheckTest, CudaPtrCheck_NullPointer) {
  ncclResult_t result = CudaPtrCheck(nullptr, comm, "invalidPtr", "TestOp");
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ArgCheckTest, CudaPtrCheck_DifferentDevicePointer) {
  int *devicePtr = nullptr;
  hipSetDevice(1);
  hipError_t err = hipMalloc(&devicePtr, sizeof(int));
  ASSERT_EQ(err, hipSuccess);

  ncclResult_t result = CudaPtrCheck(devicePtr, comm, "devicePtr", "TestOp");
  EXPECT_EQ(result, ncclInvalidArgument);

  hipFree(devicePtr);
  hipSetDevice(comm->cudaDev);
}

TEST_F(ArgCheckTest, CudaPtrCheck_HostMemoryPointer) {
  // Test with host memory instead of device memory
  int *hostPtr = (int *)malloc(sizeof(int));
  ASSERT_NE(hostPtr, nullptr) << "Failed to allocate host memory";

  *hostPtr = 42; // Initialize the memory

  // This should fail because host memory is not device memory
  ncclResult_t result = CudaPtrCheck(hostPtr, comm, "hostPtr", "TestOp");

  // Host memory should be rejected by CudaPtrCheck
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Host memory should be rejected by CudaPtrCheck";

  free(hostPtr);
}

TEST_F(ArgCheckTest, PtrCheck_ValidPointer) {
  int value = 42;
  ncclResult_t result = PtrCheck(&value, "TestOp", "value");
  ASSERT_EQ(result, ncclSuccess);
}

TEST_F(ArgCheckTest, PtrCheck_NullPointer) {
  ncclResult_t result = PtrCheck(nullptr, "TestOp", "value");
  ASSERT_EQ(result, ncclInvalidArgument);
}

TEST_F(ArgCheckTest, CommCheck_ValidComm) {
  comm->startMagic = NCCL_MAGIC;
  comm->endMagic = NCCL_MAGIC;

  // Verify magic values are still correct (should be set in SetUp())
  ASSERT_EQ(comm->startMagic, NCCL_MAGIC) << "startMagic was corrupted";
  ASSERT_EQ(comm->endMagic, NCCL_MAGIC) << "endMagic was corrupted";

  // Call CommCheck and verify the result
  ncclResult_t result = CommCheck(comm, "TestOp", "testComm");
  EXPECT_EQ(result, ncclSuccess) << "Failed for valid communicator";
}

TEST_F(ArgCheckTest, CommCheck_NullComm) {
  ncclResult_t result = CommCheck(nullptr, "TestOp", "comm");
  ASSERT_EQ(result, ncclInvalidArgument);
}

TEST_F(ArgCheckTest, CommCheck_CorruptedStartMagic) {
  // Corrupt only startMagic, keep endMagic valid
  comm->startMagic = 1;        // Corrupt startMagic
  comm->endMagic = NCCL_MAGIC; // Keep endMagic valid

  // Call CommCheck and verify the result
  ncclResult_t result = CommCheck(comm, "TestOp", "comm");
  EXPECT_EQ(result, ncclInvalidArgument) << "Failed for corrupted startMagic";
}

TEST_F(ArgCheckTest, CommCheck_CorruptedEndMagic) {
  // Keep startMagic valid, corrupt only endMagic
  comm->startMagic = NCCL_MAGIC; // Keep startMagic valid
  comm->endMagic = 1;            // Corrupt endMagic

  // Call CommCheck and verify the result
  ncclResult_t result = CommCheck(comm, "TestOp", "comm");
  EXPECT_EQ(result, ncclInvalidArgument) << "Failed for corrupted endMagic";
}

TEST_F(ArgCheckTest, CommCheck_CorruptedBothMagics) {
  // Corrupt both startMagic and endMagic
  comm->startMagic = 1; // Corrupt startMagic
  comm->endMagic = 1;   // Corrupt endMagic

  // Call CommCheck and verify the result
  ncclResult_t result = CommCheck(comm, "TestOp", "comm");
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for corrupted both magic values";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidRoot_NegativeValue) {
  info->root = -1; // Invalid root (< 0)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid root < 0";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidRoot_ExceedsNRanks) {
  info->root = comm->nRanks; // Invalid root (>= nRanks)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid root >= nRanks";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidDatatype_NegativeValue) {
  info->datatype = (ncclDataType_t)-1; // Invalid datatype (< 0)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument) << "Failed for invalid datatype < 0";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidDatatype_ExceedsMaxValue) {
  info->datatype =
      (ncclDataType_t)ncclNumTypes; // Invalid datatype (>= ncclNumTypes)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for invalid datatype >= ncclNumTypes";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidReductionOperation_NegativeValue) {
  info->op = (ncclRedOp_t)-1; // Invalid reduction operation (< 0)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for invalid reduction operation < 0";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidReductionOperation_ExceedsMaxValue) {
  info->op =
      (ncclRedOp_t)ncclNumOps; // Invalid reduction operation (>= ncclNumOps)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for invalid reduction operation >= ncclNumOps";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidCommunicatorPointers) {
  info->op = (ncclRedOp_t)0; // Valid reduction operation
  if (info->sendbuff) {
    hipFree((void *)info->sendbuff);
    info->sendbuff = nullptr; // Invalid send buffer
  }
  if (info->recvbuff) {
    hipFree((void *)info->recvbuff);
    info->recvbuff = nullptr; // Invalid receive buffer
  }

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for invalid communicator pointers";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidReductionOperationOutOfRange) {
  info->op = (ncclRedOp_t)5; // Invalid reduction operation (out of range)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for invalid reduction operation";
}

TEST_F(ArgCheckTest, ArgsCheck_UserDefinedReductionOperationInvalid) {
  // Test case: User-defined reduction operation with freeNext != -1
  info->op = (ncclRedOp_t)(ncclNumOps +
                           1); // Set op to a user-defined reduction operation

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclInvalidArgument)
      << "Failed for user-defined reduction operation with freeNext != -1";
}

TEST_F(ArgCheckTest, ArgsCheck_SendAndRecvFunction) {
  info->recvbuff =
      recvDevicePtr; // Use allocated device pointer for receive buffer

  // Test both ncclFuncSend and ncclFuncRecv
  for (auto coll : {ncclFuncSend, ncclFuncRecv}) {
    info->coll = coll; // Set the collective operation

    // Call ArgsCheck and verify the result
    ncclResult_t result = ArgsCheck(info);
    ASSERT_EQ(result, ncclSuccess) << "Failed for coll = " << coll;
  }
}

TEST_F(ArgCheckTest, ArgsCheck_CollNotReduce) {
  // Case: info->coll != ncclFuncReduce
  info->coll = ncclFuncBroadcast; // Set coll to ncclFuncBroadcast

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclSuccess) << "Failed for coll != ncclFuncReduce";
}

TEST_F(ArgCheckTest, ArgsCheck_ReduceCollWithRootRank) {
  // Case: info->coll == ncclFuncReduce and info->comm->rank == info->root
  info->coll = ncclFuncReduce; // Set coll to ncclFuncReduce

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclSuccess)
      << "Failed for coll == ncclFuncReduce and rank == root";
}

TEST_F(ArgCheckTest, ArgsCheck_ReduceCollWithNonRootRank) {
  comm->rank = 1; // Set rank to 1 (non-root)

  ncclResult_t result = ArgsCheck(info);
  EXPECT_EQ(result, ncclSuccess)
      << "Failed for coll == ncclFuncReduce and rank != root";
}
