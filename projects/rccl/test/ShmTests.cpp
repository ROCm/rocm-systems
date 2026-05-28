/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "shm.h"

#include "proxy.h"
#include "transport.h"
#include "gtest/gtest.h"
#include <cstdlib> // For setenv

#include <cstring>

void initCeOperation();
int useMemcpySend;
int useMemcpyRecv;

ncclResult_t shmSendProxyConnect(struct ncclProxyConnection *connection,
                                 struct ncclProxyState *proxyState,
                                 void *reqBuff, int reqSize, void *respBuff,
                                 int respSize, int *done);

ncclResult_t shmRecvProxyConnect(struct ncclProxyConnection *connection,
                                 struct ncclProxyState *proxyState,
                                 void *reqBuff, int reqSize, void *respBuff,
                                 int respSize, int *done);

ncclResult_t shmSendProxyFree(struct ncclProxyConnection *connection,
                              struct ncclProxyState *proxyState);

ncclResult_t shmRecvProxyFree(struct ncclProxyConnection *connection,
                              struct ncclProxyState *proxyState);

ncclResult_t shmSendProxyProgress(struct ncclProxyState *proxyState,
                                  struct ncclProxyArgs *args);

ncclResult_t shmRecvProxyProgress(struct ncclProxyState *proxyState,
                                  struct ncclProxyArgs *args);

ncclResult_t shmSendProxySetup(struct ncclProxyConnection *connection,
                               struct ncclProxyState *proxyState, void *reqBuff,
                               int reqSize, void *respBuff, int respSize,
                               int *done);

ncclResult_t shmRecvProxySetup(struct ncclProxyConnection *connection,
                               struct ncclProxyState *proxyState, void *reqBuff,
                               int reqSize, void *respBuff, int respSize,
                               int *done);

namespace RcclUnitTesting {

class ShmTests : public ::testing::Test {
protected:
  ncclProxyConnection connection;
  ncclProxyState proxyState;
  ncclProxyArgs args;

  // Make sure to keep the following struct definitions in sync with the
  // actual implementation in shm.cc
  struct shmBuffInfo {
    void *hptr;
    void *dptr;
  };

  struct shmConnectInfo {
    int rank;
    ncclShmIpcDesc_t desc;
    struct shmBuffInfo buf;
  };

  struct shmSendResources {
    struct ncclRecvMem *remHostMem;
    struct ncclRecvMem *devRemHostMem;
    ncclShmIpcDesc_t remDesc;
    struct ncclSendMem *hostMem;
    struct ncclSendMem *devHostMem;
  };

  struct shmRecvResources {
    struct ncclSendMem *remHostMem;
    struct ncclSendMem *devRemHostMem;
    ncclShmIpcDesc_t remDesc;
    struct ncclRecvMem *hostMem;
    struct ncclRecvMem *devHostMem;
  };

  struct shmProxyInfo {
    struct ncclRecvMem *ceRecvMem;
    char *devFifo;
    char *shmFifo;
    struct ncclSendMem *sendMem;
    struct ncclRecvMem *recvMem;

    // used by progress only
    uint64_t step;
    hipStream_t stream;
    hipEvent_t events[NCCL_STEPS];

    // ipc desc
    ncclShmIpcDesc_t desc;
  };

  struct shmRequest {
    size_t size;
    bool legacy;
  };

  struct shmProxyInfo *proxyInfo;
  struct shmProxyInfo req;
  struct shmConnectInfo connectInfo;
  int done;

  void SetUp() override {
    // Initialize HIP runtime
    hipError_t hipResult = hipInit(0);
    ASSERT_EQ(hipResult, hipSuccess);

    // Get device count
    int deviceCount = 0;
    hipResult = hipGetDeviceCount(&deviceCount);
    ASSERT_EQ(hipResult, hipSuccess);
    ASSERT_GT(deviceCount, 0);

    // Select device 0
    hipResult = hipSetDevice(0);
    ASSERT_EQ(hipResult, hipSuccess);

    // Make sure device is ready
    hipDeviceSynchronize();

    // Initialize ncclProxyState
    memset(&proxyState, 0, sizeof(proxyState));
    proxyState.buffSizes[NCCL_PROTO_SIMPLE] =
        1024; // Example buffer size for NCCL_PROTO_SIMPLE
    proxyState.buffSizes[NCCL_PROTO_LL] =
        2048; // Example buffer size for NCCL_PROTO_LL
    proxyState.buffSizes[NCCL_PROTO_LL128] =
        4096; // Example buffer size for NCCL_PROTO_LL128

    // Initialize ncclProxyConnection
    memset(&connection, 0, sizeof(connection));

    // Initialize ncclProxyArgs
    memset(&args, 0, sizeof(args));

    // Allocate proxyInfo on the heap
    proxyInfo = new shmProxyInfo{
        .ceRecvMem = nullptr, // Will be allocated using hipMalloc
        .devFifo = nullptr,   // Will be allocated using hipMalloc
        .shmFifo = nullptr,   // Not used in this test
        .sendMem = nullptr,   // Will be allocated using hipMalloc
        .recvMem = nullptr,   // Will be allocated using hipMalloc
        .step = 0,            // Initialize step to 0
        .stream = nullptr,    // HIP stream (will be created)
        .events = {nullptr},  // HIP events (will be created)
        .desc = {}            // Initialize IPC descriptor to default
    };

    // Allocate memory for ceRecvMem
    hipResult = hipHostMalloc(&proxyInfo->ceRecvMem, sizeof(ncclRecvMem),
                              hipHostMallocDefault);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Initialize ceRecvMem fields
    // memset(proxyInfo->ceRecvMem, 0, sizeof(ncclRecvMem));
    proxyInfo->ceRecvMem->tail = 0; // Initialize tail

    // Allocate device memory for devFifo
    hipResult =
        hipMalloc(&proxyInfo->devFifo, proxyState.buffSizes[NCCL_PROTO_SIMPLE]);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Allocate device memory for devFifo
    hipResult = hipHostMalloc(&proxyInfo->shmFifo,
                              proxyState.buffSizes[NCCL_PROTO_SIMPLE],
                              hipHostMallocDefault);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Allocate device memory for sendMem
    hipResult = hipMalloc(&proxyInfo->sendMem, sizeof(ncclSendMem));
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Allocate device memory for recvMem
    hipResult = hipHostMalloc(&proxyInfo->recvMem, sizeof(ncclRecvMem),
                              hipHostMallocDefault);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Create a HIP stream
    hipResult =
        hipStreamCreateWithFlags(&proxyInfo->stream, hipStreamNonBlocking);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure stream creation was successful

    // Create HIP events
    for (int i = 0; i < NCCL_STEPS; i++) {
      hipResult = hipEventCreate(&proxyInfo->events[i]);
      ASSERT_EQ(hipResult, hipSuccess); // Ensure event creation was successful
    }

    // Initialize shared memory descriptor
    size_t shmSize = 2048;     // Example size for shared memory
    int useLegacy = 1;         // Set to 1 for legacy IPC, 0 for CUDA IPC
    int tpProxyRank = 0;       // Example proxy rank
    ncclShmIpcDesc_t shmDesc;  // Shared memory descriptor
    void *hostPtr = nullptr;   // Pointer to host memory
    void *devicePtr = nullptr; // Pointer to device memory

    // Allocate the shareable buffer
    ncclResult_t result = ncclShmAllocateShareableBuffer(
        shmSize, useLegacy, &shmDesc, &hostPtr, &devicePtr);
    ASSERT_EQ(result, ncclSuccess) << "Failed to allocate shareable buffer in SetUp()";

    proxyInfo->desc = shmDesc;

    // Initialize test data for connection
    connection = {
        .proxyAppend = {}, // Initialize proxyAppend to default
        .proxyAppendPtr =
            nullptr, // Pointer to proxyAppend (set to nullptr for now)
        .transportResources = proxyInfo // Use proxyInfo as transport resources
    };
  }

  void TearDown() override {
    // Cleanup
    if (proxyInfo) {
      if (proxyInfo->stream) {
        hipStreamDestroy(proxyInfo->stream);
      }

      for (int i = 0; i < NCCL_STEPS; i++) {
        if (proxyInfo->events[i]) {
          hipEventDestroy(proxyInfo->events[i]);
        }
      }

      if (proxyInfo->devFifo) {
        hipFree(proxyInfo->devFifo);
      }

      if (proxyInfo->ceRecvMem) {
        hipHostFree(proxyInfo->ceRecvMem);
      }

      if (proxyInfo->recvMem) {
        hipHostFree(proxyInfo->recvMem);
      }

      if (proxyInfo->shmFifo) {
        hipHostFree(proxyInfo->shmFifo);
      }
    }
  }
};

TEST_F(ShmTests, ShmSendProxyConnect) {
  // Initialize test data for reqInfo
  shmProxyInfo reqInfo = {
      .ceRecvMem = new ncclRecvMem(), // Allocate memory for ncclRecvMem
      .devFifo = new char[1024],      // Allocate memory for device FIFO
      .shmFifo = new char[1024],      // Allocate memory for shared memory FIFO
      .sendMem = new ncclSendMem(),   // Allocate memory for ncclSendMem
      .recvMem = new ncclRecvMem(),   // Allocate memory for ncclRecvMem
      .step = 0,                      // Initialize step to 0
      .stream = nullptr,   // HIP stream (can be initialized later if needed)
      .events = {nullptr}, // HIP events (initialize to nullptr for all steps)
      .desc = {}           // Initialize IPC descriptor to default
  };

  // Initialize test data for connection
  ncclProxyConnection connection = {
      .proxyAppend = {}, // Initialize proxyAppend to default
      .proxyAppendPtr =
          nullptr, // Pointer to proxyAppend (set to nullptr for now)
      .transportResources = &reqInfo // Use reqInfo as transport resources
  };

  // Initialize test data for proxyState
  ncclProxyState proxyState = {// Buffer sizes for different protocols
                               .buffSizes = {1024, 2048, 4096}};

  // Initialize test data for respInfo
  shmProxyInfo respInfo = {
      .ceRecvMem = new ncclRecvMem(), // Allocate memory for ncclRecvMem
      .devFifo = new char[1024],      // Allocate memory for device FIFO
      .shmFifo = new char[1024],      // Allocate memory for shared memory FIFO
      .sendMem = new ncclSendMem(),   // Allocate memory for ncclSendMem
      .recvMem = new ncclRecvMem(),   // Allocate memory for ncclRecvMem
      .step = 0,                      // Initialize step to 0
      .stream = nullptr,   // HIP stream (can be initialized later if needed)
      .events = {nullptr}, // HIP events (initialize to nullptr for all steps)
      .desc = {}           // Initialize IPC descriptor to default
  };

  int done = 0; // Initialize done flag to 0

  // Call the function
  ncclResult_t result =
      shmSendProxyConnect(&connection,      // Connection object
                          &proxyState,      // Proxy state object
                          &reqInfo,         // Request buffer
                          sizeof(reqInfo),  // Size of the request buffer
                          &respInfo,        // Response buffer
                          sizeof(respInfo), // Size of the response buffer
                          &done             // Done flag
      );

  // Validate the results
  EXPECT_EQ(result, ncclSuccess);       // Expect the function to return success
  EXPECT_EQ(done, 1);                   // Expect the done flag to be set to 1
  EXPECT_NE(respInfo.devFifo, nullptr); // Expect devFifo to be initialized
  EXPECT_NE(respInfo.ceRecvMem, nullptr); // Expect ceRecvMem to be initialized
}

TEST_F(ShmTests, ShmSendProxyConnect_InvalidSizes) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));

  ncclProxyState proxyState;
  memset(&proxyState, 0, sizeof(proxyState));

  // Create invalid request/response sizes to trigger the error path
  struct shmProxyInfo reqInfo;
  struct shmProxyInfo respInfo;

  // Set done flag
  int done = 0;

  // Call the function with invalid reqSize
  ncclResult_t result =
      shmSendProxyConnect(&connection, &proxyState, &reqInfo,
                          sizeof(reqInfo) - 1, // Invalid size to trigger error
                          &respInfo, sizeof(respInfo), &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(ShmTests, ShmRecvProxyConnect) {
  // Initialize test data for reqInfo
  shmProxyInfo reqInfo = {
      .ceRecvMem = new ncclRecvMem(), // Allocate memory for ncclRecvMem
      .devFifo = new char[1024],      // Allocate memory for device FIFO
      .shmFifo = new char[1024],      // Allocate memory for shared memory FIFO
      .sendMem = new ncclSendMem(),   // Allocate memory for ncclSendMem
      .recvMem = new ncclRecvMem(),   // Allocate memory for ncclRecvMem
      .step = 0,                      // Initialize step to 0
      .stream = nullptr,   // HIP stream (can be initialized later if needed)
      .events = {nullptr}, // HIP events (initialize to nullptr for all steps)
      .desc = {}           // Initialize IPC descriptor to default
  };

  // Initialize test data for connection
  ncclProxyConnection connection = {
      .proxyAppend = {}, // Initialize proxyAppend to default
      .proxyAppendPtr =
          nullptr, // Pointer to proxyAppend (set to nullptr for now)
      .transportResources = &reqInfo // Use reqInfo as transport resources
  };

  // Initialize test data for proxyState
  ncclProxyState proxyState = {
      .buffSizes = {1024, 2048, 4096}
      // Example buffer sizes for different protocols
  };

  // Initialize test data for respInfo
  shmProxyInfo respInfo = {
      .ceRecvMem =
          nullptr,        // Initialize to nullptr (will be set by the function)
      .devFifo = nullptr, // Initialize to nullptr (will be set by the function)
      .shmFifo = nullptr, // Initialize to nullptr (will be set by the function)
      .sendMem = nullptr, // Initialize to nullptr (will be set by the function)
      .recvMem = nullptr, // Initialize to nullptr (will be set by the function)
      .step = 0,          // Initialize step to 0
      .stream = nullptr,  // HIP stream (can be initialized later if needed)
      .events = {nullptr}, // HIP events (initialize to nullptr for all steps)
      .desc = {}           // Initialize IPC descriptor to default
  };

  int done = 0; // Initialize done flag to 0

  // Call the function
  ncclResult_t result =
      shmRecvProxyConnect(&connection,      // Connection object
                          &proxyState,      // Proxy state object
                          &reqInfo,         // Request buffer
                          sizeof(reqInfo),  // Size of the request buffer
                          &respInfo,        // Response buffer
                          sizeof(respInfo), // Size of the response buffer
                          &done             // Done flag
      );

  // Validate the results
  EXPECT_EQ(result, ncclSuccess);       // Expect the function to return success
  EXPECT_EQ(done, 1);                   // Expect the done flag to be set to 1
  EXPECT_NE(respInfo.devFifo, nullptr); // Expect devFifo to be initialized
  EXPECT_NE(respInfo.ceRecvMem, nullptr); // Expect ceRecvMem to be initialized
  EXPECT_NE(respInfo.recvMem, nullptr);   // Expect recvMem to be initialized
  EXPECT_NE(respInfo.shmFifo, nullptr);   // Expect shmFifo to be initialized
}

TEST_F(ShmTests, ShmRecvProxyConnect_InvalidSizes) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));

  ncclProxyState proxyState;
  memset(&proxyState, 0, sizeof(proxyState));

  // Create invalid request/response sizes to trigger the error path
  struct shmProxyInfo reqInfo;
  struct shmProxyInfo respInfo;

  // Set done flag
  int done = 0;

  // Call the function with invalid reqSize
  ncclResult_t result =
      shmRecvProxyConnect(&connection, &proxyState, &reqInfo,
                          sizeof(reqInfo) - 1, // Invalid size to trigger error
                          &respInfo, sizeof(respInfo), &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(ShmTests, ShmSendRecvProxyFree) {

  // Call initCeOperation to initialize useMemcpySend/useMemcpyRecv/shmLocality
  initCeOperation();

  // Note: Execute the following code twice, once for shmSendProxyFree and once
  // for shmRecvProxyFree via command line by setting environment variables
  // NCCL_SHM_USE_CUDA_MEMCPY=1 and NCCL_SHM_MEMCPY_MODE=1,
  // NCCL_SHM_USE_CUDA_MEMCPY=1 and NCCL_SHM_MEMCPY_MODE=2
  // The values are set as static, it requires to call initCeOperation to
  // update the values in separate processes.
  int res = 0;
  if (useMemcpySend) {
    res = shmSendProxyFree(&connection, &proxyState);
  }

  if (useMemcpyRecv) {
    res = shmRecvProxyFree(&connection, &proxyState);
  }

  // Note: Execute w/ NCCL_SHM_USE_CUDA_MEMCPY=0 and NCCL_SHM_LOCALITY=3,
  // initCeOperation() will be called in this test case
  // to generate an warning message

  // Validate the results
  EXPECT_EQ(res, ncclSuccess); // Expect the function to return success
}

TEST_F(ShmTests, ShmSendProxyProgress) {
  // Initialize arguments
  args.state = ncclProxyOpReady;
  args.protocol = NCCL_PROTO_SIMPLE;
  args.nsubs = 1;
  args.chunkSteps = 1;
  args.sliceSteps = 1;
  args.subs[0].nsteps = 4;

  // Set up connection
  args.subs[0].connection = &connection;

  // Test Ready State
  ncclResult_t result = shmSendProxyProgress(&proxyState, &args);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(args.state, ncclProxyOpProgress);
  EXPECT_EQ(args.subs[0].base, 0);
  EXPECT_EQ(args.subs[0].posted, 0);
  EXPECT_EQ(args.subs[0].transmitted, 0);
  EXPECT_EQ(args.subs[0].done, 0);

  // Test Progress State
  for (int i = 0; i < args.subs[0].nsteps; i++) {
    // Set up data for this step - for send proxy we need to set up
    // ceRecvMem->tail
    proxyInfo->ceRecvMem->tail = i + 1;
    proxyInfo->ceRecvMem->connFifo[i % NCCL_STEPS].size =
        proxyState.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

    // Process the step
    result = shmSendProxyProgress(&proxyState, &args);
    EXPECT_EQ(result, ncclSuccess);

    // Wait for the hipMemcpyAsync to complete
    hipStreamSynchronize(proxyInfo->stream);

    // Process again - should now increase done since event is complete
    result = shmSendProxyProgress(&proxyState, &args);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GE(args.subs[0].transmitted, i + 1);
    EXPECT_GE(args.subs[0].done, i + 1);
  }

  // Test None State
  EXPECT_EQ(args.state, ncclProxyOpNone);
  EXPECT_EQ(args.subs[0].transmitted, args.subs[0].nsteps);
  EXPECT_EQ(args.subs[0].done, args.subs[0].nsteps);
  EXPECT_EQ(args.done, args.nsubs);

  // Verify that recvMem->tail was updated (specific to send proxy)
  EXPECT_EQ(proxyInfo->recvMem->tail, args.subs[0].base + args.subs[0].done);
}

TEST_F(ShmTests, ShmRecvProxyProgress) {
  // Initialize arguments
  args.state = ncclProxyOpReady;
  args.protocol = NCCL_PROTO_SIMPLE;
  args.nsubs = 1;
  args.chunkSteps = 1;
  args.sliceSteps = 1;
  args.subs[0].nsteps = 4;

  // Set up connection
  args.subs[0].connection = &connection;

  // Test Ready State
  ncclResult_t result = shmRecvProxyProgress(&proxyState, &args);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(args.state, ncclProxyOpProgress);
  EXPECT_EQ(args.subs[0].base, 0);
  EXPECT_EQ(args.subs[0].posted, 0);
  EXPECT_EQ(args.subs[0].transmitted, 0);
  EXPECT_EQ(args.subs[0].done, 0);

  // Test Progress State
  for (int i = 0; i < args.subs[0].nsteps; i++) {
    // Set up data for this step
    proxyInfo->recvMem->tail = i + 1;
    proxyInfo->recvMem->connFifo[i % NCCL_STEPS].size =
        proxyState.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

    // Process the step
    result = shmRecvProxyProgress(&proxyState, &args);
    EXPECT_EQ(result, ncclSuccess);

    // Wait for the hipMemcpyAsync to complete
    hipStreamSynchronize(proxyInfo->stream);

    // Process again - should now increase done since event is complete
    result = shmRecvProxyProgress(&proxyState, &args);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GE(args.subs[0].transmitted, i + 1);
    EXPECT_GE(args.subs[0].done, i + 1);
  }

  // Test None State
  EXPECT_EQ(args.state, ncclProxyOpNone);
  EXPECT_EQ(args.subs[0].transmitted, args.subs[0].nsteps);
  EXPECT_EQ(args.subs[0].done, args.subs[0].nsteps);
  EXPECT_EQ(args.done, args.nsubs);
}

TEST_F(ShmTests, ShmSendProxyProgress_ProtoLL) {
  // Initialize test data for args
  args.state = ncclProxyOpReady; // Set the state to ready
  args.nsubs = 1;                // Number of sub-operations
  args.protocol = NCCL_PROTO_LL; // Use a supported protocol
  args.chunkSteps = 4;           // Set chunk steps
  args.sliceSteps = 2;           // Set slice steps
  args.subs[0].connection =
      &connection; // Link the connection to the sub-operation

  // Initialize proxyState
  proxyState.buffSizes[NCCL_PROTO_LL] =
      1024; // Set buffer size for the protocol

  // Call the function
  ncclResult_t result = shmSendProxyProgress(&proxyState, &args);

  // Validate the results
  EXPECT_EQ(result, ncclSuccess); // Expect the function to return success
  EXPECT_EQ(args.state,
            ncclProxyOpNone); // Expect the state to transition to none
}

TEST_F(ShmTests, ShmRecvProxyProgress_ProtoLL) {
  // Initialize test data for args
  args.state = ncclProxyOpReady; // Set the state to ready
  args.nsubs = 1;                // Number of sub-operations
  args.protocol = NCCL_PROTO_LL; // Use a supported protocol
  args.chunkSteps = 4;           // Set chunk steps
  args.sliceSteps = 2;           // Set slice steps
  args.subs[0].connection =
      &connection; // Link the connection to the sub-operation

  // Initialize proxyState
  proxyState.buffSizes[NCCL_PROTO_LL] =
      1024; // Set buffer size for the protocol

  // Call the function
  ncclResult_t result = shmRecvProxyProgress(&proxyState, &args);

  // Validate the results
  EXPECT_EQ(result, ncclSuccess); // Expect the function to return success
  EXPECT_EQ(args.state,
            ncclProxyOpNone); // Expect the state to transition to none
}

TEST_F(ShmTests, ShmAllocateShareableBuffer_ValidParameters) {
  // Setup variables
  int tpProxyRank = 0;
  size_t size = 1024;
  bool legacy = true;
  ncclShmIpcDesc_t desc;
  void *hptr = nullptr;
  void *dptr = nullptr;

  // Test: All valid parameters - should succeed
  ncclResult_t result = ncclShmAllocateShareableBuffer(size, legacy, &desc, &hptr, &dptr);
  EXPECT_EQ(result, hipSuccess);
  EXPECT_NE(hptr, nullptr);
  EXPECT_NE(dptr, nullptr);
}

TEST_F(ShmTests, ShmAllocateShareableBuffer_NullDesc) {
  // Setup variables
  size_t size = 1024;
  bool legacy = true;
  void *hptr = nullptr;
  void *dptr = nullptr;

  // Test: NULL desc - should fail
  ncclResult_t result = ncclShmAllocateShareableBuffer(size, legacy, nullptr, &hptr, &dptr);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmAllocateShareableBuffer_NullHptr) {
  // Setup variables
  size_t size = 1024;
  bool legacy = true;
  ncclShmIpcDesc_t desc;
  void *dptr = nullptr;

  // Test: NULL hptr - should fail
  ncclResult_t result = ncclShmAllocateShareableBuffer(size, legacy, &desc, nullptr, &dptr);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmAllocateShareableBuffer_NullDptr) {
  // Setup variables
  size_t size = 1024;
  bool legacy = true;
  ncclShmIpcDesc_t desc;
  void *hptr = nullptr;

  // Test: NULL dptr - should succeed (dptr is optional)
  ncclResult_t result = ncclShmAllocateShareableBuffer(size, legacy, &desc, &hptr, nullptr);
  EXPECT_EQ(result, hipSuccess);
  EXPECT_NE(hptr, nullptr);

  // Cleanup
  if (hptr) {
    hipHostFree(hptr);
  }
}

TEST_F(ShmTests, ShmAllocateShareableBuffer_MultipleNullParameters) {
  // Setup variables
  size_t size = 1024;
  bool legacy = true;

  // Test: Multiple NULL parameters - should fail
  ncclResult_t result = ncclShmAllocateShareableBuffer(size, legacy, nullptr, nullptr, nullptr);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmImportShareableBuffer_NullComm) {
  // Setup variables
  ncclShmIpcDesc_t desc;
  void *hptr = nullptr;
  void *dptr = nullptr;
  ncclShmIpcDesc_t descOut;
  int tpProxyRank = 0;

  // Test: NULL comm - should fail
  ncclResult_t result = ncclShmImportShareableBuffer(nullptr, tpProxyRank, &desc, &hptr, &dptr, &descOut);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmImportShareableBuffer_NullDesc) {
  // Setup variables
  ncclComm_t comm = (ncclComm_t)1; // Non-null dummy pointer
  void *hptr = nullptr;
  void *dptr = nullptr;
  ncclShmIpcDesc_t descOut;
  int tpProxyRank = 0;

  // Test: NULL desc - should fail
  ncclResult_t result = ncclShmImportShareableBuffer(comm, tpProxyRank, nullptr, &hptr, &dptr, &descOut);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmImportShareableBuffer_NullHptr) {
  // Setup variables
  ncclComm_t comm = (ncclComm_t)1; // Non-null dummy pointer
  ncclShmIpcDesc_t desc;
  void *dptr = nullptr;
  ncclShmIpcDesc_t descOut;
  int tpProxyRank = 0;

  // Test: NULL hptr - should fail
  ncclResult_t result = ncclShmImportShareableBuffer(comm, tpProxyRank, &desc, nullptr, &dptr, &descOut);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmImportShareableBuffer_NullDescOut) {
  // Setup variables
  ncclComm_t comm = (ncclComm_t)1; // Non-null dummy pointer
  ncclShmIpcDesc_t desc;
  void *hptr = nullptr;
  void *dptr = nullptr;
  int tpProxyRank = 0;

  // Test: NULL descOut - should fail
  ncclResult_t result = ncclShmImportShareableBuffer(comm, tpProxyRank, &desc, &hptr, &dptr, nullptr);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmImportShareableBuffer_MultipleNullParameters) {
  // Setup variables
  int tpProxyRank = 0;

  // Test: Multiple NULL parameters - should fail
  ncclResult_t result = ncclShmImportShareableBuffer(nullptr, tpProxyRank, nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(result, ncclInvalidArgument);
}

TEST_F(ShmTests, ShmSendProxySetupSuccess) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));

  // Set proxyState.tpRank
  proxyState.tpRank = 0;

  // Create request buffer
  struct shmRequest req;
  req.size = 1024;
  req.legacy = false;
  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmSendProxySetup(&connection, &proxyState, &req,
                                          reqSize, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection.transportResources, nullptr);
  EXPECT_NE(resp->buf.hptr, nullptr);
  EXPECT_NE(resp->buf.dptr, nullptr);

  // Validate descriptor was copied
  struct shmProxyInfo *proxyInfo =
      (struct shmProxyInfo *)connection.transportResources;
  memset(&proxyInfo->desc, 0, sizeof(ncclShmIpcDesc_t));
  memset(&resp->desc, 0, sizeof(ncclShmIpcDesc_t));
  EXPECT_EQ(memcmp(&proxyInfo->desc, &resp->desc, sizeof(ncclShmIpcDesc_t)), 0);

  // Cleanup
  if (proxyInfo) {
    ncclShmIpcClose(&proxyInfo->desc);
    free(proxyInfo);
    connection.transportResources = nullptr;
  }
}

TEST_F(ShmTests, ShmSendProxySetupInvalidReqSize) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));

  // Create request buffer
  struct shmRequest req;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function with invalid reqSize
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmSendProxySetup(&connection, &proxyState, &req,
                                          reqSize - 1, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
}

TEST_F(ShmTests, ShmSendProxySetupInvalidRespSize) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));

  // Create request buffer
  struct shmRequest req;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function with invalid respSize
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmSendProxySetup(&connection, &proxyState, &req,
                                          reqSize, resp, respSize - 1, &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
}

TEST_F(ShmTests, ShmSendProxySetupWithLegacy) {
  // Setup variables
  struct ncclProxyConnection *connection = (struct ncclProxyConnection *)calloc(
      1, sizeof(struct ncclProxyConnection));

  // Set proxyState.tpRank
  proxyState.tpRank = 0;

  // Create request buffer with legacy=true
  struct shmRequest req;
  req.size = 1024;
  req.legacy = true;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmSendProxySetup(connection, &proxyState, &req,
                                          reqSize, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection->transportResources, nullptr);
  EXPECT_NE(resp->buf.hptr, nullptr);
  EXPECT_NE(resp->buf.dptr, nullptr);

  // Cleanup
  struct shmProxyInfo *proxyInfo =
      (struct shmProxyInfo *)connection->transportResources;
  if (proxyInfo) {
    ncclShmIpcClose(&proxyInfo->desc);
    free(proxyInfo);
    connection->transportResources = nullptr;
  }
  if (connection) {
    free(connection);
    connection = nullptr;
  }
}

TEST_F(ShmTests, ShmRecvProxySetupSuccess) {
  // Setup variables
  struct ncclProxyConnection *connection = (struct ncclProxyConnection *)calloc(
      1, sizeof(struct ncclProxyConnection));

  proxyState.tpRank = 0;

  // Create request buffer
  struct shmRequest req;
  req.size = 1024;
  req.legacy = false;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmRecvProxySetup(connection, &proxyState, &req,
                                          reqSize, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection->transportResources, nullptr);
  EXPECT_NE(resp->buf.hptr, nullptr);
  EXPECT_NE(resp->buf.dptr, nullptr);

  // Validate descriptor was copied
  struct shmProxyInfo *proxyInfo =
      (struct shmProxyInfo *)connection->transportResources;
  memset(&proxyInfo->desc, 0, sizeof(ncclShmIpcDesc_t));
  memset(&resp->desc, 0, sizeof(ncclShmIpcDesc_t));
  EXPECT_EQ(memcmp(&proxyInfo->desc, &resp->desc, sizeof(ncclShmIpcDesc_t)), 0);

  // Cleanup
  if (proxyInfo) {
    ncclShmIpcClose(&proxyInfo->desc);
    free(proxyInfo);
    connection->transportResources = nullptr;
  }
  if (connection) {
    free(connection);
    connection = nullptr;
  }
}

TEST_F(ShmTests, ShmRecvProxySetupInvalidReqSize) {
  // Setup variables
  struct ncclProxyConnection *connection = (struct ncclProxyConnection *)calloc(
      1, sizeof(struct ncclProxyConnection));

  // Create request buffer
  struct shmRequest req;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function with invalid reqSize
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmRecvProxySetup(connection, &proxyState, &req,
                                          reqSize - 1, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection->transportResources, nullptr);

  if (connection) {
    free(connection);
    connection = nullptr;
  }
}

TEST_F(ShmTests, ShmRecvProxySetupInvalidRespSize) {
  // Setup variables
  struct ncclProxyConnection *connection = (struct ncclProxyConnection *)calloc(
      1, sizeof(struct ncclProxyConnection));

  // Create request buffer
  struct shmRequest req;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function with invalid respSize
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmRecvProxySetup(connection, &proxyState, &req,
                                          reqSize, resp, respSize - 1, &done);

  // Validate results
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection->transportResources, nullptr);

  if (connection) {
    free(connection);
    connection = nullptr;
  }
}

TEST_F(ShmTests, ShmRecvProxySetupWithLegacy) {
  // Setup variables
  struct ncclProxyConnection *connection = (struct ncclProxyConnection *)calloc(
      1, sizeof(struct ncclProxyConnection));

  // Set proxyState.tpRank
  proxyState.tpRank = 0;

  // Create request buffer with legacy=true
  struct shmRequest req;
  req.size = 1024;
  req.legacy = true;

  // Allocate the response buffer as a pointer
  struct shmConnectInfo *resp =
      (struct shmConnectInfo *)calloc(1, sizeof(struct shmConnectInfo));

  // Set done flag
  int done = 0;

  // Call the function
  int reqSize = sizeof(struct shmRequest);
  int respSize = sizeof(struct shmConnectInfo);
  ncclResult_t result = shmRecvProxySetup(connection, &proxyState, &req,
                                          reqSize, resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection->transportResources, nullptr);
  EXPECT_NE(resp->buf.hptr, nullptr);
  EXPECT_NE(resp->buf.dptr, nullptr);

  // Cleanup
  struct shmProxyInfo *proxyInfo =
      (struct shmProxyInfo *)connection->transportResources;
  if (proxyInfo) {
    ncclShmIpcClose(&proxyInfo->desc);
    free(proxyInfo);
    connection->transportResources = nullptr;
  }
  if (connection) {
    free(connection);
    connection = nullptr;
  }
}

} // namespace RcclUnitTesting
