/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <iostream>

#include "hip/hip_runtime.h"

#include "comm.h"
#include "graph/topo.h"
#include "rccl/rccl.h"
#include "register.h"
#include "shm.h"
#include "transport.h"

enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };

extern int useMemcpy;

struct ncclP2pBuff {
  void *directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct ncclP2pRequest {
  size_t size;
  int refcount;
};

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};

struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm *shm;
  struct p2pShm *devShm;
  ncclShmIpcDesc_t desc;

  // Intermediate step for sender
  struct ncclRecvMem *ceRecvMem;
  char *ceDevBuff;

  // Receiver buffer
  char *recvFifo;

  // Used by CE memcpy progress only
  uint64_t step;
  hipStream_t stream;
  hipEvent_t events[NCCL_STEPS];
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // Used by CE memcpy
  ncclShmIpcDesc_t desc;
};

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem *sendDevMem;
    struct ncclRecvMem *recvDevMem;
  };
  void *sendMemIpc;
  int sendMemSameProc;
  void *recvMemIpc;
  int recvMemSameProc;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm *shm;
  struct p2pShm *devShm;
  int shmSize;
  ncclShmHandle_t handle;
  ncclShmIpcDesc_t desc;
  uint32_t *next_hdp_reg; // Next GPU in ring (for p2p transport use only)
};

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm *comm;
  struct ncclReg *reg;
};

ncclResult_t p2pCanConnect(int *ret, struct ncclComm *comm,
                           struct ncclTopoGraph *graph,
                           struct ncclPeerInfo *info1,
                           struct ncclPeerInfo *info2);

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount,
                                            ncclIpcDesc *ipcDesc, void **ptr);

ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer,
                                          size_t size, ncclIpcDesc *ipcDesc,
                                          void **devMemPtr);

ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);

ncclResult_t p2pMap(struct ncclComm *comm, struct ncclProxyConnector *proxyConn,
                    struct ncclPeerInfo *myInfo, struct ncclPeerInfo *peerInfo,
                    struct ncclP2pBuff *p2pBuff, void **devMem, void **ipcPtr);

ncclResult_t p2pSendSetup(struct ncclComm *comm, struct ncclTopoGraph *graph,
                          struct ncclPeerInfo *myInfo,
                          struct ncclPeerInfo *peerInfo,
                          struct ncclConnect *connectInfo,
                          struct ncclConnector *send, int channelId,
                          int connIndex);

ncclResult_t p2pRecvSetup(struct ncclComm *comm, struct ncclTopoGraph *graph,
                          struct ncclPeerInfo *myInfo,
                          struct ncclPeerInfo *peerInfo,
                          struct ncclConnect *connectInfo,
                          struct ncclConnector *recv, int channelId,
                          int connIndex);

ncclResult_t p2pSendProxyConnect(struct ncclProxyConnection *connection,
                                 struct ncclProxyState *proxyState,
                                 void *reqBuff, int reqSize, void *respBuff,
                                 int respSize, int *done);

ncclResult_t p2pSendProxySetup(struct ncclProxyConnection *connection,
                               struct ncclProxyState *proxyState, void *reqBuff,
                               int reqSize, void *respBuff, int respSize,
                               int *done);

ncclResult_t p2pRecvProxySetup(struct ncclProxyConnection *connection,
                               struct ncclProxyState *proxyState, void *reqBuff,
                               int reqSize, void *respBuff, int respSize,
                               int *done);

ncclResult_t p2pSendProxyFree(struct ncclProxyConnection *connection,
                              struct ncclProxyState *proxyState);

ncclResult_t p2pRecvProxyFree(struct ncclProxyConnection *connection,
                              struct ncclProxyState *proxyState);

ncclResult_t p2pSendProxyProgress(struct ncclProxyState *proxyState,
                                  struct ncclProxyArgs *args);

ncclResult_t ipcRegisterBuffer(ncclComm *comm, const void *userbuff,
                               size_t buffSize, int *peerRanks, int nPeers,
                               ncclIpcRegType type, struct ncclReg *regRecord,
                               int *regBufFlag, uintptr_t *offsetOut,
                               uintptr_t **peerRmtAddrsOut, bool *isLegacyIpc);

ncclResult_t cleanupIpc(struct ncclComm *comm, struct ncclCommCallback *cb);

ncclResult_t ncclIpcGraphRegisterBuffer(
    ncclComm *comm, const void *userbuff, size_t buffSize, int *peerRanks,
    int nPeers, ncclIpcRegType type, int *regBufFlag, uintptr_t *offsetOut,
    uintptr_t **peerRmtAddrsOut, void *cleanupQueuePtr, int *nCleanupQueueElts);

ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void *buff,
                                   size_t size, void **handle);

ncclResult_t p2pProxyRegister(struct ncclProxyConnection *connection,
                              struct ncclProxyState *proxyState, void *reqBuff,
                              int reqSize, void *respBuff, int respSize,
                              int *done);

ncclResult_t p2pProxyDeregister(struct ncclProxyConnection *connection,
                                struct ncclProxyState *proxyState,
                                void *reqBuff, int reqSize, int *done);

class P2pTests : public ::testing::Test {
protected:
  int deviceCount;
  std::vector<hipDeviceProp_t> props;

  ncclProxyConnection connection;
  ncclProxyState proxyState;
  ncclProxyArgs args;
  p2pShmProxyInfo *proxyInfo;

  void SetUp() override {
    // Initialize HIP runtime
    hipError_t hipResult = hipInit(0);
    ASSERT_EQ(hipResult, hipSuccess);

    // Get device count
    int count = 0;
    hipResult = hipGetDeviceCount(&count);
    ASSERT_EQ(hipResult, hipSuccess);
    ASSERT_GT(count, 0);

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

    proxyInfo = (p2pShmProxyInfo *)calloc(1, sizeof(p2pShmProxyInfo));
    proxyInfo->shm = nullptr;
    proxyInfo->devShm = nullptr;
    memset(&proxyInfo->desc, 0, sizeof(proxyInfo->desc));
    proxyInfo->ceRecvMem = nullptr;
    proxyInfo->ceDevBuff = nullptr;
    proxyInfo->recvFifo = nullptr;
    proxyInfo->step = 0;
    proxyInfo->stream = nullptr;
    for (int i = 0; i < NCCL_STEPS; ++i)
      proxyInfo->events[i] = nullptr;

    // Allocate memory for ceRecvMem
    hipResult = hipHostMalloc(&proxyInfo->ceRecvMem, sizeof(ncclRecvMem),
                              hipHostMallocDefault);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Initialize ceRecvMem fields
    proxyInfo->ceRecvMem->tail = 0; // Initialize tail

    // Allocate device memory for recvFifo
    hipResult = hipMalloc(&proxyInfo->recvFifo,
                          proxyState.buffSizes[NCCL_PROTO_SIMPLE]);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Allocate device memory for ceDevBuff
    hipResult = hipMalloc(&proxyInfo->ceDevBuff,
                          proxyState.buffSizes[NCCL_PROTO_SIMPLE]);
    ASSERT_EQ(hipResult, hipSuccess); // Ensure allocation was successful

    // Allocate memory for shm and devShm structs
    proxyInfo->shm = (struct p2pShm *)calloc(1, sizeof(struct p2pShm));
    proxyInfo->devShm = (struct p2pShm *)calloc(1, sizeof(struct p2pShm));
    ASSERT_NE(proxyInfo->shm, nullptr);
    ASSERT_NE(proxyInfo->devShm, nullptr);

    hipStreamCreate(&proxyInfo->stream);
    for (int i = 0; i < NCCL_STEPS; ++i) {
      hipEventCreate(&proxyInfo->events[i]);
    }

    // Initialize shared memory descriptor
    size_t shmSize = 1024;     // Example size for shared memory
    int useLegacy = 1;         // Set to 1 for legacy IPC, 0 for CUDA IPC
    int tpProxyRank = 0;       // Example proxy rank
    void *hostPtr = nullptr;   // Pointer to host memory
    void *devicePtr = nullptr; // Pointer to device memory

    // Allocate the shareable buffer
    ncclResult_t result = ncclShmAllocateShareableBuffer(
        shmSize, useLegacy, &proxyInfo->desc, &hostPtr, &devicePtr);
    if (result != ncclSuccess) {
      fprintf(stderr, "Failed to allocate shareable buffer: %d\n", result);
      return;
    }

    connection.transportResources = proxyInfo;

    ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
    ASSERT_GE(deviceCount, 3) << "At least three GPU required";
    props.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
    }
  }

  void setupCommAndPeers(struct ncclComm *comm, struct ncclTopoSystem *system,
                         uint64_t hostHash, int shmDev, int rank, int cudaDev,
                         bool hasFineGrain) {
    memset(comm, 0, sizeof(struct ncclComm));
    memset(system, 0, sizeof(struct ncclTopoSystem));
    comm->topo = system;
    comm->nRanks = 3;
    comm->rank = rank;
    comm->magic = NCCL_MAGIC; // Replace with the actual macro or value
    comm->regCache.pageSize = 4096;
    comm->peerInfo =
        (struct ncclPeerInfo *)calloc(3, sizeof(struct ncclPeerInfo));
    ASSERT_NE(comm->peerInfo, nullptr);

    comm->peerInfo[rank].rank = rank;
    comm->peerInfo[rank].hostHash = hostHash;
    comm->peerInfo[rank].pidHash = getpid();
    comm->peerInfo[rank].cudaDev = cudaDev;
    comm->peerInfo[rank].shmDev = shmDev;
    ASSERT_LT(cudaDev, props.size());

    comm->peerInfo[rank].busId = props[cudaDev].pciBusID;
    comm->peerInfo[rank].hasFineGrain = hasFineGrain;

    system->nodes[GPU].count = 3;

    for (int i = 0; i < 3; i++) {
      system->nodes[GPU].nodes[i].type = GPU;
      system->nodes[GPU].nodes[i].id = props[i].pciBusID;
      system->nodes[GPU].nodes[i].gpu.dev = i;
      system->nodes[GPU].nodes[i].gpu.rank = i;
      snprintf(system->nodes[GPU].nodes[i].gpu.gcn,
               sizeof(system->nodes[GPU].nodes[i].gpu.gcn), "gfx900");
    }

    system->nodes[NET].count = 1;
    system->nodes[NET].nodes[0].type = NET;
    system->nodes[NET].nodes[0].id = 0x100; // Arbitrary NET id

    // Connect each GPU to NET
    for (int i = 0; i < 3; i++) {
      system->nodes[GPU].nodes[i].paths[NET] =
          (struct ncclTopoLinkList *)calloc(1, sizeof(struct ncclTopoLinkList));
      struct ncclTopoLink *link =
          (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
      link->type = PATH_NET;
      link->remNode = &system->nodes[NET].nodes[0];
      system->nodes[GPU].nodes[i].paths[NET][0].count = 1;
      system->nodes[GPU].nodes[i].paths[NET][0].list[0] = link;
      system->nodes[GPU].nodes[i].paths[NET][0].type = PATH_PXB;
      system->nodes[GPU].nodes[i].paths[NET][0].bw = 200.0;
    }
  }

  void setupPaths(struct ncclTopoSystem *system, int pathType = PATH_PXB,
                  float bw = 100.0) {
    int gpuCount = system->nodes[GPU].count;
    for (int i = 0; i < gpuCount; i++) {
      // Allocate paths array for each GPU node
      system->nodes[GPU].nodes[i].paths[GPU] =
          (struct ncclTopoLinkList *)calloc(gpuCount,
                                            sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < gpuCount; j++) {
        if (i == j)
          continue;
        // Initialize path
        struct ncclTopoLink *link =
            (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
        link->type = pathType;
        link->bw = bw;
        link->remNode = &system->nodes[GPU].nodes[j];

        struct ncclTopoLinkList *path =
            &system->nodes[GPU].nodes[i].paths[GPU][j];
        path->count = 1;
        path->list[0] = link;
        path->type = pathType;
        path->bw = bw;
      }
    }
  }

  void setupPathsWithIntermediateGpu(struct ncclTopoSystem *system, int gpuSrc,
                                     int gpuIntermediate, int gpuDst) {
    for (int i = 0; i < system->nodes[GPU].count; i++) {
      system->nodes[GPU].nodes[i].paths[GPU] =
          (struct ncclTopoLinkList *)calloc(system->nodes[GPU].count,
                                            sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < system->nodes[GPU].count; j++) {
        system->nodes[GPU].nodes[i].paths[GPU][j].count = 0;
        memset(system->nodes[GPU].nodes[i].paths[GPU][j].list, 0,
               sizeof(system->nodes[GPU].nodes[i].paths[GPU][j].list));
      }
    }

    // gpuSrc -> gpuIntermediate
    struct ncclTopoLink *link1 =
        (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
    link1->type = PATH_PXB;
    link1->remNode = &system->nodes[GPU].nodes[gpuIntermediate];

    // gpuIntermediate -> gpuDst
    struct ncclTopoLink *link2 =
        (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
    link2->type = PATH_PXB;
    link2->remNode = &system->nodes[GPU].nodes[gpuDst];

    // Set path from gpuSrc to gpuDst via gpuIntermediate
    struct ncclTopoLinkList *pathSrcDst =
        &system->nodes[GPU].nodes[gpuSrc].paths[GPU][gpuDst];
    pathSrcDst->count = 2;
    pathSrcDst->list[0] = link1;
    pathSrcDst->list[1] = link2;
    pathSrcDst->type = PATH_PXB;

    // Set direct path from gpuIntermediate to gpuDst
    struct ncclTopoLinkList *intermediatePath =
        &system->nodes[GPU].nodes[gpuIntermediate].paths[GPU][gpuDst];
    intermediatePath->count = 2;
    intermediatePath->list[0] = link2;
    intermediatePath->type = PATH_PXB;
  }

  void cleanupPaths(struct ncclTopoSystem *system) {
    for (int src = 0; src < system->nodes[GPU].count; src++) {
      if (system->nodes[GPU].nodes[src].paths[GPU]) {
        for (int dst = 0; dst < system->nodes[GPU].count; dst++) {
          if (src == dst)
            continue;
          struct ncclTopoLinkList *path =
              &system->nodes[GPU].nodes[src].paths[GPU][dst];
          for (int linkIdx = 0; linkIdx < path->count; linkIdx++) {
            if (path->list[linkIdx]) {
              // free(path->list[linkIdx]);
              path->list[linkIdx] = nullptr;
            }
          }
        }
        free(system->nodes[GPU].nodes[src].paths[GPU]);
        system->nodes[GPU].nodes[src].paths[GPU] = nullptr;
      }
    }
  }
};

TEST_F(P2pTests, P2pAllocateShareableBuffer_ValidParameters) {
  // Setup variables
  size_t size = 1024;
  ncclIpcDesc desc;
  void *hptr = nullptr;

  // Test: All valid parameters - should succeed
  ncclResult_t result = ncclP2pAllocateShareableBuffer(size, 0, &desc, &hptr);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(hptr, nullptr);

  // Cleanup
  if (hptr) {
    hipFree(hptr);
    hptr = nullptr;
  }
}

TEST_F(P2pTests, P2pAllocateShareableBuffer_NullDesc) {
  // Setup variables
  size_t size = 1024;
  void *hptr = nullptr;

  // Test: NULL desc - should fail
  ncclResult_t result = ncclP2pAllocateShareableBuffer(size, 0, nullptr, &hptr);
  EXPECT_EQ(result, (ncclResult_t)hipErrorInvalidValue);
}

TEST_F(P2pTests, P2pFreeShareableBuffer) {
  // Setup variables
  ncclIpcDesc desc;

  // Test 1: All valid parameters - should succeed
  ncclResult_t result = ncclP2pFreeShareableBuffer(&desc);
  EXPECT_EQ(result, (ncclResult_t)hipSuccess);
}

TEST_F(P2pTests, P2pMap_DeviceEnablePeerAccessFailure) {
  // Skip test if we don't have at least 2 devices
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  // Setup test structures
  struct ncclComm comm;
  struct ncclProxyConnector proxyConn;
  struct ncclPeerInfo myInfo, peerInfo;
  struct ncclP2pBuff p2pBuff;
  void *devMem = nullptr;
  void *ipcPtr = nullptr;
  ncclResult_t result;

  // Initialize with zeroes
  memset(&comm, 0, sizeof(comm));
  memset(&proxyConn, 0, sizeof(proxyConn));
  memset(&myInfo, 0, sizeof(myInfo));
  memset(&peerInfo, 0, sizeof(peerInfo));
  memset(&p2pBuff, 0, sizeof(p2pBuff));

  // Configure peer info to have same PID but different devices
  myInfo.hostHash = 0x12345678;
  myInfo.pidHash = 0x87654321;
  myInfo.cudaDev = 0; // First GPU

  peerInfo.hostHash = 0x12345678; // Same host
  peerInfo.pidHash = 0x87654321;  // Same PID

  // Create some memory for p2pBuff
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMalloc(&p2pBuff.directPtr, 1024), hipSuccess);
  p2pBuff.size = 1024;

  // Get properties of the available GPUs
  std::vector<hipDeviceProp_t> props(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
  }

  // Find a device that cannot peer with device 0
  int incompatibleDevice = -1;
  for (int i = 1; i < deviceCount; i++) {
    int canAccessPeer = 0;
    ASSERT_EQ(hipDeviceCanAccessPeer(&canAccessPeer, 0, i), hipSuccess);
    if (!canAccessPeer) {
      incompatibleDevice = i;
      break;
    }
  }

  // If we can't find an incompatible device, force the failure by using an
  // invalid device ID
  if (incompatibleDevice == -1) {
    peerInfo.cudaDev = 999; // Invalid device ID
  } else {
    peerInfo.cudaDev = incompatibleDevice;
  }

  // Set current device to 0 to ensure we're in the right context
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  // Call the function under test - should fail at the hipDeviceEnablePeerAccess
  // call
  result =
      p2pMap(&comm, &proxyConn, &myInfo, &peerInfo, &p2pBuff, &devMem, &ipcPtr);

  // Verify that the function returned an error
  EXPECT_EQ(result, ncclInternalError);

  // Clean up
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipFree(p2pBuff.directPtr), hipSuccess);
}

TEST_F(P2pTests, P2pCanConnectForkTest) {
  // Skip test if we don't have at least 2 devices
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  // Create a pipe to communicate between parent and child
  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1) << "Failed to create pipe";

  // Fork the process
  pid_t childPid = fork();
  ASSERT_NE(childPid, -1) << "Failed to fork process";

  // Get device properties for all GPUs before we fork
  std::vector<hipDeviceProp_t> props(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
  }

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    // Child process
    close(pipefd[1]); // Close write end

    // Setup structures for child process
    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.topo = &system;
    comm.nRanks = 2;
    comm.rank = 1; // Child is rank 1

    // Set up peer info
    comm.peerInfo =
        (struct ncclPeerInfo *)calloc(2, sizeof(struct ncclPeerInfo));

    // Setup my info (rank 1)
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash;
    comm.peerInfo[1].pidHash = getpid(); // Child's PID hash
    comm.peerInfo[1].cudaDev = 1;
    comm.peerInfo[1].shmDev = 0; // Same shmDev for both ranks
    comm.peerInfo[1].busId = props[1].pciBusID;
    comm.peerInfo[1].hasFineGrain = true; // Set hasFineGrain to true

    // Wait for parent process to send their data
    struct ncclPeerInfo peerInfo;
    read(pipefd[0], &peerInfo,
         sizeof(peerInfo)); // FIXED: was using pipefd[1] incorrectly

    // Set parent's peer info
    comm.peerInfo[0] = peerInfo;

    // Initialize paths between GPUs
    for (int src = 0; src < 2; src++) {
      // Allocate memory for paths
      system.nodes[GPU].nodes[src].paths[GPU] =
          (struct ncclTopoLinkList *)calloc(2, sizeof(struct ncclTopoLinkList));

      for (int dst = 0; dst < 2; dst++) {
        if (src == dst)
          continue;

        // Create link and path
        struct ncclTopoLink *link =
            (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
        link->type = PATH_PXB; // PCIe connection
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[dst];

        // Set path properties
        system.nodes[GPU].nodes[src].paths[GPU][dst].count = 1;
        system.nodes[GPU].nodes[src].paths[GPU][dst].list[0] = link;
        system.nodes[GPU].nodes[src].paths[GPU][dst].type = PATH_PXB;
        system.nodes[GPU].nodes[src].paths[GPU][dst].bw = 100.0;
      }
    }

    // Test p2pCanConnect between the two processes
    int ret = 0;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1],
                                        &comm.peerInfo[0]);

    // Write result back to parent
    write(pipefd[0], &result,
          sizeof(result)); // FIXED: was using pipefd[1] incorrectly
    write(pipefd[0], &ret,
          sizeof(ret)); // FIXED: was using pipefd[1] incorrectly
    close(pipefd[0]);

    // Clean up
    for (int src = 0; src < 2; src++) {
      for (int dst = 0; dst < 2; dst++) {
        if (src == dst)
          continue;
        free(system.nodes[GPU].nodes[src].paths[GPU][dst].list[0]);
      }
      free(system.nodes[GPU].nodes[src].paths[GPU]);
    }
    free(comm.peerInfo);

    exit(0);
  } else {
    // Parent process
    close(
        pipefd[0]); // Close read end - FIXED: was incorrectly closing write end

    // Setup structures for parent process
    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.topo = &system;
    comm.nRanks = 2;
    comm.rank = 0; // Parent is rank 0

    // Set up peer info
    comm.peerInfo =
        (struct ncclPeerInfo *)calloc(2, sizeof(struct ncclPeerInfo));

    // Setup my info (rank 0)
    comm.peerInfo[0].rank = 0;
    comm.peerInfo[0].hostHash = hostHash;
    comm.peerInfo[0].pidHash = getpid(); // Parent's PID hash
    comm.peerInfo[0].cudaDev = 0;
    comm.peerInfo[0].shmDev = 0; // Same shmDev for both ranks
    comm.peerInfo[0].busId = props[0].pciBusID;
    comm.peerInfo[0].hasFineGrain = true; // Set hasFineGrain to true

    // Send my info to child
    write(pipefd[1], &comm.peerInfo[0], sizeof(struct ncclPeerInfo));

    // Set up system nodes (minimal needed for topology traversal)
    system.nodes[GPU].count = 2;

    // Initialize GPU nodes with real device data
    system.nodes[GPU].nodes[0].type = GPU;
    system.nodes[GPU].nodes[0].id = props[0].pciBusID;
    system.nodes[GPU].nodes[0].gpu.dev = 0;
    system.nodes[GPU].nodes[0].gpu.rank = 0;
    snprintf(system.nodes[GPU].nodes[0].gpu.gcn,
             sizeof(system.nodes[GPU].nodes[0].gpu.gcn), "gfx900");

    system.nodes[GPU].nodes[1].type = GPU;
    system.nodes[GPU].nodes[1].id = props[1].pciBusID;
    system.nodes[GPU].nodes[1].gpu.dev = 1;
    system.nodes[GPU].nodes[1].gpu.rank = 1;
    snprintf(system.nodes[GPU].nodes[1].gpu.gcn,
             sizeof(system.nodes[GPU].nodes[1].gpu.gcn), "gfx900");

    // Initialize paths between GPUs
    for (int src = 0; src < 2; src++) {
      // Allocate memory for paths
      system.nodes[GPU].nodes[src].paths[GPU] =
          (struct ncclTopoLinkList *)calloc(2, sizeof(struct ncclTopoLinkList));

      for (int dst = 0; dst < 2; dst++) {
        if (src == dst)
          continue;

        // Create link and path
        struct ncclTopoLink *link =
            (struct ncclTopoLink *)calloc(1, sizeof(struct ncclTopoLink));
        link->type = PATH_PXB; // PCIe connection
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[dst];

        // Set path properties
        system.nodes[GPU].nodes[src].paths[GPU][dst].count = 1;
        system.nodes[GPU].nodes[src].paths[GPU][dst].list[0] = link;
        system.nodes[GPU].nodes[src].paths[GPU][dst].type = PATH_PXB;
        system.nodes[GPU].nodes[src].paths[GPU][dst].bw = 100.0;
      }
    }

    // Child needs to send rank 1 info to parent process
    struct ncclPeerInfo childInfo;
    childInfo.rank = 1;
    childInfo.hostHash = hostHash;
    childInfo.busId = props[1].pciBusID;
    childInfo.cudaDev = 1;
    childInfo.hasFineGrain = true; // Set hasFineGrain to true for rank 1
    comm.peerInfo[1] = childInfo;

    // Wait for child results
    ncclResult_t childResult;
    int childRet;
    read(pipefd[1], &childResult,
         sizeof(childResult)); // FIXED: was using pipefd[0] incorrectly
    read(pipefd[1], &childRet,
         sizeof(childRet)); // FIXED: was using pipefd[0] incorrectly
    close(pipefd[1]);

    // Now test our p2pCanConnect as well
    int ret = 0;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[0],
                                        &comm.peerInfo[1]);

    // Check results
    EXPECT_EQ(result, ncclSuccess)
        << "Parent process p2pCanConnect failed with " << result;
    EXPECT_EQ(childResult, ncclSuccess)
        << "Child process p2pCanConnect failed with " << childResult;

    // Clean up
    for (int src = 0; src < 2; src++) {
      for (int dst = 0; dst < 2; dst++) {
        if (src == dst)
          continue;
        free(system.nodes[GPU].nodes[src].paths[GPU][dst].list[0]);
      }
      free(system.nodes[GPU].nodes[src].paths[GPU]);
    }
    free(comm.peerInfo);

    // Wait for child to complete
    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTests, P2pCanConnectShmDevDiffForkTest) {
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 1) {
    GTEST_SKIP() << "Test requires at least 3 GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2,
                      true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 2 ->
    // GPU 1 -> GPU 0)
    setupPathsWithIntermediateGpu(&system, 2, 1, 0);

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1],
                                        &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 0, 0,
                      true); // parent rank 0, shmDev=0, cudaDev=0

    // Setup child's peer info explicitly
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash; // Different host hash
    comm.peerInfo[1].pidHash = childPid;
    comm.peerInfo[1].cudaDev = 2;
    // comm.peerInfo[1].shmDev = 1; // Different shmDev
    comm.peerInfo[1].busId = props[2].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 0 ->
    // GPU 1 -> GPU 2)
    setupPathsWithIntermediateGpu(&system, 0, 1, 2);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));
    read(pipefd[1], &intermediateRank, sizeof(intermediateRank));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0) << "P2P should be disabled due to different host hashes";
    EXPECT_NE(intermediateRank, -1) << "Intermediate GPU should be set";

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTests, P2pCanConnectHostHashDiffForkTest) {
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 1) {
    GTEST_SKIP() << "Test requires at least 1 GPU";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Use different host hashes for all
  uint64_t parentHostHash = 0x1234567890ABCDEF;
  uint64_t childHostHash = 0xFEDCBA0987654321;
  uint64_t info2HostHash = 0x1111111111111111;

  if (childPid == 0) {
    // Child process
    close(pipefd[0]); // Close read end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    // Child rank 1, with unique hostHash
    setupCommAndPeers(&comm, &system, childHostHash, 0, 1, 0, true);

    // Receive parent's peer info (rank 0)
    read(pipefd[1], comm.peerInfo, sizeof(struct ncclPeerInfo));
    // Set hasFineGrain to true for parent info (rank 0)
    comm.peerInfo[0].hasFineGrain = true;

    // Prepare info2 with a third, different hostHash
    struct ncclPeerInfo info2 = comm.peerInfo[0];
    info2.hostHash = info2HostHash;

    int ret = -1;
    ncclResult_t result =
        p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &info2);

    // Send result back to parent
    write(pipefd[1], &result, sizeof(result));
    write(pipefd[1], &ret, sizeof(ret));

    free(comm.peerInfo);
    close(pipefd[1]);
    exit(0);

  } else {
    // Parent process
    close(pipefd[1]); // Close write end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    // Parent rank 0, with unique hostHash
    setupCommAndPeers(&comm, &system, parentHostHash, 0, 0, 0, true);
    // Set hasFineGrain to true for child info (rank 1)
    comm.peerInfo[1].hasFineGrain = true;

    // Send parent info to child (only rank 0 info)
    write(pipefd[0], &comm.peerInfo[0], sizeof(struct ncclPeerInfo));

    // Receive result from child
    ncclResult_t result;
    int ret;
    read(pipefd[0], &result, sizeof(result));
    read(pipefd[0], &ret, sizeof(ret));

    EXPECT_EQ(result, ncclSuccess)
        << "p2pCanConnect should return success with different hostHashes";
    // ret is not set in this branch, so its value is undefined; do not check
    // ret

    free(comm.peerInfo);
    close(pipefd[0]);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTests, P2pCanConnectWithIntermediateRankForkTest) {
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 1) {
    GTEST_SKIP() << "Test requires at least 3 GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2,
                      true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 2 ->
    // GPU 1 -> GPU 0)
    setupPathsWithIntermediateGpu(&system, 2, 1, 0);

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1],
                                        &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 0, 0, 0,
                      true); // parent rank 0, shmDev=0, cudaDev=0

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 0 ->
    // GPU 1 -> GPU 2)
    setupPathsWithIntermediateGpu(&system, 0, 1, 2);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));
    read(pipefd[1], &intermediateRank, sizeof(intermediateRank));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0) << "P2P should be disabled due to different host hashes";
    EXPECT_NE(intermediateRank, -1) << "Intermediate GPU should be set";

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTests, IpcRegisterBufferFailures) {
  const size_t TEST_BUFFER_ELEMENTS = 32;
  struct ncclComm comm;
  struct ncclTopoSystem system;
  setupCommAndPeers(&comm, &system, gethostid(), 0, 0, 0, true);
  comm.nRanks = 2;
  comm.rankToLocalRank = (int *)calloc(comm.nRanks, sizeof(int));
  for (int i = 0; i < comm.nRanks; ++i)
    comm.rankToLocalRank[i] = i;

  void *dptr = nullptr;
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  hipError_t err = hipMalloc(&dptr, TEST_BUFFER_ELEMENTS * sizeof(float));
  ASSERT_EQ(err, hipSuccess);

  struct ncclReg regRecord;
  memset(&regRecord, 0, sizeof(regRecord));
  regRecord.begAddr = (uintptr_t)dptr;
  regRecord.endAddr = regRecord.begAddr + (TEST_BUFFER_ELEMENTS * sizeof(float)); // Calculate end address based on size
  for (int i = 0; i < 2; ++i)
    regRecord.ipcInfos[i] = nullptr;

  int peerRanks[1] = {1};
  int regBufFlag = 0;
  uintptr_t offsetOut = 0;
  uintptr_t *peerRmtAddrsOut = nullptr;
  bool isLegacyIpc = false;
  ncclIpcRegType type = NCCL_IPC_COLLECTIVE;

  // Test 1: HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE is not supported
  ncclResult_t result = ipcRegisterBuffer(
      &comm, dptr, TEST_BUFFER_ELEMENTS * sizeof(float), peerRanks, 1, type, &regRecord,
      &regBufFlag, &offsetOut, &peerRmtAddrsOut, &isLegacyIpc);
  EXPECT_EQ(result, ncclUnhandledCudaError);

  hipFree(dptr);
  free(comm.peerInfo);
  free(comm.rankToLocalRank);
}

TEST_F(P2pTests, P2pSendProxyConnectInvalidSize) {
  // Setup valid test data for reuse
  // struct ncclProxyConnection validConnection = {};
  // struct ncclProxyState validProxyState = {};
  char reqBuffer[256] = {0};
  // char respBuffer[256] = {0};
  int done = 0;

  // Test Case 1: Invalid size (negative)
  {
    ncclResult_t result = p2pSendProxyConnect(&connection, &proxyState, nullptr,
                                              256, nullptr, 256, &done);
    EXPECT_NE(result, ncclSuccess);
    EXPECT_EQ(result, ncclInternalError);
  }
}

TEST_F(P2pTests, P2pSendProxyFree) {
  // Dummy call to p2pCanConnect to trigger static initialization (e.g.,
  // initCeOperation)
  int dummyRet = 0;
  struct ncclComm dummyComm = {};
  struct ncclTopoGraph dummyGraph = {};
  struct ncclPeerInfo dummyInfo1 = {};
  struct ncclPeerInfo dummyInfo2 = {};
  p2pCanConnect(&dummyRet, &dummyComm, &dummyGraph, &dummyInfo1, &dummyInfo2);

  int result = 0;
  if (useMemcpy) {
    result = p2pSendProxyFree(&connection, &proxyState);
    EXPECT_EQ(result, ncclSuccess); // Expect the function to return success
  }
}

TEST_F(P2pTests, P2pRecvProxyFree) {
  int result = p2pRecvProxyFree(&connection, &proxyState);
  EXPECT_EQ(result, ncclSuccess); // Expect the function to return success
}

TEST_F(P2pTests, P2pSendProxySetup) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  proxyState.tpRank = 0;

  // Allocate and initialize reqBuff
  struct ncclP2pRequest *req =
      (struct ncclP2pRequest *)calloc(1, sizeof(struct ncclP2pRequest));

  // Allocate respBuff
  struct ncclP2pBuff *resp =
      (struct ncclP2pBuff *)calloc(1, sizeof(struct ncclP2pBuff));

  int done = 0;

  int reqSize = sizeof(struct ncclP2pRequest);
  int respSize = sizeof(struct ncclP2pBuff);
  ncclResult_t result = p2pSendProxySetup(&connection, &proxyState, &req,
                                          reqSize, &resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection.transportResources, nullptr);
  EXPECT_EQ(1, done);

  // Cleanup
  ncclResult_t freeResult = p2pRecvProxyFree(&connection, &proxyState);
  EXPECT_EQ(freeResult, ncclSuccess);
  connection.transportResources = nullptr;
}

TEST_F(P2pTests, P2pSendProxySetupInvalidReqSize) {
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  struct ncclP2pRequest req;
  int invalidReqSize = sizeof(ncclP2pRequest) - 1;
  memset(&req, 0, invalidReqSize);
  struct ncclP2pBuff resp;
  memset(&resp, 0, sizeof(struct ncclP2pBuff));
  int done = 0;
  ncclResult_t result =
      p2pSendProxySetup(&connection, &proxyState, &req, invalidReqSize, &resp,
                        sizeof(resp), &done);
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(P2pTests, P2pSendProxySetupInvalidRespSize) {
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  struct ncclP2pRequest req;
  memset(&req, 0, sizeof(struct ncclP2pRequest));
  struct ncclP2pBuff resp;
  memset(&resp, 0, sizeof(struct ncclP2pBuff) - 1);
  int done = 0;

  ncclResult_t result =
      p2pSendProxySetup(&connection, &proxyState, &req, sizeof(req), &resp,
                        sizeof(resp) - 1, &done);
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(P2pTests, P2pSendProxySetupMemCpy) {
  // Dummy call to p2pCanConnect to trigger static initialization (e.g.,
  // initCeOperation)
  int dummyRet = 0;
  struct ncclComm dummyComm = {};
  struct ncclTopoGraph dummyGraph = {};
  struct ncclPeerInfo dummyInfo1 = {};
  struct ncclPeerInfo dummyInfo2 = {};
  p2pCanConnect(&dummyRet, &dummyComm, &dummyGraph, &dummyInfo1, &dummyInfo2);

  int result = 0;
  if (useMemcpy) {
    // Setup variables
    struct ncclProxyConnection connection;
    memset(&connection, 0, sizeof(connection));
    proxyState.tpRank = 0;

    // Create request buffer
    struct p2pShmProxyInfo reqInfo;
    memset(&reqInfo, 0, sizeof(reqInfo));

    // Create response buffer
    struct p2pShmProxyInfo respInfo;
    memset(&respInfo, 0, sizeof(respInfo));

    // Set done flag
    int done = 0;

    // Call the function
    ncclResult_t result =
        p2pSendProxySetup(&connection, &proxyState, &reqInfo, sizeof(reqInfo),
                          &respInfo, sizeof(respInfo), &done);

    // Validate results
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(connection.transportResources, nullptr);

    // Cleanup
    struct p2pShmProxyInfo *proxyInfo =
        (struct p2pShmProxyInfo *)connection.transportResources;
    if (proxyInfo) {
      // Free any dynamically allocated fields if needed
      free(proxyInfo);
      connection.transportResources = nullptr;
    }
  }
}

TEST_F(P2pTests, P2pSendProxySetupMemCpyInvalidRespSize) {
  // Dummy call to p2pCanConnect to trigger static initialization (e.g.,
  // initCeOperation)
  int dummyRet = 0;
  struct ncclComm dummyComm = {};
  struct ncclTopoGraph dummyGraph = {};
  struct ncclPeerInfo dummyInfo1 = {};
  struct ncclPeerInfo dummyInfo2 = {};
  p2pCanConnect(&dummyRet, &dummyComm, &dummyGraph, &dummyInfo1, &dummyInfo2);

  int result = 0;
  if (useMemcpy) {
    struct ncclProxyConnection connection;
    memset(&connection, 0, sizeof(connection));
    struct p2pShmProxyInfo reqInfo;
    memset(&reqInfo, 0, sizeof(reqInfo));
    struct p2pShmProxyInfo respInfo;
    memset(&respInfo, 0, sizeof(respInfo));
    int done = 0;
    ncclResult_t result =
        p2pSendProxySetup(&connection, &proxyState, &reqInfo, sizeof(reqInfo),
                          &respInfo, sizeof(respInfo) - 1, &done);
    EXPECT_EQ(result, ncclInternalError);
    EXPECT_EQ(connection.transportResources, nullptr);
  }
}

TEST_F(P2pTests, P2pRecvProxySetup) {
  // Setup variables
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  proxyState.tpRank = 0;

  // Allocate and initialize reqBuff
  struct ncclP2pRequest *req =
      (struct ncclP2pRequest *)calloc(1, sizeof(struct ncclP2pRequest));

  // Allocate respBuff
  struct ncclP2pBuff *resp =
      (struct ncclP2pBuff *)calloc(1, sizeof(struct ncclP2pBuff));

  int done = 0;

  int reqSize = sizeof(struct ncclP2pRequest);
  int respSize = sizeof(struct ncclP2pBuff);
  ncclResult_t result = p2pRecvProxySetup(&connection, &proxyState, &req,
                                          reqSize, &resp, respSize, &done);

  // Validate results
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_NE(connection.transportResources, nullptr);
  EXPECT_EQ(1, done);

  // Cleanup
  ncclResult_t freeResult = p2pRecvProxyFree(&connection, &proxyState);
  EXPECT_EQ(freeResult, ncclSuccess);
  connection.transportResources = nullptr;
}

TEST_F(P2pTests, P2pRecvProxySetupInvalidReqSize) {
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  struct ncclP2pRequest req;
  int invalidReqSize = sizeof(ncclP2pRequest) - 1;
  memset(&req, 0, invalidReqSize);
  struct ncclP2pBuff resp;
  memset(&resp, 0, sizeof(struct ncclP2pBuff));
  int done = 0;
  ncclResult_t result =
      p2pRecvProxySetup(&connection, &proxyState, &req, invalidReqSize, &resp,
                        sizeof(resp), &done);
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(P2pTests, P2pRecvProxySetupInvalidRespSize) {
  struct ncclProxyConnection connection;
  memset(&connection, 0, sizeof(connection));
  struct ncclP2pRequest req;
  memset(&req, 0, sizeof(struct ncclP2pRequest));
  struct ncclP2pBuff resp;
  memset(&resp, 0, sizeof(struct ncclP2pBuff) - 1);
  int done = 0;

  ncclResult_t result =
      p2pRecvProxySetup(&connection, &proxyState, &req, sizeof(req), &resp,
                        sizeof(resp) - 1, &done);
  EXPECT_EQ(result, ncclInternalError);
  EXPECT_EQ(connection.transportResources, nullptr);
}

TEST_F(P2pTests, P2pSendProxyProgress) {
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
  ncclResult_t result = p2pSendProxyProgress(&proxyState, &args);
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
    result = p2pSendProxyProgress(&proxyState, &args);
    EXPECT_EQ(result, ncclSuccess);

    // Wait for the hipMemcpyAsync to complete
    hipStreamSynchronize(proxyInfo->stream);

    // Process again - should now increase done since event is complete
    result = p2pSendProxyProgress(&proxyState, &args);
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
  EXPECT_EQ(proxyInfo->shm->recvMem.tail,
            args.subs[0].base + args.subs[0].done);
}

TEST_F(P2pTests, P2pSendProxyProgress_ProtoLL) {
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
  ncclResult_t result = p2pSendProxyProgress(&proxyState, &args);

  // Validate the results
  EXPECT_EQ(result, ncclSuccess); // Expect the function to return success
  EXPECT_EQ(
      args.state,
      ncclProxyOpProgress); // Expect the state to transition to in progress
}
