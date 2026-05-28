/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "gtest/gtest.h"

#include "collectives.h"
#include "comm.h"
#include "info.h"
#include "profiler.h"
#include "shmutils.h"
#include "socket.h"
#define ENABLE_TIMER 0
#include "profiler.h"
#include "proxy.h"
#include "timer.h"
#include "transport.h"

#include <assert.h>
#include <poll.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define NCCL_MAX_OPS (2048)
#define OP_INDEX(op) ((op) ? (op) - state->pools->elems : -1)
#define OP_SEEN 0x100000

ncclResult_t getOpIndex(struct ncclProxyArgs *op,
                        struct ncclProxyProgressState *state, int *poolIndex,
                        int *opIndex);
ncclResult_t dumpProxyState(struct ncclProxyProgressState *state);
ncclResult_t printProxyOp(struct ncclProxyArgs *op, int poolIndex, int opIndex);
ncclResult_t dumpProxyState(struct ncclProxyProgressState *state);
ncclResult_t ncclProxyCallBlockingUDS(struct ncclComm *comm,
                                      struct ncclProxyConnector *proxyConn,
                                      int type, void *reqBuff, int reqSize,
                                      void *respBuff, int respSize, int *reqFd,
                                      int *respFd);
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm *comm, int proxyRank,
                                          void *handle, int *convertedFd);
ncclResult_t
ncclProxyClientQueryFdBlocking(struct ncclComm *comm,
                               struct ncclProxyConnector *proxyConn,
                               int localFd, int *rmtFd);

void ncclDumpProxyState(int signal);

#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

void init_ncclProxyArgs_struct(ncclProxyArgs *pool_ptr) {
  // init pool_ptr
  pool_ptr->send = 2;
  pool_ptr->nextRank = 4;
  pool_ptr->prevRank = 5;
  pool_ptr->pattern = ncclPatternRing;
  pool_ptr->nsubs = 1;
  pool_ptr->state = ncclProxyOpNone;
  pool_ptr->retry_total = 2;
}
namespace RcclUnitTesting {
TEST(ProxyTests,
     getOpIndex) { // Tests what is the index of the pool being passed within
                   // the known valid pools in state ptr
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");

  // Init Dummy structs
  struct ncclProxyArgs *pool_ptr = new ncclProxyArgs;
  struct ncclProxyPool *pools_ptr = new ncclProxyPool;
  struct ncclProxyPool *pools2_ptr = new ncclProxyPool;
  struct ncclProxyProgressState *state_ptr = new ncclProxyProgressState;

  // state_ptr = &state;
  state_ptr->active = &pools_ptr->elems[1]; // chk
  state_ptr->pool = pool_ptr;
  state_ptr->pools = pools_ptr;

  pools_ptr->next = pools2_ptr;

  struct ncclProxyArgs *x =
      &pools_ptr->elems[5]; // Passing the 5th element of the pool
  struct ncclProxyProgressState *y = state_ptr;
  y->pools->next = y->pools; // next points to self

  INFO(NCCL_LOG_INFO, "[ProxyTests] x=%u y->pools=%u x-y=%u \n", x,
       y->pools->elems, x - y->pools->elems);

  int pool_idx, opIndex;
  ncclResult_t res = getOpIndex(x, y, &pool_idx, &opIndex);

  ASSERT_EQ(pool_idx, 0);
  ASSERT_EQ(opIndex, 5);

  INFO(NCCL_LOG_INFO, "[ProxyTests] pool_idx %d opIndex %d \n", pool_idx,
       opIndex);
  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", res);
  assert(res == ncclSuccess);

    delete pool_ptr;
    delete pools_ptr;
    delete pools2_ptr;
    delete state_ptr;
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

TEST(ProxyTests, printProxyOp) {
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");
  // Init Dummy structs

  struct ncclProxyArgs *pool_ptr = new ncclProxyArgs;

  struct ncclProxyPool *pools_ptr = new ncclProxyPool;
  struct ncclProxyPool *pools2_ptr = new ncclProxyPool;

  struct ncclProxyProgressState *state_ptr = new ncclProxyProgressState;

  // state_ptr = &state;
  state_ptr->active = &pools_ptr->elems[1]; // chk
  state_ptr->pool = pool_ptr;
  state_ptr->pools = pools_ptr;

  pools_ptr->next = pools2_ptr;

  struct ncclProxyArgs *x = &pools_ptr->elems[5];
  struct ncclProxyProgressState *y = state_ptr;
  y->pools->next = y->pools; // next points to self

  INFO(NCCL_LOG_INFO, "[ProxyTests] x=%u y->pools=%u x-y=%u \n", x,
       y->pools->elems, x - y->pools->elems);

  init_ncclProxyArgs_struct(pool_ptr);

  int pool_idx = 2, opIndex = 3; // random vals
  ncclResult_t res = printProxyOp(pool_ptr, pool_idx, opIndex);

  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", res);
  assert(res == ncclSuccess);

    delete pools_ptr;
    delete pools2_ptr;
    delete pool_ptr;
    delete state_ptr;
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

TEST(ProxyTests, dumpProxyState) {
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");

  // Init Dummy structs
  struct ncclProxyArgs *pool_ptr;
  struct ncclProxyPool *pools_ptr = new ncclProxyPool;
  struct ncclProxyPool *pools2_ptr = new ncclProxyPool;

  struct ncclProxyProgressState *state_ptr = new ncclProxyProgressState;

  state_ptr->active = &pools_ptr->elems[1];
  pool_ptr = &pools_ptr->elems[4];
  pool_ptr->next = NULL;
  pool_ptr->nextPeer = NULL;

  state_ptr->pool = pool_ptr;
  state_ptr->pool->next = NULL;
  state_ptr->pool->nextPeer = NULL;
  state_ptr->pool->state = OP_SEEN;
  state_ptr->pools = pools_ptr;
  state_ptr->pools->next = NULL;

  struct ncclProxyArgs *op = state_ptr->active;
  op->state = OP_SEEN;
  op->nextPeer = NULL;
  op->next = NULL;

  pools_ptr->next = NULL;

  init_ncclProxyArgs_struct(pool_ptr);

  int pool_idx = 2, opIndex = 3; // random vals
  ncclResult_t res = dumpProxyState(state_ptr);

  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", res);
  ASSERT_EQ(res, ncclSuccess);

    delete pools_ptr;
    
    delete pools2_ptr;
    
    
    delete state_ptr;
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

TEST(ProxyTests, ncclProxyCallBlockingUDS) {
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");

  // Init Dummy structs
  struct ncclComm *comm = new ncclComm;
  int *arr = new int[100];
  for (int i = 0; i < 100; i++) {
    arr[i] = i;
  }

  comm->topParentLocalRanks = arr;
  comm->localRank = 10;

  int *arr_x = new int[20];
  for (int i = 0; i < 20; i++) {
    arr_x[i] = i;
  }
  comm->topParentRanks = arr_x;

  struct ncclProxyState *sharedProxyState = new ncclProxyState;
  uint64_t *arr2 = new uint64_t[10];
  for (int i = 0; i < 10; i++) {
    arr2[i] = 122567 + i; // random
  }

  INFO(NCCL_LOG_INFO, "[ProxyTests] sizeof(ncclProxyConnector) = %u\n",
       sizeof(ncclProxyConnector));
  struct ncclProxyConnector *proxyConn =
      new (std::nothrow) ncclProxyConnector[20];
  if (proxyConn == nullptr) {
    // Handle allocation failure
    INFO(NCCL_LOG_INFO, "[ProxyTests] Allocation failed\n");
    ASSERT_NE(proxyConn, nullptr);
  }

  proxyConn->tpRank = 2;

  comm->proxyState = sharedProxyState;

  comm->proxyState->peerAddressesUDS = arr2;

  comm->abortFlag = NULL;

  int rank = comm->topParentLocalRanks[comm->localRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] rank %d\n", rank);
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] pidHash %u \n", pidHash);

  int type = ncclProxyMsgGetFd;
  // some memory on stack for storing request and response buffers
  uint64_t *x_mem = new uint64_t[10];
  uint64_t *x_mem2 = new uint64_t[10];
  void *reqBuff = (void *)x_mem;
  int reqSize = sizeof(uint64_t) * 5;
  void *respBuff = NULL;
  int respSize = 0;
  int *reqFd = NULL;
  int *respFd = (int *)x_mem2;

  ncclResult_t res =
      ncclProxyCallBlockingUDS(comm, proxyConn, type, reqBuff, reqSize,
                               respBuff, respSize, reqFd, respFd);

  bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", bool_res);
  ASSERT_EQ(bool_res, true);
    delete comm;
    delete sharedProxyState;
    delete proxyConn;
    delete[] arr_x;
    delete[] arr;
    delete[] arr2;
    delete[] x_mem;
    delete[] x_mem2;

  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

TEST(ProxyTests, ncclProxyClientGetFdBlocking) {
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");

  // Init Dummy structs
  struct ncclComm *comm = new ncclComm;
  int *arr = new int[100];
  for (int i = 0; i < 100; i++) {
    arr[i] = i;
  }

  comm->topParentLocalRanks = arr;
  comm->localRank = 10;
  struct ncclProxyState *sharedProxyState = new ncclProxyState;

  int *arr_x = new int[20];
  for (int i = 0; i < 20; i++) {
    arr_x[i] = i;
  }
  comm->topParentRanks = arr_x;

  uint64_t *arr2 = new uint64_t[10];
  for (int i = 0; i < 10; i++) {
    arr2[i] = 122567 + i; // random
  }

  struct ncclProxyConnector *proxyConn =
      new (std::nothrow) ncclProxyConnector[20];
  if (proxyConn == nullptr) {
    // Handle allocation failure
    INFO(NCCL_LOG_INFO, "[ProxyTests] Allocation failed\n");
    ASSERT_NE(proxyConn, nullptr);
  }

  proxyConn->tpRank = 2;
  comm->proxyState = sharedProxyState;
  comm->proxyState->peerAddressesUDS = arr2;
  comm->abortFlag = NULL;

  int rank = comm->topParentLocalRanks[comm->localRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] rank %d\n", rank);
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] pidHash %u \n", pidHash);

  int type = ncclProxyMsgGetFd;
  // some memory on stack for storing request and response buffers
  uint64_t *x_mem = new uint64_t[10];
  uint64_t *x_mem2 = new uint64_t[10];
  void *reqBuff = (void *)x_mem;
  int reqSize = sizeof(uint64_t) * 5;
  void *respBuff = NULL;
  int respSize = 0;
  int *reqFd = NULL;
  int *respFd = (int *)x_mem2;

  comm->gproxyConn = proxyConn;
  comm->gproxyConn[rank].initialized = true;

  ncclResult_t res = ncclProxyClientGetFdBlocking(comm, rank, reqBuff, respFd);

  bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", bool_res);
  ASSERT_EQ(bool_res, true);

    delete comm;
    delete sharedProxyState;
    delete proxyConn;
    delete[] arr_x;
    delete[] arr;
    delete[] arr2;
    delete[] x_mem;
    delete[] x_mem2;
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

TEST(ProxyTests, ncclProxyClientQueryFdBlocking) {
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Start \n");

  // Init Dummy structs
  struct ncclComm *comm = new ncclComm;
  int *arr = new int[100];
  for (int i = 0; i < 5; i++) {
    arr[i] = i;
  }

  comm->topParentLocalRanks = arr;
  comm->localRank = 0;

  int *arr_x = new int[20];
  for (int i = 0; i < 20; i++) {
    arr_x[i] = i;
  }
  comm->topParentRanks = arr_x;

  struct ncclProxyState *sharedProxyState = new ncclProxyState;

  uint64_t *arr2 = new uint64_t[10];
  for (int i = 0; i < 10; i++) {
    arr2[i] = 122567 + i; // random
  }

  struct ncclProxyConnector *proxyConn =
      new (std::nothrow) ncclProxyConnector[20];
  if (proxyConn == nullptr) {
    // Handle allocation failure
    INFO(NCCL_LOG_INFO, "[ProxyTests] Allocation failed\n");
    ASSERT_NE(proxyConn, nullptr);
  }

  proxyConn->tpRank = 2;

  comm->proxyState = sharedProxyState;

  comm->proxyState->peerAddressesUDS = arr2;

  comm->abortFlag = NULL;

  int rank = comm->topParentLocalRanks[comm->localRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] rank %d\n", rank);
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
  INFO(NCCL_LOG_INFO, "[ProxyTests] pidHash %u \n", pidHash);

  int type = ncclProxyMsgGetFd;
  // some memory on stack for storing request and response buffers
  uint64_t *x_mem = new uint64_t[10];
  uint64_t *x_mem2 = new uint64_t[10];
  void *reqBuff = (void *)x_mem;
  int reqSize = sizeof(uint64_t) * 5;
  void *respBuff = NULL;
  int respSize = 0;
  int *reqFd = NULL;
  int *respFd = (int *)x_mem2;

  comm->gproxyConn = proxyConn;
  comm->gproxyConn[rank].initialized = true;

  int localFd = 0;
  int dummy_int = 20;
  respBuff = &dummy_int;
  ncclResult_t res =
      ncclProxyClientQueryFdBlocking(comm, proxyConn, localFd, (int *)respBuff);

  bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
  INFO(NCCL_LOG_INFO, "[ProxyTests] res %u \n", bool_res);
  ASSERT_EQ(bool_res, true);

    delete comm;
    delete sharedProxyState;
    delete proxyConn;
    delete[] arr_x;
    delete[] arr;
    delete[] arr2;
    delete[] x_mem;
    delete[] x_mem2;
  INFO(NCCL_LOG_INFO, "[ProxyTests] Test Complete \n");
}

} // namespace RcclUnitTesting
