/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <climits>
#include "collectives.h"
#include "comm.h"
#include "gtest/gtest.h"
#include "info.h"
#include "profiler.h"
#include "shmutils.h"
#include "socket.h"
#define ENABLE_TIMER 0
#include <assert.h>
#include <poll.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "os.h"
#include "profiler.h"
#include "proxy.h"
#include "timer.h"
#include "transport.h"

#define OP_INDEX(op) ((op) ? (op) - state->pools->elems : -1)
#define OP_SEEN 0x100000

ncclResult_t getOpIndex(
    struct ncclProxyArgs* op, struct ncclProxyProgressState* state, int* poolIndex, int* opIndex
);
ncclResult_t dumpProxyState(struct ncclProxyProgressState* state);
ncclResult_t printProxyOp(struct ncclProxyArgs* op, int poolIndex, int opIndex);
ncclResult_t dumpProxyState(struct ncclProxyProgressState* state);
ncclResult_t ncclProxyCallBlockingUDS(
    struct ncclComm*           comm,
    struct ncclProxyConnector* proxyConn,
    int                        type,
    void*                      reqBuff,
    int                        reqSize,
    void*                      respBuff,
    int                        respSize,
    int*                       reqFd,
    int*                       respFd
);
ncclResult_t ncclProxyClientGetFdBlocking(
    struct ncclComm* comm, int proxyRank, void* handle, int* convertedFd
);
ncclResult_t ncclProxyClientQueryFdBlocking(
    struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd
);

void ncclDumpProxyState(int signal);

// Defined in proxy.cc, not exported through proxy.h.
ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd);
ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState);

#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS

struct ncclProxyPool
{
    struct ncclProxyPool* next;
    struct ncclProxyArgs  elems[PROXYARGS_ALLOCATE_SIZE];
};

void init_ncclProxyArgs_struct(ncclProxyArgs* pool_ptr)
{
    // init pool_ptr
    pool_ptr->send        = 2;
    pool_ptr->nextRank    = 4;
    pool_ptr->prevRank    = 5;
    pool_ptr->pattern     = ncclPatternRing;
    pool_ptr->nsubs       = 1;
    pool_ptr->state       = ncclProxyOpNone;
    pool_ptr->retry_total = 2;
}

namespace RcclUnitTesting
{
TEST(ProxyTests, getOpIndex)
{ // Tests what is the index of the pool being passed within
  // the known valid pools in state ptr
    TEST_INFO("[ProxyTests] Test Start");

    // Init Dummy structs
    struct ncclProxyArgs*          pool_ptr   = new ncclProxyArgs;
    struct ncclProxyPool*          pools_ptr  = new ncclProxyPool;
    struct ncclProxyPool*          pools2_ptr = new ncclProxyPool;
    struct ncclProxyProgressState* state_ptr  = new ncclProxyProgressState;

    // state_ptr = &state;
    state_ptr->active = &pools_ptr->elems[1]; // chk
    state_ptr->pool   = pool_ptr;
    state_ptr->pools  = pools_ptr;

    pools_ptr->next = pools2_ptr;

    struct ncclProxyArgs*          x = &pools_ptr->elems[5]; // Passing the 5th element of the pool
    struct ncclProxyProgressState* y = state_ptr;
    y->pools->next                   = y->pools; // next points to self

    TEST_INFO(
        "[ProxyTests] x=%p y->pools=%p x-y=%ld",
        (void*)x,
        (void*)y->pools->elems,
        x - y->pools->elems
    );

    int          pool_idx, opIndex;
    ncclResult_t res = getOpIndex(x, y, &pool_idx, &opIndex);

    ASSERT_EQ(pool_idx, 0);
    ASSERT_EQ(opIndex, 5);

    TEST_INFO("[ProxyTests] pool_idx %d opIndex %d", pool_idx, opIndex);
    TEST_INFO("[ProxyTests] res %u", res);
    assert(res == ncclSuccess);

    delete pool_ptr;
    delete pools_ptr;
    delete pools2_ptr;
    delete state_ptr;
    TEST_INFO("[ProxyTests] Test Complete");
}

TEST(ProxyTests, printProxyOp)
{
    TEST_INFO("[ProxyTests] Test Start");
    // Init Dummy structs

    struct ncclProxyArgs* pool_ptr = new ncclProxyArgs;

    struct ncclProxyPool* pools_ptr  = new ncclProxyPool;
    struct ncclProxyPool* pools2_ptr = new ncclProxyPool;

    struct ncclProxyProgressState* state_ptr = new ncclProxyProgressState;

    // state_ptr = &state;
    state_ptr->active = &pools_ptr->elems[1]; // chk
    state_ptr->pool   = pool_ptr;
    state_ptr->pools  = pools_ptr;

    pools_ptr->next = pools2_ptr;

    struct ncclProxyArgs*          x = &pools_ptr->elems[5];
    struct ncclProxyProgressState* y = state_ptr;
    y->pools->next                   = y->pools; // next points to self

    TEST_INFO(
        "[ProxyTests] x=%p y->pools=%p x-y=%ld",
        (void*)x,
        (void*)y->pools->elems,
        x - y->pools->elems
    );

    init_ncclProxyArgs_struct(pool_ptr);

    int          pool_idx = 2, opIndex = 3; // random vals
    ncclResult_t res = printProxyOp(pool_ptr, pool_idx, opIndex);

    TEST_INFO("[ProxyTests] res %u", res);
    assert(res == ncclSuccess);

    delete pools_ptr;
    delete pools2_ptr;
    delete pool_ptr;
    delete state_ptr;
    TEST_INFO("[ProxyTests] Test Complete");
}

TEST(ProxyTests, dumpProxyState)
{
    TEST_INFO("[ProxyTests] Test Start");

    // Init Dummy structs
    struct ncclProxyArgs* pool_ptr;
    struct ncclProxyPool* pools_ptr  = new ncclProxyPool;
    struct ncclProxyPool* pools2_ptr = new ncclProxyPool;

    struct ncclProxyProgressState* state_ptr = new ncclProxyProgressState;

    state_ptr->active  = &pools_ptr->elems[1];
    pool_ptr           = &pools_ptr->elems[4];
    pool_ptr->next     = NULL;
    pool_ptr->nextPeer = NULL;

    state_ptr->pool           = pool_ptr;
    state_ptr->pool->next     = NULL;
    state_ptr->pool->nextPeer = NULL;
    state_ptr->pool->state    = OP_SEEN;
    state_ptr->pools          = pools_ptr;
    state_ptr->pools->next    = NULL;

    struct ncclProxyArgs* op = state_ptr->active;
    op->state                = OP_SEEN;
    op->nextPeer             = NULL;
    op->next                 = NULL;

    pools_ptr->next = NULL;

    init_ncclProxyArgs_struct(pool_ptr);

    int          pool_idx = 2, opIndex = 3; // random vals
    ncclResult_t res = dumpProxyState(state_ptr);

    TEST_INFO("[ProxyTests] res %u", res);
    ASSERT_EQ(res, ncclSuccess);

    delete pools_ptr;

    delete pools2_ptr;

    delete state_ptr;
    TEST_INFO("[ProxyTests] Test Complete");
}

TEST(ProxyTests, ncclProxyCallBlockingUDS)
{
    TEST_INFO("[ProxyTests] Test Start");

    // Init Dummy structs
    struct ncclComm* comm = new ncclComm;
    int*             arr  = new int[100];
    for(int i = 0; i < 100; i++)
    {
        arr[i] = i;
    }

    comm->topParentLocalRanks = arr;
    comm->localRank           = 10;

    int* arr_x = new int[20];
    for(int i = 0; i < 20; i++)
    {
        arr_x[i] = i;
    }
    comm->topParentRanks = arr_x;

    struct ncclProxyState* sharedProxyState = new ncclProxyState;
    uint64_t*              arr2             = new uint64_t[10];
    for(int i = 0; i < 10; i++)
    {
        arr2[i] = 122567 + i; // random
    }

    TEST_INFO("[ProxyTests] sizeof(ncclProxyConnector) = %zu", sizeof(ncclProxyConnector));
    struct ncclProxyConnector* proxyConn = new(std::nothrow) ncclProxyConnector[20];
    if(proxyConn == nullptr)
    {
        // Handle allocation failure
        TEST_INFO("[ProxyTests] Allocation failed");
        ASSERT_NE(proxyConn, nullptr);
    }

    proxyConn->tpRank = 2;

    comm->proxyState = sharedProxyState;

    comm->proxyState->peerAddressesUDS = arr2;

    comm->abortFlag = NULL;

    int rank = comm->topParentLocalRanks[comm->localRank];
    TEST_INFO("[ProxyTests] rank %d", rank);
    uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
    TEST_INFO("[ProxyTests] pidHash %lu ", pidHash);

    int type = ncclProxyMsgGetFd;
    // some memory on stack for storing request and response buffers
    uint64_t* x_mem    = new uint64_t[10];
    uint64_t* x_mem2   = new uint64_t[10];
    void*     reqBuff  = (void*)x_mem;
    int       reqSize  = sizeof(uint64_t) * 5;
    void*     respBuff = NULL;
    int       respSize = 0;
    int*      reqFd    = NULL;
    int*      respFd   = (int*)x_mem2;

    ncclResult_t res = ncclProxyCallBlockingUDS(
        comm,
        proxyConn,
        type,
        reqBuff,
        reqSize,
        respBuff,
        respSize,
        reqFd,
        respFd
    );

    bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
    TEST_INFO("[ProxyTests] res %u", bool_res);
    ASSERT_EQ(bool_res, true);
    delete comm;
    delete sharedProxyState;
    delete[] proxyConn;
    delete[] arr_x;
    delete[] arr;
    delete[] arr2;
    delete[] x_mem;
    delete[] x_mem2;

    TEST_INFO("[ProxyTests] Test Complete");
}

TEST(ProxyTests, ncclProxyClientGetFdBlocking)
{
    RUN_ISOLATED_TEST(
        "ncclProxyClientGetFdBlocking",
        []()
        {
            TEST_INFO("[ProxyTests] Test Start");

            // Init Dummy structs
            struct ncclComm* comm = new ncclComm;
            int*             arr  = new int[100];
            for(int i = 0; i < 100; i++)
            {
                arr[i] = i;
            }

            comm->topParentLocalRanks               = arr;
            comm->localRank                         = 10;
            struct ncclProxyState* sharedProxyState = new ncclProxyState;

            int* arr_x = new int[20];
            for(int i = 0; i < 20; i++)
            {
                arr_x[i] = i;
            }
            comm->topParentRanks = arr_x;

            uint64_t* arr2 = new uint64_t[10];
            for(int i = 0; i < 10; i++)
            {
                arr2[i] = 122567 + i; // random
            }

            struct ncclProxyConnector* proxyConn = new(std::nothrow) ncclProxyConnector[20];
            if(proxyConn == nullptr)
            {
                // Handle allocation failure
                TEST_INFO("[ProxyTests] Allocation failed");
                ASSERT_NE(proxyConn, nullptr);
            }

            proxyConn->tpRank                  = 2;
            comm->proxyState                   = sharedProxyState;
            comm->proxyState->peerAddressesUDS = arr2;
            comm->abortFlag                    = NULL;

            int rank = comm->topParentLocalRanks[comm->localRank];
            TEST_INFO("[ProxyTests] rank %d", rank);
            uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
            TEST_INFO("[ProxyTests] pidHash %lu", pidHash);

            int type = ncclProxyMsgGetFd;
            // some memory on stack for storing request and response buffers
            uint64_t* x_mem    = new uint64_t[10];
            uint64_t* x_mem2   = new uint64_t[10];
            void*     reqBuff  = (void*)x_mem;
            int       reqSize  = sizeof(uint64_t) * 5;
            void*     respBuff = NULL;
            int       respSize = 0;
            int*      reqFd    = NULL;
            int*      respFd   = (int*)x_mem2;

            comm->gproxyConn                   = proxyConn;
            comm->gproxyConn[rank].initialized = true;

            ncclResult_t res = ncclProxyClientGetFdBlocking(comm, rank, reqBuff, respFd);

            bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
            TEST_INFO("[ProxyTests] res %u", bool_res);
            ASSERT_EQ(bool_res, true);

            delete comm;
            delete sharedProxyState;
            delete[] proxyConn;
            delete[] arr_x;
            delete[] arr;
            delete[] arr2;
            delete[] x_mem;
            delete[] x_mem2;
            TEST_INFO("[ProxyTests] Test Complete");
            TEST_INFO("Test 'ncclProxyClientGetFdBlocking' PASSED");
        }
    );
}

TEST(ProxyTests, ncclProxyClientQueryFdBlocking)
{
    RUN_ISOLATED_TEST(
        "ncclProxyClientQueryFdBlocking",
        []()
        {
            TEST_INFO("[ProxyTests] Test Start");

            // Init Dummy structs
            struct ncclComm* comm = new ncclComm;
            int*             arr  = new int[100];
            for(int i = 0; i < 5; i++)
            {
                arr[i] = i;
            }

            comm->topParentLocalRanks = arr;
            comm->localRank           = 0;

            int* arr_x = new int[20];
            for(int i = 0; i < 20; i++)
            {
                arr_x[i] = i;
            }
            comm->topParentRanks = arr_x;

            struct ncclProxyState* sharedProxyState = new ncclProxyState;

            uint64_t* arr2 = new uint64_t[10];
            for(int i = 0; i < 10; i++)
            {
                arr2[i] = 122567 + i; // random
            }

            struct ncclProxyConnector* proxyConn = new(std::nothrow) ncclProxyConnector[20];
            if(proxyConn == nullptr)
            {
                // Handle allocation failure
                TEST_INFO("[ProxyTests] Allocation failed");
                ASSERT_NE(proxyConn, nullptr);
            }

            proxyConn->tpRank = 2;

            comm->proxyState = sharedProxyState;

            comm->proxyState->peerAddressesUDS = arr2;

            comm->abortFlag = NULL;

            int rank = comm->topParentLocalRanks[comm->localRank];
            TEST_INFO("[ProxyTests] rank %d", rank);
            uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];
            TEST_INFO("[ProxyTests] pidHash %lu", pidHash);

            int type = ncclProxyMsgGetFd;
            // some memory on stack for storing request and response buffers
            uint64_t* x_mem    = new uint64_t[10];
            uint64_t* x_mem2   = new uint64_t[10];
            void*     reqBuff  = (void*)x_mem;
            int       reqSize  = sizeof(uint64_t) * 5;
            void*     respBuff = NULL;
            int       respSize = 0;
            int*      reqFd    = NULL;
            int*      respFd   = (int*)x_mem2;

            comm->gproxyConn                   = proxyConn;
            comm->gproxyConn[rank].initialized = true;

            int localFd   = 0;
            int dummy_int = 20;
            respBuff      = &dummy_int;
            ncclResult_t res
                = ncclProxyClientQueryFdBlocking(comm, proxyConn, localFd, (int*)respBuff);

            bool bool_res = (res >= ncclSuccess && res <= ncclRemoteError);
            TEST_INFO("[ProxyTests] res %u", bool_res);
            ASSERT_EQ(bool_res, true);

            delete comm;
            delete sharedProxyState;
            delete[] proxyConn;
            delete[] arr_x;
            delete[] arr;
            delete[] arr2;
            delete[] x_mem;
            delete[] x_mem2;
            TEST_INFO("[ProxyTests] Test Complete");
            TEST_INFO("Test 'ncclProxyClientQueryFdBlocking' PASSED");
        }
    );
}

// Regression tests for proxy connection pool bounds checking.
// Bug 1: before the fix, the wire sent a raw void* that the server dereferenced directly.
// These tests verify that ncclProxyGetConnection rejects every malformed integer ID that an
// attacker could substitute for a legitimate connId received over the proxy socket.
TEST(ProxyTests, ProxyConnectionPoolBoundsCheck)
{
    TEST_INFO("[ProxyTests] ProxyConnectionPoolBoundsCheck start");

    // Build a minimal pool manually: 1 bank, 2 initialized slots (offset = 2).
    // We bypass ncclProxyNewConnection so the test has no runtime dependencies.
    struct ncclProxyConnection conns[NCCL_PROXY_CONN_POOL_SIZE] = {};
    struct ncclProxyConnection* bank0 = conns;
    struct ncclProxyConnectionPool pool;
    pool.pools  = &bank0;
    pool.banks  = 1;
    pool.offset = 2; // slots 0 and 1 are valid

    struct ncclProxyConnection* out = nullptr;

    // Valid IDs must succeed.
    EXPECT_EQ(ncclProxyGetConnection(&pool, 0, &out), ncclSuccess);
    EXPECT_EQ(out, &conns[0]);
    EXPECT_EQ(ncclProxyGetConnection(&pool, 1, &out), ncclSuccess);
    EXPECT_EQ(out, &conns[1]);

    // Negative ID — primary regression: wire attacker sends e.g. -1 to force arbitrary deref.
    EXPECT_EQ(ncclProxyGetConnection(&pool, -1, &out), ncclInvalidArgument);
    EXPECT_EQ(ncclProxyGetConnection(&pool, INT_MIN, &out), ncclInvalidArgument);

    // ID at or past high-water mark — slot was allocated but never initialized.
    EXPECT_EQ(ncclProxyGetConnection(&pool, 2, &out), ncclInvalidArgument);
    EXPECT_EQ(ncclProxyGetConnection(&pool, NCCL_PROXY_CONN_POOL_SIZE - 1, &out), ncclInvalidArgument);

    // ID whose bank index exceeds the number of allocated banks.
    EXPECT_EQ(ncclProxyGetConnection(&pool, NCCL_PROXY_CONN_POOL_SIZE, &out), ncclInvalidArgument);
    EXPECT_EQ(ncclProxyGetConnection(&pool, INT_MAX, &out), ncclInvalidArgument);

    // Null pool (no banks allocated).
    struct ncclProxyConnectionPool emptyPool = {nullptr, 0, 0};
    EXPECT_EQ(ncclProxyGetConnection(&emptyPool, 0, &out), ncclInvalidArgument);

    TEST_INFO("[ProxyTests] ProxyConnectionPoolBoundsCheck PASSED");
}

// std::mutex / std::condition_variable modernization coverage.
//
// ncclProxyOpsPool synchronizes the main thread (producer, ncclProxyPost) and the
// progress thread (consumer, ncclProxyGetPostedOps) through a std::mutex + a
// std::condition_variable that were ported from pthread_mutex_t / pthread_cond_t.
// Because the pool lives in cross-process shared memory, RCCL re-arms those
// std::-typed primitives as PTHREAD_PROCESS_SHARED via ncclOsSetMutexCondShared()
// using their native_handle(). The following two tests exercise that machinery
// directly: the round-trip test validates the intra-process RAII lock + wait/notify
// protocol, and the process-shared test validates the same std::mutex / condvar work
// across a fork() the way the real proxy uses them.

static constexpr int kNoOps = -1;
static constexpr int kPostedOpIndex = 7;
static constexpr auto kWaitTimeout = std::chrono::seconds(10);

// Single-process producer/consumer round-trip: producer calls the real ncclProxyPost(),
// consumer mirrors ncclProxyGetPostedOps() on the pool's std::mutex/condition_variable.
TEST(ProxyTests, ProxyOpsPoolMutexCondRoundTrip)
{
    TEST_INFO("[ProxyTests] ProxyOpsPoolMutexCondRoundTrip start");

    auto pool = std::make_unique<ncclProxyOpsPool>();
    pool->nextOps    = kNoOps;
    pool->nextOpsEnd = kNoOps;

    ncclOsSetMutexCondShared(pool->mutex, pool->cond, &pool->syncObjectsInitialized);

    int  observedOps  = kNoOps;
    bool waitSucceeded = false;
    bool stop          = false;

    // Consumer: block until an op is posted or a stop is requested.
    std::thread consumer([&]() {
        std::unique_lock<std::mutex> lock(pool->mutex);
        waitSucceeded = pool->cond.wait_for(
            lock, kWaitTimeout, [&]() { return pool->nextOps != kNoOps || stop; });
        observedOps    = pool->nextOps;
        pool->nextOps  = kNoOps;
    });

    ASSERT_EQ(ncclProxyPost(pool.get(), kPostedOpIndex, kPostedOpIndex), ncclSuccess);

    consumer.join();

    EXPECT_TRUE(waitSucceeded) << "consumer timed out waiting on the pool condvar";
    EXPECT_EQ(observedOps, kPostedOpIndex);

    // Shutdown wake path: stop is set and notified, consumer must wake with no op posted.
    observedOps   = kPostedOpIndex;
    waitSucceeded = false;
    pool->nextOps = kNoOps;

    std::thread stopper([&]() {
        std::unique_lock<std::mutex> lock(pool->mutex);
        waitSucceeded = pool->cond.wait_for(
            lock, kWaitTimeout, [&]() { return pool->nextOps != kNoOps || stop; });
        observedOps = pool->nextOps;
    });

    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        stop = true;
        pool->cond.notify_one();
    }

    stopper.join();

    EXPECT_TRUE(waitSucceeded) << "consumer did not wake on stop request";
    EXPECT_EQ(observedOps, kNoOps) << "stop wake must not fabricate a posted op";

    TEST_INFO("[ProxyTests] ProxyOpsPoolMutexCondRoundTrip PASSED");
}

// Covers when an op is already queued (nextOps != -1), a subsequent
// post chains onto ops[nextOpsEnd].next and advances nextOpsEnd, without
// touching nextOps or notifying.
TEST(ProxyTests, ProxyOpsPoolPostChainsWhenPending)
{
    TEST_INFO("[ProxyTests] ProxyOpsPoolPostChainsWhenPending start");

    constexpr int kFirstOp  = 3;
    constexpr int kSecondOp = 5;

    auto pool = std::make_unique<ncclProxyOpsPool>();
    pool->nextOps    = kNoOps;
    pool->nextOpsEnd = kNoOps;

    // First post takes the nextOps == -1 branch and becomes the head of the chain.
    ASSERT_EQ(ncclProxyPost(pool.get(), kFirstOp, kFirstOp), ncclSuccess);
    ASSERT_EQ(pool->nextOps, kFirstOp);
    ASSERT_EQ(pool->nextOpsEnd, kFirstOp);

    // Second post takes the else branch: it links onto the pending tail, leaving the head
    // untouched and only advancing nextOpsEnd.
    ASSERT_EQ(ncclProxyPost(pool.get(), kSecondOp, kSecondOp), ncclSuccess);
    EXPECT_EQ(pool->nextOps, kFirstOp) << "head must not change while an op is pending";
    EXPECT_EQ(pool->ops[kFirstOp].next, kSecondOp) << "new op must chain onto the tail";
    EXPECT_EQ(pool->nextOpsEnd, kSecondOp);

    TEST_INFO("[ProxyTests] ProxyOpsPoolPostChainsWhenPending PASSED");
}

struct ProxyOpsPoolShmRegion
{
    struct ncclProxyOpsPool pool;
    volatile int            childObserved; // op index the child read, or sentinel
};

// Process-shared: Arm the pool's mutex/condvar PTHREAD_PROCESS_SHARED, fork, have the child block in
// cond.wait() and the parent publish an op + notify. Proves a std::mutex / std::condition_variable 
// synchronizes correctly across processes through their pthread native handles.
TEST(ProxyTests, ProxyOpsPoolMutexCondProcessShared)
{
    RUN_ISOLATED_TEST(
        "ProxyOpsPoolMutexCondProcessShared",
        []() {
            TEST_INFO("[ProxyTests] ProxyOpsPoolMutexCondProcessShared start");

            // Anonymous shared memory: MAP_SHARED makes writes visible across fork()
            void* map = mmap(
                nullptr,
                sizeof(ProxyOpsPoolShmRegion),
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS,
                -1,
                0);
            ASSERT_NE(map, MAP_FAILED);

            auto* region        = static_cast<ProxyOpsPoolShmRegion*>(map);
            auto& pool          = region->pool;
            pool.nextOps        = kNoOps;
            pool.nextOpsEnd     = kNoOps;
            region->childObserved = kNoOps - 1;

            ncclOsSetMutexCondShared(pool.mutex, pool.cond, &pool.syncObjectsInitialized);

            pid_t pid = fork();
            ASSERT_NE(pid, -1);

            if(pid == 0)
            {
                // Child = consumer. Watchdog so a broken pshared cond can never hang CI.
                alarm(30);
                std::unique_lock<std::mutex> lock(pool.mutex);
                bool woke = pool.cond.wait_for(
                    lock, kWaitTimeout, [&]() { return pool.nextOps != kNoOps; });
                region->childObserved = woke ? pool.nextOps : (kNoOps - 2);
                lock.unlock();
                _exit(woke ? 0 : 1);
            }

            ASSERT_EQ(ncclProxyPost(&pool, kPostedOpIndex, kPostedOpIndex), ncclSuccess);

            int status = 0;
            ASSERT_EQ(waitpid(pid, &status, 0), pid);
            EXPECT_TRUE(WIFEXITED(status)) << "child did not exit normally";
            EXPECT_EQ(WEXITSTATUS(status), 0) << "child timed out on the shared condvar";
            EXPECT_EQ(region->childObserved, kPostedOpIndex)
                << "child did not observe the cross-process posted op";

            ASSERT_EQ(munmap(map, sizeof(ProxyOpsPoolShmRegion)), 0);
            TEST_INFO("[ProxyTests] ProxyOpsPoolMutexCondProcessShared PASSED");
        });
}

// Drive the real ncclProxyProgressDestroy: it must wake and join the progress thread via
// the opsPool condvar (stop path), then free the whole pools chain. The joined thread is a
// stand-in consumer mirroring ncclProxyProgress's stop check; the destroy itself is real.
TEST(ProxyTests, ProxyProgressDestroyJoinsAndFreesPools)
{
    TEST_INFO("[ProxyTests] ProxyProgressDestroyJoinsAndFreesPools start");

    auto state   = std::make_unique<ncclProxyState>();
    auto opsPool = std::make_unique<ncclProxyOpsPool>();

    auto& ps    = state->progressState;
    ps.opsPool  = opsPool.get();
    ps.stop     = 0;

    // Stand-in progress thread: block on the pool condvar until destroy sets stop.
    ps.thread = std::thread([&]() {
        std::unique_lock<std::mutex> lock(opsPool->mutex);
        opsPool->cond.wait_for(lock, kWaitTimeout, [&]() { return ps.stop != 0; });
    });

    // Two-node pool chain for the free loop to walk and release.
    auto* pool0 = static_cast<ncclProxyPool*>(calloc(1, sizeof(ncclProxyPool)));
    auto* pool1 = static_cast<ncclProxyPool*>(calloc(1, sizeof(ncclProxyPool)));
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);
    pool0->next = pool1;
    pool1->next = nullptr;
    ps.pools    = pool0;

    EXPECT_EQ(ncclProxyProgressDestroy(state.get()), ncclSuccess);

    EXPECT_EQ(ps.stop, 1) << "destroy must request the progress thread to stop";
    EXPECT_FALSE(ps.thread.joinable()) << "destroy must join the progress thread";
    EXPECT_EQ(ps.pools, nullptr) << "destroy must free the entire pools chain";

    TEST_INFO("[ProxyTests] ProxyProgressDestroyJoinsAndFreesPools PASSED");
}

} // namespace RcclUnitTesting
