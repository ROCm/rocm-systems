/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "rccl/rccl.h"

#include "nccl_net.h"
#include "nccl_gin.h"

extern ncclGin_t IbCastGinIbProxy;

namespace RCCLGinTests
{

using ::MPITestConstants::kMinProcessesForMPI;
using ::MPITestConstants::kNoProcessLimit;
using ::MPITestConstants::kNoPowerOfTwoRequired;
using ::MPITestConstants::kNoNodeLimit;

inline ncclGin_t* GetGinPlugin()
{
    const char* envNet = std::getenv("NCCL_NET");
    if (envNet == nullptr) return nullptr;

    if (strcasecmp(envNet, "IB-CAST") == 0) return &IbCastGinIbProxy;
    // if (strcasecmp(envNet, "IB") == 0) return &ncclGinIbProxy;

    return nullptr;
}

class GinMPITestBase : public MPITestBase
{
protected:
    int defaultDevice_  = 0;
    int defaultSignals_ = 8;
    int defaultCounters_ = 8;
    int defaultQueueDepth_ = 0;

    // Polling
    static constexpr int kDefaultPollTimeoutMs = 5000;
    static constexpr int kPollSleepUs          = 100;

    // Plugin handle table under test (resolved in SetUp from env var).
    ncclGin_t* gin_ = nullptr;

    // Plugin / connection state (populated by SetUpFixture).
    void* pluginCtx_  = nullptr;
    void* listenComm_ = nullptr;
    void* collComm_   = nullptr;
    void* ginCtx_     = nullptr;
    ncclNetDeviceHandle_v11_t* devHandle_ = nullptr;

    int nDevices_  = 0;
    int worldRank_ = -1;
    int worldSize_ = 0;

    // MRs registered through RegMr released via deregMrSym in TearDown
    std::vector<void*> registeredMhandles_;

    // Device buffers allocated through AllocBuf released via hipFree in TearDown
    std::vector<void*> allocatedDeviceBuffers_;

    virtual int  GetNumContexts() const { return 1; }
    virtual bool UseDmaBuf()     const { return false; }

    void SetUp() override
    {
        MPITestBase::SetUp();
        gin_ = GetGinPlugin();
        if (gin_ == nullptr)
        {
            // GetGinPlugin() returned null -> NCCL_NET selects a transport
            // with no GIN backend ported to this RCCL build (or is unset).
            static bool warned = false;
            if (!warned)
            {
                const char* envNet = std::getenv("NCCL_NET");
                std::fprintf(stderr,
                             "[GinMPITest] WARN: NCCL_NET=%s has no GIN "
                             "backend in this RCCL build. Currently only "
                             "NCCL_NET=ib-cast is supported. Skipping all "
                             "GIN tests.\n",
                             envNet ? envNet : "<unset>");
                warned = true;
            }
            GTEST_SKIP() << "No GIN backend available for current NCCL_NET";
            return;
        }
    }

    void TearDown() override
    {
        if(gin_ != nullptr)
        {
            if(collComm_ != nullptr && gin_->deregMrSym)
            {
                for(void* mh : registeredMhandles_)
                {
                    if(mh != nullptr) (void)gin_->deregMrSym(collComm_, mh);
                }
            }
            registeredMhandles_.clear();
        }

        for(void* p : allocatedDeviceBuffers_)
        {
            RCCLTestGuards::hipFreeWrapper(p);   // null-safe + warns on hipFree errors
        }
        allocatedDeviceBuffers_.clear();

        if(gin_ != nullptr)
        {
            if(ginCtx_ != nullptr && gin_->destroyContext) {
                (void)gin_->destroyContext(ginCtx_);
            }
            if(collComm_ != nullptr && gin_->closeColl) {
                (void)gin_->closeColl(collComm_);
            }
            if(listenComm_ != nullptr && gin_->closeListen) {
                (void)gin_->closeListen(listenComm_);
            }
            if(pluginCtx_ != nullptr && gin_->finalize) {
                (void)gin_->finalize(pluginCtx_);
            }
        }
        ginCtx_ = nullptr;
        collComm_ = nullptr;
        devHandle_ = nullptr;
        pluginCtx_ = nullptr;
        listenComm_ = nullptr;

        MPITestBase::TearDown();
    }

    bool SetUpFixture(int minProcesses = kMinProcessesForMPI,
                      int maxProcesses = kNoProcessLimit,
                      int minNodes = 1)
    {
        // SetUp() already recorded the skip + warning when NCCL_NET wasn't
        // ib-cast. Short-circuit so the per-test body doesn't proceed to
        // dereference gin_->init() etc.
        if (gin_ == nullptr) return false;

        if(!validateTestPrerequisites(minProcesses, maxProcesses,
                                      kNoPowerOfTwoRequired,
                                      minNodes, kNoNodeLimit))
        {
            return false;
        }

        // GPU is a hard prerequisite — all buffers are device-allocated.
        int nGpus = 0;
        if(hipGetDeviceCount(&nGpus) != hipSuccess || nGpus <= 0)
        {
            ADD_FAILURE() << "No HIP devices visible; GIN tests require GPU";
            return false;
        }

        worldRank_ = MPIEnvironment::world_rank;
        worldSize_ = MPIEnvironment::world_size;

        // 1. init
        ncclResult_t r = gin_->init(&pluginCtx_, /*commId=*/0, nullptr);
        if(r != ncclSuccess)
        {
            ADD_FAILURE() << "gin->init failed: " << r;
            return false;
        }

        // 2. devices — skip if no IB hardware
        r = gin_->devices(&nDevices_);
        if(r != ncclSuccess)
        {
            ADD_FAILURE() << "gin->devices failed: " << r;
            return false;
        }
        if(nDevices_ <= 0)
        {
            // See comment above on the bool-vs-void GTEST_SKIP wrapping.
            [&]() { GTEST_SKIP() << "No IB GIN-capable devices on this host"; }();
            return false;
        }

        // 3. listen — produce this rank's listen handle
        std::vector<char> localHandle(NCCL_NET_HANDLE_MAXSIZE, 0);
        r = gin_->listen(pluginCtx_, defaultDevice_,
                         localHandle.data(), &listenComm_);
        if(r != ncclSuccess)
        {
            ADD_FAILURE() << "gin->listen failed: " << r;
            return false;
        }

        // 4. allgather all listen handles
        std::vector<char> allHandles(NCCL_NET_HANDLE_MAXSIZE * worldSize_, 0);
        int mpiRet = MPI_Allgather(localHandle.data(), NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE,
                                   allHandles.data(),  NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE,
                                   MPI_COMM_WORLD);
        if(mpiRet != MPI_SUCCESS)
        {
            ADD_FAILURE() << "MPI_Allgather of listen handles failed: " << mpiRet;
            return false;
        }

        std::vector<void*> handlePtrs(worldSize_, nullptr);
        for(int i = 0; i < worldSize_; ++i)
        {
            handlePtrs[i] = allHandles.data() + i * NCCL_NET_HANDLE_MAXSIZE;
        }

        // 5. connect (N-way collective)
        r = gin_->connect(pluginCtx_, handlePtrs.data(),
                          worldSize_, worldRank_,
                          listenComm_, &collComm_);
        if(r != ncclSuccess)
        {
            ADD_FAILURE() << "gin->connect failed: " << r;
            return false;
        }

        // 6. createContext — number of contexts driven by the parameter
        ncclGinConfig_t cfg = {};
        cfg.nSignals = defaultSignals_;
        cfg.nCounters = defaultCounters_;
        cfg.nContexts = GetNumContexts();
        cfg.queueDepth = defaultQueueDepth_;
        cfg.trafficClass = -1;

        r = gin_->createContext(collComm_, &cfg, &ginCtx_, &devHandle_);
        if(r != ncclSuccess)
        {
            ADD_FAILURE() << "gin->createContext failed: " << r;
            return false;
        }

        TEST_INFO("GinMPI fixture ready: rank=%d/%d nDevices=%d nContexts=%d regMethod=%s",
                  worldRank_, worldSize_, nDevices_, GetNumContexts(),
                  UseDmaBuf() ? "DmaBuf" : "RegMr");
        return true;
    }

    ncclResult_t RegMr(void* data, size_t size,
                       void** mhandle, void** ginHandle)
    {
        if(UseDmaBuf())
        {
            return RegMrDmaBuf(data, size, mhandle, ginHandle);
        }
        ncclResult_t r = gin_->regMrSym(collComm_, data, size,
                                        NCCL_PTR_CUDA, /*mrFlags=*/0,
                                        mhandle, ginHandle);
        if(r == ncclSuccess && mhandle != nullptr && *mhandle != nullptr)
        {
            registeredMhandles_.push_back(*mhandle);
        }
        return r;
    }

    ncclResult_t RegMrDmaBuf(void* data, size_t size,
                             void** mhandle, void** ginHandle)
    {
        if(gin_->regMrSymDmaBuf == nullptr)
        {
            ADD_FAILURE() << "regMrSymDmaBuf is NULL (SetUpFixture should have skipped)";
            return ncclInternalError;
        }

        const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        uintptr_t aligned = reinterpret_cast<uintptr_t>(data) & ~(pageSize - 1);
        size_t    offset  = reinterpret_cast<uintptr_t>(data) - aligned;
        size_t    alignedSize = (size + offset + pageSize - 1) & ~(pageSize - 1);

        int       fd = -1;
        uint64_t  exportOffset = 0;
        hsa_status_t hrc = hsa_amd_portable_export_dmabuf(
            reinterpret_cast<const void*>(aligned), alignedSize,
            &fd, &exportOffset);
        if(hrc != HSA_STATUS_SUCCESS || fd < 0)
        {
            ADD_FAILURE() << "hsa_amd_portable_export_dmabuf failed: hsa_status=" << hrc;
            return ncclSystemError;
        }

        ncclResult_t r = gin_->regMrSymDmaBuf(collComm_, data, size,
                                               NCCL_PTR_CUDA,
                                               exportOffset + offset, fd,
                                               /*mrFlags=*/0,
                                               mhandle, ginHandle);
        (void)close(fd);

        if(r == ncclSuccess && mhandle != nullptr && *mhandle != nullptr)
        {
            registeredMhandles_.push_back(*mhandle);
        }
        return r;
    }

    bool PollUntilDone(void* request, int timeoutMs = kDefaultPollTimeoutMs)
    {
        if(request == nullptr)
        {
            ADD_FAILURE() << "PollUntilDone called with null request "
                             "(plugin returned ncclSuccess but did not "
                             "produce a request handle)";
            return false;
        }

        const auto deadline = std::chrono::steady_clock::now()
                              + std::chrono::milliseconds(timeoutMs);
        for(;;)
        {
            int done = 0;
            ncclResult_t r = gin_->test(collComm_, request, &done);
            if(r != ncclSuccess)
            {
                ADD_FAILURE() << "gin->test failed: " << r;
                return false;
            }
            if(done) return true;

            if(std::chrono::steady_clock::now() >= deadline)
            {
                ADD_FAILURE() << "PollUntilDone timed out after " << timeoutMs << " ms";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(kPollSleepUs));
        }
    }

    /** MPI barrier across the world communicator. */
    void Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

    void* AllocBuf(size_t size)
    {
        void* p = nullptr;
        if(hipMalloc(&p, size) != hipSuccess) return nullptr;

        hipError_t err = hipMemset(p, 0, size);
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "hipMemset failed in AllocBuf: "
                          << hipGetErrorString(err);
            (void)hipFree(p);
            return nullptr;
        }

        err = hipDeviceSynchronize();
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "hipDeviceSynchronize failed in AllocBuf: "
                          << hipGetErrorString(err);
            (void)hipFree(p);
            return nullptr;
        }

        allocatedDeviceBuffers_.push_back(p);
        return p;
    }

    /** Fill device `buf` with the byte pattern (seed + i) % 256. */
    void FillBuf(void* buf, size_t size, int seed)
    {
        std::vector<uint8_t> tmp(size);
        for(size_t i = 0; i < size; ++i)
            tmp[i] = static_cast<uint8_t>((seed + i) % 256);
        (void)hipMemcpy(buf, tmp.data(), size, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
    }

    /** Return true iff device `buf` matches (seed + i) % 256 for all i. */
    bool VerifyBuf(const void* buf, size_t size, int seed)
    {
        std::vector<uint8_t> staging(size);
        if(hipMemcpy(staging.data(), buf, size, hipMemcpyDeviceToHost) != hipSuccess)
            return false;
        (void)hipDeviceSynchronize();
        for(size_t i = 0; i < size; ++i)
        {
            if(staging[i] != static_cast<uint8_t>((seed + i) % 256)) return false;
        }
        return true;
    }

    /** Fill device `buf` with a constant byte (sentinel). */
    void FillSentinel(void* buf, size_t size, uint8_t value)
    {
        (void)hipMemset(buf, value, size);
        (void)hipDeviceSynchronize();
    }

    /** Return true iff every byte in device `buf` equals `value`. */
    bool AllSentinel(const void* buf, size_t size, uint8_t value)
    {
        std::vector<uint8_t> staging(size);
        if(hipMemcpy(staging.data(), buf, size, hipMemcpyDeviceToHost) != hipSuccess)
            return false;
        (void)hipDeviceSynchronize();
        for(size_t i = 0; i < size; ++i)
        {
            if(staging[i] != value) return false;
        }
        return true;
    }

    /** Read a 64-bit value from device `signalBuf` at byte offset
     *  `signalOff`. Returns 0 on copy failure. */
    uint64_t ReadSignal(const void* signalBuf, size_t signalOff = 0)
    {
        uint64_t v = 0;
        const auto* p = static_cast<const uint8_t*>(signalBuf) + signalOff;
        if(hipMemcpy(&v, p, sizeof(v), hipMemcpyDeviceToHost) != hipSuccess)
            return 0;
        (void)hipDeviceSynchronize();
        return v;
    }
};

// ---------------------------------------------------------------------------
// Parameterized fixture — sweep nContexts × messageSize.
//
// Used by the 8 tests where both axes are meaningful. The 2 tests
// where the size is fixed by the test point itself (zero-size) or
// irrelevant (invalid-op) use GinMPIFixedSizeTest instead so they
// don't produce redundant size variants.
// ---------------------------------------------------------------------------
class GinMPITest : public GinMPITestBase,
                   public ::testing::WithParamInterface<std::tuple<int, size_t, bool>>
{
protected:
    int    NumContexts() const { return std::get<0>(GetParam()); }
    size_t MessageSize() const { return std::get<1>(GetParam()); }
    bool   IsDmaBuf()    const { return std::get<2>(GetParam()); }
    int    GetNumContexts() const override { return NumContexts(); }
    bool   UseDmaBuf()      const override { return IsDmaBuf(); }
};

// ---------------------------------------------------------------------------
// Parameterized fixture — sweep nContexts only.
//
// Used by IPutSignalZeroSize (size IS the test point) and
// IPutSignalInvalidSignalOp (size is irrelevant to the assertion).
// ---------------------------------------------------------------------------
class GinMPIFixedSizeTest : public GinMPITestBase,
                            public ::testing::WithParamInterface<std::tuple<int, bool>>
{
protected:
    int  NumContexts() const { return std::get<0>(GetParam()); }
    bool IsDmaBuf()    const { return std::get<1>(GetParam()); }
    int  GetNumContexts() const override { return NumContexts(); }
    bool UseDmaBuf()      const override { return IsDmaBuf(); }
};

// ---------------------------------------------------------------------------
// Non-parameterized fixture — used by long-running stress tests that have
// their own internal "iterations" dimension and need predictable runtime.
// Single context, fixed config; the iteration count IS the stress axis.
// Filterable as a group via `--gtest_filter='*Stress*'`.
// ---------------------------------------------------------------------------
class GinMPIStressTest : public GinMPITestBase
{
protected:
    int GetNumContexts() const override { return 1; }
};

} // namespace RCCLGinTests

#endif // RCCL_HAS_GIN_IB_PROXY
#endif // MPI_TESTS_ENABLED
