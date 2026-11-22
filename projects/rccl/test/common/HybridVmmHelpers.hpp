/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef HYBRID_VMM_HELPERS_HPP
#define HYBRID_VMM_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "MPIHelpers.hpp"
#include "MultiSegmentVmmHelpers.hpp"
#include "ipcsocket.h"
#include "rocmwrap.h"

namespace RCCLHybridVmmTests
{

constexpr size_t kHybridVmmAlignment = 2u * 1024 * 1024;

// Hybrid VMM layout [GPU][CPU local-rank 0]...[CPU local-rank N-1]; every local rank maps the same host handles.
struct HybridVmmBuffer
{
    void*                                        ptr       = nullptr;
    hipDeviceptr_t                               base      = 0;
    size_t                                       totalSize = 0;
    size_t                                       gpuBytes  = 0;
    int                                          localRank = -1;
    int                                          localSize = 0;
    int                                          exportFd  = -1;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<size_t>                          segSizes;
    // Owner's DMA-BUF fd per segment (-1 for GPU and local); an imported handle cannot re-export one.
    std::vector<int>                             segFds;

    bool valid() const
    {
        return ptr != nullptr && handles.size() == static_cast<size_t>(localSize + 1);
    }
};

// True when a host VMM handle imported from its own fd maps with device and host access.
inline bool ProbeImportedHostCpuAccess(int dev)
{
    bool supported = false;
    hipMemAllocationProp prop = RCCLTestHelpers::MakeVmmProp(hipMemLocationTypeHost, 0, /*gpuDirectRdma=*/false);
    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) != hipSuccess ||
        granularity == 0)
    {
        return false;
    }
    const size_t bytes = ((kHybridVmmAlignment + granularity - 1) / granularity) * granularity;

    hipMemGenericAllocationHandle_t original = 0;
    hipMemGenericAllocationHandle_t imported = 0;
    int fd = -1;
    if (hipMemCreate(&original, bytes, &prop, 0) == hipSuccess &&
        hipMemExportToShareableHandle(&fd, original, hipMemHandleTypePosixFileDescriptor, 0) == hipSuccess &&
        hipMemImportFromShareableHandle(&imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
                                        hipMemHandleTypePosixFileDescriptor) == hipSuccess)
    {
        RCCLTestHelpers::MultiSegmentVmmBuffer mapped;
        supported = RCCLTestHelpers::AllocSegmentedVmm(
            dev, {{bytes, hipMemLocationTypeHost, imported}}, &mapped, 0, /*requireHostAccess=*/true);
        RCCLTestHelpers::FreeMultiSegmentVmm(mapped);
    }
    else if (imported != 0)
    {
        (void)hipMemRelease(imported);
    }
    if (fd >= 0) (void)close(fd);
    if (original != 0) (void)hipMemRelease(original);
    return supported;
}

enum class HybridVmmStatus { Ok, Skip, Fail };

// RCCL_TEST_HYBRID_VMM_REQUIRE=1 (hybrid CI job) fails instead of skipping on a missing ROCM-29812 fix, host gate,
// or hybrid allocation failure.
inline HybridVmmStatus HybridVmmUnsupportedStatus()
{
    const char* require = std::getenv("RCCL_TEST_HYBRID_VMM_REQUIRE");
    return (require && std::string(require) == "1") ? HybridVmmStatus::Fail : HybridVmmStatus::Skip;
}

// Collective: every rank must call it. Skips on ROCM-29812. Imported handles are not re-exported.
// Hybrid elastic windows require NCCL_SYM_REUSE_SYSMEM_HANDLES=1 at registration time.
inline bool CheckHybridVmmRuntimeSupport(int dev, std::string* reason = nullptr)
{
    bool cpuAccessSupported = MPIHelpers::allRanksTrue(ProbeImportedHostCpuAccess(dev));

    // TODO(ROCM-29812): Remove this skip gate when imported host VMM CPU access is fixed.
    if (!cpuAccessSupported)
    {
        if (reason) *reason = "ROCM-29812: imported host VMM CPU access is unsupported";
        return false;
    }
    return true;
}

inline void FreeHybridVmm(HybridVmmBuffer& b)
{
    RCCLTestHelpers::ReleaseMappedVmm(b.base, b.totalSize, b.handles, b.segSizes,
                                      /*uniformSegSize=*/0);
    if (b.exportFd >= 0) (void)close(b.exportFd);
    for (int fd : b.segFds)
    {
        if (fd >= 0) (void)close(fd);
    }
    b = HybridVmmBuffer{};
}

// The hybrid reason for a failed segment map; GPU-segment allocation and map failures carry none.
inline void SetHybridMapFailureReason(const RCCLTestHelpers::VmmSegFailure& f, std::string* reason)
{
    using RCCLTestHelpers::VmmSegStep;
    if (reason == nullptr) return;
    const std::string err = hipGetErrorString(f.err);
    if (f.step == VmmSegStep::Map && f.segment > 0)
    {
        *reason = "hipMemMap(imported host segment): " + err;
    }
    else if (f.step == VmmSegStep::DeviceAccess || f.step == VmmSegStep::HostAccess)
    {
        const char* prefix = f.step == VmmSegStep::HostAccess ? "hipMemSetAccess(hybrid host segment "
                             : f.segment > 0                  ? "hipMemSetAccess(hybrid imported host segment "
                                                              : "hipMemSetAccess(hybrid GPU segment ";
        *reason = prefix + std::to_string(f.segment) + "): " + err;
    }
}

// Collective within the node: each local rank exports one host allocation and imports its peers' over SCM_RIGHTS.
// Returns false on every local rank when host VMM is unsupported.
inline bool AllocHybridVmm(int dev, size_t gpuBytes, size_t localCpuBytes,
                           HybridVmmBuffer* out, std::string* reason = nullptr)
{
    if (out == nullptr || gpuBytes == 0 || localCpuBytes == 0 ||
        gpuBytes % kHybridVmmAlignment != 0 || localCpuBytes % kHybridVmmAlignment != 0)
    {
        if (reason) *reason = "hybrid sizes must be non-zero and 2 MiB aligned";
        return false;
    }
    *out = HybridVmmBuffer{};

    MPI_Comm localComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm) != MPI_SUCCESS)
    {
        if (reason) *reason = "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) failed";
        return false;
    }

    int localRank = 0;
    int localSize = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    hipMemAllocationProp gpuProp =
        RCCLTestHelpers::MakeVmmProp(hipMemLocationTypeDevice, dev, /*gpuDirectRdma=*/true);
    hipMemAllocationProp cpuProp = RCCLTestHelpers::MakeVmmProp(hipMemLocationTypeHost, 0, /*gpuDirectRdma=*/false);

    hipMemGenericAllocationHandle_t localCpuHandle = 0;
    int exportFd = -1;
    // Every local rank takes the same failure exit: free what this rank holds and report `why`.
    auto fail = [&](const char* why) {
        if (exportFd >= 0) (void)close(exportFd);
        if (localCpuHandle != 0) (void)hipMemRelease(localCpuHandle);
        if (reason) *reason = why;
        MPI_Comm_free(&localComm);
        return false;
    };

    size_t gpuGran = 0;
    size_t cpuGran = 0;
    bool localOk =
        hipMemGetAllocationGranularity(&gpuGran, &gpuProp, hipMemAllocationGranularityMinimum) == hipSuccess &&
        hipMemGetAllocationGranularity(&cpuGran, &cpuProp, hipMemAllocationGranularityMinimum) == hipSuccess &&
        gpuGran != 0 && cpuGran != 0 &&
        kHybridVmmAlignment % gpuGran == 0 && kHybridVmmAlignment % cpuGran == 0;
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
        return fail("GPU/host VMM allocation is unavailable or incompatible with 2 MiB alignment");

    localOk = hipMemCreate(&localCpuHandle, localCpuBytes, &cpuProp, 0) == hipSuccess &&
              hipMemExportToShareableHandle(&exportFd, localCpuHandle,
                                            hipMemHandleTypePosixFileDescriptor, 0) == hipSuccess;
    if (!MPIHelpers::allRanksTrue(localOk, localComm)) return fail("host VMM allocation/export is unavailable");

    uint64_t localBytes = static_cast<uint64_t>(localCpuBytes);
    std::vector<uint64_t> cpuBytes(static_cast<size_t>(localSize), 0);
    MPI_Allgather(&localBytes, 1, MPI_UINT64_T,
                  cpuBytes.data(), 1, MPI_UINT64_T, localComm);

    // The IPC sockets are abstract (no filesystem permissions), so the name must not be predictable.
    uint64_t socketHash = 0;
    if (localRank == 0)
    {
        std::random_device entropy;
        socketHash = (static_cast<uint64_t>(entropy()) << 32) ^ entropy() ^
                     static_cast<uint64_t>(getpid());
    }
    MPI_Bcast(&socketHash, 1, MPI_UINT64_T, 0, localComm);

    volatile uint32_t abortFlag = 0;
    ncclIpcSocket ipcSocket = {};
    localOk = ncclIpcSocketInit(
        &ipcSocket, localRank, socketHash, &abortFlag) == ncclSuccess;
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        if (localOk) (void)ncclIpcSocketClose(&ipcSocket);
        return fail("RCCL IPC socket initialization failed");
    }

    MPI_Barrier(localComm); // all destination sockets must be bound before send
    for (int peer = 0; localOk && peer < localSize; ++peer)
    {
        if (peer == localRank) continue;
        int sender = localRank;
        localOk = ncclIpcSocketSendMsg(
            &ipcSocket, &sender, sizeof(sender), exportFd,
            peer, socketHash) == ncclSuccess;
    }
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        abortFlag = 1;
        (void)ncclIpcSocketClose(&ipcSocket);
        return fail("RCCL IPC socket FD send failed");
    }

    std::vector<MPIHelpers::FileDescriptor> peerFds(
        static_cast<size_t>(localSize));
    for (int received = 0; localOk && received < localSize - 1; ++received)
    {
        int sender = -1;
        int receivedFd = -1;
        localOk = ncclIpcSocketRecvMsg(
            &ipcSocket, &sender, sizeof(sender), &receivedFd) == ncclSuccess &&
            sender >= 0 && sender < localSize && sender != localRank &&
            !peerFds[static_cast<size_t>(sender)].is_valid();
        if (localOk)
        {
            peerFds[static_cast<size_t>(sender)] = MPIHelpers::FileDescriptor(receivedFd);
        }
        else if (receivedFd >= 0)
        {
            (void)close(receivedFd);
        }
    }
    (void)ncclIpcSocketClose(&ipcSocket);
    if (!MPIHelpers::allRanksTrue(localOk, localComm)) return fail("RCCL IPC socket FD receive failed");

    HybridVmmBuffer tmp;
    tmp.gpuBytes  = gpuBytes;
    tmp.localRank = localRank;
    tmp.localSize = localSize;
    tmp.exportFd  = exportFd;
    tmp.segFds.push_back(-1);

    // [GPU][CPU local-rank 0..N-1]: this rank's own host handle, then each peer's imported one in rank order.
    std::vector<RCCLTestHelpers::VmmSegSpec> specs = {{gpuBytes}};
    for (int i = 0; localOk && i < localSize; ++i)
    {
        const bool isLocal = i == localRank;
        const int  fd      = isLocal ? exportFd : peerFds[static_cast<size_t>(i)].get();
        hipMemGenericAllocationHandle_t handle = isLocal ? localCpuHandle : 0;
        if (fd < 0 || (isLocal && handle == 0))
        {
            localOk = false;
            break;
        }
        const hipError_t importResult = isLocal ? hipSuccess
            : hipMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
                                              hipMemHandleTypePosixFileDescriptor);
        if (importResult != hipSuccess)
        {
            if (reason) *reason = std::string("hipMemImportFromShareableHandle: ") + hipGetErrorString(importResult);
            if (handle != 0) (void)hipMemRelease(handle);
            localOk = false;
            break;
        }
        specs.push_back({static_cast<size_t>(cpuBytes[static_cast<size_t>(i)]), hipMemLocationTypeHost, handle});
        if (isLocal) localCpuHandle = 0; // ownership transferred to specs
        tmp.segFds.push_back(isLocal ? -1 : dup(fd));
    }

    // Device access on every segment; the host segments also need host access.
    RCCLTestHelpers::MultiSegmentVmmBuffer mapped;
    RCCLTestHelpers::VmmSegFailure failure;
    if (localOk)
    {
        localOk = RCCLTestHelpers::AllocSegmentedVmm(dev, specs, &mapped, kHybridVmmAlignment,
                                                     /*requireHostAccess=*/true, &failure);
        if (!localOk) SetHybridMapFailureReason(failure, reason);
    }
    else
    {
        for (size_t s = 1; s < specs.size(); ++s) (void)hipMemRelease(specs[s].handle);
    }
    if (localCpuHandle != 0) (void)hipMemRelease(localCpuHandle);
    tmp.base      = mapped.base;
    tmp.totalSize = mapped.totalSize;
    tmp.handles   = std::move(mapped.handles);
    tmp.segSizes  = std::move(mapped.segSizes);

    bool allOk = MPIHelpers::allRanksTrue(localOk, localComm);
    MPI_Barrier(localComm); // every peer has imported before SCM_RIGHTS FDs close
    MPI_Comm_free(&localComm);

    if (!allOk)
    {
        FreeHybridVmm(tmp);
        if (reason && reason->empty())
            *reason = "hybrid host-handle import/map failed on another rank";
        return false;
    }

    tmp.ptr = reinterpret_cast<void*>(tmp.base);
    *out = std::move(tmp);
    return true;
}

// Collective over MPI_COMM_WORLD: support check, then allocation. A failure on any rank frees *out on every rank.
inline HybridVmmStatus AllocHybridCollective(size_t gpuBytes, size_t localCpuBytes, int expectedLocalRanks,
                                             HybridVmmBuffer* out, std::string* reason)
{
    *out = HybridVmmBuffer{};
#if !NCCL_CUMEM_HOST_GATE
    // Compile-time gate, so every rank returns here without a collective.
    (void)gpuBytes; (void)localCpuBytes; (void)expectedLocalRanks;
    *reason = "host VMM is outside this build's NCCL_CUMEM_HOST_GATE window";
    return HybridVmmUnsupportedStatus();
#else
    int dev = 0;
    // The support check runs collectives; call it even if hipGetDevice failed.
    const bool haveDev = hipGetDevice(&dev) == hipSuccess;
    bool supported = CheckHybridVmmRuntimeSupport(dev, reason) && haveDev;
    if (!haveDev && reason->empty()) *reason = "hipGetDevice failed";
    // Every rank takes this branch together, so the REQUIRE-aware status is the same on all ranks.
    if (!MPIHelpers::allRanksTrue(supported))
    {
        if (reason->empty()) *reason = "hybrid VMM runtime support is unavailable on another rank";
        return HybridVmmUnsupportedStatus();
    }
    bool allocated = AllocHybridVmm(dev, gpuBytes, localCpuBytes, out, reason);
    if (allocated && out->localSize != expectedLocalRanks)
    {
        *reason = "need " + std::to_string(expectedLocalRanks) + " ranks per node, got " +
                  std::to_string(out->localSize);
        allocated = false;
    }
    if (!MPIHelpers::allRanksTrue(allocated))
    {
        FreeHybridVmm(*out);
        if (reason->empty()) *reason = "hybrid VMM allocation failed on another rank";
        return HybridVmmUnsupportedStatus();
    }
    return HybridVmmStatus::Ok;
#endif
}

// Fails or skips the calling test for a non-Ok AllocHybridCollective status; all ranks see the same status.
#define HYBRID_VMM_SKIP_OR_FAIL(status, reason)                                   \
    do {                                                                          \
        const RCCLHybridVmmTests::HybridVmmStatus hybridStatus_ = (status);       \
        if (hybridStatus_ == RCCLHybridVmmTests::HybridVmmStatus::Fail) {         \
            FAIL() << "RCCL_TEST_HYBRID_VMM_REQUIRE=1: " << (reason);             \
        }                                                                         \
        if (hybridStatus_ != RCCLHybridVmmTests::HybridVmmStatus::Ok) {           \
            GTEST_SKIP() << "DeepEP hybrid allocation unavailable: " << (reason); \
        }                                                                         \
    } while (0)

} // namespace RCCLHybridVmmTests

#endif // MPI_TESTS_ENABLED

#endif // HYBRID_VMM_HELPERS_HPP
