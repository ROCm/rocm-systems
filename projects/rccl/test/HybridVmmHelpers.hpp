/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef HYBRID_VMM_HELPERS_HPP
#define HYBRID_VMM_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "MPIHelpers.hpp"
#include "ipcsocket.h"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <unistd.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace RCCLHybridVmmTests
{

constexpr size_t kHybridVmmAlignment = 2u * 1024 * 1024;

// Hybrid device/host VMM shape:
// [GPU][CPU local-rank 0]...[CPU local-rank N-1].
// Every process on a node imports the same CPU allocation handles in local-rank
// order before registering the complete VA as one symmetric window.
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

    bool valid() const
    {
        return ptr != nullptr && handles.size() == static_cast<size_t>(localSize + 1);
    }
};

struct HybridVmmRuntimeSupport
{
    bool importedHostCpuAccess = false;
};

inline HybridVmmRuntimeSupport ProbeHybridVmmRuntimeSupport(int dev)
{
    HybridVmmRuntimeSupport support;
    hipMemGenericAllocationHandle_t original = 0;
    hipMemGenericAllocationHandle_t imported = 0;
    hipDeviceptr_t va = 0;
    size_t bytes = 0;
    int fd = -1;
    bool mapped = false;
    hipMemAccessDesc access = {};

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeHost;
    prop.location.id = 0;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(
            &granularity, &prop, hipMemAllocationGranularityMinimum) != hipSuccess ||
        granularity == 0)
        goto cleanup;
    bytes = ((kHybridVmmAlignment + granularity - 1) / granularity) * granularity;

    if (hipMemCreate(&original, bytes, &prop, 0) != hipSuccess ||
        hipMemExportToShareableHandle(
            &fd, original, hipMemHandleTypePosixFileDescriptor, 0) != hipSuccess ||
        hipMemImportFromShareableHandle(
            &imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
            hipMemHandleTypePosixFileDescriptor) != hipSuccess)
        goto cleanup;

    if (hipMemAddressReserve(&va, bytes, 0, 0, 0) != hipSuccess ||
        hipMemMap(va, bytes, 0, imported, 0) != hipSuccess)
        goto cleanup;
    mapped = true;

    access.location.type = hipMemLocationTypeDevice;
    access.location.id = dev;
    access.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(va, bytes, &access, 1) != hipSuccess)
        goto cleanup;

    access.location.type = hipMemLocationTypeHost;
    access.location.id = 0;
    support.importedHostCpuAccess =
        hipMemSetAccess(va, bytes, &access, 1) == hipSuccess;

cleanup:
    if (mapped) (void)hipMemUnmap(va, bytes);
    if (va != 0) (void)hipMemAddressFree(va, bytes);
    if (imported != 0) (void)hipMemRelease(imported);
    if (fd >= 0) (void)close(fd);
    if (original != 0) (void)hipMemRelease(original);
    return support;
}

// Hybrid elastic windows require NCCL_SYM_REUSE_SYSMEM_HANDLES=1 at registration
// time (DeepEP / NCCL contract).
inline bool CheckHybridVmmRuntimeSupport(int dev, std::string* reason = nullptr)
{
    HybridVmmRuntimeSupport local = ProbeHybridVmmRuntimeSupport(dev);
    bool cpuAccessSupported = MPIHelpers::allRanksTrue(local.importedHostCpuAccess);

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
    size_t offset = 0;
    for (size_t i = 0; i < b.handles.size(); ++i)
    {
        if (b.base != 0 && i < b.segSizes.size())
        {
            hipDeviceptr_t va = reinterpret_cast<hipDeviceptr_t>(
                reinterpret_cast<uintptr_t>(b.base) + offset);
            (void)hipMemUnmap(va, b.segSizes[i]);
            offset += b.segSizes[i];
        }
        if (b.handles[i] != 0)
            (void)hipMemRelease(b.handles[i]);
    }
    if (b.exportFd >= 0)
        (void)close(b.exportFd);
    if (b.base != 0 && b.totalSize != 0)
        (void)hipMemAddressFree(b.base, b.totalSize);
    b = HybridVmmBuffer{};
}

// Create a workload-independent hybrid VMM allocation. Each local rank creates
// and exports one host allocation, all local descriptors are exchanged, then
// each rank imports all host handles into the same ordered VA layout. Returns
// false collectively within the node on unsupported kernels or host VMM
// runtimes. File descriptors are transferred with RCCL's existing SCM_RIGHTS
// IPC socket API rather than passing process-local descriptor integers.
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

    hipMemAllocationProp gpuProp            = {};
    gpuProp.type                            = hipMemAllocationTypePinned;
    gpuProp.location.type                   = hipMemLocationTypeDevice;
    gpuProp.location.id                     = dev;
    gpuProp.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    gpuProp.allocFlags.gpuDirectRDMACapable = 1;

    hipMemAllocationProp cpuProp = {};
    cpuProp.type                = hipMemAllocationTypePinned;
    cpuProp.location.type       = hipMemLocationTypeHost;
    cpuProp.location.id         = 0;
    cpuProp.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

    size_t gpuGran = 0;
    size_t cpuGran = 0;
    bool localOk =
        hipMemGetAllocationGranularity(&gpuGran, &gpuProp, hipMemAllocationGranularityMinimum) == hipSuccess &&
        hipMemGetAllocationGranularity(&cpuGran, &cpuProp, hipMemAllocationGranularityMinimum) == hipSuccess &&
        gpuGran != 0 && cpuGran != 0 &&
        kHybridVmmAlignment % gpuGran == 0 && kHybridVmmAlignment % cpuGran == 0;
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        if (reason) *reason = "GPU/host VMM allocation is unavailable or incompatible with 2 MiB alignment";
        MPI_Comm_free(&localComm);
        return false;
    }

    hipMemGenericAllocationHandle_t localCpuHandle = 0;
    int exportFd = -1;
    localOk = hipMemCreate(&localCpuHandle, localCpuBytes, &cpuProp, 0) == hipSuccess;
    if (localOk)
    {
        localOk = hipMemExportToShareableHandle(&exportFd, localCpuHandle,
                                                hipMemHandleTypePosixFileDescriptor, 0) == hipSuccess;
    }
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        if (exportFd >= 0) (void)close(exportFd);
        if (localCpuHandle != 0) (void)hipMemRelease(localCpuHandle);
        if (reason) *reason = "host VMM allocation/export is unavailable";
        MPI_Comm_free(&localComm);
        return false;
    }

    uint64_t localBytes = static_cast<uint64_t>(localCpuBytes);
    std::vector<uint64_t> cpuBytes(static_cast<size_t>(localSize), 0);
    MPI_Allgather(&localBytes, 1, MPI_UINT64_T,
                  cpuBytes.data(), 1, MPI_UINT64_T, localComm);

    // Use RCCL's production IPC socket transport to duplicate descriptors with
    // SCM_RIGHTS. The hash only needs to be unique while these node-local
    // sockets are alive; the local leader's PID provides that scope.
    uint64_t socketHash = localRank == 0
        ? (static_cast<uint64_t>(getpid()) << 32) ^ UINT64_C(0x48594252)
        : 0;
    MPI_Bcast(&socketHash, 1, MPI_UINT64_T, 0, localComm);

    volatile uint32_t abortFlag = 0;
    ncclIpcSocket ipcSocket = {};
    localOk = ncclIpcSocketInit(
        &ipcSocket, localRank, socketHash, &abortFlag) == ncclSuccess;
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        if (localOk) (void)ncclIpcSocketClose(&ipcSocket);
        (void)close(exportFd);
        (void)hipMemRelease(localCpuHandle);
        if (reason) *reason = "RCCL IPC socket initialization failed";
        MPI_Comm_free(&localComm);
        return false;
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
        (void)close(exportFd);
        (void)hipMemRelease(localCpuHandle);
        if (reason) *reason = "RCCL IPC socket FD send failed";
        MPI_Comm_free(&localComm);
        return false;
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
            peerFds[static_cast<size_t>(sender)] =
                MPIHelpers::FileDescriptor(receivedFd);
        else if (receivedFd >= 0)
            (void)close(receivedFd);
    }
    (void)ncclIpcSocketClose(&ipcSocket);
    if (!MPIHelpers::allRanksTrue(localOk, localComm))
    {
        (void)close(exportFd);
        (void)hipMemRelease(localCpuHandle);
        if (reason) *reason = "RCCL IPC socket FD receive failed";
        MPI_Comm_free(&localComm);
        return false;
    }

    size_t totalSize = gpuBytes;
    for (uint64_t bytes : cpuBytes)
        totalSize += static_cast<size_t>(bytes);

    HybridVmmBuffer tmp;
    tmp.gpuBytes  = gpuBytes;
    tmp.totalSize = totalSize;
    tmp.localRank = localRank;
    tmp.localSize = localSize;
    tmp.exportFd  = exportFd;

    localOk = hipMemAddressReserve(&tmp.base, totalSize, kHybridVmmAlignment, 0, 0) == hipSuccess;
    if (localOk)
    {
        hipMemGenericAllocationHandle_t gpuHandle = 0;
        localOk = hipMemCreate(&gpuHandle, gpuBytes, &gpuProp, 0) == hipSuccess;
        if (localOk && hipMemMap(tmp.base, gpuBytes, 0, gpuHandle, 0) == hipSuccess)
        {
            tmp.handles.push_back(gpuHandle);
            tmp.segSizes.push_back(gpuBytes);
        }
        else
        {
            if (gpuHandle != 0) (void)hipMemRelease(gpuHandle);
            localOk = false;
        }
    }

    size_t offset = gpuBytes;
    for (int i = 0; localOk && i < localSize; ++i)
    {
        bool isLocal = i == localRank;
        int localFd = isLocal ? exportFd : peerFds[static_cast<size_t>(i)].get();
        if (localFd < 0 || (isLocal && localCpuHandle == 0))
        {
            localOk = false;
            break;
        }

        hipMemGenericAllocationHandle_t segmentHandle = localCpuHandle;
        hipError_t importResult = hipSuccess;
        if (!isLocal)
        {
            segmentHandle = 0;
            importResult = hipMemImportFromShareableHandle(
                &segmentHandle, reinterpret_cast<void*>(static_cast<intptr_t>(localFd)),
                hipMemHandleTypePosixFileDescriptor);
        }
        size_t bytes = static_cast<size_t>(cpuBytes[static_cast<size_t>(i)]);
        hipDeviceptr_t va = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<uintptr_t>(tmp.base) + offset);
        hipError_t mapResult = importResult == hipSuccess
            ? hipMemMap(va, bytes, 0, segmentHandle, 0)
            : hipErrorInvalidValue;
        if (importResult != hipSuccess || mapResult != hipSuccess)
        {
            if (reason)
                *reason = importResult != hipSuccess
                    ? std::string("hipMemImportFromShareableHandle: ") +
                          hipGetErrorString(importResult)
                    : std::string("hipMemMap(imported host segment): ") +
                          hipGetErrorString(mapResult);
            if (!isLocal && segmentHandle != 0) (void)hipMemRelease(segmentHandle);
            localOk = false;
            break;
        }
        tmp.handles.push_back(segmentHandle);
        if (isLocal) localCpuHandle = 0; // ownership transferred to tmp.handles
        tmp.segSizes.push_back(bytes);
        offset += bytes;
    }
    if (localCpuHandle != 0) (void)hipMemRelease(localCpuHandle);

    if (localOk)
    {
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id   = dev;
        access.flags         = hipMemAccessFlagsProtReadWrite;
        size_t accessOffset = 0;
        for (size_t segment = 0; segment < tmp.segSizes.size(); ++segment)
        {
            size_t bytes = tmp.segSizes[segment];
            hipDeviceptr_t segmentVa = reinterpret_cast<hipDeviceptr_t>(
                reinterpret_cast<uintptr_t>(tmp.base) + accessOffset);
            hipError_t accessResult =
                hipMemSetAccess(segmentVa, bytes, &access, 1);
            if (accessResult != hipSuccess)
            {
                if (reason)
                    *reason = std::string("hipMemSetAccess(hybrid segment ") +
                        std::to_string(segment) +
                        (segment == 0 ? " GPU): " : " imported host): ") +
                        hipGetErrorString(accessResult);
                localOk = false;
                break;
            }
            accessOffset += bytes;
        }
    }

    if (localOk)
    {
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeHost;
        access.location.id   = 0;
        access.flags         = hipMemAccessFlagsProtReadWrite;
        size_t accessOffset = gpuBytes;
        for (size_t segment = 1; segment < tmp.segSizes.size(); ++segment)
        {
            size_t bytes = tmp.segSizes[segment];
            hipDeviceptr_t segmentVa = reinterpret_cast<hipDeviceptr_t>(
                reinterpret_cast<uintptr_t>(tmp.base) + accessOffset);
            hipError_t accessResult =
                hipMemSetAccess(segmentVa, bytes, &access, 1);
            if (accessResult != hipSuccess)
            {
                if (reason)
                    *reason = std::string("hipMemSetAccess(hybrid host segment ") +
                        std::to_string(segment) + "): " +
                        hipGetErrorString(accessResult);
                localOk = false;
                break;
            }
            accessOffset += bytes;
        }
    }

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

} // namespace RCCLHybridVmmTests

#endif // MPI_TESTS_ENABLED

#endif // HYBRID_VMM_HELPERS_HPP
