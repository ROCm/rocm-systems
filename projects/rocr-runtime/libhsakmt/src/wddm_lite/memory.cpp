/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * Memory management for wddm_lite backend.
 * Replaces stubs with real implementations that send escape commands
 * to the custom WDDM driver for GPU memory allocation and mapping.
 */

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"
#include <string.h>

extern struct WddmLiteDevice g_wddm_lite_dev;

#define WDDM_LITE_HANDLE_MARKER 0xDEAD000000000000ULL

/*
 * Decode a memory handle. Handles from VRAM-only allocations are
 * encoded as (MARKER | slot_index). System memory handles are
 * the CPU virtual address directly.
 */
static inline ULONGLONG decode_mem_handle(void *addr)
{
    ULONGLONG val = (ULONGLONG)(uintptr_t)addr;
    if ((val & 0xFFFF000000000000ULL) == WDDM_LITE_HANDLE_MARKER)
        return val & 0x0000FFFFFFFFFFFFULL;
    return val;
}

/*
 * Convert HsaMemFlags to AMDGPU_MEM_FLAG_* bitfield.
 */
static ULONG mem_flags_to_escape(HsaMemFlags flags)
{
    ULONG f = 0;

    if (flags.ui32.NonPaged)
        f |= AMDGPU_MEM_FLAG_NONPAGED;
    if (flags.ui32.ReadOnly)
        f |= AMDGPU_MEM_FLAG_READONLY;
    if (flags.ui32.HostAccess)
        f |= AMDGPU_MEM_FLAG_HOST_ACCESS;
    if (flags.ui32.NoSubstitute)
        f |= AMDGPU_MEM_FLAG_NO_SUBSTITUTE;
    if (flags.ui32.GDSMemory)
        f |= AMDGPU_MEM_FLAG_GDS;
    if (flags.ui32.Scratch)
        f |= AMDGPU_MEM_FLAG_SCRATCH;
    if (flags.ui32.ExecuteAccess)
        f |= AMDGPU_MEM_FLAG_EXECUTABLE;
    if (flags.ui32.AQLQueueMemory)
        f |= AMDGPU_MEM_FLAG_AQL_QUEUE;
    if (flags.ui32.Uncached)
        f |= AMDGPU_MEM_FLAG_UNCACHED;
    if (flags.ui32.Contiguous)
        f |= AMDGPU_MEM_FLAG_CONTIGUOUS;
    if (flags.ui32.CoarseGrain)
        f |= AMDGPU_MEM_FLAG_COHERENT;

    return f;
}

/*
 * Determine the memory type from flags and node ID.
 * Node 0 = CPU (system memory), Node 1+ = GPU.
 */
static ULONG mem_type_from_flags(HSAuint32 node, HsaMemFlags flags)
{
    if (flags.ui32.Scratch || flags.ui32.GDSMemory)
        return AMDGPU_MEM_TYPE_VRAM;

    /* Node 0 = CPU → system/GTT memory */
    if (node == 0)
        return AMDGPU_MEM_TYPE_SYSTEM;

    /* GPU node: VRAM unless explicitly host-accessible without NonPaged */
    if (flags.ui32.HostAccess && !flags.ui32.NonPaged)
        return AMDGPU_MEM_TYPE_GTT;

    return AMDGPU_MEM_TYPE_VRAM;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAllocMemory(HSAuint32 Node, HSAuint64 SizeInBytes,
                   HsaMemFlags MemFlags, void **MemoryAddress)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    data.Header.Size = sizeof(data);
    data.GpuId = Node;
    data.SizeInBytes = SizeInBytes;
    data.Flags = mem_type_from_flags(Node, MemFlags) |
                 mem_flags_to_escape(MemFlags);

    if (MemFlags.ui32.FixedAddress && *MemoryAddress)
        data.VaAddress = (ULONGLONG)*MemoryAddress;

    fprintf(stderr, "hsaKmtAllocMemory: node=%u size=0x%llx flags=0x%x sizeof(data)=%u\n",
            Node, (unsigned long long)SizeInBytes, (unsigned)data.Flags, (unsigned)sizeof(data));
    fflush(stderr);

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0) {
        pr_err("hsaKmtAllocMemory: escape failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    if (data.Header.Status != STATUS_SUCCESS) {
        pr_err("hsaKmtAllocMemory: driver returned 0x%lx\n",
               (unsigned long)data.Header.Status);
        return HSAKMT_STATUS_NO_MEMORY;
    }

    /*
     * Use CpuAddress if available (system memory with user mapping),
     * otherwise encode the driver handle as a fake pointer for tracking.
     */
    if (data.CpuAddress)
        *MemoryAddress = data.CpuAddress;
    else
        *MemoryAddress = (void *)(uintptr_t)(0xDEAD000000000000ULL | data.Handle);

    fprintf(stderr, "hsaKmtAllocMemory: OK handle=%llu cpu=%p gpu=0x%llx -> %p\n",
            (unsigned long long)data.Handle, data.CpuAddress,
            (unsigned long long)data.GpuAddress, *MemoryAddress);
    fflush(stderr);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAllocMemoryAlign(HSAuint32 PreferredNode, HSAuint64 SizeInBytes,
                        HSAuint64 Alignment, HsaMemFlags MemFlags,
                        void **MemoryAddress)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    data.Header.Size = sizeof(data);
    data.GpuId = PreferredNode;
    data.SizeInBytes = SizeInBytes;
    data.Alignment = Alignment;
    data.Flags = mem_type_from_flags(PreferredNode, MemFlags) |
                 mem_flags_to_escape(MemFlags);

    if (MemFlags.ui32.FixedAddress && *MemoryAddress)
        data.VaAddress = (ULONGLONG)*MemoryAddress;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0) {
        pr_err("hsaKmtAllocMemoryAlign: escape failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_NO_MEMORY;

    if (data.CpuAddress)
        *MemoryAddress = data.CpuAddress;
    else
        *MemoryAddress = (void *)(uintptr_t)(0xDEAD000000000000ULL | data.Handle);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtFreeMemory(void *MemoryAddress, HSAuint64 SizeInBytes)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_FREE_MEMORY_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_FREE_MEMORY;
    data.Header.Size = sizeof(data);
    data.Handle = decode_mem_handle(MemoryAddress);

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMapMemoryToGPU(void *MemoryAddress, HSAuint64 MemorySize,
                      HSAuint64 *AlternateVAGPU)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /*
     * No-op: GPU page table management is not implemented yet.
     * System memory is already CPU-accessible, and VRAM-only allocs
     * return virtual addresses. Return the CPU address as the GPU address.
     */
    pr_info("hsaKmtMapMemoryToGPU: no-op for %p size=0x%llx\n",
            MemoryAddress, (unsigned long long)MemorySize);

    if (AlternateVAGPU)
        *AlternateVAGPU = (HSAuint64)MemoryAddress;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtMapMemoryToGPUNodes(void *MemoryAddress, HSAuint64 MemorySize,
                           HSAuint64 *AlternateVAGPU,
                           HsaMemMapFlags MapFlags,
                           HSAuint64 NumberOfNodes,
                           HSAuint32 *NodeArray)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /*
     * No-op: GPU page table management is not implemented yet.
     * Return the CPU address as the GPU address.
     */
    pr_info("hsaKmtMapMemoryToGPUNodes: no-op for %p size=0x%llx nodes=%llu\n",
            MemoryAddress, (unsigned long long)MemorySize,
            (unsigned long long)NumberOfNodes);

    if (AlternateVAGPU)
        *AlternateVAGPU = (HSAuint64)MemoryAddress;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtUnmapMemoryToGPU(void *MemoryAddress)
{
    CHECK_KFD_OPEN();

    if (!MemoryAddress)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /* No-op: GPU page table management is not implemented yet. */
    pr_info("hsaKmtUnmapMemoryToGPU: no-op for %p\n", MemoryAddress);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetMemoryPolicy(HSAuint32 Node, HSAuint32 DefaultPolicy,
                       HSAuint32 AlternatePolicy,
                       void *MemoryAddressAlternate,
                       HSAuint64 MemorySizeInBytes)
{
    CHECK_KFD_OPEN();

    AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_SET_MEMORY_POLICY;
    data.Header.Size = sizeof(data);
    data.GpuId = Node;
    data.DefaultPolicy = DefaultPolicy;
    data.AlternatePolicy = AlternatePolicy;
    data.AlternateApertureBase = (ULONGLONG)MemoryAddressAlternate;
    data.AlternateApertureSize = MemorySizeInBytes;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAvailableMemory(HSAuint32 Node, HSAuint64 *AvailableBytes)
{
    CHECK_KFD_OPEN();

    if (!AvailableBytes)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /* For GPU nodes, report VRAM size from device info.
     * For CPU node, report system memory. */
    if (Node == 0) {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        GlobalMemoryStatusEx(&mem);
        *AvailableBytes = mem.ullAvailPhys;
    } else {
        *AvailableBytes = g_wddm_lite_dev.vram_size;
    }

    return HSAKMT_STATUS_SUCCESS;
}

/*
 * RegisterMemory / DeregisterMemory - On WDDM these are no-ops since
 * the driver handles page pinning internally via escape commands.
 */
HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemory(void *MemoryAddress, HSAuint64 MemorySize)
{
    CHECK_KFD_OPEN();
    /* No-op on WDDM: memory is managed by driver */
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemoryToNodes(void *MemoryAddress, HSAuint64 MemorySize,
                             HSAuint64 NumberOfNodes,
                             HSAuint32 *NodeArray)
{
    CHECK_KFD_OPEN();
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtRegisterMemoryWithFlags(void *MemoryAddress, HSAuint64 MemorySize,
                               HsaMemFlags MemFlags)
{
    CHECK_KFD_OPEN();
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDeregisterMemory(void *MemoryAddress)
{
    CHECK_KFD_OPEN();
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtQueryPointerInfo(const void *Pointer,
                        HsaPointerInfo *PointerInfo)
{
    CHECK_KFD_OPEN();
    /* TODO: implement pointer tracking if needed by ROCR */
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetMemoryUserData(const void *Pointer, void *UserData)
{
    CHECK_KFD_OPEN();
    return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetClockCounters(HSAuint32 NodeId, HsaClockCounters *Counters)
{
    CHECK_KFD_OPEN();

    if (!Counters)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_GET_CLOCK_COUNTERS;
    data.Header.Size = sizeof(data);
    data.GpuId = NodeId;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    Counters->GPUClockCounter = data.GpuClockCounter;
    Counters->CPUClockCounter = data.CpuClockCounter;
    Counters->SystemClockCounter = data.SystemClockCounter;
    Counters->SystemClockFrequencyHz = data.SystemClockFrequencyHz;

    return HSAKMT_STATUS_SUCCESS;
}
