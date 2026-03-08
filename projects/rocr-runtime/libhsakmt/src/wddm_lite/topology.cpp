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
 * Synthesized topology for the wddm_lite backend.
 * Same approach as amdgpu_lite: 2-node topology (CPU + GPU),
 * all data derived from the WDDM escape GET_INFO response.
 */

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"

#include <string.h>

extern struct WddmLiteDevice g_wddm_lite_dev;

#define WDDM_LITE_NUM_NODES  2   /* node 0 = CPU, node 1 = GPU */

static bool topology_acquired = false;

/*
 * GFX12 (RDNA4) hardware constants for RX 9070 XT / RX 9070.
 */
#define GFX12_NUM_SE            4
#define GFX12_CU_PER_SE         16
#define GFX12_SIMD_PER_CU       2
#define GFX12_WAVES_PER_SIMD    16
#define GFX12_MAX_ENGINE_MHZ    2970
#define GFX12_VGPR_PER_CU       (384 * 1024)  /* 384 KB */
#define GFX12_SGPR_PER_CU       (16 * 1024)   /* 16 KB */

static UINT get_cpu_core_count(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

static ULONGLONG get_system_memory_bytes(void)
{
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem))
        return mem.ullTotalPhys;
    return 0;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAcquireSystemProperties(HsaSystemProperties *SystemProperties)
{
    CHECK_KFD_OPEN();

    if (!SystemProperties)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    memset(SystemProperties, 0, sizeof(*SystemProperties));
    SystemProperties->NumNodes = WDDM_LITE_NUM_NODES;
    SystemProperties->PlatformOem = 0;
    SystemProperties->PlatformId = 0;
    SystemProperties->PlatformRev = 0;

    topology_acquired = true;
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtReleaseSystemProperties(void)
{
    topology_acquired = false;
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeProperties(HSAuint32 NodeId, HsaNodeProperties *NodeProperties)
{
    CHECK_KFD_OPEN();

    if (!NodeProperties)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    if (NodeId >= WDDM_LITE_NUM_NODES)
        return HSAKMT_STATUS_INVALID_NODE_UNIT;

    memset(NodeProperties, 0, sizeof(*NodeProperties));

    if (NodeId == 0) {
        /* CPU node */
        NodeProperties->NumCPUCores = get_cpu_core_count();
        NodeProperties->NumMemoryBanks = 1;
        NodeProperties->NumIOLinks = 1;
    } else {
        /* GPU node */
        struct WddmLiteDevice *dev = &g_wddm_lite_dev;

        NodeProperties->DeviceId = dev->device_id;
        NodeProperties->LocationId = 0; /* PCI location not easily available via D3DKMT */

        /* GFX version */
        NodeProperties->EngineId.ui32.Major = (dev->gfx_version >> 16) & 0xFF;
        NodeProperties->EngineId.ui32.Minor = (dev->gfx_version >> 8) & 0xFF;
        NodeProperties->EngineId.ui32.Stepping = dev->gfx_version & 0xFF;
        NodeProperties->EngineId.ui32.uCode = 1;

        /* Compute info */
        UINT total_cu = GFX12_NUM_SE * GFX12_CU_PER_SE;
        NodeProperties->NumFComputeCores = total_cu * GFX12_SIMD_PER_CU;
        NodeProperties->NumShaderBanks = GFX12_NUM_SE;
        NodeProperties->MaxWavesPerSIMD = GFX12_WAVES_PER_SIMD;
        NodeProperties->NumArrays = GFX12_NUM_SE;
        NodeProperties->NumCUPerArray = GFX12_CU_PER_SE;
        NodeProperties->NumSIMDPerCU = GFX12_SIMD_PER_CU;
        NodeProperties->MaxEngineClockMhzFCompute = GFX12_MAX_ENGINE_MHZ;

        /* VGPR/SGPR sizes */
        NodeProperties->VGPRSizePerCU = GFX12_VGPR_PER_CU;
        NodeProperties->SGPRSizePerCU = GFX12_SGPR_PER_CU;

        /* Firmware versions */
        NodeProperties->uCodeEngineVersions.uCodeSDMA = 1;

        /* Wavefront size */
        NodeProperties->WaveFrontSize = 32;

        /* Memory */
        NodeProperties->NumMemoryBanks = 2; /* VRAM + GTT */
        NodeProperties->NumIOLinks = 1;

        /* Mark as dGPU */
        NodeProperties->Capability.ui32.HSAMMUPresent = 0;
        NodeProperties->Capability.ui32.AQLQueueDoubleMap = 0;
    }

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeMemoryProperties(HSAuint32 NodeId, HSAuint32 NumBanks,
                              HsaMemoryProperties *MemoryProperties)
{
    CHECK_KFD_OPEN();

    if (!MemoryProperties)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    if (NodeId >= WDDM_LITE_NUM_NODES)
        return HSAKMT_STATUS_INVALID_NODE_UNIT;

    memset(MemoryProperties, 0, NumBanks * sizeof(*MemoryProperties));

    if (NodeId == 0) {
        /* CPU: system memory */
        if (NumBanks >= 1) {
            MemoryProperties[0].HeapType = HSA_HEAPTYPE_SYSTEM;
            MemoryProperties[0].SizeInBytes = get_system_memory_bytes();
            MemoryProperties[0].Flags.MemoryProperty = 0;
            MemoryProperties[0].Width = 64;
        }
    } else {
        /* GPU: VRAM + GTT */
        struct WddmLiteDevice *dev = &g_wddm_lite_dev;

        if (NumBanks >= 1) {
            if (dev->info.VisibleVramSizeBytes >= dev->vram_size) {
                MemoryProperties[0].HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC;
            } else {
                MemoryProperties[0].HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE;
            }
            MemoryProperties[0].SizeInBytes = dev->vram_size;
            MemoryProperties[0].Flags.MemoryProperty = 0;
            MemoryProperties[0].Width = 256;
        }
        if (NumBanks >= 2) {
            MemoryProperties[1].HeapType = HSA_HEAPTYPE_GPU_GDS;
            MemoryProperties[1].SizeInBytes = get_system_memory_bytes() / 2;
            MemoryProperties[1].Flags.MemoryProperty = 0;
            MemoryProperties[1].Width = 64;
        }
    }

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeCacheProperties(HSAuint32 NodeId,
                             HSAuint32 ProcessorIdLow,
                             HSAuint32 NumCaches,
                             HsaCacheProperties *CacheProperties)
{
    CHECK_KFD_OPEN();

    if (NodeId >= WDDM_LITE_NUM_NODES)
        return HSAKMT_STATUS_INVALID_NODE_UNIT;

    /* No cache info synthesized for now */
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId,
                              HSAuint32 NumIoLinks,
                              HsaIoLinkProperties *IoLinkProperties)
{
    CHECK_KFD_OPEN();

    if (!IoLinkProperties)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    if (NodeId >= WDDM_LITE_NUM_NODES)
        return HSAKMT_STATUS_INVALID_NODE_UNIT;

    memset(IoLinkProperties, 0, NumIoLinks * sizeof(*IoLinkProperties));

    if (NumIoLinks >= 1) {
        IoLinkProperties[0].IoLinkType = HSA_IOLINKTYPE_PCIEXPRESS;
        IoLinkProperties[0].NodeFrom = NodeId;
        IoLinkProperties[0].NodeTo = (NodeId == 0) ? 1 : 0;
        IoLinkProperties[0].Weight = 20;
        IoLinkProperties[0].MinimumLatency = 0;
        IoLinkProperties[0].MaximumLatency = 0;
        IoLinkProperties[0].MinimumBandwidth = 0;
        IoLinkProperties[0].MaximumBandwidth = 0;
    }

    return HSAKMT_STATUS_SUCCESS;
}
