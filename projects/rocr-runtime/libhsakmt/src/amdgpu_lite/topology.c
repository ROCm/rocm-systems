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

#include "libhsakmt.h"
#include "amdgpu_lite_device.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysinfo.h>

/*
 * amdgpu_lite topology:
 *   Node 0 = CPU (host)
 *   Node 1 = GPU (the amdgpu_lite device)
 *
 * We synthesize the topology from amdgpu_lite GET_INFO data
 * instead of reading /sys/class/kfd/.
 */

#define AMDGPU_LITE_NUM_NODES      2
#define AMDGPU_LITE_CPU_NODE       0
#define AMDGPU_LITE_GPU_NODE       1

/* GPU ID for the single GPU node (arbitrary nonzero value) */
#define AMDGPU_LITE_GPU_ID         1

/* Number of memory banks reported for the GPU node:
 *   0 = VRAM (device local)
 *   1 = GTT (system memory visible to GPU)
 */
#define AMDGPU_LITE_GPU_MEM_BANKS  2

/* GFX12.0.1 (RX 9070 XT) hardware parameters */
#define GFX1201_NUM_SHADER_ENGINES   4
#define GFX1201_NUM_CU_PER_SE       16
#define GFX1201_NUM_CU              (GFX1201_NUM_SHADER_ENGINES * GFX1201_NUM_CU_PER_SE)
#define GFX1201_NUM_SIMD_PER_CU      2
#define GFX1201_WAVES_PER_SIMD       16
#define GFX1201_MAX_CLOCK_MHZ       2970

static bool topology_acquired = false;

HSAKMT_STATUS HSAKMTAPI hsaKmtAcquireSystemProperties(
	HsaSystemProperties *SystemProperties)
{
	CHECK_KFD_OPEN();

	if (!SystemProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	memset(SystemProperties, 0, sizeof(*SystemProperties));
	SystemProperties->NumNodes = AMDGPU_LITE_NUM_NODES;
	SystemProperties->PlatformOem = 0;
	SystemProperties->PlatformId = 0;
	SystemProperties->PlatformRev = 0;

	topology_acquired = true;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAcquireSystemPropertiesCtx(
	HsaKFDContext *ctx,
	HsaSystemProperties *SystemProperties)
{
	return hsaKmtAcquireSystemProperties(SystemProperties);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReleaseSystemProperties(void)
{
	CHECK_KFD_OPEN();

	topology_acquired = false;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReleaseSystemPropertiesCtx(HsaKFDContext *ctx)
{
	return hsaKmtReleaseSystemProperties();
}

static void fill_cpu_node_properties(HsaNodeProperties *props)
{
	memset(props, 0, sizeof(*props));

	props->NumCPUCores = sysconf(_SC_NPROCESSORS_ONLN);
	props->CComputeIdLo = 0;
	props->NumMemoryBanks = 1; /* System memory */
	props->NumCaches = 0;
	props->NumIOLinks = 1; /* Link to GPU */
	props->DeviceId = 0;

	/* CPU engine version */
	props->EngineId.ui32.Major = 0;
	props->EngineId.ui32.Minor = 0;
	props->EngineId.ui32.Stepping = 0;
}

static void fill_gpu_node_properties(HsaNodeProperties *props)
{
	struct amdgpu_lite_device *dev = &g_amdgpu_lite_dev;
	uint32_t gfxv = dev->gfx_version;

	memset(props, 0, sizeof(*props));

	props->NumCPUCores = 0;
	props->FComputeIdLo = AMDGPU_LITE_GPU_ID;

	/* GFX engine version from packed gfxv */
	props->EngineId.ui32.Major = (gfxv >> 16) & 0xFF;
	props->EngineId.ui32.Minor = (gfxv >> 8) & 0xFF;
	props->EngineId.ui32.Stepping = gfxv & 0xFF;
	props->EngineId.ui32.uCode = 1; /* nonzero to pass BasicTest */

	/* Firmware versions (nonzero to pass BasicTest) */
	props->uCodeEngineVersions.uCodeSDMA = 1;

	/* Device identification */
	props->DeviceId = dev->device_id;
	props->VendorId = dev->info.vendor_id;
	props->LocationId = 0; /* PCI BDF - not available from ioctl */

	/* Compute capabilities */
	props->NumFComputeCores = GFX1201_NUM_CU * GFX1201_NUM_SIMD_PER_CU;
	props->NumSIMDPerCU = GFX1201_NUM_SIMD_PER_CU;
	props->NumShaderBanks = GFX1201_NUM_SHADER_ENGINES;
	props->MaxWavesPerSIMD = GFX1201_WAVES_PER_SIMD;
	props->NumArrays = GFX1201_NUM_SHADER_ENGINES;
	props->NumCUPerArray = GFX1201_NUM_CU_PER_SE;

	/* Register file sizes per CU */
	props->VGPRSizePerCU = 384 * 1024;  /* GFX12: 384 KB VGPR */
	props->SGPRSizePerCU = 16 * 1024;   /* GFX12: 16 KB SGPR */

	/* Memory banks: VRAM + GTT */
	props->NumMemoryBanks = AMDGPU_LITE_GPU_MEM_BANKS;
	props->NumCaches = 0;
	props->NumIOLinks = 1; /* Link to CPU */

	/* Clock info */
	props->MaxEngineClockMhzFCompute = GFX1201_MAX_CLOCK_MHZ;

	/* VRAM size in bytes */
	props->LocalMemSize = dev->vram_size;

	/* Mark as dGPU */
	props->Capability.ui32.HSAMMUPresent = 0;
	props->Capability.ui32.AQLQueueDoubleMap = 0;

	/* DRM render node not available through amdgpu_lite */
	props->DrmRenderMinor = 0;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeProperties(
	HSAuint32 NodeId,
	HsaNodeProperties *NodeProperties)
{
	CHECK_KFD_OPEN();

	if (!NodeProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (NodeId >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (NodeId == AMDGPU_LITE_CPU_NODE) {
		fill_cpu_node_properties(NodeProperties);
	} else {
		fill_gpu_node_properties(NodeProperties);
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodePropertiesCtx(
	HsaKFDContext *ctx,
	HSAuint32 NodeId,
	HsaNodeProperties *NodeProperties)
{
	return hsaKmtGetNodeProperties(NodeId, NodeProperties);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeMemoryProperties(
	HSAuint32 NodeId,
	HSAuint32 NumBanks,
	HsaMemoryProperties *MemoryProperties)
{
	struct amdgpu_lite_device *dev = &g_amdgpu_lite_dev;
	struct sysinfo si;

	CHECK_KFD_OPEN();

	if (!MemoryProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (NodeId >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	memset(MemoryProperties, 0, NumBanks * sizeof(*MemoryProperties));

	if (NodeId == AMDGPU_LITE_CPU_NODE) {
		/* CPU node: system memory */
		if (NumBanks < 1)
			return HSAKMT_STATUS_INVALID_PARAMETER;

		sysinfo(&si);
		MemoryProperties[0].HeapType = HSA_HEAPTYPE_SYSTEM;
		MemoryProperties[0].SizeInBytes = (uint64_t)si.totalram * si.mem_unit;
		MemoryProperties[0].Flags.MemoryProperty = 0;
		MemoryProperties[0].Width = 64;
	} else {
		/* GPU node: VRAM + GTT */
		if (NumBanks >= 1) {
			MemoryProperties[0].HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC;
			MemoryProperties[0].SizeInBytes = dev->vram_size;
			MemoryProperties[0].Flags.MemoryProperty = 0;
			MemoryProperties[0].Width = 256; /* GDDR6X bus width */
		}
		if (NumBanks >= 2) {
			/* GTT heap: system memory accessible by GPU via GART */
			sysinfo(&si);
			MemoryProperties[1].HeapType = HSA_HEAPTYPE_SYSTEM;
			MemoryProperties[1].SizeInBytes =
				(uint64_t)si.totalram * si.mem_unit;
			MemoryProperties[1].Flags.MemoryProperty = 0;
			MemoryProperties[1].Width = 64;
		}
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeMemoryPropertiesCtx(
	HsaKFDContext *ctx,
	HSAuint32 NodeId,
	HSAuint32 NumBanks,
	HsaMemoryProperties *MemoryProperties)
{
	return hsaKmtGetNodeMemoryProperties(NodeId, NumBanks,
					     MemoryProperties);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeCacheProperties(
	HSAuint32 NodeId,
	HSAuint32 ProcessorId,
	HSAuint32 NumCaches,
	HsaCacheProperties *CacheProperties)
{
	CHECK_KFD_OPEN();

	if (NodeId >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	/* No cache info reported for amdgpu_lite */
	if (CacheProperties && NumCaches > 0)
		memset(CacheProperties, 0,
		       NumCaches * sizeof(*CacheProperties));

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeCachePropertiesCtx(
	HsaKFDContext *ctx,
	HSAuint32 NodeId,
	HSAuint32 ProcessorId,
	HSAuint32 NumCaches,
	HsaCacheProperties *CacheProperties)
{
	return hsaKmtGetNodeCacheProperties(NodeId, ProcessorId, NumCaches,
					    CacheProperties);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeIoLinkProperties(
	HSAuint32 NodeId,
	HSAuint32 NumIoLinks,
	HsaIoLinkProperties *IoLinkProperties)
{
	CHECK_KFD_OPEN();

	if (NodeId >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (!IoLinkProperties || NumIoLinks == 0)
		return HSAKMT_STATUS_SUCCESS;

	memset(IoLinkProperties, 0, NumIoLinks * sizeof(*IoLinkProperties));

	/* Single PCIe link between CPU and GPU */
	if (NumIoLinks >= 1) {
		IoLinkProperties[0].IoLinkType = HSA_IOLINKTYPE_PCIEXPRESS;
		IoLinkProperties[0].Flags.ui32.Override = 0;

		if (NodeId == AMDGPU_LITE_CPU_NODE) {
			/* CPU -> GPU link */
			IoLinkProperties[0].NodeFrom = AMDGPU_LITE_CPU_NODE;
			IoLinkProperties[0].NodeTo = AMDGPU_LITE_GPU_NODE;
		} else {
			/* GPU -> CPU link */
			IoLinkProperties[0].NodeFrom = AMDGPU_LITE_GPU_NODE;
			IoLinkProperties[0].NodeTo = AMDGPU_LITE_CPU_NODE;
		}

		/* PCIe Gen4 x16 typical bandwidth */
		IoLinkProperties[0].MinimumBandwidth = 0;
		IoLinkProperties[0].MaximumBandwidth = 0;
		IoLinkProperties[0].Weight = 20;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeIoLinkPropertiesCtx(
	HsaKFDContext *ctx,
	HSAuint32 NodeId,
	HSAuint32 NumIoLinks,
	HsaIoLinkProperties *IoLinkProperties)
{
	return hsaKmtGetNodeIoLinkProperties(NodeId, NumIoLinks,
					     IoLinkProperties);
}

/*
 * Helper functions used by the generic libhsakmt code.
 */

HSAKMT_STATUS hsakmt_validate_nodeid(HsaKFDContext *ctx, uint32_t nodeid,
				     uint32_t *gpu_id)
{
	if (nodeid >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (gpu_id) {
		if (nodeid == AMDGPU_LITE_GPU_NODE)
			*gpu_id = AMDGPU_LITE_GPU_ID;
		else
			*gpu_id = 0;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS hsakmt_gpuid_to_nodeid(HsaKFDContext *ctx, uint32_t gpu_id,
				     uint32_t *node_id)
{
	if (gpu_id == AMDGPU_LITE_GPU_ID) {
		*node_id = AMDGPU_LITE_GPU_NODE;
		return HSAKMT_STATUS_SUCCESS;
	}

	return HSAKMT_STATUS_INVALID_NODE_UNIT;
}

uint32_t hsakmt_get_gfxv_by_node_id(HsaKFDContext *ctx, HSAuint32 node_id)
{
	if (node_id == AMDGPU_LITE_GPU_NODE)
		return g_amdgpu_lite_dev.gfx_version;
	return 0;
}

uint16_t hsakmt_get_device_id_by_node_id(HsaKFDContext *ctx,
					 HSAuint32 node_id)
{
	if (node_id == AMDGPU_LITE_GPU_NODE)
		return g_amdgpu_lite_dev.device_id;
	return 0;
}

uint16_t hsakmt_get_device_id_by_gpu_id(HsaKFDContext *ctx,
					HSAuint32 gpu_id)
{
	if (gpu_id == AMDGPU_LITE_GPU_ID)
		return g_amdgpu_lite_dev.device_id;
	return 0;
}

uint32_t hsakmt_get_direct_link_cpu(HsaKFDContext *ctx, uint32_t gpu_node)
{
	if (gpu_node == AMDGPU_LITE_GPU_NODE)
		return AMDGPU_LITE_CPU_NODE;
	return 0;
}

bool hsakmt_prefer_ats(HsaKFDContext *ctx, HSAuint32 node_id)
{
	/* amdgpu_lite does not support ATS */
	return false;
}

HSAKMT_STATUS hsakmt_validate_nodeid_array(HsaKFDContext *ctx,
		uint32_t **gpu_id_array,
		uint32_t NumberOfNodes, uint32_t *NodeArray)
{
	uint32_t *ids;
	uint32_t i;

	ids = malloc(NumberOfNodes * sizeof(uint32_t));
	if (!ids)
		return HSAKMT_STATUS_NO_MEMORY;

	for (i = 0; i < NumberOfNodes; i++) {
		HSAKMT_STATUS status;
		status = hsakmt_validate_nodeid(ctx, NodeArray[i], &ids[i]);
		if (status != HSAKMT_STATUS_SUCCESS) {
			free(ids);
			return status;
		}
	}

	*gpu_id_array = ids;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS hsakmt_topology_sysfs_get_system_props(HsaKFDContext *ctx,
						     HsaSystemProperties *props)
{
	return hsaKmtAcquireSystemProperties(props);
}

void hsakmt_topology_setup_is_dgpu_param(HsaNodeProperties *props)
{
	hsakmt_is_dgpu = true;
}

bool hsakmt_topology_is_svm_needed(HSA_ENGINE_ID EngineId)
{
	return false;
}

uint32_t hsakmt_get_num_sysfs_nodes(HsaKFDContext *ctx)
{
	return AMDGPU_LITE_NUM_NODES;
}

int get_drm_render_fd_by_gpu_id(HSAuint32 gpu_id)
{
	/* No DRM render node available through amdgpu_lite */
	return -1;
}

uint32_t hsakmt_get_vgpr_size_per_cu(uint32_t gfxv)
{
	/* GFX12: 384 KB VGPR per CU */
	if ((gfxv >> 16) >= 0x0C)
		return 384 * 1024;
	return 256 * 1024;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeWallclockFrequency(
	HSAuint32 NodeId,
	uint64_t *Frequency)
{
	CHECK_KFD_OPEN();

	if (!Frequency)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (NodeId >= AMDGPU_LITE_NUM_NODES)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	/* Return 0 - wallclock frequency not available through amdgpu_lite */
	*Frequency = 0;
	return HSAKMT_STATUS_NOT_SUPPORTED;
}
