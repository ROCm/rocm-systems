/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
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
#include "hsakmt/linux/kfd_ioctl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <xf86drm.h>

#include "fmm.h"

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryPolicyCtx(HsaKFDContext *ctx,
						  HSAuint32 Node,
					      HSAuint32 DefaultPolicy,
					      HSAuint32 AlternatePolicy,
					      void *MemoryAddressAlternate,
					      HSAuint64 MemorySizeInBytes)
{
	struct kfd_ioctl_set_memory_policy_args args = {0};
	HSAKMT_STATUS result;
	uint32_t gpu_id;

	CHECK_KFD_OPEN();

	pr_debug("[%s] node %d; default %d; alternate %d\n",
		__func__, Node, DefaultPolicy, AlternatePolicy);

	result = hsakmt_validate_nodeid(ctx, Node, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	if (hsakmt_get_gfxv_by_node_id(ctx, Node) != GFX_VERSION_KAVERI)
		/* This is a legacy API useful on Kaveri only. On dGPU
		 * the alternate aperture is setup and used
		 * automatically for coherent allocations. Don't let
		 * app override it.
		 */
		return HSAKMT_STATUS_NOT_IMPLEMENTED;

	/*
	 * We accept any legal policy and alternate address location.
	 * You get CC everywhere anyway.
	 */
	if ((DefaultPolicy != HSA_CACHING_CACHED &&
		DefaultPolicy != HSA_CACHING_NONCACHED) ||
			(AlternatePolicy != HSA_CACHING_CACHED &&
			AlternatePolicy != HSA_CACHING_NONCACHED))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	CHECK_PAGE_MULTIPLE(MemoryAddressAlternate);
	CHECK_PAGE_MULTIPLE(MemorySizeInBytes);

	args.gpu_id = gpu_id;
	args.default_policy = (DefaultPolicy == HSA_CACHING_CACHED) ?
					KFD_IOC_CACHE_POLICY_COHERENT :
					KFD_IOC_CACHE_POLICY_NONCOHERENT;

	args.alternate_policy = (AlternatePolicy == HSA_CACHING_CACHED) ?
					KFD_IOC_CACHE_POLICY_COHERENT :
					KFD_IOC_CACHE_POLICY_NONCOHERENT;

	args.alternate_aperture_base = (uintptr_t) MemoryAddressAlternate;
	args.alternate_aperture_size = MemorySizeInBytes;

	int err = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_SET_MEMORY_POLICY, &args);

	return (err == -1) ? HSAKMT_STATUS_ERROR : HSAKMT_STATUS_SUCCESS;
}

HSAuint32 hsakmt_PageSizeFromFlags(unsigned int pageSizeFlags)
{
	switch (pageSizeFlags) {
	case HSA_PAGE_SIZE_4KB: return 4*1024;
	case HSA_PAGE_SIZE_64KB: return 64*1024;
	case HSA_PAGE_SIZE_2MB: return 2*1024*1024;
	case HSA_PAGE_SIZE_1GB: return 1024*1024*1024;
	default:
		assert(false);
		return 4*1024;
	}
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemoryCtx(HsaKFDContext *ctx,
					  HSAuint32 PreferredNode,
					  HSAuint64 SizeInBytes,
					  HsaMemFlags MemFlags,
					  void **MemoryAddress)
{
	return hsaKmtAllocMemoryAlignCtx(ctx, PreferredNode, SizeInBytes, 0, MemFlags, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemoryAlignCtx(HsaKFDContext *ctx,
					  HSAuint32 PreferredNode,
					  HSAuint64 SizeInBytes,
					  HSAuint64 Alignment,
					  HsaMemFlags MemFlags,
					  void **MemoryAddress)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;
	HSAuint64 page_size;

	CHECK_KFD_OPEN();

	if (MemFlags.ui32.Contiguous)
		CHECK_KFD_MINOR_VERSION(16);

	pr_debug("[%s] node %d\n", __func__, PreferredNode);

	result = hsakmt_validate_nodeid(ctx, PreferredNode, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS) {
		pr_err("[%s] invalid node ID: %d\n", __func__, PreferredNode);
		return result;
	}

	page_size = hsakmt_PageSizeFromFlags(MemFlags.ui32.PageSize);

	if (Alignment && (Alignment < page_size || !POWER_OF_2(Alignment)))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (!MemoryAddress || !SizeInBytes || (SizeInBytes & (page_size-1)))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (MemFlags.ui32.FixedAddress) {
		if (*MemoryAddress == NULL)
			return HSAKMT_STATUS_INVALID_PARAMETER;
	} else
		*MemoryAddress = NULL;

	if ((MemFlags.ui32.CoarseGrain && MemFlags.ui32.ExtendedCoherent) ||
	    (MemFlags.ui32.ExtendedCoherent && MemFlags.ui32.Uncached))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (MemFlags.ui32.Scratch) {
		if (Alignment) {
			// Scratch memory currently forced to SCRATCH_ALIGN
			pr_err("[%s] Alignment not supported for scratch memory: %d\n", __func__, PreferredNode);
			return HSAKMT_STATUS_NOT_IMPLEMENTED;
		}

		*MemoryAddress = hsakmt_fmm_allocate_scratch(ctx, gpu_id, *MemoryAddress, SizeInBytes);

		if (!(*MemoryAddress)) {
			pr_err("[%s] failed to allocate %lu bytes from scratch\n",
				__func__, SizeInBytes);
			return HSAKMT_STATUS_NO_MEMORY;
		}

		pr_debug("[%s] node %d address %p size %lu from scratch\n", __func__, PreferredNode, *MemoryAddress, SizeInBytes);
		return HSAKMT_STATUS_SUCCESS;
	}

	/* GPU allocated system memory */
	if (!gpu_id || !MemFlags.ui32.NonPaged || hsakmt_zfb_support || MemFlags.ui32.GTTAccess
		|| MemFlags.ui32.OnlyAddress) {
		/* Backwards compatibility hack: Allocate system memory if app
		 * asks for paged memory from a GPU node.
		 */

		/* If allocate VRAM under ZFB mode */
		if (hsakmt_zfb_support && gpu_id && MemFlags.ui32.NonPaged == 1)
			MemFlags.ui32.CoarseGrain = 1;

		*MemoryAddress = hsakmt_fmm_allocate_host(ctx, gpu_id, MemFlags.ui32.GTTAccess ? 0 : PreferredNode,
						   *MemoryAddress, SizeInBytes, Alignment, MemFlags);

		if (!(*MemoryAddress)) {
			pr_err("[%s] failed to allocate %lu bytes from host\n",
				__func__, SizeInBytes);
			return HSAKMT_STATUS_ERROR;
		}

		pr_debug("[%s] node %d address %p size %lu from host\n", __func__, PreferredNode, *MemoryAddress, SizeInBytes);
		return HSAKMT_STATUS_SUCCESS;
	}

	/* GPU allocated VRAM */
	/* sanity check cannot do OnlyAddress and NoAddress alloc at same time */
	if (MemFlags.ui32.OnlyAddress && MemFlags.ui32.NoAddress) {
		pr_err("[%s] allocate addr-only and memory-only at same time\n",
			__func__);
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	*MemoryAddress = hsakmt_fmm_allocate_device(ctx, gpu_id, PreferredNode, *MemoryAddress,
					     SizeInBytes, Alignment, MemFlags);

	if (!(*MemoryAddress)) {
		pr_err("[%s] failed to allocate %lu bytes from device\n",
			__func__, SizeInBytes);
		return HSAKMT_STATUS_NO_MEMORY;
	}

	pr_debug("[%s] node %d address %p size %lu from device\n", __func__, PreferredNode, *MemoryAddress, SizeInBytes);
	return HSAKMT_STATUS_SUCCESS;

}

HSAKMT_STATUS HSAKMTAPI hsaKmtFreeMemoryCtx(HsaKFDContext *ctx,
					 void *MemoryAddress,
					 HSAuint64 SizeInBytes)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	if (!MemoryAddress) {
		pr_err("FIXME: freeing NULL pointer\n");
		return HSAKMT_STATUS_ERROR;
	}

	return hsakmt_fmm_release(ctx, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAvailableMemoryCtx(HsaKFDContext *ctx,
						  HSAuint32 Node,
						  HSAuint64 *AvailableBytes)
{
	struct kfd_ioctl_get_available_memory_args args = {};
	HSAKMT_STATUS result;

	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(9);

	pr_debug("[%s] node %d\n", __func__, Node);

	result = hsakmt_validate_nodeid(ctx, Node, &args.gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS) {
		pr_err("[%s] invalid node ID: %d\n", __func__, Node);
		return result;
	}

	if (hsakmt_ioctl(ctx->fd, AMDKFD_IOC_AVAILABLE_MEMORY, &args))
		return HSAKMT_STATUS_ERROR;

	*AvailableBytes = args.available;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryCtx(HsaKFDContext *ctx,
					     void *MemoryAddress,
					     HSAuint64 MemorySizeInBytes)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p size %lu\n", __func__, MemoryAddress, MemorySizeInBytes);

	if (!hsakmt_is_dgpu)
		/* TODO: support mixed APU and dGPU configurations */
		return HSAKMT_STATUS_SUCCESS;

	HsaMemFlags flags;
	flags.ui32.CoarseGrain = 1;
	flags.ui32.ExtendedCoherent = 0;
	return hsakmt_fmm_register_memory(ctx,
				   MemoryAddress, MemorySizeInBytes,
				   NULL, 0, flags);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryToNodesCtx(HsaKFDContext *ctx,
						    void *MemoryAddress,
						    HSAuint64 MemorySizeInBytes,
						    HSAuint64 NumberOfNodes,
						    HSAuint32 *NodeArray)
{
	CHECK_KFD_OPEN();
	uint32_t *gpu_id_array;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] address %p size %lu number of nodes %lu\n",
		__func__, MemoryAddress, MemorySizeInBytes, NumberOfNodes);

	if (!hsakmt_is_dgpu)
		/* TODO: support mixed APU and dGPU configurations */
		return HSAKMT_STATUS_NOT_SUPPORTED;

	ret = hsakmt_validate_nodeid_array(ctx, &gpu_id_array,
			NumberOfNodes, NodeArray);

	if (ret == HSAKMT_STATUS_SUCCESS) {
		HsaMemFlags flags;
		flags.ui32.CoarseGrain = 1;
		flags.ui32.ExtendedCoherent = 0;

		ret = hsakmt_fmm_register_memory(ctx,
					  MemoryAddress, MemorySizeInBytes,
					  gpu_id_array,
					  NumberOfNodes*sizeof(uint32_t),
					  flags);
		if (ret != HSAKMT_STATUS_SUCCESS)
			free(gpu_id_array);
	}

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryWithFlagsCtx(HsaKFDContext *ctx,
						    void *MemoryAddress,
						    HSAuint64 MemorySizeInBytes,
						    HsaMemFlags MemFlags)
{
	CHECK_KFD_OPEN();
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] address %p size %lu\n",
		__func__, MemoryAddress, MemorySizeInBytes);

	if (MemFlags.ui32.ExtendedCoherent && MemFlags.ui32.CoarseGrain)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	// Registered memory should be ordinary paged host memory.
	if ((MemFlags.ui32.HostAccess != 1) || (MemFlags.ui32.NonPaged == 1))
		return HSAKMT_STATUS_NOT_SUPPORTED;

	if (!hsakmt_is_dgpu)
		/* TODO: support mixed APU and dGPU configurations */
		return HSAKMT_STATUS_NOT_SUPPORTED;

	ret = hsakmt_fmm_register_memory(ctx,
		MemoryAddress, MemorySizeInBytes,
		NULL, 0, MemFlags);

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodesCtx(HsaKFDContext *ctx,
							    HSAuint64 GraphicsResourceHandle,
							    HsaGraphicsResourceInfo *GraphicsResourceInfo,
							    HSAuint64 NumberOfNodes,
							    HSAuint32 *NodeArray)
{
       HSA_REGISTER_MEM_FLAGS regFlags;
       regFlags.Value = 0;

       return hsaKmtRegisterGraphicsHandleToNodesExtCtx(ctx,
						     GraphicsResourceHandle,
						     GraphicsResourceInfo,
						     NumberOfNodes,
						     NodeArray,
						     regFlags);

}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodesExtCtx(HsaKFDContext *ctx,
							       HSAuint64 GraphicsResourceHandle,
							       HsaGraphicsResourceInfo *GraphicsResourceInfo,
							       HSAuint64 NumberOfNodes,
							       HSAuint32 *NodeArray,
							       HSA_REGISTER_MEM_FLAGS RegisterFlags)
{
	CHECK_KFD_OPEN();
	uint32_t *gpu_id_array = NULL;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] number of nodes %lu\n", __func__, NumberOfNodes);

	if (NodeArray != NULL || NumberOfNodes != 0) {
		ret = hsakmt_validate_nodeid_array(ctx, &gpu_id_array,
				NumberOfNodes, NodeArray);
	}

	if (ret == HSAKMT_STATUS_SUCCESS) {
		ret = hsakmt_fmm_register_graphics_handle(ctx,
			GraphicsResourceHandle, GraphicsResourceInfo,
			gpu_id_array, NumberOfNodes * sizeof(uint32_t), RegisterFlags);
		if (ret != HSAKMT_STATUS_SUCCESS)
			free(gpu_id_array);
	}

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtExportDMABufHandleCtx(HsaKFDContext *ctx,
						 void *MemoryAddress,
						 HSAuint64 MemorySizeInBytes,
						 int *DMABufFd,
						 HSAuint64 *Offset)
{
	CHECK_KFD_OPEN();
	CHECK_KFD_MINOR_VERSION(12);

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	return hsakmt_fmm_export_dma_buf_fd(ctx,
				     MemoryAddress, MemorySizeInBytes,
				     DMABufFd, Offset);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtShareMemoryCtx(HsaKFDContext *ctx,
					  void *MemoryAddress,
					  HSAuint64 SizeInBytes,
					  HsaSharedMemoryHandle *SharedMemoryHandle)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	if (!SharedMemoryHandle)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	return hsakmt_fmm_share_memory(ctx, MemoryAddress, SizeInBytes, SharedMemoryHandle);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterSharedHandleCtx(HsaKFDContext *ctx,
					  const HsaSharedMemoryHandle *SharedMemoryHandle,
					  void **MemoryAddress,
					  HSAuint64 *SizeInBytes)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] handle %p\n", __func__, SharedMemoryHandle);

	return hsaKmtRegisterSharedHandleToNodesCtx(ctx,
						 SharedMemoryHandle,
						 MemoryAddress,
						 SizeInBytes,
						 0,
						 NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterSharedHandleToNodesCtx(HsaKFDContext *ctx,
							  const HsaSharedMemoryHandle *SharedMemoryHandle,
							  void **MemoryAddress,
							  HSAuint64 *SizeInBytes,
							  HSAuint64 NumberOfNodes,
							  HSAuint32 *NodeArray)
{
	CHECK_KFD_OPEN();

	uint32_t *gpu_id_array = NULL;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] handle %p number of nodes %lu\n",
		__func__, SharedMemoryHandle, NumberOfNodes);

	if (!SharedMemoryHandle)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (NodeArray) {
		ret = hsakmt_validate_nodeid_array(ctx, &gpu_id_array, NumberOfNodes, NodeArray);
		if (ret != HSAKMT_STATUS_SUCCESS)
			goto error;
	}

	ret = hsakmt_fmm_register_shared_memory(ctx,
					 SharedMemoryHandle,
					 SizeInBytes,
					 MemoryAddress,
					 gpu_id_array,
					 NumberOfNodes*sizeof(uint32_t));
	if (ret != HSAKMT_STATUS_SUCCESS)
		goto error;

	return ret;

error:
	if (gpu_id_array)
		free(gpu_id_array);
	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtProcessVMRead(HSAuint32 Pid,
					    HsaMemoryRange *LocalMemoryArray,
					    HSAuint64 LocalMemoryArrayCount,
					    HsaMemoryRange *RemoteMemoryArray,
					    HSAuint64 RemoteMemoryArrayCount,
					    HSAuint64 *SizeCopied)
{
	pr_err("[%s] Deprecated\n", __func__);

	return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtProcessVMWrite(HSAuint32 Pid,
					     HsaMemoryRange *LocalMemoryArray,
					     HSAuint64 LocalMemoryArrayCount,
					     HsaMemoryRange *RemoteMemoryArray,
					     HSAuint64 RemoteMemoryArrayCount,
					     HSAuint64 *SizeCopied)
{
	pr_err("[%s] Deprecated\n", __func__);

	return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDeregisterMemoryCtx(HsaKFDContext *ctx, void *MemoryAddress)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	return hsakmt_fmm_deregister_memory(ctx, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUCtx(HsaKFDContext *ctx,
					     void *MemoryAddress,
					     HSAuint64 MemorySizeInBytes,
					     HSAuint64 *AlternateVAGPU)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	if (!MemoryAddress) {
		pr_err("FIXME: mapping NULL pointer\n");
		return HSAKMT_STATUS_ERROR;
	}

	if (AlternateVAGPU)
		*AlternateVAGPU = 0;

	return hsakmt_fmm_map_to_gpu(ctx, MemoryAddress, MemorySizeInBytes, AlternateVAGPU);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUNodesCtx(HsaKFDContext *ctx,
						  void *MemoryAddress,
						  HSAuint64 MemorySizeInBytes,
						  HSAuint64 *AlternateVAGPU,
						  HsaMemMapFlags MemMapFlags,
						  HSAuint64 NumberOfNodes,
						  HSAuint32 *NodeArray)
{
	uint32_t *gpu_id_array;
	HSAKMT_STATUS ret;

	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p number of nodes %lu\n",
		__func__, MemoryAddress, NumberOfNodes);

	if (!MemoryAddress) {
		pr_err("FIXME: mapping NULL pointer\n");
		return HSAKMT_STATUS_ERROR;
	}

	if (!hsakmt_is_dgpu && NumberOfNodes == 1)
		return hsaKmtMapMemoryToGPUCtx(ctx, MemoryAddress,
					MemorySizeInBytes, AlternateVAGPU);

	ret = hsakmt_validate_nodeid_array(ctx, &gpu_id_array,
				NumberOfNodes, NodeArray);
	if (ret != HSAKMT_STATUS_SUCCESS)
		return ret;

	ret = hsakmt_fmm_map_to_gpu_nodes(ctx, MemoryAddress, MemorySizeInBytes,
		gpu_id_array, NumberOfNodes, AlternateVAGPU);

	if (gpu_id_array)
		free(gpu_id_array);

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapMemoryToGPUCtx(HsaKFDContext *ctx, void *MemoryAddress)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] address %p\n", __func__, MemoryAddress);

	if (!MemoryAddress) {
		/* Workaround for runtime bug */
		pr_err("FIXME: Unmapping NULL pointer\n");
		return HSAKMT_STATUS_SUCCESS;
	}

	if (!hsakmt_fmm_unmap_from_gpu(ctx, MemoryAddress))
		return HSAKMT_STATUS_SUCCESS;
	else
		return HSAKMT_STATUS_ERROR;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapGraphicHandle(HSAuint32 NodeId,
					       HSAuint64 GraphicDeviceHandle,
					       HSAuint64 GraphicResourceHandle,
					       HSAuint64 GraphicResourceOffset,
					       HSAuint64 GraphicResourceSize,
					       HSAuint64 *FlatMemoryAddress)
{
	/* This API was only ever implemented in KFD for Kaveri and
	 * was never upstreamed. There are no open-source users of
	 * this interface. It has been superseded by
	 * RegisterGraphicsHandleToNodes.
	 */
	return HSAKMT_STATUS_NOT_IMPLEMENTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapGraphicHandleCtx(HsaKFDContext *ctx,
						 HSAuint32 NodeId,
						 HSAuint64 FlatMemoryAddress,
						 HSAuint64 SizeInBytes)
{
	return hsaKmtUnmapMemoryToGPUCtx(ctx, PORT_UINT64_TO_VPTR(FlatMemoryAddress));
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetTileConfigCtx(HsaKFDContext *ctx,
						 HSAuint32 NodeId, HsaGpuTileConfig *config)
{
	struct kfd_ioctl_get_tile_config_args args = {0};
	uint32_t gpu_id;
	HSAKMT_STATUS result;

	CHECK_KFD_OPEN();

	pr_debug("[%s] node %d\n", __func__, NodeId);

	result = hsakmt_validate_nodeid(ctx, NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	/* Avoid Valgrind warnings about uninitialized data. Valgrind doesn't
	 * know that KFD writes this.
	 */
	memset(config->TileConfig, 0, sizeof(*config->TileConfig) * config->NumTileConfigs);
	memset(config->MacroTileConfig, 0, sizeof(*config->MacroTileConfig) * config->NumMacroTileConfigs);

	args.gpu_id = gpu_id;
	args.tile_config_ptr = (uint64_t)config->TileConfig;
	args.macro_tile_config_ptr = (uint64_t)config->MacroTileConfig;
	args.num_tile_configs = config->NumTileConfigs;
	args.num_macro_tile_configs = config->NumMacroTileConfigs;

	if (hsakmt_ioctl(ctx->fd, AMDKFD_IOC_GET_TILE_CONFIG, &args) != 0)
		return HSAKMT_STATUS_ERROR;

	config->NumTileConfigs = args.num_tile_configs;
	config->NumMacroTileConfigs = args.num_macro_tile_configs;

	config->GbAddrConfig = args.gb_addr_config;

	config->NumBanks = args.num_banks;
	config->NumRanks = args.num_ranks;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryPointerInfoCtx(HsaKFDContext *ctx,
					       const void *Pointer,
					       HsaPointerInfo *PointerInfo)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] pointer %p\n", __func__, Pointer);

	if (!PointerInfo)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	return hsakmt_fmm_get_mem_info(ctx, Pointer, PointerInfo);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryUserDataCtx(HsaKFDContext *ctx,
						const void *Pointer,
						void *UserData)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] pointer %p\n", __func__, Pointer);

	return hsakmt_fmm_set_mem_user_data(ctx, Pointer, UserData);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReplaceAsanHeaderPageCtx(HsaKFDContext *ctx, void *addr)
{
#ifdef SANITIZER_AMDGPU
	pr_debug("[%s] address %p\n", __func__, addr);
	CHECK_KFD_OPEN();

	return hsakmt_fmm_replace_asan_header_page(ctx, addr);
#else
	return HSAKMT_STATUS_NOT_SUPPORTED;
#endif
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReturnAsanHeaderPageCtx(HsaKFDContext *ctx, void *addr)
{
#ifdef SANITIZER_AMDGPU
	pr_debug("[%s] address %p\n", __func__, addr);
	CHECK_KFD_OPEN();

	return hsakmt_fmm_return_asan_header_page(ctx, addr);
#else
	return HSAKMT_STATUS_NOT_SUPPORTED;
#endif
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetAMDGPUDeviceHandleCtx(HsaKFDContext *ctx,
						HSAuint32 NodeId,
						HsaAMDGPUDeviceHandle   *DeviceHandle)
{
	CHECK_KFD_OPEN();

	return hsakmt_fmm_get_amdgpu_device_handle(ctx, NodeId, DeviceHandle);
}


HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryPolicy(HSAuint32 Node,
					      HSAuint32 DefaultPolicy,
					      HSAuint32 AlternatePolicy,
					      void *MemoryAddressAlternate,
					      HSAuint64 MemorySizeInBytes)
{
	return hsaKmtSetMemoryPolicyCtx(&hsakmt_primary_kfd_ctx, Node,
					  DefaultPolicy, AlternatePolicy,
					  MemoryAddressAlternate, MemorySizeInBytes);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemory(HSAuint32 PreferredNode,
					  HSAuint64 SizeInBytes,
					  HsaMemFlags MemFlags,
					  void **MemoryAddress)
{
	return hsaKmtAllocMemoryAlign(PreferredNode, SizeInBytes, 0, MemFlags, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocMemoryAlign(HSAuint32 PreferredNode,
					  HSAuint64 SizeInBytes,
					  HSAuint64 Alignment,
					  HsaMemFlags MemFlags,
					  void **MemoryAddress)
{
	return hsaKmtAllocMemoryAlignCtx(&hsakmt_primary_kfd_ctx, PreferredNode,
					  SizeInBytes, Alignment, MemFlags, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtFreeMemory(void *MemoryAddress,
					  HSAuint64 SizeInBytes)
{
	return hsaKmtFreeMemoryCtx(&hsakmt_primary_kfd_ctx, MemoryAddress, SizeInBytes);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAvailableMemory(HSAuint32 Node,
					  HSAuint64 *AvailableBytes)
{
	return hsaKmtAvailableMemoryCtx(&hsakmt_primary_kfd_ctx, Node, AvailableBytes);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemory(void *MemoryAddress,
					      HSAuint64 MemorySizeInBytes)
{
	return hsaKmtRegisterMemoryCtx(&hsakmt_primary_kfd_ctx, MemoryAddress, MemorySizeInBytes);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryToNodes(void *MemoryAddress,
					      HSAuint64 MemorySizeInBytes,
					      HSAuint64 NumberOfNodes,
					      HSAuint32 *NodeArray)
{
	return hsaKmtRegisterMemoryToNodesCtx(&hsakmt_primary_kfd_ctx,
					      MemoryAddress, MemorySizeInBytes,
					      NumberOfNodes, NodeArray);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterMemoryWithFlags(void *MemoryAddress,
					      HSAuint64 MemorySizeInBytes,
					      HsaMemFlags MemFlags)
{
	return hsaKmtRegisterMemoryWithFlagsCtx(&hsakmt_primary_kfd_ctx,
					      MemoryAddress, MemorySizeInBytes, MemFlags);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodes(HSAuint64 GraphicsResourceHandle,
							    HsaGraphicsResourceInfo *GraphicsResourceInfo,
							    HSAuint64 NumberOfNodes,
							    HSAuint32 *NodeArray)
{
       HSA_REGISTER_MEM_FLAGS regFlags;
       regFlags.Value = 0;

       return hsaKmtRegisterGraphicsHandleToNodesExt(GraphicsResourceHandle,
						     GraphicsResourceInfo,
						     NumberOfNodes,
						     NodeArray,
						     regFlags);

}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterGraphicsHandleToNodesExt(HSAuint64 GraphicsResourceHandle,
							       HsaGraphicsResourceInfo *GraphicsResourceInfo,
							       HSAuint64 NumberOfNodes,
							       HSAuint32 *NodeArray,
							       HSA_REGISTER_MEM_FLAGS RegisterFlags)
{
	return hsaKmtRegisterGraphicsHandleToNodesExtCtx(&hsakmt_primary_kfd_ctx,
							       GraphicsResourceHandle,
							       GraphicsResourceInfo,
							       NumberOfNodes,
							       NodeArray,
							       RegisterFlags);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtExportDMABufHandle(void *MemoryAddress,
						 HSAuint64 MemorySizeInBytes,
						 int *DMABufFd,
						 HSAuint64 *Offset)
{
	return hsaKmtExportDMABufHandleCtx(&hsakmt_primary_kfd_ctx,
					     MemoryAddress, MemorySizeInBytes,
					     DMABufFd, Offset);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtShareMemory(void *MemoryAddress,
					  HSAuint64 SizeInBytes,
					  HsaSharedMemoryHandle *SharedMemoryHandle)
{
	return hsaKmtShareMemoryCtx(&hsakmt_primary_kfd_ctx,
				     MemoryAddress, SizeInBytes, SharedMemoryHandle);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterSharedHandle(
					  const HsaSharedMemoryHandle *SharedMemoryHandle,
					  void **MemoryAddress,
					  HSAuint64 *SizeInBytes)
{
	CHECK_KFD_OPEN();

	pr_debug("[%s] handle %p\n", __func__, SharedMemoryHandle);

	return hsaKmtRegisterSharedHandleToNodes(SharedMemoryHandle,
						     MemoryAddress,
						     SizeInBytes,
						     0,
						     NULL);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtRegisterSharedHandleToNodes(const HsaSharedMemoryHandle *SharedMemoryHandle,
							  void **MemoryAddress,
							  HSAuint64 *SizeInBytes,
							  HSAuint64 NumberOfNodes,
							  HSAuint32 *NodeArray)
{
	return hsaKmtRegisterSharedHandleToNodesCtx(&hsakmt_primary_kfd_ctx,
						     SharedMemoryHandle,
						     MemoryAddress,
						     SizeInBytes,
						     NumberOfNodes,
						     NodeArray);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDeregisterMemory(void *MemoryAddress)
{
	return hsaKmtDeregisterMemoryCtx(&hsakmt_primary_kfd_ctx, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPU(void *MemoryAddress,
					  HSAuint64 MemorySizeInBytes,
					  HSAuint64 *AlternateVAGPU)
{
	return hsaKmtMapMemoryToGPUCtx(&hsakmt_primary_kfd_ctx, MemoryAddress, MemorySizeInBytes, AlternateVAGPU);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMapMemoryToGPUNodes(
					  void *MemoryAddress,
					  HSAuint64 MemorySizeInBytes,
					  HSAuint64 *AlternateVAGPU,
					  HsaMemMapFlags MemMapFlags,
					  HSAuint64 NumberOfNodes,
					  HSAuint32 *NodeArray)
{
	return hsaKmtMapMemoryToGPUNodesCtx(&hsakmt_primary_kfd_ctx, MemoryAddress,
				MemorySizeInBytes, AlternateVAGPU, MemMapFlags, NumberOfNodes, NodeArray);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapMemoryToGPU(void *MemoryAddress)
{
	return hsaKmtUnmapMemoryToGPUCtx(&hsakmt_primary_kfd_ctx, MemoryAddress);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUnmapGraphicHandle(HSAuint32 NodeId,
					  HSAuint64 FlatMemoryAddress,
					  HSAuint64 SizeInBytes)
{
	return hsaKmtUnmapMemoryToGPU(PORT_UINT64_TO_VPTR(FlatMemoryAddress));
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetTileConfig(HSAuint32 NodeId, HsaGpuTileConfig *config)
{
	return hsaKmtGetTileConfigCtx(&hsakmt_primary_kfd_ctx, NodeId, config);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueryPointerInfo(const void *Pointer,
					       HsaPointerInfo *PointerInfo)
{
	return hsaKmtQueryPointerInfoCtx(&hsakmt_primary_kfd_ctx, Pointer, PointerInfo);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetMemoryUserData(const void *Pointer,
						void *UserData)
{
	return hsaKmtSetMemoryUserDataCtx(&hsakmt_primary_kfd_ctx, Pointer, UserData);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReplaceAsanHeaderPage(void *addr)
{
	return hsaKmtReplaceAsanHeaderPageCtx(&hsakmt_primary_kfd_ctx, addr);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReturnAsanHeaderPage(void *addr)
{
	return hsaKmtReturnAsanHeaderPageCtx(&hsakmt_primary_kfd_ctx, addr);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetAMDGPUDeviceHandle(HSAuint32 NodeId,
						HsaAMDGPUDeviceHandle   *DeviceHandle)
{
	CHECK_KFD_OPEN();

	return hsaKmtGetAMDGPUDeviceHandleCtx(&hsakmt_primary_kfd_ctx, NodeId, DeviceHandle);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetMemoryHandle(void* va, void* MemoryAddress, HSAuint64 SizeInBytes,
                      uint64_t* SharedMemoryHandle) {
	CHECK_KFD_OPEN();

	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtHandleImport(const HsaExternalHandleDesc* import_desc,
    					HsaHandleImportResult* import_res, HsaHandleImportFlags* flags)
{
	CHECK_KFD_OPEN();
	amdgpu_device_handle devhandle =  (amdgpu_device_handle)import_desc->device_handle;
	enum amdgpu_bo_handle_type type;
	switch (import_desc->type) {
	case HSA_EXTERNAL_HANDLE_GEM_FLINK_NAME:
		type = amdgpu_bo_handle_type_gem_flink_name;
		break;
	case HSA_EXTERNAL_HANDLE_KMS:
		type = amdgpu_bo_handle_type_kms;
		break;
	case HSA_EXTERNAL_HANDLE_DMA_BUF:
	default:
		type = amdgpu_bo_handle_type_dma_buf_fd;
		break;
	}
	struct amdgpu_bo_import_result res;
	int ret = amdgpu_bo_import(devhandle, type, import_desc->fd, &res);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}

	if (flags->ui32.IPCHandle) {
		//query buffer object for pre existing metadata
    	struct amdgpu_bo_info info = {0};
		ret = amdgpu_bo_query_info(res.buf_handle, &info);
		if (ret) {
			return HSAKMT_STATUS_INVALID_HANDLE;
		}
		uint32_t metadata = info.metadata.umd_metadata[0];
		uint32_t size_metadata = info.metadata.size_metadata;
		if (flags->ui32.UpdateMetadata && !flags->ui32.SysMem) {
			if (!!size_metadata) { // return pre-existing metadata
				import_res->metadata = (HSAuint32)metadata;
			} else {
    			struct amdgpu_bo_metadata buf_info = {0};
    			buf_info.size_metadata = sizeof(HSAuint32);
    			buf_info.umd_metadata[0] = (uint32_t)import_desc->metadata;
    			amdgpu_bo_set_metadata(res.buf_handle, &buf_info);
			}
		} else if (import_desc->metadata != metadata) {
			import_res->metadata = (HSAuint32)metadata;
			return HSAKMT_STATUS_INVALID_PARAMETER;
		}
	}

	import_res->buf_handle = (HsaMemoryObjectHandle)res.buf_handle;
	import_res->alloc_size = (HSAuint64)res.alloc_size;
	return HSAKMT_STATUS_SUCCESS;
}

HSAuint64 MapDrmPerm(HsaMemoryMapFlags flags) {
  switch (flags) {
  case HSA_MEMORY_ACCESS_RO:
    return AMDGPU_VM_PAGE_READABLE;
  case HSA_MEMORY_ACCESS_WO:
    return AMDGPU_VM_PAGE_WRITEABLE;
  case HSA_MEMORY_ACCESS_RW:
    return AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
  case HSA_MEMORY_ACCESS_NONE:
  default:
    return 0;
  }
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaMap(HsaMemoryObjectHandle Handle,
    					HSAuint64 offset, HSAuint64 size, HSAuint64 addr,
						HsaMemoryMapFlags flags)
{
	CHECK_KFD_OPEN();
	amdgpu_bo_handle drmhandle = (amdgpu_bo_handle)(Handle);
    if (!drmhandle) {
    	return HSAKMT_STATUS_ERROR;
	}

    int ret = amdgpu_bo_va_op(drmhandle, offset, size, addr,
                      		  MapDrmPerm(flags), AMDGPU_VA_OP_MAP);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryVaUnmap(HsaMemoryObjectHandle Handle,
    					HSAuint64 offset, HSAuint64 size, HSAuint64 addr)
{
	CHECK_KFD_OPEN();
	amdgpu_bo_handle drmhandle = (amdgpu_bo_handle)(Handle);
    if (!drmhandle) {
    	return HSAKMT_STATUS_ERROR;
	}

    int ret = amdgpu_bo_va_op(drmhandle, offset, size, addr, 0,
							  AMDGPU_VA_OP_UNMAP);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemHandleFree(HsaMemoryObjectHandle Handle)
{
	CHECK_KFD_OPEN();
	// Reset metadata for the handle
    struct amdgpu_bo_metadata zero_metadata = {0};
    memset(zero_metadata.umd_metadata, 0, sizeof(uint32_t));
    int ret = amdgpu_bo_set_metadata((amdgpu_bo_handle)Handle, &zero_metadata);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}
	ret = amdgpu_bo_free((amdgpu_bo_handle)Handle);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryCpuMap(HsaMemoryObjectHandle Handle,
						void** out_cpu_ptr)
{
	int ret = amdgpu_bo_cpu_map((amdgpu_bo_handle)Handle, out_cpu_ptr);
	if (ret) {
		return HSAKMT_STATUS_ERROR;
	}
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtMemoryGetCpuAddr(HsaAMDGPUDeviceHandle DeviceHandle,
  						HsaMemoryObjectHandle MemoryHandle, HSAint32* fd, HSAuint64* cpu_addr)
{
  	amdgpu_device_handle devhandle = (amdgpu_device_handle)DeviceHandle;
  	int renderFd = amdgpu_device_get_fd(devhandle);
  	if (renderFd < 0) return HSAKMT_STATUS_ERROR;

  	uint32_t gem_handle = 0;
  	int ret = amdgpu_bo_export((amdgpu_bo_handle)MemoryHandle, amdgpu_bo_handle_type_kms, &gem_handle);
  	if (ret) {
  			return HSAKMT_STATUS_ERROR;
  	}

  	union drm_amdgpu_gem_mmap args;
  	memset(&args, 0, sizeof(args));
  	/* Query the buffer address (args.addr_ptr).
  	 * The kernel driver ignores the offset and size parameters. */
  	args.in.handle = gem_handle;
  	ret = drmCommandWriteRead(renderFd, DRM_AMDGPU_GEM_MMAP, &args, sizeof(args));
  	if (ret) {
  	  return HSAKMT_STATUS_ERROR;
  	}
  	*fd = (HSAint32)renderFd;
  	*cpu_addr = (HSAuint64)args.out.addr_ptr;
  	return HSAKMT_STATUS_SUCCESS;
}