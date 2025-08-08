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

#ifndef FMM_H_
#define FMM_H_

#include "hsakmt/hsakmttypes.h"
#include <stddef.h>

typedef enum {
	FMM_FIRST_APERTURE_TYPE = 0,
	FMM_GPUVM = FMM_FIRST_APERTURE_TYPE,
	FMM_LDS,
	FMM_SCRATCH,
	FMM_SVM,
	FMM_MMIO,
	FMM_LAST_APERTURE_TYPE
} aperture_type_e;

typedef struct {
	aperture_type_e app_type;
	uint64_t size;
	void *start_address;
} aperture_properties_t;

HSAKMT_STATUS hsakmt_fmm_get_amdgpu_device_handle(uint32_t node_id,  HsaAMDGPUDeviceHandle *DeviceHandle);
HSAKMT_STATUS hsakmt_fmm_init_process_apertures(unsigned int NumNodes);
void hsakmt_fmm_destroy_process_apertures(void);

/* Memory interface */
void *hsakmt_fmm_allocate_scratch(uint32_t gpu_id, void *address, uint64_t MemorySizeInBytes);
void *hsakmt_fmm_allocate_device(uint32_t gpu_id, uint32_t node_id, void *address,
			uint64_t MemorySizeInBytes, uint64_t alignment, HsaMemFlags flags);
void *hsakmt_fmm_allocate_doorbell(uint32_t gpu_id, uint64_t MemorySizeInBytes, uint64_t doorbell_offset);
void *hsakmt_fmm_allocate_host(uint32_t gpu_id, uint32_t node_id, void *address, uint64_t MemorySizeInBytes,
			uint64_t alignment, HsaMemFlags flags);
void hsakmt_fmm_print(uint32_t node);
HSAKMT_STATUS hsakmt_fmm_release(void *address);
HSAKMT_STATUS hsakmt_fmm_map_to_gpu(void *address, uint64_t size, uint64_t *gpuvm_address);
int hsakmt_fmm_unmap_from_gpu(void *address);
bool hsakmt_fmm_get_handle(void *address, uint64_t *handle);
HSAKMT_STATUS hsakmt_fmm_get_mem_info(const void *address, HsaPointerInfo *info);
HSAKMT_STATUS hsakmt_fmm_set_mem_user_data(const void *mem, void *usr_data);
#ifdef SANITIZER_AMDGPU
HSAKMT_STATUS hsakmt_fmm_replace_asan_header_page(void* address);
HSAKMT_STATUS hsakmt_fmm_return_asan_header_page(void* address);
#endif

/* Topology interface*/
HSAKMT_STATUS hsakmt_fmm_get_aperture_base_and_limit(aperture_type_e aperture_type, HSAuint32 gpu_id,
		HSAuint64 *aperture_base, HSAuint64 *aperture_limit);

HSAKMT_STATUS hsakmt_fmm_register_memory(void *address, uint64_t size_in_bytes,
								  uint32_t *gpu_id_array,
								  uint32_t gpu_id_array_size,
								  bool coarse_grain,
								  bool ext_coherent);
HSAKMT_STATUS hsakmt_fmm_register_graphics_handle(HSAuint64 GraphicsResourceHandle,
					   HsaGraphicsResourceInfo *GraphicsResourceInfo,
					   uint32_t *gpu_id_array,
					   uint32_t gpu_id_array_size,
					   HSA_REGISTER_MEM_FLAGS RegisterFlags);
HSAKMT_STATUS hsakmt_fmm_deregister_memory(void *address);
HSAKMT_STATUS hsakmt_fmm_export_dma_buf_fd(void *MemoryAddress,
				    HSAuint64 MemorySizeInBytes,
				    int *DMABufFd,
				    HSAuint64 *Offset);
HSAKMT_STATUS hsakmt_fmm_share_memory(void *MemoryAddress,
			       HSAuint64 SizeInBytes,
			       HsaSharedMemoryHandle *SharedMemoryHandle);
HSAKMT_STATUS hsakmt_fmm_register_shared_memory(const HsaSharedMemoryHandle *SharedMemoryHandle,
					 HSAuint64 *SizeInBytes,
					 void **MemoryAddress,
					 uint32_t *gpu_id_array,
					 uint32_t gpu_id_array_size);
HSAKMT_STATUS hsakmt_fmm_map_to_gpu_nodes(void *address, uint64_t size,
		uint32_t *nodes_to_map, uint64_t num_of_nodes, uint64_t *gpuvm_address);

int hsakmt_open_drm_render_device(int minor);
void *hsakmt_mmap_allocate_aligned(int prot, int flags, uint64_t size, uint64_t align,
			    uint64_t guard_size, void *aper_base, void *aper_limit, int fd);

extern int (*hsakmt_fn_amdgpu_device_get_fd)(HsaAMDGPUDeviceHandle device_handle);
#endif /* FMM_H_ */
