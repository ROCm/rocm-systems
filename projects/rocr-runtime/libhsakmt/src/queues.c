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
#include "fmm.h"
#include "hsakmt/linux/kfd_ioctl.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

/* 1024 doorbells, 4 or 8 bytes each doorbell depending on ASIC generation */
#define DOORBELL_SIZE(gfxv)	(((gfxv) >= 0x90000) ? 8 : 4)
#define DOORBELLS_PAGE_SIZE(ds)	(1024 * (ds))

#define WG_CONTEXT_DATA_SIZE_PER_CU(gfxv, node) 		\
	(hsakmt_get_vgpr_size_per_cu(gfxv) + SGPR_SIZE_PER_CU +	\
	 (node.LDSSizeInKB << 10) + HWREG_SIZE_PER_CU)

#define CNTL_STACK_BYTES_PER_WAVE(gfxv)	\
	((gfxv) >= GFX_VERSION_NAVI10 ? 12 : 8)

#define HWREG_SIZE_PER_CU	0x1000
#define DEBUGGER_BYTES_ALIGN	64
#define DEBUGGER_BYTES_PER_WAVE	32

struct queue {
	uint32_t queue_id;
	uint64_t wptr;
	uint64_t rptr;
	void *eop_buffer;
	void *ctx_save_restore;
	uint32_t ctx_save_restore_size;
	uint32_t ctl_stack_size;
	uint32_t debug_memory_size;
	uint32_t eop_buffer_size;
	uint32_t total_mem_alloc_size;
	uint32_t gfxv;
	bool use_ats;
	bool unified_ctx_save_restore;
	/* This queue structure is allocated from GPU with page aligned size
	 * but only small bytes are used. We use the extra space in the end for
	 * cu_mask bits array.
	 */
	uint32_t cu_mask_count; /* in bits */
	uint32_t cu_mask[0];
};

struct process_doorbells {
	bool use_gpuvm;
	uint32_t size;
	void *mapping;
	pthread_mutex_t mutex;
};

static unsigned int num_doorbells;
static struct process_doorbells *doorbells;

uint32_t hsakmt_get_vgpr_size_per_cu(uint32_t gfxv)
{
	uint32_t vgpr_size = 0x40000;

	if (gfxv == GFX_VERSION_GFX950 ||
		(gfxv & ~(0xff)) == GFX_VERSION_AQUA_VANJARAM ||
		 gfxv == GFX_VERSION_ALDEBARAN ||
		 gfxv == GFX_VERSION_ARCTURUS)
		vgpr_size = 0x80000;

	else if (gfxv == GFX_VERSION_PLUM_BONITO ||
		 gfxv == GFX_VERSION_WHEAT_NAS ||
		 gfxv == GFX_VERSION_GFX1200 ||
		 gfxv == GFX_VERSION_GFX1201)
		vgpr_size = 0x60000;

	return vgpr_size;
}

HSAKMT_STATUS hsakmt_init_process_doorbells(unsigned int NumNodes)
{
	unsigned int i;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	/* doorbells[] is accessed using Topology NodeId. This means doorbells[0],
	 * which corresponds to CPU only Node, might not be used
	 */
	doorbells = malloc(NumNodes * sizeof(struct process_doorbells));
	if (!doorbells)
		return HSAKMT_STATUS_NO_MEMORY;

	for (i = 0; i < NumNodes; i++) {
		doorbells[i].use_gpuvm = false;
		doorbells[i].size = 0;
		doorbells[i].mapping = NULL;
		pthread_mutex_init(&doorbells[i].mutex, NULL);
	}

	num_doorbells = NumNodes;

	return ret;
}

static void get_doorbell_map_info(uint32_t node_id,
				  struct process_doorbells *doorbell)
{
	/*
	 * GPUVM doorbell on Tonga requires a workaround for VM TLB ACTIVE bit
	 * lookup bug. Remove ASIC check when this is implemented in amdgpu.
	 */
	uint32_t gfxv = hsakmt_get_gfxv_by_node_id(node_id);
	doorbell->use_gpuvm = (hsakmt_is_dgpu && gfxv != GFX_VERSION_TONGA);
	doorbell->size = DOORBELLS_PAGE_SIZE(DOORBELL_SIZE(gfxv));

	if (doorbell->size < (uint32_t) PAGE_SIZE) {
		doorbell->size = PAGE_SIZE;
	}

	return;
}

void hsakmt_destroy_process_doorbells(void)
{
	unsigned int i;

	if (!doorbells)
		return;

	for (i = 0; i < num_doorbells; i++) {
		if (!doorbells[i].size)
			continue;

		if (doorbells[i].use_gpuvm) {
			hsakmt_fmm_unmap_from_gpu(doorbells[i].mapping);
			hsakmt_fmm_release(doorbells[i].mapping);
		} else
			munmap(doorbells[i].mapping, doorbells[i].size);
	}

	free(doorbells);
	doorbells = NULL;
	num_doorbells = 0;
}

/* This is a special funcion that should be called only from the child process
 * after a fork(). This will clear doorbells duplicated from the parent.
 */
void hsakmt_clear_process_doorbells(void)
{
	unsigned int i;

	if (!doorbells)
		return;

	for (i = 0; i < num_doorbells; i++) {
		if (!doorbells[i].size)
			continue;

		if (!doorbells[i].use_gpuvm)
			munmap(doorbells[i].mapping, doorbells[i].size);
	}

	free(doorbells);
	doorbells = NULL;
	num_doorbells = 0;
}

static HSAKMT_STATUS map_doorbell_apu(HSAuint32 NodeId, HSAuint32 gpu_id,
				      HSAuint64 doorbell_mmap_offset)
{
	void *ptr;

	ptr = mmap(0, doorbells[NodeId].size, PROT_READ|PROT_WRITE,
		   MAP_SHARED, hsakmt_kfd_fd, doorbell_mmap_offset);

	if (ptr == MAP_FAILED)
		return HSAKMT_STATUS_ERROR;

	doorbells[NodeId].mapping = ptr;

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS map_doorbell_dgpu(HSAuint32 NodeId, HSAuint32 gpu_id,
				       HSAuint64 doorbell_mmap_offset)
{
	void *ptr;

	ptr = hsakmt_fmm_allocate_doorbell(gpu_id, doorbells[NodeId].size,
				doorbell_mmap_offset);

	if (!ptr)
		return HSAKMT_STATUS_ERROR;

	/* map for GPU access */
	if (hsakmt_fmm_map_to_gpu(ptr, doorbells[NodeId].size, NULL)) {
		hsakmt_fmm_release(ptr);
		return HSAKMT_STATUS_ERROR;
	}

	doorbells[NodeId].mapping = ptr;

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS map_doorbell(HSAuint32 NodeId, HSAuint32 gpu_id,
				  HSAuint64 doorbell_mmap_offset)
{
	HSAKMT_STATUS status = HSAKMT_STATUS_SUCCESS;

	pthread_mutex_lock(&doorbells[NodeId].mutex);
	if (doorbells[NodeId].size) {
		pthread_mutex_unlock(&doorbells[NodeId].mutex);
		return HSAKMT_STATUS_SUCCESS;
	}

	get_doorbell_map_info(NodeId, &doorbells[NodeId]);

	if (doorbells[NodeId].use_gpuvm) {
		status = map_doorbell_dgpu(NodeId, gpu_id, doorbell_mmap_offset);
		if (status != HSAKMT_STATUS_SUCCESS) {
			/* Fall back to the old method if KFD doesn't
			 * support doorbells in GPUVM
			 */
			doorbells[NodeId].use_gpuvm = false;
			status = map_doorbell_apu(NodeId, gpu_id, doorbell_mmap_offset);
		}
	} else
		status = map_doorbell_apu(NodeId, gpu_id, doorbell_mmap_offset);

	if (status != HSAKMT_STATUS_SUCCESS)
		doorbells[NodeId].size = 0;

	pthread_mutex_unlock(&doorbells[NodeId].mutex);

	return status;
}

static void *allocate_exec_aligned_memory_cpu(uint32_t size)
{
	void *ptr;

	/* mmap will return a pointer with alignment equal to
	 * sysconf(_SC_PAGESIZE).
	 *
	 * MAP_ANONYMOUS initializes the memory to zero.
	 */
	ptr = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
				MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

	if (ptr == MAP_FAILED)
		return NULL;
	return ptr;
}

/* The bool return indicate whether the queue needs a context-save-restore area*/
static bool update_ctx_save_restore_size(uint32_t nodeid, struct queue *q)
{
	HsaNodeProperties node;

	if (q->gfxv < GFX_VERSION_CARRIZO)
		return false;
	if (hsaKmtGetNodeProperties(nodeid, &node))
		return false;
	if (node.NumFComputeCores && node.NumSIMDPerCU) {
		uint32_t ctl_stack_size, wg_data_size;
		uint32_t cu_num = node.NumFComputeCores / node.NumSIMDPerCU / node.NumXcc;
		uint32_t wave_num = (q->gfxv < GFX_VERSION_NAVI10)
			? MIN(cu_num * 40, node.NumShaderBanks / node.NumArrays * 512)
			: cu_num * 32;

		ctl_stack_size = wave_num * CNTL_STACK_BYTES_PER_WAVE(q->gfxv) + 8;
		wg_data_size = cu_num * WG_CONTEXT_DATA_SIZE_PER_CU(q->gfxv, node);
		q->ctl_stack_size = PAGE_ALIGN_UP(sizeof(HsaUserContextSaveAreaHeader)
					+ ctl_stack_size);
		if ((q->gfxv & 0x3f0000) == 0xA0000) {
			/* HW design limits control stack size to 0x7000.
			 * This is insufficient for theoretical PM4 cases
			 * but sufficient for AQL, limited by SPI events.
			 */
			q->ctl_stack_size = MIN(q->ctl_stack_size, 0x7000);
		}

		q->debug_memory_size =
			ALIGN_UP(wave_num * DEBUGGER_BYTES_PER_WAVE, DEBUGGER_BYTES_ALIGN);

		q->ctx_save_restore_size = q->ctl_stack_size
					+ PAGE_ALIGN_UP(wg_data_size);
		return true;
	}
	return false;
}

void *hsakmt_allocate_exec_aligned_memory_gpu(uint32_t size, uint32_t align, uint32_t gpu_id,
				       uint32_t NodeId, bool nonPaged,
				       bool DeviceLocal,
				       bool Uncached)
{
	void *mem = NULL;
	HSAuint64 gpu_va;
	HsaMemFlags flags;
	HSAuint32 cpu_id = 0;

	flags.Value = 0;
	flags.ui32.HostAccess = !DeviceLocal;
	flags.ui32.ExecuteAccess = 1;
	flags.ui32.NonPaged = nonPaged;
	flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
	flags.ui32.CoarseGrain = DeviceLocal;
	flags.ui32.Uncached = Uncached;

	size = ALIGN_UP(size, align);

	if (DeviceLocal && !hsakmt_zfb_support)
		mem = hsakmt_fmm_allocate_device(gpu_id, NodeId, mem, size, 0, flags);
	else {
		/* VRAM under ZFB mode should be supported here without any
		 * additional code
		 */
		/* Get the closest cpu_id to GPU NodeId for system memory allocation
		 * nonPaged=0 system memory allocation uses GTT path
		 */
		if (!nonPaged) {
			cpu_id = hsakmt_get_direct_link_cpu(NodeId);
			if (cpu_id == INVALID_NODEID) {
				flags.ui32.NoNUMABind = 1;
				cpu_id = 0;
			}
		}
		mem = hsakmt_fmm_allocate_host(gpu_id, cpu_id, mem, size, 0, flags);
	}

	if (!mem) {
		pr_err("Alloc %s memory failed size %d\n",
		       DeviceLocal ? "VRAM" : "GTT", size);
		return NULL;
	}

	if (NodeId != 0) {
		uint32_t nodes_array[1] = {NodeId};
		HsaMemMapFlags map_flags = {0};
		HSAKMT_STATUS result;

		result = hsaKmtMapMemoryToGPUNodes(mem, size, &gpu_va, map_flags, 1, nodes_array);
		if (result != HSAKMT_STATUS_SUCCESS) {
			hsaKmtFreeMemory(mem, size);
			return NULL;
		}

		return mem;
	}

	if (hsaKmtMapMemoryToGPU(mem, size, &gpu_va) != HSAKMT_STATUS_SUCCESS) {
		hsaKmtFreeMemory(mem, size);
		return NULL;
	}

	return mem;
}

void hsakmt_free_exec_aligned_memory_gpu(void *addr, uint32_t size, uint32_t align)
{
	size = ALIGN_UP(size, align);

	if (hsaKmtUnmapMemoryToGPU(addr) == HSAKMT_STATUS_SUCCESS)
		hsaKmtFreeMemory(addr, size);
}

/*
 * Allocates memory aligned to sysconf(_SC_PAGESIZE)
 */
static void *allocate_exec_aligned_memory(uint32_t size,
					  bool use_ats,
					  uint32_t gpu_id,
					  uint32_t NodeId,
					  bool nonPaged,
					  bool DeviceLocal,
					  bool Uncached)
{
	if (!use_ats)
		return hsakmt_allocate_exec_aligned_memory_gpu(size, PAGE_SIZE, gpu_id, NodeId,
							nonPaged, DeviceLocal,
							Uncached);
	return allocate_exec_aligned_memory_cpu(size);
}

static void free_exec_aligned_memory(void *addr, uint32_t size, uint32_t align,
				     bool use_ats)
{
	if (!use_ats)
		hsakmt_free_exec_aligned_memory_gpu(addr, size, align);
	else
		munmap(addr, size);
}

static HSAKMT_STATUS register_svm_range(void *mem, uint32_t size,
				uint32_t gpuNode, uint32_t prefetchNode,
				uint32_t preferredNode, bool alwaysMapped)
{
	HSA_SVM_ATTRIBUTE *attrs;
	HSAuint64 s_attr;
	HSAuint32 nattr;
	HSAuint32 flags;

	flags = HSA_SVM_FLAG_HOST_ACCESS | HSA_SVM_FLAG_GPU_EXEC;

	if (alwaysMapped) {
		CHECK_KFD_MINOR_VERSION(11);
		flags |= HSA_SVM_FLAG_GPU_ALWAYS_MAPPED;
	}

	nattr = 6;
	s_attr = sizeof(*attrs) * nattr;
	attrs = (HSA_SVM_ATTRIBUTE *)alloca(s_attr);

	attrs[0].type = HSA_SVM_ATTR_PREFETCH_LOC;
	attrs[0].value = prefetchNode;
	attrs[1].type = HSA_SVM_ATTR_PREFERRED_LOC;
	attrs[1].value = preferredNode;
	attrs[2].type = HSA_SVM_ATTR_CLR_FLAGS;
	attrs[2].value = ~flags;
	attrs[3].type = HSA_SVM_ATTR_SET_FLAGS;
	attrs[3].value = flags;
	attrs[4].type = HSA_SVM_ATTR_ACCESS;
	attrs[4].value = gpuNode;
	attrs[5].type = HSA_SVM_ATTR_GRANULARITY;
	attrs[5].value = 0xFF;

	return hsaKmtSVMSetAttr(mem, size, nattr, attrs);
}

static void free_queue(struct queue *q)
{
	if (q->eop_buffer)
		free_exec_aligned_memory(q->eop_buffer,
					 q->eop_buffer_size,
					 PAGE_SIZE, q->use_ats);
	if (q->unified_ctx_save_restore)
		munmap(q->ctx_save_restore, q->total_mem_alloc_size);
	else if (q->ctx_save_restore)
		free_exec_aligned_memory(q->ctx_save_restore,
					 q->total_mem_alloc_size,
					 PAGE_SIZE, q->use_ats);

	free_exec_aligned_memory((void *)q, sizeof(*q), PAGE_SIZE, q->use_ats);
}

static inline void fill_cwsr_header(struct queue *q, void *addr,
		HsaEvent *Event, volatile HSAint64 *ErrPayload, HSAuint32 NumXcc)
{
	uint32_t i;
	HsaUserContextSaveAreaHeader *header;

	for (i = 0; i < NumXcc; i++) {
		header = (HsaUserContextSaveAreaHeader *)
			((uintptr_t)addr + (i * q->ctx_save_restore_size));
		header->ErrorEventId = 0;
		if (Event)
			header->ErrorEventId = Event->EventId;
		header->ErrorReason = ErrPayload;
		header->DebugOffset = (NumXcc - i) * q->ctx_save_restore_size;
		header->DebugSize = q->debug_memory_size * NumXcc;
	}
}

static int handle_concrete_asic(struct queue *q,
				struct kfd_ioctl_create_queue_args *args,
				uint32_t gpu_id,
				uint32_t NodeId,
				HsaEvent *Event,
				volatile HSAint64 *ErrPayload)
{
	bool ret;

	if (args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA ||
	    args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA_XGMI)
		return HSAKMT_STATUS_SUCCESS;

	if (q->eop_buffer_size > 0) {
		pr_info("Allocating VRAM for EOP\n");
		q->eop_buffer = allocate_exec_aligned_memory(q->eop_buffer_size,
				q->use_ats, gpu_id,
				NodeId, true, true, /* Unused for VRAM */false);
		if (!q->eop_buffer)
			return HSAKMT_STATUS_NO_MEMORY;

		args->eop_buffer_address = (uintptr_t)q->eop_buffer;
		args->eop_buffer_size = q->eop_buffer_size;
	}

	ret = update_ctx_save_restore_size(NodeId, q);

	if (ret) {
		HsaNodeProperties node;

		if (hsaKmtGetNodeProperties(NodeId, &node))
			return HSAKMT_STATUS_ERROR;

		args->ctx_save_restore_size = q->ctx_save_restore_size;
		args->ctl_stack_size = q->ctl_stack_size;

		/* Total memory to be allocated is =
		 * (Control Stack size + WG size +
		 *  Debug memory area size) * num_xcc
		 */
		q->total_mem_alloc_size = (q->ctx_save_restore_size +
					q->debug_memory_size) * node.NumXcc;

		/* Allocate unified memory for context save restore
		 * area on dGPU.
		 */
		if (!q->use_ats && hsakmt_is_svm_api_supported) {
			uint32_t size = PAGE_ALIGN_UP(q->total_mem_alloc_size);

			pr_info("Allocating GTT for CWSR\n");
			void *addr = hsakmt_mmap_allocate_aligned(PROT_READ | PROT_WRITE,
						     MAP_ANONYMOUS | MAP_PRIVATE,
						     size, GPU_HUGE_PAGE_SIZE, 0,
						     0, (void *)LONG_MAX, -1);
			if (!addr) {
				pr_err("mmap failed to alloc ctx area size 0x%x: %s\n",
					size, strerror(errno));
			} else {
				/*
				 * To avoid fork child process COW MMU notifier
				 * callback evict parent process queues.
				 */
				if (madvise(addr, size, MADV_DONTFORK))
					pr_err("madvise failed -%d\n", errno);

				fill_cwsr_header(q, addr, Event, ErrPayload, node.NumXcc);

				HSAKMT_STATUS r = register_svm_range(addr, size,
						NodeId, NodeId, 0, true);

				if (r == HSAKMT_STATUS_SUCCESS) {
					q->ctx_save_restore = addr;
					q->unified_ctx_save_restore = true;
				} else {
					munmap(addr, size);
				}
			}
		}

		if (!q->unified_ctx_save_restore) {
			q->ctx_save_restore = allocate_exec_aligned_memory(
							q->total_mem_alloc_size,
							q->use_ats, gpu_id, NodeId,
							false, false, false);

			if (!q->ctx_save_restore)
				return HSAKMT_STATUS_NO_MEMORY;

			fill_cwsr_header(q, q->ctx_save_restore, Event, ErrPayload, node.NumXcc);
		}

		args->ctx_save_restore_address = (uintptr_t)q->ctx_save_restore;
	}

	return HSAKMT_STATUS_SUCCESS;
}

/* A map to translate thunk queue priority (-3 to +3)
 * to KFD queue priority (0 to 15)
 * Indexed by thunk_queue_priority+3
 */
static uint32_t priority_map[] = {0, 3, 5, 7, 9, 11, 15};

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateQueue(HSAuint32 NodeId,
					  HSA_QUEUE_TYPE Type,
					  HSAuint32 QueuePercentage,
					  HSA_QUEUE_PRIORITY Priority,
					  void *QueueAddress,
					  HSAuint64 QueueSizeInBytes,
					  HsaEvent *Event,
					  HsaQueueResource *QueueResource)
{
	if (Type == HSA_QUEUE_SDMA_BY_ENG_ID)
		return HSAKMT_STATUS_ERROR;

	return hsaKmtCreateQueueExt(NodeId, Type, QueuePercentage, Priority, 0,
				    QueueAddress, QueueSizeInBytes, Event,
				    QueueResource);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateQueueExt(HSAuint32 NodeId,
					     HSA_QUEUE_TYPE Type,
					     HSAuint32 QueuePercentage,
					     HSA_QUEUE_PRIORITY Priority,
					     HSAuint32 SdmaEngineId,
					     void *QueueAddress,
					     HSAuint64 QueueSizeInBytes,
					     HsaEvent *Event,
					     HsaQueueResource *QueueResource)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;
	uint64_t doorbell_mmap_offset;
	unsigned int doorbell_offset;
	int err;
	HsaNodeProperties props;
	uint32_t cu_num, i;

	CHECK_KFD_OPEN();

	if (Priority < HSA_QUEUE_PRIORITY_MINIMUM ||
		Priority > HSA_QUEUE_PRIORITY_MAXIMUM)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	struct queue *q = allocate_exec_aligned_memory(sizeof(*q),
			false, gpu_id, NodeId, true, false, true);
	if (!q)
		return HSAKMT_STATUS_NO_MEMORY;

	memset(q, 0, sizeof(*q));

	q->gfxv = hsakmt_get_gfxv_by_node_id(NodeId);
	q->use_ats = false;

	if (q->gfxv == GFX_VERSION_TONGA)
		q->eop_buffer_size = TONGA_PAGE_SIZE;
	else if ((q->gfxv & ~(0xff)) == GFX_VERSION_AQUA_VANJARAM)
		q->eop_buffer_size = ((Type == HSA_QUEUE_COMPUTE) ? 4096 : 0);
	else if (q->gfxv >= 0x80000)
		q->eop_buffer_size = 4096;

	/* By default, CUs are all turned on. Initialize cu_mask to '1
	 * for all CU bits.
	 */
	if (hsaKmtGetNodeProperties(NodeId, &props))
		q->cu_mask_count = 0;
	else {
		cu_num = props.NumFComputeCores / props.NumSIMDPerCU;
		/* cu_mask_count counts bits. It must be multiple of 32 */
		q->cu_mask_count = ALIGN_UP_32(cu_num, 32);
		for (i = 0; i < cu_num; i++)
			q->cu_mask[i/32] |= (1 << (i % 32));
	}

	struct kfd_ioctl_create_queue_args args = {0};

	args.gpu_id = gpu_id;

	switch (Type) {
	case HSA_QUEUE_COMPUTE:
		args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
		break;
	case HSA_QUEUE_SDMA:
		args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
		break;
	case HSA_QUEUE_SDMA_XGMI:
		args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA_XGMI;
		break;
	case HSA_QUEUE_SDMA_BY_ENG_ID:
		args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID;
		break;
	case HSA_QUEUE_COMPUTE_AQL:
		args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
		break;
	default:
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	if (Type != HSA_QUEUE_COMPUTE_AQL) {
		QueueResource->QueueRptrValue = (uintptr_t)&q->rptr;
		QueueResource->QueueWptrValue = (uintptr_t)&q->wptr;
	}

	err = handle_concrete_asic(q, &args, gpu_id, NodeId, Event, QueueResource->ErrorReason);
	if (err != HSAKMT_STATUS_SUCCESS) {
		free_queue(q);
		return err;
	}

	args.read_pointer_address = QueueResource->QueueRptrValue;
	args.write_pointer_address = QueueResource->QueueWptrValue;
	args.ring_base_address = (uintptr_t)QueueAddress;
	args.ring_size = QueueSizeInBytes;
	args.queue_percentage = QueuePercentage;
	args.queue_priority = priority_map[Priority+3];
	args.sdma_engine_id = SdmaEngineId;

	err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);

	if (err == -1) {
		free_queue(q);
		return HSAKMT_STATUS_ERROR;
	}

	q->queue_id = args.queue_id;

	if (IS_SOC15(q->gfxv)) {
		HSAuint64 mask = DOORBELLS_PAGE_SIZE(DOORBELL_SIZE(q->gfxv)) - 1;

		/* On SOC15 chips, the doorbell offset within the
		 * doorbell page is included in the doorbell offset
		 * returned by KFD. This allows CP queue doorbells to be
		 * allocated dynamically (while SDMA queue doorbells fixed)
		 * rather than based on the its process queue ID.
		 */
		doorbell_mmap_offset = args.doorbell_offset & ~mask;
		doorbell_offset = args.doorbell_offset & mask;
	} else {
		/* On older chips, the doorbell offset within the
		 * doorbell page is based on the queue ID.
		 */
		doorbell_mmap_offset = args.doorbell_offset;
		doorbell_offset = q->queue_id * DOORBELL_SIZE(q->gfxv);
	}

	err = map_doorbell(NodeId, gpu_id, doorbell_mmap_offset);
	if (err != HSAKMT_STATUS_SUCCESS) {
		hsaKmtDestroyQueue(q->queue_id);
		return HSAKMT_STATUS_ERROR;
	}

	QueueResource->QueueId = PORT_VPTR_TO_UINT64(q);
	QueueResource->Queue_DoorBell = VOID_PTR_ADD(doorbells[NodeId].mapping,
						     doorbell_offset);

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUpdateQueue(HSA_QUEUEID QueueId,
					  HSAuint32 QueuePercentage,
					  HSA_QUEUE_PRIORITY Priority,
					  void *QueueAddress,
					  HSAuint64 QueueSize,
					  HsaEvent *Event)
{
	struct kfd_ioctl_update_queue_args arg = {0};
	struct queue *q = PORT_UINT64_TO_VPTR(QueueId);

	CHECK_KFD_OPEN();

	if (Priority < HSA_QUEUE_PRIORITY_MINIMUM ||
		Priority > HSA_QUEUE_PRIORITY_MAXIMUM)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (!q)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	arg.queue_id = (HSAuint32)q->queue_id;
	arg.ring_base_address = (uintptr_t)QueueAddress;
	arg.ring_size = QueueSize;
	arg.queue_percentage = QueuePercentage;
	arg.queue_priority = priority_map[Priority+3];

	int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_UPDATE_QUEUE, &arg);

	if (err == -1)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyQueue(HSA_QUEUEID QueueId)
{
	CHECK_KFD_OPEN();

	struct queue *q = PORT_UINT64_TO_VPTR(QueueId);
	struct kfd_ioctl_destroy_queue_args args = {0};

	if (!q)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	args.queue_id = q->queue_id;

	int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, &args);

	if (err == -1) {
		pr_err("Failed to destroy queue: %s\n", strerror(errno));
		return HSAKMT_STATUS_ERROR;
	}

	free_queue(q);
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetQueueCUMask(HSA_QUEUEID QueueId,
					     HSAuint32 CUMaskCount,
					     HSAuint32 *QueueCUMask)
{
	struct queue *q = PORT_UINT64_TO_VPTR(QueueId);
	struct kfd_ioctl_set_cu_mask_args args = {0};

	CHECK_KFD_OPEN();

	if (CUMaskCount == 0 || !QueueCUMask || ((CUMaskCount % 32) != 0))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	args.queue_id = q->queue_id;
	args.num_cu_mask = CUMaskCount;
	args.cu_mask_ptr = (uintptr_t)QueueCUMask;

	int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SET_CU_MASK, &args);

	if (err == -1)
		return HSAKMT_STATUS_ERROR;

	memcpy(q->cu_mask, QueueCUMask, CUMaskCount / 8);
	q->cu_mask_count = CUMaskCount;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS
HSAKMTAPI
hsaKmtGetQueueInfo(
	HSA_QUEUEID QueueId,
	HsaQueueInfo *QueueInfo
)
{
	struct queue *q = PORT_UINT64_TO_VPTR(QueueId);
	struct kfd_ioctl_get_queue_wave_state_args args = {0};

	CHECK_KFD_OPEN();

	if (QueueInfo == NULL || q == NULL)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (q->ctx_save_restore == NULL)
		return HSAKMT_STATUS_ERROR;

	args.queue_id = q->queue_id;
	args.ctl_stack_address = (uintptr_t)q->ctx_save_restore;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_GET_QUEUE_WAVE_STATE, &args) < 0)
		return HSAKMT_STATUS_ERROR;

	QueueInfo->ControlStackTop = (void *)(args.ctl_stack_address +
				q->ctl_stack_size - args.ctl_stack_used_size);
	QueueInfo->UserContextSaveArea = (void *)
				 (args.ctl_stack_address + q->ctl_stack_size);
	QueueInfo->SaveAreaSizeInBytes = args.save_area_used_size;
	QueueInfo->ControlStackUsedInBytes = args.ctl_stack_used_size;
	QueueInfo->NumCUAssigned = q->cu_mask_count;
	QueueInfo->CUMaskInfo = q->cu_mask;
	QueueInfo->QueueDetailError = 0;
	QueueInfo->QueueTypeExtended = 0;
	QueueInfo->SaveAreaHeader = q->ctx_save_restore;

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetTrapHandler(HSAuint32 Node,
					     void *TrapHandlerBaseAddress,
					     HSAuint64 TrapHandlerSizeInBytes,
					     void *TrapBufferBaseAddress,
					     HSAuint64 TrapBufferSizeInBytes)
{
	struct kfd_ioctl_set_trap_handler_args args = {0};
	HSAKMT_STATUS result;
	uint32_t gpu_id;

	CHECK_KFD_OPEN();

	result = hsakmt_validate_nodeid(Node, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	args.gpu_id = gpu_id;
	args.tba_addr = (uintptr_t)TrapHandlerBaseAddress;
	args.tma_addr = (uintptr_t)TrapBufferBaseAddress;

	int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_SET_TRAP_HANDLER, &args);

	return (err == -1) ? HSAKMT_STATUS_ERROR : HSAKMT_STATUS_SUCCESS;
}

uint32_t *hsakmt_convert_queue_ids(HSAuint32 NumQueues, HSA_QUEUEID *Queues)
{
	uint32_t *queue_ids_ptr;
	unsigned int i;

	if (NumQueues == 0 || Queues == NULL)
		return NULL;

	queue_ids_ptr = malloc(NumQueues * sizeof(uint32_t));
	if (!queue_ids_ptr)
		return NULL;

	for (i = 0; i < NumQueues; i++) {
		struct queue *q = PORT_UINT64_TO_VPTR(Queues[i]);

		if (q == NULL) {
			free(queue_ids_ptr);
			return NULL;
		}

		queue_ids_ptr[i] = q->queue_id;
	}
	return queue_ids_ptr;
}

HSAKMT_STATUS
HSAKMTAPI
hsaKmtAllocQueueGWS(
                HSA_QUEUEID        QueueId,
                HSAuint32          nGWS,
                HSAuint32          *firstGWS)
{
	struct kfd_ioctl_alloc_queue_gws_args args = {0};
	struct queue *q = PORT_UINT64_TO_VPTR(QueueId);

	CHECK_KFD_OPEN();

	args.queue_id = (HSAuint32)q->queue_id;
	args.num_gws = nGWS;

	int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_ALLOC_QUEUE_GWS, &args);

	if (!err && firstGWS)
		*firstGWS = args.first_gws;

	if (!err)
		return HSAKMT_STATUS_SUCCESS;
	else if (errno == EINVAL)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	else if (errno == EBUSY)
		return HSAKMT_STATUS_OUT_OF_RESOURCES;
	else if (errno == ENODEV)
		return HSAKMT_STATUS_NOT_SUPPORTED;
	else
		return HSAKMT_STATUS_ERROR;
}
