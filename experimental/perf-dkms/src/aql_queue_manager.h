/*
 * aql_queue_manager.h - AQL GPU Queue Lifecycle Management
 *
 * Manages the full lifecycle of AQL compute queues for GPU performance
 * counter collection. Uses KFD ioctls via the bridge to allocate GPU
 * memory, create queues, and map doorbells. The hot path (submit + wait)
 * operates on pinned kernel VAs without any ioctl calls.
 */

#ifndef _AQL_QUEUE_MANAGER_H
#define _AQL_QUEUE_MANAGER_H

#include <linux/types.h>
#include <linux/mutex.h>
#include <linux/mm_types.h>
#include "aql_c/aql_queue.h"

/* Buffer sizing */
#define AQL_QUEUE_RING_SIZE     (64 * 1024)   /* 64KB ring buffer */
#define AQL_QUEUE_IB_SIZE       (64 * 1024)   /* 64KB IB buffer */
#define AQL_QUEUE_DATA_SIZE     (4 * 1024)    /* 4KB data buffer per counter block */
#define AQL_QUEUE_EOP_SIZE      (4 * 1024)    /* 4KB EOP buffer */
#define AQL_QUEUE_SIGNAL_SIZE   PAGE_SIZE     /* Signal buffer (1 page, 8 bytes used) */

/* Number of GPU memory buffers per queue */
#define AQL_QUEUE_NUM_BUFFERS   7

/* Buffer indices */
#define AQL_BUF_RING    0
#define AQL_BUF_IB      1
#define AQL_BUF_DATA    2
#define AQL_BUF_EOP     3
#define AQL_BUF_SIGNAL  4
#define AQL_BUF_CWSR    5
#define AQL_BUF_WPTR    6  /* GTT buffer for wptr/rptr (GART-pinnable) */

/**
 * struct aql_gpu_buffer - Tracks a single GPU memory allocation
 */
struct aql_gpu_buffer {
	uint64_t handle;          /* KFD memory handle (for cleanup) */
	uint64_t gpu_addr;        /* GPU virtual address */
	uint64_t mmap_offset;    /* KFD mmap offset (for vm_mmap to CPU-map) */
	void    *cpu_addr;        /* Kernel VA (pinned + vmapped) */
	uint32_t size;            /* Buffer size in bytes */
	unsigned long user_addr;  /* User VA from vm_mmap (for cleanup) */

	/* Page pinning tracking */
	struct page **pages;      /* Pinned pages array */
	int nr_pages;             /* Number of pinned pages */
	bool is_gtt;              /* GTT alloc (pages from get_user_pages) */
};

/**
 * struct aql_gpu_queue - Per-GPU AQL queue context
 *
 * Contains all state needed for submitting PM4 commands to a GPU
 * via an AQL compute queue. After initialization, the hot path
 * (submit + wait) uses only kernel VAs and MMIO writes.
 */
struct aql_gpu_queue {
	/* Queue identity */
	uint32_t gpu_id;
	uint32_t queue_id;        /* Assigned by KFD CREATE_QUEUE */

	/* Ring buffer and IB pool (pure C, from aql_queue.h) */
	aql_ring_t    ring;
	aql_ib_pool_t ib_pool;

	/* Data buffer for counter results */
	void    *data_buf;        /* Kernel VA */
	uint64_t data_gpu_addr;   /* GPU VA */
	uint32_t data_size;

	/* Completion signal (allocated as GTT, like wptr)
	 * GPU firmware writes to this via AQL completion_signal mechanism.
	 * Must be GTT (not USERPTR) because GPU cannot write to USERPTR buffers.
	 * Accessed via get_user/put_user with kthread_use_mm. */
	volatile uint64_t *signal;      /* User VA for polling (value field at +8) */
	uint64_t           signal_gpu_addr;

	/* Doorbell (MMIO via KFD mmap).
	 * Doorbell is VM_PFNMAP (io_remap_pfn_range), so no struct pages.
	 * We walk page tables during init to get phys addr, then ioremap_wc
	 * for direct kernel MMIO access. User VA kept for vm_munmap cleanup. */
	unsigned long doorbell_user_addr;  /* User VA of doorbell page (for cleanup) */
	uint32_t doorbell_byte_offset;    /* Byte offset within mmap for our queue */
	uint32_t doorbell_mmap_size;      /* Size of doorbell mmap (for munmap) */
	uint64_t doorbell_offset;         /* Raw value from KFD CREATE_QUEUE */
	void __iomem *doorbell_kaddr;     /* Kernel MMIO VA from ioremap_wc */

	/* Write/read pointer addresses (must be in user-accessible memory
	 * for HWS/MES to access via GPU page tables) */
	volatile uint64_t *wptr_addr;   /* Write pointer (user-mapped) */
	volatile uint64_t *rptr_addr;   /* Read pointer (user-mapped) */

	/* GPU buffer allocations (for cleanup) */
	struct aql_gpu_buffer bufs[AQL_QUEUE_NUM_BUFFERS];

	/* CWSR sizes (from KFD topology, needed for CREATE_QUEUE) */
	uint32_t ctl_stack_size;
	uint32_t cwsr_size;        /* ctx_save_restore_size reported to KFD */
	uint32_t cwsr_alloc_size;  /* actual buffer alloc (cwsr_size + debug_mem) */

	/* DRM render node (for mmap of GPU buffers) */
	int render_minor;         /* e.g. 128 for renderD128 */
	struct file *drm_filp;    /* DRM render node file (kept open for mmap) */
	int drm_fd;               /* DRM fd in process fd table (-1 if closed) */

	/* Process mm (for kthread_use_mm to access GTT wptr page) */
	struct mm_struct *mm;

	/* Header-polling completion tracking.
	 * MES invalidates the AQL packet header (writes 0x0001 = INVALID)
	 * after processing. We poll this instead of using completion_signal
	 * because MES crashes on GFX12 when completion_signal != 0. */
	uint64_t last_submit_slot_offset;  /* byte offset of last submitted slot in ring */

	/* State */
	bool initialized;
	struct mutex lock;
};

/**
 * aql_queue_create - Create and initialize a GPU AQL queue
 * @queue: Queue structure to initialize
 * @kfd_filp: Open /dev/kfd file pointer
 * @gpu_id: KFD GPU ID
 *
 * Full setup sequence:
 *   1. ALLOC_MEMORY_OF_GPU x5 via KFD bridge
 *   2. MAP_MEMORY_TO_GPU via KFD bridge
 *   3. vm_mmap buffers to get user VAs
 *   4. pin_user_pages + vmap for kernel VAs
 *   5. CREATE_QUEUE via KFD bridge -> doorbell_offset
 *   6. Map doorbell via vm_mmap + pin
 *
 * Must be called from process context with valid current->mm.
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_queue_create(struct aql_gpu_queue *queue,
		     struct file *kfd_filp, uint32_t gpu_id);

/**
 * aql_queue_destroy - Tear down a GPU AQL queue
 * @queue: Queue to destroy
 * @kfd_filp: Open /dev/kfd file pointer
 *
 * Reverse of aql_queue_create:
 *   1. DESTROY_QUEUE via KFD bridge
 *   2. UNMAP_MEMORY + FREE_MEMORY via KFD bridge
 *   3. vunmap + unpin pages
 *
 * Must be called from process context with valid current->mm.
 */
void aql_queue_destroy(struct aql_gpu_queue *queue,
		       struct file *kfd_filp);

/**
 * aql_queue_submit - Submit PM4 commands via AQL queue (hot path)
 * @queue: AQL queue
 * @pm4_data: PM4 command buffer (DWORD array)
 * @pm4_size_dw: Number of DWORDs in the PM4 buffer
 * @out_dispatch_idx: Output: dispatch index to wait for completion.
 *                    Caller must pass this to aql_queue_wait().
 *                    Can be NULL if caller doesn't need to wait.
 *
 * Hot path - no ioctl calls, no mm needed:
 *   1. Allocate IB space from pool
 *   2. Copy PM4 data to IB
 *   3. Build AQL PM4 IB packet
 *   4. Write packet to ring
 *   5. Update write pointer
 *   6. Ring doorbell
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_queue_submit(struct aql_gpu_queue *queue,
		     const uint32_t *pm4_data, uint32_t pm4_size_dw,
		     uint64_t *out_dispatch_idx);

/**
 * aql_queue_wait - Wait for GPU to complete a specific submission
 * @queue: AQL queue
 * @expected_dispatch_idx: Dispatch index returned by aql_queue_submit()
 * @timeout_ms: Maximum wait time in milliseconds
 *
 * Polls the rptr (dispatch index) until it reaches the expected value.
 * Uses progressive backoff: spin -> udelay -> usleep_range.
 *
 * Returns: 0 on completion, -ETIMEDOUT if timeout exceeded
 */
int aql_queue_wait(struct aql_gpu_queue *queue,
		   uint64_t expected_dispatch_idx,
		   unsigned long timeout_ms);

/**
 * aql_queue_get_data_buffer - Get the data buffer for counter results
 * @queue: AQL queue
 * @out_cpu: Output: kernel VA of data buffer
 * @out_gpu: Output: GPU VA of data buffer
 *
 * Returns: 0 on success, -EINVAL if queue not initialized
 */
int aql_queue_get_data_buffer(struct aql_gpu_queue *queue,
			      void **out_cpu, uint64_t *out_gpu);

#endif /* _AQL_QUEUE_MANAGER_H */
