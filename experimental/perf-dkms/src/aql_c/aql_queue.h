/**
 * @file aql_queue.h
 * @brief AQL queue ring buffer, IB pool, and PM4 IB packet structures
 *
 * Pure C data structures and functions for AQL queue-based GPU submission.
 * Compiles in both kernel and userspace (USERSPACE_BUILD) contexts.
 */

#ifndef AQL_QUEUE_H
#define AQL_QUEUE_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/string.h>
#else
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#endif

/* AQL queue constants */
#define AQL_RING_DEFAULT_SIZE    (64 * 1024)   /* 64KB default ring size */
#define AQL_IB_POOL_DEFAULT_SIZE (64 * 1024)   /* 64KB default IB pool size */
#define AQL_PACKET_SIZE          64             /* AQL packets are always 64 bytes */
#define AQL_IB_ALIGNMENT         256            /* IB allocations aligned to 256 bytes */

/* HSA packet header constants */
#define HSA_PACKET_TYPE_VENDOR_SPECIFIC 0
#define HSA_PACKET_HEADER_TYPE_SHIFT    0
#define HSA_PACKET_HEADER_BARRIER_SHIFT 8
#define HSA_PACKET_HEADER_ACQUIRE_SHIFT 9
#define HSA_PACKET_HEADER_RELEASE_SHIFT 11

/* PM4 INDIRECT_BUFFER opcode */
#define PACKET3_INDIRECT_BUFFER 0x3F
#define IB_VALID_BIT            (1u << 23)

/* PM4 Type 3 header for INDIRECT_BUFFER */
#define PM4_IB_HEADER (3u << 30 | (PACKET3_INDIRECT_BUFFER << 8) | (2u << 16))

/**
 * Ring buffer context for AQL queue
 *
 * Manages a circular ring buffer where AQL packets are written.
 * The ring size must be a power of 2.
 */
typedef struct {
	void     *base;       /* kernel VA of ring buffer */
	uint64_t  gpu_addr;   /* GPU VA of ring buffer */
	uint32_t  size;       /* ring size in bytes (power of 2) */
	uint64_t  wptr;       /* write pointer (bytes, monotonically increasing) */
	uint64_t  rptr;       /* read pointer (bytes, monotonically increasing) */
} aql_ring_t;

/**
 * IB (Indirect Buffer) pool - bump allocator within a single buffer
 *
 * PM4 commands are written here and referenced by AQL packets via
 * INDIRECT_BUFFER PM4 commands in the ring.
 */
typedef struct {
	void     *base;       /* kernel VA */
	uint64_t  gpu_addr;   /* GPU VA */
	uint32_t  size;       /* total size in bytes */
	uint32_t  offset;     /* current write offset */
} aql_ib_pool_t;

/**
 * 64-byte AQL PM4 IB vendor packet
 *
 * This is the packet format written to the AQL ring buffer.
 * It contains a PM4 INDIRECT_BUFFER command that points to
 * actual PM4 commands in the IB pool.
 */
typedef struct {
	uint16_t header;              /* HSA packet header (type + barrier) */
	uint16_t pm4_ib_format;       /* = 1 (AMD_AQL_PM4_IB_FORMAT) */
	uint32_t pm4_ib_command[4];   /* INDIRECT_BUFFER PM4 command */
	uint32_t dw_count_remain;     /* = 10 (remaining DWORDs after pm4_ib_command) */
	uint32_t reserved[8];         /* reserved, must be 0 */
	uint64_t completion_signal;   /* GPU addr of signal (decremented on completion) */
} __attribute__((packed)) aql_pm4_ib_packet_t;

#ifndef __KERNEL__
/* Userspace static assert */
_Static_assert(sizeof(aql_pm4_ib_packet_t) == 64, "AQL packet must be 64 bytes");
#endif

/* ========================================================================
 * Ring buffer functions
 * ======================================================================== */

/**
 * aql_ring_init - Initialize a ring buffer context
 * @ring: Ring buffer to initialize
 * @base: Kernel VA of the ring buffer memory
 * @gpu_addr: GPU VA of the ring buffer memory
 * @size: Size of the ring buffer in bytes (must be power of 2)
 *
 * Returns: 0 on success, -1 on invalid parameters
 */
int aql_ring_init(aql_ring_t *ring, void *base, uint64_t gpu_addr, uint32_t size);

/**
 * aql_ring_write_packet - Write a 64-byte AQL packet to the ring
 * @ring: Ring buffer
 * @pkt: Packet to write (64 bytes)
 *
 * Copies the packet to the current write slot. The header is written last
 * with a write memory barrier to ensure the GPU sees complete packet data
 * before the header indicates validity.
 *
 * Advances wptr by 64 bytes after writing.
 *
 * Returns: 0 on success, -1 if ring is full
 */
int aql_ring_write_packet(aql_ring_t *ring, const aql_pm4_ib_packet_t *pkt);

/**
 * aql_ring_slot_index - Get current slot index from write pointer
 * @ring: Ring buffer
 *
 * Returns: Current slot index (0 to ring_size/64 - 1)
 */
uint32_t aql_ring_slot_index(const aql_ring_t *ring);

/**
 * aql_ring_available_slots - Get number of available ring slots
 * @ring: Ring buffer
 *
 * Returns: Number of free 64-byte slots
 */
uint32_t aql_ring_available_slots(const aql_ring_t *ring);

/* ========================================================================
 * IB pool functions
 * ======================================================================== */

/**
 * aql_ib_pool_init - Initialize an IB pool
 * @pool: IB pool to initialize
 * @base: Kernel VA of the IB buffer memory
 * @gpu_addr: GPU VA of the IB buffer memory
 * @size: Total size of the IB buffer in bytes
 *
 * Returns: 0 on success, -1 on invalid parameters
 */
int aql_ib_pool_init(aql_ib_pool_t *pool, void *base, uint64_t gpu_addr, uint32_t size);

/**
 * aql_ib_pool_alloc - Allocate a chunk from the IB pool
 * @pool: IB pool
 * @size_bytes: Number of bytes to allocate (will be rounded up to AQL_IB_ALIGNMENT)
 * @out_cpu: Output: kernel VA of the allocated chunk
 * @out_gpu: Output: GPU VA of the allocated chunk
 *
 * Bump allocator with wrapping. When the pool doesn't have enough space
 * at the current offset, it wraps to the beginning.
 *
 * Returns: 0 on success, -1 if size_bytes exceeds pool capacity
 */
int aql_ib_pool_alloc(aql_ib_pool_t *pool, uint32_t size_bytes,
		      void **out_cpu, uint64_t *out_gpu);

/**
 * aql_ib_pool_reset - Reset the IB pool offset to 0
 * @pool: IB pool to reset
 */
void aql_ib_pool_reset(aql_ib_pool_t *pool);

/* ========================================================================
 * AQL PM4 IB packet builder
 * ======================================================================== */

/**
 * aql_build_pm4_ib_packet - Build a 64-byte AQL PM4 IB vendor packet
 * @pkt: Output packet structure (64 bytes)
 * @ib_gpu_addr: GPU address of the indirect buffer containing PM4 commands
 * @ib_size_dw: Size of the indirect buffer in DWORDs
 * @signal_gpu_addr: GPU address of the completion signal
 *
 * Fills in the AQL vendor packet with:
 *   - header = VENDOR_SPECIFIC | BARRIER
 *   - pm4_ib_format = 1
 *   - pm4_ib_command[0..3] = INDIRECT_BUFFER PM4 pointing to ib_gpu_addr
 *   - completion_signal = signal_gpu_addr
 *
 * Returns: 0 on success, -1 on invalid parameters
 */
int aql_build_pm4_ib_packet(aql_pm4_ib_packet_t *pkt,
			    uint64_t ib_gpu_addr, uint32_t ib_size_dw,
			    uint64_t signal_gpu_addr);

#endif /* AQL_QUEUE_H */
