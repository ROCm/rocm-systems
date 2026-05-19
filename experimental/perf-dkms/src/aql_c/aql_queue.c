/**
 * @file aql_queue.c
 * @brief AQL queue ring buffer, IB pool, and PM4 IB packet implementation
 *
 * Pure C implementation for AQL queue-based GPU submission.
 * Compiles in both kernel and userspace (USERSPACE_BUILD) contexts.
 */

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/string.h>
#include <asm/barrier.h>
#else
#include <stdint.h>
#include <string.h>
/* Userspace stubs for kernel barriers */
#define smp_wmb() __asm__ __volatile__("" ::: "memory")
#define WRITE_ONCE(x, val) ((x) = (val))
#endif

#include "aql_queue.h"

/* ========================================================================
 * Helper: check if value is a power of 2
 * ======================================================================== */
static int is_power_of_2(uint32_t v)
{
	return v && !(v & (v - 1));
}

/* ========================================================================
 * Helper: align up to AQL_IB_ALIGNMENT
 * ======================================================================== */
static uint32_t align_up(uint32_t val, uint32_t alignment)
{
	return (val + alignment - 1) & ~(alignment - 1);
}

/* ========================================================================
 * Ring buffer implementation
 * ======================================================================== */

int aql_ring_init(aql_ring_t *ring, void *base, uint64_t gpu_addr, uint32_t size)
{
	if (!ring || !base || !size || !is_power_of_2(size))
		return -1;

	/* Ring must hold at least one 64-byte packet */
	if (size < AQL_PACKET_SIZE)
		return -1;

	ring->base = base;
	ring->gpu_addr = gpu_addr;
	ring->size = size;
	ring->wptr = 0;
	ring->rptr = 0;

	return 0;
}

int aql_ring_write_packet(aql_ring_t *ring, const aql_pm4_ib_packet_t *pkt)
{
	uint64_t slot_offset;
	uint8_t *dst;

	if (!ring || !pkt)
		return -1;

	/* Check if ring is full */
	if (ring->wptr - ring->rptr >= ring->size)
		return -1;

	/* Calculate slot offset using mask (power of 2 size) */
	slot_offset = ring->wptr & (ring->size - 1);
	dst = (uint8_t *)ring->base + slot_offset;

	/* Copy everything EXCEPT the header first (bytes 2..63) */
	memcpy(dst + 2, (const uint8_t *)pkt + 2, AQL_PACKET_SIZE - 2);

	/* Write memory barrier: ensure packet body is visible before header */
	smp_wmb();

	/* Write header last (2 bytes) - signals packet validity to GPU */
	WRITE_ONCE(*(uint16_t *)dst, pkt->header);

	/* Advance write pointer */
	ring->wptr += AQL_PACKET_SIZE;

	return 0;
}

uint32_t aql_ring_slot_index(const aql_ring_t *ring)
{
	if (!ring || !ring->size)
		return 0;

	return (uint32_t)((ring->wptr & (ring->size - 1)) / AQL_PACKET_SIZE);
}

uint32_t aql_ring_available_slots(const aql_ring_t *ring)
{
	uint64_t used;

	if (!ring || !ring->size)
		return 0;

	used = ring->wptr - ring->rptr;
	return (ring->size - (uint32_t)used) / AQL_PACKET_SIZE;
}

/* ========================================================================
 * IB pool implementation
 * ======================================================================== */

int aql_ib_pool_init(aql_ib_pool_t *pool, void *base, uint64_t gpu_addr, uint32_t size)
{
	if (!pool || !base || !size)
		return -1;

	pool->base = base;
	pool->gpu_addr = gpu_addr;
	pool->size = size;
	pool->offset = 0;

	return 0;
}

int aql_ib_pool_alloc(aql_ib_pool_t *pool, uint32_t size_bytes,
		      void **out_cpu, uint64_t *out_gpu)
{
	uint32_t aligned_size;
	uint32_t alloc_offset;

	if (!pool || !out_cpu || !out_gpu || !size_bytes)
		return -1;

	aligned_size = align_up(size_bytes, AQL_IB_ALIGNMENT);

	/* Check if single allocation exceeds pool capacity */
	if (aligned_size > pool->size)
		return -1;

	/* Wrap if not enough space at current offset */
	if (pool->offset + aligned_size > pool->size)
		pool->offset = 0;

	alloc_offset = pool->offset;

	*out_cpu = (uint8_t *)pool->base + alloc_offset;
	*out_gpu = pool->gpu_addr + alloc_offset;

	pool->offset = alloc_offset + aligned_size;

	return 0;
}

void aql_ib_pool_reset(aql_ib_pool_t *pool)
{
	if (pool)
		pool->offset = 0;
}

/* ========================================================================
 * AQL PM4 IB packet builder
 * ======================================================================== */

int aql_build_pm4_ib_packet(aql_pm4_ib_packet_t *pkt,
			    uint64_t ib_gpu_addr, uint32_t ib_size_dw,
			    uint64_t signal_gpu_addr)
{
	if (!pkt || !ib_size_dw)
		return -1;

	/* Zero the entire packet first */
	memset(pkt, 0, sizeof(*pkt));

	/* Header: VENDOR_SPECIFIC type (0) with BARRIER bit set.
	 * BARRIER ensures MES waits for any prior packets before processing.
	 * This gives header = 0x0100 which is non-zero, ensuring MES
	 * doesn't misinterpret the packet as an empty slot on GFX12. */
	pkt->header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE_SHIFT)
		    | (1 << HSA_PACKET_HEADER_BARRIER_SHIFT);

	/* PM4 IB format = 1 (AMD-specific) */
	pkt->pm4_ib_format = 1;

	/* INDIRECT_BUFFER PM4 command (4 DWORDs) */
	pkt->pm4_ib_command[0] = PM4_IB_HEADER;
	pkt->pm4_ib_command[1] = (uint32_t)(ib_gpu_addr & 0xFFFFFFFC); /* DWORD-aligned, low */
	pkt->pm4_ib_command[2] = (uint32_t)(ib_gpu_addr >> 32);        /* high */
	pkt->pm4_ib_command[3] = ib_size_dw | IB_VALID_BIT;            /* size + valid bit */

	/* Remaining DWORDs after pm4_ib_command in the vendor-specific region */
	pkt->dw_count_remain = 10;

	/* Completion signal GPU address */
	pkt->completion_signal = signal_gpu_addr;

	return 0;
}
