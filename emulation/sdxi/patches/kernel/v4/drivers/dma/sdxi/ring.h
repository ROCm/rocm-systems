/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright Advanced Micro Devices, Inc. */
#ifndef DMA_SDXI_RING_H
#define DMA_SDXI_RING_H

#include <linux/bug.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <linux/range.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/wait.h>
#include <asm/barrier.h>
#include <asm/byteorder.h>
#include <asm/div64.h>
#include <asm/rwonce.h>

#include "hw.h"

struct sdxi_sq;

/*
 * struct sdxi_ring_state - Software state for driving a submission queue.
 *
 * @lock: Guards the write index.
 * @sq: The submission queue being driven.
 * @wqh: Pending reservations.
 */
struct sdxi_ring_state {
	spinlock_t lock;
	struct sdxi_sq *sq;
	wait_queue_head_t wqh;
};

/*
 * A claim on a contiguous span of the ring's logical index space. The
 * indexes are monotonic and unwrapped; sdxi_ring_desc() maps them onto
 * ring entries.
 */
struct sdxi_ring_resv {
	const struct sdxi_ring_state *rs;
	struct range range;
};

void sdxi_ring_state_init(struct sdxi_ring_state *ring, struct sdxi_sq *sq);
void sdxi_ring_wake_up(struct sdxi_ring_state *rs);
int sdxi_ring_reserve(struct sdxi_ring_state *ring, size_t nr,
		      struct sdxi_ring_resv *resv);
int sdxi_ring_try_reserve(struct sdxi_ring_state *ring, size_t nr,
			  struct sdxi_ring_resv *resv);
int sdxi_ring_try_reserve_range(struct sdxi_ring_state *rs, size_t min_nr,
				size_t max_nr, struct sdxi_ring_resv *resv);
struct sdxi_desc *sdxi_ring_desc(const struct sdxi_ring_state *rs, u64 index);

/* Number of slots held by @resv. */
static inline size_t sdxi_ring_resv_count(const struct sdxi_ring_resv *resv)
{
	return range_len(&resv->range);
}

/*
 * Logical index of slot @i of @resv, suitable for passing to
 * sdxi_ring_desc(). @i must be less than sdxi_ring_resv_count().
 */
static inline u64 sdxi_ring_resv_index(const struct sdxi_ring_resv *resv,
				       unsigned int i)
{
	/* Keep a caller's mistake inside the slots it owns. */
	if (WARN_ON_ONCE(i >= sdxi_ring_resv_count(resv)))
		return resv->range.start;
	return resv->range.start + i;
}

/* A reserved descriptor slot identified by its ring and logical index. */
struct sdxi_slot {
	const struct sdxi_ring_state *rs;
	u64 index;
};

/* Slot @i of @resv. @i must be less than sdxi_ring_resv_count(). */
static inline struct sdxi_slot sdxi_ring_resv_slot(const struct sdxi_ring_resv *resv,
						   unsigned int i)
{
	return (struct sdxi_slot) {
		.rs = resv->rs,
		.index = sdxi_ring_resv_index(resv, i),
	};
}

static inline struct sdxi_desc *sdxi_slot_desc(struct sdxi_slot slot)
{
	return sdxi_ring_desc(slot.rs, slot.index);
}

#endif /* DMA_SDXI_RING_H */
