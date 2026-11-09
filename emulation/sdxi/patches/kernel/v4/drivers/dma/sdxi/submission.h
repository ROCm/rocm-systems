/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright Advanced Micro Devices, Inc. */
#ifndef DMA_SDXI_SUBMISSION_H
#define DMA_SDXI_SUBMISSION_H

#include <linux/cache.h>
#include <linux/compiler_attributes.h>
#include <linux/dma-mapping.h>
#include <linux/overflow.h>
#include <linux/types.h>

#include "hw.h"
#include "sdxi.h"

/*
 * Submission queue: one DMA-coherent allocation holding the
 * descriptor ring, write index, and context status (which includes
 * the read index). cxt_sts gets its own cache line to avoid sharing
 * with write_index.
 */
struct sdxi_sq {
	dma_addr_t handle;
	void __iomem *doorbell;
	__le64 write_index;
	u32 count;
	struct sdxi_cxt_sts cxt_sts ____cacheline_aligned;
	struct sdxi_desc ring[] __counted_by(count);
};

static inline struct sdxi_sq *sdxi_sq_alloc(struct sdxi_dev *sdxi, unsigned int count)
{
	dma_addr_t handle;
	struct sdxi_sq *sq;
	size_t sz = struct_size(sq, ring, count);

	sq = dma_alloc_coherent(sdxi->dev, sz, &handle, GFP_KERNEL);
	if (!sq)
		return NULL;

	sq->count = count;
	sq->handle = handle;
	return sq;
}

static inline void sdxi_sq_free(struct sdxi_dev *sdxi, struct sdxi_sq *sq)
{
	dma_free_coherent(sdxi->dev, struct_size(sq, ring, sq->count), sq,
			  sq->handle);
}

static inline dma_addr_t sdxi_sq_ring_dma(const struct sdxi_sq *sq)
{
	return sq->handle + struct_offset(sq, ring);
}

static inline dma_addr_t sdxi_sq_write_index_dma(const struct sdxi_sq *sq)
{
	return sq->handle + struct_offset(sq, write_index);
}

static inline dma_addr_t sdxi_sq_cxt_sts_dma(const struct sdxi_sq *sq)
{
	return sq->handle + struct_offset(sq, cxt_sts);
}

#endif /* DMA_SDXI_SUBMISSION_H */
