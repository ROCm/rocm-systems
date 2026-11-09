/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * SDXI dma-buf importer helpers
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#ifndef DMA_SDXI_DMABUF_H
#define DMA_SDXI_DMABUF_H

#include <linux/dma-buf.h>
#include <linux/dma-direction.h>

struct sdxi_dev;

/**
 * struct sdxi_dmabuf_attach - tracks one imported dma-buf attachment.
 *
 * Valid between sdxi_dmabuf_import() and sdxi_dmabuf_release().  Callers must
 * not access the dma-buf fd after import — this struct owns the reference.
 */
struct sdxi_dmabuf_attach {
	struct dma_buf            *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table           *sgt;
	enum dma_data_direction    dir;
};

/**
 * sdxi_dmabuf_import() - attach and map a dma-buf for DMA by the SDXI device.
 * @sdxi: SDXI device that will perform the DMA.
 * @dmabuf: dma-buf to import (caller holds a reference; this function takes
 *          ownership of that reference on success).
 * @dir: DMA direction (DMA_TO_DEVICE for source, DMA_FROM_DEVICE for dest).
 *
 * Returns a pointer to the attach struct on success, ERR_PTR on failure.  The
 * caller must eventually call sdxi_dmabuf_release().
 *
 * Only single-entry sg_tables are supported; returns -ENOTSUPP for
 * scatter-gather buffers that cannot be mapped contiguously for the device.
 */
struct sdxi_dmabuf_attach *
sdxi_dmabuf_import(struct sdxi_dev *sdxi, struct dma_buf *dmabuf,
		   enum dma_data_direction dir);

/**
 * sdxi_dmabuf_release() - unmap and detach a previously imported dma-buf.
 * @a: attachment returned by sdxi_dmabuf_import(), or NULL (no-op).
 */
void sdxi_dmabuf_release(struct sdxi_dmabuf_attach *a);

/**
 * sdxi_dmabuf_dma_addr() - return the device-side DMA address for the buffer.
 * @a: a valid (non-NULL) attachment returned by sdxi_dmabuf_import().
 *
 * Valid only until sdxi_dmabuf_release() is called.
 */
static inline dma_addr_t sdxi_dmabuf_dma_addr(const struct sdxi_dmabuf_attach *a)
{
	return sg_dma_address(a->sgt->sgl);
}

#endif /* DMA_SDXI_DMABUF_H */
