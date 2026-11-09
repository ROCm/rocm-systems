/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright Advanced Micro Devices, Inc. */

#ifndef DMA_SDXI_DMA_H
#define DMA_SDXI_DMA_H

#include <linux/dmaengine.h>

struct dma_buf;
struct sdxi_dev;

int sdxi_dma_register(struct sdxi_dev *sdxi);

/**
 * sdxi_dma_prep_dma_buf_memcpy() - prepare a DMA copy between two dma-bufs.
 * @chan:     DMA channel (from dmaengine_get() or similar).
 * @dst_buf:  dma-buf for the destination.  Caller must hold a reference; this
 *            function takes ownership on success.
 * @src_buf:  dma-buf for the source.  Same ownership rules as dst_buf.
 * @len:      bytes to copy.
 * @flags:    DMA_PREP_* flags (e.g. DMA_PREP_INTERRUPT).
 *
 * Returns a descriptor on success, NULL on failure.  On failure neither
 * dma-buf reference is consumed.  dma-buf references are released when the
 * descriptor is freed (after dma_async_is_tx_complete() reports done).
 *
 * Both dma-bufs must map to a single sg entry for the SDXI device; use
 * dma-heap or other physically-contiguous allocators to satisfy this.
 */
struct dma_async_tx_descriptor *
sdxi_dma_prep_dma_buf_memcpy(struct dma_chan *chan,
			      struct dma_buf *dst_buf, struct dma_buf *src_buf,
			      size_t len, unsigned long flags);

#endif /* DMA_SDXI_DMA_H */
