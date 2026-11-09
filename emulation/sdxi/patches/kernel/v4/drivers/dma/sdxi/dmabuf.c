// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI dma-buf importer helpers
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#include <linux/dma-buf.h>
#include <linux/dma-direction.h>
#include <linux/err.h>
#include <linux/slab.h>

#include "dmabuf.h"
#include "sdxi.h"

struct sdxi_dmabuf_attach *
sdxi_dmabuf_import(struct sdxi_dev *sdxi, struct dma_buf *dmabuf,
		   enum dma_data_direction dir)
{
	struct sdxi_dmabuf_attach *a;
	int err;

	a = kzalloc(sizeof(*a), GFP_KERNEL);
	if (!a)
		return ERR_PTR(-ENOMEM);

	a->dmabuf = dmabuf;
	a->dir    = dir;

	a->attach = dma_buf_attach(dmabuf, sdxi->dev);
	if (IS_ERR(a->attach)) {
		err = PTR_ERR(a->attach);
		goto free;
	}

	a->sgt = dma_buf_map_attachment_unlocked(a->attach, dir);
	if (IS_ERR(a->sgt)) {
		err = PTR_ERR(a->sgt);
		goto detach;
	}

	/*
	 * SDXI copy descriptors take a single (src, dst) address pair; we do
	 * not support chaining across multiple sg entries.  Reject anything
	 * that cannot be represented as one contiguous DMA region.
	 */
	if (a->sgt->nents != 1) {
		err = -ENOTSUPP;
		goto unmap;
	}

	return a;

unmap:
	dma_buf_unmap_attachment_unlocked(a->attach, a->sgt, dir);
detach:
	dma_buf_detach(dmabuf, a->attach);
free:
	kfree(a);
	return ERR_PTR(err);
}

void sdxi_dmabuf_release(struct sdxi_dmabuf_attach *a)
{
	if (!a)
		return;

	dma_buf_unmap_attachment_unlocked(a->attach, a->sgt, a->dir);
	dma_buf_detach(a->dmabuf, a->attach);
	dma_buf_put(a->dmabuf);
	kfree(a);
}
