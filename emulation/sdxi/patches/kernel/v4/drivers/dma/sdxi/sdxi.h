/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * SDXI device driver header
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#ifndef DMA_SDXI_H
#define DMA_SDXI_H

#include <linux/bug.h>
#include <linux/compiler_types.h>
#include <linux/dev_printk.h>
#include <linux/idr.h>
#include <linux/io.h>
#include <linux/types.h>
#include <linux/xarray.h>

#include "mmio.h"

#define ID_TO_L1_INDEX(id)	((id) & 0x7F)

#define DESC_RING_BASE_PTR_SHIFT	6
#define CXT_STATUS_PTR_SHIFT		4
#define WRT_INDEX_PTR_SHIFT		3

#define L1_CXT_CTRL_PTR_SHIFT		6
#define L1_CXT_AKEY_PTR_SHIFT		12

enum {
	/* Reserve vector 0 for the error log interrupt. */
	SDXI_ERROR_VECTOR = 0,

	/*
	 * Request at least one vector to account for the error log
	 * interrupt. Increment this if the driver gains more
	 * dedicated interrupts (e.g. one for the admin context).
	 */
	SDXI_MIN_VECTORS = 1,
};

struct sdxi_dev;

/**
 * struct sdxi_bus_ops - Bus-specific methods for SDXI devices.
 */
struct sdxi_bus_ops {
	/**
	 * @init: Map control registers and doorbell region, allocate
	 *        IRQ ranges. Invoked before bus-agnostic SDXI
	 *        function initialization.
	 */
	int (*init)(struct sdxi_dev *sdxi);
	/**
	 * @reset: Bus-specific recovery when the native stop sequence fails.
	 */
	int (*reset)(struct sdxi_dev *sdxi);
	/**
	 * @get_irq: Map device interrupt index to Linux IRQ number.
	 */
	int (*get_irq)(struct sdxi_dev *sdxi, unsigned int index);
};

struct sdxi_dev {
	struct device *dev;
	void __iomem *ctrl_regs;	/* virt addr of ctrl registers */
	void __iomem *dbs;		/* virt addr of doorbells */

	/* hardware capabilities (from cap0 & cap1) */
	u32 db_stride;			/* doorbell stride in bytes */
	u16 max_cxtid;			/* Maximum context ID allowed. */
	u8 max_buffer;			/* Configured log2 buffer size minus 21. */
	u32 op_grp_cap;			/* supported operation group cap */

	struct sdxi_cxt_L2_table *L2_table;
	dma_addr_t L2_dma;
	struct sdxi_cxt_L1_table *L1_table;
	dma_addr_t L1_dma;

	struct dma_pool *cxt_ctl_pool;

	unsigned int nr_vectors;
	struct ida vectors;

	struct sdxi_cxt *admin_cxt;
	struct xarray client_cxts; /* context id -> (struct sdxi_cxt *) */

	bool has_mailbox;

	const struct sdxi_bus_ops *bus_ops;
};

/**
 * sdxi_alloc_vector() - Allocate an interrupt vector.
 *
 * A vector that will have the same lifetime as the device does not
 * need to be released explicitly. Otherwise the vector must be
 * released with sdxi_free_vector().
 */
static inline int sdxi_alloc_vector(struct sdxi_dev *sdxi)
{
	return ida_alloc_range(&sdxi->vectors, SDXI_MIN_VECTORS,
			       sdxi->nr_vectors - 1, GFP_KERNEL);
}

/**
 * sdxi_free_vector() - Release a previously allocated index.
 */
static inline void sdxi_free_vector(struct sdxi_dev *sdxi, unsigned int nr)
{
	ida_free(&sdxi->vectors, nr);
}

/**
 * sdxi_vector_to_irq() - Translate an allocated interrupt vector to
 *                        Linux IRQ number suitable for passing to
 *                        request_irq() et al.
 */
static inline int sdxi_vector_to_irq(struct sdxi_dev *sdxi, unsigned int nr)
{
	WARN_ON_ONCE(!ida_exists(&sdxi->vectors, nr));
	return sdxi->bus_ops->get_irq(sdxi, nr);
}

int sdxi_register(struct device *dev, const struct sdxi_bus_ops *ops);
void sdxi_unregister(struct device *dev);

static inline u64 sdxi_read64(const struct sdxi_dev *sdxi, enum sdxi_reg reg)
{
	return readq(sdxi->ctrl_regs + reg);
}

static inline void sdxi_write64(struct sdxi_dev *sdxi, enum sdxi_reg reg, u64 val)
{
	writeq(val, sdxi->ctrl_regs + reg);
}

#endif /* DMA_SDXI_H */
