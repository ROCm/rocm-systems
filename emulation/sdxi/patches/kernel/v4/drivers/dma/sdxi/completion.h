/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright Advanced Micro Devices, Inc. */
#ifndef DMA_SDXI_COMPLETION_H
#define DMA_SDXI_COMPLETION_H

#include <linux/compiler_attributes.h>
#include "sdxi.h"
#include "hw.h"


/* A completion status block and the device address the hardware needs. */
struct sdxi_completion {
	struct sdxi_cst_blk *cst_blk;
	dma_addr_t cst_blk_dma;
};

struct sdxi_cst_pool;

/* Software completion states derived from CST_BLK. */
enum sdxi_completion_state {
	SDXI_COMPLETION_PENDING,
	SDXI_COMPLETION_DONE,
	SDXI_COMPLETION_ERROR,
};

struct sdxi_cst_pool *sdxi_cst_pool_create(struct sdxi_dev *sdxi,
					unsigned int count);
/* Hardware and software must no longer access any blocks before destruction. */
void sdxi_cst_pool_destroy(struct sdxi_dev *sdxi, struct sdxi_cst_pool *pool);
/* Claims and initializes a block; false means the fixed pool is exhausted. */
bool sdxi_cst_pool_get(struct sdxi_cst_pool *pool, struct sdxi_completion *comp);
int sdxi_cst_pool_get_wait(struct sdxi_cst_pool *pool,
			 struct sdxi_completion *comp);
/* Release only after consumption, unpublished rollback, or hardware quiescence. */
void sdxi_cst_pool_put(struct sdxi_cst_pool *pool, struct sdxi_cst_blk *cst);

#if IS_ENABLED(CONFIG_SDXI_KUNIT_TEST)
struct sdxi_cst_pool *sdxi_cst_pool_test_create(unsigned int count);
void sdxi_cst_pool_test_destroy(struct sdxi_cst_pool *pool);
#endif

static inline void sdxi_cst_blk_init(struct sdxi_cst_blk *cst)
{
	*cst = (typeof(*cst)) {
		.signal = cpu_to_le64(SDXI_CST_BLK_SIGNAL_INIT),
	};
}

int __must_check sdxi_cst_blk_poll(const struct sdxi_cst_blk *cst);
enum sdxi_completion_state sdxi_cst_blk_check(const struct sdxi_cst_blk *cst);

#endif /* DMA_SDXI_COMPLETION_H */
