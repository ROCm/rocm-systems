// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI Descriptor Completion Status Block handling.
 *
 * Copyright Advanced Micro Devices, Inc.
 */
#include <kunit/visibility.h>
#include <linux/bitfield.h>
#include <linux/bitmap.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/wait.h>

#include "completion.h"
#include "hw.h"

/* Only the status blocks themselves are visible to hardware. */
struct sdxi_cst_pool {
	struct sdxi_cst_blk *blocks;
	dma_addr_t dma;
	unsigned int count;
	spinlock_t lock;
	wait_queue_head_t wait;
	unsigned long *allocated;
};

static void sdxi_cst_pool_init(struct sdxi_cst_pool *pool, unsigned int count)
{
	pool->count = count;
	spin_lock_init(&pool->lock);
	init_waitqueue_head(&pool->wait);
}

struct sdxi_cst_pool *sdxi_cst_pool_create(struct sdxi_dev *sdxi,
					unsigned int count)
{
	struct sdxi_cst_pool *pool;

	if (!count)
		return NULL;

	pool = kzalloc_obj(*pool, GFP_KERNEL);
	if (!pool)
		return NULL;

	pool->allocated = bitmap_zalloc(count, GFP_KERNEL);
	if (!pool->allocated) {
		kfree(pool);
		return NULL;
	}

	/* SDXI 1.0a 6.1.1 requires a 32-byte-aligned completion block. */
	pool->blocks = dma_alloc_coherent(sdxi->dev,
					 array_size(count, sizeof(*pool->blocks)),
					 &pool->dma, GFP_KERNEL);
	if (!pool->blocks) {
		bitmap_free(pool->allocated);
		kfree(pool);
		return NULL;
	}

	sdxi_cst_pool_init(pool, count);
	return pool;
}

void sdxi_cst_pool_destroy(struct sdxi_dev *sdxi, struct sdxi_cst_pool *pool)
{
	dma_free_coherent(sdxi->dev, array_size(pool->count, sizeof(*pool->blocks)),
			  pool->blocks, pool->dma);
	bitmap_free(pool->allocated);
	kfree(pool);
}

bool sdxi_cst_pool_get(struct sdxi_cst_pool *pool, struct sdxi_completion *comp)
{
	unsigned long flags;
	unsigned int index;

	spin_lock_irqsave(&pool->lock, flags);
	index = find_first_zero_bit(pool->allocated, pool->count);
	if (index < pool->count)
		__set_bit(index, pool->allocated);
	spin_unlock_irqrestore(&pool->lock, flags);
	if (index == pool->count)
		return false;

	comp->cst_blk = &pool->blocks[index];
	comp->cst_blk_dma = pool->dma + index * sizeof(*pool->blocks);
	sdxi_cst_blk_init(comp->cst_blk);
	return true;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_cst_pool_get);

int sdxi_cst_pool_get_wait(struct sdxi_cst_pool *pool,
			 struct sdxi_completion *comp)
{
	long ret;

	ret = wait_event_killable_timeout(pool->wait,
					 sdxi_cst_pool_get(pool, comp), HZ);
	if (ret < 0)
		return ret;
	return ret ? 0 : -ETIMEDOUT;
}

void sdxi_cst_pool_put(struct sdxi_cst_pool *pool, struct sdxi_cst_blk *cst)
{
	unsigned int index = cst - pool->blocks;
	unsigned long flags;

	if (WARN_ON_ONCE(index >= pool->count))
		return;

	spin_lock_irqsave(&pool->lock, flags);
	WARN_ON_ONCE(!__test_and_clear_bit(index, pool->allocated));
	spin_unlock_irqrestore(&pool->lock, flags);
	wake_up_all(&pool->wait);
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_cst_pool_put);

#if IS_ENABLED(CONFIG_SDXI_KUNIT_TEST)
/* Exercise allocation without requiring a device capable of coherent DMA. */
struct sdxi_cst_pool *sdxi_cst_pool_test_create(unsigned int count)
{
	struct sdxi_cst_pool *pool;

	pool = kzalloc_obj(*pool, GFP_KERNEL);
	if (!pool)
		return NULL;
	pool->allocated = bitmap_zalloc(count, GFP_KERNEL);
	if (!pool->allocated) {
		kfree(pool);
		return NULL;
	}
	pool->blocks = kcalloc(count, sizeof(*pool->blocks), GFP_KERNEL);
	if (!pool->blocks) {
		bitmap_free(pool->allocated);
		kfree(pool);
		return NULL;
	}
	pool->dma = 0x1000;
	sdxi_cst_pool_init(pool, count);
	return pool;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_cst_pool_test_create);

void sdxi_cst_pool_test_destroy(struct sdxi_cst_pool *pool)
{
	kfree(pool->blocks);
	bitmap_free(pool->allocated);
	kfree(pool);
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_cst_pool_test_destroy);
#endif

enum sdxi_completion_state sdxi_cst_blk_check(const struct sdxi_cst_blk *cst)
{
	u64 signal = le64_to_cpu(READ_ONCE(cst->signal));

	if (signal != SDXI_CST_BLK_SIGNAL_TERMINAL)
		return SDXI_COMPLETION_PENDING;
	/*
	 * The final ER state (and the operation's effects) are
	 * guaranteed visible only after a cleared signal is observed.
	 * See SDXI 1.0 5.6 Memory Consistency Model.
	 */
	dma_rmb();

	if (FIELD_GET(SDXI_CST_BLK_ER, le32_to_cpu(cst->flags)))
		return SDXI_COMPLETION_ERROR;

	return SDXI_COMPLETION_DONE;
}

int sdxi_cst_blk_poll(const struct sdxi_cst_blk *cst)
{
	enum sdxi_completion_state state;
	int ret;

	ret = read_poll_timeout(sdxi_cst_blk_check, state,
				state != SDXI_COMPLETION_PENDING,
				10, USEC_PER_SEC, false, cst);
	if (ret)
		return ret;		/* -ETIMEDOUT */

	return state == SDXI_COMPLETION_ERROR ? -EIO : 0;
}
