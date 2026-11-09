// SPDX-License-Identifier: GPL-2.0-only
/* Copyright Advanced Micro Devices, Inc. */
#include <kunit/test.h>
#include <linux/module.h>

#include "completion.h"

MODULE_IMPORT_NS("EXPORTED_FOR_KUNIT_TESTING");

static void destroy_pool(void *pool)
{
	sdxi_cst_pool_test_destroy(pool);
}

static struct sdxi_cst_pool *alloc_pool(struct kunit *test, unsigned int count)
{
	struct sdxi_cst_pool *pool = sdxi_cst_pool_test_create(count);

	if (!pool || kunit_add_action_or_reset(test, destroy_pool, pool))
		return NULL;
	return pool;
}

static void exhaustion(struct kunit *test)
{
	/* Cross a bitmap word boundary without filling the final word. */
	const unsigned int count = BITS_PER_LONG + 3;
	struct sdxi_completion *held, extra = {};
	struct sdxi_cst_pool *pool;

	pool = alloc_pool(test, count);
	KUNIT_ASSERT_NOT_NULL(test, pool);
	held = kunit_kcalloc(test, count, sizeof(*held), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, held);
	for (unsigned int i = 0; i < count; i++) {
		KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &held[i]));
		KUNIT_EXPECT_EQ(test, held[i].cst_blk_dma % 32, 0);
		KUNIT_EXPECT_EQ(test, le64_to_cpu(held[i].cst_blk->signal),
				(u64)SDXI_CST_BLK_SIGNAL_INIT);
		for (unsigned int j = 0; j < i; j++) {
			KUNIT_EXPECT_PTR_NE(test, held[i].cst_blk, held[j].cst_blk);
			KUNIT_EXPECT_NE(test, held[i].cst_blk_dma, held[j].cst_blk_dma);
		}
	}
	KUNIT_EXPECT_FALSE(test, sdxi_cst_pool_get(pool, &extra));
	KUNIT_EXPECT_PTR_EQ(test, extra.cst_blk, NULL);
	for (unsigned int i = 0; i < count; i++)
		sdxi_cst_pool_put(pool, held[i].cst_blk);
}

static void arbitrary_release(struct kunit *test)
{
	struct sdxi_completion held[5], comp;
	struct sdxi_cst_pool *pool;
	const unsigned int order[] = { 3, 1, 4, 0, 2 };

	pool = alloc_pool(test, ARRAY_SIZE(held));
	KUNIT_ASSERT_NOT_NULL(test, pool);
	for (unsigned int i = 0; i < ARRAY_SIZE(held); i++)
		KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &held[i]));

	/* A released block is reusable while all other owners retain theirs. */
	for (unsigned int i = 0; i < ARRAY_SIZE(order); i++) {
		unsigned int index = order[i];

		sdxi_cst_pool_put(pool, held[index].cst_blk);
		KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &comp));
		KUNIT_EXPECT_PTR_EQ(test, comp.cst_blk, held[index].cst_blk);
		KUNIT_EXPECT_EQ(test, comp.cst_blk_dma, held[index].cst_blk_dma);
		KUNIT_EXPECT_FALSE(test, sdxi_cst_pool_get(pool, &comp));
	}
	for (unsigned int i = 0; i < ARRAY_SIZE(held); i++)
		sdxi_cst_pool_put(pool, held[i].cst_blk);
}

static void reuse_initializes_status(struct kunit *test)
{
	const struct sdxi_cst_blk initial = {
		.signal = cpu_to_le64(SDXI_CST_BLK_SIGNAL_INIT),
	};
	struct sdxi_completion first, second;
	struct sdxi_cst_pool *pool;

	pool = alloc_pool(test, 1);
	KUNIT_ASSERT_NOT_NULL(test, pool);
	KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &first));
	memset(first.cst_blk, 0xff, sizeof(*first.cst_blk));
	first.cst_blk->signal = cpu_to_le64(SDXI_CST_BLK_SIGNAL_TERMINAL);
	sdxi_cst_pool_put(pool, first.cst_blk);
	KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &second));
	KUNIT_EXPECT_PTR_EQ(test, first.cst_blk, second.cst_blk);
	KUNIT_EXPECT_MEMEQ(test, second.cst_blk, &initial, sizeof(initial));
	sdxi_cst_pool_put(pool, second.cst_blk);
}

static void retained_completion(struct kunit *test)
{
	struct sdxi_completion timed_out, other;
	struct sdxi_cst_pool *pool;

	pool = alloc_pool(test, 1);
	KUNIT_ASSERT_NOT_NULL(test, pool);
	KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &timed_out));
	KUNIT_EXPECT_FALSE(test, sdxi_cst_pool_get(pool, &other));
	/* Even a late completion remains allocated until its owner releases it. */
	timed_out.cst_blk->signal = cpu_to_le64(SDXI_CST_BLK_SIGNAL_TERMINAL);
	KUNIT_EXPECT_FALSE(test, sdxi_cst_pool_get(pool, &other));
	sdxi_cst_pool_put(pool, timed_out.cst_blk);
	KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &other));
	/* A never-published claim can be returned without a terminal signal. */
	sdxi_cst_pool_put(pool, other.cst_blk);
	KUNIT_ASSERT_TRUE(test, sdxi_cst_pool_get(pool, &other));
	sdxi_cst_pool_put(pool, other.cst_blk);
}

static struct kunit_case completion_testcases[] = {
	KUNIT_CASE(exhaustion),
	KUNIT_CASE(arbitrary_release),
	KUNIT_CASE(reuse_initializes_status),
	KUNIT_CASE(retained_completion),
	{}
};

static struct kunit_suite completion_tests = {
	.name = "SDXI completion pool",
	.test_cases = completion_testcases,
};
kunit_test_suite(completion_tests);

MODULE_DESCRIPTION("SDXI completion pool tests");
MODULE_LICENSE("GPL");
