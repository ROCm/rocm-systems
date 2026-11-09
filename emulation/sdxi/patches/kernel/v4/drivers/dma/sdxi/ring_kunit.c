// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI descriptor ring management tests.
 *
 * Copyright Advanced Micro Devices, Inc.
 */
#include <kunit/device.h>
#include <kunit/test-bug.h>
#include <kunit/test.h>
#include <linux/container_of.h>
#include <linux/dma-mapping.h>
#include <linux/module.h>
#include <linux/overflow.h>
#include <linux/packing.h>
#include <linux/string.h>

#include "ring.h"
#include "submission.h"

MODULE_IMPORT_NS("EXPORTED_FOR_KUNIT_TESTING");

/* A submission queue backed by plain memory; no device or DMA required. */
static struct sdxi_sq *alloc_sq(struct kunit *t, u32 count)
{
	struct sdxi_sq *sq;

	sq = kunit_kmalloc(t, struct_size(sq, ring, count),
			   GFP_KERNEL | __GFP_ZERO);
	if (sq)
		sq->count = count;
	return sq;
}

static void valid(struct kunit *t)
{
	struct sdxi_ring_state r;
	struct sdxi_ring_resv resv;
	struct sdxi_sq *sq;

	sq = alloc_sq(t, SZ_1K);
	KUNIT_ASSERT_NOT_NULL(t, sq);

	sdxi_ring_state_init(&r, sq);

	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&r, sq->count, &resv), 0);
	KUNIT_EXPECT_EQ(t, resv.range.start, 0);
	KUNIT_EXPECT_EQ(t, resv.range.end, sq->count - 1);
	KUNIT_EXPECT_EQ(t, sdxi_ring_resv_count(&resv), sq->count);
	KUNIT_EXPECT_EQ(t, le64_to_cpu(sq->write_index), sq->count);

	/* A ring-sized reservation maps slot i onto ring entry i. */
	for (unsigned int i = 0; i < sq->count; i++) {
		KUNIT_EXPECT_EQ(t, sdxi_ring_resv_index(&resv, i), i);
		KUNIT_EXPECT_PTR_EQ(t, sdxi_ring_desc(&r, i), &sq->ring[i]);
	}

	sq->cxt_sts.read_index = cpu_to_le64(1);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&r, 1, &resv), 0);
	KUNIT_EXPECT_EQ(t, le64_to_cpu(sq->write_index), sq->count + 1);
	KUNIT_EXPECT_EQ(t, sdxi_ring_resv_count(&resv), 1);

	/* The index space is unwrapped: index sq->count is entry 0. */
	KUNIT_EXPECT_EQ(t, sdxi_ring_resv_index(&resv, 0), sq->count);
	KUNIT_EXPECT_PTR_EQ(t, sdxi_ring_desc(&r, sq->count), &sq->ring[0]);
}

static void invalid(struct kunit *t)
{
	struct sdxi_ring_state rs;
	struct sdxi_ring_resv resv;
	struct sdxi_sq *sq;

	sq = alloc_sq(t, SZ_1K);
	KUNIT_ASSERT_NOT_NULL(t, sq);

	sdxi_ring_state_init(&rs, sq);

	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&rs, 0, &resv), -EINVAL);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&rs, sq->count + 1, &resv), -EINVAL);

	sq->cxt_sts.read_index = cpu_to_le64(1);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&rs, 1, &resv), -EIO);

	sq->cxt_sts.read_index = 0;
	sq->write_index = cpu_to_le64(sq->count);
	sdxi_ring_state_init(&rs, sq);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&rs, 1, &resv), -EBUSY);

	sq->cxt_sts.read_index = cpu_to_le64(sq->count);
	sq->write_index = cpu_to_le64(sq->count + 1);
	sdxi_ring_state_init(&rs, sq);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve(&rs, sq->count, &resv), -EBUSY);
}

static void bounded_reservations(struct kunit *t)
{
	struct sdxi_ring_state rs;
	struct sdxi_ring_resv resv;
	struct sdxi_sq *sq = alloc_sq(t, SZ_1K);

	KUNIT_ASSERT_NOT_NULL(t, sq);
	sdxi_ring_state_init(&rs, sq);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve_range(&rs, 0, 2, &resv), -EINVAL);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve_range(&rs, 3, 2, &resv), -EINVAL);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve_range(&rs, 2, sq->count + 1, &resv),
			-EINVAL);

	sq->write_index = cpu_to_le64(sq->count - 2);
	KUNIT_ASSERT_EQ(t, sdxi_ring_try_reserve_range(&rs, 2, 10, &resv), 0);
	KUNIT_EXPECT_EQ(t, sdxi_ring_resv_count(&resv), 2);
	KUNIT_EXPECT_EQ(t, resv.range.start, sq->count - 2);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve_range(&rs, 2, 10, &resv), -EBUSY);
	sq->cxt_sts.read_index = cpu_to_le64(1);
	KUNIT_EXPECT_EQ(t, sdxi_ring_try_reserve_range(&rs, 2, 10, &resv), -EBUSY);
	sq->cxt_sts.read_index = cpu_to_le64(5);
	KUNIT_ASSERT_EQ(t, sdxi_ring_try_reserve_range(&rs, 2, 10, &resv), 0);
	KUNIT_EXPECT_EQ(t, sdxi_ring_resv_count(&resv), 5);
	KUNIT_EXPECT_PTR_EQ(t, sdxi_slot_desc(sdxi_ring_resv_slot(&resv, 0)),
			    &sq->ring[0]);
}

static struct kunit_case testcases[] = {
	KUNIT_CASE(valid),
	KUNIT_CASE(invalid),
	KUNIT_CASE(bounded_reservations),
	{}
};

static int setup_device(struct kunit *t)
{
	struct device *dev = kunit_device_register(t, "sdxi-mock-device");

	KUNIT_ASSERT_NOT_ERR_OR_NULL(t, dev);
	t->priv = dev;
	return 0;
}

static struct kunit_suite generic_desc_ts = {
	.name = "SDXI descriptor ring management",
	.test_cases = testcases,
	.init = setup_device,
};
kunit_test_suite(generic_desc_ts);

MODULE_DESCRIPTION("SDXI descriptor ring tests");
MODULE_AUTHOR("Nathan Lynch");
MODULE_LICENSE("GPL");
