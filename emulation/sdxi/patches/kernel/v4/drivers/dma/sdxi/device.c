// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI hardware device driver
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#include <linux/bitfield.h>
#include <linux/delay.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/dmapool.h>
#include <linux/export.h>
#include <linux/idr.h>
#include <linux/iopoll.h>
#include <linux/jiffies.h>
#include <linux/limits.h>
#include <linux/log2.h>
#include <linux/minmax.h>
#include <linux/sizes.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/xarray.h>

#include "context.h"
#include "dma.h"
#include "hw.h"
#include "mmio.h"
#include "sdxi.h"

enum sdxi_fn_gsv {
	SDXI_GSV_STOP     = 0,
	SDXI_GSV_INIT     = 1,
	SDXI_GSV_ACTIVE   = 2,
	SDXI_GSV_STOPG_SF = 3,
	SDXI_GSV_STOPG_HD = 4,
	SDXI_GSV_ERROR    = 5,
};

static const char *const gsv_strings[] = {
	[SDXI_GSV_STOP]     = "stopped",
	[SDXI_GSV_INIT]     = "initializing",
	[SDXI_GSV_ACTIVE]   = "active",
	[SDXI_GSV_STOPG_SF] = "soft stopping",
	[SDXI_GSV_STOPG_HD] = "hard stopping",
	[SDXI_GSV_ERROR]    = "error",
};

static const char *gsv_str(enum sdxi_fn_gsv gsv)
{
	if ((size_t)gsv < ARRAY_SIZE(gsv_strings))
		return gsv_strings[(size_t)gsv];

	return "unknown";
}

enum sdxi_fn_gsr {
	SDXI_GSRV_RESET   = 0,
	SDXI_GSRV_STOP_SF = 1,
	SDXI_GSRV_STOP_HD = 2,
	SDXI_GSRV_ACTIVE  = 3,
};

static const char *const gsr_strings[] = {
	[SDXI_GSRV_RESET]   = "reset",
	[SDXI_GSRV_STOP_SF] = "soft stop",
	[SDXI_GSRV_STOP_HD] = "hard stop",
	[SDXI_GSRV_ACTIVE]  = "activate",
};

static const char *gsr_str(enum sdxi_fn_gsr gsr)
{
	if ((size_t)gsr < ARRAY_SIZE(gsr_strings))
		return gsr_strings[(size_t)gsr];

	return "unknown";
}

static enum sdxi_fn_gsv sdxi_dev_gsv(const struct sdxi_dev *sdxi)
{
	u64 sts0 = sdxi_read64(sdxi, SDXI_MMIO_STS0);
	enum sdxi_fn_gsv gsv = FIELD_GET(SDXI_MMIO_STS0_FN_GSV, sts0);

	dev_dbg(sdxi->dev, "MMIO_STS0.fn_gsv = %#x (%s)\n", gsv, gsv_str(gsv));

	switch (gsv) {
	case SDXI_GSV_STOP ... SDXI_GSV_ERROR:
		break;
	default:
		dev_warn_ratelimited(sdxi->dev, "unknown gsv %u\n", gsv);
		break;
	}

	return gsv;
}

static const unsigned long gsv_poll_interval_us = USEC_PER_MSEC * 100;
static const unsigned long gsv_transition_timeout_us = USEC_PER_SEC * 1;

#define sdxi_dev_gsv_poll(sdxi, val, cond)				\
	read_poll_timeout(sdxi_dev_gsv, val, cond, gsv_poll_interval_us, \
			  gsv_transition_timeout_us, false, sdxi)

static void sdxi_write_fn_gsr(struct sdxi_dev *sdxi, enum sdxi_fn_gsr cmd)
{
	u64 ctl0 = sdxi_read64(sdxi, SDXI_MMIO_CTL0);

	dev_dbg(sdxi->dev, "MMIO_CTL0.fn_gsr = %#x (%s)\n", cmd, gsr_str(cmd));

	FIELD_MODIFY(SDXI_MMIO_CTL0_FN_GSR, &ctl0, cmd);
	sdxi_write64(sdxi, SDXI_MMIO_CTL0, ctl0);
}

/*
 * How to make progress toward GSV_STOP from each function state. See
 * SDXI 1.0 Figure 4-1: SDXI Function States and State Transitions.
 *
 * The function ordinarily leaves a @transient state without software
 * intervention, so @cmd applies only to the others. @stall_cmd, if any,
 * is issued when the state outlasts the poll timeout. @next is the set
 * of states software may observe after the step.
 */
struct gsv_stop_step {
	bool transient:1;
	bool issue_stall_cmd:1;
	enum sdxi_fn_gsr cmd;
	enum sdxi_fn_gsr stall_cmd;
	unsigned long next;
};

static const struct gsv_stop_step gsv_stop_steps[] = {
	[SDXI_GSV_STOP] = { },
	[SDXI_GSV_INIT] = {
		/*
		 * Any fn_gsr write other than SDXI_GSRV_ACTIVE during
		 * initialization forces HaltErr:Fn -> GSV_ERROR.
		 * GSV_STOPG_HD and GSV_STOP are valid next values since
		 * the stop command could race with a GSV_INIT ->
		 * GSV_ACTIVE transition.
		 */
		.transient = true,
		.issue_stall_cmd = true,
		.stall_cmd = SDXI_GSRV_STOP_HD,
		.next = BIT(SDXI_GSV_ACTIVE) | BIT(SDXI_GSV_ERROR) |
			BIT(SDXI_GSV_STOPG_HD) | BIT(SDXI_GSV_STOP),
	},
	[SDXI_GSV_ACTIVE] = {
		.cmd = SDXI_GSRV_STOP_SF,
		.next = BIT(SDXI_GSV_STOPG_SF) | BIT(SDXI_GSV_STOP) |
			BIT(SDXI_GSV_ERROR),
	},
	[SDXI_GSV_STOPG_SF] = {
		.transient = true,
		.issue_stall_cmd = true,
		.stall_cmd = SDXI_GSRV_STOP_HD,
		.next = BIT(SDXI_GSV_STOPG_HD) | BIT(SDXI_GSV_STOP) |
			BIT(SDXI_GSV_ERROR),
	},
	[SDXI_GSV_STOPG_HD] = {
		/* All fn_gsr writes are ignored here; only waiting helps. */
		.transient = true,
		.next = BIT(SDXI_GSV_STOP) | BIT(SDXI_GSV_ERROR),
	},
	[SDXI_GSV_ERROR] = {
		.cmd = SDXI_GSRV_RESET,
		.next = BIT(SDXI_GSV_STOP),
	},
};

/* Drive the function to GSV_STOP from any state. */
static int sdxi_dev_stop(struct sdxi_dev *sdxi)
{
	unsigned long stepped_from = 0;
	enum sdxi_fn_gsv gsv;

	/*
	 * Each step must reach a state we have not already stepped from.
	 * The table admits no cycles, so revisiting one means the device
	 * is not behaving as specified.
	 */
	for (gsv = sdxi_dev_gsv(sdxi); gsv != SDXI_GSV_STOP; ) {
		const struct gsv_stop_step *step;
		enum sdxi_fn_gsv from = gsv;

		if ((size_t)gsv >= ARRAY_SIZE(gsv_stop_steps))
			break;
		if (stepped_from & BIT(gsv))
			break;
		stepped_from |= BIT(gsv);

		step = &gsv_stop_steps[gsv];
		if (!step->transient)
			sdxi_write_fn_gsr(sdxi, step->cmd);

		if (sdxi_dev_gsv_poll(sdxi, gsv, gsv != from)) {
			if (!step->issue_stall_cmd)
				break;
			dev_notice(sdxi->dev, "stop: %s, issuing %s\n",
				   gsv_str(from), gsr_str(step->stall_cmd));
			sdxi_write_fn_gsr(sdxi, step->stall_cmd);
			if (sdxi_dev_gsv_poll(sdxi, gsv, gsv != from))
				break;
		}

		WARN_ONCE(!(BIT(gsv) & step->next), "unexpected %s -> %s\n",
			  gsv_str(from), gsv_str(gsv));
	}

	if (gsv == SDXI_GSV_STOP)
		return 0;

	dev_err(sdxi->dev, "stop: stuck at MMIO_STS0.fn_gsv = %#x (%s)\n",
		gsv, gsv_str(gsv));

	return sdxi->bus_ops->reset ? sdxi->bus_ops->reset(sdxi) : -EIO;
}

/*
 * Activate a stopped function. GSRV_ACTIVE takes it through the
 * transient GSV_INIT state to GSV_ACTIVE or GSV_ERROR. See SDXI 1.0
 * 4.1.8.9.
 */
static int sdxi_dev_start(struct sdxi_dev *sdxi)
{
	const unsigned long settled = BIT(SDXI_GSV_ACTIVE) | BIT(SDXI_GSV_ERROR);
	enum sdxi_fn_gsv gsv = sdxi_dev_gsv(sdxi);

	if (gsv != SDXI_GSV_STOP) {
		dev_err(sdxi->dev,
			"start: MMIO_STS0.fn_gsv = %#x (%s) on entry\n",
			gsv, gsv_str(gsv));
		return -EBUSY;
	}

	sdxi_write_fn_gsr(sdxi, SDXI_GSRV_ACTIVE);

	if (sdxi_dev_gsv_poll(sdxi, gsv, BIT(gsv) & settled) ||
	    gsv != SDXI_GSV_ACTIVE) {
		dev_err(sdxi->dev,
			"start: MMIO_STS0.fn_gsv = %#x (%s) after activate\n",
			gsv, gsv_str(gsv));
		return -EIO;
	}

	return 0;
}

/* Stop the function and discard inherited MMIO configuration. */
static int sdxi_fn_prepare(struct sdxi_dev *sdxi)
{
	u64 cap1, ctl0, ctl2, err_cfg, rkey;
	int err;

	/* Disable inherited error logging before stop can provoke an error. */
	err_cfg = sdxi_read64(sdxi, SDXI_MMIO_ERR_CFG);
	FIELD_MODIFY(SDXI_MMIO_ERR_CFG_EN, &err_cfg, 0);
	sdxi_write64(sdxi, SDXI_MMIO_ERR_CFG, err_cfg);
	err_cfg = sdxi_read64(sdxi, SDXI_MMIO_ERR_CFG);
	if (FIELD_GET(SDXI_MMIO_ERR_CFG_EN, err_cfg)) {
		dev_err(sdxi->dev, "failed to disable error logging\n");
		return -EIO;
	}

	/*
	 * Ensure the function is in GSV_STOP state, then clear ctl0's
	 * pasid and error interrupt configuration while preserving
	 * any assigned group ID (fn_grp_id).
	 */
	err = sdxi_dev_stop(sdxi);
	if (err)
		return err;

	ctl0 = sdxi_read64(sdxi, SDXI_MMIO_CTL0);
	FIELD_MODIFY(SDXI_MMIO_CTL0_FN_ERR_INTR_EN, &ctl0, 0);
	FIELD_MODIFY(SDXI_MMIO_CTL0_FN_PASID_VL, &ctl0, 0);
	FIELD_MODIFY(SDXI_MMIO_CTL0_FN_PASID, &ctl0, 0);
	sdxi_write64(sdxi, SDXI_MMIO_CTL0, ctl0);

	/* Restore CTL2's reset limits, with operation groups disabled. */
	cap1 = sdxi_read64(sdxi, SDXI_MMIO_CAP1);
	ctl2 = FIELD_PREP(SDXI_MMIO_CTL2_MAX_BUFFER,
			  FIELD_GET(SDXI_MMIO_CAP1_MAX_BUFFER, cap1));
	ctl2 |= FIELD_PREP(SDXI_MMIO_CTL2_MAX_AKEY_SZ,
			   FIELD_GET(SDXI_MMIO_CAP1_MAX_AKEY_SZ, cap1));
	ctl2 |= FIELD_PREP(SDXI_MMIO_CTL2_MAX_CXT,
			   FIELD_GET(SDXI_MMIO_CAP1_MAX_CXT, cap1));
	sdxi_write64(sdxi, SDXI_MMIO_CTL2, ctl2);

	/* No inherited context or remote-access tables are owned by us. */
	sdxi_write64(sdxi, SDXI_MMIO_CXT_L2, 0);
	if (FIELD_GET(SDXI_MMIO_CAP1_RKEY_CAP, cap1)) {
		rkey = sdxi_read64(sdxi, SDXI_MMIO_RKEY);
		FIELD_MODIFY(SDXI_MMIO_RKEY_EN, &rkey, 0);
		sdxi_write64(sdxi, SDXI_MMIO_RKEY, rkey);
		rkey = sdxi_read64(sdxi, SDXI_MMIO_RKEY);
		if (FIELD_GET(SDXI_MMIO_RKEY_EN, rkey)) {
			dev_err(sdxi->dev, "failed to disable RKey access\n");
			return -EIO;
		}
		sdxi_write64(sdxi, SDXI_MMIO_RKEY, 0);
	}

	/* Clear error-log state using the status register's RW1C semantics. */
	sdxi_write64(sdxi, SDXI_MMIO_ERR_CTL, 0);
	sdxi_write64(sdxi, SDXI_MMIO_ERR_STS,
		     SDXI_MMIO_ERR_STS_STS_BIT | SDXI_MMIO_ERR_STS_OVF_BIT |
		     SDXI_MMIO_ERR_STS_ERR_BIT);
	sdxi_write64(sdxi, SDXI_MMIO_ERR_CFG, 0);
	sdxi_write64(sdxi, SDXI_MMIO_ERR_WRT, 0);
	sdxi_write64(sdxi, SDXI_MMIO_ERR_RD, 0);

	/* Mailbox contents and commands belong to the peer protocol. */
	if (sdxi->has_mailbox)
		sdxi_write64(sdxi, SDXI_MMIO_MB_CTL, 0);

	return 0;
}

/*
 * See SDXI 1.0 4.1.8 Activation of the SDXI Function by Software.
 */
static int sdxi_fn_activate(struct sdxi_dev *sdxi)
{
	u64 version, cap0, cap1, ctl2, cxt_l2, lv01_ptr;
	struct sdxi_cxt_L2_ent *L2_ent;
	int err;

	version = sdxi_read64(sdxi, SDXI_MMIO_VERSION);
	dev_info(sdxi->dev, "SDXI %llu.%llu device found\n",
		  FIELD_GET(SDXI_MMIO_VERSION_MAJOR, version),
		  FIELD_GET(SDXI_MMIO_VERSION_MINOR, version));

	/* Read capabilities and features. */
	cap0 = sdxi_read64(sdxi, SDXI_MMIO_CAP0);
	sdxi->db_stride = SZ_4K;
	sdxi->db_stride *= 1U << FIELD_GET(SDXI_MMIO_CAP0_DB_STRIDE, cap0);

	cap1 = sdxi_read64(sdxi, SDXI_MMIO_CAP1);
	sdxi->op_grp_cap = FIELD_GET(SDXI_MMIO_CAP1_OPB_000_CAP, cap1);
	sdxi->max_buffer = FIELD_GET(SDXI_MMIO_CAP1_MAX_BUFFER, cap1);
	dma_set_max_seg_size(sdxi->dev,
			     min((u64)SZ_2M << sdxi->max_buffer, U32_MAX));

	/*
	 * Constrain the number of client contexts supported by the
	 * driver to what fits in a single L1 table.
	 */
	sdxi->max_cxtid = min(SDXI_L1_TABLE_ENTRIES - 1,
			      FIELD_GET(SDXI_MMIO_CAP1_MAX_CXT, cap1));

	/* Apply our configuration. */
	ctl2 = FIELD_PREP(SDXI_MMIO_CTL2_MAX_CXT, sdxi->max_cxtid);
	ctl2 |= FIELD_PREP(SDXI_MMIO_CTL2_MAX_BUFFER,
			   sdxi->max_buffer);
	ctl2 |= FIELD_PREP(SDXI_MMIO_CTL2_MAX_AKEY_SZ,
			   FIELD_GET(SDXI_MMIO_CAP1_MAX_AKEY_SZ, cap1));
	ctl2 |= FIELD_PREP(SDXI_MMIO_CTL2_OPB_000_AVL,
			   FIELD_GET(SDXI_MMIO_CAP1_OPB_000_CAP, cap1));
	sdxi_write64(sdxi, SDXI_MMIO_CTL2, ctl2);

	/* SDXI 1.0 4.1.8.2 Context Level 2 Table Setup */
	sdxi->L2_table = dmam_alloc_coherent(sdxi->dev,
					     sizeof(*sdxi->L2_table),
					     &sdxi->L2_dma, GFP_KERNEL);
	if (!sdxi->L2_table)
		return -ENOMEM;

	cxt_l2 = FIELD_PREP(SDXI_MMIO_CXT_L2_PTR, sdxi->L2_dma >> ilog2(SZ_4K));
	sdxi_write64(sdxi, SDXI_MMIO_CXT_L2, cxt_l2);

	/* SDXI 1.0 4.1.8.3 Context Level 1 Table Setup */
	sdxi->L1_table = dmam_alloc_coherent(sdxi->dev,
					     sizeof(*sdxi->L1_table),
					     &sdxi->L1_dma, GFP_KERNEL);
	if (!sdxi->L1_table)
		return -ENOMEM;
	/*
	 * SDXI 1.0 4.1.8.3.c: Initialize the Context level 2 table to
	 * point to the Context Level 1 [table].
	 */
	L2_ent = &sdxi->L2_table->entry[0];
	lv01_ptr = FIELD_PREP(SDXI_CXT_L2_ENT_VL, 1) |
		   FIELD_PREP(SDXI_CXT_L2_ENT_LV01_PTR,
			      sdxi->L1_dma >> ilog2(SZ_4K));
	L2_ent->lv01_ptr = cpu_to_le64(lv01_ptr);

	/*
	 * SDXI 1.0 4.1.8.4 Administrative Context
	 *
	 * The admin context will not consume descriptors until we
	 * write its doorbell later.
	 */
	err = sdxi_admin_cxt_init(sdxi);
	if (err)
		return err;

	/*
	 * SDXI 1.0 4.1.8.9: Set MMIO_CTL0.fn_gsr to GSRV_ACTIVE and
	 * wait for MMIO_STS0.fn_gsv to reach GSV_ACTIVE or GSV_ERROR.
	 *
	 * We haven't queued any descriptors to the admin context at
	 * this point, so the admin context jump start (SDXI 1.0
	 * 4.1.8.10.b) is accomplished later when other
	 * code starts client contexts.
	 */
	err = sdxi_dev_start(sdxi);
	if (err)
		sdxi_dev_stop(sdxi);

	return err;
}

static int sdxi_device_init(struct sdxi_dev *sdxi)
{
	struct device *dev = sdxi->dev;
	size_t size, align;
	int err;

	size = align = sizeof(struct sdxi_cxt_ctl);
	sdxi->cxt_ctl_pool = dmam_pool_create("CXT_CTL", dev, size, align, 0);
	if (!sdxi->cxt_ctl_pool)
		return -ENOMEM;

	err = sdxi_fn_activate(sdxi);
	if (err)
		return err;

	sdxi_dma_register(sdxi);
	return 0;
}

int sdxi_register(struct device *dev, const struct sdxi_bus_ops *ops)
{
	struct sdxi_dev *sdxi;
	int err;

	sdxi = devm_kzalloc(dev, sizeof(*sdxi), GFP_KERNEL);
	if (!sdxi)
		return -ENOMEM;

	sdxi->dev = dev;
	sdxi->bus_ops = ops;
	ida_init(&sdxi->vectors);
	xa_init_flags(&sdxi->client_cxts, XA_FLAGS_ALLOC1);
	dev_set_drvdata(dev, sdxi);

	err = sdxi->bus_ops->init(sdxi);
	if (err)
		return err;

	err = sdxi_fn_prepare(sdxi);
	if (err)
		return err;

	return sdxi_device_init(sdxi);
}
EXPORT_SYMBOL_NS_GPL(sdxi_register, "SDXI");

void sdxi_unregister(struct device *dev)
{
	struct sdxi_dev *sdxi = dev_get_drvdata(dev);

	/* Stop/reset is best effort if the hardware cannot be quiesced. */
	sdxi_dev_stop(sdxi);
	sdxi_client_cxts_exit(sdxi);
	ida_destroy(&sdxi->vectors);
}
EXPORT_SYMBOL_NS_GPL(sdxi_unregister, "SDXI");

MODULE_AUTHOR("Wei Huang");
MODULE_AUTHOR("Nathan Lynch");
MODULE_DESCRIPTION("SDXI core");
MODULE_LICENSE("GPL");
