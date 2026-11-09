// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI descriptor encoding.
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#include <kunit/test-bug.h>
#include <kunit/visibility.h>
#include <linux/bitfield.h>
#include <linux/bug.h>
#include <linux/range.h>
#include <linux/sizes.h>
#include <linux/types.h>
#include <asm/byteorder.h>

#include "hw.h"
#include "descriptor.h"

VISIBLE_IF_KUNIT int __must_check sdxi_encode_size32(u64 size, __le32 *dest)
{
	/*
	 * Sizes are encoded as value - 1:
	 * value    encoding
	 *     1           0
	 *     2           1
	 *   ...
	 *    4G  0xffffffff
	 * See SDXI 1.0 Table 6-8: DSC_DMAB_COPY Descriptor Format,
	 * for example.
	 */
	if (size > 0 && size <= SZ_4G) {
		*dest = cpu_to_le32((u32)(size - 1));
		return 0;
	}

	/* Inhibit warning stack traces while testing. */
	if (!kunit_get_current_test()) {
		WARN_ON_ONCE(size == 0);
		WARN_ON_ONCE(size > SZ_4G);
	}

	return -EINVAL;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_size32);

int sdxi_encode_copy(struct sdxi_desc *desc, const struct sdxi_copy *params)
{
	u64 csb_ptr;
	u32 opcode;
	__le32 size;
	int err;

	err = sdxi_encode_size32(params->len, &size);
	if (err)
		return err;
	/*
	 * Reject overlapping src and dst. "Software ... shall not
	 * overlap the source buffer, destination buffer, Atomic
	 * Return Data, or completion status block." - SDXI 1.0 5.6
	 * Memory Consistency Model
	 */
	if (range_overlaps(&(const struct range) {
				   .start = params->src,
				   .end   = params->src + params->len - 1,
			   },
			   &(const struct range) {
				   .start = params->dst,
				   .end   = params->dst + params->len - 1,
			   }))
		return -EINVAL;

	opcode = (FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_COPY) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_DMAB));

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.copy = (typeof(desc->copy)) {
			.opcode = cpu_to_le32(opcode),
			.size = size,
			.akey0 = cpu_to_le16(params->src_akey),
			.akey1 = cpu_to_le16(params->dst_akey),
			.addr0 = cpu_to_le64(params->src),
			.addr1 = cpu_to_le64(params->dst),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_copy);

int sdxi_encode_intr(struct sdxi_desc *desc,
		     const struct sdxi_intr *params)
{
	u64 csb_ptr;
	u32 opcode;

	opcode = (FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_INTR) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_INTR));

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.intr = (typeof(desc->intr)) {
			.opcode = cpu_to_le32(opcode),
			.akey = cpu_to_le16(params->akey),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_intr);

int sdxi_encode_cxt_start(struct sdxi_desc *desc,
			  const struct sdxi_cxt_start *params)
{
	u64 csb_ptr;
	u32 opcode;

	opcode = (FIELD_PREP(SDXI_DSC_FE, 1) |
		  FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_CXT_START_NM) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_ADMIN));

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.cxt_start = (typeof(desc->cxt_start)) {
			.opcode = cpu_to_le32(opcode),
			.cxt_start = cpu_to_le16(params->range.cxt_start),
			.cxt_end = cpu_to_le16(params->range.cxt_end),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_cxt_start);

int sdxi_encode_cxt_stop(struct sdxi_desc *desc,
			  const struct sdxi_cxt_stop *params)
{
	u64 csb_ptr;
	u32 opcode;

	opcode = (FIELD_PREP(SDXI_DSC_FE, 1) |
		  FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_CXT_STOP) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_ADMIN));

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.cxt_stop = (typeof(desc->cxt_stop)) {
			.opcode = cpu_to_le32(opcode),
			.cxt_start = cpu_to_le16(params->range.cxt_start),
			.cxt_end = cpu_to_le16(params->range.cxt_end),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_cxt_stop);

int sdxi_encode_cxt_upd(struct sdxi_desc *desc,
			const struct sdxi_cxt_upd *params)
{
	u64 csb_ptr;
	u32 opcode;
	u8 cflags;

	opcode = (FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_CXT_UPD) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_ADMIN));

	cflags = FIELD_PREP(SDXI_DSC_CXT_UPD_PCD, params->level);

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.cxt_upd = (typeof(desc->cxt_upd)) {
			.opcode = cpu_to_le32(opcode),
			.cflags = cflags,
			.cxt_start = cpu_to_le16(params->range.cxt_start),
			.cxt_end = cpu_to_le16(params->range.cxt_end),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_cxt_upd);

int sdxi_encode_sync(struct sdxi_desc *desc, const struct sdxi_sync *params)
{
	u64 csb_ptr;
	u32 opcode;
	u8 cflags;

	opcode = (FIELD_PREP(SDXI_DSC_SUBTYPE, SDXI_DSC_OP_SUBTYPE_SYNC) |
		  FIELD_PREP(SDXI_DSC_TYPE, SDXI_DSC_OP_TYPE_ADMIN));

	cflags = FIELD_PREP(SDXI_DSC_SYNC_FLT, params->filter);

	csb_ptr = FIELD_PREP(SDXI_DSC_NP, 1);

	*desc = (typeof(*desc)) {
		.sync = (typeof(desc->sync)) {
			.opcode = cpu_to_le32(opcode),
			.cflags = cflags,
			.cxt_start = cpu_to_le16(params->range.cxt_start),
			.cxt_end = cpu_to_le16(params->range.cxt_end),
			.csb_ptr = cpu_to_le64(csb_ptr),
		},
	};

	return 0;
}
EXPORT_SYMBOL_IF_KUNIT(sdxi_encode_sync);
