/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef DMA_SDXI_DESCRIPTOR_H
#define DMA_SDXI_DESCRIPTOR_H

/*
 * Facilities for encoding SDXI descriptors.
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#include <linux/bitfield.h>
#include <linux/kconfig.h>
#include <linux/minmax.h>
#include <linux/ratelimit.h>
#include <linux/types.h>
#include <asm/byteorder.h>

#include "hw.h"

#if IS_ENABLED(CONFIG_KUNIT)
int __must_check sdxi_encode_size32(u64 size, __le32 *dest);
#endif

static inline void sdxi_desc_vl_expect(const struct sdxi_desc *desc, bool expected)
{
	u8 vl = FIELD_GET(SDXI_DSC_VL, le32_to_cpu(desc->opcode));

	WARN_RATELIMIT(vl != expected, "expected vl=%u but got %u\n", expected, vl);
}

static inline void sdxi_desc_set_csb(struct sdxi_desc *desc, dma_addr_t addr)
{
	sdxi_desc_vl_expect(desc, 0);
	desc->csb_ptr = cpu_to_le64(FIELD_PREP(SDXI_DSC_CSB_PTR, addr >> 5));
}

static inline void sdxi_desc_make_valid(struct sdxi_desc *desc)
{
	u32 opcode = le32_to_cpu(desc->opcode);

	sdxi_desc_vl_expect(desc, 0);
	FIELD_MODIFY(SDXI_DSC_VL, &opcode, 1);
	/*
	 * Once vl is set, no more modifications to the descriptor
	 * payload are allowed. Ensure the vl update is ordered after
	 * all other initialization of the descriptor.
	 */
	dma_wmb();
	WRITE_ONCE(desc->opcode, cpu_to_le32(opcode));
}

static inline void sdxi_desc_set_fence(struct sdxi_desc *desc)
{
	u32 opcode = le32_to_cpu(desc->opcode);

	sdxi_desc_vl_expect(desc, 0);
	FIELD_MODIFY(SDXI_DSC_FE, &opcode, 1);
	desc->opcode = cpu_to_le32(opcode);
}

static inline void sdxi_desc_set_csr(struct sdxi_desc *desc)
{
	u32 opcode = le32_to_cpu(desc->opcode);

	sdxi_desc_vl_expect(desc, 0);
	FIELD_MODIFY(SDXI_DSC_CSR, &opcode, 1);
	desc->opcode = cpu_to_le32(opcode);
}

struct sdxi_cxt_range {
	u16 cxt_start;
	u16 cxt_end;
};

static inline struct sdxi_cxt_range sdxi_cxt_range(u16 a, u16 b)
{
	return (struct sdxi_cxt_range) {
		.cxt_start = min(a, b),
		.cxt_end   = max(a, b),
	};
}

static inline struct sdxi_cxt_range sdxi_cxt_range_single(u16 nr)
{
	return sdxi_cxt_range(nr, nr);
}

struct sdxi_copy {
	dma_addr_t src;
	dma_addr_t dst;
	u64 len;
	u16 src_akey;
	u16 dst_akey;
};

int sdxi_encode_copy(struct sdxi_desc *desc,
		     const struct sdxi_copy *params);

struct sdxi_intr {
	u16 akey;
};

int sdxi_encode_intr(struct sdxi_desc *desc,
		     const struct sdxi_intr *params);

struct sdxi_cxt_start {
	struct sdxi_cxt_range range;
};

int sdxi_encode_cxt_start(struct sdxi_desc *desc,
			  const struct sdxi_cxt_start *params);

struct sdxi_cxt_stop {
	struct sdxi_cxt_range range;
};

int sdxi_encode_cxt_stop(struct sdxi_desc *desc,
			  const struct sdxi_cxt_stop *params);

struct sdxi_cxt_upd {
	enum sdxi_cxt_upd_level {
		SDXI_CXT_UPD_LVL_CTL = 0x4,
		SDXI_CXT_UPD_LVL_L1  = 0x6,
		SDXI_CXT_UPD_LVL_L2  = 0x7,
	} level;
	struct sdxi_cxt_range range;
};

int sdxi_encode_cxt_upd(struct sdxi_desc *desc,
			const struct sdxi_cxt_upd *params);

struct sdxi_sync {
	enum sdxi_sync_filter  {
		SDXI_SYNC_FLT_CXT  = 0x0,
		SDXI_SYNC_FLT_STOP = 0x1,
		SDXI_SYNC_FLT_AKEY = 0x2,
		SDXI_SYNC_FLT_RKEY = 0x3,
		SDXI_SYNC_FLT_FN   = 0x4,
	} filter;
	struct sdxi_cxt_range range;
};

int sdxi_encode_sync(struct sdxi_desc *desc, const struct sdxi_sync *params);

#endif /* DMA_SDXI_DESCRIPTOR_H */
