/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef PMC_TABLE_H
#define PMC_TABLE_H

#include "libhsakmt.h"

enum perf_block_id {
	PERFCOUNTER_BLOCKID__FIRST = 0,
	/* non-privileged */
	PERFCOUNTER_BLOCKID__CB = PERFCOUNTER_BLOCKID__FIRST,
	PERFCOUNTER_BLOCKID__CPC,
	PERFCOUNTER_BLOCKID__CPF,
	PERFCOUNTER_BLOCKID__CPG,
	PERFCOUNTER_BLOCKID__DB,
	PERFCOUNTER_BLOCKID__GDS,
	PERFCOUNTER_BLOCKID__GRBM,
	PERFCOUNTER_BLOCKID__GRBMSE,
	PERFCOUNTER_BLOCKID__IA,
	PERFCOUNTER_BLOCKID__MC,
	PERFCOUNTER_BLOCKID__PASC,
	PERFCOUNTER_BLOCKID__PASU,
	PERFCOUNTER_BLOCKID__SPI,
	PERFCOUNTER_BLOCKID__SRBM,
	PERFCOUNTER_BLOCKID__SQ,
	PERFCOUNTER_BLOCKID__SX,
	PERFCOUNTER_BLOCKID__TA,
	PERFCOUNTER_BLOCKID__TCA,
	PERFCOUNTER_BLOCKID__TCC,
	PERFCOUNTER_BLOCKID__TCP,
	PERFCOUNTER_BLOCKID__TCS,
	PERFCOUNTER_BLOCKID__TD,
	PERFCOUNTER_BLOCKID__VGT,
	PERFCOUNTER_BLOCKID__WD,
	/* privileged */
	PERFCOUNTER_BLOCKID__MAX
};

struct perf_counter_block {
	uint32_t    num_of_slots;
	uint32_t    num_of_counters;
	uint32_t    *counter_ids;
	uint32_t    counter_size_in_bits;
	uint64_t    counter_mask;
};

HSAKMT_STATUS hsakmt_get_block_properties(uint32_t node_id,
				   enum perf_block_id block_id,
				   struct perf_counter_block *block);

#endif // PMC_TABLE_H
