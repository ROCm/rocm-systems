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

#include "libhsakmt.h"
#include "kfd_ioctl.h"
#include <time.h>

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCountersCtx(HsaKFDContext *ctx,
					       HSAuint32 NodeId,
					       HsaClockCounters *Counters)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;
	struct kfd_ioctl_get_clock_counters_args args = {0};
	struct kfd_ioctl_get_clock_counters_args best = {0};
	uint64_t best_elapsed = UINT64_MAX;
	unsigned int i, samples;

	CHECK_KFD_OPEN();

	result = hsakmt_validate_nodeid(ctx, NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	/* KFD reads the GPU counter before the CPU clocks. An interrupt or a
	 * reschedule between those reads can displace the correlation by tens of
	 * microseconds. Reduce that uncertainty by selecting the shortest of a
	 * small, fixed number of queries. Keep all counters from the same sample.
	 * CPU-only nodes have no GPU/CPU correlation to establish.
	 */
	samples = gpu_id ? 4 : 1;
	for (i = 0; i < samples; i++) {
		struct timespec before, after;
		uint64_t elapsed;

		args.gpu_id = gpu_id;
		if (clock_gettime(CLOCK_MONOTONIC_RAW, &before))
			return HSAKMT_STATUS_ERROR;
		if (hsakmt_ioctl(ctx->fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args) < 0)
			return HSAKMT_STATUS_ERROR;
		if (clock_gettime(CLOCK_MONOTONIC_RAW, &after))
			return HSAKMT_STATUS_ERROR;

		elapsed = (uint64_t)(after.tv_sec - before.tv_sec) * 1000000000 +
			  after.tv_nsec - before.tv_nsec;
		if (elapsed < best_elapsed) {
			best_elapsed = elapsed;
			best = args;
		}
	}

	Counters->GPUClockCounter = best.gpu_clock_counter;
	Counters->CPUClockCounter = best.cpu_clock_counter;
	Counters->SystemClockCounter = best.system_clock_counter;
	Counters->SystemClockFrequencyHz = best.system_clock_freq;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCounters(HSAuint32 NodeId,
					       HsaClockCounters *Counters)
{
	return hsaKmtGetClockCountersCtx(&hsakmt_primary_kfd_ctx, NodeId, Counters);
}
