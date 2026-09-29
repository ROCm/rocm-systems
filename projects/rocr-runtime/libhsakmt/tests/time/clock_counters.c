/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "libhsakmt.h"
#include <string.h>
#include <time.h>

/* Link the real time.c against a deterministic clock and ioctl transport.
 * No KFD device or GPU is needed. Checks remain enabled in release builds.
 */
#define CHECK(condition) do { \
	if (!(condition)) { \
		fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
		exit(1); \
	} \
} while (0)

unsigned long hsakmt_kfd_open_count = 1;
bool hsakmt_forked;
HsaKFDContext hsakmt_primary_kfd_ctx = {.fd = 19};

struct sample {
	uint64_t latency;
	struct kfd_ioctl_get_clock_counters_args value;
};

static struct sample samples[4];
static uint64_t now;
static unsigned int query_count, clock_count;
static int fail_query, fail_clock;

HSAKMT_STATUS hsakmt_validate_nodeid(HsaKFDContext *ctx, uint32_t nodeid,
				   uint32_t *gpu_id)
{
	CHECK(ctx == &hsakmt_primary_kfd_ctx);
	if (nodeid > 1)
		return HSAKMT_STATUS_INVALID_PARAMETER;
	*gpu_id = nodeid ? 12345 : 0;
	return HSAKMT_STATUS_SUCCESS;
}

int clock_gettime(clockid_t clock_id, struct timespec *ts)
{
	CHECK(clock_id == CLOCK_MONOTONIC_RAW);
	if ((int)clock_count++ == fail_clock)
		return -1;
	ts->tv_sec = now / 1000000000;
	ts->tv_nsec = now % 1000000000;
	return 0;
}

int hsakmt_ioctl(int fd, unsigned long request, void *arg)
{
	struct kfd_ioctl_get_clock_counters_args *out = arg;
	unsigned int index = query_count++;

	CHECK(fd == 19);
	CHECK(request == AMDKFD_IOC_GET_CLOCK_COUNTERS);
	CHECK(out->gpu_id == 12345 || out->gpu_id == 0);
	CHECK(index < 4);
	if ((int)index == fail_query)
		return -1;
	now += samples[index].latency;
	*out = samples[index].value;
	return 0;
}

static void reset(void)
{
	unsigned int i;

	hsakmt_kfd_open_count = 1;
	hsakmt_forked = false;
	now = 1999999000;
	query_count = clock_count = 0;
	fail_query = fail_clock = -1;
	for (i = 0; i < 4; i++) {
		samples[i].latency = 2500;
		samples[i].value = (struct kfd_ioctl_get_clock_counters_args) {
			.gpu_clock_counter = 100 + i,
			.cpu_clock_counter = 1000 + i,
			.system_clock_counter = 2000 + i,
			.system_clock_freq = 1000000000,
		};
	}
}

static void expect_sample(HsaClockCounters *value, unsigned int index)
{
	CHECK(value->GPUClockCounter == samples[index].value.gpu_clock_counter);
	CHECK(value->CPUClockCounter == samples[index].value.cpu_clock_counter);
	CHECK(value->SystemClockCounter == samples[index].value.system_clock_counter);
	CHECK(value->SystemClockFrequencyHz == samples[index].value.system_clock_freq);
}

static void rejects_delayed_correlation(void)
{
	HsaClockCounters value = {0};
	unsigned int position;

	/* Exercise every possible position of the good sample, including after
	 * an interrupt-delayed pair and across a timespec second boundary.
	 */
	for (position = 0; position < 4; position++) {
		unsigned int fastest = (position + 1) % 4;
		unsigned int i;

		reset();
		for (i = 0; i < 4; i++) {
			samples[i].latency = 50000;
			samples[i].value.system_clock_counter += 46628;
		}
		samples[fastest].latency = 2200;
		samples[fastest].value.system_clock_counter -= 46628;
		CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_SUCCESS);
		expect_sample(&value, fastest);
		CHECK(query_count == 4);
		CHECK(clock_count == 8);
	}
}

static void equal_latency_keeps_one_complete_pair(void)
{
	HsaClockCounters value = {0};

	reset();
	CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_SUCCESS);
	expect_sample(&value, 0);
}

static void cpu_node_uses_one_query(void)
{
	HsaClockCounters value = {0};

	reset();
	samples[0].value.gpu_clock_counter = 0;
	CHECK(hsaKmtGetClockCounters(0, &value) == HSAKMT_STATUS_SUCCESS);
	CHECK(query_count == 1);
	expect_sample(&value, 0);
}

static void errors_do_not_publish_partial_samples(void)
{
	HsaClockCounters value, original;
	int i;

	memset(&original, 0xa5, sizeof(original));
	for (i = 0; i < 4; i++) {
		reset();
		value = original;
		fail_query = i;
		CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_ERROR);
		CHECK(memcmp(&value, &original, sizeof(value)) == 0);
		CHECK(query_count == (unsigned int)i + 1);
	}
	for (i = 0; i < 8; i++) {
		reset();
		value = original;
		fail_clock = i;
		CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_ERROR);
		CHECK(memcmp(&value, &original, sizeof(value)) == 0);
		CHECK(clock_count == (unsigned int)i + 1);
	}
}

static void rejects_invalid_context_and_node(void)
{
	HsaClockCounters value = {0};

	reset();
	hsakmt_kfd_open_count = 0;
	CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
	CHECK(query_count == 0);
	reset();
	hsakmt_forked = true;
	CHECK(hsaKmtGetClockCounters(1, &value) == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED);
	CHECK(query_count == 0);
	reset();
	CHECK(hsaKmtGetClockCounters(2, &value) == HSAKMT_STATUS_INVALID_PARAMETER);
	CHECK(query_count == 0);
}

int main(void)
{
	rejects_delayed_correlation();
	equal_latency_keeps_one_complete_pair();
	cpu_node_uses_one_query();
	errors_do_not_publish_partial_samples();
	rejects_invalid_context_and_node();
	puts("Clock-counter sampling tests passed");
	return 0;
}
