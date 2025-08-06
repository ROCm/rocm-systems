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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include "libhsakmt.h"
#include "pmc_table.h"
#include "hsakmt/linux/kfd_ioctl.h"
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>

#define BITS_PER_BYTE		CHAR_BIT

#define HSA_PERF_MAGIC4CC	0x54415348

enum perf_trace_state {
	PERF_TRACE_STATE__STOPPED = 0,
	PERF_TRACE_STATE__STARTED
};

struct perf_trace_block {
	enum perf_block_id block_id;
	uint32_t num_counters;
	uint64_t *counter_id;
	int *perf_event_fd;
};

struct perf_trace {
	uint32_t magic4cc;
	uint32_t gpu_id;
	enum perf_trace_state state;
	uint32_t num_blocks;
	void *buf;
	uint64_t buf_size;
	struct perf_trace_block blocks[0];
};

struct perf_counts_values {
	union {
		struct {
			uint64_t val;
			uint64_t ena;
			uint64_t run;
		};
		uint64_t values[3];
	};
};

static HsaCounterProperties **counter_props;
static unsigned int counter_props_count;

static ssize_t readn(int fd, void *buf, size_t n)
{
	size_t left = n;
	ssize_t bytes;

	while (left) {
		bytes = read(fd, buf, left);
		if (!bytes) /* reach EOF */
			return (n - left);
		if (bytes < 0) {
			if (errno == EINTR) /* read got interrupted */
				continue;
			else
				return -errno;
		}
		left -= bytes;
		buf = VOID_PTR_ADD(buf, bytes);
	}
	return n;
}

HSAKMT_STATUS hsakmt_init_counter_props(unsigned int NumNodes)
{
	counter_props = calloc(NumNodes, sizeof(struct HsaCounterProperties *));
	if (!counter_props) {
		pr_warn("Profiling is not available.\n");
		return HSAKMT_STATUS_NO_MEMORY;
	}

	counter_props_count = NumNodes;

	return HSAKMT_STATUS_SUCCESS;
}

void hsakmt_destroy_counter_props(void)
{
	unsigned int i;

	if (!counter_props)
		return;

	for (i = 0; i < counter_props_count; i++)
		if (counter_props[i]) {
			free(counter_props[i]);
			counter_props[i] = NULL;
		}

	free(counter_props);
}

static int blockid2uuid(enum perf_block_id block_id, HSA_UUID *uuid)
{
	int rc = 0;

	switch (block_id) {
	case PERFCOUNTER_BLOCKID__CB:
		*uuid = HSA_PROFILEBLOCK_AMD_CB;
		break;
	case PERFCOUNTER_BLOCKID__CPF:
		*uuid = HSA_PROFILEBLOCK_AMD_CPF;
		break;
	case PERFCOUNTER_BLOCKID__CPG:
		*uuid = HSA_PROFILEBLOCK_AMD_CPG;
		break;
	case PERFCOUNTER_BLOCKID__DB:
		*uuid = HSA_PROFILEBLOCK_AMD_DB;
		break;
	case PERFCOUNTER_BLOCKID__GDS:
		*uuid = HSA_PROFILEBLOCK_AMD_GDS;
		break;
	case PERFCOUNTER_BLOCKID__GRBM:
		*uuid = HSA_PROFILEBLOCK_AMD_GRBM;
		break;
	case PERFCOUNTER_BLOCKID__GRBMSE:
		*uuid = HSA_PROFILEBLOCK_AMD_GRBMSE;
		break;
	case PERFCOUNTER_BLOCKID__IA:
		*uuid = HSA_PROFILEBLOCK_AMD_IA;
		break;
	case PERFCOUNTER_BLOCKID__MC:
		*uuid = HSA_PROFILEBLOCK_AMD_MC;
		break;
	case PERFCOUNTER_BLOCKID__PASC:
		*uuid = HSA_PROFILEBLOCK_AMD_PASC;
		break;
	case PERFCOUNTER_BLOCKID__PASU:
		*uuid = HSA_PROFILEBLOCK_AMD_PASU;
		break;
	case PERFCOUNTER_BLOCKID__SPI:
		*uuid = HSA_PROFILEBLOCK_AMD_SPI;
		break;
	case PERFCOUNTER_BLOCKID__SRBM:
		*uuid = HSA_PROFILEBLOCK_AMD_SRBM;
		break;
	case PERFCOUNTER_BLOCKID__SQ:
		*uuid = HSA_PROFILEBLOCK_AMD_SQ;
		break;
	case PERFCOUNTER_BLOCKID__SX:
		*uuid = HSA_PROFILEBLOCK_AMD_SX;
		break;
	case PERFCOUNTER_BLOCKID__TA:
		*uuid = HSA_PROFILEBLOCK_AMD_TA;
		break;
	case PERFCOUNTER_BLOCKID__TCA:
		*uuid = HSA_PROFILEBLOCK_AMD_TCA;
		break;
	case PERFCOUNTER_BLOCKID__TCC:
		*uuid = HSA_PROFILEBLOCK_AMD_TCC;
		break;
	case PERFCOUNTER_BLOCKID__TCP:
		*uuid = HSA_PROFILEBLOCK_AMD_TCP;
		break;
	case PERFCOUNTER_BLOCKID__TCS:
		*uuid = HSA_PROFILEBLOCK_AMD_TCS;
		break;
	case PERFCOUNTER_BLOCKID__TD:
		*uuid = HSA_PROFILEBLOCK_AMD_TD;
		break;
	case PERFCOUNTER_BLOCKID__VGT:
		*uuid = HSA_PROFILEBLOCK_AMD_VGT;
		break;
	case PERFCOUNTER_BLOCKID__WD:
		*uuid = HSA_PROFILEBLOCK_AMD_WD;
		break;
	default:
		/* If we reach this point, it's a bug */
		rc = -1;
		break;
	}

	return rc;
}

static HSAuint32 get_block_concurrent_limit(uint32_t node_id,
						HSAuint32 block_id)
{
	uint32_t i;
	HsaCounterBlockProperties *block = &counter_props[node_id]->Blocks[0];

	for (i = 0; i < PERFCOUNTER_BLOCKID__MAX; i++) {
		if (block->Counters[0].BlockIndex == block_id)
			return block->NumConcurrent;
		block = (HsaCounterBlockProperties *)&block->Counters[block->NumCounters];
	}

	return 0;
}

static HSAKMT_STATUS perf_trace_ioctl(struct perf_trace_block *block,
				      uint32_t cmd)
{
	uint32_t i;

	for (i = 0; i < block->num_counters; i++) {
		if (block->perf_event_fd[i] < 0)
			return HSAKMT_STATUS_UNAVAILABLE;
		if (ioctl(block->perf_event_fd[i], cmd, NULL))
			return HSAKMT_STATUS_ERROR;
	}

	return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS query_trace(int fd, uint64_t *buf)
{
	struct perf_counts_values content;

	if (fd < 0)
		return HSAKMT_STATUS_ERROR;
	if (readn(fd, &content, sizeof(content)) != sizeof(content))
		return HSAKMT_STATUS_ERROR;

	*buf = content.val;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcGetCounterProperties(HSAuint32 NodeId,
						      HsaCounterProperties **CounterProperties)
{
	HSAKMT_STATUS rc = HSAKMT_STATUS_SUCCESS;
	uint32_t gpu_id, i, block_id;
	uint32_t counter_props_size = 0;
	uint32_t total_counters = 0;
	uint32_t total_concurrent = 0;
	struct perf_counter_block block = {0};
	uint32_t total_blocks = 0;
	HsaCounterBlockProperties *block_prop;

	if (!counter_props)
		return HSAKMT_STATUS_NO_MEMORY;

	if (!CounterProperties)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (hsakmt_validate_nodeid(NodeId, &gpu_id) != HSAKMT_STATUS_SUCCESS)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	if (counter_props[NodeId]) {
		*CounterProperties = counter_props[NodeId];
		return HSAKMT_STATUS_SUCCESS;
	}

	for (i = 0; i < PERFCOUNTER_BLOCKID__MAX; i++) {
		rc = hsakmt_get_block_properties(NodeId, i, &block);
		if (rc != HSAKMT_STATUS_SUCCESS)
			return rc;
		total_concurrent += block.num_of_slots;
		total_counters += block.num_of_counters;
		/* If num_of_slots=0, this block doesn't exist */
		if (block.num_of_slots)
			total_blocks++;
	}

	counter_props_size = sizeof(HsaCounterProperties) +
			sizeof(HsaCounterBlockProperties) * (total_blocks - 1) +
			sizeof(HsaCounter) * (total_counters - total_blocks);

	counter_props[NodeId] = malloc(counter_props_size);
	if (!counter_props[NodeId])
		return HSAKMT_STATUS_NO_MEMORY;

	counter_props[NodeId]->NumBlocks = total_blocks;
	counter_props[NodeId]->NumConcurrent = total_concurrent;

	block_prop = &counter_props[NodeId]->Blocks[0];
	for (block_id = 0; block_id < PERFCOUNTER_BLOCKID__MAX; block_id++) {
		rc = hsakmt_get_block_properties(NodeId, block_id, &block);
		if (rc != HSAKMT_STATUS_SUCCESS) {
			free(counter_props[NodeId]);
			counter_props[NodeId] = NULL;
			return rc;
		}

		if (!block.num_of_slots) /* not a valid block */
			continue;

		blockid2uuid(block_id, &block_prop->BlockId);
		block_prop->NumCounters = block.num_of_counters;
		block_prop->NumConcurrent = block.num_of_slots;
		for (i = 0; i < block.num_of_counters; i++) {
			block_prop->Counters[i].BlockIndex = block_id;
			block_prop->Counters[i].CounterId = block.counter_ids[i];
			block_prop->Counters[i].CounterSizeInBits = block.counter_size_in_bits;
			block_prop->Counters[i].CounterMask = block.counter_mask;
			block_prop->Counters[i].Flags.ui32.Global = 1;
			block_prop->Counters[i].Type = HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE;
		}

		block_prop = (HsaCounterBlockProperties *)&block_prop->Counters[block_prop->NumCounters];
	}

	*CounterProperties = counter_props[NodeId];

	return HSAKMT_STATUS_SUCCESS;
}

/* Registers a set of (HW) counters to be used for tracing/profiling */
HSAKMT_STATUS HSAKMTAPI hsaKmtPmcRegisterTrace(HSAuint32 NodeId,
					       HSAuint32 NumberOfCounters,
					       HsaCounter *Counters,
					       HsaPmcTraceRoot *TraceRoot)
{
	uint32_t gpu_id, i, j;
	uint64_t min_buf_size = 0;
	struct perf_trace *trace = NULL;
	uint32_t concurrent_limit;
	const uint32_t MAX_COUNTERS = 512;

	/* Declare performance counter ID 2D array as a contiguous block */
	uint64_t *counter_id = malloc(
			PERFCOUNTER_BLOCKID__MAX * MAX_COUNTERS * sizeof(uint64_t));
	uint32_t num_counters[PERFCOUNTER_BLOCKID__MAX] = {0};
	uint32_t block, num_blocks = 0, total_counters = 0;
	uint64_t *counter_id_ptr;
	int *fd_ptr;

	pr_debug("[%s] Number of counters %d\n", __func__, NumberOfCounters);

	if (counter_id == NULL) {
		pr_err("Failed to allocate memory for counter_id. Requested %zu bytes.\n",
				PERFCOUNTER_BLOCKID__MAX * MAX_COUNTERS * sizeof(uint64_t));
		return HSAKMT_STATUS_NO_MEMORY;
	}

	if (!counter_props) {
		pr_err("Profiling is not available, counter_props is NULL.\n");
		goto no_memory_exit;
	}

	if (!Counters || !TraceRoot || NumberOfCounters == 0)
		goto invalid_parameter_exit;

	if (hsakmt_validate_nodeid(NodeId, &gpu_id) != HSAKMT_STATUS_SUCCESS) {
		free(counter_id);
		return HSAKMT_STATUS_INVALID_NODE_UNIT;
	}

	if (NumberOfCounters > MAX_COUNTERS) {
		pr_err("MAX_COUNTERS is too small for %d.\n", NumberOfCounters);
		goto no_memory_exit;
	}

	/* Calculating the minimum buffer size */
	for (i = 0; i < NumberOfCounters; i++) {
		if (Counters[i].BlockIndex >= PERFCOUNTER_BLOCKID__MAX)
			goto invalid_parameter_exit;
		/* Only privileged counters need to register */
		if (Counters[i].Type > HSA_PROFILE_TYPE_PRIVILEGED_STREAMING)
			continue;
		min_buf_size += Counters[i].CounterSizeInBits/BITS_PER_BYTE;
		/* j: the first blank entry in the block to record counter_id */
		j = num_counters[Counters[i].BlockIndex];
		/* Make sure counter_id stays within bounds */
		if (j >= MAX_COUNTERS) {
			pr_err("Counter ID exceeded MAX_COUNTERS for block %d.\n",
					Counters[i].BlockIndex);
			goto invalid_parameter_exit;
		}
		/* Initialize counter_id */
		counter_id[Counters[i].BlockIndex * MAX_COUNTERS + j] = Counters[i].CounterId;
		num_counters[Counters[i].BlockIndex]++;
		total_counters++;
	}

	/* Verify that the number of counters per block is not larger than the
	 * number of slots.
	 */
	for (i = 0; i < PERFCOUNTER_BLOCKID__MAX; i++) {
		if (!num_counters[i])
			continue;
		concurrent_limit = get_block_concurrent_limit(NodeId, i);
		if (!concurrent_limit) {
			pr_err("Invalid block ID: %d\n", i);
			goto invalid_parameter_exit;
		}
		if (num_counters[i] > concurrent_limit) {
			pr_err("Counters exceed the limit.\n");
			goto invalid_parameter_exit;
		}
		num_blocks++;
	}

	if (!num_blocks)
		goto invalid_parameter_exit;

	/* Now we have sorted blocks/counters information in
	 * num_counters[block_id] and counter_id[block_id][]. Allocate trace
	 * and record the information.
	 */
	trace = (struct perf_trace *)calloc(sizeof(struct perf_trace)
			+ sizeof(struct perf_trace_block) * num_blocks
			+ sizeof(uint64_t) * total_counters
			+ sizeof(int) * total_counters,
			1);
	if (!trace) {
		pr_err("Failed to allocate memory for trace. Requested %zu bytes.\n",
				sizeof(struct perf_trace)
				+ sizeof(struct perf_trace_block) * num_blocks
				+ sizeof(uint64_t) * total_counters
				+ sizeof(int) * total_counters);
		goto no_memory_exit;
	}

	/* Allocated area is partitioned as:
	 * +---------------------------------+ trace
	 * |    perf_trace                   |
	 * |---------------------------------| trace->blocks[0]
	 * | perf_trace_block 0              |
	 * | ....                            |
	 * | perf_trace_block N-1            | trace->blocks[N-1]
	 * |---------------------------------| <-- counter_id_ptr starts here
	 * | block 0's counter IDs(uint64_t) |
	 * | ......                          |
	 * | block N-1's counter IDs         |
	 * |---------------------------------| <-- perf_event_fd starts here
	 * | block 0's perf_event_fds(int)   |
	 * | ......                          |
	 * | block N-1's perf_event_fds      |
	 * +---------------------------------+
	 */
	block = 0;
	counter_id_ptr = (uint64_t *)((char *)
			trace + sizeof(struct perf_trace)
			+ sizeof(struct perf_trace_block) * num_blocks);
	fd_ptr = (int *)(counter_id_ptr + total_counters);
	/* Fill in each block's information to the TraceId */
	for (i = 0; i < PERFCOUNTER_BLOCKID__MAX; i++) {
		if (!num_counters[i]) /* not a block to trace */
			continue;
		/* Following perf_trace + perf_trace_block x N are those
		 * counter_id arrays. Assign the counter_id array belonging to
		 * this block.
		 */
		trace->blocks[block].counter_id = counter_id_ptr;
		/* Fill in counter IDs to the counter_id array. */
		for (j = 0; j < num_counters[i]; j++)
			trace->blocks[block].counter_id[j] = counter_id[i * MAX_COUNTERS + j];
		trace->blocks[block].perf_event_fd = fd_ptr;
		/* how many counters to trace */
		trace->blocks[block].num_counters = num_counters[i];
		/* block index in "enum perf_block_id" */
		trace->blocks[block].block_id = i;
		block++; /* move to next */
		counter_id_ptr += num_counters[i];
		fd_ptr += num_counters[i];
	}

	trace->magic4cc = HSA_PERF_MAGIC4CC;
	trace->gpu_id = gpu_id;
	trace->state = PERF_TRACE_STATE__STOPPED;
	trace->num_blocks = num_blocks;

	TraceRoot->NumberOfPasses = 1;
	TraceRoot->TraceBufferMinSizeBytes = PAGE_ALIGN_UP(min_buf_size);
	TraceRoot->TraceId = PORT_VPTR_TO_UINT64(trace);

	free(trace);
	free(counter_id);
	return HSAKMT_STATUS_SUCCESS;

	no_memory_exit:
		free(counter_id);
		return HSAKMT_STATUS_NO_MEMORY;

	invalid_parameter_exit:
		free(counter_id);
		return HSAKMT_STATUS_INVALID_PARAMETER;
}

/* Unregisters a set of (HW) counters used for tracing/profiling */

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcUnregisterTrace(HSAuint32 NodeId,
						 HSATraceId TraceId)
{
	uint32_t gpu_id;
	struct perf_trace *trace;

	pr_debug("[%s] Trace ID 0x%lx\n", __func__, TraceId);

	if (TraceId == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (hsakmt_validate_nodeid(NodeId, &gpu_id) != HSAKMT_STATUS_SUCCESS)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	trace = (struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	if (trace->gpu_id != gpu_id)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	/* If the trace is in the running state, stop it */
	if (trace->state == PERF_TRACE_STATE__STARTED) {
		HSAKMT_STATUS status = hsaKmtPmcStopTrace(TraceId);

		if (status != HSAKMT_STATUS_SUCCESS)
			return status;
	}

	free(trace);

	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcAcquireTraceAccess(HSAuint32 NodeId,
						    HSATraceId TraceId)
{
	struct perf_trace *trace;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint32_t gpu_id;

	pr_debug("[%s] Trace ID 0x%lx\n", __func__, TraceId);

	if (TraceId == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	trace = (struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	if (hsakmt_validate_nodeid(NodeId, &gpu_id) != HSAKMT_STATUS_SUCCESS)
		return HSAKMT_STATUS_INVALID_NODE_UNIT;

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcReleaseTraceAccess(HSAuint32 NodeId,
						    HSATraceId TraceId)
{
	struct perf_trace *trace;

	pr_debug("[%s] Trace ID 0x%lx\n", __func__, TraceId);

	if (TraceId == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	trace = (struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	return HSAKMT_STATUS_SUCCESS;
}


/* Starts tracing operation on a previously established set of performance counters */
HSAKMT_STATUS HSAKMTAPI hsaKmtPmcStartTrace(HSATraceId TraceId,
					    void *TraceBuffer,
					    HSAuint64 TraceBufferSizeBytes)
{
	struct perf_trace *trace =
			(struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);
	uint32_t i;
	int32_t j;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] Trace ID 0x%lx\n", __func__, TraceId);

	if (TraceId == 0 || !TraceBuffer || TraceBufferSizeBytes == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	for (i = 0; i < trace->num_blocks; i++) {
		ret = perf_trace_ioctl(&trace->blocks[i],
					PERF_EVENT_IOC_ENABLE);
		if (ret != HSAKMT_STATUS_SUCCESS)
			break;
	}
	if (ret != HSAKMT_STATUS_SUCCESS) {
		/* Disable enabled blocks before returning the failure. */
		j = (int32_t)i;
		while (--j >= 0)
			perf_trace_ioctl(&trace->blocks[j],
					PERF_EVENT_IOC_DISABLE);
		return ret;
	}

	trace->state = PERF_TRACE_STATE__STARTED;
	trace->buf = TraceBuffer;
	trace->buf_size = TraceBufferSizeBytes;

	return HSAKMT_STATUS_SUCCESS;
}


/*Forces an update of all the counters that a previously started trace operation has registered */

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcQueryTrace(HSATraceId TraceId)
{
	struct perf_trace *trace =
			(struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);
	uint32_t i, j;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
	uint64_t *buf;
	uint64_t buf_filled = 0;

	if (TraceId == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	buf = (uint64_t *)trace->buf;
	pr_debug("[%s] Trace buffer(%p): ", __func__, buf);
	for (i = 0; i < trace->num_blocks; i++)
		for (j = 0; j < trace->blocks[i].num_counters; j++) {
			buf_filled += sizeof(uint64_t);
			if (buf_filled > trace->buf_size)
				return HSAKMT_STATUS_NO_MEMORY;
			ret = query_trace(trace->blocks[i].perf_event_fd[j],
					buf);
			if (ret != HSAKMT_STATUS_SUCCESS)
				return ret;
			pr_debug("%lu_", *buf);
			buf++;
		}
	pr_debug("\n");

	return HSAKMT_STATUS_SUCCESS;
}


/* Stops tracing operation on a previously established set of performance counters */
HSAKMT_STATUS HSAKMTAPI hsaKmtPmcStopTrace(HSATraceId TraceId)
{
	struct perf_trace *trace =
			(struct perf_trace *)PORT_UINT64_TO_VPTR(TraceId);
	uint32_t i;
	HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;

	pr_debug("[%s] Trace ID 0x%lx\n", __func__, TraceId);

	if (TraceId == 0)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	if (trace->magic4cc != HSA_PERF_MAGIC4CC)
		return HSAKMT_STATUS_INVALID_HANDLE;

	for (i = 0; i < trace->num_blocks; i++) {
		ret = perf_trace_ioctl(&trace->blocks[i],
					PERF_EVENT_IOC_DISABLE);
		if (ret != HSAKMT_STATUS_SUCCESS)
			return ret;
	}

	trace->state = PERF_TRACE_STATE__STOPPED;

	return ret;
}
