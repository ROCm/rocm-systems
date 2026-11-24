/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * FreeBSD stand-in for src/perfctr.c.
 *
 * The Linux implementation drives the GPU performance counters through
 * perf_event_open(2) and <linux/perf_event.h>, neither of which FreeBSD
 * provides.  Until an hwpmc(4)-based backend exists, this file supplies the
 * same symbol set so that libhsakmt links and callers get a well-defined
 * HSAKMT_STATUS_NOT_SUPPORTED instead of a missing symbol.  CMake selects
 * exactly one of the two implementations (see HSAKMT_SRC in CMakeLists.txt).
 */

#include "libhsakmt.h"

int hsakmt_kfdcontext_init_perf_context(HsaKFDContext *ctx)
{
	CHECK_CTX(ctx, -1);

	/* The Linux build allocates a struct hsa_kfd_perf_context here to hold the
	 * per-node counter properties.  Nothing below records any, so leave
	 * ctx->perf_context NULL; hsakmt_kfdcontext_clear_context() free()s it
	 * unconditionally, which is well defined for NULL.
	 */
	return 0;
}

HSAKMT_STATUS hsakmt_init_counter_props(HsaKFDContext *ctx, unsigned int NumNodes)
{
	/* Nothing to set up: no node ever reports counter properties. */
	return HSAKMT_STATUS_SUCCESS;
}

void hsakmt_destroy_counter_props(HsaKFDContext *ctx)
{
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcGetCounterPropertiesCtx(HsaKFDContext *ctx,
						      HSAuint32 NodeId,
						      HsaCounterProperties **CounterProperties)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcRegisterTraceCtx(HsaKFDContext* ctx,
					       HSAuint32 NodeId,
					       HSAuint32 NumberOfCounters,
					       HsaCounter *Counters,
					       HsaPmcTraceRoot *TraceRoot)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcUnregisterTraceCtx(HsaKFDContext* ctx,
						 HSAuint32 NodeId,
						 HSATraceId TraceId)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcAcquireTraceAccessCtx(HsaKFDContext* ctx,
						    HSAuint32 NodeId,
						    HSATraceId TraceId)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcReleaseTraceAccess(HSAuint32 NodeId,
						    HSATraceId TraceId)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcStartTrace(HSATraceId TraceId,
					    void *TraceBuffer,
					    HSAuint64 TraceBufferSizeBytes)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcQueryTrace(HSATraceId TraceId)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcStopTrace(HSATraceId TraceId)
{
	pr_warn_once("performance counters are not supported on FreeBSD\n");
	return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcGetCounterProperties(HSAuint32 NodeId,
						      HsaCounterProperties **CounterProperties)
{
	return hsaKmtPmcGetCounterPropertiesCtx(&hsakmt_primary_kfd_ctx, NodeId,
						CounterProperties);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcRegisterTrace(HSAuint32 NodeId,
					       HSAuint32 NumberOfCounters,
					       HsaCounter *Counters,
					       HsaPmcTraceRoot *TraceRoot)
{
	return hsaKmtPmcRegisterTraceCtx(&hsakmt_primary_kfd_ctx, NodeId,
					 NumberOfCounters, Counters, TraceRoot);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcUnregisterTrace(HSAuint32 NodeId,
						 HSATraceId TraceId)
{
	return hsaKmtPmcUnregisterTraceCtx(&hsakmt_primary_kfd_ctx, NodeId, TraceId);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPmcAcquireTraceAccess(HSAuint32 NodeId,
						    HSATraceId TraceId)
{
	return hsaKmtPmcAcquireTraceAccessCtx(&hsakmt_primary_kfd_ctx, NodeId, TraceId);
}
