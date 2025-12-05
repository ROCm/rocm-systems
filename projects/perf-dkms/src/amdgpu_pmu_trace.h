/*
 * amdgpu_pmu_trace.h - Tracepoint definitions for AMDGPU PMU
 *
 * This header defines tracepoints for GPU performance counter sampling,
 * allowing perf record to capture counter samples as trace events.
 *
 * Usage:
 *   perf record -e amdgpu_pmu:counter_sample -a ./app
 *   perf script
 */

#undef TRACE_SYSTEM
#define TRACE_SYSTEM amdgpu_pmu

#if !defined(_AMDGPU_PMU_TRACE_H) || defined(TRACE_HEADER_MULTI_READ)
#define _AMDGPU_PMU_TRACE_H

#include <linux/tracepoint.h>

/**
 * counter_sample - Emitted when a GPU counter is sampled
 * @counter_id: Numeric counter identifier (from counter_id_t enum)
 * @counter_name: Human-readable counter name (e.g., "SQ_WAVES")
 * @value: Current counter value
 * @delta: Change since last sample (for rate calculation)
 * @timestamp_ns: High-precision timestamp (ktime_get_ns())
 *
 * This tracepoint is emitted by the timer handler for each active sampling
 * event. It provides full visibility into GPU counter behavior over time.
 *
 * Example output:
 *   test_32_waves-1234 [000] 1000.123: amdgpu_pmu:counter_sample: \
 *       id=30 name=SQ_WAVES value=64 delta=32 ts=1000123456789
 */
TRACE_EVENT(counter_sample,

	    TP_PROTO(u32 counter_id, const char *counter_name, u64 value, u64 delta,
		     u64 timestamp_ns),

	    TP_ARGS(counter_id, counter_name, value, delta, timestamp_ns),

	    TP_STRUCT__entry(__field(u32, counter_id) __string(counter_name, counter_name)
				     __field(u64, value) __field(u64, delta)
					     __field(u64, timestamp_ns)),

	    TP_fast_assign(__entry->counter_id = counter_id; __assign_str(counter_name);
			   __entry->value = value; __entry->delta = delta;
			   __entry->timestamp_ns = timestamp_ns;),

	    TP_printk("id=%u name=%s value=%llu delta=%llu ts=%llu", __entry->counter_id,
		      __get_str(counter_name), __entry->value, __entry->delta,
		      __entry->timestamp_ns));

#endif /* _AMDGPU_PMU_TRACE_H */

/* This part must be outside protection */
#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH ../../ src
#undef TRACE_INCLUDE_FILE
#define TRACE_INCLUDE_FILE amdgpu_pmu_trace
#include <trace/define_trace.h>
